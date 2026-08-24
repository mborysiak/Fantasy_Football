"""Research-only bounded Sequential Target Board for the Brown/Tuten state."""

from __future__ import annotations

import argparse
import ast
import faulthandler
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import types
import warnings

import pandas as pd

# Avoid the intermittent Windows numerical-thread stalls observed in repeated
# research replays. These must be set before importing NumPy/SciPy transitively.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings(
    "ignore",
    message="Unrecognized options detected: .*threads.*",
    category=RuntimeWarning,
)

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import run_experiment as experiment  # noqa: E402


audit = experiment.audit
sequential = experiment.sequential
RESULTS_DIR = STUDY_DIR / "results"


def save_board_state(results, summary, curves, prefix: str):
    base_path = RESULTS_DIR / prefix
    base_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(f"{base_path}_results.csv"), index=False)
    curves.to_csv(Path(f"{base_path}_curves.csv"), index=False)
    with Path(f"{base_path}_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit.json_value(summary), handle, indent=2, sort_keys=True)
    with Path(f"{base_path}_state.pkl").open("wb") as handle:
        pickle.dump((results, summary, curves), handle)


def run_board(
    variation: int,
    compute_budget: int,
    *,
    prior_results=None,
    prior_summary=None,
    prior_curves=None,
    output_prefix: str | None = None,
):
    audit.FIXED_SALARIES = {
        "Chase Brown": 34.0,
        "Bhayshul Tuten": 11.0,
    }
    database_uri = f"file:{audit.APP_DB.as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    production_simulator = sequential.simulate_history_only_branch
    production_market_inflation = sequential.history_market_inflation
    try:
        state = experiment.build_state(connection)
        sim = state["sim"]
        production_required_solver = (
            sim.solve_managed_roster_with_required_players
        )
        sim.solve_managed_roster_with_required_players = types.MethodType(
            experiment.direct_highs_required_roster_solve,
            sim,
        )

        def bounded_simulator(*args, **kwargs):
            return experiment.simulate_budget_aware_branch(
                *args,
                **kwargs,
                replan_mode="bounded",
                slack_floor=5.0,
            )

        def safe_history_market_inflation(
            observed_sales,
            minimum=0.75,
            maximum=1.35,
        ):
            if not observed_sales:
                return 1.0
            actual_excess = 0.0
            forecast_excess = 0.0
            for actual, forecast in observed_sales:
                actual_excess += max(float(actual) - 1.0, 0.0)
                forecast_excess += max(float(forecast) - 1.0, 0.0)
            return sequential._history_market_inflation_from_totals(
                actual_excess,
                forecast_excess,
                len(observed_sales),
                minimum=minimum,
                maximum=maximum,
            )

        sequential.simulate_history_only_branch = bounded_simulator
        sequential.history_market_inflation = safe_history_market_inflation
        seed = audit.evidence_seed(
            sim,
            state["waiver_baselines"],
            enforce_top_n=True,
            use_selection_premium=False,
            variation=int(variation),
        )
        results, summary, curves = sequential.run_sequential_target_board(
            sim,
            state["to_add"],
            state["to_drop"],
            compute_budget=int(compute_budget),
            require_top_n=audit.REQUIRE_TOP_N,
            enforce_top_n=True,
            roster_size=audit.ROSTER_SIZE,
            lineup_require=audit.LINEUP_REQUIRE,
            pos_min_counts=audit.POS_MIN,
            pos_max_counts=audit.POS_MAX,
            waiver_baselines=state["waiver_baselines"],
            remaining_market_budget=state["remaining_market_budget"],
            remaining_market_slots=state["remaining_market_slots"],
            use_selection_premium=False,
            random_seed=seed,
            parallel_workers=1,
            prior_results=prior_results,
            prior_summary=prior_summary,
            prior_curves=prior_curves,
        )
    finally:
        sequential.simulate_history_only_branch = production_simulator
        sequential.history_market_inflation = production_market_inflation
        if "sim" in locals() and "production_required_solver" in locals():
            sim.solve_managed_roster_with_required_players = (
                production_required_solver
            )
        connection.close()

    prefix = output_prefix or f"brown_tuten_bounded_board_v{int(variation)}"
    save_board_state(results, summary, curves, prefix)

    confirmed = results.loc[
        results.EvidenceStage.eq("Confirmed")
    ].sort_values("TargetRank", kind="mergesort")
    columns = [
        "TargetRank",
        "Player",
        "Pos",
        "MarketPrice",
        "Recommendation",
        "SequentialGain",
        "SequentialLCB80",
        "BuyEV",
        "PassEV",
        "BuySeasonP10",
        "PassSeasonP10",
        "BuyCompletion",
        "PassCompletion",
        "BlockPositiveRate",
    ]
    print(confirmed[columns].head(10).to_string(index=False), flush=True)
    print("\nSummary", flush=True)
    print(json.dumps(audit.json_value(summary), indent=2, sort_keys=True), flush=True)
    return results, summary, curves


def load_evidence_state(prefix: str):
    path = RESULTS_DIR / f"{prefix}_state.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    results = pd.read_csv(RESULTS_DIR / f"{prefix}_results.csv")
    curves = pd.read_csv(RESULTS_DIR / f"{prefix}_curves.csv")
    with (RESULTS_DIR / f"{prefix}_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    evidence_columns = (
        "_CurrentBlockEvidence",
        "_KeeperBlockEvidence",
        "_CandidateBenchFlags",
        "_CandidateKeeperWinRates",
    )
    for frame in (results, curves):
        for column in evidence_columns:
            if column not in frame.columns:
                continue
            frame[column] = frame[column].map(
                lambda value: (
                    ast.literal_eval(value)
                    if isinstance(value, str) and value.strip()
                    else tuple()
                )
            )
    if not sequential.sequential_evidence_can_accumulate(results, curves):
        raise ValueError(f"{prefix} does not contain reusable evidence blocks")
    return results, summary, curves


def add_evidence_batches(
    variation: int,
    compute_budget: int,
    batch_count: int,
):
    base_prefix = f"brown_tuten_bounded_board_v{int(variation)}"
    print(
        "Rebuilding the initial board in-process so reusable block evidence "
        "is retained...",
        flush=True,
    )
    prior_results, prior_summary, prior_curves = run_board(
        variation,
        compute_budget,
        output_prefix=base_prefix,
    )

    for batch_index in range(1, int(batch_count) + 1):
        next_variation = int(variation) + batch_index
        output_prefix = f"{base_prefix}_plus{batch_index}_evidence"
        print(
            f"\nAdding evidence batch {batch_index}/{batch_count} "
            f"with variation {next_variation}...",
            flush=True,
        )
        prior_results, prior_summary, prior_curves = run_board(
            next_variation,
            compute_budget,
            prior_results=prior_results,
            prior_summary=prior_summary,
            prior_curves=prior_curves,
            output_prefix=output_prefix,
        )
    return prior_results, prior_summary, prior_curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variation", type=int, default=14)
    parser.add_argument("--compute-budget", type=int, default=320)
    parser.add_argument("--add-evidence", type=int, default=0)
    parser.add_argument("--prior-prefix")
    parser.add_argument("--output-prefix")
    parser.add_argument("--confirmed-only", action="store_true")
    parser.add_argument("--only-player")
    parser.add_argument("--watchdog-seconds", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.watchdog_seconds:
        faulthandler.dump_traceback_later(
            int(arguments.watchdog_seconds),
            repeat=True,
        )
    if arguments.prior_prefix:
        previous_results, previous_summary, previous_curves = (
            load_evidence_state(arguments.prior_prefix)
        )
        if arguments.confirmed_only:
            previous_results = previous_results.loc[
                previous_results.EvidenceStage.astype(str).str.lower().eq(
                    "confirmed"
                )
            ].copy()
        if arguments.only_player:
            previous_results = previous_results.loc[
                previous_results.Player.eq(arguments.only_player)
                & previous_results.EvidenceStage.astype(str).str.lower().eq(
                    "confirmed"
                )
            ].copy()
            previous_curves = previous_curves.loc[
                previous_curves.Player.eq(arguments.only_player)
            ].copy()
            if len(previous_results) != 1 or len(previous_curves) == 0:
                raise ValueError(
                    f"Frozen player evidence is incomplete: "
                    f"{arguments.only_player}"
                )
        run_board(
            arguments.variation,
            arguments.compute_budget,
            prior_results=previous_results,
            prior_summary=previous_summary,
            prior_curves=previous_curves,
            output_prefix=arguments.output_prefix,
        )
    elif arguments.add_evidence:
        add_evidence_batches(
            arguments.variation,
            arguments.compute_budget,
            arguments.add_evidence,
        )
    else:
        run_board(arguments.variation, arguments.compute_budget)
