"""Frozen 2022-2024 Auction mean-versus-expected-excess replay."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
POWER_SCRIPT = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-27_auction_power_win_objective"
    / "run_power_win_test.py"
)
RESULTS_DIR = STUDY_DIR / "results"
REPLAY_YEARS = (2022, 2023, 2024)
ARMS = ("mean_frontier", "excess_pure", "excess_half_mean")
ARM_LABELS = {
    "mean_frontier": "Expected score",
    "excess_pure": "Pure expected excess",
    "excess_half_mean": "50/50 mean + expected excess",
}
FROZEN_RULES_VERSION = "2026-08-27_pre_multi_origin_excess_v1"
FROZEN_OBJECTIVE_RULES = {
    "mean_frontier": "maximum construction-bank managed-season mean",
    "excess_pure": "maximum construction-bank expected positive winning margin",
    "excess_half_mean": (
        "0.5 standardized construction mean + 0.5 standardized expected "
        "positive winning margin"
    ),
}

SPEC = importlib.util.spec_from_file_location(
    "auction_power_objective_frozen",
    POWER_SCRIPT,
)
power = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = power
SPEC.loader.exec_module(power)
base = power.base


def staged_simulation_database(year: int) -> Path:
    return (
        STUDY_DIR
        / "staging"
        / str(int(year))
        / "databases"
        / "Simulation.sqlite3"
    )


def configure_origin(year: int) -> Path:
    database = staged_simulation_database(year).resolve()
    if not database.is_file():
        raise FileNotFoundError(
            f"Build the isolated {year} replay before this study: {database}"
        )
    base.YEAR = int(year)
    base.SIMULATION_DB = database
    base.DEAD_ZONE_RBS = set()
    return database


def paired_lcb(values: pd.Series) -> float:
    numeric = values.to_numpy(dtype=np.float64)
    if len(numeric) < 2:
        return float("nan")
    return float(
        numeric.mean()
        - base.LCB80_Z * numeric.std(ddof=1) / np.sqrt(len(numeric))
    )


def select_frozen_arms(metrics: pd.DataFrame, plans: dict) -> dict:
    all_choices = power.select_arms(metrics, plans)
    return {arm: all_choices[arm] for arm in ARMS}


def build_origin(
    year: int,
    args: argparse.Namespace,
) -> dict:
    database = configure_origin(year)
    origin_seed = int(args.seed + 10_000 * (year - min(REPLAY_YEARS)))
    state = base.prepare_state(
        block_count=args.blocks,
        construction_contexts=args.construction_contexts,
        seed=origin_seed,
    )
    candidate_frames = []
    plan_frames = []
    cell_frames = []
    roster_rows = []
    all_candidate_rosters: set[tuple[str, ...]] = set()
    static_cache = {}
    validation_seeds = np.random.SeedSequence(origin_seed + 500).spawn(
        args.blocks
    )
    try:
        for block_idx in range(args.blocks):
            plans, sources = power.compile_candidates(
                state,
                block_idx,
                args.candidate_starts,
                origin_seed + 1000 * (block_idx + 1),
                static_cache,
            )
            metrics, rosters, _ = power.score_candidates(
                state,
                block_idx,
                plans,
                sources,
            )
            selected = select_frozen_arms(metrics, plans)
            metrics.insert(0, "year", year)
            candidate_frames.append(metrics)
            all_candidate_rosters.update(rosters)
            validation_seed = int(
                validation_seeds[block_idx].generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )
            validation_bank = base.sequential._sample_validation_bank(
                state["sim"],
                state["predictions"],
                args.validation_contexts,
                16,
                6,
                0.65,
                validation_seed,
                canonical_predictions=state["canonical_predictions"],
            )
            plan_frame, cell_frame = power.score_validation_block(
                state,
                block_idx,
                selected,
                rosters,
                validation_bank,
            )
            plan_frame.insert(0, "year", year)
            cell_frame.insert(0, "year", year)
            plan_frames.append(plan_frame)
            cell_frames.append(cell_frame)
            position_map = (
                state["predictions"].set_index("player").pos.to_dict()
            )
            experience_map = (
                state["predictions"].set_index("player").year_exp.to_dict()
            )
            for arm, choice in selected.items():
                for player in choice["roster"]:
                    roster_rows.append(
                        {
                            "year": year,
                            "block": block_idx,
                            "arm": arm,
                            "player": player,
                            "pos": position_map[player],
                            "year_exp": float(experience_map[player]),
                            "forecast_cost": float(
                                choice["plan"]["forecast_cost"][player]
                            ),
                        }
                    )
        return {
            "year": year,
            "database": database,
            "seed": origin_seed,
            "state": state,
            "candidate_metrics": pd.concat(
                candidate_frames,
                ignore_index=True,
            ),
            "plan_rows": pd.concat(plan_frames, ignore_index=True),
            "cell_rows": pd.concat(cell_frames, ignore_index=True),
            "roster_players": pd.DataFrame(roster_rows),
            "all_candidate_rosters": all_candidate_rosters,
        }
    except Exception:
        state["conn"].close()
        raise


def score_origin_actual(origin: dict) -> tuple[pd.DataFrame, dict]:
    year = int(origin["year"])
    configure_origin(year)
    actual_rows, coverage = power.score_actual(
        origin["state"],
        origin["plan_rows"],
        origin["all_candidate_rosters"],
    )
    actual_rows.insert(0, "year", year)
    return actual_rows, coverage


def summarize(
    plan_rows: pd.DataFrame,
    actual_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = plan_rows.merge(
        actual_rows,
        on=["year", "block", "arm"],
        validate="one_to_one",
    )
    numeric_metrics = [
        column
        for column in merged.columns
        if (
            column.startswith("holdout_")
            or column.startswith("actual_")
            or column.startswith("spend_")
            or column.startswith("count_")
        )
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    annual_summary = (
        merged.groupby(["year", "arm"], as_index=False)[numeric_metrics]
        .mean()
    )
    paired_rows = []
    for year in REPLAY_YEARS:
        year_rows = merged.loc[merged.year.eq(year)]
        baseline = year_rows.loc[
            year_rows.arm.eq("mean_frontier")
        ].set_index("block")
        for arm in ARMS:
            comparison = year_rows.loc[year_rows.arm.eq(arm)].set_index("block")
            record = {"year": year, "arm": arm, "blocks": len(comparison)}
            for metric in numeric_metrics:
                delta = comparison[metric] - baseline[metric]
                record[f"{metric}_delta"] = float(delta.mean())
                record[f"{metric}_delta_lcb80"] = paired_lcb(delta)
            paired_rows.append(record)
    paired = pd.DataFrame(paired_rows)
    season_level = paired[[
        "year",
        "arm",
        "holdout_mean_delta",
        "holdout_p90_delta",
        "holdout_expected_excess_delta",
        "actual_managed_score_delta",
    ]].copy()
    return merged, annual_summary, season_level


def player_frequencies(roster_players: pd.DataFrame, blocks: int) -> pd.DataFrame:
    frequencies = (
        roster_players.groupby(
            ["year", "arm", "player", "pos", "year_exp"],
            as_index=False,
        )
        .agg(
            blocks_selected=("block", "nunique"),
            mean_cost=("forecast_cost", "mean"),
        )
    )
    frequencies["selection_rate"] = frequencies.blocks_selected / int(blocks)
    return frequencies


def write_summary(
    annual_summary: pd.DataFrame,
    season_level: pd.DataFrame,
    frequencies: pd.DataFrame,
    metadata: dict,
) -> None:
    indexed = annual_summary.set_index(["year", "arm"])
    paired = season_level.set_index(["year", "arm"])
    lines = [
        "# Frozen Expected-Excess Multi-Origin Results",
        "",
        (
            "The three policies were frozen from the 2025 experiment before "
            "these 2022-2024 outcomes were scored. Candidate construction uses "
            "preseason projections and donors through the prior year; actual "
            "auction prices define the retrospective cost surface."
        ),
        "",
        "## Annual results",
        "",
        "| Year | Arm | Holdout EV | Holdout P90 | Expected excess | Actual score | Actual delta vs mean |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for year in REPLAY_YEARS:
        for arm in ARMS:
            row = indexed.loc[(year, arm)]
            delta = paired.loc[(year, arm), "actual_managed_score_delta"]
            lines.append(
                f"| {year} | {ARM_LABELS[arm]} | {row.holdout_mean:.2f} | "
                f"{row.holdout_p90:.2f} | {row.holdout_expected_excess:.2f} | "
                f"{row.actual_managed_score:.2f} | {delta:+.2f} |"
            )
    lines.extend([
        "",
        "## Cross-season readout",
        "",
        "| Arm | Mean actual delta | Seasons positive | Mean holdout-EV delta | Mean holdout-P90 delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for arm in ARMS:
        rows = season_level.loc[season_level.arm.eq(arm)]
        lines.append(
            f"| {ARM_LABELS[arm]} | "
            f"{rows.actual_managed_score_delta.mean():+.2f} | "
            f"{int(rows.actual_managed_score_delta.gt(0).sum())}/3 | "
            f"{rows.holdout_mean_delta.mean():+.2f} | "
            f"{rows.holdout_p90_delta.mean():+.2f} |"
        )
    lines.extend(["", "## Most frequent roster changes", ""])
    mean_rates = frequencies.loc[
        frequencies.arm.eq("mean_frontier"),
        ["year", "player", "pos", "selection_rate"],
    ].rename(columns={"selection_rate": "mean_rate"})
    for arm in ("excess_pure", "excess_half_mean"):
        changes = frequencies.loc[
            frequencies.arm.eq(arm),
            ["year", "player", "pos", "selection_rate"],
        ].merge(
            mean_rates,
            on=["year", "player", "pos"],
            how="outer",
        ).fillna(0.0)
        changes["rate_delta"] = changes.selection_rate - changes.mean_rate
        gains = changes.nlargest(10, "rate_delta")
        losses = changes.nsmallest(10, "rate_delta")
        gain_text = ", ".join(
            f"{row.player} {int(row.year)} ({row.rate_delta:+.0%})"
            for row in gains.itertuples(index=False)
            if row.rate_delta > 0
        )
        loss_text = ", ".join(
            f"{row.player} {int(row.year)} ({row.rate_delta:+.0%})"
            for row in losses.itertuples(index=False)
            if row.rate_delta < 0
        )
        lines.append(f"- **{ARM_LABELS[arm]} adds:** {gain_text or 'none'}")
        lines.append(f"- **{ARM_LABELS[arm]} removes:** {loss_text or 'none'}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- These are three season-level origins; eight construction blocks within a season measure seed sensitivity, not eight independent NFL seasons.",
        "- Actual prices make this a hindsight cost replay. Projections and roster choices remain target-outcome blind.",
        "- The 2026 model specification was applied to every origin, so positive results validate the frozen objective more than they validate a historically deployable 2022 method.",
        f"- Frozen rule identifier: `{metadata['frozen_rules_version']}`.",
    ])
    (RESULTS_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace):
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    origins = []
    try:
        # Phase 1: construct and select every roster without target outcomes.
        for year in REPLAY_YEARS:
            print(f"Constructing frozen policies for {year}...")
            origins.append(build_origin(year, args))

        # Phase 2: only now load and score target-season actual weeks.
        actual_frames = []
        coverage = {}
        for origin in origins:
            year = int(origin["year"])
            print(f"Scoring held-out actual weeks for {year}...")
            actual, year_coverage = score_origin_actual(origin)
            actual_frames.append(actual)
            coverage[str(year)] = year_coverage

        candidate_metrics = pd.concat(
            [origin["candidate_metrics"] for origin in origins],
            ignore_index=True,
        )
        plan_rows = pd.concat(
            [origin["plan_rows"] for origin in origins],
            ignore_index=True,
        )
        cell_rows = pd.concat(
            [origin["cell_rows"] for origin in origins],
            ignore_index=True,
        )
        roster_players = pd.concat(
            [origin["roster_players"] for origin in origins],
            ignore_index=True,
        )
        actual_rows = pd.concat(actual_frames, ignore_index=True)
        merged, annual_summary, season_level = summarize(plan_rows, actual_rows)
        frequencies = player_frequencies(roster_players, args.blocks)
        metadata = {
            "years": list(REPLAY_YEARS),
            "league": base.LEAGUE,
            "salary_source": "actual",
            "frozen_rules_version": FROZEN_RULES_VERSION,
            "frozen_objective_rules": FROZEN_OBJECTIVE_RULES,
            "target_outcomes_loaded_after_all_policy_selection": True,
            "model_spec_asof_year": 2026,
            "blocks_per_year": args.blocks,
            "construction_contexts_per_block": args.construction_contexts,
            "candidate_start_budget": args.candidate_starts,
            "validation_contexts_per_block": args.validation_contexts,
            "seed": args.seed,
            "origin_seeds": {
                str(origin["year"]): origin["seed"] for origin in origins
            },
            "simulation_databases": {
                str(origin["year"]): str(origin["database"])
                for origin in origins
            },
            "actual_weekly_coverage": coverage,
            "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "production_changed": False,
        }
        candidate_metrics.to_csv(
            RESULTS_DIR / "candidate_rosters.csv",
            index=False,
        )
        merged.to_csv(RESULTS_DIR / "annual_block_plans.csv", index=False)
        cell_rows.to_csv(RESULTS_DIR / "holdout_utility_cells.csv", index=False)
        roster_players.to_csv(RESULTS_DIR / "roster_players.csv", index=False)
        frequencies.to_csv(RESULTS_DIR / "player_frequency.csv", index=False)
        actual_rows.to_csv(RESULTS_DIR / "actual_outcomes.csv", index=False)
        annual_summary.to_csv(RESULTS_DIR / "annual_summary.csv", index=False)
        season_level.to_csv(RESULTS_DIR / "season_level_deltas.csv", index=False)
        (RESULTS_DIR / "metadata.json").write_text(
            json.dumps(base.json_value(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_summary(annual_summary, season_level, frequencies, metadata)
        print("\nANNUAL SUMMARY")
        print(annual_summary.to_string(index=False))
        print("\nSEASON-LEVEL DELTAS VS MEAN")
        print(season_level.to_string(index=False))
        print(f"\nRuntime seconds {metadata['runtime_seconds']:.2f}")
        return annual_summary, season_level
    finally:
        for origin in origins:
            origin["state"]["conn"].close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--construction-contexts", type=int, default=64)
    parser.add_argument("--candidate-starts", type=int, default=128)
    parser.add_argument("--validation-contexts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
