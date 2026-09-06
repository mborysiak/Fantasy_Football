"""Paired validation for the Sequential shared-opportunity correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_DIR = ROOT.parent / "Fantasy_Football_App" / "app"
APP_DB = APP_DIR / "Simulation.sqlite3"
RESULTS_DIR = STUDY_DIR / "results"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import zSequential_Target as sequential  # noqa: E402
from zSim_Helper import FootballSimulation  # noqa: E402
from keeper_market import load_active_keeper_market  # noqa: E402


YEAR = 2026
LEAGUE = "nv"
PRED_VERSION = "final_ensemble"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP_REQUIRE = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
POS_MIN = {"QB": 2, "RB": 4, "WR": 4, "TE": 1}
POS_MAX = {"QB": 3, "RB": 6, "WR": 6, "TE": 2}
REQUIRE_TOP_N = 12
FIXED_SALARIES = {"Drake Maye": 18.0, "De'Von Achane": 47.0}
ARMS = {
    "baseline": {"max_swaps": 0, "mode": "disabled"},
    "shortlist": {"max_swaps": 1, "mode": "shortlist"},
    "full_exact": {"max_swaps": 1, "mode": "full_exact"},
}
SOURCE_SUFFIX = {"predicted": "pred", "actual": "_actual"}


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def prepare_source(
    source: str,
    *,
    block_count: int,
    construction_contexts: int,
    random_seed: int,
):
    conn = sqlite3.connect(APP_DB)
    sim = FootballSimulation(
        conn,
        YEAR,
        LINEUP_REQUIRE,
        SALARY_CAP,
        PRED_VERSION,
        LEAGUE,
        sal_pred_actual=SOURCE_SUFFIX[source],
    )
    sim.load_weekly_template_profiles()
    keeper_market = load_active_keeper_market(
        conn,
        sim,
        year=YEAR,
        league=LEAGUE,
        salary_source=source,
        owned_salary_map=FIXED_SALARIES,
    )
    unavailable_keepers = set(
        keeper_market["unavailable_keeper_players"]
    )
    with sim.temp_seed(random_seed):
        canonical_predictions = sim.get_predictions(
            "pred_fp_per_game",
            num_options=512,
        )
    predictions, pool_summary = sequential.apply_sequential_draft_pool_filter(
        canonical_predictions.copy(),
        sequential._sequential_draft_pool_metadata(sim),
        LEAGUE,
        required_players=set(FIXED_SALARIES),
    )
    players_before_keeper_filter = len(predictions)
    predictions = predictions.loc[
        ~predictions.player.isin(unavailable_keepers)
    ].reset_index(drop=True)
    keeper_players_excluded = players_before_keeper_filter - len(predictions)
    pool_summary = {
        **pool_summary,
        "draft_pool_players_before_keeper_filter": int(
            players_before_keeper_filter
        ),
        "active_keeper_players_excluded": int(keeper_players_excluded),
        "draft_pool_players_after_keeper_filter": int(len(predictions)),
    }
    missing = sorted(set(FIXED_SALARIES) - set(predictions.player))
    if missing:
        raise ValueError("Fixed players missing from replay: " + ", ".join(missing))

    state_indices = sequential._canonical_state_indices(
        canonical_predictions,
        predictions,
    )
    canonical_aligned = sequential._aligned_player_frame(
        sim,
        canonical_predictions,
    )
    aligned = sequential._aligned_player_frame(sim, predictions)
    canonical_market_prices = canonical_aligned.salary.to_numpy(dtype=np.float64)
    market_prices = aligned.salary.to_numpy(dtype=np.float64)
    remaining_market_budget = (
        NUM_TEAMS * SALARY_CAP - keeper_market["keeper_spend"]
    )
    remaining_market_slots = (
        NUM_TEAMS * ROSTER_SIZE - keeper_market["keeper_count"]
    )
    canonical_available_mask = (
        np.isin(
            canonical_predictions.player.to_numpy(),
            predictions.player.to_numpy(),
        )
        & ~canonical_predictions.player.isin(
            keeper_market["keeper_players"]
        ).to_numpy()
    )
    canonical_base_prices = sim.normalize_salary_market_values(
        canonical_market_prices,
        canonical_available_mask,
        remaining_market_budget=remaining_market_budget,
        remaining_market_slots=remaining_market_slots,
    )
    base_prices = canonical_base_prices[state_indices]
    predictions["salary"] = market_prices
    selection_premiums = sim.get_selection_premium_values(
        predictions.player.to_numpy(),
        fixed_players=list(FIXED_SALARIES),
        enabled=(source == "predicted"),
    )
    waiver_baselines = sim.estimate_waiver_baselines(
        num_teams=NUM_TEAMS,
        roster_size=ROSTER_SIZE,
    )
    construction_started = time.perf_counter()
    managed_blocks, construction_banks = (
        sequential._sample_construction_value_blocks(
            sim,
            canonical_predictions,
            predictions,
            list(FIXED_SALARIES),
            block_count=block_count,
            contexts_per_block=construction_contexts,
            num_weeks=16,
            waiver_baselines=waiver_baselines,
            lineup_require=LINEUP_REQUIRE,
            learn_weeks=6,
            max_learn_weight=0.65,
            random_seed=random_seed + 100,
            return_contexts=True,
        )
    )
    construction_seconds = time.perf_counter() - construction_started
    return {
        "conn": conn,
        "sim": sim,
        "canonical_predictions": canonical_predictions,
        "predictions": predictions,
        "base_prices": base_prices,
        "selection_premiums": selection_premiums,
        "waiver_baselines": waiver_baselines,
        "managed_blocks": managed_blocks,
        "construction_banks": construction_banks,
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "keeper_market": keeper_market,
        "unavailable_keeper_players": unavailable_keepers,
        "pool_summary": pool_summary,
        "construction_seconds": construction_seconds,
    }


def compile_plan(
    state,
    block_idx: int,
    max_swaps: int,
    enforce_top_n: bool,
    static_cache: dict,
):
    predictions = state["predictions"]
    return sequential.solve_history_only_plan(
        state["sim"],
        predictions,
        state["managed_blocks"][block_idx],
        state["base_prices"],
        state["selection_premiums"],
        FIXED_SALARIES,
        set(predictions.player) - set(FIXED_SALARIES),
        ROSTER_SIZE,
        POS_MIN,
        POS_MAX,
        REQUIRE_TOP_N,
        enforce_top_n,
        static_matrix_cache=static_cache,
        construction_bank=(
            None
            if max_swaps == 0
            else state["construction_banks"][block_idx]
        ),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=state["waiver_baselines"],
        joint_refinement_max_swaps=max_swaps,
    )


def plan_is_legal(state, plan, enforce_top_n: bool):
    predictions = state["predictions"]
    selected = set(plan["selected"])
    positions = predictions.set_index("player").pos.to_dict()
    counts = {
        pos: sum(positions[player] == pos for player in selected)
        for pos in ("QB", "RB", "WR", "TE")
    }
    cap_legal = sum(
        plan["forecast_cost"][player] for player in selected
    ) <= SALARY_CAP + 1e-8
    position_legal = all(
        POS_MIN[pos] <= counts[pos] <= POS_MAX[pos]
        for pos in counts
    )
    top_players = set(
        pd.DataFrame({
            "player": predictions.player,
            "price": state["base_prices"],
        }).sort_values(
            ["price", "player"],
            ascending=[False, True],
            kind="mergesort",
        ).head(REQUIRE_TOP_N).player
    )
    top_n_legal = (not enforce_top_n) or bool(selected & top_players)
    return cap_legal, position_legal, top_n_legal


def refine_compiled_plan(
    state,
    block_idx,
    baseline_plan,
    max_swaps,
    enforce_top_n,
    mode,
):
    predictions = state["predictions"]
    players = predictions.player.to_numpy()
    selected_mask = np.isin(players, list(baseline_plan["selected"]))
    forecast_costs = np.asarray([
        baseline_plan["forecast_cost"][player] for player in players
    ], dtype=np.float64)
    ordered_top = (
        pd.DataFrame({"player": players, "price": state["base_prices"]})
        .sort_values(
            ["price", "player"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(REQUIRE_TOP_N)
        .player.tolist()
    )
    refinement_kwargs = {}
    if mode == "shortlist":
        refinement_kwargs = {
            "shortlist_outgoing_size": (
                sequential.DEFAULT_SEQUENTIAL_JOINT_SHORTLIST_OUTGOING
            ),
            "shortlist_incoming_size": (
                sequential.DEFAULT_SEQUENTIAL_JOINT_SHORTLIST_INCOMING
            ),
        }
    selected_mask, refinement = (
        state["sim"].refine_managed_roster_bank_to_convergence(
            predictions,
            selected_mask,
            state["construction_banks"][block_idx]["weekly_scores"],
            state["construction_banks"][block_idx]["decision_scores"],
            played_mask=state["construction_banks"][block_idx]["played_mask"],
            max_swaps=max_swaps,
            fixed_players=set(FIXED_SALARIES),
            fixed_salary_map=FIXED_SALARIES,
            waiver_baselines=state["waiver_baselines"],
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts=POS_MIN,
            pos_max_counts=POS_MAX,
            top_n=ordered_top,
            enforce_top_n=enforce_top_n,
            salary_values=forecast_costs,
            **refinement_kwargs,
        )
    )
    selected = set(predictions.loc[selected_mask, "player"])
    return {
        **baseline_plan,
        "selected": selected,
        "targets": selected - set(FIXED_SALARIES),
        "joint_refinement": refinement,
    }


def validate_source(
    source: str,
    state,
    *,
    block_count: int,
    path_count: int,
    validation_contexts: int,
    timing_repeats: int,
    random_seed: int,
    block_indices=None,
):
    sim = state["sim"]
    predictions = state["predictions"]
    plan_rows = []
    rollout_rows = []
    player_rows = []
    path_rows = []
    static_cache = {}
    validation_seeds = np.random.SeedSequence(random_seed + 200).spawn(block_count)
    tape_seeds = np.random.SeedSequence(random_seed + 300).spawn(block_count)

    selected_blocks = (
        range(block_count) if block_indices is None else block_indices
    )
    for enforce_top_n in (False, True):
        for block_idx in selected_blocks:
            print(
                f"[{source}] top_n={enforce_top_n} block={block_idx} prepare",
                flush=True,
            )
            baseline_started = time.perf_counter()
            baseline_plan = compile_plan(
                state,
                block_idx,
                0,
                enforce_top_n,
                static_cache,
            )
            baseline_seconds = time.perf_counter() - baseline_started
            if baseline_plan is None:
                raise RuntimeError("Baseline completion plan is infeasible.")
            validation_seed = int(
                validation_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
            ) + int(enforce_top_n) * 10_000
            validation_bank = sequential._sample_validation_bank(
                sim,
                predictions,
                validation_contexts,
                16,
                6,
                0.65,
                validation_seed,
                canonical_predictions=state["canonical_predictions"],
            )
            plans = {}
            for arm, arm_config in ARMS.items():
                max_swaps = arm_config["max_swaps"]
                mode = arm_config["mode"]
                print(
                    f"[{source}] top_n={enforce_top_n} block={block_idx} "
                    f"plan={arm}",
                    flush=True,
                )
                timings = []
                if max_swaps == 0:
                    plan = baseline_plan
                    timings = [baseline_seconds]
                else:
                    for _ in range(timing_repeats):
                        started = time.perf_counter()
                        plan = refine_compiled_plan(
                            state,
                            block_idx,
                            baseline_plan,
                            max_swaps,
                            enforce_top_n,
                            mode,
                        )
                        timings.append(
                            baseline_seconds + time.perf_counter() - started
                        )
                plans[arm] = plan
                roster = sorted(plan["selected"])
                keeper_hits = sorted(
                    set(roster) & state["unavailable_keeper_players"]
                )
                if keeper_hits:
                    raise AssertionError(
                        "Completion plan selected unavailable NV keepers: "
                        + ", ".join(keeper_hits)
                    )
                score_started = time.perf_counter()
                scores = sequential._score_roster_bank(
                    sim,
                    predictions,
                    roster,
                    *validation_bank,
                    LINEUP_REQUIRE,
                    state["waiver_baselines"],
                    {},
                )
                score_seconds = time.perf_counter() - score_started
                cap_legal, position_legal, top_n_legal = plan_is_legal(
                    state,
                    plan,
                    enforce_top_n,
                )
                refinement = plan["joint_refinement"]
                plan_rows.append({
                    "source": source,
                    "enforce_top_n": enforce_top_n,
                    "block": block_idx,
                    "arm": arm,
                    "max_swaps": max_swaps,
                    "refinement_mode": mode,
                    "accepted_swaps": refinement["accepted_swaps"],
                    "construction_gain": refinement["improvement"],
                    "compile_seconds_median": float(np.median(timings)),
                    "compile_seconds_min": float(np.min(timings)),
                    "score_seconds": score_seconds,
                    "holdout_mean": float(np.mean(scores)),
                    "holdout_p10": float(np.percentile(scores, 10)),
                    "forecast_spend": float(sum(
                        plan["forecast_cost"][player] for player in roster
                    )),
                    "cap_legal": cap_legal,
                    "position_legal": position_legal,
                    "top_n_legal": top_n_legal,
                    "swaps": json.dumps(json_value(refinement["swaps"])),
                    "roster": " | ".join(roster),
                })

            if not enforce_top_n:
                continue
            tape_seed = int(
                tape_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
            )
            tapes = sequential.generate_hidden_auction_tapes(
                sim,
                predictions,
                FIXED_SALARIES,
                path_count,
                state["remaining_market_budget"],
                state["remaining_market_slots"],
                tape_seed,
                canonical_predictions=state["canonical_predictions"],
            )
            evaluation_context = sequential._build_sequential_evaluation_context(
                sim,
                predictions,
                state["managed_blocks"][block_idx],
                state["base_prices"],
                state["selection_premiums"],
                FIXED_SALARIES,
            )
            for arm, plan in plans.items():
                print(
                    f"[{source}] block={block_idx} rollout={arm}",
                    flush=True,
                )
                branch_results = []
                started = time.perf_counter()
                for path_idx in range(path_count):
                    branch_results.append(sequential.simulate_history_only_branch(
                        sim=sim,
                        predictions=predictions,
                        managed_values=state["managed_blocks"][block_idx],
                        base_prices=state["base_prices"],
                        selection_premiums=state["selection_premiums"],
                        initial_salary_map=FIXED_SALARIES,
                        candidate=None,
                        candidate_price=None,
                        force_buy=False,
                        order=tapes["orders"][path_idx],
                        revealed_prices=tapes["prices"][path_idx],
                        remaining_market_budget=state["remaining_market_budget"],
                        remaining_market_slots=state["remaining_market_slots"],
                        roster_size=ROSTER_SIZE,
                        pos_min_counts=POS_MIN,
                        pos_max_counts=POS_MAX,
                        require_top_n=REQUIRE_TOP_N,
                        enforce_top_n=True,
                        compiled_plan=plan,
                        policy_scores=evaluation_context["policy_scores"],
                        rollout_context=evaluation_context,
                    ))
                rollout_seconds = time.perf_counter() - started
                score_cache = {}
                score_started = time.perf_counter()
                complete_scores = []
                for path_idx, branch in enumerate(branch_results):
                    keeper_hits = sorted(
                        set(branch["roster"])
                        & state["unavailable_keeper_players"]
                    )
                    if keeper_hits:
                        raise AssertionError(
                            "Sequential rollout selected unavailable NV keepers: "
                            + ", ".join(keeper_hits)
                        )
                    path_rows.append({
                        "source": source,
                        "block": block_idx,
                        "path": path_idx,
                        "arm": arm,
                        "complete": branch["complete"],
                        "failure_reason": branch.get("failure_reason"),
                        "salary_spend": branch.get("salary_spend", np.nan),
                        "cap_legal": branch.get("final_cap_legal", False),
                        "position_legal": branch.get(
                            "final_position_legal",
                            False,
                        ),
                        "top_n_legal": branch.get("final_top_n_legal", False),
                        "roster": " | ".join(branch["roster"]),
                    })
                    if branch["complete"]:
                        season_scores = sequential._score_roster_bank(
                            sim,
                            predictions,
                            branch["roster"],
                            *validation_bank,
                            LINEUP_REQUIRE,
                            state["waiver_baselines"],
                            score_cache,
                            evaluation_context=evaluation_context,
                        )
                        complete_scores.append(season_scores)
                    for player in branch["roster"]:
                        player_rows.append({
                            "source": source,
                            "block": block_idx,
                            "path": path_idx,
                            "arm": arm,
                            "complete": branch["complete"],
                            "player": player,
                        })
                scoring_seconds = time.perf_counter() - score_started
                complete_scores = (
                    np.concatenate(complete_scores)
                    if complete_scores
                    else np.array([], dtype=np.float64)
                )
                rollout_rows.append({
                    "source": source,
                    "block": block_idx,
                    "arm": arm,
                    "paths": path_count,
                    "completion_rate": float(np.mean([
                        branch["complete"] for branch in branch_results
                    ])),
                    "cap_legal_rate": float(np.mean([
                        branch.get("final_cap_legal", False)
                        for branch in branch_results
                    ])),
                    "position_legal_rate": float(np.mean([
                        branch.get("final_position_legal", False)
                        for branch in branch_results
                    ])),
                    "top_n_legal_rate": float(np.mean([
                        branch.get("final_top_n_legal", False)
                        for branch in branch_results
                    ])),
                    "rollout_seconds": rollout_seconds,
                    "scoring_seconds": scoring_seconds,
                    "unique_rosters": len(score_cache),
                    "mean_spend": float(np.mean([
                        branch.get("salary_spend", np.nan)
                        for branch in branch_results
                        if branch["complete"]
                    ])),
                    "validation_mean": (
                        float(np.mean(complete_scores))
                        if len(complete_scores) else np.nan
                    ),
                    "validation_p10": (
                        float(np.percentile(complete_scores, 10))
                        if len(complete_scores) else np.nan
                    ),
                    "rosters": json.dumps([
                        list(branch["roster"]) for branch in branch_results
                    ]),
                })
    return (
        pd.DataFrame(plan_rows),
        pd.DataFrame(rollout_rows),
        pd.DataFrame(player_rows),
        pd.DataFrame(path_rows),
    )


def summarize(plan_rows, rollout_rows, player_rows, path_rows, metadata):
    plan_summary = (
        plan_rows.groupby(["source", "enforce_top_n", "arm"], as_index=False)
        .agg(
            blocks=("block", "nunique"),
            compile_seconds=("compile_seconds_median", "mean"),
            accepted_swaps=("accepted_swaps", "mean"),
            construction_gain=("construction_gain", "mean"),
            holdout_mean=("holdout_mean", "mean"),
            holdout_p10=("holdout_p10", "mean"),
            forecast_spend=("forecast_spend", "mean"),
            legal=("cap_legal", "all"),
        )
    )
    rollout_summary = (
        rollout_rows.groupby(["source", "arm"], as_index=False)
        .agg(
            blocks=("block", "nunique"),
            completion_rate=("completion_rate", "mean"),
            cap_legal_rate=("cap_legal_rate", "mean"),
            position_legal_rate=("position_legal_rate", "mean"),
            top_n_legal_rate=("top_n_legal_rate", "mean"),
            rollout_seconds=("rollout_seconds", "sum"),
            scoring_seconds=("scoring_seconds", "sum"),
            validation_mean=("validation_mean", "mean"),
            validation_p10=("validation_p10", "mean"),
            mean_spend=("mean_spend", "mean"),
            unique_rosters=("unique_rosters", "sum"),
        )
    )
    baseline_paths = path_rows.loc[
        path_rows.arm == "baseline",
        ["source", "block", "path", "roster"],
    ].rename(columns={"roster": "baseline_roster"})
    path_comparison = path_rows.merge(
        baseline_paths,
        on=["source", "block", "path"],
        how="left",
        validate="many_to_one",
    )
    path_comparison["roster_changed"] = (
        path_comparison.roster != path_comparison.baseline_roster
    )
    change_summary = (
        path_comparison.groupby(["source", "arm"], as_index=False)
        .roster_changed.mean()
        .rename(columns={"roster_changed": "roster_change_rate"})
    )
    rollout_summary = rollout_summary.merge(
        change_summary,
        on=["source", "arm"],
        how="left",
        validate="one_to_one",
    )
    player_frequency = (
        player_rows.groupby(["source", "arm", "player"], as_index=False)
        .agg(selections=("path", "size"))
    )
    totals = (
        rollout_rows.groupby(["source", "arm"])
        .paths.sum()
        .to_dict()
    )
    player_frequency["selection_rate"] = player_frequency.apply(
        lambda row: row.selections / totals[(row.source, row.arm)],
        axis=1,
    )
    plan_summary.to_csv(RESULTS_DIR / "plan_summary.csv", index=False)
    rollout_summary.to_csv(RESULTS_DIR / "rollout_summary.csv", index=False)
    player_frequency.to_csv(RESULTS_DIR / "player_frequency.csv", index=False)
    path_comparison.to_csv(
        RESULTS_DIR / "rollout_path_comparison.csv",
        index=False,
    )
    (RESULTS_DIR / "metadata.json").write_text(
        json.dumps(json_value(metadata), indent=2),
        encoding="utf-8",
    )
    return plan_summary, rollout_summary, player_frequency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["predicted"])
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--construction-contexts", type=int, default=32)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--validation-contexts", type=int, default=64)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--block-index", type=int)
    parser.add_argument("--combine-blocks", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    invalid = sorted(set(args.sources) - set(SOURCE_SUFFIX))
    if invalid:
        raise ValueError("Unknown salary sources: " + ", ".join(invalid))
    if args.block_index is not None:
        if len(args.sources) != 1:
            raise ValueError("Isolated block runs require exactly one source.")
        if not 0 <= args.block_index < args.blocks:
            raise ValueError("block-index must be within the configured blocks.")
    if sum((args.aggregate_only, args.combine_blocks)) > 1:
        raise ValueError("Choose only one aggregation mode.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_plan_rows = []
    all_rollout_rows = []
    all_player_rows = []
    all_path_rows = []
    source_metadata = {}
    started = time.perf_counter()
    if args.combine_blocks:
        stems = {
            "plan": "plan_blocks",
            "rollout": "rollout_blocks",
            "player": "rollout_players",
            "path": "rollout_paths",
        }
        for source in args.sources:
            combined = {}
            metadata_parts = []
            for label, stem in stems.items():
                parts = [
                    pd.read_csv(
                        RESULTS_DIR / f"{stem}_{source}_block{block}.csv"
                    )
                    for block in range(args.blocks)
                ]
                combined[label] = pd.concat(parts, ignore_index=True)
                observed = set(combined[label].block.astype(int))
                expected = set(range(args.blocks))
                if observed != expected:
                    raise ValueError(
                        f"Incomplete isolated {source} {label} blocks: "
                        f"{sorted(observed)}"
                    )
                combined[label].to_csv(
                    RESULTS_DIR / f"{stem}_{source}.csv",
                    index=False,
                )
            for block in range(args.blocks):
                metadata_parts.append(json.loads(
                    (
                        RESULTS_DIR
                        / f"metadata_{source}_block{block}.json"
                    ).read_text(encoding="utf-8")
                ))
            metadata = dict(metadata_parts[0])
            metadata["blocks"] = args.blocks
            metadata["block_indices"] = list(range(args.blocks))
            metadata["isolated_block_processes"] = True
            metadata["construction_seconds_per_process"] = [
                part["construction_seconds"] for part in metadata_parts
            ]
            metadata["construction_seconds"] = float(np.mean(
                metadata["construction_seconds_per_process"]
            ))
            (RESULTS_DIR / f"metadata_{source}.json").write_text(
                json.dumps(json_value(metadata), indent=2),
                encoding="utf-8",
            )
            print(
                f"Combined {args.blocks} isolated {source} blocks.",
                flush=True,
            )
        return
    if args.aggregate_only:
        for source in SOURCE_SUFFIX:
            all_plan_rows.append(pd.read_csv(
                RESULTS_DIR / f"plan_blocks_{source}.csv"
            ))
            all_rollout_rows.append(pd.read_csv(
                RESULTS_DIR / f"rollout_blocks_{source}.csv"
            ))
            all_player_rows.append(pd.read_csv(
                RESULTS_DIR / f"rollout_players_{source}.csv"
            ))
            all_path_rows.append(pd.read_csv(
                RESULTS_DIR / f"rollout_paths_{source}.csv"
            ))
        plan_rows = pd.concat(all_plan_rows, ignore_index=True)
        rollout_rows = pd.concat(all_rollout_rows, ignore_index=True)
        player_rows = pd.concat(all_player_rows, ignore_index=True)
        path_rows = pd.concat(all_path_rows, ignore_index=True)
        prior_metadata = {}
        for source in SOURCE_SUFFIX:
            source_path = RESULTS_DIR / f"metadata_{source}.json"
            prior_metadata[source] = json.loads(
                source_path.read_text(encoding="utf-8")
            )
        metadata = {
            "year": YEAR,
            "league": LEAGUE,
            "fixed_salaries": FIXED_SALARIES,
            "arms": ARMS,
            "source_runs": prior_metadata,
            "aggregation_elapsed_seconds": time.perf_counter() - started,
        }
        plan_rows.to_csv(RESULTS_DIR / "plan_blocks.csv", index=False)
        rollout_rows.to_csv(RESULTS_DIR / "rollout_blocks.csv", index=False)
        player_rows.to_csv(RESULTS_DIR / "rollout_players.csv", index=False)
        path_rows.to_csv(RESULTS_DIR / "rollout_paths.csv", index=False)
        plan_summary, rollout_summary, _ = summarize(
            plan_rows,
            rollout_rows,
            player_rows,
            path_rows,
            metadata,
        )
        print("\nPLAN SUMMARY")
        print(plan_summary.to_string(index=False))
        print("\nROLLOUT SUMMARY")
        print(rollout_summary.to_string(index=False))
        return
    for source_index, source in enumerate(args.sources):
        print(f"Preparing {source} source", flush=True)
        state = prepare_source(
            source,
            block_count=args.blocks,
            construction_contexts=args.construction_contexts,
            random_seed=args.seed + source_index * 100_000,
        )
        try:
            plan_rows, rollout_rows, player_rows, path_rows = validate_source(
                source,
                state,
                block_count=args.blocks,
                path_count=args.paths,
                validation_contexts=args.validation_contexts,
                timing_repeats=args.timing_repeats,
                random_seed=args.seed + source_index * 100_000,
                block_indices=(
                    None
                    if args.block_index is None
                    else [args.block_index]
                ),
            )
            all_plan_rows.append(plan_rows)
            all_rollout_rows.append(rollout_rows)
            all_player_rows.append(player_rows)
            all_path_rows.append(path_rows)
            source_metadata[source] = {
                "pool_summary": state["pool_summary"],
                "keeper_market": state["keeper_market"],
                "remaining_market_budget": state["remaining_market_budget"],
                "remaining_market_slots": state["remaining_market_slots"],
                "waiver_baselines": state["waiver_baselines"],
                "construction_seconds": state["construction_seconds"],
                "state_players": len(state["predictions"]),
            }
            output_suffix = (
                source
                if args.block_index is None
                else f"{source}_block{args.block_index}"
            )
            plan_rows.to_csv(
                RESULTS_DIR / f"plan_blocks_{output_suffix}.csv",
                index=False,
            )
            rollout_rows.to_csv(
                RESULTS_DIR / f"rollout_blocks_{output_suffix}.csv",
                index=False,
            )
            player_rows.to_csv(
                RESULTS_DIR / f"rollout_players_{output_suffix}.csv",
                index=False,
            )
            path_rows.to_csv(
                RESULTS_DIR / f"rollout_paths_{output_suffix}.csv",
                index=False,
            )
            (RESULTS_DIR / f"metadata_{output_suffix}.json").write_text(
                json.dumps(json_value({
                    "source": source,
                    "blocks": args.blocks,
                    "block_indices": (
                        list(range(args.blocks))
                        if args.block_index is None
                        else [args.block_index]
                    ),
                    "construction_contexts_per_block": (
                        args.construction_contexts
                    ),
                    "paths_per_block": args.paths,
                    "validation_contexts_per_block": args.validation_contexts,
                    "timing_repeats": args.timing_repeats,
                    "seed": args.seed + source_index * 100_000,
                    **source_metadata[source],
                }), indent=2),
                encoding="utf-8",
            )
        finally:
            state["conn"].close()
    if args.block_index is not None:
        print(
            f"Completed isolated {args.sources[0]} block "
            f"{args.block_index}.",
            flush=True,
        )
        return
    plan_rows = pd.concat(all_plan_rows, ignore_index=True)
    rollout_rows = pd.concat(all_rollout_rows, ignore_index=True)
    player_rows = pd.concat(all_player_rows, ignore_index=True)
    path_rows = pd.concat(all_path_rows, ignore_index=True)
    plan_rows.to_csv(RESULTS_DIR / "plan_blocks.csv", index=False)
    rollout_rows.to_csv(RESULTS_DIR / "rollout_blocks.csv", index=False)
    player_rows.to_csv(RESULTS_DIR / "rollout_players.csv", index=False)
    path_rows.to_csv(RESULTS_DIR / "rollout_paths.csv", index=False)
    metadata = {
        "year": YEAR,
        "league": LEAGUE,
        "fixed_salaries": FIXED_SALARIES,
        "arms": ARMS,
        "blocks": args.blocks,
        "construction_contexts_per_block": args.construction_contexts,
        "paths_per_block": args.paths,
        "validation_contexts_per_block": args.validation_contexts,
        "timing_repeats": args.timing_repeats,
        "seed": args.seed,
        "sources": source_metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    plan_summary, rollout_summary, _ = summarize(
        plan_rows,
        rollout_rows,
        player_rows,
        path_rows,
        metadata,
    )
    print("\nPLAN SUMMARY")
    print(plan_summary.to_string(index=False))
    print("\nROLLOUT SUMMARY")
    print(rollout_summary.to_string(index=False))
    print(f"\nElapsed: {metadata['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
