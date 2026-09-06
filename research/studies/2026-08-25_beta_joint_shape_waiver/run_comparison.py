"""Paired beta construction comparison for joint, shape, and waiver arms."""

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
SHARED_STUDY_DIR = (
    ROOT / "research" / "studies" / "2026-08-24_sequential_shared_opportunity"
)
RESULTS_DIR = STUDY_DIR / "results"
for import_path in (APP_DIR, SHARED_STUDY_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import zSequential_Target as sequential  # noqa: E402
from zSim_Helper import FootballSimulation  # noqa: E402
from keeper_market import load_active_keeper_market  # noqa: E402


YEAR = 2026
LEAGUE = "beta"
PRED_VERSION = "final_ensemble"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
CURRENT_MIN = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
CURRENT_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 2}
FIXED_SHAPE_MIN = {"QB": 1, "RB": 5, "WR": 5, "TE": 1}
FIXED_SHAPE_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 1}
REQUIRE_TOP_N = 12
FIXED_SALARIES = {"Chase Brown": 34.0, "Bhayshul Tuten": 11.0}
WAIVER_BUMP = 1.5
ARMS = {
    "current_additive": {
        "joint_swaps": 0,
        "position_min": CURRENT_MIN,
        "position_max": CURRENT_MAX,
        "waiver_mode": "current",
    },
    "joint_one_swap": {
        "joint_swaps": 1,
        "position_min": CURRENT_MIN,
        "position_max": CURRENT_MAX,
        "waiver_mode": "current",
    },
    "fixed_shape_additive": {
        "joint_swaps": 0,
        "position_min": FIXED_SHAPE_MIN,
        "position_max": FIXED_SHAPE_MAX,
        "waiver_mode": "current",
    },
    "waiver_plus_1_5_additive": {
        "joint_swaps": 0,
        "position_min": CURRENT_MIN,
        "position_max": CURRENT_MAX,
        "waiver_mode": "plus_1_5",
    },
}


def json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def prepare_state(*, block_count, construction_contexts, random_seed):
    conn = sqlite3.connect(APP_DB)
    sim = FootballSimulation(
        conn,
        YEAR,
        LINEUP_REQUIRE,
        SALARY_CAP,
        PRED_VERSION,
        LEAGUE,
        sal_pred_actual="pred",
    )
    sim.load_weekly_template_profiles()
    keeper_market = load_active_keeper_market(
        conn,
        sim,
        year=YEAR,
        league=LEAGUE,
        salary_source="predicted",
        owned_salary_map=FIXED_SALARIES,
    )
    unavailable_keepers = set(keeper_market["unavailable_keeper_players"])
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
    before_keeper_filter = len(predictions)
    predictions = predictions.loc[
        ~predictions.player.isin(unavailable_keepers)
    ].reset_index(drop=True)
    pool_summary = {
        **pool_summary,
        "draft_pool_players_before_keeper_filter": int(before_keeper_filter),
        "active_keeper_players_excluded": int(
            before_keeper_filter - len(predictions)
        ),
        "draft_pool_players_after_keeper_filter": int(len(predictions)),
    }
    missing = sorted(set(FIXED_SALARIES) - set(predictions.player))
    if missing:
        raise ValueError("Fixed keepers missing from beta pool: " + ", ".join(missing))

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
        enabled=True,
    )
    current_waivers = sim.estimate_waiver_baselines(
        num_teams=NUM_TEAMS,
        roster_size=ROSTER_SIZE,
    )
    raised_waivers = {
        pos: float(value) + WAIVER_BUMP
        for pos, value in current_waivers.items()
    }
    construction_started = time.perf_counter()
    current_blocks, construction_banks = (
        sequential._sample_construction_value_blocks(
            sim,
            canonical_predictions,
            predictions,
            list(FIXED_SALARIES),
            block_count=block_count,
            contexts_per_block=construction_contexts,
            num_weeks=16,
            waiver_baselines=current_waivers,
            lineup_require=LINEUP_REQUIRE,
            learn_weeks=6,
            max_learn_weight=0.65,
            random_seed=random_seed + 100,
            return_contexts=True,
        )
    )
    raised_blocks = []
    for bank in construction_banks:
        raised_blocks.append(
            sim.managed_marginal_values_multi_context_batch(
                bank["weekly_scores"],
                predictions.pos.to_numpy(),
                bank["decision_scores"],
                predictions.player.to_numpy(),
                [list(FIXED_SALARIES)],
                waiver_baselines=raised_waivers,
                lineup_require=LINEUP_REQUIRE,
                played_mask=bank["played_mask"],
            )[0]
        )
    raised_blocks = np.stack(raised_blocks)
    construction_seconds = time.perf_counter() - construction_started
    return {
        "conn": conn,
        "sim": sim,
        "canonical_predictions": canonical_predictions,
        "predictions": predictions,
        "base_prices": base_prices,
        "selection_premiums": selection_premiums,
        "current_waivers": current_waivers,
        "raised_waivers": raised_waivers,
        "current_blocks": current_blocks,
        "raised_blocks": raised_blocks,
        "construction_banks": construction_banks,
        "keeper_market": keeper_market,
        "unavailable_keepers": unavailable_keepers,
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "pool_summary": pool_summary,
        "construction_seconds": construction_seconds,
    }


def compile_arm(state, block_idx, arm_name, static_cache):
    config = ARMS[arm_name]
    predictions = state["predictions"]
    waiver_baselines = (
        state["raised_waivers"]
        if config["waiver_mode"] == "plus_1_5"
        else state["current_waivers"]
    )
    managed_blocks = (
        state["raised_blocks"]
        if config["waiver_mode"] == "plus_1_5"
        else state["current_blocks"]
    )
    return sequential.solve_history_only_plan(
        state["sim"],
        predictions,
        managed_blocks[block_idx],
        state["base_prices"],
        state["selection_premiums"],
        FIXED_SALARIES,
        set(predictions.player) - set(FIXED_SALARIES),
        ROSTER_SIZE,
        config["position_min"],
        config["position_max"],
        REQUIRE_TOP_N,
        True,
        static_matrix_cache=static_cache,
        construction_bank=(
            state["construction_banks"][block_idx]
            if config["joint_swaps"]
            else None
        ),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=waiver_baselines,
        joint_refinement_max_swaps=config["joint_swaps"],
        joint_refinement_mode="full_exact",
    )


def roster_counts(predictions, roster):
    position_map = predictions.set_index("player").pos.to_dict()
    return {
        pos: sum(position_map[player] == pos for player in roster)
        for pos in ("QB", "RB", "WR", "TE")
    }


def run_comparison(
    state,
    *,
    block_count,
    validation_contexts,
    timing_repeats,
    random_seed,
):
    rows = []
    roster_rows = []
    static_cache = {}
    validation_seeds = np.random.SeedSequence(random_seed + 200).spawn(block_count)
    predictions = state["predictions"]
    for block_idx in range(block_count):
        validation_seed = int(
            validation_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
        )
        validation_bank = sequential._sample_validation_bank(
            state["sim"],
            predictions,
            validation_contexts,
            16,
            6,
            0.65,
            validation_seed,
            canonical_predictions=state["canonical_predictions"],
        )
        block_plans = {}
        for arm_name in ARMS:
            timings = []
            plan = None
            for _ in range(timing_repeats):
                started = time.perf_counter()
                plan = compile_arm(state, block_idx, arm_name, static_cache)
                timings.append(time.perf_counter() - started)
            if plan is None:
                raise RuntimeError(f"Infeasible beta plan for {arm_name} block {block_idx}.")
            roster = sorted(plan["selected"])
            keeper_hits = sorted(set(roster) & state["unavailable_keepers"])
            if keeper_hits:
                raise AssertionError(
                    "Plan selected unavailable beta keepers: " + ", ".join(keeper_hits)
                )
            counts = roster_counts(predictions, roster)
            config = ARMS[arm_name]
            if not all(
                int(config["position_min"][pos])
                <= counts[pos]
                <= int(config["position_max"][pos])
                for pos in counts
            ):
                raise AssertionError(f"Illegal position counts for {arm_name}: {counts}")
            scores = sequential._score_roster_bank(
                state["sim"],
                predictions,
                roster,
                *validation_bank,
                LINEUP_REQUIRE,
                state["current_waivers"],
                {},
            )
            own_waiver_scores = sequential._score_roster_bank(
                state["sim"],
                predictions,
                roster,
                *validation_bank,
                LINEUP_REQUIRE,
                (
                    state["raised_waivers"]
                    if config["waiver_mode"] == "plus_1_5"
                    else state["current_waivers"]
                ),
                {},
            )
            refinement = plan["joint_refinement"]
            forecast_spend = sum(plan["forecast_cost"][player] for player in roster)
            rows.append({
                "block": block_idx,
                "arm": arm_name,
                "compile_seconds": float(np.median(timings)),
                "holdout_mean_common_waiver": float(np.mean(scores)),
                "holdout_p10_common_waiver": float(np.percentile(scores, 10)),
                "holdout_mean_own_waiver": float(np.mean(own_waiver_scores)),
                "holdout_p10_own_waiver": float(np.percentile(own_waiver_scores, 10)),
                "forecast_spend": float(forecast_spend),
                **{f"count_{pos.lower()}": counts[pos] for pos in counts},
                "accepted_swaps": int(refinement.get("accepted_swaps", 0)),
                "construction_gain": float(refinement.get("improvement", 0.0)),
                "swaps": json.dumps(json_value(refinement.get("swaps", []))),
                "roster": " | ".join(roster),
            })
            block_plans[arm_name] = set(roster)
            for player in roster:
                roster_rows.append({
                    "block": block_idx,
                    "arm": arm_name,
                    "player": player,
                    "pos": predictions.set_index("player").loc[player, "pos"],
                    "forecast_cost": plan["forecast_cost"][player],
                })
        baseline = block_plans["current_additive"]
        for arm_name, roster in block_plans.items():
            if arm_name == "current_additive":
                continue
            outgoing = sorted(baseline - roster)
            incoming = sorted(roster - baseline)
            if len(outgoing) != len(incoming):
                raise AssertionError("Paired roster differences must preserve roster size.")
            for row in rows:
                if row["block"] == block_idx and row["arm"] == arm_name:
                    row["out_vs_current"] = " | ".join(outgoing)
                    row["in_vs_current"] = " | ".join(incoming)
                    break
    return pd.DataFrame(rows), pd.DataFrame(roster_rows)


def summarize(plan_rows, roster_rows):
    summary = (
        plan_rows.groupby("arm", as_index=False)
        .agg(
            blocks=("block", "nunique"),
            compile_seconds=("compile_seconds", "mean"),
            holdout_mean=("holdout_mean_common_waiver", "mean"),
            holdout_p10=("holdout_p10_common_waiver", "mean"),
            own_waiver_mean=("holdout_mean_own_waiver", "mean"),
            own_waiver_p10=("holdout_p10_own_waiver", "mean"),
            forecast_spend=("forecast_spend", "mean"),
            qb=("count_qb", "mean"),
            rb=("count_rb", "mean"),
            wr=("count_wr", "mean"),
            te=("count_te", "mean"),
            accepted_swaps=("accepted_swaps", "mean"),
            construction_gain=("construction_gain", "mean"),
        )
    )
    baseline = summary.loc[
        summary.arm.eq("current_additive"),
        ["holdout_mean", "holdout_p10"],
    ].iloc[0]
    summary["mean_delta_vs_current"] = summary.holdout_mean - baseline.holdout_mean
    summary["p10_delta_vs_current"] = summary.holdout_p10 - baseline.holdout_p10
    frequencies = (
        roster_rows.groupby(["arm", "player", "pos"], as_index=False)
        .agg(blocks_selected=("block", "nunique"))
    )
    frequencies["selection_rate"] = frequencies.blocks_selected / plan_rows.block.nunique()
    return summary, frequencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--construction-contexts", type=int, default=32)
    parser.add_argument("--validation-contexts", type=int, default=64)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = prepare_state(
        block_count=args.blocks,
        construction_contexts=args.construction_contexts,
        random_seed=args.seed,
    )
    try:
        plan_rows, roster_rows = run_comparison(
            state,
            block_count=args.blocks,
            validation_contexts=args.validation_contexts,
            timing_repeats=args.timing_repeats,
            random_seed=args.seed,
        )
        summary, frequencies = summarize(plan_rows, roster_rows)
        plan_rows.to_csv(RESULTS_DIR / "plan_blocks.csv", index=False)
        roster_rows.to_csv(RESULTS_DIR / "roster_players.csv", index=False)
        summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
        frequencies.to_csv(RESULTS_DIR / "player_frequency.csv", index=False)
        metadata = {
            "year": YEAR,
            "league": LEAGUE,
            "fixed_salaries": FIXED_SALARIES,
            "lineup_require": LINEUP_REQUIRE,
            "current_position_min": CURRENT_MIN,
            "current_position_max": CURRENT_MAX,
            "fixed_shape_min": FIXED_SHAPE_MIN,
            "fixed_shape_max": FIXED_SHAPE_MAX,
            "current_waivers": state["current_waivers"],
            "raised_waivers": state["raised_waivers"],
            "keeper_market": state["keeper_market"],
            "remaining_market_budget": state["remaining_market_budget"],
            "remaining_market_slots": state["remaining_market_slots"],
            "pool_summary": state["pool_summary"],
            "construction_seconds": state["construction_seconds"],
            "blocks": args.blocks,
            "construction_contexts_per_block": args.construction_contexts,
            "validation_contexts_per_block": args.validation_contexts,
            "timing_repeats": args.timing_repeats,
            "seed": args.seed,
        }
        (RESULTS_DIR / "metadata.json").write_text(
            json.dumps(json_value(metadata), indent=2),
            encoding="utf-8",
        )
        print("\nSUMMARY")
        print(summary.to_string(index=False))
        print("\nBLOCK PLANS")
        print(plan_rows[[
            "block",
            "arm",
            "holdout_mean_common_waiver",
            "holdout_p10_common_waiver",
            "forecast_spend",
            "count_qb",
            "count_rb",
            "count_wr",
            "count_te",
            "out_vs_current",
            "in_vs_current",
            "swaps",
        ]].to_string(index=False))
    finally:
        state["conn"].close()


if __name__ == "__main__":
    main()
