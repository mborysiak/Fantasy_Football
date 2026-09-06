"""Paired organic-auction comparison of three 2026 beta second keepers."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_DIR = ROOT.parent / "Fantasy_Football_App" / "app"
APP_DB = APP_DIR / "Simulation.sqlite3"
SHARED_STUDY_DIR = (
    ROOT / "research" / "studies" / "2026-08-24_sequential_shared_opportunity"
)
SALARY_RESULTS = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-26_beta_nonkeeper_salary_counterfactual"
    / "results"
    / "all_salary_deltas.csv"
)
RESULTS_DIR = STUDY_DIR / "results"
for import_path in (APP_DIR, SHARED_STUDY_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import zSequential_Target as sequential  # noqa: E402
from zSim_Helper import SALARY_RESID_COLS, FootballSimulation  # noqa: E402


YEAR = 2026
LEAGUE = "beta"
PRED_VERSION = "final_ensemble"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POSITION_MIN = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
POSITION_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 2}
REQUIRE_TOP_N = 12
REFERENCE_SCENARIO = "tuten"
SCENARIOS = {
    "tuten": {"Chase Brown": 34.0, "Bhayshul Tuten": 11.0},
    "burden": {"Chase Brown": 34.0, "Luther Burden III": 11.0},
    "loveland": {"Chase Brown": 34.0, "Colston Loveland": 11.0},
}
COUNTERFACTUAL_CANDIDATES = {
    "Bhayshul Tuten",
    "Luther Burden III",
    "Colston Loveland",
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


def apply_counterfactual_salary_surface(sim: FootballSimulation) -> dict:
    surface = pd.read_csv(SALARY_RESULTS)
    required = {
        "player_key",
        "player",
        "counterfactual_salary",
        "counterfactual_std_dev",
        "counterfactual_min",
        "counterfactual_max",
        "counterfactual_salary_resid_10",
        "counterfactual_salary_resid_90",
    }
    if not required.issubset(surface.columns):
        raise ValueError(
            "Counterfactual salary output lacks columns: "
            f"{sorted(required - set(surface.columns))}"
        )
    if surface.player_key.duplicated().any():
        raise ValueError("Counterfactual salary surface contains duplicate keys")

    live_keys = set(sim.player_data.player_key.astype(str))
    counterfactual_keys = set(surface.player_key.astype(str))
    if live_keys != counterfactual_keys:
        raise ValueError("Counterfactual salaries lack exact app player-key parity")
    mapped = surface.set_index("player_key")
    key_order = sim.player_data.player_key.astype(str)
    for target_column, source_column in (
        ("salary", "counterfactual_salary"),
        ("salary_std_dev", "counterfactual_std_dev"),
        ("salary_min_score", "counterfactual_min"),
        ("salary_max_score", "counterfactual_max"),
    ):
        sim.player_data[target_column] = key_order.map(mapped[source_column]).to_numpy()
    if sim.player_data[[
        "salary", "salary_std_dev", "salary_min_score", "salary_max_score"
    ]].isna().any().any():
        raise ValueError("Counterfactual salary mapping produced missing values")

    # Tuten's live row is a deterministic keeper override. Reconstruct a
    # monotone six-knot salary residual shape from the counterfactual model's
    # exact min, P10, P90, and max so the non-keeper arms do not treat his
    # clearing price as known. Burden/Loveland already retain non-keeper shapes.
    tuten = surface.loc[surface.player.eq("Bhayshul Tuten")].iloc[0]
    center = float(tuten.counterfactual_salary)
    q5 = float(tuten.counterfactual_min) - center
    q10 = float(tuten.counterfactual_salary_resid_10)
    q90 = float(tuten.counterfactual_salary_resid_90)
    q95 = float(tuten.counterfactual_max) - center
    tuten_residuals = np.maximum.accumulate(
        np.array([q5, q10, 0.5 * q10, 0.5 * q90, q90, q95], dtype=float)
    )
    tuten_mask = sim.player_data.player.eq("Bhayshul Tuten")
    if int(tuten_mask.sum()) != 1:
        raise ValueError("Expected exactly one Bhayshul Tuten row")
    sim.player_data.loc[tuten_mask, SALARY_RESID_COLS] = tuten_residuals

    return {
        row.player: {
            "salary": float(row.counterfactual_salary),
            "std_dev": float(row.counterfactual_std_dev),
            "min": float(row.counterfactual_min),
            "p10": float(
                row.counterfactual_salary
                + row.counterfactual_salary_resid_10
            ),
            "p90": float(
                row.counterfactual_salary
                + row.counterfactual_salary_resid_90
            ),
            "max": float(row.counterfactual_max),
        }
        for row in surface.loc[
            surface.player.isin(COUNTERFACTUAL_CANDIDATES)
        ].itertuples(index=False)
    }


def load_keeper_context(conn, sim):
    keepers = pd.read_sql_query(
        """
        SELECT player_key, player AS source_player, keeper_salary
          FROM League_Keepers
         WHERE year=? AND league=?
        """,
        conn,
        params=(YEAR, LEAGUE),
    )
    canonical = sim.player_data[["player_key", "player"]]
    keepers = keepers.merge(
        canonical,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    if keepers.player.isna().any():
        raise ValueError("Active keeper labels did not resolve to the app population")
    removed = {"Chase Brown", "Bhayshul Tuten"}
    other = keepers.loc[~keepers.player.isin(removed)].copy()
    if len(keepers) != 14 or len(other) != 12:
        raise ValueError(
            f"Expected 14 active and 12 other keepers; found {len(keepers)}/{len(other)}"
        )
    if not np.isclose(other.keeper_salary.sum(), 396.0):
        raise ValueError("Other-keeper spend is not the expected $396")
    return {
        "active_keeper_count": int(len(keepers)),
        "active_keeper_spend": float(keepers.keeper_salary.sum()),
        "other_keeper_count": int(len(other)),
        "other_keeper_spend": float(other.keeper_salary.sum()),
        "unavailable_players": tuple(sorted(other.player)),
    }


def managed_values_from_banks(sim, predictions, fixed_players, banks, waivers):
    values = []
    for bank in banks:
        values.append(
            sim.managed_marginal_values_multi_context_batch(
                bank["weekly_scores"],
                predictions.pos.to_numpy(),
                bank["decision_scores"],
                predictions.player.to_numpy(),
                [list(fixed_players)],
                waiver_baselines=waivers,
                lineup_require=LINEUP_REQUIRE,
                played_mask=bank["played_mask"],
            )[0]
        )
    return np.stack(values)


def roster_counts(predictions, roster):
    position_map = predictions.set_index("player").pos.to_dict()
    counts = Counter(position_map[player] for player in roster)
    return {pos: int(counts.get(pos, 0)) for pos in ("QB", "RB", "WR", "TE")}


def prepare_state(conn, *, blocks, construction_contexts, validation_contexts, seed):
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
    candidate_market = apply_counterfactual_salary_surface(sim)
    keeper_context = load_keeper_context(conn, sim)
    required_players = set().union(*(set(value) for value in SCENARIOS.values()))

    seeds = np.random.SeedSequence(seed).spawn(5)
    seed_values = [
        int(value.generate_state(1, dtype=np.uint32)[0])
        for value in seeds
    ]
    with sim.temp_seed(seed_values[0]):
        canonical_predictions = sim.get_predictions(
            "pred_fp_per_game",
            num_options=512,
        )
    predictions, pool_summary = sequential.apply_sequential_draft_pool_filter(
        canonical_predictions.copy(),
        sequential._sequential_draft_pool_metadata(sim),
        LEAGUE,
        required_players=required_players,
    )
    predictions = predictions.loc[
        ~predictions.player.isin(keeper_context["unavailable_players"])
    ].reset_index(drop=True)
    missing = sorted(required_players - set(predictions.player))
    if missing:
        raise ValueError("Scenario players missing from draft pool: " + ", ".join(missing))

    state_indices = sequential._canonical_state_indices(
        canonical_predictions,
        predictions,
    )
    canonical_aligned = sequential._aligned_player_frame(sim, canonical_predictions)
    aligned = sequential._aligned_player_frame(sim, predictions)
    canonical_market_prices = canonical_aligned.salary.to_numpy(dtype=np.float64)
    market_prices = aligned.salary.to_numpy(dtype=np.float64)
    predictions["salary"] = market_prices
    waivers = sim.estimate_waiver_baselines(
        num_teams=NUM_TEAMS,
        roster_size=ROSTER_SIZE,
    )

    _, construction_banks = sequential._sample_construction_value_blocks(
        sim,
        canonical_predictions,
        predictions,
        list(SCENARIOS[REFERENCE_SCENARIO]),
        block_count=blocks,
        contexts_per_block=construction_contexts,
        num_weeks=16,
        waiver_baselines=waivers,
        lineup_require=LINEUP_REQUIRE,
        learn_weeks=6,
        max_learn_weight=0.65,
        random_seed=seed_values[1],
        return_contexts=True,
    )
    validation_seeds = np.random.SeedSequence(seed_values[2]).spawn(blocks)
    validation_banks = []
    for block_idx in range(blocks):
        bank_seed = int(
            validation_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
        )
        validation_banks.append(
            sequential._sample_validation_bank(
                sim,
                predictions,
                validation_contexts,
                16,
                6,
                0.65,
                bank_seed,
                canonical_predictions=canonical_predictions,
            )
        )

    return {
        "sim": sim,
        "canonical_predictions": canonical_predictions,
        "predictions": predictions,
        "state_indices": state_indices,
        "canonical_market_prices": canonical_market_prices,
        "market_prices": market_prices,
        "waivers": waivers,
        "construction_banks": construction_banks,
        "validation_banks": validation_banks,
        "tape_seed": seed_values[3],
        "candidate_market": candidate_market,
        "keeper_context": keeper_context,
        "pool_summary": pool_summary,
    }


def run_scenarios(state, *, blocks, paths_per_block):
    sim = state["sim"]
    predictions = state["predictions"]
    canonical_predictions = state["canonical_predictions"]
    position_array = predictions.pos.to_numpy()
    player_array = predictions.player.to_numpy()
    remaining_market_budget = (
        NUM_TEAMS * SALARY_CAP
        - state["keeper_context"]["other_keeper_spend"]
        - 45.0
    )
    remaining_market_slots = (
        NUM_TEAMS * ROSTER_SIZE
        - state["keeper_context"]["other_keeper_count"]
        - 2
    )
    if not np.isclose(remaining_market_budget, 3135.0) or remaining_market_slots != 142:
        raise ValueError("Scenario keeper market does not reconcile to 142/$3,135")

    managed_by_scenario = {
        scenario: managed_values_from_banks(
            sim,
            predictions,
            fixed,
            state["construction_banks"],
            state["waivers"],
        )
        for scenario, fixed in SCENARIOS.items()
    }
    tape_seeds = [
        int(seed.generate_state(1, dtype=np.uint32)[0])
        for seed in np.random.SeedSequence(state["tape_seed"]).spawn(blocks)
    ]
    score_rows = []
    path_rows = []
    roster_player_rows = []
    root_plan_rows = []
    reference_orders = {}
    static_caches = {scenario: {} for scenario in SCENARIOS}
    score_caches = [{} for _ in range(blocks)]

    for scenario, fixed in SCENARIOS.items():
        fixed_players = set(fixed)
        canonical_available_mask = (
            np.isin(
                canonical_predictions.player.to_numpy(),
                predictions.player.to_numpy(),
            )
            & ~canonical_predictions.player.isin(fixed_players).to_numpy()
        )
        canonical_base_prices = sim.normalize_salary_market_values(
            state["canonical_market_prices"],
            canonical_available_mask,
            remaining_market_budget=remaining_market_budget,
            remaining_market_slots=remaining_market_slots,
        )
        base_prices = canonical_base_prices[state["state_indices"]]
        selection_premiums = np.zeros(len(predictions), dtype=np.float64)

        for block_idx in range(blocks):
            managed_values = managed_by_scenario[scenario][block_idx]
            plan = sequential.solve_history_only_plan(
                sim,
                predictions,
                managed_values,
                base_prices,
                selection_premiums,
                fixed,
                set(predictions.player) - fixed_players,
                ROSTER_SIZE,
                POSITION_MIN,
                POSITION_MAX,
                REQUIRE_TOP_N,
                True,
                static_matrix_cache=static_caches[scenario],
            )
            if plan is None:
                raise RuntimeError(f"No root plan for {scenario} block {block_idx}")
            root_roster = tuple(sorted(plan["selected"]))
            root_plan_rows.append({
                "scenario": scenario,
                "block": block_idx,
                "forecast_spend": float(sum(
                    plan["forecast_cost"][player] for player in root_roster
                )),
                "roster": " | ".join(root_roster),
                **{
                    f"count_{pos.lower()}": count
                    for pos, count in roster_counts(predictions, root_roster).items()
                },
            })
            evaluation_context = sequential._build_sequential_evaluation_context(
                sim,
                predictions,
                managed_values,
                base_prices,
                selection_premiums,
                fixed,
            )
            block_tape_seed = tape_seeds[block_idx]
            tapes = sequential.generate_hidden_auction_tapes(
                sim,
                predictions,
                fixed_players,
                paths_per_block,
                remaining_market_budget,
                remaining_market_slots,
                block_tape_seed,
                canonical_predictions=canonical_predictions,
            )
            if block_idx not in reference_orders:
                reference_orders[block_idx] = tapes["orders"].copy()
            elif not np.array_equal(reference_orders[block_idx], tapes["orders"]):
                raise AssertionError("Nomination orders differ across keeper scenarios")

            validation_bank = state["validation_banks"][block_idx]
            for path_idx in range(paths_per_block):
                rollout = sequential.simulate_history_only_branch(
                    sim=sim,
                    predictions=predictions,
                    managed_values=managed_values,
                    base_prices=base_prices,
                    selection_premiums=selection_premiums,
                    initial_salary_map=fixed,
                    candidate=None,
                    candidate_price=None,
                    force_buy=False,
                    order=tapes["orders"][path_idx],
                    revealed_prices=tapes["prices"][path_idx],
                    remaining_market_budget=remaining_market_budget,
                    remaining_market_slots=remaining_market_slots,
                    roster_size=ROSTER_SIZE,
                    pos_min_counts=POSITION_MIN,
                    pos_max_counts=POSITION_MAX,
                    require_top_n=REQUIRE_TOP_N,
                    enforce_top_n=True,
                    compiled_plan=plan,
                    policy_scores=evaluation_context["policy_scores"],
                    rollout_context=evaluation_context,
                    budget_reinvestment=True,
                )
                roster_players = set(rollout["roster"])
                if not fixed_players.issubset(roster_players):
                    raise AssertionError(
                        f"{scenario} block {block_idx} path {path_idx} omitted a fixed keeper"
                    )
                unavailable_keepers = set(
                    state["keeper_context"]["unavailable_players"]
                )
                if roster_players & unavailable_keepers:
                    raise AssertionError(
                        f"{scenario} block {block_idx} path {path_idx} selected an unavailable keeper"
                    )
                path_record = {
                    "scenario": scenario,
                    "block": block_idx,
                    "path": path_idx,
                    "complete": bool(rollout["complete"]),
                    "salary_spend": float(rollout["salary_spend"]),
                    "unused_salary": float(rollout["final_unused_salary"]),
                    "replans": int(rollout["replans"]),
                    "bounded_swaps": int(rollout["bounded_swaps"]),
                    "failure_reason": rollout.get("failure_reason"),
                    "roster": " | ".join(rollout["roster"]),
                    **{
                        f"count_{pos.lower()}": count
                        for pos, count in roster_counts(
                            predictions, rollout["roster"]
                        ).items()
                    },
                }
                if not rollout["complete"]:
                    path_rows.append(path_record)
                    continue
                scores = sequential._score_roster_bank(
                    sim,
                    predictions,
                    rollout["roster"],
                    *validation_bank,
                    LINEUP_REQUIRE,
                    state["waivers"],
                    score_caches[block_idx],
                    evaluation_context=evaluation_context,
                )
                path_record["holdout_mean"] = float(np.mean(scores))
                path_record["holdout_p10"] = float(np.percentile(scores, 10))
                path_rows.append(path_record)
                salary_map = dict(rollout["salary_map"])
                for player in rollout["roster"]:
                    roster_player_rows.append({
                        "scenario": scenario,
                        "block": block_idx,
                        "path": path_idx,
                        "player": player,
                        "pos": evaluation_context["position_map"][player],
                        "paid_salary": float(salary_map[player]),
                        "fixed": player in fixed_players,
                    })
                for context_idx, score in enumerate(scores):
                    score_rows.append({
                        "scenario": scenario,
                        "block": block_idx,
                        "path": path_idx,
                        "context": context_idx,
                        "managed_season_score": float(score),
                    })

    return (
        pd.DataFrame(root_plan_rows),
        pd.DataFrame(path_rows),
        pd.DataFrame(roster_player_rows),
        pd.DataFrame(score_rows),
        remaining_market_budget,
        remaining_market_slots,
    )


def summarize(path_rows, roster_players, score_rows):
    completed = path_rows.loc[path_rows.complete].copy()
    score_summary = (
        score_rows.groupby("scenario", as_index=False)
        .agg(
            managed_ev=("managed_season_score", "mean"),
            managed_sd=("managed_season_score", "std"),
            score_cells=("managed_season_score", "size"),
        )
    )
    percentiles = (
        score_rows.groupby("scenario").managed_season_score
        .quantile([0.10, 0.50, 0.90])
        .unstack()
        .rename(columns={0.10: "managed_p10", 0.50: "managed_p50", 0.90: "managed_p90"})
        .reset_index()
    )
    path_summary = (
        path_rows.groupby("scenario", as_index=False)
        .agg(paths=("path", "size"), completed_paths=("complete", "sum"))
    )
    completed_summary = (
        completed.groupby("scenario", as_index=False)
        .agg(
            avg_spend=("salary_spend", "mean"),
            avg_unused=("unused_salary", "mean"),
            avg_qb=("count_qb", "mean"),
            avg_rb=("count_rb", "mean"),
            avg_wr=("count_wr", "mean"),
            avg_te=("count_te", "mean"),
            avg_replans=("replans", "mean"),
            avg_bounded_swaps=("bounded_swaps", "mean"),
            unique_rosters=("roster", "nunique"),
        )
    )
    summary = (
        score_summary.merge(percentiles, on="scenario", validate="one_to_one")
        .merge(path_summary, on="scenario", validate="one_to_one")
        .merge(completed_summary, on="scenario", validate="one_to_one")
    )
    summary["completion_rate"] = summary.completed_paths / summary.paths
    reference_ev = float(
        summary.loc[summary.scenario.eq(REFERENCE_SCENARIO), "managed_ev"].iloc[0]
    )
    reference_p10 = float(
        summary.loc[summary.scenario.eq(REFERENCE_SCENARIO), "managed_p10"].iloc[0]
    )
    summary["ev_delta_vs_tuten"] = summary.managed_ev - reference_ev
    summary["p10_delta_vs_tuten"] = summary.managed_p10 - reference_p10

    wide = score_rows.pivot(
        index=["block", "path", "context"],
        columns="scenario",
        values="managed_season_score",
    ).dropna(subset=list(SCENARIOS))
    paired_rows = []
    block_rows = []
    for scenario in SCENARIOS:
        delta = wide[scenario] - wide[REFERENCE_SCENARIO]
        block_delta = delta.groupby(level="block").mean()
        block_se = (
            float(block_delta.std(ddof=1) / np.sqrt(len(block_delta)))
            if len(block_delta) > 1
            else 0.0
        )
        paired_rows.append({
            "scenario": scenario,
            "reference": REFERENCE_SCENARIO,
            "paired_cells": int(len(delta)),
            "paired_mean_delta": float(delta.mean()),
            "paired_p10_delta": float(np.percentile(delta, 10)),
            "paired_median_delta": float(np.median(delta)),
            "paired_win_rate": float(np.mean(delta > 0)),
            "positive_blocks": int((block_delta > 0).sum()),
            "blocks": int(len(block_delta)),
            "block_mean_se": block_se,
            "block_lcb80": float(delta.mean() - 0.8416212335729143 * block_se),
        })
        for block, value in block_delta.items():
            block_rows.append({
                "scenario": scenario,
                "reference": REFERENCE_SCENARIO,
                "block": int(block),
                "mean_delta": float(value),
            })

    frequencies = (
        roster_players.groupby(["scenario", "player", "pos", "fixed"], as_index=False)
        .agg(
            paths_selected=("path", "size"),
            avg_paid_salary=("paid_salary", "mean"),
        )
    )
    completed_counts = completed.groupby("scenario").size()
    frequencies["selection_rate"] = frequencies.apply(
        lambda row: row.paths_selected / completed_counts[row.scenario],
        axis=1,
    )
    return (
        summary.sort_values("managed_ev", ascending=False),
        pd.DataFrame(paired_rows),
        pd.DataFrame(block_rows),
        frequencies,
        wide,
    )


def representative_rosters(path_rows, summary):
    records = []
    for scenario, rows in path_rows.loc[path_rows.complete].groupby("scenario"):
        target_ev = float(
            summary.loc[summary.scenario.eq(scenario), "managed_ev"].iloc[0]
        )
        target_spend = float(rows.salary_spend.mean())
        central = rows.assign(
            distance=(rows.holdout_mean - target_ev).abs()
            + 0.10 * (rows.salary_spend - target_spend).abs()
        ).sort_values(["distance", "block", "path"]).iloc[0]
        records.append({
            "scenario": scenario,
            "block": int(central.block),
            "path": int(central.path),
            "managed_ev": float(central.holdout_mean),
            "managed_p10": float(central.holdout_p10),
            "salary_spend": float(central.salary_spend),
            "unused_salary": float(central.unused_salary),
            "qb": int(central.count_qb),
            "rb": int(central.count_rb),
            "wr": int(central.count_wr),
            "te": int(central.count_te),
            "roster": central.roster,
        })
    return pd.DataFrame(records)


def write_summary(summary, paired, representatives, frequencies, metadata):
    labels = {
        "tuten": "Brown + Tuten",
        "burden": "Brown + Burden",
        "loveland": "Brown + Loveland",
    }
    lines = [
        "# Beta Keeper Roster Tournament Results",
        "",
        (
            f"All three scenarios use {metadata['blocks']} shared construction blocks, "
            f"{metadata['paths_per_block']} hidden auction paths per block, and "
            f"{metadata['validation_contexts_per_block']} held-out managed seasons "
            "per completed roster."
        ),
        "",
        "## Aggregate managed-season result",
        "",
        "| Keeper start | EV | Delta vs Tuten | P10 | P10 delta | Completion | Avg spend | Avg QB/RB/WR/TE |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {labels[row.scenario]} | {row.managed_ev:.2f} | "
            f"{row.ev_delta_vs_tuten:+.2f} | {row.managed_p10:.2f} | "
            f"{row.p10_delta_vs_tuten:+.2f} | {row.completion_rate:.1%} | "
            f"${row.avg_spend:.2f} | {row.avg_qb:.2f}/{row.avg_rb:.2f}/"
            f"{row.avg_wr:.2f}/{row.avg_te:.2f} |"
        )
    lines.extend([
        "",
        "## Paired differences versus Brown + Tuten",
        "",
        "| Scenario | Mean delta | P10 of paired delta | Win rate | Positive blocks | LCB80 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in paired.itertuples(index=False):
        if row.scenario == REFERENCE_SCENARIO:
            continue
        lines.append(
            f"| {labels[row.scenario]} | {row.paired_mean_delta:+.2f} | "
            f"{row.paired_p10_delta:+.2f} | {row.paired_win_rate:.1%} | "
            f"{row.positive_blocks}/{row.blocks} | {row.block_lcb80:+.2f} |"
        )
    lines.extend([
        "",
        "## Representative completed rosters",
        "",
    ])
    position_lookup = metadata["position_lookup"]
    for row in representatives.itertuples(index=False):
        lines.append(
            f"### {labels[row.scenario]} — {row.managed_ev:.1f} EV, "
            f"{row.managed_p10:.1f} P10, ${row.salary_spend:.0f} spend"
        )
        lines.append("")
        players = str(row.roster).split(" | ")
        for pos in ("QB", "RB", "WR", "TE"):
            selected = [player for player in players if position_lookup[player] == pos]
            lines.append(f"- {pos}: " + ", ".join(selected))
        lines.append("")
    lines.extend([
        "## Frequent non-fixed targets",
        "",
    ])
    for scenario in SCENARIOS:
        top = frequencies.loc[
            frequencies.scenario.eq(scenario) & ~frequencies.fixed
        ].nlargest(10, ["selection_rate", "avg_paid_salary"])
        target_text = "; ".join(
            f"{row.player} ({row.selection_rate:.0%}, ${row.avg_paid_salary:.1f})"
            for row in top.itertuples(index=False)
        )
        lines.append(f"- **{labels[scenario]}:** {target_text}")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This isolates current-season roster value under the current Sequential "
        "policy. It does not add a separate next-year keeper-option bonus. The "
        "selection-premium reserve is disabled because the stored calibration "
        "belongs to the production Brown/Tuten keeper state.",
        "",
    ])
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--paths", type=int, default=24)
    parser.add_argument("--construction-contexts", type=int, default=64)
    parser.add_argument("--validation-contexts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    for name in ("blocks", "paths", "construction_contexts", "validation_contexts"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    conn = sqlite3.connect(APP_DB)
    try:
        state = prepare_state(
            conn,
            blocks=args.blocks,
            construction_contexts=args.construction_contexts,
            validation_contexts=args.validation_contexts,
            seed=args.seed,
        )
        (
            root_plans,
            path_rows,
            roster_players,
            score_rows,
            remaining_market_budget,
            remaining_market_slots,
        ) = run_scenarios(
            state,
            blocks=args.blocks,
            paths_per_block=args.paths,
        )
    finally:
        conn.close()

    summary, paired, block_deltas, frequencies, wide_scores = summarize(
        path_rows,
        roster_players,
        score_rows,
    )
    representatives = representative_rosters(path_rows, summary)
    position_lookup = state["predictions"].set_index("player").pos.to_dict()
    metadata = {
        "year": YEAR,
        "league": LEAGUE,
        "scenarios": SCENARIOS,
        "counterfactual_candidate_market": state["candidate_market"],
        "keeper_context": state["keeper_context"],
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "salary_cap": SALARY_CAP,
        "roster_size": ROSTER_SIZE,
        "lineup_require": LINEUP_REQUIRE,
        "position_min": POSITION_MIN,
        "position_max": POSITION_MAX,
        "require_top_n": REQUIRE_TOP_N,
        "selection_premium_enabled": False,
        "joint_refinement_swaps": 0,
        "bounded_reinvestment": True,
        "waiver_baselines": state["waivers"],
        "blocks": args.blocks,
        "paths_per_block": args.paths,
        "construction_contexts_per_block": args.construction_contexts,
        "validation_contexts_per_block": args.validation_contexts,
        "paired_score_cells": int(len(wide_scores)),
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "position_lookup": position_lookup,
    }

    root_plans.to_csv(RESULTS_DIR / "root_plans.csv", index=False)
    path_rows.to_csv(RESULTS_DIR / "organic_paths.csv", index=False)
    roster_players.to_csv(RESULTS_DIR / "roster_players.csv", index=False)
    score_rows.to_csv(RESULTS_DIR / "managed_score_cells.csv", index=False)
    summary.to_csv(RESULTS_DIR / "scenario_summary.csv", index=False)
    paired.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)
    block_deltas.to_csv(RESULTS_DIR / "block_deltas.csv", index=False)
    frequencies.to_csv(RESULTS_DIR / "player_frequencies.csv", index=False)
    representatives.to_csv(RESULTS_DIR / "representative_rosters.csv", index=False)
    (RESULTS_DIR / "metadata.json").write_text(
        json.dumps(json_value(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(summary, paired, representatives, frequencies, metadata)

    print("\nSCENARIO SUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED COMPARISONS")
    print(paired.to_string(index=False))
    print("\nREPRESENTATIVE ROSTERS")
    print(representatives.to_string(index=False))
    print(f"\nRuntime: {metadata['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
