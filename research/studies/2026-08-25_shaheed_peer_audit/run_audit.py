"""Paired Shaheed/Coker/Doubs Sequential decision audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

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
from zTemplate_Explorer import read_weekly_template_comp_data  # noqa: E402


YEAR = 2026
LEAGUE = "beta"
PRED_VERSION = "final_ensemble"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
COMPUTE_BUDGET = 320
LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POSITION_MIN = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
POSITION_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 2}
REQUIRE_TOP_N = 12
OWNED_SALARIES = {
    "Chase Brown": 34.0,
    "Bhayshul Tuten": 11.0,
    "Jordyn Tyson": 7.0,
    "Jonah Coleman": 1.0,
}
CANDIDATES = ("Rashid Shaheed", "Jalen Coker", "Romeo Doubs")


def json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (tuple, list, set)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def prepare_state(seed_variation):
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
    keepers = pd.read_sql_query(
        """
        SELECT player_key, keeper_salary
        FROM League_Keepers
        WHERE year = :year AND league = :league
        ORDER BY player_key
        """,
        conn,
        params={"year": YEAR, "league": LEAGUE},
    )
    keepers = keepers.merge(
        sim.player_data[["player_key", "player"]],
        on="player_key",
        how="left",
        validate="one_to_one",
    ).sort_values("player")
    if len(keepers) != 14 or keepers.player.duplicated().any():
        raise ValueError("Expected 14 unique active beta keepers.")
    keeper_players = set(keepers.player)
    owned_keeper_players = keeper_players & set(OWNED_SALARIES)
    if owned_keeper_players != {"Chase Brown", "Bhayshul Tuten"}:
        raise ValueError("Displayed owned keeper state does not match beta keepers.")
    unavailable = sorted(keeper_players - owned_keeper_players)
    keeper_spend = float(keepers.keeper_salary.sum())
    drafted_nonkeeper_spend = sum(
        salary
        for player, salary in OWNED_SALARIES.items()
        if player not in keeper_players
    )
    drafted_nonkeeper_count = sum(
        player not in keeper_players for player in OWNED_SALARIES
    )
    remaining_market_budget = (
        NUM_TEAMS * SALARY_CAP - keeper_spend - drafted_nonkeeper_spend
    )
    remaining_market_slots = (
        NUM_TEAMS * ROSTER_SIZE - len(keepers) - drafted_nonkeeper_count
    )
    waiver_baselines = sim.estimate_waiver_baselines(
        num_teams=NUM_TEAMS,
        roster_size=ROSTER_SIZE,
    )
    seed_components = [
        YEAR,
        LEAGUE,
        PRED_VERSION,
        "pred",
        COMPUTE_BUDGET,
        True,
        REQUIRE_TOP_N,
        tuple(sorted(LINEUP_REQUIRE.items())),
        ROSTER_SIZE,
        tuple(sorted(POSITION_MIN.items())),
        tuple(sorted(POSITION_MAX.items())),
        tuple(sorted(waiver_baselines.items())),
        False,
    ]
    if int(seed_variation) > 0:
        seed_components.extend(("user_variation", int(seed_variation)))
    evidence_seed = sequential.stable_sequential_evidence_seed(*seed_components)
    return {
        "conn": conn,
        "sim": sim,
        "keepers": keepers,
        "unavailable": unavailable,
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "waiver_baselines": waiver_baselines,
        "evidence_seed": evidence_seed,
    }


def construction_diagnostics(state):
    sim = state["sim"]
    seeds = np.random.SeedSequence(state["evidence_seed"]).spawn(8)
    seed_values = [
        int(seed.generate_state(1, dtype=np.uint32)[0]) for seed in seeds
    ]
    with sim.temp_seed(seed_values[0]):
        canonical = sim.get_predictions("pred_fp_per_game", num_options=512)
    predictions = sim.drop_players(canonical, state["unavailable"])
    predictions, _ = sequential.apply_sequential_draft_pool_filter(
        predictions,
        sequential._sequential_draft_pool_metadata(sim),
        LEAGUE,
        required_players=set(OWNED_SALARIES) | set(CANDIDATES),
    )
    blocks, banks = sequential._sample_construction_value_blocks(
        sim,
        canonical,
        predictions,
        list(OWNED_SALARIES),
        block_count=4,
        contexts_per_block=32,
        num_weeks=16,
        waiver_baselines=state["waiver_baselines"],
        lineup_require=LINEUP_REQUIRE,
        learn_weeks=6,
        max_learn_weight=0.65,
        random_seed=seed_values[1],
        return_contexts=True,
    )
    player_index = {
        player: idx for idx, player in enumerate(predictions.player)
    }
    aligned = sequential._aligned_player_frame(sim, predictions)
    state_indices = sequential._canonical_state_indices(canonical, predictions)
    canonical_aligned = sequential._aligned_player_frame(sim, canonical)
    canonical_available_mask = (
        np.isin(canonical.player.to_numpy(), predictions.player.to_numpy())
        & ~canonical.player.isin(OWNED_SALARIES).to_numpy()
    )
    canonical_base_prices = sim.normalize_salary_market_values(
        canonical_aligned.salary.to_numpy(dtype=np.float64),
        canonical_available_mask,
        remaining_market_budget=state["remaining_market_budget"],
        remaining_market_slots=state["remaining_market_slots"],
    )
    base_prices = canonical_base_prices[state_indices]
    base_price_map = dict(zip(predictions.player, base_prices))
    zero_premiums = np.zeros(len(predictions), dtype=np.float64)
    rows = []
    plan_rows = []
    fixed_swap_rows = []
    dimensions = sequential._sequential_evidence_dimensions(COMPUTE_BUDGET)
    validation_seeds = np.random.SeedSequence(seed_values[6]).spawn(4)
    validation_banks = [
        sequential._sample_validation_bank(
            sim,
            predictions,
            dimensions["confirm_validation"],
            16,
            6,
            0.65,
            int(seed.generate_state(1, dtype=np.uint32)[0]),
            canonical_predictions=canonical,
        )
        for seed in validation_seeds
    ]
    for player in CANDIDATES:
        idx = player_index[player]
        values = blocks[:, idx]
        weekly = np.concatenate([
            np.asarray(bank["weekly_scores"])[:, idx, :]
            for bank in banks
        ])
        played = np.concatenate([
            np.asarray(bank["played_mask"])[:, idx, :]
            for bank in banks
        ])
        waiver = float(state["waiver_baselines"]["WR"])
        rows.append({
            "player": player,
            "pred_ppg": float(aligned.loc[player, "pred_fp_per_game"]),
            "model_salary": float(aligned.loc[player, "salary"]),
            "managed_value_mean": float(np.mean(values)),
            "managed_value_min": float(np.min(values)),
            "managed_value_max": float(np.max(values)),
            "context_total_points_mean": float(weekly.sum(axis=1).mean()),
            "context_played_weeks_mean": float(played.sum(axis=1).mean()),
            "context_weeks_above_waiver": float((weekly > waiver).sum(axis=1).mean()),
            "context_weeks_10_plus": float((weekly >= 10.0).sum(axis=1).mean()),
            "context_weeks_15_plus": float((weekly >= 15.0).sum(axis=1).mean()),
            "context_points_above_waiver": float(
                np.maximum(weekly - waiver, 0.0).sum(axis=1).mean()
            ),
            **{
                f"managed_value_block_{block_idx}": float(value)
                for block_idx, value in enumerate(values)
            },
        })
        for block_idx, managed_values in enumerate(blocks):
            static_cache = {}
            buy_owned = {**OWNED_SALARIES, player: float(round(aligned.loc[player, "salary"]))}
            buy = sequential.solve_history_only_plan(
                sim,
                predictions,
                managed_values,
                base_prices,
                zero_premiums,
                buy_owned,
                set(predictions.player) - set(buy_owned),
                ROSTER_SIZE,
                POSITION_MIN,
                POSITION_MAX,
                REQUIRE_TOP_N,
                True,
                observed_sales=[(buy_owned[player], base_price_map[player])],
                static_matrix_cache=static_cache,
            )
            pass_owned = dict(OWNED_SALARIES)
            pass_payment = max(float(round(aligned.loc[player, "salary"])) - 1.0, 1.0)
            passed = sequential.solve_history_only_plan(
                sim,
                predictions,
                managed_values,
                base_prices,
                zero_premiums,
                pass_owned,
                set(predictions.player) - set(pass_owned) - {player},
                ROSTER_SIZE,
                POSITION_MIN,
                POSITION_MAX,
                REQUIRE_TOP_N,
                True,
                observed_sales=[(pass_payment, base_price_map[player])],
                static_matrix_cache=static_cache,
            )
            buy_roster = set(buy["selected"])
            pass_roster = set(passed["selected"])
            plan_rows.append({
                "player": player,
                "block": block_idx,
                "buy_only": " | ".join(sorted(buy_roster - pass_roster)),
                "pass_only": " | ".join(sorted(pass_roster - buy_roster)),
                "buy_roster": " | ".join(sorted(buy_roster)),
                "pass_roster": " | ".join(sorted(pass_roster)),
                "coker_in_buy": "Jalen Coker" in buy_roster,
                "coker_in_pass": "Jalen Coker" in pass_roster,
                "doubs_in_buy": "Romeo Doubs" in buy_roster,
                "doubs_in_pass": "Romeo Doubs" in pass_roster,
                "shaheed_in_buy": "Rashid Shaheed" in buy_roster,
                "shaheed_in_pass": "Rashid Shaheed" in pass_roster,
            })
            if player == "Rashid Shaheed":
                for substitute in CANDIDATES:
                    variant = (buy_roster - {"Rashid Shaheed"}) | {substitute}
                    scores = sequential._score_roster_bank(
                        sim,
                        predictions,
                        sorted(variant),
                        *validation_banks[block_idx],
                        LINEUP_REQUIRE,
                        state["waiver_baselines"],
                        {},
                    )
                    fixed_swap_rows.append({
                        "block": block_idx,
                        "player": substitute,
                        "mean": float(np.mean(scores)),
                        "p10": float(np.percentile(scores, 10)),
                    })
    fixed_swaps = pd.DataFrame(fixed_swap_rows)
    baseline = fixed_swaps.loc[
        fixed_swaps.player.eq("Rashid Shaheed"),
        ["block", "mean", "p10"],
    ].rename(columns={"mean": "shaheed_mean", "p10": "shaheed_p10"})
    fixed_swaps = fixed_swaps.merge(baseline, on="block", validate="many_to_one")
    fixed_swaps["mean_delta_vs_shaheed"] = (
        fixed_swaps["mean"] - fixed_swaps["shaheed_mean"]
    )
    fixed_swaps["p10_delta_vs_shaheed"] = (
        fixed_swaps["p10"] - fixed_swaps["shaheed_p10"]
    )
    return pd.DataFrame(rows), pd.DataFrame(plan_rows), fixed_swaps


def template_diagnostics(state):
    _, comps = read_weekly_template_comp_data(
        state["conn"],
        YEAR,
        LEAGUE,
        PRED_VERSION,
    )
    rows = []
    for player in CANDIDATES:
        frame = comps.loc[comps.target_player.eq(player)].copy()
        probs = pd.to_numeric(
            frame.template_sample_prob,
            errors="coerce",
        ).to_numpy(dtype=float)
        probs = probs / probs.sum()
        residuals = (
            pd.to_numeric(frame.active_ppg_resid, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        residual_mean = float(np.sum(probs * residuals))
        centered = residuals - residual_mean
        played = (
            pd.to_numeric(frame.played_games, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        def weighted_quantile(values, quantile):
            order = np.argsort(values, kind="mergesort")
            ordered_values = values[order]
            ordered_probs = probs[order]
            return float(ordered_values[
                np.searchsorted(np.cumsum(ordered_probs), quantile, side="left")
            ])

        rows.append({
            "player": player,
            "donor_count": int(len(frame)),
            "donor_effective_n": float(1.0 / np.sum(np.square(probs))),
            "raw_residual_mean": residual_mean,
            "raw_positive_probability": float(probs[residuals > 0].sum()),
            "raw_negative_probability": float(probs[residuals < 0].sum()),
            "centered_residual_sd": float(
                np.sqrt(np.sum(probs * np.square(centered)))
            ),
            "centered_residual_p10": weighted_quantile(centered, 0.10),
            "centered_residual_p90": weighted_quantile(centered, 0.90),
            "donor_expected_played": float(np.sum(probs * played)),
        })
    return pd.DataFrame(rows)


def run_decisions(state, candidates):
    rows = []
    block_rows = []
    raw = {}
    sim = state["sim"]
    price_lookup = sim.player_data.set_index("player").salary
    for player in candidates:
        price = int(round(float(price_lookup[player])))
        result = sequential.run_sequential_nomination_analysis(
            sim,
            {
                "players": list(OWNED_SALARIES),
                "salaries": list(OWNED_SALARIES.values()),
            },
            state["unavailable"],
            player,
            price,
            compute_budget=COMPUTE_BUDGET,
            require_top_n=REQUIRE_TOP_N,
            enforce_top_n=True,
            roster_size=ROSTER_SIZE,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts=POSITION_MIN,
            pos_max_counts=POSITION_MAX,
            waiver_baselines=state["waiver_baselines"],
            remaining_market_budget=state["remaining_market_budget"],
            remaining_market_slots=state["remaining_market_slots"],
            use_selection_premium=False,
            random_seed=state["evidence_seed"],
            profile_bid=False,
        )
        rows.append({
            "player": player,
            "price": price,
            "recommendation": result["recommendation"],
            "sequential_gain": result["SequentialGain"],
            "lcb80": result["SequentialLCB80"],
            "gain_p10": result["GainP10"],
            "win_rate": result["WinRate"],
            "buy_ev": result["BuyEV"],
            "pass_ev": result["PassEV"],
            "buy_season_p10": result["BuySeasonP10"],
            "pass_season_p10": result["PassSeasonP10"],
            "season_p10_delta": result["SeasonP10Delta"],
            "block_positive_rate": result["BlockPositiveRate"],
            "block_gain_min": result["BlockGainMin"],
            "block_gain_max": result["BlockGainMax"],
            "common_fallback": result["CommonFallback"],
            "top_alternatives": " | ".join(result["TopAlternatives"]),
            "buy_completion_core": result["BuyCompletionCore"],
            "buy_completion": result["BuyCompletion"],
            "pass_completion": result["PassCompletion"],
            "paired_rate": result["PairedRate"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
        for block in result["_CurrentBlockEvidence"]:
            block_rows.append({"player": player, **json_value(block)})
        raw[player] = json_value(result)
    return pd.DataFrame(rows), pd.DataFrame(block_rows), raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = prepare_state(args.variation)
    try:
        construction, plan_pairs, fixed_swaps = construction_diagnostics(state)
        templates = template_diagnostics(state)
        candidates = (args.candidate,) if args.candidate else CANDIDATES
        decisions, blocks, raw = run_decisions(state, candidates)
        output = (
            decisions
            .merge(construction, on="player", validate="one_to_one")
            .merge(templates, on="player", validate="one_to_one")
        )
        suffix = (
            "_" + args.candidate.lower().replace(" ", "_")
            if args.candidate else ""
        )
        suffix += f"_v{int(args.variation)}"
        output.to_csv(
            RESULTS_DIR / f"candidate_comparison{suffix}.csv",
            index=False,
        )
        blocks.to_csv(
            RESULTS_DIR / f"block_comparison{suffix}.csv",
            index=False,
        )
        plan_pairs.loc[plan_pairs.player.isin(candidates)].to_csv(
            RESULTS_DIR / f"compiled_plan_pairs{suffix}.csv",
            index=False,
        )
        fixed_swaps.to_csv(
            RESULTS_DIR / f"fixed_roster_swaps{suffix}.csv",
            index=False,
        )
        metadata = {
            "calculation_version": sequential.SEQUENTIAL_TARGET_VERSION,
            "screen_joint_swaps": sequential.DEFAULT_SEQUENTIAL_SCREEN_JOINT_SWAPS,
            "confirm_joint_swaps": sequential.DEFAULT_SEQUENTIAL_CONFIRM_JOINT_SWAPS,
            "owned_salaries": OWNED_SALARIES,
            "unavailable_keepers": state["unavailable"],
            "remaining_market_budget": state["remaining_market_budget"],
            "remaining_market_slots": state["remaining_market_slots"],
            "waiver_baselines": state["waiver_baselines"],
            "evidence_seed": state["evidence_seed"],
            "compute_budget": COMPUTE_BUDGET,
            "seed_variation": int(args.variation),
        }
        (RESULTS_DIR / "metadata.json").write_text(
            json.dumps(json_value(metadata), indent=2),
            encoding="utf-8",
        )
        (RESULTS_DIR / f"raw_results{suffix}.json").write_text(
            json.dumps(raw, indent=2),
            encoding="utf-8",
        )
        if not args.quiet:
            print(output.to_string(index=False))
            print("\nBLOCKS")
            print(blocks.to_string(index=False))
    finally:
        state["conn"].close()


if __name__ == "__main__":
    main()
