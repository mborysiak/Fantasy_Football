"""Audit experimental current-plan salary/PPG structure buckets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_WORKTREE = ROOT.parent / "Fantasy_Football_App_target_tiers"
APP_DIR = APP_WORKTREE / "app"
APP_DB = ROOT.parent / "Fantasy_Football_App" / "app" / "Simulation.sqlite3"
SHARED_STUDY = (
    ROOT / "research" / "studies" / "2026-08-24_sequential_shared_opportunity"
)
RESULTS_DIR = STUDY_DIR / "results"
for import_path in (APP_DIR, SHARED_STUDY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import zSequential_Target as sequential  # noqa: E402
from keeper_market import load_active_keeper_market  # noqa: E402
from zSim_Helper import FootballSimulation  # noqa: E402


YEAR = 2026
PRED_VERSION = "final_ensemble"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
COMPUTE_BUDGET = 320
REQUIRE_TOP_N = 12
CONFIG = {
    "beta": {
        "lineup": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2},
        "pos_min": {"QB": 1, "RB": 4, "WR": 4, "TE": 1},
        "pos_max": {"QB": 1, "RB": 6, "WR": 6, "TE": 2},
        "owned": {
            "Chase Brown": 34.0,
            "Bhayshul Tuten": 11.0,
            "Jordyn Tyson": 7.0,
            "Jonah Coleman": 1.0,
        },
        "owned_keepers": {"Chase Brown": 34.0, "Bhayshul Tuten": 11.0},
    },
    "nv": {
        "lineup": {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        "pos_min": {"QB": 2, "RB": 4, "WR": 4, "TE": 1},
        "pos_max": {"QB": 3, "RB": 6, "WR": 6, "TE": 2},
        "owned": {"Drake Maye": 18.0, "De'Von Achane": 47.0},
        "owned_keepers": {"Drake Maye": 18.0, "De'Von Achane": 47.0},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=tuple(CONFIG), required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    config = CONFIG[args.league]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(APP_DB)
    try:
        sim = FootballSimulation(
            conn,
            YEAR,
            config["lineup"],
            SALARY_CAP,
            PRED_VERSION,
            args.league,
            sal_pred_actual="pred",
        )
        sim.load_weekly_template_profiles()
        keeper_market = load_active_keeper_market(
            conn,
            sim,
            year=YEAR,
            league=args.league,
            salary_source="predicted",
            owned_salary_map=config["owned_keepers"],
        )
        nonkeeper_owned = (
            set(config["owned"]) - set(config["owned_keepers"])
        )
        nonkeeper_spend = sum(
            config["owned"][player] for player in nonkeeper_owned
        )
        remaining_market_budget = (
            NUM_TEAMS * SALARY_CAP
            - keeper_market["keeper_spend"]
            - nonkeeper_spend
        )
        remaining_market_slots = (
            NUM_TEAMS * ROSTER_SIZE
            - keeper_market["keeper_count"]
            - len(nonkeeper_owned)
        )
        waiver_baselines = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )
        seeds = np.random.SeedSequence(args.seed).spawn(2)
        seed_values = [
            int(seed.generate_state(1, dtype=np.uint32)[0])
            for seed in seeds
        ]
        with sim.temp_seed(seed_values[0]):
            canonical = sim.get_predictions(
                "pred_fp_per_game",
                num_options=512,
            )
        predictions = sim.drop_players(
            canonical,
            keeper_market["unavailable_keeper_players"],
        )
        predictions, pool_summary = sequential.apply_sequential_draft_pool_filter(
            predictions,
            sequential._sequential_draft_pool_metadata(sim),
            args.league,
            required_players=set(config["owned"]),
        )
        state_indices = sequential._canonical_state_indices(
            canonical,
            predictions,
        )
        canonical_aligned = sequential._aligned_player_frame(sim, canonical)
        aligned = sequential._aligned_player_frame(sim, predictions)
        market_prices = aligned.salary.to_numpy(dtype=np.float64)
        canonical_available = (
            np.isin(canonical.player.to_numpy(), predictions.player.to_numpy())
            & ~canonical.player.isin(config["owned"]).to_numpy()
        )
        canonical_base_prices = sim.normalize_salary_market_values(
            canonical_aligned.salary.to_numpy(dtype=np.float64),
            canonical_available,
            remaining_market_budget=remaining_market_budget,
            remaining_market_slots=remaining_market_slots,
        )
        base_prices = canonical_base_prices[state_indices]
        selection_premiums = sim.get_selection_premium_values(
            predictions.player.to_numpy(),
            fixed_players=list(config["owned"]),
            enabled=False,
        )
        dimensions = sequential._sequential_evidence_dimensions(COMPUTE_BUDGET)
        managed_blocks = sequential._sample_construction_value_blocks(
            sim,
            canonical,
            predictions,
            list(config["owned"]),
            block_count=4,
            contexts_per_block=dimensions["construction_contexts"],
            num_weeks=16,
            waiver_baselines=waiver_baselines,
            lineup_require=config["lineup"],
            learn_weeks=6,
            max_learn_weight=0.65,
            random_seed=seed_values[1],
            return_contexts=False,
        )
        mean_values = np.asarray(managed_blocks).mean(axis=0)
        raw_prices = {
            player: sequential._round_price(price)
            for player, price in zip(predictions.player, market_prices)
        }
        fixed_counts = Counter(
            predictions.loc[
                predictions.player.isin(config["owned"]),
                "pos",
            ]
        )
        personal_open_after = ROSTER_SIZE - len(config["owned"]) - 1
        maximum_price = max(1, int(np.floor(min(
            SALARY_CAP - sum(config["owned"].values()) - personal_open_after,
            remaining_market_budget - (remaining_market_slots - 1),
        ))))
        eligible = np.array([
            player not in config["owned"]
            and fixed_counts.get(pos, 0) < config["pos_max"].get(pos, ROSTER_SIZE)
            and raw_prices[player] <= maximum_price
            for player, pos in zip(predictions.player, predictions.pos)
        ])
        peer_frame = pd.DataFrame({
            "player": predictions.loc[eligible, "player"].to_numpy(),
            "pos": predictions.loc[eligible, "pos"].to_numpy(),
            "market_price": np.asarray([
                raw_prices[player]
                for player in predictions.loc[eligible, "player"]
            ], dtype=np.float64),
            "ppg": aligned.loc[
                eligible, "pred_fp_per_game"
            ].to_numpy(dtype=np.float64),
            "value": mean_values[eligible],
        })
        started = time.perf_counter()
        buckets, plan_count = sequential.build_sequential_structure_buckets(
            sim,
            predictions,
            managed_blocks,
            base_prices,
            selection_premiums,
            config["owned"],
            peer_frame,
            ROSTER_SIZE,
            config["pos_min"],
            config["pos_max"],
            REQUIRE_TOP_N,
            True,
            static_matrix_cache={},
        )
        runtime = time.perf_counter() - started
    finally:
        conn.close()

    display_records = [
        {key: value for key, value in bucket.items() if not key.startswith("_")}
        for bucket in buckets
    ]
    pd.DataFrame(display_records).to_csv(
        RESULTS_DIR / f"{args.league}_structure_buckets.csv",
        index=False,
    )
    metadata = {
        "league": args.league,
        "calculation_version": sequential.SEQUENTIAL_TARGET_VERSION,
        "seed": args.seed,
        "owned": config["owned"],
        "unavailable_keepers": keeper_market["unavailable_keeper_players"],
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "waiver_baselines": waiver_baselines,
        "plan_count": plan_count,
        "bucket_count": len(display_records),
        "structure_runtime_seconds": runtime,
        "pool_summary": pool_summary,
    }
    (RESULTS_DIR / f"{args.league}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(display_records).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
