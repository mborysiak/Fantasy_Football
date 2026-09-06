"""Validate LCB-aligned conditional structure families on the beta draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_WORKTREE = ROOT.parent / "Fantasy_Football_App_target_tiers"
APP_DIR = APP_WORKTREE / "app"
APP_DB = ROOT.parent / "Fantasy_Football_App" / "app" / "Simulation.sqlite3"
RESULTS_DIR = STUDY_DIR / "results"
SHARED_STUDY = (
    ROOT / "research" / "studies" / "2026-08-24_sequential_shared_opportunity"
)
for import_path in (APP_DIR, SHARED_STUDY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from keeper_market import load_active_keeper_market  # noqa: E402
from zSequential_Target import (  # noqa: E402
    SEQUENTIAL_TARGET_VERSION,
    run_sequential_target_board,
)
from zSim_Helper import FootballSimulation  # noqa: E402


YEAR = 2026
LEAGUE = "beta"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POS_MIN = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
POS_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 2}
OWNED = {"Chase Brown": 34.0, "Bhayshul Tuten": 11.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=320)
    parser.add_argument("--batches", type=int, choices=(1, 2), default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(APP_DB)
    try:
        sim = FootballSimulation(
            conn,
            YEAR,
            LINEUP,
            SALARY_CAP,
            "final_ensemble",
            LEAGUE,
            sal_pred_actual="pred",
        )
        keeper_market = load_active_keeper_market(
            conn,
            sim,
            year=YEAR,
            league=LEAGUE,
            salary_source="predicted",
            owned_salary_map=OWNED,
        )
        remaining_market_budget = (
            NUM_TEAMS * SALARY_CAP - keeper_market["keeper_spend"]
        )
        remaining_market_slots = (
            NUM_TEAMS * ROSTER_SIZE - keeper_market["keeper_count"]
        )
        waiver_baselines = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )
        prior = (None, None, None)
        runtimes = []
        for batch in range(args.batches):
            started = time.perf_counter()
            prior = run_sequential_target_board(
                sim,
                {
                    "players": list(OWNED),
                    "salaries": list(OWNED.values()),
                },
                keeper_market["unavailable_keeper_players"],
                compute_budget=args.budget,
                roster_size=ROSTER_SIZE,
                lineup_require=LINEUP,
                pos_min_counts=POS_MIN,
                pos_max_counts=POS_MAX,
                waiver_baselines=waiver_baselines,
                remaining_market_budget=remaining_market_budget,
                remaining_market_slots=remaining_market_slots,
                use_selection_premium=False,
                random_seed=args.seed + batch,
                parallel_workers=args.workers,
                prior_results=prior[0],
                prior_summary=prior[1],
                prior_curves=prior[2],
                profile_curves=False,
            )
            runtimes.append(time.perf_counter() - started)
    finally:
        conn.close()

    results, summary, _ = prior
    targets = summary.get("structure_targets", [])
    target_names = {target["Player"] for target in targets}
    examples = summary.get("structure_paths", [])
    if not target_names:
        raise AssertionError("No positive confirmed LCB anchors were summarized.")
    if {path.get("Anchor") for path in examples} != target_names:
        raise AssertionError("Conditional examples do not match LCB anchors.")
    for path in examples:
        if path["Anchor"] not in {
            player["Player"] for player in path.get("Players", [])
        }:
            raise AssertionError(
                f"Conditional path omitted its anchor: {path['Anchor']}"
            )

    payload = {
        "calculation_version": SEQUENTIAL_TARGET_VERSION,
        "budget": args.budget,
        "batches": args.batches,
        "workers": args.workers,
        "batch_runtime_seconds": runtimes,
        "evidence_blocks": summary.get("evidence_blocks"),
        "structure_source": summary.get("structure_source"),
        "conditional_outcomes": summary.get("structure_plan_count"),
        "structure_runtime_seconds": summary.get("structure_runtime_seconds"),
        "targets": targets,
        "families": summary.get("structure_families", []),
        "examples": examples,
        "confirmed_board": results.loc[
            results.EvidenceStage.eq("Confirmed"),
            [
                "Player", "MarketPrice", "SequentialLCB80",
                "SequentialGain", "BlockPositiveRate", "Recommendation",
            ],
        ].to_dict("records"),
    }
    output_path = RESULTS_DIR / "beta_validation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "structure_source": payload["structure_source"],
        "conditional_outcomes": payload["conditional_outcomes"],
        "targets": [target["Player"] for target in targets],
        "families": [
            (family["Name"], family["Support"])
            for family in payload["families"][:3]
        ],
        "batch_runtime_seconds": runtimes,
    }, indent=2))


if __name__ == "__main__":
    main()
