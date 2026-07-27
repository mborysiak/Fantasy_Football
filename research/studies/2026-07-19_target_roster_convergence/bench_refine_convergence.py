import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
APP_DIR = REPO_ROOT.parent / "Fantasy_Football_App" / "app"
APP_DB = APP_DIR / "Simulation.sqlite3"
RESULTS_DIR = STUDY_DIR / "results"
SEED = 20260719

sys.path.insert(0, str(APP_DIR))
from zSim_Helper import FootballSimulation  # noqa: E402


def build_sim():
    conn = sqlite3.connect(APP_DB)
    sim = FootballSimulation(
        conn,
        2026,
        {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2},
        300,
        pred_vers="final_ensemble",
        league="beta",
        sal_pred_actual="pred",
    )
    sim.load_weekly_template_profiles()
    return conn, sim


def run_case(label, max_swaps, to_add, market_budget, market_slots, num_iters):
    original_refine = FootballSimulation.refine_managed_roster_to_convergence
    original_contrib = FootballSimulation.managed_roster_buy_pass_contributions
    stats = {
        "refine_seconds": 0.0,
        "accepted_swaps": [],
        "contribution_seconds": 0.0,
        "contribution_calls": 0,
    }

    def timed_refinement(self, *args, **kwargs):
        kwargs["max_swaps"] = int(max_swaps)
        start = time.perf_counter()
        mask, info = original_refine(self, *args, **kwargs)
        stats["refine_seconds"] += time.perf_counter() - start
        stats["accepted_swaps"].append(int(info["accepted_swaps"]))
        return mask, info

    def timed_contribution(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_contrib(self, *args, **kwargs)
        stats["contribution_seconds"] += time.perf_counter() - start
        stats["contribution_calls"] += 1
        return result

    FootballSimulation.refine_managed_roster_to_convergence = timed_refinement
    FootballSimulation.managed_roster_buy_pass_contributions = timed_contribution
    conn = None
    try:
        conn, sim = build_sim()
        np.random.seed(SEED)
        start = time.perf_counter()
        sim.run_sim(
            to_add,
            [],
            int(num_iters),
            remaining_market_budget=market_budget,
            remaining_market_slots=market_slots,
        )
        elapsed = time.perf_counter() - start
    finally:
        FootballSimulation.refine_managed_roster_to_convergence = original_refine
        FootballSimulation.managed_roster_buy_pass_contributions = original_contrib
        if conn is not None:
            conn.close()

    swaps = np.asarray(stats["accepted_swaps"], dtype=int)
    return {
        "label": label,
        "max_swaps": int(max_swaps),
        "trials": int(num_iters),
        "elapsed_seconds": float(elapsed),
        "milliseconds_per_trial": float(elapsed / num_iters * 1000),
        "refine_seconds": float(stats["refine_seconds"]),
        "refine_share": float(stats["refine_seconds"] / elapsed),
        "accepted_swaps_total": int(swaps.sum()),
        "accepted_swaps_mean": float(swaps.mean()),
        "accepted_swaps_max": int(swaps.max()),
        "zero_swap_trials": int((swaps == 0).sum()),
        "contribution_seconds": float(stats["contribution_seconds"]),
        "contribution_calls": int(stats["contribution_calls"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=16)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")

    cases = [
        {
            "name": "empty_roster",
            "to_add": {"players": [], "salaries": []},
            "market_budget": 12 * 300,
            "market_slots": 12 * 13,
        },
        {
            "name": "mid_draft_roster",
            "to_add": {
                "players": [
                    "Bucky Irving",
                    "Chase Brown",
                    "Josh Allen",
                    "Ja'Marr Chase",
                ],
                "salaries": [12, 19, 45, 105],
            },
            "market_budget": 12 * 300 - 181 - 900,
            "market_slots": 12 * 13 - 4 - 42,
        },
    ]

    output = {"seed": SEED, "cases": {}}
    for case in cases:
        old = run_case(
            "old behavior",
            1,
            case["to_add"],
            case["market_budget"],
            case["market_slots"],
            args.trials,
        )
        converged = run_case(
            "converged behavior",
            12,
            case["to_add"],
            case["market_budget"],
            case["market_slots"],
            args.trials,
        )
        output["cases"][case["name"]] = {
            "old": old,
            "converged": converged,
            "runtime_ratio": (
                converged["elapsed_seconds"] / old["elapsed_seconds"]
            ),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
