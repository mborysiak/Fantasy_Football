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


def run_case(label, num_iters, max_swaps):
    swap_counts = []
    original = FootballSimulation.refine_managed_roster_to_convergence

    def counted_refinement(self, *args, **kwargs):
        kwargs["max_swaps"] = int(max_swaps)
        mask, info = original(self, *args, **kwargs)
        swap_counts.append(int(info["accepted_swaps"]))
        return mask, info

    FootballSimulation.refine_managed_roster_to_convergence = counted_refinement
    conn = None
    try:
        conn, sim = build_sim()
        np.random.seed(SEED)
        start = time.perf_counter()
        results = sim.run_sim(
            {"players": [], "salaries": []},
            [],
            int(num_iters),
            remaining_market_budget=12 * 300,
            remaining_market_slots=12 * 13,
        )
        elapsed = time.perf_counter() - start
        summary = sim.get_managed_summary()
    finally:
        FootballSimulation.refine_managed_roster_to_convergence = original
        if conn is not None:
            conn.close()

    counts = np.asarray(swap_counts, dtype=int)
    top_players = results[
        ["player", "SelectionCounts", "ExpectedRosterGain"]
    ].head(5)
    output = {
        "label": label,
        "max_swaps": int(max_swaps),
        "seed": SEED,
        "requested_trials": int(num_iters),
        "completed_trials": int(summary["trials"]),
        "elapsed_seconds": float(elapsed),
        "accepted_swaps_mean": float(counts.mean()),
        "accepted_swaps_max": int(counts.max()),
        "hit_cap_count": int((counts >= int(max_swaps)).sum()),
        "season_ev": float(summary["season_ev"]),
        "season_p10": float(summary["season_p10"]),
        "season_p90": float(summary["season_p90"]),
        "salary_spend": float(summary["salary_spend"]),
        "top_players": top_players.to_dict(orient="records"),
    }
    print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")

    old = run_case("old behavior", args.trials, max_swaps=1)
    converged = run_case("converged behavior", args.trials, max_swaps=12)
    output = {
        "old": old,
        "converged": converged,
        "season_ev_change": converged["season_ev"] - old["season_ev"],
        "season_p10_change": converged["season_p10"] - old["season_p10"],
        "season_p90_change": converged["season_p90"] - old["season_p90"],
        "runtime_ratio": converged["elapsed_seconds"] / old["elapsed_seconds"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "verification.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison": output}, indent=2))


if __name__ == "__main__":
    main()
