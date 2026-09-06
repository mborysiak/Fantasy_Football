"""Fresh-process wall-time check for the complete Sequential Target board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time


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


LINEUP_REQUIRE = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
POS_MIN = {"QB": 2, "RB": 4, "WR": 4, "TE": 1}
POS_MAX = {"QB": 3, "RB": 6, "WR": 6, "TE": 2}
FIXED_SALARIES = {"Drake Maye": 18.0, "De'Von Achane": 47.0}
SOURCE_SUFFIX = {"predicted": "pred", "actual": "_actual"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-swaps", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--joint-mode",
        choices=("full_exact", "shortlist"),
        default="full_exact",
    )
    parser.add_argument("--confirm-only", action="store_true")
    parser.add_argument(
        "--source",
        choices=tuple(SOURCE_SUFFIX),
        default="predicted",
    )
    parser.add_argument("--compute-budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--expanded-candidate-limit", type=int, default=64)
    parser.add_argument("--confirm-limit", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sequential.DEFAULT_SEQUENTIAL_SCREEN_JOINT_SWAPS = (
        0 if args.confirm_only else args.joint_swaps
    )
    sequential.DEFAULT_SEQUENTIAL_CONFIRM_JOINT_SWAPS = args.joint_swaps
    sequential.DEFAULT_SEQUENTIAL_SCREEN_JOINT_MODE = args.joint_mode
    sequential.DEFAULT_SEQUENTIAL_CONFIRM_JOINT_MODE = args.joint_mode

    conn = sqlite3.connect(APP_DB)
    try:
        sim = FootballSimulation(
            conn,
            2026,
            LINEUP_REQUIRE,
            298,
            "final_ensemble",
            "nv",
            sal_pred_actual=SOURCE_SUFFIX[args.source],
        )
        sim.load_weekly_template_profiles()
        keeper_market = load_active_keeper_market(
            conn,
            sim,
            year=2026,
            league="nv",
            salary_source=args.source,
            owned_salary_map=FIXED_SALARIES,
        )
        waiver_baselines = sim.estimate_waiver_baselines(
            num_teams=12,
            roster_size=13,
        )
        started = time.perf_counter()
        board, summary, curves = sequential.run_sequential_target_board(
            sim,
            {
                "players": list(FIXED_SALARIES),
                "salaries": list(FIXED_SALARIES.values()),
            },
            list(keeper_market["unavailable_keeper_players"]),
            compute_budget=args.compute_budget,
            require_top_n=12,
            enforce_top_n=True,
            roster_size=13,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts=POS_MIN,
            pos_max_counts=POS_MAX,
            waiver_baselines=waiver_baselines,
            remaining_market_budget=12 * 298 - keeper_market["keeper_spend"],
            remaining_market_slots=12 * 13 - keeper_market["keeper_count"],
            use_selection_premium=(args.source == "predicted"),
            candidate_limit=args.candidate_limit,
            expanded_candidate_limit=args.expanded_candidate_limit,
            confirm_limit=args.confirm_limit,
            parallel_workers=args.workers,
            random_seed=args.seed,
            profile_curves=False,
        )
        wall_seconds = time.perf_counter() - started
    finally:
        conn.close()

    player_column = "player" if "player" in board.columns else "Player"
    keeper_hits = sorted(
        set(board[player_column])
        & set(keeper_market["unavailable_keeper_players"])
    )
    if keeper_hits:
        raise AssertionError(
            "Target board included unavailable NV keepers: "
            + ", ".join(keeper_hits)
        )

    if not args.joint_swaps:
        label = "baseline"
    elif args.confirm_only:
        label = f"{args.joint_mode}_confirm_only"
    else:
        label = args.joint_mode
    source_prefix = "" if args.source == "predicted" else "actual_"
    run_label = (
        f"{source_prefix}{label}_b{args.compute_budget}"
        f"_x{args.expanded_candidate_limit}"
        f"_w{args.workers}"
    )
    board.to_csv(RESULTS_DIR / f"board_{run_label}.csv", index=False)
    output = {
        "arm": label,
        "salary_source": args.source,
        "joint_swaps": args.joint_swaps,
        "joint_mode": args.joint_mode,
        "screen_joint_swaps": (
            sequential.DEFAULT_SEQUENTIAL_SCREEN_JOINT_SWAPS
        ),
        "confirm_joint_swaps": (
            sequential.DEFAULT_SEQUENTIAL_CONFIRM_JOINT_SWAPS
        ),
        "compute_budget": args.compute_budget,
        "workers": args.workers,
        "candidate_limit": args.candidate_limit,
        "expanded_candidate_limit": args.expanded_candidate_limit,
        "confirm_limit": args.confirm_limit,
        "seed": args.seed,
        "wall_seconds": wall_seconds,
        "reported_runtime_seconds": summary["runtime_seconds"],
        "candidate_count": summary["candidate_count"],
        "confirmed_count": summary["confirmed_count"],
        "screen_paths": summary["screen_paths"],
        "confirm_paths": summary["confirm_paths"],
        "construction_contexts_total": summary[
            "construction_contexts_total"
        ],
        "parallel_workers_used": summary["parallel_workers_used"],
        "parallel_fallback": summary["parallel_fallback"],
        "curves_rows": len(curves),
        "keeper_count": keeper_market["keeper_count"],
        "keeper_spend": keeper_market["keeper_spend"],
        "unavailable_keeper_count": len(
            keeper_market["unavailable_keeper_players"]
        ),
    }
    generic_label = (
        label if args.source == "predicted" else f"actual_{label}"
    )
    board.to_csv(
        RESULTS_DIR / f"board_{generic_label}.csv",
        index=False,
    )
    (RESULTS_DIR / f"board_timing_{generic_label}.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"board_timing_{run_label}.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
