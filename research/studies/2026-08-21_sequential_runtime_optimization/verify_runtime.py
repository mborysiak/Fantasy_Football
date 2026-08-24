"""Parity and timing checks for sequential rollout and deferred bid curves."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys
import tempfile

# Avoid native numerical oversubscription inside spawned candidate workers.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = ROOT.parent / "Fantasy_Football_App"
APP_DIR = APP_ROOT / "app"
SHADOW_DIR = ROOT / "research" / "studies" / "2026-08-21_bounded_app_shadow"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SHADOW_DIR) not in sys.path:
    sys.path.insert(0, str(SHADOW_DIR))

from zSequential_Target import (  # noqa: E402
    attach_sequential_curve_profiles,
    run_sequential_target_board,
)
from zSim_Helper import (  # noqa: E402
    DEFAULT_WAIVER_BASELINES,
    FootballSimulation,
    MANAGED_POS_MAX,
)
import run_shadow as shadow  # noqa: E402


LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POSITION_MINIMUMS = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
POSITION_MAXIMUMS = {**MANAGED_POS_MAX, "RB": 6, "WR": 6}
ROSTER_SIZE = 13
SEED = 3962081362


def run_board(
    *,
    profile_curves,
    curve_profile_only=False,
    prior_results=None,
    prior_summary=None,
    prior_curves=None,
    random_seed=SEED,
    parallel_workers=4,
    compute_budget=120,
):
    connection = sqlite3.connect(APP_DIR / "Simulation.sqlite3")
    try:
        sim = FootballSimulation(
            connection,
            2026,
            LINEUP_REQUIRE,
            298,
            "final_ensemble",
            "beta",
            sal_pred_actual="pred",
        )
        return run_sequential_target_board(
            sim,
            {"players": [], "salaries": []},
            [],
            compute_budget=compute_budget,
            roster_size=ROSTER_SIZE,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts=POSITION_MINIMUMS,
            pos_max_counts=POSITION_MAXIMUMS,
            waiver_baselines=DEFAULT_WAIVER_BASELINES,
            remaining_market_budget=12 * 298,
            remaining_market_slots=12 * ROSTER_SIZE,
            random_seed=random_seed,
            parallel_workers=parallel_workers,
            prior_results=prior_results,
            prior_summary=prior_summary,
            prior_curves=prior_curves,
            profile_curves=profile_curves,
            curve_profile_only=curve_profile_only,
        )
    finally:
        connection.close()


def verify_rollout_parity():
    scenario_name = "early_brown_tuten_gibbs"
    scenario = shadow.SCENARIOS[scenario_name]
    database_uri = f"file:{shadow.audit.APP_DB.as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    original = shadow.sequential.simulate_history_only_branch
    try:
        state = shadow.build_state(connection, scenario)

        def run_mode(fast_checks):
            def simulator(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs["fast_reinvestment_checks"] = bool(fast_checks)
                return original(*args, **kwargs)

            shadow.sequential.simulate_history_only_branch = simulator
            baseline, _, plans = shadow.run_arm(
                state,
                scenario_name,
                scenario,
                0,
                "baseline",
            )
            bounded, paths, _ = shadow.run_arm(
                state,
                scenario_name,
                scenario,
                0,
                "bounded",
                compiled_plan_bank=plans,
            )
            return bounded, paths

        with tempfile.TemporaryDirectory(prefix="ff-runtime-parity-") as temp_dir:
            shadow.RESULTS_DIR = Path(temp_dir)
            reference, reference_paths = run_mode(False)
            optimized, optimized_paths = run_mode(True)
        comparable = sorted(set(reference) - {"runtime_seconds"})
        for key in comparable:
            if reference[key] != optimized[key]:
                raise AssertionError(
                    f"Rollout result mismatch for {key}: "
                    f"{reference[key]!r} != {optimized[key]!r}"
                )
        pd.testing.assert_frame_equal(reference_paths, optimized_paths)
        return {
            "reference_seconds": float(reference["runtime_seconds"]),
            "optimized_seconds": float(optimized["runtime_seconds"]),
        }
    finally:
        shadow.sequential.simulate_history_only_branch = original
        connection.close()


def verify_deferred_curves():
    immediate_results, immediate_summary, immediate_curves = run_board(
        profile_curves=True,
    )
    market_results, market_summary, market_curves = run_board(
        profile_curves=False,
    )
    _, curve_summary, extra_curves = run_board(
        profile_curves=False,
        curve_profile_only=True,
        prior_results=market_results,
    )
    attached_results, attached_curves = attach_sequential_curve_profiles(
        market_results,
        market_curves,
        extra_curves,
    )
    pd.testing.assert_frame_equal(immediate_results, attached_results)
    curve_sort = ["Player", "Price"]
    pd.testing.assert_frame_equal(
        immediate_curves.sort_values(curve_sort).reset_index(drop=True),
        attached_curves.sort_values(curve_sort).reset_index(drop=True),
    )

    next_seed = SEED + 1
    after_profile_results, after_profile_summary, after_profile_curves = run_board(
        profile_curves=False,
        prior_results=immediate_results,
        prior_summary=immediate_summary,
        prior_curves=immediate_curves,
        random_seed=next_seed,
    )
    market_add_results, market_add_summary, market_add_curves = run_board(
        profile_curves=False,
        prior_results=market_results,
        prior_summary=market_summary,
        prior_curves=market_curves,
        random_seed=next_seed,
    )
    pd.testing.assert_frame_equal(after_profile_results, market_add_results)
    pd.testing.assert_frame_equal(after_profile_curves, market_add_curves)
    if after_profile_summary['curves_profiled']:
        raise AssertionError('Add Evidence did not defer prior max-bid profiles.')
    if after_profile_summary['evidence_seeds'] != market_add_summary['evidence_seeds']:
        raise AssertionError('Add Evidence changed retained evidence seeds.')

    return {
        "immediate_seconds": float(immediate_summary["runtime_seconds"]),
        "market_only_seconds": float(market_summary["runtime_seconds"]),
        "curve_only_seconds": float(curve_summary["runtime_seconds"]),
        "result_rows": int(len(immediate_results)),
        "market_curve_rows": int(len(market_curves)),
        "full_curve_rows": int(len(immediate_curves)),
        "add_evidence_seconds": float(
            after_profile_summary['last_batch_runtime_seconds']
        ),
    }


def main():
    rollout = verify_rollout_parity()
    curves = verify_deferred_curves()
    print("Rollout parity: exact")
    print(
        "Reference legality: "
        f"{rollout['reference_seconds']:.3f}s; optimized: "
        f"{rollout['optimized_seconds']:.3f}s"
    )
    print("Deferred-curve parity: exact")
    print("Add Evidence after profiling: exact market-only parity")
    print(
        "Immediate board: "
        f"{curves['immediate_seconds']:.3f}s; market-only board: "
        f"{curves['market_only_seconds']:.3f}s; deferred curves: "
        f"{curves['curve_only_seconds']:.3f}s"
    )
    print(
        f"Rows: {curves['result_rows']} board / "
        f"{curves['market_curve_rows']} market anchors / "
        f"{curves['full_curve_rows']} full anchors"
    )
    print(
        "Market-only Add Evidence batch: "
        f"{curves['add_evidence_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
