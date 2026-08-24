"""Paired production shadow test for bounded auction-budget reinvestment.

The baseline and candidate arms share the same evidence seed, hidden auction
tapes, and compiled completion plans.  The only policy difference is the
``budget_reinvestment`` flag passed to the production branch simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import types

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
SOURCE_STUDY = (
    ROOT / "research" / "studies" / "2026-08-20_bijan_fourth_rb_audit"
)
REINVESTMENT_STUDY = (
    ROOT / "research" / "studies" / "2026-08-20_sequential_budget_reinvestment"
)
RESULTS_DIR = STUDY_DIR / "results"
if str(SOURCE_STUDY) not in sys.path:
    sys.path.insert(0, str(SOURCE_STUDY))
if str(REINVESTMENT_STUDY) not in sys.path:
    sys.path.insert(0, str(REINVESTMENT_STUDY))

import run_audit as audit  # noqa: E402
import run_experiment as reinvestment_study  # noqa: E402


sequential = audit.sequential
FootballSimulation = audit.FootballSimulation

SCENARIOS = {
    "early_brown_tuten_gibbs": {
        "stage": "early",
        "candidate": "Jahmyr Gibbs",
        "candidate_price": 110,
        "fixed_salaries": {
            "Chase Brown": 34.0,
            "Bhayshul Tuten": 11.0,
        },
    },
    "middle_balanced_bowers": {
        "stage": "middle",
        "candidate": "Brock Bowers",
        "candidate_price": 51,
        "fixed_salaries": {
            "Brock Purdy": 3.0,
            "Jahmyr Gibbs": 110.0,
            "Chase Brown": 34.0,
            "Bhayshul Tuten": 11.0,
            "KC Concepcion": 4.0,
            "Makai Lemon": 7.0,
        },
    },
    "late_cap_squeeze_pitts": {
        "stage": "late_cap_squeeze",
        "candidate": "Kyle Pitts",
        "candidate_price": 11,
        "fixed_salaries": {
            "Brock Purdy": 3.0,
            "Jahmyr Gibbs": 110.0,
            "Bijan Robinson": 105.0,
            "Chase Brown": 34.0,
            "Bhayshul Tuten": 11.0,
            "KC Concepcion": 4.0,
            "Makai Lemon": 7.0,
        },
    },
}


def build_state(conn, scenario):
    sim = FootballSimulation(
        conn,
        audit.YEAR,
        audit.LINEUP_REQUIRE,
        audit.SALARY_CAP,
        audit.PRED_VERSION,
        audit.LEAGUE,
        sal_pred_actual=audit.SALARY_SOURCE,
    )
    sim.load_weekly_template_profiles()
    fixed_salaries = dict(scenario["fixed_salaries"])
    required = set(fixed_salaries) | {scenario["candidate"]}
    missing = sorted(required - set(sim.player_data.player))
    if missing:
        raise ValueError("Scenario players missing from pool: " + ", ".join(missing))

    keepers = pd.read_sql_query(
        """
        SELECT player, player_key, keeper_salary
        FROM League_Keepers
        WHERE year = :year AND league = :league
        """,
        conn,
        params={"year": audit.YEAR, "league": audit.LEAGUE},
    )
    canonical_by_key = sim.player_data.set_index("player_key").player
    keepers["canonical_player"] = keepers.player_key.map(canonical_by_key)
    if keepers.canonical_player.isna().any():
        raise ValueError("Keeper keys failed to map to the current player pool.")
    keeper_salary_map = dict(zip(
        keepers.canonical_player,
        keepers.keeper_salary.astype(float),
    ))
    nonkeeper_fixed_spend = sum(
        salary
        for player, salary in fixed_salaries.items()
        if player not in keeper_salary_map
    )
    remaining_market_budget = float(
        audit.NUM_TEAMS * audit.SALARY_CAP
        - sum(keeper_salary_map.values())
        - nonkeeper_fixed_spend
    )
    remaining_market_slots = int(
        audit.NUM_TEAMS * audit.ROSTER_SIZE
        - len(keeper_salary_map)
        - sum(player not in keeper_salary_map for player in fixed_salaries)
    )
    return {
        "sim": sim,
        "to_add": {
            "players": list(fixed_salaries),
            "salaries": list(fixed_salaries.values()),
        },
        "to_drop": sorted(set(keeper_salary_map) - set(fixed_salaries)),
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "waiver_baselines": sim.estimate_waiver_baselines(
            num_teams=audit.NUM_TEAMS,
            roster_size=audit.ROSTER_SIZE,
        ),
    }


def run_arm(
    state,
    scenario_name,
    scenario,
    variation,
    arm,
    compiled_plan_bank=None,
):
    sim = state["sim"]
    production_simulator = sequential.simulate_history_only_branch
    production_plan_solver = sequential.solve_history_only_plan
    production_required_solver = sim.solve_managed_roster_with_required_players
    captured_paths = []
    captured_plans = []
    plan_cursor = 0

    if arm == "baseline":
        def plan_solver(*args, **kwargs):
            plan = production_plan_solver(*args, **kwargs)
            captured_plans.append(plan)
            return plan
    elif arm == "bounded":
        if compiled_plan_bank is None:
            raise ValueError("Bounded arm requires the paired baseline plan bank.")

        def plan_solver(*args, **kwargs):
            del args, kwargs
            nonlocal plan_cursor
            if plan_cursor >= len(compiled_plan_bank):
                raise RuntimeError("Bounded arm requested too many compiled plans.")
            plan = compiled_plan_bank[plan_cursor]
            plan_cursor += 1
            return plan
    else:
        raise ValueError(f"Unknown arm: {arm}")

    def capture_simulator(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["budget_reinvestment"] = arm == "bounded"
        branch = production_simulator(*args, **kwargs)
        captured_paths.append({
            "branch": "buy" if kwargs.get("force_buy") else "pass",
            "complete": bool(branch.get("complete")),
            "failure_reason": branch.get("failure_reason"),
            "salary_spend": float(branch.get("salary_spend", np.nan)),
            "unused_salary": float(branch.get("final_unused_salary", np.nan)),
            "cap_legal": bool(branch.get("final_cap_legal", False)),
            "position_legal": bool(branch.get("final_position_legal", False)),
            "top_n_legal": bool(branch.get("final_top_n_legal", False)),
            "bounded_triggers": int(branch.get("bounded_triggers", 0)),
            "bounded_swaps": int(branch.get("bounded_swaps", 0)),
            "bounded_rebuilds": int(branch.get("bounded_rebuilds", 0)),
            "max_projected_slack": float(
                branch.get("max_projected_slack", 0.0)
            ),
            "roster": " | ".join(branch.get("roster", ())),
            "salary_map": json.dumps(
                dict(branch.get("salary_map", ())),
                sort_keys=True,
            ),
        })
        return branch

    audit.CANDIDATE = scenario["candidate"]
    audit.FIXED_SALARIES = dict(scenario["fixed_salaries"])
    audit.RESULTS_DIR = RESULTS_DIR
    label = f"{scenario_name}_variation{variation}_{arm}"
    sequential.solve_history_only_plan = plan_solver
    sequential.simulate_history_only_branch = capture_simulator
    sim.solve_managed_roster_with_required_players = types.MethodType(
        reinvestment_study.stable_required_roster_solve,
        sim,
    )
    started = time.perf_counter()
    try:
        result = audit.run_case(
            sim,
            to_add=state["to_add"],
            to_drop=state["to_drop"],
            remaining_market_budget=state["remaining_market_budget"],
            remaining_market_slots=state["remaining_market_slots"],
            waiver_baselines=state["waiver_baselines"],
            candidate_price=int(scenario["candidate_price"]),
            label=label,
            variation=int(variation),
            enforce_top_n=True,
            use_selection_premium=False,
            profile_bid=False,
            capture_paths=False,
        )
    finally:
        sequential.solve_history_only_plan = production_plan_solver
        sequential.simulate_history_only_branch = production_simulator
        sim.solve_managed_roster_with_required_players = production_required_solver

    expected_paths = 2 * int(result["requested_paths"])
    if len(captured_paths) != expected_paths:
        raise AssertionError(
            f"{label}: captured {len(captured_paths)} branches; "
            f"expected {expected_paths}."
        )
    if arm == "bounded" and plan_cursor != len(compiled_plan_bank):
        raise AssertionError(
            f"{label}: replayed {plan_cursor} plans; "
            f"expected {len(compiled_plan_bank)}."
        )

    paths = pd.DataFrame(captured_paths)
    paths.insert(0, "path_index", np.arange(len(paths)) // 2)
    paths.insert(0, "arm", arm)
    paths.insert(0, "variation", int(variation))
    paths.insert(0, "stage", scenario["stage"])
    paths.insert(0, "scenario", scenario_name)
    decision = {
        "scenario": scenario_name,
        "stage": scenario["stage"],
        "candidate": scenario["candidate"],
        "candidate_price": int(scenario["candidate_price"]),
        "fixed_count": len(scenario["fixed_salaries"]),
        "fixed_spend": float(sum(scenario["fixed_salaries"].values())),
        "variation": int(variation),
        "arm": arm,
        "recommendation": result.get("recommendation"),
        "gain": float(result.get("SequentialGain", np.nan)),
        "lcb80": float(result.get("SequentialLCB80", np.nan)),
        "buy_completion": float(result.get("BuyCompletion", np.nan)),
        "pass_completion": float(result.get("PassCompletion", np.nan)),
        "paired_rate": float(result.get("PairedRate", np.nan)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return decision, paths, captured_plans


def summarize_paths(paths):
    completed = paths.loc[paths.complete].copy()
    summary = (
        paths.groupby(["scenario", "stage", "arm", "branch"], as_index=False)
        .agg(
            paths=("complete", "size"),
            completion_rate=("complete", "mean"),
            failures=("complete", lambda values: int((~values).sum())),
        )
    )
    complete_summary = (
        completed.groupby(
            ["scenario", "stage", "arm", "branch"],
            as_index=False,
        )
        .agg(
            mean_spend=("salary_spend", "mean"),
            mean_unused=("unused_salary", "mean"),
            median_unused=("unused_salary", "median"),
            p90_unused=("unused_salary", lambda values: np.percentile(values, 90)),
            max_unused=("unused_salary", "max"),
            cap_legal_rate=("cap_legal", "mean"),
            position_legal_rate=("position_legal", "mean"),
            top_n_legal_rate=("top_n_legal", "mean"),
            mean_bounded_swaps=("bounded_swaps", "mean"),
            mean_bounded_rebuilds=("bounded_rebuilds", "mean"),
        )
    )
    return summary.merge(
        complete_summary,
        on=["scenario", "stage", "arm", "branch"],
        how="left",
        validate="one_to_one",
    )


def evaluate_gates(decisions, path_summary):
    candidate = path_summary[path_summary.arm.eq("bounded")]
    legality_columns = [
        "cap_legal_rate",
        "position_legal_rate",
        "top_n_legal_rate",
    ]
    legality_pass = bool(
        candidate[legality_columns].fillna(0.0).ge(1.0 - 1e-12).all().all()
    )

    paired = path_summary.pivot(
        index=["scenario", "stage", "branch"],
        columns="arm",
        values=["completion_rate", "mean_unused"],
    ).reset_index()
    paired.columns = [
        "_".join(str(value) for value in column if value).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in paired.columns
    ]
    paired["completion_delta"] = (
        paired["completion_rate_bounded"]
        - paired["completion_rate_baseline"]
    )
    paired["unused_delta"] = (
        paired["mean_unused_bounded"]
        - paired["mean_unused_baseline"]
    )
    completion_pass = bool(paired.completion_delta.min() >= -0.01)
    pass_rows = paired[paired.branch.eq("pass")]
    spend_pass = bool(
        pass_rows.unused_delta.mean() <= 1e-9
        and (pass_rows.unused_delta < -1.0).any()
    )

    decision_pivot = decisions.pivot(
        index=["scenario", "stage", "candidate", "variation"],
        columns="arm",
        values=["recommendation", "gain", "lcb80"],
    ).reset_index()
    decision_pivot.columns = [
        "_".join(str(value) for value in column if value).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in decision_pivot.columns
    ]
    gates = {
        "completed_roster_legality": legality_pass,
        "completion_not_materially_worse": completion_pass,
        "pass_path_budget_reinvestment": spend_pass,
    }
    return gates, paired, decision_pivot


def accumulate_decision_evidence(variations):
    rows = []
    for scenario_name, scenario in SCENARIOS.items():
        for arm in ("baseline", "bounded"):
            evidence = []
            for variation in variations:
                path = RESULTS_DIR / (
                    f"decision_{scenario_name}_variation{variation}_{arm}.json"
                )
                with path.open(encoding="utf-8") as handle:
                    evidence.append(json.load(handle))
            combined = evidence[0]
            for fresh in evidence[1:]:
                combined = sequential.combine_sequential_evidence_row(
                    combined,
                    fresh,
                )
            if (
                min(combined["BuyCompletion"], combined["PassCompletion"]) < 90.0
                or combined["PairedRate"] < 90.0
            ):
                recommendation = "INCOMPLETE"
            elif combined["SequentialLCB80"] > 0:
                recommendation = "TARGET"
            elif combined["SequentialGain"] > 0:
                recommendation = "WATCH"
            else:
                recommendation = "PASS"
            rows.append({
                "scenario": scenario_name,
                "stage": scenario["stage"],
                "candidate": scenario["candidate"],
                "candidate_price": scenario["candidate_price"],
                "arm": arm,
                "recommendation": recommendation,
                "gain": combined["SequentialGain"],
                "lcb80": combined["SequentialLCB80"],
                "evidence_blocks": combined["EvidenceBlocks"],
                "buy_completion": combined["BuyCompletion"],
                "pass_completion": combined["PassCompletion"],
                "paired_rate": combined["PairedRate"],
            })
    return pd.DataFrame(rows)


def main(variations):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    database_uri = f"file:{audit.APP_DB.as_posix()}?mode=ro"
    conn = sqlite3.connect(database_uri, uri=True)
    decision_rows = []
    path_frames = []
    try:
        for scenario_name, scenario in SCENARIOS.items():
            print(
                f"scenario={scenario_name} candidate={scenario['candidate']} "
                f"price={scenario['candidate_price']}",
                flush=True,
            )
            state = build_state(conn, scenario)
            for variation in variations:
                baseline, baseline_paths, plan_bank = run_arm(
                    state,
                    scenario_name,
                    scenario,
                    variation,
                    "baseline",
                )
                bounded, bounded_paths, _ = run_arm(
                    state,
                    scenario_name,
                    scenario,
                    variation,
                    "bounded",
                    compiled_plan_bank=plan_bank,
                )
                decision_rows.extend([baseline, bounded])
                path_frames.extend([baseline_paths, bounded_paths])
                print(
                    f"  variation={variation} "
                    f"baseline={baseline['recommendation']} "
                    f"({baseline['gain']:+.2f}, {baseline['lcb80']:+.2f}) "
                    f"bounded={bounded['recommendation']} "
                    f"({bounded['gain']:+.2f}, {bounded['lcb80']:+.2f})",
                    flush=True,
                )
    finally:
        conn.close()

    decisions = pd.DataFrame(decision_rows)
    paths = pd.concat(path_frames, ignore_index=True)
    path_summary = summarize_paths(paths)
    gates, paired_paths, paired_decisions = evaluate_gates(
        decisions,
        path_summary,
    )
    run_tag = "variation" + "_".join(str(value) for value in variations)
    decisions.to_csv(
        RESULTS_DIR / f"{run_tag}_decision_summary.csv",
        index=False,
    )
    paths.to_csv(RESULTS_DIR / f"{run_tag}_all_paths.csv", index=False)
    path_summary.to_csv(
        RESULTS_DIR / f"{run_tag}_path_summary.csv",
        index=False,
    )
    paired_paths.to_csv(
        RESULTS_DIR / f"{run_tag}_paired_path_deltas.csv",
        index=False,
    )
    paired_decisions.to_csv(
        RESULTS_DIR / f"{run_tag}_paired_decision_deltas.csv",
        index=False,
    )
    metadata = {
        "variations": [int(value) for value in variations],
        "scenarios": SCENARIOS,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "comparison": (
            "production simulate_history_only_branch with "
            "budget_reinvestment=False versus True"
        ),
        "paired_inputs": [
            "evidence seed",
            "hidden auction tape",
            "managed-value block",
            "compiled completion plan",
        ],
        "stable_required_roster_solver": True,
    }
    with (RESULTS_DIR / f"{run_tag}_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print("\nGates", flush=True)
    for gate, passed in gates.items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}", flush=True)
    print("\nPaired path deltas", flush=True)
    print(paired_paths.to_string(index=False), flush=True)


def aggregate_results(variations):
    decision_frames = []
    path_frames = []
    for variation in variations:
        tag = f"variation{int(variation)}"
        decision_frames.append(pd.read_csv(
            RESULTS_DIR / f"{tag}_decision_summary.csv"
        ))
        path_frames.append(pd.read_csv(
            RESULTS_DIR / f"{tag}_all_paths.csv"
        ))
    decisions = pd.concat(decision_frames, ignore_index=True)
    paths = pd.concat(path_frames, ignore_index=True)
    path_summary = summarize_paths(paths)
    gates, paired_paths, paired_decisions = evaluate_gates(
        decisions,
        path_summary,
    )
    accumulated = accumulate_decision_evidence(variations)
    accumulated_pivot = accumulated.pivot(
        index=["scenario", "stage", "candidate", "candidate_price"],
        columns="arm",
        values=["recommendation", "gain", "lcb80"],
    ).reset_index()
    accumulated_pivot.columns = [
        "_".join(str(value) for value in column if value).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in accumulated_pivot.columns
    ]
    anchor_scenarios = {
        "early_brown_tuten_gibbs",
        "late_cap_squeeze_pitts",
    }
    anchor_rows = accumulated_pivot[
        accumulated_pivot.scenario.isin(anchor_scenarios)
    ]
    gates["accumulated_anchor_recommendations_stable"] = bool(
        (
            anchor_rows.recommendation_baseline
            == anchor_rows.recommendation_bounded
        ).all()
    )
    decisions.to_csv(RESULTS_DIR / "decision_summary.csv", index=False)
    paths.to_csv(RESULTS_DIR / "all_paths.csv", index=False)
    path_summary.to_csv(RESULTS_DIR / "path_summary.csv", index=False)
    paired_paths.to_csv(RESULTS_DIR / "paired_path_deltas.csv", index=False)
    paired_decisions.to_csv(
        RESULTS_DIR / "paired_decision_deltas.csv",
        index=False,
    )
    accumulated.to_csv(
        RESULTS_DIR / "accumulated_decision_summary.csv",
        index=False,
    )
    accumulated_pivot.to_csv(
        RESULTS_DIR / "accumulated_decision_deltas.csv",
        index=False,
    )
    metadata = {
        "variations": [int(value) for value in variations],
        "scenarios": SCENARIOS,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "comparison": (
            "production simulate_history_only_branch with "
            "budget_reinvestment=False versus True"
        ),
        "paired_inputs": [
            "evidence seed",
            "hidden auction tape",
            "managed-value block",
            "compiled completion plan",
        ],
        "stable_required_roster_solver": True,
        "behavioral_diagnostic": {
            "middle_bowers_recommendation_changed": bool(
                accumulated_pivot.loc[
                    accumulated_pivot.scenario.eq("middle_balanced_bowers"),
                    "recommendation_baseline",
                ].iloc[0]
                != accumulated_pivot.loc[
                    accumulated_pivot.scenario.eq("middle_balanced_bowers"),
                    "recommendation_bounded",
                ].iloc[0]
            ),
            "interpretation": (
                "The middle-state Bowers decision is an opportunity-cost "
                "sensitivity check, not a promotion-stability anchor."
            ),
        },
    }
    with (RESULTS_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variations", nargs="+", type=int, default=[0, 1, 2, 14])
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate_results(args.variations)
    else:
        main(args.variations)
