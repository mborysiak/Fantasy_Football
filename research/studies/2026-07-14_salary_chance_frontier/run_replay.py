"""Feasibility-first replay of auction salary chance constraints.

The optimizer maximizes the existing managed-value objective while requiring the
roster to fit the cap in a specified fraction of normalized five-draw salary markets.
Affordability is evaluated on salary markets that were not used by the optimizer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cvxopt import matrix


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
CURRENT_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_current_salary_buffer_replay"
    / "run_replay.py"
)
CHANCE_LEVELS = (0.60, 0.70, 0.80, 0.90)
SALARY_DRAWS_PER_MARKET = 5
SOLVER_CAP_MARGIN = 0.01
AUDIT_SALARY_TOLERANCE = 1e-6


def load_current_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_salary_chance_current_runner", CURRENT_RUNNER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import current salary replay: {CURRENT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


current = load_current_runner()
base = current.base


def build_normalized_market_bank(
    sim: Any,
    salary_draws: np.ndarray,
    remaining_budget: float,
    remaining_slots: int,
) -> np.ndarray:
    """Average five marginal draws and reconcile every market to league dollars."""
    count = salary_draws.shape[1] // SALARY_DRAWS_PER_MARKET
    if count <= 0:
        raise ValueError("Not enough salary draws to build a five-draw market.")
    trimmed = np.asarray(
        salary_draws[:, : count * SALARY_DRAWS_PER_MARKET], dtype=np.float64
    )
    raw = trimmed.reshape(
        len(trimmed), count, SALARY_DRAWS_PER_MARKET
    ).mean(axis=2)
    markets = base.normalize_market_draws(
        sim,
        raw,
        remaining_budget,
        remaining_slots,
    )
    if not np.isfinite(markets).all() or (markets < 1.0 - 1e-8).any():
        raise AssertionError("A normalized salary market is invalid.")
    return np.asarray(markets, dtype=np.float64)


def chance_counts(level: float, scenario_count: int) -> tuple[int, int]:
    required_hits = int(math.ceil(level * scenario_count - 1e-12))
    return required_hits, scenario_count - required_hits


def solve_chance_roster(
    sim: Any,
    managed_values: np.ndarray,
    static: dict[str, Any],
    scenario_markets: np.ndarray,
    chance_level: float,
    roster_size: int,
    salary_cap: float,
) -> dict[str, Any] | None:
    """Solve the exact sampled chance constraint with binary miss indicators."""
    scenario_markets = np.asarray(scenario_markets, dtype=np.float64)
    if scenario_markets.ndim != 2:
        raise ValueError("Scenario markets must be players by scenarios.")
    num_players, num_scenarios = scenario_markets.shape
    if num_players != len(managed_values):
        raise ValueError("Managed values and salary scenarios must align.")

    required_hits, allowed_misses = chance_counts(chance_level, num_scenarios)
    solver_cap = salary_cap - SOLVER_CAP_MARGIN
    # A scenario-specific tight upper bound makes z_s=1 sufficient for every
    # possible 13-player roster without using a numerically loose constant.
    top_spend = np.partition(
        scenario_markets,
        kth=num_players - roster_size,
        axis=0,
    )[-roster_size:, :].sum(axis=0)
    big_m = np.maximum(top_spend - solver_cap, 0.0)

    static_g = np.asarray(static["G_static"], dtype=np.float64)
    static_h = np.asarray(static["h_static"], dtype=np.float64).reshape(-1, 1)
    static_extended = np.hstack(
        [static_g, np.zeros((len(static_g), num_scenarios), dtype=np.float64)]
    )

    scenario_g = np.zeros(
        (num_scenarios, num_players + num_scenarios), dtype=np.float64
    )
    scenario_g[:, :num_players] = scenario_markets.T
    scenario_g[:, num_players:] = -np.diag(big_m)
    miss_count_g = np.zeros((1, num_players + num_scenarios), dtype=np.float64)
    miss_count_g[0, num_players:] = 1.0
    g = np.vstack([scenario_g, miss_count_g, static_extended])
    h = np.vstack(
        [
            np.full((num_scenarios, 1), solver_cap, dtype=np.float64),
            [[float(allowed_misses)]],
            static_h,
        ]
    )

    a_static = np.asarray(static["A"], dtype=np.float64)
    a = np.hstack(
        [a_static, np.zeros((a_static.shape[0], num_scenarios), dtype=np.float64)]
    )
    b = np.asarray(static["b"], dtype=np.float64)
    objective = np.concatenate(
        [-np.asarray(managed_values, dtype=np.float64), np.zeros(num_scenarios)]
    )
    status, solution = sim.solve_ilp(
        matrix(objective, tc="d"),
        matrix(g, tc="d"),
        matrix(h, tc="d"),
        matrix(a, tc="d"),
        matrix(b, tc="d"),
    )
    infeasible_statuses = {
        "infeasible problem",
        "LP relaxation is primal infeasible",
    }
    if status in infeasible_statuses:
        return None
    if status != "optimal":
        raise RuntimeError(f"Chance ILP did not finish optimally (status={status!r}).")

    values = np.asarray(solution, dtype=np.float64).reshape(-1)
    selected = values[:num_players] > 0.5
    indicated_misses = values[num_players:] > 0.5
    spend = scenario_markets[selected].sum(axis=0)
    actual_hits = spend <= salary_cap + AUDIT_SALARY_TOLERANCE
    if selected.sum() != roster_size:
        raise AssertionError("Chance solver returned the wrong roster size.")
    if actual_hits.sum() < required_hits:
        unindicated = ~indicated_misses
        max_excess = float(
            np.max(np.maximum(spend[unindicated] - salary_cap, 0.0))
            if unindicated.any()
            else 0.0
        )
        raise AssertionError(
            "Chance solver violated the required scenario hit count; "
            f"maximum unindicated cap excess was ${max_excess:.8f}."
        )
    if indicated_misses.sum() > allowed_misses:
        raise AssertionError("Chance solver exceeded the miss-indicator allowance.")
    return {
        "selected_mask": selected,
        "construction_spend": spend,
        "construction_hits": actual_hits,
        "required_hits": required_hits,
        "allowed_misses": allowed_misses,
        "indicated_misses": int(indicated_misses.sum()),
    }


def run_trials(
    year: int,
    sim: Any,
    predictions: pd.DataFrame,
    market_bank: np.ndarray,
    point_salary: np.ndarray,
    salary_model_matched: np.ndarray,
    espn_source_matched: np.ndarray,
    environment: dict[str, Any],
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    evaluation_weekly: np.ndarray,
    evaluation_decisions: np.ndarray,
    evaluation_played: np.ndarray,
    managed_values: np.ndarray,
    waiver_baseline: dict[str, float],
    trials: int,
    context_draws: int,
    construction_scenarios: int,
    evaluation_scenarios: int,
    seed: int,
) -> pd.DataFrame:
    predictions["salary"] = point_salary
    top_n = predictions.nlargest(
        min(base.TOP_N, len(predictions)), "salary"
    ).player.tolist()
    static = sim.build_managed_ilp_static_matrices(
        predictions,
        {},
        [],
        top_n,
        base.ROSTER_SIZE,
        base.POS_MIN,
        base.POS_MAX,
        enforce_top_n=True,
    )
    if construction_scenarios + evaluation_scenarios > market_bank.shape[1]:
        raise ValueError(
            "Construction and evaluation scenario counts exceed the salary market bank."
        )

    rng = np.random.default_rng(seed + year * 101)
    context_plan = rng.integers(0, weekly.shape[0], size=(trials, context_draws))
    market_plan = np.vstack(
        [
            rng.choice(
                market_bank.shape[1],
                size=construction_scenarios + evaluation_scenarios,
                replace=False,
            )
            for _ in range(trials)
        ]
    )
    rows: list[dict[str, Any]] = []
    forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}
    player_values = predictions.player.to_numpy(dtype=object)
    position_values = predictions.pos.to_numpy(dtype=object)
    top_n_set = set(top_n)

    for trial in range(trials):
        objective = managed_values[:, context_plan[trial]].mean(axis=1)
        scenario_indices = market_plan[trial]
        construction = market_bank[
            :, scenario_indices[:construction_scenarios]
        ]
        evaluation = market_bank[
            :, scenario_indices[construction_scenarios:]
        ]
        for level in CHANCE_LEVELS:
            solved = solve_chance_roster(
                sim,
                objective,
                static,
                construction,
                level,
                base.ROSTER_SIZE,
                base.SALARY_CAP,
            )
            if solved is None:
                rows.append(
                    {
                        "year": year,
                        "trial": trial,
                        "chance_level": level,
                        "status": "infeasible",
                    }
                )
                continue
            selected = solved["selected_mask"]
            roster = tuple(sorted(player_values[selected].tolist()))
            forecast_ev = base.forecast_roster_ev(
                roster,
                current.CURRENT_WAIVER,
                waiver_baseline,
                predictions,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                forecast_cache,
            )
            actual = base.score_actual_roster(environment, roster)
            actual_feasible = actual["actual_salary_spend"] <= base.SALARY_CAP + 1e-8
            evaluation_spend = evaluation[selected].sum(axis=0)
            evaluation_hits = evaluation_spend <= base.SALARY_CAP + 1e-8
            pos_counts = pd.Series(position_values[selected]).value_counts().to_dict()
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "chance_level": level,
                    "status": "optimal",
                    "roster": "|".join(roster),
                    "required_construction_hits": solved["required_hits"],
                    "allowed_construction_misses": solved["allowed_misses"],
                    "construction_hit_count": int(solved["construction_hits"].sum()),
                    "construction_cap_probability": float(
                        solved["construction_hits"].mean()
                    ),
                    "heldout_cap_probability": float(evaluation_hits.mean()),
                    "heldout_salary_spend_mean": float(evaluation_spend.mean()),
                    "heldout_salary_spend_p90": float(
                        np.quantile(evaluation_spend, 0.90)
                    ),
                    "heldout_salary_spend_p95": float(
                        np.quantile(evaluation_spend, 0.95)
                    ),
                    "point_salary_spend": float(point_salary[selected].sum()),
                    "managed_forecast_season_points": float(forecast_ev),
                    "actual_cap_feasible": bool(actual_feasible),
                    "actual_cap_overage": float(
                        max(actual["actual_salary_spend"] - base.SALARY_CAP, 0.0)
                    ),
                    "actual_points_if_affordable": (
                        float(actual["actual_points"]) if actual_feasible else np.nan
                    ),
                    "raw_actual_points_audit_only": float(actual["actual_points"]),
                    "salary_model_fallback_players": int(
                        (~salary_model_matched[selected]).sum()
                    ),
                    "minimum_salary_fallback_players": int(
                        (
                            (~salary_model_matched[selected])
                            & (~espn_source_matched[selected])
                        ).sum()
                    ),
                    "contains_top_n": bool(set(roster) & top_n_set),
                    "qb_count": int(pos_counts.get("QB", 0)),
                    "rb_count": int(pos_counts.get("RB", 0)),
                    "wr_count": int(pos_counts.get("WR", 0)),
                    "te_count": int(pos_counts.get("TE", 0)),
                    **actual,
                }
            )
        if (trial + 1) % max(1, min(25, trials)) == 0:
            print(f"{year}: completed {trial + 1}/{trials} trials", flush=True)
    return pd.DataFrame(rows)


def paired_frontier(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "managed_forecast_season_points",
        "heldout_cap_probability",
        "actual_cap_feasible",
        "actual_cap_overage",
        "actual_salary_spend",
        "point_salary_spend",
    ]
    pairs = []
    for lower, higher in zip(CHANCE_LEVELS[:-1], CHANCE_LEVELS[1:]):
        left = trials[trials.chance_level.eq(lower)]
        right = trials[trials.chance_level.eq(higher)]
        keep = ["year", "trial", "roster", "actual_points", *metrics]
        merged = left[keep].merge(
            right[keep],
            on=["year", "trial"],
            suffixes=("_lower", "_higher"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            values = row._asdict()
            both_feasible = bool(
                values["actual_cap_feasible_lower"]
                and values["actual_cap_feasible_higher"]
            )
            output: dict[str, Any] = {
                "comparison": f"{int(higher * 100)}_minus_{int(lower * 100)}",
                "lower_chance_level": lower,
                "higher_chance_level": higher,
                "year": int(values["year"]),
                "trial": int(values["trial"]),
                "roster_changed": values["roster_lower"] != values["roster_higher"],
                "both_actual_cap_feasible": both_feasible,
                "joint_feasible_actual_points_effect": (
                    values["actual_points_higher"] - values["actual_points_lower"]
                    if both_feasible
                    else np.nan
                ),
            }
            for metric in metrics:
                output[f"{metric}_effect"] = float(values[f"{metric}_higher"]) - float(
                    values[f"{metric}_lower"]
                )
            pairs.append(output)
    return pd.DataFrame(pairs)


def summarize(
    trials: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year = trials.groupby(["year", "chance_level"], as_index=False).agg(
        trials=("trial", "size"),
        unique_rosters=("roster", "nunique"),
        construction_cap_probability=("construction_cap_probability", "mean"),
        heldout_cap_probability=("heldout_cap_probability", "mean"),
        managed_forecast_season_points=("managed_forecast_season_points", "mean"),
        actual_cap_feasible_rate=("actual_cap_feasible", "mean"),
        actual_cap_overage=("actual_cap_overage", "mean"),
        actual_salary_spend=("actual_salary_spend", "mean"),
        affordable_actual_rosters=("actual_points_if_affordable", "count"),
        actual_points_if_affordable=("actual_points_if_affordable", "mean"),
        point_salary_spend=("point_salary_spend", "mean"),
    )
    period_rows = []
    for level, group in by_year.groupby("chance_level"):
        row: dict[str, Any] = {"chance_level": level}
        for col in [
            "construction_cap_probability",
            "heldout_cap_probability",
            "managed_forecast_season_points",
            "actual_cap_feasible_rate",
            "actual_cap_overage",
            "actual_salary_spend",
            "point_salary_spend",
        ]:
            row[f"development_2022_2024_{col}"] = float(
                group.loc[group.year.le(2024), col].mean()
            )
            check = group.loc[group.year.eq(2025), col]
            row[f"temporal_check_2025_{col}"] = (
                float(check.iloc[0]) if len(check) else np.nan
            )
        row["development_2022_2024_affordable_actual_rosters"] = int(
            group.loc[group.year.le(2024), "affordable_actual_rosters"].sum()
        )
        check_count = group.loc[group.year.eq(2025), "affordable_actual_rosters"]
        row["temporal_check_2025_affordable_actual_rosters"] = (
            int(check_count.iloc[0]) if len(check_count) else 0
        )
        period_rows.append(row)
    periods = pd.DataFrame(period_rows)

    pair_by_year = pairs.groupby(["comparison", "year"], as_index=False).agg(
        comparisons=("trial", "size"),
        roster_changed_rate=("roster_changed", "mean"),
        both_actual_cap_feasible_rate=("both_actual_cap_feasible", "mean"),
        **{
            f"mean_{col}": (col, "mean")
            for col in pairs.columns
            if col.endswith("_effect")
        },
    )
    pair_period_rows = []
    for comparison, group in pair_by_year.groupby("comparison"):
        row = {"comparison": comparison}
        metric_cols = [
            col
            for col in pair_by_year
            if col.startswith("mean_") or col.endswith("_rate")
        ]
        for col in metric_cols:
            row[f"development_2022_2024_{col}"] = float(
                group.loc[group.year.le(2024), col].mean()
            )
            check = group.loc[group.year.eq(2025), col]
            row[f"temporal_check_2025_{col}"] = (
                float(check.iloc[0]) if len(check) else np.nan
            )
        pair_period_rows.append(row)
    return by_year, periods, pair_by_year, pd.DataFrame(pair_period_rows)


def validate_trials(
    trials: pd.DataFrame,
    construction_scenarios: int,
) -> dict[str, Any]:
    if not trials.status.eq("optimal").all():
        failed = trials.loc[~trials.status.eq("optimal"), ["year", "trial", "chance_level"]]
        raise AssertionError(f"At least one chance cell was infeasible:\n{failed.head()}")
    if trials.duplicated(["year", "trial", "chance_level"]).any():
        raise AssertionError("Replay contains duplicate chance cells.")
    if not trials.groupby(["year", "trial"]).size().eq(len(CHANCE_LEVELS)).all():
        raise AssertionError("A paired chance trial is incomplete.")
    rosters = trials.roster.str.split("|")
    if not rosters.map(len).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A roster does not contain 13 players.")
    if not rosters.map(lambda values: len(set(values))).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A roster contains duplicate players.")
    required = trials.chance_level.map(
        lambda value: chance_counts(float(value), construction_scenarios)[0]
    )
    if (trials.construction_hit_count < required).any():
        raise AssertionError("A construction chance threshold was violated.")
    if (~trials.contains_top_n).any():
        raise AssertionError("A Top-N constraint was violated.")
    for pos in base.POSITIONS:
        col = f"{pos.lower()}_count"
        if trials[col].lt(base.POS_MIN[pos]).any() or trials[col].gt(base.POS_MAX[pos]).any():
            raise AssertionError(f"A {pos} roster limit was violated.")
    expected = trials.actual_salary_spend.le(base.SALARY_CAP + 1e-8)
    if not trials.actual_cap_feasible.eq(expected).all():
        raise AssertionError("Historical cap feasibility does not match spend.")
    if not trials.loc[~expected, "actual_points_if_affordable"].isna().all():
        raise AssertionError("An unaffordable roster entered feasible-only point scoring.")
    return {
        "all_cells_optimal": True,
        "all_rosters_size_13_unique": True,
        "all_construction_chance_thresholds_satisfied": True,
        "all_position_and_top_n_constraints_satisfied": True,
        "historical_cap_flags_match_spend": True,
        "unaffordable_points_excluded_from_policy_summary": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--contexts", type=int, default=250)
    parser.add_argument("--context-draws", type=int, default=5)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--construction-scenarios", type=int, default=20)
    parser.add_argument("--evaluation-scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    args = parser.parse_args()
    if sorted(set(args.years) - {2022, 2023, 2024, 2025}):
        parser.error("Only salary-table origins 2022-2025 are supported.")
    positive = [
        args.trials,
        args.contexts,
        args.context_draws,
        args.projection_draws,
        args.salary_draws,
        args.construction_scenarios,
        args.evaluation_scenarios,
    ]
    if min(positive) <= 0:
        parser.error("Draw, context, trial, and scenario counts must be positive.")
    available_markets = args.salary_draws // SALARY_DRAWS_PER_MARKET
    if args.construction_scenarios + args.evaluation_scenarios > available_markets:
        parser.error(
            "salary-draws must provide at least five draws for every construction "
            "and evaluation market scenario."
        )
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    salary_rows, source_rows = current.load_salary_tables()
    raw_weekly = base.load_raw_weekly(max_year=max(args.years))
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    all_trials = []
    all_surfaces = []
    all_template_audit = []
    origins: dict[str, Any] = {}

    for year in args.years:
        year_started = time.perf_counter()
        print(f"\n=== Origin {year} ===", flush=True)
        conn, source_manifest = base.open_frozen_source(base.FROZEN_SOURCES[year])
        try:
            forecast, ppg_draws, projection_meta = base.load_frozen_forecast(
                year,
                conn,
                features[features.season.eq(year)],
                args.projection_draws,
                args.seed,
            )
        finally:
            conn.close()
        environment, outcome_labels = base.build_actual_environment(
            year, forecast, raw_weekly, features, actual
        )
        cache, template_audit = base.build_template_cache(
            year, forecast, features, raw_weekly
        )
        template_audit["max_donor_is_causal"] = template_audit.max_donor_season.lt(year)
        if not template_audit.max_donor_is_causal.all():
            raise AssertionError("A construction template crossed the origin.")
        all_template_audit.append(template_audit)

        player_data = forecast[["player", "player_key", "pos", "pred_fp_per_game"]].copy()
        player_data["salary"] = 1.0
        sim = base.make_simulation(year, player_data, cache)
        waiver_baseline = sim.estimate_waiver_baselines(
            num_teams=base.NUM_TEAMS, roster_size=base.ROSTER_SIZE
        )
        candidate_idx = np.flatnonzero(~outcome_labels.is_keeper.to_numpy(dtype=bool))
        candidate_forecast = forecast.iloc[candidate_idx].reset_index(drop=True)
        candidate_ppg = ppg_draws[candidate_idx]
        remaining_budget = base.TOTAL_MARKET_BUDGET - environment["keeper_spend"]
        remaining_slots = base.TOTAL_MARKET_SLOTS - environment["keeper_count"]
        surface, salary_draws, salary_meta = current.build_salary_surface(
            year,
            candidate_forecast,
            salary_rows,
            source_rows,
            sim,
            remaining_budget,
            remaining_slots,
            args.salary_draws,
            args.seed,
        )
        point_salary = surface.point_salary.to_numpy(dtype=float)
        candidate_forecast["salary"] = point_salary
        predictions = base.build_predictions(candidate_forecast, candidate_ppg)
        market_bank = build_normalized_market_bank(
            sim,
            salary_draws,
            remaining_budget,
            remaining_slots,
        )
        weekly, decisions, played = base.generate_construction_contexts(
            sim, predictions, args.contexts, args.seed + year
        )
        eval_weekly, eval_decisions, eval_played = base.generate_construction_contexts(
            sim, predictions, args.contexts, args.seed + 100_000 + year
        )
        value_banks = base.managed_value_banks(
            weekly,
            decisions,
            played,
            predictions,
            {current.CURRENT_WAIVER: waiver_baseline},
        )
        trials = run_trials(
            year,
            sim,
            predictions,
            market_bank,
            point_salary,
            surface.salary_model_matched.to_numpy(dtype=bool),
            surface.espn_source_matched.to_numpy(dtype=bool),
            environment,
            weekly,
            decisions,
            played,
            eval_weekly,
            eval_decisions,
            eval_played,
            value_banks[(current.CURRENT_WAIVER, current.CURRENT_BENCH_WEIGHT)],
            waiver_baseline,
            args.trials,
            args.context_draws,
            args.construction_scenarios,
            args.evaluation_scenarios,
            args.seed,
        )
        surface["year"] = year
        all_surfaces.append(surface)
        all_trials.append(trials)
        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(
            {
                "keeper_count": environment["keeper_count"],
                "keeper_spend": environment["keeper_spend"],
                "remaining_budget": remaining_budget,
                "remaining_slots": remaining_slots,
                "waiver_baseline": waiver_baseline,
                "normalized_market_scenarios": int(market_bank.shape[1]),
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        origins[str(year)] = source_manifest
        print(f"{year}: complete in {time.perf_counter() - year_started:.1f}s", flush=True)

    trials = pd.concat(all_trials, ignore_index=True)
    surfaces = pd.concat(all_surfaces, ignore_index=True)
    template_audit = pd.concat(all_template_audit, ignore_index=True)
    expected_rows = len(args.years) * args.trials * len(CHANCE_LEVELS)
    if len(trials) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} cells, found {len(trials)}.")
    validation = validate_trials(trials, args.construction_scenarios)
    pairs = paired_frontier(trials)
    by_year, periods, pair_by_year, pair_periods = summarize(trials, pairs)
    outputs = {
        "roster_trials.csv": trials,
        "salary_surface_audit.csv": surfaces,
        "template_pool_audit.csv": template_audit,
        "paired_frontier_effects.csv": pairs,
        "frontier_summary_by_year.csv": by_year,
        "frontier_summary_periods.csv": periods,
        "paired_frontier_by_year.csv": pair_by_year,
        "paired_frontier_periods.csv": pair_periods,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    lines = [
        "# Salary Chance-Constraint Frontier",
        "",
        (
            f"{args.trials} paired trials per origin; each roster was constructed "
            f"against {args.construction_scenarios} normalized five-draw salary "
            f"markets and evaluated on {args.evaluation_scenarios} unseen markets."
        ),
        "",
        "## Frontier by year",
        "",
        base.markdown_table(
            by_year,
            [
                "year",
                "chance_level",
                "managed_forecast_season_points",
                "heldout_cap_probability",
                "actual_cap_feasible_rate",
                "actual_cap_overage",
                "affordable_actual_rosters",
            ],
            digits=3,
        ),
        "",
        "## Development and temporal-check frontier",
        "",
        base.markdown_table(
            periods,
            [
                "chance_level",
                "development_2022_2024_managed_forecast_season_points",
                "development_2022_2024_heldout_cap_probability",
                "development_2022_2024_actual_cap_feasible_rate",
                "development_2022_2024_actual_cap_overage",
                "temporal_check_2025_managed_forecast_season_points",
                "temporal_check_2025_heldout_cap_probability",
                "temporal_check_2025_actual_cap_feasible_rate",
                "temporal_check_2025_actual_cap_overage",
            ],
            digits=3,
        ),
        "",
        "## Adjacent-threshold paired effects",
        "",
        "Effects are higher threshold minus lower threshold.",
        "",
        base.markdown_table(
            pair_periods,
            [
                "comparison",
                "development_2022_2024_mean_managed_forecast_season_points_effect",
                "development_2022_2024_mean_heldout_cap_probability_effect",
                "development_2022_2024_mean_actual_cap_feasible_effect",
                "development_2022_2024_mean_actual_cap_overage_effect",
                "temporal_check_2025_mean_managed_forecast_season_points_effect",
                "temporal_check_2025_mean_heldout_cap_probability_effect",
                "temporal_check_2025_mean_actual_cap_feasible_effect",
                "temporal_check_2025_mean_actual_cap_overage_effect",
            ],
            digits=3,
        ),
        "",
        "## Interpretation limits",
        "",
        "- Managed forecast points are independently simulated preseason EV, not realized historical points.",
        "- Raw points for historically unaffordable rosters are audit-only and are excluded from the policy summary.",
        "- Feasible-only historical points select on future realized prices and cannot identify the best policy.",
        "- The one-swap refiner is disabled because it cannot enforce the multi-scenario chance constraint; every threshold uses the same unrefined optimizer.",
        "- Market scenarios reconcile the shared league budget, but player residuals are sampled marginally; cross-player auction-price correlation is not learned.",
        "- Salary training data roll by origin, but the 2026 model specification is retrospective rather than a fresh method holdout.",
        "- Historical final prices are exogenous, and missing actual prices retain the intentional `$1` fallback.",
        "- Four seasons are four outcome units; trial counts measure Monte Carlo stability, not additional independent seasons.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "chance_levels": list(CHANCE_LEVELS),
        "fixed_settings": {
            "salary_draws_per_market": SALARY_DRAWS_PER_MARKET,
            "solver_cap_margin": SOLVER_CAP_MARGIN,
            "audit_salary_tolerance": AUDIT_SALARY_TOLERANCE,
            "salary_cap": base.SALARY_CAP,
            "top_n": True,
            "waiver_source": current.CURRENT_WAIVER,
            "bench_weight": current.CURRENT_BENCH_WEIGHT,
            "roster_refinement": False,
            "position_min": base.POS_MIN,
            "position_max": base.POS_MAX,
        },
        "salary_identity": {
            "method_version": current.SALARY_METHOD,
            "model_spec_asof_year": current.MODEL_SPEC_YEAR,
            "data_rolling_origin": True,
            "fresh_method_holdout": False,
            "database": str(current.VALIDATION_DB),
            "database_sha256": base.sha256_file(current.VALIDATION_DB),
        },
        "sources": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": base.sha256_file(Path(__file__).resolve()),
            "current_salary_runner": str(CURRENT_RUNNER),
            "current_salary_runner_sha256": base.sha256_file(CURRENT_RUNNER),
            "base_runner": str(current.BASE_RUNNER),
            "base_runner_sha256": base.sha256_file(current.BASE_RUNNER),
            "simulation_helper": str(base.APP_HELPER),
            "simulation_helper_sha256": base.sha256_file(base.APP_HELPER),
            "raw_weekly_database_sha256": base.sha256_file(base.DAILY_DB),
        },
        "origins": origins,
        "validation": {
            **validation,
            "expected_rows": expected_rows,
            "all_template_donors_pre_origin": True,
        },
        "output_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"Replay complete in {time.perf_counter() - started:.1f}s: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
