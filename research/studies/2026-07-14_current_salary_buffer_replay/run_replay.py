"""Paired $5 versus $10 replay using rolling current-method salary tables."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
BASE_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-13_managed_auction_rolling_replay"
    / "run_replay.py"
)
VALIDATION_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
SALARY_METHOD = "current_locked_spec_v1"
MODEL_SPEC_YEAR = 2026
BUFFERS: tuple[tuple[str, float | None], ...] = (
    ("none", None),
    ("5", 5.0),
    ("10", 10.0),
)
CURRENT_WAIVER = "current_projected"
CURRENT_BENCH_WEIGHT = 0.25
RESID_COLS = [
    "salary_resid_5",
    "salary_resid_10",
    "salary_resid_25",
    "salary_resid_75",
    "salary_resid_90",
    "salary_resid_95",
]


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("_current_salary_replay_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base replay: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def load_salary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(f"file:{VALIDATION_DB}?mode=ro", uri=True) as conn:
        salary = pd.read_sql_query(
            """SELECT * FROM Salary_Backtest_Predictions
                WHERE league='beta' AND method_version=?
                  AND model_spec_asof_year=?""",
            conn,
            params=(SALARY_METHOD, MODEL_SPEC_YEAR),
        )
    if set(salary.year.astype(int)) != {2022, 2023, 2024, 2025}:
        raise ValueError("Salary backtest origins are incomplete.")
    salary = base.add_identity(salary)
    if salary.duplicated(["year", "player_key"]).any():
        raise ValueError("Salary backtest contains duplicate cleaned player keys.")
    if not salary.normalization_uses_target_actuals.eq(0).all():
        raise ValueError("A salary origin used target actuals for normalization.")
    if not (salary.training_through_year == salary.year - 1).all():
        raise ValueError("A salary-model training cutoff crossed its origin.")
    if not (salary.resid_training_through_year == salary.year - 1).all():
        raise ValueError("A salary residual cutoff crossed its origin.")
    if not (np.diff(salary[RESID_COLS].to_numpy(float), axis=1) >= -1e-10).all():
        raise ValueError("Salary residual quantiles are not monotone.")

    with sqlite3.connect(f"file:{base.SIM_DB}?mode=ro", uri=True) as conn:
        source = pd.read_sql_query(
            "SELECT player, year, salary FROM Salaries WHERE league='beta'",
            conn,
        )
    source = base.add_identity(source)
    source["salary"] = pd.to_numeric(source.salary, errors="coerce")
    source = source.sort_values("salary", ascending=False).drop_duplicates(
        ["year", "player_key"]
    )
    return salary, source


def interpolate_fallback_quantiles(surface: pd.DataFrame) -> np.ndarray:
    output = surface[RESID_COLS].to_numpy(dtype=float)
    missing = ~surface.salary_model_matched.to_numpy(dtype=bool)
    for idx in np.flatnonzero(missing):
        pos = surface.loc[idx, "pos"]
        donors = surface[
            surface.salary_model_matched & surface.pos.eq(pos)
        ].sort_values("stored_pred_salary")
        if len(donors) < 2:
            donors = surface[surface.salary_model_matched].sort_values(
                "stored_pred_salary"
            )
        if len(donors) == 0:
            output[idx] = 0.0
            continue
        x = donors.stored_pred_salary.to_numpy(dtype=float)
        target = float(surface.loc[idx, "point_salary"])
        for col_idx, col in enumerate(RESID_COLS):
            output[idx, col_idx] = np.interp(
                target,
                x,
                donors[col].to_numpy(dtype=float),
            )
    return np.maximum.accumulate(output, axis=1)


def sample_residual_quantiles(
    means: np.ndarray,
    residuals: np.ndarray,
    num_draws: int,
    seed: int,
) -> np.ndarray:
    """Seeded equivalent of the live app's piecewise residual sampler."""
    probs = np.array([0.00, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 1.00])
    q5, q10, q25, q75, q90, q95 = residuals.T
    knots = np.column_stack(
        [(2 * q5) - q10, q5, q10, q25, q75, q90, q95, (2 * q95) - q90]
    )
    knots = np.maximum.accumulate(knots, axis=1)
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(0, 1, size=(len(means), num_draws))
    knot_idx = np.searchsorted(probs, uniforms, side="right") - 1
    knot_idx = np.clip(knot_idx, 0, len(probs) - 2)
    left_prob = probs[knot_idx]
    right_prob = probs[knot_idx + 1]
    left = np.take_along_axis(knots, knot_idx, axis=1)
    right = np.take_along_axis(knots, knot_idx + 1, axis=1)
    weights = (uniforms - left_prob) / (right_prob - left_prob)
    return np.rint(np.maximum(means[:, None] + left + weights * (right - left), 1.0))


def build_salary_surface(
    year: int,
    candidate_forecast: pd.DataFrame,
    salary_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    sim: Any,
    remaining_budget: float,
    remaining_slots: int,
    num_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    keep = [
        "player_key",
        "pred_salary",
        "pred_salary_raw",
        "base_salary_observed",
        *RESID_COLS,
    ]
    origin = salary_rows[salary_rows.year.eq(year)][keep].copy()
    origin = origin.rename(columns={"pred_salary": "stored_pred_salary"})
    source = source_rows[source_rows.year.eq(year)][["player_key", "salary"]].rename(
        columns={"salary": "espn_source_salary"}
    )
    surface = candidate_forecast[
        ["player", "player_key", "pos", "pred_fp_per_game"]
    ].merge(origin, on="player_key", how="left")
    surface = surface.merge(source, on="player_key", how="left")
    surface["salary_model_matched"] = surface.pred_salary_raw.notna()
    surface["espn_source_matched"] = surface.espn_source_salary.notna()
    raw_center = surface.pred_salary_raw.where(
        surface.salary_model_matched,
        surface.espn_source_salary,
    ).fillna(0.0)
    raw_center = pd.to_numeric(raw_center, errors="coerce").fillna(0.0).clip(lower=1.0)
    point_salary = base.normalize_market_draws(
        sim,
        raw_center.to_numpy(dtype=float)[:, None],
        remaining_budget,
        remaining_slots,
    )[:, 0]
    surface["raw_point_salary"] = raw_center.to_numpy(dtype=float)
    surface["point_salary"] = point_salary
    residuals = interpolate_fallback_quantiles(surface)
    salary_draws = sample_residual_quantiles(
        point_salary,
        residuals,
        num_draws,
        seed + year * 37,
    ).astype(np.float32)
    top_total = float(np.sort(point_salary)[-remaining_slots:].sum())
    if not math.isclose(top_total, remaining_budget, abs_tol=1e-3, rel_tol=0.0):
        raise AssertionError("Point salary market does not equal remaining budget.")
    for col_idx, col in enumerate(RESID_COLS):
        surface[col] = residuals[:, col_idx]
    return surface, salary_draws, {
        "candidate_players": int(len(surface)),
        "salary_model_matches": int(surface.salary_model_matched.sum()),
        "salary_model_fallbacks": int((~surface.salary_model_matched).sum()),
        "espn_fallback_matches": int(
            ((~surface.salary_model_matched) & surface.espn_source_matched).sum()
        ),
        "minimum_fallbacks": int(
            ((~surface.salary_model_matched) & ~surface.espn_source_matched).sum()
        ),
        "point_market_total": top_total,
    }


def run_trials(
    year: int,
    sim: Any,
    predictions: pd.DataFrame,
    salary_draws: np.ndarray,
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
    seed: int,
) -> pd.DataFrame:
    remaining_budget = base.TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = base.TOTAL_MARKET_SLOTS - environment["keeper_count"]
    predictions["salary"] = point_salary
    top_n = predictions.nlargest(min(base.TOP_N, len(predictions)), "salary").player.tolist()
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
    ref_weekly = weekly.mean(axis=0)
    ref_decisions = decisions.mean(axis=0)
    ref_played = np.where(
        np.any(played >= 0, axis=0),
        np.any(played > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)
    rng = np.random.default_rng(seed + year * 101)
    salary_plan = rng.integers(0, salary_draws.shape[1], size=(trials, 5))
    context_plan = rng.integers(0, weekly.shape[0], size=(trials, context_draws))
    raw_markets = np.column_stack(
        [salary_draws[:, indices].mean(axis=1) for indices in salary_plan]
    )
    markets = base.normalize_market_draws(
        sim,
        raw_markets,
        remaining_budget,
        remaining_slots,
    )
    rows = []
    forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}
    for trial in range(trials):
        objective = managed_values[:, context_plan[trial]].mean(axis=1)
        market = markets[:, trial]
        predictions["salary"] = market
        for label, buffer_value in BUFFERS:
            nominal_cap = None if buffer_value is None else base.SALARY_CAP + buffer_value
            solved = sim._solve_managed_scenario(
                predictions,
                objective,
                ref_weekly,
                ref_decisions,
                static,
                [],
                {},
                top_n,
                base.ROSTER_SIZE,
                base.POS_MIN,
                base.POS_MAX,
                waiver_baseline,
                base.LINEUP_REQUIRE,
                True,
                refine_roster=True,
                score_roster=False,
                salary_values=market,
                played_mask=ref_played,
                nominal_salary_values=None if nominal_cap is None else point_salary,
                nominal_salary_cap=nominal_cap,
            )
            if solved is None:
                rows.append(
                    {"year": year, "trial": trial, "buffer": label, "status": "infeasible"}
                )
                continue
            selected = np.asarray(solved["selected_mask"], dtype=bool)
            roster = tuple(sorted(solved["selected_players"]))
            actual = base.score_actual_roster(environment, roster)
            forecast_ev = base.forecast_roster_ev(
                roster,
                CURRENT_WAIVER,
                waiver_baseline,
                predictions,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                forecast_cache,
            )
            pos_counts = predictions.loc[selected, "pos"].value_counts().to_dict()
            actual_feasible = actual["actual_salary_spend"] <= base.SALARY_CAP + 1e-8
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "buffer": label,
                    "buffer_dollars": buffer_value,
                    "nominal_cap": nominal_cap,
                    "status": "optimal",
                    "roster": "|".join(roster),
                    "sampled_salary_spend": float(market[selected].sum()),
                    "point_salary_spend": float(point_salary[selected].sum()),
                    "forecast_ev": float(forecast_ev),
                    "actual_cap_feasible": bool(actual_feasible),
                    "actual_cap_overage": float(
                        max(actual["actual_salary_spend"] - base.SALARY_CAP, 0.0)
                    ),
                    "salary_model_fallback_players": int((~salary_model_matched[selected]).sum()),
                    "minimum_salary_fallback_players": int(
                        ((~salary_model_matched[selected]) & (~espn_source_matched[selected])).sum()
                    ),
                    "contains_top_n": bool(set(roster) & set(top_n)),
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


PAIR_METRICS = [
    "actual_points",
    "drafted_only_points",
    "actual_cap_feasible",
    "actual_cap_overage",
    "actual_salary_spend",
    "forecast_ev",
    "sampled_salary_spend",
    "point_salary_spend",
    "actual_waiver_starts",
    "salary_model_fallback_players",
    "minimum_salary_fallback_players",
]


def pair_comparison(trials: pd.DataFrame, default: str, candidate: str) -> pd.DataFrame:
    left = trials[trials.buffer.eq(default)]
    right = trials[trials.buffer.eq(candidate)]
    keep = ["year", "trial", "roster", *PAIR_METRICS]
    merged = left[keep].merge(
        right[keep],
        on=["year", "trial"],
        suffixes=("_default", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        values = row._asdict()
        jaccard, changed = base.roster_jaccard(
            values["roster_default"], values["roster_candidate"]
        )
        output = {
            "comparison": f"{candidate}_minus_{default}",
            "default_buffer": default,
            "candidate_buffer": candidate,
            "year": int(values["year"]),
            "trial": int(values["trial"]),
            "roster_jaccard": jaccard,
            "roster_slots_changed": changed,
            "roster_changed": changed > 0,
            "both_actual_cap_feasible": bool(
                values["actual_cap_feasible_default"]
                and values["actual_cap_feasible_candidate"]
            ),
        }
        for metric in PAIR_METRICS:
            output[f"{metric}_effect"] = float(values[f"{metric}_candidate"]) - float(
                values[f"{metric}_default"]
            )
        output["joint_feasible_actual_points_effect"] = (
            values["actual_points_candidate"] - values["actual_points_default"]
            if output["both_actual_cap_feasible"]
            else np.nan
        )
        rows.append(output)
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year = trials.groupby(["year", "buffer"], as_index=False).agg(
        trials=("trial", "size"),
        unique_rosters=("roster", "nunique"),
        actual_points=("actual_points", "mean"),
        cap_feasible_rate=("actual_cap_feasible", "mean"),
        mean_cap_overage=("actual_cap_overage", "mean"),
        actual_salary_spend=("actual_salary_spend", "mean"),
        point_salary_spend=("point_salary_spend", "mean"),
        sampled_salary_spend=("sampled_salary_spend", "mean"),
        forecast_ev=("forecast_ev", "mean"),
        actual_waiver_starts=("actual_waiver_starts", "mean"),
        salary_model_fallback_players=("salary_model_fallback_players", "mean"),
        minimum_salary_fallback_players=("minimum_salary_fallback_players", "mean"),
    )
    pair_by_year = pairs.groupby(["comparison", "year"], as_index=False).agg(
        comparisons=("trial", "size"),
        roster_changed_rate=("roster_changed", "mean"),
        roster_jaccard=("roster_jaccard", "mean"),
        both_feasible_rate=("both_actual_cap_feasible", "mean"),
        **{
            f"mean_{col}": (col, "mean")
            for col in pairs.columns
            if col.endswith("_effect")
        },
    )
    periods = []
    for comparison, group in pair_by_year.groupby("comparison"):
        row: dict[str, Any] = {"comparison": comparison}
        for col in [c for c in pair_by_year if c.startswith("mean_") or c.endswith("_rate")]:
            row[f"development_2022_2024_{col}"] = float(group.loc[group.year.le(2024), col].mean())
            check = group.loc[group.year.eq(2025), col]
            row[f"temporal_check_2025_{col}"] = float(check.iloc[0]) if len(check) else np.nan
        periods.append(row)
    return by_year, pair_by_year, pd.DataFrame(periods)


def validate_trials(trials: pd.DataFrame) -> dict[str, Any]:
    if not trials.status.eq("optimal").all():
        raise AssertionError("At least one replay cell was infeasible.")
    if trials.duplicated(["year", "trial", "buffer"]).any():
        raise AssertionError("Replay contains duplicate cells.")
    if not trials.groupby(["year", "trial"]).size().eq(len(BUFFERS)).all():
        raise AssertionError("A paired trial is incomplete.")
    rosters = trials.roster.str.split("|")
    if not rosters.map(len).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A roster does not contain 13 players.")
    if not rosters.map(lambda x: len(set(x))).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A roster contains duplicate players.")
    if (trials.sampled_salary_spend > base.SALARY_CAP + 1e-4).any():
        raise AssertionError("A sampled salary cap was exceeded.")
    constrained = trials.buffer.ne("none")
    if (trials.loc[constrained, "point_salary_spend"] > trials.loc[constrained, "nominal_cap"] + 1e-4).any():
        raise AssertionError("A nominal salary cap was exceeded.")
    if (~trials.contains_top_n).any():
        raise AssertionError("A Top-N constraint was violated.")
    for pos in base.POSITIONS:
        col = f"{pos.lower()}_count"
        if trials[col].lt(base.POS_MIN[pos]).any() or trials[col].gt(base.POS_MAX[pos]).any():
            raise AssertionError(f"A {pos} roster limit was violated.")
    expected = trials.actual_salary_spend.le(base.SALARY_CAP + 1e-8)
    if not trials.actual_cap_feasible.eq(expected).all():
        raise AssertionError("Actual cap feasibility does not match spend.")
    return {
        "all_cells_optimal": True,
        "all_rosters_size_13_unique": True,
        "all_sampled_caps_satisfied": True,
        "all_nominal_caps_satisfied": True,
        "all_position_and_top_n_constraints_satisfied": True,
        "actual_cap_flags_match_spend": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--contexts", type=int, default=250)
    parser.add_argument("--context-draws", type=int, default=5)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    args = parser.parse_args()
    if sorted(set(args.years) - {2022, 2023, 2024, 2025}):
        parser.error("Only salary-table origins 2022-2025 are supported.")
    if min(args.trials, args.contexts, args.context_draws, args.projection_draws, args.salary_draws) <= 0:
        parser.error("Draw and trial counts must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    salary_rows, source_rows = load_salary_tables()
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
        surface, salary_draws, salary_meta = build_salary_surface(
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
        candidate_forecast["salary"] = surface.point_salary.to_numpy(dtype=float)
        predictions = base.build_predictions(candidate_forecast, candidate_ppg)
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
            {CURRENT_WAIVER: waiver_baseline},
        )
        trials = run_trials(
            year,
            sim,
            predictions,
            salary_draws,
            surface.point_salary.to_numpy(dtype=float),
            surface.salary_model_matched.to_numpy(dtype=bool),
            surface.espn_source_matched.to_numpy(dtype=bool),
            environment,
            weekly,
            decisions,
            played,
            eval_weekly,
            eval_decisions,
            eval_played,
            value_banks[(CURRENT_WAIVER, CURRENT_BENCH_WEIGHT)],
            waiver_baseline,
            args.trials,
            args.context_draws,
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
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        origins[str(year)] = source_manifest
        print(f"{year}: complete in {time.perf_counter() - year_started:.1f}s", flush=True)

    trials = pd.concat(all_trials, ignore_index=True)
    surfaces = pd.concat(all_surfaces, ignore_index=True)
    template_audit = pd.concat(all_template_audit, ignore_index=True)
    expected_rows = len(args.years) * args.trials * len(BUFFERS)
    if len(trials) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} cells, found {len(trials)}.")
    validation = validate_trials(trials)
    if not template_audit.max_donor_is_causal.all():
        raise AssertionError("Template timing validation failed.")
    pairs = pd.concat(
        [
            pair_comparison(trials, "none", "5"),
            pair_comparison(trials, "none", "10"),
            pair_comparison(trials, "10", "5"),
        ],
        ignore_index=True,
    )
    variant_by_year, pair_by_year, pair_periods = summarize(trials, pairs)
    outputs = {
        "roster_trials.csv": trials,
        "salary_surface_audit.csv": surfaces,
        "template_pool_audit.csv": template_audit,
        "paired_effects.csv": pairs,
        "variant_summary_by_year.csv": variant_by_year,
        "paired_effects_by_year.csv": pair_by_year,
        "paired_effects_development_check.csv": pair_periods,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    direct = pair_periods[pair_periods.comparison.eq("5_minus_10")]
    lines = [
        "# Current-Method $5 versus $10 Replay",
        "",
        f"{args.trials} paired five-draw trials per origin; development is 2022-2024 and 2025 is a temporal check.",
        "",
        "## Variant outcomes by year",
        "",
        base.markdown_table(
            variant_by_year,
            ["year", "buffer", "actual_points", "cap_feasible_rate", "mean_cap_overage", "actual_salary_spend", "point_salary_spend", "salary_model_fallback_players"],
            digits=3,
        ),
        "",
        "## Direct $5 minus $10 effects",
        "",
        "Positive points/feasibility favor $5; negative overage/spend favor $5.",
        "",
        base.markdown_table(
            direct,
            [
                "comparison",
                "development_2022_2024_mean_actual_points_effect",
                "development_2022_2024_mean_actual_cap_feasible_effect",
                "development_2022_2024_mean_actual_cap_overage_effect",
                "development_2022_2024_roster_changed_rate",
                "temporal_check_2025_mean_actual_points_effect",
                "temporal_check_2025_mean_actual_cap_feasible_effect",
                "temporal_check_2025_mean_actual_cap_overage_effect",
            ],
            digits=3,
        ),
        "",
        "## Limits",
        "",
        "- Salary training data roll by origin, but the 2026 model specification is retrospective rather than a fresh method holdout.",
        "- Historical final prices remain exogenous and missing actual prices retain the intentional $1 scoring fallback.",
        "- Frozen point forecasts and the current salary pool differ for some players; every salary-model and minimum fallback is recorded.",
        "- Four seasons are four outcome units; trial counts measure Monte Carlo stability, not additional independent seasons.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "buffers": [{"label": label, "dollars": value} for label, value in BUFFERS],
        "fixed_settings": {
            "salary_draw_count": 5,
            "sampled_salary_cap": base.SALARY_CAP,
            "top_n": True,
            "waiver_source": CURRENT_WAIVER,
            "bench_weight": CURRENT_BENCH_WEIGHT,
            "position_min": base.POS_MIN,
            "position_max": base.POS_MAX,
        },
        "salary_identity": {
            "method_version": SALARY_METHOD,
            "model_spec_asof_year": MODEL_SPEC_YEAR,
            "data_rolling_origin": True,
            "fresh_method_holdout": False,
            "database": str(VALIDATION_DB),
            "database_sha256": base.sha256_file(VALIDATION_DB),
        },
        "sources": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": base.sha256_file(Path(__file__).resolve()),
            "base_runner": str(BASE_RUNNER),
            "base_runner_sha256": base.sha256_file(BASE_RUNNER),
            "simulation_helper": str(base.APP_HELPER),
            "simulation_helper_sha256": base.sha256_file(base.APP_HELPER),
            "raw_weekly_database_sha256": base.sha256_file(base.DAILY_DB),
        },
        "origins": origins,
        "validation": {**validation, "expected_rows": expected_rows, "all_template_donors_pre_origin": True},
        "output_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"Replay complete in {time.perf_counter() - started:.1f}s: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
