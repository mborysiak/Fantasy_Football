"""Rolling replay of targeted optimizer-selection salary surcharges."""

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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
CURRENT_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_current_salary_buffer_replay"
    / "run_replay.py"
)
SELECTION_DIAGNOSTIC = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-16_salary_v5_replay"
    / "results"
    / "selected_residuals_v5"
    / "candidate_diagnostic.csv"
)
BASELINE_FRONTIER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-16_salary_v5_replay"
    / "results"
    / "frontier_v5"
    / "roster_trials.csv"
)
SALARY_METHOD = "current_locked_spec_v5_compact_salary_features"
MODEL_SPEC_YEAR = 2026
RIDGE_ALPHA = 100.0
SURCHARGE_CAP = 10.0
BLANKET_CAP = 285.0
VARIANTS = (
    "baseline_298",
    "blanket_285",
    "targeted_half",
    "targeted_full",
)
PAIR_METRICS = [
    "actual_points",
    "drafted_only_points",
    "actual_cap_feasible",
    "actual_cap_overage",
    "actual_salary_spend",
    "forecast_ev",
    "sampled_salary_spend",
    "decision_salary_spend",
    "point_salary_spend",
    "calibrated_point_salary_spend",
    "surcharge_spend",
    "actual_waiver_starts",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import replay helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


current = load_module(CURRENT_RUNNER, "_selection_surcharge_current_replay")
current.SALARY_METHOD = SALARY_METHOD
current.MODEL_SPEC_YEAR = MODEL_SPEC_YEAR
base = current.base


def load_selection_diagnostic() -> pd.DataFrame:
    diagnostic = pd.read_csv(SELECTION_DIAGNOSTIC)
    required = {
        "year",
        "player",
        "player_key",
        "pos",
        "point_salary",
        "selection_rate",
        "salary_residual",
        "actual_salary_recorded",
    }
    missing = sorted(required - set(diagnostic.columns))
    if missing:
        raise ValueError(f"Selection diagnostic is missing columns: {missing}")
    diagnostic["year"] = diagnostic.year.astype(int)
    diagnostic["selection_rate"] = (
        pd.to_numeric(diagnostic.selection_rate, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    diagnostic["point_salary"] = pd.to_numeric(
        diagnostic.point_salary, errors="coerce"
    )
    diagnostic["salary_residual"] = pd.to_numeric(
        diagnostic.salary_residual, errors="coerce"
    )
    diagnostic["actual_salary_recorded"] = (
        diagnostic.actual_salary_recorded.fillna(0).astype(int)
    )
    if diagnostic.duplicated(["year", "player_key"]).any():
        raise ValueError("Selection diagnostic contains duplicate player origins.")
    return diagnostic


def calibration_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    salary = pd.to_numeric(frame.point_salary, errors="coerce").clip(lower=1.0)
    selection = (
        pd.to_numeric(frame.selection_rate, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    output["pos"] = frame.pos.fillna("UNK").astype(str)
    output["pred_salary"] = salary
    output["pred_salary_sq"] = (salary / 25.0) ** 2
    output["selection_rate"] = selection
    output["selection_x_salary"] = selection * salary
    for pos in ("QB", "RB", "TE"):
        output[f"selection_x_{pos}"] = selection * frame.pos.eq(pos).astype(float)
    return output


def fit_origin_surcharge(
    target_year: int,
    target_surface: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_rates = diagnostic.loc[
        diagnostic.year.eq(target_year),
        ["player_key", "selection_rate"],
    ]
    calibration = target_surface[
        ["player", "player_key", "pos", "point_salary"]
    ].merge(
        target_rates,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    missing_selection = int(calibration.selection_rate.isna().sum())
    calibration["selection_rate"] = calibration.selection_rate.fillna(0.0)

    if target_year == diagnostic.year.min():
        calibration["predicted_salary_residual"] = 0.0
        calibration["surcharge_full"] = 0.0
        calibration["surcharge_half"] = 0.0
        return calibration, pd.DataFrame(), {
            "target_year": target_year,
            "training_through_year": None,
            "training_rows": 0,
            "selection_rate_matches": int(len(calibration) - missing_selection),
            "selection_rate_fallbacks": missing_selection,
            "ridge_alpha": RIDGE_ALPHA,
            "surcharge_cap": SURCHARGE_CAP,
        }

    training = diagnostic[
        diagnostic.year.lt(target_year)
        & diagnostic.actual_salary_recorded.eq(1)
        & diagnostic.salary_residual.notna()
    ].copy()
    if training.empty:
        raise ValueError(f"No prior-origin calibration rows exist for {target_year}.")
    if training.year.max() >= target_year:
        raise AssertionError("Calibration training crossed its target origin.")

    x_train = calibration_features(training)
    x_target = calibration_features(calibration)
    numeric = [col for col in x_train.columns if col != "pos"]
    preprocessing = ColumnTransformer(
        [
            (
                "pos",
                OneHotEncoder(handle_unknown="ignore"),
                ["pos"],
            ),
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(x_train, training.salary_residual.to_numpy(dtype=float))
    predicted = model.predict(x_target)
    calibration["predicted_salary_residual"] = predicted
    calibration["surcharge_full"] = np.clip(predicted, 0.0, SURCHARGE_CAP)
    calibration["surcharge_half"] = calibration.surcharge_full * 0.5

    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "target_year": target_year,
            "training_through_year": int(training.year.max()),
            "feature": feature_names,
            "coefficient": model.named_steps["ridge"].coef_,
        }
    )
    coefficients = pd.concat(
        [
            pd.DataFrame(
                {
                    "target_year": [target_year],
                    "training_through_year": [int(training.year.max())],
                    "feature": ["intercept"],
                    "coefficient": [model.named_steps["ridge"].intercept_],
                }
            ),
            coefficients,
        ],
        ignore_index=True,
    )
    return calibration, coefficients, {
        "target_year": target_year,
        "training_through_year": int(training.year.max()),
        "training_rows": int(len(training)),
        "training_origins": sorted(training.year.unique().astype(int).tolist()),
        "selection_rate_matches": int(len(calibration) - missing_selection),
        "selection_rate_fallbacks": missing_selection,
        "ridge_alpha": RIDGE_ALPHA,
        "surcharge_cap": SURCHARGE_CAP,
        "mean_predicted_residual": float(np.mean(predicted)),
        "mean_full_surcharge": float(calibration.surcharge_full.mean()),
        "selection_weighted_full_surcharge": float(
            np.average(
                calibration.surcharge_full,
                weights=np.maximum(calibration.selection_rate, 1e-12),
            )
        ),
        "max_full_surcharge": float(calibration.surcharge_full.max()),
    }


def run_trials(
    year: int,
    sim: Any,
    predictions: pd.DataFrame,
    salary_draws: np.ndarray,
    calibration: pd.DataFrame,
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
    point_salary = calibration.point_salary.to_numpy(dtype=float)
    surcharge_half = calibration.surcharge_half.to_numpy(dtype=float)
    surcharge_full = calibration.surcharge_full.to_numpy(dtype=float)
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
    rows: list[dict[str, Any]] = []
    forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}

    for trial in range(trials):
        objective = managed_values[:, context_plan[trial]].mean(axis=1)
        market = markets[:, trial]
        variant_prices = {
            "baseline_298": (market, None, None, np.zeros_like(market)),
            "blanket_285": (market, market, BLANKET_CAP, np.zeros_like(market)),
            "targeted_half": (
                market + surcharge_half,
                None,
                None,
                surcharge_half,
            ),
            "targeted_full": (
                market + surcharge_full,
                None,
                None,
                surcharge_full,
            ),
        }
        for variant in VARIANTS:
            decision_market, nominal_values, nominal_cap, surcharge = variant_prices[
                variant
            ]
            predictions["salary"] = decision_market
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
                salary_values=decision_market,
                played_mask=ref_played,
                nominal_salary_values=nominal_values,
                nominal_salary_cap=nominal_cap,
            )
            if solved is None:
                rows.append(
                    {
                        "year": year,
                        "trial": trial,
                        "variant": variant,
                        "status": "infeasible",
                    }
                )
                continue
            selected = np.asarray(solved["selected_mask"], dtype=bool)
            roster = tuple(sorted(solved["selected_players"]))
            actual = base.score_actual_roster(environment, roster)
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
            pos_counts = predictions.loc[selected, "pos"].value_counts().to_dict()
            actual_feasible = actual["actual_salary_spend"] <= base.SALARY_CAP + 1e-8
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "variant": variant,
                    "status": "optimal",
                    "roster": "|".join(roster),
                    "policy_cap": (
                        BLANKET_CAP if variant == "blanket_285" else base.SALARY_CAP
                    ),
                    "sampled_salary_spend": float(market[selected].sum()),
                    "decision_salary_spend": float(decision_market[selected].sum()),
                    "point_salary_spend": float(point_salary[selected].sum()),
                    "calibrated_point_salary_spend": float(
                        (point_salary + surcharge)[selected].sum()
                    ),
                    "surcharge_spend": float(surcharge[selected].sum()),
                    "forecast_ev": float(forecast_ev),
                    "actual_cap_feasible": bool(actual_feasible),
                    "actual_cap_overage": float(
                        max(actual["actual_salary_spend"] - base.SALARY_CAP, 0.0)
                    ),
                    "salary_model_fallback_players": int(
                        (~salary_model_matched[selected]).sum()
                    ),
                    "minimum_salary_fallback_players": int(
                        (
                            (~salary_model_matched[selected])
                            & (~espn_source_matched[selected])
                        ).sum()
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


def pair_comparison(
    trials: pd.DataFrame,
    candidate: str,
) -> pd.DataFrame:
    baseline = trials[trials.variant.eq("baseline_298")]
    contender = trials[trials.variant.eq(candidate)]
    keep = ["year", "trial", "roster", *PAIR_METRICS]
    merged = baseline[keep].merge(
        contender[keep],
        on=["year", "trial"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        values = row._asdict()
        jaccard, changed = base.roster_jaccard(
            values["roster_baseline"], values["roster_candidate"]
        )
        output = {
            "comparison": f"{candidate}_minus_baseline_298",
            "candidate": candidate,
            "year": int(values["year"]),
            "trial": int(values["trial"]),
            "roster_jaccard": jaccard,
            "roster_slots_changed": changed,
            "roster_changed": changed > 0,
            "both_actual_cap_feasible": bool(
                values["actual_cap_feasible_baseline"]
                and values["actual_cap_feasible_candidate"]
            ),
        }
        for metric in PAIR_METRICS:
            output[f"{metric}_effect"] = float(
                values[f"{metric}_candidate"]
            ) - float(values[f"{metric}_baseline"])
        output["joint_feasible_actual_points_effect"] = (
            values["actual_points_candidate"] - values["actual_points_baseline"]
            if output["both_actual_cap_feasible"]
            else np.nan
        )
        rows.append(output)
    return pd.DataFrame(rows)


def summarize(
    trials: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year = trials.groupby(["year", "variant"], as_index=False).agg(
        trials=("trial", "size"),
        unique_rosters=("roster", "nunique"),
        actual_points=("actual_points", "mean"),
        cap_feasible_rate=("actual_cap_feasible", "mean"),
        mean_cap_overage=("actual_cap_overage", "mean"),
        actual_salary_spend=("actual_salary_spend", "mean"),
        sampled_salary_spend=("sampled_salary_spend", "mean"),
        decision_salary_spend=("decision_salary_spend", "mean"),
        point_salary_spend=("point_salary_spend", "mean"),
        calibrated_point_salary_spend=(
            "calibrated_point_salary_spend",
            "mean",
        ),
        surcharge_spend=("surcharge_spend", "mean"),
        forecast_ev=("forecast_ev", "mean"),
        actual_waiver_starts=("actual_waiver_starts", "mean"),
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
        metric_cols = [
            col
            for col in pair_by_year
            if col.startswith("mean_") or col.endswith("_rate")
        ]
        for col in metric_cols:
            development = group.loc[group.year.isin([2023, 2024]), col]
            check = group.loc[group.year.eq(2025), col]
            seed = group.loc[group.year.eq(2022), col]
            row[f"development_2023_2024_{col}"] = float(development.mean())
            row[f"temporal_check_2025_{col}"] = (
                float(check.iloc[0]) if len(check) else np.nan
            )
            row[f"seed_2022_{col}"] = (
                float(seed.iloc[0]) if len(seed) else np.nan
            )
        periods.append(row)
    return by_year, pair_by_year, pd.DataFrame(periods)


def calibration_error_summary(
    calibrations: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    observed = calibrations.merge(
        diagnostic[
            diagnostic.actual_salary_recorded.eq(1)
        ][["year", "player_key", "salary_residual"]],
        on=["year", "player_key"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for period, years in (
        ("seed_2022", [2022]),
        ("development_2023_2024", [2023, 2024]),
        ("temporal_check_2025", [2025]),
    ):
        period_rows = observed[observed.year.isin(years)]
        if period_rows.empty:
            continue
        for variant, surcharge_col in (
            ("baseline_298", None),
            ("targeted_half", "surcharge_half"),
            ("targeted_full", "surcharge_full"),
        ):
            surcharge = (
                np.zeros(len(period_rows))
                if surcharge_col is None
                else period_rows[surcharge_col].to_numpy(dtype=float)
            )
            error = period_rows.salary_residual.to_numpy(dtype=float) - surcharge
            weights = period_rows.selection_rate.to_numpy(dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(len(period_rows))
            rows.append(
                {
                    "period": period,
                    "variant": variant,
                    "rows": int(len(period_rows)),
                    "mean_error": float(error.mean()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "selection_weighted_mean_error": float(
                        np.average(error, weights=weights)
                    ),
                    "selection_weighted_mae": float(
                        np.average(np.abs(error), weights=weights)
                    ),
                    "selection_weighted_rmse": float(
                        np.sqrt(np.average(error**2, weights=weights))
                    ),
                    "mean_surcharge": float(surcharge.mean()),
                    "selection_weighted_surcharge": float(
                        np.average(surcharge, weights=weights)
                    ),
                }
            )
    return pd.DataFrame(rows)


def fixed_roster_gap_summary(calibrations: pd.DataFrame) -> pd.DataFrame:
    rosters = pd.read_csv(BASELINE_FRONTIER)
    rosters = rosters[
        rosters.year.isin(calibrations.year.unique())
    ].copy()
    exploded = rosters[
        ["year", "trial", "chance_level", "roster"]
    ].copy()
    exploded["player"] = exploded.roster.str.split("|")
    exploded = exploded.explode("player")
    merged = exploded.merge(
        calibrations[
            [
                "year",
                "player",
                "point_salary",
                "surcharge_half",
                "surcharge_full",
            ]
        ],
        on=["year", "player"],
        how="left",
        validate="many_to_one",
    )
    diagnostic = pd.read_csv(SELECTION_DIAGNOSTIC)[
        ["year", "player", "actual_salary_used_in_replay"]
    ]
    merged = merged.merge(
        diagnostic,
        on=["year", "player"],
        how="left",
        validate="many_to_one",
    )
    if merged.point_salary.isna().any() or merged.actual_salary_used_in_replay.isna().any():
        raise ValueError("A baseline frontier roster did not reconcile to calibration rows.")
    merged["baseline_298"] = merged.point_salary
    merged["targeted_half"] = merged.point_salary + merged.surcharge_half
    merged["targeted_full"] = merged.point_salary + merged.surcharge_full
    roster_spend = merged.groupby(
        ["year", "trial", "chance_level"],
        as_index=False,
    ).agg(
        actual_salary=("actual_salary_used_in_replay", "sum"),
        baseline_298=("baseline_298", "sum"),
        targeted_half=("targeted_half", "sum"),
        targeted_full=("targeted_full", "sum"),
    )
    rows = []
    for period, years in (
        ("seed_2022", [2022]),
        ("development_2023_2024", [2023, 2024]),
        ("temporal_check_2025", [2025]),
    ):
        period_rows = roster_spend[roster_spend.year.isin(years)]
        if period_rows.empty:
            continue
        for variant in ("baseline_298", "targeted_half", "targeted_full"):
            gap = period_rows.actual_salary - period_rows[variant]
            rows.append(
                {
                    "period": period,
                    "variant": variant,
                    "rosters": int(len(period_rows)),
                    "mean_actual_minus_price_gap": float(gap.mean()),
                    "mean_absolute_gap": float(np.abs(gap).mean()),
                    "mean_modeled_spend": float(period_rows[variant].mean()),
                    "mean_actual_spend": float(period_rows.actual_salary.mean()),
                }
            )
    return pd.DataFrame(rows)


def feasible_roster_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, variant), group in trials.groupby(["year", "variant"]):
        feasible = group[group.actual_cap_feasible]
        rows.append(
            {
                "year": int(year),
                "variant": variant,
                "trials": int(len(group)),
                "feasible_trials": int(len(feasible)),
                "cap_feasible_rate": float(group.actual_cap_feasible.mean()),
                "actual_points_all": float(group.actual_points.mean()),
                "actual_points_feasible_only": float(
                    feasible.actual_points.mean()
                ),
                "forecast_ev_feasible_only": float(feasible.forecast_ev.mean()),
                "actual_salary_spend_feasible_only": float(
                    feasible.actual_salary_spend.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_trials(trials: pd.DataFrame) -> dict[str, Any]:
    if not trials.status.eq("optimal").all():
        raise AssertionError("At least one replay cell was infeasible.")
    if trials.duplicated(["year", "trial", "variant"]).any():
        raise AssertionError("Replay contains duplicate cells.")
    if not trials.groupby(["year", "trial"]).size().eq(len(VARIANTS)).all():
        raise AssertionError("A paired replay trial is incomplete.")
    rosters = trials.roster.str.split("|")
    if not rosters.map(len).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A replay roster does not contain 13 players.")
    if not rosters.map(lambda value: len(set(value))).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A replay roster contains duplicate players.")
    if (trials.sampled_salary_spend > base.SALARY_CAP + 1e-4).any():
        raise AssertionError("A sampled market cap was exceeded.")
    policy_violation = trials.decision_salary_spend > trials.policy_cap + 1e-4
    if policy_violation.any():
        raise AssertionError("A decision-price policy cap was exceeded.")
    if (trials.surcharge_spend < -1e-10).any():
        raise AssertionError("A negative surcharge was applied.")
    if (~trials.contains_top_n).any():
        raise AssertionError("A Top-N constraint was violated.")
    for pos in base.POSITIONS:
        col = f"{pos.lower()}_count"
        if (
            trials[col].lt(base.POS_MIN[pos]).any()
            or trials[col].gt(base.POS_MAX[pos]).any()
        ):
            raise AssertionError(f"A {pos} roster limit was violated.")
    expected = trials.actual_salary_spend.le(base.SALARY_CAP + 1e-8)
    if not trials.actual_cap_feasible.eq(expected).all():
        raise AssertionError("Actual cap feasibility does not match spend.")
    seed = trials[trials.year.eq(2022)].pivot(
        index="trial",
        columns="variant",
        values="roster",
    )
    for variant in ("targeted_half", "targeted_full"):
        if not seed[variant].eq(seed.baseline_298).all():
            raise AssertionError("The no-history seed surcharge changed a 2022 roster.")
    return {
        "all_cells_optimal": True,
        "all_rosters_size_13_unique": True,
        "all_policy_caps_satisfied": True,
        "all_position_and_top_n_constraints_satisfied": True,
        "actual_cap_flags_match_spend": True,
        "seed_surcharge_is_zero": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
    )
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--contexts", type=int, default=250)
    parser.add_argument("--context-draws", type=int, default=5)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    args = parser.parse_args()
    if sorted(set(args.years) - {2022, 2023, 2024, 2025}):
        parser.error("Only rolling salary origins 2022-2025 are supported.")
    if min(
        args.trials,
        args.contexts,
        args.context_draws,
        args.projection_draws,
        args.salary_draws,
    ) <= 0:
        parser.error("Draw and trial counts must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    diagnostic = load_selection_diagnostic()
    salary_rows, source_rows = current.load_salary_tables()
    raw_weekly = base.load_raw_weekly(max_year=max(args.years))
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    all_trials = []
    all_surfaces = []
    all_calibrations = []
    all_coefficients = []
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
            year,
            forecast,
            raw_weekly,
            features,
            actual,
        )
        cache, template_audit = base.build_template_cache(
            year,
            forecast,
            features,
            raw_weekly,
        )
        template_audit["max_donor_is_causal"] = template_audit.max_donor_season.lt(
            year
        )
        if not template_audit.max_donor_is_causal.all():
            raise AssertionError("A construction template crossed the origin.")
        all_template_audit.append(template_audit)

        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game"]
        ].copy()
        player_data["salary"] = 1.0
        sim = base.make_simulation(year, player_data, cache)
        waiver_baseline = sim.estimate_waiver_baselines(
            num_teams=base.NUM_TEAMS,
            roster_size=base.ROSTER_SIZE,
        )
        candidate_idx = np.flatnonzero(
            ~outcome_labels.is_keeper.to_numpy(dtype=bool)
        )
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
        calibration, coefficients, calibration_meta = fit_origin_surcharge(
            year,
            surface,
            diagnostic,
        )
        if not np.allclose(
            calibration.point_salary.to_numpy(dtype=float),
            surface.point_salary.to_numpy(dtype=float),
        ):
            raise AssertionError("Calibration changed the v5 point-salary center.")
        calibration["year"] = year
        coefficients["year"] = year
        all_calibrations.append(calibration)
        all_coefficients.append(coefficients)

        candidate_forecast["salary"] = surface.point_salary.to_numpy(dtype=float)
        predictions = base.build_predictions(candidate_forecast, candidate_ppg)
        weekly, decisions, played = base.generate_construction_contexts(
            sim,
            predictions,
            args.contexts,
            args.seed + year,
        )
        eval_weekly, eval_decisions, eval_played = (
            base.generate_construction_contexts(
                sim,
                predictions,
                args.contexts,
                args.seed + 100_000 + year,
            )
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
            salary_draws,
            calibration,
            surface.salary_model_matched.to_numpy(dtype=bool),
            surface.espn_source_matched.to_numpy(dtype=bool),
            environment,
            weekly,
            decisions,
            played,
            eval_weekly,
            eval_decisions,
            eval_played,
            value_banks[
                (current.CURRENT_WAIVER, current.CURRENT_BENCH_WEIGHT)
            ],
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
        source_manifest.update(calibration_meta)
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
        print(
            f"{year}: complete in {time.perf_counter() - year_started:.1f}s",
            flush=True,
        )

    trials = pd.concat(all_trials, ignore_index=True)
    surfaces = pd.concat(all_surfaces, ignore_index=True)
    calibrations = pd.concat(all_calibrations, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    template_audit = pd.concat(all_template_audit, ignore_index=True)
    expected_rows = len(args.years) * args.trials * len(VARIANTS)
    if len(trials) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} cells, found {len(trials)}.")
    validation = validate_trials(trials)
    if not template_audit.max_donor_is_causal.all():
        raise AssertionError("Template timing validation failed.")

    pairs = pd.concat(
        [pair_comparison(trials, variant) for variant in VARIANTS[1:]],
        ignore_index=True,
    )
    variant_by_year, pair_by_year, pair_periods = summarize(trials, pairs)
    calibration_errors = calibration_error_summary(calibrations, diagnostic)
    fixed_roster_gaps = fixed_roster_gap_summary(calibrations)
    feasible_rosters = feasible_roster_summary(trials)
    outputs = {
        "roster_trials.csv": trials,
        "salary_surface_audit.csv": surfaces,
        "calibration_predictions.csv": calibrations,
        "calibration_coefficients.csv": coefficients,
        "calibration_error_summary.csv": calibration_errors,
        "fixed_roster_gap_summary.csv": fixed_roster_gaps,
        "feasible_roster_summary.csv": feasible_rosters,
        "template_pool_audit.csv": template_audit,
        "paired_effects.csv": pairs,
        "variant_summary_by_year.csv": variant_by_year,
        "paired_effects_by_year.csv": pair_by_year,
        "paired_effects_development_check.csv": pair_periods,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    display_periods = pair_periods[
        [
            "comparison",
            "development_2023_2024_mean_actual_points_effect",
            "development_2023_2024_mean_actual_cap_feasible_effect",
            "development_2023_2024_mean_actual_cap_overage_effect",
            "development_2023_2024_mean_forecast_ev_effect",
            "development_2023_2024_roster_changed_rate",
            "temporal_check_2025_mean_actual_points_effect",
            "temporal_check_2025_mean_actual_cap_feasible_effect",
            "temporal_check_2025_mean_actual_cap_overage_effect",
            "temporal_check_2025_mean_forecast_ev_effect",
        ]
    ]
    lines = [
        "# Optimizer Selection Surcharge Replay",
        "",
        (
            f"{args.trials} paired five-draw trials per origin. "
            "2022 seeds the calibration; development is 2023-2024 and "
            "2025 is the temporal check."
        ),
        "",
        "## Variant outcomes by year",
        "",
        base.markdown_table(
            variant_by_year,
            [
                "year",
                "variant",
                "actual_points",
                "cap_feasible_rate",
                "mean_cap_overage",
                "actual_salary_spend",
                "decision_salary_spend",
                "surcharge_spend",
                "forecast_ev",
            ],
            digits=3,
        ),
        "",
        "## Paired effects versus baseline",
        "",
        (
            "Positive points/feasibility favor the candidate; negative "
            "overage favors the candidate."
        ),
        "",
        base.markdown_table(display_periods, list(display_periods.columns), digits=3),
        "",
        "## Player-level calibration",
        "",
        base.markdown_table(
            calibration_errors,
            [
                "period",
                "variant",
                "mean_error",
                "mae",
                "selection_weighted_mean_error",
                "selection_weighted_mae",
                "selection_weighted_surcharge",
            ],
            digits=3,
        ),
        "",
        "## Fixed baseline-roster spend gap",
        "",
        base.markdown_table(
            fixed_roster_gaps,
            [
                "period",
                "variant",
                "mean_actual_minus_price_gap",
                "mean_absolute_gap",
                "mean_modeled_spend",
                "mean_actual_spend",
            ],
            digits=3,
        ),
        "",
        "## Feasible-only roster quality",
        "",
        (
            "These point means are conditional on each policy producing an "
            "actually affordable roster, so they are descriptive rather than "
            "a paired causal point comparison."
        ),
        "",
        base.markdown_table(
            feasible_rosters,
            [
                "year",
                "variant",
                "feasible_trials",
                "cap_feasible_rate",
                "actual_points_feasible_only",
                "forecast_ev_feasible_only",
                "actual_salary_spend_feasible_only",
            ],
            digits=3,
        ),
        "",
        "## Limits",
        "",
        (
            "- Selection frequency is a causal preseason seed-pass feature, "
            "but production use requires that initial optimizer pass."
        ),
        (
            "- The v5 method specification is retrospective even though every "
            "salary and surcharge fit rolls strictly by data origin."
        ),
        (
            "- Four seasons are four outcome units; trial counts measure Monte "
            "Carlo stability rather than additional independent seasons."
        ),
        (
            "- The surcharge is a decision-price reserve, not a claim that the "
            "coherent league-wide salary market has a larger total budget."
        ),
        "",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    manifest = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "variants": list(VARIANTS),
        "calibration": {
            "target": "actual_salary_minus_v5_point_salary",
            "ridge_alpha": RIDGE_ALPHA,
            "surcharge_cap": SURCHARGE_CAP,
            "positive_residuals_only": True,
            "features": list(
                calibration_features(calibrations.head(1)).columns
            ),
            "strict_prior_origin_fit": True,
            "seed_origin_without_surcharge": 2022,
        },
        "fixed_settings": {
            "salary_draw_count": 5,
            "baseline_cap": base.SALARY_CAP,
            "blanket_cap": BLANKET_CAP,
            "top_n": True,
            "waiver_source": current.CURRENT_WAIVER,
            "bench_weight": current.CURRENT_BENCH_WEIGHT,
            "position_min": base.POS_MIN,
            "position_max": base.POS_MAX,
        },
        "salary_identity": {
            "method_version": SALARY_METHOD,
            "model_spec_asof_year": MODEL_SPEC_YEAR,
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
            "selection_diagnostic": str(SELECTION_DIAGNOSTIC),
            "selection_diagnostic_sha256": base.sha256_file(
                SELECTION_DIAGNOSTIC
            ),
            "baseline_frontier": str(BASELINE_FRONTIER),
            "baseline_frontier_sha256": base.sha256_file(BASELINE_FRONTIER),
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
        "output_rows": {
            name: int(len(frame)) for name, frame in outputs.items()
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(
        f"Replay complete in {time.perf_counter() - started:.1f}s: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
