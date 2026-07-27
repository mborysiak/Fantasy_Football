"""Paired walk-forward ablation of v1 and v2 salary predictions."""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
RESULTS = STUDY_DIR / "results"
VALIDATION_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
CURRENT_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_current_salary_buffer_replay"
    / "run_replay.py"
)
CHANCE_RESULTS = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_salary_chance_frontier"
    / "results"
)
DIAGNOSTIC_RESULTS = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_selected_roster_salary_residuals"
    / "results"
)
METHODS = {
    "v1": "current_locked_spec_v1",
    "v2": "current_locked_spec_v2_ensemble_features",
}
MODEL_SPEC_YEAR = 2026
PRICE_TIER_ORDER = ["$1-5", "$6-15", "$16-30", "$31-50", "$51+"]
QUANTILES = {
    0.05: "salary_resid_5",
    0.10: "salary_resid_10",
    0.25: "salary_resid_25",
    0.75: "salary_resid_75",
    0.90: "salary_resid_90",
    0.95: "salary_resid_95",
}


def load_current_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_salary_ensemble_ablation_current", CURRENT_RUNNER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import replay helper: {CURRENT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


current = load_current_runner()
base = current.base


def periods(
    data: pd.DataFrame,
    include_full_validation_development: bool = True,
) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all_years", data
    if include_full_validation_development:
        yield "development_2021_2024", data[data.year.le(2024)]
    yield "replay_development_2022_2024", data[data.year.between(2022, 2024)]
    yield "temporal_check_2025", data[data.year.eq(2025)]
    for year, group in data.groupby("year", sort=True):
        yield str(int(year)), group


def prediction_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    residual = actual_values - predicted_values
    squared = residual**2
    denominator = np.sum((actual_values - np.mean(actual_values)) ** 2)
    return {
        "mean_residual": float(np.mean(residual)),
        "absolute_bias": float(abs(np.mean(residual))),
        "mae": float(np.mean(np.abs(residual))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(squared))),
        "r2": float(1.0 - np.sum(squared) / denominator)
        if denominator > 0
        else np.nan,
        "positive_residual_rate": float(np.mean(residual > 0)),
        "mean_predicted_salary": float(np.mean(predicted_values)),
        "mean_actual_salary": float(np.mean(actual_values)),
    }


def read_salary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    placeholders = ",".join("?" for _ in METHODS)
    params = [*METHODS.values(), MODEL_SPEC_YEAR]
    with sqlite3.connect(f"file:{VALIDATION_DB}?mode=ro", uri=True) as conn:
        validations = pd.read_sql_query(
            f"""SELECT * FROM Salary_Validations_Resid
                WHERE league='beta'
                  AND method_version IN ({placeholders})
                  AND model_spec_asof_year=?
                  AND included_in_residual_evaluation=1""",
            conn,
            params=params,
        )
        backtest = pd.read_sql_query(
            f"""SELECT * FROM Salary_Backtest_Predictions
                WHERE league='beta'
                  AND method_version IN ({placeholders})
                  AND model_spec_asof_year=?""",
            conn,
            params=params,
        )
    validations = base.add_identity(validations)
    backtest = base.add_identity(backtest)
    for name, frame in [("validations", validations), ("backtest", backtest)]:
        if frame.duplicated(["method_version", "year", "player_key"]).any():
            raise AssertionError(f"{name} contain duplicate method/player/year rows.")
        if not frame.normalization_uses_target_actuals.eq(0).all():
            raise AssertionError(f"{name} normalization used target-year actuals.")
        if not (
            frame.training_through_year.eq(frame.year.astype(int) - 1)
        ).all():
            raise AssertionError(f"{name} training cutoff crossed an origin.")
    if validations.is_keeper.ne(0).any():
        raise AssertionError("Residual validation rows unexpectedly contain keepers.")
    if set(backtest.year.astype(int)) != {2022, 2023, 2024, 2025}:
        raise AssertionError("Backtest origins are incomplete.")
    return validations, backtest


def pair_validation_rows(
    validations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "year",
        "player_key",
        "player",
        "pos",
        "actual_salary",
        "pred_salary",
        "pred_salary_raw",
        "actual_resid",
        "actual_resid_raw",
        "resid_training_rows",
        *QUANTILES.values(),
    ]
    sides = {}
    for label, method in METHODS.items():
        rename = {
            column: f"{column}_{label}"
            for column in columns
            if column not in {"year", "player_key"}
        }
        sides[label] = (
            validations[validations.method_version.eq(method)][columns]
            .copy()
            .rename(columns=rename)
        )
    outer = sides["v1"].merge(
        sides["v2"],
        on=["year", "player_key"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    coverage_rows = []
    for period, group in periods(outer, include_full_validation_development=True):
        counts = group["_merge"].value_counts()
        coverage_rows.append(
            {
                "period": period,
                "v1_rows": int(counts.get("both", 0) + counts.get("left_only", 0)),
                "v2_rows": int(counts.get("both", 0) + counts.get("right_only", 0)),
                "paired_rows": int(counts.get("both", 0)),
                "v1_only_rows": int(counts.get("left_only", 0)),
                "v2_only_rows": int(counts.get("right_only", 0)),
            }
        )
    unmatched = outer[outer["_merge"].ne("both")].copy()
    paired = outer[outer["_merge"].eq("both")].drop(columns="_merge").copy()
    if not np.allclose(
        paired.actual_salary_v1,
        paired.actual_salary_v2,
        atol=1e-10,
    ):
        raise AssertionError("Paired methods disagree on actual salary.")
    paired["reference_pred_salary"] = (
        paired.pred_salary_v1 + paired.pred_salary_v2
    ) / 2.0
    paired["reference_price_tier"] = pd.cut(
        paired.reference_pred_salary,
        bins=[-np.inf, 5.0, 15.0, 30.0, 50.0, np.inf],
        labels=PRICE_TIER_ORDER,
    )
    paired["normalized_price_shift_v2_minus_v1"] = (
        paired.pred_salary_v2 - paired.pred_salary_v1
    )
    paired["raw_price_shift_v2_minus_v1"] = (
        paired.pred_salary_raw_v2 - paired.pred_salary_raw_v1
    )
    paired["normalized_abs_error_v1"] = (
        paired.actual_salary_v1 - paired.pred_salary_v1
    ).abs()
    paired["normalized_abs_error_v2"] = (
        paired.actual_salary_v2 - paired.pred_salary_v2
    ).abs()
    paired["normalized_abs_error_improvement_v2"] = (
        paired.normalized_abs_error_v1 - paired.normalized_abs_error_v2
    )
    return paired, pd.DataFrame(coverage_rows), unmatched


def paired_accuracy_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, suffix in [("normalized", ""), ("raw", "_raw")]:
        for period, group in periods(paired, include_full_validation_development=True):
            row: dict[str, Any] = {
                "prediction_scale": scale,
                "period": period,
                "player_years": int(len(group)),
            }
            for label in METHODS:
                metrics = prediction_metrics(
                    group[f"actual_salary_{label}"],
                    group[f"pred_salary{suffix}_{label}"],
                )
                row.update({f"{label}_{key}": value for key, value in metrics.items()})
            for metric in [
                "absolute_bias",
                "mae",
                "median_absolute_error",
                "rmse",
            ]:
                row[f"{metric}_delta_v2_minus_v1"] = (
                    row[f"v2_{metric}"] - row[f"v1_{metric}"]
                )
            row["r2_delta_v2_minus_v1"] = row["v2_r2"] - row["v1_r2"]
            rows.append(row)
    return pd.DataFrame(rows)


def grouped_paired_accuracy(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = [
        ("position", ["pos_v1"]),
        ("reference_price_tier", ["reference_price_tier"]),
        ("year_position", ["year", "pos_v1"]),
    ]
    for grouping, columns in group_specs:
        for keys, group in paired.groupby(columns, observed=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "grouping": grouping,
                "player_years": int(len(group)),
            }
            row.update(dict(zip(columns, keys)))
            for label in METHODS:
                metrics = prediction_metrics(
                    group[f"actual_salary_{label}"],
                    group[f"pred_salary_{label}"],
                )
                row.update({f"{label}_{key}": value for key, value in metrics.items()})
            row["mae_delta_v2_minus_v1"] = row["v2_mae"] - row["v1_mae"]
            row["rmse_delta_v2_minus_v1"] = row["v2_rmse"] - row["v1_rmse"]
            row["absolute_bias_delta_v2_minus_v1"] = (
                row["v2_absolute_bias"] - row["v1_absolute_bias"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_calibration(validations: pd.DataFrame) -> pd.DataFrame:
    usable = validations[validations.resid_training_rows.gt(0)].copy()
    rows = []
    for label, method in METHODS.items():
        method_rows = usable[usable.method_version.eq(method)]
        for period, group in periods(
            method_rows, include_full_validation_development=False
        ):
            if group.empty:
                continue
            row: dict[str, Any] = {
                "method": label,
                "period": period,
                "player_years": int(len(group)),
                "mean_actual_residual": float(group.actual_resid.mean()),
            }
            for probability, column in QUANTILES.items():
                empirical = float(group.actual_resid.le(group[column]).mean())
                row[f"empirical_le_q{int(probability * 100)}"] = empirical
                row[f"coverage_error_q{int(probability * 100)}"] = (
                    empirical - probability
                )
            row["central_50_coverage"] = float(
                group.actual_resid.between(
                    group.salary_resid_25,
                    group.salary_resid_75,
                    inclusive="both",
                ).mean()
            )
            row["central_80_coverage"] = float(
                group.actual_resid.between(
                    group.salary_resid_10,
                    group.salary_resid_90,
                    inclusive="both",
                ).mean()
            )
            row["central_90_coverage"] = float(
                group.actual_resid.between(
                    group.salary_resid_5,
                    group.salary_resid_95,
                    inclusive="both",
                ).mean()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def normalized_point_values(
    raw_values: np.ndarray,
    remaining_budget: float,
    remaining_slots: int,
) -> np.ndarray:
    return np.asarray(
        base.FootballSimulation.normalize_salary_market_values(
            np.asarray(raw_values, dtype=float),
            np.ones(len(raw_values), dtype=bool),
            remaining_market_budget=remaining_budget,
            remaining_market_slots=remaining_slots,
        ),
        dtype=float,
    )


def add_value_ranks(candidates: pd.DataFrame, label: str) -> pd.DataFrame:
    output = candidates.copy()
    output[f"projection_strength_pct_{label}"] = output.groupby(
        ["year", "pos"]
    ).pred_fp_per_game.rank(method="average", pct=True, ascending=True)
    output[f"price_strength_pct_{label}"] = output.groupby(
        ["year", "pos"]
    )[f"point_salary_{label}"].rank(method="average", pct=True, ascending=True)
    output[f"value_rank_gap_{label}"] = (
        output[f"projection_strength_pct_{label}"]
        - output[f"price_strength_pct_{label}"]
    )
    output[f"value_quintile_{label}"] = output.groupby(
        ["year", "pos"]
    )[f"value_rank_gap_{label}"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            5,
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    output[f"predicted_salary_tier_{label}"] = pd.cut(
        output[f"point_salary_{label}"],
        bins=[-np.inf, 5.0, 15.0, 30.0, 50.0, np.inf],
        labels=PRICE_TIER_ORDER,
    )
    return output


def build_candidate_surface(backtest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = pd.read_csv(DIAGNOSTIC_RESULTS / "candidate_diagnostic.csv")
    manifest = json.loads(
        (CHANCE_RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    if candidates.duplicated(["year", "player_key"]).any():
        raise AssertionError("Prior candidate surface has duplicate player-origins.")
    for label, method in METHODS.items():
        prices = backtest[backtest.method_version.eq(method)][
            ["year", "player_key", "pred_salary", "pred_salary_raw"]
        ].rename(
            columns={
                "pred_salary": f"stored_pred_salary_{label}",
                "pred_salary_raw": f"pred_salary_raw_{label}",
            }
        )
        candidates = candidates.merge(
            prices,
            on=["year", "player_key"],
            how="left",
            validate="one_to_one",
        )
        candidates[f"salary_model_matched_{label}"] = candidates[
            f"pred_salary_raw_{label}"
        ].notna()
        candidates[f"raw_point_salary_{label}"] = (
            candidates[f"pred_salary_raw_{label}"]
            .where(
                candidates[f"salary_model_matched_{label}"],
                candidates.espn_source_salary,
            )
            .fillna(0.0)
            .clip(lower=1.0)
        )
        candidates[f"point_salary_{label}"] = np.nan
        for year, indices in candidates.groupby("year", sort=True).groups.items():
            origin = manifest["origins"][str(int(year))]
            normalized = normalized_point_values(
                candidates.loc[indices, f"raw_point_salary_{label}"].to_numpy(),
                float(origin["remaining_budget"]),
                int(origin["remaining_slots"]),
            )
            candidates.loc[indices, f"point_salary_{label}"] = normalized
            top_total = float(
                np.sort(normalized)[-int(origin["remaining_slots"]) :].sum()
            )
            if not math.isclose(
                top_total,
                float(origin["remaining_budget"]),
                abs_tol=1e-6,
                rel_tol=0.0,
            ):
                raise AssertionError(f"{label} {year} normalized market is imbalanced.")
        candidates = add_value_ranks(candidates, label)
        candidates[f"salary_residual_{label}"] = (
            candidates.actual_salary - candidates[f"point_salary_{label}"]
        )
    reconstruction_error = float(
        np.max(np.abs(candidates.point_salary_v1 - candidates.point_salary))
    )
    if reconstruction_error > 1e-5:
        raise AssertionError("Could not reproduce the prior v1 point-salary surface.")
    candidates["point_salary_shift_v2_minus_v1"] = (
        candidates.point_salary_v2 - candidates.point_salary_v1
    )
    return candidates, {
        "v1_point_surface_max_abs_reconstruction_error": reconstruction_error,
        "candidate_player_origins": int(len(candidates)),
        "recorded_actual_player_origins": int(candidates.actual_salary_recorded.sum()),
        "v1_salary_model_matches": int(candidates.salary_model_matched_v1.sum()),
        "v2_salary_model_matches": int(candidates.salary_model_matched_v2.sum()),
    }


def candidate_accuracy_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[candidates.actual_salary_recorded].copy()
    rows = []
    for period, group in periods(observed, include_full_validation_development=False):
        for label in METHODS:
            residual = group[f"salary_residual_{label}"].to_numpy(dtype=float)
            weights = group.selection_slots.to_numpy(dtype=float)
            rows.append(
                {
                    "period": period,
                    "method": label,
                    "player_origins": int(len(group)),
                    "selected_slots": int(weights.sum()),
                    "mean_residual": float(np.mean(residual)),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "positive_residual_rate": float(np.mean(residual > 0)),
                    "selection_weighted_mean_residual": float(
                        np.average(residual, weights=weights)
                    )
                    if weights.sum() > 0
                    else np.nan,
                    "selection_weighted_mae": float(
                        np.average(np.abs(residual), weights=weights)
                    )
                    if weights.sum() > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def value_tail_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[candidates.actual_salary_recorded].copy()
    rows = []
    for label in METHODS:
        for period, period_rows in periods(
            observed, include_full_validation_development=False
        ):
            for quintile, group in period_rows.groupby(
                f"value_quintile_{label}", observed=True
            ):
                residual = group[f"salary_residual_{label}"].to_numpy(dtype=float)
                weights = group.selection_slots.to_numpy(dtype=float)
                rows.append(
                    {
                        "method": label,
                        "period": period,
                        "value_quintile": int(quintile),
                        "player_origins": int(len(group)),
                        "selected_slots_under_prior_v1_optimizer": int(weights.sum()),
                        "mean_residual": float(np.mean(residual)),
                        "mae": float(np.mean(np.abs(residual))),
                        "positive_residual_rate": float(np.mean(residual > 0)),
                        "selection_weighted_mean_residual": float(
                            np.average(residual, weights=weights)
                        )
                        if weights.sum() > 0
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def selection_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[candidates.actual_salary_recorded].copy()
    rows = []
    for label in METHODS:
        for bucket, group in observed.groupby("selection_bucket", observed=True):
            residual = group[f"salary_residual_{label}"].to_numpy(dtype=float)
            weights = group.selection_slots.to_numpy(dtype=float)
            rows.append(
                {
                    "method": label,
                    "selection_bucket_from_prior_v1_optimizer": str(bucket),
                    "player_origins": int(len(group)),
                    "selected_slots": int(weights.sum()),
                    "mean_residual": float(np.mean(residual)),
                    "positive_residual_rate": float(np.mean(residual > 0)),
                    "selection_weighted_mean_residual": float(
                        np.average(residual, weights=weights)
                    )
                    if weights.sum() > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def price_shift_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = [
        ("period", None),
        ("year", ["year"]),
        ("position", ["pos"]),
        ("year_position", ["year", "pos"]),
        ("prior_v1_selection_bucket", ["selection_bucket"]),
        ("prior_v1_value_quintile", ["value_quintile_v1"]),
    ]
    for grouping, columns in group_specs:
        groups: Iterable[tuple[Any, pd.DataFrame]]
        if columns is None:
            groups = periods(candidates, include_full_validation_development=False)
        else:
            groups = candidates.groupby(columns, observed=True, dropna=False)
        for keys, group in groups:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "grouping": grouping,
                "player_origins": int(len(group)),
                "mean_shift_v2_minus_v1": float(
                    group.point_salary_shift_v2_minus_v1.mean()
                ),
                "median_shift_v2_minus_v1": float(
                    group.point_salary_shift_v2_minus_v1.median()
                ),
                "p10_shift_v2_minus_v1": float(
                    group.point_salary_shift_v2_minus_v1.quantile(0.10)
                ),
                "p90_shift_v2_minus_v1": float(
                    group.point_salary_shift_v2_minus_v1.quantile(0.90)
                ),
                "mean_prior_v1_selection_rate": float(group.selection_rate.mean()),
            }
            if columns is None:
                row["group_value"] = keys[0]
            else:
                row["group_value"] = "|".join(str(value) for value in keys)
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_roster_repricing(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rosters = pd.read_csv(CHANCE_RESULTS / "roster_trials.csv")
    slots = rosters[["year", "trial", "chance_level", "roster"]].copy()
    slots["player"] = slots.roster.str.split("|")
    slots = slots.explode("player", ignore_index=True)
    slots = base.add_identity(slots)
    slots = slots.merge(
        candidates[
            [
                "year",
                "player_key",
                "point_salary_v1",
                "point_salary_v2",
                "actual_salary_used_in_replay",
            ]
        ],
        on=["year", "player_key"],
        how="left",
        validate="many_to_one",
    )
    if slots.point_salary_v2.isna().any():
        raise AssertionError("A selected roster slot did not join to the v2 surface.")
    repriced = slots.groupby(
        ["year", "trial", "chance_level"], as_index=False
    ).agg(
        point_salary_spend_v1=("point_salary_v1", "sum"),
        point_salary_spend_v2=("point_salary_v2", "sum"),
        actual_salary_spend_reconstructed=("actual_salary_used_in_replay", "sum"),
    )
    repriced = repriced.merge(
        rosters[
            [
                "year",
                "trial",
                "chance_level",
                "point_salary_spend",
                "actual_salary_spend",
            ]
        ],
        on=["year", "trial", "chance_level"],
        validate="one_to_one",
    )
    point_error = float(
        np.max(
            np.abs(
                repriced.point_salary_spend_v1 - repriced.point_salary_spend
            )
        )
    )
    actual_error = float(
        np.max(
            np.abs(
                repriced.actual_salary_spend_reconstructed
                - repriced.actual_salary_spend
            )
        )
    )
    if point_error > 1e-4 or actual_error > 1e-8:
        raise AssertionError("Could not reproduce prior roster spending.")
    repriced["point_spend_shift_v2_minus_v1"] = (
        repriced.point_salary_spend_v2 - repriced.point_salary_spend_v1
    )
    repriced["actual_minus_point_v1"] = (
        repriced.actual_salary_spend - repriced.point_salary_spend_v1
    )
    repriced["actual_minus_point_v2"] = (
        repriced.actual_salary_spend - repriced.point_salary_spend_v2
    )
    repriced["point_v1_at_or_below_cap"] = repriced.point_salary_spend_v1.le(
        base.SALARY_CAP
    )
    repriced["point_v2_at_or_below_cap"] = repriced.point_salary_spend_v2.le(
        base.SALARY_CAP
    )
    rows = []
    for period, group in periods(repriced, include_full_validation_development=False):
        chance_groups: list[tuple[str, pd.DataFrame]] = [("all", group)]
        chance_groups.extend(
            (f"{chance:.1f}", chance_rows)
            for chance, chance_rows in group.groupby("chance_level", sort=True)
        )
        for chance, chance_rows in chance_groups:
            rows.append(
                {
                    "period": period,
                    "chance_level": chance,
                    "rosters": int(len(chance_rows)),
                    "mean_point_spend_v1": float(
                        chance_rows.point_salary_spend_v1.mean()
                    ),
                    "mean_point_spend_v2": float(
                        chance_rows.point_salary_spend_v2.mean()
                    ),
                    "mean_point_spend_shift_v2_minus_v1": float(
                        chance_rows.point_spend_shift_v2_minus_v1.mean()
                    ),
                    "mean_actual_minus_point_v1": float(
                        chance_rows.actual_minus_point_v1.mean()
                    ),
                    "mean_actual_minus_point_v2": float(
                        chance_rows.actual_minus_point_v2.mean()
                    ),
                    "actual_minus_point_gap_change_v2_minus_v1": float(
                        (
                            chance_rows.actual_minus_point_v2
                            - chance_rows.actual_minus_point_v1
                        ).mean()
                    ),
                    "point_v1_at_or_below_cap_rate": float(
                        chance_rows.point_v1_at_or_below_cap.mean()
                    ),
                    "point_v2_at_or_below_cap_rate": float(
                        chance_rows.point_v2_at_or_below_cap.mean()
                    ),
                }
            )
    return repriced, pd.DataFrame(rows), {
        "fixed_roster_rows": int(len(repriced)),
        "fixed_roster_slots": int(len(slots)),
        "v1_roster_point_spend_max_abs_reconstruction_error": point_error,
        "actual_roster_spend_max_abs_reconstruction_error": actual_error,
    }


def write_summary(
    accuracy: pd.DataFrame,
    coverage: pd.DataFrame,
    candidate_accuracy: pd.DataFrame,
    value_tail: pd.DataFrame,
    repricing_summary: pd.DataFrame,
) -> None:
    overall = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.isin(
            ["all_years", "replay_development_2022_2024", "temporal_check_2025"]
        )
    ]
    candidate_overall = candidate_accuracy[
        candidate_accuracy.period.isin(
            ["all_years", "replay_development_2022_2024", "temporal_check_2025"]
        )
    ]
    top_value = value_tail[
        value_tail.period.eq("all_years") & value_tail.value_quintile.eq(5)
    ]
    fixed = repricing_summary[
        repricing_summary.chance_level.eq("all")
        & repricing_summary.period.isin(
            ["all_years", "replay_development_2022_2024", "temporal_check_2025"]
        )
    ]
    lines = [
        "# Salary Ensemble-Feature Ablation",
        "",
        "## Paired observed salary accuracy",
        "",
        base.markdown_table(
            overall,
            [
                "period",
                "player_years",
                "v1_mean_residual",
                "v2_mean_residual",
                "v1_mae",
                "v2_mae",
                "mae_delta_v2_minus_v1",
                "v1_rmse",
                "v2_rmse",
            ],
            digits=3,
        ),
        "",
        "Negative MAE/RMSE deltas favor v2. Residual is actual minus predicted.",
        "",
        "## Coverage",
        "",
        base.markdown_table(
            coverage[coverage.period.isin(["all_years", "temporal_check_2025"])],
            [
                "period",
                "v1_rows",
                "v2_rows",
                "paired_rows",
                "v1_only_rows",
                "v2_only_rows",
            ],
            digits=0,
        ),
        "",
        "## Frozen replay candidate universe",
        "",
        base.markdown_table(
            candidate_overall,
            [
                "period",
                "method",
                "player_origins",
                "mean_residual",
                "mae",
                "rmse",
                "selection_weighted_mean_residual",
            ],
            digits=3,
        ),
        "",
        "## Strongest within-position value quintile",
        "",
        base.markdown_table(
            top_value,
            [
                "method",
                "player_origins",
                "mean_residual",
                "positive_residual_rate",
                "selection_weighted_mean_residual",
            ],
            digits=3,
        ),
        "",
        "## Fixed prior-v1 roster repricing",
        "",
        base.markdown_table(
            fixed,
            [
                "period",
                "rosters",
                "mean_point_spend_shift_v2_minus_v1",
                "mean_actual_minus_point_v1",
                "mean_actual_minus_point_v2",
                "actual_minus_point_gap_change_v2_minus_v1",
            ],
            digits=3,
        ),
        "",
        "These are the same rosters selected under v1. The repricing isolates the salary surface; it does not measure which rosters a v2 optimizer would select.",
        "",
        "## Interpretation limits",
        "",
        "- The data cutoff rolls by origin, but the model family and features are retrospectively locked as of 2026.",
        "- Player-year rows within a season are not independent season-level outcome units.",
        "- Fixed-roster repricing is deliberately conditional on old v1 selections and cannot replace a v2 optimizer replay.",
        "- Historical actual-price fallbacks retain the prior replay's intentional `$1` treatment for unrecorded auction prices.",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_decision_readout(
    accuracy: pd.DataFrame,
    candidate_accuracy: pd.DataFrame,
    value_tail: pd.DataFrame,
    repricing_summary: pd.DataFrame,
) -> None:
    paired = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.eq("all_years")
    ].iloc[0]
    check = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.eq("temporal_check_2025")
    ].iloc[0]
    candidate = candidate_accuracy[
        candidate_accuracy.period.eq("all_years")
    ].set_index("method")
    top = value_tail[
        value_tail.period.eq("all_years") & value_tail.value_quintile.eq(5)
    ].set_index("method")
    dev_rosters = repricing_summary[
        repricing_summary.period.eq("replay_development_2022_2024")
        & repricing_summary.chance_level.eq("all")
    ].iloc[0]
    check_rosters = repricing_summary[
        repricing_summary.period.eq("temporal_check_2025")
        & repricing_summary.chance_level.eq("all")
    ].iloc[0]
    lines = [
        "# Decision Readout",
        "",
        "## Finding",
        "",
        (
            "v2 reduces the salary model's average underprediction bias, but it "
            "does not improve ordinary absolute error consistently."
        ),
        "",
        (
            f"Across {int(paired.player_years)} common observed player-years, "
            f"mean residual moved from {paired.v1_mean_residual:+.2f} to "
            f"{paired.v2_mean_residual:+.2f}, while MAE changed from "
            f"{paired.v1_mae:.2f} to {paired.v2_mae:.2f} "
            f"({paired.mae_delta_v2_minus_v1:+.2f})."
        ),
        (
            f"In the 2025 temporal check, MAE changed from {check.v1_mae:.2f} "
            f"to {check.v2_mae:.2f} "
            f"({check.mae_delta_v2_minus_v1:+.2f})."
        ),
        "",
        "## Optimizer-relevant tail",
        "",
        (
            "On the frozen replay candidate universe, the strongest within-position "
            "value quintile's unique-player mean residual changed from "
            f"{top.loc['v1', 'mean_residual']:+.2f} to "
            f"{top.loc['v2', 'mean_residual']:+.2f}. Its old-v1-selection-weighted "
            "residual changed from "
            f"{top.loc['v1', 'selection_weighted_mean_residual']:+.2f} to "
            f"{top.loc['v2', 'selection_weighted_mean_residual']:+.2f}."
        ),
        (
            "Across every recorded candidate, the old-v1-selection-weighted "
            "residual changed from "
            f"{candidate.loc['v1', 'selection_weighted_mean_residual']:+.2f} to "
            f"{candidate.loc['v2', 'selection_weighted_mean_residual']:+.2f}."
        ),
        "",
        "## Fixed old-roster repricing",
        "",
        (
            "v2 prices the old v1-selected development rosters "
            f"{dev_rosters.mean_point_spend_shift_v2_minus_v1:+.2f} per roster "
            "relative to v1, changing their actual-minus-point gap from "
            f"{dev_rosters.mean_actual_minus_point_v1:.2f} to "
            f"{dev_rosters.mean_actual_minus_point_v2:.2f}."
        ),
        (
            "For 2025, v2 changes those same roster prices by "
            f"{check_rosters.mean_point_spend_shift_v2_minus_v1:+.2f}, and the "
            "actual-minus-point gap moves from "
            f"{check_rosters.mean_actual_minus_point_v1:.2f} to "
            f"{check_rosters.mean_actual_minus_point_v2:.2f}."
        ),
        "",
        "## Action",
        "",
        (
            "Do not declare v2 a general point-accuracy upgrade. Keep the added "
            "features as an optimizer-tail candidate, but require a paired v2 "
            "optimizer replay before making it the production salary surface. "
            "The replay is necessary because v2 improves the apparent strongest-"
            "value tail while repricing old rosters differently by season; the "
            "optimizer will respond by selecting a different set of players."
        ),
        "",
    ]
    (RESULTS / "decision_readout.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    validations, backtest = read_salary_tables()
    paired, coverage, unmatched = pair_validation_rows(validations)
    accuracy = paired_accuracy_summary(paired)
    grouped_accuracy = grouped_paired_accuracy(paired)
    calibration = quantile_calibration(validations)
    candidates, candidate_validation = build_candidate_surface(backtest)
    candidate_accuracy = candidate_accuracy_summary(candidates)
    value_tail = value_tail_summary(candidates)
    selection = selection_summary(candidates)
    price_shift = price_shift_summary(candidates)
    repriced, repricing_summary, roster_validation = fixed_roster_repricing(
        candidates
    )

    outputs = {
        "paired_validation_predictions.csv": paired,
        "validation_coverage.csv": coverage,
        "unmatched_validation_rows.csv": unmatched,
        "paired_accuracy_summary.csv": accuracy,
        "grouped_paired_accuracy.csv": grouped_accuracy,
        "residual_quantile_calibration.csv": calibration,
        "candidate_surface_v1_v2.csv": candidates,
        "candidate_accuracy_summary.csv": candidate_accuracy,
        "value_tail_summary.csv": value_tail,
        "prior_v1_selection_bucket_summary.csv": selection,
        "point_salary_shift_summary.csv": price_shift,
        "fixed_v1_rosters_repriced.csv": repriced,
        "fixed_roster_repricing_summary.csv": repricing_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS / filename, index=False)

    write_summary(
        accuracy,
        coverage,
        candidate_accuracy,
        value_tail,
        repricing_summary,
    )
    write_decision_readout(
        accuracy,
        candidate_accuracy,
        value_tail,
        repricing_summary,
    )
    manifest = {
        "study": STUDY_DIR.name,
        "methods": METHODS,
        "model_spec_asof_year": MODEL_SPEC_YEAR,
        "validation": {
            "common_observed_player_years": int(len(paired)),
            "unmatched_observed_player_years": int(len(unmatched)),
            **candidate_validation,
            **roster_validation,
        },
        "sources": {
            "validation_database": str(VALIDATION_DB),
            "validation_database_sha256": base.sha256_file(VALIDATION_DB),
            "current_salary_runner": str(CURRENT_RUNNER),
            "current_salary_runner_sha256": base.sha256_file(CURRENT_RUNNER),
            "chance_rosters": str(CHANCE_RESULTS / "roster_trials.csv"),
            "chance_rosters_sha256": base.sha256_file(
                CHANCE_RESULTS / "roster_trials.csv"
            ),
            "prior_candidate_diagnostic": str(
                DIAGNOSTIC_RESULTS / "candidate_diagnostic.csv"
            ),
            "prior_candidate_diagnostic_sha256": base.sha256_file(
                DIAGNOSTIC_RESULTS / "candidate_diagnostic.csv"
            ),
        },
        "outputs": {
            filename: int(len(frame)) for filename, frame in outputs.items()
        },
    }
    (RESULTS / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print((RESULTS / "decision_readout.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
