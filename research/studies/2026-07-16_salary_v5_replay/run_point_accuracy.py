"""Compare v1, v3, and v5 rolling salary accuracy and market coherence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
RESULTS = STUDY_DIR / "results"
VALIDATIONS_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
METHODS = {
    "v1": "current_locked_spec_v1",
    "v3": "current_locked_spec_v3_resid_share_features",
    "v5": "current_locked_spec_v5_compact_salary_features",
}
MODEL_SPEC_YEAR = 2026


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Format a small DataFrame without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def periods(data: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all_years", data
    yield "development_2021_2024", data[data.year.le(2024)]
    yield "replay_development_2022_2024", data[
        data.year.between(2022, 2024)
    ]
    yield "temporal_check_2025", data[data.year.eq(2025)]
    for year, group in data.groupby("year", sort=True):
        yield str(int(year)), group


def metrics(
    actual: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
    residual = actual.to_numpy(dtype=float) - prediction.to_numpy(dtype=float)
    denominator = np.sum(
        np.square(actual.to_numpy(dtype=float) - float(actual.mean()))
    )
    return {
        "mean_residual": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1 - np.square(residual).sum() / denominator)
        if denominator > 0
        else np.nan,
        "positive_residual_rate": float(np.mean(residual > 0)),
    }


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    placeholders = ",".join("?" for _ in METHODS)
    params = [*METHODS.values(), MODEL_SPEC_YEAR]
    with sqlite3.connect(VALIDATIONS_DB) as connection:
        validation = pd.read_sql_query(
            f"""
            SELECT *
            FROM Salary_Validations_Resid
            WHERE league='beta'
              AND method_version IN ({placeholders})
              AND model_spec_asof_year=?
              AND included_in_residual_evaluation=1
            """,
            connection,
            params=params,
        )
        backtest = pd.read_sql_query(
            f"""
            SELECT *
            FROM Salary_Backtest_Predictions
            WHERE league='beta'
              AND method_version IN ({placeholders})
              AND model_spec_asof_year=?
            """,
            connection,
            params=params,
        )
    for name, frame in [
        ("validation", validation),
        ("backtest", backtest),
    ]:
        if frame.duplicated(
            ["method_version", "year", "player"]
        ).any():
            raise AssertionError(f"{name} contains duplicate method/player rows.")
        if not frame.normalization_uses_target_actuals.eq(0).all():
            raise AssertionError(f"{name} used target actuals in normalization.")
    if set(backtest.year.astype(int)) != {2022, 2023, 2024, 2025}:
        raise AssertionError("Backtest origins are incomplete.")
    return validation, backtest


def accuracy_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    method_labels = {value: key for key, value in METHODS.items()}
    for method, method_rows in validation.groupby("method_version"):
        for scale, prediction_column in [
            ("normalized", "pred_salary"),
            ("raw", "pred_salary_raw"),
        ]:
            for period, selected in periods(method_rows):
                if selected.empty:
                    continue
                rows.append(
                    {
                        "method": method_labels[method],
                        "method_version": method,
                        "prediction_scale": scale,
                        "period": period,
                        "player_years": int(len(selected)),
                        **metrics(
                            selected.actual_salary,
                            selected[prediction_column],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def common_v1_v5(validation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "player",
        "pos",
        "actual_salary",
        "pred_salary",
        "pred_salary_raw",
        "base_salary",
    ]
    sides = {}
    for label in ["v1", "v5"]:
        method = METHODS[label]
        sides[label] = validation[
            validation.method_version.eq(method)
        ][columns].rename(
            columns={
                column: f"{column}_{label}"
                for column in columns
                if column not in {"year", "player"}
            }
        )
    paired = sides["v1"].merge(
        sides["v5"],
        on=["year", "player"],
        how="inner",
        validate="one_to_one",
    )
    if not np.allclose(
        paired.actual_salary_v1,
        paired.actual_salary_v5,
    ):
        raise AssertionError("Paired v1/v5 rows disagree on actual salary.")
    for scale, suffix in [("normalized", ""), ("raw", "_raw")]:
        for label in ["v1", "v5"]:
            paired[f"residual_{scale}_{label}"] = (
                paired[f"actual_salary_{label}"]
                - paired[f"pred_salary{suffix}_{label}"]
            )
            paired[f"absolute_error_{scale}_{label}"] = paired[
                f"residual_{scale}_{label}"
            ].abs()
        paired[f"absolute_error_improvement_{scale}_v5"] = (
            paired[f"absolute_error_{scale}_v1"]
            - paired[f"absolute_error_{scale}_v5"]
        )
    return paired


def grouped_paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paired = paired.copy()
    paired["actual_salary_tier"] = pd.cut(
        paired.actual_salary_v5,
        bins=[-np.inf, 5, 15, 30, 50, np.inf],
        labels=["$1-5", "$6-15", "$16-30", "$31-50", "$51+"],
    )
    groupings = [
        ("position", ["pos_v5"]),
        ("actual_salary_tier", ["actual_salary_tier"]),
        ("year", ["year"]),
        ("year_position", ["year", "pos_v5"]),
    ]
    for grouping, columns in groupings:
        for keys, group in paired.groupby(
            columns,
            observed=True,
            dropna=False,
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "grouping": grouping,
                "player_years": int(len(group)),
                **dict(zip(columns, keys)),
            }
            for scale in ["normalized", "raw"]:
                for label in ["v1", "v5"]:
                    residual = group[
                        f"residual_{scale}_{label}"
                    ].to_numpy(dtype=float)
                    row[f"{scale}_{label}_mean_residual"] = float(
                        residual.mean()
                    )
                    row[f"{scale}_{label}_mae"] = float(
                        np.abs(residual).mean()
                    )
                row[f"{scale}_mae_effect_v5_minus_v1"] = (
                    row[f"{scale}_v5_mae"] - row[f"{scale}_v1_mae"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def market_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method_version",
        "year",
        "available_slots",
        "available_budget",
        "pre_normalized_total",
        "post_normalized_total",
        "pred_salary_scale",
        "pred_salary_shift",
        "normalization_method",
    ]
    output = backtest[columns].drop_duplicates().copy()
    output["method"] = output.method_version.map(
        {value: key for key, value in METHODS.items()}
    )
    output["raw_minus_budget"] = (
        output.pre_normalized_total - output.available_budget
    )
    output["raw_budget_absolute_gap"] = output.raw_minus_budget.abs()
    output["raw_budget_gap_pct"] = (
        output.raw_minus_budget / output.available_budget
    )
    output["normalized_budget_error"] = (
        output.post_normalized_total - output.available_budget
    )
    return output.sort_values(["method", "year"])


def quantile_coverage(validation: pd.DataFrame) -> pd.DataFrame:
    v5 = validation[
        validation.method_version.eq(METHODS["v5"])
        & validation.resid_training_rows.gt(0)
    ].copy()
    rows = []
    for period, selected in periods(v5):
        if selected.empty:
            continue
        row: dict[str, Any] = {
            "period": period,
            "player_years": int(len(selected)),
        }
        for probability in [5, 10, 25, 75, 90, 95]:
            row[f"empirical_le_q{probability}"] = float(
                selected.actual_resid.le(
                    selected[f"salary_resid_{probability}"]
                ).mean()
            )
        row["central_80_coverage"] = float(
            selected.actual_resid.between(
                selected.salary_resid_10,
                selected.salary_resid_90,
                inclusive="both",
            ).mean()
        )
        row["central_90_coverage"] = float(
            selected.actual_resid.between(
                selected.salary_resid_5,
                selected.salary_resid_95,
                inclusive="both",
            ).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    accuracy: pd.DataFrame,
    market: pd.DataFrame,
) -> None:
    selected = accuracy[
        accuracy.period.isin(
            ["all_years", "replay_development_2022_2024", "temporal_check_2025"]
        )
    ].copy()
    v5_market = market[market.method.eq("v5")]
    lines = [
        "# Salary v5 Point-Accuracy Readout",
        "",
        "## Rolling player accuracy",
        "",
        markdown_table(
            selected[
                [
                    "method",
                    "prediction_scale",
                    "period",
                    "player_years",
                    "mean_residual",
                    "mae",
                    "rmse",
                    "r2",
                ]
            ],
        ),
        "",
        "## v5 raw market coherence",
        "",
        markdown_table(
            v5_market[
                [
                    "year",
                    "available_budget",
                    "pre_normalized_total",
                    "raw_minus_budget",
                    "pred_salary_shift",
                    "post_normalized_total",
                ]
            ],
        ),
        "",
        (
            "v5 raw predictions are not forced to match the budget. The additive "
            "shift is the final exact reconciliation and does not use realized "
            "target-auction spending."
        ),
        "",
    ]
    (RESULTS / "point_accuracy_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    validation, backtest = read_tables()
    accuracy = accuracy_summary(validation)
    paired = common_v1_v5(validation)
    grouped = grouped_paired_summary(paired)
    market = market_summary(backtest)
    coverage = quantile_coverage(validation)

    outputs = {
        "point_accuracy_by_period.csv": accuracy,
        "paired_v1_v5_player_rows.csv": paired,
        "paired_v1_v5_grouped_accuracy.csv": grouped,
        "market_total_calibration.csv": market,
        "v5_residual_quantile_coverage.csv": coverage,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS / filename, index=False)
    write_summary(accuracy, market)
    manifest = {
        "methods": METHODS,
        "model_spec_asof_year": MODEL_SPEC_YEAR,
        "paired_v1_v5_player_years": int(len(paired)),
        "sources": {
            "validations_database": str(VALIDATIONS_DB),
        },
        "outputs": {
            filename: int(len(frame))
            for filename, frame in outputs.items()
        },
    }
    (RESULTS / "point_accuracy_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        (RESULTS / "point_accuracy_summary.md").read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
