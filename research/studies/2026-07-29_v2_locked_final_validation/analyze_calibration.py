"""Test compact strictly-prior point-calibration policies on the locked replay."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
RESULTS_DIR = STUDY_DIR / "results"
OUTPUT_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3"
PRIMARY_METHOD = "conditional_ppg_primary_blend"
MIN_POSITION_ROWS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-db",
        type=Path,
        default=OUTPUT_DB_PATH,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    return parser.parse_args()


def _rmse(frame: pd.DataFrame, prediction: str) -> float:
    return float(
        np.sqrt(mean_squared_error(frame["actual"], frame[prediction]))
    )


def build_calibration_diagnostics(
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = evaluation[
        evaluation["target_name"].eq("conditional_ppg")
        & evaluation["method"].eq(PRIMARY_METHOD)
    ].copy()
    strategies = (
        "uncalibrated",
        "expanding_global_intercept",
        "last3_global_intercept",
        "expanding_global_affine",
        "expanding_position_intercept",
        "expanding_position_affine",
    )
    for strategy in strategies:
        frame[strategy] = frame["prediction"]

    parameter_rows: list[dict[str, object]] = []
    for season in sorted(frame["season"].unique()):
        current = frame["season"].eq(season)
        prior = frame[frame["season"].lt(season)]
        if prior.empty:
            parameter_rows.append(
                {
                    "forecast_origin": season,
                    "calibration_scope": "global",
                    "prior_rows": 0,
                    "prior_start_season": pd.NA,
                    "prior_end_season": pd.NA,
                    "intercept_correction": 0.0,
                    "affine_intercept": 0.0,
                    "affine_slope": 1.0,
                }
            )
            continue

        global_bias = float((prior["prediction"] - prior["actual"]).mean())
        last3 = prior[prior["season"].ge(season - 3)]
        last3_bias = float(
            (last3["prediction"] - last3["actual"]).mean()
        )
        global_affine = LinearRegression().fit(
            prior[["prediction"]], prior["actual"]
        )
        frame.loc[current, "expanding_global_intercept"] = (
            frame.loc[current, "prediction"] - global_bias
        )
        frame.loc[current, "last3_global_intercept"] = (
            frame.loc[current, "prediction"] - last3_bias
        )
        frame.loc[current, "expanding_global_affine"] = (
            global_affine.predict(frame.loc[current, ["prediction"]])
        )
        parameter_rows.append(
            {
                "forecast_origin": season,
                "calibration_scope": "global",
                "prior_rows": len(prior),
                "prior_start_season": int(prior["season"].min()),
                "prior_end_season": int(prior["season"].max()),
                "intercept_correction": global_bias,
                "affine_intercept": float(global_affine.intercept_),
                "affine_slope": float(global_affine.coef_[0]),
            }
        )

        for position in sorted(frame.loc[current, "position"].unique()):
            position_prior = prior[prior["position"].eq(position)]
            position_current = current & frame["position"].eq(position)
            if len(position_prior) < MIN_POSITION_ROWS:
                frame.loc[
                    position_current, "expanding_position_intercept"
                ] = (
                    frame.loc[position_current, "prediction"] - global_bias
                )
                frame.loc[
                    position_current, "expanding_position_affine"
                ] = global_affine.predict(
                    frame.loc[position_current, ["prediction"]]
                )
                intercept_correction = global_bias
                affine_intercept = float(global_affine.intercept_)
                affine_slope = float(global_affine.coef_[0])
            else:
                position_bias = float(
                    (
                        position_prior["prediction"]
                        - position_prior["actual"]
                    ).mean()
                )
                position_affine = LinearRegression().fit(
                    position_prior[["prediction"]],
                    position_prior["actual"],
                )
                frame.loc[
                    position_current, "expanding_position_intercept"
                ] = (
                    frame.loc[position_current, "prediction"] - position_bias
                )
                frame.loc[
                    position_current, "expanding_position_affine"
                ] = position_affine.predict(
                    frame.loc[position_current, ["prediction"]]
                )
                intercept_correction = position_bias
                affine_intercept = float(position_affine.intercept_)
                affine_slope = float(position_affine.coef_[0])
            parameter_rows.append(
                {
                    "forecast_origin": season,
                    "calibration_scope": position,
                    "prior_rows": len(position_prior),
                    "prior_start_season": int(prior["season"].min()),
                    "prior_end_season": int(prior["season"].max()),
                    "intercept_correction": intercept_correction,
                    "affine_intercept": affine_intercept,
                    "affine_slope": affine_slope,
                }
            )

    base_rmse = _rmse(frame, "uncalibrated")
    base_recent = _rmse(
        frame[frame["season"].ge(2023)], "uncalibrated"
    )
    rows = []
    for strategy in strategies:
        season_deltas = []
        for _, season_frame in frame.groupby("season"):
            season_deltas.append(
                _rmse(season_frame, strategy)
                - _rmse(season_frame, "uncalibrated")
            )
        recent = frame[frame["season"].ge(2023)]
        rows.append(
            {
                "calibration_strategy": strategy,
                "n_rows": len(frame),
                "n_seasons": frame["season"].nunique(),
                "pooled_rmse": _rmse(frame, strategy),
                "pooled_rmse_delta": _rmse(frame, strategy) - base_rmse,
                "pooled_bias": float(
                    (frame[strategy] - frame["actual"]).mean()
                ),
                "recent_rmse": _rmse(recent, strategy),
                "recent_rmse_delta": _rmse(recent, strategy) - base_recent,
                "recent_bias": float(
                    (recent[strategy] - recent["actual"]).mean()
                ),
                "season_wins": int(np.sum(np.asarray(season_deltas) < 0)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows)


def main() -> None:
    args = parse_args()
    evaluation = pd.read_csv(
        args.results_dir / "locked_whole_season_predictions.csv"
    )
    summary, parameters = build_calibration_diagnostics(evaluation)
    run_id = str(evaluation["model_run_id"].dropna().iloc[0])
    lock_version = str(evaluation["lock_version"].dropna().iloc[0])
    for frame in (summary, parameters):
        frame.insert(0, "model_run_id", run_id)
        frame.insert(0, "lock_version", lock_version)
    summary.to_csv(
        args.results_dir / "locked_point_calibration_comparisons.csv",
        index=False,
    )
    parameters.to_csv(
        args.results_dir / "locked_point_calibration_parameters.csv",
        index=False,
    )
    with sqlite3.connect(args.output_db) as connection:
        summary.to_sql(
            "locked_point_calibration_comparisons",
            connection,
            if_exists="replace",
            index=False,
        )
        parameters.to_sql(
            "locked_point_calibration_parameters",
            connection,
            if_exists="replace",
            index=False,
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
