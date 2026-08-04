"""Frozen rolling-origin projection and residual validation for the Ridge swap."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import (
    LOCKED_RANDOM_SEED,
    PRIMARY_PPG_FEATURES,
    lock_version_for_scoring,
)


REFERENCE_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_locked_final_validation"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ridge_swap_locked_reference", REFERENCE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(REFERENCE_PATH)
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
BASELINE = "production"
CHALLENGER = "ridge_swap"
HISTORICAL_ORIGINS = tuple(range(2017, 2026))
ALL_ORIGINS = (*HISTORICAL_ORIGINS, 2026)
RIDGE_ALPHA = 10.0
MIN_INTERVAL_ROWS = 100
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
RESID_COLS = {
    0.05: "pred_resid_5",
    0.10: "pred_resid_10",
    0.25: "pred_resid_25",
    0.50: "pred_resid_50",
    0.75: "pred_resid_75",
    0.90: "pred_resid_90",
    0.95: "pred_resid_95",
}
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
DISTRIBUTION_PERIODS = {
    "all_2018_2025": (2018, 2025),
    "development_2018_2022": (2018, 2022),
    "temporal_2023_2025": (2023, 2025),
}
BOOTSTRAP_REPEATS = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=sorted(DATABASES), required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA, max_iter=10_000)),
        ]
    )


def ridge_predictions(
    ppg: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    feature_columns = list(PRIMARY_PPG_FEATURES)
    rows = []
    for origin in ALL_ORIGINS:
        train = ppg[ppg.season.lt(origin)]
        hold = candidates[
            candidates.season.eq(origin)
            & candidates.expert_ppg_team_game_median.notna()
        ].copy()
        if train.empty or hold.empty:
            raise ValueError(f"Missing Ridge train/hold rows for {origin}")
        model = ridge_pipeline()
        model.fit(
            train[feature_columns].apply(pd.to_numeric, errors="coerce"),
            train.actual.to_numpy(float),
        )
        hold["ridge_prediction"] = model.predict(
            hold[feature_columns].apply(pd.to_numeric, errors="coerce")
        )
        rows.append(
            hold[["player_key", "season", "position", "ridge_prediction"]]
        )
        print(
            f"Ridge origin {origin}: train={len(train):,}, hold={len(hold):,}",
            flush=True,
        )
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["player_key", "season"]).any():
        raise ValueError("Ridge predictions are not unique")
    return output


def load_live_predictions(
    database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    methods = (
        "conditional_ppg_lasso",
        "conditional_ppg_random_forest",
        "conditional_ppg_lightgbm",
        "conditional_ppg_primary_blend",
    )
    placeholders = ",".join("?" for _ in methods)
    with sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro", uri=True
    ) as connection:
        historical = pd.read_sql_query(
            f"""
            SELECT *
            FROM locked_whole_season_predictions
            WHERE target_name='conditional_ppg'
              AND method IN ({placeholders})
            """,
            connection,
            params=methods,
        )
        shadow = pd.read_sql_query(
            """
            SELECT player_key, display_name, season, position, team,
                   identity_status, expert_ppg_team_game_median,
                   conditional_ppg_lasso,
                   conditional_ppg_random_forest,
                   conditional_ppg_lightgbm,
                   conditional_ppg_primary_blend,
                   participation_probability, publication_status
            FROM locked_2026_shadow_predictions
            """,
            connection,
        )
        run = pd.read_sql_query(
            "SELECT * FROM locked_candidate_runs", connection
        )
    if len(run) != 1:
        raise ValueError("Expected exactly one active locked candidate run")
    return historical, shadow, run.iloc[0].to_dict()


def assemble_historical(
    historical: pd.DataFrame, ridge: pd.DataFrame
) -> pd.DataFrame:
    key = ["player_key", "season", "position"]
    wide = historical.pivot(
        index=key, columns="method", values="prediction"
    ).reset_index()
    primary = historical[
        historical.method.eq("conditional_ppg_primary_blend")
    ].drop(columns=["prediction", "method", "residual", "target_name"])
    if primary.duplicated(key).any():
        raise ValueError("Production historical metadata are not unique")
    wide = wide.merge(primary, on=key, how="inner", validate="one_to_one")
    expected = wide[
        [
            "conditional_ppg_lasso",
            "conditional_ppg_random_forest",
            "conditional_ppg_lightgbm",
        ]
    ].mean(axis=1)
    if not np.allclose(
        expected,
        wide.conditional_ppg_primary_blend,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Stored primary blend is not the equal-third blend")
    wide = wide.merge(ridge, on=key, how="inner", validate="one_to_one")
    if len(wide) != len(primary):
        raise ValueError(
            f"Ridge/baseline cohort mismatch: {len(wide)} vs {len(primary)}"
        )
    wide[BASELINE] = wide.conditional_ppg_primary_blend
    wide[CHALLENGER] = wide[
        [
            "ridge_prediction",
            "conditional_ppg_random_forest",
            "conditional_ppg_lightgbm",
        ]
    ].mean(axis=1)
    metadata = [
        "lock_version",
        "model_run_id",
        *key,
        "history_depth",
        "limited_history",
        "is_rookie",
        "year_exp",
        "has_prior_outcome",
        "projection_provider_count",
        "projection_trajectory_prior_year_available",
        "adp_median",
        "actual",
    ]
    rows = []
    for method in (BASELINE, CHALLENGER):
        current = wide[metadata].copy()
        current["method"] = method
        current["prediction"] = wide[method].to_numpy(float)
        current["residual"] = current.actual - current.prediction
        rows.append(current)
    evaluation = pd.concat(rows, ignore_index=True)
    evaluation.sort_values(
        ["method", "season", "player_key"], inplace=True
    )
    return evaluation.reset_index(drop=True)


def assemble_shadow(shadow: pd.DataFrame, ridge: pd.DataFrame) -> pd.DataFrame:
    current_ridge = ridge[ridge.season.eq(2026)]
    output = shadow.merge(
        current_ridge,
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    output[BASELINE] = output.conditional_ppg_primary_blend
    output[CHALLENGER] = output[
        [
            "ridge_prediction",
            "conditional_ppg_random_forest",
            "conditional_ppg_lightgbm",
        ]
    ].mean(axis=1, skipna=False)
    output["ridge_swap_minus_production"] = output[CHALLENGER] - output[BASELINE]
    return output


def strict_prior_residuals(evaluation: pd.DataFrame) -> pd.DataFrame:
    output_frames = []
    for method, method_frame in evaluation.groupby("method", sort=True):
        frame = method_frame.copy().sort_values(["season", "player_key"])
        for column in RESID_COLS.values():
            frame[column] = np.nan
        frame["resid_calibration_available"] = 0
        frame["resid_calibration_pool"] = "unavailable"
        frame["resid_calibration_rows"] = 0
        for season in HISTORICAL_ORIGINS:
            prior = frame[frame.season.lt(season)]
            for index, row in frame[frame.season.eq(season)].iterrows():
                pools = (
                    (
                        prior[
                            prior.position.eq(row.position)
                            & prior.history_depth.eq(row.history_depth)
                        ],
                        "position_history",
                    ),
                    (prior[prior.position.eq(row.position)], "position"),
                    (prior, "global"),
                )
                donors = pd.DataFrame()
                pool_name = "unavailable"
                for pool, name in pools:
                    if len(pool) >= MIN_INTERVAL_ROWS:
                        donors = pool
                        pool_name = name
                        break
                if donors.empty:
                    continue
                quantiles = donors.residual.quantile(list(QUANTILES))
                for quantile, column in RESID_COLS.items():
                    frame.loc[index, column] = float(quantiles.loc[quantile])
                frame.loc[index, "resid_calibration_available"] = 1
                frame.loc[index, "resid_calibration_pool"] = pool_name
                frame.loc[index, "resid_calibration_rows"] = len(donors)
        output_frames.append(frame)
    output = pd.concat(output_frames, ignore_index=True)
    output.sort_values(["method", "season", "player_key"], inplace=True)
    return output.reset_index(drop=True)


def point_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices: list[tuple[str, str, pd.Series]] = [
        ("all", "all", pd.Series(True, index=evaluation.index))
    ]
    for position in sorted(evaluation.position.unique()):
        slices.append(
            ("position", position, evaluation.position.eq(position))
        )
    for depth in sorted(evaluation.history_depth.unique()):
        slices.append(
            ("history_depth", depth, evaluation.history_depth.eq(depth))
        )
    for season in HISTORICAL_ORIGINS:
        slices.append(
            ("season", str(season), evaluation.season.eq(season))
        )
    for slice_type, slice_value, mask in slices:
        scoped = evaluation[mask]
        for period, (start, end) in PERIODS.items():
            period_frame = scoped[scoped.season.between(start, end)]
            if period_frame.empty:
                continue
            for method, group in period_frame.groupby("method", sort=True):
                error = group.prediction - group.actual
                correlation = spearmanr(
                    group.actual.to_numpy(float),
                    group.prediction.to_numpy(float),
                ).statistic
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        "period": period,
                        "method": method,
                        "rows": len(group),
                        "rmse": float(np.sqrt(np.mean(np.square(error)))),
                        "mae": float(np.mean(np.abs(error))),
                        "bias": float(np.mean(error)),
                        "spearman": float(correlation),
                    }
                )
    summary = pd.DataFrame(rows)
    metrics = ("rmse", "mae", "bias", "spearman")
    baseline = summary[summary.method.eq(BASELINE)][
        ["slice_type", "slice_value", "period", *metrics]
    ].rename(columns={metric: f"{metric}_baseline" for metric in metrics})
    summary = summary.merge(
        baseline,
        on=["slice_type", "slice_value", "period"],
        how="left",
        validate="many_to_one",
    )
    for metric in metrics:
        summary[f"{metric}_delta"] = (
            summary[metric] - summary[f"{metric}_baseline"]
        )
    return summary


def residual_knots(row: pd.Series) -> np.ndarray:
    q5, q10, q25, q75, q90, q95 = [
        float(row[RESID_COLS[quantile]])
        for quantile in (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
    ]
    return np.maximum.accumulate(
        np.array(
            [
                (2 * q5) - q10,
                q5,
                q10,
                q25,
                q75,
                q90,
                q95,
                (2 * q95) - q90,
            ],
            dtype=float,
        )
    )


def interpolate_residuals(knots: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    probabilities = np.array(
        [0.00, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 1.00]
    )
    return np.interp(uniforms, probabilities, knots)


def add_distribution_rows(calibrated: pd.DataFrame) -> pd.DataFrame:
    frame = calibrated[
        calibrated.resid_calibration_available.eq(1)
    ].copy()
    uniforms = (np.arange(4096, dtype=float) + 0.5) / 4096
    crps = []
    for _, row in frame.iterrows():
        samples = row.prediction + interpolate_residuals(
            residual_knots(row), uniforms
        )
        samples = np.maximum(samples, 0.0)
        n = len(samples)
        coefficients = 2 * np.arange(1, n + 1) - n - 1
        pair_term = float(np.sum(coefficients * samples) / (n * n))
        crps.append(float(np.mean(np.abs(samples - row.actual)) - pair_term))
    frame["crps"] = crps
    frame["covered_50"] = frame.residual.between(
        frame.pred_resid_25, frame.pred_resid_75
    )
    frame["covered_80"] = frame.residual.between(
        frame.pred_resid_10, frame.pred_resid_90
    )
    frame["interval_width_50"] = (
        frame.pred_resid_75 - frame.pred_resid_25
    )
    frame["interval_width_80"] = (
        frame.pred_resid_90 - frame.pred_resid_10
    )
    return frame


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices: list[tuple[str, str, pd.Series]] = [
        ("all", "all", pd.Series(True, index=frame.index))
    ]
    for position in sorted(frame.position.unique()):
        slices.append(("position", position, frame.position.eq(position)))
    for slice_type, slice_value, mask in slices:
        scoped = frame[mask]
        for period, (start, end) in DISTRIBUTION_PERIODS.items():
            period_frame = scoped[scoped.season.between(start, end)]
            for method, group in period_frame.groupby("method", sort=True):
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        "period": period,
                        "method": method,
                        "rows": len(group),
                        "crps": float(group.crps.mean()),
                        "coverage_50": float(group.covered_50.mean()),
                        "coverage_80": float(group.covered_80.mean()),
                        "interval_width_50": float(group.interval_width_50.mean()),
                        "interval_width_80": float(group.interval_width_80.mean()),
                    }
                )
    summary = pd.DataFrame(rows)
    metrics = (
        "crps",
        "coverage_50",
        "coverage_80",
        "interval_width_50",
        "interval_width_80",
    )
    baseline = summary[summary.method.eq(BASELINE)][
        ["slice_type", "slice_value", "period", *metrics]
    ].rename(columns={metric: f"{metric}_baseline" for metric in metrics})
    summary = summary.merge(
        baseline,
        on=["slice_type", "slice_value", "period"],
        how="left",
        validate="many_to_one",
    )
    for metric in metrics:
        summary[f"{metric}_delta"] = (
            summary[metric] - summary[f"{metric}_baseline"]
        )
    return summary


def cluster_bootstrap(evaluation: pd.DataFrame) -> pd.DataFrame:
    wide = evaluation.pivot(
        index=["player_key", "season", "position"],
        columns="method",
        values=["actual", "prediction"],
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in wide.columns
    ]
    wide["baseline_sq"] = np.square(
        wide[f"prediction_{BASELINE}"] - wide[f"actual_{BASELINE}"]
    )
    wide["challenger_sq"] = np.square(
        wide[f"prediction_{CHALLENGER}"] - wide[f"actual_{CHALLENGER}"]
    )
    rows = []
    rng = np.random.default_rng(LOCKED_RANDOM_SEED + 829)
    for period, (start, end) in PERIODS.items():
        scoped = wide[wide.season.between(start, end)]
        groups = [group for _, group in scoped.groupby("player_key", sort=True)]
        if not groups:
            continue
        draws = np.empty(BOOTSTRAP_REPEATS, dtype=float)
        for repeat in range(BOOTSTRAP_REPEATS):
            sampled = rng.integers(0, len(groups), len(groups))
            baseline_error = np.concatenate(
                [groups[index].baseline_sq.to_numpy(float) for index in sampled]
            )
            challenger_error = np.concatenate(
                [groups[index].challenger_sq.to_numpy(float) for index in sampled]
            )
            draws[repeat] = np.sqrt(challenger_error.mean()) - np.sqrt(
                baseline_error.mean()
            )
        rows.append(
            {
                "period": period,
                "player_clusters": len(groups),
                "candidate_minus_baseline": float(draws.mean()),
                "bootstrap_p025": float(np.quantile(draws, 0.025)),
                "bootstrap_p975": float(np.quantile(draws, 0.975)),
                "probability_candidate_better": float(np.mean(draws < 0)),
            }
        )
    return pd.DataFrame(rows)


def shadow_summary(shadow: pd.DataFrame) -> pd.DataFrame:
    available = shadow.dropna(subset=[BASELINE, CHALLENGER]).copy()
    rows = []
    for slice_type, slice_value, group in [
        ("all", "all", available),
        *[
            ("position", position, current)
            for position, current in available.groupby("position", sort=True)
        ],
    ]:
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "rows": len(group),
                "mean_production": float(group[BASELINE].mean()),
                "mean_ridge_swap": float(group[CHALLENGER].mean()),
                "mean_delta": float(group.ridge_swap_minus_production.mean()),
                "median_abs_delta": float(
                    group.ridge_swap_minus_production.abs().median()
                ),
                "max_abs_delta": float(
                    group.ridge_swap_minus_production.abs().max()
                ),
                "spearman": float(
                    spearmanr(group[BASELINE], group[CHALLENGER]).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    league = args.league
    database = DATABASES[league]
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else STUDY_DIR / f"results_projection_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    reference.ACTIVE_OUTPUT_DB_PATH = database
    reference.ACTIVE_SCORING_OBJECTIVE = league
    reference.ACTIVE_LOCK_VERSION = lock_version_for_scoring(league)
    features, _, feature_run_id = reference._load_inputs()
    ppg, _, candidates = reference._target_frames(features)
    ridge = ridge_predictions(ppg, candidates)
    historical, current, run = load_live_predictions(database)
    evaluation = assemble_historical(historical, ridge)
    shadow = assemble_shadow(current, ridge)
    calibrated = strict_prior_residuals(evaluation)
    distribution_rows = add_distribution_rows(calibrated)

    ppg_scores = point_summary(evaluation)
    distribution_scores = distribution_summary(distribution_rows)
    bootstrap = cluster_bootstrap(evaluation)
    current_summary = shadow_summary(shadow)

    evaluation.to_csv(results_dir / "paired_point_predictions.csv", index=False)
    ppg_scores.to_csv(results_dir / "point_summary.csv", index=False)
    bootstrap.to_csv(results_dir / "point_player_cluster_bootstrap.csv", index=False)
    calibrated.to_csv(results_dir / "strict_prior_residuals.csv", index=False)
    distribution_rows.to_csv(results_dir / "distribution_rows.csv", index=False)
    distribution_scores.to_csv(results_dir / "distribution_summary.csv", index=False)
    shadow.to_csv(results_dir / "shadow_2026_predictions.csv", index=False)
    current_summary.to_csv(results_dir / "shadow_2026_summary.csv", index=False)
    metadata = {
        "league": league,
        "database": str(database),
        "source_model_run_id": run["model_run_id"],
        "source_feature_run_id": run["feature_run_id"],
        "loaded_feature_run_id": feature_run_id,
        "source_lock_version": run["lock_version"],
        "ridge_alpha": RIDGE_ALPHA,
        "feature_count": len(PRIMARY_PPG_FEATURES),
        "historical_origins": list(HISTORICAL_ORIGINS),
        "point_rows_per_method": int(len(evaluation) / 2),
        "distribution_rows_per_method": int(len(distribution_rows) / 2),
        "shadow_rows": int(len(shadow)),
        "shadow_complete_rows": int(
            shadow[[BASELINE, CHALLENGER]].notna().all(axis=1).sum()
        ),
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        ppg_scores[
            ppg_scores.slice_type.eq("all")
            & ppg_scores.method.eq(CHALLENGER)
        ].to_string(index=False),
        flush=True,
    )
    print(
        distribution_scores[
            distribution_scores.slice_type.eq("all")
            & distribution_scores.method.eq(CHALLENGER)
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
