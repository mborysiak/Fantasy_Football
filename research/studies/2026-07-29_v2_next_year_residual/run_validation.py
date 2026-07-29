"""Validate V2 following-season residual models and publish 2027 shadows."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import (  # noqa: E402
    COMPLETED_THROUGH_SEASON,
    OUTPUT_DB_PATH as DEFAULT_OUTPUT_DB_PATH,
    POSITIONS,
)
from Scripts.V2.contracts import (  # noqa: E402
    create_run_id,
    publish_tables_atomic,
    scoring_hash,
    utc_now,
)
from Scripts.V2.locked_candidates import LOCKED_RANDOM_SEED  # noqa: E402
from Scripts.V2.modeling import add_modeling_features  # noqa: E402
from Scripts.V2.next_year import (  # noqa: E402
    NEXT_YEAR_PARTICIPATION_FEATURES,
    NEXT_YEAR_RESIDUAL_FEATURES,
    NEXT_YEAR_TARGET_VERSION,
    build_next_year_target_audit,
    build_next_year_targets,
    feature_hash,
)


DEFAULT_RESULTS_DIR = STUDY_DIR / "results"
VALIDATION_ORIGINS = tuple(range(2017, 2025))
SHADOW_ORIGINS = (2025, 2026)
PREDICTION_ORIGINS = tuple(range(2011, 2027))
GRID_ORIGINS = tuple(range(2012, 2025))
TARGET_HORIZON = 1
MIN_SELECTION_SEASONS = 3
RESIDUAL_QUANTILES = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)

RESIDUAL_COMPONENTS = (
    "next_residual_lasso",
    "next_residual_random_forest",
    "next_residual_lightgbm",
)
PRIMARY_RESIDUAL_METHOD = "next_residual_primary_blend"
PRIMARY_PPG_METHOD = "next_ppg_primary_blend"
PRIMARY_PARTICIPATION_METHOD = "next_participation_lightgbm"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBM.*",
    category=UserWarning,
)


LASSO_GRID = (
    {"alpha": 0.01},
    {"alpha": 0.03},
    {"alpha": 0.10},
)
RANDOM_FOREST_GRID = (
    {
        "n_estimators": 250,
        "max_depth": 6,
        "min_samples_leaf": 15,
        "max_features": 0.5,
    },
    {
        "n_estimators": 250,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": 0.75,
    },
    {
        "n_estimators": 250,
        "max_depth": 10,
        "min_samples_leaf": 15,
        "max_features": 0.75,
    },
    {
        "n_estimators": 250,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": 1.0,
    },
)
LIGHTGBM_GRID = (
    {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "reg_lambda": 5.0,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 4,
        "min_child_samples": 40,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "max_depth": 4,
        "min_child_samples": 20,
        "reg_lambda": 5.0,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 3,
        "min_child_samples": 40,
        "reg_lambda": 5.0,
    },
)
LOGISTIC_GRID = ({"C": 0.1}, {"C": 1.0}, {"C": 10.0})

MODEL_GRIDS: dict[str, Sequence[Mapping[str, object]]] = {
    "next_residual_lasso": LASSO_GRID,
    "next_residual_random_forest": RANDOM_FOREST_GRID,
    "next_residual_lightgbm": LIGHTGBM_GRID,
    "next_participation_logistic": LOGISTIC_GRID,
    "next_participation_lightgbm": LIGHTGBM_GRID,
}
DEFAULT_CANDIDATE_IDS = {
    "next_residual_lasso": 1,
    "next_residual_random_forest": 0,
    "next_residual_lightgbm": 0,
    "next_participation_logistic": 1,
    "next_participation_lightgbm": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-db", type=Path, default=DEFAULT_OUTPUT_DB_PATH
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument("--league", default="dk")
    return parser.parse_args()


def load_inputs(
    output_db: Path,
    league: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with sqlite3.connect(output_db) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features", connection
        )
        outcomes = pd.read_sql_query(
            "SELECT * FROM player_season_outcomes", connection
        )
    feature_run_ids = features["run_id"].dropna().astype(str).unique()
    if len(feature_run_ids) != 1:
        raise ValueError("Expected exactly one active feature run")
    observed_leagues = set(features["league"].dropna().astype(str))
    if observed_leagues != {league}:
        raise ValueError(
            f"Feature league mismatch: {sorted(observed_leagues)} vs {league}"
        )
    expected_scoring_hash = scoring_hash(league)
    observed_hashes = set(features["scoring_hash"].dropna().astype(str))
    if observed_hashes != {expected_scoring_hash}:
        raise ValueError("Feature scoring hash does not match requested league")
    return (
        add_modeling_features(features),
        outcomes,
        str(feature_run_ids[0]),
    )


def model_pipeline(
    model_name: str,
    parameters: Mapping[str, object],
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ]
    if model_name in {
        "next_residual_lasso",
        "next_participation_logistic",
    }:
        steps.append(("scale", StandardScaler()))
    if model_name == "next_residual_lasso":
        estimator = Lasso(
            max_iter=20_000,
            tol=1e-6,
            **parameters,
        )
    elif model_name == "next_residual_random_forest":
        estimator = RandomForestRegressor(
            bootstrap=True,
            random_state=LOCKED_RANDOM_SEED,
            n_jobs=1,
            **parameters,
        )
    elif model_name == "next_residual_lightgbm":
        estimator = LGBMRegressor(
            objective="regression",
            verbosity=-1,
            subsample=1.0,
            colsample_bytree=1.0,
            deterministic=True,
            force_col_wise=True,
            random_state=LOCKED_RANDOM_SEED,
            n_jobs=1,
            **parameters,
        )
    elif model_name == "next_participation_logistic":
        estimator = LogisticRegression(
            max_iter=3_000,
            solver="lbfgs",
            random_state=LOCKED_RANDOM_SEED,
            **parameters,
        )
    elif model_name == "next_participation_lightgbm":
        estimator = LGBMClassifier(
            objective="binary",
            verbosity=-1,
            subsample=1.0,
            colsample_bytree=1.0,
            deterministic=True,
            force_col_wise=True,
            random_state=LOCKED_RANDOM_SEED,
            n_jobs=1,
            **parameters,
        )
    else:
        raise ValueError(f"Unsupported next-year model: {model_name}")
    steps.append(("model", estimator))
    return Pipeline(steps)


def model_inputs(
    model_name: str,
) -> tuple[tuple[str, ...], str, bool]:
    if model_name.startswith("next_residual"):
        return (
            NEXT_YEAR_RESIDUAL_FEATURES,
            "next_residual_vs_expert",
            False,
        )
    return NEXT_YEAR_PARTICIPATION_FEATURES, "next_appeared", True


def fit_model(
    model_name: str,
    parameters: Mapping[str, object],
    train: pd.DataFrame,
) -> Pipeline:
    features, target, _ = model_inputs(model_name)
    X = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train[target], errors="raise")
    model = model_pipeline(model_name, parameters)
    model.fit(X, y)
    return model


def predict_model(
    model_name: str,
    model: Pipeline,
    hold: pd.DataFrame,
) -> np.ndarray:
    features, _, probability = model_inputs(model_name)
    X = hold.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if probability:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def training_rows(
    targets: pd.DataFrame,
    model_name: str,
    forecast_origin: int,
) -> pd.DataFrame:
    latest_origin = int(forecast_origin) - TARGET_HORIZON - 1
    if model_name.startswith("next_residual"):
        eligible = targets["next_conditional_ppg_training_eligible"].eq(1)
    else:
        eligible = targets["next_participation_target_available"].eq(1)
    return targets[
        eligible & targets["origin_season"].le(latest_origin)
    ].copy()


def hold_rows(
    targets: pd.DataFrame,
    model_name: str,
    forecast_origin: int,
    require_actual: bool,
) -> pd.DataFrame:
    hold = targets[targets["origin_season"].eq(forecast_origin)].copy()
    if model_name.startswith("next_residual"):
        hold = hold[hold["origin_expert_ppg"].notna()].copy()
        if require_actual:
            hold = hold[
                hold["next_conditional_ppg_training_eligible"].eq(1)
            ].copy()
    elif require_actual:
        hold = hold[
            hold["next_participation_target_available"].eq(1)
        ].copy()
    return hold


def grid_predictions(
    targets: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for origin in GRID_ORIGINS:
        train = training_rows(targets, model_name, origin)
        hold = hold_rows(
            targets, model_name, origin, require_actual=True
        )
        if train.empty or hold.empty:
            continue
        for candidate_id, parameters in enumerate(MODEL_GRIDS[model_name]):
            model = fit_model(model_name, parameters, train)
            prediction = predict_model(model_name, model, hold)
            current = hold[
                ["player_key", "origin_season", "position"]
            ].copy()
            current["actual"] = hold[
                model_inputs(model_name)[1]
            ].to_numpy()
            current["model_name"] = model_name
            current["candidate_id"] = candidate_id
            current["prediction"] = prediction
            current["training_through_origin"] = origin - 2
            current["target_outcome_through"] = origin - 1
            rows.append(current)
    if not rows:
        raise ValueError(f"No grid predictions for {model_name}")
    return pd.concat(rows, ignore_index=True)


def metric(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    probability: bool,
) -> float:
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if probability:
        return float(
            brier_score_loss(
                actual_values,
                np.clip(prediction_values, 1e-6, 1 - 1e-6),
            )
        )
    return float(
        np.sqrt(mean_squared_error(actual_values, prediction_values))
    )


def select_hyperparameters(
    grid: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    probability = model_inputs(model_name)[2]
    rows: list[dict[str, object]] = []
    for origin in PREDICTION_ORIGINS:
        prior = grid[grid["origin_season"].le(origin - 2)]
        prior_seasons = sorted(prior["origin_season"].unique())
        default_id = DEFAULT_CANDIDATE_IDS[model_name]
        if len(prior_seasons) < MIN_SELECTION_SEASONS:
            selected_id = default_id
            selected_score = np.nan
            source = "fixed_default_insufficient_prior"
        else:
            scores: list[tuple[float, int]] = []
            for candidate_id, candidate in prior.groupby("candidate_id"):
                season_scores = [
                    metric(
                        season["actual"],
                        season["prediction"],
                        probability,
                    )
                    for _, season in candidate.groupby("origin_season")
                ]
                scores.append(
                    (float(np.mean(season_scores)), int(candidate_id))
                )
            selected_score, selected_id = min(scores)
            source = "strict_prior_season_mean"
        rows.append(
            {
                "model_name": model_name,
                "forecast_origin": origin,
                "candidate_id": int(selected_id),
                "parameters_json": json.dumps(
                    MODEL_GRIDS[model_name][selected_id], sort_keys=True
                ),
                "selection_metric": "brier" if probability else "rmse",
                "selection_score": selected_score,
                "selection_start_origin": (
                    min(prior_seasons) if prior_seasons else pd.NA
                ),
                "selection_end_origin": (
                    max(prior_seasons) if prior_seasons else pd.NA
                ),
                "selection_seasons": len(prior_seasons),
                "selection_source": source,
                "latest_usable_inner_origin": origin - 2,
            }
        )
    return pd.DataFrame(rows)


def selected_predictions(
    targets: pd.DataFrame,
    model_name: str,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    probability = model_inputs(model_name)[2]
    for origin in PREDICTION_ORIGINS:
        train = training_rows(targets, model_name, origin)
        hold = hold_rows(
            targets, model_name, origin, require_actual=False
        )
        selection = selections[
            selections["forecast_origin"].eq(origin)
            & selections["model_name"].eq(model_name)
        ]
        if len(selection) != 1:
            raise ValueError(f"Missing selected parameters for {model_name}")
        if train.empty or hold.empty:
            continue
        parameters = json.loads(selection.iloc[0]["parameters_json"])
        model = fit_model(model_name, parameters, train)
        prediction = predict_model(model_name, model, hold)
        current = hold[
            ["player_key", "origin_season", "position"]
        ].copy()
        current["model_name"] = model_name
        current["prediction"] = (
            np.clip(prediction, 1e-6, 1 - 1e-6)
            if probability
            else prediction
        )
        current["training_through_origin"] = origin - 2
        current["target_outcome_through"] = origin - 1
        current["selected_candidate_id"] = int(
            selection.iloc[0]["candidate_id"]
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def experience_group(frame: pd.DataFrame) -> pd.Series:
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    group = pd.Series("unknown", index=frame.index, dtype=object)
    group.loc[rookie | year_exp.eq(0)] = "rookie"
    group.loc[~rookie & year_exp.eq(1)] = "second_year"
    group.loc[year_exp.between(2, 3)] = "year_2_3"
    group.loc[year_exp.between(4, 6)] = "year_4_6"
    group.loc[year_exp.between(7, 9)] = "year_7_9"
    group.loc[year_exp.ge(10)] = "year_10_plus"
    return group


def smoothed_group_predictions(
    targets: pd.DataFrame,
    probability: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    target_column = "next_appeared" if probability else "next_residual_vs_expert"
    model_name = (
        "next_participation_position_experience_prior"
        if probability
        else "next_residual_position_experience_aging"
    )
    for origin in PREDICTION_ORIGINS:
        source_name = (
            "next_participation_lightgbm"
            if probability
            else "next_residual_lightgbm"
        )
        train = training_rows(targets, source_name, origin)
        hold = hold_rows(
            targets, source_name, origin, require_actual=False
        )
        if train.empty or hold.empty:
            continue
        train = train.copy()
        hold = hold.copy()
        train["experience_group"] = experience_group(train)
        hold["experience_group"] = experience_group(hold)
        global_value = float(train[target_column].mean())
        position = train.groupby("position")[target_column].agg(
            ["mean", "count"]
        )
        position["smoothed"] = (
            position["count"] * position["mean"] + 75.0 * global_value
        ) / (position["count"] + 75.0)
        group = train.groupby(
            ["position", "experience_group"]
        )[target_column].agg(["mean", "count"])
        position_prior = group.index.get_level_values("position").map(
            position["smoothed"]
        )
        group["smoothed"] = (
            group["count"] * group["mean"] + 30.0 * position_prior
        ) / (group["count"] + 30.0)
        keys = pd.MultiIndex.from_frame(
            hold[["position", "experience_group"]]
        )
        prediction = pd.Series(
            group["smoothed"].reindex(keys).to_numpy(),
            index=hold.index,
        )
        position_fallback = hold["position"].map(position["smoothed"])
        prediction = prediction.fillna(position_fallback).fillna(global_value)
        current = hold[
            ["player_key", "origin_season", "position"]
        ].copy()
        current["model_name"] = model_name
        current["prediction"] = (
            prediction.clip(1e-6, 1 - 1e-6)
            if probability
            else prediction
        )
        current["training_through_origin"] = origin - 2
        current["target_outcome_through"] = origin - 1
        current["selected_candidate_id"] = pd.NA
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def assemble_wide(
    targets: pd.DataFrame,
    prediction_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    metadata = [
        "player_key",
        "gsis_id",
        "display_name",
        "origin_season",
        "target_season",
        "position",
        "team",
        "identity_status",
        "origin_expert_ppg",
        "next_appeared",
        "next_conditional_ppg",
        "next_residual_vs_expert",
        "next_participation_target_available",
        "next_conditional_ppg_training_eligible",
        "next_target_join_status",
        "is_rookie",
        "year_exp",
        "has_prior_outcome",
        "projection_provider_count",
        "adp_median",
    ]
    frame = targets.loc[
        targets["origin_season"].isin(PREDICTION_ORIGINS), metadata
    ].copy()
    long = pd.concat(prediction_frames, ignore_index=True)
    if long.duplicated(
        ["player_key", "origin_season", "model_name"]
    ).any():
        raise ValueError("Next-year model predictions contain duplicate keys")
    wide = long.pivot(
        index=["player_key", "origin_season"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    frame = frame.merge(
        wide,
        on=["player_key", "origin_season"],
        how="left",
        validate="one_to_one",
    )
    frame[PRIMARY_RESIDUAL_METHOD] = frame[
        list(RESIDUAL_COMPONENTS)
    ].mean(axis=1, skipna=False)
    frame["next_ppg_expert_carry_forward"] = frame["origin_expert_ppg"]
    frame["next_ppg_position_experience_aging"] = (
        frame["origin_expert_ppg"]
        + frame["next_residual_position_experience_aging"]
    ).clip(lower=0)
    for component in RESIDUAL_COMPONENTS:
        frame[component.replace("next_residual", "next_ppg")] = (
            frame["origin_expert_ppg"] + frame[component]
        ).clip(lower=0)
    frame[PRIMARY_PPG_METHOD] = (
        frame["origin_expert_ppg"] + frame[PRIMARY_RESIDUAL_METHOD]
    ).clip(lower=0)
    frame["predicted_next_year_residual"] = frame[
        PRIMARY_RESIDUAL_METHOD
    ]
    frame["predicted_next_year_conditional_ppg"] = frame[
        PRIMARY_PPG_METHOD
    ]
    frame["predicted_next_year_appearance_probability"] = frame[
        PRIMARY_PARTICIPATION_METHOD
    ]
    return frame


def history_depth(frame: pd.DataFrame) -> pd.Series:
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    prior = pd.to_numeric(
        frame["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    output = pd.Series("other_no_history", index=frame.index, dtype=object)
    output.loc[rookie] = "rookie"
    output.loc[~rookie & year_exp.eq(1)] = "second_year"
    output.loc[~rookie & year_exp.ge(2) & prior] = "veteran_with_history"
    output.loc[year_exp.isna()] = "unknown_experience"
    return output


def evaluation_long(frame: pd.DataFrame) -> pd.DataFrame:
    valid_origins = frame["origin_season"].isin(VALIDATION_ORIGINS)
    ppg_eligible = (
        valid_origins
        & frame["next_conditional_ppg_training_eligible"].eq(1)
    )
    participation_eligible = (
        valid_origins
        & frame["next_participation_target_available"].eq(1)
    )
    metadata = [
        "player_key",
        "origin_season",
        "target_season",
        "position",
        "is_rookie",
        "year_exp",
        "has_prior_outcome",
        "projection_provider_count",
        "adp_median",
    ]
    rows: list[pd.DataFrame] = []
    ppg_methods = (
        "next_ppg_expert_carry_forward",
        "next_ppg_position_experience_aging",
        "next_ppg_lasso",
        "next_ppg_random_forest",
        "next_ppg_lightgbm",
        PRIMARY_PPG_METHOD,
    )
    for method in ppg_methods:
        current = frame.loc[
            ppg_eligible, metadata + ["next_conditional_ppg"]
        ].copy()
        current.rename(
            columns={"next_conditional_ppg": "actual"}, inplace=True
        )
        current["method"] = method
        current["prediction"] = frame.loc[
            ppg_eligible, method
        ].to_numpy()
        current["target_name"] = "next_conditional_ppg"
        rows.append(current)
    participation_methods = (
        "next_participation_position_experience_prior",
        "next_participation_logistic",
        "next_participation_lightgbm",
    )
    for method in participation_methods:
        current = frame.loc[
            participation_eligible, metadata + ["next_appeared"]
        ].copy()
        current.rename(columns={"next_appeared": "actual"}, inplace=True)
        current["method"] = method
        current["prediction"] = frame.loc[
            participation_eligible, method
        ].to_numpy()
        current["target_name"] = "next_participation"
        rows.append(current)
    output = pd.concat(rows, ignore_index=True)
    output["history_depth"] = history_depth(output)
    output["experience_group"] = experience_group(output)
    output["residual"] = output["actual"] - output["prediction"]
    return output


def score_values(
    group: pd.DataFrame,
    target_name: str,
) -> dict[str, float]:
    actual = group["actual"].to_numpy(dtype=float)
    prediction = group["prediction"].to_numpy(dtype=float)
    if target_name == "next_conditional_ppg":
        return {
            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
            "mae": float(mean_absolute_error(actual, prediction)),
            "bias": float(np.mean(prediction - actual)),
            "spearman": float(
                spearmanr(actual, prediction).statistic
            ),
        }
    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(actual, clipped)),
        "log_loss": float(log_loss(actual, clipped, labels=[0, 1])),
        "calibration_bias": float(np.mean(clipped - actual)),
        "roc_auc": (
            float(roc_auc_score(actual, clipped))
            if len(np.unique(actual)) > 1
            else np.nan
        ),
    }


def score_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target_name, method), model in evaluation.groupby(
        ["target_name", "method"], sort=True
    ):
        slices: list[tuple[str, str, pd.DataFrame]] = [
            ("pooled", "all", model),
            (
                "recent",
                "2022_2024",
                model[model["origin_season"].ge(2022)],
            ),
        ]
        for column in (
            "origin_season",
            "position",
            "history_depth",
            "experience_group",
        ):
            slices.extend(
                (column, str(value), group)
                for value, group in model.groupby(column)
            )
        for slice_type, slice_value, group in slices:
            if group.empty:
                continue
            for metric_name, value in score_values(
                group, target_name
            ).items():
                rows.append(
                    {
                        "target_name": target_name,
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        "metric": metric_name,
                        "n_rows": len(group),
                        "n_origins": group["origin_season"].nunique(),
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    signs = np.array(
        [
            [
                1 if (mask >> bit) & 1 else -1
                for bit in range(len(values))
            ]
            for mask in range(2 ** len(values))
        ],
        dtype=float,
    )
    observed = abs(float(values.mean()))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def comparison_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(LOCKED_RANDOM_SEED)
    comparisons = (
        (
            PRIMARY_PPG_METHOD,
            "next_ppg_expert_carry_forward",
            "primary_vs_expert_carry",
            "next_conditional_ppg",
            False,
        ),
        (
            "next_ppg_position_experience_aging",
            "next_ppg_expert_carry_forward",
            "aging_vs_expert_carry",
            "next_conditional_ppg",
            False,
        ),
        (
            "next_ppg_lasso",
            "next_ppg_expert_carry_forward",
            "lasso_vs_expert_carry",
            "next_conditional_ppg",
            False,
        ),
        (
            "next_ppg_random_forest",
            "next_ppg_expert_carry_forward",
            "rf_vs_expert_carry",
            "next_conditional_ppg",
            False,
        ),
        (
            "next_ppg_lightgbm",
            "next_ppg_expert_carry_forward",
            "lightgbm_vs_expert_carry",
            "next_conditional_ppg",
            False,
        ),
        (
            PRIMARY_PARTICIPATION_METHOD,
            "next_participation_position_experience_prior",
            "participation_lgbm_vs_prior",
            "next_participation",
            True,
        ),
        (
            PRIMARY_PARTICIPATION_METHOD,
            "next_participation_logistic",
            "participation_lgbm_vs_logistic",
            "next_participation",
            True,
        ),
    )
    for challenger, reference, name, target_name, probability in comparisons:
        target = evaluation[evaluation["target_name"].eq(target_name)]
        season_rows = []
        for origin, group in target.groupby("origin_season"):
            challenger_group = group[group["method"].eq(challenger)]
            reference_group = group[group["method"].eq(reference)]
            season_rows.append(
                {
                    "origin_season": int(origin),
                    "challenger": metric(
                        challenger_group["actual"],
                        challenger_group["prediction"],
                        probability,
                    ),
                    "reference": metric(
                        reference_group["actual"],
                        reference_group["prediction"],
                        probability,
                    ),
                }
            )
        season = pd.DataFrame(season_rows)
        delta = (season["challenger"] - season["reference"]).to_numpy()
        indices = rng.integers(
            0, len(delta), size=(20_000, len(delta))
        )
        bootstrap = delta[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        challenger_rows = target[target["method"].eq(challenger)]
        reference_rows = target[target["method"].eq(reference)]
        rows.append(
            {
                "comparison": name,
                "challenger": challenger,
                "reference": reference,
                "metric": "brier" if probability else "rmse",
                "origins": len(delta),
                "mean_origin_delta": float(delta.mean()),
                "pooled_challenger": metric(
                    challenger_rows["actual"],
                    challenger_rows["prediction"],
                    probability,
                ),
                "pooled_reference": metric(
                    reference_rows["actual"],
                    reference_rows["prediction"],
                    probability,
                ),
                "origin_wins": int(np.sum(delta < 0)),
                "recent_mean_delta": float(
                    season.loc[
                        season["origin_season"].ge(2022),
                        "challenger",
                    ].sub(
                        season.loc[
                            season["origin_season"].ge(2022),
                            "reference",
                        ]
                    ).mean()
                ),
                "bootstrap_95_lower": float(lower),
                "bootstrap_95_upper": float(upper),
                "exact_sign_flip_pvalue": sign_flip_pvalue(delta),
            }
        )
    return pd.DataFrame(rows)


def attach_shadow_residual_quantiles(
    current: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    output = current.copy()
    primary = evaluation[
        evaluation["target_name"].eq("next_conditional_ppg")
        & evaluation["method"].eq(PRIMARY_PPG_METHOD)
    ].copy()
    for quantile in RESIDUAL_QUANTILES:
        output[f"pred_resid_{int(quantile * 100)}_ny_shadow"] = np.nan
    output["next_residual_interval_pool"] = "unavailable"
    output["next_residual_interval_rows"] = 0
    output["history_depth"] = history_depth(output)
    for index, row in output.iterrows():
        pools = (
            (
                primary[
                    primary["position"].eq(row["position"])
                    & primary["history_depth"].eq(row["history_depth"])
                ],
                "position_history",
            ),
            (
                primary[primary["position"].eq(row["position"])],
                "position",
            ),
            (primary, "global"),
        )
        donors = pd.DataFrame()
        pool_name = "unavailable"
        for pool, candidate_name in pools:
            if len(pool) >= 100:
                donors = pool
                pool_name = candidate_name
                break
        if donors.empty:
            continue
        quantiles = donors["residual"].quantile(RESIDUAL_QUANTILES)
        for quantile, value in quantiles.items():
            output.loc[
                index,
                f"pred_resid_{int(float(quantile) * 100)}_ny_shadow",
            ] = float(value)
        output.loc[index, "next_residual_interval_pool"] = pool_name
        output.loc[index, "next_residual_interval_rows"] = len(donors)
    return output


def specification_table(run_id: str, league: str) -> pd.DataFrame:
    rows = []
    for model_name, grid in MODEL_GRIDS.items():
        features, target, probability = model_inputs(model_name)
        rows.append(
            {
                "run_id": run_id,
                "target_version": NEXT_YEAR_TARGET_VERSION,
                "league": league,
                "model_name": model_name,
                "target_column": target,
                "prediction_kind": (
                    "probability" if probability else "expert_residual"
                ),
                "feature_count": len(features),
                "feature_hash": feature_hash(features),
                "feature_names_json": json.dumps(list(features)),
                "grid_json": json.dumps(list(grid), sort_keys=True),
                "target_horizon": TARGET_HORIZON,
                "training_origin_embargo": 1,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(
    frame: pd.DataFrame,
    float_digits: int = 4,
) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{float(value):.{float_digits}f}"
            )
        )
    headers = [str(column) for column in display.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def summary_markdown(
    league: str,
    run_id: str,
    feature_run_id: str,
    targets: pd.DataFrame,
    current: pd.DataFrame,
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["slice_value"].eq("all")
    ][["target_name", "method", "metric", "n_rows", "value"]]
    comparison_lines = markdown_table(
        comparisons[
            [
                "comparison",
                "pooled_challenger",
                "pooled_reference",
                "mean_origin_delta",
                "origin_wins",
                "bootstrap_95_lower",
                "bootstrap_95_upper",
            ]
        ]
    )
    pooled_lines = markdown_table(pooled)
    return f"""# V2 Next-Year Residual Validation ({league})

## Scope

- Run: `{run_id}`
- Feature run: `{feature_run_id}`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: {min(VALIDATION_ORIGINS)}-{max(VALIDATION_ORIGINS)}.
- Each origin uses training labels only through origin minus two; the latest
  training target outcome season is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Pooled scores

{pooled_lines}

## Causal comparisons

{comparison_lines}

## 2027 shadow

- Candidate origin rows: {len(current):,}
- Conditional-PPG centers: {current[PRIMARY_PPG_METHOD].notna().sum():,}
- Following-season participation probabilities:
  {current[PRIMARY_PARTICIPATION_METHOD].notna().sum():,}
- Historical conditional training rows:
  {int(targets['next_conditional_ppg_training_eligible'].sum()):,}
- Historical participation labels:
  {int(targets['next_participation_target_available'].sum()):,}

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: {runtime_seconds:.1f} seconds.
"""


def main() -> None:
    args = parse_args()
    output_db = args.output_db.resolve()
    results_dir = args.results_dir.resolve()
    league = str(args.league).lower()
    started = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)

    features, outcomes, feature_run_id = load_inputs(output_db, league)
    missing = [
        feature
        for feature in (
            *NEXT_YEAR_RESIDUAL_FEATURES,
            *NEXT_YEAR_PARTICIPATION_FEATURES,
        )
        if feature not in features
    ]
    if missing:
        raise ValueError(f"Next-year feature lock is missing: {sorted(set(missing))}")
    targets = build_next_year_targets(
        features,
        outcomes,
        completed_through_season=COMPLETED_THROUGH_SEASON,
    )
    target_audit = build_next_year_target_audit(targets)
    run_id = create_run_id(f"v2_next_year_{league}")

    grid_frames: dict[str, pd.DataFrame] = {}
    selection_frames = []
    prediction_frames = []
    for model_name in MODEL_GRIDS:
        print(f"Fitting grid: {model_name}", flush=True)
        grid = grid_predictions(targets, model_name)
        selected = select_hyperparameters(grid, model_name)
        grid_frames[model_name] = grid
        selection_frames.append(selected)
        prediction_frames.append(
            selected_predictions(targets, model_name, selected)
        )
    prediction_frames.extend(
        [
            smoothed_group_predictions(targets, probability=False),
            smoothed_group_predictions(targets, probability=True),
        ]
    )
    selections = pd.concat(selection_frames, ignore_index=True)
    wide = assemble_wide(targets, prediction_frames)
    evaluation = evaluation_long(wide)
    scores = score_table(evaluation)
    comparisons = comparison_table(evaluation)

    current = wide[wide["origin_season"].eq(2026)].copy()
    current = attach_shadow_residual_quantiles(current, evaluation)
    current["run_id"] = run_id
    current["feature_run_id"] = feature_run_id
    current["target_version"] = NEXT_YEAR_TARGET_VERSION
    current["league"] = league
    current["scoring_hash"] = scoring_hash(league)
    current["publication_status"] = "shadow"

    handoff = wide[
        wide["origin_season"].isin(PREDICTION_ORIGINS)
    ].copy()
    handoff["run_id"] = run_id
    handoff["target_version"] = NEXT_YEAR_TARGET_VERSION
    handoff["league"] = league
    handoff["match_next_residual_rank_pct"] = handoff.groupby(
        ["origin_season", "position"]
    )["predicted_next_year_residual"].rank(
        method="average", pct=True
    )
    handoff["match_next_participation_probability"] = handoff[
        "predicted_next_year_appearance_probability"
    ]
    handoff["training_through_origin"] = handoff["origin_season"] - 2
    handoff["target_outcome_through"] = handoff["origin_season"] - 1
    handoff["forecast_status"] = np.where(
        handoff["origin_season"].isin(VALIDATION_ORIGINS),
        "strict_oos_validation",
        np.where(
            handoff["origin_season"].isin(SHADOW_ORIGINS),
            "shadow",
            "causal_historical_handoff",
        ),
    )
    handoff_columns = [
        "run_id",
        "target_version",
        "league",
        "player_key",
        "gsis_id",
        "display_name",
        "origin_season",
        "target_season",
        "position",
        "team",
        "origin_expert_ppg",
        "predicted_next_year_residual",
        "predicted_next_year_conditional_ppg",
        "predicted_next_year_appearance_probability",
        "match_next_residual_rank_pct",
        "match_next_participation_probability",
        "training_through_origin",
        "target_outcome_through",
        "forecast_status",
    ]
    handoff = handoff.loc[:, handoff_columns]
    if handoff.duplicated(["player_key", "origin_season"]).any():
        raise ValueError("Next-year template handoff contains duplicate keys")
    if (
        handoff["training_through_origin"]
        >= handoff["origin_season"] - 1
    ).any():
        raise ValueError("Next-year handoff violates the outcome embargo")

    specifications = specification_table(run_id, league)
    runtime_seconds = time.perf_counter() - started
    run = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "target_version": NEXT_YEAR_TARGET_VERSION,
                "league": league,
                "scoring_hash": scoring_hash(league),
                "feature_run_id": feature_run_id,
                "created_at": utc_now(),
                "validation_start_origin": min(VALIDATION_ORIGINS),
                "validation_end_origin": max(VALIDATION_ORIGINS),
                "current_origin": 2026,
                "forecast_target_season": 2027,
                "training_origin_embargo": 1,
                "primary_residual_method": PRIMARY_RESIDUAL_METHOD,
                "primary_participation_method": PRIMARY_PARTICIPATION_METHOD,
                "runtime_seconds": runtime_seconds,
                "status": "shadow",
            }
        ]
    )
    tables = {
        "next_year_candidate_runs": run,
        "next_year_candidate_specifications": specifications,
        "next_year_targets": targets,
        "next_year_target_audit": target_audit,
        "next_year_selected_hyperparameters": selections,
        "next_year_whole_season_predictions": evaluation,
        "next_year_model_scores": scores,
        "next_year_model_comparisons": comparisons,
        "next_year_template_handoff": handoff,
        "next_year_2027_shadow_predictions": current,
    }
    publish_tables_atomic(output_db, tables)
    for table, frame in tables.items():
        frame.to_csv(results_dir / f"{table}.csv", index=False)
    metadata = {
        "run_id": run_id,
        "target_version": NEXT_YEAR_TARGET_VERSION,
        "league": league,
        "feature_run_id": feature_run_id,
        "validation_origins": list(VALIDATION_ORIGINS),
        "prediction_origins": list(PREDICTION_ORIGINS),
        "training_origin_embargo": 1,
        "primary_residual_method": PRIMARY_RESIDUAL_METHOD,
        "primary_participation_method": PRIMARY_PARTICIPATION_METHOD,
        "runtime_seconds": runtime_seconds,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary = summary_markdown(
        league,
        run_id,
        feature_run_id,
        targets,
        current,
        scores,
        comparisons,
        runtime_seconds,
    )
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(comparisons.round(5).to_string(index=False), flush=True)
    print(
        f"Published {len(current)} {league} 2027 shadow rows to {output_db}",
        flush=True,
    )


if __name__ == "__main__":
    main()
