"""Run the locked V2 whole-season replay and publish 2026 shadow outputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Keep LightGBM after scikit-learn: on Windows this makes both packages use
# one vcomp runtime instead of loading separate copies of vcomp140.dll.
from lightgbm import LGBMClassifier, LGBMRegressor


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import OUTPUT_DB_PATH as DEFAULT_OUTPUT_DB_PATH, POSITIONS
from Scripts.V2.contracts import (
    create_run_id,
    publish_tables_atomic,
    scoring_hash,
    utc_now,
)
from Scripts.V2.locked_candidates import (
    HISTORY_GAP_PPG_FEATURES,
    LIGHTGBM_GRID,
    LOCKED_FEATURE_SETS,
    LOCKED_INNER_VALIDATION_START,
    LOCKED_RANDOM_SEED,
    LOCKED_VALIDATION_SEASONS,
    LOCK_VERSION,
    LOG_ADP_LASSO_FEATURES,
    MODEL_GRIDS,
    PARTICIPATION_FEATURES,
    PRIMARY_PPG_FEATURES,
    PROJECTION_ONLY_PPG_FEATURES,
    QB_STYLE_PPG_FEATURES,
    locked_metadata,
    lock_version_for_scoring,
    specification_table,
    validate_feature_lock,
)
from Scripts.V2.native_runtime import assert_single_openmp_runtime
from Scripts.V2.modeling import add_modeling_features, rolling_position_rate


DEFAULT_RESULTS_DIR = STUDY_DIR / "results"
ACTIVE_OUTPUT_DB_PATH = DEFAULT_OUTPUT_DB_PATH
ACTIVE_RESULTS_DIR = DEFAULT_RESULTS_DIR
ACTIVE_SCORING_OBJECTIVE = "dk"
ACTIVE_LOCK_VERSION = LOCK_VERSION
OUTER_SEASONS = LOCKED_VALIDATION_SEASONS
CURRENT_SEASON = max(OUTER_SEASONS) + 1
GRID_ORIGINS = tuple(
    range(LOCKED_INNER_VALIDATION_START, max(OUTER_SEASONS) + 1)
)
MIN_INNER_SEASONS = 3
MIN_ROUTER_ROWS = 50
MIN_INTERVAL_ROWS = 100

PRIMARY_COMPONENTS = (
    "conditional_ppg_lasso",
    "conditional_ppg_random_forest",
    "conditional_ppg_lightgbm",
)
PRIMARY_METHOD = "conditional_ppg_primary_blend"
PUBLISHED_METHOD = PRIMARY_METHOD
PPG_METHODS = (
    "expert_team_game",
    "expert_recalibrated",
    *PRIMARY_COMPONENTS,
    PRIMARY_METHOD,
    "conditional_ppg_log_lasso_tree_blend",
    "conditional_ppg_history_gap_no_history_route",
    "conditional_ppg_projection_history_router",
    "conditional_ppg_qb_style_wrte_route",
)
PARTICIPATION_METHODS = (
    "participation_prior_position_rate",
    "participation_logistic",
    "participation_lightgbm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-db",
        type=Path,
        default=DEFAULT_OUTPUT_DB_PATH,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--league", default="dk")
    return parser.parse_args()


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with sqlite3.connect(ACTIVE_OUTPUT_DB_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features", connection
        )
        manifests = pd.read_sql_query(
            "SELECT * FROM feature_manifests", connection
        )
    feature_run_ids = features["run_id"].dropna().astype(str).unique()
    if len(feature_run_ids) != 1:
        raise ValueError("Expected exactly one active V2 feature run")
    feature_run_id = str(feature_run_ids[0])
    manifest_run_ids = set(manifests["run_id"].dropna().astype(str))
    if manifest_run_ids != {feature_run_id}:
        raise ValueError("Feature manifests do not match the active mart")
    features = add_modeling_features(features)
    validate_feature_lock(features, manifests)
    observed_leagues = set(features["league"].dropna().astype(str))
    if observed_leagues != {ACTIVE_SCORING_OBJECTIVE}:
        raise ValueError(
            "Active feature mart league does not match the requested scoring "
            f"objective: observed={sorted(observed_leagues)}, "
            f"requested={ACTIVE_SCORING_OBJECTIVE}"
        )
    observed_scoring_hashes = set(
        features["scoring_hash"].dropna().astype(str)
    )
    expected_scoring_hash = scoring_hash(ACTIVE_SCORING_OBJECTIVE)
    if observed_scoring_hashes != {expected_scoring_hash}:
        raise ValueError(
            "Active feature mart scoring hash does not match the requested "
            f"objective: observed={sorted(observed_scoring_hashes)}, "
            f"expected={expected_scoring_hash}"
        )
    features["season"] = pd.to_numeric(
        features["season"], errors="raise"
    ).astype(int)
    features.sort_values(["season", "player_key"], inplace=True)
    features.reset_index(drop=True, inplace=True)
    return features, manifests, feature_run_id


def _target_frames(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ppg = features[
        features["position"].isin(POSITIONS)
        & features["season"].le(max(OUTER_SEASONS))
        & features["conditional_ppg_training_eligible"].eq(1)
        & features["conditional_ppg"].notna()
        & features["expert_ppg_team_game_median"].notna()
    ].copy()
    ppg["actual"] = pd.to_numeric(ppg["conditional_ppg"], errors="raise")

    participation = features[
        features["position"].isin(POSITIONS)
        & features["season"].le(max(OUTER_SEASONS))
        & features["active_target_available"].eq(1)
        & features["appeared"].notna()
    ].copy()
    participation["actual"] = pd.to_numeric(
        participation["appeared"], errors="raise"
    )

    candidates = features[
        features["position"].isin(POSITIONS)
        & features["season"].between(min(OUTER_SEASONS), CURRENT_SEASON)
    ].copy()
    for frame in (ppg, participation, candidates):
        frame.sort_values(["season", "player_key"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return ppg, participation, candidates


def _model_pipeline(
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
        "expert_recalibrated_ridge",
        "conditional_ppg_lasso",
        "participation_logistic",
    }:
        steps.append(("scale", StandardScaler()))

    if model_name == "expert_recalibrated_ridge":
        estimator = Ridge(max_iter=5_000, **parameters)
    elif model_name == "conditional_ppg_lasso":
        estimator = Lasso(max_iter=20_000, tol=1e-6, **parameters)
    elif model_name == "conditional_ppg_random_forest":
        estimator = RandomForestRegressor(**parameters)
    elif model_name == "conditional_ppg_lightgbm":
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
    elif model_name == "participation_logistic":
        estimator = LogisticRegression(
            max_iter=3_000,
            solver="lbfgs",
            random_state=LOCKED_RANDOM_SEED,
            **parameters,
        )
    elif model_name == "participation_lightgbm":
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
        raise ValueError(f"Unsupported locked model: {model_name}")
    steps.append(("model", estimator))
    return Pipeline(steps)


def _predict(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    probability: bool,
) -> np.ndarray:
    X = frame.loc[:, list(feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if probability:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _fit(
    model_name: str,
    parameters: Mapping[str, object],
    train: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Pipeline:
    X = train.loc[:, list(feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    y = pd.to_numeric(train["actual"], errors="raise")
    model = _model_pipeline(model_name, parameters)
    model.fit(X, y)
    return model


def _grid_predictions(
    target: pd.DataFrame,
    feature_columns: Sequence[str],
    model_name: str,
    grid: Sequence[Mapping[str, object]],
    probability: bool = False,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for origin in GRID_ORIGINS:
        train = target[target["season"].lt(origin)]
        hold = target[target["season"].eq(origin)]
        if train.empty or hold.empty:
            continue
        for candidate_id, parameters in enumerate(grid):
            model = _fit(
                model_name, parameters, train, feature_columns
            )
            prediction = _predict(
                model, hold, feature_columns, probability=probability
            )
            current = hold[
                ["player_key", "season", "position", "actual"]
            ].copy()
            current["model_name"] = model_name
            current["candidate_id"] = candidate_id
            current["prediction"] = prediction
            rows.append(current)
    if not rows:
        raise ValueError(f"No grid predictions generated for {model_name}")
    return pd.concat(rows, ignore_index=True)


def _metric(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    probability: bool,
) -> float:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(prediction, dtype=float)
    if probability:
        return float(
            brier_score_loss(
                actual_values, np.clip(predicted_values, 1e-6, 1 - 1e-6)
            )
        )
    return float(
        np.sqrt(mean_squared_error(actual_values, predicted_values))
    )


def _select_hyperparameters(
    grid_predictions: pd.DataFrame,
    grid: Sequence[Mapping[str, object]],
    model_name: str,
    probability: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for origin in (*OUTER_SEASONS, CURRENT_SEASON):
        prior = grid_predictions[grid_predictions["season"].lt(origin)]
        prior_seasons = sorted(prior["season"].unique())
        if len(prior_seasons) < MIN_INNER_SEASONS:
            raise ValueError(
                f"{model_name}/{origin} has only {len(prior_seasons)} "
                "strict-prior inner seasons"
            )
        scores: list[tuple[float, int]] = []
        for candidate_id, candidate in prior.groupby("candidate_id"):
            season_scores = [
                _metric(
                    season["actual"],
                    season["prediction"],
                    probability=probability,
                )
                for _, season in candidate.groupby("season")
            ]
            scores.append((float(np.mean(season_scores)), int(candidate_id)))
        selected_score, selected_id = min(scores)
        rows.append(
            {
                "model_name": model_name,
                "forecast_origin": origin,
                "candidate_id": selected_id,
                "parameters_json": json.dumps(
                    grid[selected_id], sort_keys=True
                ),
                "selection_metric": "brier" if probability else "rmse",
                "selection_score": selected_score,
                "selection_start_season": min(prior_seasons),
                "selection_end_season": max(prior_seasons),
                "selection_seasons": len(prior_seasons),
            }
        )
    return pd.DataFrame(rows)


def _selected_predictions(
    train_target: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    fit_model_name: str,
    output_model_name: str,
    selected: pd.DataFrame,
    probability: bool = False,
    require_expert: bool = True,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for origin in (*OUTER_SEASONS, CURRENT_SEASON):
        train = train_target[train_target["season"].lt(origin)]
        hold = candidates[candidates["season"].eq(origin)].copy()
        if require_expert:
            hold = hold[hold["expert_ppg_team_game_median"].notna()].copy()
        selection = selected[
            selected["forecast_origin"].eq(origin)
            & selected["model_name"].eq(fit_model_name)
        ]
        if len(selection) != 1:
            raise ValueError(
                f"Expected one selected {fit_model_name} row for {origin}"
            )
        parameters = json.loads(selection.iloc[0]["parameters_json"])
        model = _fit(
            fit_model_name, parameters, train, feature_columns
        )
        prediction = _predict(
            model, hold, feature_columns, probability=probability
        )
        current = hold[
            ["player_key", "season", "position"]
        ].copy()
        current["model_name"] = output_model_name
        current["prediction"] = prediction
        current["training_through_season"] = origin - 1
        current["selected_candidate_id"] = int(
            selection.iloc[0]["candidate_id"]
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def _prior_position_predictions(
    participation: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for origin in (*OUTER_SEASONS, CURRENT_SEASON):
        prior = participation[participation["season"].lt(origin)].copy()
        current = candidates[candidates["season"].eq(origin)][
            ["player_key", "season", "position"]
        ].copy()
        if prior.empty:
            current["prediction"] = 0.5
        else:
            global_rate = float(prior["actual"].mean())
            stats = prior.groupby("position")["actual"].agg(["sum", "count"])
            position_rates = {}
            for position in POSITIONS:
                if position in stats.index:
                    position_rates[position] = (
                        float(stats.loc[position, "sum"])
                        + 25.0 * global_rate
                    ) / (float(stats.loc[position, "count"]) + 25.0)
                else:
                    position_rates[position] = global_rate
            current["prediction"] = (
                current["position"].map(position_rates).fillna(global_rate)
            )
        current["model_name"] = "participation_prior_position_rate"
        current["training_through_season"] = origin - 1
        current["selected_candidate_id"] = pd.NA
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def _wide_predictions(
    candidates: pd.DataFrame,
    prediction_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    metadata = [
        "player_key",
        "gsis_id",
        "display_name",
        "season",
        "position",
        "team",
        "identity_status",
        "outcome_join_status",
        "conditional_ppg_training_eligible",
        "conditional_ppg_target_available",
        "conditional_ppg",
        "active_target_available",
        "appeared",
        "opportunity_games",
        "has_prior_outcome",
        "is_rookie",
        "year_exp",
        "projection_provider_count",
        "projection_trajectory_prior_year_available",
        "adp_median",
        "expert_ppg_team_game_median",
        "expert_ppg_active_median",
    ]
    frame = candidates.loc[:, metadata].copy()
    long = pd.concat(prediction_frames, ignore_index=True)
    if long.duplicated(["player_key", "season", "model_name"]).any():
        raise ValueError("Candidate predictions are not unique by model/key")
    wide = long.pivot(
        index=["player_key", "season"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    frame = frame.merge(
        wide,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    return frame


def _add_history_depth(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    year_exp = pd.to_numeric(output["year_exp"], errors="coerce")
    rookie = pd.to_numeric(
        output["is_rookie"], errors="coerce"
    ).fillna(0).eq(1)
    prior = pd.to_numeric(
        output["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    depth = pd.Series("other_no_history", index=output.index, dtype=object)
    depth.loc[rookie] = "rookie"
    depth.loc[~rookie & year_exp.eq(1)] = "second_year"
    depth.loc[~rookie & year_exp.ge(2) & prior] = "veteran_with_history"
    depth.loc[year_exp.isna()] = "unknown_experience"
    output["history_depth"] = depth
    output["limited_history"] = (
        rookie | year_exp.le(1) | ~prior
    ).astype(int)
    return output


def _blend(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame.loc[:, list(columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    return values.mean(axis=1, skipna=False)


def _projection_history_router(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    prediction = frame[PRIMARY_METHOD].copy()
    source = pd.Series(PRIMARY_METHOD, index=frame.index, dtype=object)
    decisions: list[dict[str, object]] = []
    for origin in (*OUTER_SEASONS, CURRENT_SEASON):
        for position in POSITIONS:
            prior = frame[
                frame["season"].lt(origin)
                & frame["season"].isin(OUTER_SEASONS)
                & frame["position"].eq(position)
                & frame["limited_history"].eq(1)
                & frame["conditional_ppg_training_eligible"].eq(1)
                & frame["conditional_ppg"].notna()
                & frame[PRIMARY_METHOD].notna()
                & frame["conditional_ppg_projection_only"].notna()
            ]
            use_projection = False
            primary_rmse = np.nan
            projection_rmse = np.nan
            if len(prior) >= MIN_ROUTER_ROWS:
                primary_rmse = _metric(
                    prior["conditional_ppg"],
                    prior[PRIMARY_METHOD],
                    probability=False,
                )
                projection_rmse = _metric(
                    prior["conditional_ppg"],
                    prior["conditional_ppg_projection_only"],
                    probability=False,
                )
                use_projection = projection_rmse < primary_rmse
            current = (
                frame["season"].eq(origin)
                & frame["position"].eq(position)
                & frame["limited_history"].eq(1)
            )
            if use_projection:
                prediction.loc[current] = frame.loc[
                    current, "conditional_ppg_projection_only"
                ]
                source.loc[current] = "conditional_ppg_projection_only"
            decisions.append(
                {
                    "forecast_origin": origin,
                    "position": position,
                    "history_group": "limited_history",
                    "prior_rows": len(prior),
                    "primary_rmse": primary_rmse,
                    "projection_only_rmse": projection_rmse,
                    "selected_source": (
                        "conditional_ppg_projection_only"
                        if use_projection
                        else PRIMARY_METHOD
                    ),
                }
            )
    return prediction, source, pd.DataFrame(decisions)


def _assemble_methods(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = _add_history_depth(frame)
    output["expert_team_game"] = output["expert_ppg_team_game_median"]
    output[PRIMARY_METHOD] = _blend(output, PRIMARY_COMPONENTS)
    output["conditional_ppg_log_lasso_tree_blend"] = _blend(
        output,
        (
            "conditional_ppg_log_adp_lasso",
            "conditional_ppg_random_forest",
            "conditional_ppg_lightgbm",
        ),
    )
    output["conditional_ppg_history_gap_blend"] = _blend(
        output,
        (
            "conditional_ppg_history_gap_lasso",
            "conditional_ppg_history_gap_random_forest",
            "conditional_ppg_history_gap_lightgbm",
        ),
    )
    no_history = pd.to_numeric(
        output["has_prior_outcome"], errors="coerce"
    ).fillna(0).ne(1)
    output["conditional_ppg_history_gap_no_history_route"] = output[
        PRIMARY_METHOD
    ]
    output.loc[
        no_history, "conditional_ppg_history_gap_no_history_route"
    ] = output.loc[no_history, "conditional_ppg_history_gap_blend"]

    output["conditional_ppg_qb_style_blend"] = _blend(
        output,
        (
            "conditional_ppg_qb_style_lasso",
            "conditional_ppg_qb_style_random_forest",
            "conditional_ppg_qb_style_lightgbm",
        ),
    )
    pass_catcher = output["position"].isin(("WR", "TE"))
    output["conditional_ppg_qb_style_wrte_route"] = output[PRIMARY_METHOD]
    output.loc[
        pass_catcher, "conditional_ppg_qb_style_wrte_route"
    ] = output.loc[pass_catcher, "conditional_ppg_qb_style_blend"]

    (
        output["conditional_ppg_projection_history_router"],
        output["projection_router_source"],
        router_decisions,
    ) = _projection_history_router(output)
    return output, router_decisions


def _evaluation_long(
    frame: pd.DataFrame,
    methods: Sequence[str],
    target_column: str,
    eligible: pd.Series,
    target_name: str,
) -> pd.DataFrame:
    metadata = [
        "player_key",
        "season",
        "position",
        "history_depth",
        "limited_history",
        "is_rookie",
        "year_exp",
        "has_prior_outcome",
        "projection_provider_count",
        "projection_trajectory_prior_year_available",
        "adp_median",
    ]
    base = frame.loc[eligible, metadata + [target_column]].rename(
        columns={target_column: "actual"}
    )
    rows = []
    for method in methods:
        current = base.copy()
        current["method"] = method
        current["prediction"] = frame.loc[eligible, method].to_numpy()
        current["target_name"] = target_name
        rows.append(current)
    output = pd.concat(rows, ignore_index=True)
    output = output[output["prediction"].notna()].copy()
    output["residual"] = output["actual"] - output["prediction"]
    return output


def _score_group(
    group: pd.DataFrame,
    target_name: str,
) -> dict[str, float]:
    actual = group["actual"].to_numpy(dtype=float)
    prediction = group["prediction"].to_numpy(dtype=float)
    if target_name == "conditional_ppg":
        return {
            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
            "mae": float(mean_absolute_error(actual, prediction)),
            "bias": float(np.mean(prediction - actual)),
            "spearman": float(spearmanr(actual, prediction).statistic),
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


def _score_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target_name, method), model in evaluation.groupby(
        ["target_name", "method"], sort=True
    ):
        slices: list[tuple[str, str, pd.DataFrame]] = [
            ("pooled", "all", model),
            (
                "provider_era",
                "2023_2025",
                model[model["season"].ge(2023)],
            ),
        ]
        slices.extend(
            ("season", str(value), group)
            for value, group in model.groupby("season")
        )
        slices.extend(
            ("position", str(value), group)
            for value, group in model.groupby("position")
        )
        slices.extend(
            ("history_depth", str(value), group)
            for value, group in model.groupby("history_depth")
        )
        for slice_type, slice_value, group in slices:
            if group.empty:
                continue
            for metric_name, value in _score_group(
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
                        "n_seasons": group["season"].nunique(),
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def _sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan
    signs = np.array(
        [
            [1 if (mask >> bit) & 1 else -1 for bit in range(n)]
            for mask in range(2**n)
        ],
        dtype=float,
    )
    observed = abs(float(values.mean()))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def _comparison_table(
    ppg_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    season = (
        ppg_evaluation.groupby(["method", "season"])
        .apply(
            lambda group: _metric(
                group["actual"], group["prediction"], probability=False
            ),
            include_groups=False,
        )
        .rename("rmse")
        .reset_index()
    )
    comparisons = [
        (PRIMARY_METHOD, "expert_recalibrated", "primary_vs_expert"),
        (
            "conditional_ppg_log_lasso_tree_blend",
            PRIMARY_METHOD,
            "log_lasso_vs_primary",
        ),
        (
            "conditional_ppg_history_gap_no_history_route",
            PRIMARY_METHOD,
            "history_gap_route_vs_primary",
        ),
        (
            "conditional_ppg_projection_history_router",
            PRIMARY_METHOD,
            "projection_router_vs_primary",
        ),
        (
            "conditional_ppg_qb_style_wrte_route",
            PRIMARY_METHOD,
            "qb_style_route_vs_primary",
        ),
    ]
    rows = []
    rng = np.random.default_rng(LOCKED_RANDOM_SEED)
    for challenger, reference, comparison_name in comparisons:
        merged = season[season["method"].eq(challenger)].merge(
            season[season["method"].eq(reference)],
            on="season",
            suffixes=("_challenger", "_reference"),
            validate="one_to_one",
        )
        delta = (
            merged["rmse_challenger"] - merged["rmse_reference"]
        ).to_numpy(dtype=float)
        if len(delta):
            indices = rng.integers(
                0, len(delta), size=(20_000, len(delta))
            )
            bootstrap = delta[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        else:
            lower = upper = np.nan
        recent = merged[merged["season"].ge(2023)]
        rows.append(
            {
                "comparison": comparison_name,
                "challenger": challenger,
                "reference": reference,
                "seasons": len(merged),
                "mean_season_rmse_delta": float(np.mean(delta)),
                "pooled_rmse_challenger": _metric(
                    ppg_evaluation.loc[
                        ppg_evaluation["method"].eq(challenger), "actual"
                    ],
                    ppg_evaluation.loc[
                        ppg_evaluation["method"].eq(challenger), "prediction"
                    ],
                    probability=False,
                ),
                "pooled_rmse_reference": _metric(
                    ppg_evaluation.loc[
                        ppg_evaluation["method"].eq(reference), "actual"
                    ],
                    ppg_evaluation.loc[
                        ppg_evaluation["method"].eq(reference), "prediction"
                    ],
                    probability=False,
                ),
                "season_wins": int(np.sum(delta < 0)),
                "recent_mean_delta": float(
                    np.mean(
                        recent["rmse_challenger"]
                        - recent["rmse_reference"]
                    )
                ),
                "bootstrap_95_lower": float(lower),
                "bootstrap_95_upper": float(upper),
                "exact_sign_flip_pvalue": _sign_flip_pvalue(delta),
            }
        )
    return pd.DataFrame(rows)


def _calibration_bins(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (target_name, method), model in evaluation.groupby(
        ["target_name", "method"], sort=True
    ):
        model = model.copy()
        try:
            model["prediction_bin"] = pd.qcut(
                model["prediction"],
                q=10,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            model["prediction_bin"] = 0
        for prediction_bin, group in model.groupby("prediction_bin"):
            rows.append(
                {
                    "target_name": target_name,
                    "method": method,
                    "prediction_bin": int(prediction_bin),
                    "n_rows": len(group),
                    "prediction_min": float(group["prediction"].min()),
                    "prediction_max": float(group["prediction"].max()),
                    "mean_prediction": float(group["prediction"].mean()),
                    "mean_actual": float(group["actual"].mean()),
                    "bias": float(
                        (group["prediction"] - group["actual"]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _strict_prior_intervals(
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = primary.copy()
    frame["q10"] = np.nan
    frame["q25"] = np.nan
    frame["q50"] = np.nan
    frame["q75"] = np.nan
    frame["q90"] = np.nan
    frame["interval_donor_rows"] = 0
    frame["interval_pool"] = "unavailable"
    for index, row in frame.iterrows():
        prior = frame[frame["season"].lt(row["season"])]
        pools = (
            (
                prior[
                    prior["position"].eq(row["position"])
                    & prior["history_depth"].eq(row["history_depth"])
                ],
                "position_history",
            ),
            (
                prior[prior["position"].eq(row["position"])],
                "position",
            ),
            (prior, "global"),
        )
        donors = pd.DataFrame()
        pool_name = "unavailable"
        for pool, candidate_name in pools:
            if len(pool) >= MIN_INTERVAL_ROWS:
                donors = pool
                pool_name = candidate_name
                break
        if donors.empty:
            continue
        quantiles = donors["residual"].quantile(
            [0.10, 0.25, 0.50, 0.75, 0.90]
        )
        frame.loc[index, ["q10", "q25", "q50", "q75", "q90"]] = (
            quantiles.to_numpy()
        )
        frame.loc[index, "interval_donor_rows"] = len(donors)
        frame.loc[index, "interval_pool"] = pool_name
    available = frame["q10"].notna() & frame["q90"].notna()
    frame["covered_50"] = np.where(
        frame["q25"].notna(),
        frame["residual"].between(frame["q25"], frame["q75"]),
        np.nan,
    )
    frame["covered_80"] = np.where(
        available,
        frame["residual"].between(frame["q10"], frame["q90"]),
        np.nan,
    )
    rows = []
    for slice_type, column in (
        ("pooled", None),
        ("position", "position"),
        ("history_depth", "history_depth"),
        ("season", "season"),
    ):
        groups = (
            [("all", frame)]
            if column is None
            else list(frame.groupby(column))
        )
        for value, group in groups:
            valid = group[group["q10"].notna() & group["q90"].notna()]
            if valid.empty:
                continue
            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": str(value),
                    "n_rows": len(group),
                    "n_interval_rows": len(valid),
                    "interval_coverage": len(valid) / len(group),
                    "p25_p75_coverage": float(valid["covered_50"].mean()),
                    "p10_p90_coverage": float(valid["covered_80"].mean()),
                    "mean_interval_width": float(
                        (valid["q90"] - valid["q10"]).mean()
                    ),
                    "median_residual": float(valid["residual"].median()),
                }
            )
    return frame, pd.DataFrame(rows)


def _template_handoff(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = frame[frame["season"].isin(OUTER_SEASONS)].copy()
    current = frame[frame["season"].eq(CURRENT_SEASON)].copy()
    handoff = pd.concat([historical, current], ignore_index=True)
    handoff["historical_pred_fp_per_game"] = handoff[PUBLISHED_METHOD]
    # This is the V2 season target used for model validation. It is not the
    # weekly-template active_ppg field, whose first-16-week and short-QB rules
    # remain owned by s4_Best_Ball_Weekly.py.
    handoff["v2_conditional_ppg_actual"] = pd.to_numeric(
        handoff["conditional_ppg"], errors="coerce"
    )
    handoff["v2_conditional_ppg_training_residual"] = (
        handoff["v2_conditional_ppg_actual"]
        - handoff["historical_pred_fp_per_game"]
    )
    handoff.loc[
        handoff["season"].eq(CURRENT_SEASON),
        "v2_conditional_ppg_training_residual",
    ] = np.nan
    handoff["point_center_source"] = ACTIVE_LOCK_VERSION
    handoff["joint_template_draw_required"] = 1
    handoff["independent_model_residual_draw_allowed"] = 0
    handoff["template_active_ppg_resid_recompute_required"] = 1
    handoff["template_center_available"] = handoff[
        "historical_pred_fp_per_game"
    ].notna().astype(int)

    valid = handoff[
        handoff["v2_conditional_ppg_training_residual"].notna()
        & handoff["historical_pred_fp_per_game"].notna()
    ]
    reconstruction_error = (
        valid["historical_pred_fp_per_game"]
        + valid["v2_conditional_ppg_training_residual"]
        - valid["v2_conditional_ppg_actual"]
    ).abs()
    residual_variance = float(
        valid["v2_conditional_ppg_training_residual"].var(ddof=0)
    )
    rows = [
        {
            "audit": "point_center_coverage_all",
            "slice": "all",
            "n_rows": len(handoff),
            "value": float(handoff["template_center_available"].mean()),
        },
        {
            "audit": "point_center_coverage_current",
            "slice": str(CURRENT_SEASON),
            "n_rows": len(current),
            "value": float(
                current[PUBLISHED_METHOD].notna().mean()
                if len(current)
                else np.nan
            ),
        },
        {
            "audit": "residual_reconstruction_max_abs_error",
            "slice": "historical_actual_available",
            "n_rows": len(valid),
            "value": float(reconstruction_error.max()),
        },
        {
            "audit": "single_template_residual_variance",
            "slice": "historical_actual_available",
            "n_rows": len(valid),
            "value": residual_variance,
        },
        {
            "audit": "independent_double_draw_variance",
            "slice": "prohibited_counterfactual",
            "n_rows": len(valid),
            "value": 2.0 * residual_variance,
        },
        {
            "audit": "independent_double_draw_variance_ratio",
            "slice": "prohibited_counterfactual",
            "n_rows": len(valid),
            "value": 2.0,
        },
    ]
    for (season, position), group in handoff.groupby(
        ["season", "position"]
    ):
        rows.append(
            {
                "audit": "point_center_coverage",
                "slice": f"{season}_{position}",
                "n_rows": len(group),
                "value": float(group["template_center_available"].mean()),
            }
        )
    columns = [
        "player_key",
        "gsis_id",
        "display_name",
        "season",
        "position",
        "team",
        "historical_pred_fp_per_game",
        "participation_lightgbm",
        "v2_conditional_ppg_actual",
        "v2_conditional_ppg_training_residual",
        "point_center_source",
        "joint_template_draw_required",
        "independent_model_residual_draw_allowed",
        "template_active_ppg_resid_recompute_required",
        "template_center_available",
    ]
    return handoff.loc[:, columns], pd.DataFrame(rows)


def _summary_markdown(
    run_id: str,
    feature_run_id: str,
    comparisons: pd.DataFrame,
    scores: pd.DataFrame,
    intervals: pd.DataFrame,
    template_audit: pd.DataFrame,
    current: pd.DataFrame,
) -> str:
    ppg = scores[
        scores["target_name"].eq("conditional_ppg")
        & scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    participation = scores[
        scores["target_name"].eq("participation")
        & scores["slice_type"].eq("pooled")
        & scores["metric"].eq("brier")
    ].sort_values("value")
    interval = intervals[
        intervals["slice_type"].eq("pooled")
    ].iloc[0]
    reconstruction = template_audit[
        template_audit["audit"].eq(
            "residual_reconstruction_max_abs_error"
        )
    ]["value"].iloc[0]

    def score_rows(frame: pd.DataFrame) -> str:
        return "\n".join(
            f"| `{row.method}` | {row.value:.4f} |"
            for row in frame.itertuples(index=False)
        )

    comparison_rows = "\n".join(
        "| `{}` | {:+.4f} | {}/{} | {:+.4f} | [{:+.4f}, {:+.4f}] |".format(
            row.comparison,
            row.mean_season_rmse_delta,
            row.season_wins,
            row.seasons,
            row.recent_mean_delta,
            row.bootstrap_95_lower,
            row.bootstrap_95_upper,
        )
        for row in comparisons.itertuples(index=False)
    )
    return f"""# Locked V2 Whole-Season and Shadow Results

- Lock: `{ACTIVE_LOCK_VERSION}`
- Run: `{run_id}`
- Feature run: `{feature_run_id}`
- Whole-season forecast origins: {min(OUTER_SEASONS)}-{max(OUTER_SEASONS)}
- Current shadow season: {CURRENT_SEASON}
- Current candidate rows: {len(current)}
- Current conditional-PPG centers: {current[PUBLISHED_METHOD].notna().sum()}

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
{score_rows(ppg)}

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
{comparison_rows}

## Participation

| Method | Pooled Brier |
|---|---:|
{score_rows(participation)}

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: {interval.p25_p75_coverage:.3f}
- Strict-prior P10-P90 residual coverage: {interval.p10_p90_coverage:.3f}
- Residual reconstruction maximum absolute error: {reconstruction:.3g}
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
"""


def main() -> None:
    global ACTIVE_OUTPUT_DB_PATH
    global ACTIVE_RESULTS_DIR
    global ACTIVE_SCORING_OBJECTIVE
    global ACTIVE_LOCK_VERSION

    assert_single_openmp_runtime()
    args = parse_args()
    ACTIVE_OUTPUT_DB_PATH = args.output_db
    ACTIVE_RESULTS_DIR = args.results_dir
    ACTIVE_SCORING_OBJECTIVE = str(args.league).strip().lower()
    ACTIVE_LOCK_VERSION = lock_version_for_scoring(
        ACTIVE_SCORING_OBJECTIVE
    )
    ACTIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = create_run_id(f"v2_locked_final_{ACTIVE_SCORING_OBJECTIVE}")
    features, _, feature_run_id = _load_inputs()
    ppg, participation, candidates = _target_frames(features)
    specifications = specification_table(ACTIVE_LOCK_VERSION)

    model_inputs = {
        "expert_recalibrated_ridge": (
            ppg,
            LOCKED_FEATURE_SETS["expert_recalibration"],
            False,
        ),
        "conditional_ppg_lasso": (
            ppg,
            PRIMARY_PPG_FEATURES,
            False,
        ),
        "conditional_ppg_random_forest": (
            ppg,
            PRIMARY_PPG_FEATURES,
            False,
        ),
        "conditional_ppg_lightgbm": (
            ppg,
            PRIMARY_PPG_FEATURES,
            False,
        ),
        "participation_logistic": (
            participation,
            PARTICIPATION_FEATURES,
            True,
        ),
        "participation_lightgbm": (
            participation,
            PARTICIPATION_FEATURES,
            True,
        ),
    }
    selections = []
    selected_prediction_frames = []
    for model_name, (target, feature_columns, probability) in model_inputs.items():
        print(f"Nested whole-season grid: {model_name}", flush=True)
        grid_predictions = _grid_predictions(
            target,
            feature_columns,
            model_name,
            MODEL_GRIDS[model_name],
            probability=probability,
        )
        selected = _select_hyperparameters(
            grid_predictions,
            MODEL_GRIDS[model_name],
            model_name,
            probability=probability,
        )
        selections.append(selected)
        selected_prediction_frames.append(
            _selected_predictions(
                target,
                candidates,
                feature_columns,
                fit_model_name=model_name,
                output_model_name=(
                    "expert_recalibrated"
                    if model_name == "expert_recalibrated_ridge"
                    else model_name
                ),
                selected=selected,
                probability=probability,
                require_expert=not model_name.startswith("participation_"),
            )
        )
    selected_hyperparameters = pd.concat(selections, ignore_index=True)

    print("Nested whole-season grid: conditional_ppg_log_adp_lasso", flush=True)
    log_grid = _grid_predictions(
        ppg,
        LOG_ADP_LASSO_FEATURES,
        "conditional_ppg_lasso",
        MODEL_GRIDS["conditional_ppg_lasso"],
    )
    log_selected = _select_hyperparameters(
        log_grid,
        MODEL_GRIDS["conditional_ppg_lasso"],
        "conditional_ppg_lasso",
    )
    log_selected["model_name"] = "conditional_ppg_log_adp_lasso"
    selected_hyperparameters = pd.concat(
        [selected_hyperparameters, log_selected], ignore_index=True
    )
    log_fit_selection = log_selected.copy()
    log_fit_selection["model_name"] = "conditional_ppg_lasso"
    selected_prediction_frames.append(
        _selected_predictions(
            ppg,
            candidates,
            LOG_ADP_LASSO_FEATURES,
            fit_model_name="conditional_ppg_lasso",
            output_model_name="conditional_ppg_log_adp_lasso",
            selected=log_fit_selection,
        )
    )

    primary_selections = {
        model: selected_hyperparameters[
            selected_hyperparameters["model_name"].eq(model)
        ].copy()
        for model in PRIMARY_COMPONENTS
    }
    for variant_name, feature_columns in (
        ("conditional_ppg_history_gap", HISTORY_GAP_PPG_FEATURES),
        ("conditional_ppg_qb_style", QB_STYLE_PPG_FEATURES),
    ):
        for component in PRIMARY_COMPONENTS:
            suffix = component.removeprefix("conditional_ppg_")
            output_name = f"{variant_name}_{suffix}"
            print(f"Selected-grid replay: {output_name}", flush=True)
            selected_prediction_frames.append(
                _selected_predictions(
                    ppg,
                    candidates,
                    feature_columns,
                    fit_model_name=component,
                    output_model_name=output_name,
                    selected=primary_selections[component],
                )
            )
    print("Selected-grid replay: conditional_ppg_projection_only", flush=True)
    selected_prediction_frames.append(
        _selected_predictions(
            ppg,
            candidates,
            PROJECTION_ONLY_PPG_FEATURES,
            fit_model_name="conditional_ppg_lightgbm",
            output_model_name="conditional_ppg_projection_only",
            selected=primary_selections["conditional_ppg_lightgbm"],
        )
    )
    selected_prediction_frames.append(
        _prior_position_predictions(participation, candidates)
    )

    wide = _wide_predictions(candidates, selected_prediction_frames)
    wide, router_decisions = _assemble_methods(wide)

    ppg_eligible = (
        wide["season"].isin(OUTER_SEASONS)
        & wide["conditional_ppg_training_eligible"].eq(1)
        & wide["conditional_ppg"].notna()
        & wide[PRIMARY_METHOD].notna()
    )
    ppg_evaluation = _evaluation_long(
        wide,
        PPG_METHODS,
        target_column="conditional_ppg",
        eligible=ppg_eligible,
        target_name="conditional_ppg",
    )

    participation_eligible = (
        wide["season"].isin(OUTER_SEASONS)
        & wide["active_target_available"].eq(1)
        & wide["appeared"].notna()
    )
    participation_evaluation = _evaluation_long(
        wide,
        PARTICIPATION_METHODS,
        target_column="appeared",
        eligible=participation_eligible,
        target_name="participation",
    )
    evaluation = pd.concat(
        [ppg_evaluation, participation_evaluation], ignore_index=True
    )
    scores = _score_table(evaluation)
    comparisons = _comparison_table(ppg_evaluation)
    calibration_bins = _calibration_bins(evaluation)
    primary_rows = ppg_evaluation[
        ppg_evaluation["method"].eq(PRIMARY_METHOD)
    ].copy()
    interval_rows, interval_summary = _strict_prior_intervals(primary_rows)

    template_handoff, template_audit = _template_handoff(wide)
    current = wide[wide["season"].eq(CURRENT_SEASON)].copy()
    current["lock_version"] = ACTIVE_LOCK_VERSION
    current["model_run_id"] = run_id
    current["conditional_ppg_shadow_unclipped"] = current[PUBLISHED_METHOD]
    current["conditional_ppg_shadow"] = current[
        PUBLISHED_METHOD
    ].clip(lower=0)
    current["conditional_ppg_shadow_method"] = PUBLISHED_METHOD
    current["participation_probability"] = current[
        "participation_lightgbm"
    ].clip(1e-6, 1 - 1e-6)
    current["publication_status"] = "shadow"
    current_columns = [
        "lock_version",
        "model_run_id",
        "player_key",
        "gsis_id",
        "display_name",
        "season",
        "position",
        "team",
        "identity_status",
        "expert_ppg_team_game_median",
        "expert_ppg_active_median",
        "conditional_ppg_lasso",
        "conditional_ppg_random_forest",
        "conditional_ppg_lightgbm",
        PRIMARY_METHOD,
        "conditional_ppg_log_lasso_tree_blend",
        "conditional_ppg_history_gap_no_history_route",
        "conditional_ppg_projection_history_router",
        "projection_router_source",
        "conditional_ppg_qb_style_wrte_route",
        "conditional_ppg_shadow_unclipped",
        "conditional_ppg_shadow",
        "conditional_ppg_shadow_method",
        "participation_logistic",
        "participation_lightgbm",
        "participation_probability",
        "publication_status",
    ]
    current = current.loc[:, current_columns]

    for frame in (
        specifications,
        selected_hyperparameters,
        evaluation,
        scores,
        comparisons,
        calibration_bins,
        interval_rows,
        interval_summary,
        router_decisions,
        template_handoff,
        template_audit,
    ):
        if "model_run_id" not in frame:
            frame.insert(0, "model_run_id", run_id)
        if "lock_version" not in frame:
            frame.insert(0, "lock_version", ACTIVE_LOCK_VERSION)

    run_table = pd.DataFrame(
        [
            {
                "lock_version": ACTIVE_LOCK_VERSION,
                "model_run_id": run_id,
                "feature_run_id": feature_run_id,
                "created_at_utc": utc_now(),
                "validation_start_season": min(OUTER_SEASONS),
                "validation_end_season": max(OUTER_SEASONS),
                "current_shadow_season": CURRENT_SEASON,
                "conditional_ppg_rows": int(ppg_eligible.sum()),
                "participation_rows": int(participation_eligible.sum()),
                "current_candidate_rows": len(current),
                "current_ppg_rows": int(
                    current["conditional_ppg_shadow"].notna().sum()
                ),
                "status": "complete_shadow",
                "metadata_json": json.dumps(
                    locked_metadata(
                        ACTIVE_SCORING_OBJECTIVE,
                        ACTIVE_LOCK_VERSION,
                    ),
                    sort_keys=True,
                ),
            }
        ]
    )

    tables = {
        "locked_candidate_runs": run_table,
        "locked_candidate_specifications": specifications,
        "locked_selected_hyperparameters": selected_hyperparameters,
        "locked_whole_season_predictions": evaluation,
        "locked_whole_season_scores": scores,
        "locked_model_comparisons": comparisons,
        "locked_calibration_bins": calibration_bins,
        "locked_residual_intervals": interval_rows,
        "locked_residual_interval_summary": interval_summary,
        "locked_router_decisions": router_decisions,
        "locked_2026_shadow_predictions": current,
        "locked_template_handoff": template_handoff,
        "locked_template_handoff_audit": template_audit,
    }
    publish_tables_atomic(ACTIVE_OUTPUT_DB_PATH, tables)

    for table_name, frame in tables.items():
        frame.to_csv(
            ACTIVE_RESULTS_DIR / f"{table_name}.csv",
            index=False,
        )
    summary = _summary_markdown(
        run_id,
        feature_run_id,
        comparisons,
        scores,
        interval_summary,
        template_audit,
        current,
    )
    (ACTIVE_RESULTS_DIR / "summary.md").write_text(
        summary,
        encoding="utf-8",
    )
    (ACTIVE_RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                **locked_metadata(
                    ACTIVE_SCORING_OBJECTIVE,
                    ACTIVE_LOCK_VERSION,
                ),
                "model_run_id": run_id,
                "feature_run_id": feature_run_id,
                "output_database": str(
                    ACTIVE_OUTPUT_DB_PATH.resolve()
                ),
                "tables": {
                    name: len(frame) for name, frame in tables.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "current_rows": len(current),
                "current_ppg_rows": int(
                    current["conditional_ppg_shadow"].notna().sum()
                ),
                "results": str(ACTIVE_RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
