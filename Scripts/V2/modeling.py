"""Leakage-safe datasets, compact pipelines, OOF predictions, and metrics for V2."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from Scripts.V2.config import POSITIONS, REPO_ROOT
from Scripts.V2.contracts import (
    MODEL_FOLD_COLUMNS,
    MODEL_HYPERPARAMETER_COLUMNS,
    MODEL_OOF_COLUMNS,
    MODEL_SCORE_COLUMNS,
    MODEL_SLICE_COLUMNS,
    MODEL_SPECIFICATION_COLUMNS,
    align_columns,
    require_columns,
)


CONDITIONAL_PPG_TARGET = "conditional_ppg"
PARTICIPATION_TARGET = "participation"
POSITION_FEATURES = tuple(f"position_{position}" for position in POSITIONS)
CALIBRATION_FEATURES = tuple(
    f"expert_ppg_x_{position}" for position in POSITIONS
) + tuple(
    f"expert_active_ppg_x_{position}" for position in POSITIONS
) + POSITION_FEATURES

RESIDUAL_COMPACT_FEATURES = (
    "expert_ppg_team_game_median",
    "expert_ppg_active_median",
    "expert_ppg_team_game_std",
    "projection_provider_count",
    "proj_games",
    "projection_adp_percentile_diff",
    "age",
    "year_exp",
    "is_rookie",
    "draft_pick_log",
    "career_weighted_ppg",
    "prior_year_ppg_residual",
    "prior_3year_ppg_std",
    "projected_rush_point_share",
    "projected_receiving_point_share",
    "consensus_room_share",
    "room_gap_to_leader_median",
    "team_qb1_ppg",
    "team_changed_from_prior_candidate",
)

PARTICIPATION_COMPACT_FEATURES = (
    "expert_ppg_team_game_median",
    "projection_provider_count",
    "proj_games",
    "adp_median",
    "age",
    "year_exp",
    "is_rookie",
    "draft_pick_log",
    "career_observed_seasons",
    "prior_year_appeared",
    "prior_year_opportunity_games",
    "team_changed_from_prior_candidate",
)

PPG_METRICS = ("rmse", "mae", "bias", "spearman")
PARTICIPATION_METRICS = (
    "brier",
    "log_loss",
    "calibration_bias",
    "roc_auc",
)


@dataclass(frozen=True)
class ModelSpec:
    target_name: str
    model_name: str
    model_family: str
    prediction_kind: str
    feature_set: str
    pipeline_variant: str
    model_piece: str | None
    parameters: Mapping[str, Sequence[object]]
    search_iterations: int


class ModelColumnSelector(BaseEstimator, TransformerMixin):
    """Keep split-control columns available to CV but out of fitted models."""

    def __init__(self, columns: Sequence[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> "ModelColumnSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, list(self.columns)]


def _ridge_parameters(extra: Mapping[str, Sequence[object]] | None = None):
    parameters: dict[str, Sequence[object]] = {
        "ridge__alpha": (1.0, 10.0, 100.0),
    }
    if extra:
        parameters.update(extra)
    return parameters


def _logistic_parameters(
    extra: Mapping[str, Sequence[object]] | None = None,
):
    parameters: dict[str, Sequence[object]] = {
        "lr_c__C": (0.1, 1.0, 10.0),
    }
    if extra:
        parameters.update(extra)
    return parameters


def _lgbm_parameters(step: str) -> dict[str, Sequence[object]]:
    return {
        f"{step}__n_estimators": (100, 200),
        f"{step}__learning_rate": (0.03, 0.05),
        f"{step}__num_leaves": (7, 15),
        f"{step}__max_depth": (3, 4),
        f"{step}__min_child_samples": (20, 40),
        f"{step}__reg_lambda": (1.0, 5.0),
        f"{step}__subsample": (0.8,),
        f"{step}__colsample_bytree": (0.8,),
    }


def initial_model_specs(search_iterations: int = 4) -> tuple[ModelSpec, ...]:
    """Return the intentionally small M4A comparison surface."""
    return (
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "expert_team_game_consensus",
            "baseline",
            "team_game_baseline",
            "team_game_consensus",
            "none",
            None,
            {},
            0,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "expert_consensus_hybrid",
            "baseline",
            "baseline",
            "consensus",
            "none",
            None,
            {},
            0,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "consensus_recalibrated_ridge",
            "ridge",
            "direct",
            "calibration",
            "raw",
            "ridge",
            _ridge_parameters(),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "residual_ridge_compact",
            "ridge",
            "residual",
            "compact",
            "raw",
            "ridge",
            _ridge_parameters(),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "residual_ridge_full",
            "ridge",
            "residual",
            "full_manifest",
            "raw",
            "ridge",
            _ridge_parameters(),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "residual_ridge_kbest",
            "ridge",
            "residual",
            "full_manifest",
            "kbest",
            "ridge",
            _ridge_parameters({"k_best__k": (10, 15, 20)}),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "residual_ridge_pca",
            "ridge",
            "residual",
            "full_manifest",
            "pca",
            "ridge",
            _ridge_parameters({"pca__n_components": (8, 12, 16)}),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "residual_ridge_agg",
            "ridge",
            "residual",
            "full_manifest",
            "agglomeration",
            "ridge",
            _ridge_parameters(
                {"agglomeration__n_clusters": (8, 12, 16)}
            ),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "residual_lgbm_shallow",
            "lightgbm",
            "residual",
            "full_manifest",
            "raw",
            "lgbm",
            _lgbm_parameters("lgbm"),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "direct_ridge_full",
            "ridge",
            "direct",
            "full_manifest",
            "raw",
            "ridge",
            _ridge_parameters(),
            search_iterations,
        ),
        ModelSpec(
            CONDITIONAL_PPG_TARGET,
            "direct_lgbm_shallow",
            "lightgbm",
            "direct",
            "full_manifest",
            "raw",
            "lgbm",
            _lgbm_parameters("lgbm"),
            search_iterations,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "prior_position_rate",
            "baseline",
            "baseline",
            "prior_rate",
            "none",
            None,
            {},
            0,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "participation_logistic_compact",
            "logistic",
            "probability",
            "compact",
            "raw",
            "lr_c",
            _logistic_parameters(),
            search_iterations,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "participation_logistic_full",
            "logistic",
            "probability",
            "full_manifest",
            "raw",
            "lr_c",
            _logistic_parameters(),
            search_iterations,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "participation_logistic_kbest",
            "logistic",
            "probability",
            "full_manifest",
            "kbest",
            "lr_c",
            _logistic_parameters({"k_best_c__k": (8, 12, 16)}),
            search_iterations,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "participation_logistic_pca",
            "logistic",
            "probability",
            "full_manifest",
            "pca",
            "lr_c",
            _logistic_parameters({"pca__n_components": (6, 10, 14)}),
            search_iterations,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "participation_logistic_agg",
            "logistic",
            "probability",
            "full_manifest",
            "agglomeration",
            "lr_c",
            _logistic_parameters(
                {"agglomeration__n_clusters": (6, 10, 14)}
            ),
            search_iterations,
        ),
        ModelSpec(
            PARTICIPATION_TARGET,
            "participation_lgbm_shallow",
            "lightgbm",
            "probability",
            "full_manifest",
            "raw",
            "lgbm_c",
            _lgbm_parameters("lgbm_c"),
            search_iterations,
        ),
    )


def add_modeling_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    baseline = pd.to_numeric(
        frame["expert_ppg_team_game_median"], errors="coerce"
    )
    active_baseline = pd.to_numeric(
        frame["expert_ppg_active_median"], errors="coerce"
    )
    for position in POSITIONS:
        indicator = frame["position"].eq(position).astype(float)
        frame[f"position_{position}"] = indicator
        frame[f"expert_ppg_x_{position}"] = baseline * indicator
        frame[f"expert_active_ppg_x_{position}"] = (
            active_baseline * indicator
        )
    return frame


def build_feature_sets(
    manifests: pd.DataFrame,
) -> dict[str, dict[str, tuple[str, ...]]]:
    require_columns(
        manifests,
        ("manifest_name", "feature_name"),
        "feature_manifests",
    )

    def manifest(name: str) -> tuple[str, ...]:
        values = (
            manifests.loc[manifests["manifest_name"].eq(name), "feature_name"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not values:
            raise ValueError(f"Missing or empty feature manifest: {name}")
        return tuple(values)

    residual_full = manifest("residual_candidate_v1")
    participation_full = manifest("participation_candidate_v1")
    if not set(RESIDUAL_COMPACT_FEATURES).issubset(residual_full):
        raise ValueError("Residual compact feature set is outside its manifest")
    if not set(PARTICIPATION_COMPACT_FEATURES).issubset(participation_full):
        raise ValueError(
            "Participation compact feature set is outside its manifest"
        )
    return {
        CONDITIONAL_PPG_TARGET: {
            "team_game_consensus": ("expert_ppg_team_game_median",),
            "consensus": (
                "expert_ppg_team_game_median",
                "expert_ppg_active_median",
            ),
            "calibration": CALIBRATION_FEATURES,
            "compact": RESIDUAL_COMPACT_FEATURES + POSITION_FEATURES,
            "full_manifest": residual_full + POSITION_FEATURES,
        },
        PARTICIPATION_TARGET: {
            "prior_rate": POSITION_FEATURES,
            "compact": PARTICIPATION_COMPACT_FEATURES + POSITION_FEATURES,
            "full_manifest": participation_full + POSITION_FEATURES,
        },
    }


def build_target_frames(
    features: pd.DataFrame,
    validation_end_season: int,
) -> dict[str, pd.DataFrame]:
    require_columns(
        features,
        (
            "player_key",
            "season",
            "position",
            "team",
            "outcome_complete",
            "active_target_available",
            "conditional_ppg_training_eligible",
            "conditional_ppg",
            "appeared",
            "expert_ppg_team_game_median",
            "expert_ppg_active_median",
        ),
        "player_season_features",
    )
    frame = add_modeling_features(features)
    frame = frame[
        frame["season"].le(validation_end_season)
        & frame["position"].isin(POSITIONS)
    ].copy()
    ppg = frame[
        frame["conditional_ppg_training_eligible"].eq(1)
        & frame["conditional_ppg"].notna()
        & frame["expert_ppg_team_game_median"].notna()
    ].copy()
    participation = frame[
        frame["active_target_available"].eq(1) & frame["appeared"].notna()
    ].copy()
    for target in (ppg, participation):
        target.sort_values(["season", "player_key"], inplace=True)
        target.reset_index(drop=True, inplace=True)
    ppg["actual_target"] = pd.to_numeric(
        ppg["conditional_ppg"], errors="coerce"
    )
    ppg["team_game_prediction"] = pd.to_numeric(
        ppg["expert_ppg_team_game_median"], errors="coerce"
    )
    ppg["baseline_prediction"] = pd.to_numeric(
        ppg["expert_ppg_active_median"], errors="coerce"
    ).combine_first(ppg["team_game_prediction"])
    participation["actual_target"] = pd.to_numeric(
        participation["appeared"], errors="coerce"
    )
    participation["baseline_prediction"] = rolling_position_rate(
        participation
    )
    participation["team_game_prediction"] = np.nan
    return {
        CONDITIONAL_PPG_TARGET: ppg,
        PARTICIPATION_TARGET: participation,
    }


def rolling_position_rate(
    frame: pd.DataFrame,
    prior_strength: float = 25.0,
) -> pd.Series:
    """Return a position rate using only seasons earlier than each row."""
    require_columns(frame, ("season", "position", "appeared"), "participation")
    predictions = pd.Series(index=frame.index, dtype=float)
    seasons = sorted(pd.to_numeric(frame["season"]).dropna().unique())
    for season in seasons:
        prior = frame[frame["season"].lt(season)]
        current = frame["season"].eq(season)
        if prior.empty:
            predictions.loc[current] = 0.5
            continue
        global_rate = float(pd.to_numeric(prior["appeared"]).mean())
        position_stats = prior.groupby("position")["appeared"].agg(
            ["sum", "count"]
        )
        for position in frame.loc[current, "position"].unique():
            if position in position_stats.index:
                successes = float(position_stats.loc[position, "sum"])
                count = float(position_stats.loc[position, "count"])
                rate = (
                    successes + prior_strength * global_rate
                ) / (count + prior_strength)
            else:
                rate = global_rate
            predictions.loc[current & frame["position"].eq(position)] = rate
    return predictions.clip(1e-6, 1 - 1e-6)


def make_fold_assignments(
    frame: pd.DataFrame,
    target_name: str,
    run_id: str,
    validation_start_season: int,
    n_splits: int,
    random_seed: int,
) -> pd.DataFrame:
    require_columns(
        frame,
        ("player_key", "season", "position"),
        f"{target_name}_target_frame",
    )
    validation = frame[frame["season"].ge(validation_start_season)].copy()
    if validation.empty:
        raise ValueError(f"No validation rows for target {target_name}")
    season_counts = validation.groupby("season").size()
    if season_counts.min() < n_splits:
        raise ValueError(
            f"{target_name} has a season with fewer than {n_splits} rows"
        )
    validation.reset_index(drop=True, inplace=True)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    validation["fold"] = -1
    for fold, (_, hold_index) in enumerate(
        splitter.split(validation, validation["season"])
    ):
        validation.loc[hold_index, "fold"] = fold
    validation["training_start_season"] = int(frame["season"].min())
    validation["training_through_season"] = validation["season"].astype(int) - 1
    validation["run_id"] = run_id
    validation["target_name"] = target_name
    assignments = align_columns(
        validation,
        MODEL_FOLD_COLUMNS,
        "model_fold_assignments",
    )
    validate_fold_assignments(
        assignments,
        validation_start_season=validation_start_season,
        n_splits=n_splits,
    )
    return assignments


def validate_fold_assignments(
    assignments: pd.DataFrame,
    validation_start_season: int,
    n_splits: int,
) -> None:
    if assignments.duplicated(["target_name", "player_key", "season"]).any():
        raise ValueError("A target player-season was assigned to multiple folds")
    if not assignments["fold"].between(0, n_splits - 1).all():
        raise ValueError("Fold assignment is outside the configured range")
    if assignments["season"].lt(validation_start_season).any():
        raise ValueError("Pre-validation rows were assigned to holdout folds")
    if (
        assignments["training_through_season"]
        >= assignments["season"]
    ).any():
        raise ValueError("An OOF row can train on its target season")
    folds_by_season = assignments.groupby("season")["fold"].nunique()
    if not folds_by_season.eq(n_splits).all():
        raise ValueError("Each validation season must appear in every fold")


def _load_scikit_model():
    warning_filter = (
        "ignore:pkg_resources is deprecated as an API:UserWarning"
    )
    existing_filters = os.environ.get("PYTHONWARNINGS", "")
    if warning_filter not in existing_filters:
        os.environ["PYTHONWARNINGS"] = ",".join(
            value for value in (existing_filters, warning_filter) if value
        )
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    try:
        return importlib.import_module("skmodel.run_models").SciKitModel
    except ModuleNotFoundError:
        sibling = REPO_ROOT.parent / "Scikit_Model"
        if not sibling.exists():
            raise ModuleNotFoundError(
                "SciKitModel is not installed and the sibling Scikit_Model "
                f"repository was not found at {sibling}"
            )
        sys.path.insert(0, str(sibling))
        return importlib.import_module("skmodel.run_models").SciKitModel


def _feature_columns(
    spec: ModelSpec,
    feature_sets: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    try:
        return tuple(feature_sets[spec.target_name][spec.feature_set])
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature set {spec.target_name}/{spec.feature_set}"
        ) from exc


def _build_pipeline(
    skm: object,
    spec: ModelSpec,
    feature_columns: Sequence[str],
):
    steps: list[tuple[str, object]] = [
        ("model_columns", ModelColumnSelector(feature_columns)),
        skm.piece("impute"),
    ]
    scale_required = spec.model_family in {
        "ridge",
        "lasso",
        "elastic_net",
        "logistic",
        "knn",
    }
    if scale_required:
        steps.append(skm.piece("std_scale"))
    if spec.pipeline_variant == "kbest":
        selector = (
            "k_best_c"
            if spec.target_name == PARTICIPATION_TARGET
            else "k_best"
        )
        steps.append(skm.piece(selector))
    elif spec.pipeline_variant == "pca":
        steps.append(skm.piece("pca"))
    elif spec.pipeline_variant == "agglomeration":
        steps.append(skm.piece("agglomeration"))
    elif spec.pipeline_variant != "raw":
        raise ValueError(f"Unsupported pipeline variant: {spec.pipeline_variant}")
    if spec.model_piece is None:
        raise ValueError("A fitted model specification needs a model piece")
    steps.append(skm.piece(spec.model_piece))
    pipe = skm.model_pipe(steps)
    pipe.set_params(
        impute__strategy="median",
        impute__add_indicator=True,
        impute__keep_empty_features=True,
    )
    if spec.model_family == "ridge":
        pipe.set_params(ridge__max_iter=5000)
    elif spec.model_family == "lasso":
        pipe.set_params(lasso__max_iter=20_000, lasso__tol=1e-6)
    elif spec.model_family == "elastic_net":
        pipe.set_params(enet__max_iter=20_000, enet__tol=1e-6)
    elif spec.model_family == "logistic":
        pipe.set_params(lr_c__max_iter=3000, lr_c__solver="lbfgs")
    elif spec.model_family == "lightgbm":
        pipe.set_params(**{f"{spec.model_piece}__verbosity": -1})
    return pipe


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def parameters_json(parameters: Mapping[str, object]) -> str:
    return json.dumps(
        {str(key): _json_value(value) for key, value in parameters.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


def specification_row(
    spec: ModelSpec,
    feature_columns: Sequence[str],
    run_id: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "target_name": spec.target_name,
        "model_name": spec.model_name,
        "model_family": spec.model_family,
        "prediction_kind": spec.prediction_kind,
        "feature_set": spec.feature_set,
        "pipeline_variant": spec.pipeline_variant,
        "feature_count": len(feature_columns),
        "feature_names_json": json.dumps(list(feature_columns)),
        "hyperparameters_json": parameters_json(spec.parameters),
        "search_iterations": spec.search_iterations,
        "status": (
            "diagnostic"
            if spec.prediction_kind == "team_game_baseline"
            else (
                "incumbent"
                if spec.model_family == "baseline"
                else "challenger"
            )
        ),
    }


def _hyperparameter_rows(
    scores: pd.DataFrame,
    spec: ModelSpec,
    run_id: str,
    n_splits: int,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=MODEL_HYPERPARAMETER_COLUMNS)
    if len(scores) % n_splits:
        raise ValueError(
            f"Unexpected hyperparameter score rows for {spec.model_name}"
        )
    trials_per_fold = len(scores) // n_splits
    rows: list[dict[str, object]] = []
    for fold in range(n_splits):
        start = fold * trials_per_fold
        fold_scores = scores.iloc[start : start + trials_per_fold].reset_index(
            drop=True
        )
        selected_trial = int(
            pd.to_numeric(fold_scores["scores"], errors="coerce").idxmin()
        )
        for trial, row in fold_scores.iterrows():
            params = {
                column: row[column]
                for column in fold_scores.columns
                if column != "scores"
            }
            rows.append(
                {
                    "run_id": run_id,
                    "target_name": spec.target_name,
                    "model_name": spec.model_name,
                    "fold": fold,
                    "trial": int(trial),
                    "parameters_json": parameters_json(params),
                    "validation_score": float(row["scores"]),
                    "selected": int(trial == selected_trial),
                }
            )
    return align_columns(
        pd.DataFrame(rows),
        MODEL_HYPERPARAMETER_COLUMNS,
        "model_hyperparameter_results",
    )


def _base_oof_rows(
    target_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    spec: ModelSpec,
    run_id: str,
    feature_run_id: str,
) -> pd.DataFrame:
    metadata_columns = [
        "player_key",
        "season",
        "position",
        "team",
        "actual_target",
        "baseline_prediction",
        "team_game_prediction",
        "opportunity_games",
        "has_prior_outcome",
        "is_rookie",
        "year_exp",
        "projection_provider_count",
    ]
    frame = assignments.merge(
        target_frame.loc[:, metadata_columns],
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    frame["run_id"] = run_id
    frame["feature_run_id"] = feature_run_id
    frame["model_name"] = spec.model_name
    frame["model_family"] = spec.model_family
    frame["prediction_kind"] = spec.prediction_kind
    frame["feature_set"] = spec.feature_set
    frame["pipeline_variant"] = spec.pipeline_variant
    frame["actual"] = frame["actual_target"]
    return frame


def _baseline_oof(
    target_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    spec: ModelSpec,
    run_id: str,
    feature_run_id: str,
) -> pd.DataFrame:
    frame = _base_oof_rows(
        target_frame, assignments, spec, run_id, feature_run_id
    )
    if spec.prediction_kind == "team_game_baseline":
        frame["model_prediction"] = frame["team_game_prediction"]
        frame["final_prediction"] = frame["team_game_prediction"]
    else:
        frame["model_prediction"] = frame["baseline_prediction"]
        frame["final_prediction"] = frame["baseline_prediction"]
    if spec.target_name == CONDITIONAL_PPG_TARGET:
        frame["residual_actual"] = (
            frame["actual"] - frame["baseline_prediction"]
        )
        frame["residual_prediction"] = (
            frame["final_prediction"] - frame["baseline_prediction"]
        )
    else:
        frame["residual_actual"] = np.nan
        frame["residual_prediction"] = np.nan
    return align_columns(frame, MODEL_OOF_COLUMNS, "model_oof_predictions")


def run_model_spec(
    target_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    spec: ModelSpec,
    feature_columns: Sequence[str],
    run_id: str,
    feature_run_id: str,
    validation_start_season: int,
    n_splits: int,
    random_seed: int,
    quiet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.model_family == "baseline":
        return (
            _baseline_oof(
                target_frame,
                assignments,
                spec,
                run_id,
                feature_run_id,
            ),
            pd.DataFrame(columns=MODEL_HYPERPARAMETER_COLUMNS),
        )
    missing = [
        feature for feature in feature_columns if feature not in target_frame
    ]
    if missing:
        raise ValueError(f"{spec.model_name} is missing features: {missing}")
    model_data = target_frame.copy()
    model_data["player"] = model_data["player_key"]
    model_data["week"] = 1
    model_data["year"] = model_data["season"].astype(int)
    model_data["game_date"] = model_data["season"].astype(int)
    if spec.prediction_kind == "residual":
        model_data["y_act"] = (
            model_data["actual_target"]
            - model_data["baseline_prediction"]
        )
    else:
        model_data["y_act"] = model_data["actual_target"]
    model_data.reset_index(drop=True, inplace=True)
    numeric_columns = list(feature_columns) + ["game_date"]
    X = model_data.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    y = pd.to_numeric(model_data["y_act"], errors="coerce")
    if y.isna().any():
        raise ValueError(f"{spec.model_name} target contains nulls")

    SciKitModel = _load_scikit_model()
    objective = (
        "class"
        if spec.target_name == PARTICIPATION_TARGET
        else "reg"
    )
    skm = SciKitModel(model_data, model_obj=objective, set_seed=random_seed)
    pipe = _build_pipeline(skm, spec, feature_columns)
    sink = io.StringIO()
    context = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with context:
        _, oof_data, parameter_scores, _ = skm.time_series_cv(
            pipe,
            X,
            y,
            dict(spec.parameters),
            col_split="game_date",
            time_split=validation_start_season,
            n_splits=n_splits,
            n_iter=spec.search_iterations,
            bayes_rand="rand",
            proba=spec.target_name == PARTICIPATION_TARGET,
            random_seed=random_seed,
        )
    predicted = oof_data["full_hold"].rename(
        columns={"player": "player_key", "year": "season"}
    )
    predicted = predicted.loc[:, ["player_key", "season", "pred"]]
    if predicted.duplicated(["player_key", "season"]).any():
        raise ValueError(f"{spec.model_name} produced duplicate OOF rows")
    frame = _base_oof_rows(
        target_frame, assignments, spec, run_id, feature_run_id
    )
    frame = frame.merge(
        predicted,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    if frame["pred"].isna().any():
        raise ValueError(f"{spec.model_name} did not predict every OOF row")
    frame["model_prediction"] = frame["pred"]
    if spec.prediction_kind == "residual":
        frame["residual_actual"] = (
            frame["actual"] - frame["baseline_prediction"]
        )
        frame["residual_prediction"] = frame["model_prediction"]
        frame["final_prediction"] = (
            frame["baseline_prediction"] + frame["model_prediction"]
        )
    elif spec.target_name == CONDITIONAL_PPG_TARGET:
        frame["residual_actual"] = (
            frame["actual"] - frame["baseline_prediction"]
        )
        frame["final_prediction"] = frame["model_prediction"]
        frame["residual_prediction"] = (
            frame["final_prediction"] - frame["baseline_prediction"]
        )
    else:
        frame["residual_actual"] = np.nan
        frame["residual_prediction"] = np.nan
        frame["final_prediction"] = frame["model_prediction"].clip(
            1e-6, 1 - 1e-6
        )
        frame["model_prediction"] = frame["final_prediction"]
    oof = align_columns(frame, MODEL_OOF_COLUMNS, "model_oof_predictions")
    parameters = _hyperparameter_rows(
        parameter_scores,
        spec,
        run_id,
        n_splits,
    )
    return oof, parameters


def _metric_value(metric: str, actual: pd.Series, prediction: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    prediction_values = pd.to_numeric(
        prediction, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(prediction_values)
    actual_values = actual_values[valid]
    prediction_values = prediction_values[valid]
    if not len(actual_values):
        return np.nan
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(actual_values, prediction_values)))
    if metric == "mae":
        return float(mean_absolute_error(actual_values, prediction_values))
    if metric == "bias":
        return float(np.mean(prediction_values - actual_values))
    if metric == "spearman":
        return float(
            pd.Series(actual_values).corr(
                pd.Series(prediction_values), method="spearman"
            )
        )
    clipped = np.clip(prediction_values, 1e-6, 1 - 1e-6)
    if metric == "brier":
        return float(brier_score_loss(actual_values, clipped))
    if metric == "log_loss":
        return float(log_loss(actual_values, clipped, labels=[0, 1]))
    if metric == "calibration_bias":
        return float(np.mean(clipped - actual_values))
    if metric == "roc_auc":
        if len(np.unique(actual_values)) < 2:
            return np.nan
        return float(roc_auc_score(actual_values, clipped))
    raise ValueError(f"Unsupported metric: {metric}")


def build_score_summary(oof: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target_name, model_name), group in oof.groupby(
        ["target_name", "model_name"], sort=True
    ):
        metrics = (
            PPG_METRICS
            if target_name == CONDITIONAL_PPG_TARGET
            else PARTICIPATION_METRICS
        )
        for aggregation in ("pooled_oof", "season_mean"):
            for metric in metrics:
                if aggregation == "pooled_oof":
                    value = _metric_value(
                        metric, group["actual"], group["final_prediction"]
                    )
                    baseline_value = _metric_value(
                        metric, group["actual"], group["baseline_prediction"]
                    )
                else:
                    season_values = []
                    season_baselines = []
                    for _, season in group.groupby("season"):
                        season_values.append(
                            _metric_value(
                                metric,
                                season["actual"],
                                season["final_prediction"],
                            )
                        )
                        season_baselines.append(
                            _metric_value(
                                metric,
                                season["actual"],
                                season["baseline_prediction"],
                            )
                        )
                    value = float(np.nanmean(season_values))
                    baseline_value = float(np.nanmean(season_baselines))
                rows.append(
                    {
                        "run_id": run_id,
                        "target_name": target_name,
                        "model_name": model_name,
                        "aggregation": aggregation,
                        "metric": metric,
                        "n_rows": len(group),
                        "n_seasons": group["season"].nunique(),
                        "value": value,
                        "baseline_value": baseline_value,
                        "delta": value - baseline_value,
                    }
                )
    return align_columns(
        pd.DataFrame(rows), MODEL_SCORE_COLUMNS, "model_score_summary"
    )


def _history_depth(frame: pd.DataFrame) -> pd.Series:
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    prior = pd.to_numeric(
        frame["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    result = pd.Series("other_no_history", index=frame.index, dtype=object)
    result.loc[rookie] = "rookie"
    result.loc[~rookie & year_exp.eq(1)] = "second_year"
    result.loc[~rookie & year_exp.ge(2) & prior] = "veteran_with_history"
    result.loc[year_exp.isna()] = "unknown_experience"
    return result


def _provider_depth(frame: pd.DataFrame) -> pd.Series:
    count = pd.to_numeric(
        frame["projection_provider_count"], errors="coerce"
    ).fillna(0)
    result = pd.Series("none", index=frame.index, dtype=object)
    result.loc[count.eq(1)] = "one"
    result.loc[count.eq(2)] = "two"
    result.loc[count.ge(3)] = "three_plus"
    return result


def _provider_era(frame: pd.DataFrame) -> pd.Series:
    season = pd.to_numeric(frame["season"], errors="coerce")
    result = pd.Series("other", index=frame.index, dtype=object)
    result.loc[season.between(2017, 2019)] = "2017_2019"
    result.loc[season.between(2020, 2022)] = "2020_2022"
    result.loc[season.ge(2023)] = "2023_plus"
    return result


def build_slice_summary(
    oof: pd.DataFrame,
    run_id: str,
    min_slice_rows: int = 25,
) -> pd.DataFrame:
    frame = oof.copy()
    frame["history_depth"] = _history_depth(frame)
    frame["provider_depth"] = _provider_depth(frame)
    frame["provider_era"] = _provider_era(frame)
    slice_columns = {
        "position": "position",
        "season": "season",
        "history_depth": "history_depth",
        "provider_depth": "provider_depth",
        "provider_era": "provider_era",
    }
    rows: list[dict[str, object]] = []
    for (target_name, model_name), model in frame.groupby(
        ["target_name", "model_name"], sort=True
    ):
        metrics = (
            PPG_METRICS
            if target_name == CONDITIONAL_PPG_TARGET
            else PARTICIPATION_METRICS
        )
        for slice_type, column in slice_columns.items():
            for slice_value, group in model.groupby(column, dropna=False):
                if len(group) < min_slice_rows:
                    continue
                for metric in metrics:
                    value = _metric_value(
                        metric, group["actual"], group["final_prediction"]
                    )
                    baseline_value = _metric_value(
                        metric, group["actual"], group["baseline_prediction"]
                    )
                    rows.append(
                        {
                            "run_id": run_id,
                            "target_name": target_name,
                            "model_name": model_name,
                            "slice_type": slice_type,
                            "slice_value": str(slice_value),
                            "metric": metric,
                            "n_rows": len(group),
                            "n_seasons": group["season"].nunique(),
                            "value": value,
                            "baseline_value": baseline_value,
                            "delta": value - baseline_value,
                        }
                    )
    return align_columns(
        pd.DataFrame(rows), MODEL_SLICE_COLUMNS, "model_slice_summary"
    )


def validate_oof_predictions(
    oof: pd.DataFrame,
    assignments: pd.DataFrame,
    specs: Sequence[ModelSpec],
) -> None:
    expected = {
        target: len(group)
        for target, group in assignments.groupby("target_name")
    }
    spec_counts = pd.Series([spec.target_name for spec in specs]).value_counts()
    for target, rows in expected.items():
        observed = len(oof[oof["target_name"].eq(target)])
        if observed != rows * int(spec_counts[target]):
            raise ValueError(
                f"{target} OOF row count {observed} did not match expected "
                f"{rows * int(spec_counts[target])}"
            )
    if oof.duplicated(
        ["target_name", "model_name", "player_key", "season"]
    ).any():
        raise ValueError("OOF predictions are not unique by model/player-season")
    if oof["final_prediction"].isna().any():
        raise ValueError("OOF predictions contain null final predictions")
    if (oof["training_through_season"] >= oof["season"]).any():
        raise ValueError("OOF provenance permits target-season training")
    participation = oof["target_name"].eq(PARTICIPATION_TARGET)
    if not oof.loc[participation, "final_prediction"].between(0, 1).all():
        raise ValueError("Participation predictions must be probabilities")
