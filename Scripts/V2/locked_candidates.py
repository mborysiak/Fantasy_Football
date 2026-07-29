"""Versioned feature and model lock for the 2026 V2 shadow projection."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import Mapping, Sequence

import pandas as pd

from Scripts.V2.modeling import POSITION_FEATURES


LOCK_VERSION = "v2_conditional_ppg_2026_candidate_v1"
LOCKED_SCORING_OBJECTIVE = "dk"
LOCKED_VALIDATION_SEASONS = tuple(range(2017, 2026))
LOCKED_INNER_VALIDATION_START = 2013
LOCKED_RANDOM_SEED = 1234
LOCKED_BLEND_WEIGHTS = {
    "conditional_ppg_lasso": 1.0 / 3.0,
    "conditional_ppg_random_forest": 1.0 / 3.0,
    "conditional_ppg_lightgbm": 1.0 / 3.0,
}


RESIDUAL_CANDIDATE_FEATURES = (
    "adp_median",
    "age",
    "career_observed_seasons",
    "career_weighted_ppg",
    "consensus_room_share",
    "draft_pick_log",
    "expert_points_iqr",
    "expert_ppg_active_median",
    "expert_ppg_team_game_median",
    "expert_ppg_team_game_std",
    "is_rookie",
    "prior_3year_ppg_std",
    "prior_3year_weighted_ppg",
    "prior_year_ppg",
    "prior_year_ppg_residual",
    "proj_games",
    "proj_pass_attempts",
    "proj_rush_attempts",
    "proj_targets",
    "projected_pass_point_share",
    "projected_receiving_point_share",
    "projected_rush_point_share",
    "projection_adp_percentile_diff",
    "projection_provider_count",
    "room_gap_to_leader_median",
    "room_hhi_median",
    "room_share_median",
    "seasons_since_observed",
    "team_changed_from_prior_candidate",
    "team_qb1_ppg",
    "year_exp",
)

PROJECTION_TRAJECTORY_FEATURES = (
    "projection_trajectory_change_1year",
    "projection_trajectory_change_3year",
    "projection_trajectory_prior_3year_count",
    "projection_trajectory_prior_3year_std",
    "projection_trajectory_prior_year_available",
)

PARTICIPATION_CANDIDATE_FEATURES = (
    "adp_median",
    "adp_source_count",
    "age",
    "career_observed_seasons",
    "career_opportunity_games",
    "career_useful_seasons",
    "draft_pick_log",
    "experience_known",
    "expert_ppg_team_game_median",
    "is_rookie",
    "last_observed_opportunity_games",
    "prior_year_appeared",
    "prior_year_candidate",
    "prior_year_opportunity_games",
    "proj_games",
    "projection_provider_count",
    "seasons_since_observed",
    "team_changed_from_prior_candidate",
    "year_exp",
)

PROJECTION_CORE_FEATURES = (
    "expert_ppg_team_game_median",
    "expert_ppg_active_median",
    "expert_ppg_team_game_std",
    "expert_ppg_team_game_iqr",
    "expert_points_iqr",
    "projection_provider_count",
    "configured_projection_provider_count",
    "proj_games",
    "proj_pass_attempts",
    "proj_passing_yards",
    "proj_passing_tds",
    "proj_interceptions",
    "proj_rush_attempts",
    "proj_rushing_yards",
    "proj_rushing_tds",
    "proj_targets",
    "proj_receptions",
    "proj_receiving_yards",
    "proj_receiving_tds",
    "projected_pass_point_share",
    "projected_rush_point_share",
    "projected_receiving_point_share",
)

HISTORY_GAP_REPLACED_FEATURES = frozenset(
    {
        "career_weighted_ppg",
        "prior_year_ppg",
        "prior_year_ppg_residual",
        "prior_3year_weighted_ppg",
        "seasons_since_observed",
    }
)
HISTORY_GAP_COMMON_FEATURES = (
    "history_career_opportunity_games_log",
    "history_prior_year_opportunity_games_log",
    "history_prior_3year_opportunity_games_log",
    "history_prior_year_ppg_available",
    "history_prior_3year_ppg_available",
    "history_prior_year_residual_neutral",
    "history_seasons_since_observed_neutral",
)
HISTORY_GAP_RAW_FEATURES = (
    "history_career_ppg_gap",
    "history_prior_year_ppg_gap",
    "history_prior_3year_ppg_gap",
)

EXPERT_RECALIBRATION_FEATURES = (
    "expert_ppg_team_game_median",
    "expert_ppg_active_median",
    *(f"expert_ppg_x_{position}" for position in ("QB", "RB", "WR", "TE")),
    *(
        f"expert_active_ppg_x_{position}"
        for position in ("QB", "RB", "WR", "TE")
    ),
    *POSITION_FEATURES,
)


def _unique(features: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(features))


PRIMARY_PPG_FEATURES = _unique(
    (*RESIDUAL_CANDIDATE_FEATURES, *PROJECTION_TRAJECTORY_FEATURES, *POSITION_FEATURES)
)
LOG_ADP_LASSO_FEATURES = tuple(
    "adp_log" if feature == "adp_median" else feature
    for feature in PRIMARY_PPG_FEATURES
)
QB_STYLE_PPG_FEATURES = _unique(
    (*PRIMARY_PPG_FEATURES, "team_qb1_rush_point_share")
)
HISTORY_GAP_PPG_FEATURES = _unique(
    (
        *(
            feature
            for feature in RESIDUAL_CANDIDATE_FEATURES
            if feature not in HISTORY_GAP_REPLACED_FEATURES
        ),
        *HISTORY_GAP_COMMON_FEATURES,
        *HISTORY_GAP_RAW_FEATURES,
        *PROJECTION_TRAJECTORY_FEATURES,
        *POSITION_FEATURES,
    )
)
PROJECTION_ONLY_PPG_FEATURES = _unique(
    (*PROJECTION_CORE_FEATURES, *POSITION_FEATURES)
)
PARTICIPATION_FEATURES = _unique(
    (*PARTICIPATION_CANDIDATE_FEATURES, *POSITION_FEATURES)
)

LOCKED_FEATURE_SETS = {
    "expert_recalibration": EXPERT_RECALIBRATION_FEATURES,
    "conditional_ppg_primary": PRIMARY_PPG_FEATURES,
    "conditional_ppg_log_adp_lasso": LOG_ADP_LASSO_FEATURES,
    "conditional_ppg_qb_style": QB_STYLE_PPG_FEATURES,
    "conditional_ppg_history_gap": HISTORY_GAP_PPG_FEATURES,
    "conditional_ppg_projection_only": PROJECTION_ONLY_PPG_FEATURES,
    "participation_primary": PARTICIPATION_FEATURES,
}


def _params(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)


LASSO_GRID = tuple(
    _params(alpha=value)
    for value in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
)
RIDGE_GRID = tuple(_params(alpha=value) for value in (1.0, 10.0, 100.0))
LOGISTIC_GRID = tuple(_params(C=value) for value in (0.1, 1.0, 10.0))
RANDOM_FOREST_GRID = tuple(
    _params(
        n_estimators=250,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=LOCKED_RANDOM_SEED,
        n_jobs=1,
    )
    for max_depth, min_samples_leaf, max_features in product(
        (6, 10), (5, 15), (0.5, 1.0)
    )
)
LIGHTGBM_GRID = (
    _params(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=40,
        reg_lambda=5.0,
    ),
    _params(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=4,
        min_child_samples=40,
        reg_lambda=1.0,
    ),
    _params(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=4,
        min_child_samples=20,
        reg_lambda=5.0,
    ),
    _params(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=3,
        min_child_samples=40,
        reg_lambda=5.0,
    ),
    _params(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        reg_lambda=1.0,
    ),
    _params(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        reg_lambda=5.0,
    ),
    _params(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=3,
        min_child_samples=20,
        reg_lambda=1.0,
    ),
    _params(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=20,
        reg_lambda=1.0,
    ),
)

MODEL_GRIDS = {
    "expert_recalibrated_ridge": RIDGE_GRID,
    "conditional_ppg_lasso": LASSO_GRID,
    "conditional_ppg_random_forest": RANDOM_FOREST_GRID,
    "conditional_ppg_lightgbm": LIGHTGBM_GRID,
    "participation_logistic": LOGISTIC_GRID,
    "participation_lightgbm": LIGHTGBM_GRID,
}


def feature_hash(features: Sequence[str]) -> str:
    payload = json.dumps(list(features), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_feature_lock(
    features: pd.DataFrame,
    manifests: pd.DataFrame,
) -> None:
    """Fail when the active mart no longer matches the reviewed lock."""
    missing = sorted(
        {
            feature
            for feature_set in LOCKED_FEATURE_SETS.values()
            for feature in feature_set
            if feature not in features.columns
        }
    )
    if missing:
        raise ValueError(f"Locked feature columns are missing: {missing}")

    expected_manifests = {
        "residual_candidate_v1": set(RESIDUAL_CANDIDATE_FEATURES),
        "residual_projection_trajectory_challenger_v1": set(
            PROJECTION_TRAJECTORY_FEATURES
        ),
        "participation_candidate_v1": set(PARTICIPATION_CANDIDATE_FEATURES),
    }
    for manifest_name, expected in expected_manifests.items():
        observed = set(
            manifests.loc[
                manifests["manifest_name"].eq(manifest_name),
                "feature_name",
            ]
        )
        if observed != expected:
            added = sorted(observed - expected)
            removed = sorted(expected - observed)
            raise ValueError(
                f"Feature lock mismatch for {manifest_name}; "
                f"added={added}, removed={removed}"
            )


def lock_version_for_scoring(scoring_objective: str) -> str:
    objective = str(scoring_objective).strip().lower()
    if not objective:
        raise ValueError("scoring_objective cannot be empty")
    if objective == LOCKED_SCORING_OBJECTIVE:
        return LOCK_VERSION
    return f"v2_conditional_ppg_2026_candidate_{objective}_v1"


def specification_table(
    lock_version: str = LOCK_VERSION,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_set, features in LOCKED_FEATURE_SETS.items():
        rows.append(
            {
                "lock_version": lock_version,
                "record_type": "feature_set",
                "specification_name": feature_set,
                "feature_count": len(features),
                "feature_hash": feature_hash(features),
                "feature_names_json": json.dumps(list(features)),
                "parameters_json": pd.NA,
            }
        )
    for model_name, grid in MODEL_GRIDS.items():
        rows.append(
            {
                "lock_version": lock_version,
                "record_type": "model_grid",
                "specification_name": model_name,
                "feature_count": pd.NA,
                "feature_hash": pd.NA,
                "feature_names_json": pd.NA,
                "parameters_json": json.dumps(grid, sort_keys=True),
            }
        )
    rows.append(
        {
            "lock_version": lock_version,
            "record_type": "blend",
            "specification_name": "conditional_ppg_equal_thirds",
            "feature_count": pd.NA,
            "feature_hash": pd.NA,
            "feature_names_json": pd.NA,
            "parameters_json": json.dumps(
                LOCKED_BLEND_WEIGHTS, sort_keys=True
            ),
        }
    )
    return pd.DataFrame(rows)


def locked_metadata(
    scoring_objective: str = LOCKED_SCORING_OBJECTIVE,
    lock_version: str | None = None,
) -> Mapping[str, object]:
    active_lock = lock_version or lock_version_for_scoring(
        scoring_objective
    )
    return {
        "lock_version": active_lock,
        "scoring_objective": scoring_objective,
        "validation_seasons": list(LOCKED_VALIDATION_SEASONS),
        "inner_validation_start": LOCKED_INNER_VALIDATION_START,
        "random_seed": LOCKED_RANDOM_SEED,
        "primary_feature_hash": feature_hash(PRIMARY_PPG_FEATURES),
        "participation_feature_hash": feature_hash(PARTICIPATION_FEATURES),
        "blend_weights": LOCKED_BLEND_WEIGHTS,
    }
