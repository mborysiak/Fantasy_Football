"""Run the leakage-safe V2 projection consensus ladder."""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import OUTPUT_DB_PATH
from Scripts.V2.contracts import create_run_id
from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    POSITION_FEATURES,
    ModelSpec,
    build_score_summary,
    build_slice_summary,
    build_target_frames,
    make_fold_assignments,
    run_model_spec,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234
MIN_PROVIDER_ACCURACY_SEASONS = 3
MIN_PROVIDER_ACCURACY_ROWS = 50
INNER_ORIGINS = 3
GLOBAL_LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0)
POSITION_LAMBDAS = (0.01, 0.1, 1.0, 10.0)
POSITIONS = ("QB", "RB", "WR", "TE")

CONSENSUS_CORE_FEATURES = (
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

SHAPE_FEATURES = (
    "proj_total_touches",
    "proj_total_opportunities",
    "proj_pass_yards_per_attempt",
    "proj_pass_td_rate",
    "proj_interception_rate",
    "proj_rush_yards_per_attempt",
    "proj_rush_td_rate",
    "proj_receiving_yards_per_reception",
    "proj_receiving_td_rate",
    "proj_catch_rate",
)

STACK_FEATURES = (
    "causal_provider_stack_global",
    "causal_provider_stack_position",
    "causal_provider_stack_minus_median",
    "causal_provider_weighted_std",
    "causal_provider_eligible_count",
)

ROOM_FEATURES = (
    "projection_room_share_std",
    "projection_room_share_range",
    "projection_room_rank_std",
    "projection_room_gap_std",
    "projection_room_leader_vote_share",
    "projection_room_provider_count",
)

ACTIVE_FEATURES = (
    "projection_active_ppg_std",
    "projection_active_provider_count",
    "projection_active_minus_team_game",
    "projection_projected_games_std",
)


def _load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    with sqlite3.connect(OUTPUT_DB_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features", connection
        )
        manifests = pd.read_sql_query(
            "SELECT * FROM feature_manifests", connection
        )
        projection_values = pd.read_sql_query(
            "SELECT * FROM player_season_projection_values", connection
        )
    run_ids = features["run_id"].dropna().astype(str).unique()
    if len(run_ids) != 1:
        raise ValueError("Expected one active feature run")
    if set(manifests["run_id"].dropna().astype(str).unique()) != set(run_ids):
        raise ValueError("Feature manifests do not match active features")
    if set(
        projection_values["run_id"].dropna().astype(str).unique()
    ) != set(run_ids):
        raise ValueError("Projection values do not match active features")
    return features, manifests, projection_values, str(run_ids[0])


def _provider_wide(
    features: pd.DataFrame,
    projection_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    configured = projection_values[
        projection_values["configured_points_complete"].eq(1)
        & projection_values["provider_points_per_team_game"].notna()
    ].copy()
    providers = tuple(sorted(configured["provider"].astype(str).unique()))
    pivot = (
        configured.pivot_table(
            index=["player_key", "season"],
            columns="provider",
            values="provider_points_per_team_game",
            aggfunc="median",
        )
        .reindex(columns=providers)
        .reset_index()
    )
    columns = [
        "player_key",
        "season",
        "position",
        "expert_ppg_team_game_median",
        "outcome_complete",
        "unconditional_season_points",
    ]
    stack_data = features[columns].merge(
        pivot,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    schedule_games = np.where(
        pd.to_numeric(stack_data["season"]).ge(2021), 17.0, 16.0
    )
    stack_data["actual_team_game_ppg"] = (
        pd.to_numeric(
            stack_data["unconditional_season_points"], errors="coerce"
        )
        / schedule_games
    ).where(stack_data["outcome_complete"].eq(1))
    return stack_data, configured, providers


def _eligible_providers(
    configured: pd.DataFrame,
    stack_data: pd.DataFrame,
    providers: Sequence[str],
    origin_season: int,
) -> tuple[str, ...]:
    outcomes = stack_data[
        ["player_key", "season", "actual_team_game_ppg"]
    ]
    history = configured[
        configured["season"].lt(origin_season)
        & configured["provider"].isin(providers)
    ].merge(
        outcomes,
        on=["player_key", "season"],
        how="inner",
        validate="many_to_one",
    )
    history = history[history["actual_team_game_ppg"].notna()]
    if history.empty:
        return ()
    coverage = history.groupby("provider").agg(
        seasons=("season", "nunique"),
        rows=("player_key", "size"),
    )
    return tuple(
        provider
        for provider in providers
        if provider in coverage.index
        and int(coverage.loc[provider, "seasons"])
        >= MIN_PROVIDER_ACCURACY_SEASONS
        and int(coverage.loc[provider, "rows"])
        >= MIN_PROVIDER_ACCURACY_ROWS
    )


def _stack_matrix(
    frame: pd.DataFrame,
    providers: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    baseline = pd.to_numeric(
        frame["expert_ppg_team_game_median"], errors="coerce"
    ).to_numpy(dtype=float)
    if providers:
        raw = frame.loc[:, list(providers)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        provider_matrix = np.where(
            np.isfinite(raw), raw, baseline[:, np.newaxis]
        )
    else:
        provider_matrix = np.empty((len(frame), 0), dtype=float)
    matrix = np.column_stack([baseline, provider_matrix])
    return matrix, baseline


def _fit_convex_weights(
    frame: pd.DataFrame,
    providers: Sequence[str],
    prior_weights: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, int]:
    matrix, baseline = _stack_matrix(frame, providers)
    target = pd.to_numeric(
        frame["actual_team_game_ppg"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(target) & np.isfinite(baseline)
    matrix = matrix[valid]
    target = target[valid]
    if len(target) < MIN_PROVIDER_ACCURACY_ROWS:
        return prior_weights.copy(), int(len(target))

    def objective(weights: np.ndarray) -> float:
        error = matrix @ weights - target
        return float(
            np.mean(np.square(error))
            + penalty * np.sum(np.square(weights - prior_weights))
        )

    result = minimize(
        objective,
        prior_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(prior_weights),
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
        },
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or not np.isclose(result.x.sum(), 1.0, atol=1e-6)
    ):
        return prior_weights.copy(), int(len(target))
    return np.clip(result.x, 0.0, 1.0), int(len(target))


def _predict_stack(
    frame: pd.DataFrame,
    providers: Sequence[str],
    weights: np.ndarray,
) -> np.ndarray:
    matrix, baseline = _stack_matrix(frame, providers)
    prediction = matrix @ weights
    return np.where(np.isfinite(prediction), prediction, baseline)


def _inner_seasons(
    stack_data: pd.DataFrame,
    origin_season: int,
) -> tuple[int, ...]:
    seasons = sorted(
        int(season)
        for season in stack_data.loc[
            stack_data["season"].lt(origin_season)
            & stack_data["actual_team_game_ppg"].notna(),
            "season",
        ].unique()
    )
    return tuple(seasons[-INNER_ORIGINS:])


def _global_fit(
    stack_data: pd.DataFrame,
    configured: pd.DataFrame,
    providers: Sequence[str],
    origin_season: int,
    penalty: float,
) -> tuple[tuple[str, ...], np.ndarray, int]:
    eligible = _eligible_providers(
        configured, stack_data, providers, origin_season
    )
    prior = np.zeros(len(eligible) + 1, dtype=float)
    prior[0] = 1.0
    training = stack_data[
        stack_data["season"].lt(origin_season)
        & stack_data["actual_team_game_ppg"].notna()
    ]
    weights, rows = _fit_convex_weights(
        training, eligible, prior, penalty
    )
    return eligible, weights, rows


def _select_global_penalty(
    stack_data: pd.DataFrame,
    configured: pd.DataFrame,
    providers: Sequence[str],
    origin_season: int,
) -> float:
    inner = _inner_seasons(stack_data, origin_season)
    if len(inner) < 2:
        return 1.0
    scores: list[tuple[float, float]] = []
    for penalty in GLOBAL_LAMBDAS:
        actual_parts = []
        prediction_parts = []
        for season in inner:
            eligible, weights, _ = _global_fit(
                stack_data,
                configured,
                providers,
                season,
                penalty,
            )
            holdout = stack_data[
                stack_data["season"].eq(season)
                & stack_data["actual_team_game_ppg"].notna()
            ]
            prediction = _predict_stack(holdout, eligible, weights)
            actual = pd.to_numeric(
                holdout["actual_team_game_ppg"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(prediction)
            actual_parts.append(actual[valid])
            prediction_parts.append(prediction[valid])
        if actual_parts and sum(len(values) for values in actual_parts):
            actual = np.concatenate(actual_parts)
            prediction = np.concatenate(prediction_parts)
            scores.append(
                (
                    float(
                        np.sqrt(np.mean(np.square(prediction - actual)))
                    ),
                    float(penalty),
                )
            )
    if not scores:
        return 1.0
    return min(scores, key=lambda item: (item[0], -item[1]))[1]


def _position_fit(
    stack_data: pd.DataFrame,
    providers: Sequence[str],
    origin_season: int,
    position: str,
    global_weights: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, int]:
    training = stack_data[
        stack_data["season"].lt(origin_season)
        & stack_data["position"].eq(position)
        & stack_data["actual_team_game_ppg"].notna()
    ]
    return _fit_convex_weights(
        training, providers, global_weights, penalty
    )


def _select_position_penalty(
    stack_data: pd.DataFrame,
    configured: pd.DataFrame,
    providers: Sequence[str],
    origin_season: int,
    global_penalty: float,
) -> float:
    inner = _inner_seasons(stack_data, origin_season)
    if len(inner) < 2:
        return 1.0
    scores: list[tuple[float, float]] = []
    for penalty in POSITION_LAMBDAS:
        actual_parts = []
        prediction_parts = []
        for season in inner:
            eligible, global_weights, _ = _global_fit(
                stack_data,
                configured,
                providers,
                season,
                global_penalty,
            )
            holdout = stack_data[
                stack_data["season"].eq(season)
                & stack_data["actual_team_game_ppg"].notna()
            ].copy()
            prediction = np.full(len(holdout), np.nan, dtype=float)
            for position in POSITIONS:
                position_mask = holdout["position"].eq(position).to_numpy()
                if not position_mask.any():
                    continue
                weights, _ = _position_fit(
                    stack_data,
                    eligible,
                    season,
                    position,
                    global_weights,
                    penalty,
                )
                prediction[position_mask] = _predict_stack(
                    holdout.loc[position_mask],
                    eligible,
                    weights,
                )
            actual = pd.to_numeric(
                holdout["actual_team_game_ppg"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(prediction)
            actual_parts.append(actual[valid])
            prediction_parts.append(prediction[valid])
        if actual_parts and sum(len(values) for values in actual_parts):
            actual = np.concatenate(actual_parts)
            prediction = np.concatenate(prediction_parts)
            scores.append(
                (
                    float(
                        np.sqrt(np.mean(np.square(prediction - actual)))
                    ),
                    float(penalty),
                )
            )
    if not scores:
        return 1.0
    return min(scores, key=lambda item: (item[0], -item[1]))[1]


def _weighted_provider_std(
    frame: pd.DataFrame,
    providers: Sequence[str],
    weights: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    if not providers:
        return np.full(len(frame), np.nan, dtype=float)
    raw = frame.loc[:, list(providers)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    provider_weights = weights[1:]
    valid = np.isfinite(raw)
    row_weights = valid * provider_weights[np.newaxis, :]
    denominator = row_weights.sum(axis=1)
    squared = np.square(raw - center[:, np.newaxis])
    numerator = np.nansum(row_weights * squared, axis=1)
    output = np.sqrt(
        np.divide(
            numerator,
            denominator,
            out=np.full(len(frame), np.nan, dtype=float),
            where=denominator > 0,
        )
    )
    return output


def build_causal_provider_stack(
    features: pd.DataFrame,
    projection_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stack_data, configured, providers = _provider_wide(
        features, projection_values
    )
    prediction_parts = []
    weight_rows: list[dict[str, object]] = []
    seasons = sorted(int(value) for value in stack_data["season"].unique())
    for index, season in enumerate(seasons, start=1):
        current = stack_data[stack_data["season"].eq(season)].copy()
        global_penalty = _select_global_penalty(
            stack_data, configured, providers, season
        )
        eligible, global_weights, global_rows = _global_fit(
            stack_data,
            configured,
            providers,
            season,
            global_penalty,
        )
        position_penalty = _select_position_penalty(
            stack_data,
            configured,
            providers,
            season,
            global_penalty,
        )
        global_prediction = _predict_stack(
            current, eligible, global_weights
        )
        position_prediction = np.full(len(current), np.nan, dtype=float)
        weighted_std = np.full(len(current), np.nan, dtype=float)
        components = ("configured_median", *eligible)
        for component, weight in zip(components, global_weights):
            weight_rows.append(
                {
                    "origin_season": season,
                    "position": "ALL",
                    "component": component,
                    "weight": float(weight),
                    "global_penalty": global_penalty,
                    "position_penalty": position_penalty,
                    "training_rows": global_rows,
                }
            )
        for position in POSITIONS:
            mask = current["position"].eq(position).to_numpy()
            if not mask.any():
                continue
            position_weights, position_rows = _position_fit(
                stack_data,
                eligible,
                season,
                position,
                global_weights,
                position_penalty,
            )
            position_prediction[mask] = _predict_stack(
                current.loc[mask], eligible, position_weights
            )
            weighted_std[mask] = _weighted_provider_std(
                current.loc[mask],
                eligible,
                position_weights,
                position_prediction[mask],
            )
            for component, weight in zip(components, position_weights):
                weight_rows.append(
                    {
                        "origin_season": season,
                        "position": position,
                        "component": component,
                        "weight": float(weight),
                        "global_penalty": global_penalty,
                        "position_penalty": position_penalty,
                        "training_rows": position_rows,
                    }
                )
        baseline = pd.to_numeric(
            current["expert_ppg_team_game_median"], errors="coerce"
        ).to_numpy(dtype=float)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "player_key": current["player_key"].to_numpy(),
                    "season": season,
                    "position": current["position"].to_numpy(),
                    "actual_team_game_ppg": current[
                        "actual_team_game_ppg"
                    ].to_numpy(),
                    "configured_median": baseline,
                    "causal_provider_stack_global": global_prediction,
                    "causal_provider_stack_position": position_prediction,
                    "causal_provider_stack_minus_median": (
                        position_prediction - baseline
                    ),
                    "causal_provider_weighted_std": weighted_std,
                    "causal_provider_eligible_count": len(eligible),
                }
            )
        )
        print(
            f"[stack {index}/{len(seasons)}] {season}: "
            f"{len(eligible)} eligible providers",
            flush=True,
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    return (
        predictions[
            [
                "player_key",
                "season",
                *STACK_FEATURES,
            ]
        ],
        predictions,
        weights,
    )


def build_room_disagreement(
    projection_values: pd.DataFrame,
) -> pd.DataFrame:
    configured = projection_values[
        projection_values["configured_points_complete"].eq(1)
    ].copy()
    keys = ["player_key", "season"]
    grouped = configured.groupby(keys, sort=True)
    output = grouped.agg(
        projection_room_share_std=("provider_room_share", "std"),
        projection_room_rank_std=("provider_room_rank", "std"),
        projection_room_gap_std=("provider_room_gap_to_leader", "std"),
        projection_room_provider_count=("provider_room_share", "count"),
    ).reset_index()
    share_range = grouped["provider_room_share"].agg(
        lambda values: (
            float(values.max() - values.min())
            if values.notna().any()
            else np.nan
        )
    )
    leader_vote = grouped["provider_room_rank"].agg(
        lambda values: (
            float(values.dropna().eq(1).mean())
            if values.notna().any()
            else np.nan
        )
    )
    output = output.merge(
        share_range.rename("projection_room_share_range").reset_index(),
        on=keys,
        how="left",
        validate="one_to_one",
    ).merge(
        leader_vote.rename(
            "projection_room_leader_vote_share"
        ).reset_index(),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    return output


def build_active_alignment(
    projection_values: pd.DataFrame,
) -> pd.DataFrame:
    active = projection_values[
        projection_values["configured_points_complete"].eq(1)
        & projection_values[
            "provider_points_per_projected_game"
        ].notna()
    ].copy()
    keys = ["player_key", "season"]
    output = (
        active.groupby(keys)
        .agg(
            projection_active_ppg_median=(
                "provider_points_per_projected_game",
                "median",
            ),
            projection_active_ppg_std=(
                "provider_points_per_projected_game",
                "std",
            ),
            projection_active_provider_count=(
                "provider_points_per_projected_game",
                "count",
            ),
            projection_projected_games_std=("projected_games", "std"),
        )
        .reset_index()
    )
    return output


def add_research_features(
    features: pd.DataFrame,
    projection_values: pd.DataFrame,
    stack_features: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["player_key", "season"]
    output = (
        features.merge(
            stack_features,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        .merge(
            build_room_disagreement(projection_values),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        .merge(
            build_active_alignment(projection_values),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    )
    output["projection_active_minus_team_game"] = (
        pd.to_numeric(
            output["projection_active_ppg_median"], errors="coerce"
        )
        - pd.to_numeric(
            output["expert_ppg_team_game_median"], errors="coerce"
        )
    )
    required = {
        *CONSENSUS_CORE_FEATURES,
        *SHAPE_FEATURES,
        *STACK_FEATURES,
        *ROOM_FEATURES,
        *ACTIVE_FEATURES,
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        raise ValueError(f"Missing research features: {missing}")
    return output


def _manifest_features(
    manifests: pd.DataFrame,
    manifest_name: str,
) -> tuple[str, ...]:
    return tuple(
        manifests.loc[
            manifests["manifest_name"].eq(manifest_name),
            "feature_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def _variants(
    manifests: pd.DataFrame,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    position = tuple(POSITION_FEATURES)
    core = tuple(dict.fromkeys((*CONSENSUS_CORE_FEATURES, *position)))
    projection_only = {
        "median": (
            "expert_ppg_team_game_median",
            *position,
        ),
        "core": core,
        "plus_shape": tuple(
            dict.fromkeys((*core, *SHAPE_FEATURES))
        ),
        "plus_stack": tuple(
            dict.fromkeys((*core, *STACK_FEATURES))
        ),
        "plus_room": tuple(
            dict.fromkeys((*core, *ROOM_FEATURES))
        ),
        "plus_active": tuple(
            dict.fromkeys((*core, *ACTIVE_FEATURES))
        ),
        "plus_all": tuple(
            dict.fromkeys(
                (
                    *core,
                    *SHAPE_FEATURES,
                    *STACK_FEATURES,
                    *ROOM_FEATURES,
                    *ACTIVE_FEATURES,
                )
            )
        ),
    }
    incumbent = tuple(
        dict.fromkeys(
            (
                *_manifest_features(
                    manifests, "residual_candidate_v1"
                ),
                *position,
            )
        )
    )
    full = {
        "base": incumbent,
        "plus_stack": tuple(
            dict.fromkeys((*incumbent, *STACK_FEATURES))
        ),
        "plus_room": tuple(
            dict.fromkeys((*incumbent, *ROOM_FEATURES))
        ),
        "plus_active": tuple(
            dict.fromkeys((*incumbent, *ACTIVE_FEATURES))
        ),
        "plus_targeted": tuple(
            dict.fromkeys(
                (
                    *incumbent,
                    *STACK_FEATURES,
                    *ROOM_FEATURES,
                    *ACTIVE_FEATURES,
                )
            )
        ),
        "plus_all_projection": tuple(
            dict.fromkeys(
                (
                    *incumbent,
                    *SHAPE_FEATURES,
                    *STACK_FEATURES,
                    *ROOM_FEATURES,
                    *ACTIVE_FEATURES,
                )
            )
        ),
    }
    return projection_only, full


def _model_spec(
    stage: str,
    model_family: str,
    variant: str,
) -> ModelSpec:
    if model_family == "lasso":
        model_piece = "lasso"
        parameters = {
            "lasso__alpha": (
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
            )
        }
        iterations = 20
    elif model_family == "lightgbm":
        model_piece = "lgbm"
        parameters = {
            "lgbm__n_estimators": (100, 200),
            "lgbm__learning_rate": (0.03, 0.05),
            "lgbm__num_leaves": (7, 15),
            "lgbm__max_depth": (3, 4),
            "lgbm__min_child_samples": (20, 40),
            "lgbm__reg_lambda": (1.0, 5.0),
            "lgbm__subsample": (1.0,),
            "lgbm__colsample_bytree": (1.0,),
            "lgbm__deterministic": (True,),
            "lgbm__force_col_wise": (True,),
        }
        iterations = 4
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    return ModelSpec(
        CONDITIONAL_PPG_TARGET,
        f"{stage}_{model_family}_{variant}",
        model_family,
        "direct",
        variant,
        "raw",
        model_piece,
        parameters,
        iterations,
    )


def _experiments(
    manifests: pd.DataFrame,
) -> list[tuple[ModelSpec, tuple[str, ...], str]]:
    projection_only, full = _variants(manifests)
    experiments = [
        (
            _model_spec("projection_only", family, variant),
            columns,
            "projection_only",
        )
        for family in ("lasso", "lightgbm")
        for variant, columns in projection_only.items()
    ]
    experiments.extend(
        (
            _model_spec("full", "lightgbm", variant),
            columns,
            "full",
        )
        for variant, columns in full.items()
    )
    return experiments


def _pooled_rmse(scores: pd.DataFrame, model_name: str) -> float:
    selected = scores[
        scores["model_name"].eq(model_name)
        & scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("rmse")
    ]
    if len(selected) != 1:
        raise ValueError(f"Missing pooled RMSE for {model_name}")
    return float(selected.iloc[0]["value"])


def _season_rmse(slices: pd.DataFrame, model_name: str) -> pd.Series:
    selected = slices[
        slices["model_name"].eq(model_name)
        & slices["slice_type"].eq("season")
        & slices["metric"].eq("rmse")
    ]
    return selected.set_index("slice_value")["value"].astype(float)


def _comparison_summary(
    scores: pd.DataFrame,
    slices: pd.DataFrame,
    experiments: list[tuple[ModelSpec, tuple[str, ...], str]],
) -> pd.DataFrame:
    rows = []
    for index, (spec, columns, stage) in enumerate(experiments):
        if stage == "projection_only":
            if spec.feature_set == "median":
                continue
            reference_variant = (
                "median" if spec.feature_set == "core" else "core"
            )
            reference = (
                f"projection_only_{spec.model_family}_"
                f"{reference_variant}"
            )
        else:
            if spec.feature_set == "base":
                continue
            reference = "full_lightgbm_base"
        challenger_rmse = _pooled_rmse(scores, spec.model_name)
        reference_rmse = _pooled_rmse(scores, reference)
        deltas = _season_rmse(slices, spec.model_name) - _season_rmse(
            slices, reference
        )
        values = deltas.to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + index)
        draws = np.array(
            [
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "stage": stage,
                "model_family": spec.model_family,
                "variant": spec.feature_set,
                "feature_count": len(columns),
                "reference_model": reference,
                "reference_rmse": reference_rmse,
                "challenger_rmse": challenger_rmse,
                "pooled_delta": challenger_rmse - reference_rmse,
                "mean_season_delta": float(deltas.mean()),
                "median_season_delta": float(deltas.median()),
                "challenger_wins": int(deltas.lt(0).sum()),
                "season_count": len(deltas),
                "bootstrap_95_low": float(np.quantile(draws, 0.025)),
                "bootstrap_95_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stage", "model_family", "pooled_delta"]
    )


def _metric_rows(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_columns: Iterable[str],
    slice_type: str,
    slice_column: str | None,
) -> list[dict[str, object]]:
    rows = []
    groups = (
        [("all", frame)]
        if slice_column is None
        else frame.groupby(slice_column, dropna=False)
    )
    for slice_value, group in groups:
        actual = pd.to_numeric(
            group[actual_column], errors="coerce"
        ).to_numpy(dtype=float)
        for method in prediction_columns:
            prediction = pd.to_numeric(
                group[method], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(prediction)
            a = actual[valid]
            p = prediction[valid]
            if not len(a):
                continue
            rows.extend(
                [
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "rmse",
                        "n_rows": len(a),
                        "value": float(
                            np.sqrt(np.mean(np.square(p - a)))
                        ),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "mae",
                        "n_rows": len(a),
                        "value": float(np.mean(np.abs(p - a))),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "bias",
                        "n_rows": len(a),
                        "value": float(np.mean(p - a)),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "spearman",
                        "n_rows": len(a),
                        "value": float(
                            pd.Series(a).corr(
                                pd.Series(p), method="spearman"
                            )
                        ),
                    },
                ]
            )
    return rows


def provider_stack_scores(
    stack_predictions: pd.DataFrame,
) -> pd.DataFrame:
    validation = stack_predictions[
        stack_predictions["season"].between(
            VALIDATION_START, VALIDATION_END
        )
        & stack_predictions["actual_team_game_ppg"].notna()
        & stack_predictions["configured_median"].notna()
    ].copy()
    methods = (
        "configured_median",
        "causal_provider_stack_global",
        "causal_provider_stack_position",
    )
    rows = []
    rows.extend(
        _metric_rows(
            validation,
            "actual_team_game_ppg",
            methods,
            "pooled",
            None,
        )
    )
    rows.extend(
        _metric_rows(
            validation,
            "actual_team_game_ppg",
            methods,
            "season",
            "season",
        )
    )
    rows.extend(
        _metric_rows(
            validation,
            "actual_team_game_ppg",
            methods,
            "position",
            "position",
        )
    )
    return pd.DataFrame(rows)


def calibration_by_expert_tier(oof: pd.DataFrame) -> pd.DataFrame:
    frame = oof.copy()
    frame["_expert_rank"] = frame.groupby(
        ["model_name", "season", "position"]
    )["baseline_prediction"].rank(method="first", pct=True)
    frame["expert_tier"] = pd.cut(
        frame["_expert_rank"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=("bottom", "low", "middle", "high", "top"),
        include_lowest=True,
    )
    rows = []
    for (model_name, tier), group in frame.groupby(
        ["model_name", "expert_tier"],
        observed=True,
    ):
        actual = pd.to_numeric(group["actual"], errors="coerce")
        prediction = pd.to_numeric(
            group["final_prediction"], errors="coerce"
        )
        rows.append(
            {
                "model_name": model_name,
                "expert_tier": str(tier),
                "n_rows": len(group),
                "actual_mean": float(actual.mean()),
                "prediction_mean": float(prediction.mean()),
                "bias": float((prediction - actual).mean()),
                "rmse": float(
                    np.sqrt(np.square(prediction - actual).mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_coverage(
    features: pd.DataFrame,
) -> pd.DataFrame:
    families = {
        "consensus_core": CONSENSUS_CORE_FEATURES,
        "shape": SHAPE_FEATURES,
        "causal_stack": STACK_FEATURES,
        "room_disagreement": ROOM_FEATURES,
        "active_alignment": ACTIVE_FEATURES,
    }
    training = features["conditional_ppg_training_eligible"].eq(1)
    current = features["season"].eq(features["season"].max())
    rows = []
    for family, columns in families.items():
        for column in columns:
            values = pd.to_numeric(features[column], errors="coerce")
            rows.append(
                {
                    "family": family,
                    "feature_name": column,
                    "training_coverage": float(
                        values[training].notna().mean()
                    ),
                    "current_coverage": float(
                        values[current].notna().mean()
                    ),
                    "first_available_season": (
                        int(features.loc[values.notna(), "season"].min())
                        if values.notna().any()
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summary_markdown(
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    stack_scores: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    direct_stack = stack_scores[
        stack_scores["slice_type"].eq("pooled")
        & stack_scores["metric"].eq("rmse")
    ].sort_values("value")
    lines = [
        "# Projection Consensus Ladder Results",
        "",
        "Negative deltas favor the challenger.",
        "",
        "## Provider stack on realized team-game PPG",
        "",
        "| Method | RMSE |",
        "|---|---:|",
    ]
    for row in direct_stack.itertuples(index=False):
        lines.append(f"| `{row.method}` | {float(row.value):.4f} |")
    lines.extend(
        [
            "",
            "## Conditional-PPG OOF models",
            "",
            "| Model | RMSE |",
            "|---|---:|",
        ]
    )
    for row in pooled.itertuples(index=False):
        lines.append(f"| `{row.model_name}` | {float(row.value):.4f} |")
    lines.extend(
        [
            "",
            "## Fold-identical comparisons",
            "",
            "| Stage | Model | Variant | Reference | Delta | "
            "95% interval | Wins |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.stage}` | `{row.model_family}` | "
            f"`{row.variant}` | `{row.reference_model}` | "
            f"{row.pooled_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.challenger_wins}/{row.season_count} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    features, manifests, projection_values, feature_run_id = _load_inputs()
    stack_features, stack_predictions, stack_weights = (
        build_causal_provider_stack(features, projection_values)
    )
    research_features = add_research_features(
        features, projection_values, stack_features
    )
    target = build_target_frames(
        research_features, VALIDATION_END
    )[CONDITIONAL_PPG_TARGET]
    run_id = create_run_id("m4a_projection_consensus_ladder")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    experiments = _experiments(manifests)
    oof_frames = []
    parameter_frames = []
    for index, (spec, columns, stage) in enumerate(
        experiments, start=1
    ):
        print(
            f"[model {index}/{len(experiments)}] "
            f"{spec.model_name} ({stage}, {len(columns)} features)",
            flush=True,
        )
        oof, parameters = run_model_spec(
            target,
            assignments,
            spec,
            columns,
            run_id,
            feature_run_id,
            VALIDATION_START,
            N_SPLITS,
            RANDOM_SEED,
            quiet=True,
        )
        oof_frames.append(oof)
        parameter_frames.append(parameters)

    oof = pd.concat(oof_frames, ignore_index=True)
    hyperparameters = pd.concat(parameter_frames, ignore_index=True)
    scores = build_score_summary(oof, run_id)
    slices = build_slice_summary(oof, run_id)
    comparisons = _comparison_summary(scores, slices, experiments)
    stack_score_table = provider_stack_scores(stack_predictions)
    calibration = calibration_by_expert_tier(oof)
    coverage = feature_coverage(research_features)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "model_slices.csv", index=False)
    hyperparameters.to_csv(
        RESULTS_DIR / "hyperparameters.csv", index=False
    )
    comparisons.to_csv(
        RESULTS_DIR / "model_comparisons.csv", index=False
    )
    stack_predictions.to_csv(
        RESULTS_DIR / "provider_stack_predictions.csv", index=False
    )
    stack_weights.to_csv(
        RESULTS_DIR / "provider_stack_weights.csv", index=False
    )
    stack_score_table.to_csv(
        RESULTS_DIR / "provider_stack_scores.csv", index=False
    )
    calibration.to_csv(
        RESULTS_DIR / "calibration_by_expert_tier.csv", index=False
    )
    coverage.to_csv(
        RESULTS_DIR / "feature_coverage.csv", index=False
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, comparisons, stack_score_table),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "target_rows": len(target),
                "experiments": len(experiments),
                "results_directory": str(RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
