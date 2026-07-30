"""Test direct-total versus separate QB passing/rushing PPG targets."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import (
    LOCKED_INNER_VALIDATION_START,
    LOCKED_RANDOM_SEED,
    LOCKED_VALIDATION_SEASONS,
    PRIMARY_PPG_FEATURES,
)
from Scripts.V2.modeling import add_modeling_features


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": (
        REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3"
    ),
}
RESULT_DIRECTORIES = {
    "dk": STUDY_DIR / "results",
    "beta": STUDY_DIR / "results_beta",
}
OUTER_SEASONS = LOCKED_VALIDATION_SEASONS
GRID_ORIGINS = tuple(
    range(LOCKED_INNER_VALIDATION_START, max(OUTER_SEASONS) + 1)
)
MIN_INNER_SEASONS = 3
MODEL_GRIDS = {
    "lasso": tuple(
        {"alpha": value}
        for value in (0.01, 0.03, 0.1, 0.3)
    ),
    "random_forest": tuple(
        {
            "n_estimators": 125,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": 0.5,
            "bootstrap": True,
            "random_state": LOCKED_RANDOM_SEED,
            "n_jobs": 1,
        }
        for max_depth in (6, 10)
        for min_samples_leaf in (5, 15)
    ),
    "lightgbm": (
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
    ),
}
TARGET_COLUMNS = {
    "total": "actual_total_ppg",
    "pass": "actual_pass_ppg",
    "rush": "actual_rush_ppg",
}
COMPONENT_FEATURES = (
    "expert_pass_ppg_team_game_median",
    "expert_rush_ppg_team_game_median",
    "expert_component_sum_ppg_team_game",
    "prior_year_pass_ppg",
    "prior_year_rush_ppg",
)
FEATURE_COLUMNS = tuple(
    dict.fromkeys((*PRIMARY_PPG_FEATURES, *COMPONENT_FEATURES))
)
LOCKED_METHODS = (
    "expert_recalibrated",
    "conditional_ppg_primary_blend",
)
EVALUATION_METHODS = (
    "expert_direct_team_game",
    "expert_component_sum",
    "expert_component_sum_plus_prior_other",
    "expert_recalibrated",
    "conditional_ppg_primary_blend",
    "qb_direct_total_lasso",
    "qb_component_sum_lasso",
    "qb_direct_total_random_forest",
    "qb_component_sum_random_forest",
    "qb_direct_total_lightgbm",
    "qb_component_sum_lightgbm",
    "qb_direct_total_blend",
    "qb_component_sum_blend",
    "qb_component_sum_plus_prior_other",
)


def _load_inputs(
    database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(database) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features", connection
        )
        outcomes = pd.read_sql_query(
            "SELECT * FROM player_season_outcomes", connection
        )
        projection_values = pd.read_sql_query(
            "SELECT * FROM player_season_projection_values", connection
        )
        locked = pd.read_sql_query(
            """
            SELECT player_key, season, method, prediction
            FROM locked_whole_season_predictions
            WHERE position = 'QB'
              AND target_name = 'conditional_ppg'
              AND method IN (?, ?)
            """,
            connection,
            params=LOCKED_METHODS,
        )
    return features, outcomes, projection_values, locked


def _component_projection_consensus(
    projection_values: pd.DataFrame,
) -> pd.DataFrame:
    frame = projection_values[
        projection_values["position"].eq("QB")
        & projection_values["configured_points_complete"].eq(1)
    ].copy()
    frame["schedule_games"] = np.where(
        pd.to_numeric(frame["season"], errors="raise").ge(2021),
        17.0,
        16.0,
    )
    frame["expert_pass_ppg_team_game"] = (
        pd.to_numeric(frame["passing_points"], errors="coerce")
        / frame["schedule_games"]
    )
    frame["expert_rush_ppg_team_game"] = (
        pd.to_numeric(frame["rushing_points"], errors="coerce")
        / frame["schedule_games"]
    )
    consensus = (
        frame.groupby(["player_key", "season"], as_index=False)
        .agg(
            expert_pass_ppg_team_game_median=(
                "expert_pass_ppg_team_game",
                "median",
            ),
            expert_rush_ppg_team_game_median=(
                "expert_rush_ppg_team_game",
                "median",
            ),
            component_provider_count=("provider", "nunique"),
        )
    )
    consensus["expert_component_sum_ppg_team_game"] = (
        consensus["expert_pass_ppg_team_game_median"]
        + consensus["expert_rush_ppg_team_game_median"]
    )
    return consensus


def _locked_wide(locked: pd.DataFrame) -> pd.DataFrame:
    if locked.duplicated(["player_key", "season", "method"]).any():
        raise ValueError("Locked predictions are not unique")
    wide = locked.pivot(
        index=["player_key", "season"],
        columns="method",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    missing = sorted(set(LOCKED_METHODS).difference(wide.columns))
    if missing:
        raise ValueError(f"Missing locked comparator methods: {missing}")
    return wide


def _build_target(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    projection_values: pd.DataFrame,
    locked: pd.DataFrame,
) -> pd.DataFrame:
    frame = add_modeling_features(features)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame = frame[
        frame["position"].eq("QB")
        & frame["season"].le(max(OUTER_SEASONS))
        & frame["conditional_ppg_training_eligible"].eq(1)
        & frame["conditional_ppg"].notna()
        & frame["expert_ppg_team_game_median"].notna()
    ].copy()
    outcome_columns = (
        "player_key",
        "season",
        "opportunity_games",
        "season_points",
        "passing_points",
        "rushing_points",
        "receiving_points",
        "fumble_points",
        "two_point_points",
        "special_teams_points",
    )
    outcome = outcomes.loc[:, outcome_columns].copy()
    outcome["season"] = pd.to_numeric(
        outcome["season"], errors="raise"
    ).astype(int)
    frame = frame.drop(columns=["opportunity_games"], errors="ignore").merge(
        outcome,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        _component_projection_consensus(projection_values),
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    opportunity_games = pd.to_numeric(
        frame["opportunity_games"], errors="raise"
    )
    frame["actual_total_ppg"] = pd.to_numeric(
        frame["season_points"], errors="raise"
    ) / opportunity_games
    frame["actual_pass_ppg"] = pd.to_numeric(
        frame["passing_points"], errors="raise"
    ) / opportunity_games
    frame["actual_rush_ppg"] = pd.to_numeric(
        frame["rushing_points"], errors="raise"
    ) / opportunity_games
    frame["actual_other_ppg"] = (
        frame["actual_total_ppg"]
        - frame["actual_pass_ppg"]
        - frame["actual_rush_ppg"]
    )
    reconstructed = (
        frame["actual_pass_ppg"]
        + frame["actual_rush_ppg"]
        + frame["actual_other_ppg"]
    )
    if not np.allclose(
        reconstructed,
        frame["actual_total_ppg"],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("QB point components do not reconstruct total PPG")
    required_components = [
        "expert_pass_ppg_team_game_median",
        "expert_rush_ppg_team_game_median",
    ]
    if frame[required_components].isna().any().any():
        missing = frame.loc[
            frame[required_components].isna().any(axis=1),
            ["player_key", "season", "display_name"],
        ]
        raise ValueError(
            "Eligible QB rows lack projection components:\n"
            + missing.head(10).to_string(index=False)
        )
    frame["prior_year_pass_ppg"] = (
        pd.to_numeric(frame["prior_year_ppg"], errors="coerce")
        * pd.to_numeric(
            frame["prior_year_pass_point_share"], errors="coerce"
        )
    )
    frame["prior_year_rush_ppg"] = (
        pd.to_numeric(frame["prior_year_ppg"], errors="coerce")
        * pd.to_numeric(
            frame["prior_year_rush_point_share"], errors="coerce"
        )
    )
    frame = frame.merge(
        _locked_wide(locked),
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    outer = frame["season"].isin(OUTER_SEASONS)
    if frame.loc[outer, list(LOCKED_METHODS)].isna().any().any():
        raise ValueError("Outer QB rows lack locked comparator predictions")
    missing_features = sorted(set(FEATURE_COLUMNS).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Missing study features: {missing_features}")
    frame.sort_values(["season", "player_key"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


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
    if model_name == "lasso":
        steps.extend(
            (
                ("scale", StandardScaler()),
                (
                    "model",
                    Lasso(max_iter=20_000, tol=1e-6, **parameters),
                ),
            )
        )
    elif model_name == "random_forest":
        steps.append(("model", RandomForestRegressor(**parameters)))
    elif model_name == "lightgbm":
        steps.append(
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    verbosity=-1,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    deterministic=True,
                    force_col_wise=True,
                    random_state=LOCKED_RANDOM_SEED,
                    n_jobs=1,
                    **parameters,
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported model family: {model_name}")
    return Pipeline(steps)


def _fit_predict(
    train: pd.DataFrame,
    hold: pd.DataFrame,
    target_column: str,
    model_name: str,
    parameters: Mapping[str, object],
) -> np.ndarray:
    X_train = train.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    X_hold = hold.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    y_train = pd.to_numeric(train[target_column], errors="raise")
    model = _model_pipeline(model_name, parameters)
    model.fit(X_train, y_train)
    return model.predict(X_hold)


def _grid_predictions(
    target: pd.DataFrame,
    target_name: str,
    model_name: str,
    grid: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    target_column = TARGET_COLUMNS[target_name]
    rows: list[pd.DataFrame] = []
    for origin in GRID_ORIGINS:
        train = target[target["season"].lt(origin)]
        hold = target[target["season"].eq(origin)]
        if train.empty or hold.empty:
            continue
        for candidate_id, parameters in enumerate(grid):
            current = hold[
                ["player_key", "season", target_column]
            ].rename(columns={target_column: "actual"})
            current["prediction"] = _fit_predict(
                train,
                hold,
                target_column,
                model_name,
                parameters,
            )
            current["candidate_id"] = candidate_id
            rows.append(current)
    if not rows:
        raise ValueError(f"No grid predictions for {target_name}/{model_name}")
    return pd.concat(rows, ignore_index=True)


def _rmse(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                np.asarray(actual, dtype=float),
                np.asarray(prediction, dtype=float),
            )
        )
    )


def _select_parameters(
    grid_predictions: pd.DataFrame,
    grid: Sequence[Mapping[str, object]],
    target_name: str,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for origin in OUTER_SEASONS:
        prior = grid_predictions[grid_predictions["season"].lt(origin)]
        prior_seasons = sorted(prior["season"].unique())
        if len(prior_seasons) < MIN_INNER_SEASONS:
            raise ValueError(
                f"{target_name}/{model_name}/{origin} has insufficient "
                "strict-prior inner seasons"
            )
        candidates = []
        for candidate_id, candidate in prior.groupby("candidate_id"):
            annual_scores = [
                _rmse(season["actual"], season["prediction"])
                for _, season in candidate.groupby("season")
            ]
            candidates.append(
                (float(np.mean(annual_scores)), int(candidate_id))
            )
        selection_score, selected_id = min(candidates)
        rows.append(
            {
                "target_name": target_name,
                "model_name": model_name,
                "forecast_origin": origin,
                "candidate_id": selected_id,
                "parameters_json": json.dumps(
                    grid[selected_id], sort_keys=True
                ),
                "selection_score": selection_score,
                "selection_start_season": min(prior_seasons),
                "selection_end_season": max(prior_seasons),
                "selection_seasons": len(prior_seasons),
            }
        )
    return pd.DataFrame(rows)


def _outer_predictions(
    target: pd.DataFrame,
    target_name: str,
    model_name: str,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    target_column = TARGET_COLUMNS[target_name]
    rows = []
    for origin in OUTER_SEASONS:
        train = target[target["season"].lt(origin)]
        hold = target[target["season"].eq(origin)]
        selection = selections[
            selections["forecast_origin"].eq(origin)
            & selections["target_name"].eq(target_name)
            & selections["model_name"].eq(model_name)
        ]
        if len(selection) != 1:
            raise ValueError(
                f"Expected one selection for {target_name}/{model_name}/{origin}"
            )
        parameters = json.loads(selection.iloc[0]["parameters_json"])
        current = hold[
            ["player_key", "season", target_column]
        ].rename(columns={target_column: "actual"})
        current["target_name"] = target_name
        current["model_name"] = model_name
        current["prediction"] = _fit_predict(
            train,
            hold,
            target_column,
            model_name,
            parameters,
        )
        current["training_through_season"] = origin - 1
        current["selected_candidate_id"] = int(
            selection.iloc[0]["candidate_id"]
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def _prior_other_adjustment(target: pd.DataFrame) -> dict[int, float]:
    return {
        origin: float(
            target.loc[
                target["season"].lt(origin),
                "actual_other_ppg",
            ].mean()
        )
        for origin in OUTER_SEASONS
    }


def _assemble_predictions(
    target: pd.DataFrame,
    model_predictions: pd.DataFrame,
) -> pd.DataFrame:
    outer = target[target["season"].isin(OUTER_SEASONS)].copy()
    keys = ["player_key", "season"]
    if model_predictions.duplicated(
        [*keys, "target_name", "model_name"]
    ).any():
        raise ValueError("Duplicate modeled QB predictions")
    wide = model_predictions.pivot(
        index=keys,
        columns=["target_name", "model_name"],
        values="prediction",
    )
    wide.columns = [
        f"pred_{target_name}_{model_name}"
        for target_name, model_name in wide.columns
    ]
    outer = outer.merge(
        wide.reset_index(),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    outer["prior_other_ppg_adjustment"] = outer["season"].map(
        _prior_other_adjustment(target)
    )
    outer["expert_direct_team_game"] = outer[
        "expert_ppg_team_game_median"
    ]
    outer["expert_component_sum"] = outer[
        "expert_component_sum_ppg_team_game"
    ]
    outer["expert_component_sum_plus_prior_other"] = (
        outer["expert_component_sum"]
        + outer["prior_other_ppg_adjustment"]
    )
    for model_name in MODEL_GRIDS:
        outer[f"qb_direct_total_{model_name}"] = outer[
            f"pred_total_{model_name}"
        ]
        outer[f"qb_component_sum_{model_name}"] = (
            outer[f"pred_pass_{model_name}"]
            + outer[f"pred_rush_{model_name}"]
        )
    outer["qb_direct_total_blend"] = outer[
        [f"qb_direct_total_{model}" for model in MODEL_GRIDS]
    ].mean(axis=1)
    outer["qb_component_sum_blend"] = outer[
        [f"qb_component_sum_{model}" for model in MODEL_GRIDS]
    ].mean(axis=1)
    outer["qb_component_sum_plus_prior_other"] = (
        outer["qb_component_sum_blend"]
        + outer["prior_other_ppg_adjustment"]
    )
    outer["experience_group"] = np.select(
        [
            pd.to_numeric(outer["is_rookie"], errors="coerce").eq(1),
            pd.to_numeric(outer["year_exp"], errors="coerce").eq(1),
            pd.to_numeric(outer["has_prior_outcome"], errors="coerce").eq(1),
        ],
        ["rookie", "second_year", "veteran_with_history"],
        default="other_no_history",
    )
    rush_share = pd.to_numeric(
        outer["projected_rush_point_share"], errors="coerce"
    )
    outer["qb_style"] = np.select(
        [rush_share.ge(0.25), rush_share.ge(0.15)],
        ["high_rush", "balanced"],
        default="pass_heavy",
    )
    return outer


def _score_group(
    frame: pd.DataFrame,
    slice_type: str,
    slice_value: str,
) -> list[dict[str, object]]:
    rows = []
    actual = frame["actual_total_ppg"]
    for method in EVALUATION_METHODS:
        prediction = frame[method]
        rows.append(
            {
                "method": method,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "n_rows": len(frame),
                "rmse": _rmse(actual, prediction),
                "mae": float(mean_absolute_error(actual, prediction)),
                "bias": float((prediction - actual).mean()),
                "spearman": float(prediction.corr(actual, method="spearman")),
            }
        )
    return rows


def _scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = _score_group(predictions, "all", "all")
    recent = predictions[predictions["season"].ge(2023)]
    rows.extend(_score_group(recent, "window", "2023-2025"))
    for column in ("season", "experience_group", "qb_style"):
        for value, group in predictions.groupby(column, dropna=False):
            rows.extend(
                _score_group(group, column, str(value))
            )
    return pd.DataFrame(rows)


def _component_scores(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target_name, actual_column, expert_column in (
        (
            "pass",
            "actual_pass_ppg",
            "expert_pass_ppg_team_game_median",
        ),
        (
            "rush",
            "actual_rush_ppg",
            "expert_rush_ppg_team_game_median",
        ),
    ):
        methods = {"expert_component": expert_column}
        methods.update(
            {
                model_name: f"pred_{target_name}_{model_name}"
                for model_name in MODEL_GRIDS
            }
        )
        for method, prediction_column in methods.items():
            rows.append(
                {
                    "target_name": target_name,
                    "method": method,
                    "n_rows": len(predictions),
                    "rmse": _rmse(
                        predictions[actual_column],
                        predictions[prediction_column],
                    ),
                    "mae": float(
                        mean_absolute_error(
                            predictions[actual_column],
                            predictions[prediction_column],
                        )
                    ),
                    "bias": float(
                        (
                            predictions[prediction_column]
                            - predictions[actual_column]
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_delta(
    frame: pd.DataFrame,
    challenger: str,
    reference: str,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    errors = frame[["player_key"]].copy()
    actual = frame["actual_total_ppg"].to_numpy(dtype=float)
    errors["challenger_squared_error"] = np.square(
        frame[challenger].to_numpy(dtype=float) - actual
    )
    errors["reference_squared_error"] = np.square(
        frame[reference].to_numpy(dtype=float) - actual
    )
    clusters = errors.groupby("player_key", as_index=False).agg(
        n_rows=("player_key", "size"),
        challenger_squared_error=("challenger_squared_error", "sum"),
        reference_squared_error=("reference_squared_error", "sum"),
    )
    n_players = len(clusters)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n_players,
        np.repeat(1.0 / n_players, n_players),
        size=draws,
    )
    sampled_rows = counts @ clusters["n_rows"].to_numpy(dtype=float)
    challenger_rmse = np.sqrt(
        (
            counts
            @ clusters["challenger_squared_error"].to_numpy(dtype=float)
        )
        / sampled_rows
    )
    reference_rmse = np.sqrt(
        (
            counts
            @ clusters["reference_squared_error"].to_numpy(dtype=float)
        )
        / sampled_rows
    )
    deltas = challenger_rmse - reference_rmse
    return (
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def _comparisons(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    pooled = scores[
        scores["slice_type"].eq("all")
        & scores["slice_value"].eq("all")
    ].set_index("method")["rmse"]
    season = scores[
        scores["slice_type"].eq("season")
    ].pivot(index="slice_value", columns="method", values="rmse")
    pairs = (
        ("expert_component_sum", "expert_direct_team_game"),
        (
            "expert_component_sum_plus_prior_other",
            "expert_direct_team_game",
        ),
        ("qb_component_sum_lasso", "qb_direct_total_lasso"),
        ("qb_component_sum_random_forest", "qb_direct_total_random_forest"),
        ("qb_component_sum_lightgbm", "qb_direct_total_lightgbm"),
        ("qb_component_sum_blend", "qb_direct_total_blend"),
        (
            "qb_component_sum_plus_prior_other",
            "qb_direct_total_blend",
        ),
        (
            "qb_component_sum_plus_prior_other",
            "conditional_ppg_primary_blend",
        ),
        ("qb_component_sum_plus_prior_other", "expert_recalibrated"),
    )
    rows = []
    for index, (challenger, reference) in enumerate(pairs):
        season_delta = season[challenger] - season[reference]
        low, high = _bootstrap_delta(
            predictions,
            challenger,
            reference,
            LOCKED_RANDOM_SEED + index,
        )
        rows.append(
            {
                "challenger": challenger,
                "reference": reference,
                "challenger_rmse": float(pooled[challenger]),
                "reference_rmse": float(pooled[reference]),
                "pooled_delta": float(
                    pooled[challenger] - pooled[reference]
                ),
                "mean_season_delta": float(season_delta.mean()),
                "season_wins": int(season_delta.lt(0).sum()),
                "season_count": len(season_delta),
                "player_cluster_95_low": low,
                "player_cluster_95_high": high,
            }
        )
    return pd.DataFrame(rows)


def _findings_markdown(
    league: str,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    component_scores: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["slice_type"].eq("all")
        & scores["slice_value"].eq("all")
    ].sort_values("rmse")
    primary = comparisons[
        comparisons["challenger"].eq(
            "qb_component_sum_plus_prior_other"
        )
        & comparisons["reference"].eq("qb_direct_total_blend")
    ].iloc[0]
    production = comparisons[
        comparisons["challenger"].eq(
            "qb_component_sum_plus_prior_other"
        )
        & comparisons["reference"].eq(
            "conditional_ppg_primary_blend"
        )
    ].iloc[0]
    lines = [
        f"# QB Component-Target Findings ({league})",
        "",
        (
            "Negative deltas favor independently modeling passing and rushing "
            "PPG and summing the predictions."
        ),
        "",
        "## Pooled 2017-2025",
        "",
        "| Method | RMSE | MAE | Bias |",
        "|---|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.rmse:.4f} | "
            f"{row.mae:.4f} | {row.bias:+.4f} |"
        )
    lines.extend(
        (
            "",
            "## Key same-model comparison",
            "",
            (
                f"The component blend plus its strictly-prior other-points "
                f"adjustment changes RMSE by **{primary.pooled_delta:+.4f}** "
                f"versus the QB-only direct-total blend, wins "
                f"{primary.season_wins}/{primary.season_count} seasons, and "
                f"has player-cluster interval "
                f"[{primary.player_cluster_95_low:+.4f}, "
                f"{primary.player_cluster_95_high:+.4f}]."
            ),
            "",
            (
                f"Versus the locked pooled production candidate, the same "
                f"component challenger changes RMSE by "
                f"**{production.pooled_delta:+.4f}**."
            ),
            "",
            "## Passing and rushing targets",
            "",
            "| Target | Method | RMSE | MAE | Bias |",
            "|---|---|---:|---:|---:|",
        )
    )
    for row in component_scores.sort_values(
        ["target_name", "rmse"]
    ).itertuples(index=False):
        lines.append(
            f"| {row.target_name} | `{row.method}` | {row.rmse:.4f} | "
            f"{row.mae:.4f} | {row.bias:+.4f} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            (
                "This is a target-decomposition test, not a template-weight "
                "test. Production remains unchanged. The direct-total and "
                "component models use identical QB samples, features, model "
                "families, grids, and strictly-prior selection rules."
            ),
        )
    )
    return "\n".join(lines) + "\n"


def run_league(league: str, database: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    features, outcomes, projection_values, locked = _load_inputs(database)
    target = _build_target(
        features,
        outcomes,
        projection_values,
        locked,
    )
    prediction_parts = []
    selection_parts = []
    for target_name in TARGET_COLUMNS:
        for model_name, grid in MODEL_GRIDS.items():
            print(
                f"{league}: {target_name}/{model_name}",
                flush=True,
            )
            grid_predictions = _grid_predictions(
                target,
                target_name,
                model_name,
                grid,
            )
            selections = _select_parameters(
                grid_predictions,
                grid,
                target_name,
                model_name,
            )
            prediction_parts.append(
                _outer_predictions(
                    target,
                    target_name,
                    model_name,
                    selections,
                )
            )
            selection_parts.append(selections)
    model_predictions = pd.concat(prediction_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    predictions = _assemble_predictions(target, model_predictions)
    scores = _scores(predictions)
    components = _component_scores(predictions)
    comparisons = _comparisons(predictions, scores)
    findings = _findings_markdown(
        league,
        predictions,
        scores,
        comparisons,
        components,
    )
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    model_predictions.to_csv(
        results_dir / "model_predictions.csv", index=False
    )
    selections.to_csv(
        results_dir / "selected_hyperparameters.csv", index=False
    )
    scores.to_csv(results_dir / "scores.csv", index=False)
    components.to_csv(results_dir / "component_scores.csv", index=False)
    comparisons.to_csv(results_dir / "comparisons.csv", index=False)
    (results_dir / "findings.md").write_text(findings, encoding="utf-8")
    metadata = {
        "league": league,
        "database": str(database),
        "validation_seasons": list(OUTER_SEASONS),
        "inner_validation_start": LOCKED_INNER_VALIDATION_START,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "outer_rows": len(predictions),
        "outer_unique_players": int(predictions["player_key"].nunique()),
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(findings, flush=True)


def main() -> None:
    for league, database in DATABASES.items():
        run_league(league, database, RESULT_DIRECTORIES[league])


if __name__ == "__main__":
    main()
