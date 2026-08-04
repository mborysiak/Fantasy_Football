"""Expand LightGBM/CatBoost tuning without touching the temporal holdout."""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import (
    LIGHTGBM_GRID,
    LOCKED_RANDOM_SEED,
    PRIMARY_PPG_FEATURES,
)
from Scripts.V2.modeling import add_modeling_features
from Scripts.V2.native_runtime import (
    RANDOM_FOREST_N_JOBS,
    run_module_function_in_fresh_process,
)


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
BASELINE_STUDY = REPO_ROOT / "research" / "studies" / "2026-08-02_extra_trees_cb_ensembles"
LGBM_WORKER = STUDY_DIR / "lgbm_worker.py"
CURRENT_FAMILIES = (
    "conditional_ppg_lasso",
    "conditional_ppg_random_forest",
    "conditional_ppg_lightgbm",
)
TUNING_ORIGINS = tuple(range(2013, 2023))
HOLDOUT_SEASONS = (2023, 2024, 2025)
BOOTSTRAP_DRAWS = 20_000

LGBM_NEW_PROFILES = (
    {
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
    },
    {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 40,
        "reg_lambda": 5.0,
    },
)
LGBM_NEW_SCHEDULES = (
    (500, 0.01),
    (250, 0.02),
    (125, 0.04),
    (50, 0.10),
)
EXPANDED_LGBM_GRID = tuple(dict(parameters) for parameters in LIGHTGBM_GRID) + tuple(
    {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        **profile,
    }
    for profile, (n_estimators, learning_rate) in product(
        LGBM_NEW_PROFILES, LGBM_NEW_SCHEDULES
    )
)

CATBOOST_ORIGINAL_GRID = tuple(
    {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "random_strength": random_strength,
    }
    for depth, l2_leaf_reg, random_strength in product(
        (3, 5), (5.0, 20.0), (0.5, 2.0)
    )
)
CATBOOST_SCHEDULE_EXPANSION = tuple(
    {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
    }
    for iterations, learning_rate in (
        (900, 0.01),
        (450, 0.02),
        (180, 0.05),
        (120, 0.075),
    )
)
CATBOOST_BOUNDARY_EXPANSION = (
    {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
    },
    {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 1.0,
        "random_strength": 0.5,
    },
    {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.0,
    },
    {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "random_strength": 0.0,
    },
)
EXPANDED_CATBOOST_GRID = (
    *CATBOOST_ORIGINAL_GRID,
    *CATBOOST_SCHEDULE_EXPANSION,
    *CATBOOST_BOUNDARY_EXPANSION,
)

ENSEMBLE_METHODS = (
    "current_single",
    "expanded_lgbm_replacement_equal3",
    "expanded_catboost_equal4",
    "current_plus_extra_trees_equal4",
    "expanded_lgbm_plus_extra_trees_equal4",
    "expanded_lgbm_plus_catboost_equal4",
)
BOOTSTRAP_COMPARISONS = (
    ("expanded_lgbm_replacement_equal3", "current_single"),
    ("expanded_catboost_equal4", "current_single"),
    ("current_plus_extra_trees_equal4", "current_single"),
    ("expanded_lgbm_plus_extra_trees_equal4", "current_single"),
    ("expanded_lgbm_plus_extra_trees_equal4", "current_plus_extra_trees_equal4"),
    ("expanded_lgbm_plus_catboost_equal4", "expanded_lgbm_replacement_equal3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=sorted(DATABASES), required=True)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args()


def _load_target(league: str) -> pd.DataFrame:
    database = DATABASES[league]
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query("SELECT * FROM player_season_features", connection)
    frame = add_modeling_features(frame)
    missing = [feature for feature in PRIMARY_PPG_FEATURES if feature not in frame]
    if missing:
        raise ValueError(f"Missing locked primary features: {missing}")
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame = frame[
        frame["position"].isin(("QB", "RB", "WR", "TE"))
        & frame["season"].le(max(HOLDOUT_SEASONS))
        & frame["conditional_ppg_training_eligible"].eq(1)
        & frame["conditional_ppg"].notna()
        & frame["expert_ppg_team_game_median"].notna()
    ].copy()
    frame["actual"] = pd.to_numeric(frame["conditional_ppg"], errors="raise")
    frame.sort_values(["season", "player_key"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if frame.duplicated(["player_key", "season"]).any():
        raise ValueError("Duplicate player-season targets")
    return frame


def _catboost_pipeline(parameters: Mapping[str, object]) -> Pipeline:
    estimator = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=LOCKED_RANDOM_SEED,
        thread_count=RANDOM_FOREST_N_JOBS,
        verbose=False,
        allow_writing_files=False,
        **dict(parameters),
    )
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
            ("model", estimator),
        ]
    )


def _fit_predict_catboost(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    parameters: Mapping[str, object],
) -> np.ndarray:
    features = list(PRIMARY_PPG_FEATURES)
    model = _catboost_pipeline(parameters)
    model.fit(
        train[features].apply(pd.to_numeric, errors="coerce"),
        train["actual"].to_numpy(float),
    )
    prediction = np.asarray(
        model.predict(predict[features].apply(pd.to_numeric, errors="coerce")),
        dtype=float,
    )
    del model
    gc.collect()
    return prediction


def _select_catboost(train: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    rows = []
    for origin in TUNING_ORIGINS:
        print(f"CatBoost origin {origin}", flush=True)
        origin_train = train[train["season"].lt(origin)]
        origin_hold = train[train["season"].eq(origin)]
        actual = origin_hold["actual"].to_numpy(float)
        for candidate_id, parameters in enumerate(EXPANDED_CATBOOST_GRID):
            prediction = _fit_predict_catboost(origin_train, origin_hold, parameters)
            rows.append(
                {
                    "model_family": "conditional_ppg_catboost_expanded",
                    "candidate_id": candidate_id,
                    "candidate_source": "original" if candidate_id < 8 else "expanded",
                    "origin": origin,
                    "rows": len(origin_hold),
                    "rmse": float(np.sqrt(np.mean(np.square(actual - prediction)))),
                    "parameters_json": json.dumps(parameters, sort_keys=True),
                }
            )
    scores = pd.DataFrame(rows)
    ranked = scores.groupby("candidate_id", as_index=False).agg(
        selection_score=("rmse", "mean")
    )
    winner = ranked.sort_values(["selection_score", "candidate_id"]).iloc[0]
    return int(winner["candidate_id"]), float(winner["selection_score"]), scores


def _chunked(values: Sequence[object], size: int) -> list[Sequence[object]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _select_lgbm(train: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    rows = []
    candidates = list(enumerate(EXPANDED_LGBM_GRID))
    for origin in TUNING_ORIGINS:
        print(f"LightGBM origin {origin}", flush=True)
        origin_train = train[train["season"].lt(origin)]
        origin_hold = train[train["season"].eq(origin)]
        for chunk in _chunked(candidates, 8):
            chunk_rows = run_module_function_in_fresh_process(
                LGBM_WORKER,
                "score_candidate_chunk",
                args=(origin_train, origin_hold, chunk, LOCKED_RANDOM_SEED),
            )
            for row in chunk_rows:
                candidate_id = int(row["candidate_id"])
                rows.append(
                    {
                        "model_family": "conditional_ppg_lightgbm_expanded",
                        "candidate_id": candidate_id,
                        "candidate_source": "original" if candidate_id < 8 else "expanded",
                        "origin": origin,
                        "rows": int(row["rows"]),
                        "rmse": float(row["rmse"]),
                        "parameters_json": json.dumps(
                            EXPANDED_LGBM_GRID[candidate_id], sort_keys=True
                        ),
                    }
                )
    scores = pd.DataFrame(rows)
    ranked = scores.groupby("candidate_id", as_index=False).agg(
        selection_score=("rmse", "mean")
    )
    winner = ranked.sort_values(["selection_score", "candidate_id"]).iloc[0]
    return int(winner["candidate_id"]), float(winner["selection_score"]), scores


def _fit_selected_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parameters: Mapping[str, object],
) -> np.ndarray:
    prediction = run_module_function_in_fresh_process(
        LGBM_WORKER,
        "fit_selected",
        args=(train, test, parameters, LOCKED_RANDOM_SEED),
    )
    return np.asarray(prediction, dtype=float)


def _load_baseline_predictions(league: str, test: pd.DataFrame) -> pd.DataFrame:
    directory = BASELINE_STUDY / f"results_{league}"
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["training_max_season"] != 2022:
        raise ValueError("Baseline training boundary is not sealed at 2022")
    predictions = pd.read_csv(directory / "holdout_predictions.csv")
    selected = predictions[
        (
            predictions["method"].eq("current_single")
            & predictions["model_family"].isin(CURRENT_FAMILIES)
        )
        | (
            predictions["method"].eq("challenger_standalone")
            & predictions["model_family"].eq("conditional_ppg_extra_trees")
        )
    ].copy()
    keys = ["player_key", "season", "position"]
    wide = selected.pivot(index=keys, columns="model_family", values="prediction")
    wide.reset_index(inplace=True)
    current = test[keys + ["actual"]].merge(wide, on=keys, how="left", validate="one_to_one")
    required = [*CURRENT_FAMILIES, "conditional_ppg_extra_trees"]
    if current[required].isna().any().any():
        raise ValueError("Baseline component predictions are incomplete")
    baseline_actual = selected[keys + ["actual"]].drop_duplicates(keys)
    check = current.merge(
        baseline_actual,
        on=keys,
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    if not np.allclose(check["actual"], check["actual_baseline"]):
        raise ValueError("Target values changed since the baseline study")
    return current


def _prediction_frame(
    frame: pd.DataFrame,
    league: str,
    method: str,
    model_family: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    output = frame[["player_key", "season", "position", "actual"]].copy()
    output.insert(0, "league", league)
    output["method"] = method
    output["model_family"] = model_family
    output["prediction"] = np.asarray(prediction, dtype=float)
    return output


def _score_group(group: pd.DataFrame) -> pd.Series:
    actual = group["actual"].to_numpy(float)
    prediction = group["prediction"].to_numpy(float)
    return pd.Series(
        {
            "rows": len(group),
            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
            "mae": float(mean_absolute_error(actual, prediction)),
            "bias": float(np.mean(prediction - actual)),
            "spearman": float(
                pd.Series(actual).corr(pd.Series(prediction), method="spearman")
            ),
        }
    )


def _scores(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    slices = {
        "all": pd.Series("all_2023_2025", index=predictions.index),
        "season": predictions["season"].astype(str),
        "position": predictions["position"],
    }
    for slice_type, values in slices.items():
        current = predictions.copy()
        current["slice_type"] = slice_type
        current["slice_value"] = values
        frames.append(
            current.groupby(
                ["league", "method", "model_family", "slice_type", "slice_value"],
                sort=True,
            )
            .apply(_score_group, include_groups=False)
            .reset_index()
        )
    return pd.concat(frames, ignore_index=True)


def _player_cluster_bootstrap(predictions: pd.DataFrame, league: str) -> pd.DataFrame:
    blend = predictions[predictions["model_family"].eq("primary_blend")]
    wide = blend.pivot(
        index=["player_key", "season", "actual"],
        columns="method",
        values="prediction",
    ).reset_index()
    player_rows = []
    for player_key, group in wide.groupby("player_key", sort=False):
        record: dict[str, object] = {"player_key": player_key, "rows": len(group)}
        actual = group["actual"].to_numpy(float)
        for method in ENSEMBLE_METHODS:
            record[f"sse__{method}"] = float(
                np.square(actual - group[method].to_numpy(float)).sum()
            )
        player_rows.append(record)
    players = pd.DataFrame(player_rows)
    rng = np.random.default_rng(111_001 if league == "dk" else 112_001)
    counts = players["rows"].to_numpy(float)
    n_players = len(players)
    rows = []
    for challenger, baseline in BOOTSTRAP_COMPARISONS:
        base_sse = players[f"sse__{baseline}"].to_numpy(float)
        challenger_sse = players[f"sse__{challenger}"].to_numpy(float)
        deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for start in range(0, BOOTSTRAP_DRAWS, 500):
            stop = min(start + 500, BOOTSTRAP_DRAWS)
            sample = rng.integers(0, n_players, size=(stop - start, n_players))
            sampled_rows = counts[sample].sum(axis=1)
            base_rmse = np.sqrt(base_sse[sample].sum(axis=1) / sampled_rows)
            challenger_rmse = np.sqrt(challenger_sse[sample].sum(axis=1) / sampled_rows)
            deltas[start:stop] = challenger_rmse - base_rmse
        rows.append(
            {
                "league": league,
                "method": challenger,
                "baseline_method": baseline,
                "comparison": f"{challenger}_vs_{baseline}",
                "rmse_delta": float(
                    np.sqrt(challenger_sse.sum() / counts.sum())
                    - np.sqrt(base_sse.sum() / counts.sum())
                ),
                "bootstrap_low": float(np.quantile(deltas, 0.025)),
                "bootstrap_high": float(np.quantile(deltas, 0.975)),
                "player_clusters": n_players,
                "draws": BOOTSTRAP_DRAWS,
            }
        )
    return pd.DataFrame(rows)


def _component_correlations(wide: pd.DataFrame, league: str) -> pd.DataFrame:
    components = (
        "conditional_ppg_lightgbm",
        "conditional_ppg_lightgbm_expanded",
        "conditional_ppg_catboost_expanded",
        "conditional_ppg_extra_trees",
    )
    rows = []
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            rows.append(
                {
                    "league": league,
                    "component_left": left,
                    "component_right": right,
                    "prediction_correlation": float(wide[left].corr(wide[right])),
                    "error_correlation": float(
                        (wide[left] - wide["actual"]).corr(wide[right] - wide["actual"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(league: str, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = _load_target(league)
    train = target[target["season"].lt(min(HOLDOUT_SEASONS))].copy()
    test = target[target["season"].isin(HOLDOUT_SEASONS)].copy()
    if int(train["season"].max()) != 2022:
        raise ValueError("Training boundary is not sealed at 2022")

    print(f"{league}: selecting expanded LightGBM", flush=True)
    lgbm_id, lgbm_selection_score, lgbm_scores = _select_lgbm(train)
    lgbm_parameters = EXPANDED_LGBM_GRID[lgbm_id]
    print(f"{league}: selected LightGBM candidate {lgbm_id}", flush=True)
    lgbm_prediction = _fit_selected_lgbm(train, test, lgbm_parameters)

    print(f"{league}: selecting expanded CatBoost", flush=True)
    cat_id, cat_selection_score, cat_scores = _select_catboost(train)
    cat_parameters = EXPANDED_CATBOOST_GRID[cat_id]
    print(f"{league}: selected CatBoost candidate {cat_id}", flush=True)
    cat_prediction = _fit_predict_catboost(train, test, cat_parameters)

    wide = _load_baseline_predictions(league, test)
    wide["conditional_ppg_lightgbm_expanded"] = lgbm_prediction
    wide["conditional_ppg_catboost_expanded"] = cat_prediction
    lasso = wide["conditional_ppg_lasso"].to_numpy(float)
    random_forest = wide["conditional_ppg_random_forest"].to_numpy(float)
    current_lgbm = wide["conditional_ppg_lightgbm"].to_numpy(float)
    expanded_lgbm = lgbm_prediction
    catboost = cat_prediction
    extra_trees = wide["conditional_ppg_extra_trees"].to_numpy(float)

    blends = {
        "current_single": (lasso + random_forest + current_lgbm) / 3.0,
        "expanded_lgbm_replacement_equal3": (lasso + random_forest + expanded_lgbm) / 3.0,
        "expanded_catboost_equal4": (lasso + random_forest + current_lgbm + catboost) / 4.0,
        "current_plus_extra_trees_equal4": (lasso + random_forest + current_lgbm + extra_trees) / 4.0,
        "expanded_lgbm_plus_extra_trees_equal4": (
            lasso + random_forest + expanded_lgbm + extra_trees
        ) / 4.0,
        "expanded_lgbm_plus_catboost_equal4": (
            lasso + random_forest + expanded_lgbm + catboost
        ) / 4.0,
    }
    prediction_frames = []
    for family in CURRENT_FAMILIES:
        prediction_frames.append(
            _prediction_frame(wide, league, "current_component", family, wide[family].to_numpy(float))
        )
    for family in (
        "conditional_ppg_extra_trees",
        "conditional_ppg_lightgbm_expanded",
        "conditional_ppg_catboost_expanded",
    ):
        prediction_frames.append(
            _prediction_frame(wide, league, "challenger_standalone", family, wide[family].to_numpy(float))
        )
    for method, prediction in blends.items():
        prediction_frames.append(
            _prediction_frame(wide, league, method, "primary_blend", prediction)
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.sort_values(
        ["model_family", "method", "season", "player_key"], inplace=True
    )
    predictions.reset_index(drop=True, inplace=True)
    scores = _scores(predictions)
    bootstrap = _player_cluster_bootstrap(predictions, league)
    correlations = _component_correlations(wide, league)
    origin_scores = pd.concat([lgbm_scores, cat_scores], ignore_index=True)
    selection = pd.DataFrame(
        [
            {
                "league": league,
                "model_family": "conditional_ppg_lightgbm_expanded",
                "candidate_id": lgbm_id,
                "candidate_source": "original" if lgbm_id < 8 else "expanded",
                "selection_score": lgbm_selection_score,
                "parameters_json": json.dumps(lgbm_parameters, sort_keys=True),
            },
            {
                "league": league,
                "model_family": "conditional_ppg_catboost_expanded",
                "candidate_id": cat_id,
                "candidate_source": "original" if cat_id < 8 else "expanded",
                "selection_score": cat_selection_score,
                "parameters_json": json.dumps(cat_parameters, sort_keys=True),
            },
        ]
    )
    origin_scores.insert(0, "league", league)
    origin_scores.to_csv(results_dir / "origin_candidate_scores.csv", index=False)
    selection.to_csv(results_dir / "selected_parameters.csv", index=False)
    predictions.to_csv(results_dir / "holdout_predictions.csv", index=False)
    scores.to_csv(results_dir / "scores.csv", index=False)
    bootstrap.to_csv(results_dir / "player_cluster_bootstrap.csv", index=False)
    correlations.to_csv(results_dir / "component_correlations.csv", index=False)
    metadata = {
        "league": league,
        "training_max_season": int(train["season"].max()),
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "tuning_origins": list(TUNING_ORIGINS),
        "training_rows": len(train),
        "holdout_rows": len(test),
        "feature_count": len(PRIMARY_PPG_FEATURES),
        "lightgbm_candidate_count": len(EXPANDED_LGBM_GRID),
        "catboost_candidate_count": len(EXPANDED_CATBOOST_GRID),
        "baseline_source": str(BASELINE_STUDY.relative_to(REPO_ROOT)),
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    key_scores = scores[
        scores["slice_type"].isin(("all", "season"))
        & (
            scores["model_family"].eq("primary_blend")
            | scores["method"].eq("challenger_standalone")
        )
    ]
    print(selection.to_string(index=False), flush=True)
    print(key_scores.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or STUDY_DIR / f"results_{args.league}"
    run(args.league, results_dir)


if __name__ == "__main__":
    main()
