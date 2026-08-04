"""Test Extra Trees and CatBoost as fixed-weight fourth ensemble members."""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import LOCKED_RANDOM_SEED, PRIMARY_PPG_FEATURES
from Scripts.V2.modeling import add_modeling_features
from Scripts.V2.native_runtime import RANDOM_FOREST_N_JOBS


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
BASELINE_STUDY = REPO_ROOT / "research" / "studies" / "2026-08-02_skm_fold_ensemble_holdout"
CURRENT_FAMILIES = (
    "conditional_ppg_lasso",
    "conditional_ppg_random_forest",
    "conditional_ppg_lightgbm",
)
CHALLENGER_FAMILIES = (
    "conditional_ppg_extra_trees",
    "conditional_ppg_catboost",
)
ENSEMBLE_METHODS = (
    "current_single",
    "current_plus_extra_trees_equal4",
    "current_plus_catboost_equal4",
    "current_plus_both_equal5",
)
TUNING_ORIGINS = tuple(range(2013, 2023))
HOLDOUT_SEASONS = (2023, 2024, 2025)
BOOTSTRAP_DRAWS = 20_000

EXTRA_TREES_GRID = tuple(
    {
        "n_estimators": 400,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": False,
    }
    for max_depth, min_samples_leaf, max_features in product(
        (6, 10), (3, 10), (0.5, 1.0)
    )
)
CATBOOST_GRID = tuple(
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
MODEL_GRIDS = {
    "conditional_ppg_extra_trees": EXTRA_TREES_GRID,
    "conditional_ppg_catboost": CATBOOST_GRID,
}


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


def _pipeline(
    model_name: str,
    parameters: Mapping[str, object],
    estimator_seed: int = LOCKED_RANDOM_SEED,
) -> Pipeline:
    parameters = dict(parameters)
    if model_name == "conditional_ppg_extra_trees":
        estimator = ExtraTreesRegressor(
            random_state=estimator_seed,
            n_jobs=RANDOM_FOREST_N_JOBS,
            **parameters,
        )
    elif model_name == "conditional_ppg_catboost":
        estimator = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=estimator_seed,
            thread_count=RANDOM_FOREST_N_JOBS,
            verbose=False,
            allow_writing_files=False,
            **parameters,
        )
    else:
        raise ValueError(f"Unsupported model family: {model_name}")
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


def _fit_predict(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    model_name: str,
    parameters: Mapping[str, object],
    estimator_seed: int = LOCKED_RANDOM_SEED,
) -> np.ndarray:
    features = list(PRIMARY_PPG_FEATURES)
    model = _pipeline(model_name, parameters, estimator_seed)
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


def _select_candidate(
    train: pd.DataFrame,
    model_name: str,
) -> tuple[int, float, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    grid = MODEL_GRIDS[model_name]
    for origin in TUNING_ORIGINS:
        origin_train = train[train["season"].lt(origin)]
        origin_hold = train[train["season"].eq(origin)]
        if origin_train.empty or origin_hold.empty:
            raise ValueError(f"Missing selection rows for {model_name}/{origin}")
        actual = origin_hold["actual"].to_numpy(float)
        for candidate_id, parameters in enumerate(grid):
            prediction = _fit_predict(
                origin_train, origin_hold, model_name, parameters
            )
            rows.append(
                {
                    "model_family": model_name,
                    "candidate_id": candidate_id,
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


def _load_current_predictions(league: str, test: pd.DataFrame) -> pd.DataFrame:
    baseline_dir = BASELINE_STUDY / f"results_{league}"
    metadata = json.loads(
        (baseline_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    if metadata["training_max_season"] != 2022:
        raise ValueError("Baseline training boundary is not sealed at 2022")
    if metadata["holdout_seasons"] != list(HOLDOUT_SEASONS):
        raise ValueError("Baseline holdout seasons do not match")
    baseline = pd.read_csv(baseline_dir / "holdout_predictions.csv")
    baseline = baseline[
        baseline["method"].eq("current_single")
        & baseline["model_family"].isin(CURRENT_FAMILIES)
    ].copy()
    keys = ["player_key", "season", "position"]
    wide = baseline.pivot(index=keys, columns="model_family", values="prediction")
    wide.reset_index(inplace=True)
    current = test[keys + ["actual"]].merge(wide, on=keys, how="left", validate="one_to_one")
    if current[list(CURRENT_FAMILIES)].isna().any().any():
        raise ValueError("Current-model baseline is incomplete")
    baseline_actual = baseline[keys + ["actual"]].drop_duplicates(keys)
    check = current.merge(
        baseline_actual,
        on=keys,
        how="left",
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
    rng = np.random.default_rng(101_001 if league == "dk" else 102_001)
    n_players = len(players)
    counts = players["rows"].to_numpy(float)
    baseline_sse = players["sse__current_single"].to_numpy(float)
    rows = []
    for method in ENSEMBLE_METHODS[1:]:
        challenger_sse = players[f"sse__{method}"].to_numpy(float)
        deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for start in range(0, BOOTSTRAP_DRAWS, 500):
            stop = min(start + 500, BOOTSTRAP_DRAWS)
            sample = rng.integers(0, n_players, size=(stop - start, n_players))
            sampled_rows = counts[sample].sum(axis=1)
            base_rmse = np.sqrt(baseline_sse[sample].sum(axis=1) / sampled_rows)
            challenger_rmse = np.sqrt(challenger_sse[sample].sum(axis=1) / sampled_rows)
            deltas[start:stop] = challenger_rmse - base_rmse
        rows.append(
            {
                "league": league,
                "method": method,
                "comparison": f"{method}_vs_current_single",
                "rmse_delta": float(
                    np.sqrt(challenger_sse.sum() / counts.sum())
                    - np.sqrt(baseline_sse.sum() / counts.sum())
                ),
                "bootstrap_low": float(np.quantile(deltas, 0.025)),
                "bootstrap_high": float(np.quantile(deltas, 0.975)),
                "player_clusters": n_players,
                "draws": BOOTSTRAP_DRAWS,
            }
        )
    return pd.DataFrame(rows)


def _component_correlations(wide: pd.DataFrame, league: str) -> pd.DataFrame:
    component_names = [*CURRENT_FAMILIES, *CHALLENGER_FAMILIES]
    rows = []
    for left_index, left in enumerate(component_names):
        for right in component_names[left_index + 1 :]:
            rows.append(
                {
                    "league": league,
                    "component_left": left,
                    "component_right": right,
                    "prediction_correlation": float(wide[left].corr(wide[right])),
                    "error_correlation": float(
                        (wide[left] - wide["actual"]).corr(
                            wide[right] - wide["actual"]
                        )
                    ),
                }
            )
    current_prediction = wide[list(CURRENT_FAMILIES)].mean(axis=1)
    for challenger in CHALLENGER_FAMILIES:
        rows.append(
            {
                "league": league,
                "component_left": "current_primary_blend",
                "component_right": challenger,
                "prediction_correlation": float(current_prediction.corr(wide[challenger])),
                "error_correlation": float(
                    (current_prediction - wide["actual"]).corr(
                        wide[challenger] - wide["actual"]
                    )
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
    if set(test["season"]) != set(HOLDOUT_SEASONS):
        raise ValueError("Unexpected holdout seasons")

    current = _load_current_predictions(league, test)
    selection_frames = []
    selection_rows = []
    challenger_predictions: dict[str, np.ndarray] = {}
    for model_name in CHALLENGER_FAMILIES:
        print(f"{league}: selecting {model_name}", flush=True)
        candidate_id, selection_score, origin_scores = _select_candidate(
            train, model_name
        )
        parameters = MODEL_GRIDS[model_name][candidate_id]
        selection_frames.append(origin_scores.assign(league=league))
        selection_rows.append(
            {
                "league": league,
                "model_family": model_name,
                "selection_method": "current_mean_season_rmse",
                "candidate_id": candidate_id,
                "selection_score": selection_score,
                "parameters_json": json.dumps(parameters, sort_keys=True),
            }
        )
        print(
            f"{league}: refitting {model_name} candidate {candidate_id}",
            flush=True,
        )
        challenger_predictions[model_name] = _fit_predict(
            train, test, model_name, parameters
        )

    wide = current.copy()
    for model_name, prediction in challenger_predictions.items():
        wide[model_name] = prediction

    current_blend = wide[list(CURRENT_FAMILIES)].mean(axis=1).to_numpy(float)
    extra_blend = wide[
        [*CURRENT_FAMILIES, "conditional_ppg_extra_trees"]
    ].mean(axis=1).to_numpy(float)
    cat_blend = wide[
        [*CURRENT_FAMILIES, "conditional_ppg_catboost"]
    ].mean(axis=1).to_numpy(float)
    both_blend = wide[[*CURRENT_FAMILIES, *CHALLENGER_FAMILIES]].mean(axis=1).to_numpy(float)

    prediction_frames = []
    for model_name in CURRENT_FAMILIES:
        prediction_frames.append(
            _prediction_frame(
                wide, league, "current_single", model_name, wide[model_name].to_numpy(float)
            )
        )
    for model_name in CHALLENGER_FAMILIES:
        prediction_frames.append(
            _prediction_frame(
                wide, league, "challenger_standalone", model_name, wide[model_name].to_numpy(float)
            )
        )
    for method, prediction in (
        ("current_single", current_blend),
        ("current_plus_extra_trees_equal4", extra_blend),
        ("current_plus_catboost_equal4", cat_blend),
        ("current_plus_both_equal5", both_blend),
    ):
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

    pd.concat(selection_frames, ignore_index=True).to_csv(
        results_dir / "origin_candidate_scores.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        results_dir / "selected_parameters.csv", index=False
    )
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
        "candidate_count_per_family": 8,
        "ensemble_methods": list(ENSEMBLE_METHODS),
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
    print(key_scores.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or STUDY_DIR / f"results_{args.league}"
    run(args.league, results_dir)


if __name__ == "__main__":
    main()
