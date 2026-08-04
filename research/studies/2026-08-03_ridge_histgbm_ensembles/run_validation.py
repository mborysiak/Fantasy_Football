"""Test wider Ridge and HistGradientBoosting on the locked PPG surface."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
BASELINE_STUDY = (
    REPO_ROOT / "research" / "studies" / "2026-08-02_skm_fold_ensemble_holdout"
)
BASE_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-02_extra_trees_cb_ensembles"
    / "run_validation.py"
)


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "extra_trees_cb_validation_helpers", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load validation helpers from {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_runner()
PRIMARY_PPG_FEATURES = base.PRIMARY_PPG_FEATURES
CURRENT_FAMILIES = base.CURRENT_FAMILIES
LOCKED_RANDOM_SEED = base.LOCKED_RANDOM_SEED
TUNING_ORIGINS = base.TUNING_ORIGINS
HOLDOUT_SEASONS = base.HOLDOUT_SEASONS

CHALLENGER_FAMILIES = (
    "conditional_ppg_ridge",
    "conditional_ppg_hist_gradient_boosting",
)
ENSEMBLE_METHODS = (
    "current_single",
    "ridge_replaces_lasso_equal3",
    "lasso_ridge_split_linear_third",
    "current_plus_ridge_equal4",
    "histgbm_replaces_lightgbm_equal3",
    "current_plus_histgbm_equal4",
)

RIDGE_GRID = tuple(
    {"alpha": alpha}
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
)
HISTGBM_GRID = tuple(
    {
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": 20,
        "l2_regularization": l2_regularization,
    }
    for (
        max_iter,
        learning_rate,
        max_depth,
        max_leaf_nodes,
    ), l2_regularization in product(
        ((150, 0.03, 3, 7), (100, 0.05, 4, 15)),
        (0.0, 0.1, 1.0, 10.0, 100.0),
    )
)
MODEL_GRIDS = {
    "conditional_ppg_ridge": RIDGE_GRID,
    "conditional_ppg_hist_gradient_boosting": HISTGBM_GRID,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=sorted(base.DATABASES), required=True)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args()


def _pipeline(model_name: str, parameters: Mapping[str, object]) -> Pipeline:
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
    if model_name == "conditional_ppg_ridge":
        steps.extend(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(max_iter=10_000, **dict(parameters))),
            ]
        )
    elif model_name == "conditional_ppg_hist_gradient_boosting":
        steps.append(
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    early_stopping=False,
                    random_state=LOCKED_RANDOM_SEED,
                    **dict(parameters),
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported model family: {model_name}")
    return Pipeline(steps)


def _fit_predict(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    model_name: str,
    parameters: Mapping[str, object],
) -> np.ndarray:
    features = list(PRIMARY_PPG_FEATURES)
    model = _pipeline(model_name, parameters)
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
                origin_train,
                origin_hold,
                model_name,
                parameters,
            )
            rows.append(
                {
                    "model_family": model_name,
                    "candidate_id": candidate_id,
                    "origin": origin,
                    "rows": len(origin_hold),
                    "rmse": float(
                        np.sqrt(np.mean(np.square(actual - prediction)))
                    ),
                    "parameters_json": json.dumps(parameters, sort_keys=True),
                }
            )
    scores = pd.DataFrame(rows)
    ranked = scores.groupby("candidate_id", as_index=False).agg(
        selection_score=("rmse", "mean")
    )
    winner = ranked.sort_values(["selection_score", "candidate_id"]).iloc[0]
    return int(winner["candidate_id"]), float(winner["selection_score"]), scores


def run(league: str, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = base._load_target(league)
    train = target[target["season"].lt(min(HOLDOUT_SEASONS))].copy()
    test = target[target["season"].isin(HOLDOUT_SEASONS)].copy()
    if int(train["season"].max()) != 2022:
        raise ValueError("Training boundary is not sealed at 2022")
    if set(test["season"]) != set(HOLDOUT_SEASONS):
        raise ValueError("Unexpected holdout seasons")

    current = base._load_current_predictions(league, test)
    selection_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
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
        print(f"{league}: refitting {model_name} candidate {candidate_id}", flush=True)
        challenger_predictions[model_name] = _fit_predict(
            train, test, model_name, parameters
        )

    wide = current.copy()
    for model_name, prediction in challenger_predictions.items():
        wide[model_name] = prediction

    lasso = wide["conditional_ppg_lasso"].to_numpy(float)
    random_forest = wide["conditional_ppg_random_forest"].to_numpy(float)
    lightgbm = wide["conditional_ppg_lightgbm"].to_numpy(float)
    ridge = wide["conditional_ppg_ridge"].to_numpy(float)
    histgbm = wide["conditional_ppg_hist_gradient_boosting"].to_numpy(float)
    method_predictions = {
        "current_single": (lasso + random_forest + lightgbm) / 3.0,
        "ridge_replaces_lasso_equal3": (ridge + random_forest + lightgbm) / 3.0,
        "lasso_ridge_split_linear_third": (
            (lasso + ridge) / 2.0 + random_forest + lightgbm
        )
        / 3.0,
        "current_plus_ridge_equal4": (lasso + random_forest + lightgbm + ridge)
        / 4.0,
        "histgbm_replaces_lightgbm_equal3": (lasso + random_forest + histgbm)
        / 3.0,
        "current_plus_histgbm_equal4": (
            lasso + random_forest + lightgbm + histgbm
        )
        / 4.0,
    }

    prediction_frames = []
    for model_name in CURRENT_FAMILIES:
        prediction_frames.append(
            base._prediction_frame(
                wide,
                league,
                "current_single",
                model_name,
                wide[model_name].to_numpy(float),
            )
        )
    for model_name in CHALLENGER_FAMILIES:
        prediction_frames.append(
            base._prediction_frame(
                wide,
                league,
                "challenger_standalone",
                model_name,
                wide[model_name].to_numpy(float),
            )
        )
    for method, prediction in method_predictions.items():
        prediction_frames.append(
            base._prediction_frame(
                wide, league, method, "primary_blend", prediction
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.sort_values(
        ["model_family", "method", "season", "player_key"], inplace=True
    )
    predictions.reset_index(drop=True, inplace=True)
    scores = base._scores(predictions)
    base.ENSEMBLE_METHODS = ENSEMBLE_METHODS
    base.CHALLENGER_FAMILIES = CHALLENGER_FAMILIES
    bootstrap = base._player_cluster_bootstrap(predictions, league)
    correlations = base._component_correlations(wide, league)

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
        "candidate_counts": {
            model_name: len(grid) for model_name, grid in MODEL_GRIDS.items()
        },
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

