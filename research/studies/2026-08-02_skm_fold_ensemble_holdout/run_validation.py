"""Compare current single fits with deployment-matched SKM fold/seed bags."""

from __future__ import annotations

import argparse
import gc
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import (
    LOCKED_RANDOM_SEED,
    MODEL_GRIDS,
    PRIMARY_PPG_FEATURES,
)
from Scripts.V2.modeling import add_modeling_features
from Scripts.V2.native_runtime import RANDOM_FOREST_N_JOBS


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
MODEL_FAMILIES = (
    "conditional_ppg_lasso",
    "conditional_ppg_random_forest",
    "conditional_ppg_lightgbm",
)
METHODS = (
    "current_single",
    "current_seed_bag",
    "skm_fold_param_bag",
    "skm_fold_seed_bag",
)
TUNING_ORIGINS = tuple(range(2013, 2023))
HOLDOUT_SEASONS = (2023, 2024, 2025)
N_FOLDS = 5
FOLD_SEEDS = {
    "conditional_ppg_lasso": 133,
    "conditional_ppg_random_forest": 170,
    "conditional_ppg_lightgbm": 201,
}
BOOTSTRAP_DRAWS = 20_000


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
    missing = [column for column in PRIMARY_PPG_FEATURES if column not in frame]
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
    if not set(HOLDOUT_SEASONS).issubset(set(frame["season"])):
        raise ValueError("Incomplete sealed holdout")
    return frame


def _member_seed(model_name: str, member: int) -> int:
    family_offset = MODEL_FAMILIES.index(model_name) * 1_000
    return LOCKED_RANDOM_SEED + family_offset + member * 101


def _pipeline(
    model_name: str,
    parameters: Mapping[str, object],
    estimator_seed: int,
) -> Pipeline:
    parameters = dict(parameters)
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
    if model_name == "conditional_ppg_lasso":
        steps.append(("scale", StandardScaler()))
        estimator = Lasso(max_iter=50_000, tol=1e-6, **parameters)
    elif model_name == "conditional_ppg_random_forest":
        parameters.pop("random_state", None)
        estimator = RandomForestRegressor(
            n_jobs=RANDOM_FOREST_N_JOBS,
            random_state=estimator_seed,
            **parameters,
        )
    elif model_name == "conditional_ppg_lightgbm":
        estimator = LGBMRegressor(
            objective="regression",
            verbosity=-1,
            subsample=1.0,
            colsample_bytree=1.0,
            deterministic=True,
            force_col_wise=True,
            random_state=estimator_seed,
            n_jobs=1,
            **parameters,
        )
    else:
        raise ValueError(f"Unsupported model family: {model_name}")
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    if model_name.endswith("lightgbm"):
        pipeline.set_output(transform="pandas")
    return pipeline


def _fit_predict(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    model_name: str,
    parameters: Mapping[str, object],
    estimator_seed: int,
) -> np.ndarray:
    features = list(PRIMARY_PPG_FEATURES)
    model = _pipeline(model_name, parameters, estimator_seed)
    model.fit(
        train[features].apply(pd.to_numeric, errors="coerce"),
        pd.to_numeric(train["actual"], errors="raise"),
    )
    prediction = model.predict(
        predict[features].apply(pd.to_numeric, errors="coerce")
    )
    del model
    if model_name.endswith("lightgbm"):
        gc.collect()
    return np.asarray(prediction, dtype=float)


def _origin_candidate_scores(
    selector: pd.DataFrame,
    model_name: str,
    origin: int,
) -> pd.DataFrame:
    train = selector[selector["season"].lt(origin)]
    hold = selector[selector["season"].eq(origin)]
    if train.empty or hold.empty:
        raise ValueError(f"Missing selection rows for {model_name}/{origin}")
    rows = []
    for candidate_id, parameters in enumerate(MODEL_GRIDS[model_name]):
        prediction = _fit_predict(
            train,
            hold,
            model_name,
            parameters,
            LOCKED_RANDOM_SEED,
        )
        actual = hold["actual"].to_numpy(float)
        error = actual - prediction
        rows.append(
            {
                "candidate_id": candidate_id,
                "origin": origin,
                "rows": len(hold),
                "sse": float(np.square(error).sum()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
            }
        )
    return pd.DataFrame(rows)


def _grid_scores(selector: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return pd.concat(
        [
            _origin_candidate_scores(selector, model_name, origin)
            for origin in TUNING_ORIGINS
        ],
        ignore_index=True,
    )


def _select_current(scores: pd.DataFrame) -> tuple[int, float]:
    ranked = scores.groupby("candidate_id", as_index=False).agg(
        selection_score=("rmse", "mean")
    )
    winner = ranked.sort_values(["selection_score", "candidate_id"]).iloc[0]
    return int(winner["candidate_id"]), float(winner["selection_score"])


def _select_legacy_fold(scores: pd.DataFrame) -> tuple[int, float]:
    ranked = scores.groupby("candidate_id", as_index=False).agg(
        sse=("sse", "sum"), rows=("rows", "sum")
    )
    ranked["selection_score"] = np.sqrt(ranked["sse"] / ranked["rows"])
    winner = ranked.sort_values(["selection_score", "candidate_id"]).iloc[0]
    return int(winner["candidate_id"]), float(winner["selection_score"])


def _fold_assignments(train: pd.DataFrame, model_name: str) -> pd.Series:
    development = train[train["season"].isin(TUNING_ORIGINS)].copy()
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=FOLD_SEEDS[model_name],
    )
    assignments = pd.Series(-1, index=train.index, dtype=int)
    for fold, (_, hold_positions) in enumerate(
        splitter.split(development, development["season"])
    ):
        assignments.loc[development.iloc[hold_positions].index] = fold
    if not assignments.loc[development.index].between(0, N_FOLDS - 1).all():
        raise ValueError(f"Incomplete fold assignments for {model_name}")
    return assignments


def _prediction_frame(
    test: pd.DataFrame,
    league: str,
    method: str,
    model_name: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    output = test[["player_key", "season", "position", "actual"]].copy()
    output.insert(0, "league", league)
    output["method"] = method
    output["model_family"] = model_name
    output["prediction"] = prediction
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
            "spearman": float(pd.Series(actual).corr(pd.Series(prediction), method="spearman")),
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
        scored = (
            current.groupby(
                ["league", "method", "model_family", "slice_type", "slice_value"],
                sort=True,
            )
            .apply(_score_group, include_groups=False)
            .reset_index()
        )
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def _player_cluster_bootstrap(
    predictions: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    blend = predictions[predictions["model_family"].eq("primary_blend")]
    wide = blend.pivot(
        index=["player_key", "season", "actual"],
        columns="method",
        values="prediction",
    ).reset_index()
    baseline = "current_single"
    player_rows = []
    for player_key, group in wide.groupby("player_key", sort=False):
        record = {"player_key": player_key, "rows": len(group)}
        actual = group["actual"].to_numpy(float)
        for method in METHODS:
            record[f"sse__{method}"] = float(
                np.square(actual - group[method].to_numpy(float)).sum()
            )
        player_rows.append(record)
    players = pd.DataFrame(player_rows)
    rng = np.random.default_rng(91_001 if league == "dk" else 92_001)
    n_players = len(players)
    counts = players["rows"].to_numpy(float)
    rows = []
    for method in METHODS[1:]:
        baseline_sse = players[f"sse__{baseline}"].to_numpy(float)
        challenger_sse = players[f"sse__{method}"].to_numpy(float)
        deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for start in range(0, BOOTSTRAP_DRAWS, 500):
            stop = min(start + 500, BOOTSTRAP_DRAWS)
            sample = rng.integers(0, n_players, size=(stop - start, n_players))
            sampled_rows = counts[sample].sum(axis=1)
            base_rmse = np.sqrt(baseline_sse[sample].sum(axis=1) / sampled_rows)
            challenger_rmse = np.sqrt(challenger_sse[sample].sum(axis=1) / sampled_rows)
            deltas[start:stop] = challenger_rmse - base_rmse
        point = float(
            np.sqrt(challenger_sse.sum() / counts.sum())
            - np.sqrt(baseline_sse.sum() / counts.sum())
        )
        rows.append(
            {
                "league": league,
                "method": method,
                "comparison": f"{method}_vs_{baseline}",
                "rmse_delta": point,
                "bootstrap_low": float(np.quantile(deltas, 0.025)),
                "bootstrap_high": float(np.quantile(deltas, 0.975)),
                "player_clusters": n_players,
                "draws": BOOTSTRAP_DRAWS,
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

    prediction_frames = []
    selection_rows = []
    fold_summary_rows = []

    for model_name in MODEL_FAMILIES:
        print(f"{league}: selecting current {model_name}", flush=True)
        current_scores = _grid_scores(train, model_name)
        current_id, current_score = _select_current(current_scores)
        current_params = MODEL_GRIDS[model_name][current_id]
        selection_rows.append(
            {
                "league": league,
                "model_family": model_name,
                "selection_method": "current_mean_season_rmse",
                "fold": pd.NA,
                "candidate_id": current_id,
                "selection_score": current_score,
                "parameters_json": json.dumps(current_params, sort_keys=True),
            }
        )

        current_prediction = _fit_predict(
            train, test, model_name, current_params, LOCKED_RANDOM_SEED
        )
        prediction_frames.append(
            _prediction_frame(test, league, "current_single", model_name, current_prediction)
        )
        seeded_members = [
            _fit_predict(
                train,
                test,
                model_name,
                current_params,
                _member_seed(model_name, member),
            )
            for member in range(N_FOLDS)
        ]
        prediction_frames.append(
            _prediction_frame(
                test,
                league,
                "current_seed_bag",
                model_name,
                np.mean(seeded_members, axis=0),
            )
        )

        assignments = _fold_assignments(train, model_name)
        development = train[train["season"].isin(TUNING_ORIGINS)]
        for fold in range(N_FOLDS):
            held = development.loc[assignments.loc[development.index].eq(fold)]
            fold_summary_rows.append(
                {
                    "league": league,
                    "model_family": model_name,
                    "fold": fold,
                    "fold_seed": FOLD_SEEDS[model_name],
                    "held_rows": len(held),
                    "held_start_season": int(held["season"].min()),
                    "held_end_season": int(held["season"].max()),
                }
            )

        fold_params = []
        for fold in range(N_FOLDS):
            print(f"{league}: selecting SKM {model_name} fold {fold + 1}/{N_FOLDS}", flush=True)
            selector = train[
                ~assignments.eq(fold)
            ].copy()
            scores = _grid_scores(selector, model_name)
            candidate_id, selection_score = _select_legacy_fold(scores)
            parameters = MODEL_GRIDS[model_name][candidate_id]
            fold_params.append(parameters)
            selection_rows.append(
                {
                    "league": league,
                    "model_family": model_name,
                    "selection_method": "skm_fold_pooled_rmse",
                    "fold": fold,
                    "candidate_id": candidate_id,
                    "selection_score": selection_score,
                    "parameters_json": json.dumps(parameters, sort_keys=True),
                }
            )

        fixed_members = [
            _fit_predict(
                train, test, model_name, parameters, LOCKED_RANDOM_SEED
            )
            for parameters in fold_params
        ]
        seeded_fold_members = [
            _fit_predict(
                train,
                test,
                model_name,
                parameters,
                _member_seed(model_name, member),
            )
            for member, parameters in enumerate(fold_params)
        ]
        prediction_frames.extend(
            [
                _prediction_frame(
                    test,
                    league,
                    "skm_fold_param_bag",
                    model_name,
                    np.mean(fixed_members, axis=0),
                ),
                _prediction_frame(
                    test,
                    league,
                    "skm_fold_seed_bag",
                    model_name,
                    np.mean(seeded_fold_members, axis=0),
                ),
            ]
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    blend_frames = []
    for method in METHODS:
        current = predictions[predictions["method"].eq(method)]
        wide = current.pivot(
            index=["league", "player_key", "season", "position", "actual"],
            columns="model_family",
            values="prediction",
        ).reset_index()
        if wide[list(MODEL_FAMILIES)].isna().any().any():
            raise ValueError(f"Incomplete family predictions for {method}")
        wide["prediction"] = wide[list(MODEL_FAMILIES)].mean(axis=1)
        wide["method"] = method
        wide["model_family"] = "primary_blend"
        blend_frames.append(
            wide[
                [
                    "league", "player_key", "season", "position", "actual",
                    "method", "model_family", "prediction",
                ]
            ]
        )
    predictions = pd.concat([predictions, *blend_frames], ignore_index=True)
    predictions.sort_values(
        ["model_family", "method", "season", "player_key"], inplace=True
    )
    predictions.reset_index(drop=True, inplace=True)

    scores = _scores(predictions)
    bootstrap = _player_cluster_bootstrap(predictions, league)
    pd.DataFrame(selection_rows).to_csv(results_dir / "selected_parameters.csv", index=False)
    pd.DataFrame(fold_summary_rows).to_csv(results_dir / "fold_summary.csv", index=False)
    predictions.to_csv(results_dir / "holdout_predictions.csv", index=False)
    scores.to_csv(results_dir / "scores.csv", index=False)
    bootstrap.to_csv(results_dir / "player_cluster_bootstrap.csv", index=False)
    metadata = {
        "league": league,
        "training_max_season": int(train["season"].max()),
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "tuning_origins": list(TUNING_ORIGINS),
        "training_rows": len(train),
        "holdout_rows": len(test),
        "folds": N_FOLDS,
        "feature_count": len(PRIMARY_PPG_FEATURES),
        "methods": list(METHODS),
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    blend_scores = scores[
        scores["model_family"].eq("primary_blend")
        & scores["slice_type"].isin(["all", "season"])
    ]
    print(blend_scores.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or STUDY_DIR / f"results_{args.league}"
    run(args.league, results_dir)


if __name__ == "__main__":
    main()
