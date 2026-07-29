"""Attribute the LightGBM provider-family result to individual providers."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
BASE_MANIFEST = "residual_candidate_v1"
PROVIDER_FAMILY = "provider_projection"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with sqlite3.connect(OUTPUT_DB_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features", connection
        )
        manifests = pd.read_sql_query(
            "SELECT * FROM feature_manifests", connection
        )
    feature_run_ids = features["run_id"].dropna().astype(str).unique()
    manifest_run_ids = manifests["run_id"].dropna().astype(str).unique()
    if (
        len(feature_run_ids) != 1
        or set(manifest_run_ids) != set(feature_run_ids)
    ):
        raise ValueError("Feature tables do not share one active run ID")
    return features, manifests, str(feature_run_ids[0])


def _manifest_features(
    manifests: pd.DataFrame,
    manifest_name: str,
) -> tuple[str, ...]:
    return tuple(
        manifests.loc[
            manifests["manifest_name"].eq(manifest_name), "feature_name"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def _specification(variant: str) -> ModelSpec:
    return ModelSpec(
        CONDITIONAL_PPG_TARGET,
        f"direct_lightgbm_{variant}",
        "lightgbm",
        "direct",
        variant,
        "raw",
        "lgbm",
        {
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
        },
        4,
    )


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


def _comparisons(
    scores: pd.DataFrame,
    slices: pd.DataFrame,
    provider_features: tuple[str, ...],
) -> pd.DataFrame:
    reference = "direct_lightgbm_base"
    reference_rmse = _pooled_rmse(scores, reference)
    reference_seasons = _season_rmse(slices, reference)
    rows = []
    for index, feature_name in enumerate(provider_features):
        variant = feature_name.removeprefix("provider_").removesuffix(
            "_ppg_team_game"
        )
        model_name = f"direct_lightgbm_plus_{variant}"
        challenger_rmse = _pooled_rmse(scores, model_name)
        deltas = _season_rmse(slices, model_name) - reference_seasons
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
                "provider": variant,
                "feature_name": feature_name,
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
    return pd.DataFrame(rows).sort_values("pooled_delta")


def main() -> None:
    features, manifests, feature_run_id = _load_inputs()
    target = build_target_frames(features, VALIDATION_END)[
        CONDITIONAL_PPG_TARGET
    ]
    base_features = _manifest_features(manifests, BASE_MANIFEST)
    provider_features = tuple(
        manifests.loc[
            manifests["family"].eq(PROVIDER_FAMILY), "feature_name"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not provider_features:
        raise ValueError(f"No {PROVIDER_FAMILY} features found")

    variants = [("base", base_features)]
    variants.extend(
        (
            "plus_"
            + feature_name.removeprefix("provider_").removesuffix(
                "_ppg_team_game"
            ),
            tuple(dict.fromkeys((*base_features, feature_name))),
        )
        for feature_name in provider_features
    )

    run_id = create_run_id("m4a_provider_lightgbm_attribution")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    oof_frames = []
    parameter_frames = []
    for index, (variant, columns) in enumerate(variants, start=1):
        spec = _specification(variant)
        print(
            f"[{index}/{len(variants)}] {spec.model_name}",
            flush=True,
        )
        oof, parameters = run_model_spec(
            target,
            assignments,
            spec,
            columns + POSITION_FEATURES,
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
    comparisons = _comparisons(scores, slices, provider_features)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(
        RESULTS_DIR / "individual_provider_lightgbm_oof_predictions.csv",
        index=False,
    )
    scores.to_csv(
        RESULTS_DIR / "individual_provider_lightgbm_model_scores.csv",
        index=False,
    )
    slices.to_csv(
        RESULTS_DIR / "individual_provider_lightgbm_model_slices.csv",
        index=False,
    )
    hyperparameters.to_csv(
        RESULTS_DIR / "individual_provider_lightgbm_hyperparameters.csv",
        index=False,
    )
    comparisons.to_csv(
        RESULTS_DIR / "individual_provider_lightgbm_comparisons.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "target_rows": len(target),
                "experiments": len(variants),
                "results_directory": str(RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
