"""Run pooled KNN and random-forest V2 conditional-PPG challengers."""

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
    build_target_frames,
    make_fold_assignments,
    run_model_spec,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
LIGHTGBM_RESULTS = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_projection_consensus_ladder"
    / "results"
    / "oof_predictions.csv"
)
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234
LIGHTGBM_MODELS = {
    "projection_core_lightgbm": "projection_only_lightgbm_core",
    "full_lightgbm": "full_lightgbm_base",
}

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

MODEL_CONFIGS = {
    "knn": {
        "model_piece": "knn",
        "search_iterations": 12,
        "parameters": {
            "knn__n_neighbors": (15, 35, 75),
            "knn__weights": ("uniform", "distance"),
            "knn__p": (1, 2),
            "knn__algorithm": ("brute",),
        },
    },
    "random_forest": {
        "model_piece": "rf",
        "search_iterations": 8,
        "parameters": {
            "rf__n_estimators": (250,),
            "rf__max_depth": (6, 10),
            "rf__min_samples_leaf": (5, 15),
            "rf__max_features": (0.5, 1.0),
            "rf__bootstrap": (True,),
            "rf__random_state": (RANDOM_SEED,),
            "rf__n_jobs": (1,),
        },
    },
}


def _load_inputs() -> tuple[pd.DataFrame, tuple[str, ...], str]:
    with sqlite3.connect(OUTPUT_DB_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features",
            connection,
        )
        manifests = pd.read_sql_query(
            "SELECT * FROM feature_manifests",
            connection,
        )
    run_ids = features["run_id"].dropna().astype(str).unique()
    if len(run_ids) != 1:
        raise ValueError("Expected one active feature run")
    feature_run_id = str(run_ids[0])
    if set(
        manifests["run_id"].dropna().astype(str).unique()
    ) != {feature_run_id}:
        raise ValueError("Feature manifest lineage does not match the mart")
    full_features = tuple(
        manifests.loc[
            manifests["manifest_name"].eq("residual_candidate_v1"),
            "feature_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(full_features) != 31:
        raise ValueError(
            f"Expected 31 full features, found {len(full_features)}"
        )
    return features, full_features, feature_run_id


def _spec(model_family: str, feature_set: str) -> ModelSpec:
    config = MODEL_CONFIGS[model_family]
    return ModelSpec(
        target_name=CONDITIONAL_PPG_TARGET,
        model_name=f"{feature_set}_{model_family}",
        model_family=model_family,
        prediction_kind="direct",
        feature_set=feature_set,
        pipeline_variant="raw",
        model_piece=str(config["model_piece"]),
        parameters=config["parameters"],
        search_iterations=int(config["search_iterations"]),
    )


def _history_depth(frame: pd.DataFrame) -> pd.Series:
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    prior = pd.to_numeric(
        frame["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    result = pd.Series(
        "other_no_history",
        index=frame.index,
        dtype=object,
    )
    result.loc[rookie] = "rookie"
    result.loc[~rookie & year_exp.eq(1)] = "second_year"
    result.loc[~rookie & year_exp.ge(2) & prior] = (
        "veteran_with_history"
    )
    result.loc[year_exp.isna()] = "unknown_experience"
    return result


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    error = (
        pd.to_numeric(prediction, errors="coerce")
        - pd.to_numeric(actual, errors="coerce")
    )
    return float(np.sqrt(np.square(error).mean()))


def _score_rows(
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    slice_type: str,
    slice_column: str | None,
) -> list[dict[str, object]]:
    groups = (
        [("all", frame)]
        if slice_column is None
        else frame.groupby(slice_column, dropna=False)
    )
    rows = []
    for slice_value, group in groups:
        actual = pd.to_numeric(group["actual"], errors="coerce")
        for method in methods:
            prediction = pd.to_numeric(
                group[method],
                errors="coerce",
            )
            rows.extend(
                (
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "rmse",
                        "n_rows": len(group),
                        "value": _rmse(actual, prediction),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "mae",
                        "n_rows": len(group),
                        "value": float(
                            (prediction - actual).abs().mean()
                        ),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "bias",
                        "n_rows": len(group),
                        "value": float((prediction - actual).mean()),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "spearman",
                        "n_rows": len(group),
                        "value": float(
                            prediction.corr(actual, method="spearman")
                        ),
                    },
                )
            )
    return rows


def _score_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    methods = tuple(
        column
        for column in frame.columns
        if column
        in {
            "expert_baseline",
            "projection_core_lightgbm",
            "full_lightgbm",
            "projection_core_knn",
            "full_knn",
            "projection_core_random_forest",
            "full_random_forest",
            "projection_core_knn_lgbm_average",
            "full_knn_lgbm_average",
            "projection_core_rf_lgbm_average",
            "full_rf_lgbm_average",
        }
    )
    rows = _score_rows(frame, methods, "pooled", None)
    for slice_type, column in (
        ("season", "season"),
        ("position", "position"),
        ("history_depth", "history_depth"),
        ("history_group", "history_group"),
    ):
        rows.extend(
            _score_rows(
                frame,
                methods,
                slice_type,
                column,
            )
        )
    return pd.DataFrame(rows)


def _comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].set_index("method")["value"]
    season = scores[
        scores["slice_type"].eq("season")
        & scores["metric"].eq("rmse")
    ].pivot(
        index="slice_value",
        columns="method",
        values="value",
    )
    pairs = (
        ("projection_core_knn", "projection_core_lightgbm"),
        ("full_knn", "full_lightgbm"),
        (
            "projection_core_random_forest",
            "projection_core_lightgbm",
        ),
        ("full_random_forest", "full_lightgbm"),
        (
            "projection_core_knn_lgbm_average",
            "projection_core_lightgbm",
        ),
        ("full_knn_lgbm_average", "full_lightgbm"),
        (
            "projection_core_rf_lgbm_average",
            "projection_core_lightgbm",
        ),
        ("full_rf_lgbm_average", "full_lightgbm"),
    )
    rows = []
    for index, (challenger, reference) in enumerate(pairs):
        delta = (
            season[challenger] - season[reference]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + index)
        draws = np.array(
            [
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        recent = season.index.astype(int) >= 2023
        rows.append(
            {
                "challenger": challenger,
                "reference": reference,
                "challenger_rmse": float(pooled[challenger]),
                "reference_rmse": float(pooled[reference]),
                "pooled_delta": float(
                    pooled[challenger] - pooled[reference]
                ),
                "mean_season_delta": float(delta.mean()),
                "recent_mean_season_delta": float(
                    delta[recent].mean()
                ),
                "season_wins": int((delta < 0).sum()),
                "season_count": len(delta),
                "bootstrap_95_low": float(
                    np.quantile(draws, 0.025)
                ),
                "bootstrap_95_high": float(
                    np.quantile(draws, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows)


def _error_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    methods = (
        "projection_core_lightgbm",
        "full_lightgbm",
        "projection_core_knn",
        "full_knn",
        "projection_core_random_forest",
        "full_random_forest",
    )
    errors = pd.DataFrame(
        {
            method: pd.to_numeric(
                frame[method],
                errors="coerce",
            )
            - pd.to_numeric(frame["actual"], errors="coerce")
            for method in methods
        }
    )
    return errors.corr().rename_axis("method").reset_index()


def _summary_markdown(
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    lines = [
        "# KNN and Random-Forest Results",
        "",
        "Negative deltas favor the challenger.",
        "",
        "## Pooled OOF",
        "",
        "| Method | RMSE |",
        "|---|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| `{row.method}` | {row.value:.4f} |")
    lines.extend(
        (
            "",
            "## Paired season comparisons",
            "",
            "| Challenger | Reference | Delta | Recent delta | "
            "95% interval | Wins |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.challenger}` | `{row.reference}` | "
            f"{row.pooled_delta:+.4f} | "
            f"{row.recent_mean_season_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.season_wins}/{row.season_count} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features, full_features, feature_run_id = _load_inputs()
    target = build_target_frames(
        features,
        VALIDATION_END,
    )[CONDITIONAL_PPG_TARGET]
    run_id = create_run_id("m4a_knn_rf_models")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    feature_sets = {
        "projection_core": tuple(
            dict.fromkeys(
                (*PROJECTION_CORE_FEATURES, *POSITION_FEATURES)
            )
        ),
        "full": tuple(
            dict.fromkeys((*full_features, *POSITION_FEATURES))
        ),
    }

    oof_parts = []
    parameter_parts = []
    specification_rows = []
    experiments = [
        (model_family, feature_set)
        for model_family in MODEL_CONFIGS
        for feature_set in feature_sets
    ]
    for index, (model_family, feature_set) in enumerate(
        experiments,
        start=1,
    ):
        feature_columns = feature_sets[feature_set]
        spec = _spec(model_family, feature_set)
        print(
            f"[{index}/{len(experiments)}] {spec.model_name} "
            f"({len(feature_columns)} features)",
            flush=True,
        )
        oof, parameters = run_model_spec(
            target,
            assignments,
            spec,
            feature_columns,
            run_id,
            feature_run_id,
            VALIDATION_START,
            N_SPLITS,
            RANDOM_SEED,
        )
        oof_parts.append(oof)
        parameter_parts.append(parameters)
        specification_rows.append(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "model_name": spec.model_name,
                "model_family": model_family,
                "feature_set": feature_set,
                "feature_count": len(feature_columns),
                "feature_names_json": json.dumps(feature_columns),
                "hyperparameters_json": json.dumps(
                    MODEL_CONFIGS[model_family]["parameters"],
                    default=list,
                    sort_keys=True,
                ),
                "search_iterations": spec.search_iterations,
            }
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    parameters = pd.concat(parameter_parts, ignore_index=True)
    keys = ["player_key", "season"]
    prediction_wide = oof.pivot(
        index=keys,
        columns="model_name",
        values="final_prediction",
    ).reset_index()
    metadata = oof[
        oof["model_name"].eq("projection_core_knn")
    ][
        [
            *keys,
            "position",
            "actual",
            "baseline_prediction",
            "year_exp",
            "is_rookie",
            "has_prior_outcome",
        ]
    ].rename(columns={"baseline_prediction": "expert_baseline"})
    lightgbm = pd.read_csv(LIGHTGBM_RESULTS)
    lightgbm = lightgbm[
        lightgbm["model_name"].isin(LIGHTGBM_MODELS.values())
    ].copy()
    lightgbm["method"] = lightgbm["model_name"].map(
        {value: key for key, value in LIGHTGBM_MODELS.items()}
    )
    lightgbm_wide = lightgbm.pivot(
        index=keys,
        columns="method",
        values="final_prediction",
    ).reset_index()
    comparison = (
        metadata.merge(
            prediction_wide,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            lightgbm_wide,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    )
    expected = int(
        target["season"].between(
            VALIDATION_START,
            VALIDATION_END,
        ).sum()
    )
    if len(comparison) != expected:
        raise ValueError(
            f"Expected {expected} paired OOF rows, found "
            f"{len(comparison)}"
        )
    comparison["projection_core_knn_lgbm_average"] = (
        comparison["projection_core_knn"]
        + comparison["projection_core_lightgbm"]
    ) / 2
    comparison["full_knn_lgbm_average"] = (
        comparison["full_knn"] + comparison["full_lightgbm"]
    ) / 2
    comparison["projection_core_rf_lgbm_average"] = (
        comparison["projection_core_random_forest"]
        + comparison["projection_core_lightgbm"]
    ) / 2
    comparison["full_rf_lgbm_average"] = (
        comparison["full_random_forest"]
        + comparison["full_lightgbm"]
    ) / 2
    comparison["history_depth"] = _history_depth(comparison)
    comparison["history_group"] = np.where(
        comparison["history_depth"].isin(
            ("rookie", "second_year", "other_no_history")
        ),
        "limited",
        "veteran",
    )
    scores = _score_predictions(comparison)
    comparisons = _comparisons(scores)
    correlations = _error_correlations(comparison)

    oof.to_csv(RESULTS_DIR / "model_oof.csv", index=False)
    parameters.to_csv(
        RESULTS_DIR / "hyperparameters.csv",
        index=False,
    )
    assignments.to_csv(
        RESULTS_DIR / "fold_assignments.csv",
        index=False,
    )
    pd.DataFrame(specification_rows).to_csv(
        RESULTS_DIR / "model_specifications.csv",
        index=False,
    )
    comparison.to_csv(
        RESULTS_DIR / "comparison_predictions.csv",
        index=False,
    )
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    comparisons.to_csv(
        RESULTS_DIR / "model_comparisons.csv",
        index=False,
    )
    correlations.to_csv(
        RESULTS_DIR / "error_correlations.csv",
        index=False,
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, comparisons),
        encoding="utf-8",
    )
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
