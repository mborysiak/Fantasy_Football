"""Test standardized provider, projection-shape, and disagreement features."""

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
CHALLENGER_MANIFEST = "residual_projection_challenger_v1"
MODEL_FAMILIES = ("lasso", "lightgbm")


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
    feature_run_ids = features["run_id"].dropna().astype(str).unique()
    manifest_run_ids = manifests["run_id"].dropna().astype(str).unique()
    projection_run_ids = (
        projection_values["run_id"].dropna().astype(str).unique()
    )
    if (
        len(feature_run_ids) != 1
        or set(manifest_run_ids) != set(feature_run_ids)
        or set(projection_run_ids) != set(feature_run_ids)
    ):
        raise ValueError("Feature tables do not share one active run ID")
    for manifest_name in (BASE_MANIFEST, CHALLENGER_MANIFEST):
        if not manifests["manifest_name"].eq(manifest_name).any():
            raise ValueError(f"Missing feature manifest: {manifest_name}")
    return (
        features,
        manifests,
        projection_values,
        str(feature_run_ids[0]),
    )


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


def _feature_variants(
    manifests: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    base = _manifest_features(manifests, BASE_MANIFEST)
    challengers = manifests[
        manifests["manifest_name"].eq(CHALLENGER_MANIFEST)
    ]
    variants: dict[str, tuple[str, ...]] = {"base": base}
    for family, group in challengers.groupby("family", sort=True):
        additions = tuple(
            group["feature_name"].drop_duplicates().sort_values().tolist()
        )
        variants[f"plus_{family}"] = tuple(
            dict.fromkeys((*base, *additions))
        )
    all_additions = tuple(
        challengers["feature_name"].drop_duplicates().sort_values().tolist()
    )
    variants["plus_all"] = tuple(
        dict.fromkeys((*base, *all_additions))
    )
    return variants


def _model_specification(
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
        search_iterations = 20
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
        search_iterations = 4
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    return ModelSpec(
        CONDITIONAL_PPG_TARGET,
        f"direct_{model_family}_{variant}",
        model_family,
        "direct",
        variant,
        "raw",
        model_piece,
        parameters,
        search_iterations,
    )


def _experiments(
    manifests: pd.DataFrame,
) -> list[tuple[ModelSpec, tuple[str, ...], int]]:
    variants = _feature_variants(manifests)
    base_count = len(variants["base"])
    return [
        (
            _model_specification(model_family, variant),
            features + POSITION_FEATURES,
            len(features) - base_count,
        )
        for model_family in MODEL_FAMILIES
        for variant, features in variants.items()
    ]


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
    experiments: list[tuple[ModelSpec, tuple[str, ...], int]],
) -> pd.DataFrame:
    rows = []
    for index, (spec, features, added_count) in enumerate(experiments):
        if spec.feature_set == "base":
            continue
        reference = f"direct_{spec.model_family}_base"
        challenger_rmse = _pooled_rmse(scores, spec.model_name)
        reference_rmse = _pooled_rmse(scores, reference)
        deltas = _season_rmse(slices, spec.model_name) - _season_rmse(
            slices, reference
        )
        rng = np.random.default_rng(RANDOM_SEED + index)
        values = deltas.to_numpy(dtype=float)
        draws = np.array(
            [
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "model_family": spec.model_family,
                "variant": spec.feature_set,
                "added_feature_count": added_count,
                "total_feature_count": len(features),
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
        ["model_family", "pooled_delta"]
    )


def _feature_coverage(
    features: pd.DataFrame,
    manifests: pd.DataFrame,
) -> pd.DataFrame:
    challenger = manifests[
        manifests["manifest_name"].eq(CHALLENGER_MANIFEST)
    ][["feature_name", "family"]].drop_duplicates()
    training = features["conditional_ppg_training_eligible"].eq(1)
    current = features["season"].eq(features["season"].max())
    rows = []
    for row in challenger.itertuples(index=False):
        values = pd.to_numeric(features[row.feature_name], errors="coerce")
        available_seasons = features.loc[values.notna(), "season"]
        rows.append(
            {
                "family": row.family,
                "feature_name": row.feature_name,
                "training_rows": int(training.sum()),
                "training_non_null": int(values[training].notna().sum()),
                "training_coverage": float(values[training].notna().mean()),
                "current_rows": int(current.sum()),
                "current_non_null": int(values[current].notna().sum()),
                "current_coverage": float(values[current].notna().mean()),
                "first_available_season": (
                    int(available_seasons.min())
                    if not available_seasons.empty
                    else pd.NA
                ),
                "last_available_season": (
                    int(available_seasons.max())
                    if not available_seasons.empty
                    else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "feature_name"])


def _provider_diagnostics(
    projection_values: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "player_key",
        "season",
        "outcome_complete",
        "unconditional_season_points",
        "expert_ppg_team_game_median",
    ]
    joined = projection_values.merge(
        features[columns],
        on=["player_key", "season"],
        how="inner",
        validate="many_to_one",
    )
    joined = joined[
        joined["season"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    coverage = (
        joined.groupby("provider", as_index=False)
        .agg(
            first_season=("season", "min"),
            last_season=("season", "max"),
            season_count=("season", "nunique"),
            rows=("player_key", "size"),
            configured_rows=(
                "configured_points_complete",
                lambda values: int(values.eq(1).sum()),
            ),
            imputed_configured_rows=(
                "configured_points_imputed_component_count",
                lambda values: int(values.gt(0).sum()),
            ),
        )
        .sort_values("provider")
    )

    observed = joined[
        joined["outcome_complete"].eq(1)
        & joined["unconditional_season_points"].notna()
        & joined["configured_points_complete"].eq(1)
        & joined["provider_points_per_team_game"].notna()
    ].copy()
    observed["actual_team_game_ppg"] = (
        observed["unconditional_season_points"]
        / np.where(observed["season"].ge(2021), 17.0, 16.0)
    )
    rows = []
    for provider, group in observed.groupby("provider", sort=True):
        provider_error = (
            group["provider_points_per_team_game"]
            - group["actual_team_game_ppg"]
        )
        consensus_error = (
            group["expert_ppg_team_game_median"]
            - group["actual_team_game_ppg"]
        )
        rows.append(
            {
                "provider": provider,
                "rows": len(group),
                "season_count": group["season"].nunique(),
                "first_season": int(group["season"].min()),
                "last_season": int(group["season"].max()),
                "provider_rmse": float(
                    np.sqrt(np.square(provider_error).mean())
                ),
                "consensus_rmse_same_rows": float(
                    np.sqrt(np.square(consensus_error).mean())
                ),
                "delta_provider_minus_consensus": float(
                    np.sqrt(np.square(provider_error).mean())
                    - np.sqrt(np.square(consensus_error).mean())
                ),
                "provider_bias": float(provider_error.mean()),
                "spearman": float(
                    group["provider_points_per_team_game"].corr(
                        group["actual_team_game_ppg"],
                        method="spearman",
                    )
                ),
            }
        )
    return coverage, pd.DataFrame(rows).sort_values(
        "delta_provider_minus_consensus"
    )


def _summary_markdown(
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    lines = [
        "# Projection Feature Challenger Results",
        "",
        "Negative deltas favor the added feature family.",
        "",
        "## Pooled OOF",
        "",
        "| Model | RMSE |",
        "|---|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| `{row.model_name}` | {float(row.value):.4f} |")
    lines.extend(
        [
            "",
            "## Fold-identical family comparisons",
            "",
            "| Model | Variant | Added | Pooled delta | Mean season delta | "
            "95% interval | Wins |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.model_family}` | `{row.variant}` | "
            f"{row.added_feature_count} | {row.pooled_delta:+.4f} | "
            f"{row.mean_season_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.challenger_wins}/{row.season_count} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    features, manifests, projection_values, feature_run_id = _load_inputs()
    target = build_target_frames(features, VALIDATION_END)[
        CONDITIONAL_PPG_TARGET
    ]
    run_id = create_run_id("m4a_projection_feature_challengers")
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
    for index, (spec, columns, added_count) in enumerate(
        experiments, start=1
    ):
        print(
            f"[{index}/{len(experiments)}] {spec.model_name} "
            f"(+{added_count})",
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
    coverage = _feature_coverage(features, manifests)
    provider_coverage, provider_accuracy = _provider_diagnostics(
        projection_values, features
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "model_slices.csv", index=False)
    hyperparameters.to_csv(
        RESULTS_DIR / "hyperparameters.csv", index=False
    )
    comparisons.to_csv(RESULTS_DIR / "family_comparisons.csv", index=False)
    coverage.to_csv(RESULTS_DIR / "feature_coverage.csv", index=False)
    provider_coverage.to_csv(
        RESULTS_DIR / "provider_coverage.csv", index=False
    )
    provider_accuracy.to_csv(
        RESULTS_DIR / "provider_accuracy_diagnostic.csv", index=False
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, comparisons), encoding="utf-8"
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
    main()
