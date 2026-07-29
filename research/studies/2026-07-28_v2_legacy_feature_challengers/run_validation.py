"""Test legacy-inspired V2 feature families on identical rolling OOF folds."""

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
SEARCH_ITERATIONS = 4
BASE_MANIFEST = "residual_candidate_v1"
CHALLENGER_MANIFEST = "residual_legacy_challenger_v1"
MODEL_FAMILIES = ("ridge", "lightgbm")


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
    if len(feature_run_ids) != 1 or set(manifest_run_ids) != set(
        feature_run_ids
    ):
        raise ValueError("Feature mart and manifests do not share one run ID")
    for manifest_name in (BASE_MANIFEST, CHALLENGER_MANIFEST):
        if not manifests["manifest_name"].eq(manifest_name).any():
            raise ValueError(f"Missing feature manifest: {manifest_name}")
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


def _lgbm_parameters() -> dict[str, tuple[object, ...]]:
    return {
        "lgbm__n_estimators": (100, 200),
        "lgbm__learning_rate": (0.03, 0.05),
        "lgbm__num_leaves": (7, 15),
        "lgbm__max_depth": (3, 4),
        "lgbm__min_child_samples": (20, 40),
        "lgbm__reg_lambda": (1.0, 5.0),
        "lgbm__subsample": (0.8,),
        "lgbm__colsample_bytree": (0.8,),
    }


def _specification(
    model_family: str,
    variant: str,
) -> ModelSpec:
    if model_family == "ridge":
        model_piece = "ridge"
        parameters = {"ridge__alpha": (1.0, 10.0, 100.0)}
    elif model_family == "lightgbm":
        model_piece = "lgbm"
        parameters = _lgbm_parameters()
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
        SEARCH_ITERATIONS,
    )


def _experiments(
    manifests: pd.DataFrame,
) -> list[tuple[ModelSpec, tuple[str, ...], str, int]]:
    base = _manifest_features(manifests, BASE_MANIFEST)
    challenger = manifests[
        manifests["manifest_name"].eq(CHALLENGER_MANIFEST)
    ].copy()
    family_features = {
        str(family): tuple(
            group["feature_name"].drop_duplicates().sort_values().tolist()
        )
        for family, group in challenger.groupby("family", sort=True)
    }
    variants: dict[str, tuple[str, ...]] = {"base": base}
    for family, features in family_features.items():
        variants[f"plus_{family}"] = tuple(
            dict.fromkeys((*base, *features))
        )
    variants["plus_all_legacy"] = tuple(
        dict.fromkeys(
            (
                *base,
                *challenger["feature_name"]
                .drop_duplicates()
                .sort_values()
                .tolist(),
            )
        )
    )

    output = []
    for model_family in MODEL_FAMILIES:
        for variant, features in variants.items():
            added = len(features) - len(base)
            output.append(
                (
                    _specification(model_family, variant),
                    features + POSITION_FEATURES,
                    variant,
                    added,
                )
            )
    return output


def _score_value(
    scores: pd.DataFrame,
    model_name: str,
    metric: str,
) -> float:
    selected = scores[
        scores["model_name"].eq(model_name)
        & scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"Missing pooled {metric} for {model_name}")
    return float(selected.iloc[0]["value"])


def _paired_summary(
    scores: pd.DataFrame,
    slices: pd.DataFrame,
    experiments: list[tuple[ModelSpec, tuple[str, ...], str, int]],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for spec, features, variant, added_count in experiments:
        if variant == "base":
            continue
        base_model = f"direct_{spec.model_family}_base"
        variant_value = _score_value(scores, spec.model_name, "rmse")
        base_value = _score_value(scores, base_model, "rmse")
        season = slices[
            slices["model_name"].eq(spec.model_name)
            & slices["slice_type"].eq("season")
            & slices["metric"].eq("rmse")
        ].set_index("slice_value")["value"]
        base_season = slices[
            slices["model_name"].eq(base_model)
            & slices["slice_type"].eq("season")
            & slices["metric"].eq("rmse")
        ].set_index("slice_value")["value"]
        delta = (season - base_season).dropna()
        bootstrap = np.array(
            [
                rng.choice(delta.to_numpy(), len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "model_family": spec.model_family,
                "variant": variant,
                "added_feature_count": added_count,
                "total_feature_count": len(features),
                "base_rmse": base_value,
                "variant_rmse": variant_value,
                "pooled_delta_variant_minus_base": variant_value - base_value,
                "mean_season_delta_variant_minus_base": float(delta.mean()),
                "median_season_delta_variant_minus_base": float(delta.median()),
                "variant_wins": int(delta.lt(0).sum()),
                "season_count": len(delta),
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_family", "pooled_delta_variant_minus_base"]
    )


def _slice_comparison(slices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_family in MODEL_FAMILIES:
        base_model = f"direct_{model_family}_base"
        base = slices[
            slices["model_name"].eq(base_model)
            & slices["metric"].eq("rmse")
        ][["slice_type", "slice_value", "n_rows", "value"]].rename(
            columns={"value": "base_rmse"}
        )
        variants = slices[
            slices["model_name"].str.startswith(
                f"direct_{model_family}_plus_"
            )
            & slices["metric"].eq("rmse")
        ]
        for model_name, group in variants.groupby("model_name"):
            compared = group.merge(
                base,
                on=["slice_type", "slice_value", "n_rows"],
                how="inner",
                validate="one_to_one",
            )
            compared["model_family"] = model_family
            compared["variant"] = model_name.removeprefix(
                f"direct_{model_family}_"
            )
            compared["variant_rmse"] = compared["value"]
            compared["delta_variant_minus_base"] = (
                compared["variant_rmse"] - compared["base_rmse"]
            )
            rows.extend(
                compared[
                    [
                        "model_family",
                        "variant",
                        "slice_type",
                        "slice_value",
                        "n_rows",
                        "base_rmse",
                        "variant_rmse",
                        "delta_variant_minus_base",
                    ]
                ].to_dict("records")
            )
    return pd.DataFrame(rows)


def _summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Legacy-Inspired Feature Challenger Results",
        "",
        "Negative deltas mean the added feature family improved conditional-PPG "
        "RMSE versus the same model on the original 31-feature manifest.",
        "",
    ]
    for model_family, group in summary.groupby("model_family", sort=True):
        lines.extend(
            [
                f"## {model_family}",
                "",
                "| Variant | Added | RMSE | Pooled delta | Mean season delta | "
                "95% season bootstrap | Wins |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"| `{row.variant}` | {row.added_feature_count} | "
                f"{row.variant_rmse:.4f} | "
                f"{row.pooled_delta_variant_minus_base:+.4f} | "
                f"{row.mean_season_delta_variant_minus_base:+.4f} | "
                f"[{row.bootstrap_95_low:+.4f}, "
                f"{row.bootstrap_95_high:+.4f}] | "
                f"{row.variant_wins}/{row.season_count} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    features, manifests, feature_run_id = _load_inputs()
    target = build_target_frames(features, VALIDATION_END)[
        CONDITIONAL_PPG_TARGET
    ]
    run_id = create_run_id("m4a_legacy_feature_challengers")
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
    for index, (spec, columns, variant, added_count) in enumerate(
        experiments, start=1
    ):
        print(
            f"[{index}/{len(experiments)}] {spec.model_family} "
            f"{variant} (+{added_count})",
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
    parameters = pd.concat(parameter_frames, ignore_index=True)
    scores = build_score_summary(oof, run_id)
    slices = build_slice_summary(oof, run_id)
    summary = _paired_summary(scores, slices, experiments)
    slice_comparison = _slice_comparison(slices)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "model_slices.csv", index=False)
    parameters.to_csv(RESULTS_DIR / "hyperparameters.csv", index=False)
    summary.to_csv(RESULTS_DIR / "family_augmentation_summary.csv", index=False)
    slice_comparison.to_csv(
        RESULTS_DIR / "family_augmentation_slices.csv", index=False
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
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
