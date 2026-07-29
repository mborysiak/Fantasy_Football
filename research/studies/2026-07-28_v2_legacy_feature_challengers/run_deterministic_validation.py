"""Run deterministic full-column LightGBM attribution for all challengers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_validation import (
    BASE_MANIFEST,
    CHALLENGER_MANIFEST,
    N_SPLITS,
    POSITION_FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
    SEARCH_ITERATIONS,
    VALIDATION_END,
    VALIDATION_START,
    _load_inputs,
    _manifest_features,
)
from Scripts.V2.contracts import create_run_id
from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    ModelSpec,
    build_score_summary,
    build_slice_summary,
    build_target_frames,
    make_fold_assignments,
    run_model_spec,
)


MODEL_PREFIX = "direct_lgbm_fullcols_"


def _specification(variant: str) -> ModelSpec:
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
    return ModelSpec(
        CONDITIONAL_PPG_TARGET,
        f"{MODEL_PREFIX}{variant}",
        "lightgbm",
        "direct",
        variant,
        "raw",
        "lgbm",
        parameters,
        SEARCH_ITERATIONS,
    )


def _experiments(
    manifests: pd.DataFrame,
) -> list[tuple[ModelSpec, tuple[str, ...], str, str]]:
    base = _manifest_features(manifests, BASE_MANIFEST)
    challenger = manifests[
        manifests["manifest_name"].eq(CHALLENGER_MANIFEST)
    ].copy()
    variants: dict[str, tuple[str, ...]] = {"base": base}
    variant_type = {"base": "base"}
    for family, group in challenger.groupby("family", sort=True):
        family_features = tuple(
            group["feature_name"].drop_duplicates().sort_values().tolist()
        )
        variant = f"family_{family}"
        variants[variant] = tuple(dict.fromkeys((*base, *family_features)))
        variant_type[variant] = "family"
    all_features = tuple(
        challenger["feature_name"].drop_duplicates().sort_values().tolist()
    )
    variants["family_all_legacy"] = tuple(
        dict.fromkeys((*base, *all_features))
    )
    variant_type["family_all_legacy"] = "family"
    for feature in all_features:
        variant = f"feature_{feature}"
        variants[variant] = base + (feature,)
        variant_type[variant] = "feature"
    return [
        (
            _specification(variant),
            features + POSITION_FEATURES,
            variant,
            variant_type[variant],
        )
        for variant, features in variants.items()
    ]


def _pooled_value(
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
    experiments: list[tuple[ModelSpec, tuple[str, ...], str, str]],
    base_feature_count: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    base_model = f"{MODEL_PREFIX}base"
    base_rmse = _pooled_value(scores, base_model, "rmse")
    rows = []
    for spec, columns, variant, variant_type in experiments:
        if variant == "base":
            continue
        variant_rmse = _pooled_value(scores, spec.model_name, "rmse")
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
                "variant_type": variant_type,
                "variant": variant,
                "added_feature_count": len(columns)
                - base_feature_count
                - len(POSITION_FEATURES),
                "base_rmse": base_rmse,
                "variant_rmse": variant_rmse,
                "pooled_delta_variant_minus_base": variant_rmse - base_rmse,
                "mean_season_delta_variant_minus_base": float(delta.mean()),
                "median_season_delta_variant_minus_base": float(delta.median()),
                "variant_wins": int(delta.lt(-1e-12).sum()),
                "exact_ties": int(delta.abs().le(1e-12).sum()),
                "season_count": len(delta),
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["variant_type", "pooled_delta_variant_minus_base"]
    )


def _slice_comparison(slices: pd.DataFrame) -> pd.DataFrame:
    base_model = f"{MODEL_PREFIX}base"
    base = slices[
        slices["model_name"].eq(base_model)
        & slices["metric"].eq("rmse")
    ][["slice_type", "slice_value", "n_rows", "value"]].rename(
        columns={"value": "base_rmse"}
    )
    rows = []
    for model_name, group in slices[
        ~slices["model_name"].eq(base_model) & slices["metric"].eq("rmse")
    ].groupby("model_name"):
        compared = group.merge(
            base,
            on=["slice_type", "slice_value", "n_rows"],
            how="inner",
            validate="one_to_one",
        )
        compared["variant"] = model_name.removeprefix(MODEL_PREFIX)
        compared["variant_rmse"] = compared["value"]
        compared["delta_variant_minus_base"] = (
            compared["variant_rmse"] - compared["base_rmse"]
        )
        rows.append(
            compared[
                [
                    "variant",
                    "slice_type",
                    "slice_value",
                    "n_rows",
                    "base_rmse",
                    "variant_rmse",
                    "delta_variant_minus_base",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Deterministic Full-Column LightGBM Attribution",
        "",
        "Negative deltas mean the addition improved RMSE. Full row/column "
        "sampling and deterministic LightGBM settings prevent an unavailable "
        "feature from changing which incumbent columns are sampled.",
        "",
    ]
    for variant_type, group in summary.groupby("variant_type", sort=True):
        lines.extend(
            [
                f"## {variant_type}",
                "",
                "| Variant | Added | RMSE | Pooled delta | Mean season delta | "
                "95% season bootstrap | Wins | Ties |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
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
                f"{row.variant_wins}/{row.season_count} | "
                f"{row.exact_ties}/{row.season_count} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    features, manifests, feature_run_id = _load_inputs()
    base_feature_count = len(_manifest_features(manifests, BASE_MANIFEST))
    target = build_target_frames(features, VALIDATION_END)[
        CONDITIONAL_PPG_TARGET
    ]
    run_id = create_run_id("m4a_legacy_full_column")
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
    for index, (spec, columns, variant, _) in enumerate(
        experiments, start=1
    ):
        print(f"[{index}/{len(experiments)}] {variant}", flush=True)
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
    summary = _paired_summary(
        scores,
        slices,
        experiments,
        base_feature_count,
    )
    slice_comparison = _slice_comparison(slices)

    scores.to_csv(RESULTS_DIR / "deterministic_model_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "deterministic_model_slices.csv", index=False)
    parameters.to_csv(
        RESULTS_DIR / "deterministic_hyperparameters.csv", index=False
    )
    summary.to_csv(
        RESULTS_DIR / "deterministic_attribution_summary.csv", index=False
    )
    slice_comparison.to_csv(
        RESULTS_DIR / "deterministic_attribution_slices.csv", index=False
    )
    (RESULTS_DIR / "deterministic_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "experiments": len(experiments),
                "results_directory": str(RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
