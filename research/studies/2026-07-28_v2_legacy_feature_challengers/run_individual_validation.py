"""Test each legacy-inspired feature alone on the direct LightGBM model."""

from __future__ import annotations

import json

import pandas as pd

from run_validation import (
    BASE_MANIFEST,
    CHALLENGER_MANIFEST,
    N_SPLITS,
    POSITION_FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
    VALIDATION_END,
    VALIDATION_START,
    _load_inputs,
    _manifest_features,
    _paired_summary,
    _specification,
)
from Scripts.V2.contracts import create_run_id
from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    build_score_summary,
    build_slice_summary,
    build_target_frames,
    make_fold_assignments,
    run_model_spec,
)


def _summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Individual LightGBM Feature Additions",
        "",
        "Negative deltas mean that adding the one feature improved the direct "
        "LightGBM model versus the original 31-feature manifest.",
        "",
        "| Added feature | RMSE | Pooled delta | Mean season delta | "
        "95% season bootstrap | Wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        feature = row.variant.removeprefix("plus_feature_")
        lines.append(
            f"| `{feature}` | {row.variant_rmse:.4f} | "
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
    run_id = create_run_id("m4a_legacy_individual_features")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    base = _manifest_features(manifests, BASE_MANIFEST)
    challenger = manifests[
        manifests["manifest_name"].eq(CHALLENGER_MANIFEST)
    ]["feature_name"].drop_duplicates().sort_values()
    experiments = []
    for feature in challenger:
        variant = f"plus_feature_{feature}"
        experiments.append(
            (
                _specification("lightgbm", variant),
                base + (feature,) + POSITION_FEATURES,
                variant,
                1,
            )
        )

    oof_frames = []
    parameter_frames = []
    for index, (spec, columns, variant, _) in enumerate(
        experiments, start=1
    ):
        print(
            f"[{index}/{len(experiments)}] {variant}",
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

    family_scores = pd.read_csv(RESULTS_DIR / "model_scores.csv")
    family_slices = pd.read_csv(RESULTS_DIR / "model_slices.csv")
    base_scores = family_scores[
        family_scores["model_name"].eq("direct_lightgbm_base")
    ]
    base_slices = family_slices[
        family_slices["model_name"].eq("direct_lightgbm_base")
    ]
    comparison_scores = pd.concat([base_scores, scores], ignore_index=True)
    comparison_slices = pd.concat([base_slices, slices], ignore_index=True)
    summary = _paired_summary(
        comparison_scores,
        comparison_slices,
        experiments,
    )

    slice_rows = []
    base_slice = base_slices[
        base_slices["metric"].eq("rmse")
    ][["slice_type", "slice_value", "n_rows", "value"]].rename(
        columns={"value": "base_rmse"}
    )
    for model_name, group in slices[
        slices["metric"].eq("rmse")
    ].groupby("model_name"):
        compared = group.merge(
            base_slice,
            on=["slice_type", "slice_value", "n_rows"],
            how="inner",
            validate="one_to_one",
        )
        compared["feature_name"] = model_name.removeprefix(
            "direct_lightgbm_plus_feature_"
        )
        compared["variant_rmse"] = compared["value"]
        compared["delta_variant_minus_base"] = (
            compared["variant_rmse"] - compared["base_rmse"]
        )
        slice_rows.append(
            compared[
                [
                    "feature_name",
                    "slice_type",
                    "slice_value",
                    "n_rows",
                    "base_rmse",
                    "variant_rmse",
                    "delta_variant_minus_base",
                ]
            ]
        )
    individual_slices = pd.concat(slice_rows, ignore_index=True)

    scores.to_csv(RESULTS_DIR / "individual_model_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "individual_model_slices.csv", index=False)
    parameters.to_csv(
        RESULTS_DIR / "individual_hyperparameters.csv", index=False
    )
    summary.to_csv(
        RESULTS_DIR / "individual_feature_summary.csv", index=False
    )
    individual_slices.to_csv(
        RESULTS_DIR / "individual_feature_slices.csv", index=False
    )
    (RESULTS_DIR / "individual_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "features_tested": len(experiments),
                "results_directory": str(RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
