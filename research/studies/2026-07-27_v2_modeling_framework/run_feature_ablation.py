"""Run fold-identical feature-family dropouts for the simple linear models."""

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
    PARTICIPATION_TARGET,
    POSITION_FEATURES,
    ModelSpec,
    build_score_summary,
    build_slice_summary,
    build_target_frames,
    make_fold_assignments,
    run_model_spec,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PRIMARY_METRICS = {
    CONDITIONAL_PPG_TARGET: "rmse",
    PARTICIPATION_TARGET: "brier",
}
REFERENCE_MODELS = {
    CONDITIONAL_PPG_TARGET: "residual_ridge_full",
    PARTICIPATION_TARGET: "participation_logistic_full",
}
MANIFESTS = {
    CONDITIONAL_PPG_TARGET: "residual_candidate_v1",
    PARTICIPATION_TARGET: "participation_candidate_v1",
}


def _load_inputs():
    with sqlite3.connect(OUTPUT_DB_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features", connection
        )
        manifests = pd.read_sql_query(
            "SELECT * FROM feature_manifests", connection
        )
        model_runs = pd.read_sql_query(
            "SELECT * FROM model_runs WHERE status='complete'", connection
        )
        reference_scores = pd.read_sql_query(
            "SELECT * FROM model_score_summary", connection
        )
        reference_slices = pd.read_sql_query(
            "SELECT * FROM model_slice_summary", connection
        )
    if len(model_runs) != 1:
        raise ValueError("Expected exactly one active M4A model run")
    model_run = model_runs.iloc[0]
    feature_run_ids = features["run_id"].dropna().astype(str).unique()
    if set(feature_run_ids) != {str(model_run["feature_run_id"])}:
        raise ValueError("Model and feature runs do not match")
    return (
        features,
        manifests,
        model_run,
        reference_scores,
        reference_slices,
    )


def _dropout_specs(manifests: pd.DataFrame):
    output = []
    for target_name, manifest_name in MANIFESTS.items():
        selected = manifests[manifests["manifest_name"].eq(manifest_name)]
        for family in sorted(selected["family"].unique()):
            kept = tuple(
                selected.loc[
                    ~selected["family"].eq(family), "feature_name"
                ]
                .drop_duplicates()
                .sort_values()
                .tolist()
            ) + POSITION_FEATURES
            if target_name == CONDITIONAL_PPG_TARGET:
                spec = ModelSpec(
                    target_name,
                    f"residual_ridge_drop_{family}",
                    "ridge",
                    "residual",
                    f"drop_{family}",
                    "raw",
                    "ridge",
                    {"ridge__alpha": (1.0, 10.0, 100.0)},
                    4,
                )
            else:
                spec = ModelSpec(
                    target_name,
                    f"participation_logistic_drop_{family}",
                    "logistic",
                    "probability",
                    f"drop_{family}",
                    "raw",
                    "lr_c",
                    {"lr_c__C": (0.1, 1.0, 10.0)},
                    4,
                )
            output.append(
                (
                    spec,
                    kept,
                    family,
                    int(selected["family"].eq(family).sum()),
                )
            )
    return output


def _paired_summary(
    scores: pd.DataFrame,
    slices: pd.DataFrame,
    reference_scores: pd.DataFrame,
    reference_slices: pd.DataFrame,
    specifications: list[tuple[ModelSpec, tuple[str, ...], str, int]],
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    for spec, features, family, dropped_count in specifications:
        target = spec.target_name
        metric = PRIMARY_METRICS[target]
        reference = REFERENCE_MODELS[target]
        dropout_score = scores[
            scores["target_name"].eq(target)
            & scores["model_name"].eq(spec.model_name)
            & scores["aggregation"].eq("pooled_oof")
            & scores["metric"].eq(metric)
        ]["value"].iloc[0]
        full_score = reference_scores[
            reference_scores["target_name"].eq(target)
            & reference_scores["model_name"].eq(reference)
            & reference_scores["aggregation"].eq("pooled_oof")
            & reference_scores["metric"].eq(metric)
        ]["value"].iloc[0]
        dropout_seasons = slices[
            slices["target_name"].eq(target)
            & slices["model_name"].eq(spec.model_name)
            & slices["slice_type"].eq("season")
            & slices["metric"].eq(metric)
        ].set_index("slice_value")["value"]
        full_seasons = reference_slices[
            reference_slices["target_name"].eq(target)
            & reference_slices["model_name"].eq(reference)
            & reference_slices["slice_type"].eq("season")
            & reference_slices["metric"].eq(metric)
        ].set_index("slice_value")["value"]
        season_delta = (dropout_seasons - full_seasons).dropna()
        boot = np.array(
            [
                rng.choice(
                    season_delta.to_numpy(),
                    len(season_delta),
                    replace=True,
                ).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "target_name": target,
                "reference_model": reference,
                "dropped_family": family,
                "dropped_feature_count": dropped_count,
                "remaining_feature_count": len(features),
                "metric": metric,
                "dropout_value": dropout_score,
                "full_value": full_score,
                "pooled_delta_dropout_minus_full": dropout_score - full_score,
                "mean_season_delta_dropout_minus_full": season_delta.mean(),
                "median_season_delta_dropout_minus_full": season_delta.median(),
                "dropout_worse_seasons": int(season_delta.gt(0).sum()),
                "season_count": len(season_delta),
                "bootstrap_95_low": np.quantile(boot, 0.025),
                "bootstrap_95_high": np.quantile(boot, 0.975),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target_name", "pooled_delta_dropout_minus_full"],
        ascending=[True, False],
    )


def _summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Feature-Family Dropout Results",
        "",
        "Positive deltas mean that removing the family made the primary error "
        "metric worse, so the family added OOF value in the full linear model.",
        "",
    ]
    for target, group in summary.groupby("target_name", sort=True):
        lines.extend(
            [
                f"## {target}",
                "",
                "| Dropped family | Features | Pooled delta | Mean season delta | "
                "95% season bootstrap | Worse seasons |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"| `{row.dropped_family}` | {row.dropped_feature_count} | "
                f"{row.pooled_delta_dropout_minus_full:+.4f} | "
                f"{row.mean_season_delta_dropout_minus_full:+.4f} | "
                f"[{row.bootstrap_95_low:+.4f}, "
                f"{row.bootstrap_95_high:+.4f}] | "
                f"{row.dropout_worse_seasons}/{row.season_count} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    (
        features,
        manifests,
        model_run,
        reference_scores,
        reference_slices,
    ) = _load_inputs()
    validation_start = int(model_run["validation_start_season"])
    validation_end = int(model_run["validation_end_season"])
    n_splits = int(model_run["n_splits"])
    random_seed = int(model_run["random_seed"])
    target_frames = build_target_frames(features, validation_end)
    run_id = create_run_id("m4a_family_ablation")
    specifications = _dropout_specs(manifests)
    assignments = {
        target: make_fold_assignments(
            target_frames[target],
            target,
            run_id,
            validation_start,
            n_splits,
            random_seed,
        )
        for target in MANIFESTS
    }
    oof_frames = []
    parameter_frames = []
    for index, (spec, columns, family, _) in enumerate(
        specifications, start=1
    ):
        print(
            f"[{index}/{len(specifications)}] {spec.target_name}: "
            f"drop {family}",
            flush=True,
        )
        oof, parameters = run_model_spec(
            target_frames[spec.target_name],
            assignments[spec.target_name],
            spec,
            columns,
            run_id,
            str(model_run["feature_run_id"]),
            validation_start,
            n_splits,
            random_seed,
            quiet=True,
        )
        oof_frames.append(oof)
        parameter_frames.append(parameters)
    oof = pd.concat(oof_frames, ignore_index=True)
    scores = build_score_summary(oof, run_id)
    slices = build_slice_summary(oof, run_id)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = _paired_summary(
        scores,
        slices,
        reference_scores,
        reference_slices,
        specifications,
        random_seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "feature_family_ablation_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "feature_family_ablation_slices.csv", index=False)
    parameters.to_csv(
        RESULTS_DIR / "feature_family_ablation_hyperparameters.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS_DIR / "feature_family_ablation_summary.csv", index=False
    )
    (RESULTS_DIR / "feature_family_ablation_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": model_run["feature_run_id"],
                "dropout_models": len(specifications),
                "summary_rows": len(summary),
                "results_directory": str(RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
