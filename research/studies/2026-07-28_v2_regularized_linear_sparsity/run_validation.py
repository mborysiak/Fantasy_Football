"""Compare sparse linear V2 PPG models on identical rolling OOF folds."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import OUTPUT_DB_PATH
from Scripts.V2.contracts import create_run_id
from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    POSITION_FEATURES,
    ModelSpec,
    _build_pipeline,
    _load_scikit_model,
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
SEARCH_ITERATIONS = 20
BASE_MANIFEST = "residual_candidate_v1"
CHALLENGER_MANIFEST = "residual_legacy_challenger_v1"
MODEL_FAMILIES = ("ridge", "lasso", "elastic_net")
VARIANTS = ("incumbent", "expanded")
NONZERO_TOLERANCE = 1e-8


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


def _feature_variants(
    manifests: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    base = _manifest_features(manifests, BASE_MANIFEST)
    challengers = _manifest_features(manifests, CHALLENGER_MANIFEST)
    return {
        "incumbent": base + POSITION_FEATURES,
        "expanded": tuple(dict.fromkeys((*base, *challengers)))
        + POSITION_FEATURES,
    }


def _model_parameters(
    model_family: str,
) -> tuple[str, dict[str, tuple[object, ...]]]:
    if model_family == "ridge":
        return "ridge", {
            "ridge__alpha": (0.1, 1.0, 10.0, 100.0),
        }
    if model_family == "lasso":
        return "lasso", {
            "lasso__alpha": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
        }
    if model_family == "elastic_net":
        return "enet", {
            "enet__alpha": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
            "enet__l1_ratio": (0.1, 0.3, 0.5, 0.7, 0.9),
        }
    raise ValueError(f"Unsupported model family: {model_family}")


def _experiments(
    manifests: pd.DataFrame,
) -> list[tuple[ModelSpec, tuple[str, ...]]]:
    variants = _feature_variants(manifests)
    experiments = []
    for variant in VARIANTS:
        for model_family in MODEL_FAMILIES:
            model_piece, parameters = _model_parameters(model_family)
            experiments.append(
                (
                    ModelSpec(
                        CONDITIONAL_PPG_TARGET,
                        f"direct_{model_family}_{variant}",
                        model_family,
                        "direct",
                        variant,
                        "raw",
                        model_piece,
                        parameters,
                        SEARCH_ITERATIONS,
                    ),
                    variants[variant],
                )
            )
    return experiments


def _pooled_rmse(
    scores: pd.DataFrame,
    model_name: str,
) -> float:
    selected = scores[
        scores["model_name"].eq(model_name)
        & scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("rmse")
    ]
    if len(selected) != 1:
        raise ValueError(f"Missing pooled RMSE for {model_name}")
    return float(selected.iloc[0]["value"])


def _season_rmse(
    slices: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    selected = slices[
        slices["model_name"].eq(model_name)
        & slices["slice_type"].eq("season")
        & slices["metric"].eq("rmse")
    ]
    return selected.set_index("slice_value")["value"].astype(float)


def _bootstrap_interval(
    deltas: pd.Series,
    seed_offset: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    values = deltas.dropna().to_numpy(dtype=float)
    draws = np.array(
        [
            rng.choice(values, len(values), replace=True).mean()
            for _ in range(20_000)
        ]
    )
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def _comparison_rows(
    scores: pd.DataFrame,
    slices: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = (
        ("lasso_vs_ridge_incumbent", "direct_lasso_incumbent",
         "direct_ridge_incumbent"),
        ("elastic_net_vs_ridge_incumbent", "direct_elastic_net_incumbent",
         "direct_ridge_incumbent"),
        ("lasso_expanded_vs_incumbent", "direct_lasso_expanded",
         "direct_lasso_incumbent"),
        ("elastic_net_expanded_vs_incumbent", "direct_elastic_net_expanded",
         "direct_elastic_net_incumbent"),
        ("ridge_expanded_vs_incumbent", "direct_ridge_expanded",
         "direct_ridge_incumbent"),
    )
    rows = []
    for index, (comparison, challenger, reference) in enumerate(comparisons):
        challenger_rmse = _pooled_rmse(scores, challenger)
        reference_rmse = _pooled_rmse(scores, reference)
        deltas = _season_rmse(slices, challenger) - _season_rmse(
            slices, reference
        )
        low, high = _bootstrap_interval(deltas, index)
        rows.append(
            {
                "comparison": comparison,
                "challenger_model": challenger,
                "reference_model": reference,
                "challenger_rmse": challenger_rmse,
                "reference_rmse": reference_rmse,
                "pooled_delta": challenger_rmse - reference_rmse,
                "mean_season_delta": float(deltas.mean()),
                "median_season_delta": float(deltas.median()),
                "challenger_wins": int(deltas.lt(0).sum()),
                "season_count": len(deltas),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    return pd.DataFrame(rows)


def _selected_parameters(
    hyperparameters: pd.DataFrame,
    model_name: str,
    fold: int,
) -> dict[str, object]:
    selected = hyperparameters[
        hyperparameters["model_name"].eq(model_name)
        & hyperparameters["fold"].eq(fold)
        & hyperparameters["selected"].eq(1)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one selected trial for {model_name}, fold {fold}"
        )
    return json.loads(str(selected.iloc[0]["parameters_json"]))


def _training_mask(
    target: pd.DataFrame,
    fold_lookup: pd.Series,
    season: int,
    fold: int,
) -> pd.Series:
    row_folds = pd.MultiIndex.from_frame(
        target[["player_key", "season"]]
    ).map(fold_lookup)
    return target["season"].lt(season) & (
        pd.isna(row_folds) | (row_folds != fold)
    )


def _coefficient_rows(
    target: pd.DataFrame,
    assignments: pd.DataFrame,
    experiments: Sequence[tuple[ModelSpec, tuple[str, ...]]],
    hyperparameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    SciKitModel = _load_scikit_model()
    model_data = target.copy()
    model_data["player"] = model_data["player_key"]
    model_data["week"] = 1
    model_data["year"] = model_data["season"].astype(int)
    model_data["game_date"] = model_data["season"].astype(int)
    model_data["y_act"] = model_data["actual_target"]
    fold_lookup = assignments.set_index(["player_key", "season"])["fold"]
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for spec, feature_columns in experiments:
        skm = SciKitModel(model_data, model_obj="reg", set_seed=RANDOM_SEED)
        base_pipeline = _build_pipeline(skm, spec, feature_columns)
        for fold in range(N_SPLITS):
            parameters = _selected_parameters(
                hyperparameters, spec.model_name, fold
            )
            for season in range(VALIDATION_START, VALIDATION_END + 1):
                mask = _training_mask(
                    model_data, fold_lookup, season, fold
                )
                train = model_data.loc[mask].copy()
                pipeline = clone(base_pipeline).set_params(**parameters)
                X_train = train.loc[
                    :, list(feature_columns) + ["game_date"]
                ].apply(pd.to_numeric, errors="coerce")
                y_train = pd.to_numeric(train["y_act"], errors="coerce")
                pipeline.fit(X_train, y_train)
                hold_mask = model_data["season"].eq(season) & (
                    pd.MultiIndex.from_frame(
                        model_data[["player_key", "season"]]
                    ).map(fold_lookup)
                    == fold
                )
                hold = model_data.loc[hold_mask].copy()
                X_hold = hold.loc[
                    :, list(feature_columns) + ["game_date"]
                ].apply(pd.to_numeric, errors="coerce")
                hold_predictions = pipeline.predict(X_hold)
                for player_key, prediction in zip(
                    hold["player_key"], hold_predictions
                ):
                    prediction_rows.append(
                        {
                            "model_name": spec.model_name,
                            "fold": fold,
                            "season": season,
                            "player_key": player_key,
                            "refit_prediction": float(prediction),
                        }
                    )
                imputer = pipeline.named_steps["impute"]
                feature_names = imputer.get_feature_names_out(
                    list(feature_columns)
                )
                estimator = pipeline.named_steps[str(spec.model_piece)]
                coefficients = np.asarray(estimator.coef_).reshape(-1)
                if len(feature_names) != len(coefficients):
                    raise ValueError(
                        f"Coefficient names do not align for "
                        f"{spec.model_name}"
                    )
                for feature_name, coefficient in zip(
                    feature_names, coefficients
                ):
                    is_indicator = str(feature_name).startswith(
                        "missingindicator_"
                    )
                    raw_feature = str(feature_name).removeprefix(
                        "missingindicator_"
                    )
                    rows.append(
                        {
                            "model_name": spec.model_name,
                            "model_family": spec.model_family,
                            "variant": spec.feature_set,
                            "fold": fold,
                            "season": season,
                            "training_rows": len(train),
                            "feature_name": raw_feature,
                            "coefficient_name": str(feature_name),
                            "is_missing_indicator": int(is_indicator),
                            "coefficient": float(coefficient),
                            "absolute_coefficient": float(abs(coefficient)),
                            "selected": int(
                                abs(coefficient) > NONZERO_TOLERANCE
                            ),
                            "parameters_json": json.dumps(
                                parameters,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def _validate_refit_predictions(
    oof: pd.DataFrame,
    refit_predictions: pd.DataFrame,
) -> pd.DataFrame:
    expected = oof[
        ["model_name", "fold", "season", "player_key", "model_prediction"]
    ]
    checked = refit_predictions.merge(
        expected,
        on=["model_name", "fold", "season", "player_key"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError("Coefficient refits do not cover the exact OOF rows")
    checked["absolute_prediction_delta"] = (
        checked["refit_prediction"] - checked["model_prediction"]
    ).abs()
    summary = (
        checked.groupby("model_name", as_index=False)
        .agg(
            prediction_rows=("absolute_prediction_delta", "size"),
            mean_absolute_prediction_delta=(
                "absolute_prediction_delta",
                "mean",
            ),
            max_absolute_prediction_delta=(
                "absolute_prediction_delta",
                "max",
            ),
        )
        .sort_values("model_name")
    )
    if summary["max_absolute_prediction_delta"].max() > 1e-8:
        raise ValueError("Coefficient refits do not reproduce OOF predictions")
    return summary


def _selection_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    return (
        coefficients.groupby(
            [
                "model_name",
                "model_family",
                "variant",
                "feature_name",
                "coefficient_name",
                "is_missing_indicator",
            ],
            as_index=False,
        )
        .agg(
            fit_count=("selected", "size"),
            selected_count=("selected", "sum"),
            selection_frequency=("selected", "mean"),
            mean_coefficient=("coefficient", "mean"),
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
            median_absolute_coefficient=("absolute_coefficient", "median"),
        )
        .sort_values(
            [
                "model_name",
                "is_missing_indicator",
                "selection_frequency",
                "mean_absolute_coefficient",
            ],
            ascending=[True, True, False, False],
        )
    )


def _sparsity_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    enriched = coefficients.assign(
        raw_coefficient=coefficients["is_missing_indicator"].eq(0).astype(int),
        raw_selected=(
            coefficients["selected"]
            * coefficients["is_missing_indicator"].eq(0).astype(int)
        ),
        indicator_coefficient=coefficients[
            "is_missing_indicator"
        ].astype(int),
        indicator_selected=(
            coefficients["selected"]
            * coefficients["is_missing_indicator"].astype(int)
        ),
    )
    per_fit = (
        enriched.groupby(
            ["model_name", "model_family", "variant", "fold", "season"],
            as_index=False,
        )
        .agg(
            coefficient_count=("selected", "size"),
            selected_count=("selected", "sum"),
            raw_coefficient_count=("raw_coefficient", "sum"),
            raw_selected_count=("raw_selected", "sum"),
            indicator_coefficient_count=("indicator_coefficient", "sum"),
            indicator_selected_count=("indicator_selected", "sum"),
        )
    )
    per_fit["selected_fraction"] = (
        per_fit["selected_count"] / per_fit["coefficient_count"]
    )
    per_fit["raw_selected_fraction"] = (
        per_fit["raw_selected_count"] / per_fit["raw_coefficient_count"]
    )
    return (
        per_fit.groupby(
            ["model_name", "model_family", "variant"], as_index=False
        )
        .agg(
            fit_count=("selected_count", "size"),
            mean_selected_count=("selected_count", "mean"),
            min_selected_count=("selected_count", "min"),
            max_selected_count=("selected_count", "max"),
            mean_selected_fraction=("selected_fraction", "mean"),
            raw_feature_count=("raw_coefficient_count", "max"),
            mean_raw_selected_count=("raw_selected_count", "mean"),
            min_raw_selected_count=("raw_selected_count", "min"),
            max_raw_selected_count=("raw_selected_count", "max"),
            mean_raw_selected_fraction=("raw_selected_fraction", "mean"),
            indicator_count=("indicator_coefficient_count", "max"),
            mean_indicator_selected_count=("indicator_selected_count", "mean"),
        )
        .sort_values("model_name")
    )


def _stability_summary(selection: pd.DataFrame) -> pd.DataFrame:
    raw = selection[selection["is_missing_indicator"].eq(0)].copy()
    return (
        raw.groupby(
            ["model_name", "model_family", "variant"], as_index=False
        )
        .agg(
            raw_feature_count=("feature_name", "size"),
            selected_at_least_95pct=(
                "selection_frequency",
                lambda values: int(values.ge(0.95).sum()),
            ),
            selected_at_least_80pct=(
                "selection_frequency",
                lambda values: int(values.ge(0.80).sum()),
            ),
            selected_at_most_20pct=(
                "selection_frequency",
                lambda values: int(values.le(0.20).sum()),
            ),
            never_selected=(
                "selection_frequency",
                lambda values: int(values.eq(0).sum()),
            ),
        )
        .sort_values("model_name")
    )


def _challenger_selection_summary(
    target: pd.DataFrame,
    manifests: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    challengers = manifests[
        manifests["manifest_name"].eq(CHALLENGER_MANIFEST)
    ][["family", "feature_name"]].drop_duplicates()
    rows = []
    for row in challengers.itertuples(index=False):
        values = pd.to_numeric(target[row.feature_name], errors="coerce")
        output: dict[str, object] = {
            "family": row.family,
            "feature_name": row.feature_name,
            "available_rows": int(values.notna().sum()),
            "availability_rate": float(values.notna().mean()),
            "first_available_season": (
                int(target.loc[values.notna(), "season"].min())
                if values.notna().any()
                else pd.NA
            ),
        }
        for model_family in ("lasso", "elastic_net"):
            model_name = f"direct_{model_family}_expanded"
            selected = selection[
                selection["model_name"].eq(model_name)
                & selection["coefficient_name"].eq(row.feature_name)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Missing raw coefficient summary for "
                    f"{model_name}/{row.feature_name}"
                )
            output[f"{model_family}_selection_frequency"] = float(
                selected.iloc[0]["selection_frequency"]
            )
            output[f"{model_family}_mean_coefficient"] = float(
                selected.iloc[0]["mean_coefficient"]
            )
        rows.append(output)
    return pd.DataFrame(rows).sort_values(
        ["family", "lasso_selection_frequency"],
        ascending=[True, False],
    )


def _summary_markdown(
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    sparsity: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    lines = [
        "# V2 Regularized Linear Sparsity Results",
        "",
        "Negative comparison deltas favor the challenger. Confidence intervals "
        "bootstrap the nine validation-season RMSE differences.",
        "",
        "## Pooled OOF",
        "",
        "| Model | Features | RMSE |",
        "|---|---:|---:|",
    ]
    feature_counts = {
        "incumbent": 35,
        "expanded": 47,
    }
    for row in pooled.itertuples(index=False):
        variant = str(row.model_name).rsplit("_", 1)[-1]
        lines.append(
            f"| `{row.model_name}` | {feature_counts[variant]} | "
            f"{float(row.value):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired season comparisons",
            "",
            "| Comparison | Pooled delta | Mean season delta | 95% interval | "
            "Wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.comparison}` | {row.pooled_delta:+.4f} | "
            f"{row.mean_season_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.challenger_wins}/{row.season_count} |"
        )
    lines.extend(
        [
            "",
            "## Sparsity",
            "",
            "| Model | Mean raw selected | Raw range | Raw fraction | "
            "Mean indicators selected |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sparsity.itertuples(index=False):
        lines.append(
            f"| `{row.model_name}` | {row.mean_raw_selected_count:.1f} | "
            f"{int(row.min_raw_selected_count)}-"
            f"{int(row.max_raw_selected_count)} | "
            f"{row.mean_raw_selected_fraction:.1%} | "
            f"{row.mean_indicator_selected_count:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    features, manifests, feature_run_id = _load_inputs()
    target = build_target_frames(features, VALIDATION_END)[
        CONDITIONAL_PPG_TARGET
    ]
    run_id = create_run_id("m4a_regularized_linear_sparsity")
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
    for index, (spec, columns) in enumerate(experiments, start=1):
        print(
            f"[{index}/{len(experiments)}] {spec.model_name} "
            f"({len(columns)} features)",
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
    comparisons = _comparison_rows(scores, slices)
    print("[coefficients] refitting 270 exact season-fold models", flush=True)
    coefficients, refit_predictions = _coefficient_rows(
        target, assignments, experiments, hyperparameters
    )
    refit_checks = _validate_refit_predictions(oof, refit_predictions)
    selection = _selection_summary(coefficients)
    sparsity = _sparsity_summary(coefficients)
    stability = _stability_summary(selection)
    challenger_selection = _challenger_selection_summary(
        target, manifests, selection
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    slices.to_csv(RESULTS_DIR / "model_slices.csv", index=False)
    hyperparameters.to_csv(
        RESULTS_DIR / "hyperparameters.csv", index=False
    )
    comparisons.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / "fold_season_coefficients.csv", index=False)
    selection.to_csv(RESULTS_DIR / "feature_selection_summary.csv", index=False)
    sparsity.to_csv(RESULTS_DIR / "model_sparsity_summary.csv", index=False)
    stability.to_csv(
        RESULTS_DIR / "feature_stability_summary.csv", index=False
    )
    challenger_selection.to_csv(
        RESULTS_DIR / "challenger_selection_summary.csv", index=False
    )
    refit_checks.to_csv(
        RESULTS_DIR / "refit_prediction_checks.csv", index=False
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, comparisons, sparsity),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "target_rows": len(target),
                "experiments": len(experiments),
                "coefficient_fits": int(
                    coefficients[
                        ["model_name", "fold", "season"]
                    ].drop_duplicates().shape[0]
                ),
                "results_directory": str(RESULTS_DIR.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
