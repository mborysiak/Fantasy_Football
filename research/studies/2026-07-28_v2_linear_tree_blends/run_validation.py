"""Test fixed and causal blends of V2 Lasso, LightGBM, and RF."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
CONSENSUS_OOF = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_projection_consensus_ladder"
    / "results"
    / "oof_predictions.csv"
)
RF_OOF = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_knn_rf_models"
    / "results"
    / "model_oof.csv"
)
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234
MIN_CAUSAL_SEASONS = 2
SOURCE_MODELS = {
    "projection_core_lasso": "projection_only_lasso_core",
    "projection_active_lasso": "projection_only_lasso_plus_active",
    "full_lightgbm": "full_lightgbm_base",
}
LINEAR_CANDIDATES = (
    "full_lasso",
    "projection_core_lasso",
    "projection_active_lasso",
)


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


def _lasso_spec() -> ModelSpec:
    return ModelSpec(
        target_name=CONDITIONAL_PPG_TARGET,
        model_name="full_lasso",
        model_family="lasso",
        prediction_kind="direct",
        feature_set="full",
        pipeline_variant="raw",
        model_piece="lasso",
        parameters={
            "lasso__alpha": (
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
            )
        },
        search_iterations=7,
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


def _load_source_predictions() -> pd.DataFrame:
    keys = ["player_key", "season"]
    consensus = pd.read_csv(CONSENSUS_OOF)
    consensus = consensus[
        consensus["model_name"].isin(SOURCE_MODELS.values())
    ].copy()
    consensus["method"] = consensus["model_name"].map(
        {value: key for key, value in SOURCE_MODELS.items()}
    )
    consensus_wide = consensus.pivot(
        index=keys,
        columns="method",
        values="final_prediction",
    ).reset_index()
    rf = pd.read_csv(RF_OOF)
    rf = rf[rf["model_name"].eq("full_random_forest")][
        [*keys, "final_prediction"]
    ].rename(columns={"final_prediction": "full_random_forest"})
    return consensus_wide.merge(
        rf,
        on=keys,
        how="inner",
        validate="one_to_one",
    )


def _add_fixed_blends(frame: pd.DataFrame) -> list[str]:
    methods = []
    for linear in LINEAR_CANDIDATES:
        lgbm_name = f"{linear}_lgbm_average"
        rf_name = f"{linear}_rf_average"
        thirds_name = f"{linear}_rf_lgbm_equal_thirds"
        frame[lgbm_name] = (
            frame[linear] + frame["full_lightgbm"]
        ) / 2
        frame[rf_name] = (
            frame[linear] + frame["full_random_forest"]
        ) / 2
        frame[thirds_name] = (
            frame[linear]
            + frame["full_random_forest"]
            + frame["full_lightgbm"]
        ) / 3
        methods.extend((lgbm_name, rf_name, thirds_name))
    frame["full_rf_lgbm_average"] = (
        frame["full_random_forest"] + frame["full_lightgbm"]
    ) / 2
    return methods


def _fit_convex_weights(
    prior: pd.DataFrame,
    components: Sequence[str],
    default_weights: np.ndarray,
) -> np.ndarray:
    matrix = prior.loc[:, list(components)].to_numpy(dtype=float)
    actual = pd.to_numeric(
        prior["actual"],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    actual = actual[valid]
    if not len(actual):
        return default_weights.copy()

    def objective(weights: np.ndarray) -> float:
        return float(np.mean(np.square(matrix @ weights - actual)))

    fitted = minimize(
        objective,
        x0=default_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(components),
        constraints={
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
        },
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        return default_weights.copy()
    weights = np.clip(fitted.x, 0.0, 1.0)
    return weights / weights.sum()


def _add_causal_blends(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    specifications = {
        "causal_full_lasso_lgbm": (
            ("full_lasso", "full_lightgbm"),
            np.asarray((0.0, 1.0)),
        ),
        "causal_full_lasso_rf": (
            ("full_lasso", "full_random_forest"),
            np.asarray((0.0, 1.0)),
        ),
        "causal_full_lasso_tree_average": (
            ("full_lasso", "full_rf_lgbm_average"),
            np.asarray((0.0, 1.0)),
        ),
        "causal_full_lasso_rf_lgbm": (
            (
                "full_lasso",
                "full_random_forest",
                "full_lightgbm",
            ),
            np.asarray((0.0, 0.5, 0.5)),
        ),
    }
    weight_rows = []
    for method, (components, default_weights) in (
        specifications.items()
    ):
        frame[method] = np.nan
        for season, current in frame.groupby("season", sort=True):
            prior = frame[frame["season"].lt(season)]
            weights = default_weights.copy()
            if prior["season"].nunique() >= MIN_CAUSAL_SEASONS:
                weights = _fit_convex_weights(
                    prior,
                    components,
                    default_weights,
                )
            matrix = current.loc[
                :, list(components)
            ].to_numpy(dtype=float)
            frame.loc[current.index, method] = matrix @ weights
            weight_rows.extend(
                {
                    "method": method,
                    "season": int(season),
                    "component": component,
                    "weight": float(weight),
                    "prior_rows": len(prior),
                    "prior_seasons": int(
                        prior["season"].nunique()
                    ),
                }
                for component, weight in zip(components, weights)
            )
    return pd.DataFrame(weight_rows), list(specifications)


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    error = (
        pd.to_numeric(prediction, errors="coerce")
        - pd.to_numeric(actual, errors="coerce")
    )
    return float(np.sqrt(np.square(error).mean()))


def _score_rows(
    frame: pd.DataFrame,
    methods: Sequence[str],
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


def _score_predictions(
    frame: pd.DataFrame,
    methods: Sequence[str],
) -> pd.DataFrame:
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


def _comparison_reference(method: str) -> str:
    if method.endswith("_rf_average"):
        return "full_random_forest"
    if "equal_thirds" in method:
        return "full_rf_lgbm_average"
    if method.endswith("_lgbm_average"):
        return "full_lightgbm"
    if method == "causal_full_lasso_rf":
        return "full_random_forest"
    if method in {
        "causal_full_lasso_tree_average",
        "causal_full_lasso_rf_lgbm",
    }:
        return "full_rf_lgbm_average"
    return "full_lightgbm"


def _comparisons(
    scores: pd.DataFrame,
    challengers: Sequence[str],
) -> pd.DataFrame:
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
    rows = []
    for index, challenger in enumerate(challengers):
        reference = _comparison_reference(challenger)
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
        post_warmup = season.index.astype(int) >= 2019
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
                "post_warmup_mean_delta": float(
                    delta[post_warmup].mean()
                ),
                "recent_mean_delta": float(delta[recent].mean()),
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
    return pd.DataFrame(rows).sort_values("pooled_delta")


def _error_correlations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    methods = (
        "full_lasso",
        "projection_core_lasso",
        "projection_active_lasso",
        "full_random_forest",
        "full_lightgbm",
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
        "# Linear and Tree Blend Results",
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
            "| Challenger | Reference | Delta | Post-warmup | Recent | "
            "95% interval | Wins |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.challenger}` | `{row.reference}` | "
            f"{row.pooled_delta:+.4f} | "
            f"{row.post_warmup_mean_delta:+.4f} | "
            f"{row.recent_mean_delta:+.4f} | "
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
    run_id = create_run_id("m4a_linear_tree_blends")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    spec = _lasso_spec()
    feature_columns = tuple(
        dict.fromkeys((*full_features, *POSITION_FEATURES))
    )
    print("Fitting current-lineage governed full Lasso", flush=True)
    lasso_oof, parameters = run_model_spec(
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
    keys = ["player_key", "season"]
    metadata = lasso_oof[
        [
            *keys,
            "position",
            "actual",
            "year_exp",
            "is_rookie",
            "has_prior_outcome",
        ]
    ]
    full_lasso = lasso_oof[
        [*keys, "final_prediction"]
    ].rename(columns={"final_prediction": "full_lasso"})
    comparison = (
        metadata.merge(
            full_lasso,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            _load_source_predictions(),
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
            f"Expected {expected} current-lineage OOF rows, found "
            f"{len(comparison)}"
        )
    fixed_methods = _add_fixed_blends(comparison)
    weight_rows, causal_methods = _add_causal_blends(comparison)
    comparison["history_depth"] = _history_depth(comparison)
    comparison["history_group"] = np.where(
        comparison["history_depth"].isin(
            ("rookie", "second_year", "other_no_history")
        ),
        "limited",
        "veteran",
    )
    base_methods = [
        *LINEAR_CANDIDATES,
        "full_random_forest",
        "full_lightgbm",
        "full_rf_lgbm_average",
    ]
    methods = [*base_methods, *fixed_methods, *causal_methods]
    scores = _score_predictions(comparison, methods)
    challengers = [*fixed_methods, *causal_methods]
    comparisons = _comparisons(scores, challengers)
    correlations = _error_correlations(comparison)

    lasso_oof.to_csv(RESULTS_DIR / "full_lasso_oof.csv", index=False)
    parameters.to_csv(
        RESULTS_DIR / "hyperparameters.csv",
        index=False,
    )
    assignments.to_csv(
        RESULTS_DIR / "fold_assignments.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "feature_run_id": feature_run_id,
                "model_name": spec.model_name,
                "feature_count": len(feature_columns),
                "feature_names_json": json.dumps(feature_columns),
                "hyperparameters_json": json.dumps(
                    spec.parameters,
                    default=list,
                    sort_keys=True,
                ),
            }
        ]
    ).to_csv(
        RESULTS_DIR / "model_specification.csv",
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
    weight_rows.to_csv(
        RESULTS_DIR / "causal_blend_weights.csv",
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
