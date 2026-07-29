"""Test preseason projection trajectory and logged ADP in V2 PPG models."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
HISTORY_GAP_RESULTS = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_history_gap_features"
    / "results"
)
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234

TRAJECTORY_FEATURES = (
    "projection_trajectory_change_1year",
    "projection_trajectory_change_3year",
    "projection_trajectory_prior_year_available",
    "projection_trajectory_prior_3year_count",
    "projection_trajectory_prior_3year_std",
)
VARIANTS = (
    "incumbent",
    "trajectory",
    "log_adp",
    "trajectory_log_adp",
)
FAMILIES = ("lasso", "random_forest", "lightgbm")

MODEL_PARAMETERS = {
    "lasso": {
        "model_piece": "lasso",
        "search_iterations": 7,
        "parameters": {
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
    "lightgbm": {
        "model_piece": "lgbm",
        "search_iterations": 4,
        "parameters": {
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
    },
}


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
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
    manifest_runs = set(
        manifests["run_id"].dropna().astype(str).unique()
    )
    if manifest_runs != {feature_run_id}:
        raise ValueError("Feature manifests do not match the active mart")
    return features, manifests, feature_run_id


def _manifest(
    manifests: pd.DataFrame,
    name: str,
) -> tuple[str, ...]:
    values = tuple(
        manifests.loc[
            manifests["manifest_name"].eq(name),
            "feature_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not values:
        raise ValueError(f"Missing feature manifest: {name}")
    return values


def _feature_variants(
    manifests: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    incumbent = _manifest(manifests, "residual_candidate_v1")
    trajectory = set(
        _manifest(
            manifests,
            "residual_projection_trajectory_challenger_v1",
        )
    )
    adp_transform = set(
        _manifest(
            manifests,
            "residual_adp_transform_challenger_v1",
        )
    )
    if trajectory != set(TRAJECTORY_FEATURES):
        raise ValueError("Projection-trajectory manifest is unexpected")
    if adp_transform != {"adp_log"}:
        raise ValueError("ADP-transform manifest is unexpected")
    logged = tuple(
        "adp_log" if feature == "adp_median" else feature
        for feature in incumbent
    )
    return {
        "incumbent": tuple(
            dict.fromkeys((*incumbent, *POSITION_FEATURES))
        ),
        "trajectory": tuple(
            dict.fromkeys(
                (*incumbent, *TRAJECTORY_FEATURES, *POSITION_FEATURES)
            )
        ),
        "log_adp": tuple(
            dict.fromkeys((*logged, *POSITION_FEATURES))
        ),
        "trajectory_log_adp": tuple(
            dict.fromkeys(
                (*logged, *TRAJECTORY_FEATURES, *POSITION_FEATURES)
            )
        ),
    }


def _model_spec(family: str, variant: str) -> ModelSpec:
    config = MODEL_PARAMETERS[family]
    return ModelSpec(
        target_name=CONDITIONAL_PPG_TARGET,
        model_name=f"{variant}_{family}",
        model_family=family,
        prediction_kind="direct",
        feature_set=variant,
        pipeline_variant="raw",
        model_piece=str(config["model_piece"]),
        parameters=config["parameters"],
        search_iterations=int(config["search_iterations"]),
    )


def _load_incumbent_oof(
    target: pd.DataFrame,
    assignments: pd.DataFrame,
    incumbent_features: Sequence[str],
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(HISTORY_GAP_RESULTS / "oof_predictions.csv")
    source = source[
        source["model_name"].isin(
            tuple(f"incumbent_{family}" for family in FAMILIES)
        )
    ].copy()
    specifications = pd.read_csv(
        HISTORY_GAP_RESULTS / "model_specifications.csv"
    )
    specifications = specifications[
        specifications["model_name"].isin(source["model_name"].unique())
    ].copy()
    expected_features = set(incumbent_features)
    for row in specifications.itertuples(index=False):
        old_features = set(json.loads(row.feature_names_json))
        if old_features != expected_features:
            raise ValueError(
                f"Incumbent feature bridge failed for {row.model_name}"
            )
    keys = ["player_key", "season"]
    validation_target = target[
        target["season"].between(VALIDATION_START, VALIDATION_END)
    ][[*keys, "actual_target", "baseline_prediction"]]
    source_check = source[
        [
            *keys,
            "actual",
            "baseline_prediction",
        ]
    ].drop_duplicates(keys)
    check = validation_target.merge(
        source_check,
        on=keys,
        how="outer",
        suffixes=("_current", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not check["_merge"].eq("both").all():
        raise ValueError("Incumbent OOF target keys changed")
    for column in ("actual", "baseline_prediction"):
        current_name = (
            "actual_target"
            if column == "actual"
            else f"{column}_current"
        )
        source_name = (
            "actual"
            if column == "actual"
            else f"{column}_source"
        )
        if not np.allclose(
            check[current_name],
            check[source_name],
            equal_nan=True,
        ):
            raise ValueError(f"Incumbent OOF {column} changed")

    old_assignments = pd.read_csv(
        HISTORY_GAP_RESULTS / "fold_assignments.csv"
    )
    fold_check = assignments[
        [*keys, "fold", "training_through_season"]
    ].merge(
        old_assignments[
            [*keys, "fold", "training_through_season"]
        ],
        on=keys,
        how="outer",
        suffixes=("_current", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not fold_check["_merge"].eq("both").all():
        raise ValueError("Incumbent OOF fold keys changed")
    if not (
        fold_check["fold_current"].eq(fold_check["fold_source"])
        & fold_check["training_through_season_current"].eq(
            fold_check["training_through_season_source"]
        )
    ).all():
        raise ValueError("Incumbent OOF fold assignments changed")
    oof_frames = [
        group.copy()
        for _, group in source.groupby("model_name", sort=True)
    ]
    bridge = pd.DataFrame(
        [
            {
                "source_study": str(HISTORY_GAP_RESULTS.parent),
                "source_feature_run_id": source["feature_run_id"].iloc[0],
                "current_feature_run_id": target["run_id"].iloc[0],
                "target_rows": len(validation_target),
                "target_keys_equal": 1,
                "actuals_equal": 1,
                "baselines_equal": 1,
                "folds_equal": 1,
                "incumbent_feature_names_equal": 1,
                "bridge_reason": (
                    "additive trajectory/adp mart columns; incumbent "
                    "features and targets unchanged"
                ),
            }
        ]
    )
    return oof_frames, specifications, bridge


def _history_depth(frame: pd.DataFrame) -> pd.Series:
    rookie = pd.to_numeric(
        frame["is_rookie"], errors="coerce"
    ).fillna(0).eq(1)
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
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
    result.loc[~rookie & year_exp.ge(2) & prior] = "veteran"
    return result


def _projection_history_depth(frame: pd.DataFrame) -> pd.Series:
    prior_year = pd.to_numeric(
        frame["projection_trajectory_prior_year_available"],
        errors="coerce",
    ).fillna(0).eq(1)
    count = pd.to_numeric(
        frame["projection_trajectory_prior_3year_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    result = pd.Series("no_prior_projection", index=frame.index, dtype=object)
    result.loc[count.eq(1)] = "one_prior_projection"
    result.loc[count.eq(2)] = "two_prior_projections"
    result.loc[count.ge(3)] = "three_prior_projections"
    result.loc[
        prior_year & count.eq(1)
    ] = "exact_prior_only"
    return result


def _adp_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=(-np.inf, 50, 100, 200, np.inf),
        labels=("top_50", "50_to_100", "100_to_200", "200_plus"),
        right=False,
    ).astype(object).fillna("missing")


def _add_blends(comparison: pd.DataFrame) -> list[str]:
    methods: list[str] = []
    for variant in VARIANTS:
        lasso = f"{variant}_lasso"
        rf = f"{variant}_random_forest"
        lgbm = f"{variant}_lightgbm"
        tree = f"{variant}_tree_average"
        equal = f"{variant}_equal_thirds"
        comparison[tree] = (comparison[rf] + comparison[lgbm]) / 2
        comparison[equal] = (
            comparison[lasso] + comparison[rf] + comparison[lgbm]
        ) / 3
        methods.extend((tree, equal))
    return methods


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
    rows: list[dict[str, object]] = []
    for slice_value, group in groups:
        actual = pd.to_numeric(group["actual"], errors="coerce")
        for method in methods:
            prediction = pd.to_numeric(group[method], errors="coerce")
            valid = actual.notna() & prediction.notna()
            observed = actual[valid]
            predicted = prediction[valid]
            error = predicted - observed
            correlation = (
                float(spearmanr(observed, predicted).statistic)
                if len(observed) > 1
                else np.nan
            )
            metrics = {
                "rmse": float(np.sqrt(np.square(error).mean())),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
                "spearman": correlation,
            }
            rows.extend(
                {
                    "method": method,
                    "slice_type": slice_type,
                    "slice_value": str(slice_value),
                    "metric": metric,
                    "n_rows": int(valid.sum()),
                    "value": value,
                }
                for metric, value in metrics.items()
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
        ("projection_history", "projection_history"),
        ("adp_band", "adp_band"),
    ):
        rows.extend(_score_rows(frame, methods, slice_type, column))
    return pd.DataFrame(rows)


def _sign_flip_pvalue(delta: np.ndarray) -> float:
    observed = abs(float(delta.mean()))
    values = []
    for mask in range(1 << len(delta)):
        signs = np.asarray(
            [1 if mask & (1 << index) else -1 for index in range(len(delta))]
        )
        values.append(abs(float((delta * signs).mean())))
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def _comparison_pairs() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for variant in VARIANTS[1:]:
        for family in FAMILIES:
            rows.append(
                (
                    f"{variant}_{family}",
                    f"incumbent_{family}",
                    f"{variant}_vs_incumbent_{family}",
                )
            )
        for suffix in ("tree_average", "equal_thirds"):
            rows.append(
                (
                    f"{variant}_{suffix}",
                    f"incumbent_{suffix}",
                    f"{variant}_vs_incumbent_{suffix}",
                )
            )
        rows.append(
            (
                f"{variant}_equal_thirds",
                f"{variant}_tree_average",
                f"{variant}_equal_thirds_vs_tree",
            )
        )
    return rows


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
    rows: list[dict[str, object]] = []
    for index, (challenger, reference, comparison) in enumerate(
        _comparison_pairs()
    ):
        delta = (
            season[challenger] - season[reference]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + index)
        draws = np.asarray(
            [
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        recent = season.index.astype(int) >= 2023
        rows.append(
            {
                "comparison": comparison,
                "challenger": challenger,
                "reference": reference,
                "challenger_rmse": float(pooled[challenger]),
                "reference_rmse": float(pooled[reference]),
                "pooled_delta": float(
                    pooled[challenger] - pooled[reference]
                ),
                "mean_season_delta": float(delta.mean()),
                "recent_mean_delta": float(delta[recent].mean()),
                "season_wins": int((delta < 0).sum()),
                "season_count": len(delta),
                "sign_flip_pvalue": _sign_flip_pvalue(delta),
                "bootstrap_95_low": float(np.quantile(draws, 0.025)),
                "bootstrap_95_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_delta")


def _feature_audit(target: pd.DataFrame) -> pd.DataFrame:
    columns = [*TRAJECTORY_FEATURES, "adp_median", "adp_log"]
    rows = []
    for column in columns:
        values = pd.to_numeric(target[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "non_null_rows": int(values.notna().sum()),
                "missing_rows": int(values.isna().sum()),
                "zero_rows": int(values.fillna(np.inf).eq(0).sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        )
    no_prior = pd.to_numeric(
        target["projection_trajectory_prior_3year_count"],
        errors="coerce",
    ).fillna(0).eq(0)
    if target.loc[
        no_prior, "projection_trajectory_change_1year"
    ].ne(0).any():
        raise ValueError("No-projection-history one-year gap must be zero")
    if target.loc[
        no_prior, "projection_trajectory_change_3year"
    ].ne(0).any():
        raise ValueError("No-projection-history three-year gap must be zero")
    return pd.DataFrame(rows)


def _summary_markdown(
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    lines = [
        "# Projection Trajectory and Logged-ADP Results",
        "",
        "Negative deltas favor the challenger.",
        "",
        "## Pooled OOF",
        "",
        "| Method | RMSE |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{row.method}` | {row.value:.4f} |"
        for row in pooled.itertuples(index=False)
    )
    lines.extend(
        (
            "",
            "## Paired season comparisons",
            "",
            "| Challenger | Reference | Delta | Recent | "
            "95% interval | Wins | Sign-flip p |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.challenger}` | `{row.reference}` | "
            f"{row.pooled_delta:+.4f} | "
            f"{row.recent_mean_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.season_wins}/{row.season_count} | "
            f"{row.sign_flip_pvalue:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features, manifests, feature_run_id = _load_inputs()
    target = build_target_frames(
        features,
        VALIDATION_END,
    )[CONDITIONAL_PPG_TARGET]
    variants = _feature_variants(manifests)
    run_id = create_run_id("m4a_projection_trajectory_adp")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    incumbent_oof, incumbent_specs, bridge = _load_incumbent_oof(
        target,
        assignments,
        variants["incumbent"],
    )
    oof_frames = list(incumbent_oof)
    parameter_frames: list[pd.DataFrame] = []
    specification_rows: list[dict[str, object]] = []
    for variant in VARIANTS[1:]:
        feature_columns = variants[variant]
        for family in FAMILIES:
            spec = _model_spec(family, variant)
            print(f"Fitting {spec.model_name}", flush=True)
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
            oof_frames.append(oof)
            parameter_frames.append(parameters)
            specification_rows.append(
                {
                    "run_id": run_id,
                    "feature_run_id": feature_run_id,
                    "model_name": spec.model_name,
                    "model_family": family,
                    "feature_variant": variant,
                    "feature_count": len(feature_columns),
                    "feature_names_json": json.dumps(feature_columns),
                    "hyperparameters_json": json.dumps(
                        spec.parameters,
                        default=list,
                        sort_keys=True,
                    ),
                }
            )

    oof_all = pd.concat(oof_frames, ignore_index=True)
    expected_rows = int(
        target["season"].between(
            VALIDATION_START,
            VALIDATION_END,
        ).sum()
    )
    counts = oof_all.groupby("model_name").size()
    if len(counts) != len(VARIANTS) * len(FAMILIES):
        raise ValueError(f"Unexpected model count: {counts.to_dict()}")
    if not counts.eq(expected_rows).all():
        raise ValueError(f"Incomplete OOF models: {counts.to_dict()}")
    keys = ["player_key", "season"]
    metadata = target.loc[
        target["season"].between(VALIDATION_START, VALIDATION_END),
        [
            *keys,
            "position",
            "baseline_prediction",
            "year_exp",
            "is_rookie",
            "has_prior_outcome",
            "adp_median",
            "actual_target",
            *TRAJECTORY_FEATURES,
        ],
    ].rename(columns={"actual_target": "actual"})
    wide = oof_all.pivot(
        index=keys,
        columns="model_name",
        values="final_prediction",
    ).reset_index()
    comparison = metadata.merge(
        wide,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} comparison rows, found "
            f"{len(comparison)}"
        )
    blend_methods = _add_blends(comparison)
    comparison["history_depth"] = _history_depth(comparison)
    comparison["projection_history"] = _projection_history_depth(comparison)
    comparison["adp_band"] = _adp_band(comparison["adp_median"])
    model_methods = sorted(oof_all["model_name"].unique())
    methods = [*model_methods, *blend_methods]
    scores = _score_predictions(comparison, methods)
    paired = _comparisons(scores)
    audit = _feature_audit(
        target[target["season"].between(VALIDATION_START, VALIDATION_END)]
    )

    oof_all.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "hyperparameters.csv",
        index=False,
    )
    assignments.to_csv(
        RESULTS_DIR / "fold_assignments.csv",
        index=False,
    )
    current_specs = pd.DataFrame(specification_rows)
    incumbent_specs.to_csv(
        RESULTS_DIR / "source_incumbent_specifications.csv",
        index=False,
    )
    current_specs.to_csv(
        RESULTS_DIR / "model_specifications.csv",
        index=False,
    )
    bridge.to_csv(RESULTS_DIR / "lineage_bridge.csv", index=False)
    comparison.to_csv(
        RESULTS_DIR / "comparison_predictions.csv",
        index=False,
    )
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    paired.to_csv(
        RESULTS_DIR / "model_comparisons.csv",
        index=False,
    )
    audit.to_csv(RESULTS_DIR / "feature_audit.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, paired),
        encoding="utf-8",
    )
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
