"""Compare pooled-median history levels with projection-anchored gaps."""

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
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234
MIN_CAUSAL_SEASONS = 2

REPLACED_HISTORY_FEATURES = {
    "career_weighted_ppg",
    "prior_year_ppg",
    "prior_year_ppg_residual",
    "prior_3year_weighted_ppg",
    "seasons_since_observed",
}
GAP_COMMON_FEATURES = (
    "history_career_opportunity_games_log",
    "history_prior_year_opportunity_games_log",
    "history_prior_3year_opportunity_games_log",
    "history_prior_year_ppg_available",
    "history_prior_3year_ppg_available",
    "history_prior_year_residual_neutral",
    "history_seasons_since_observed_neutral",
)
RAW_GAP_FEATURES = (
    "history_career_ppg_gap",
    "history_prior_year_ppg_gap",
    "history_prior_3year_ppg_gap",
)
SHRUNK_GAP_FEATURES = (
    "history_career_ppg_gap_shrunk",
    "history_prior_year_ppg_gap_shrunk",
    "history_prior_3year_ppg_gap_shrunk",
)

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
    gap_manifest = set(
        _manifest(manifests, "residual_history_gap_challenger_v1")
    )
    required = set(
        (*GAP_COMMON_FEATURES, *RAW_GAP_FEATURES, *SHRUNK_GAP_FEATURES)
    )
    if not required.issubset(gap_manifest):
        raise ValueError("History-gap manifest is incomplete")
    retained = tuple(
        feature
        for feature in incumbent
        if feature not in REPLACED_HISTORY_FEATURES
    )
    return {
        "incumbent": tuple(
            dict.fromkeys((*incumbent, *POSITION_FEATURES))
        ),
        "gap_raw": tuple(
            dict.fromkeys(
                (
                    *retained,
                    *GAP_COMMON_FEATURES,
                    *RAW_GAP_FEATURES,
                    *POSITION_FEATURES,
                )
            )
        ),
        "gap_shrunk": tuple(
            dict.fromkeys(
                (
                    *retained,
                    *GAP_COMMON_FEATURES,
                    *SHRUNK_GAP_FEATURES,
                    *POSITION_FEATURES,
                )
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


def _projection_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=(-np.inf, 8.0, 12.0, 16.0, np.inf),
        labels=("under_8", "8_to_12", "12_to_16", "16_plus"),
        right=False,
    ).astype(object)


def _history_sample(frame: pd.DataFrame) -> pd.Series:
    rookie = pd.to_numeric(
        frame["is_rookie"], errors="coerce"
    ).fillna(0).eq(1)
    career_games = pd.to_numeric(
        frame["career_opportunity_games"], errors="coerce"
    ).fillna(0)
    prior_games = pd.to_numeric(
        frame["prior_year_opportunity_games"], errors="coerce"
    ).fillna(0)
    recency = pd.to_numeric(
        frame["seasons_since_observed"], errors="coerce"
    )
    result = pd.Series(
        "prior_8_plus_games",
        index=frame.index,
        dtype=object,
    )
    result.loc[rookie] = "rookie"
    result.loc[~rookie & career_games.eq(0)] = "no_career_history"
    result.loc[
        ~rookie & career_games.gt(0) & recency.ge(2)
    ] = "returning_after_gap"
    result.loc[
        ~rookie & prior_games.between(1, 3)
    ] = "prior_1_to_3_games"
    result.loc[
        ~rookie & prior_games.between(4, 7)
    ] = "prior_4_to_7_games"
    return result


def _fit_causal_lasso_weight(
    prior: pd.DataFrame,
    lasso: str,
    tree_average: str,
) -> float:
    linear_delta = (
        pd.to_numeric(prior[lasso], errors="coerce")
        - pd.to_numeric(prior[tree_average], errors="coerce")
    ).to_numpy(dtype=float)
    target_delta = (
        pd.to_numeric(prior["actual"], errors="coerce")
        - pd.to_numeric(prior[tree_average], errors="coerce")
    ).to_numpy(dtype=float)
    valid = np.isfinite(linear_delta) & np.isfinite(target_delta)
    linear_delta = linear_delta[valid]
    target_delta = target_delta[valid]
    denominator = float(np.square(linear_delta).sum())
    if denominator <= 0:
        return 0.0
    weight = float(np.dot(linear_delta, target_delta) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def _add_blends(
    comparison: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    methods: list[str] = []
    weight_rows: list[dict[str, object]] = []
    for variant in ("incumbent", "gap_raw", "gap_shrunk"):
        lasso = f"{variant}_lasso"
        rf = f"{variant}_random_forest"
        lgbm = f"{variant}_lightgbm"
        tree = f"{variant}_tree_average"
        equal = f"{variant}_equal_thirds"
        causal = f"{variant}_causal_lasso_tree"
        comparison[tree] = (comparison[rf] + comparison[lgbm]) / 2
        comparison[equal] = (
            comparison[lasso] + comparison[rf] + comparison[lgbm]
        ) / 3
        comparison[causal] = np.nan
        for season, current in comparison.groupby("season", sort=True):
            prior = comparison[comparison["season"].lt(season)]
            weight = 0.0
            if prior["season"].nunique() >= MIN_CAUSAL_SEASONS:
                weight = _fit_causal_lasso_weight(prior, lasso, tree)
            comparison.loc[current.index, causal] = (
                weight * comparison.loc[current.index, lasso]
                + (1 - weight) * comparison.loc[current.index, tree]
            )
            weight_rows.append(
                {
                    "variant": variant,
                    "season": int(season),
                    "lasso_weight": weight,
                    "tree_weight": 1 - weight,
                    "prior_rows": len(prior),
                    "prior_seasons": int(prior["season"].nunique()),
                }
            )
        methods.extend((tree, equal, causal))
    return methods, pd.DataFrame(weight_rows)


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
            values = {
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
                for metric, value in values.items()
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
        ("history_sample", "history_sample"),
        ("projection_band", "projection_band"),
    ):
        rows.extend(
            _score_rows(frame, methods, slice_type, column)
        )
    return pd.DataFrame(rows)


def _sign_flip_pvalue(delta: np.ndarray) -> float:
    observed = abs(float(delta.mean()))
    means = []
    for mask in range(1 << len(delta)):
        signs = np.asarray(
            [1 if mask & (1 << index) else -1 for index in range(len(delta))]
        )
        means.append(abs(float((delta * signs).mean())))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def _comparison_pairs() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for variant in ("gap_raw", "gap_shrunk"):
        for family in ("lasso", "random_forest", "lightgbm"):
            rows.append(
                (
                    f"{variant}_{family}",
                    f"incumbent_{family}",
                    f"{variant}_vs_incumbent_{family}",
                )
            )
        for suffix in (
            "tree_average",
            "equal_thirds",
            "causal_lasso_tree",
        ):
            rows.append(
                (
                    f"{variant}_{suffix}",
                    f"incumbent_{suffix}",
                    f"{variant}_vs_incumbent_{suffix}",
                )
            )
    rows.extend(
        (
            (
                "incumbent_equal_thirds",
                "incumbent_tree_average",
                "incumbent_equal_thirds_vs_tree",
            ),
            (
                "gap_raw_equal_thirds",
                "gap_raw_tree_average",
                "gap_raw_equal_thirds_vs_tree",
            ),
            (
                "gap_shrunk_equal_thirds",
                "gap_shrunk_tree_average",
                "gap_shrunk_equal_thirds_vs_tree",
            ),
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


def _gap_audit(target: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *RAW_GAP_FEATURES,
        *SHRUNK_GAP_FEATURES,
        *GAP_COMMON_FEATURES,
    ]
    missing = target[columns].isna().sum()
    no_career = pd.to_numeric(
        target["career_opportunity_games"], errors="coerce"
    ).fillna(0).eq(0)
    no_prior = pd.to_numeric(
        target["prior_year_opportunity_games"], errors="coerce"
    ).fillna(0).eq(0)
    if target.loc[no_career, "history_career_ppg_gap"].ne(0).any():
        raise ValueError("No-career rows must have a neutral career gap")
    if target.loc[no_prior, "history_prior_year_ppg_gap"].ne(0).any():
        raise ValueError("No-prior-year rows must have a neutral prior gap")
    rows = []
    for column in columns:
        rows.append(
            {
                "feature": column,
                "non_null_rows": int(target[column].notna().sum()),
                "missing_rows": int(missing[column]),
                "zero_rows": int(
                    pd.to_numeric(target[column], errors="coerce")
                    .fillna(np.inf)
                    .eq(0)
                    .sum()
                ),
            }
        )
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
        "# Projection-Anchored History Gap Results",
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
    run_id = create_run_id("m4a_history_gap")
    assignments = make_fold_assignments(
        target,
        CONDITIONAL_PPG_TARGET,
        run_id,
        VALIDATION_START,
        N_SPLITS,
        RANDOM_SEED,
    )
    oof_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    specification_rows: list[dict[str, object]] = []
    for variant, feature_columns in variants.items():
        for family in ("lasso", "random_forest", "lightgbm"):
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
            "career_opportunity_games",
            "prior_year_opportunity_games",
            "seasons_since_observed",
            "actual_target",
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
    blend_methods, weights = _add_blends(comparison)
    comparison["history_sample"] = _history_sample(comparison)
    comparison["projection_band"] = _projection_band(
        comparison["baseline_prediction"]
    )
    model_methods = sorted(oof_all["model_name"].unique())
    methods = [*model_methods, *blend_methods]
    scores = _score_predictions(comparison, methods)
    paired = _comparisons(scores)
    audit = _gap_audit(
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
    pd.DataFrame(specification_rows).to_csv(
        RESULTS_DIR / "model_specifications.csv",
        index=False,
    )
    comparison.to_csv(
        RESULTS_DIR / "comparison_predictions.csv",
        index=False,
    )
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    paired.to_csv(
        RESULTS_DIR / "model_comparisons.csv",
        index=False,
    )
    weights.to_csv(
        RESULTS_DIR / "causal_blend_weights.csv",
        index=False,
    )
    audit.to_csv(RESULTS_DIR / "gap_feature_audit.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, paired),
        encoding="utf-8",
    )
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
