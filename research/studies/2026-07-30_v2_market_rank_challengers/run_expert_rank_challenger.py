"""Test normalized expert-rank features against the locked V2 PPG model.

The study is attribution-only.  It reuses the incumbent's strictly-prior,
per-origin selected hyperparameters and changes only the feature matrix.
Nothing is written to the V2 database.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.contracts import scoring_hash
from Scripts.V2.locked_candidates import (
    LOCKED_BLEND_WEIGHTS,
    PRIMARY_PPG_FEATURES,
    lock_version_for_scoring,
)


LOCKED_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_locked_final_validation"
    / "run_validation.py"
)
DEFAULT_DATABASES = {
    "dk": (
        STUDY_ROOT
        / "artifacts"
        / "local"
        / "Projection_V2_single_nffc.sqlite3"
    ),
    "beta": (
        STUDY_ROOT
        / "artifacts"
        / "local"
        / "Projection_V2_beta_single_nffc.sqlite3"
    ),
}
MODEL_COMPONENTS = tuple(LOCKED_BLEND_WEIGHTS)
RANDOM_FOREST_COMPONENT = "conditional_ppg_random_forest"
FULL_COLUMN_RF_METHOD = "random_forest_full_columns"
CONTROLLED_BLEND_METHOD = "controlled_equal_thirds"
VARIANT_FEATURES = {
    "incumbent": (),
    "rank_level": ("expert_rank_position_percentile_median",),
    "rank_gap": ("expert_projection_percentile_diff",),
}
RANDOM_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), default="dk")
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    return parser.parse_args()


def _load_locked_runner():
    spec = importlib.util.spec_from_file_location(
        "v2_locked_rank_attribution_runner",
        LOCKED_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load locked runner: {LOCKED_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rank_percentiles(market_values: pd.DataFrame) -> pd.DataFrame:
    ranks = market_values[
        market_values["expert_rank"].notna()
        & market_values["position"].isin(("QB", "RB", "WR", "TE"))
    ].copy()
    ranks["expert_rank"] = pd.to_numeric(
        ranks["expert_rank"], errors="raise"
    )
    keys = ["source", "season", "position"]
    ranks["_source_position_order"] = ranks.groupby(keys)[
        "expert_rank"
    ].rank(method="average", ascending=True)
    ranks["_source_position_size"] = ranks.groupby(keys)[
        "player_key"
    ].transform("count")
    denominator = ranks["_source_position_size"].sub(1)
    ranks["expert_rank_position_percentile"] = 1 - (
        ranks["_source_position_order"].sub(1)
        / denominator.where(denominator.gt(0))
    )
    ranks.loc[
        ranks["_source_position_size"].eq(1),
        "expert_rank_position_percentile",
    ] = 0.5
    return ranks


def _iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def build_normalized_rank_features(
    market_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = _rank_percentiles(market_values)
    keys = ["player_key", "season"]
    features = (
        ranks.groupby(keys, sort=True)
        .agg(
            expert_rank_position_percentile_median=(
                "expert_rank_position_percentile",
                "median",
            ),
            expert_rank_position_percentile_iqr=(
                "expert_rank_position_percentile",
                _iqr,
            ),
            expert_rank_source_count=("source", "nunique"),
        )
        .reset_index()
    )
    no_etr = (
        ranks[~ranks["source"].eq("etr_rank")]
        .groupby(keys, sort=True)["expert_rank_position_percentile"]
        .median()
        .rename("expert_rank_position_percentile_median_without_etr")
        .reset_index()
    )
    features = features.merge(
        no_etr,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    etr = (
        ranks[ranks["source"].eq("etr_rank")]
        .loc[
            :,
            keys + ["expert_rank_position_percentile"],
        ]
        .rename(
            columns={
                "expert_rank_position_percentile": (
                    "etr_rank_position_percentile"
                )
            }
        )
    )
    features = features.merge(
        etr,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    coverage = (
        ranks.groupby(["season", "source"], sort=True)
        .agg(
            ranked_players=("player_key", "nunique"),
            positions=("position", "nunique"),
        )
        .reset_index()
    )
    return features, coverage


def _feature_columns(variant: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*PRIMARY_PPG_FEATURES, *VARIANT_FEATURES[variant]))
    )


def _load_inputs(
    database: Path,
    league: str,
):
    locked = _load_locked_runner()
    locked.ACTIVE_OUTPUT_DB_PATH = database
    locked.ACTIVE_RESULTS_DIR = STUDY_ROOT / "artifacts" / "local"
    locked.ACTIVE_SCORING_OBJECTIVE = league
    locked.ACTIVE_LOCK_VERSION = lock_version_for_scoring(league)
    features, _, feature_run_id = locked._load_inputs()
    with sqlite3.connect(database) as connection:
        market_values = pd.read_sql_query(
            "SELECT * FROM player_season_market_values",
            connection,
        )
        selected = pd.read_sql_query(
            "SELECT * FROM locked_selected_hyperparameters",
            connection,
        )
        locked_predictions = pd.read_sql_query(
            "SELECT * FROM locked_whole_season_predictions",
            connection,
        )
    normalized, coverage = build_normalized_rank_features(market_values)
    features = features.merge(
        normalized,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    features["expert_projection_percentile_diff"] = (
        features["expert_rank_position_percentile_median"]
        - features["projection_position_percentile"]
    )
    return (
        locked,
        features,
        selected,
        locked_predictions,
        normalized,
        coverage,
        feature_run_id,
    )


def _run_predictions(
    locked,
    features: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    ppg, _, candidates = locked._target_frames(features)
    prediction_frames: list[pd.DataFrame] = []
    for variant in VARIANT_FEATURES:
        columns = _feature_columns(variant)
        missing = sorted(set(columns).difference(features.columns))
        if missing:
            raise ValueError(f"{variant} is missing feature columns: {missing}")
        for component in MODEL_COMPONENTS:
            component_selection = selected[
                selected["model_name"].eq(component)
            ].copy()
            prediction_frames.append(
                locked._selected_predictions(
                    ppg,
                    candidates,
                    columns,
                    fit_model_name=component,
                    output_model_name=f"{variant}__{component}",
                    selected=component_selection,
                )
            )

    full_column_selected = selected[
        selected["model_name"].eq(RANDOM_FOREST_COMPONENT)
    ].copy()
    full_column_selected["parameters_json"] = full_column_selected[
        "parameters_json"
    ].map(
        lambda value: json.dumps(
            {
                **json.loads(value),
                "max_features": 1.0,
            },
            sort_keys=True,
        )
    )
    for variant in VARIANT_FEATURES:
        prediction_frames.append(
            locked._selected_predictions(
                ppg,
                candidates,
                _feature_columns(variant),
                fit_model_name=RANDOM_FOREST_COMPONENT,
                output_model_name=(
                    f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns"
                ),
                selected=full_column_selected,
            )
        )

    long = pd.concat(prediction_frames, ignore_index=True)
    wide = long.pivot(
        index=["player_key", "season", "position"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    metadata_columns = [
        "player_key",
        "season",
        "position",
        "conditional_ppg",
        "conditional_ppg_training_eligible",
        "has_prior_outcome",
        "is_rookie",
        "year_exp",
        "expert_rank_position_percentile_median",
        "expert_rank_position_percentile_iqr",
        "expert_rank_source_count",
        "expert_rank_position_percentile_median_without_etr",
        "etr_rank_position_percentile",
        "expert_projection_percentile_diff",
        "projection_position_percentile",
    ]
    output = candidates.loc[:, metadata_columns].merge(
        wide,
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    output = locked._add_history_depth(output)
    for variant in VARIANT_FEATURES:
        component_columns = [
            f"{variant}__{component}" for component in MODEL_COMPONENTS
        ]
        component_values = output[component_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        weights = np.asarray(
            [LOCKED_BLEND_WEIGHTS[component] for component in MODEL_COMPONENTS],
            dtype=float,
        )
        output[f"{variant}__equal_thirds"] = component_values.to_numpy().dot(
            weights
        )
        output.loc[
            component_values.isna().any(axis=1),
            f"{variant}__equal_thirds",
        ] = np.nan
        controlled_columns = [
            f"{variant}__conditional_ppg_lasso",
            f"{variant}__conditional_ppg_lightgbm",
            f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns",
        ]
        controlled_values = output[controlled_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        output[f"{variant}__{CONTROLLED_BLEND_METHOD}"] = (
            controlled_values.mean(axis=1)
        )
        output.loc[
            controlled_values.isna().any(axis=1),
            f"{variant}__{CONTROLLED_BLEND_METHOD}",
        ] = np.nan
    return output


def _assert_incumbent_reproduces(
    predictions: pd.DataFrame,
    locked_predictions: pd.DataFrame,
) -> float:
    checks = {
        **{
            component: f"incumbent__{component}"
            for component in MODEL_COMPONENTS
        },
        "conditional_ppg_primary_blend": "incumbent__equal_thirds",
    }
    differences: list[float] = []
    for locked_method, challenger_column in checks.items():
        expected = locked_predictions[
            locked_predictions["method"].eq(locked_method)
        ][["player_key", "season", "prediction"]]
        observed = predictions[
            ["player_key", "season", challenger_column]
        ].rename(columns={challenger_column: "observed"})
        compared = expected.merge(
            observed,
            on=["player_key", "season"],
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["prediction", "observed"])
        if compared.empty:
            raise ValueError(f"No reproduction rows for {locked_method}")
        differences.extend(
            (compared["prediction"] - compared["observed"]).abs().tolist()
        )
    maximum = float(max(differences))
    if maximum > 1e-10:
        raise ValueError(
            "Incumbent attribution replay does not reproduce the locked "
            f"predictions: max_abs_delta={maximum}"
        )
    return maximum


def _evaluation_long(predictions: pd.DataFrame, outer_seasons: Sequence[int]):
    eligible = (
        predictions["season"].isin(outer_seasons)
        & predictions["conditional_ppg_training_eligible"].eq(1)
        & predictions["conditional_ppg"].notna()
    )
    metadata = [
        "player_key",
        "season",
        "position",
        "history_depth",
        "conditional_ppg",
        "expert_rank_source_count",
    ]
    rows = []
    for variant in VARIANT_FEATURES:
        methods = {
            component.removeprefix("conditional_ppg_"): (
                f"{variant}__{component}"
            )
            for component in MODEL_COMPONENTS
        }
        methods["equal_thirds"] = f"{variant}__equal_thirds"
        methods[FULL_COLUMN_RF_METHOD] = (
            f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns"
        )
        methods[CONTROLLED_BLEND_METHOD] = (
            f"{variant}__{CONTROLLED_BLEND_METHOD}"
        )
        for method, column in methods.items():
            current = predictions.loc[eligible, metadata].copy()
            current["variant"] = variant
            current["method"] = method
            current["actual"] = current.pop("conditional_ppg")
            current["prediction"] = predictions.loc[eligible, column].to_numpy()
            current = current[current["prediction"].notna()].copy()
            current["squared_error"] = (
                current["actual"] - current["prediction"]
            ) ** 2
            rows.append(current)
    return pd.concat(rows, ignore_index=True)


def _score_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, method), model in evaluation.groupby(
        ["variant", "method"], sort=True
    ):
        slices: list[tuple[str, str, pd.DataFrame]] = [
            ("pooled", "all", model),
            ("recent", "2023_2025", model[model["season"].ge(2023)]),
        ]
        slices.extend(
            ("season", str(value), group)
            for value, group in model.groupby("season", sort=True)
        )
        slices.extend(
            ("position", str(value), group)
            for value, group in model.groupby("position", sort=True)
        )
        slices.extend(
            ("history_depth", str(value), group)
            for value, group in model.groupby("history_depth", sort=True)
        )
        for slice_type, slice_value, group in slices:
            if group.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "n_rows": len(group),
                    "n_seasons": group["season"].nunique(),
                    "rmse": float(
                        np.sqrt(
                            mean_squared_error(
                                group["actual"],
                                group["prediction"],
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cluster_interval(
    compared: pd.DataFrame,
    cluster: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    grouped = (
        compared.groupby(cluster, sort=True)
        .agg(
            incumbent_sse=("incumbent_squared_error", "sum"),
            variant_sse=("variant_squared_error", "sum"),
            n_rows=("player_key", "size"),
        )
        .reset_index(drop=True)
    )
    values = grouped[
        ["incumbent_sse", "variant_sse", "n_rows"]
    ].to_numpy(dtype=float)
    deltas = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = values[
            rng.integers(0, len(values), size=len(values))
        ].sum(axis=0)
        deltas[index] = np.sqrt(sampled[1] / sampled[2]) - np.sqrt(
            sampled[0] / sampled[2]
        )
    return (
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def _variant_summary(
    evaluation: pd.DataFrame,
    scores: pd.DataFrame,
    iterations: int,
    *,
    method: str,
) -> pd.DataFrame:
    blend = evaluation[evaluation["method"].eq(method)].copy()
    incumbent = blend[blend["variant"].eq("incumbent")][
        ["player_key", "season", "squared_error"]
    ].rename(columns={"squared_error": "incumbent_squared_error"})
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for variant in ("rank_level", "rank_gap"):
        challenger = blend[blend["variant"].eq(variant)][
            ["player_key", "season", "squared_error"]
        ].rename(columns={"squared_error": "variant_squared_error"})
        compared = incumbent.merge(
            challenger,
            on=["player_key", "season"],
            how="inner",
            validate="one_to_one",
        )
        season_rmse = (
            compared.groupby("season", sort=True)
            .agg(
                incumbent_rmse=(
                    "incumbent_squared_error",
                    lambda value: float(np.sqrt(value.mean())),
                ),
                variant_rmse=(
                    "variant_squared_error",
                    lambda value: float(np.sqrt(value.mean())),
                ),
            )
            .reset_index()
        )
        season_rmse["delta"] = (
            season_rmse["variant_rmse"] - season_rmse["incumbent_rmse"]
        )
        pooled = scores[
            scores["variant"].isin(("incumbent", variant))
            & scores["method"].eq(method)
            & scores["slice_type"].eq("pooled")
        ].set_index("variant")["rmse"]
        recent = scores[
            scores["variant"].isin(("incumbent", variant))
            & scores["method"].eq(method)
            & scores["slice_type"].eq("recent")
        ].set_index("variant")["rmse"]
        season_low, season_high = _cluster_interval(
            compared,
            "season",
            iterations,
            rng,
        )
        player_low, player_high = _cluster_interval(
            compared,
            "player_key",
            iterations,
            rng,
        )
        rows.append(
            {
                "comparison_method": method,
                "variant": variant,
                "incumbent_rmse": float(pooled["incumbent"]),
                "variant_rmse": float(pooled[variant]),
                "pooled_delta_variant_minus_incumbent": float(
                    pooled[variant] - pooled["incumbent"]
                ),
                "recent_delta_variant_minus_incumbent": float(
                    recent[variant] - recent["incumbent"]
                ),
                "mean_season_delta": float(season_rmse["delta"].mean()),
                "median_season_delta": float(season_rmse["delta"].median()),
                "season_wins": int(season_rmse["delta"].lt(0).sum()),
                "season_count": len(season_rmse),
                "season_bootstrap_95_low": season_low,
                "season_bootstrap_95_high": season_high,
                "player_cluster_95_low": player_low,
                "player_cluster_95_high": player_high,
            }
        )
    return pd.DataFrame(rows)


def _etr_diagnostic(normalized: pd.DataFrame) -> pd.DataFrame:
    diagnostic = normalized[
        normalized["etr_rank_position_percentile"].notna()
    ].copy()
    diagnostic["all_minus_without_etr"] = (
        diagnostic["expert_rank_position_percentile_median"]
        - diagnostic["expert_rank_position_percentile_median_without_etr"]
    )
    rows = []
    for season, group in diagnostic.groupby("season", sort=True):
        delta = group["all_minus_without_etr"].dropna()
        rows.append(
            {
                "season": int(season),
                "etr_ranked_players": len(group),
                "leave_etr_out_comparable_players": len(delta),
                "mean_all_minus_without_etr": (
                    float(delta.mean()) if len(delta) else np.nan
                ),
                "median_absolute_all_minus_without_etr": (
                    float(delta.abs().median()) if len(delta) else np.nan
                ),
                "maximum_absolute_all_minus_without_etr": (
                    float(delta.abs().max()) if len(delta) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _findings_markdown(
    league: str,
    database: Path,
    feature_run_id: str,
    reproduction_delta: float,
    summary: pd.DataFrame,
    production_surface_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    lines = [
        f"# Normalized Expert-Rank Challenger — {league.upper()}",
        "",
        "## Method",
        "",
        "- Expert ranks are converted to within-source, within-season, "
        "within-position percentiles before taking the cross-source median.",
        "- The incumbent, rank-level, and expert-minus-projection gap matrices "
        "reuse the incumbent's strictly-prior selected hyperparameters.",
        "- Primary attribution replaces the random forest's 50% feature "
        "subsampling with full-column forests on both sides. This prevents the "
        "added column from changing which incumbent columns are sampled.",
        "- Every forecast origin is fit only on earlier seasons. Fold-local "
        "median imputation and missing indicators are unchanged.",
        "- This is an attribution study, not a production promotion.",
        "",
        "## Results",
        "",
        "| Variant | RMSE | Pooled delta | Recent delta | Mean season delta | "
        "Season wins | Season 95% | Player 95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.variant_rmse:.4f} | "
            f"{row.pooled_delta_variant_minus_incumbent:+.4f} | "
            f"{row.recent_delta_variant_minus_incumbent:+.4f} | "
            f"{row.mean_season_delta:+.4f} | "
            f"{row.season_wins}/{row.season_count} | "
            f"[{row.season_bootstrap_95_low:+.4f}, "
            f"{row.season_bootstrap_95_high:+.4f}] | "
            f"[{row.player_cluster_95_low:+.4f}, "
            f"{row.player_cluster_95_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "The production-surface sensitivity retains the locked 50% "
            "random-forest feature subsampling. It is reported separately "
            "because adding a column changes the sampled feature set:",
            "",
            "| Variant | Production-surface RMSE | Delta |",
            "|---|---:|---:|",
        ]
    )
    for row in production_surface_summary.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.variant_rmse:.4f} | "
            f"{row.pooled_delta_variant_minus_incumbent:+.4f} |"
        )
    current_sources = coverage.loc[
        coverage["season"].eq(2026), "source"
    ].nunique()
    lines.extend(
        [
            "",
            "Negative deltas improve on the incumbent.",
            "",
            "## Governance",
            "",
            f"- Feature run: `{feature_run_id}`",
            f"- Staged database: `{database.resolve()}`",
            f"- Locked-incumbent reproduction max delta: "
            f"`{reproduction_delta:.3g}`",
            f"- 2026 normalized rank providers: {current_sources}",
            "- A dedicated ETR coefficient is not tested because ETR has only "
            "2024-2026 half-PPR history and 2025-2026 full-PPR history. ETR "
            "instead contributes one normalized vote to the cross-provider "
            "rank consensus.",
            "- Promotion requires a favorable pooled result without a recent "
            "reversal and uncertainty intervals that support a stable gain.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    league = args.league
    database = (
        args.output_db
        if args.output_db is not None
        else DEFAULT_DATABASES[league]
    )
    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else STUDY_ROOT / "results" / f"expert_rank_{league}"
    )
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    results_dir.mkdir(parents=True, exist_ok=True)

    (
        locked,
        features,
        selected,
        locked_predictions,
        normalized,
        coverage,
        feature_run_id,
    ) = _load_inputs(database, league)
    observed_hashes = set(features["scoring_hash"].dropna().astype(str))
    expected_hash = scoring_hash(league)
    if observed_hashes != {expected_hash}:
        raise ValueError(
            f"Scoring mismatch: observed={observed_hashes}, "
            f"expected={expected_hash}"
        )

    predictions = _run_predictions(locked, features, selected)
    reproduction_delta = _assert_incumbent_reproduces(
        predictions,
        locked_predictions,
    )
    evaluation = _evaluation_long(predictions, locked.OUTER_SEASONS)
    scores = _score_table(evaluation)
    production_surface_summary = _variant_summary(
        evaluation,
        scores,
        args.bootstrap_iterations,
        method="equal_thirds",
    )
    summary = _variant_summary(
        evaluation,
        scores,
        args.bootstrap_iterations,
        method=CONTROLLED_BLEND_METHOD,
    )
    etr = _etr_diagnostic(normalized)
    shadow = predictions[predictions["season"].eq(locked.CURRENT_SEASON)].copy()

    normalized.to_csv(
        results_dir / "normalized_expert_rank_features.csv",
        index=False,
    )
    coverage.to_csv(results_dir / "rank_source_coverage.csv", index=False)
    evaluation.to_csv(results_dir / "oof_predictions.csv", index=False)
    scores.to_csv(results_dir / "model_scores.csv", index=False)
    summary.to_csv(results_dir / "variant_summary.csv", index=False)
    production_surface_summary.to_csv(
        results_dir / "production_surface_variant_summary.csv",
        index=False,
    )
    etr.to_csv(results_dir / "etr_leaveout_diagnostic.csv", index=False)
    shadow.to_csv(results_dir / "shadow_predictions.csv", index=False)
    (results_dir / "findings.md").write_text(
        _findings_markdown(
            league,
            database,
            feature_run_id,
            reproduction_delta,
            summary,
            production_surface_summary,
            coverage,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "league": league,
                "database": str(database.resolve()),
                "feature_run_id": feature_run_id,
                "locked_reproduction_max_abs_delta": reproduction_delta,
                "summary": summary.to_dict("records"),
                "results_directory": str(results_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
