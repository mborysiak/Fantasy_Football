"""Evaluate logged within-position expert-rank disagreement in V2.

This is an isolated, read-only study. It reuses the locked point-model replay
and scoring-specific rank lineage from the 2026-07-30 rank challenger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parent
PRIOR_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-30_v2_market_rank_challengers"
    / "run_raw_rank_challenger.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.contracts import scoring_hash


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
VARIANT_FEATURES = {
    "incumbent": (),
    "rank_level": ("scoring_specific_rank_position_percentile_median",),
    "rank_level_logged": (
        "scoring_specific_rank_position_percentile_median",
        "expert_rank_logged_mad",
        "rank_source_coverage",
        "observed_rank_source_count",
    ),
    "rank_level_excess": (
        "scoring_specific_rank_position_percentile_median",
        "expert_rank_logged_mad_excess",
        "rank_source_coverage",
        "observed_rank_source_count",
    ),
}
CONTROLLED_METHOD = "controlled_equal_thirds"
PRODUCTION_METHOD = "equal_thirds"
POINT_COMPARISONS = {
    "rank_level": "incumbent",
    "rank_level_logged": "rank_level",
    "rank_level_excess": "rank_level",
}
SCALE_FEATURES = {
    "scale_rank_level": (),
    "scale_logged": ("expert_rank_logged_mad",),
    "scale_excess": ("expert_rank_logged_mad_excess",),
}
RANDOM_SEED = 1234
MIN_EXPECTED_GROUP_ROWS = 20
GAUSSIAN_80_Z = 1.2815515655446004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta", "all"), default="all")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--combine-existing",
        action="store_true",
        help="Combine completed DK/beta result files without refitting models.",
    )
    return parser.parse_args()


def _load_prior_runner():
    spec = importlib.util.spec_from_file_location(
        "v2_raw_rank_runner_for_logged_disagreement",
        PRIOR_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prior runner: {PRIOR_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.VARIANT_FEATURES = VARIANT_FEATURES.copy()
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def _mad(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(numeric) < 2:
        return np.nan
    center = float(np.median(numeric))
    return float(np.median(np.abs(numeric - center)))


def logged_rank_mad(common_position_ranks: Iterable[float]) -> float:
    """Return MAD(log1p(rank)); one observation is unknown, not agreement."""
    values = pd.Series(list(common_position_ranks), dtype=float)
    return _mad(np.log1p(values))


def _strictly_prior_expected_mad(features: pd.DataFrame) -> pd.DataFrame:
    """Estimate expected MAD from prior seasons through fixed fallbacks."""
    output = features.copy()
    output["expert_rank_level_decile"] = np.floor(
        (1 - output["scoring_specific_rank_position_percentile_median"]) * 10
    ).clip(0, 9)
    output["expert_rank_source_count_bucket"] = output[
        "expert_rank_logged_source_count"
    ].clip(upper=4)
    output["expert_rank_logged_mad_expected_prior"] = np.nan
    output["expert_rank_logged_mad_expected_level"] = pd.NA
    output["expert_rank_logged_mad_expected_rows"] = 0

    candidates = output[output["expert_rank_logged_mad"].notna()].copy()
    hierarchies = (
        (
            "position_rank_decile_source_count",
            ["position", "expert_rank_level_decile", "expert_rank_source_count_bucket"],
        ),
        ("position_rank_decile", ["position", "expert_rank_level_decile"]),
        (
            "rank_decile_source_count",
            ["expert_rank_level_decile", "expert_rank_source_count_bucket"],
        ),
        ("position", ["position"]),
        ("global", []),
    )
    for season in sorted(output["season"].dropna().astype(int).unique()):
        target_index = output.index[output["season"].eq(season)]
        history = candidates[candidates["season"].lt(season)]
        if history.empty:
            continue
        for index in target_index:
            row = output.loc[index]
            if pd.isna(row["expert_rank_logged_mad"]):
                continue
            for label, columns in hierarchies:
                selected = history
                for column in columns:
                    if pd.isna(row[column]):
                        selected = selected.iloc[0:0]
                        break
                    selected = selected[selected[column].eq(row[column])]
                if len(selected) >= MIN_EXPECTED_GROUP_ROWS or label == "global":
                    output.at[index, "expert_rank_logged_mad_expected_prior"] = float(
                        selected["expert_rank_logged_mad"].median()
                    )
                    output.at[index, "expert_rank_logged_mad_expected_level"] = label
                    output.at[index, "expert_rank_logged_mad_expected_rows"] = len(selected)
                    break
    output["expert_rank_logged_mad_excess"] = (
        output["expert_rank_logged_mad"]
        - output["expert_rank_logged_mad_expected_prior"]
    )
    return output


def build_logged_rank_features(
    rank_rows: pd.DataFrame,
    feature_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build common-depth logged dispersion and its causal sensitivity."""
    ranks = rank_rows.copy()
    keys = ["source", "season", "position"]
    ranks["source_position_order"] = ranks.groupby(keys, sort=True)[
        "expert_rank"
    ].rank(method="average", ascending=True)
    ranks["source_position_size"] = ranks.groupby(keys, sort=True)[
        "player_key"
    ].transform("count")
    denominator = ranks["source_position_size"].sub(1)
    ranks["source_position_percentile"] = 1 - (
        ranks["source_position_order"].sub(1)
        / denominator.where(denominator.gt(0))
    )
    ranks.loc[
        ranks["source_position_size"].eq(1), "source_position_percentile"
    ] = 0.5
    observed_percentiles = ranks["source_position_percentile"].dropna()
    if not observed_percentiles.between(0, 1).all():
        raise ValueError("Source-position rank percentiles escaped [0, 1]")

    universe = feature_universe[
        ["player_key", "season", "position"]
    ].drop_duplicates()
    common_depth = (
        universe.groupby(["season", "position"], sort=True)["player_key"]
        .nunique()
        .rename("common_position_depth")
        .reset_index()
    )
    ranks = ranks.merge(
        common_depth,
        on=["season", "position"],
        how="left",
        validate="many_to_one",
    )
    if ranks["common_position_depth"].isna().any():
        raise ValueError("Missing common position depth for published ranks")
    ranks["common_position_rank"] = 1 + (
        1 - ranks["source_position_percentile"]
    ) * (ranks["common_position_depth"] - 1)
    ranks["logged_common_position_rank"] = np.log1p(
        ranks["common_position_rank"]
    )

    player_features = (
        ranks.groupby(["player_key", "season", "position"], sort=True)
        .agg(
            scoring_specific_rank_position_percentile_median=(
                "source_position_percentile",
                "median",
            ),
            expert_rank_common_position_rank_median=(
                "common_position_rank",
                "median",
            ),
            expert_rank_logged_mad=("logged_common_position_rank", _mad),
            expert_rank_logged_iqr=("logged_common_position_rank", _iqr),
            expert_rank_logged_range=(
                "logged_common_position_rank",
                lambda value: float(value.max() - value.min()),
            ),
            expert_rank_logged_source_count=("source", "nunique"),
        )
        .reset_index()
    )
    single = player_features["expert_rank_logged_source_count"].lt(2)
    player_features.loc[
        single,
        ["expert_rank_logged_mad", "expert_rank_logged_iqr", "expert_rank_logged_range"],
    ] = np.nan
    player_features = _strictly_prior_expected_mad(player_features)
    return player_features, ranks


def _paired_rows(
    evaluation: pd.DataFrame,
    method: str,
    baseline: str,
    challenger: str,
    score_column: str = "squared_error",
) -> pd.DataFrame:
    selected = evaluation[evaluation["method"].eq(method)]
    base = selected[selected["variant"].eq(baseline)][
        ["player_key", "season", "position", score_column]
    ].rename(columns={score_column: "baseline_score"})
    challenge = selected[selected["variant"].eq(challenger)][
        ["player_key", "season", score_column]
    ].rename(columns={score_column: "challenger_score"})
    compared = base.merge(
        challenge,
        on=["player_key", "season"],
        how="inner",
        validate="one_to_one",
    )
    if len(compared) != len(base) or len(compared) != len(challenge):
        raise ValueError(f"Unmatched comparison rows: {challenger} vs {baseline}")
    return compared


def _mean_cluster_interval(
    compared: pd.DataFrame,
    cluster: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    working = compared.copy()
    working["delta"] = working["challenger_score"] - working["baseline_score"]
    grouped = working.groupby(cluster, sort=True).agg(
        delta_sum=("delta", "sum"), n_rows=("player_key", "size")
    )
    values = grouped[["delta_sum", "n_rows"]].to_numpy(dtype=float)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        draws[index] = sampled[0] / sampled[1]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _point_comparison_summary(
    normalized_runner,
    evaluation: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in (CONTROLLED_METHOD, PRODUCTION_METHOD):
        for challenger, baseline in POINT_COMPARISONS.items():
            compared = _paired_rows(evaluation, method, baseline, challenger)
            base_rmse = float(np.sqrt(compared["baseline_score"].mean()))
            challenger_rmse = float(np.sqrt(compared["challenger_score"].mean()))
            by_season = compared.groupby("season", sort=True).agg(
                baseline_mse=("baseline_score", "mean"),
                challenger_mse=("challenger_score", "mean"),
            )
            by_season["delta"] = np.sqrt(by_season["challenger_mse"]) - np.sqrt(
                by_season["baseline_mse"]
            )
            recent = compared[compared["season"].ge(2023)]
            recent_delta = float(
                np.sqrt(recent["challenger_score"].mean())
                - np.sqrt(recent["baseline_score"].mean())
            )
            interval_input = compared.rename(
                columns={
                    "baseline_score": "incumbent_squared_error",
                    "challenger_score": "variant_squared_error",
                }
            )
            rng = np.random.default_rng(RANDOM_SEED)
            season_low, season_high = normalized_runner._cluster_interval(
                interval_input, "season", iterations, rng
            )
            player_low, player_high = normalized_runner._cluster_interval(
                interval_input, "player_key", iterations, rng
            )
            position_deltas = {}
            for position, group in compared.groupby("position", sort=True):
                position_deltas[str(position)] = float(
                    np.sqrt(group["challenger_score"].mean())
                    - np.sqrt(group["baseline_score"].mean())
                )
            rows.append(
                {
                    "method": method,
                    "baseline_variant": baseline,
                    "challenger_variant": challenger,
                    "baseline_rmse": base_rmse,
                    "challenger_rmse": challenger_rmse,
                    "pooled_delta": challenger_rmse - base_rmse,
                    "recent_delta": recent_delta,
                    "season_wins": int(by_season["delta"].lt(0).sum()),
                    "season_count": int(len(by_season)),
                    "season_95_low": season_low,
                    "season_95_high": season_high,
                    "player_95_low": player_low,
                    "player_95_high": player_high,
                    "nonworse_positions": int(
                        sum(value <= 0 for value in position_deltas.values())
                    ),
                    "position_count": len(position_deltas),
                    "position_deltas_json": json.dumps(position_deltas, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def _scale_design(frame: pd.DataFrame, extra: Sequence[str]) -> pd.DataFrame:
    design = frame[
        [
            "scoring_specific_rank_position_percentile_median",
            "rank_source_coverage",
            "observed_rank_source_count",
            *extra,
        ]
    ].copy()
    level = design["scoring_specific_rank_position_percentile_median"]
    design["expert_rank_level_squared"] = level**2
    for position in ("QB", "RB", "WR", "TE"):
        design[f"position_{position}"] = frame["position"].eq(position).astype(int)
    return design


def _gaussian_crps(actual: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    z = (np.asarray(actual, dtype=float) - np.asarray(mean, dtype=float)) / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def _run_scale_models(
    predictions: pd.DataFrame,
    outer_seasons: Sequence[int],
) -> pd.DataFrame:
    center_column = f"rank_level__{CONTROLLED_METHOD}"
    required = [
        "player_key",
        "season",
        "position",
        "conditional_ppg",
        "conditional_ppg_training_eligible",
        center_column,
        "scoring_specific_rank_position_percentile_median",
        "rank_source_coverage",
        "observed_rank_source_count",
        "expert_rank_logged_mad",
        "expert_rank_logged_mad_excess",
    ]
    data = predictions.loc[:, required].copy()
    eligible = (
        data["season"].isin(outer_seasons)
        & data["conditional_ppg_training_eligible"].eq(1)
        & data["conditional_ppg"].notna()
        & data[center_column].notna()
    )
    data = data[eligible].copy()
    data.rename(
        columns={"conditional_ppg": "actual", center_column: "point_prediction"},
        inplace=True,
    )
    data["absolute_error"] = (data["actual"] - data["point_prediction"]).abs()
    frames: list[pd.DataFrame] = []
    seasons = sorted(data["season"].astype(int).unique())
    for target_season in seasons[1:]:
        train = data[data["season"].lt(target_season)].copy()
        target = data[data["season"].eq(target_season)].copy()
        if train.empty or target.empty:
            continue
        for variant, extra in SCALE_FEATURES.items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=10.0)),
                ]
            )
            train_x = _scale_design(train, extra)
            target_x = _scale_design(target, extra)
            pipeline.fit(train_x, np.log1p(train["absolute_error"]))
            training_mae = np.maximum(np.expm1(pipeline.predict(train_x)), 0.25)
            calibration = float(train["absolute_error"].sum() / training_mae.sum())
            target_mae = np.maximum(np.expm1(pipeline.predict(target_x)), 0.25)
            target_sigma = target_mae * calibration * np.sqrt(np.pi / 2)
            current = target[
                [
                    "player_key",
                    "season",
                    "position",
                    "actual",
                    "point_prediction",
                    "absolute_error",
                    "expert_rank_logged_mad",
                    "expert_rank_logged_mad_excess",
                ]
            ].copy()
            current["variant"] = variant
            current["method"] = "strict_prior_ridge_gaussian"
            current["predicted_mae"] = target_mae * calibration
            current["sigma"] = target_sigma
            current["crps"] = _gaussian_crps(
                current["actual"].to_numpy(),
                current["point_prediction"].to_numpy(),
                target_sigma,
            )
            current["covered_80"] = (
                (current["actual"] - current["point_prediction"]).abs()
                <= GAUSSIAN_80_Z * target_sigma
            ).astype(int)
            current["training_seasons"] = train["season"].nunique()
            current["calibration_multiplier"] = calibration
            frames.append(current)
    if not frames:
        raise ValueError("No strictly-prior scale predictions were produced")
    output = pd.concat(frames, ignore_index=True)
    counts = output.groupby("variant").size()
    if counts.nunique() != 1:
        raise ValueError(f"Scale variants have unequal rows: {counts.to_dict()}")
    return output


def _scale_summary(evaluation: pd.DataFrame, iterations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = "scale_rank_level"
    for challenger in ("scale_logged", "scale_excess"):
        base = evaluation[evaluation["variant"].eq(baseline)][
            ["player_key", "season", "position", "crps", "covered_80"]
        ].rename(columns={"crps": "baseline_score", "covered_80": "baseline_covered"})
        challenge = evaluation[evaluation["variant"].eq(challenger)][
            ["player_key", "season", "crps", "covered_80"]
        ].rename(
            columns={"crps": "challenger_score", "covered_80": "challenger_covered"}
        )
        compared = base.merge(
            challenge,
            on=["player_key", "season"],
            how="inner",
            validate="one_to_one",
        )
        by_season = compared.groupby("season", sort=True).agg(
            baseline=("baseline_score", "mean"),
            challenger=("challenger_score", "mean"),
        )
        by_season["delta"] = by_season["challenger"] - by_season["baseline"]
        recent = compared[compared["season"].ge(2023)]
        rng = np.random.default_rng(RANDOM_SEED)
        season_low, season_high = _mean_cluster_interval(
            compared, "season", iterations, rng
        )
        player_low, player_high = _mean_cluster_interval(
            compared, "player_key", iterations, rng
        )
        baseline_crps = float(compared["baseline_score"].mean())
        challenger_crps = float(compared["challenger_score"].mean())
        baseline_coverage = float(compared["baseline_covered"].mean())
        challenger_coverage = float(compared["challenger_covered"].mean())
        rows.append(
            {
                "baseline_variant": baseline,
                "challenger_variant": challenger,
                "baseline_crps": baseline_crps,
                "challenger_crps": challenger_crps,
                "pooled_delta": challenger_crps - baseline_crps,
                "pooled_relative_delta": challenger_crps / baseline_crps - 1,
                "recent_delta": float(
                    recent["challenger_score"].mean() - recent["baseline_score"].mean()
                ),
                "season_wins": int(by_season["delta"].lt(0).sum()),
                "season_count": int(len(by_season)),
                "season_95_low": season_low,
                "season_95_high": season_high,
                "player_95_low": player_low,
                "player_95_high": player_high,
                "baseline_80_coverage": baseline_coverage,
                "challenger_80_coverage": challenger_coverage,
                "baseline_coverage_error": abs(baseline_coverage - 0.8),
                "challenger_coverage_error": abs(challenger_coverage - 0.8),
            }
        )
    return pd.DataFrame(rows)


def _disagreement_diagnostics(
    point_evaluation: pd.DataFrame,
    logged_features: pd.DataFrame,
) -> pd.DataFrame:
    point = point_evaluation[
        point_evaluation["method"].eq(CONTROLLED_METHOD)
        & point_evaluation["variant"].eq("rank_level")
    ][["player_key", "season", "position", "actual", "prediction"]].copy()
    point["absolute_error"] = (point["actual"] - point["prediction"]).abs()
    point = point.merge(
        logged_features[
            [
                "player_key",
                "season",
                "expert_rank_logged_mad",
                "expert_rank_common_position_rank_median",
                "expert_rank_logged_source_count",
            ]
        ],
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    available = point[point["expert_rank_logged_mad"].notna()].copy()
    available["disagreement_quartile"] = pd.qcut(
        available["expert_rank_logged_mad"].rank(method="first"),
        4,
        labels=("Q1", "Q2", "Q3", "Q4"),
    )
    rows: list[dict[str, object]] = []
    for scope, group in [("all", available), *available.groupby("position", sort=True)]:
        rows.append(
            {
                "scope": str(scope),
                "disagreement_quartile": "all",
                "n_rows": len(group),
                "median_logged_mad": float(group["expert_rank_logged_mad"].median()),
                "mean_absolute_error": float(group["absolute_error"].mean()),
                "spearman_logged_mad_absolute_error": float(
                    group["expert_rank_logged_mad"].corr(
                        group["absolute_error"], method="spearman"
                    )
                ),
            }
        )
    for quartile, group in available.groupby("disagreement_quartile", observed=True):
        rows.append(
            {
                "scope": "all",
                "disagreement_quartile": str(quartile),
                "n_rows": len(group),
                "median_logged_mad": float(group["expert_rank_logged_mad"].median()),
                "mean_absolute_error": float(group["absolute_error"].mean()),
                "spearman_logged_mad_absolute_error": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _gate_audit(point_summary: pd.DataFrame, scale_summary: pd.DataFrame) -> dict[str, object]:
    controlled = point_summary[
        point_summary["method"].eq(CONTROLLED_METHOD)
        & point_summary["challenger_variant"].eq("rank_level_logged")
    ].iloc[0]
    production = point_summary[
        point_summary["method"].eq(PRODUCTION_METHOD)
        & point_summary["challenger_variant"].eq("rank_level_logged")
    ].iloc[0]
    scale = scale_summary[
        scale_summary["challenger_variant"].eq("scale_logged")
    ].iloc[0]
    point_gates = {
        "controlled_pooled_improvement_at_least_0_001": bool(controlled.pooled_delta <= -0.001),
        "controlled_recent_nonworse": bool(controlled.recent_delta <= 0),
        "controlled_at_least_6_season_wins": bool(controlled.season_wins >= 6),
        "controlled_season_interval_upper_nonpositive": bool(controlled.season_95_high <= 0),
        "controlled_player_interval_upper_nonpositive": bool(controlled.player_95_high <= 0),
        "production_pooled_nonworse": bool(production.pooled_delta <= 0),
        "production_recent_nonworse": bool(production.recent_delta <= 0),
        "controlled_at_least_3_positions_nonworse": bool(controlled.nonworse_positions >= 3),
    }
    scale_gates = {
        "pooled_crps_improvement_at_least_0_25_percent": bool(scale.pooled_relative_delta <= -0.0025),
        "recent_crps_nonworse": bool(scale.recent_delta <= 0),
        "at_least_5_season_wins": bool(scale.season_wins >= 5),
        "season_interval_upper_nonpositive": bool(scale.season_95_high <= 0),
        "player_interval_upper_nonpositive": bool(scale.player_95_high <= 0),
        "coverage_error_nonworse": bool(scale.challenger_coverage_error <= scale.baseline_coverage_error),
    }
    return {
        "point_gates": point_gates,
        "scale_gates": scale_gates,
        "point_all_gates_pass": all(point_gates.values()),
        "scale_all_gates_pass": all(scale_gates.values()),
        "point_next_action": (
            "advance_to_nested_retune" if all(point_gates.values()) else "retain_outside_production"
        ),
        "scale_next_action": (
            "advance_to_downstream_residual_template_validation"
            if all(scale_gates.values())
            else "retain_outside_production"
        ),
    }


def _findings_markdown(
    league: str,
    point: pd.DataFrame,
    scale: pd.DataFrame,
    gates: dict[str, object],
    db_hash: str,
    feature_run_id: str,
) -> str:
    lines = [
        f"# Logged Rank Disagreement - {league.upper()}",
        "",
        "## Point-model comparisons",
        "",
        "| Surface | Baseline | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Pos nonworse |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in point.itertuples(index=False):
        lines.append(
            f"| {row.method} | `{row.baseline_variant}` | `{row.challenger_variant}` | "
            f"{row.pooled_delta:+.5f} | {row.recent_delta:+.5f} | "
            f"{row.season_wins}/{row.season_count} | "
            f"[{row.season_95_low:+.5f}, {row.season_95_high:+.5f}] | "
            f"[{row.player_95_low:+.5f}, {row.player_95_high:+.5f}] | "
            f"{row.nonworse_positions}/{row.position_count} |"
        )
    lines.extend(
        [
            "",
            "## Residual-scale comparisons",
            "",
            "| Baseline | Challenger | Delta CRPS | Relative | Recent | Wins | Season 95% | Player 95% | 80% coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale.itertuples(index=False):
        lines.append(
            f"| `{row.baseline_variant}` | `{row.challenger_variant}` | "
            f"{row.pooled_delta:+.5f} | {row.pooled_relative_delta:+.2%} | "
            f"{row.recent_delta:+.5f} | {row.season_wins}/{row.season_count} | "
            f"[{row.season_95_low:+.5f}, {row.season_95_high:+.5f}] | "
            f"[{row.player_95_low:+.5f}, {row.player_95_high:+.5f}] | "
            f"{row.baseline_80_coverage:.1%} -> {row.challenger_80_coverage:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Logged point feature passes every gate: `{gates['point_all_gates_pass']}`.",
            f"- Logged scale feature passes every gate: `{gates['scale_all_gates_pass']}`.",
            f"- Point next action: `{gates['point_next_action']}`.",
            f"- Scale next action: `{gates['scale_next_action']}`.",
            "- The excess-disagreement variant is a sensitivity only.",
            "- No production table, feature contract, or model lock was changed.",
            "",
            "## Lineage",
            "",
            f"- Feature run: `{feature_run_id}`",
            f"- Read-only database SHA-256: `{db_hash}`",
            "",
        ]
    )
    return "\n".join(lines)


def _run_league(
    league: str,
    database: Path,
    results_dir: Path,
    iterations: int,
) -> dict[str, object]:
    if not database.is_file():
        raise FileNotFoundError(database)
    before_hash = _file_sha256(database)
    raw = _load_prior_runner()
    (
        normalized_runner,
        locked,
        features,
        selected,
        locked_predictions,
        raw_features,
        source_coverage,
        depth_audit,
        feature_run_id,
        raw_median_reproduction_delta,
        ppr_resolution,
        input_manifest,
        position_audit,
    ) = raw._load_inputs(database, league)
    observed_hashes = set(features["scoring_hash"].dropna().astype(str))
    expected_hash = scoring_hash(league)
    if observed_hashes != {expected_hash}:
        raise ValueError(f"Scoring mismatch: observed={observed_hashes}, expected={expected_hash}")

    with raw._read_only_connection(database) as connection:
        market_values = pd.read_sql_query(
            "SELECT * FROM player_season_market_values", connection
        )
    rank_rows, _ = raw._scoring_specific_rank_rows(market_values, database, league)
    rank_rows, rebuilt_position_audit = raw._canonicalize_rank_positions(
        rank_rows, features
    )
    logged_features, rank_detail = build_logged_rank_features(rank_rows, features)
    comparator = features[
        ["player_key", "season", "scoring_specific_rank_position_percentile_median"]
    ].merge(
        logged_features[
            ["player_key", "season", "scoring_specific_rank_position_percentile_median"]
        ],
        on=["player_key", "season"],
        how="outer",
        suffixes=("_prior", "_rebuilt"),
        validate="one_to_one",
    )
    missing_mismatch = comparator.iloc[:, -2].isna() ^ comparator.iloc[:, -1].isna()
    if missing_mismatch.any():
        raise ValueError("Rebuilt rank level has different missingness from prior study")
    complete = comparator.dropna()
    rank_level_delta = float(
        (complete.iloc[:, -2] - complete.iloc[:, -1]).abs().max()
    )
    if rank_level_delta > 1e-12:
        raise ValueError(f"Rebuilt rank level mismatch: {rank_level_delta}")

    new_columns = [
        column
        for column in logged_features.columns
        if column not in ("position", "scoring_specific_rank_position_percentile_median")
    ]
    features = features.merge(
        logged_features[new_columns],
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    predictions = raw._run_predictions(locked, features, selected)
    reproduction_delta = raw._assert_incumbent_reproduces_exactly(
        predictions, locked_predictions
    )
    predictions = predictions.merge(
        logged_features[
            [
                "player_key",
                "season",
                "expert_rank_logged_mad",
                "expert_rank_logged_mad_excess",
            ]
        ],
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    point_evaluation = raw._evaluation_long(predictions, locked.OUTER_SEASONS)
    point_scores = raw._score_table(point_evaluation)
    point_summary = _point_comparison_summary(
        normalized_runner, point_evaluation, iterations
    )
    scale_evaluation = _run_scale_models(predictions, locked.OUTER_SEASONS)
    scale_summary = _scale_summary(scale_evaluation, iterations)
    diagnostics = _disagreement_diagnostics(point_evaluation, logged_features)
    gates = _gate_audit(point_summary, scale_summary)

    after_hash = _file_sha256(database)
    if after_hash != before_hash:
        raise RuntimeError("V2 database changed during read-only study execution")
    results_dir.mkdir(parents=True, exist_ok=True)
    logged_features.to_csv(results_dir / "logged_rank_features.csv", index=False)
    rank_detail.to_csv(results_dir / "rank_source_detail.csv", index=False)
    raw_features.to_csv(results_dir / "inherited_raw_rank_features.csv", index=False)
    source_coverage.to_csv(results_dir / "rank_source_coverage.csv", index=False)
    depth_audit.to_csv(results_dir / "rank_source_depths.csv", index=False)
    rebuilt_position_audit.to_csv(results_dir / "rank_position_audit.csv", index=False)
    point_evaluation.to_csv(results_dir / "point_oof_predictions.csv", index=False)
    point_scores.to_csv(results_dir / "point_scores.csv", index=False)
    point_summary.to_csv(results_dir / "point_comparison_summary.csv", index=False)
    scale_evaluation.to_csv(results_dir / "scale_oof_predictions.csv", index=False)
    scale_summary.to_csv(results_dir / "scale_comparison_summary.csv", index=False)
    diagnostics.to_csv(results_dir / "disagreement_diagnostics.csv", index=False)
    manifest = {
        **input_manifest,
        "database_sha256_before": before_hash,
        "database_sha256_after": after_hash,
        "rank_level_reproduction_max_abs_delta": rank_level_delta,
        "incumbent_reproduction_max_abs_delta": reproduction_delta,
        "raw_median_reproduction_max_abs_delta": raw_median_reproduction_delta,
        "logged_feature_spec": {
            "common_depth": "feature_universe_season_position_player_count",
            "transform": "log1p_common_position_rank",
            "dispersion": "median_absolute_deviation",
            "single_source_policy": "missing",
            "expected_min_group_rows": MIN_EXPECTED_GROUP_ROWS,
        },
        "bootstrap_iterations": iterations,
    }
    (results_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "ppr_identity_resolution.json").write_text(
        json.dumps(ppr_resolution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "gate_audit.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "findings.md").write_text(
        _findings_markdown(
            league, point_summary, scale_summary, gates, before_hash, feature_run_id
        ),
        encoding="utf-8",
    )
    return {
        "league": league,
        "database": str(database.resolve()),
        "database_sha256": before_hash,
        "feature_run_id": feature_run_id,
        "point": point_summary.to_dict("records"),
        "scale": scale_summary.to_dict("records"),
        "gates": gates,
    }


def _combine(payloads: Sequence[dict[str, object]], results_dir: Path) -> dict[str, object]:
    by_league = {str(payload["league"]): payload for payload in payloads}
    if set(by_league) != {"dk", "beta"}:
        raise ValueError("Combined decision requires DK and beta")
    point_pass = all(bool(payload["gates"]["point_all_gates_pass"]) for payload in payloads)
    scale_pass = all(bool(payload["gates"]["scale_all_gates_pass"]) for payload in payloads)
    decision = {
        "point_both_leagues_pass": point_pass,
        "scale_both_leagues_pass": scale_pass,
        "point_next_action": "run_nested_retune" if point_pass else "retain_outside_production",
        "scale_next_action": (
            "run_downstream_residual_template_validation"
            if scale_pass
            else "retain_outside_production"
        ),
        "league_results": by_league,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Logged Rank Disagreement Cross-League Decision",
        "",
        f"- Point feature passes both leagues: `{point_pass}`.",
        f"- Scale feature passes both leagues: `{scale_pass}`.",
        f"- Point next action: `{decision['point_next_action']}`.",
        f"- Scale next action: `{decision['scale_next_action']}`.",
        "",
        "## Primary logged-disagreement deltas",
        "",
        "| League | Point controlled RMSE | Point production RMSE | Scale CRPS | Scale relative |",
        "|---|---:|---:|---:|---:|",
    ]
    for league in ("dk", "beta"):
        point = pd.DataFrame(by_league[league]["point"])
        scale = pd.DataFrame(by_league[league]["scale"])
        controlled = point[
            point["method"].eq(CONTROLLED_METHOD)
            & point["challenger_variant"].eq("rank_level_logged")
        ].iloc[0]
        production = point[
            point["method"].eq(PRODUCTION_METHOD)
            & point["challenger_variant"].eq("rank_level_logged")
        ].iloc[0]
        scale_logged = scale[scale["challenger_variant"].eq("scale_logged")].iloc[0]
        lines.append(
            f"| {league.upper()} | {controlled.pooled_delta:+.5f} | "
            f"{production.pooled_delta:+.5f} | {scale_logged.pooled_delta:+.5f} | "
            f"{scale_logged.pooled_relative_delta:+.2%} |"
        )
    lines.extend(
        [
            "",
            "No production feature, model lock, template, or SQLite table was changed.",
            "",
        ]
    )
    (results_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def _load_existing_payload(league: str) -> dict[str, object]:
    results_dir = STUDY_ROOT / "results" / league
    manifest = json.loads(
        (results_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    gates = json.loads(
        (results_dir / "gate_audit.json").read_text(encoding="utf-8")
    )
    return {
        "league": league,
        "database": manifest["staged_database"],
        "database_sha256": manifest["database_sha256_before"],
        "feature_run_id": manifest["locked_lineage"]["feature_run_id"],
        "point": pd.read_csv(
            results_dir / "point_comparison_summary.csv"
        ).to_dict("records"),
        "scale": pd.read_csv(
            results_dir / "scale_comparison_summary.csv"
        ).to_dict("records"),
        "gates": gates,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    if args.league == "all" and args.database is not None:
        raise ValueError("--database cannot be used with --league all")
    if args.league == "all":
        if args.combine_existing:
            payloads = [
                _load_existing_payload(league) for league in ("dk", "beta")
            ]
        else:
            payloads = []
            for league in ("dk", "beta"):
                league_dir = STUDY_ROOT / "results" / league
                payloads.append(
                    _run_league(
                        league,
                        DATABASES[league],
                        league_dir,
                        args.bootstrap_iterations,
                    )
                )
        combined_dir = args.results_dir or STUDY_ROOT / "results"
        decision = _combine(payloads, combined_dir)
        print(json.dumps(decision, indent=2))
        return
    database = args.database or DATABASES[args.league]
    results_dir = args.results_dir or STUDY_ROOT / "results" / args.league
    payload = _run_league(
        args.league, database, results_dir, args.bootstrap_iterations
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
