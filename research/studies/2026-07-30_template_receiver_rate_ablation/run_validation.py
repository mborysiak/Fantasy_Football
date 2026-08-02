"""Test projected receiving efficiency in weekly-template matching."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFERENCE_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_template_projection_weight_bump"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "receiver_rate_projection_weight_reference",
    REFERENCE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import template replay from {REFERENCE_PATH}")
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)
pruning = reference.pruning
base = reference.base
builder = reference.builder


DEFAULT_RESULTS = STUDY_DIR / "results"
BASELINE_METHOD = "production"
PRIMARY_METHOD = "both_w050_wrte"
RECENCY_HALF_LIFE = 12.0
RECEPTION_SHRINKAGE = 10.0
BOOTSTRAP_SAMPLES = 2_000
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}
BOOTSTRAP_PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
YPR_FEATURE = "match_proj_rec_ypr_profile"
TD_RATE_FEATURE = "match_proj_rec_td_rate_profile"
RAW_RATE_COLUMNS = [
    "proj_receptions",
    "proj_receiving_yards_per_reception",
    "proj_receiving_td_rate",
]
PROFILE_COLUMNS = [
    *RAW_RATE_COLUMNS,
    "proj_reception_rate_reliability",
    "proj_receiving_ypr_position_pct",
    "proj_receiving_td_rate_position_pct",
    YPR_FEATURE,
    TD_RATE_FEATURE,
    "receiver_rate_source_position",
    "receiver_rate_available",
]
VARIANTS = {
    BASELINE_METHOD: {
        "ypr_weight": 0.0,
        "td_rate_weight": 0.0,
        "positions": (),
        "primary": 0,
    },
    "ypr_w050_wrte": {
        "ypr_weight": 0.50,
        "td_rate_weight": 0.0,
        "positions": ("WR", "TE"),
        "primary": 0,
    },
    "tdrate_w050_wrte": {
        "ypr_weight": 0.0,
        "td_rate_weight": 0.50,
        "positions": ("WR", "TE"),
        "primary": 0,
    },
    "both_w025_wrte": {
        "ypr_weight": 0.25,
        "td_rate_weight": 0.25,
        "positions": ("WR", "TE"),
        "primary": 0,
    },
    PRIMARY_METHOD: {
        "ypr_weight": 0.50,
        "td_rate_weight": 0.50,
        "positions": ("WR", "TE"),
        "primary": 1,
    },
    "both_w100_wrte": {
        "ypr_weight": 1.00,
        "td_rate_weight": 1.00,
        "positions": ("WR", "TE"),
        "primary": 0,
    },
    "both_w050_rbwrte": {
        "ypr_weight": 0.50,
        "td_rate_weight": 0.50,
        "positions": ("RB", "WR", "TE"),
        "primary": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="dk")
    parser.add_argument(
        "--v2-db",
        type=Path,
        default=None,
        help=(
            "League-specific V2 database containing preseason rate features. "
            "Defaults to the configured database for --league."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    methods: dict[str, dict] = {}
    metadata = []
    for method, variant in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        for position in variant["positions"]:
            if variant["ypr_weight"] > 0:
                weights[position][YPR_FEATURE] = variant["ypr_weight"]
            if variant["td_rate_weight"] > 0:
                weights[position][TD_RATE_FEATURE] = variant["td_rate_weight"]
        methods[method] = {
            "weights": weights,
            "recency_half_life": RECENCY_HALF_LIFE,
            "variant": method,
            "removed_families": (),
        }
        metadata.append(
            {
                "method": method,
                "ypr_weight": variant["ypr_weight"],
                "td_rate_weight": variant["td_rate_weight"],
                "positions": ",".join(variant["positions"]),
                "primary": variant["primary"],
                "recency_half_life": RECENCY_HALF_LIFE,
                "reception_shrinkage": RECEPTION_SHRINKAGE,
                "total_match_weight": sum(
                    sum(position_weights.values())
                    for position_weights in weights.values()
                ),
                **{
                    f"{position.lower()}_feature_count": len(weights[position])
                    for position in builder.POSITIONS
                },
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def load_receiver_rate_features(
    v2_database: Path,
    max_season: int,
) -> pd.DataFrame:
    query = """
        SELECT player_key,
               CAST(season AS INTEGER) season,
               position receiver_rate_source_position,
               proj_receptions,
               proj_receiving_yards_per_reception,
               proj_receiving_td_rate
        FROM player_season_features
        WHERE season <= ?
              AND position IN ('RB', 'WR', 'TE')
    """
    with sqlite3.connect(v2_database) as connection:
        rates = pd.read_sql_query(query, connection, params=(int(max_season),))
    duplicate = rates.duplicated(["player_key", "season"], keep=False)
    if duplicate.any():
        preview = rates.loc[
            duplicate,
            ["player_key", "season", "receiver_rate_source_position"],
        ].head(10)
        raise ValueError(
            "V2 receiver rates are not unique by player_key and season:\n"
            f"{preview.to_string(index=False)}"
        )

    for column in RAW_RATE_COLUMNS:
        rates[column] = pd.to_numeric(rates[column], errors="coerce")
    rates["proj_receiving_ypr_position_pct"] = (
        rates.groupby(
            ["season", "receiver_rate_source_position"],
            observed=True,
        )["proj_receiving_yards_per_reception"]
        .rank(method="average", pct=True)
    )
    rates["proj_receiving_td_rate_position_pct"] = (
        rates.groupby(
            ["season", "receiver_rate_source_position"],
            observed=True,
        )["proj_receiving_td_rate"]
        .rank(method="average", pct=True)
    )
    receptions = rates["proj_receptions"].clip(lower=0)
    rates["proj_reception_rate_reliability"] = (
        receptions / (receptions + RECEPTION_SHRINKAGE)
    ).fillna(0)
    rates[YPR_FEATURE] = 0.5 + (
        rates["proj_receiving_ypr_position_pct"] - 0.5
    ) * rates["proj_reception_rate_reliability"]
    rates[TD_RATE_FEATURE] = 0.5 + (
        rates["proj_receiving_td_rate_position_pct"] - 0.5
    ) * rates["proj_reception_rate_reliability"]
    rates["receiver_rate_available"] = (
        rates["proj_receiving_yards_per_reception"].notna()
        & rates["proj_receiving_td_rate"].notna()
    ).astype(int)
    return rates


def attach_receiver_rate_features(
    frame: pd.DataFrame,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    overlapping = sorted(set(PROFILE_COLUMNS).intersection(frame.columns))
    if overlapping:
        raise ValueError(
            "Receiver-rate output columns already exist: "
            + ", ".join(overlapping)
        )
    output = frame.merge(
        rates[["player_key", "season", *PROFILE_COLUMNS]],
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    for column in [YPR_FEATURE, TD_RATE_FEATURE]:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .fillna(builder.MATCH_FILL_VALUE)
            .clip(lower=0, upper=1)
        )
    output["receiver_rate_available"] = (
        pd.to_numeric(output["receiver_rate_available"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return output


def reattach_template_player_keys(
    templates: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Restore the exact projection key dropped by the intermediate builder."""
    if "player_key" in templates.columns:
        return templates
    key_map = projections[
        ["player_key", "player", "pos", "season"]
    ].drop_duplicates()
    if key_map.duplicated(["player", "pos", "season"]).any():
        preview = key_map[
            key_map.duplicated(
                ["player", "pos", "season"],
                keep=False,
            )
        ].head(10)
        raise ValueError(
            "Projection keys are not unique on the weekly-template grain:\n"
            f"{preview.to_string(index=False)}"
        )
    output = templates.merge(
        key_map,
        on=["player", "pos", "season"],
        how="left",
        validate="one_to_one",
    )
    if output["player_key"].isna().any():
        preview = output.loc[
            output["player_key"].isna(),
            ["player", "pos", "season"],
        ].head(10)
        raise ValueError(
            "Weekly templates are missing exact projection keys:\n"
            f"{preview.to_string(index=False)}"
        )
    return output


def coverage_audit(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for population, frame in [
        ("historical_templates", templates),
        ("rolling_targets", targets),
    ]:
        for position, group in frame.groupby("pos", sort=True):
            rows.append(
                {
                    "population": population,
                    "pos": position,
                    "rows": len(group),
                    "rate_available": int(
                        group["receiver_rate_available"].sum()
                    ),
                    "rate_missing": int(
                        group["receiver_rate_available"].eq(0).sum()
                    ),
                    "coverage": float(
                        group["receiver_rate_available"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def attach_target_profile(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    profile = targets[
        [
            "player_key",
            "player",
            "pos",
            "season",
            *PROFILE_COLUMNS,
        ]
    ].rename(columns={"player_key": "target_player_key"})
    return predictions.merge(
        profile,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )


def refresh_row_event_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute row losses after the final target-profile merge."""
    output = predictions.copy()
    pairs = [
        ("prob_plus3", "observed_plus3", "plus3_brier_row"),
        ("prob_plus5", "observed_plus5", "plus5_brier_row"),
        ("prob_impact", "observed_impact", "impact_brier_row"),
        (
            "prob_zero_contribution",
            "observed_zero_contribution",
            "zero_brier_row",
        ),
        (
            "prob_extended_absence",
            "observed_extended_absence",
            "extended_absence_brier_row",
        ),
    ]
    for probability, outcome, loss in pairs:
        output[loss] = np.square(
            pd.to_numeric(output[probability], errors="raise")
            - pd.to_numeric(output[outcome], errors="raise")
        )
    if not np.isfinite(output[[loss for _, _, loss in pairs]]).all().all():
        raise ValueError("Receiver-rate replay contains non-finite event losses.")
    return output


def scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return frame
    if scope == "wr_te":
        return frame[frame["pos"].isin(["WR", "TE"])]
    return frame[frame["pos"].eq(scope.upper())]


def grouped_period_summaries(
    predictions: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    output = []
    scoped = scope_frame(predictions, scope)
    for period, (start, end) in PERIODS.items():
        frame = scoped[scoped.season.between(start, end)]
        summary = pruning.grouped_summary(frame, ["method"])
        summary.insert(0, "scope", scope)
        summary.insert(1, "period", period)
        output.append(summary)
    return pd.concat(output, ignore_index=True).merge(
        METHOD_METADATA,
        on="method",
        how="left",
        validate="many_to_one",
    )


def bootstrap_methods_for_scope(scope: str) -> list[str]:
    candidates = [
        method for method in METHODS if method != BASELINE_METHOD
    ]
    if scope in {"wr_te", "wr", "te"}:
        return [
            method
            for method in candidates
            if method != "both_w050_rbwrte"
        ]
    if scope == "rb":
        return ["both_w050_rbwrte"]
    return candidates


def clustered_bootstrap(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["target_player_key", "pos", "season"]
    metric_columns = pruning.LOWER_IS_BETTER
    output = []
    for scope in ["all", "wr_te", "wr", "te", "rb"]:
        scoped = scope_frame(predictions, scope)
        baseline = scoped[
            scoped.method.eq(BASELINE_METHOD)
        ][key_cols + metric_columns]
        for method in bootstrap_methods_for_scope(scope):
            candidate = scoped[
                scoped.method.eq(method)
            ][key_cols + metric_columns]
            paired = candidate.merge(
                baseline,
                on=key_cols,
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for period, (start, end) in BOOTSTRAP_PERIODS.items():
                frame = paired[paired.season.between(start, end)].copy()
                for cluster_type, cluster_column in [
                    ("season", "season"),
                    ("player", "target_player_key"),
                ]:
                    clusters = frame[cluster_column].drop_duplicates().tolist()
                    cluster_index = {
                        cluster: index
                        for index, cluster in enumerate(clusters)
                    }
                    cluster_ids = (
                        frame[cluster_column].map(cluster_index).to_numpy()
                    )
                    rng = np.random.default_rng(
                        builder.stable_seed(
                            "receiver_rate_bootstrap",
                            scope,
                            method,
                            period,
                            cluster_type,
                        )
                    )
                    sampled_index = rng.integers(
                        0,
                        len(clusters),
                        size=(BOOTSTRAP_SAMPLES, len(clusters)),
                    )
                    cluster_counts = np.bincount(
                        cluster_ids,
                        minlength=len(clusters),
                    ).astype(float)
                    sampled_counts = cluster_counts[sampled_index].sum(axis=1)
                    for metric in metric_columns:
                        delta = (
                            frame[f"{metric}_candidate"]
                            - frame[f"{metric}_baseline"]
                        ).to_numpy(dtype=float)
                        cluster_sums = np.bincount(
                            cluster_ids,
                            weights=delta,
                            minlength=len(clusters),
                        )
                        draws = (
                            cluster_sums[sampled_index].sum(axis=1)
                            / sampled_counts
                        )
                        observed = float(delta.mean())
                        output.append(
                            {
                                "scope": scope,
                                "candidate_method": method,
                                "baseline_method": BASELINE_METHOD,
                                "period": period,
                                "metric": metric,
                                "cluster_type": cluster_type,
                                "n": len(frame),
                                "clusters": len(clusters),
                                "candidate_minus_baseline": observed,
                                "bootstrap_p025": float(
                                    np.quantile(draws, 0.025)
                                ),
                                "bootstrap_p975": float(
                                    np.quantile(draws, 0.975)
                                ),
                                "probability_candidate_better": float(
                                    np.mean(draws < 0)
                                ),
                            }
                        )
    return pd.DataFrame(output)


def pool_profile_audit(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped_donors = {
        (season, pos): group.reset_index(drop=True)
        for (season, pos), group in templates.groupby(["season", "pos"])
    }
    donor_seasons = sorted(templates.season.unique())
    donors_by_origin_pos = {}
    for season in sorted(targets.season.unique()):
        for position in builder.POSITIONS:
            donors = pd.concat(
                [
                    grouped_donors[(donor_season, position)]
                    for donor_season in donor_seasons
                    if donor_season < season
                    and (donor_season, position) in grouped_donors
                ],
                ignore_index=True,
            )
            donors_by_origin_pos[(season, position)] = donors[
                donors.template_eligible.eq(1)
            ].reset_index(drop=True)

    rows = []
    for target in targets.itertuples(index=False):
        eligible = donors_by_origin_pos[(target.season, target.pos)]
        candidate_method = (
            "both_w050_rbwrte"
            if target.pos == "RB"
            else PRIMARY_METHOD
        )
        pools = {
            method: pruning.selected_pool(
                target,
                eligible,
                METHODS[method],
            )
            for method in [BASELINE_METHOD, candidate_method]
        }
        donor_keys = {}
        metrics = {}
        for method, pool in pools.items():
            donors = pool["donors"]
            probabilities = pool["probabilities"]
            donor_keys[method] = set(
                zip(
                    donors["player_key"].astype(str),
                    donors["season"].astype(int),
                )
            )
            metrics[method] = {
                "ypr_profile_distance": float(
                    np.sum(
                        probabilities
                        * np.abs(
                            donors[YPR_FEATURE].to_numpy(dtype=float)
                            - float(getattr(target, YPR_FEATURE))
                        )
                    )
                ),
                "td_rate_profile_distance": float(
                    np.sum(
                        probabilities
                        * np.abs(
                            donors[TD_RATE_FEATURE].to_numpy(dtype=float)
                            - float(getattr(target, TD_RATE_FEATURE))
                        )
                    )
                ),
                "effective_sample_size": float(
                    1 / np.square(probabilities).sum()
                ),
            }
        overlap = len(
            donor_keys[BASELINE_METHOD].intersection(
                donor_keys[candidate_method]
            )
        ) / len(donor_keys[BASELINE_METHOD])
        rows.append(
            {
                "player_key": target.player_key,
                "player": target.player,
                "pos": target.pos,
                "season": int(target.season),
                "candidate_method": candidate_method,
                "pool_overlap_share": overlap,
                "baseline_ypr_profile_distance": metrics[
                    BASELINE_METHOD
                ]["ypr_profile_distance"],
                "primary_ypr_profile_distance": metrics[
                    candidate_method
                ]["ypr_profile_distance"],
                "ypr_profile_distance_delta": (
                    metrics[candidate_method]["ypr_profile_distance"]
                    - metrics[BASELINE_METHOD]["ypr_profile_distance"]
                ),
                "baseline_td_rate_profile_distance": metrics[
                    BASELINE_METHOD
                ]["td_rate_profile_distance"],
                "primary_td_rate_profile_distance": metrics[
                    candidate_method
                ]["td_rate_profile_distance"],
                "td_rate_profile_distance_delta": (
                    metrics[candidate_method]["td_rate_profile_distance"]
                    - metrics[BASELINE_METHOD]["td_rate_profile_distance"]
                ),
                "baseline_effective_sample_size": metrics[
                    BASELINE_METHOD
                ]["effective_sample_size"],
                "primary_effective_sample_size": metrics[
                    candidate_method
                ]["effective_sample_size"],
            }
        )
    audit = pd.DataFrame(rows)
    summary_frames = []
    for scope in ["all", "wr_te", "wr", "te", "rb"]:
        scoped = scope_frame(audit, scope)
        summary_frames.append(
            pd.DataFrame(
                [
                    {
                        "scope": scope,
                        "n": len(scoped),
                        "mean_pool_overlap_share": (
                            scoped.pool_overlap_share.mean()
                        ),
                        "median_pool_overlap_share": (
                            scoped.pool_overlap_share.median()
                        ),
                        "mean_ypr_profile_distance_delta": (
                            scoped.ypr_profile_distance_delta.mean()
                        ),
                        "mean_td_rate_profile_distance_delta": (
                            scoped.td_rate_profile_distance_delta.mean()
                        ),
                        "mean_effective_sample_size_delta": (
                            scoped.primary_effective_sample_size
                            - scoped.baseline_effective_sample_size
                        ).mean(),
                    }
                ]
            )
        )
    return audit, pd.concat(summary_frames, ignore_index=True)


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    return reference.markdown_table(frame, digits=digits)


def write_summary(
    results_dir: Path,
    league: str,
    target_count: int,
    summaries: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pool_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    metric_columns = [
        "scope",
        "period",
        "method",
        "n",
        "ppg_crps",
        "contribution_crps",
        "played_crps",
        "plus3_brier",
        "impact_brier",
        "impact_auc",
        "effective_sample_size",
    ]
    focus = summaries[
        summaries["scope"].isin(["all", "wr_te"])
        & summaries["period"].isin(
            ["all_2017_2025", "temporal_2023_2025"]
        )
        & summaries["method"].isin(
            [
                BASELINE_METHOD,
                "ypr_w050_wrte",
                "tdrate_w050_wrte",
                PRIMARY_METHOD,
                "both_w050_rbwrte",
            ]
        )
    ][metric_columns]
    bootstrap_focus = bootstrap[
        bootstrap["scope"].eq("wr_te")
        & bootstrap["candidate_method"].eq(PRIMARY_METHOD)
        & bootstrap["period"].isin(
            ["all_2017_2025", "temporal_2023_2025"]
        )
        & bootstrap["metric"].isin(
            [
                "ppg_crps",
                "contribution_crps",
                "played_crps",
                "plus3_brier_row",
                "impact_brier_row",
            ]
        )
    ]
    text = f"""# Receiver-Rate Weekly-Template Replay ({league})

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: {target_count:,}.
- Primary comparison: `{PRIMARY_METHOD}` versus `{BASELINE_METHOD}` for WR/TE.
- Rates are preseason V2 projections, not realized outcomes.
- Every donor precedes its target season.
- The production pool size, kernel, recency prior, donor cap, and joint outcome
  transport are unchanged.
- Production code and databases are unchanged.

## Coverage

{markdown_table(coverage)}

## Outcome summary

{markdown_table(focus)}

## Primary WR/TE clustered comparisons

{markdown_table(bootstrap_focus)}

Lower CRPS and Brier scores are better. `candidate_minus_baseline < 0` favors
the receiver-rate matcher.

## Pool-composition audit

{markdown_table(pool_summary)}

Negative profile-distance deltas mean the position-relevant candidate selected
donors closer to the target on that projected rate. The candidate is the primary
WR/TE arm for WR/TE and the RB-extension arm for RB. Pool overlap is the share
of baseline top-80 donors retained.

Runtime: {runtime_seconds:.1f} seconds.
"""
    (results_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    league = str(args.league).lower()
    v2_database = (
        Path(args.v2_db).resolve()
        if args.v2_db is not None
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    builder.set_active_league(league)
    base.builder.LEAGUE = league
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    pruning.PERIODS = PERIODS

    max_season = builder.get_daily_max_template_season()
    rates = load_receiver_rate_features(v2_database, max_season)
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(
        projections,
        weekly,
        league=league,
    )
    templates = reattach_template_player_keys(templates, projections)
    templates = attach_receiver_rate_features(templates, rates)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)

    predictions = pruning.run_replay(templates, targets)
    expected_rows = len(targets) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} predictions; found {len(predictions)}."
        )
    predictions = attach_target_profile(predictions, targets)
    predictions = refresh_row_event_losses(predictions)

    coverage = coverage_audit(templates, targets)
    summaries = pd.concat(
        [
            grouped_period_summaries(predictions, "all"),
            grouped_period_summaries(predictions, "wr_te"),
            grouped_period_summaries(predictions, "wr"),
            grouped_period_summaries(predictions, "te"),
            grouped_period_summaries(predictions, "rb"),
        ],
        ignore_index=True,
    )
    bootstrap = clustered_bootstrap(predictions)
    pool_audit, pool_summary = pool_profile_audit(templates, targets)
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    coverage.to_csv(results_dir / "feature_coverage.csv", index=False)
    summaries.to_csv(results_dir / "summary_by_scope_period.csv", index=False)
    bootstrap.to_csv(results_dir / "clustered_bootstrap.csv", index=False)
    pool_audit.to_csv(results_dir / "pool_profile_audit.csv", index=False)
    pool_summary.to_csv(results_dir / "pool_profile_summary.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_template_season": int(max_season),
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "primary_method": PRIMARY_METHOD,
        "recency_half_life": RECENCY_HALF_LIFE,
        "reception_shrinkage": RECEPTION_SHRINKAGE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "runtime_seconds": runtime_seconds,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_summary(
        results_dir,
        league,
        len(targets),
        summaries,
        bootstrap,
        pool_summary,
        coverage,
        runtime_seconds,
    )
    print(
        summaries[
            summaries.scope.eq("wr_te")
            & summaries.period.eq("all_2017_2025")
        ][
            [
                "method",
                "ppg_crps",
                "contribution_crps",
                "played_crps",
                "plus3_brier",
                "impact_brier",
                "impact_auc",
            ]
        ]
        .round(6)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
