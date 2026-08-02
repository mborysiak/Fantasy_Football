"""Test preseason projection trajectory in WR weekly-template matching."""

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
REFERENCE_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-30_template_receiver_rate_ablation"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "template_projection_trajectory_reference",
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
PRIMARY_METHOD = "full_w025_wr"
RECENCY_HALF_LIFE = 12.0
PERIODS = reference.PERIODS
ONE_YEAR_FEATURE = "match_projection_trajectory_1year_pct"
THREE_YEAR_FEATURE = "match_projection_trajectory_3year_pct"
PRIOR_AVAILABLE_FEATURE = "match_projection_trajectory_prior_year_available"
HISTORY_DEPTH_FEATURE = "match_projection_trajectory_history_depth"
RAW_TRAJECTORY_COLUMNS = [
    "projection_trajectory_change_1year",
    "projection_trajectory_change_3year",
    "projection_trajectory_prior_year_available",
    "projection_trajectory_prior_3year_count",
    "projection_trajectory_prior_3year_std",
]
PROFILE_COLUMNS = [
    *RAW_TRAJECTORY_COLUMNS,
    ONE_YEAR_FEATURE,
    THREE_YEAR_FEATURE,
    PRIOR_AVAILABLE_FEATURE,
    HISTORY_DEPTH_FEATURE,
]
VARIANTS = {
    BASELINE_METHOD: {
        "one_year_weight": 0.00,
        "three_year_weight": 0.00,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    "one_year_w025_wr": {
        "one_year_weight": 0.25,
        "three_year_weight": 0.00,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    "one_year_w050_wr": {
        "one_year_weight": 0.50,
        "three_year_weight": 0.00,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    "three_year_w025_wr": {
        "one_year_weight": 0.00,
        "three_year_weight": 0.25,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    "three_year_w050_wr": {
        "one_year_weight": 0.00,
        "three_year_weight": 0.50,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    "both_w025_wr": {
        "one_year_weight": 0.25,
        "three_year_weight": 0.25,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    "both_w050_wr": {
        "one_year_weight": 0.50,
        "three_year_weight": 0.50,
        "prior_available_weight": 0.00,
        "history_depth_weight": 0.00,
    },
    PRIMARY_METHOD: {
        "one_year_weight": 0.25,
        "three_year_weight": 0.25,
        "prior_available_weight": 0.25,
        "history_depth_weight": 0.25,
    },
    "full_w050_wr": {
        "one_year_weight": 0.50,
        "three_year_weight": 0.50,
        "prior_available_weight": 0.25,
        "history_depth_weight": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="dk")
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    methods = {}
    metadata = []
    for method, variant in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        additions = {
            ONE_YEAR_FEATURE: variant["one_year_weight"],
            THREE_YEAR_FEATURE: variant["three_year_weight"],
            PRIOR_AVAILABLE_FEATURE: variant["prior_available_weight"],
            HISTORY_DEPTH_FEATURE: variant["history_depth_weight"],
        }
        for feature, weight in additions.items():
            if weight > 0:
                weights["WR"][feature] = weight
        methods[method] = {
            "weights": weights,
            "recency_half_life": RECENCY_HALF_LIFE,
            "variant": method,
            "removed_families": (),
        }
        metadata.append(
            {
                "method": method,
                **variant,
                "primary": int(method == PRIMARY_METHOD),
                "recency_half_life": RECENCY_HALF_LIFE,
                "wr_feature_count": len(weights["WR"]),
                "wr_total_match_weight": sum(weights["WR"].values()),
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def configure_reference_globals() -> None:
    reference.METHODS = METHODS
    reference.METHOD_METADATA = METHOD_METADATA
    reference.BASELINE_METHOD = BASELINE_METHOD
    reference.PRIMARY_METHOD = PRIMARY_METHOD
    reference.PERIODS = PERIODS
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    pruning.PERIODS = PERIODS


def load_trajectory_features(
    v2_database: Path,
    max_season: int,
) -> pd.DataFrame:
    query = """
        SELECT player_key,
               CAST(season AS INTEGER) season,
               position trajectory_source_position,
               projection_trajectory_change_1year,
               projection_trajectory_change_3year,
               projection_trajectory_prior_year_available,
               projection_trajectory_prior_3year_count,
               projection_trajectory_prior_3year_std
        FROM player_season_features
        WHERE season <= ?
              AND position IN ('QB', 'RB', 'WR', 'TE')
    """
    with sqlite3.connect(v2_database) as connection:
        trajectory = pd.read_sql_query(
            query,
            connection,
            params=(int(max_season),),
        )
    duplicate = trajectory.duplicated(["player_key", "season"], keep=False)
    if duplicate.any():
        raise ValueError("Trajectory features are not unique by player-season.")
    for column in RAW_TRAJECTORY_COLUMNS:
        trajectory[column] = pd.to_numeric(
            trajectory[column],
            errors="coerce",
        )
    prior_available = trajectory[
        "projection_trajectory_prior_year_available"
    ].eq(1)
    prior_history = trajectory[
        "projection_trajectory_prior_3year_count"
    ].gt(0)
    trajectory[ONE_YEAR_FEATURE] = np.nan
    trajectory.loc[prior_available, ONE_YEAR_FEATURE] = (
        trajectory.loc[prior_available]
        .groupby(
            ["season", "trajectory_source_position"],
            observed=True,
        )["projection_trajectory_change_1year"]
        .rank(method="average", pct=True)
    )
    trajectory[THREE_YEAR_FEATURE] = np.nan
    trajectory.loc[prior_history, THREE_YEAR_FEATURE] = (
        trajectory.loc[prior_history]
        .groupby(
            ["season", "trajectory_source_position"],
            observed=True,
        )["projection_trajectory_change_3year"]
        .rank(method="average", pct=True)
    )
    # A zero raw gap is the neutral trajectory. No-history players, including
    # rookies, receive that neutral profile while explicit history fields keep
    # missing history distinguishable from an observed stable veteran.
    trajectory[ONE_YEAR_FEATURE] = trajectory[ONE_YEAR_FEATURE].fillna(0.5)
    trajectory[THREE_YEAR_FEATURE] = trajectory[THREE_YEAR_FEATURE].fillna(0.5)
    trajectory[PRIOR_AVAILABLE_FEATURE] = (
        trajectory["projection_trajectory_prior_year_available"]
        .fillna(0)
        .clip(lower=0, upper=1)
    )
    trajectory[HISTORY_DEPTH_FEATURE] = (
        trajectory["projection_trajectory_prior_3year_count"]
        .fillna(0)
        .clip(lower=0, upper=3)
        .div(3.0)
    )
    return trajectory


def attach_trajectory_features(
    frame: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    overlapping = sorted(set(PROFILE_COLUMNS).intersection(frame.columns))
    if overlapping:
        raise ValueError(
            "Trajectory output columns already exist: "
            + ", ".join(overlapping)
        )
    output = frame.merge(
        trajectory[["player_key", "season", *PROFILE_COLUMNS]],
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    for column in [ONE_YEAR_FEATURE, THREE_YEAR_FEATURE]:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .fillna(0.5)
            .clip(lower=0, upper=1)
        )
    for column in [PRIOR_AVAILABLE_FEATURE, HISTORY_DEPTH_FEATURE]:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
        )
    return output


def attach_target_profile(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    profile = targets[
        ["player_key", "player", "pos", "season", *PROFILE_COLUMNS]
    ].rename(columns={"player_key": "target_player_key"})
    return predictions.merge(
        profile,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )


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
                    "prior_year_available": int(
                        group[PRIOR_AVAILABLE_FEATURE].eq(1).sum()
                    ),
                    "prior_3year_available": int(
                        group[HISTORY_DEPTH_FEATURE].gt(0).sum()
                    ),
                    "prior_year_coverage": float(
                        group[PRIOR_AVAILABLE_FEATURE].eq(1).mean()
                    ),
                    "prior_3year_coverage": float(
                        group[HISTORY_DEPTH_FEATURE].gt(0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def current_ladd_audit(
    league: str,
    v2_database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simulation = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
    with sqlite3.connect(simulation) as connection:
        templates = pd.read_sql_query(
            "SELECT * FROM Best_Ball_Weekly_Templates WHERE league = ?",
            connection,
            params=(league,),
        )
        target = pd.read_sql_query(
            """
            SELECT *
            FROM Best_Ball_Weekly_Player_Map
            WHERE version = ? AND player = 'Ladd McConkey'
            """,
            connection,
            params=(league,),
        )
    if templates.empty or target.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trajectory = load_trajectory_features(
        v2_database,
        int(target["year"].max()),
    )
    templates = attach_trajectory_features(templates, trajectory)
    target["season"] = target["year"]
    target = attach_trajectory_features(target, trajectory)
    player_row = next(target.itertuples(index=False))
    summaries = []
    top_rows = []
    original_weights = builder.MATCH_FEATURE_WEIGHTS
    try:
        for method, specification in METHODS.items():
            builder.MATCH_FEATURE_WEIGHTS = specification["weights"]
            members, _ = builder.select_player_template_pool(
                player_row,
                templates,
            )
            pool = members.merge(
                templates[
                    [
                        "league",
                        "template_id",
                        "player",
                        "season",
                        "historical_pred_fp_per_game",
                        "active_ppg_resid",
                        "played_games",
                    ]
                ],
                left_on=["template_league", "template_id"],
                right_on=["league", "template_id"],
                how="left",
                validate="one_to_one",
                suffixes=("", "_template"),
            )
            probabilities = pool["template_sample_prob"].to_numpy(dtype=float)
            residuals = pool["active_ppg_resid"].to_numpy(dtype=float)
            residual_mean = float(np.average(residuals, weights=probabilities))
            pryor = pool[pool["player"].eq("Terrelle Pryor")]
            pryor_row = pryor.iloc[0] if not pryor.empty else None
            summaries.append(
                {
                    "league": league,
                    "method": method,
                    "expected_played": float(
                        np.average(
                            pool["played_games"].to_numpy(dtype=float),
                            weights=probabilities,
                        )
                    ),
                    "pool_residual_sd": float(
                        np.sqrt(
                            np.average(
                                np.square(residuals - residual_mean),
                                weights=probabilities,
                            )
                        )
                    ),
                    "top12_weight": float(
                        pool.nsmallest(12, "match_rank")[
                            "template_sample_prob"
                        ].sum()
                    ),
                    "pryor_rank": (
                        int(pryor_row["match_rank"])
                        if pryor_row is not None
                        else np.nan
                    ),
                    "pryor_distance": (
                        float(pryor_row["template_distance"])
                        if pryor_row is not None
                        else np.nan
                    ),
                    "pryor_weight": (
                        float(pryor_row["template_sample_prob"])
                        if pryor_row is not None
                        else 0.0
                    ),
                }
            )
            top = pool.nsmallest(12, "match_rank")[
                [
                    "match_rank",
                    "player",
                    "season_template",
                    "template_distance",
                    "template_sample_prob",
                    "historical_pred_fp_per_game",
                    "played_games",
                ]
            ].copy()
            top.insert(0, "method", method)
            top.insert(0, "league", league)
            top_rows.append(top)
    finally:
        builder.MATCH_FEATURE_WEIGHTS = original_weights

    pryor_profile = templates[
        templates["player"].eq("Terrelle Pryor")
        & templates["season"].eq(2017)
    ][["player", "season", *PROFILE_COLUMNS]]
    target_profile = target[["player", "season", *PROFILE_COLUMNS]]
    profiles = pd.concat(
        [target_profile, pryor_profile],
        ignore_index=True,
    )
    profiles.insert(0, "league", league)
    return (
        pd.DataFrame(summaries),
        pd.concat(top_rows, ignore_index=True),
        profiles,
    )


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

    configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    max_season = builder.get_daily_max_template_season()
    trajectory = load_trajectory_features(v2_database, max_season)
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
    templates = reference.reattach_template_player_keys(
        templates,
        projections,
    )
    templates = attach_trajectory_features(templates, trajectory)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)
    targets = targets[targets["pos"].eq("WR")].reset_index(drop=True)

    predictions = pruning.run_replay(templates, targets)
    expected_rows = len(targets) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} predictions; found {len(predictions)}."
        )
    predictions = attach_target_profile(predictions, targets)
    predictions = reference.refresh_row_event_losses(predictions)
    summaries = reference.grouped_period_summaries(predictions, "wr")
    bootstrap = pd.concat(
        [
            pruning.bootstrap_comparison(
                predictions,
                method,
                reference.BOOTSTRAP_PERIODS,
                baseline_method=BASELINE_METHOD,
            )
            for method in METHODS
            if method != BASELINE_METHOD
        ],
        ignore_index=True,
    )
    coverage = coverage_audit(templates, targets)
    ladd_summary, ladd_top12, ladd_profiles = current_ladd_audit(
        league,
        v2_database,
    )
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    coverage.to_csv(results_dir / "feature_coverage.csv", index=False)
    summaries.to_csv(results_dir / "summary_by_period.csv", index=False)
    bootstrap.to_csv(results_dir / "clustered_bootstrap.csv", index=False)
    ladd_summary.to_csv(results_dir / "current_ladd_pool_audit.csv", index=False)
    ladd_top12.to_csv(results_dir / "current_ladd_top12.csv", index=False)
    ladd_profiles.to_csv(
        results_dir / "current_ladd_pryor_profiles.csv",
        index=False,
    )
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_template_season": int(max_season),
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "primary_method": PRIMARY_METHOD,
        "runtime_seconds": runtime_seconds,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        summaries[
            summaries["period"].isin(
                ["all_2017_2025", "temporal_2023_2025"]
            )
        ][
            [
                "period",
                "method",
                "ppg_crps",
                "contribution_crps",
                "played_crps",
                "impact_brier",
                "impact_auc",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
