"""Test tighter WR PPG matching jointly with projected receiver-rate profiles."""

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
    "wr_ppg_profile_receiver_rate_reference",
    REFERENCE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import receiver-rate replay from {REFERENCE_PATH}")
receiver_rate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receiver_rate
SPEC.loader.exec_module(receiver_rate)

pruning = receiver_rate.pruning
base = receiver_rate.base
builder = receiver_rate.builder


DEFAULT_RESULTS = STUDY_DIR / "results"
BASELINE_METHOD = "production"
PRIMARY_METHOD = "ppg225_ypr050_wr"
RECENCY_HALF_LIFE = 12.0
PERIODS = receiver_rate.PERIODS
VARIANTS = {
    BASELINE_METHOD: {
        "ppg_weight": 1.50,
        "ypr_weight": 0.00,
        "td_rate_weight": 0.00,
    },
    "ppg225_wr": {
        "ppg_weight": 2.25,
        "ypr_weight": 0.00,
        "td_rate_weight": 0.00,
    },
    "ypr050_wr": {
        "ppg_weight": 1.50,
        "ypr_weight": 0.50,
        "td_rate_weight": 0.00,
    },
    "tdrate050_wr": {
        "ppg_weight": 1.50,
        "ypr_weight": 0.00,
        "td_rate_weight": 0.50,
    },
    "both050_wr": {
        "ppg_weight": 1.50,
        "ypr_weight": 0.50,
        "td_rate_weight": 0.50,
    },
    PRIMARY_METHOD: {
        "ppg_weight": 2.25,
        "ypr_weight": 0.50,
        "td_rate_weight": 0.00,
    },
    "ppg225_tdrate050_wr": {
        "ppg_weight": 2.25,
        "ypr_weight": 0.00,
        "td_rate_weight": 0.50,
    },
    "ppg225_both025_wr": {
        "ppg_weight": 2.25,
        "ypr_weight": 0.25,
        "td_rate_weight": 0.25,
    },
    "ppg225_both050_wr": {
        "ppg_weight": 2.25,
        "ypr_weight": 0.50,
        "td_rate_weight": 0.50,
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
        weights["WR"]["match_projection_ppg_scaled"] = variant["ppg_weight"]
        if variant["ypr_weight"] > 0:
            weights["WR"][receiver_rate.YPR_FEATURE] = variant["ypr_weight"]
        if variant["td_rate_weight"] > 0:
            weights["WR"][receiver_rate.TD_RATE_FEATURE] = variant["td_rate_weight"]
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
    receiver_rate.METHODS = METHODS
    receiver_rate.METHOD_METADATA = METHOD_METADATA
    receiver_rate.BASELINE_METHOD = BASELINE_METHOD
    receiver_rate.PRIMARY_METHOD = PRIMARY_METHOD
    receiver_rate.PERIODS = PERIODS
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    pruning.PERIODS = PERIODS


def current_ladd_audit(
    league: str,
    v2_database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        return pd.DataFrame(), pd.DataFrame()

    rates = receiver_rate.load_receiver_rate_features(
        v2_database,
        int(target["year"].max()),
    )
    templates = receiver_rate.attach_receiver_rate_features(templates, rates)
    target["season"] = target["year"]
    target = receiver_rate.attach_receiver_rate_features(target, rates)
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
            ppg_gap = np.abs(
                pool["historical_pred_fp_per_game"].to_numpy(dtype=float)
                - float(player_row.pred_fp_per_game)
            )
            pryor = pool[pool["player"].eq("Terrelle Pryor")]
            pryor_row = pryor.iloc[0] if not pryor.empty else None
            summaries.append(
                {
                    "league": league,
                    "method": method,
                    "weighted_abs_pred_ppg_gap": float(
                        np.average(ppg_gap, weights=probabilities)
                    ),
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
    return pd.DataFrame(summaries), pd.concat(top_rows, ignore_index=True)


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
    rates = receiver_rate.load_receiver_rate_features(
        v2_database,
        max_season,
    )
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
    templates = receiver_rate.reattach_template_player_keys(
        templates,
        projections,
    )
    templates = receiver_rate.attach_receiver_rate_features(
        templates,
        rates,
    )
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)
    targets = targets[targets["pos"].eq("WR")].reset_index(drop=True)

    predictions = pruning.run_replay(templates, targets)
    predictions = receiver_rate.attach_target_profile(
        predictions,
        targets,
    )
    predictions = receiver_rate.refresh_row_event_losses(predictions)
    summaries = receiver_rate.grouped_period_summaries(predictions, "wr")
    bootstrap = pd.concat(
        [
            pruning.bootstrap_comparison(
                predictions,
                method,
                receiver_rate.BOOTSTRAP_PERIODS,
                baseline_method=BASELINE_METHOD,
            )
            for method in METHODS
            if method != BASELINE_METHOD
        ],
        ignore_index=True,
    )
    ladd_summary, ladd_top12 = current_ladd_audit(league, v2_database)
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    summaries.to_csv(results_dir / "summary_by_period.csv", index=False)
    bootstrap.to_csv(results_dir / "clustered_bootstrap.csv", index=False)
    ladd_summary.to_csv(results_dir / "current_ladd_pool_audit.csv", index=False)
    ladd_top12.to_csv(results_dir / "current_ladd_top12.csv", index=False)
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
