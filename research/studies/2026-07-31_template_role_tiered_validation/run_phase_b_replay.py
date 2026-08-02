"""Fresh corrected-lineage replay of frozen role-tiered template finalists."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
REFERENCE_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-30_wr_template_ppg_profile_tradeoff"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "role_tiered_wr_profile_reference",
    REFERENCE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import replay reference from {REFERENCE_PATH}")
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

receiver_rate = reference.receiver_rate
pruning = reference.pruning
base = reference.base
builder = reference.builder

BASELINE_METHOD = "production"
RECENCY_HALF_LIFE = 12.0
EXPANDED_TARGET_COUNTS = {"QB": 48, "RB": 90, "WR": 120, "TE": 48}
VARIANTS = {
    BASELINE_METHOD: {
        "all_weight_multiplier": 1.0,
        "wr_ppg_weight": 1.50,
        "wr_ypr_weight": 0.00,
        "wr_td_rate_weight": 0.00,
        "te_ypr_weight": 0.00,
    },
    "flatter_w025_all": {
        "all_weight_multiplier": 0.25,
        "wr_ppg_weight": 1.50,
        "wr_ypr_weight": 0.00,
        "wr_td_rate_weight": 0.00,
        "te_ypr_weight": 0.00,
    },
    "wr_ppg225_tdrate050": {
        "all_weight_multiplier": 1.0,
        "wr_ppg_weight": 2.25,
        "wr_ypr_weight": 0.00,
        "wr_td_rate_weight": 0.50,
        "te_ypr_weight": 0.00,
    },
    "wr_ppg225_both025": {
        "all_weight_multiplier": 1.0,
        "wr_ppg_weight": 2.25,
        "wr_ypr_weight": 0.25,
        "wr_td_rate_weight": 0.25,
        "te_ypr_weight": 0.00,
    },
    "te_ypr050": {
        "all_weight_multiplier": 1.0,
        "wr_ppg_weight": 1.50,
        "wr_ypr_weight": 0.00,
        "wr_td_rate_weight": 0.00,
        "te_ypr_weight": 0.50,
    },
    "wr_ppg225_tdrate050__te_ypr050": {
        "all_weight_multiplier": 1.0,
        "wr_ppg_weight": 2.25,
        "wr_ypr_weight": 0.00,
        "wr_td_rate_weight": 0.50,
        "te_ypr_weight": 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    methods = {}
    metadata = []
    for method, variant in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        multiplier = float(variant["all_weight_multiplier"])
        if multiplier != 1.0:
            for position in builder.POSITIONS:
                for feature in weights[position]:
                    weights[position][feature] *= multiplier
        weights["WR"]["match_projection_ppg_scaled"] = float(
            variant["wr_ppg_weight"]
        ) * multiplier
        if variant["wr_ypr_weight"] > 0:
            weights["WR"][receiver_rate.YPR_FEATURE] = float(
                variant["wr_ypr_weight"]
            )
        if variant["wr_td_rate_weight"] > 0:
            weights["WR"][receiver_rate.TD_RATE_FEATURE] = float(
                variant["wr_td_rate_weight"]
            )
        if variant["te_ypr_weight"] > 0:
            weights["TE"][receiver_rate.YPR_FEATURE] = float(
                variant["te_ypr_weight"]
            )
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
                "recency_half_life": RECENCY_HALF_LIFE,
                "data_note": (
                    "receiver-rate fields are preseason projection profiles; "
                    "no new historical performance source is introduced"
                ),
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def configure_reference_globals() -> None:
    reference.METHODS = METHODS
    receiver_rate.METHODS = METHODS
    receiver_rate.METHOD_METADATA = METHOD_METADATA
    receiver_rate.BASELINE_METHOD = BASELINE_METHOD
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    base.TARGET_COUNTS = EXPANDED_TARGET_COUNTS


def add_target_metadata(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "player",
        "pos",
        "season",
        "preseason_pos_rank",
        "qb_team_rank",
        "qb_team_rank_bucket",
    ]
    if "team" in targets.columns:
        columns.append("team")
    metadata = targets[columns].drop_duplicates(["player", "pos", "season"])
    output = predictions.merge(
        metadata,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )
    if output.preseason_pos_rank.isna().any():
        raise ValueError("Fresh predictions are missing preseason position rank.")
    return output


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else STUDY_DIR / f"results_phase_b_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db is not None
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    rates = receiver_rate.load_receiver_rate_features(v2_database, max_season)
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
    templates = receiver_rate.attach_receiver_rate_features(templates, rates)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = (
        targets.groupby(["season", "pos"]).cumcount() + 1
    )

    predictions = pruning.run_replay(templates, targets)
    predictions = receiver_rate.attach_target_profile(predictions, targets)
    predictions = receiver_rate.refresh_row_event_losses(predictions)
    predictions = add_target_metadata(predictions, targets)
    coverage = receiver_rate.coverage_audit(templates, targets)
    cohort = (
        targets.groupby(["season", "pos"], as_index=False)
        .agg(
            targets=("player", "size"),
            team_qb1=("qb_team_rank", lambda values: int(values.eq(1).sum())),
        )
    )
    summaries = pd.concat(
        [
            receiver_rate.grouped_period_summaries(predictions, scope)
            for scope in ("all", "wr", "te", "rb", "qb")
        ],
        ignore_index=True,
    )
    ladd_summary, ladd_top12 = reference.current_ladd_audit(
        league,
        v2_database,
    )

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    summaries.to_csv(results_dir / "summary_by_period.csv", index=False)
    coverage.to_csv(results_dir / "receiver_rate_coverage.csv", index=False)
    cohort.to_csv(results_dir / "target_cohort.csv", index=False)
    ladd_summary.to_csv(results_dir / "current_ladd_pool_audit.csv", index=False)
    ladd_top12.to_csv(results_dir / "current_ladd_top12.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_template_season": int(max_season),
        "expanded_target_counts": EXPANDED_TARGET_COUNTS,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
