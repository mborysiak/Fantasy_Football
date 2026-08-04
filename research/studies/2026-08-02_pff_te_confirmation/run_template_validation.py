"""Replay prespecified prior-season PFF TE features in template distance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
for root in (REPO_ROOT, STUDY_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pff_te_features import (
    AUDIT_COLUMNS,
    AVAILABLE,
    MTF_MATCH,
    TEMPLATE_FEATURES,
    YAC_MATCH,
    attach_template_features,
    build_te_profiles,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROFILE_REFERENCE = _load_module(
    "pff_te_template_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_fastr_receiver_profiles"
    / "run_validation.py",
)
TAIL_REFERENCE = _load_module(
    "pff_te_tail_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-01_upside_objective_audit"
    / "run_player_tail_replay.py",
)

builder = PROFILE_REFERENCE.builder
base = PROFILE_REFERENCE.base
pruning = PROFILE_REFERENCE.pruning
receiver_rate = PROFILE_REFERENCE.receiver_rate

RAW_DB = REPO_ROOT / "Data" / "Databases" / "Season_Stats_New.sqlite3"
BASELINE_METHOD = "production"
PRIMARY_METHOD = "te_pff_mtf_w025"
RECENCY_HALF_LIFE = 12.0
TARGET_COUNTS = {"QB": 48, "RB": 90, "WR": 120, "TE": 48}
CORE_COUNTS = {"QB": 18, "RB": 36, "WR": 48, "TE": 18}
VARIANTS = {
    BASELINE_METHOD: {},
    PRIMARY_METHOD: {MTF_MATCH: 0.25},
    "te_pff_mtf_w050": {MTF_MATCH: 0.50},
    "te_pff_yac_w025": {YAC_MATCH: 0.25},
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
    for method, additions in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        weights["TE"].update(additions)
        methods[method] = {
            "weights": weights,
            "recency_half_life": RECENCY_HALF_LIFE,
            "variant": method,
            "removed_families": (),
        }
        metadata.append(
            {
                "method": method,
                "primary": int(method == PRIMARY_METHOD),
                "weight_mtf_per_reception": additions.get(MTF_MATCH, 0.0),
                "weight_yac_per_route": additions.get(YAC_MATCH, 0.0),
                "affected_position": "TE" if additions else "none",
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def configure_reference_globals() -> None:
    PROFILE_REFERENCE.METHODS = METHODS
    receiver_rate.METHODS = METHODS
    receiver_rate.METHOD_METADATA = METHOD_METADATA
    receiver_rate.BASELINE_METHOD = BASELINE_METHOD
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    base.TARGET_COUNTS = TARGET_COUNTS


def attach_prediction_metadata(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "player_key",
        "player",
        "pos",
        "season",
        "preseason_pos_rank",
        "qb_team_rank",
        "qb_team_rank_bucket",
        *AUDIT_COLUMNS,
    ]
    if "team" in targets:
        columns.append("team")
    metadata = targets[columns].rename(columns={"player_key": "target_player_key"})
    output = predictions.merge(
        metadata,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )
    if output["target_player_key"].isna().any():
        raise ValueError("Template predictions are missing target player keys")
    return receiver_rate.refresh_row_event_losses(output)


def coverage_audit(templates: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, frame in (("historical_templates", templates), ("rolling_targets", targets)):
        for position, group in frame.groupby("pos", sort=True):
            available = pd.to_numeric(group[AVAILABLE], errors="coerce").fillna(0).eq(1)
            rows.append(
                {
                    "population": population,
                    "position": position,
                    "rows": len(group),
                    "available": int(available.sum()),
                    "coverage": float(available.mean()),
                    "median_routes_when_available": float(group.loc[available, "pff_te_routes"].median()),
                    "median_receptions_when_available": float(group.loc[available, "pff_te_receptions"].median()),
                }
            )
    return pd.DataFrame(rows)


def non_te_parity(predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pool_size", "ppg_mean", "ppg_crps", "contribution_mean",
        "contribution_crps", "played_mean", "played_crps",
        "prob_league_winner_q90", "tail_utility_q90_crps",
    ]
    subset = predictions[~predictions["pos"].eq("TE")]
    baseline = subset[subset["method"].eq(BASELINE_METHOD)][
        ["player", "pos", "season", *metrics]
    ]
    rows = []
    for method in sorted(set(subset["method"]) - {BASELINE_METHOD}):
        challenger = subset[subset["method"].eq(method)][
            ["player", "pos", "season", *metrics]
        ]
        joined = baseline.merge(
            challenger,
            on=["player", "pos", "season"],
            suffixes=("_base", "_challenger"),
            validate="one_to_one",
        )
        for metric in metrics:
            delta = (
                pd.to_numeric(joined[f"{metric}_challenger"], errors="coerce")
                - pd.to_numeric(joined[f"{metric}_base"], errors="coerce")
            ).abs()
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "rows": len(joined),
                    "max_absolute_difference": float(delta.max()),
                }
            )
    output = pd.DataFrame(rows)
    if output["max_absolute_difference"].max() > 1e-12:
        raise AssertionError("A TE-only matcher changed a non-TE prediction")
    return output


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = args.results_dir.resolve() if args.results_dir else STUDY_DIR / f"results_template_{league}"
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    base.evaluate_distribution = TAIL_REFERENCE.evaluate_tail_distribution
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    projections = builder.load_historical_projection_context(max_season, v2_database=v2_database)
    profiles = build_te_profiles(v2_database, RAW_DB, max_season)
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    templates = receiver_rate.reattach_template_player_keys(templates, projections)
    templates = attach_template_features(templates, profiles)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(templates, forecasts)
    targets = base.build_targets(target_templates)
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = targets.groupby(["season", "pos"]).cumcount() + 1
    thresholds = TAIL_REFERENCE.causal_thresholds(templates, targets)
    targets = TAIL_REFERENCE.add_tail_targets(targets, thresholds)

    predictions = pruning.run_replay(templates, targets)
    predictions = attach_prediction_metadata(predictions, targets)
    tail_columns = [
        "player", "pos", "season", "threshold_history_start",
        "threshold_history_end", "threshold_n",
        "league_winner_contribution_q90", "league_winner_contribution_q95",
        "observed_league_winner_q90", "observed_league_winner_q95",
        "observed_tail_utility_q90", "observed_tail_utility_q95",
    ]
    predictions = predictions.merge(
        targets[tail_columns].drop_duplicates(["player", "pos", "season"]),
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )
    predictions["is_core"] = predictions["preseason_pos_rank"].le(
        predictions["pos"].map(CORE_COUNTS)
    )
    for severity in (90, 95):
        predictions[f"league_winner_q{severity}_brier_row"] = np.square(
            predictions[f"prob_league_winner_q{severity}"]
            - predictions[f"observed_league_winner_q{severity}"]
        )

    expected = len(targets) * len(METHODS)
    if len(predictions) != expected:
        raise AssertionError(f"Expected {expected} predictions; found {len(predictions)}")
    coverage = coverage_audit(templates, targets)
    parity = non_te_parity(predictions)
    tail_summary = TAIL_REFERENCE.add_baseline_deltas(
        TAIL_REFERENCE.summarize_predictions(predictions)
    )

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    profiles.to_csv(results_dir / "pff_te_profiles.csv", index=False)
    thresholds.to_csv(results_dir / "causal_thresholds.csv", index=False)
    coverage.to_csv(results_dir / "feature_coverage.csv", index=False)
    parity.to_csv(results_dir / "non_te_parity.csv", index=False)
    tail_summary.to_csv(results_dir / "tail_summary.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_template_season": int(max_season),
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "methods": list(METHODS),
        "primary_method": PRIMARY_METHOD,
        "non_te_parity_max_abs": float(parity["max_absolute_difference"].max()),
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(tail_summary[(tail_summary["scope"].eq("core")) & (tail_summary["severity"].eq("q90"))].to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

