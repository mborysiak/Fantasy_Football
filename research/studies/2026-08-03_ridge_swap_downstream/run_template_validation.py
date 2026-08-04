"""Frozen weekly-template replay for production versus the Ridge point center."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
for root in (REPO_ROOT, STUDY_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase_b = load_module(
    "ridge_swap_phase_b_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_b_replay.py",
)
phase_a = load_module(
    "ridge_swap_phase_a_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_a_rescore.py",
)

receiver_rate = phase_b.receiver_rate
pruning = phase_b.pruning
base = phase_b.base
builder = phase_b.builder
METHODS = ("production", "ridge_swap")
RESID_COLS = [
    "pred_resid_5",
    "pred_resid_10",
    "pred_resid_25",
    "pred_resid_75",
    "pred_resid_90",
    "pred_resid_95",
]
MATCH_SPEC = {
    "weights": builder.MATCH_FEATURE_WEIGHTS,
    "recency_half_life": 12.0,
    "removed_families": (),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--projection-results-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_forecasts(projection_results_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        projection_results_dir / "strict_prior_residuals.csv"
    )
    frame = frame[
        frame.season.between(2018, 2025)
        & frame.resid_calibration_available.eq(1)
        & frame.method.isin(METHODS)
    ].copy()
    frame.rename(columns={"position": "pos"}, inplace=True)
    required = {
        "player_key",
        "season",
        "pos",
        "method",
        "prediction",
        *RESID_COLS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Projection results are missing {sorted(missing)}")
    if frame.duplicated(["player_key", "season", "pos", "method"]).any():
        raise ValueError("Strict-prior forecasts are not unique")
    return frame


def build_target_templates(
    templates: pd.DataFrame,
    forecasts: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    selected = forecasts[forecasts.method.eq(method)].copy()
    selected.rename(
        columns={"prediction": "center_prediction"}, inplace=True
    )
    target_templates = templates[
        templates.season.between(
            int(selected.season.min()), int(selected.season.max())
        )
    ].copy()
    target_templates = target_templates.rename(
        columns={
            "historical_pred_fp_per_game": "builder_historical_pred_fp_per_game",
            "historical_projection_source": "builder_historical_projection_source",
        }
    ).merge(
        selected[
            [
                "player_key",
                "season",
                "pos",
                "center_prediction",
                "resid_calibration_available",
                *RESID_COLS,
            ]
        ],
        on=["player_key", "season", "pos"],
        how="inner",
        validate="one_to_one",
    )
    target_templates["historical_pred_fp_per_game"] = target_templates[
        "center_prediction"
    ]
    target_templates["historical_projection_source"] = (
        f"v2_conditional_ppg_{method}_strict_prior"
    )
    target_templates = builder.add_projection_buckets(
        target_templates,
        value_col="historical_pred_fp_per_game",
        group_cols=["season", "pos"],
    )
    target_templates["match_projection_rank_pct"] = target_templates[
        "projection_rank_pct"
    ]
    target_templates["match_projection_ppg_scaled"] = (
        target_templates.historical_pred_fp_per_game
        .clip(lower=0)
        .div(builder.PROJECTION_PPG_SCALE)
    )
    target_templates["projection_x_exp"] = (
        target_templates.match_projection_rank_pct
        * target_templates.year_exp_scaled
    )
    target_templates["market_projection_gap"] = (
        target_templates.adp_rank_pct
        - target_templates.match_projection_rank_pct
    )
    return target_templates


def freeze_target_cohort(
    target_templates: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    baseline = base.build_targets(target_templates["production"])
    baseline = baseline.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    baseline["preseason_pos_rank"] = (
        baseline.groupby(["season", "pos"]).cumcount() + 1
    )
    cohort = baseline[
        ["player_key", "season", "pos", "preseason_pos_rank"]
    ]
    challenger_templates = target_templates["ridge_swap"].merge(
        cohort,
        on=["player_key", "season", "pos"],
        how="inner",
        validate="one_to_one",
    )
    if len(challenger_templates) != len(baseline):
        raise ValueError(
            "Ridge target surface does not cover the frozen production cohort"
        )
    challenger = base.build_targets(challenger_templates)
    if len(challenger) != len(baseline):
        raise ValueError("Target construction changed the frozen cohort")
    key_order = cohort.assign(_cohort_order=np.arange(len(cohort)))
    challenger = challenger.drop(
        columns=["preseason_pos_rank"], errors="ignore"
    ).merge(
        key_order,
        on=["player_key", "season", "pos"],
        how="inner",
        validate="one_to_one",
    ).sort_values("_cohort_order")
    challenger.drop(columns="_cohort_order", inplace=True)
    challenger.reset_index(drop=True, inplace=True)
    baseline_keys = baseline[["player_key", "season", "pos"]].reset_index(drop=True)
    challenger_keys = challenger[["player_key", "season", "pos"]].reset_index(drop=True)
    if not baseline_keys.equals(challenger_keys):
        raise ValueError("Baseline and Ridge target order differs")
    return {"production": baseline, "ridge_swap": challenger}


def run_method(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    specification = {**MATCH_SPEC, "variant": method}
    pruning.METHODS = {method: specification}
    predictions = pruning.run_replay(templates, targets)
    predictions = receiver_rate.attach_target_profile(predictions, targets)
    predictions = receiver_rate.refresh_row_event_losses(predictions)
    predictions = phase_b.add_target_metadata(predictions, targets)
    if set(predictions.method.unique()) != {method}:
        raise ValueError(f"Unexpected replay method for {method}")
    return predictions


def summarize_role_tiers(
    predictions: pd.DataFrame, league: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = phase_a.Source(
        "2026-08-03_ridge_swap_downstream",
        f"results_template_{league}",
        league,
        "production",
        "current_direct",
    )
    experiment = source.study + "__" + source.result_dir
    frame = predictions.copy()
    frame["study"] = source.study
    frame["result_dir"] = source.result_dir
    frame["league"] = league
    frame["baseline_method"] = source.baseline
    frame["evidence_class"] = source.evidence_class
    frame["experiment"] = experiment
    metrics = phase_a.metric_table(frame, sources=(source,))
    deltas = phase_a.add_baseline_deltas(metrics)
    intervals = phase_a.bootstrap_intervals(frame, sources=(source,))
    return metrics, deltas, intervals


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else STUDY_DIR / f"results_template_{league}"
    )
    projection_results_dir = (
        args.projection_results_dir.resolve()
        if args.projection_results_dir
        else STUDY_DIR / f"results_projection_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phase_b.configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    projections = builder.load_historical_projection_context(
        max_season, v2_database=v2_database
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(
        projections, weekly, league=league
    )
    templates = receiver_rate.reattach_template_player_keys(
        templates, projections
    )
    rates = receiver_rate.load_receiver_rate_features(
        v2_database, max_season
    )
    templates = receiver_rate.attach_receiver_rate_features(templates, rates)
    forecasts = load_forecasts(projection_results_dir)
    target_templates = {
        method: build_target_templates(templates, forecasts, method)
        for method in METHODS
    }
    targets = freeze_target_cohort(target_templates)
    predictions = pd.concat(
        [run_method(templates, targets[method], method) for method in METHODS],
        ignore_index=True,
    )
    metrics, deltas, intervals = summarize_role_tiers(predictions, league)

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    for method in METHODS:
        targets[method].to_csv(
            results_dir / f"target_rows_{method}.csv", index=False
        )
    metrics.to_csv(results_dir / "role_tier_metrics.csv", index=False)
    deltas.to_csv(results_dir / "role_tier_deltas.csv", index=False)
    intervals.to_csv(results_dir / "role_tier_bootstrap.csv", index=False)
    cohort_summary = (
        targets["production"]
        .groupby(["season", "pos"], as_index=False)
        .agg(targets=("player_key", "size"))
    )
    cohort_summary.to_csv(results_dir / "target_cohort.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "projection_results_dir": str(projection_results_dir),
        "max_template_season": int(max_season),
        "target_rows_per_method": int(len(targets["production"])),
        "prediction_rows": int(len(predictions)),
        "methods": list(METHODS),
        "cohort_policy": "production top-N frozen for both methods",
        "matcher_policy": "identical production weights and donor rules",
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        deltas[
            deltas.method.eq("ridge_swap")
            & deltas.tier.isin(["core_main", "depth_main"])
            & deltas.period.isin(
                ["development_2017_2022", "temporal_2023_2025"]
            )
        ][
            [
                "tier",
                "period",
                "n",
                "ppg_crps_relative_delta",
                "contribution_crps_relative_delta",
                "played_crps_relative_delta",
                "ppg_80_coverage_delta",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
