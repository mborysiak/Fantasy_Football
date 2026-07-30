"""Test higher projection-level weights in weekly-template matching."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRUNING_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-23_template_feature_pruning"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "projection_weight_pruning_reference",
    PRUNING_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import template replay from {PRUNING_PATH}")
pruning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pruning
SPEC.loader.exec_module(pruning)
base = pruning.base
builder = pruning.builder


DEFAULT_RESULTS = STUDY_DIR / "results"
BASELINE_METHOD = "production"
RECENCY_HALF_LIFE = 12.0
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}
COMPONENT_RANK_FEATURES = {
    "QB": ("rush_proj_rank_pct", "pass_proj_rank_pct"),
    "RB": ("rush_proj_rank_pct", "rec_proj_rank_pct"),
    "WR": ("rec_proj_rank_pct",),
    "TE": ("rec_proj_rank_pct",),
}
RAW_COMPONENT_FEATURES = {
    "QB": (
        "match_rush_component_ppg_scaled",
        "match_pass_component_ppg_scaled",
    ),
    "RB": (
        "match_rush_component_ppg_scaled",
        "match_rec_component_ppg_scaled",
    ),
    "WR": ("match_rec_component_ppg_scaled",),
    "TE": ("match_rec_component_ppg_scaled",),
}
VARIANTS = {
    BASELINE_METHOD: {
        "ppg_weight": 1.50,
        "component_rank_weight": 1.00,
        "raw_component_weight": 0.00,
    },
    "ppg_w225": {
        "ppg_weight": 2.25,
        "component_rank_weight": 1.00,
        "raw_component_weight": 0.00,
    },
    "ppg_w300": {
        "ppg_weight": 3.00,
        "component_rank_weight": 1.00,
        "raw_component_weight": 0.00,
    },
    "component_rank_w150": {
        "ppg_weight": 1.50,
        "component_rank_weight": 1.50,
        "raw_component_weight": 0.00,
    },
    "raw_component_w100": {
        "ppg_weight": 1.50,
        "component_rank_weight": 1.00,
        "raw_component_weight": 1.00,
    },
    "ppg225_component_rank150": {
        "ppg_weight": 2.25,
        "component_rank_weight": 1.50,
        "raw_component_weight": 0.00,
    },
    "ppg225_component_rank150_raw100": {
        "ppg_weight": 2.25,
        "component_rank_weight": 1.50,
        "raw_component_weight": 1.00,
    },
    "ppg300_component_rank200_raw150": {
        "ppg_weight": 3.00,
        "component_rank_weight": 2.00,
        "raw_component_weight": 1.50,
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
            "League-specific V2 database used for historical centers and "
            "canonical identity. Defaults to the configured database for "
            "--league."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def add_raw_component_ppg_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add scoring-aligned component PPG on the existing /10 distance scale."""
    output = frame.copy()
    total = pd.to_numeric(
        output["avg_proj_points"],
        errors="coerce",
    ).replace(0, np.nan)
    point_center = pd.to_numeric(
        output["historical_pred_fp_per_game"],
        errors="coerce",
    ).clip(lower=0)
    for component in ("rush", "rec", "pass"):
        component_points = pd.to_numeric(
            output[f"avg_proj_{component}_points"],
            errors="coerce",
        ).clip(lower=0)
        share = (
            component_points
            .div(total)
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0, upper=1)
            .fillna(0)
        )
        output[f"match_{component}_component_ppg_scaled"] = (
            share * point_center / builder.PROJECTION_PPG_SCALE
        )
    return output


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    methods: dict[str, dict] = {}
    metadata = []
    for method, variant in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        for position in builder.POSITIONS:
            weights[position]["match_projection_ppg_scaled"] = variant[
                "ppg_weight"
            ]
            for feature in COMPONENT_RANK_FEATURES[position]:
                weights[position][feature] = variant[
                    "component_rank_weight"
                ]
            if variant["raw_component_weight"] > 0:
                for feature in RAW_COMPONENT_FEATURES[position]:
                    weights[position][feature] = variant[
                        "raw_component_weight"
                    ]
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


def comparison_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        if method == BASELINE_METHOD:
            continue
        rows.append(
            pruning.bootstrap_comparison(
                predictions,
                method,
                PERIODS,
                baseline_method=BASELINE_METHOD,
            )
        )
    return pd.concat(rows, ignore_index=True)


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: (
                "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
        )
    columns = list(display.columns)
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *[
                "| "
                + " | ".join(
                    str(value).replace("|", "\\|")
                    for value in row
                )
                + " |"
                for row in display.itertuples(index=False, name=None)
            ],
        ]
    )


def write_summary(
    results_dir: Path,
    league: str,
    target_count: int,
    period_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    focus = period_summary[
        period_summary["period"].isin(
            ("all_2017_2025", "temporal_2023_2025")
        )
    ][
        [
            "period",
            "method",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
            "ppg_bias",
            "played_bias",
        ]
    ]
    position_focus = position_summary[
        position_summary["period"].eq("all_2017_2025")
    ][
        [
            "method",
            "pos",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
        ]
    ]
    comparison_focus = comparison[
        comparison["metric"].isin(
            ("ppg_crps", "contribution_crps", "played_crps")
        )
        & comparison["period"].isin(
            ("all_2017_2025", "temporal_2023_2025")
        )
    ][
        [
            "candidate_method",
            "period",
            "metric",
            "candidate_minus_baseline",
            "bootstrap_p025",
            "bootstrap_p975",
            "probability_candidate_better",
        ]
    ]
    text = f"""# Projection-Weight Weekly-Template Replay ({league})

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: {target_count:,}.
- Every weekly donor precedes its target season.
- All variants retain the production donor eligibility, pool size, kernel,
  12-season recency prior, and 5% donor cap.
- Production configuration and databases are unchanged.

## Period results

{markdown_table(focus)}

## Position results

{markdown_table(position_focus)}

## Paired candidate-minus-production results

{markdown_table(comparison_focus)}

Lower CRPS is better. Raw component magnitudes are scoring-aligned component
PPG estimates on the same `/10` scale as absolute projected PPG.

Runtime: {runtime_seconds:.1f} seconds.
"""
    (results_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    league = str(args.league).lower()
    v2_db = (
        Path(args.v2_db).resolve()
        if args.v2_db is not None
        else builder.resolve_v2_database(league=league)
    )
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    builder.LEAGUE = league
    base.builder.LEAGUE = league
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    pruning.PERIODS = PERIODS

    max_season = builder.get_daily_max_template_season()
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_db,
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(
        projections,
        weekly,
        league=league,
    )
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    templates = add_raw_component_ppg_features(templates)
    target_templates = add_raw_component_ppg_features(target_templates)
    targets = base.build_targets(target_templates)

    predictions = pruning.run_replay(templates, targets)
    expected = len(targets) * len(METHODS)
    if len(predictions) != expected:
        raise AssertionError(
            f"Expected {expected} prediction rows; found {len(predictions)}"
        )

    period_summary = pruning.period_summaries(predictions)
    position_frames = []
    for period, (start, end) in PERIODS.items():
        summary = pruning.grouped_summary(
            predictions[predictions["season"].between(start, end)],
            ["method", "pos"],
        )
        summary.insert(0, "period", period)
        position_frames.append(summary)
    position_summary = pd.concat(position_frames, ignore_index=True)
    comparison = comparison_table(predictions)
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(
        results_dir / "method_metadata.csv",
        index=False,
    )
    period_summary.to_csv(
        results_dir / "summary_by_period.csv",
        index=False,
    )
    position_summary.to_csv(
        results_dir / "summary_by_position.csv",
        index=False,
    )
    comparison.to_csv(
        results_dir / "candidate_bootstrap.csv",
        index=False,
    )
    metadata = {
        "league": league,
        "max_template_season": int(max_season),
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "recency_half_life": RECENCY_HALF_LIFE,
        "runtime_seconds": runtime_seconds,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_summary(
        results_dir,
        league,
        len(targets),
        period_summary,
        position_summary,
        comparison,
        runtime_seconds,
    )
    print(
        period_summary[
            period_summary["period"].eq("all_2017_2025")
        ][
            [
                "method",
                "ppg_crps",
                "contribution_crps",
                "played_crps",
            ]
        ]
        .round(5)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
