"""Test V2 next-year trajectory fields in the weekly-template matcher."""

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

PRUNING_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-23_template_feature_pruning"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "next_year_template_pruning_reference",
    PRUNING_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import template replay from {PRUNING_PATH}")
pruning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pruning
SPEC.loader.exec_module(pruning)
base = pruning.base
builder = pruning.builder


DEFAULT_V2_DB = REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3"
DEFAULT_RESULTS = STUDY_DIR / "template_results"
BASELINE_METHOD = "production_no_next"
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}
NEXT_FEATURES = (
    "match_next_residual_rank_pct",
    "match_next_participation_probability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="dk")
    parser.add_argument("--v2-db", type=Path, default=DEFAULT_V2_DB)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    baseline_weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    variants = {
        BASELINE_METHOD: {},
        "next_residual_w025": {
            "match_next_residual_rank_pct": 0.25,
        },
        "next_residual_w050": {
            "match_next_residual_rank_pct": 0.50,
        },
        "next_residual_w100": {
            "match_next_residual_rank_pct": 1.00,
        },
        "next_participation_w025": {
            "match_next_participation_probability": 0.25,
        },
        "next_participation_w050": {
            "match_next_participation_probability": 0.50,
        },
        "next_both_w050": {
            "match_next_residual_rank_pct": 0.50,
            "match_next_participation_probability": 0.50,
        },
    }
    methods: dict[str, dict] = {}
    metadata = []
    for method, additions in variants.items():
        weights = deepcopy(baseline_weights)
        for position in builder.POSITIONS:
            weights[position].update(additions)
        feature_counts = {
            position: len(weights[position])
            for position in builder.POSITIONS
        }
        methods[method] = {
            "weights": weights,
            "recency_half_life": 12.0,
            "variant": method,
            "removed_families": (),
        }
        metadata.append(
            {
                "method": method,
                "next_residual_weight": additions.get(
                    "match_next_residual_rank_pct", 0.0
                ),
                "next_participation_weight": additions.get(
                    "match_next_participation_probability", 0.0
                ),
                "recency_half_life": 12.0,
                "feature_count_total": sum(feature_counts.values()),
                "complexity_score": sum(feature_counts.values()),
                **{
                    f"{position.lower()}_feature_count": count
                    for position, count in feature_counts.items()
                },
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def load_handoff(v2_db: Path, league: str) -> pd.DataFrame:
    with sqlite3.connect(v2_db) as connection:
        handoff = pd.read_sql_query(
            """
            SELECT *
            FROM next_year_template_handoff
            WHERE league=?
            """,
            connection,
            params=(league,),
        )
    if handoff.empty:
        raise ValueError(f"No next-year template handoff for {league}")
    if handoff.duplicated(["player_key", "origin_season"]).any():
        raise ValueError("Next-year handoff has duplicate player-origin rows")
    if (
        handoff["training_through_origin"]
        >= handoff["origin_season"] - 1
    ).any():
        raise ValueError("Next-year handoff violates its outcome embargo")
    return handoff


def attach_next_context(
    frame: pd.DataFrame,
    handoff: pd.DataFrame,
) -> pd.DataFrame:
    if "player_key" not in frame:
        raise ValueError("Template replay rows do not have canonical player_key")
    context = handoff[
        [
            "player_key",
            "origin_season",
            *NEXT_FEATURES,
            "training_through_origin",
            "target_outcome_through",
            "forecast_status",
        ]
    ].rename(columns={"origin_season": "season"})
    output = frame.merge(
        context,
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    output["next_context_available"] = output[
        "match_next_residual_rank_pct"
    ].notna().astype(int)
    for feature in NEXT_FEATURES:
        output[feature] = pd.to_numeric(
            output[feature], errors="coerce"
        ).fillna(builder.MATCH_FILL_VALUE)
    return output


def comparison_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        if method == BASELINE_METHOD:
            continue
        bootstrap = pruning.bootstrap_comparison(
            predictions,
            method,
            PERIODS,
            baseline_method=BASELINE_METHOD,
        )
        rows.append(bootstrap)
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
                    str(value).replace("|", "\\|") for value in row
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
    template_audit: pd.DataFrame,
    period_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    focus_metrics = ("ppg_crps", "contribution_crps", "played_crps")
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
    comparison_focus = comparison[
        comparison["metric"].isin(focus_metrics)
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
    coverage = template_audit[
        [
            "universe",
            "rows",
            "next_context_coverage",
        ]
    ]
    text = f"""# Next-Year Weekly-Template Feature Replay ({league})

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: {target_count:,}.
- Every weekly donor season precedes its target season.
- The next-year fields are themselves causal forecasts with a one-origin
  outcome embargo.
- Baseline is the current production matcher with the 12-season recency prior.
- Production templates and optimizer inputs remain unchanged.

## Context coverage

{markdown_table(coverage)}

## Period results

{markdown_table(focus)}

## Paired candidate-minus-baseline results

{markdown_table(comparison_focus)}

The residual feature is the within-position percentile of predicted
following-season PPG change versus the origin expert projection. The
participation feature is the predicted probability of any following-season
appearance. Both are matching context only; neither creates another residual
draw or directly changes current-season games played.

Runtime: {runtime_seconds:.1f} seconds.
"""
    (results_dir / "template_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    league = str(args.league).lower()
    v2_db = args.v2_db.resolve()
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
    print(
        f"Building {league} weekly templates through {max_season}",
        flush=True,
    )
    projections = builder.load_historical_projection_context(max_season)
    weekly = builder.load_weekly_points(max_season)
    templates = builder.build_weekly_templates(projections, weekly)
    templates = builder.attach_v2_player_keys(
        templates,
        v2_db,
        season_column="season",
    )
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    handoff = load_handoff(v2_db, league)
    templates = attach_next_context(templates, handoff)
    target_templates = attach_next_context(target_templates, handoff)
    targets = base.build_targets(target_templates)

    template_audit = pd.DataFrame(
        [
            {
                "universe": "all_templates",
                "rows": len(templates),
                "next_context_coverage": float(
                    templates["next_context_available"].mean()
                ),
            },
            {
                "universe": "held_out_targets",
                "rows": len(targets),
                "next_context_coverage": float(
                    targets["next_context_available"].mean()
                ),
            },
        ]
    )
    predictions = pruning.run_replay(templates, targets)
    expected = len(targets) * len(METHODS)
    if len(predictions) != expected:
        raise AssertionError(
            f"Expected {expected} prediction rows; found {len(predictions)}"
        )
    period_summary = pruning.period_summaries(predictions)
    position_summary = pruning.grouped_summary(
        predictions[predictions["season"].ge(2020)],
        ["method", "pos"],
    ).merge(
        METHOD_METADATA,
        on="method",
        how="left",
        validate="many_to_one",
    )
    comparison = comparison_table(predictions)
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(
        results_dir / "template_target_predictions.csv", index=False
    )
    METHOD_METADATA.to_csv(
        results_dir / "template_method_metadata.csv", index=False
    )
    template_audit.to_csv(
        results_dir / "template_context_audit.csv", index=False
    )
    period_summary.to_csv(
        results_dir / "template_summary_by_period.csv", index=False
    )
    position_summary.to_csv(
        results_dir / "template_summary_by_position.csv", index=False
    )
    comparison.to_csv(
        results_dir / "template_candidate_bootstrap.csv", index=False
    )
    metadata = {
        "league": league,
        "v2_db": str(v2_db),
        "max_template_season": int(max_season),
        "target_rows": len(targets),
        "prediction_rows": len(predictions),
        "method_count": len(METHODS),
        "baseline_method": BASELINE_METHOD,
        "recency_half_life": 12.0,
        "runtime_seconds": runtime_seconds,
    }
    (results_dir / "template_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_summary(
        results_dir,
        league,
        len(targets),
        template_audit,
        period_summary,
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
