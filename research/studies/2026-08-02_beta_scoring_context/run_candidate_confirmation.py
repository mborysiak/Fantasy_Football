"""Confirm the beta-scored absolute-PPG matcher on strict rolling targets."""

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
PLAYER_RUNNER_PATH = STUDY_DIR / "run_validation.py"
BASELINE_RESULTS = STUDY_DIR / "results" / "target_predictions.csv"
BASELINE = "production_hybrid"
CANDIDATE = "beta_scored_ppg_rank_w050"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("beta_context_confirmation_runner", PLAYER_RUNNER_PATH)
builder = runner.builder
base = runner.base
phase_a = runner.phase_a
pruning = runner.pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-db", type=Path, required=True)
    parser.add_argument("--simulation-db", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=STUDY_DIR / "results_confirmation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    v2_database = args.v2_db.resolve()
    simulation_database = args.simulation_db.resolve()
    started = time.perf_counter()

    runner.phase_b.configure_reference_globals()
    builder.set_simulation_db(simulation_database)
    builder.set_active_league("beta")
    base.builder.LEAGUE = "beta"
    max_season = builder.get_daily_max_template_season()
    weekly = builder.load_weekly_points(max_season, league="beta")
    forecasts = base.load_production_oos_forecasts(max_season)
    baseline_projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
        scoring_matched_context=False,
        scoring_matched_fallback_center=False,
    )
    templates, template_audit = runner.build_arm_templates(
        CANDIDATE,
        runner.DECOUPLED_OPTIONS,
        v2_database=v2_database,
        baseline_projections=baseline_projections,
        weekly=weekly,
    )
    targets = runner.build_targets(templates, forecasts)

    specification = deepcopy(runner.phase_b.METHODS["production"])
    for position_weights in specification["weights"].values():
        position_weights["match_projection_rank_pct"] = 0.5
        position_weights["market_projection_gap"] = 0.0
    pruning.METHODS = {CANDIDATE: specification}
    candidate = pruning.run_replay(templates, targets)
    candidate = runner.phase_b.add_target_metadata(candidate, targets)

    baseline = pd.read_csv(BASELINE_RESULTS)
    baseline = baseline[baseline.method.eq(BASELINE)].copy()
    keys = ["player", "pos", "season"]
    baseline_keys = baseline[keys].sort_values(keys).reset_index(drop=True)
    candidate_keys = candidate[keys].sort_values(keys).reset_index(drop=True)
    if not baseline_keys.equals(candidate_keys):
        raise ValueError("Confirmation target cohort changed from baseline.")
    paired = baseline.merge(
        candidate,
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    for column in (
        "predicted_ppg",
        "observed_ppg",
        "observed_contribution",
        "observed_played",
        "observed_active",
    ):
        if not np.allclose(
            paired[f"{column}_baseline"],
            paired[f"{column}_candidate"],
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"Confirmation target changed: {column}")
    predictions = pd.concat([baseline, candidate], ignore_index=True)
    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    template_audit.to_csv(
        results_dir / "template_policy_audit.csv", index=False
    )

    source = phase_a.Source(
        STUDY_DIR.name,
        results_dir.name,
        "beta",
        BASELINE,
        "fresh_confirmation",
    )
    scored = phase_a.read_source(source)
    sources = (source,)
    metrics = phase_a.metric_table(scored, sources)
    deltas = phase_a.add_baseline_deltas(metrics)
    intervals = phase_a.bootstrap_intervals(scored, sources)
    position_guardrails = phase_a.position_ppg_guardrails(scored, sources)
    decisions = runner.decision_table(deltas, position_guardrails)
    metrics.to_csv(results_dir / "role_tier_metrics.csv", index=False)
    deltas.to_csv(results_dir / "candidate_deltas.csv", index=False)
    intervals.to_csv(
        results_dir / "season_cluster_bootstrap.csv", index=False
    )
    position_guardrails.to_csv(
        results_dir / "position_ppg_guardrails.csv", index=False
    )
    decisions.to_csv(results_dir / "decision_gates.csv", index=False)

    passed = bool(decisions.player_level_pass.iloc[0])
    findings = [
        "# Candidate confirmation findings",
        "",
        runner.markdown_table(decisions),
        "",
        (
            f"`{CANDIDATE}` passes strict player-level confirmation."
            if passed
            else f"`{CANDIDATE}` fails strict player-level confirmation."
        ),
        "",
    ]
    (results_dir / "findings.md").write_text(
        "\n".join(findings), encoding="utf-8"
    )
    metadata = {
        "league": "beta",
        "v2_database": str(v2_database),
        "simulation_database": str(simulation_database),
        "candidate": CANDIDATE,
        "max_template_season": int(max_season),
        "targets": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "player_level_pass": passed,
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(decisions.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
