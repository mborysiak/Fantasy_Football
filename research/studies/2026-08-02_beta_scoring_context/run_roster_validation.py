"""Paired roster replay of hybrid versus fully beta-scored context."""

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
PLAYER_RUNNER_PATH = STUDY_DIR / "run_validation.py"
PHASE_C_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_c_roster_replay.py"
)
BASELINE = "production_hybrid"
CONTEXT_ONLY = "beta_context_only"
EXPERT_CENTER = "beta_scored_full"
CANDIDATE = "beta_scored_ppg_rank_w050"
CANDIDATES = (CONTEXT_ONLY, EXPERT_CENTER, CANDIDATE)
ARMS = (BASELINE, *CANDIDATES)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


player_runner = load_module("beta_context_player_runner", PLAYER_RUNNER_PATH)
phase_c = load_module("beta_context_phase_c", PHASE_C_PATH)
builder = player_runner.builder
base = player_runner.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-db", type=Path, required=True)
    parser.add_argument("--simulation-db", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=STUDY_DIR / "results_roster",
    )
    return parser.parse_args()


def configure_reference_globals(simulation_database: Path) -> None:
    player_runner.phase_b.configure_reference_globals()
    builder.set_simulation_db(simulation_database)
    builder.set_active_league("beta")
    base.builder.LEAGUE = "beta"

    # Phase C was written as a reusable research harness. Point its module
    # globals at the same builder stack used to construct the challenger so
    # both arms share one staged database and identity contract.
    phase_c.builder = builder
    phase_c.base = base
    phase_c.pruning = player_runner.pruning
    phase_c.roster_reference.builder = builder
    phase_c.roster_reference.builder.LEAGUE = "beta"
    phase_c.roster_reference.ROOMS_PER_ORIGIN = phase_c.ROOMS_PER_ORIGIN
    phase_c.roster_reference.TEAMS_PER_ROOM = phase_c.TEAMS_PER_ROOM
    phase_c.roster_reference.ROSTER_SIZE = phase_c.ROSTER_SIZE
    phase_c.roster_reference.ROOT_SEED = phase_c.ROOT_SEED


def build_roster_targets(
    templates: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    targets = phase_c.build_targets(templates, forecasts)
    # A legal best-ball roster requires QBs. Because the complete 2018 beta QB
    # context is quarantined, retaining only that season's skill positions
    # would either make drafting impossible or silently restore DK QB units.
    # Exclude 2018 symmetrically from every arm.
    return targets[~targets.season.eq(2018)].reset_index(drop=True)


def assert_paired_targets(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    keys = ["player", "pos", "season"]
    baseline_keys = baseline[keys].sort_values(keys).reset_index(drop=True)
    candidate_keys = candidate[keys].sort_values(keys).reset_index(drop=True)
    if not baseline_keys.equals(candidate_keys):
        raise ValueError("Roster target cohort changed between scoring arms.")
    paired = baseline.merge(
        candidate,
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    observed = [
        "active_ppg",
        "active_games",
        *phase_c.WEEK_COLS,
        *phase_c.PLAYED_WEEK_COLS,
    ]
    for column in observed:
        if not np.allclose(
            pd.to_numeric(
                paired[f"{column}_baseline"], errors="coerce"
            ),
            pd.to_numeric(
                paired[f"{column}_candidate"], errors="coerce"
            ),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"Observed roster outcome changed: {column}")


def replay_arm(
    arm: str,
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    if arm == CANDIDATE:
        for position_weights in weights.values():
            position_weights["match_projection_rank_pct"] = 0.5
            position_weights["market_projection_gap"] = 0.0
    phase_c.MATCHERS = {arm: weights}
    pools = phase_c.build_matcher_pools(templates, targets)
    return phase_c.score_rosters(targets, pools)


def season_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    rng = np.random.default_rng(phase_c.ROOT_SEED + 99)
    metrics = (
        "score_crps",
        "zero_player_weeks_crps",
        "zero_active_players_crps",
    )
    for period, (start, end) in {
        "development_2017_2022": (2017, 2022),
        "temporal_2023_2025": (2023, 2025),
    }.items():
        scoped = frame[frame.season.between(start, end)]
        baseline = scoped[scoped.matcher.eq(BASELINE)]
        for candidate_name in CANDIDATES:
            candidate = scoped[scoped.matcher.eq(candidate_name)]
            keys = ["season", "room", "team", "roster_id"]
            paired = candidate.merge(
                baseline,
                on=keys,
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            seasons = np.sort(paired.season.unique())
            for metric in metrics:
                season_delta = (
                    paired.assign(
                        delta=(
                            paired[f"{metric}_candidate"]
                            - paired[f"{metric}_baseline"]
                        )
                    )
                    .groupby("season")
                    .delta.mean()
                    .reindex(seasons)
                    .to_numpy(dtype=float)
                )
                sampled = rng.integers(
                    0,
                    len(seasons),
                    size=(phase_c.BOOTSTRAP_REPEATS, len(seasons)),
                )
                draws = season_delta[sampled].mean(axis=1)
                output.append(
                    {
                        "period": period,
                        "candidate": candidate_name,
                        "baseline": BASELINE,
                        "metric": metric,
                        "n": len(paired),
                        "season_clusters": len(seasons),
                        "candidate_minus_baseline": float(
                            season_delta.mean()
                        ),
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


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name in CANDIDATES:
        gates = []
        candidate_rows = []
        for period in ("development_2017_2022", "temporal_2023_2025"):
            scoped = summary[summary.period.eq(period)].set_index("matcher")
            baseline = float(scoped.loc[BASELINE, "score_crps"])
            candidate = float(scoped.loc[candidate_name, "score_crps"])
            relative_delta = (candidate - baseline) / baseline
            passed = relative_delta <= 0.005
            gates.append(passed)
            candidate_rows.append(
                {
                    "candidate": candidate_name,
                    "period": period,
                    "baseline_score_crps": baseline,
                    "candidate_score_crps": candidate,
                    "score_crps_relative_delta": relative_delta,
                    "gate_within_0_5_percent": bool(passed),
                }
            )
        for row in candidate_rows:
            rows.append({**row, "roster_level_pass": bool(all(gates))})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    v2_database = args.v2_db.resolve()
    simulation_database = args.simulation_db.resolve()
    started = time.perf_counter()
    configure_reference_globals(simulation_database)

    max_season = builder.get_daily_max_template_season()
    weekly = builder.load_weekly_points(max_season, league="beta")
    forecasts = phase_c.load_oos_forecasts("beta", max_season)
    baseline_projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
        scoring_matched_context=False,
        scoring_matched_fallback_center=False,
    )

    templates_by_arm = {}
    targets_by_arm = {}
    template_audits = []
    for arm in ARMS:
        print(f"Building roster arm {arm}", flush=True)
        options = (
            player_runner.DECOUPLED_OPTIONS
            if arm == CANDIDATE
            else player_runner.ARMS[arm]
        )
        templates, audit = player_runner.build_arm_templates(
            arm,
            options,
            v2_database=v2_database,
            baseline_projections=baseline_projections,
            weekly=weekly,
        )
        templates_by_arm[arm] = templates
        targets_by_arm[arm] = build_roster_targets(templates, forecasts)
        template_audits.append(audit)
    for arm in ARMS[1:]:
        assert_paired_targets(targets_by_arm[BASELINE], targets_by_arm[arm])

    prediction_frames = []
    pool_audits = []
    for arm in ARMS:
        print(f"Replaying roster arm {arm}", flush=True)
        predictions, pool_audit = replay_arm(
            arm,
            templates_by_arm[arm],
            targets_by_arm[arm],
        )
        prediction_frames.append(predictions)
        pool_audits.append(pool_audit)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = phase_c.summarize(predictions)
    bootstrap = season_bootstrap(predictions)
    decisions = decision_table(summary)

    predictions.to_csv(results_dir / "roster_predictions.csv", index=False)
    pd.concat(pool_audits, ignore_index=True).to_csv(
        results_dir / "target_pool_audit.csv", index=False
    )
    summary.to_csv(results_dir / "summary.csv", index=False)
    bootstrap.to_csv(results_dir / "season_bootstrap.csv", index=False)
    decisions.to_csv(results_dir / "decision_gates.csv", index=False)
    pd.concat(template_audits, ignore_index=True).to_csv(
        results_dir / "template_policy_audit.csv", index=False
    )

    passed = bool(
        decisions.loc[
            decisions.candidate.eq(CANDIDATE), "roster_level_pass"
        ].iloc[0]
    )
    findings = [
        "# Roster validation findings",
        "",
        player_runner.markdown_table(decisions),
        "",
        (
            "At least one fully beta-scored representation passes the roster gate."
            if passed
            else "No fully beta-scored representation passes the roster gate."
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
        "max_template_season": int(max_season),
        "target_players": int(len(targets_by_arm[BASELINE])),
        "rooms_per_origin": phase_c.ROOMS_PER_ORIGIN,
        "teams_per_room": phase_c.TEAMS_PER_ROOM,
        "roster_size": phase_c.ROSTER_SIZE,
        "scenarios": phase_c.SCENARIOS,
        "rosters": int(predictions.roster_id.nunique()),
        "prediction_rows": int(len(predictions)),
        "matchers": list(ARMS),
        "future_donor_rows": 0,
        "roster_level_pass": passed,
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(decisions.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
