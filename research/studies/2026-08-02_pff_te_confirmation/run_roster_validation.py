"""Replacement-aware roster/championship replay for the TE YAC finalist."""

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
for root in (REPO_ROOT, STUDY_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pff_te_features import YAC_MATCH, attach_template_features, build_te_profiles


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = _load(
    "pff_te_roster_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-01_upside_objective_audit"
    / "run_roster_championship_replay.py",
)

phase_c = reference.phase_c
builder = reference.builder
base = reference.base
receiver_rate = reference.receiver_rate
roster_reference = reference.roster_reference
RAW_DB = REPO_ROOT / "Data" / "Databases" / "Season_Stats_New.sqlite3"
BASELINE = "production"
FINALIST = "te_pff_yac_w025"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def matcher_weights() -> dict[str, dict]:
    production = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    challenger = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    challenger["TE"][YAC_MATCH] = 0.25
    return {BASELINE: production, FINALIST: challenger}


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = args.results_dir.resolve() if args.results_dir else STUDY_DIR / f"results_roster_{league}"
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    weights = matcher_weights()
    reference.MATCHER_WEIGHTS = weights
    reference.MATCHERS = tuple(weights)
    reference.BASELINE_MATCHER = BASELINE
    phase_c.MATCHERS = weights
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    roster_reference.builder.LEAGUE = league
    roster_reference.ROOMS_PER_ORIGIN = phase_c.ROOMS_PER_ORIGIN
    roster_reference.TEAMS_PER_ROOM = phase_c.TEAMS_PER_ROOM
    roster_reference.ROSTER_SIZE = phase_c.ROSTER_SIZE
    roster_reference.ROOT_SEED = phase_c.ROOT_SEED
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    projections = builder.load_historical_projection_context(max_season, v2_database=v2_database)
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    templates = receiver_rate.reattach_template_player_keys(templates, projections)
    profiles = build_te_profiles(v2_database, RAW_DB, max_season)
    templates = attach_template_features(templates, profiles)
    forecasts = phase_c.load_oos_forecasts(league, max_season)
    targets = phase_c.build_targets(templates, forecasts)
    pools = phase_c.build_matcher_pools(templates, targets)
    predictions = reference.score_rooms(targets, pools)
    summary = reference.summarize(predictions)

    predictions.to_csv(results_dir / "roster_championship_predictions.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_season": int(max_season),
        "target_players": int(len(targets)),
        "rooms_per_origin": phase_c.ROOMS_PER_ORIGIN,
        "teams_per_room": phase_c.TEAMS_PER_ROOM,
        "roster_size": phase_c.ROSTER_SIZE,
        "scenarios": phase_c.SCENARIOS,
        "rooms": int(predictions.groupby(["season", "room"]).ngroups),
        "prediction_rows": int(len(predictions)),
        "matchers": list(weights),
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

