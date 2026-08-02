"""Score Phase-C roster banks on room-level championship probability."""

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
for import_root in (REPO_ROOT, REPO_ROOT / "Scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

PHASE_C_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_c_roster_replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "upside_phase_c_reference",
    PHASE_C_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import Phase-C replay from {PHASE_C_PATH}")
phase_c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase_c
SPEC.loader.exec_module(phase_c)

phase_b = phase_c.phase_b
roster_reference = phase_c.roster_reference
builder = phase_c.builder
base = phase_c.base
receiver_rate = phase_b.receiver_rate
WEEK_COLS = phase_c.WEEK_COLS
MATCHER_WEIGHTS = {
    "production": phase_b.METHODS["production"]["weights"],
    "flatter_w025_all": phase_b.METHODS["flatter_w025_all"]["weights"],
    "wr_ppg225_both025": phase_b.METHODS["wr_ppg225_both025"]["weights"],
}
MATCHERS = tuple(MATCHER_WEIGHTS)
BASELINE_MATCHER = "production"
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def scenario_top_probabilities(scores: np.ndarray, top_n: int) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("Room scores must align as teams by scenarios.")
    team_count, scenario_count = scores.shape
    if not 1 <= top_n <= team_count:
        raise ValueError("Invalid room finish threshold.")
    if top_n == 1:
        maxima = scores.max(axis=0, keepdims=True)
        tied = np.isclose(scores, maxima)
        return (tied / tied.sum(axis=0, keepdims=True)).mean(axis=1)
    order = np.argsort(-scores, axis=0, kind="stable")[:top_n]
    finish = np.zeros((team_count, scenario_count), dtype=float)
    np.put_along_axis(finish, order, 1.0, axis=0)
    return finish.mean(axis=1)


def score_rooms(targets: pd.DataFrame, pools: dict) -> pd.DataFrame:
    records = []
    for season, season_targets in targets.groupby("season", sort=True):
        season_targets = season_targets.reset_index(drop=True)
        banks, _ = phase_c.build_season_banks(season_targets, pools)
        actual_weekly = (
            season_targets[WEEK_COLS].to_numpy(dtype=np.float32)
            * season_targets.active_ppg.to_numpy(dtype=np.float32)[:, None]
        )
        positions = season_targets.pos.to_numpy()
        rosters = roster_reference.draft_rooms(season_targets, int(season))
        for room, room_rosters in pd.DataFrame(rosters).groupby("room", sort=True):
            room_roster_records = room_rosters.to_dict("records")
            teams = np.asarray([int(roster["team"]) for roster in room_roster_records])
            actual_scores = []
            scenario_scores = {
                matcher: [] for matcher in MATCHERS
            }
            for roster in room_roster_records:
                indices = np.asarray(roster["roster_indices"], dtype=int)
                roster_positions = positions[indices]
                actual_scores.append(
                    float(
                        roster_reference.best_ball_score(
                            actual_weekly[indices][None, :, :],
                            roster_positions,
                        )[0]
                    )
                )
                for matcher in MATCHERS:
                    scenario_scores[matcher].append(
                        roster_reference.best_ball_score(
                            banks[matcher]["weekly_scores"][:, indices, :],
                            roster_positions,
                        )
                    )
            actual_scores = np.asarray(actual_scores, dtype=float)
            actual_order = np.argsort(-actual_scores, kind="stable")
            actual_champion = np.zeros(len(teams), dtype=int)
            actual_champion[actual_order[0]] = 1
            actual_top3 = np.zeros(len(teams), dtype=int)
            actual_top3[actual_order[:3]] = 1
            for matcher in MATCHERS:
                score_matrix = np.asarray(scenario_scores[matcher], dtype=float)
                champion_probability = scenario_top_probabilities(score_matrix, 1)
                top3_probability = scenario_top_probabilities(score_matrix, 3)
                for team_idx, team in enumerate(teams):
                    scores = score_matrix[team_idx]
                    records.append(
                        {
                            "season": int(season),
                            "room": int(room),
                            "team": int(team),
                            "matcher": matcher,
                            "actual_score": float(actual_scores[team_idx]),
                            "actual_champion": int(actual_champion[team_idx]),
                            "actual_top3": int(actual_top3[team_idx]),
                            "predicted_mean": float(scores.mean()),
                            "predicted_q90": float(np.quantile(scores, 0.90)),
                            "score_crps": roster_reference.empirical_crps(
                                scores,
                                actual_scores[team_idx],
                            ),
                            "championship_probability": float(
                                champion_probability[team_idx]
                            ),
                            "top3_probability": float(top3_probability[team_idx]),
                            "championship_brier_row": float(
                                (
                                    champion_probability[team_idx]
                                    - actual_champion[team_idx]
                                )
                                ** 2
                            ),
                            "top3_brier_row": float(
                                (top3_probability[team_idx] - actual_top3[team_idx])
                                ** 2
                            ),
                        }
                    )
        print(f"Scored championship probabilities for {int(season)}", flush=True)
    return pd.DataFrame(records)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        period_frame = predictions[predictions.season.between(start, end)]
        for matcher, group in period_frame.groupby("matcher", sort=True):
            room_winners = group[group.actual_champion.eq(1)]
            room_top3 = group[group.actual_top3.eq(1)]
            champion_hits = []
            top3_overlap = []
            for _, room_group in group.groupby(["season", "room"], sort=True):
                champion_pick = room_group.championship_probability.idxmax()
                champion_hits.append(int(room_group.loc[champion_pick, "actual_champion"]))
                predicted_top3 = room_group.nlargest(3, "top3_probability").index
                top3_overlap.append(int(room_group.loc[predicted_top3, "actual_top3"].sum()))
            rows.append(
                {
                    "period": period,
                    "matcher": matcher,
                    "teams": int(len(group)),
                    "rooms": int(group.groupby(["season", "room"]).ngroups),
                    "score_crps": float(group.score_crps.mean()),
                    "championship_brier": float(group.championship_brier_row.mean()),
                    "championship_log_loss": float(
                        -np.log(
                            np.clip(
                                room_winners.championship_probability.to_numpy(float),
                                1e-9,
                                1.0,
                            )
                        ).mean()
                    ),
                    "actual_winner_probability": float(
                        room_winners.championship_probability.mean()
                    ),
                    "champion_pick_hit_rate": float(np.mean(champion_hits)),
                    "top3_brier": float(group.top3_brier_row.mean()),
                    "actual_top3_probability": float(
                        room_top3.top3_probability.mean()
                    ),
                    "predicted_top3_overlap": float(np.mean(top3_overlap)),
                }
            )
    output = pd.DataFrame(rows)
    metric_cols = [
        "score_crps",
        "championship_brier",
        "championship_log_loss",
        "actual_winner_probability",
        "champion_pick_hit_rate",
        "top3_brier",
        "actual_top3_probability",
        "predicted_top3_overlap",
    ]
    baseline = output[output.matcher.eq(BASELINE_MATCHER)][["period", *metric_cols]]
    baseline = baseline.rename(
        columns={metric: f"{metric}_baseline" for metric in metric_cols}
    )
    output = output.merge(baseline, on="period", how="left", validate="many_to_one")
    for metric in metric_cols:
        output[f"{metric}_delta"] = output[metric] - output[f"{metric}_baseline"]
    return output


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else STUDY_DIR / f"results_roster_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phase_b.configure_reference_globals()
    phase_c.MATCHERS = MATCHER_WEIGHTS
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    roster_reference.builder.LEAGUE = league
    roster_reference.ROOMS_PER_ORIGIN = phase_c.ROOMS_PER_ORIGIN
    roster_reference.TEAMS_PER_ROOM = phase_c.TEAMS_PER_ROOM
    roster_reference.ROSTER_SIZE = phase_c.ROSTER_SIZE
    roster_reference.ROOT_SEED = phase_c.ROOT_SEED
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db is not None
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    rates = receiver_rate.load_receiver_rate_features(v2_database, max_season)
    templates = receiver_rate.reattach_template_player_keys(templates, projections)
    templates = receiver_rate.attach_receiver_rate_features(templates, rates)
    forecasts = phase_c.load_oos_forecasts(league, max_season)
    targets = phase_c.build_targets(templates, forecasts)
    pools = phase_c.build_matcher_pools(templates, targets)
    predictions = score_rooms(targets, pools)
    summary = summarize(predictions)

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
        "matchers": list(MATCHERS),
        "production_changed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
