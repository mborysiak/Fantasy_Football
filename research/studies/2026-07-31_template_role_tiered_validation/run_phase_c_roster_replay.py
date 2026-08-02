"""Paired roster-level replay of production versus the Phase-B finalist."""

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
SNAKE_ROOT = REPO_ROOT.parent / "Fantasy_Football_Snake"
PHASE_B_PATH = STUDY_DIR / "run_phase_b_replay.py"
ROSTER_REFERENCE_PATH = (
    SNAKE_ROOT
    / "research"
    / "studies"
    / "2026-07-22_joint_template_blend_rolling_validation"
    / "run_validation.py"
)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_b = import_module("role_tiered_phase_b", PHASE_B_PATH)
roster_reference = import_module(
    "role_tiered_roster_reference",
    ROSTER_REFERENCE_PATH,
)
receiver_rate = phase_b.receiver_rate
pruning = phase_b.pruning
base = phase_b.base
builder = phase_b.builder

ORIGIN_START = 2017
RECENT_START = 2020
ROOMS_PER_ORIGIN = 12
TEAMS_PER_ROOM = 12
ROSTER_SIZE = 20
SCENARIOS = 384
BOOTSTRAP_REPEATS = 5_000
ROOT_SEED = 20260733
WEEK_COLS = [f"week_{week}" for week in builder.WEEKS]
PLAYED_WEEK_COLS = [f"played_week_{week}" for week in builder.WEEKS]
RESID_COLS = roster_reference.RESID_COLS
MATCHERS = {
    "production": deepcopy(builder.MATCH_FEATURE_WEIGHTS),
    "flatter_w025_all": {
        position: {
            feature: weight * 0.25
            for feature, weight in position_weights.items()
        }
        for position, position_weights in builder.MATCH_FEATURE_WEIGHTS.items()
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_oos_forecasts(league: str, max_season: int) -> pd.DataFrame:
    select_cols = ", ".join(RESID_COLS)
    forecasts = builder.dm.read(
        f"""
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               pred_fp_per_game production_oos_pred_fp_per_game,
               y_act production_validation_y_act,
               resid_calibration_available,
               {select_cols}
        FROM Final_Validations_Resid
        WHERE version='{league}'
              AND model_spec_asof_year={builder.YEAR}
              AND data_oos=1
              AND season BETWEEN {ORIGIN_START} AND {max_season}
        """,
        "Validations",
    )
    forecasts = builder.clean_player_names(forecasts)
    if forecasts.duplicated(["player", "season", "pos"]).any():
        raise ValueError(f"Duplicate {league} final-validation forecast rows.")
    forecasts[RESID_COLS] = forecasts[RESID_COLS].fillna(0)
    forecasts["resid_calibration_available"] = (
        forecasts.resid_calibration_available.fillna(0).astype(int)
    )
    return forecasts


def build_targets(
    templates: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    targets = base.build_production_oos_target_templates(templates, forecasts)
    targets["actual_zero_player_weeks"] = (
        targets[PLAYED_WEEK_COLS].to_numpy(dtype=float) <= 0
    ).sum(axis=1)
    targets["actual_zero_active"] = targets.active_games.eq(0).astype(int)
    return targets.reset_index(drop=True)


def build_matcher_pools(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[tuple[str, int, str, str], dict]:
    grouped_donors = {
        (season, pos): group.reset_index(drop=True)
        for (season, pos), group in templates.groupby(["season", "pos"])
    }
    donor_seasons = sorted(templates.season.unique())
    donors_by_origin_pos = {}
    for season in sorted(targets.season.unique()):
        for pos in builder.POSITIONS:
            donors = pd.concat(
                [
                    grouped_donors[(donor_season, pos)]
                    for donor_season in donor_seasons
                    if donor_season < season
                    and (donor_season, pos) in grouped_donors
                ],
                ignore_index=True,
            )
            donors_by_origin_pos[(season, pos)] = donors[
                donors.template_eligible.eq(1)
            ].reset_index(drop=True)

    pools = {}
    total = len(targets) * len(MATCHERS)
    completed = 0
    for target in targets.itertuples(index=False):
        donors = donors_by_origin_pos[(target.season, target.pos)]
        for matcher, weights in MATCHERS.items():
            specification = {
                "weights": weights,
                "recency_half_life": 12.0,
            }
            selected = pruning.selected_pool(target, donors, specification)
            selected_donors = selected["donors"]
            probabilities = selected["probabilities"]
            residuals = selected_donors.active_ppg_resid.to_numpy(dtype=float)
            residual_mean = float(np.average(residuals, weights=probabilities))
            centered_residuals = residuals - residual_mean
            residual_sd = float(
                np.sqrt(
                    np.average(
                        np.square(centered_residuals),
                        weights=probabilities,
                    )
                )
            )
            pools[(matcher, int(target.season), target.player, target.pos)] = {
                "profiles": selected_donors[WEEK_COLS].to_numpy(
                    dtype=np.float32
                ),
                "played_profiles": selected_donors[
                    PLAYED_WEEK_COLS
                ].to_numpy(dtype=np.int8),
                "zero_active": selected_donors.active_games.eq(0).to_numpy(),
                "centered_residuals": centered_residuals.astype(np.float32),
                "probabilities": probabilities.astype(np.float64),
                "residual_sd": residual_sd,
                "raw_residual_mean": residual_mean,
                "zero_active_probability": float(
                    probabilities[
                        selected_donors.active_games.eq(0).to_numpy()
                    ].sum()
                ),
                "expected_active_games": float(
                    np.average(
                        selected_donors.active_games.to_numpy(dtype=float),
                        weights=probabilities,
                    )
                ),
            }
            completed += 1
        if completed % 600 == 0 or completed == total:
            print(f"Built {completed}/{total} matcher-target pools", flush=True)
    return pools


def build_season_banks(
    season_targets: pd.DataFrame,
    pools: dict,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    player_count = len(season_targets)
    point_forecast = season_targets.historical_pred_fp_per_game.to_numpy(
        dtype=np.float32
    )
    banks = {}
    audit_rows = []
    for matcher in MATCHERS:
        sampled_profiles = np.empty(
            (SCENARIOS, player_count, len(builder.WEEKS)),
            dtype=np.float32,
        )
        sampled_residuals = np.empty(
            (SCENARIOS, player_count), dtype=np.float32
        )
        sampled_played = np.empty(
            (SCENARIOS, player_count, len(builder.WEEKS)),
            dtype=np.int8,
        )
        sampled_zero_active = np.empty(
            (SCENARIOS, player_count), dtype=bool
        )
        for player_index, target in enumerate(
            season_targets.itertuples(index=False)
        ):
            pool = pools[
                (matcher, int(target.season), target.player, target.pos)
            ]
            rng = np.random.default_rng(
                builder.stable_seed(
                    ROOT_SEED,
                    target.season,
                    target.player,
                    target.pos,
                    "outcomes",
                )
            )
            donor_indices = np.searchsorted(
                np.cumsum(pool["probabilities"]),
                rng.random(SCENARIOS),
                side="right",
            )
            donor_indices = np.minimum(
                donor_indices, len(pool["probabilities"]) - 1
            )
            profiles = pool["profiles"][donor_indices]
            played_profiles = pool["played_profiles"][donor_indices]
            template_residuals = pool["centered_residuals"][donor_indices]
            knots = roster_reference.residual_knots(target)
            model_grid = roster_reference.interpolate_residuals(
                knots,
                (np.arange(4096, dtype=float) + 0.5) / 4096,
            )
            model_sd = float(np.std(model_grid))
            if pool["residual_sd"] > 1e-6 and model_sd > 1e-6:
                residuals = (
                    template_residuals * model_sd / pool["residual_sd"]
                )
            else:
                residuals = np.zeros_like(template_residuals)
            sampled_profiles[:, player_index, :] = profiles
            sampled_played[:, player_index, :] = played_profiles
            sampled_zero_active[:, player_index] = pool["zero_active"][
                donor_indices
            ]
            sampled_residuals[:, player_index] = residuals
            audit_rows.append(
                {
                    "matcher": matcher,
                    "season": int(target.season),
                    "player": target.player,
                    "pos": target.pos,
                    "pool_raw_residual_mean": pool["raw_residual_mean"],
                    "pool_residual_sd": pool["residual_sd"],
                    "model_residual_sd": model_sd,
                    "pool_zero_active_probability": pool[
                        "zero_active_probability"
                    ],
                    "pool_expected_active_games": pool[
                        "expected_active_games"
                    ],
                }
            )
        sampled_ppg = np.maximum(
            point_forecast[None, :] + sampled_residuals,
            0,
        ).astype(np.float32)
        banks[matcher] = {
            "weekly_scores": sampled_profiles * sampled_ppg[:, :, None],
            "missed_player_weeks": sampled_played <= 0,
            "zero_active_players": sampled_zero_active,
        }
    return banks, pd.DataFrame(audit_rows)


def score_rosters(
    targets: pd.DataFrame,
    pools: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    audits = []
    for season, season_targets in targets.groupby("season", sort=True):
        season_targets = season_targets.reset_index(drop=True)
        banks, audit = build_season_banks(season_targets, pools)
        audits.append(audit)
        actual_weekly = (
            season_targets[WEEK_COLS].to_numpy(dtype=np.float32)
            * season_targets.active_ppg.to_numpy(dtype=np.float32)[:, None]
        )
        positions = season_targets.pos.to_numpy()
        rosters = roster_reference.draft_rooms(season_targets, int(season))
        for roster in rosters:
            indices = np.asarray(roster["roster_indices"], dtype=int)
            roster_positions = positions[indices]
            actual_score = float(
                roster_reference.best_ball_score(
                    actual_weekly[indices][None, :, :],
                    roster_positions,
                )[0]
            )
            actual_zero_player_weeks = int(
                (
                    season_targets.iloc[indices][PLAYED_WEEK_COLS]
                    .to_numpy(dtype=float)
                    <= 0
                ).sum()
            )
            actual_zero_active_players = int(
                season_targets.iloc[indices].active_games.eq(0).sum()
            )
            common = {
                "season": int(season),
                "room": int(roster["room"]),
                "team": int(roster["team"]),
                "roster_id": (
                    f"{int(season)}_{int(roster['room'])}_{int(roster['team'])}"
                ),
                "actual_score": actual_score,
                "actual_zero_player_weeks": actual_zero_player_weeks,
                "actual_zero_active_players": actual_zero_active_players,
                "actual_zero_lineup_slots": (
                    roster_reference.actual_lineup_zero_slots(
                        actual_weekly[indices], roster_positions
                    )
                ),
            }
            for matcher, bank in banks.items():
                scores = roster_reference.best_ball_score(
                    bank["weekly_scores"][:, indices, :],
                    roster_positions,
                )
                missed = bank["missed_player_weeks"][:, indices, :].sum(
                    axis=(1, 2)
                )
                zero_active = bank["zero_active_players"][:, indices].sum(
                    axis=1
                )
                q10, q50, q90 = np.quantile(scores, [0.10, 0.50, 0.90])
                records.append(
                    {
                        **common,
                        "matcher": matcher,
                        "predicted_mean": float(scores.mean()),
                        "predicted_q10": float(q10),
                        "predicted_q50": float(q50),
                        "predicted_q90": float(q90),
                        "score_crps": roster_reference.empirical_crps(
                            scores, actual_score
                        ),
                        "score_covered_80": int(
                            q10 <= actual_score <= q90
                        ),
                        "predicted_zero_player_weeks_mean": float(
                            missed.mean()
                        ),
                        "zero_player_weeks_crps": (
                            roster_reference.empirical_crps(
                                missed,
                                actual_zero_player_weeks,
                            )
                        ),
                        "predicted_zero_active_players_mean": float(
                            zero_active.mean()
                        ),
                        "zero_active_players_crps": (
                            roster_reference.empirical_crps(
                                zero_active,
                                actual_zero_active_players,
                            )
                        ),
                    }
                )
        print(
            f"Scored {len(rosters)} rosters for {int(season)}",
            flush=True,
        )
    return pd.DataFrame(records), pd.concat(audits, ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {
        "all_2017_2025": (2017, 2025),
        "development_2017_2022": (2017, 2022),
        "recent_2020_2025": (2020, 2025),
        "temporal_2023_2025": (2023, 2025),
    }
    for period, (start, end) in periods.items():
        scoped = frame[frame.season.between(start, end)]
        for matcher, group in scoped.groupby("matcher"):
            score_error = group.predicted_mean - group.actual_score
            missed_error = (
                group.predicted_zero_player_weeks_mean
                - group.actual_zero_player_weeks
            )
            rows.append(
                {
                    "period": period,
                    "matcher": matcher,
                    "n": len(group),
                    "score_crps": float(group.score_crps.mean()),
                    "score_bias": float(score_error.mean()),
                    "score_coverage_80": float(
                        group.score_covered_80.mean()
                    ),
                    "mean_interval_width": float(
                        (group.predicted_q90 - group.predicted_q10).mean()
                    ),
                    "zero_player_weeks_crps": float(
                        group.zero_player_weeks_crps.mean()
                    ),
                    "zero_player_weeks_bias": float(missed_error.mean()),
                    "zero_active_players_crps": float(
                        group.zero_active_players_crps.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def season_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    rng = np.random.default_rng(ROOT_SEED + 99)
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
        baseline = scoped[scoped.matcher.eq("production")]
        candidate = scoped[scoped.matcher.eq("flatter_w025_all")]
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
                size=(BOOTSTRAP_REPEATS, len(seasons)),
            )
            draws = season_delta[sampled].mean(axis=1)
            output.append(
                {
                    "period": period,
                    "candidate": "flatter_w025_all",
                    "baseline": "production",
                    "metric": metric,
                    "n": len(paired),
                    "season_clusters": len(seasons),
                    "candidate_minus_baseline": float(season_delta.mean()),
                    "bootstrap_p025": float(np.quantile(draws, 0.025)),
                    "bootstrap_p975": float(np.quantile(draws, 0.975)),
                    "probability_candidate_better": float(
                        np.mean(draws < 0)
                    ),
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else STUDY_DIR / f"results_phase_c_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phase_b.configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    roster_reference.builder.LEAGUE = league
    roster_reference.ROOMS_PER_ORIGIN = ROOMS_PER_ORIGIN
    roster_reference.TEAMS_PER_ROOM = TEAMS_PER_ROOM
    roster_reference.ROSTER_SIZE = ROSTER_SIZE
    roster_reference.ROOT_SEED = ROOT_SEED
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
    templates = builder.build_weekly_templates(
        projections,
        weekly,
        league=league,
    )
    forecasts = load_oos_forecasts(league, max_season)
    targets = build_targets(templates, forecasts)
    pools = build_matcher_pools(templates, targets)
    predictions, pool_audit = score_rosters(targets, pools)
    summary = summarize(predictions)
    bootstrap = season_bootstrap(predictions)

    predictions.to_csv(results_dir / "roster_predictions.csv", index=False)
    pool_audit.to_csv(results_dir / "target_pool_audit.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    bootstrap.to_csv(results_dir / "season_bootstrap.csv", index=False)
    metadata = {
        "league": league,
        "max_season": int(max_season),
        "target_players": int(len(targets)),
        "rooms_per_origin": ROOMS_PER_ORIGIN,
        "teams_per_room": TEAMS_PER_ROOM,
        "roster_size": ROSTER_SIZE,
        "scenarios": SCENARIOS,
        "rosters": int(predictions.roster_id.nunique()),
        "prediction_rows": int(len(predictions)),
        "matchers": list(MATCHERS),
        "future_donor_rows": 0,
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
