"""Fixed-roster Snake score and championship replay for the Ridge point center."""

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase_c = load_module(
    "ridge_swap_phase_c_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_c_roster_replay.py",
)
upside = load_module(
    "ridge_swap_upside_reference",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-01_upside_objective_audit"
    / "run_roster_championship_replay.py",
)

phase_b = phase_c.phase_b
receiver_rate = phase_b.receiver_rate
builder = phase_c.builder
base = phase_c.base
roster_reference = phase_c.roster_reference
METHODS = ("production", "ridge_swap")
PERIODS = {
    "all_2018_2025": (2018, 2025),
    "development_2018_2022": (2018, 2022),
    "temporal_2023_2025": (2023, 2025),
}
BOOTSTRAP_REPEATS = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--template-results-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_targets(template_results_dir: Path) -> dict[str, pd.DataFrame]:
    targets = {
        method: pd.read_csv(
            template_results_dir / f"target_rows_{method}.csv"
        )
        for method in METHODS
    }
    keys = ["player_key", "season", "pos"]
    baseline_keys = targets["production"][keys].reset_index(drop=True)
    for method, frame in targets.items():
        if not frame[keys].reset_index(drop=True).equals(baseline_keys):
            raise ValueError(f"{method} target rows do not match frozen order")
        if frame.duplicated(keys).any():
            raise ValueError(f"{method} target rows are not unique")
    baseline = targets["production"]
    valid_seasons = [
        int(season)
        for season, group in baseline.groupby("season", sort=True)
        if set(group.pos.unique()) == set(builder.POSITIONS)
    ]
    dropped_seasons = sorted(
        set(int(value) for value in baseline.season.unique())
        - set(valid_seasons)
    )
    if dropped_seasons:
        print(
            "Dropping roster origins without all four positions: "
            + ", ".join(str(value) for value in dropped_seasons),
            flush=True,
        )
        targets = {
            method: frame[frame.season.isin(valid_seasons)]
            .reset_index(drop=True)
            for method, frame in targets.items()
        }
    return targets


def build_pools(
    templates: pd.DataFrame, targets: dict[str, pd.DataFrame]
) -> dict:
    pools = {}
    for method in METHODS:
        phase_c.MATCHERS = {
            method: deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        }
        pools.update(
            phase_c.build_matcher_pools(templates, targets[method])
        )
    return pools


def build_method_banks(
    season: int,
    season_targets: dict[str, pd.DataFrame],
    pools: dict,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    banks = {}
    audits = []
    for method in METHODS:
        phase_c.MATCHERS = {
            method: deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        }
        method_banks, audit = phase_c.build_season_banks(
            season_targets[method], pools
        )
        banks[method] = method_banks[method]
        audits.append(audit)
    return banks, pd.concat(audits, ignore_index=True)


def score_fixed_rosters(
    targets: dict[str, pd.DataFrame], pools: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    audits = []
    baseline = targets["production"]
    for season in sorted(baseline.season.unique()):
        season_targets = {
            method: targets[method][targets[method].season.eq(season)]
            .reset_index(drop=True)
            for method in METHODS
        }
        keys = ["player_key", "pos"]
        if not season_targets["production"][keys].equals(
            season_targets["ridge_swap"][keys]
        ):
            raise ValueError(f"Target order changed in {season}")
        banks, audit = build_method_banks(
            int(season), season_targets, pools
        )
        audits.append(audit)
        fixed = season_targets["production"]
        actual_weekly = (
            fixed[phase_c.WEEK_COLS].to_numpy(dtype=np.float32)
            * fixed.active_ppg.to_numpy(dtype=np.float32)[:, None]
        )
        positions = fixed.pos.to_numpy()
        rosters = pd.DataFrame(
            roster_reference.draft_rooms(fixed, int(season))
        )
        for room, room_rosters in rosters.groupby("room", sort=True):
            roster_rows = room_rosters.to_dict("records")
            actual_scores = []
            actual_missed = []
            actual_zero_active = []
            score_arrays = {method: [] for method in METHODS}
            missed_arrays = {method: [] for method in METHODS}
            zero_active_arrays = {method: [] for method in METHODS}
            for roster in roster_rows:
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
                actual_missed.append(
                    int(
                        (
                            fixed.iloc[indices][phase_c.PLAYED_WEEK_COLS]
                            .to_numpy(dtype=float)
                            <= 0
                        ).sum()
                    )
                )
                actual_zero_active.append(
                    int(fixed.iloc[indices].active_games.eq(0).sum())
                )
                for method in METHODS:
                    bank = banks[method]
                    score_arrays[method].append(
                        roster_reference.best_ball_score(
                            bank["weekly_scores"][:, indices, :],
                            roster_positions,
                        )
                    )
                    missed_arrays[method].append(
                        bank["missed_player_weeks"][:, indices, :].sum(
                            axis=(1, 2)
                        )
                    )
                    zero_active_arrays[method].append(
                        bank["zero_active_players"][:, indices].sum(axis=1)
                    )
            actual_scores_array = np.asarray(actual_scores, dtype=float)
            actual_order = np.argsort(-actual_scores_array, kind="stable")
            actual_champion = np.zeros(len(roster_rows), dtype=int)
            actual_champion[actual_order[0]] = 1
            actual_top3 = np.zeros(len(roster_rows), dtype=int)
            actual_top3[actual_order[:3]] = 1
            for method in METHODS:
                score_matrix = np.asarray(score_arrays[method], dtype=float)
                champion_probability = upside.scenario_top_probabilities(
                    score_matrix, 1
                )
                top3_probability = upside.scenario_top_probabilities(
                    score_matrix, 3
                )
                for index, roster in enumerate(roster_rows):
                    scores = score_matrix[index]
                    missed = missed_arrays[method][index]
                    zero_active = zero_active_arrays[method][index]
                    q10, q50, q90 = np.quantile(
                        scores, [0.10, 0.50, 0.90]
                    )
                    records.append(
                        {
                            "season": int(season),
                            "room": int(room),
                            "team": int(roster["team"]),
                            "roster_id": (
                                f"{int(season)}_{int(room)}_{int(roster['team'])}"
                            ),
                            "method": method,
                            "actual_score": actual_scores[index],
                            "actual_zero_player_weeks": actual_missed[index],
                            "actual_zero_active_players": actual_zero_active[index],
                            "actual_champion": int(actual_champion[index]),
                            "actual_top3": int(actual_top3[index]),
                            "predicted_mean": float(scores.mean()),
                            "predicted_q10": float(q10),
                            "predicted_q50": float(q50),
                            "predicted_q90": float(q90),
                            "score_crps": roster_reference.empirical_crps(
                                scores, actual_scores[index]
                            ),
                            "score_covered_80": int(
                                q10 <= actual_scores[index] <= q90
                            ),
                            "predicted_zero_player_weeks_mean": float(
                                missed.mean()
                            ),
                            "zero_player_weeks_crps": (
                                roster_reference.empirical_crps(
                                    missed, actual_missed[index]
                                )
                            ),
                            "predicted_zero_active_players_mean": float(
                                zero_active.mean()
                            ),
                            "zero_active_players_crps": (
                                roster_reference.empirical_crps(
                                    zero_active, actual_zero_active[index]
                                )
                            ),
                            "championship_probability": float(
                                champion_probability[index]
                            ),
                            "top3_probability": float(top3_probability[index]),
                            "championship_brier_row": float(
                                (
                                    champion_probability[index]
                                    - actual_champion[index]
                                )
                                ** 2
                            ),
                            "top3_brier_row": float(
                                (top3_probability[index] - actual_top3[index])
                                ** 2
                            ),
                        }
                    )
        print(
            f"Scored {len(rosters)} fixed rosters for {int(season)}",
            flush=True,
        )
    return pd.DataFrame(records), pd.concat(audits, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        scoped = predictions[predictions.season.between(start, end)]
        for method, group in scoped.groupby("method", sort=True):
            winners = group[group.actual_champion.eq(1)]
            top3 = group[group.actual_top3.eq(1)]
            champion_hits = []
            top3_overlap = []
            for _, room in group.groupby(["season", "room"], sort=True):
                champion_pick = room.championship_probability.idxmax()
                champion_hits.append(
                    int(room.loc[champion_pick, "actual_champion"])
                )
                predicted_top3 = room.nlargest(3, "top3_probability").index
                top3_overlap.append(
                    int(room.loc[predicted_top3, "actual_top3"].sum())
                )
            rows.append(
                {
                    "period": period,
                    "method": method,
                    "rosters": len(group),
                    "rooms": group.groupby(["season", "room"]).ngroups,
                    "score_crps": float(group.score_crps.mean()),
                    "score_bias": float(
                        (group.predicted_mean - group.actual_score).mean()
                    ),
                    "score_coverage_80": float(
                        group.score_covered_80.mean()
                    ),
                    "score_interval_width_80": float(
                        (group.predicted_q90 - group.predicted_q10).mean()
                    ),
                    "zero_player_weeks_crps": float(
                        group.zero_player_weeks_crps.mean()
                    ),
                    "zero_active_players_crps": float(
                        group.zero_active_players_crps.mean()
                    ),
                    "championship_brier": float(
                        group.championship_brier_row.mean()
                    ),
                    "championship_log_loss": float(
                        -np.log(
                            np.clip(
                                winners.championship_probability.to_numpy(float),
                                1e-9,
                                1.0,
                            )
                        ).mean()
                    ),
                    "actual_winner_probability": float(
                        winners.championship_probability.mean()
                    ),
                    "champion_pick_hit_rate": float(np.mean(champion_hits)),
                    "top3_brier": float(group.top3_brier_row.mean()),
                    "actual_top3_probability": float(
                        top3.top3_probability.mean()
                    ),
                    "predicted_top3_overlap": float(np.mean(top3_overlap)),
                }
            )
    output = pd.DataFrame(rows)
    metrics = [
        column
        for column in output.columns
        if column not in {"period", "method", "rosters", "rooms"}
    ]
    baseline = output[output.method.eq("production")][
        ["period", *metrics]
    ].rename(columns={metric: f"{metric}_baseline" for metric in metrics})
    output = output.merge(
        baseline, on="period", how="left", validate="many_to_one"
    )
    for metric in metrics:
        output[f"{metric}_delta"] = (
            output[metric] - output[f"{metric}_baseline"]
        )
        if metric.endswith("crps") or metric.endswith("brier"):
            output[f"{metric}_relative_delta"] = (
                output[metric] / output[f"{metric}_baseline"] - 1.0
            )
    return output


def season_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "room", "team", "roster_id"]
    baseline = predictions[predictions.method.eq("production")]
    candidate = predictions[predictions.method.eq("ridge_swap")]
    paired = candidate.merge(
        baseline,
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    metrics = (
        "score_crps",
        "championship_brier_row",
        "top3_brier_row",
    )
    rng = np.random.default_rng(phase_c.ROOT_SEED + 803)
    rows = []
    for period, (start, end) in PERIODS.items():
        scoped = paired[paired.season.between(start, end)]
        seasons = np.sort(scoped.season.unique())
        for metric in metrics:
            deltas = (
                scoped.assign(
                    delta=(
                        scoped[f"{metric}_candidate"]
                        - scoped[f"{metric}_baseline"]
                    )
                )
                .groupby("season")
                .delta.mean()
                .reindex(seasons)
                .to_numpy(float)
            )
            sampled = rng.integers(
                0,
                len(seasons),
                size=(BOOTSTRAP_REPEATS, len(seasons)),
            )
            draws = deltas[sampled].mean(axis=1)
            rows.append(
                {
                    "period": period,
                    "metric": metric,
                    "season_clusters": len(seasons),
                    "candidate_minus_baseline": float(deltas.mean()),
                    "bootstrap_p025": float(np.quantile(draws, 0.025)),
                    "bootstrap_p975": float(np.quantile(draws, 0.975)),
                    "probability_candidate_better": float(
                        np.mean(draws < 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else STUDY_DIR / f"results_roster_{league}"
    )
    template_results_dir = (
        args.template_results_dir.resolve()
        if args.template_results_dir
        else STUDY_DIR / f"results_template_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phase_b.configure_reference_globals()
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
    rates = receiver_rate.load_receiver_rate_features(v2_database, max_season)
    templates = receiver_rate.attach_receiver_rate_features(templates, rates)
    targets = load_targets(template_results_dir)
    pools = build_pools(templates, targets)
    predictions, audit = score_fixed_rosters(targets, pools)
    summary = summarize(predictions)
    bootstrap = season_bootstrap(predictions)

    predictions.to_csv(results_dir / "roster_predictions.csv", index=False)
    audit.to_csv(results_dir / "target_pool_audit.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    bootstrap.to_csv(results_dir / "season_bootstrap.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "template_results_dir": str(template_results_dir),
        "target_rows_per_method": int(len(targets["production"])),
        "rooms_per_origin": phase_c.ROOMS_PER_ORIGIN,
        "teams_per_room": phase_c.TEAMS_PER_ROOM,
        "roster_size": phase_c.ROSTER_SIZE,
        "scenarios": phase_c.SCENARIOS,
        "seasons": sorted(int(value) for value in targets["production"].season.unique()),
        "rooms": int(predictions.groupby(["season", "room"]).ngroups),
        "prediction_rows": int(len(predictions)),
        "roster_policy": "production drafts frozen and shared across methods",
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        summary[summary.method.eq("ridge_swap")][
            [
                "period",
                "rosters",
                "score_crps_relative_delta",
                "score_bias_delta",
                "score_coverage_80_delta",
                "championship_brier_relative_delta",
                "championship_log_loss_delta",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
