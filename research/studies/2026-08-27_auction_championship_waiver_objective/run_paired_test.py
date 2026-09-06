"""Paired 2025 Auction test of waiver and championship objectives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_DIR = ROOT.parent / "Fantasy_Football_App" / "app"
SHARED_STUDY_DIR = (
    ROOT / "research" / "studies" / "2026-08-24_sequential_shared_opportunity"
)
SIMULATION_DB = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-26_auction_2025_historical_replay"
    / "staging"
    / "databases"
    / "Simulation.sqlite3"
)
TAIL_PREDICTIONS = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-01_upside_objective_audit"
    / "results_player_beta"
    / "target_predictions.csv"
)
RESULTS_DIR = STUDY_DIR / "results"
for import_path in (APP_DIR, SHARED_STUDY_DIR, ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import zSequential_Target as sequential  # noqa: E402
from zSim_Helper import FootballSimulation  # noqa: E402
from keeper_market import load_active_keeper_market  # noqa: E402


YEAR = 2025
LEAGUE = "beta"
PRED_VERSION = "final_ensemble"
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POSITION_MIN = {"QB": 1, "RB": 4, "WR": 4, "TE": 1}
POSITION_MAX = {"QB": 1, "RB": 6, "WR": 6, "TE": 2}
REQUIRE_TOP_N = 12
MEAN_NONINFERIORITY_FRAC = 0.0025
CHURN_WAIVER_FLOORS = {"QB": 15.5, "RB": 9.0, "WR": 9.0}
TAIL_RESIDUAL_STRIKE = 5.0
TAIL_WAIVER_BASELINES = {"QB": 15.0, "RB": 7.0, "WR": 7.0, "TE": 5.0}
LCB80_Z = 0.8416212335729143
DEAD_ZONE_RBS = {"Aaron Jones", "Isiah Pacheco", "James Conner"}
ARMS = (
    "baseline",
    "waiver_proxy",
    "championship_tiebreak",
    "combined",
)


def json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def empirical_championship_proxy(
    scores: np.ndarray,
    reference_scores: np.ndarray,
    opponents: int = NUM_TEAMS - 1,
) -> float:
    """Return mean same-context probability of beating every opponent.

    Reference rows are feasible rosters and columns are common weekly-season
    scenarios. Midranks make the result deterministic in tied score cells.
    """

    scores = np.asarray(scores, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    if scores.ndim != 1 or reference_scores.ndim != 2:
        raise ValueError("Championship inputs must be scores and rosters-by-contexts.")
    if reference_scores.shape[1] != len(scores) or reference_scores.shape[0] < 2:
        raise ValueError("Championship reference bank must align and contain two rosters.")
    less = (reference_scores < scores[None, :]).sum(axis=0)
    equal = np.isclose(reference_scores, scores[None, :]).sum(axis=0)
    percentile = (less + 0.5 * equal) / reference_scores.shape[0]
    return float(np.mean(np.power(percentile, int(opponents))))


def choose_lexicographic_candidate(
    metrics: pd.DataFrame,
    tolerance_frac: float = MEAN_NONINFERIORITY_FRAC,
) -> pd.Series:
    """Choose championship utility only inside the expected-score guardrail."""

    if metrics.empty:
        raise ValueError("Candidate metrics cannot be empty.")
    best_mean = float(metrics.construction_mean.max())
    tolerance = abs(best_mean) * float(tolerance_frac)
    eligible = metrics.loc[
        metrics.construction_mean.ge(best_mean - tolerance - 1e-9)
    ].copy()
    return eligible.sort_values(
        [
            "construction_championship_proxy",
            "construction_prob_two_difference_makers",
            "construction_mean",
            "roster_key",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def difference_maker_events(
    weekly_scores: np.ndarray,
    played_mask: np.ndarray,
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Return context-by-player q90 difference-maker indicators."""

    weekly_scores = np.asarray(weekly_scores, dtype=np.float64)
    played_mask = np.asarray(played_mask)
    if weekly_scores.ndim != 3 or weekly_scores.shape != played_mask.shape:
        raise ValueError("Difference-maker banks must align by context/player/week.")
    if weekly_scores.shape[1] != len(predictions):
        raise ValueError("Difference-maker player axis does not match predictions.")
    known_played = played_mask >= 0
    active = np.where(known_played, played_mask > 0, weekly_scores > 0.05)
    active_games = active.sum(axis=2)
    active_ppg = np.divide(
        weekly_scores.sum(axis=2),
        active_games,
        out=np.zeros_like(active_games, dtype=np.float64),
        where=active_games > 0,
    )
    projected = predictions.pred_fp_per_game.to_numpy(dtype=np.float64)
    residual_hit = active_ppg >= projected[None, :] + TAIL_RESIDUAL_STRIKE

    contributions = np.zeros_like(active_ppg, dtype=np.float64)
    contribution_hit = np.zeros_like(residual_hit, dtype=bool)
    positions = predictions.pos.to_numpy()
    for pos in ("QB", "RB", "WR", "TE"):
        pos_mask = positions == pos
        if not pos_mask.any():
            continue
        contribution = np.maximum(
            weekly_scores[:, pos_mask, :] - TAIL_WAIVER_BASELINES[pos],
            0.0,
        ).sum(axis=2)
        contributions[:, pos_mask] = contribution
        contribution_hit[:, pos_mask] = contribution >= float(thresholds[pos])
    return residual_hit & contribution_hit


def roster_difference_metrics(
    events: np.ndarray,
    roster_mask: np.ndarray,
) -> dict[str, float]:
    counts = np.asarray(events[:, roster_mask].sum(axis=1), dtype=np.float64)
    return {
        "expected_difference_makers": float(counts.mean()),
        "prob_one_difference_maker": float(np.mean(counts >= 1)),
        "prob_two_difference_makers": float(np.mean(counts >= 2)),
        "prob_three_difference_makers": float(np.mean(counts >= 3)),
    }


def load_tail_thresholds() -> dict[str, float]:
    if not TAIL_PREDICTIONS.exists():
        raise FileNotFoundError(TAIL_PREDICTIONS)
    frame = pd.read_csv(TAIL_PREDICTIONS)
    rows = frame.loc[
        frame.season.eq(YEAR) & frame.method.eq("production"),
        ["pos", "league_winner_contribution_q90"],
    ].drop_duplicates()
    if rows.pos.duplicated().any() or set(rows.pos) != {"QB", "RB", "WR", "TE"}:
        raise ValueError("2025 production q90 contribution thresholds are incomplete.")
    return rows.set_index("pos").league_winner_contribution_q90.astype(float).to_dict()


def load_player_context(conn: sqlite3.Connection) -> pd.DataFrame:
    context = pd.read_sql_query(
        """
        SELECT player_key, player, year_exp, avg_pick
        FROM Best_Ball_Weekly_Player_Map
        WHERE year=? AND version=? AND dataset=?
        """,
        conn,
        params=(YEAR, LEAGUE, PRED_VERSION),
    )
    if context.player_key.duplicated().any():
        raise ValueError("Historical replay player context is not unique by key.")
    if context.player.duplicated().any():
        raise ValueError("Historical replay player context is not unique by label.")
    return context


def prepare_state(*, block_count: int, construction_contexts: int, seed: int):
    if not SIMULATION_DB.exists():
        raise FileNotFoundError(
            f"Build the isolated 2025 replay before running this study: {SIMULATION_DB}"
        )
    conn = sqlite3.connect(SIMULATION_DB)
    sim = FootballSimulation(
        conn,
        YEAR,
        LINEUP_REQUIRE,
        SALARY_CAP,
        PRED_VERSION,
        LEAGUE,
        sal_pred_actual="_actual",
    )
    sim.load_weekly_template_profiles()
    keeper_market = load_active_keeper_market(
        conn,
        sim,
        year=YEAR,
        league=LEAGUE,
        salary_source="actual",
        owned_salary_map={},
    )
    unavailable = set(keeper_market["unavailable_keeper_players"])
    with sim.temp_seed(seed):
        canonical_predictions = sim.get_predictions(
            "pred_fp_per_game",
            num_options=512,
        )
    predictions, pool_summary = sequential.apply_sequential_draft_pool_filter(
        canonical_predictions.copy(),
        sequential._sequential_draft_pool_metadata(sim),
        LEAGUE,
        required_players=set(),
    )
    before_keeper_filter = len(predictions)
    predictions = predictions.loc[
        ~predictions.player.isin(unavailable)
    ].reset_index(drop=True)
    if len(predictions) != before_keeper_filter - keeper_market["keeper_count"]:
        raise ValueError("Historical keepers did not remove one draft-pool row each.")
    context = load_player_context(conn)
    projection_lookup = sim.player_data.set_index("player").pred_fp_per_game
    predictions["pred_fp_per_game"] = predictions.player.map(projection_lookup)
    if predictions.pred_fp_per_game.isna().any():
        raise ValueError("Draft-pool players are missing canonical PPG projections.")
    predictions = predictions.merge(
        context[["player", "year_exp", "avg_pick"]],
        on="player",
        how="left",
        validate="one_to_one",
    )
    if predictions.year_exp.isna().any():
        raise ValueError("Draft-pool players are missing experience context.")

    aligned = sequential._aligned_player_frame(sim, predictions)
    base_prices = aligned.salary.to_numpy(dtype=np.float64)
    predictions["salary"] = base_prices
    selection_premiums = np.zeros(len(predictions), dtype=np.float64)
    current_waivers = sim.estimate_waiver_baselines(
        num_teams=NUM_TEAMS,
        roster_size=ROSTER_SIZE,
    )
    churn_waivers = {
        pos: max(float(value), float(CHURN_WAIVER_FLOORS.get(pos, value)))
        for pos, value in current_waivers.items()
    }
    current_values, construction_banks = sequential._sample_construction_value_blocks(
        sim,
        canonical_predictions,
        predictions,
        [],
        block_count=block_count,
        contexts_per_block=construction_contexts,
        num_weeks=16,
        waiver_baselines=current_waivers,
        lineup_require=LINEUP_REQUIRE,
        learn_weeks=6,
        max_learn_weight=0.65,
        random_seed=seed + 100,
        return_contexts=True,
    )
    churn_values = []
    for bank in construction_banks:
        churn_values.append(
            sim.managed_marginal_values_multi_context_batch(
                bank["weekly_scores"],
                predictions.pos.to_numpy(),
                bank["decision_scores"],
                predictions.player.to_numpy(),
                [[]],
                waiver_baselines=churn_waivers,
                lineup_require=LINEUP_REQUIRE,
                played_mask=bank["played_mask"],
            )[0]
        )
    return {
        "conn": conn,
        "sim": sim,
        "canonical_predictions": canonical_predictions,
        "predictions": predictions,
        "base_prices": base_prices,
        "selection_premiums": selection_premiums,
        "current_waivers": current_waivers,
        "churn_waivers": churn_waivers,
        "current_values": np.stack(current_values),
        "churn_values": np.stack(churn_values),
        "construction_banks": construction_banks,
        "keeper_market": keeper_market,
        "pool_summary": {
            **pool_summary,
            "before_keeper_filter": int(before_keeper_filter),
            "after_keeper_filter": int(len(predictions)),
        },
        "tail_thresholds": load_tail_thresholds(),
    }


def solve_plan(state, managed_values, static_cache):
    predictions = state["predictions"]
    return sequential.solve_history_only_plan(
        state["sim"],
        predictions,
        managed_values,
        state["base_prices"],
        state["selection_premiums"],
        {},
        set(predictions.player),
        ROSTER_SIZE,
        POSITION_MIN,
        POSITION_MAX,
        REQUIRE_TOP_N,
        True,
        static_matrix_cache=static_cache,
        joint_refinement_max_swaps=0,
    )


def candidate_value_vectors(
    state,
    block_idx: int,
    waiver_mode: str,
    starts: int,
    contexts_per_start: int,
    seed: int,
):
    bank = state["construction_banks"][block_idx]
    waivers = state[f"{waiver_mode}_waivers"]
    values = state[f"{waiver_mode}_values"][block_idx]
    output = [("production", values)]
    context_count = bank["weekly_scores"].shape[0]
    subset_size = min(max(1, int(contexts_per_start)), context_count)
    rng = np.random.default_rng(seed)
    for start_idx in range(int(starts)):
        subset = np.sort(rng.choice(context_count, size=subset_size, replace=False))
        subset_values = state["sim"].managed_marginal_values_multi_context_batch(
            bank["weekly_scores"][subset],
            state["predictions"].pos.to_numpy(),
            bank["decision_scores"][subset],
            state["predictions"].player.to_numpy(),
            [[]],
            waiver_baselines=waivers,
            lineup_require=LINEUP_REQUIRE,
            played_mask=bank["played_mask"][subset],
        )[0]
        output.append((f"subset_{start_idx:02d}", subset_values))
    return output


def compile_candidates(
    state,
    block_idx: int,
    waiver_mode: str,
    starts: int,
    contexts_per_start: int,
    seed: int,
    static_cache: dict,
):
    plans = {}
    source_labels = {}
    for label, values in candidate_value_vectors(
        state,
        block_idx,
        waiver_mode,
        starts,
        contexts_per_start,
        seed,
    ):
        plan = solve_plan(state, values, static_cache)
        if plan is None:
            continue
        roster = tuple(sorted(plan["selected"]))
        if roster not in plans:
            plans[roster] = plan
            source_labels[roster] = []
        source_labels[roster].append(label)
    if not plans:
        raise RuntimeError(f"No feasible candidates for block {block_idx} {waiver_mode}.")
    return plans, source_labels


def score_candidate_set(
    state,
    block_idx: int,
    waiver_mode: str,
    plans: dict,
    source_labels: dict,
):
    bank = state["construction_banks"][block_idx]
    predictions = state["predictions"]
    waivers = state[f"{waiver_mode}_waivers"]
    cache = {}
    rosters = sorted(plans)
    score_matrix = np.stack([
        sequential._score_roster_bank(
            state["sim"],
            predictions,
            roster,
            bank["weekly_scores"],
            bank["decision_scores"],
            bank["played_mask"],
            LINEUP_REQUIRE,
            waivers,
            cache,
        )
        for roster in rosters
    ])
    events = difference_maker_events(
        bank["weekly_scores"],
        bank["played_mask"],
        predictions,
        state["tail_thresholds"],
    )
    players = predictions.player.to_numpy()
    rows = []
    for roster_idx, roster in enumerate(rosters):
        roster_mask = np.isin(players, roster)
        diff = roster_difference_metrics(events, roster_mask)
        rows.append({
            "block": block_idx,
            "waiver_mode": waiver_mode,
            "roster_key": " | ".join(roster),
            "candidate_sources": " | ".join(source_labels[roster]),
            "production_candidate": "production" in source_labels[roster],
            "construction_mean": float(score_matrix[roster_idx].mean()),
            "construction_p10": float(np.percentile(score_matrix[roster_idx], 10)),
            "construction_p90": float(np.percentile(score_matrix[roster_idx], 90)),
            "construction_championship_proxy": empirical_championship_proxy(
                score_matrix[roster_idx],
                score_matrix,
            ),
            "construction_expected_difference_makers": diff[
                "expected_difference_makers"
            ],
            "construction_prob_one_difference_maker": diff[
                "prob_one_difference_maker"
            ],
            "construction_prob_two_difference_makers": diff[
                "prob_two_difference_makers"
            ],
            "construction_prob_three_difference_makers": diff[
                "prob_three_difference_makers"
            ],
            "forecast_spend": float(sum(
                plans[roster]["forecast_cost"][player] for player in roster
            )),
        })
    return pd.DataFrame(rows), plans


def select_arm_rosters(current_metrics, current_plans, churn_metrics, churn_plans):
    current_production = current_metrics.loc[current_metrics.production_candidate]
    churn_production = churn_metrics.loc[churn_metrics.production_candidate]
    if len(current_production) != 1 or len(churn_production) != 1:
        raise ValueError("Each waiver mode requires one production candidate.")
    selected_rows = {
        "baseline": current_production.iloc[0],
        "waiver_proxy": churn_production.iloc[0],
        "championship_tiebreak": choose_lexicographic_candidate(current_metrics),
        "combined": choose_lexicographic_candidate(churn_metrics),
    }
    selected = {}
    for arm, row in selected_rows.items():
        roster = tuple(row.roster_key.split(" | "))
        plan_lookup = churn_plans if arm in {"waiver_proxy", "combined"} else current_plans
        selected[arm] = {"row": row, "roster": roster, "plan": plan_lookup[roster]}
    return selected


def score_validation_block(
    state,
    selected,
    candidate_rosters,
    validation_bank,
    block_idx: int,
):
    predictions = state["predictions"]
    players = predictions.player.to_numpy()
    caches = {"current": {}, "churn": {}}
    all_rosters = sorted(set(candidate_rosters) | {value["roster"] for value in selected.values()})
    reference = {}
    for mode in ("current", "churn"):
        reference[mode] = np.stack([
            sequential._score_roster_bank(
                state["sim"],
                predictions,
                roster,
                *validation_bank,
                LINEUP_REQUIRE,
                state[f"{mode}_waivers"],
                caches[mode],
            )
            for roster in all_rosters
        ])
    events = difference_maker_events(
        validation_bank[0],
        validation_bank[2],
        predictions,
        state["tail_thresholds"],
    )
    rows = []
    score_rows = []
    for arm, choice in selected.items():
        roster = choice["roster"]
        roster_mask = np.isin(players, roster)
        diff = roster_difference_metrics(events, roster_mask)
        scores_by_mode = {}
        for mode in ("current", "churn"):
            scores = sequential._score_roster_bank(
                state["sim"],
                predictions,
                roster,
                *validation_bank,
                LINEUP_REQUIRE,
                state[f"{mode}_waivers"],
                caches[mode],
            )
            scores_by_mode[mode] = scores
            for context_idx, score in enumerate(scores):
                score_rows.append({
                    "block": block_idx,
                    "arm": arm,
                    "scoring_waiver_mode": mode,
                    "context": context_idx,
                    "managed_season_score": float(score),
                })
        counts = Counter(predictions.loc[roster_mask, "pos"])
        rb_experience = predictions.loc[roster_mask & predictions.pos.eq("RB"), "year_exp"]
        rows.append({
            "block": block_idx,
            "arm": arm,
            "roster": " | ".join(roster),
            "forecast_spend": float(choice["row"].forecast_spend),
            "holdout_mean_current": float(scores_by_mode["current"].mean()),
            "holdout_p10_current": float(np.percentile(scores_by_mode["current"], 10)),
            "holdout_p90_current": float(np.percentile(scores_by_mode["current"], 90)),
            "holdout_mean_churn": float(scores_by_mode["churn"].mean()),
            "holdout_p10_churn": float(np.percentile(scores_by_mode["churn"], 10)),
            "holdout_p90_churn": float(np.percentile(scores_by_mode["churn"], 90)),
            "holdout_championship_proxy_current": empirical_championship_proxy(
                scores_by_mode["current"], reference["current"]
            ),
            "holdout_championship_proxy_churn": empirical_championship_proxy(
                scores_by_mode["churn"], reference["churn"]
            ),
            "holdout_expected_difference_makers": diff["expected_difference_makers"],
            "holdout_prob_one_difference_maker": diff["prob_one_difference_maker"],
            "holdout_prob_two_difference_makers": diff["prob_two_difference_makers"],
            "holdout_prob_three_difference_makers": diff["prob_three_difference_makers"],
            "dead_zone_rb_count": int(len(set(roster) & DEAD_ZONE_RBS)),
            "rb_mean_year_exp": float(rb_experience.mean()),
            "rookie_rb_count": int((rb_experience == 0).sum()),
            **{f"count_{pos.lower()}": int(counts.get(pos, 0)) for pos in ("QB", "RB", "WR", "TE")},
        })
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def normalize_player_name(value: str) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def load_actual_2025_bank(state):
    """Load held-out 2025 weekly results after policy construction is complete."""

    from Scripts.Modeling import s4_Best_Ball_Weekly as weekly_builder

    weekly = weekly_builder.load_weekly_points(YEAR, league=LEAGUE)
    weekly = weekly.loc[
        weekly.season.eq(YEAR) & weekly.week.between(1, 16)
    ].copy()
    weekly["name_key"] = weekly.player.map(normalize_player_name)
    predictions = state["predictions"].copy()
    predictions["name_key"] = predictions.player.map(normalize_player_name)
    prediction_lookup = predictions.set_index(["name_key", "pos"])
    if prediction_lookup.index.duplicated().any():
        raise ValueError("Actual weekly mapping is ambiguous in the draft population.")
    weekly = weekly.loc[
        weekly.set_index(["name_key", "pos"]).index.isin(prediction_lookup.index)
    ].copy()
    weekly = (
        weekly.groupby(["name_key", "pos", "week"], as_index=False)
        .agg(
            managed_fantasy_pts=("managed_fantasy_pts", "sum"),
            played_week=("played_week", "max"),
        )
    )
    player_count = len(predictions)
    scores = np.zeros((player_count, 16), dtype=np.float32)
    played = np.zeros((player_count, 16), dtype=np.int8)
    index_lookup = {
        (row.name_key, row.pos): idx
        for idx, row in predictions.reset_index(drop=True).iterrows()
    }
    for row in weekly.itertuples(index=False):
        idx = index_lookup[(row.name_key, row.pos)]
        week_idx = int(row.week) - 1
        scores[idx, week_idx] = (
            0.0 if pd.isna(row.managed_fantasy_pts) else float(row.managed_fantasy_pts)
        )
        played[idx, week_idx] = int(bool(row.played_week))
    decisions = state["sim"].build_managed_decision_scores(
        scores,
        preseason_ppg=predictions.pred_fp_per_game.to_numpy(dtype=np.float64),
        learn_weeks=6,
        max_learn_weight=0.65,
        played_mask=played,
    )
    return (
        scores[None, :, :],
        decisions[None, :, :],
        played[None, :, :],
    ), {
        "players_with_appearances": int((played.sum(axis=1) > 0).sum()),
        "draft_pool_players": int(player_count),
        "weekly_rows": int(len(weekly)),
    }


def score_actual_outcomes(state, plan_rows, actual_bank):
    predictions = state["predictions"]
    players = predictions.player.to_numpy()
    unique = plan_rows[["block", "arm", "roster"]].drop_duplicates()
    all_rosters = sorted({tuple(value.split(" | ")) for value in unique.roster})
    reference_by_mode = {}
    cache_by_mode = {"current": {}, "churn": {}}
    for mode in ("current", "churn"):
        reference_by_mode[mode] = np.stack([
            sequential._score_roster_bank(
                state["sim"],
                predictions,
                roster,
                *actual_bank,
                LINEUP_REQUIRE,
                state[f"{mode}_waivers"],
                cache_by_mode[mode],
            )
            for roster in all_rosters
        ])
    actual_events = difference_maker_events(
        actual_bank[0],
        actual_bank[2],
        predictions,
        state["tail_thresholds"],
    )
    rows = []
    for row in unique.itertuples(index=False):
        roster = tuple(row.roster.split(" | "))
        roster_mask = np.isin(players, roster)
        diff = roster_difference_metrics(actual_events, roster_mask)
        actual_difference_makers = sorted(
            predictions.loc[roster_mask & actual_events[0], "player"].tolist()
        )
        record = {
            "block": int(row.block),
            "arm": row.arm,
            "actual_difference_maker_count": diff["expected_difference_makers"],
            "actual_difference_makers": " | ".join(actual_difference_makers),
        }
        for mode in ("current", "churn"):
            scores = sequential._score_roster_bank(
                state["sim"],
                predictions,
                roster,
                *actual_bank,
                LINEUP_REQUIRE,
                state[f"{mode}_waivers"],
                cache_by_mode[mode],
            )
            record[f"actual_managed_score_{mode}"] = float(scores[0])
            record[f"actual_championship_proxy_{mode}"] = empirical_championship_proxy(
                scores,
                reference_by_mode[mode],
            )
        rows.append(record)
    return pd.DataFrame(rows)


def paired_lcb(delta: pd.Series) -> float:
    values = delta.to_numpy(dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    return float(values.mean() - LCB80_Z * values.std(ddof=1) / np.sqrt(len(values)))


def summarize(plan_rows: pd.DataFrame, actual_rows: pd.DataFrame):
    merged = plan_rows.merge(
        actual_rows,
        on=["block", "arm"],
        how="left",
        validate="one_to_one",
    )
    metrics = [
        "holdout_mean_current",
        "holdout_p10_current",
        "holdout_p90_current",
        "holdout_mean_churn",
        "holdout_p10_churn",
        "holdout_p90_churn",
        "holdout_championship_proxy_current",
        "holdout_championship_proxy_churn",
        "holdout_expected_difference_makers",
        "holdout_prob_two_difference_makers",
        "dead_zone_rb_count",
        "rb_mean_year_exp",
        "rookie_rb_count",
        "actual_managed_score_current",
        "actual_managed_score_churn",
        "actual_championship_proxy_churn",
        "actual_difference_maker_count",
    ]
    summary = merged.groupby("arm", as_index=False)[metrics].mean()
    baseline = merged.loc[merged.arm.eq("baseline")].set_index("block")
    paired_rows = []
    for arm in ARMS:
        arm_rows = merged.loc[merged.arm.eq(arm)].set_index("block")
        record = {"arm": arm, "blocks": int(len(arm_rows))}
        for metric in metrics:
            delta = arm_rows[metric] - baseline[metric]
            record[f"{metric}_delta"] = float(delta.mean())
            record[f"{metric}_delta_lcb80"] = paired_lcb(delta)
        paired_rows.append(record)
    return merged, summary, pd.DataFrame(paired_rows)


def write_summary(summary, paired, frequencies, metadata):
    labels = {
        "baseline": "Baseline",
        "waiver_proxy": "Waiver proxy",
        "championship_tiebreak": "Championship tie-break",
        "combined": "Combined",
    }
    lines = [
        "# Paired 2025 Auction Objective Results",
        "",
        (
            f"Eight blocks use current waiver estimates `{metadata['current_waivers']}` "
            f"and the churn proxy `{metadata['churn_waivers']}`. All selection "
            "evidence uses donors through 2024; actual 2025 weekly results are holdout only."
        ),
        "",
        "## Validation summary",
        "",
        "| Arm | Managed EV, churn | Championship proxy, churn | P(2+ difference-makers) | Dead-zone RBs | Rookie RBs | Actual 2025 score, churn | Actual difference-makers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.set_index("arm").loc[list(ARMS)].itertuples():
        lines.append(
            f"| {labels[row.Index]} | {row.holdout_mean_churn:.2f} | "
            f"{row.holdout_championship_proxy_churn:.3%} | "
            f"{row.holdout_prob_two_difference_makers:.2%} | "
            f"{row.dead_zone_rb_count:.2f} | {row.rookie_rb_count:.2f} | "
            f"{row.actual_managed_score_churn:.2f} | "
            f"{row.actual_difference_maker_count:.2f} |"
        )
    lines.extend([
        "",
        "## Paired deltas versus baseline",
        "",
        "| Arm | EV delta | EV LCB80 | Championship delta | Championship LCB80 | P(2+) delta | Dead-zone RB delta | Actual score delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in paired.set_index("arm").loc[list(ARMS)].itertuples():
        lines.append(
            f"| {labels[row.Index]} | {row.holdout_mean_churn_delta:+.2f} | "
            f"{row.holdout_mean_churn_delta_lcb80:+.2f} | "
            f"{row.holdout_championship_proxy_churn_delta:+.3%} | "
            f"{row.holdout_championship_proxy_churn_delta_lcb80:+.3%} | "
            f"{row.holdout_prob_two_difference_makers_delta:+.2%} | "
            f"{row.dead_zone_rb_count_delta:+.2f} | "
            f"{row.actual_managed_score_churn_delta:+.2f} |"
        )
    lines.extend([
        "",
        "## Player movement",
        "",
        "The most frequently selected RBs by arm are:",
        "",
    ])
    for arm in ARMS:
        rows = frequencies.loc[
            frequencies.arm.eq(arm) & frequencies.pos.eq("RB")
        ].nlargest(10, "selection_rate")
        values = ", ".join(
            f"{row.player} ({row.selection_rate:.0%})"
            for row in rows.itertuples(index=False)
        )
        lines.append(f"- **{labels[arm]}:** {values}")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- The championship value is a common-bank relative proxy, not an absolute calibrated league-win probability.",
        "- The churn arm is a frozen best-available PPG sensitivity, not a complete transaction, learning, or waiver-competition model.",
        "- A single 2025 realized season can diagnose behavior but cannot justify production promotion by itself.",
        "- The q90 difference-maker event is position- and history-aware; it does not directly reward youth.",
    ])
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(args):
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = prepare_state(
        block_count=args.blocks,
        construction_contexts=args.construction_contexts,
        seed=args.seed,
    )
    candidate_frames = []
    plan_frames = []
    score_frames = []
    roster_rows = []
    static_cache = {}
    validation_seeds = np.random.SeedSequence(args.seed + 200).spawn(args.blocks)
    all_selected = {}
    try:
        for block_idx in range(args.blocks):
            mode_outputs = {}
            for mode_idx, mode in enumerate(("current", "churn")):
                plans, sources = compile_candidates(
                    state,
                    block_idx,
                    mode,
                    args.candidate_starts,
                    args.contexts_per_start,
                    args.seed + 1000 * (block_idx + 1) + 100 * mode_idx,
                    static_cache,
                )
                metrics, plans = score_candidate_set(
                    state,
                    block_idx,
                    mode,
                    plans,
                    sources,
                )
                candidate_frames.append(metrics)
                mode_outputs[mode] = (metrics, plans)
            selected = select_arm_rosters(
                mode_outputs["current"][0],
                mode_outputs["current"][1],
                mode_outputs["churn"][0],
                mode_outputs["churn"][1],
            )
            all_selected[block_idx] = selected
            candidate_rosters = set(mode_outputs["current"][1]) | set(
                mode_outputs["churn"][1]
            )
            validation_seed = int(
                validation_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
            )
            validation_bank = sequential._sample_validation_bank(
                state["sim"],
                state["predictions"],
                args.validation_contexts,
                16,
                6,
                0.65,
                validation_seed,
                canonical_predictions=state["canonical_predictions"],
            )
            plan_frame, score_frame = score_validation_block(
                state,
                selected,
                candidate_rosters,
                validation_bank,
                block_idx,
            )
            plan_frames.append(plan_frame)
            score_frames.append(score_frame)
            position_map = state["predictions"].set_index("player").pos.to_dict()
            experience_map = state["predictions"].set_index("player").year_exp.to_dict()
            for arm, choice in selected.items():
                for player in choice["roster"]:
                    roster_rows.append({
                        "block": block_idx,
                        "arm": arm,
                        "player": player,
                        "pos": position_map[player],
                        "year_exp": float(experience_map[player]),
                        "forecast_cost": float(choice["plan"]["forecast_cost"][player]),
                    })

        candidate_metrics = pd.concat(candidate_frames, ignore_index=True)
        plan_rows = pd.concat(plan_frames, ignore_index=True)
        score_rows = pd.concat(score_frames, ignore_index=True)
        roster_players = pd.DataFrame(roster_rows)

        actual_bank, actual_coverage = load_actual_2025_bank(state)
        actual_rows = score_actual_outcomes(state, plan_rows, actual_bank)
        merged, summary, paired = summarize(plan_rows, actual_rows)
        frequencies = (
            roster_players.groupby(["arm", "player", "pos", "year_exp"], as_index=False)
            .agg(blocks_selected=("block", "nunique"))
        )
        frequencies["selection_rate"] = frequencies.blocks_selected / args.blocks
        metadata = {
            "year": YEAR,
            "league": LEAGUE,
            "salary_source": "actual",
            "simulation_database": str(SIMULATION_DB),
            "current_waivers": state["current_waivers"],
            "churn_waivers": state["churn_waivers"],
            "tail_thresholds": state["tail_thresholds"],
            "tail_residual_strike": TAIL_RESIDUAL_STRIKE,
            "mean_noninferiority_frac": MEAN_NONINFERIORITY_FRAC,
            "blocks": args.blocks,
            "construction_contexts_per_block": args.construction_contexts,
            "candidate_starts_per_mode_block": args.candidate_starts,
            "contexts_per_candidate_start": args.contexts_per_start,
            "validation_contexts_per_block": args.validation_contexts,
            "seed": args.seed,
            "keeper_market": state["keeper_market"],
            "pool_summary": state["pool_summary"],
            "actual_weekly_coverage": actual_coverage,
            "runtime_seconds": time.perf_counter() - started,
            "production_changed": False,
        }
        candidate_metrics.to_csv(RESULTS_DIR / "candidate_rosters.csv", index=False)
        merged.to_csv(RESULTS_DIR / "plan_blocks.csv", index=False)
        score_rows.to_csv(RESULTS_DIR / "holdout_score_cells.csv", index=False)
        roster_players.to_csv(RESULTS_DIR / "roster_players.csv", index=False)
        frequencies.to_csv(RESULTS_DIR / "player_frequency.csv", index=False)
        summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
        paired.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)
        actual_rows.to_csv(RESULTS_DIR / "actual_2025_outcomes.csv", index=False)
        (RESULTS_DIR / "metadata.json").write_text(
            json.dumps(json_value(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_summary(summary, paired, frequencies, metadata)
        print("\nSUMMARY")
        print(summary.to_string(index=False))
        print("\nPAIRED")
        print(paired.to_string(index=False))
        print("\nRuntime seconds", metadata["runtime_seconds"])
        return summary, paired
    finally:
        state["conn"].close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--construction-contexts", type=int, default=32)
    parser.add_argument("--candidate-starts", type=int, default=24)
    parser.add_argument("--contexts-per-start", type=int, default=8)
    parser.add_argument("--validation-contexts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


if __name__ == "__main__":
    run_study(parse_args())
