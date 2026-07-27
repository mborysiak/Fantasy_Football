"""Rolling-origin replay of bench call-option and waiver-hurdle policies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
BASE_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-13_managed_auction_rolling_replay"
    / "run_replay.py"
)

spec = importlib.util.spec_from_file_location("managed_auction_base_replay", BASE_STUDY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load base replay module: {BASE_STUDY}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

FootballSimulation = base.FootballSimulation
POSITIONS = base.POSITIONS
LINEUP_REQUIRE = base.LINEUP_REQUIRE
POS_MIN = base.POS_MIN
POS_MAX = base.POS_MAX
ROSTER_SIZE = base.ROSTER_SIZE
SALARY_CAP = base.SALARY_CAP
NUM_TEAMS = base.NUM_TEAMS
TOTAL_MARKET_BUDGET = base.TOTAL_MARKET_BUDGET
TOTAL_MARKET_SLOTS = base.TOTAL_MARKET_SLOTS
TOP_N = base.TOP_N


@dataclass(frozen=True)
class Policy:
    name: str
    hurdle_delta: float
    bench_weight: float
    option_lambda: float


POLICIES = (
    Policy("current_bench025", 0.0, 0.25, 0.0),
    Policy("hurdle_plus1", 1.0, 0.25, 0.0),
    Policy("hurdle_plus2", 2.0, 0.25, 0.0),
    Policy("hurdle_plus3", 3.0, 0.25, 0.0),
    Policy("bench0", 0.0, 0.0, 0.0),
    Policy("sustained_option025", 0.0, 0.0, 0.25),
    Policy("sustained_option050", 0.0, 0.0, 0.50),
)
BASELINE_POLICY = "current_bench025"
PLAYOFF_WEEK_START = 12  # zero-based week 13 in the 16-week replay
BREAKOUT_THRESHOLDS = (12.0, 15.0)


def construction_waiver(
    baseline: dict[str, float],
    hurdle_delta: float,
) -> dict[str, float]:
    """Raise only bench-eligible positions; QB has a one-player roster maximum."""
    return {
        pos: float(value + (hurdle_delta if pos in {"RB", "WR", "TE"} else 0.0))
        for pos, value in baseline.items()
    }


def league_starter_thresholds(
    full_forecast: pd.DataFrame,
    waiver_baseline: dict[str, float],
    clearance_ppg: float = 2.0,
) -> dict[str, float]:
    """Estimate a league-average impact hurdle from direct and FLEX starters."""
    frame = full_forecast[["player", "pos", "pred_fp_per_game"]].copy()
    frame = frame.sort_values(
        ["pred_fp_per_game", "player"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected: set[int] = set()
    for pos in POSITIONS:
        direct_count = int(NUM_TEAMS * LINEUP_REQUIRE.get(pos, 0))
        selected.update(frame[frame.pos.eq(pos)].head(direct_count).index.tolist())
    flex_count = int(NUM_TEAMS * LINEUP_REQUIRE.get("FLEX", 0))
    flex = frame[
        frame.pos.isin(["RB", "WR", "TE"]) & ~frame.index.isin(selected)
    ].head(flex_count)
    selected.update(flex.index.tolist())

    thresholds: dict[str, float] = {}
    selected_frame = frame.loc[sorted(selected)] if selected else frame.iloc[0:0]
    for pos in POSITIONS:
        pos_starters = selected_frame[selected_frame.pos.eq(pos)]
        starter_floor = (
            float(pos_starters.pred_fp_per_game.min())
            if len(pos_starters)
            else float(waiver_baseline[pos])
        )
        thresholds[pos] = float(
            max(
                waiver_baseline[pos] + clearance_ppg,
                starter_floor + clearance_ppg,
            )
        )
    return thresholds


def sustained_option_bank(
    weekly: np.ndarray,
    played: np.ndarray,
    predictions: pd.DataFrame,
    impact_thresholds: dict[str, float],
    lookback: int = 3,
    minimum_prior_games: int = 2,
    validation_window: int = 4,
    minimum_validation_games: int = 3,
) -> np.ndarray:
    """Expected post-detection impact from sustained simulated breakouts.

    Detection at week ``t`` uses only weeks before ``t``. Future weeks determine
    the payoff of that draft-time scenario, just as future holdouts score any
    other preseason decision. Players already above the impact hurdle in the
    preseason mean receive no bench-option bonus.
    """
    weekly = np.asarray(weekly, dtype=np.float32)
    played = np.asarray(played)
    if weekly.ndim != 3 or weekly.shape != played.shape:
        raise ValueError("Weekly and played option banks must align by context/player/week.")
    preseason = predictions[FootballSimulation.sample_value_columns(predictions)].mean(
        axis=1
    ).to_numpy(dtype=float)
    positions = predictions.pos.to_numpy()
    result = np.zeros((weekly.shape[1], weekly.shape[0]), dtype=np.float32)

    for context_idx in range(weekly.shape[0]):
        for player_idx, pos in enumerate(positions):
            if pos == "QB":
                continue
            threshold = float(impact_thresholds[pos])
            if preseason[player_idx] >= threshold:
                continue
            scores = weekly[context_idx, player_idx]
            available = played[context_idx, player_idx] > 0
            for week_idx in range(lookback, weekly.shape[2]):
                prior_slice = slice(max(0, week_idx - lookback), week_idx)
                prior_available = available[prior_slice]
                if int(prior_available.sum()) < minimum_prior_games:
                    continue
                prior_scores = scores[prior_slice][prior_available]
                if float(prior_scores.mean()) < threshold:
                    continue

                validation_end = min(weekly.shape[2], week_idx + validation_window)
                validation_available = available[week_idx:validation_end]
                if int(validation_available.sum()) < minimum_validation_games:
                    continue
                validation_scores = scores[week_idx:validation_end][validation_available]
                if float(validation_scores.mean()) < threshold:
                    continue

                future_scores = scores[week_idx:]
                future_available = available[week_idx:]
                result[player_idx, context_idx] = float(
                    np.maximum(future_scores[future_available] - threshold, 0.0).sum()
                )
                break
    return result


def policy_value_banks(
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    predictions: pd.DataFrame,
    current_waiver: dict[str, float],
    option_bank: np.ndarray,
) -> dict[str, np.ndarray]:
    cache: dict[tuple[float, float], np.ndarray] = {}
    values: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        cache_key = (policy.hurdle_delta, policy.bench_weight)
        if cache_key not in cache:
            waiver = construction_waiver(current_waiver, policy.hurdle_delta)
            samples = []
            for context_idx in range(len(weekly)):
                samples.append(
                    FootballSimulation.managed_marginal_values(
                        weekly[context_idx],
                        predictions.pos.to_numpy(),
                        decisions[context_idx],
                        predictions.player.to_numpy(),
                        base_players=[],
                        waiver_baselines=waiver,
                        lineup_require=LINEUP_REQUIRE,
                        bench_upside_weight=policy.bench_weight,
                        played_mask=played[context_idx],
                    )
                )
            cache[cache_key] = np.column_stack(samples).astype(np.float32)
        values[policy.name] = (
            cache[cache_key] + policy.option_lambda * option_bank
        ).astype(np.float32)
    return values


def exact_reference_score(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    weekly_scores: np.ndarray,
    decision_scores: np.ndarray,
    played_mask: np.ndarray,
    waiver_baseline: dict[str, float],
) -> float:
    return float(
        FootballSimulation.managed_lineup_weekly_scores(
            weekly_scores[selected_mask],
            predictions.loc[selected_mask, "pos"].to_numpy(),
            decision_scores=decision_scores[selected_mask],
            player_names=predictions.loc[selected_mask, "player"].to_numpy(),
            lineup_require=LINEUP_REQUIRE,
            waiver_baselines=waiver_baseline,
            played_mask=played_mask[selected_mask],
        ).sum()
    )


def refine_policy_roster(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    weekly_scores: np.ndarray,
    decision_scores: np.ndarray,
    played_mask: np.ndarray,
    salary_values: np.ndarray,
    top_n: list[str],
    waiver_baseline: dict[str, float],
    option_values: np.ndarray,
    option_lambda: float,
    max_swaps: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Converge on exact reference score plus an additive option utility."""
    selected_mask = np.asarray(selected_mask, dtype=bool).copy()
    salary_values = np.asarray(salary_values, dtype=float)
    option_values = np.asarray(option_values, dtype=float)
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    top_n_set = set(top_n)
    accepted = 0

    for _ in range(max_swaps):
        current_lineup = exact_reference_score(
            predictions,
            selected_mask,
            weekly_scores,
            decision_scores,
            played_mask,
            waiver_baseline,
        )
        current_option = float(option_values[selected_mask].sum())
        current_objective = current_lineup + option_lambda * current_option
        out_indices = np.flatnonzero(selected_mask)
        base_masks = []
        for out_idx in out_indices:
            base_mask = selected_mask.copy()
            base_mask[out_idx] = False
            base_masks.append(base_mask)
        incoming_rows = FootballSimulation.managed_marginal_values_batch(
            weekly_scores,
            positions,
            decision_scores,
            players,
            [players[base_mask].tolist() for base_mask in base_masks],
            waiver_baselines=waiver_baseline,
            lineup_require=LINEUP_REQUIRE,
            bench_upside_weight=0.0,
            played_mask=played_mask,
        )

        best: tuple[float, int, int] | None = None
        for out_idx, base_mask, incoming in zip(out_indices, base_masks, incoming_rows):
            base_salary = float(salary_values[base_mask].sum())
            base_counts = {
                pos: int(np.sum(positions[base_mask] == pos)) for pos in POSITIONS
            }
            eligible = ~selected_mask
            eligible &= base_salary + salary_values <= SALARY_CAP + 1e-8
            allowed_positions = [
                candidate_pos
                for candidate_pos in POSITIONS
                if all(
                    int(POS_MIN[pos])
                    <= base_counts[pos] + int(pos == candidate_pos)
                    <= int(POS_MAX[pos])
                    for pos in POSITIONS
                )
            ]
            eligible &= np.isin(positions, allowed_positions)
            if not any(player in top_n_set for player in players[base_mask]):
                eligible &= np.isin(players, list(top_n_set))
            eligible_idx = np.flatnonzero(eligible & np.isfinite(incoming))
            if len(eligible_idx) == 0:
                continue

            estimated_lineup = current_lineup - float(incoming[out_idx]) + incoming[
                eligible_idx
            ]
            estimated_option = (
                current_option
                - float(option_values[out_idx])
                + option_values[eligible_idx]
            )
            estimated_objective = estimated_lineup + option_lambda * estimated_option
            local = int(np.argmax(estimated_objective))
            candidate = (
                float(estimated_objective[local]),
                int(out_idx),
                int(eligible_idx[local]),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None or best[0] <= current_objective + 1e-5:
            break
        _, out_idx, in_idx = best
        replacement = selected_mask.copy()
        replacement[out_idx] = False
        replacement[in_idx] = True
        exact_replacement = exact_reference_score(
            predictions,
            replacement,
            weekly_scores,
            decision_scores,
            played_mask,
            waiver_baseline,
        ) + option_lambda * float(option_values[replacement].sum())
        if exact_replacement <= current_objective + 1e-5:
            break
        selected_mask = replacement
        accepted += 1

    return selected_mask, {"accepted_swaps": accepted}


def nominal_starter_mask(predictions: pd.DataFrame, roster_mask: np.ndarray) -> np.ndarray:
    frame = predictions.loc[roster_mask].copy()
    value_columns = FootballSimulation.sample_value_columns(predictions)
    frame["_ppg"] = predictions.loc[roster_mask, value_columns].mean(axis=1)
    remaining = set(frame.index)
    starters: list[int] = []
    for pos in POSITIONS:
        count = int(LINEUP_REQUIRE.get(pos, 0))
        chosen = (
            frame.loc[sorted(remaining)]
            .loc[lambda value: value.pos.eq(pos)]
            .sort_values(["_ppg", "player"], ascending=[False, True])
            .head(count)
            .index.tolist()
        )
        starters.extend(chosen)
        remaining -= set(chosen)
    flex_count = int(LINEUP_REQUIRE.get("FLEX", 0))
    chosen = (
        frame.loc[sorted(remaining)]
        .loc[lambda value: value.pos.isin(["RB", "WR", "TE"])]
        .sort_values(["_ppg", "player"], ascending=[False, True])
        .head(flex_count)
        .index.tolist()
    )
    starters.extend(chosen)
    result = np.zeros(len(predictions), dtype=bool)
    result[starters] = True
    return result


def max_active_window_ppg(
    scores: np.ndarray,
    played: np.ndarray,
    window: int = 4,
    minimum_games: int = 3,
) -> float:
    best = 0.0
    for start in range(0, len(scores) - window + 1):
        available = played[start : start + window] > 0
        if int(available.sum()) < minimum_games:
            continue
        best = max(best, float(scores[start : start + window][available].mean()))
    return best


def actual_policy_metrics(
    environment: dict[str, Any],
    roster: tuple[str, ...],
    bench_players: tuple[str, ...],
) -> dict[str, Any]:
    base_score = base.score_actual_roster(environment, roster)
    labels = environment["labels"]
    roster_mask = labels.player.isin(roster).to_numpy()
    roster_keys = set(labels.loc[roster_mask, "player_key"])
    waiver_scores, waiver_decisions, waiver_played, waiver_pos, waiver_names = (
        base.dynamic_waiver_slots(environment["waiver_pool"], roster_keys)
    )
    weekly = FootballSimulation.managed_lineup_weekly_scores(
        np.vstack([environment["scores"][roster_mask], waiver_scores]),
        np.concatenate([labels.loc[roster_mask, "pos"].to_numpy(), waiver_pos]),
        decision_scores=np.vstack(
            [environment["decisions"][roster_mask], waiver_decisions]
        ),
        player_names=np.concatenate(
            [labels.loc[roster_mask, "player"].to_numpy(), waiver_names]
        ),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=base.ZERO_WAIVERS,
        played_mask=np.vstack([environment["played"][roster_mask], waiver_played]),
    )

    bench_mask = labels.player.isin(bench_players).to_numpy()
    max_windows = [
        max_active_window_ppg(
            environment["scores"][idx],
            environment["played"][idx],
        )
        for idx in np.flatnonzero(bench_mask)
    ]
    metrics: dict[str, Any] = {
        **base_score,
        "actual_playoff_points": float(np.asarray(weekly)[PLAYOFF_WEEK_START:].sum()),
        "bench_actual_max4_ppg": float(np.mean(max_windows)) if max_windows else 0.0,
        "bench_dead_below10": int(sum(value < 10.0 for value in max_windows)),
    }
    for threshold in BREAKOUT_THRESHOLDS:
        metrics[f"bench_sustained_{int(threshold)}_hits"] = int(
            sum(value >= threshold for value in max_windows)
        )
    return metrics


def forecast_metrics(
    predictions: pd.DataFrame,
    roster: tuple[str, ...],
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    waiver_baseline: dict[str, float],
    cache: dict[tuple[str, ...], dict[str, float]],
) -> dict[str, float]:
    key = tuple(sorted(roster))
    if key in cache:
        return cache[key]
    mask = predictions.player.isin(roster).to_numpy()
    scores, _ = FootballSimulation.managed_lineup_multi_context_scores(
        weekly[:, mask, :],
        predictions.loc[mask, "pos"].to_numpy(),
        decisions[:, mask, :],
        predictions.loc[mask, "player"].to_numpy(),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=waiver_baseline,
        played_mask=played[:, mask, :],
    )
    result = {
        "forecast_ev": float(np.mean(scores)),
        "forecast_p10": float(np.percentile(scores, 10)),
        "forecast_p90": float(np.percentile(scores, 90)),
    }
    cache[key] = result
    return result


def run_year_trials(
    year: int,
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    salary_draws: np.ndarray,
    environment: dict[str, Any],
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    evaluation_weekly: np.ndarray,
    evaluation_decisions: np.ndarray,
    evaluation_played: np.ndarray,
    value_banks: dict[str, np.ndarray],
    option_bank: np.ndarray,
    current_waiver: dict[str, float],
    trials: int,
    context_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    remaining_budget = TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = TOTAL_MARKET_SLOTS - environment["keeper_count"]
    top_n = predictions.nlargest(min(TOP_N, len(predictions)), "salary").player.tolist()
    static = sim.build_managed_ilp_static_matrices(
        predictions,
        {},
        [],
        top_n,
        ROSTER_SIZE,
        POS_MIN,
        POS_MAX,
        enforce_top_n=True,
    )
    ref_weekly = weekly.mean(axis=0)
    ref_decisions = decisions.mean(axis=0)
    ref_played = np.where(
        np.any(played >= 0, axis=0),
        np.any(played > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)
    ref_option = option_bank.mean(axis=1)

    rng = np.random.default_rng(seed + year * 101)
    salary_plan = rng.integers(0, salary_draws.shape[1], size=(trials, 5))
    context_plan = rng.integers(0, weekly.shape[0], size=(trials, context_draws))
    raw_market = np.column_stack(
        [salary_draws[:, row].mean(axis=1) for row in salary_plan]
    )
    markets = base.normalize_market_draws(
        sim,
        raw_market,
        remaining_budget,
        remaining_slots,
    )
    forecast_cache: dict[tuple[str, ...], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for trial in range(trials):
        context_idx = context_plan[trial]
        market = markets[:, trial]
        predictions["salary"] = market
        for policy in POLICIES:
            values = value_banks[policy.name][:, context_idx].mean(axis=1)
            solved = sim._solve_managed_scenario(
                predictions,
                values,
                ref_weekly,
                ref_decisions,
                static,
                [],
                {},
                top_n,
                ROSTER_SIZE,
                POS_MIN,
                POS_MAX,
                construction_waiver(current_waiver, policy.hurdle_delta),
                LINEUP_REQUIRE,
                True,
                refine_roster=False,
                score_roster=False,
                salary_values=market,
                played_mask=ref_played,
            )
            if solved is None:
                rows.append(
                    {
                        "year": year,
                        "trial": trial,
                        "policy": policy.name,
                        "solve_status": "infeasible",
                    }
                )
                continue
            selected_mask, refine_info = refine_policy_roster(
                predictions,
                solved["selected_mask"],
                ref_weekly,
                ref_decisions,
                ref_played,
                market,
                top_n,
                construction_waiver(current_waiver, policy.hurdle_delta),
                ref_option,
                policy.option_lambda,
            )
            roster = tuple(sorted(predictions.loc[selected_mask, "player"]))
            starter_mask = nominal_starter_mask(predictions, selected_mask)
            bench_mask = selected_mask & ~starter_mask
            expected_bench = ROSTER_SIZE - int(sum(LINEUP_REQUIRE.values()))
            if int(bench_mask.sum()) != expected_bench:
                raise AssertionError(
                    f"Expected {expected_bench} nominal bench players, "
                    f"found {int(bench_mask.sum())}."
                )
            bench_players = tuple(sorted(predictions.loc[bench_mask, "player"]))
            actual_metrics = actual_policy_metrics(environment, roster, bench_players)
            projected = forecast_metrics(
                predictions,
                roster,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                current_waiver,
                forecast_cache,
            )
            selected_prices = np.sort(market[selected_mask])[::-1]
            forecast_spend = float(market[selected_mask].sum())
            actual_feasible = actual_metrics["actual_salary_spend"] <= SALARY_CAP + 1e-8
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "policy": policy.name,
                    "solve_status": "optimal",
                    "hurdle_delta": policy.hurdle_delta,
                    "bench_weight": policy.bench_weight,
                    "option_lambda": policy.option_lambda,
                    "roster": "|".join(roster),
                    "bench_players": "|".join(bench_players),
                    "accepted_swaps": int(refine_info["accepted_swaps"]),
                    "forecast_salary_spend": forecast_spend,
                    "starter_forecast_spend": float(market[starter_mask].sum()),
                    "bench_forecast_spend": float(market[bench_mask].sum()),
                    "bench_mean_price": float(market[bench_mask].mean()),
                    "bench_le5": int(np.sum(market[bench_mask] <= 5.0)),
                    "bench_le10": int(np.sum(market[bench_mask] <= 10.0)),
                    "top3_spend_share": float(selected_prices[:3].sum() / forecast_spend),
                    "stars_ge40": int(np.sum(market[selected_mask] >= 40.0)),
                    "actual_cap_feasible": bool(actual_feasible),
                    "actual_cap_overage": float(
                        max(actual_metrics["actual_salary_spend"] - SALARY_CAP, 0.0)
                    ),
                    **projected,
                    **actual_metrics,
                }
            )
        if (trial + 1) % max(1, min(25, trials)) == 0:
            print(
                f"{year}: completed {trial + 1}/{trials} paired trials "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    return pd.DataFrame(rows), {
        "top_n_players": top_n,
        "remaining_market_budget": remaining_budget,
        "remaining_market_slots": remaining_slots,
        "runtime_seconds": time.perf_counter() - started,
    }


METRICS = (
    "forecast_ev",
    "forecast_p10",
    "forecast_p90",
    "actual_points",
    "actual_playoff_points",
    "actual_waiver_starts",
    "actual_cap_feasible",
    "actual_cap_overage",
    "forecast_salary_spend",
    "starter_forecast_spend",
    "bench_forecast_spend",
    "bench_mean_price",
    "bench_le5",
    "bench_le10",
    "top3_spend_share",
    "stars_ge40",
    "bench_actual_max4_ppg",
    "bench_sustained_12_hits",
    "bench_sustained_15_hits",
    "bench_dead_below10",
)


def roster_jaccard(left: str, right: str) -> float:
    left_set = set(str(left).split("|"))
    right_set = set(str(right).split("|"))
    return float(len(left_set & right_set) / len(left_set | right_set))


def summarize_results(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    optimal = trials[trials.solve_status.eq("optimal")].copy()
    by_year = (
        optimal.groupby(["year", "policy"], as_index=False)
        .agg(
            trials=("trial", "size"),
            unique_rosters=("roster", "nunique"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )
    across = (
        by_year.groupby("policy", as_index=False)
        .agg(
            seasons=("year", "nunique"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )

    baseline = optimal[optimal.policy.eq(BASELINE_POLICY)][
        ["year", "trial", "roster", *METRICS]
    ].copy()
    paired_rows = []
    for policy in [value.name for value in POLICIES if value.name != BASELINE_POLICY]:
        candidate = optimal[optimal.policy.eq(policy)][
            ["year", "trial", "roster", *METRICS]
        ].copy()
        paired = baseline.merge(
            candidate,
            on=["year", "trial"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired["policy"] = policy
        paired["roster_changed"] = paired.roster_baseline.ne(paired.roster_candidate)
        paired["roster_jaccard"] = [
            roster_jaccard(left, right)
            for left, right in zip(paired.roster_baseline, paired.roster_candidate)
        ]
        for metric in METRICS:
            paired[f"{metric}_effect"] = (
                pd.to_numeric(paired[f"{metric}_candidate"]).astype(float)
                - pd.to_numeric(paired[f"{metric}_baseline"]).astype(float)
            )
        paired_rows.append(paired)
    paired = pd.concat(paired_rows, ignore_index=True)

    period_rows = []
    for (policy, period), frame in paired.assign(
        period=np.where(paired.year.eq(2025), "temporal_check_2025", "development_2022_2024")
    ).groupby(["policy", "period"]):
        row: dict[str, Any] = {
            "policy": policy,
            "period": period,
            "comparisons": int(len(frame)),
            "roster_changed_rate": float(frame.roster_changed.mean()),
            "roster_jaccard": float(frame.roster_jaccard.mean()),
        }
        for metric in METRICS:
            values = frame[f"{metric}_effect"].astype(float)
            row[f"mean_{metric}_effect"] = float(values.mean())
            row[f"mcse_{metric}_effect"] = float(
                values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        period_rows.append(row)
    periods = pd.DataFrame(period_rows)
    return by_year, across, paired, periods


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 2) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "|" + "|".join(["---"] * len(display.columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    across: pd.DataFrame,
    periods: pd.DataFrame,
) -> None:
    overall_columns = [
        "policy",
        "forecast_ev",
        "forecast_p10",
        "actual_points",
        "actual_playoff_points",
        "bench_forecast_spend",
        "top3_spend_share",
        "bench_sustained_15_hits",
        "actual_cap_feasible",
    ]
    period_columns = [
        "policy",
        "period",
        "roster_changed_rate",
        "mean_forecast_ev_effect",
        "mean_forecast_p10_effect",
        "mean_actual_points_effect",
        "mean_actual_playoff_points_effect",
        "mean_bench_forecast_spend_effect",
        "mean_top3_spend_share_effect",
        "mean_bench_sustained_15_hits_effect",
        "mean_actual_cap_feasible_effect",
    ]
    lines = [
        "# Bench Option And Waiver-Hurdle Results",
        "",
        f"{args.trials} paired trials per origin with {args.contexts} construction and evaluation contexts.",
        "All forecast evaluation uses the unchanged current projected waiver baseline.",
        "",
        "## Policy means across origins",
        "",
        markdown_table(across, overall_columns, digits=3),
        "",
        "## Paired effects versus current bench weight 0.25",
        "",
        markdown_table(periods, period_columns, digits=3),
        "",
        "## Interpretation boundaries",
        "",
        "- Four seasons provide four outcome units; trial counts measure Monte Carlo stability.",
        "- Actual point comparisons include historically unaffordable rosters and must be read with cap feasibility.",
        "- The sustained option is a strategy-utility sensitivity, not literal additional lineup points.",
        "- Realized waiver scoring remains optimistic and lacks opponent claim competition or transaction persistence.",
        "- The construction refinement retains the live mean-profile/OR-played-mask approximation.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--contexts", type=int, default=250)
    parser.add_argument("--context-draws", type=int, default=5)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(base.FROZEN_SOURCES))
    if invalid:
        parser.error(f"Unsupported replay years: {invalid}")
    if min(args.trials, args.contexts, args.context_draws) <= 0:
        parser.error("Trials, contexts, and context draws must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Loading shared leakage-safe replay inputs...", flush=True)
    raw_weekly = base.load_raw_weekly(max_year=max(args.years))
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    all_trials = []
    manifest: dict[str, Any] = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "policies": [asdict(policy) for policy in POLICIES],
        "base_replay": str(BASE_STUDY),
        "base_replay_sha256": base.sha256_file(BASE_STUDY),
        "simulation_helper": {
            "path": str(base.APP_HELPER),
            "sha256": base.sha256_file(base.APP_HELPER),
        },
        "origins": {},
    }

    for year in args.years:
        year_started = time.perf_counter()
        print(f"\n=== Origin {year} ===", flush=True)
        conn, source_manifest = base.open_frozen_source(base.FROZEN_SOURCES[year])
        try:
            target_features = features[features.season.eq(year)].copy()
            forecast, ppg_draws, projection_meta = base.load_frozen_forecast(
                year,
                conn,
                target_features,
                args.projection_draws,
                args.seed,
            )
            forecast, salary_draws, salary_meta = base.build_salary_forecast(
                year,
                conn,
                forecast,
                actual,
                features,
                args.salary_draws,
                args.seed,
            )
        finally:
            conn.close()

        environment, outcome_labels = base.build_actual_environment(
            year,
            forecast,
            raw_weekly,
            features,
            actual,
        )
        cache, template_audit = base.build_template_cache(
            year,
            forecast,
            features,
            raw_weekly,
        )
        if not (template_audit.max_donor_season < year).all():
            raise AssertionError("Construction template pool crossed the replay origin.")
        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game", "salary"]
        ].copy()
        sim = base.make_simulation(year, player_data, cache)
        current_waiver = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )
        impact_thresholds = league_starter_thresholds(forecast, current_waiver)

        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_idx].reset_index(drop=True)
        predictions = base.build_predictions(
            candidate_forecast,
            ppg_draws[candidate_idx],
        )
        candidate_salary_draws = salary_draws[candidate_idx]
        print(
            f"{year}: {len(predictions)} selectable players; building "
            f"{args.contexts} causal construction and evaluation contexts...",
            flush=True,
        )
        weekly, decisions, played = base.generate_construction_contexts(
            sim,
            predictions,
            args.contexts,
            args.seed + year,
        )
        evaluation_weekly, evaluation_decisions, evaluation_played = (
            base.generate_construction_contexts(
                sim,
                predictions,
                args.contexts,
                args.seed + 100_000 + year,
            )
        )
        option_bank = sustained_option_bank(
            weekly,
            played,
            predictions,
            impact_thresholds,
        )
        if option_bank.shape != (len(predictions), args.contexts):
            raise AssertionError("Sustained option bank has the wrong shape.")
        if not np.isfinite(option_bank).all() or (option_bank < 0).any():
            raise AssertionError("Sustained option values must be finite and nonnegative.")
        value_banks = policy_value_banks(
            weekly,
            decisions,
            played,
            predictions,
            current_waiver,
            option_bank,
        )
        trials, run_meta = run_year_trials(
            year,
            sim,
            predictions,
            candidate_salary_draws,
            environment,
            weekly,
            decisions,
            played,
            evaluation_weekly,
            evaluation_decisions,
            evaluation_played,
            value_banks,
            option_bank,
            current_waiver,
            args.trials,
            args.context_draws,
            args.seed,
        )
        if not trials.solve_status.eq("optimal").all():
            raise AssertionError("A policy roster solve failed.")
        all_trials.append(trials)
        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(run_meta)
        source_manifest.update(
            {
                "keeper_count": environment["keeper_count"],
                "keeper_spend": environment["keeper_spend"],
                "current_waiver_baseline": current_waiver,
                "impact_thresholds": impact_thresholds,
                "mean_option_value_by_position": {
                    pos: float(option_bank[predictions.pos.eq(pos)].mean())
                    for pos in POSITIONS
                },
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        manifest["origins"][str(year)] = source_manifest
        pd.concat(all_trials, ignore_index=True).to_csv(
            output_dir / "roster_trials.csv",
            index=False,
        )
        print(
            f"{year}: complete in {time.perf_counter() - year_started:.1f}s",
            flush=True,
        )

    trials = pd.concat(all_trials, ignore_index=True)
    expected_rows = len(args.years) * args.trials * len(POLICIES)
    if len(trials) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(trials)}.")
    if not trials.roster.str.split("|").map(len).eq(ROSTER_SIZE).all():
        raise AssertionError("A policy roster does not contain 13 players.")
    if not (trials.forecast_salary_spend <= SALARY_CAP + 1e-8).all():
        raise AssertionError("A policy roster exceeds the forecast salary cap.")

    by_year, across, paired, periods = summarize_results(trials)
    outputs = {
        "roster_trials.csv": trials,
        "policy_summary_by_year.csv": by_year,
        "policy_summary_across_years.csv": across,
        "paired_effects.csv": paired,
        "paired_effects_by_period.csv": periods,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["verification"] = {
        "expected_rows": expected_rows,
        "all_solves_optimal": True,
        "all_rosters_size_13": True,
        "all_forecast_spend_within_cap": True,
        "evaluation_waiver_locked": True,
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(output_dir, args, across, periods)
    print(f"Study complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
