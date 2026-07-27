"""Replay one-year keeper options with full-roster budget reinvestment."""

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
PORTFOLIO_STUDY = (
    ROOT / "research" / "studies" / "2026-07-20_one_year_keeper_portfolio"
)
PORTFOLIO_RUNNER = PORTFOLIO_STUDY / "run_replay.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load replay module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


portfolio = load_module("keeper_reinvestment_portfolio", PORTFOLIO_RUNNER)
base = portfolio.base
bench = portfolio.bench
keeper = portfolio.keeper
FootballSimulation = base.FootballSimulation

POSITIONS = portfolio.POSITIONS
LINEUP_REQUIRE = portfolio.LINEUP_REQUIRE
POS_MIN = portfolio.POS_MIN
POS_MAX = portfolio.POS_MAX
ROSTER_SIZE = portfolio.ROSTER_SIZE
SALARY_CAP = portfolio.SALARY_CAP
NUM_TEAMS = portfolio.NUM_TEAMS
TOTAL_MARKET_BUDGET = portfolio.TOTAL_MARKET_BUDGET
TOTAL_MARKET_SLOTS = portfolio.TOTAL_MARKET_SLOTS
TOP_N = portfolio.TOP_N
KEEPER_ESCALATION = portfolio.KEEPER_ESCALATION
STARTER_COUNT = int(sum(LINEUP_REQUIRE.values()))
BENCH_COUNT = ROSTER_SIZE - STARTER_COUNT
BASELINE_POLICY = "control"


@dataclass(frozen=True)
class Policy:
    name: str
    max_forced_options: int


POLICIES = (
    Policy("control", 0),
    Policy("reinvest_k1", 1),
    Policy("reinvest_k2", 2),
    Policy("reinvest_k3", 3),
)


def current_value_bank(
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    predictions: pd.DataFrame,
    waiver_baseline: dict[str, float],
) -> np.ndarray:
    """Build the same zero-bench-bonus marginal bank as keeper_engine0."""
    values = []
    for context_idx in range(len(weekly)):
        values.append(
            FootballSimulation.managed_marginal_values(
                weekly[context_idx],
                predictions.pos.to_numpy(),
                decisions[context_idx],
                predictions.player.to_numpy(),
                base_players=[],
                waiver_baselines=waiver_baseline,
                lineup_require=LINEUP_REQUIRE,
                bench_upside_weight=0.0,
                played_mask=played[context_idx],
            )
        )
    return np.column_stack(values).astype(np.float32)


def solve_current_roster(
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    current_values: np.ndarray,
    market: np.ndarray,
    top_n: list[str],
    forced_players: tuple[str, ...],
) -> np.ndarray | None:
    """Solve the entire roster while requiring the proposed option players."""
    h_player_add = {player: -1 for player in forced_players}
    static = sim.build_managed_ilp_static_matrices(
        predictions,
        h_player_add,
        list(forced_players),
        top_n,
        ROSTER_SIZE,
        POS_MIN,
        POS_MAX,
        enforce_top_n=True,
    )
    zeros = np.zeros(len(predictions), dtype=float)
    return keeper.solve_keeper_scenario(
        sim,
        predictions,
        current_values,
        zeros,
        0.0,
        market,
        static,
    )


def refine_current_roster_fixed(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    forced_players: tuple[str, ...],
    weekly_scores: np.ndarray,
    decision_scores: np.ndarray,
    played_mask: np.ndarray,
    salary_values: np.ndarray,
    top_n: list[str],
    waiver_baseline: dict[str, float],
    max_swaps: int = 12,
) -> tuple[np.ndarray, int]:
    """Apply the current exact one-swap refinement without dropping options."""
    selected = np.asarray(selected_mask, dtype=bool).copy()
    salary_values = np.asarray(salary_values, dtype=float)
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    forced_set = set(forced_players)
    top_n_set = set(top_n)
    accepted = 0

    ref_weekly = weekly_scores.mean(axis=0)
    ref_decisions = decision_scores.mean(axis=0)
    ref_played = np.where(
        np.any(played_mask >= 0, axis=0),
        np.any(played_mask > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)

    for _ in range(max_swaps):
        current_score = portfolio.exact_construction_score(
            predictions,
            selected,
            weekly_scores,
            decision_scores,
            played_mask,
            waiver_baseline,
        )
        out_indices = np.array(
            [
                idx
                for idx in np.flatnonzero(selected)
                if players[idx] not in forced_set
            ],
            dtype=int,
        )
        if len(out_indices) == 0:
            break
        base_masks = []
        for out_idx in out_indices:
            base_mask = selected.copy()
            base_mask[out_idx] = False
            base_masks.append(base_mask)
        incoming_rows = FootballSimulation.managed_marginal_values_batch(
            ref_weekly,
            positions,
            ref_decisions,
            players,
            [players[mask].tolist() for mask in base_masks],
            waiver_baselines=waiver_baseline,
            lineup_require=LINEUP_REQUIRE,
            bench_upside_weight=0.0,
            played_mask=ref_played,
        )

        best_approx: tuple[float, int, int] | None = None
        for out_idx, base_mask, incoming in zip(
            out_indices, base_masks, incoming_rows
        ):
            base_salary = float(salary_values[base_mask].sum())
            base_counts = {
                pos: int(np.sum(positions[base_mask] == pos)) for pos in POSITIONS
            }
            eligible = ~selected
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
            approximate = (
                current_score - float(incoming[out_idx]) + incoming[eligible_idx]
            )
            local = int(np.argmax(approximate))
            candidate = (
                float(approximate[local]),
                int(out_idx),
                int(eligible_idx[local]),
            )
            if best_approx is None or candidate[0] > best_approx[0]:
                best_approx = candidate

        if best_approx is None:
            break
        _, out_idx, in_idx = best_approx
        replacement = selected.copy()
        replacement[out_idx] = False
        replacement[in_idx] = True
        exact_score = portfolio.exact_construction_score(
            predictions,
            replacement,
            weekly_scores,
            decision_scores,
            played_mask,
            waiver_baseline,
        )
        if exact_score <= current_score + 1e-5:
            break
        selected = replacement
        accepted += 1

    if any(not selected[np.flatnonzero(players == player)[0]] for player in forced_set):
        raise AssertionError("A forced keeper option was removed during refinement.")
    return selected, accepted


def ranked_option_candidates(
    selected_mask: np.ndarray,
    surplus_draws: np.ndarray,
    positions: np.ndarray,
    market: np.ndarray,
    current_ppg: np.ndarray,
    player_names: np.ndarray,
    shortlist: int,
) -> np.ndarray:
    """Rank outside-roster candidates by marginal expected-best utility."""
    starter_mask = keeper.fast_nominal_starter_mask(
        selected_mask,
        current_ppg,
        positions,
        player_names,
    )
    bench_idx = np.flatnonzero(selected_mask & ~starter_mask)
    if len(bench_idx):
        current_best = np.max(surplus_draws[bench_idx], axis=0)
    else:
        current_best = np.zeros(surplus_draws.shape[1], dtype=float)
    current_utility = float(np.mean(current_best))
    outside = np.flatnonzero(~selected_mask & (positions != "QB"))
    if len(outside) == 0:
        return outside
    marginal = (
        np.maximum(current_best[None, :], surplus_draws[outside]).mean(axis=1)
        - current_utility
    )
    positive = marginal > 1e-8
    outside = outside[positive]
    marginal = marginal[positive]
    if len(outside) == 0:
        return outside
    # Expected surplus per acquisition dollar breaks close utility ties in favor
    # of candidates that can actually create starter-reinvestment room.
    efficiency = marginal / np.maximum(market[outside], 1.0)
    order = np.lexsort((outside, -efficiency, -marginal))
    return outside[order[: min(int(shortlist), len(order))]]


def optimize_option_step(
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    forced_players: tuple[str, ...],
    baseline_score: float,
    current_values: np.ndarray,
    construction_weekly: np.ndarray,
    construction_decisions: np.ndarray,
    construction_played: np.ndarray,
    market: np.ndarray,
    top_n: list[str],
    waiver_baseline: dict[str, float],
    surplus_draws: np.ndarray,
    shortlist: int,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    current_ppg = predictions[
        FootballSimulation.sample_value_columns(predictions)
    ].mean(axis=1).to_numpy(dtype=float)
    current_starters = keeper.fast_nominal_starter_mask(
        selected_mask, current_ppg, positions, players
    )
    current_bench = selected_mask & ~current_starters
    current_utility = portfolio.portfolio_utility(
        np.flatnonzero(current_bench), surplus_draws, "expected_best"
    )
    candidates = ranked_option_candidates(
        selected_mask,
        surplus_draws,
        positions,
        market,
        current_ppg,
        players,
        shortlist,
    )
    best: tuple[float, float, str, np.ndarray, int] | None = None
    attempts = 0

    for candidate_idx in candidates:
        candidate_name = str(players[candidate_idx])
        proposed_forced = tuple(sorted((*forced_players, candidate_name)))
        solved = solve_current_roster(
            sim,
            predictions,
            current_values,
            market,
            top_n,
            proposed_forced,
        )
        attempts += 1
        if solved is None:
            continue
        refined, refine_swaps = refine_current_roster_fixed(
            predictions,
            solved,
            proposed_forced,
            construction_weekly,
            construction_decisions,
            construction_played,
            market,
            top_n,
            waiver_baseline,
        )
        starters = keeper.fast_nominal_starter_mask(
            refined, current_ppg, positions, players
        )
        bench_mask = refined & ~starters
        if any(
            not bench_mask[np.flatnonzero(players == player)[0]]
            for player in proposed_forced
        ):
            continue
        score = portfolio.exact_construction_score(
            predictions,
            refined,
            construction_weekly,
            construction_decisions,
            construction_played,
            waiver_baseline,
        )
        if score < baseline_score - 1e-6:
            continue
        utility = portfolio.portfolio_utility(
            np.flatnonzero(bench_mask), surplus_draws, "expected_best"
        )
        if utility <= current_utility + 1e-8:
            continue
        candidate = (
            float(utility),
            float(score),
            candidate_name,
            refined,
            int(refine_swaps),
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        return selected_mask.copy(), forced_players, {
            "step_accepted": False,
            "candidate_attempts": attempts,
            "refine_swaps": 0,
        }
    utility, score, candidate_name, refined, refine_swaps = best
    return refined, tuple(sorted((*forced_players, candidate_name))), {
        "step_accepted": True,
        "candidate_attempts": attempts,
        "refine_swaps": refine_swaps,
        "step_utility": utility,
        "step_construction_score": score,
    }


def starter_raw_points(
    environment: dict[str, Any], starter_players: tuple[str, ...]
) -> float:
    mask = environment["labels"].player.isin(starter_players).to_numpy()
    played = environment["played"][mask] > 0
    return float((environment["scores"][mask] * played).sum())


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
    value_bank: np.ndarray,
    next_draws: np.ndarray,
    validation_match: np.ndarray,
    current_waiver: dict[str, float],
    observed_prices: np.ndarray,
    future_ppg: np.ndarray,
    available_horizons: list[int],
    year_exp: np.ndarray,
    trials: int,
    candidate_shortlist: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    remaining_budget = TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = TOTAL_MARKET_SLOTS - environment["keeper_count"]
    top_n = predictions.nlargest(min(TOP_N, len(predictions)), "salary").player.tolist()
    rng = np.random.default_rng(seed + year * 101)
    salary_plan = rng.integers(0, salary_draws.shape[1], size=(trials, 5))
    raw_market = np.column_stack(
        [salary_draws[:, row].mean(axis=1) for row in salary_plan]
    )
    markets = base.normalize_market_draws(
        sim, raw_market, remaining_budget, remaining_slots
    )
    forecast_cache: dict[tuple[str, ...], dict[str, float]] = {}
    starter_forecast_cache: dict[tuple[str, ...], dict[str, float]] = {}
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    current_ppg = predictions[
        FootballSimulation.sample_value_columns(predictions)
    ].mean(axis=1).to_numpy(dtype=float)
    construction_weekly = weekly.mean(axis=0, keepdims=True)
    construction_decisions = decisions.mean(axis=0, keepdims=True)
    construction_played = np.where(
        np.any(played >= 0, axis=0),
        np.any(played > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)[None, :, :]
    current_values = value_bank.mean(axis=1)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for trial in range(trials):
        market = markets[:, trial]
        predictions["salary"] = market
        curves = keeper.fit_position_market_curves(current_ppg, market, positions)
        future_market = portfolio.transform_ppg_to_market(
            next_draws, positions, curves
        )
        surplus_draws = portfolio.first_year_surplus_draws(future_market, market)
        surplus_draws[positions == "QB"] = 0.0

        solved_control = solve_current_roster(
            sim,
            predictions,
            current_values,
            market,
            top_n,
            (),
        )
        if solved_control is None:
            raise RuntimeError("The rebuilt current-only roster was infeasible.")
        control_mask, control_refine_swaps = refine_current_roster_fixed(
            predictions,
            solved_control,
            (),
            construction_weekly,
            construction_decisions,
            construction_played,
            market,
            top_n,
            current_waiver,
        )
        control_starters = keeper.fast_nominal_starter_mask(
            control_mask, current_ppg, positions, players
        )
        control_bench = control_mask & ~control_starters
        baseline_score = portfolio.exact_construction_score(
            predictions,
            control_mask,
            construction_weekly,
            construction_decisions,
            construction_played,
            current_waiver,
        )

        policy_states: dict[int, tuple[np.ndarray, tuple[str, ...], dict[str, Any]]] = {
            0: (
                control_mask.copy(),
                (),
                {
                    "candidate_attempts": 0,
                    "refine_swaps": int(control_refine_swaps),
                    "accepted_steps": 0,
                },
            )
        }
        selected = control_mask.copy()
        forced_players: tuple[str, ...] = ()
        cumulative_attempts = 0
        cumulative_refine = int(control_refine_swaps)
        accepted_steps = 0
        for max_options in range(1, 4):
            selected, forced_players, step = optimize_option_step(
                sim,
                predictions,
                selected,
                forced_players,
                baseline_score,
                current_values,
                construction_weekly,
                construction_decisions,
                construction_played,
                market,
                top_n,
                current_waiver,
                surplus_draws,
                candidate_shortlist,
            )
            cumulative_attempts += int(step["candidate_attempts"])
            cumulative_refine += int(step["refine_swaps"])
            accepted_steps += int(step["step_accepted"])
            policy_states[max_options] = (
                selected.copy(),
                forced_players,
                {
                    "candidate_attempts": cumulative_attempts,
                    "refine_swaps": cumulative_refine,
                    "accepted_steps": accepted_steps,
                },
            )

        for policy in POLICIES:
            selected, forced, search_info = policy_states[policy.max_forced_options]
            starters = keeper.fast_nominal_starter_mask(
                selected, current_ppg, positions, players
            )
            bench_mask = selected & ~starters
            if int(starters.sum()) != STARTER_COUNT or int(bench_mask.sum()) != BENCH_COUNT:
                raise AssertionError("A policy roster has the wrong starter/bench shape.")
            if any(
                not bench_mask[np.flatnonzero(players == player)[0]]
                for player in forced
            ):
                raise AssertionError("A forced option did not remain on the bench.")

            roster = tuple(sorted(players[selected]))
            starter_players = tuple(sorted(players[starters]))
            bench_players = tuple(sorted(players[bench_mask]))
            projected = bench.forecast_metrics(
                predictions,
                roster,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                current_waiver,
                forecast_cache,
            )
            starter_projected = bench.forecast_metrics(
                predictions,
                starter_players,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                current_waiver,
                starter_forecast_cache,
            )
            actual_metrics = bench.actual_policy_metrics(
                environment, roster, bench_players
            )
            keeper_metrics = portfolio.realized_keeper_metrics(
                bench_mask,
                surplus_draws,
                validation_match,
                future_ppg,
                available_horizons,
                positions,
                curves,
                market,
                observed_prices,
                players,
            )
            construction_score = portfolio.exact_construction_score(
                predictions,
                selected,
                construction_weekly,
                construction_decisions,
                construction_played,
                current_waiver,
            )
            forced_idx = np.flatnonzero(np.isin(players, list(forced)))
            selected_prices = np.sort(market[selected])[::-1]
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "policy": policy.name,
                    "solve_status": "optimal",
                    "max_forced_options": policy.max_forced_options,
                    "forced_option_count": len(forced),
                    "forced_option_players": "|".join(forced),
                    "forced_young_count": int(np.sum(year_exp[forced_idx] <= 2)),
                    "forced_mean_year_exp": float(np.mean(year_exp[forced_idx]))
                    if len(forced_idx)
                    else np.nan,
                    "candidate_attempts": search_info["candidate_attempts"],
                    "reoptimization_refine_swaps": search_info["refine_swaps"],
                    "roster": "|".join(roster),
                    "nominal_starters": "|".join(starter_players),
                    "bench_players": "|".join(bench_players),
                    "starter_changes_vs_control": int(
                        np.sum(starters & ~control_starters)
                    ),
                    "bench_changes_vs_control": int(
                        np.sum(bench_mask & ~control_bench)
                    ),
                    "roster_changes_vs_control": int(
                        np.sum(selected & ~control_mask)
                    ),
                    "forecast_salary_spend": float(market[selected].sum()),
                    "unspent_budget": float(SALARY_CAP - market[selected].sum()),
                    "starter_forecast_spend": float(market[starters].sum()),
                    "bench_forecast_spend": float(market[bench_mask].sum()),
                    "starter_projected_ppg_sum": float(current_ppg[starters].sum()),
                    "starter_raw_actual_points": starter_raw_points(
                        environment, starter_players
                    ),
                    "starter_forecast_ev": starter_projected["forecast_ev"],
                    "starter_forecast_p10": starter_projected["forecast_p10"],
                    "starter_forecast_p90": starter_projected["forecast_p90"],
                    "top3_spend_share": float(
                        selected_prices[:3].sum() / market[selected].sum()
                    ),
                    "baseline_construction_score": baseline_score,
                    "current_construction_score": construction_score,
                    "current_construction_delta": construction_score - baseline_score,
                    **projected,
                    **actual_metrics,
                    **keeper_metrics,
                }
            )
        if (trial + 1) % max(1, min(10, trials)) == 0:
            print(
                f"{year}: completed {trial + 1}/{trials} trials "
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
    "forced_option_count",
    "forced_young_count",
    "candidate_attempts",
    "reoptimization_refine_swaps",
    "starter_changes_vs_control",
    "bench_changes_vs_control",
    "roster_changes_vs_control",
    "forecast_salary_spend",
    "unspent_budget",
    "starter_forecast_spend",
    "bench_forecast_spend",
    "starter_projected_ppg_sum",
    "starter_raw_actual_points",
    "starter_forecast_ev",
    "starter_forecast_p10",
    "starter_forecast_p90",
    "top3_spend_share",
    "current_construction_delta",
    "forecast_ev",
    "forecast_p10",
    "forecast_p90",
    "actual_points",
    "drafted_only_points",
    "actual_waiver_starts",
    "actual_playoff_points",
    "predicted_expected_best_surplus",
    "predicted_probability_any_hit",
    "predicted_probability_any_10",
    "predicted_probability_any_20",
    "actual_best_keeper_surplus",
    "actual_any_keeper_hit_10",
    "actual_any_keeper_hit_20",
    "actual_best_future_ppg",
)


def summarize(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year = (
        trials.groupby(["year", "policy"], as_index=False)
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
    control_columns = ["year", "trial", "roster", *METRICS]
    control = trials[trials.policy.eq(BASELINE_POLICY)][control_columns].copy()
    control = control.rename(
        columns={
            "roster": "roster_control",
            **{metric: f"{metric}_control" for metric in METRICS},
        }
    )
    paired = trials[~trials.policy.eq(BASELINE_POLICY)].merge(
        control, on=["year", "trial"], validate="many_to_one"
    )
    paired["roster_changed"] = paired.roster.ne(paired.roster_control)
    for metric in METRICS:
        paired[f"{metric}_effect"] = (
            paired[metric] - paired[f"{metric}_control"]
        )
    return by_year, across, paired


def write_summary(
    output_dir: Path,
    by_year: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        headers = [str(column) for column in display.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in display.itertuples(index=False, name=None):
            values = ["" if pd.isna(value) else str(value) for value in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    effects = (
        paired.groupby(["year", "policy"], as_index=False)
        .agg(
            roster_changed_rate=("roster_changed", "mean"),
            forced_option_count=("forced_option_count", "mean"),
            starter_changes=("starter_changes_vs_control", "mean"),
            starter_spend_effect=("starter_forecast_spend_effect", "mean"),
            bench_spend_effect=("bench_forecast_spend_effect", "mean"),
            unspent_budget_effect=("unspent_budget_effect", "mean"),
            starter_forecast_ev_effect=("starter_forecast_ev_effect", "mean"),
            forecast_ev_effect=("forecast_ev_effect", "mean"),
            forecast_p10_effect=("forecast_p10_effect", "mean"),
            actual_points_effect=("actual_points_effect", "mean"),
            waiver_starts_effect=("actual_waiver_starts_effect", "mean"),
            playoff_effect=("actual_playoff_points_effect", "mean"),
            predicted_best_surplus_effect=(
                "predicted_expected_best_surplus_effect",
                "mean",
            ),
            actual_best_surplus_effect=("actual_best_keeper_surplus_effect", "mean"),
            actual_hit20_effect=("actual_any_keeper_hit_20_effect", "mean"),
        )
    )
    effects.to_csv(output_dir / "policy_effects_by_year.csv", index=False)
    lines = [
        "# Keeper Reinvestment Sensitivity Results",
        "",
        "Full-roster reoptimization around up to one, two, or three newly forced ",
        "keeper-oriented bench players. All accepted portfolios preserve the ",
        "full-bank expected reference score; independent outcomes remain evaluation-only.",
        "",
        "## Paired effects versus current-only control",
        "",
        markdown_table(effects.round(3)),
        "",
        "## Policy means by origin",
        "",
        markdown_table(by_year.round(3)),
        "",
        "## Interpretation boundaries",
        "",
        "- The option search is greedy and uses a bounded marginal-utility shortlist.",
        "- A forced player must remain on the nominal preseason bench.",
        "- Other starter and bench slots may change during the conditional full solve.",
        "- Keeper surplus never enters current-season fantasy-point evaluation.",
        "",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(line.rstrip() for line in lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--contexts", type=int, default=250)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--candidate-shortlist", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(base.FROZEN_SOURCES))
    if invalid:
        parser.error(f"Unsupported years: {invalid}")
    if min(
        args.trials,
        args.contexts,
        args.projection_draws,
        args.candidate_shortlist,
    ) <= 0:
        parser.error("Trial, context, draw, and shortlist counts must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "roster_trials.csv"
    manifest_path = output_dir / "source_manifest.json"
    started = time.perf_counter()

    raw_weekly = base.load_raw_weekly(max_year=max(args.years) + 1)
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()

    all_trials: list[pd.DataFrame] = []
    complete_years: set[int] = set()
    prior_manifest: dict[str, Any] = {}
    if args.resume and manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.resume and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        for year, frame in checkpoint.groupby("year"):
            counts = frame.groupby("policy").size().to_dict()
            if (
                set(counts) == {policy.name for policy in POLICIES}
                and all(count == args.trials for count in counts.values())
            ):
                complete_years.add(int(year))
                all_trials.append(frame)
        if complete_years:
            print(
                "Resuming complete origins: "
                + ", ".join(map(str, sorted(complete_years))),
                flush=True,
            )

    manifest: dict[str, Any] = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "policies": [asdict(policy) for policy in POLICIES],
        "selection_contract": {
            "one_year_keeper_escalation": KEEPER_ESCALATION,
            "portfolio_objective": "expected best surplus across nominal bench",
            "construction_tolerance": 0.0,
            "full_roster_reoptimization": True,
            "forced_options_must_remain_nominal_bench": True,
        },
        "portfolio_runner": str(PORTFOLIO_RUNNER),
        "portfolio_runner_sha256": base.sha256_file(PORTFOLIO_RUNNER),
        "origins": prior_manifest.get("origins", {}),
    }

    for year in args.years:
        if year in complete_years:
            continue
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
            year, forecast, raw_weekly, features, actual
        )
        cache, template_audit = base.build_template_cache(
            year, forecast, features, raw_weekly
        )
        if not (template_audit.max_donor_season < year).all():
            raise AssertionError("A construction template crossed the origin.")
        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game", "salary"]
        ].copy()
        sim = base.make_simulation(year, player_data, cache)
        current_waiver = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS, roster_size=ROSTER_SIZE
        )

        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_idx].reset_index(drop=True)
        predictions = base.build_predictions(
            candidate_forecast, ppg_draws[candidate_idx]
        )
        candidate_salary_draws = salary_draws[candidate_idx]
        candidate_outcomes = outcome_labels.iloc[candidate_idx].reset_index(drop=True)
        observed_prices = pd.to_numeric(
            candidate_outcomes.actual_salary, errors="coerce"
        ).to_numpy(dtype=float)
        observed_prices = np.where(
            candidate_outcomes.actual_salary_matched.to_numpy(dtype=bool),
            observed_prices,
            np.nan,
        )
        next_draws, validation_match, next_meta = portfolio.validation_next_year_draws(
            year, candidate_forecast, args.projection_draws, args.seed
        )
        future_ppg, _, available_horizons = keeper.future_actual_ppg(
            year, candidate_forecast, raw_weekly
        )
        exp_lookup = (
            target_features.sort_values("preseason_proj_ppg", ascending=False)
            .drop_duplicates(["player_key", "pos"])
            .set_index(["player_key", "pos"])
            .year_exp
            .to_dict()
        )
        year_exp = np.array(
            [
                float(exp_lookup.get((row.player_key, row.pos), 99.0))
                for row in candidate_forecast.itertuples()
            ],
            dtype=float,
        )
        weekly, decisions, played = base.generate_construction_contexts(
            sim, predictions, args.contexts, args.seed + year
        )
        evaluation_weekly, evaluation_decisions, evaluation_played = (
            base.generate_construction_contexts(
                sim,
                predictions,
                args.contexts,
                args.seed + 100_000 + year,
            )
        )
        values = current_value_bank(
            weekly, decisions, played, predictions, current_waiver
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
            values,
            next_draws,
            validation_match,
            current_waiver,
            observed_prices,
            future_ppg,
            available_horizons,
            year_exp,
            args.trials,
            args.candidate_shortlist,
            args.seed,
        )
        all_trials.append(trials)
        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(next_meta)
        source_manifest.update(run_meta)
        source_manifest.update(
            {
                "current_waiver_baseline": current_waiver,
                "realized_future_horizons": available_horizons,
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        manifest["origins"][str(year)] = source_manifest
        combined = pd.concat(all_trials, ignore_index=True)
        combined.to_csv(checkpoint_path, index=False)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"{year}: complete in {time.perf_counter() - year_started:.1f}s",
            flush=True,
        )

    trials = (
        pd.concat(all_trials, ignore_index=True)
        .sort_values(["year", "trial", "policy"])
        .reset_index(drop=True)
    )
    expected = len(args.years) * args.trials * len(POLICIES)
    if len(trials) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(trials)}.")
    if trials.duplicated(["year", "trial", "policy"]).any():
        raise AssertionError("Duplicate year-trial-policy key.")
    if not trials.roster.str.split("|").map(len).eq(ROSTER_SIZE).all():
        raise AssertionError("A roster does not contain 13 players.")
    if not trials.nominal_starters.str.split("|").map(len).eq(STARTER_COUNT).all():
        raise AssertionError("A roster does not contain eight nominal starters.")
    if not trials.bench_players.str.split("|").map(len).eq(BENCH_COUNT).all():
        raise AssertionError("A roster does not contain five nominal bench players.")
    if not (trials.forecast_salary_spend <= SALARY_CAP + 1e-8).all():
        raise AssertionError("A roster exceeds the forecast salary cap.")
    if not (trials.current_construction_delta >= -1e-5).all():
        raise AssertionError("A policy violated the no-loss construction gate.")
    if not (
        trials.forced_option_count <= trials.max_forced_options
    ).all():
        raise AssertionError("A policy exceeded its keeper-option limit.")
    if not trials.forced_young_count.le(trials.forced_option_count).all():
        raise AssertionError("Young forced-option count is invalid.")

    by_year, across, paired = summarize(trials)
    output_frames = {
        "roster_trials.csv": trials,
        "policy_summary_by_year.csv": by_year,
        "policy_summary_across_years.csv": across,
        "paired_effects.csv": paired,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(output_dir / filename, index=False)
    write_summary(output_dir, by_year, paired)
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["verification"] = {
        "expected_rows": expected,
        "all_solves_optimal": True,
        "all_rosters_size_13": True,
        "all_starters_size_8": True,
        "all_benches_size_5": True,
        "all_forecast_spend_within_cap": True,
        "all_construction_no_loss": True,
        "all_option_limits_respected": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Study complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
