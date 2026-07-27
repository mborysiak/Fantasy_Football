"""Replay a soft expected-best keeper portfolio across the whole bench."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
REINVESTMENT_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-20_keeper_reinvestment_sensitivity"
)
REINVESTMENT_RUNNER = REINVESTMENT_STUDY / "run_replay.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load replay module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reinvestment = load_module("soft_keeper_reinvestment", REINVESTMENT_RUNNER)
portfolio = reinvestment.portfolio
base = reinvestment.base
bench = reinvestment.bench
keeper = reinvestment.keeper
FootballSimulation = base.FootballSimulation

POSITIONS = reinvestment.POSITIONS
LINEUP_REQUIRE = reinvestment.LINEUP_REQUIRE
ROSTER_SIZE = reinvestment.ROSTER_SIZE
SALARY_CAP = reinvestment.SALARY_CAP
NUM_TEAMS = reinvestment.NUM_TEAMS
TOTAL_MARKET_BUDGET = reinvestment.TOTAL_MARKET_BUDGET
TOTAL_MARKET_SLOTS = reinvestment.TOTAL_MARKET_SLOTS
TOP_N = reinvestment.TOP_N
STARTER_COUNT = reinvestment.STARTER_COUNT
BENCH_COUNT = reinvestment.BENCH_COUNT
KEEPER_ESCALATION = reinvestment.KEEPER_ESCALATION
POLICIES = ("control", "soft_portfolio")
BASELINE_POLICY = "control"


def construction_metrics(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    waiver_baseline: dict[str, float],
    cache: dict[tuple[str, ...], dict[str, float]],
) -> dict[str, float]:
    players = tuple(sorted(predictions.loc[selected_mask, "player"]))
    if players in cache:
        return cache[players]
    scores, _ = FootballSimulation.managed_lineup_multi_context_scores(
        weekly[:, selected_mask, :],
        predictions.loc[selected_mask, "pos"].to_numpy(),
        decisions[:, selected_mask, :],
        predictions.loc[selected_mask, "player"].to_numpy(),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=waiver_baseline,
        played_mask=played[:, selected_mask, :],
    )
    result = {
        "mean": float(np.mean(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
    }
    cache[players] = result
    return result


def bench_fillin_metrics(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    starter_mask: np.ndarray,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    waiver_baseline: dict[str, float],
    cache: dict[tuple[str, ...], dict[str, float]],
) -> dict[str, float]:
    baseline = construction_metrics(
        predictions,
        selected_mask,
        weekly,
        decisions,
        played,
        waiver_baseline,
        cache,
    )["mean"]
    marginal = []
    for idx in np.flatnonzero(selected_mask & ~starter_mask):
        partial = selected_mask.copy()
        partial[idx] = False
        without = construction_metrics(
            predictions,
            partial,
            weekly,
            decisions,
            played,
            waiver_baseline,
            cache,
        )["mean"]
        marginal.append(max(float(baseline - without), 0.0))
    values = np.sort(np.asarray(marginal, dtype=float))[::-1]
    return {
        "bench_fillin_total": float(values.sum()),
        "bench_fillin_top2": float(values[:2].sum()),
        "bench_fillin_second": float(values[1]) if len(values) > 1 else 0.0,
        "bench_positive_fillin_count": int(np.sum(values > 0.25)),
    }


def option_concentration_metrics(
    bench_mask: np.ndarray,
    surplus_draws: np.ndarray,
    players: np.ndarray,
) -> dict[str, Any]:
    bench_idx = np.flatnonzero(bench_mask)
    values = np.asarray(surplus_draws[bench_idx], dtype=float)
    best = np.max(values, axis=0)
    positive_draw = best > 1e-8
    shares = np.zeros(len(bench_idx), dtype=float)
    if positive_draw.any():
        winners = np.isclose(values[:, positive_draw], best[positive_draw][None, :])
        allocation = winners / np.maximum(winners.sum(axis=0, keepdims=True), 1)
        shares = allocation.mean(axis=1)
    positive_shares = shares[shares > 1e-12]
    effective = (
        float(1.0 / np.sum(np.square(positive_shares)))
        if len(positive_shares)
        else 0.0
    )
    order = np.lexsort((players[bench_idx], -shares))
    share_text = "|".join(
        f"{players[bench_idx[idx]]}:{shares[idx]:.4f}"
        for idx in order
        if shares[idx] > 1e-12
    )
    return {
        "option_positive_draw_rate": float(np.mean(positive_draw)),
        "option_effective_count": effective,
        "option_active_count_5pct": int(np.sum(shares >= 0.05)),
        "option_top_winner_share": float(shares.max()) if len(shares) else 0.0,
        "option_winner_shares": share_text,
    }


def optimize_soft_portfolio(
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    baseline_mask: np.ndarray,
    baseline_gate: dict[str, float],
    current_values: np.ndarray,
    reference_weekly: np.ndarray,
    reference_decisions: np.ndarray,
    reference_played: np.ndarray,
    gate_weekly: np.ndarray,
    gate_decisions: np.ndarray,
    gate_played: np.ndarray,
    market: np.ndarray,
    top_n: list[str],
    waiver_baseline: dict[str, float],
    surplus_draws: np.ndarray,
    candidate_shortlist: int,
    gate_cache: dict[tuple[str, ...], dict[str, float]],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    selected = np.asarray(baseline_mask, dtype=bool).copy()
    anchors: tuple[str, ...] = ()
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    current_ppg = predictions[
        FootballSimulation.sample_value_columns(predictions)
    ].mean(axis=1).to_numpy(dtype=float)
    attempts = 0
    accepted = 0
    refine_swaps = 0

    while accepted < BENCH_COUNT:
        starter_mask = keeper.fast_nominal_starter_mask(
            selected, current_ppg, positions, players
        )
        bench_mask = selected & ~starter_mask
        current_utility = portfolio.portfolio_utility(
            np.flatnonzero(bench_mask), surplus_draws, "expected_best"
        )
        candidates = reinvestment.ranked_option_candidates(
            selected,
            surplus_draws,
            positions,
            market,
            current_ppg,
            players,
            candidate_shortlist,
        )
        best: tuple[float, float, float, str, np.ndarray, int] | None = None
        for candidate_idx in candidates:
            candidate_name = str(players[candidate_idx])
            proposed_anchors = tuple(sorted((*anchors, candidate_name)))
            solved = reinvestment.solve_current_roster(
                sim,
                predictions,
                current_values,
                market,
                top_n,
                proposed_anchors,
            )
            attempts += 1
            if solved is None:
                continue
            refined, local_refine = reinvestment.refine_current_roster_fixed(
                predictions,
                solved,
                proposed_anchors,
                reference_weekly,
                reference_decisions,
                reference_played,
                market,
                top_n,
                waiver_baseline,
            )
            starters = keeper.fast_nominal_starter_mask(
                refined, current_ppg, positions, players
            )
            candidate_bench = refined & ~starters
            if any(
                not candidate_bench[np.flatnonzero(players == player)[0]]
                for player in proposed_anchors
            ):
                continue
            gate = construction_metrics(
                predictions,
                refined,
                gate_weekly,
                gate_decisions,
                gate_played,
                waiver_baseline,
                gate_cache,
            )
            if gate["mean"] < baseline_gate["mean"] - 1e-6:
                continue
            if gate["p10"] < baseline_gate["p10"] - 1e-6:
                continue
            utility = portfolio.portfolio_utility(
                np.flatnonzero(candidate_bench), surplus_draws, "expected_best"
            )
            if utility <= current_utility + 1e-8:
                continue
            candidate = (
                float(utility),
                float(gate["mean"]),
                float(gate["p10"]),
                candidate_name,
                refined,
                int(local_refine),
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate
        if best is None:
            break
        _, _, _, candidate_name, selected, local_refine = best
        anchors = tuple(sorted((*anchors, candidate_name)))
        refine_swaps += local_refine
        accepted += 1

    return selected, anchors, {
        "accepted_option_additions": accepted,
        "candidate_attempts": attempts,
        "reoptimization_refine_swaps": refine_swaps,
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
    gate_contexts: int,
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
    gate_count = min(int(gate_contexts), len(weekly))
    gate_idx = np.unique(
        np.linspace(0, len(weekly) - 1, gate_count, dtype=int)
    )
    gate_weekly = weekly[gate_idx]
    gate_decisions = decisions[gate_idx]
    gate_played = played[gate_idx]
    reference_weekly = weekly.mean(axis=0, keepdims=True)
    reference_decisions = decisions.mean(axis=0, keepdims=True)
    reference_played = np.where(
        np.any(played >= 0, axis=0),
        np.any(played > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)[None, :, :]
    current_values = value_bank.mean(axis=1)
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    current_ppg = predictions[
        FootballSimulation.sample_value_columns(predictions)
    ].mean(axis=1).to_numpy(dtype=float)
    forecast_cache: dict[tuple[str, ...], dict[str, float]] = {}
    starter_forecast_cache: dict[tuple[str, ...], dict[str, float]] = {}
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

        solved_control = reinvestment.solve_current_roster(
            sim, predictions, current_values, market, top_n, ()
        )
        if solved_control is None:
            raise RuntimeError("The current-only control was infeasible.")
        control_mask, control_refine = reinvestment.refine_current_roster_fixed(
            predictions,
            solved_control,
            (),
            reference_weekly,
            reference_decisions,
            reference_played,
            market,
            top_n,
            current_waiver,
        )
        gate_cache: dict[tuple[str, ...], dict[str, float]] = {}
        control_gate = construction_metrics(
            predictions,
            control_mask,
            gate_weekly,
            gate_decisions,
            gate_played,
            current_waiver,
            gate_cache,
        )
        soft_mask, anchors, search = optimize_soft_portfolio(
            sim,
            predictions,
            control_mask,
            control_gate,
            current_values,
            reference_weekly,
            reference_decisions,
            reference_played,
            gate_weekly,
            gate_decisions,
            gate_played,
            market,
            top_n,
            current_waiver,
            surplus_draws,
            candidate_shortlist,
            gate_cache,
        )
        search["reoptimization_refine_swaps"] += int(control_refine)
        policy_states = {
            "control": (
                control_mask,
                (),
                {
                    "accepted_option_additions": 0,
                    "candidate_attempts": 0,
                    "reoptimization_refine_swaps": int(control_refine),
                },
            ),
            "soft_portfolio": (soft_mask, anchors, search),
        }
        control_starters = keeper.fast_nominal_starter_mask(
            control_mask, current_ppg, positions, players
        )
        control_bench = control_mask & ~control_starters

        for policy in POLICIES:
            selected, policy_anchors, search_info = policy_states[policy]
            starters = keeper.fast_nominal_starter_mask(
                selected, current_ppg, positions, players
            )
            bench_mask = selected & ~starters
            if int(starters.sum()) != STARTER_COUNT or int(bench_mask.sum()) != BENCH_COUNT:
                raise AssertionError("A roster has the wrong starter/bench shape.")
            if any(
                not bench_mask[np.flatnonzero(players == player)[0]]
                for player in policy_anchors
            ):
                raise AssertionError("A search anchor did not remain on the bench.")

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
            gate = construction_metrics(
                predictions,
                selected,
                gate_weekly,
                gate_decisions,
                gate_played,
                current_waiver,
                gate_cache,
            )
            fillin = bench_fillin_metrics(
                predictions,
                selected,
                starters,
                gate_weekly,
                gate_decisions,
                gate_played,
                current_waiver,
                gate_cache,
            )
            concentration = option_concentration_metrics(
                bench_mask, surplus_draws, players
            )
            selected_prices = np.sort(market[selected])[::-1]
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "policy": policy,
                    "solve_status": "optimal",
                    "search_anchor_players": "|".join(policy_anchors),
                    **search_info,
                    "roster": "|".join(roster),
                    "nominal_starters": "|".join(starter_players),
                    "bench_players": "|".join(bench_players),
                    "bench_young_le2": int(np.sum(year_exp[bench_mask] <= 2)),
                    "bench_young_le3": int(np.sum(year_exp[bench_mask] <= 3)),
                    "bench_rookies": int(np.sum(year_exp[bench_mask] <= 0)),
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
                    "construction_mean": gate["mean"],
                    "construction_p10": gate["p10"],
                    "construction_p90": gate["p90"],
                    "construction_mean_delta": gate["mean"] - control_gate["mean"],
                    "construction_p10_delta": gate["p10"] - control_gate["p10"],
                    **fillin,
                    **concentration,
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
        "gate_context_indices": gate_idx.tolist(),
        "runtime_seconds": time.perf_counter() - started,
    }


METRICS = (
    "accepted_option_additions",
    "candidate_attempts",
    "reoptimization_refine_swaps",
    "bench_young_le2",
    "bench_young_le3",
    "bench_rookies",
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
    "construction_mean_delta",
    "construction_p10_delta",
    "bench_fillin_total",
    "bench_fillin_top2",
    "bench_fillin_second",
    "bench_positive_fillin_count",
    "option_positive_draw_rate",
    "option_effective_count",
    "option_active_count_5pct",
    "option_top_winner_share",
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
    control = trials[trials.policy.eq(BASELINE_POLICY)][
        ["year", "trial", "roster", *METRICS]
    ].copy()
    control = control.rename(
        columns={
            "roster": "roster_control",
            **{metric: f"{metric}_control" for metric in METRICS},
        }
    )
    paired = trials[trials.policy.eq("soft_portfolio")].merge(
        control, on=["year", "trial"], validate="one_to_one"
    )
    paired["roster_changed"] = paired.roster.ne(paired.roster_control)
    for metric in METRICS:
        paired[f"{metric}_effect"] = paired[metric] - paired[f"{metric}_control"]
    return by_year, across, paired


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join("" if pd.isna(value) else str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def write_summary(output_dir: Path, by_year: pd.DataFrame, paired: pd.DataFrame) -> None:
    effects = (
        paired.groupby("year", as_index=False)
        .agg(
            roster_changed_rate=("roster_changed", "mean"),
            accepted_option_additions=("accepted_option_additions", "mean"),
            option_effective_count_effect=("option_effective_count_effect", "mean"),
            active_option_count_effect=("option_active_count_5pct_effect", "mean"),
            bench_fillin_top2_effect=("bench_fillin_top2_effect", "mean"),
            starter_spend_effect=("starter_forecast_spend_effect", "mean"),
            bench_spend_effect=("bench_forecast_spend_effect", "mean"),
            starter_ev_effect=("starter_forecast_ev_effect", "mean"),
            forecast_ev_effect=("forecast_ev_effect", "mean"),
            forecast_p10_effect=("forecast_p10_effect", "mean"),
            actual_points_effect=("actual_points_effect", "mean"),
            waiver_starts_effect=("actual_waiver_starts_effect", "mean"),
            playoff_effect=("actual_playoff_points_effect", "mean"),
            predicted_best_surplus_effect=(
                "predicted_expected_best_surplus_effect", "mean"
            ),
            actual_best_surplus_effect=("actual_best_keeper_surplus_effect", "mean"),
        )
    )
    effects.to_csv(output_dir / "policy_effects_by_year.csv", index=False)
    lines = [
        "# Soft Whole-Bench Keeper Portfolio Results",
        "",
        "No age, role, or option-count quotas. The policy maximizes expected-best",
        "one-year keeper surplus across all five bench players subject to",
        "construction-bank mean and p10 protection.",
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
        "- Search is greedy and uses a bounded candidate shortlist.",
        "- Search anchors coordinate full-roster reoptimization but are not",
        "  designated keeper slots; all five final bench players receive utility.",
        "- Gate contexts are construction data. Independent evaluation contexts",
        "  and realized outcomes never enter selection.",
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
    parser.add_argument("--gate-contexts", type=int, default=50)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--candidate-shortlist", type=int, default=6)
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
        args.gate_contexts,
        args.projection_draws,
        args.candidate_shortlist,
    ) <= 0:
        parser.error("Trial, context, gate, draw, and shortlist counts must be positive.")
    if args.gate_contexts > args.contexts:
        parser.error("--gate-contexts cannot exceed --contexts.")
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
            if set(counts) == set(POLICIES) and all(
                count == args.trials for count in counts.values()
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
        "policies": list(POLICIES),
        "selection_contract": {
            "keeper_objective": "expected best surplus across all five bench players",
            "keeper_escalation": KEEPER_ESCALATION,
            "hard_option_count": None,
            "age_or_role_quotas": False,
            "construction_mean_tolerance": 0.0,
            "construction_p10_tolerance": 0.0,
            "full_roster_reoptimization": True,
        },
        "reinvestment_runner": str(REINVESTMENT_RUNNER),
        "reinvestment_runner_sha256": base.sha256_file(REINVESTMENT_RUNNER),
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
                year, conn, target_features, args.projection_draws, args.seed
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
        value_bank = reinvestment.current_value_bank(
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
            value_bank,
            next_draws,
            validation_match,
            current_waiver,
            observed_prices,
            future_ppg,
            available_horizons,
            year_exp,
            args.trials,
            args.gate_contexts,
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
        raise AssertionError("Duplicate year/trial/policy key.")
    if not trials.roster.str.split("|").map(len).eq(ROSTER_SIZE).all():
        raise AssertionError("A roster does not contain 13 players.")
    if not trials.nominal_starters.str.split("|").map(len).eq(STARTER_COUNT).all():
        raise AssertionError("A roster does not contain eight nominal starters.")
    if not trials.bench_players.str.split("|").map(len).eq(BENCH_COUNT).all():
        raise AssertionError("A roster does not contain five bench players.")
    if not (trials.forecast_salary_spend <= SALARY_CAP + 1e-8).all():
        raise AssertionError("A roster exceeds the forecast salary cap.")
    soft = trials[trials.policy.eq("soft_portfolio")]
    if soft.construction_mean_delta.min() < -1e-5:
        raise AssertionError("The soft policy violated its construction mean gate.")
    if soft.construction_p10_delta.min() < -1e-5:
        raise AssertionError("The soft policy violated its construction p10 gate.")
    if soft.accepted_option_additions.max() > BENCH_COUNT:
        raise AssertionError("The search accepted more anchors than bench slots.")

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
        "all_construction_mean_gates": True,
        "all_construction_p10_gates": True,
        "no_hard_option_count_below_physical_bench_size": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Study complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
