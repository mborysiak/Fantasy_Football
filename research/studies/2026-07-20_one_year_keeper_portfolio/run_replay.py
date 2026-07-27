"""Replay a bench-local, one-year keeper portfolio objective."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
PRIOR_STUDY = ROOT / "research" / "studies" / "2026-07-19_keeper_option_bonus"
PRIOR_RUNNER = PRIOR_STUDY / "run_replay.py"
PRIOR_RESULTS = PRIOR_STUDY / "results" / "roster_trials.csv"
VALIDATIONS_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


keeper = load_module("keeper_option_replay", PRIOR_RUNNER)
base = keeper.base
bench = keeper.bench
FootballSimulation = base.FootballSimulation

POSITIONS = keeper.POSITIONS
LINEUP_REQUIRE = keeper.LINEUP_REQUIRE
POS_MIN = keeper.POS_MIN
POS_MAX = keeper.POS_MAX
ROSTER_SIZE = keeper.ROSTER_SIZE
SALARY_CAP = keeper.SALARY_CAP
NUM_TEAMS = keeper.NUM_TEAMS
TOTAL_MARKET_BUDGET = keeper.TOTAL_MARKET_BUDGET
TOTAL_MARKET_SLOTS = keeper.TOTAL_MARKET_SLOTS
TOP_N = keeper.TOP_N
KEEPER_ESCALATION = 10.0
RESID_COLS = [
    "pred_resid_5",
    "pred_resid_10",
    "pred_resid_25",
    "pred_resid_75",
    "pred_resid_90",
    "pred_resid_95",
]
RESID_PROBS = np.array([0.00, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 1.00])


@dataclass(frozen=True)
class Policy:
    name: str
    current_tolerance: float
    option_objective: str
    max_bench_swaps: int


POLICIES = (
    Policy("control", 0.0, "none", 0),
    Policy("best1_lex0", 0.0, "expected_best", 2),
    Policy("best1_lex2", 2.0, "expected_best", 2),
)
BASELINE_POLICY = "control"
VALIDATION_MODEL_SPEC = 2026
PLAYER_RESIDUAL_RHO = 0.25


def validation_next_year_draws(
    year: int,
    forecast: pd.DataFrame,
    num_draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load current-method historical next predictions with causal intervals."""
    select_cols = ", ".join(RESID_COLS)
    with sqlite3.connect(VALIDATIONS_DB) as conn:
        rows = pd.read_sql_query(
            f"""
            SELECT player, pos, pred_fp_per_game, {select_cols},
                   resid_calibration_available,
                   resid_training_through_origin,
                   resid_training_through_season,
                   resid_target_available
            FROM Model_Validations_Resid
            WHERE version='beta'
              AND year=?
              AND season=?
              AND current_or_next_year='next'
              AND rush_pass NOT IN ('rush', 'pass', 'rec')
              AND dataset NOT LIKE '%Rookie%'
            """,
            conn,
            params=(VALIDATION_MODEL_SPEC, year),
        )
    rows = base.add_identity(rows)
    rows = rows[rows.pos.isin(POSITIONS)].copy()
    numeric = [
        "pred_fp_per_game",
        *RESID_COLS,
        "resid_training_through_origin",
        "resid_training_through_season",
    ]
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows[rows.resid_calibration_available.eq(1)].copy()
    if len(rows) == 0:
        raise ValueError(f"No calibrated next validation rows found for {year}.")
    if rows.resid_training_through_season.max() > year - 1:
        raise AssertionError("Next residual calibration borrowed a future season.")

    aggregate = (
        rows.groupby(["player_key", "pos"], as_index=False)
        .agg(
            next_mean=("pred_fp_per_game", "mean"),
            next_source_rows=("player", "size"),
            training_through_origin=("resid_training_through_origin", "max"),
            training_through_season=("resid_training_through_season", "max"),
            target_available=("resid_target_available", "max"),
            **{column: (column, "mean") for column in RESID_COLS},
        )
    )
    quantiles = aggregate[RESID_COLS].to_numpy(dtype=float)
    aggregate[RESID_COLS] = np.maximum.accumulate(quantiles, axis=1)

    aligned = forecast[
        ["player", "player_key", "pos", "pred_fp_per_game"]
    ].merge(
        aggregate,
        on=["player_key", "pos"],
        how="left",
        validate="one_to_one",
    )
    validation_match = (
        aligned.next_mean.notna() & aligned[RESID_COLS].notna().all(axis=1)
    ).to_numpy(dtype=bool)

    global_fallback = aggregate[RESID_COLS].median().to_numpy(dtype=float)
    residuals = aligned[RESID_COLS].to_numpy(dtype=float)
    for pos in POSITIONS:
        target = aligned.pos.eq(pos).to_numpy()
        source = aggregate[aggregate.pos.eq(pos)]
        fallback = (
            source[RESID_COLS].median().to_numpy(dtype=float)
            if len(source)
            else global_fallback
        )
        missing = target & ~np.isfinite(residuals).all(axis=1)
        residuals[missing] = fallback
    if not np.isfinite(residuals).all():
        raise AssertionError("A proxy next-year residual vector is missing.")
    residuals = np.maximum.accumulate(residuals, axis=1)

    current = aligned.pred_fp_per_game.to_numpy(dtype=float)
    means = aligned.next_mean.to_numpy(dtype=float)
    means = np.where(np.isfinite(means), means, current)

    rng = np.random.default_rng(seed + 44_000 + year)
    common = rng.standard_normal((1, num_draws))
    independent = rng.standard_normal((len(aligned), num_draws))
    latent = (
        np.sqrt(PLAYER_RESIDUAL_RHO) * common
        + np.sqrt(1.0 - PLAYER_RESIDUAL_RHO) * independent
    )
    uniform = ndtr(latent)

    q5, q10, q25, q75, q90, q95 = residuals.T
    q0 = 2.0 * q5 - q10
    q100 = 2.0 * q95 - q90
    knots = np.maximum.accumulate(
        np.column_stack([q0, q5, q10, q25, q75, q90, q95, q100]),
        axis=1,
    )
    knot_idx = np.searchsorted(RESID_PROBS, uniform, side="right") - 1
    knot_idx = np.clip(knot_idx, 0, len(RESID_PROBS) - 2)
    left_prob = RESID_PROBS[knot_idx]
    right_prob = RESID_PROBS[knot_idx + 1]
    left = np.take_along_axis(knots, knot_idx, axis=1)
    right = np.take_along_axis(knots, knot_idx + 1, axis=1)
    weight = (uniform - left_prob) / (right_prob - left_prob)
    draws = np.maximum(means[:, None] + left + weight * (right - left), 0.0)

    metadata = {
        "next_projection_source": "Model_Validations_Resid.next_noncomponent",
        "validation_model_spec_asof_year": VALIDATION_MODEL_SPEC,
        "validation_matches": int(validation_match.sum()),
        "current_projection_proxies": int((~validation_match).sum()),
        "player_residual_copula_rho": PLAYER_RESIDUAL_RHO,
        "resid_training_through_origin": int(
            rows.resid_training_through_origin.max()
        ),
        "resid_training_through_season": int(
            rows.resid_training_through_season.max()
        ),
        "realized_validation_targets": int(rows.resid_target_available.sum()),
        "validation_db": str(VALIDATIONS_DB),
        "validation_db_sha256": base.sha256_file(VALIDATIONS_DB),
    }
    return draws.astype(np.float32), validation_match, metadata


def first_year_surplus_draws(
    future_market_values: np.ndarray,
    acquisition_prices: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        np.asarray(future_market_values, dtype=float)
        - (np.asarray(acquisition_prices, dtype=float)[:, None] + KEEPER_ESCALATION),
        0.0,
    )


def portfolio_utility(
    bench_indices: np.ndarray,
    surplus_draws: np.ndarray,
    objective: str,
) -> float:
    indices = np.asarray(bench_indices, dtype=int)
    if len(indices) == 0:
        return 0.0
    best = np.max(surplus_draws[indices], axis=0)
    if objective == "expected_best":
        return float(best.mean())
    if objective == "probability_10":
        return float(np.mean(best >= 10.0))
    if objective == "none":
        return 0.0
    raise ValueError(f"Unknown keeper portfolio objective: {objective}")


def expected_top_two_utility(
    bench_indices: np.ndarray,
    surplus_draws: np.ndarray,
) -> float:
    values = surplus_draws[np.asarray(bench_indices, dtype=int)]
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values.mean())
    partition = np.partition(values, -2, axis=0)[-2:]
    return float(partition.sum(axis=0).mean())


def transform_ppg_to_market(
    ppg_draws: np.ndarray,
    positions: np.ndarray,
    curves: dict[str, Any],
) -> np.ndarray:
    output = np.zeros_like(ppg_draws, dtype=float)
    for pos, curve in curves.items():
        mask = positions == pos
        output[mask] = curve.predict(ppg_draws[mask].reshape(-1)).reshape(
            int(mask.sum()), ppg_draws.shape[1]
        )
    return output


def exact_construction_score(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    weekly_scores: np.ndarray,
    decision_scores: np.ndarray,
    played_mask: np.ndarray,
    waiver_baseline: dict[str, float],
) -> float:
    scores, _ = FootballSimulation.managed_lineup_multi_context_scores(
        weekly_scores[:, selected_mask, :],
        predictions.loc[selected_mask, "pos"].to_numpy(),
        decision_scores[:, selected_mask, :],
        predictions.loc[selected_mask, "player"].to_numpy(),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=waiver_baseline,
        played_mask=played_mask[:, selected_mask, :],
    )
    return float(np.mean(scores))


def refine_bench_portfolio(
    predictions: pd.DataFrame,
    baseline_mask: np.ndarray,
    protected_starters: np.ndarray,
    weekly_scores: np.ndarray,
    decision_scores: np.ndarray,
    played_mask: np.ndarray,
    salary_values: np.ndarray,
    top_n: list[str],
    waiver_baseline: dict[str, float],
    surplus_draws: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.asarray(baseline_mask, dtype=bool).copy()
    protected = np.asarray(protected_starters, dtype=bool)
    salary_values = np.asarray(salary_values, dtype=float)
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    top_n_set = set(top_n)
    baseline_score = exact_construction_score(
        predictions,
        selected,
        weekly_scores,
        decision_scores,
        played_mask,
        waiver_baseline,
    )
    current_score = baseline_score
    bench_indices = np.flatnonzero(selected & ~protected)
    current_utility = portfolio_utility(
        bench_indices, surplus_draws, policy.option_objective
    )
    floor = baseline_score - policy.current_tolerance
    accepted = 0

    ref_weekly = weekly_scores.mean(axis=0)
    ref_decisions = decision_scores.mean(axis=0)
    ref_played = np.where(
        np.any(played_mask >= 0, axis=0),
        np.any(played_mask > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)

    for _ in range(policy.max_bench_swaps):
        out_indices = np.flatnonzero(selected & ~protected)
        base_masks = []
        for out_idx in out_indices:
            mask = selected.copy()
            mask[out_idx] = False
            base_masks.append(mask)
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
        best: tuple[float, float, int, int] | None = None
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

            other_bench = np.flatnonzero(base_mask & ~protected)
            if len(other_bench):
                other_best = np.max(surplus_draws[other_bench], axis=0)
            else:
                other_best = np.zeros(surplus_draws.shape[1], dtype=float)
            candidate_best = np.maximum(
                other_best[None, :], surplus_draws[eligible_idx]
            )
            if policy.option_objective == "expected_best":
                candidate_utility = candidate_best.mean(axis=1)
            elif policy.option_objective == "probability_10":
                candidate_utility = np.mean(candidate_best >= 10.0, axis=1)
            else:
                raise ValueError(policy.option_objective)
            approximate_score = (
                current_score - float(incoming[out_idx]) + incoming[eligible_idx]
            )
            approximate_ok = approximate_score >= floor - 5.0
            if not approximate_ok.any():
                continue
            available_order = np.flatnonzero(approximate_ok)
            utility_order = available_order[
                np.lexsort(
                    (
                        eligible_idx[available_order],
                        -candidate_utility[available_order],
                    )
                )[: min(15, len(available_order))]
            ]
            current_order = available_order[
                np.lexsort(
                    (
                        eligible_idx[available_order],
                        -approximate_score[available_order],
                    )
                )[: min(5, len(available_order))]
            ]
            shortlist = np.unique(np.concatenate([utility_order, current_order]))
            for local_idx in shortlist:
                utility = float(candidate_utility[local_idx])
                if utility <= current_utility + 1e-8:
                    continue
                in_idx = int(eligible_idx[local_idx])
                replacement = base_mask.copy()
                replacement[in_idx] = True
                exact_score = exact_construction_score(
                    predictions,
                    replacement,
                    weekly_scores,
                    decision_scores,
                    played_mask,
                    waiver_baseline,
                )
                if exact_score < floor - 1e-6:
                    continue
                candidate = (utility, float(exact_score), int(out_idx), in_idx)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is None:
            break
        current_utility, current_score, out_idx, in_idx = best
        selected[out_idx] = False
        selected[in_idx] = True
        accepted += 1

    if not np.all(selected[protected]):
        raise AssertionError("A protected starter was removed by keeper refinement.")
    return selected, {
        "accepted_swaps": accepted,
        "baseline_construction_score": float(baseline_score),
        "current_construction_score": float(current_score),
        "current_construction_delta": float(current_score - baseline_score),
        "portfolio_construction_utility": float(current_utility),
    }


def realized_keeper_metrics(
    bench_mask: np.ndarray,
    surplus_draws: np.ndarray,
    validation_match: np.ndarray,
    future_ppg: np.ndarray,
    available_horizons: list[int],
    positions: np.ndarray,
    curves: dict[str, Any],
    market: np.ndarray,
    observed_prices: np.ndarray,
    players: np.ndarray,
) -> dict[str, Any]:
    bench_idx = np.flatnonzero(bench_mask)
    individual_expected = surplus_draws.mean(axis=1)
    predicted_idx = bench_idx[np.argmax(individual_expected[bench_idx])]
    predicted_best_draw = np.max(surplus_draws[bench_idx], axis=0)
    output: dict[str, Any] = {
        "predicted_expected_best_surplus": float(predicted_best_draw.mean()),
        "predicted_probability_any_hit": float(np.mean(predicted_best_draw > 0)),
        "predicted_probability_any_10": float(
            np.mean(predicted_best_draw >= 10.0)
        ),
        "predicted_probability_any_20": float(
            np.mean(predicted_best_draw >= 20.0)
        ),
        "predicted_expected_top2_surplus": expected_top_two_utility(
            bench_idx, surplus_draws
        ),
        "predicted_best_player": str(players[predicted_idx]),
        "next_validation_bench_matches": int(validation_match[bench_idx].sum()),
        "next_projection_proxy_bench_count": int((~validation_match[bench_idx]).sum()),
        "future_outcome_available": int(bool(available_horizons)),
    }
    unavailable = {
        "actual_best_keeper_surplus": np.nan,
        "actual_top2_keeper_surplus": np.nan,
        "actual_any_keeper_hit": np.nan,
        "actual_any_keeper_hit_10": np.nan,
        "actual_any_keeper_hit_20": np.nan,
        "actual_best_future_ppg": np.nan,
        "actual_future_ppg_ge12_count": np.nan,
        "actual_future_ppg_ge15_count": np.nan,
        "actual_keeper_hit_players": "",
        "observed_cost_coverage": np.nan,
        "observed_best_keeper_surplus": np.nan,
        "observed_any_keeper_hit_10": np.nan,
    }
    if not available_horizons:
        output.update(unavailable)
        return output

    actual_ppg = future_ppg[:, 0]
    actual_values = np.zeros(len(actual_ppg), dtype=float)
    for pos, curve in curves.items():
        mask = positions == pos
        actual_values[mask] = curve.predict(actual_ppg[mask])
    modeled_surplus = np.maximum(
        actual_values - (market + KEEPER_ESCALATION), 0.0
    )
    bench_surplus = modeled_surplus[bench_idx]
    top = np.sort(bench_surplus)[::-1]
    hit_idx = bench_idx[bench_surplus > 0]
    observed_valid = np.isfinite(observed_prices[bench_idx])
    observed_surplus = np.where(
        observed_valid,
        np.maximum(
            actual_values[bench_idx]
            - (observed_prices[bench_idx] + KEEPER_ESCALATION),
            0.0,
        ),
        np.nan,
    )
    output.update(
        {
            "actual_best_keeper_surplus": float(top[0]) if len(top) else 0.0,
            "actual_top2_keeper_surplus": float(top[:2].sum()),
            "actual_any_keeper_hit": int(np.any(bench_surplus > 0)),
            "actual_any_keeper_hit_10": int(np.any(bench_surplus >= 10.0)),
            "actual_any_keeper_hit_20": int(np.any(bench_surplus >= 20.0)),
            "actual_best_future_ppg": float(actual_ppg[bench_idx].max()),
            "actual_future_ppg_ge12_count": int(
                np.sum(actual_ppg[bench_idx] >= 12.0)
            ),
            "actual_future_ppg_ge15_count": int(
                np.sum(actual_ppg[bench_idx] >= 15.0)
            ),
            "actual_keeper_hit_players": "|".join(sorted(players[hit_idx])),
            "observed_cost_coverage": float(observed_valid.mean()),
            "observed_best_keeper_surplus": float(np.nanmax(observed_surplus))
            if observed_valid.any()
            else np.nan,
            "observed_any_keeper_hit_10": int(
                np.nanmax(observed_surplus) >= 10.0
            )
            if observed_valid.any()
            else np.nan,
        }
    )
    return output


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
    next_draws: np.ndarray,
    validation_match: np.ndarray,
    current_waiver: dict[str, float],
    observed_prices: np.ndarray,
    future_ppg: np.ndarray,
    available_horizons: list[int],
    control_rows: pd.DataFrame,
    trials: int,
    context_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    remaining_budget = TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = TOTAL_MARKET_SLOTS - environment["keeper_count"]
    top_n = predictions.nlargest(min(TOP_N, len(predictions)), "salary").player.tolist()
    rng = np.random.default_rng(seed + year * 101)
    salary_plan = rng.integers(0, salary_draws.shape[1], size=(trials, 5))
    context_plan = rng.integers(
        0, weekly.shape[0], size=(trials, context_draws)
    )
    raw_market = np.column_stack(
        [salary_draws[:, row].mean(axis=1) for row in salary_plan]
    )
    markets = base.normalize_market_draws(
        sim,
        raw_market,
        remaining_budget,
        remaining_slots,
    )
    control_lookup = control_rows.set_index("trial").roster.to_dict()
    forecast_cache: dict[tuple[str, ...], dict[str, float]] = {}
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    current_ppg = predictions[
        FootballSimulation.sample_value_columns(predictions)
    ].mean(axis=1).to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for trial in range(trials):
        market = markets[:, trial]
        predictions["salary"] = market
        context_idx = context_plan[trial]
        construction_weekly = weekly[context_idx]
        construction_decisions = decisions[context_idx]
        construction_played = played[context_idx]
        curves = keeper.fit_position_market_curves(
            current_ppg, market, positions
        )
        future_market = transform_ppg_to_market(next_draws, positions, curves)
        surplus_draws = first_year_surplus_draws(future_market, market)
        surplus_draws[positions == "QB"] = 0.0

        roster_names = set(str(control_lookup[trial]).split("|"))
        baseline_mask = np.isin(players, list(roster_names))
        if int(baseline_mask.sum()) != ROSTER_SIZE:
            raise AssertionError("Stored control roster did not align to candidates.")
        protected = bench.nominal_starter_mask(predictions, baseline_mask)
        if int(protected.sum()) != int(sum(LINEUP_REQUIRE.values())):
            raise AssertionError("The protected starter set is not eight players.")
        baseline_score = exact_construction_score(
            predictions,
            baseline_mask,
            construction_weekly,
            construction_decisions,
            construction_played,
            current_waiver,
        )

        for policy in POLICIES:
            if policy.name == BASELINE_POLICY:
                selected = baseline_mask.copy()
                bench_idx = np.flatnonzero(selected & ~protected)
                refine_info = {
                    "accepted_swaps": 0,
                    "baseline_construction_score": float(baseline_score),
                    "current_construction_score": float(baseline_score),
                    "current_construction_delta": 0.0,
                    "portfolio_construction_utility": portfolio_utility(
                        bench_idx, surplus_draws, "expected_best"
                    ),
                }
            else:
                selected, refine_info = refine_bench_portfolio(
                    predictions,
                    baseline_mask,
                    protected,
                    construction_weekly,
                    construction_decisions,
                    construction_played,
                    market,
                    top_n,
                    current_waiver,
                    surplus_draws,
                    policy,
                )
            bench_mask = selected & ~protected
            if int(bench_mask.sum()) != ROSTER_SIZE - int(sum(LINEUP_REQUIRE.values())):
                raise AssertionError("A refined roster does not have five bench slots.")
            roster = tuple(sorted(players[selected]))
            bench_players = tuple(sorted(players[bench_mask]))
            protected_players = tuple(sorted(players[protected]))
            projected = bench.forecast_metrics(
                predictions,
                roster,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                current_waiver,
                forecast_cache,
            )
            actual_metrics = bench.actual_policy_metrics(
                environment,
                roster,
                bench_players,
            )
            keeper_metrics = realized_keeper_metrics(
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
            selected_prices = np.sort(market[selected])[::-1]
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "policy": policy.name,
                    "solve_status": "optimal",
                    "current_tolerance": policy.current_tolerance,
                    "option_objective": policy.option_objective,
                    "roster": "|".join(roster),
                    "protected_starters": "|".join(protected_players),
                    "bench_players": "|".join(bench_players),
                    "forecast_salary_spend": float(market[selected].sum()),
                    "starter_forecast_spend": float(market[protected].sum()),
                    "bench_forecast_spend": float(market[bench_mask].sum()),
                    "top3_spend_share": float(
                        selected_prices[:3].sum() / market[selected].sum()
                    ),
                    **refine_info,
                    **projected,
                    **actual_metrics,
                    **keeper_metrics,
                }
            )
        if (trial + 1) % max(1, min(25, trials)) == 0:
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
    "forecast_ev",
    "forecast_p10",
    "forecast_p90",
    "actual_points",
    "actual_playoff_points",
    "bench_forecast_spend",
    "top3_spend_share",
    "current_construction_delta",
    "accepted_swaps",
    "predicted_expected_best_surplus",
    "predicted_probability_any_hit",
    "predicted_probability_any_10",
    "predicted_probability_any_20",
    "predicted_expected_top2_surplus",
    "next_validation_bench_matches",
    "next_projection_proxy_bench_count",
    "actual_best_keeper_surplus",
    "actual_top2_keeper_surplus",
    "actual_any_keeper_hit",
    "actual_any_keeper_hit_10",
    "actual_any_keeper_hit_20",
    "actual_best_future_ppg",
    "actual_future_ppg_ge12_count",
    "actual_future_ppg_ge15_count",
    "observed_cost_coverage",
    "observed_best_keeper_surplus",
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
    baseline = trials[trials.policy.eq(BASELINE_POLICY)][
        ["year", "trial", "roster", *METRICS]
    ]
    paired_rows = []
    for policy in [p.name for p in POLICIES if p.name != BASELINE_POLICY]:
        candidate = trials[trials.policy.eq(policy)][
            ["year", "trial", "roster", *METRICS]
        ]
        paired = baseline.merge(
            candidate,
            on=["year", "trial"],
            suffixes=("_control", "_candidate"),
            validate="one_to_one",
        )
        paired["policy"] = policy
        paired["roster_changed"] = paired.roster_control.ne(
            paired.roster_candidate
        )
        for metric in METRICS:
            paired[f"{metric}_effect"] = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_control"]
            )
        paired_rows.append(paired)
    paired = pd.concat(paired_rows, ignore_index=True)
    return by_year, across, paired


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
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
            **{
                f"{metric}_effect": (f"{metric}_effect", "mean")
                for metric in METRICS
            },
        )
    )
    columns = [
        "year",
        "policy",
        "roster_changed_rate",
        "forecast_ev_effect",
        "forecast_p10_effect",
        "actual_points_effect",
        "actual_playoff_points_effect",
        "current_construction_delta_effect",
        "predicted_expected_best_surplus_effect",
        "actual_best_keeper_surplus_effect",
        "actual_any_keeper_hit_10_effect",
    ]
    lines = [
        "# One-Year Keeper Portfolio Results",
        "",
        f"{args.trials} paired trials per origin; fixed starters and at most two bench swaps.",
        "",
        "## Paired effects versus current-only control",
        "",
        markdown_table(effects[columns].round(3)),
        "",
        "## Policy means by origin",
        "",
        markdown_table(
            by_year[
                [
                    "year",
                    "policy",
                    "forecast_ev",
                    "forecast_p10",
                    "actual_points",
                    "actual_playoff_points",
                    "predicted_expected_best_surplus",
                    "actual_best_keeper_surplus",
                    "actual_any_keeper_hit_10",
                    "actual_best_future_ppg",
                ]
            ].round(3)
        ),
        "",
        "## Interpretation boundaries",
        "",
        "- One historical season is one realized outcome unit; trial counts measure construction stability.",
        "- Historical point predictions use the current 2026 model specification on OOS origin data, not a frozen old method.",
        "- Players without a dedicated next validation row use an explicit current-projection proxy.",
        "- Counterfactual modeled acquisition cost is primary; observed salary is a coverage-limited audit.",
        "- Current forecast evaluation never includes keeper surplus as fantasy points.",
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
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(base.FROZEN_SOURCES))
    if invalid:
        parser.error(f"Unsupported years: {invalid}")
    if min(
        args.trials, args.contexts, args.context_draws, args.projection_draws
    ) <= 0:
        parser.error("Trials, contexts, and projection draws must be positive.")
    if args.trials > 250:
        parser.error("The stored paired control contains at most 250 trials per year.")
    if args.salary_draws != 5000 or args.seed != 20260720:
        parser.error(
            "Stored control rosters require --salary-draws 5000 and "
            "--seed 20260720 so the trial markets remain paired."
        )
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
    control_all = pd.read_csv(PRIOR_RESULTS)
    control_all = control_all[control_all.policy.eq("keeper_engine0")].copy()
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
                set(counts) == {p.name for p in POLICIES}
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
        "keeper_contract": {
            "primary_keeper_slots": 1,
            "league_keeper_slots": 2,
            "first_year_escalation": KEEPER_ESCALATION,
            "future_years_scored": 1,
            "portfolio_slots": ROSTER_SIZE - int(sum(LINEUP_REQUIRE.values())),
        },
        "prior_study": str(PRIOR_STUDY),
        "prior_runner_sha256": base.sha256_file(PRIOR_RUNNER),
        "prior_results_sha256": base.sha256_file(PRIOR_RESULTS),
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
            raise AssertionError("A construction template crossed the origin.")
        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game", "salary"]
        ].copy()
        sim = base.make_simulation(year, player_data, cache)
        current_waiver = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )

        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_idx].reset_index(drop=True)
        predictions = base.build_predictions(
            candidate_forecast,
            ppg_draws[candidate_idx],
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
        next_draws, validation_match, next_meta = validation_next_year_draws(
            year,
            candidate_forecast,
            args.projection_draws,
            args.seed,
        )
        future_ppg, _, available_horizons = keeper.future_actual_ppg(
            year,
            candidate_forecast,
            raw_weekly,
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
        control_rows = control_all[
            control_all.year.eq(year) & control_all.trial.lt(args.trials)
        ].copy()
        if len(control_rows) != args.trials:
            raise AssertionError(
                f"Expected {args.trials} stored control rows for {year}, "
                f"found {len(control_rows)}."
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
            next_draws,
            validation_match,
            current_waiver,
            observed_prices,
            future_ppg,
            available_horizons,
            control_rows,
            args.trials,
            args.context_draws,
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
    if not trials.protected_starters.str.split("|").map(len).eq(
        int(sum(LINEUP_REQUIRE.values()))
    ).all():
        raise AssertionError("A roster does not retain eight protected starters.")
    if not trials.bench_players.str.split("|").map(len).eq(
        ROSTER_SIZE - int(sum(LINEUP_REQUIRE.values()))
    ).all():
        raise AssertionError("A roster does not contain five bench slots.")
    if not (trials.forecast_salary_spend <= SALARY_CAP + 1e-8).all():
        raise AssertionError("A roster exceeds the forecast salary cap.")
    if not (
        trials.current_construction_delta
        >= -trials.current_tolerance - 1e-5
    ).all():
        raise AssertionError("A policy exceeded its current-value tolerance.")
    if not (trials.accepted_swaps <= 2).all():
        raise AssertionError("A policy changed more than two bench players.")

    by_year, across, paired = summarize(trials)
    outputs = {
        "roster_trials.csv": trials,
        "policy_summary_by_year.csv": by_year,
        "policy_summary_across_years.csv": across,
        "paired_effects.csv": paired,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["verification"] = {
        "expected_rows": expected,
        "all_solves_optimal": True,
        "all_rosters_size_13": True,
        "all_starters_protected": True,
        "all_benches_size_5": True,
        "all_forecast_spend_within_cap": True,
        "all_current_tolerances_respected": True,
        "at_most_two_bench_swaps": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary(output_dir, args, by_year, paired)
    print(f"Study complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
