"""Rolling-origin replay of a price-aware bench keeper-option bonus."""

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
from cvxopt import matrix
from sklearn.isotonic import IsotonicRegression


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
BASE_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-13_managed_auction_rolling_replay"
    / "run_replay.py"
)
BENCH_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-19_bench_option_hurdle"
    / "run_replay.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load replay module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("keeper_option_base_replay", BASE_STUDY)
bench = load_module("keeper_option_bench_replay", BENCH_STUDY)

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

KEEPER_SLOTS = 2
KEEPER_ESCALATION = 10.0
MAX_KEEPER_YEARS = 3
MIN_FUTURE_GAMES = 4


@dataclass(frozen=True)
class Policy:
    name: str
    bench_weight: float
    keeper_lambda: float


POLICIES = (
    Policy("current_bench025", 0.25, 0.0),
    Policy("bench0", 0.0, 0.0),
    Policy("keeper_engine0", 0.0, 0.0),
    Policy("keeper_tiebreak", 0.0, 0.0001),
    Policy("keeper_0p01", 0.0, 0.01),
    Policy("keeper_1p0", 0.0, 1.0),
    Policy("keeper_10p0", 0.0, 10.0),
)
BASELINE_POLICY = "current_bench025"


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def load_frozen_next_year_draws(
    year: int,
    conn: sqlite3.Connection,
    forecast: pd.DataFrame,
    num_draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load origin-frozen next-year predictions without future borrowing."""
    final_columns = table_columns(conn, "Final_Predictions")
    if "pred_fp_per_game_ny" in final_columns:
        rows = pd.read_sql_query(
            """
            SELECT player, pos,
                   pred_fp_per_game_ny AS next_mean,
                   std_dev_ny AS next_std,
                   min_score_ny AS next_min,
                   max_score_ny AS next_max
            FROM Final_Predictions
            WHERE year=? AND version='beta' AND dataset='final_ensemble'
            """,
            conn,
            params=(year,),
        )
        source = "Final_Predictions.pred_fp_per_game_ny"
    else:
        model_columns = table_columns(conn, "Model_Predictions")
        select_min = "min_score" if "min_score" in model_columns else "NULL"
        rows = pd.read_sql_query(
            f"""
            SELECT player, pos, pred_fp_per_game AS next_mean,
                   std_dev AS next_std,
                   {select_min} AS next_min,
                   max_score AS next_max,
                   rush_pass
            FROM Model_Predictions
            WHERE year=? AND version='beta' AND current_or_next_year='next'
            """,
            conn,
            params=(year,),
        )
        rows = rows[~rows.rush_pass.isin(["rush", "pass", "rec"])].copy()
        source = "Model_Predictions.next_noncomponent_mixture"

    rows = base.add_identity(rows)
    rows = rows[rows.pos.isin(POSITIONS)].copy()
    for column in ["next_mean", "next_std", "next_min", "next_max"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        means = group.next_mean.dropna().to_numpy(dtype=float)
        stds = group.next_std.dropna().to_numpy(dtype=float)
        mean = float(np.mean(means)) if len(means) else np.nan
        if len(means):
            within = float(np.mean(np.square(stds))) if len(stds) else 0.0
            between = float(np.var(means))
            std = float(np.sqrt(max(within + between, 0.0)))
        else:
            std = np.nan
        lower = group.next_min.dropna().to_numpy(dtype=float)
        upper = group.next_max.dropna().to_numpy(dtype=float)
        return pd.Series(
            {
                "next_mean": mean,
                "next_std": std,
                "next_min": float(np.mean(lower)) if len(lower) else np.nan,
                "next_max": float(np.mean(upper)) if len(upper) else np.nan,
                "next_source_rows": int(len(group)),
            }
        )

    rows = (
        rows.groupby(["player_key", "pos"], as_index=False)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    if "level_0" in rows:
        rows = rows.drop(columns="level_0")
    if "level_1" in rows:
        rows = rows.drop(columns="level_1")

    aligned = forecast[["player_key", "pos", "pred_fp_per_game"]].merge(
        rows,
        on=["player_key", "pos"],
        how="left",
        validate="one_to_one",
    )
    available = aligned.next_mean.notna().to_numpy(dtype=bool)
    current = aligned.pred_fp_per_game.to_numpy(dtype=float)
    means = aligned.next_mean.fillna(aligned.pred_fp_per_game).to_numpy(dtype=float)
    stds = aligned.next_std.to_numpy(dtype=float)
    fallback_std = np.maximum(0.25 * means, 1.0)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, fallback_std)
    mins = aligned.next_min.to_numpy(dtype=float)
    mins = np.where(np.isfinite(mins), mins, np.maximum(means - 2.0 * stds, 0.0))
    maxs = aligned.next_max.to_numpy(dtype=float)
    maxs = np.where(np.isfinite(maxs), maxs, means + 2.0 * stds)
    maxs = np.maximum(maxs, mins + 0.01)
    draws = base.legacy_truncnorm_draws(
        means,
        stds,
        mins,
        maxs,
        num_draws=num_draws,
        seed=seed + 31_000 + year,
        floor=0.0,
    )
    draws[~available] = current[~available, None]
    metadata = {
        "next_projection_source": source,
        "next_projection_available": int(available.sum()),
        "next_projection_missing": int((~available).sum()),
        "next_projection_mean_diff": float(np.mean(means[available] - current[available]))
        if available.any()
        else 0.0,
    }
    return draws, available, metadata


def fit_position_market_curves(
    current_ppg: np.ndarray,
    market_salary: np.ndarray,
    positions: np.ndarray,
) -> dict[str, IsotonicRegression]:
    curves: dict[str, IsotonicRegression] = {}
    for pos in POSITIONS:
        mask = (
            (positions == pos)
            & np.isfinite(current_ppg)
            & np.isfinite(market_salary)
        )
        x = np.asarray(current_ppg[mask], dtype=float)
        y = np.asarray(market_salary[mask], dtype=float)
        if len(x) < 2 or len(np.unique(x)) < 2:
            raise ValueError(f"Insufficient {pos} rows for a market-value curve.")
        curves[pos] = IsotonicRegression(
            increasing=True,
            y_min=1.0,
            y_max=float(max(np.max(y), 1.0)),
            out_of_bounds="clip",
        ).fit(x, y)
    return curves


def market_value_draws(
    draws: np.ndarray,
    positions: np.ndarray,
    curves: dict[str, IsotonicRegression],
) -> np.ndarray:
    values = np.empty_like(draws, dtype=np.float32)
    for pos, curve in curves.items():
        mask = positions == pos
        values[mask] = curve.predict(draws[mask].reshape(-1)).reshape(
            int(mask.sum()), draws.shape[1]
        )
    return values


def keeper_contract_surplus(
    future_market_values: np.ndarray,
    acquisition_prices: np.ndarray,
    escalation: float = KEEPER_ESCALATION,
    max_years: int = MAX_KEEPER_YEARS,
) -> np.ndarray:
    """Return draw-level surplus for a persistent hit over the keeper term."""
    values = np.asarray(future_market_values, dtype=float)
    prices = np.asarray(acquisition_prices, dtype=float).reshape(-1, 1)
    if values.ndim != 2 or len(values) != len(prices):
        raise ValueError("Future values and acquisition prices must align.")
    surplus = np.zeros_like(values, dtype=float)
    for keep_year in range(1, int(max_years) + 1):
        surplus += np.maximum(values - (prices + escalation * keep_year), 0.0)
    return surplus


def keeper_option_values(
    predictions: pd.DataFrame,
    next_draws: np.ndarray,
    next_available: np.ndarray,
    market: np.ndarray,
    impact_thresholds: dict[str, float],
) -> tuple[np.ndarray, dict[str, IsotonicRegression], np.ndarray]:
    sample_columns = FootballSimulation.sample_value_columns(predictions)
    current_ppg = predictions[sample_columns].mean(axis=1).to_numpy(dtype=float)
    positions = predictions.pos.to_numpy()
    curves = fit_position_market_curves(current_ppg, market, positions)
    future_values = market_value_draws(next_draws, positions, curves)
    options = keeper_contract_surplus(future_values, market).mean(axis=1)
    eligible = np.array(
        [
            bool(
                pos != "QB"
                and next_available[idx]
                and current_ppg[idx] < float(impact_thresholds[pos])
            )
            for idx, pos in enumerate(positions)
        ],
        dtype=bool,
    )
    options = np.where(eligible, options, 0.0)
    return options.astype(np.float32), curves, eligible


def top_keeper_utility(
    selected_mask: np.ndarray,
    option_values: np.ndarray,
    slots: int = KEEPER_SLOTS,
    candidate_mask: np.ndarray | None = None,
) -> float:
    eligible = np.asarray(selected_mask, dtype=bool).copy()
    if candidate_mask is not None:
        eligible &= np.asarray(candidate_mask, dtype=bool)
    values = np.asarray(option_values, dtype=float)[eligible]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 0.0
    take = min(int(slots), len(values))
    return float(np.partition(values, len(values) - take)[-take:].sum())


def top_keeper_indices(
    selected_mask: np.ndarray,
    option_values: np.ndarray,
    slots: int = KEEPER_SLOTS,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    eligible = np.asarray(selected_mask, dtype=bool).copy()
    if candidate_mask is not None:
        eligible &= np.asarray(candidate_mask, dtype=bool)
    idx = np.flatnonzero(
        eligible
        & np.isfinite(option_values)
        & (np.asarray(option_values) > 0)
    )
    if len(idx) == 0:
        return idx
    order = np.lexsort((idx, -np.asarray(option_values)[idx]))
    return idx[order[: int(slots)]]


def roster_keeper_utility(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    option_values: np.ndarray,
) -> float:
    starter_mask = bench.nominal_starter_mask(predictions, selected_mask)
    nominal_bench = np.asarray(selected_mask, dtype=bool) & ~starter_mask
    return top_keeper_utility(
        selected_mask,
        option_values,
        candidate_mask=nominal_bench,
    )


def fast_nominal_starter_mask(
    selected_mask: np.ndarray,
    current_ppg: np.ndarray,
    positions: np.ndarray,
    player_names: np.ndarray,
) -> np.ndarray:
    """Array-only equivalent of the nominal preseason lineup selector."""
    selected_mask = np.asarray(selected_mask, dtype=bool)
    remaining = selected_mask.copy()
    starters = np.zeros_like(selected_mask)

    def choose(candidate_mask: np.ndarray, count: int) -> np.ndarray:
        idx = np.flatnonzero(candidate_mask)
        if count <= 0 or len(idx) == 0:
            return np.empty(0, dtype=int)
        order = np.lexsort((player_names[idx], -current_ppg[idx]))
        return idx[order[:count]]

    for pos in POSITIONS:
        chosen = choose(
            remaining & (positions == pos),
            int(LINEUP_REQUIRE.get(pos, 0)),
        )
        starters[chosen] = True
        remaining[chosen] = False
    flex = choose(
        remaining & np.isin(positions, ["RB", "WR", "TE"]),
        int(LINEUP_REQUIRE.get("FLEX", 0)),
    )
    starters[flex] = True
    return starters


def fast_roster_keeper_utility(
    selected_mask: np.ndarray,
    option_values: np.ndarray,
    current_ppg: np.ndarray,
    positions: np.ndarray,
    player_names: np.ndarray,
) -> float:
    starters = fast_nominal_starter_mask(
        selected_mask,
        current_ppg,
        positions,
        player_names,
    )
    nominal_bench = np.asarray(selected_mask, dtype=bool) & ~starters
    return top_keeper_utility(
        selected_mask,
        option_values,
        candidate_mask=nominal_bench,
    )


def solve_keeper_scenario(
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    current_values: np.ndarray,
    option_values: np.ndarray,
    keeper_lambda: float,
    market: np.ndarray,
    static: dict[str, Any],
) -> np.ndarray | None:
    """Solve roster selections with two auxiliary binary keeper activations."""
    n = len(predictions)
    salary_row = sim.create_G_salaries_from_values(
        market,
        predictions.player.to_numpy(),
        {"players": [], "salaries": []},
    )
    g_x = np.vstack([salary_row, static["G_static"]])
    h_x = np.vstack([sim.create_h_salaries(), static["h_static"]])
    zero = np.zeros((len(g_x), n))
    g_parts = [np.hstack([g_x, zero])]
    h_parts = [h_x]

    eye = np.eye(n)
    # Keeper activation requires roster selection, is nonnegative, and is
    # disabled for players with zero eligible option value.
    g_parts.extend(
        [
            np.hstack([-eye, eye]),
            np.hstack([np.zeros((n, n)), -eye]),
            np.hstack([np.zeros((n, n)), eye]),
            np.hstack([np.zeros((1, n)), np.ones((1, n))]),
        ]
    )
    h_parts.extend(
        [
            np.zeros((n, 1)),
            np.zeros((n, 1)),
            (np.asarray(option_values) > 0).astype(float).reshape(-1, 1),
            np.array([[float(KEEPER_SLOTS)]]),
        ]
    )
    a_x = np.asarray(static["A"])
    a = np.hstack([a_x, np.zeros((a_x.shape[0], n))])
    objective = -np.concatenate(
        [
            np.asarray(current_values, dtype=float),
            float(keeper_lambda) * np.asarray(option_values, dtype=float),
        ]
    )
    status, solution = sim.solve_ilp(
        matrix(objective, tc="d"),
        matrix(np.vstack(g_parts), tc="d"),
        matrix(np.vstack(h_parts), tc="d"),
        matrix(a, tc="d"),
        static["b"],
    )
    if status in {"infeasible problem", "LP relaxation is primal infeasible"}:
        return None
    if status != "optimal":
        raise RuntimeError(f"Keeper-option ILP status={status!r}.")
    return np.asarray(solution)[:n, 0] == 1


def refine_keeper_roster(
    predictions: pd.DataFrame,
    selected_mask: np.ndarray,
    weekly_scores: np.ndarray,
    decision_scores: np.ndarray,
    played_mask: np.ndarray,
    salary_values: np.ndarray,
    top_n: list[str],
    waiver_baseline: dict[str, float],
    option_values: np.ndarray,
    keeper_lambda: float,
    max_swaps: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_mask = np.asarray(selected_mask, dtype=bool).copy()
    salary_values = np.asarray(salary_values, dtype=float)
    option_values = np.asarray(option_values, dtype=float)
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    current_ppg = predictions[
        FootballSimulation.sample_value_columns(predictions)
    ].mean(axis=1).to_numpy(dtype=float)
    top_n_set = set(top_n)
    accepted = 0

    for _ in range(max_swaps):
        current_lineup = bench.exact_reference_score(
            predictions,
            selected_mask,
            weekly_scores,
            decision_scores,
            played_mask,
            waiver_baseline,
        )
        current_option = fast_roster_keeper_utility(
            selected_mask,
            option_values,
            current_ppg,
            positions,
            players,
        )
        current_objective = current_lineup + keeper_lambda * current_option
        out_indices = np.flatnonzero(selected_mask)
        base_masks = []
        for out_idx in out_indices:
            mask = selected_mask.copy()
            mask[out_idx] = False
            base_masks.append(mask)
        incoming_rows = FootballSimulation.managed_marginal_values_batch(
            weekly_scores,
            positions,
            decision_scores,
            players,
            [players[mask].tolist() for mask in base_masks],
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

            base_option = np.sort(option_values[base_mask & (option_values > 0)])[::-1]
            if len(base_option) == 0:
                approximate_option = option_values[eligible_idx]
            elif len(base_option) == 1:
                approximate_option = base_option[0] + option_values[eligible_idx]
            else:
                approximate_option = (
                    base_option[0]
                    + np.maximum(base_option[1], option_values[eligible_idx])
                )
            approximate_lineup = (
                current_lineup
                - float(incoming[out_idx])
                + incoming[eligible_idx]
            )
            approximate_objective = (
                approximate_lineup + keeper_lambda * approximate_option
            )
            shortlist_size = min(40, len(eligible_idx))
            objective_order = np.lexsort(
                (eligible_idx, -approximate_objective)
            )[:shortlist_size]
            auxiliary_size = min(10, len(eligible_idx))
            lineup_order = np.lexsort(
                (eligible_idx, -incoming[eligible_idx])
            )[:auxiliary_size]
            option_order = np.lexsort(
                (eligible_idx, -option_values[eligible_idx])
            )[:auxiliary_size]
            shortlist_order = np.unique(
                np.concatenate([objective_order, lineup_order, option_order])
            )
            for in_idx in eligible_idx[shortlist_order]:
                replacement = base_mask.copy()
                replacement[in_idx] = True
                estimated_lineup = (
                    current_lineup - float(incoming[out_idx]) + float(incoming[in_idx])
                )
                estimated = estimated_lineup + keeper_lambda * fast_roster_keeper_utility(
                    replacement,
                    option_values,
                    current_ppg,
                    positions,
                    players,
                )
                candidate = (float(estimated), int(out_idx), int(in_idx))
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None or best[0] <= current_objective + 1e-5:
            break
        _, out_idx, in_idx = best
        replacement = selected_mask.copy()
        replacement[out_idx] = False
        replacement[in_idx] = True
        exact = bench.exact_reference_score(
            predictions,
            replacement,
            weekly_scores,
            decision_scores,
            played_mask,
            waiver_baseline,
        ) + keeper_lambda * fast_roster_keeper_utility(
            replacement,
            option_values,
            current_ppg,
            positions,
            players,
        )
        if exact <= current_objective + 1e-5:
            break
        selected_mask = replacement
        accepted += 1
    return selected_mask, {"accepted_swaps": accepted}


def current_value_banks(
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    predictions: pd.DataFrame,
    waiver: dict[str, float],
    policies: tuple[Policy, ...] = POLICIES,
) -> dict[float, np.ndarray]:
    banks: dict[float, np.ndarray] = {}
    for weight in sorted({policy.bench_weight for policy in policies}):
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
                    bench_upside_weight=weight,
                    played_mask=played[context_idx],
                )
            )
        banks[weight] = np.column_stack(samples).astype(np.float32)
    return banks


def future_actual_ppg(
    year: int,
    candidate_forecast: pd.DataFrame,
    raw_weekly: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ppg = np.zeros((len(candidate_forecast), MAX_KEEPER_YEARS), dtype=np.float32)
    games = np.zeros_like(ppg, dtype=np.int16)
    available_horizons: list[int] = []
    max_outcome_year = int(raw_weekly.season.max())
    for horizon in range(1, MAX_KEEPER_YEARS + 1):
        target = year + horizon
        if target > max_outcome_year:
            continue
        labels = candidate_forecast[["player", "player_key", "pos"]].copy()
        labels["season"] = target
        scores, played, _, _ = base.raw_week_matrices(
            labels,
            raw_weekly[raw_weekly.season.eq(target)],
        )
        counts = (played > 0).sum(axis=1)
        totals = (scores * (played > 0)).sum(axis=1)
        values = np.divide(
            totals,
            counts,
            out=np.zeros(len(counts), dtype=float),
            where=counts > 0,
        )
        values[counts < MIN_FUTURE_GAMES] = 0.0
        ppg[:, horizon - 1] = values
        games[:, horizon - 1] = counts
        available_horizons.append(horizon)
    return ppg, games, available_horizons


def keeper_evaluation_metrics(
    selected_mask: np.ndarray,
    starter_mask: np.ndarray,
    option_values: np.ndarray,
    eligible: np.ndarray,
    curves: dict[str, IsotonicRegression],
    positions: np.ndarray,
    market: np.ndarray,
    actual_acquisition_prices: np.ndarray,
    year_exp: np.ndarray,
    future_ppg: np.ndarray,
    available_horizons: list[int],
) -> dict[str, Any]:
    bench_mask = selected_mask & ~starter_mask
    identified = top_keeper_indices(
        selected_mask,
        option_values,
        candidate_mask=bench_mask,
    )
    output: dict[str, Any] = {
        "predicted_keeper_option_top2": top_keeper_utility(
            selected_mask,
            option_values,
            candidate_mask=bench_mask,
        ),
        "identified_keeper_count": int(len(identified)),
        "identified_nominal_bench_count": int(np.sum(bench_mask[identified])),
        "identified_young_count": int(np.sum(year_exp[identified] <= 2)),
        "identified_keeper_players": "|".join(map(str, identified.tolist())),
        "future_outcome_years": int(len(available_horizons)),
    }
    if not available_horizons:
        output.update(
            {
                "realized_next_keeper_surplus": np.nan,
                "realized_available_contract_surplus": np.nan,
                "realized_contract_surplus_per_year": np.nan,
                "realized_next_ppg_max": np.nan,
                "realized_next_market_value_max": np.nan,
                "realized_next_keeper_hits": np.nan,
                "roster_oracle_next_keeper_surplus": np.nan,
                "keeper_identification_regret": np.nan,
                "actual_keeper_cost_coverage": np.nan,
            }
        )
        return output

    future_values = np.zeros_like(future_ppg, dtype=float)
    for pos, curve in curves.items():
        mask = positions == pos
        future_values[mask] = curve.predict(
            future_ppg[mask].reshape(-1)
        ).reshape(int(mask.sum()), future_ppg.shape[1])
    actual_prices = np.asarray(actual_acquisition_prices, dtype=float)
    actual_prices_valid = np.isfinite(actual_prices)
    realized = np.full_like(future_values, np.nan, dtype=float)
    for horizon in available_horizons:
        realized[:, horizon - 1] = np.where(
            actual_prices_valid,
            np.maximum(
                future_values[:, horizon - 1]
                - (actual_prices + KEEPER_ESCALATION * horizon),
                0.0,
            ),
            np.nan,
        )
    first = realized[:, 0]
    identified_first = first[identified]
    identified_available = np.isfinite(identified_first)
    output["actual_keeper_cost_coverage"] = float(
        identified_available.mean() if len(identified_available) else 0.0
    )
    output["realized_next_keeper_surplus"] = float(
        np.nansum(identified_first) if identified_available.any() else np.nan
    )
    output["realized_next_keeper_hits"] = float(
        np.sum(identified_first[identified_available] > 0)
        if identified_available.any()
        else np.nan
    )
    output["realized_next_ppg_max"] = float(
        np.max(future_ppg[identified, 0]) if len(identified) else 0.0
    )
    output["realized_next_market_value_max"] = float(
        np.max(future_values[identified, 0]) if len(identified) else 0.0
    )
    contract = realized[:, [h - 1 for h in available_horizons]]
    identified_contract = contract[identified]
    output["realized_available_contract_surplus"] = float(
        np.nansum(identified_contract)
        if np.isfinite(identified_contract).any()
        else np.nan
    )
    output["realized_contract_surplus_per_year"] = float(
        output["realized_available_contract_surplus"] / len(available_horizons)
        if np.isfinite(output["realized_available_contract_surplus"])
        else np.nan
    )
    oracle_pool = np.flatnonzero(selected_mask & eligible & np.isfinite(first))
    oracle_values = np.sort(first[oracle_pool])[::-1][:KEEPER_SLOTS]
    oracle = float(oracle_values.sum()) if len(oracle_values) else 0.0
    output["roster_oracle_next_keeper_surplus"] = oracle
    output["keeper_identification_regret"] = float(
        oracle - output["realized_next_keeper_surplus"]
        if np.isfinite(output["realized_next_keeper_surplus"])
        else np.nan
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
    value_banks: dict[float, np.ndarray],
    next_draws: np.ndarray,
    next_available: np.ndarray,
    impact_thresholds: dict[str, float],
    current_waiver: dict[str, float],
    actual_acquisition_prices: np.ndarray,
    year_exp: np.ndarray,
    future_ppg: np.ndarray,
    available_horizons: list[int],
    trials: int,
    context_draws: int,
    seed: int,
    policies: tuple[Policy, ...] = POLICIES,
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
    option_diagnostics = []
    started = time.perf_counter()

    for trial in range(trials):
        context_idx = context_plan[trial]
        market = markets[:, trial]
        predictions["salary"] = market
        option_values, curves, option_eligible = keeper_option_values(
            predictions,
            next_draws,
            next_available,
            market,
            impact_thresholds,
        )
        positive = option_values[option_values > 0]
        option_diagnostics.append(
            {
                "positive_players": int(len(positive)),
                "mean_positive_option": float(positive.mean()) if len(positive) else 0.0,
                "p90_positive_option": float(np.percentile(positive, 90))
                if len(positive)
                else 0.0,
            }
        )
        for policy in policies:
            current_values = value_banks[policy.bench_weight][:, context_idx].mean(axis=1)
            solved_mask = solve_keeper_scenario(
                sim,
                predictions,
                current_values,
                option_values,
                policy.keeper_lambda,
                market,
                static,
            )
            if solved_mask is None:
                rows.append(
                    {
                        "year": year,
                        "trial": trial,
                        "policy": policy.name,
                        "solve_status": "infeasible",
                    }
                )
                continue
            if policy.name in {"current_bench025", "bench0"}:
                selected_mask, refine_info = bench.refine_policy_roster(
                    predictions,
                    solved_mask,
                    ref_weekly,
                    ref_decisions,
                    ref_played,
                    market,
                    top_n,
                    current_waiver,
                    np.zeros_like(option_values),
                    0.0,
                )
            else:
                selected_mask, refine_info = refine_keeper_roster(
                    predictions,
                    solved_mask,
                    ref_weekly,
                    ref_decisions,
                    ref_played,
                    market,
                    top_n,
                    current_waiver,
                    option_values,
                    policy.keeper_lambda,
                )
            roster = tuple(sorted(predictions.loc[selected_mask, "player"]))
            starter_mask = bench.nominal_starter_mask(predictions, selected_mask)
            bench_mask = selected_mask & ~starter_mask
            expected_bench = ROSTER_SIZE - int(sum(LINEUP_REQUIRE.values()))
            if int(bench_mask.sum()) != expected_bench:
                raise AssertionError("Nominal bench size does not match league settings.")
            bench_players = tuple(sorted(predictions.loc[bench_mask, "player"]))
            actual_metrics = bench.actual_policy_metrics(
                environment,
                roster,
                bench_players,
            )
            projected = bench.forecast_metrics(
                predictions,
                roster,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                current_waiver,
                forecast_cache,
            )
            keeper_metrics = keeper_evaluation_metrics(
                selected_mask,
                starter_mask,
                option_values,
                option_eligible,
                curves,
                predictions.pos.to_numpy(),
                market,
                actual_acquisition_prices,
                year_exp,
                future_ppg,
                available_horizons,
            )
            selected_prices = np.sort(market[selected_mask])[::-1]
            forecast_spend = float(market[selected_mask].sum())
            actual_feasible = actual_metrics["actual_salary_spend"] <= SALARY_CAP + 1e-8
            identified_idx = top_keeper_indices(
                selected_mask,
                option_values,
                candidate_mask=bench_mask,
            )
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "policy": policy.name,
                    "solve_status": "optimal",
                    "bench_weight": policy.bench_weight,
                    "keeper_lambda": policy.keeper_lambda,
                    "roster": "|".join(roster),
                    "bench_players": "|".join(bench_players),
                    "identified_keeper_players": "|".join(
                        predictions.iloc[identified_idx].player.tolist()
                    ),
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
                    **{
                        key: value
                        for key, value in keeper_metrics.items()
                        if key != "identified_keeper_players"
                    },
                }
            )
        if (trial + 1) % max(1, min(25, trials)) == 0:
            print(
                f"{year}: completed {trial + 1}/{trials} paired trials "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    diagnostics = pd.DataFrame(option_diagnostics)
    return pd.DataFrame(rows), {
        "top_n_players": top_n,
        "remaining_market_budget": remaining_budget,
        "remaining_market_slots": remaining_slots,
        "runtime_seconds": time.perf_counter() - started,
        "mean_positive_option_players": float(diagnostics.positive_players.mean()),
        "mean_positive_keeper_option": float(diagnostics.mean_positive_option.mean()),
        "mean_p90_keeper_option": float(diagnostics.p90_positive_option.mean()),
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
    "predicted_keeper_option_top2",
    "identified_keeper_count",
    "identified_nominal_bench_count",
    "identified_young_count",
    "future_outcome_years",
    "realized_next_keeper_surplus",
    "realized_available_contract_surplus",
    "realized_contract_surplus_per_year",
    "realized_next_ppg_max",
    "realized_next_market_value_max",
    "realized_next_keeper_hits",
    "roster_oracle_next_keeper_surplus",
    "keeper_identification_regret",
    "actual_keeper_cost_coverage",
)


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
    ]
    paired_rows = []
    for policy in [value.name for value in POLICIES if value.name != BASELINE_POLICY]:
        candidate = optimal[optimal.policy.eq(policy)][
            ["year", "trial", "roster", *METRICS]
        ]
        paired = baseline.merge(
            candidate,
            on=["year", "trial"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired["policy"] = policy
        paired["roster_changed"] = paired.roster_baseline.ne(paired.roster_candidate)
        paired["roster_jaccard"] = [
            bench.roster_jaccard(left, right)
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
    current_period = np.where(
        paired.year.eq(2025),
        "current_temporal_2025",
        "current_development_2022_2024",
    )
    keeper_period = np.select(
        [paired.year.le(2023), paired.year.eq(2024)],
        ["keeper_development_2022_2023", "keeper_temporal_2024"],
        default="keeper_unrealized_2025",
    )
    long = pd.concat(
        [
            paired.assign(period=current_period),
            paired.assign(period=keeper_period),
        ],
        ignore_index=True,
    )
    for (policy, period), frame in long.groupby(["policy", "period"]):
        row: dict[str, Any] = {
            "policy": policy,
            "period": period,
            "comparisons": int(len(frame)),
            "roster_changed_rate": float(frame.roster_changed.mean()),
            "roster_jaccard": float(frame.roster_jaccard.mean()),
        }
        for metric in METRICS:
            values = frame[f"{metric}_effect"].astype(float).dropna()
            row[f"mean_{metric}_effect"] = float(values.mean()) if len(values) else np.nan
            row[f"mcse_{metric}_effect"] = float(
                values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        period_rows.append(row)
    return by_year, across, paired, pd.DataFrame(period_rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 2) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
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
        "predicted_keeper_option_top2",
        "realized_next_keeper_surplus",
        "realized_next_keeper_hits",
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
        "mean_predicted_keeper_option_top2_effect",
        "mean_realized_next_keeper_surplus_effect",
        "mean_realized_next_keeper_hits_effect",
        "mean_actual_cap_feasible_effect",
    ]
    lines = [
        "# Keeper Option Bonus Results",
        "",
        f"{args.trials} paired trials per origin with {args.contexts} construction and evaluation contexts.",
        "Keeper rules: two slots, +$10 per keeper year, maximum three keeper years.",
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
        "- Current-season evaluation never includes keeper utility as fantasy points.",
        "- The three-year construction payoff assumes a next-year hit persists; first-year realized surplus is primary.",
        "- Historical keeper cost uses observed acquisition salary when available, so affordability remains a required companion metric.",
        "- Four current seasons and three realized next-season origins are the independent outcome units.",
        "- Frozen legacy salary and next-year uncertainty methods differ by origin and are not the current v5 production surface.",
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
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete year-policy blocks already present in roster_trials.csv.",
    )
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
    print("Loading leakage-safe current and future outcome inputs...", flush=True)
    raw_weekly = base.load_raw_weekly(max_year=max(args.years) + MAX_KEEPER_YEARS)
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    all_trials = []
    completed_policy_years: set[tuple[int, str]] = set()
    checkpoint_path = output_dir / "roster_trials.csv"
    manifest_path = output_dir / "source_manifest.json"
    prior_manifest: dict[str, Any] = {}
    if args.resume and manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.resume and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        expected_policies = {policy.name for policy in POLICIES}
        checkpoint = checkpoint[
            checkpoint.year.isin(args.years)
            & checkpoint.policy.isin(expected_policies)
        ].copy()
        valid_blocks = []
        for (year, policy), block in checkpoint.groupby(["year", "policy"]):
            if len(block) == args.trials and block.trial.nunique() == args.trials:
                completed_policy_years.add((int(year), str(policy)))
                valid_blocks.append(block)
        if valid_blocks:
            all_trials.append(pd.concat(valid_blocks, ignore_index=True))
            print(
                "Resuming complete year-policy blocks from roster_trials.csv: "
                f"{len(completed_policy_years)}",
                flush=True,
            )
    manifest: dict[str, Any] = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "keeper_rules": {
            "keeper_slots": KEEPER_SLOTS,
            "annual_salary_escalation": KEEPER_ESCALATION,
            "maximum_future_keeper_years": MAX_KEEPER_YEARS,
        },
        "policies": [asdict(policy) for policy in POLICIES],
        "base_replay": str(BASE_STUDY),
        "base_replay_sha256": base.sha256_file(BASE_STUDY),
        "bench_replay": str(BENCH_STUDY),
        "bench_replay_sha256": base.sha256_file(BENCH_STUDY),
        "simulation_helper": {
            "path": str(base.APP_HELPER),
            "sha256": base.sha256_file(base.APP_HELPER),
        },
        "origins": prior_manifest.get("origins", {}),
    }

    for year in args.years:
        pending_policies = tuple(
            policy
            for policy in POLICIES
            if (year, policy.name) not in completed_policy_years
        )
        if not pending_policies:
            manifest["origins"].setdefault(
                str(year),
                {
                    "resumed_from_existing_roster_trials": True,
                    "rows": args.trials * len(POLICIES),
                },
            )
            continue
        year_started = time.perf_counter()
        print(
            f"\n=== Origin {year}: "
            + ", ".join(policy.name for policy in pending_policies)
            + " ===",
            flush=True,
        )
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
            next_draws, next_available, next_meta = load_frozen_next_year_draws(
                year,
                conn,
                forecast,
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
        impact_thresholds = bench.league_starter_thresholds(
            forecast,
            current_waiver,
        )

        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_idx].reset_index(drop=True)
        predictions = base.build_predictions(
            candidate_forecast,
            ppg_draws[candidate_idx],
        )
        candidate_salary_draws = salary_draws[candidate_idx]
        candidate_next_draws = next_draws[candidate_idx]
        candidate_next_available = next_available[candidate_idx]
        candidate_outcomes = outcome_labels.iloc[candidate_idx].reset_index(drop=True)
        actual_acquisition_prices = pd.to_numeric(
            candidate_outcomes.actual_salary,
            errors="coerce",
        ).to_numpy(dtype=float)
        actual_acquisition_prices = np.where(
            candidate_outcomes.actual_salary_matched.to_numpy(dtype=bool),
            actual_acquisition_prices,
            np.nan,
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
        future_ppg, future_games, available_horizons = future_actual_ppg(
            year,
            candidate_forecast,
            raw_weekly,
        )
        del future_games

        print(
            f"{year}: {len(predictions)} selectable players, "
            f"{int(candidate_next_available.sum())} next-year forecasts, "
            f"{len(available_horizons)} realized future horizons; building contexts...",
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
        value_banks = current_value_banks(
            weekly,
            decisions,
            played,
            predictions,
            current_waiver,
            pending_policies,
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
            candidate_next_draws,
            candidate_next_available,
            impact_thresholds,
            current_waiver,
            actual_acquisition_prices,
            year_exp,
            future_ppg,
            available_horizons,
            args.trials,
            args.context_draws,
            args.seed,
            pending_policies,
        )
        if not trials.solve_status.eq("optimal").all():
            raise AssertionError("A keeper policy roster solve failed.")
        all_trials.append(trials)
        source_manifest.update(projection_meta)
        source_manifest.update(next_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(run_meta)
        source_manifest.update(
            {
                "keeper_count": environment["keeper_count"],
                "keeper_spend": environment["keeper_spend"],
                "current_waiver_baseline": current_waiver,
                "impact_thresholds": impact_thresholds,
                "realized_future_horizons": available_horizons,
                "computed_policies": [policy.name for policy in pending_policies],
                "resumed_policies": [
                    policy.name
                    for policy in POLICIES
                    if (year, policy.name) in completed_policy_years
                ],
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        manifest["origins"][str(year)] = source_manifest
        pd.concat(all_trials, ignore_index=True).to_csv(
            output_dir / "roster_trials.csv",
            index=False,
        )
        (output_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
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
    expected_rows = len(args.years) * args.trials * len(POLICIES)
    if len(trials) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(trials)}.")
    if not trials.roster.str.split("|").map(len).eq(ROSTER_SIZE).all():
        raise AssertionError("A keeper policy roster does not contain 13 players.")
    if not (trials.forecast_salary_spend <= SALARY_CAP + 1e-8).all():
        raise AssertionError("A keeper policy roster exceeds the forecast salary cap.")
    if not (trials.identified_keeper_count <= KEEPER_SLOTS).all():
        raise AssertionError("A roster identifies more than two keeper options.")

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
        "at_most_two_keeper_options": True,
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
