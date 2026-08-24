"""Paired research test of budget-aware Sequential Auction recourse.

Production code is imported read-only.  The two experimental policies replace
the branch rollout only for this process and are restored after every arm.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import types

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

try:
    import highspy
except ImportError:  # pragma: no cover - optional research runtime
    highspy = None


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
SOURCE_STUDY = ROOT / "research" / "studies" / "2026-08-20_bijan_fourth_rb_audit"
RESULTS_DIR = STUDY_DIR / "results"
if str(SOURCE_STUDY) not in sys.path:
    sys.path.insert(0, str(SOURCE_STUDY))

import run_audit as audit  # noqa: E402


sequential = audit.sequential
FootballSimulation = audit.FootballSimulation
_REPLAN_STATIC_CACHE: dict[tuple, dict] = {}
_STABLE_REQUIRED_SOLVE_CACHE: dict[tuple, np.ndarray | None] = {}
_COMPILED_PLAN_BANK: dict[tuple[int, str], list[dict | None]] = {}
BOUNDED_CANDIDATE_LIMIT = 24
BOUNDED_OUTGOING_LIMIT = 8
BOUNDED_SWAP_LIMIT = 3


def compiled_plan_path(variation, plan_key=""):
    key = f"_{plan_key}" if plan_key else ""
    return RESULTS_DIR / f"compiled_plans{key}_variation{int(variation)}.json"


def write_compiled_plan_bank(variation, plans, plan_key=""):
    payload = []
    for plan in plans:
        if plan is None:
            payload.append(None)
            continue
        payload.append({
            "selected": sorted(plan["selected"]),
            "targets": sorted(plan["targets"]),
            "forecast_cost": {
                str(player): float(cost)
                for player, cost in plan["forecast_cost"].items()
            },
            "inflation": float(plan["inflation"]),
        })
    with compiled_plan_path(variation, plan_key).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_compiled_plan_bank(variation, plan_key=""):
    path = compiled_plan_path(variation, plan_key)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    plans = []
    for plan in payload:
        if plan is None:
            plans.append(None)
            continue
        plans.append({
            "selected": set(plan["selected"]),
            "targets": set(plan["targets"]),
            "forecast_cost": {
                str(player): float(cost)
                for player, cost in plan["forecast_cost"].items()
            },
            "inflation": float(plan["inflation"]),
        })
    return plans


def stable_required_roster_solve(
    self,
    predictions,
    current_values,
    salary_values,
    required_players,
    fixed_salary_map=None,
    roster_size=None,
    pos_min_counts=None,
    pos_max_counts=None,
    top_n=None,
    enforce_top_n=True,
    static_matrices=None,
):
    """Research-only SciPy/HiGHS equivalent of the small required-player ILP."""
    del static_matrices
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    values = np.asarray(current_values, dtype=np.float64).reshape(-1)
    salaries = np.asarray(salary_values, dtype=np.float64).reshape(-1).copy()
    fixed_salary_map = fixed_salary_map or {}
    for player, salary in fixed_salary_map.items():
        matches = np.flatnonzero(players == player)
        if len(matches) == 1:
            salaries[matches[0]] = float(salary)
    roster_size = int(roster_size or sum(self.pos_require_start.values()))
    pos_min_counts = pos_min_counts or {
        pos: 0 for pos in ("QB", "RB", "WR", "TE")
    }
    pos_max_counts = pos_max_counts or {
        pos: roster_size for pos in ("QB", "RB", "WR", "TE")
    }
    cache_key = (
        tuple(players),
        hashlib.blake2b(values.tobytes(), digest_size=16).digest(),
        hashlib.blake2b(salaries.tobytes(), digest_size=16).digest(),
        tuple(sorted(dict.fromkeys(required_players or []))),
        int(roster_size),
        tuple(sorted(pos_min_counts.items())),
        tuple(sorted(pos_max_counts.items())),
        tuple(sorted(top_n or [])),
        bool(enforce_top_n),
        float(self.salary_cap),
    )
    if cache_key in _STABLE_REQUIRED_SOLVE_CACHE:
        cached = _STABLE_REQUIRED_SOLVE_CACHE[cache_key]
        return None if cached is None else cached.copy()

    rows = [np.ones(len(players), dtype=np.float64), salaries]
    lower = [float(roster_size), -np.inf]
    upper = [float(roster_size), float(self.salary_cap)]
    for player in dict.fromkeys(required_players or []):
        matches = np.flatnonzero(players == player)
        if len(matches) != 1:
            return None
        row = np.zeros(len(players), dtype=np.float64)
        row[matches[0]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for pos in ("QB", "RB", "WR", "TE"):
        row = (positions == pos).astype(np.float64)
        rows.append(row)
        lower.append(float(pos_min_counts.get(pos, 0)))
        upper.append(float(pos_max_counts.get(pos, roster_size)))
    top_n = list(top_n or [])
    if enforce_top_n and top_n:
        row = np.isin(players, top_n).astype(np.float64)
        rows.append(row)
        lower.append(1.0)
        upper.append(np.inf)

    result = milp(
        c=-values,
        integrality=np.ones(len(players), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 10.0, "presolve": True},
    )
    if result.status == 2:
        _STABLE_REQUIRED_SOLVE_CACHE[cache_key] = None
        return None
    if not result.success or result.x is None:
        raise RuntimeError(
            "Research stable required-player solve failed: "
            f"status={result.status} message={result.message!r}"
        )
    selected = np.asarray(result.x) > 0.5
    _STABLE_REQUIRED_SOLVE_CACHE[cache_key] = selected.copy()
    return selected


def direct_highs_required_roster_solve(
    self,
    predictions,
    current_values,
    salary_values,
    required_players,
    fixed_salary_map=None,
    roster_size=None,
    pos_min_counts=None,
    pos_max_counts=None,
    top_n=None,
    enforce_top_n=True,
    static_matrices=None,
):
    """Direct-HiGHS equivalent that avoids repeated SciPy wrapper teardown."""
    del static_matrices
    if highspy is None:
        raise RuntimeError("The direct HiGHS research backend is unavailable")
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    values = np.asarray(current_values, dtype=np.float64).reshape(-1)
    salaries = np.asarray(salary_values, dtype=np.float64).reshape(-1).copy()
    fixed_salary_map = fixed_salary_map or {}
    for player, salary in fixed_salary_map.items():
        matches = np.flatnonzero(players == player)
        if len(matches) == 1:
            salaries[matches[0]] = float(salary)
    roster_size = int(roster_size or sum(self.pos_require_start.values()))
    pos_min_counts = pos_min_counts or {
        pos: 0 for pos in ("QB", "RB", "WR", "TE")
    }
    pos_max_counts = pos_max_counts or {
        pos: roster_size for pos in ("QB", "RB", "WR", "TE")
    }
    cache_key = (
        "direct_highs",
        tuple(players),
        hashlib.blake2b(values.tobytes(), digest_size=16).digest(),
        hashlib.blake2b(salaries.tobytes(), digest_size=16).digest(),
        tuple(sorted(dict.fromkeys(required_players or []))),
        int(roster_size),
        tuple(sorted(pos_min_counts.items())),
        tuple(sorted(pos_max_counts.items())),
        tuple(sorted(top_n or [])),
        bool(enforce_top_n),
        float(self.salary_cap),
    )
    if cache_key in _STABLE_REQUIRED_SOLVE_CACHE:
        cached = _STABLE_REQUIRED_SOLVE_CACHE[cache_key]
        return None if cached is None else cached.copy()

    rows = [np.ones(len(players), dtype=np.float64), salaries]
    lower = [float(roster_size), -highspy.kHighsInf]
    upper = [float(roster_size), float(self.salary_cap)]
    for player in dict.fromkeys(required_players or []):
        matches = np.flatnonzero(players == player)
        if len(matches) != 1:
            return None
        row = np.zeros(len(players), dtype=np.float64)
        row[matches[0]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for pos in ("QB", "RB", "WR", "TE"):
        rows.append((positions == pos).astype(np.float64))
        lower.append(float(pos_min_counts.get(pos, 0)))
        upper.append(float(pos_max_counts.get(pos, roster_size)))
    top_n = list(top_n or [])
    if enforce_top_n and top_n:
        rows.append(np.isin(players, top_n).astype(np.float64))
        lower.append(1.0)
        upper.append(highspy.kHighsInf)

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("time_limit", 10.0)
    solver.setOptionValue("mip_rel_gap", 0.0)
    count = len(players)
    indices = np.arange(count, dtype=np.int32)
    solver.addVars(count, np.zeros(count), np.ones(count))
    solver.changeColsIntegrality(
        count,
        indices,
        np.full(count, highspy.HighsVarType.kInteger),
    )
    solver.changeColsCost(count, indices, -values)
    for row, row_lower, row_upper in zip(rows, lower, upper):
        nonzero = np.flatnonzero(row).astype(np.int32)
        solver.addRow(
            float(row_lower),
            float(row_upper),
            len(nonzero),
            nonzero,
            row[nonzero],
        )
    solver.run()
    status = solver.getModelStatus()
    if status == highspy.HighsModelStatus.kInfeasible:
        _STABLE_REQUIRED_SOLVE_CACHE[cache_key] = None
        return None
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"Direct HiGHS required-roster status={status!r}")
    selected = np.asarray(solver.getSolution().col_value) > 0.5
    _STABLE_REQUIRED_SOLVE_CACHE[cache_key] = selected.copy()
    return selected


def simulate_budget_aware_branch(
    sim,
    predictions,
    managed_values,
    base_prices,
    selection_premiums,
    initial_salary_map,
    candidate,
    candidate_price,
    force_buy,
    order,
    revealed_prices,
    remaining_market_budget,
    remaining_market_slots,
    roster_size,
    pos_min_counts,
    pos_max_counts,
    require_top_n,
    enforce_top_n,
    compiled_plan,
    policy_scores,
    bargain_replan_fraction=sequential.DEFAULT_BARGAIN_REPLAN_FRACTION,
    rollout_context=None,
    *,
    replan_mode="slack",
    slack_floor=5.0,
    personal_exclusions=(),
):
    """Production-equivalent rollout with history-only target upgrades.

    ``slack`` re-solves after a purchase when projected final spend leaves more
    than max(slack_floor, open_slots) unused. ``purchase`` re-solves after every
    own purchase. ``bounded`` and ``bounded_guard`` use deterministic local
    upgrades instead; the latter also protects the final two roster slots.
    """
    if replan_mode not in {
        "baseline",
        "slack",
        "purchase",
        "bounded",
        "bounded_guard",
    }:
        raise ValueError(f"Unknown replan mode: {replan_mode}")

    personal_exclusions = set(personal_exclusions)
    if rollout_context is None:
        players = predictions.player.to_numpy()
        base_price_map = dict(zip(players, np.asarray(base_prices, dtype=np.float64)))
        premium_map = dict(zip(
            players,
            np.asarray(selection_premiums, dtype=np.float64),
        ))
        position_map = predictions.set_index("player").pos.to_dict()
        initial_unresolved = set(players) - set(initial_salary_map)
        initial_position_counts = Counter(
            position_map[player] for player in initial_salary_map
        )
        initial_owned_spend = float(sum(initial_salary_map.values()))
        policy_order = tuple(sorted(
            (player for player in players if player not in personal_exclusions),
            key=lambda player: (-policy_scores[player], player),
        ))
    else:
        players = rollout_context["players"]
        base_price_map = rollout_context["base_price_map"]
        premium_map = rollout_context["premium_map"]
        position_map = rollout_context["position_map"]
        initial_unresolved = rollout_context["initial_unresolved"]
        initial_position_counts = rollout_context["initial_position_counts"]
        initial_owned_spend = rollout_context["initial_owned_spend"]
        policy_order = tuple(
            player
            for player in rollout_context["policy_order"]
            if player not in personal_exclusions
        )
    managed_value_map = dict(zip(
        players,
        np.asarray(managed_values, dtype=np.float64),
    ))
    managed_value_order = tuple(sorted(
        (player for player in players if player not in personal_exclusions),
        key=lambda player: (-managed_value_map[player], player),
    ))

    owned = dict(initial_salary_map)
    owned_spend = float(initial_owned_spend)
    owned_position_counts = Counter(initial_position_counts)
    unresolved = set(initial_unresolved)
    initial_unresolved_count = max(len(unresolved), 1)
    observed_sales: list[tuple[float, float]] = []
    market_budget = float(remaining_market_budget)
    market_slots = int(remaining_market_slots)
    policy_refreshes = 0
    full_replans = 0
    budget_replans = 0
    bounded_triggers = 0
    bounded_swaps = 0
    bounded_rebuilds = 0
    guard_rejections = 0
    events_seen = 0
    failure_reason = None
    max_projected_slack = 0.0

    def current_market_inflation():
        return sequential.history_market_inflation(observed_sales)

    def record_sale(actual, forecast):
        observed_sales.append((float(actual), float(forecast)))

    def incomplete(reason=None):
        return {
            "complete": False,
            "roster": tuple(sorted(owned)),
            "salary_map": tuple(sorted(
                (player, float(salary)) for player, salary in owned.items()
            )),
            "salary_spend": float(owned_spend),
            "replans": int(policy_refreshes),
            "full_replans": int(full_replans),
            "budget_replans": int(budget_replans),
            "bounded_triggers": int(bounded_triggers),
            "bounded_swaps": int(bounded_swaps),
            "bounded_rebuilds": int(bounded_rebuilds),
            "guard_rejections": int(guard_rejections),
            "owned_count": int(len(owned)),
            "events_seen": int(events_seen),
            "failure_reason": reason,
            "max_projected_slack": float(max_projected_slack),
        }

    if candidate is not None:
        if candidate not in unresolved:
            return incomplete("candidate_unavailable")
        if force_buy:
            if not sequential._legal_personal_purchase(
                predictions,
                owned,
                candidate,
                candidate_price,
                roster_size,
                pos_max_counts,
                sim.salary_cap,
                position_map=position_map,
                owned_spend=owned_spend,
                owned_position_counts=owned_position_counts,
            ):
                return incomplete("candidate_illegal")
            payment = float(candidate_price)
            owned[candidate] = payment
            owned_spend += payment
            owned_position_counts[position_map[candidate]] += 1
        else:
            payment = max(float(candidate_price) - 1.0, 1.0)
        unresolved.remove(candidate)
        market_budget -= payment
        market_slots -= 1
        record_sale(payment, base_price_map[candidate])

    if compiled_plan is None:
        return incomplete("compiled_plan")
    active_targets = set(compiled_plan["targets"]) & unresolved
    compiled_cost = dict(compiled_plan["forecast_cost"])

    def forecast_price(player, inflation=None):
        if inflation is None:
            inflation = current_market_inflation()
        return (
            1.0
            + max(base_price_map[player] - 1.0, 0.0) * inflation
            + premium_map[player]
            + 1.0
        )

    def planned_cost(player, inflation):
        if player in compiled_cost:
            return compiled_cost[player]
        return forecast_price(player, inflation=inflation)

    def target_set_is_legal(targets, inflation=None):
        final_players = list(owned) + list(targets)
        if len(final_players) != int(roster_size):
            return False
        counts = Counter(owned_position_counts)
        counts.update([position_map[player] for player in targets])
        for pos, minimum in pos_min_counts.items():
            if counts.get(pos, 0) < int(minimum):
                return False
        for pos, maximum in pos_max_counts.items():
            if counts.get(pos, 0) > int(maximum):
                return False
        if inflation is None:
            inflation = current_market_inflation()
        forecast_spend = owned_spend + sum(
            planned_cost(player, inflation) for player in targets
        )
        return forecast_spend <= float(sim.salary_cap) + 1e-9

    def partial_target_is_feasible(targets, desired, inflation):
        counts = Counter(owned_position_counts)
        counts.update([position_map[player] for player in targets])
        for pos, maximum in pos_max_counts.items():
            if counts.get(pos, 0) > int(maximum):
                return False
        unfilled = desired - len(targets)
        forecast_spend = owned_spend + sum(
            planned_cost(player, inflation) for player in targets
        )
        if forecast_spend + max(unfilled, 0) > float(sim.salary_cap) + 1e-9:
            return False
        total_missing = sum(
            max(int(minimum) - counts.get(pos, 0), 0)
            for pos, minimum in pos_min_counts.items()
        )
        if total_missing > unfilled:
            return False
        for pos, minimum in pos_min_counts.items():
            missing = max(int(minimum) - counts.get(pos, 0), 0)
            if missing > unfilled:
                return False
            available = sum(
                position_map[player] == pos
                for player in unresolved - set(targets)
            )
            if available < missing:
                return False
        return True

    def refresh_targets(lost_position=None):
        """Production cached-priority repair used when no full replan fires."""
        nonlocal active_targets, policy_refreshes
        policy_refreshes += 1
        desired = int(roster_size) - len(owned)
        active_targets &= unresolved
        inflation = current_market_inflation()
        if len(active_targets) > desired:
            remove_count = len(active_targets) - desired
            removable = sorted(
                active_targets,
                key=lambda player: (policy_scores[player], player),
            )
            active_targets -= set(removable[:remove_count])
        if (
            len(active_targets) == desired
            and target_set_is_legal(active_targets, inflation=inflation)
        ):
            return True

        same_position = [
            player for player in policy_order
            if lost_position and position_map[player] == lost_position
        ]
        same_position_set = set(same_position)
        ranked = same_position + [
            player for player in policy_order
            if player not in same_position_set
        ]
        for incoming in ranked:
            if incoming not in unresolved or incoming in active_targets:
                continue
            proposed = active_targets | {incoming}
            if not partial_target_is_feasible(proposed, desired, inflation):
                continue
            if len(proposed) == desired and not target_set_is_legal(
                proposed,
                inflation=inflation,
            ):
                continue
            active_targets = proposed
            if len(active_targets) == desired:
                return True

        if len(active_targets) == desired:
            incoming_order = sorted(
                unresolved - active_targets - personal_exclusions,
                key=lambda player: (
                    planned_cost(player, inflation),
                    -policy_scores[player],
                    player,
                ),
            )
            outgoing_order = sorted(
                active_targets,
                key=lambda player: (policy_scores[player], player),
            )
            for outgoing in outgoing_order[:8]:
                for incoming in incoming_order[:24]:
                    proposed = (active_targets - {outgoing}) | {incoming}
                    if target_set_is_legal(proposed, inflation=inflation):
                        active_targets = proposed
                        return True
        return target_set_is_legal(active_targets, inflation=inflation)

    def projected_spend_slack():
        desired = int(roster_size) - len(owned)
        if len(active_targets) != desired:
            return float("inf")
        inflation = current_market_inflation()
        projected_final = owned_spend + sum(
            forecast_price(player, inflation=inflation)
            for player in active_targets
        )
        return float(sim.salary_cap) - projected_final

    def bounded_upgrade_targets():
        """Make a few deterministic positive-value upgrades without an ILP."""
        nonlocal active_targets
        nonlocal compiled_cost
        nonlocal bounded_triggers
        nonlocal bounded_swaps
        bounded_triggers += 1
        if not target_set_is_legal(active_targets):
            return False
        last_swap = None
        for _ in range(BOUNDED_SWAP_LIMIT):
            incoming_order = []
            for player in managed_value_order:
                if player not in unresolved or player in active_targets:
                    continue
                incoming_order.append(player)
                if len(incoming_order) >= BOUNDED_CANDIDATE_LIMIT:
                    break
            outgoing_order = sorted(
                active_targets,
                key=lambda player: (managed_value_map[player], player),
            )[:BOUNDED_OUTGOING_LIMIT]
            invalid_targets = [
                player for player in outgoing_order if player not in position_map
            ]
            if invalid_targets:
                raise RuntimeError(
                    "Bounded target set lost player-key alignment: "
                    f"invalid={invalid_targets!r} last_swap={last_swap!r}"
                )
            best_swap = None
            best_key = None
            inflation = current_market_inflation()
            for outgoing in outgoing_order:
                for incoming in incoming_order:
                    if position_map[incoming] != position_map[outgoing]:
                        continue
                    value_gain = (
                        managed_value_map[incoming]
                        - managed_value_map[outgoing]
                    )
                    if value_gain <= 1e-9:
                        continue
                    proposed = (active_targets - {outgoing}) | {incoming}
                    if not target_set_is_legal(proposed, inflation=inflation):
                        continue
                    key = (
                        float(value_gain),
                        float(managed_value_map[incoming]),
                        -float(forecast_price(incoming, inflation=inflation)),
                        incoming,
                        outgoing,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_swap = (outgoing, incoming)
            if best_swap is None:
                break
            outgoing, incoming = best_swap
            active_targets = (active_targets - {outgoing}) | {incoming}
            compiled_cost.pop(outgoing, None)
            compiled_cost[incoming] = forecast_price(incoming)
            bounded_swaps += 1
            last_swap = best_swap
        return target_set_is_legal(active_targets)

    def rebuild_targets_minimum_first():
        """Deterministic feasibility fallback before optional value upgrades."""
        nonlocal active_targets
        nonlocal bounded_rebuilds
        nonlocal policy_refreshes
        bounded_rebuilds += 1
        policy_refreshes += 1
        desired = int(roster_size) - len(owned)
        inflation = current_market_inflation()
        rebuilt = set()

        def can_add(player):
            proposed = rebuilt | {player}
            counts = Counter(owned_position_counts)
            counts.update([position_map[name] for name in proposed])
            if any(
                counts.get(pos, 0) > int(maximum)
                for pos, maximum in pos_max_counts.items()
            ):
                return False
            projected = owned_spend + sum(
                planned_cost(name, inflation) for name in proposed
            )
            unfilled = desired - len(proposed)
            return projected + max(unfilled, 0) <= float(sim.salary_cap) + 1e-9

        for pos in ("QB", "RB", "WR", "TE"):
            need = max(
                int(pos_min_counts.get(pos, 0))
                - int(owned_position_counts.get(pos, 0)),
                0,
            )
            candidates = sorted(
                (
                    player for player in unresolved
                    if (
                        player not in personal_exclusions
                        and position_map[player] == pos
                    )
                ),
                key=lambda player: (
                    planned_cost(player, inflation),
                    -managed_value_map[player],
                    player,
                ),
            )
            for player in candidates:
                if need <= 0:
                    break
                if player in rebuilt or not can_add(player):
                    continue
                rebuilt.add(player)
                need -= 1
            if need > 0:
                return False

        for player in policy_order:
            if len(rebuilt) >= desired:
                break
            if player not in unresolved or player in rebuilt:
                continue
            if can_add(player):
                rebuilt.add(player)
        if len(rebuilt) != desired:
            return False
        active_targets = rebuilt
        return target_set_is_legal(active_targets, inflation=inflation)

    def full_replan(reason):
        nonlocal active_targets
        nonlocal compiled_cost
        nonlocal policy_refreshes
        nonlocal full_replans
        nonlocal budget_replans
        if len(owned) >= int(roster_size):
            active_targets = set()
            return True
        policy_refreshes += 1
        full_replans += 1
        budget_replans += int(reason == "budget_slack")
        plan = sequential.solve_history_only_plan(
            sim,
            predictions,
            managed_values,
            base_prices,
            selection_premiums,
            owned,
            unresolved,
            roster_size,
            pos_min_counts,
            pos_max_counts,
            require_top_n,
            enforce_top_n,
            observed_sales=observed_sales,
            static_matrix_cache=_REPLAN_STATIC_CACHE,
        )
        if plan is None:
            return False
        active_targets = set(plan["targets"]) & unresolved
        compiled_cost = dict(plan["forecast_cost"])
        return target_set_is_legal(active_targets)

    if not refresh_targets():
        return incomplete("initial_refresh")

    for nominee_idx in np.asarray(order, dtype=int):
        if len(owned) >= int(roster_size):
            break
        nominee = players[nominee_idx]
        if nominee not in unresolved:
            continue
        if market_slots <= 0:
            failure_reason = "market_slots"
            break
        events_seen += 1

        max_market_payment = max(
            market_budget - max(market_slots - 1, 0),
            1.0,
        )
        opponent_price = min(
            float(revealed_prices[nominee_idx]),
            max_market_payment,
        )
        ask_price = min(opponent_price + 1.0, max_market_payment)
        inflation = current_market_inflation()
        expected_price = forecast_price(nominee, inflation=inflation)
        should_buy = False
        swap_out = None
        open_slots = int(roster_size) - len(owned)
        nomination_progress = events_seen / initial_unresolved_count
        target_ceiling = np.ceil(
            expected_price * (1.0 + 0.20 * nomination_progress)
        )
        completion_urgent = market_slots <= max(
            10 * open_slots,
            open_slots + 20,
        )

        personally_eligible = nominee not in personal_exclusions
        if personally_eligible and nominee in active_targets and (
            ask_price <= target_ceiling or completion_urgent
        ):
            should_buy = True
        elif (
            personally_eligible
            and ask_price <= float(bargain_replan_fraction) * expected_price
        ):
            same_position_targets = [
                player for player in active_targets
                if position_map[player] == position_map[nominee]
            ]
            if same_position_targets:
                swap_out = min(
                    same_position_targets,
                    key=lambda player: (policy_scores[player], player),
                )
                dynamic_score = policy_scores[nominee] * np.sqrt(
                    max(expected_price, 1.0) / max(ask_price, 1.0)
                )
                should_buy = dynamic_score > 1.05 * policy_scores[swap_out]
        elif personally_eligible and completion_urgent:
            for proposed_out in sorted(
                active_targets,
                key=lambda player: (policy_scores[player], player),
            ):
                remaining_targets = active_targets - {proposed_out}
                final_players = list(owned) + [nominee] + list(remaining_targets)
                counts = Counter(position_map[player] for player in final_players)
                position_legal = all(
                    counts.get(pos, 0) >= int(minimum)
                    for pos, minimum in pos_min_counts.items()
                ) and all(
                    counts.get(pos, 0) <= int(maximum)
                    for pos, maximum in pos_max_counts.items()
                )
                reserved_cost = sum(
                    planned_cost(player, inflation)
                    for player in remaining_targets
                )
                budget_legal = (
                    owned_spend + ask_price + reserved_cost
                    <= float(sim.salary_cap) + 1e-9
                )
                if position_legal and budget_legal:
                    swap_out = proposed_out
                    should_buy = True
                    break

        if should_buy:
            should_buy = sequential._legal_personal_purchase(
                predictions,
                owned,
                nominee,
                ask_price,
                roster_size,
                pos_max_counts,
                sim.salary_cap,
                position_map=position_map,
                owned_spend=owned_spend,
                owned_position_counts=owned_position_counts,
            )
        if should_buy:
            reserved_targets = active_targets - {nominee}
            if swap_out is not None:
                reserved_targets.discard(swap_out)
            reserved_cost = sum(
                planned_cost(player, inflation)
                for player in reserved_targets
            )
            should_buy = (
                owned_spend + ask_price + reserved_cost
                <= float(sim.salary_cap) + 1e-9
            )

        if (
            should_buy
            and replan_mode == "bounded_guard"
            and swap_out is not None
            and open_slots <= 2
            and not completion_urgent
        ):
            guarded_targets = active_targets - {swap_out}
            guarded_targets.discard(nominee)
            projected_after_buy = owned_spend + ask_price + sum(
                forecast_price(player, inflation=inflation)
                for player in guarded_targets
            )
            projected_after_slack = (
                float(sim.salary_cap) - projected_after_buy
            )
            guard_trigger = max(float(slack_floor), float(open_slots - 1))
            displaced_value = managed_value_map[swap_out]
            incoming_value = managed_value_map[nominee]
            if (
                projected_after_slack > guard_trigger
                and displaced_value > incoming_value + 1e-9
            ):
                should_buy = False
                swap_out = None
                guard_rejections += 1

        was_target = nominee in active_targets
        unresolved.remove(nominee)
        if should_buy:
            payment = ask_price
            owned[nominee] = payment
            owned_spend += payment
            owned_position_counts[position_map[nominee]] += 1
            active_targets.discard(nominee)
            if swap_out is not None:
                active_targets.discard(swap_out)
        else:
            payment = opponent_price
            active_targets.discard(nominee)
        market_budget -= payment
        market_slots -= 1
        record_sale(payment, base_price_map[nominee])

        if len(owned) < int(roster_size) and (should_buy or was_target):
            if replan_mode == "purchase" or (
                replan_mode == "slack" and was_target and not should_buy
            ):
                replan_ok = full_replan(
                    "purchase" if should_buy else "lost_target"
                )
            else:
                lost_position = (
                    position_map[nominee]
                    if was_target and not should_buy else None
                )
                replan_ok = refresh_targets(
                    lost_position=lost_position
                )
                if (
                    not replan_ok
                    and replan_mode in {"bounded", "bounded_guard"}
                ):
                    replan_ok = rebuild_targets_minimum_first()
                slack = projected_spend_slack() if replan_ok else float("inf")
                max_projected_slack = max(max_projected_slack, slack)
                trigger = max(float(slack_floor), float(roster_size - len(owned)))
                if (
                    replan_ok
                    and replan_mode == "slack"
                    and should_buy
                    and slack > trigger
                ):
                    replan_ok = full_replan("budget_slack")
                elif (
                    replan_ok
                    and replan_mode in {"bounded", "bounded_guard"}
                    and slack > trigger
                ):
                    replan_ok = bounded_upgrade_targets()
                    if not replan_ok:
                        replan_ok = rebuild_targets_minimum_first()
            if not replan_ok:
                failure_reason = "dynamic_replan"
                break

    if len(owned) < int(roster_size) and failure_reason is None:
        failure_reason = "order_exhausted"
    final_slack = float(sim.salary_cap) - float(owned_spend)
    return {
        "complete": len(owned) == int(roster_size),
        "roster": tuple(sorted(owned)),
        "salary_map": tuple(sorted(
            (player, float(salary)) for player, salary in owned.items()
        )),
        "salary_spend": float(owned_spend),
        "replans": int(policy_refreshes),
        "full_replans": int(full_replans),
        "budget_replans": int(budget_replans),
        "bounded_triggers": int(bounded_triggers),
        "bounded_swaps": int(bounded_swaps),
        "bounded_rebuilds": int(bounded_rebuilds),
        "guard_rejections": int(guard_rejections),
        "owned_count": int(len(owned)),
        "events_seen": int(events_seen),
        "failure_reason": failure_reason,
        "max_projected_slack": float(max_projected_slack),
        "final_unused_salary": final_slack,
    }


def summarize_paths(frame):
    rows = []
    for (arm, variation, branch), group in frame.groupby(
        ["arm", "variation", "branch"],
        sort=True,
    ):
        complete = group.loc[group.complete]
        spend = complete.roster_spend.to_numpy(dtype=float)
        unused = audit.SALARY_CAP - spend
        rows.append({
            "arm": arm,
            "variation": int(variation),
            "branch": branch,
            "paths": int(len(group)),
            "completion_rate": float(group.complete.mean()),
            "mean_spend": float(np.mean(spend)) if len(spend) else np.nan,
            "mean_unused": float(np.mean(unused)) if len(unused) else np.nan,
            "median_unused": float(np.median(unused)) if len(unused) else np.nan,
            "p90_unused": float(np.percentile(unused, 90)) if len(unused) else np.nan,
            "max_unused": float(np.max(unused)) if len(unused) else np.nan,
            "mean_full_replans": float(complete.full_replans.mean()) if len(complete) else np.nan,
            "mean_budget_replans": float(complete.budget_replans.mean()) if len(complete) else np.nan,
            "mean_bounded_triggers": float(complete.bounded_triggers.mean()) if len(complete) else np.nan,
            "mean_bounded_swaps": float(complete.bounded_swaps.mean()) if len(complete) else np.nan,
            "mean_bounded_rebuilds": float(complete.bounded_rebuilds.mean()) if len(complete) else np.nan,
            "mean_guard_rejections": float(complete.guard_rejections.mean()) if len(complete) else np.nan,
        })
    return pd.DataFrame(rows)


def run_arm(
    sim,
    *,
    arm,
    variation,
    to_add,
    to_drop,
    remaining_market_budget,
    remaining_market_slots,
    waiver_baselines,
    candidate_price,
    slack_floor,
    stable_solver,
    plan_key,
    personal_exclusions,
    artifact_prefix,
):
    production_simulator = sequential.simulate_history_only_branch
    production_plan_solver = sequential.solve_history_only_plan
    production_required_solver = sim.solve_managed_roster_with_required_players
    captured = []
    captured_plans = []
    plan_cursor = 0
    _REPLAN_STATIC_CACHE.clear()

    bank_key = (int(variation), str(plan_key))
    personal_exclusions = set(personal_exclusions)

    def solve_personal_plan(*args, **kwargs):
        args = list(args)
        if "unresolved_players" in kwargs:
            kwargs = dict(kwargs)
            kwargs["unresolved_players"] = (
                set(kwargs["unresolved_players"]) - personal_exclusions
            )
        elif len(args) > 6:
            args[6] = set(args[6]) - personal_exclusions
        else:
            raise TypeError("Could not locate unresolved_players in plan solve.")
        return production_plan_solver(*args, **kwargs)

    if arm == "baseline":
        def plan_solver(*args, **kwargs):
            plan = solve_personal_plan(*args, **kwargs)
            captured_plans.append(plan)
            return plan
    elif arm in {"bounded_replan", "bounded_guard"}:
        plan_bank = _COMPILED_PLAN_BANK.get(bank_key)
        if plan_bank is None:
            plan_bank = read_compiled_plan_bank(variation, plan_key)
            if plan_bank is not None:
                _COMPILED_PLAN_BANK[bank_key] = plan_bank
        if plan_bank is None:
            raise RuntimeError(
                "Run baseline before experimental arms so compiled plans are paired."
            )

        def plan_solver(*args, **kwargs):
            del args, kwargs
            nonlocal plan_cursor
            if plan_cursor >= len(plan_bank):
                raise RuntimeError("Experimental arm requested too many compiled plans.")
            plan = plan_bank[plan_cursor]
            plan_cursor += 1
            return plan
    else:
        plan_solver = (
            solve_personal_plan if personal_exclusions else production_plan_solver
        )

    if arm == "baseline" and not personal_exclusions:
        policy_simulator = production_simulator
    else:
        mode_by_arm = {
            "baseline": "baseline",
            "slack_replan": "slack",
            "purchase_replan": "purchase",
            "bounded_replan": "bounded",
            "bounded_guard": "bounded_guard",
        }
        mode = mode_by_arm[arm]

        def policy_simulator(*args, **kwargs):
            return simulate_budget_aware_branch(
                *args,
                **kwargs,
                replan_mode=mode,
                slack_floor=slack_floor,
                personal_exclusions=personal_exclusions,
            )

    def capture_simulator(*args, **kwargs):
        branch = policy_simulator(*args, **kwargs)
        captured.append({
            "branch": "buy" if kwargs.get("force_buy") else "pass",
            "complete": bool(branch.get("complete")),
            "failure_reason": branch.get("failure_reason"),
            "roster_spend": float(branch.get("salary_spend", np.nan)),
            "unused_salary": (
                float(audit.SALARY_CAP) - float(branch.get("salary_spend", np.nan))
            ),
            "full_replans": int(branch.get("full_replans", 0)),
            "budget_replans": int(branch.get("budget_replans", 0)),
            "bounded_triggers": int(branch.get("bounded_triggers", 0)),
            "bounded_swaps": int(branch.get("bounded_swaps", 0)),
            "bounded_rebuilds": int(branch.get("bounded_rebuilds", 0)),
            "guard_rejections": int(branch.get("guard_rejections", 0)),
            "max_projected_slack": float(branch.get("max_projected_slack", 0.0)),
            "roster": " | ".join(branch.get("roster", ())),
            "salary_map": json.dumps(
                dict(branch.get("salary_map", ())),
                sort_keys=True,
            ),
        })
        return branch

    label = "_".join(
        part
        for part in (artifact_prefix, f"variation{variation}_{arm}")
        if part
    )
    sequential.simulate_history_only_branch = capture_simulator
    sequential.solve_history_only_plan = plan_solver
    if stable_solver:
        sim.solve_managed_roster_with_required_players = types.MethodType(
            stable_required_roster_solve,
            sim,
        )
    started = time.perf_counter()
    try:
        result = audit.run_case(
            sim,
            to_add=to_add,
            to_drop=to_drop,
            remaining_market_budget=remaining_market_budget,
            remaining_market_slots=remaining_market_slots,
            waiver_baselines=waiver_baselines,
            candidate_price=candidate_price,
            label=label,
            variation=variation,
            enforce_top_n=True,
            use_selection_premium=False,
            profile_bid=False,
            capture_paths=False,
        )
    finally:
        sequential.simulate_history_only_branch = production_simulator
        sequential.solve_history_only_plan = production_plan_solver
        sim.solve_managed_roster_with_required_players = production_required_solver

    if arm == "baseline":
        _COMPILED_PLAN_BANK[bank_key] = list(captured_plans)
        write_compiled_plan_bank(variation, captured_plans, plan_key)
    elif (
        arm in {"bounded_replan", "bounded_guard"}
        and plan_cursor != len(_COMPILED_PLAN_BANK[bank_key])
    ):
        raise AssertionError(
            f"{arm} variation {variation}: replayed {plan_cursor} compiled plans, "
            f"expected {len(_COMPILED_PLAN_BANK[bank_key])}."
        )

    expected = 2 * int(result["requested_paths"])
    if len(captured) != expected:
        raise AssertionError(
            f"{arm} variation {variation}: captured {len(captured)} branches, "
            f"expected {expected}."
        )
    paths = pd.DataFrame(captured)
    paths.insert(0, "path_index", np.arange(len(paths)) // 2)
    paths.insert(0, "variation", int(variation))
    paths.insert(0, "arm", arm)
    paths.to_csv(RESULTS_DIR / f"paths_{label}.csv", index=False)

    decision = {
        "arm": arm,
        "variation": int(variation),
        "price": int(candidate_price),
        "recommendation": result.get("recommendation"),
        "gain": result.get("SequentialGain"),
        "se": result.get("SequentialSE"),
        "lcb80": result.get("SequentialLCB80"),
        "buy_ev": result.get("BuyEV"),
        "pass_ev": result.get("PassEV"),
        "buy_season_p10": result.get("BuySeasonP10"),
        "pass_season_p10": result.get("PassSeasonP10"),
        "season_p10_delta": result.get("SeasonP10Delta"),
        "gain_p10": result.get("GainP10"),
        "win_rate": result.get("WinRate"),
        "buy_completion": result.get("BuyCompletion"),
        "pass_completion": result.get("PassCompletion"),
        "paired_rate": result.get("PairedRate"),
        "block_positive_rate": result.get("BlockPositiveRate"),
        "runtime_seconds": time.perf_counter() - started,
    }
    return decision, paths


def build_state(conn):
    sim = FootballSimulation(
        conn,
        audit.YEAR,
        audit.LINEUP_REQUIRE,
        audit.SALARY_CAP,
        audit.PRED_VERSION,
        audit.LEAGUE,
        sal_pred_actual=audit.SALARY_SOURCE,
    )
    sim.load_weekly_template_profiles()
    keepers = pd.read_sql_query(
        """
        SELECT player, player_key, keeper_salary
        FROM League_Keepers
        WHERE year = :year AND league = :league
        """,
        conn,
        params={"year": audit.YEAR, "league": audit.LEAGUE},
    )
    canonical_by_key = sim.player_data.set_index("player_key").player
    keepers["canonical_player"] = keepers.player_key.map(canonical_by_key)
    if keepers.canonical_player.isna().any():
        raise ValueError("Keeper keys failed to map to the current player pool.")
    keeper_salary_map = dict(zip(
        keepers.canonical_player,
        keepers.keeper_salary.astype(float),
    ))
    to_add = {
        "players": list(audit.FIXED_SALARIES),
        "salaries": list(audit.FIXED_SALARIES.values()),
    }
    to_drop = sorted(set(keeper_salary_map) - set(audit.FIXED_SALARIES))
    nonkeeper_fixed_spend = sum(
        salary
        for player, salary in audit.FIXED_SALARIES.items()
        if player not in keeper_salary_map
    )
    remaining_market_budget = float(
        audit.NUM_TEAMS * audit.SALARY_CAP
        - sum(keeper_salary_map.values())
        - nonkeeper_fixed_spend
    )
    remaining_market_slots = int(
        audit.NUM_TEAMS * audit.ROSTER_SIZE
        - len(keeper_salary_map)
        - sum(
            player not in keeper_salary_map
            for player in audit.FIXED_SALARIES
        )
    )
    candidate_price = int(round(float(
        sim.player_data.loc[
            sim.player_data.player.eq(audit.CANDIDATE),
            "salary",
        ].iloc[0]
    )))
    waiver_baselines = sim.estimate_waiver_baselines(
        num_teams=audit.NUM_TEAMS,
        roster_size=audit.ROSTER_SIZE,
    )
    return {
        "sim": sim,
        "to_add": to_add,
        "to_drop": to_drop,
        "remaining_market_budget": remaining_market_budget,
        "remaining_market_slots": remaining_market_slots,
        "candidate_price": candidate_price,
        "waiver_baselines": waiver_baselines,
    }


def main(
    variations,
    slack_floor,
    arms,
    summary_prefix,
    stable_solver,
    plan_key,
    exclude_players,
    pool_exclude_players,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit.RESULTS_DIR = RESULTS_DIR
    valid_arms = {
        "baseline",
        "slack_replan",
        "purchase_replan",
        "bounded_replan",
        "bounded_guard",
    }
    unknown_arms = sorted(set(arms) - valid_arms)
    if unknown_arms:
        raise ValueError("Unknown arms: " + ", ".join(unknown_arms))
    if plan_key and not all(char.isalnum() or char in "-_" for char in plan_key):
        raise ValueError("plan_key may contain only letters, digits, '-' and '_'.")
    database_uri = f"file:{audit.APP_DB.as_posix()}?mode=ro"
    conn = sqlite3.connect(database_uri, uri=True)
    try:
        state = build_state(conn)
        personal_exclusions = set(exclude_players)
        pool_exclusions = set(pool_exclude_players)
        unknown_exclusions = sorted(
            (personal_exclusions | pool_exclusions)
            - set(state["sim"].player_data.player)
        )
        if unknown_exclusions:
            raise ValueError(
                "Excluded players are absent from the current pool: "
                + ", ".join(unknown_exclusions)
            )
        protected_exclusions = sorted(
            (personal_exclusions | pool_exclusions)
            & ({audit.CANDIDATE} | set(audit.FIXED_SALARIES))
        )
        if protected_exclusions:
            raise ValueError(
                "Cannot exclude the candidate or fixed roster players: "
                + ", ".join(protected_exclusions)
            )
        state["to_drop"] = sorted(set(state["to_drop"]) | pool_exclusions)
        decision_rows = []
        path_frames = []
        for variation in variations:
            for arm in arms:
                print(f"variation={variation} arm={arm}", flush=True)
                decision, paths = run_arm(
                    state["sim"],
                    arm=arm,
                    variation=int(variation),
                    to_add=state["to_add"],
                    to_drop=state["to_drop"],
                    remaining_market_budget=state["remaining_market_budget"],
                    remaining_market_slots=state["remaining_market_slots"],
                    waiver_baselines=state["waiver_baselines"],
                    candidate_price=state["candidate_price"],
                    slack_floor=float(slack_floor),
                    stable_solver=bool(stable_solver),
                    plan_key=str(plan_key),
                    personal_exclusions=personal_exclusions,
                    artifact_prefix=str(summary_prefix),
                )
                decision_rows.append(decision)
                path_frames.append(paths)
                print(
                    f"  gain={decision['gain']:+.2f} "
                    f"lcb80={decision['lcb80']:+.2f} "
                    f"decision={decision['recommendation']} "
                    f"runtime={decision['runtime_seconds']:.1f}s",
                    flush=True,
                )
        decisions = pd.DataFrame(decision_rows)
        paths = pd.concat(path_frames, ignore_index=True)
        path_summary = summarize_paths(paths)
        prefix = f"{summary_prefix}_" if summary_prefix else ""
        decisions.to_csv(
            RESULTS_DIR / f"{prefix}arm_decision_summary.csv",
            index=False,
        )
        paths.to_csv(RESULTS_DIR / f"{prefix}all_paths.csv", index=False)
        path_summary.to_csv(
            RESULTS_DIR / f"{prefix}arm_path_summary.csv",
            index=False,
        )
        metadata = {
            "candidate": audit.CANDIDATE,
            "candidate_price": state["candidate_price"],
            "fixed_salaries": audit.FIXED_SALARIES,
            "variations": [int(value) for value in variations],
            "arms": list(arms),
            "slack_trigger": "max(slack_floor, open_slots)",
            "slack_floor": float(slack_floor),
            "bounded_candidate_limit": BOUNDED_CANDIDATE_LIMIT,
            "bounded_outgoing_limit": BOUNDED_OUTGOING_LIMIT,
            "bounded_swap_limit": BOUNDED_SWAP_LIMIT,
            "stable_required_roster_solver": bool(stable_solver),
            "compiled_plan_key": str(plan_key),
            "personal_exclusions": sorted(personal_exclusions),
            "pool_exclusions": sorted(pool_exclusions),
            "production_code_changed": False,
        }
        with (RESULTS_DIR / f"{prefix}metadata.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print("\nDecision summary", flush=True)
        print(decisions.to_string(index=False), flush=True)
        print("\nPath summary", flush=True)
        print(path_summary.to_string(index=False), flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variations", nargs="+", type=int, default=[14])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["baseline", "slack_replan"],
        choices=[
            "baseline",
            "slack_replan",
            "purchase_replan",
            "bounded_replan",
            "bounded_guard",
        ],
    )
    parser.add_argument("--slack-floor", type=float, default=5.0)
    parser.add_argument("--summary-prefix", default="")
    parser.add_argument("--stable-solver", action="store_true")
    parser.add_argument("--plan-key", default="")
    parser.add_argument("--exclude-players", nargs="+", default=[])
    parser.add_argument("--pool-exclude-players", nargs="+", default=[])
    arguments = parser.parse_args()
    main(
        arguments.variations,
        arguments.slack_floor,
        arguments.arms,
        arguments.summary_prefix,
        arguments.stable_solver,
        arguments.plan_key,
        arguments.exclude_players,
        arguments.pool_exclude_players,
    )
