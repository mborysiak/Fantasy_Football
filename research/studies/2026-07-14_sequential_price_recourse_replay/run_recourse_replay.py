"""Sequential, non-anticipating guardrail replay on historical price tapes.

Historical nomination order and losing bids are unavailable.  This runner is
therefore a paired partial-equilibrium stress test, not an identified replay of
the historical auction room.  See README.md for the estimand and boundaries.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
BUFFER_STUDY_DIR = (
    ROOT / "research" / "studies" / "2026-07-14_nominal_salary_buffer_replay"
)
BUFFER_RUNNER = BUFFER_STUDY_DIR / "run_buffer_replay.py"
BASE_MANIFEST = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-13_managed_auction_rolling_replay"
    / "results"
    / "source_manifest.json"
)
RAW_SALARY_DIR = ROOT / "Data" / "OtherData" / "Salaries"


def load_buffer_replay() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_nominal_salary_buffer_replay",
        BUFFER_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import buffer replay: {BUFFER_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


buffer_replay = load_buffer_replay()
base = buffer_replay.base

ORDER_REGIMES = ("tier_early", "uniform", "position_run", "star_late")
ORDER_SEED_OFFSETS = {
    "tier_early": 0,
    "uniform": 1_000,
    "position_run": 2_000,
    "star_late": 3_000,
}
PRICE_RULES = ("clearing", "plus_one")
POLICY_MODES = ("strict", "operational")
DEFAULT_BUFFERS = (5.0, 10.0)
CURRENT_WAIVER = "current_projected"
CURRENT_BENCH_WEIGHT = 0.25
# GLPK can return continuous spend totals a fraction of a cent above the exact
# row bound. Paid-price rosters are still audited against $298 at 1e-8.
SOLVER_CAP_TOLERANCE = 1e-3
# Execution-control-only predecessor accepted for checkpoints written before
# trial sharding was added.  Simulation, policy, and output logic are unchanged.
COMPATIBLE_CHECKPOINT_RUNNER_SHA256 = {
    "330f62dd3a944e01bef6e68c95caf9b9a792b3e98c821378d12c721381f76722",
    "11f7f8bf268bb3d9cc52001f116c8360c45376549f23886616e1fef4945cfed0",
}
POSITION_ORDER = {pos: idx for idx, pos in enumerate(base.POSITIONS)}
MANUAL_2022_SKILL_POS = {
    "robbieanderson": "WR",
    "treylance": "QB",
    "michaelthomas": "WR",
}
RAW_PLAYER_ALIASES = {
    "Tetairoa McMillan": "Tet McMillan",
}
KICKER_2022_KEYS = {
    "mattprater",
    "tylerbass",
    "cadeyork",
    "brandonmcmanus",
    "nickfolk",
    "gregjoseph",
    "evanmcpherson",
    "mattgay",
    "harrisonbutker",
    "danielcarlson",
    "chrisboswell",
    "justintucker",
}
RAW_GOLDEN_TOTALS = {
    2022: {
        "rows": 179,
        "priced_rows": 169,
        "skill_rows": 156,
        "priced_skill_rows": 149,
        "special_rows": 23,
        "priced_special_rows": 20,
        "known_skill_spend": 3510.0,
        "keeper_count": 20,
        "keeper_spend": 690.0,
        "known_raw_spend": 3544.0,
    },
    2023: {
        "rows": 180,
        "priced_rows": 180,
        "skill_rows": 155,
        "priced_skill_rows": 155,
        "special_rows": 25,
        "priced_special_rows": 25,
        "known_skill_spend": 3555.0,
        "keeper_count": 21,
        "keeper_spend": 871.0,
        "known_raw_spend": 3596.0,
    },
    2024: {
        "rows": 179,
        "priced_rows": 179,
        "skill_rows": 155,
        "priced_skill_rows": 155,
        "special_rows": 24,
        "priced_special_rows": 24,
        "known_skill_spend": 3530.0,
        "keeper_count": 18,
        "keeper_spend": 563.0,
        "known_raw_spend": 3573.0,
    },
    2025: {
        "rows": 180,
        "priced_rows": 179,
        "skill_rows": 156,
        "priced_skill_rows": 156,
        "special_rows": 24,
        "priced_special_rows": 23,
        "known_skill_spend": 3550.0,
        "keeper_count": 15,
        "keeper_spend": 407.0,
        "known_raw_spend": 3586.0,
    },
}


def parse_money(value: Any) -> tuple[float, bool]:
    text = "" if value is None else str(value).strip()
    number = re.sub(r"[^0-9.]", "", text)
    if number == "":
        return 1.0, True
    return float(number), False


def strip_espn_suffix(value: Any) -> tuple[str, str | None, str | None]:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    match = re.match(
        r"^(.*?)\s+[A-Za-z0-9]{2,4},\s*(QB|RB|WR|TE|K|D/ST)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return text, None, None
    suffix = text[len(match.group(1)) :]
    team_match = re.match(
        r"^\s+([A-Za-z0-9]{2,4}),",
        suffix,
        flags=re.IGNORECASE,
    )
    nfl_team = None if team_match is None else team_match.group(1).upper()
    return match.group(1).strip(), nfl_team, match.group(2).upper()


def parse_raw_rosters(
    year: int,
    forecast: pd.DataFrame,
    target_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the 12 ESPN owner blocks and reconcile the skill positions."""
    path = RAW_SALARY_DIR / f"beta_{year}_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    roster_block_id = 0
    roster_row = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source_line, csv_row in enumerate(csv.reader(handle), start=1):
            padded = (csv_row + ["", "", ""])[:3]
            first = padded[0].strip()
            second = padded[1].strip()
            is_header = (
                first.casefold() == "player"
                and (
                    "salary" in second.casefold()
                    or "offer amount" in second.casefold()
                )
            )
            if is_header:
                roster_block_id += 1
                roster_row = 0
                continue
            if not any(value.strip() for value in padded):
                continue
            if roster_block_id == 0:
                raise ValueError(f"Roster row appeared before the first header in {path}.")
            roster_row += 1
            display, nfl_team, embedded_pos = strip_espn_suffix(first)
            display = RAW_PLAYER_ALIASES.get(display, display)
            replay_price, salary_missing = parse_money(second)
            keeper_text = padded[2].strip().casefold()
            if keeper_text not in {"", "0", "1", "false", "no"}:
                raise ValueError(
                    f"{year} line {source_line}: invalid keeper flag {padded[2]!r}."
                )
            rows.append(
                {
                    "year": year,
                    "source_line": source_line,
                    "roster_block_id": roster_block_id,
                    "roster_row": roster_row,
                    "raw_player": first,
                    "player": display,
                    "nfl_team": nfl_team,
                    "embedded_pos": embedded_pos,
                    "actual_salary": (
                        np.nan if salary_missing else replay_price
                    ),
                    "replay_price": replay_price,
                    "price_source": (
                        "imputed_blank_one" if salary_missing else "observed"
                    ),
                    "salary_missing_filled_one": salary_missing,
                    "is_keeper": keeper_text not in {"", "0", "false", "no"},
                }
            )

    roster = base.add_identity(pd.DataFrame(rows))
    if roster_block_id != base.NUM_TEAMS:
        raise AssertionError(
            f"{year}: expected 12 roster blocks, found {roster_block_id}."
        )
    if roster.player_key.duplicated().any():
        duplicates = roster.loc[roster.player_key.duplicated(False), "player"].tolist()
        raise AssertionError(f"{year}: duplicate raw roster players: {duplicates}")

    pos_lookup: dict[str, str] = {}
    for source in (target_features, forecast):
        if not {"player_key", "pos"}.issubset(source.columns):
            continue
        ordered = source.copy()
        if "preseason_proj_ppg" in ordered:
            ordered = ordered.sort_values("preseason_proj_ppg", ascending=False)
        elif "pred_fp_per_game" in ordered:
            ordered = ordered.sort_values("pred_fp_per_game", ascending=False)
        pos_lookup.update(
            ordered.drop_duplicates("player_key").set_index("player_key").pos.to_dict()
        )
    pos_lookup.update(MANUAL_2022_SKILL_POS)

    positions = []
    position_sources = []
    for row in roster.itertuples(index=False):
        if row.embedded_pos is not None:
            positions.append(row.embedded_pos)
            position_sources.append("espn_suffix")
        elif "d/st" in row.raw_player.casefold() or row.player_key == "eagles":
            positions.append("D/ST")
            position_sources.append("name_defense")
        elif row.player_key in pos_lookup:
            positions.append(base.normalize_pos(pos_lookup[row.player_key]))
            position_sources.append("frozen_skill_lookup")
        elif year == 2022 and row.player_key in KICKER_2022_KEYS:
            positions.append("K")
            position_sources.append("2022_explicit_kicker")
        else:
            raise AssertionError(
                f"{year}: unresolved raw roster position for {row.raw_player!r}."
            )
    roster["pos"] = positions
    roster["position_source"] = position_sources

    if not roster.pos.isin([*base.POSITIONS, "K", "D/ST"]).all():
        bad = roster.loc[
            ~roster.pos.isin([*base.POSITIONS, "K", "D/ST"]),
            ["player", "pos"],
        ]
        raise AssertionError(f"{year}: unsupported raw positions: {bad.to_dict('records')}")
    if not roster.groupby("roster_block_id").size().between(14, 15).all():
        raise AssertionError(f"{year}: raw roster blocks are outside 14-15 rows.")
    if (roster.groupby("roster_block_id").replay_price.sum() > 300.0 + 1e-8).any():
        raise AssertionError(f"{year}: a raw roster block exceeds the $300 league cap.")

    skill_mask = roster.pos.isin(base.POSITIONS)
    priced_mask = roster.actual_salary.notna()
    golden = RAW_GOLDEN_TOTALS[year]
    observed = {
        "rows": len(roster),
        "priced_rows": int(priced_mask.sum()),
        "skill_rows": int(skill_mask.sum()),
        "priced_skill_rows": int((skill_mask & priced_mask).sum()),
        "special_rows": int((~skill_mask).sum()),
        "priced_special_rows": int(((~skill_mask) & priced_mask).sum()),
        "known_skill_spend": float(roster.loc[skill_mask, "actual_salary"].sum()),
        "keeper_count": int(roster.loc[skill_mask, "is_keeper"].sum()),
        "keeper_spend": float(
            roster.loc[skill_mask & roster.is_keeper, "actual_salary"].sum()
        ),
        "known_raw_spend": float(roster.actual_salary.sum()),
    }
    if observed != golden:
        raise AssertionError(
            f"{year}: raw roster golden gate failed: observed={observed}, expected={golden}."
        )

    owner_rows = []
    for block_id, group in roster.groupby("roster_block_id", sort=True):
        skill = group[group.pos.isin(base.POSITIONS)]
        owner_rows.append(
            {
                "year": year,
                "roster_block_id": block_id,
                "raw_rows": len(group),
                "skill_rows": len(skill),
                "k_rows": int(group.pos.eq("K").sum()),
                "dst_rows": int(group.pos.eq("D/ST").sum()),
                "keeper_skill_rows": int(skill.is_keeper.sum()),
                "known_raw_total_spend": float(group.actual_salary.sum()),
                "replay_total_spend": float(group.replay_price.sum()),
                "known_skill_spend": float(skill.actual_salary.sum()),
                "replay_skill_spend": float(skill.replay_price.sum()),
                "missing_salary_rows": int(group.salary_missing_filled_one.sum()),
            }
        )
    return roster.reset_index(drop=True), pd.DataFrame(owner_rows)


def build_price_tape(
    year: int,
    roster: pd.DataFrame,
    candidate_forecast: pd.DataFrame,
    target_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the skill-player tape plus explicit contract-fill events."""
    skill = roster[roster.pos.isin(base.POSITIONS)].copy()
    keepers = skill[skill.is_keeper].copy()
    tape = skill[~skill.is_keeper].copy()
    expected_events = base.TOTAL_MARKET_SLOTS - len(keepers)
    if len(tape) > expected_events:
        raise AssertionError(
            f"{year}: {len(tape)} nonkeeper skill rows exceed {expected_events} open slots."
        )

    candidate_by_key = candidate_forecast.drop_duplicates("player_key").set_index(
        "player_key"
    )
    feature_ppg = (
        target_features.sort_values("preseason_proj_ppg", ascending=False)
        .drop_duplicates("player_key")
        .set_index("player_key")
        .preseason_proj_ppg.to_dict()
    )

    event_rows: list[dict[str, Any]] = []
    for row in tape.itertuples(index=False):
        actionable = row.player_key in candidate_by_key.index
        if actionable:
            candidate = candidate_by_key.loc[row.player_key]
            canonical_player = candidate.player
            preauction_score = float(candidate.pred_fp_per_game)
            raw_nominal = float(candidate.salary)
        else:
            canonical_player = row.player
            preauction_score = float(feature_ppg.get(row.player_key, 0.0))
            raw_nominal = np.nan
        event_rows.append(
            {
                "year": year,
                "source_roster_block_id": int(row.roster_block_id),
                "player": canonical_player,
                "raw_player": row.raw_player,
                "player_key": row.player_key,
                "pos": row.pos,
                "clearing_price": float(row.replay_price),
                "observed_actual_salary": float(row.actual_salary),
                "price_source": row.price_source,
                "salary_missing_filled_one": bool(row.salary_missing_filled_one),
                "actionable": bool(actionable),
                "event_source": (
                    "historical_forecast_matched"
                    if actionable
                    else "historical_opponent_only"
                ),
                "is_synthetic_contract_fill": False,
                "preauction_score": preauction_score,
                "raw_nominal_salary": raw_nominal,
            }
        )

    deficit = expected_events - len(event_rows)
    tape_keys = {row["player_key"] for row in event_rows}
    if deficit:
        for fill_idx in range(deficit):
            synthetic_key = f"__missing_skill_slot_{fill_idx + 1}__"
            event_rows.append(
                {
                    "year": year,
                    "source_roster_block_id": -1,
                    "player": f"Missing Skill Slot {fill_idx + 1}",
                    "raw_player": "",
                    "player_key": synthetic_key,
                    "pos": "UNK",
                    "clearing_price": 1.0,
                    "observed_actual_salary": np.nan,
                    "price_source": "synthetic_contract_fill_one",
                    "salary_missing_filled_one": True,
                    "actionable": False,
                    "event_source": "synthetic_contract_fill",
                    "is_synthetic_contract_fill": True,
                    "preauction_score": 0.0,
                    "raw_nominal_salary": np.nan,
                }
            )

    events = pd.DataFrame(event_rows)
    if len(events) != expected_events:
        raise AssertionError(f"{year}: price tape does not fill the modeled skill market.")
    if events.player_key.duplicated().any():
        raise AssertionError(f"{year}: duplicate player on the price tape.")
    valid_pos = events.pos.isin(base.POSITIONS) | (
        events.is_synthetic_contract_fill & events.pos.eq("UNK")
    )
    if not valid_pos.all():
        raise AssertionError(f"{year}: K/DST leaked onto the skill-player price tape.")

    opaque_historical = events[
        ~events.actionable & ~events.is_synthetic_contract_fill
    ]

    audit = pd.DataFrame(
        [
            {
                "year": year,
                "raw_skill_rows": int(len(skill)),
                "keeper_skill_rows": int(len(keepers)),
                "keeper_skill_spend": float(keepers.actual_salary.sum()),
                "historical_nonkeeper_events": int(len(tape)),
                "synthetic_contract_fill_events": int(deficit),
                "modeled_open_slots": int(expected_events),
                "forecast_matched_events": int(events.actionable.sum()),
                "opponent_only_events": int((~events.actionable).sum()),
                "historical_opponent_only_events": int(len(opaque_historical)),
                "historical_opponent_only_spend": float(
                    opaque_historical.clearing_price.sum()
                ),
                "historical_opponent_only_max_price": float(
                    opaque_historical.clearing_price.max()
                    if len(opaque_historical)
                    else 0.0
                ),
                "historical_nonkeeper_known_spend": float(tape.actual_salary.sum()),
                "historical_nonkeeper_replay_spend": float(tape.replay_price.sum()),
                "modeled_initial_market_budget": float(
                    base.TOTAL_MARKET_BUDGET - keepers.replay_price.sum()
                ),
            }
        ]
    )
    return events.reset_index(drop=True), audit


def make_order(tape: pd.DataFrame, regime: str, seed: int) -> pd.DataFrame:
    if regime not in ORDER_REGIMES:
        raise ValueError(regime)
    rng = np.random.default_rng(seed)
    frame = tape.copy().reset_index(drop=True)
    frame["order_jitter"] = rng.random(len(frame))
    score = frame.preauction_score.fillna(0.0).to_numpy(dtype=float)
    rank = pd.Series(-score).rank(method="first").to_numpy(dtype=int) - 1
    frame["projection_tier"] = np.minimum(3, (4 * rank) // max(len(frame), 1))

    if regime == "uniform":
        ordered = frame.iloc[rng.permutation(len(frame))]
    elif regime == "tier_early":
        ordered = frame.sort_values(
            ["projection_tier", "order_jitter", "player_key"],
            ascending=[True, True, True],
        )
    elif regime == "star_late":
        ordered = frame.sort_values(
            ["projection_tier", "order_jitter", "player_key"],
            ascending=[False, True, True],
        )
    else:
        queues: dict[str, list[int]] = {}
        for pos in base.POSITIONS:
            pos_frame = frame[frame.pos.eq(pos)].sort_values(
                ["projection_tier", "order_jitter", "player_key"]
            )
            queues[pos] = pos_frame.index.tolist()
        indices: list[int] = []
        while any(queues.values()):
            cycle = list(rng.permutation(base.POSITIONS))
            for pos in cycle:
                take = min(3, len(queues[pos]))
                indices.extend(queues[pos][:take])
                del queues[pos][:take]
        unknown = frame.index[~frame.pos.isin(base.POSITIONS)].tolist()
        for idx in unknown:
            insert_at = int(rng.integers(0, len(indices) + 1))
            indices.insert(insert_at, idx)
        ordered = frame.loc[indices]

    ordered = ordered.drop(columns="order_jitter").reset_index(drop=True)
    ordered["nomination_number"] = np.arange(1, len(ordered) + 1)
    return ordered


def nominal_stages(
    requested_buffer: float,
    policy_mode: str,
) -> list[float | None]:
    if policy_mode == "strict":
        return [float(requested_buffer)]
    if policy_mode != "operational":
        raise ValueError(policy_mode)
    stages: list[float | None] = [float(requested_buffer)]
    if requested_buffer < 10.0:
        stages.append(10.0)
    stages.append(None)
    output: list[float | None] = []
    for stage in stages:
        if stage not in output:
            output.append(stage)
    return output


def managed_objective(
    sim: Any,
    predictions: pd.DataFrame,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    context_idx: np.ndarray,
    base_players: tuple[str, ...],
    waiver_baseline: dict[str, float],
    cache: dict[tuple[tuple[int, ...], tuple[str, ...]], np.ndarray],
) -> np.ndarray:
    context_key = tuple(int(value) for value in context_idx)
    player_key = tuple(sorted(base_players))
    key = (context_key, player_key)
    if key in cache:
        return cache[key]
    values = []
    for idx in context_idx:
        values.append(
            sim.managed_marginal_values(
                weekly[int(idx)],
                predictions.pos.to_numpy(),
                decisions[int(idx)],
                predictions.player.to_numpy(),
                base_players=list(base_players),
                waiver_baselines=waiver_baseline,
                lineup_require=base.LINEUP_REQUIRE,
                bench_upside_weight=CURRENT_BENCH_WEIGHT,
                played_mask=played[int(idx)],
            )
        )
    objective = np.mean(values, axis=0).astype(np.float32)
    cache[key] = objective
    return objective


def solve_branch(
    *,
    sim: Any,
    predictions: pd.DataFrame,
    player_keys: np.ndarray,
    raw_salary_market: np.ndarray,
    raw_nominal_salary: np.ndarray,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    reference_weekly: np.ndarray,
    reference_decisions: np.ndarray,
    reference_played: np.ndarray,
    evaluation_weekly: np.ndarray,
    evaluation_decisions: np.ndarray,
    evaluation_played: np.ndarray,
    context_idx: np.ndarray,
    waiver_baseline: dict[str, float],
    sold_keys_after: set[str],
    fixed_salary_map_after: dict[str, float],
    post_market_budget: float,
    post_market_slots: int,
    nominal_buffer: float | None,
    objective_cache: dict[tuple[tuple[int, ...], tuple[str, ...]], np.ndarray],
    forecast_cache: dict[tuple[tuple[str, ...], str], float],
    enforce_top_n: bool = True,
) -> dict[str, Any] | None:
    fixed_players = tuple(sorted(fixed_salary_map_after))
    name_to_key = dict(zip(predictions.player, player_keys))
    fixed_keys = {name_to_key[player] for player in fixed_players}
    keep_mask = np.array(
        [key not in sold_keys_after or key in fixed_keys for key in player_keys],
        dtype=bool,
    )
    if int(keep_mask.sum()) < base.ROSTER_SIZE:
        return None

    branch_predictions = predictions.loc[keep_mask].reset_index(drop=True).copy()
    branch_keys = player_keys[keep_mask]
    available_mask = ~branch_predictions.player.isin(fixed_players).to_numpy()
    if post_market_slots > int(available_mask.sum()):
        return None
    if post_market_budget < post_market_slots - 1e-8:
        return None

    sampled_salary = sim.normalize_salary_market_values(
        raw_salary_market[keep_mask],
        available_mask,
        remaining_market_budget=post_market_budget,
        remaining_market_slots=post_market_slots,
    )
    nominal_salary = sim.normalize_salary_market_values(
        raw_nominal_salary[keep_mask],
        available_mask,
        remaining_market_budget=post_market_budget,
        remaining_market_slots=post_market_slots,
    )
    for idx, player in enumerate(branch_predictions.player):
        if player in fixed_salary_map_after:
            sampled_salary[idx] = fixed_salary_map_after[player]
            nominal_salary[idx] = fixed_salary_map_after[player]
    branch_predictions["salary"] = sampled_salary

    top_count = min(base.TOP_N, len(branch_predictions))
    top_idx = np.argsort(raw_nominal_salary[keep_mask])[-top_count:]
    top_n = branch_predictions.iloc[top_idx].player.tolist()
    h_player_add = {player: -1 for player in fixed_players}
    enforce_top_n = bool(enforce_top_n and len(fixed_players) < base.ROSTER_SIZE)
    static = sim.build_managed_ilp_static_matrices(
        branch_predictions,
        h_player_add,
        list(fixed_players),
        top_n,
        base.ROSTER_SIZE,
        base.POS_MIN,
        base.POS_MAX,
        enforce_top_n=enforce_top_n,
    )

    full_objective = managed_objective(
        sim,
        predictions,
        weekly,
        decisions,
        played,
        context_idx,
        fixed_players,
        waiver_baseline,
        objective_cache,
    )
    solved = sim._solve_managed_scenario(
        branch_predictions,
        full_objective[keep_mask],
        reference_weekly[keep_mask],
        reference_decisions[keep_mask],
        static,
        list(fixed_players),
        fixed_salary_map_after,
        top_n,
        base.ROSTER_SIZE,
        base.POS_MIN,
        base.POS_MAX,
        waiver_baseline,
        base.LINEUP_REQUIRE,
        enforce_top_n,
        refine_roster=False,
        score_roster=False,
        salary_values=sampled_salary,
        played_mask=reference_played[keep_mask],
        nominal_salary_values=(None if nominal_buffer is None else nominal_salary),
        nominal_salary_cap=(
            None if nominal_buffer is None else base.SALARY_CAP + nominal_buffer
        ),
    )
    if solved is None:
        return None

    selected = np.asarray(solved["selected_mask"], dtype=bool)
    roster = tuple(sorted(solved["selected_players"]))
    if len(roster) != base.ROSTER_SIZE or not set(fixed_players).issubset(roster):
        raise AssertionError("Branch solve failed to retain every deterministic purchase.")
    forecast_ev = base.forecast_roster_ev(
        roster,
        CURRENT_WAIVER,
        waiver_baseline,
        predictions,
        evaluation_weekly,
        evaluation_decisions,
        evaluation_played,
        forecast_cache,
    )
    sampled_spend = float(sampled_salary[selected].sum())
    nominal_spend = float(nominal_salary[selected].sum())
    if sampled_spend > base.SALARY_CAP + SOLVER_CAP_TOLERANCE:
        raise AssertionError(
            "Branch continuation exceeds the sampled $298 cap "
            f"(spend={sampled_spend:.12f}, over={sampled_spend - base.SALARY_CAP:.12g})."
        )
    if (
        nominal_buffer is not None
        and nominal_spend
        > base.SALARY_CAP + nominal_buffer + SOLVER_CAP_TOLERANCE
    ):
        raise AssertionError(
            "Branch continuation exceeds its nominal guardrail "
            f"(spend={nominal_spend:.12f}, cap={base.SALARY_CAP + nominal_buffer:.12f})."
        )
    return {
        "roster": roster,
        "forecast_ev": forecast_ev,
        "sampled_spend": sampled_spend,
        "nominal_spend": nominal_spend,
        "selected_keys": "|".join(sorted(branch_keys[selected])),
    }


def position_counts(players: list[str], predictions: pd.DataFrame) -> dict[str, int]:
    counts = (
        predictions[predictions.player.isin(players)].pos.value_counts().to_dict()
    )
    return {pos: int(counts.get(pos, 0)) for pos in base.POSITIONS}


def partial_position_feasible(counts: dict[str, int], open_slots: int) -> bool:
    if any(counts[pos] > base.POS_MAX[pos] for pos in base.POSITIONS):
        return False
    minimum_needed = sum(
        max(base.POS_MIN[pos] - counts[pos], 0) for pos in base.POSITIONS
    )
    remaining_capacity = sum(
        max(base.POS_MAX[pos] - counts[pos], 0) for pos in base.POSITIONS
    )
    return minimum_needed <= open_slots <= remaining_capacity


def run_policy_path(
    *,
    year: int,
    trial: int,
    order_regime: str,
    price_rule: str,
    policy_mode: str,
    requested_buffer: float,
    order: pd.DataFrame,
    initial_market_budget: float,
    initial_market_slots: int,
    sim: Any,
    predictions: pd.DataFrame,
    player_keys: np.ndarray,
    raw_salary_market: np.ndarray,
    raw_nominal_salary: np.ndarray,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    evaluation_weekly: np.ndarray,
    evaluation_decisions: np.ndarray,
    evaluation_played: np.ndarray,
    context_idx: np.ndarray,
    waiver_baseline: dict[str, float],
    environment: dict[str, Any],
    objective_cache: dict[tuple[tuple[int, ...], tuple[str, ...]], np.ndarray],
    forecast_cache: dict[tuple[tuple[str, ...], str], float],
    max_events: int | None = None,
    score_outcomes: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if price_rule not in PRICE_RULES:
        raise ValueError(price_rule)
    if policy_mode not in POLICY_MODES:
        raise ValueError(policy_mode)
    path_id = (
        f"{year}_{trial}_{order_regime}_{price_rule}_{policy_mode}_"
        f"buffer{requested_buffer:g}"
    )
    fixed_salary_map: dict[str, float] = {}
    fixed_clearing_map: dict[str, float] = {}
    sold_keys: set[str] = set()
    market_budget = float(initial_market_budget)
    market_slots = int(initial_market_slots)
    events_seen = 0
    forced_buys = 0
    relaxation_events = 0
    top_n_relaxation_events = 0
    generic_repairs = 0
    synthetic_contract_buys = 0
    synthetic_contract_events_seen = 0
    event_rows: list[dict[str, Any]] = []

    reference_weekly = weekly[context_idx].mean(axis=0)
    reference_decisions = decisions[context_idx].mean(axis=0)
    reference_played = np.any(played[context_idx] > 0, axis=0).astype(np.int8)
    key_to_name = dict(zip(player_keys, predictions.player))

    def append_failure_event(
        event: dict[str, Any],
        reason: str,
        *,
        buy_feasible: bool = False,
        pass_feasible: bool = False,
        buy_structural: bool = False,
        pass_structural: bool = False,
    ) -> None:
        clearing = float(event.get("clearing_price", np.nan))
        event_rows.append(
            {
                "path_id": path_id,
                "year": year,
                "trial": trial,
                "order_regime": order_regime,
                "price_rule": price_rule,
                "policy_mode": policy_mode,
                "nominal_buffer": requested_buffer,
                "nomination_number": int(event.get("nomination_number", events_seen)),
                "nominee": event.get("player", ""),
                "nominee_key": event.get("player_key", ""),
                "nominee_pos": event.get("pos", ""),
                "event_source": event.get("event_source", ""),
                "clearing_price": clearing,
                "observed_clearing_price": clearing,
                "action": "failure",
                "failure_reason": reason,
                "forced_buy": False,
                "generic_repair": False,
                "nominal_stage": f"{requested_buffer:g}",
                "nominal_relaxed": False,
                "top_n_relaxed": False,
                "buy_structural": buy_structural,
                "pass_structural": pass_structural,
                "buy_feasible": buy_feasible,
                "pass_feasible": pass_feasible,
                "strict_buy_feasible": False,
                "strict_pass_feasible": False,
                "buy_forecast_ev": np.nan,
                "pass_forecast_ev": np.nan,
                "buy_sampled_spend": np.nan,
                "pass_sampled_spend": np.nan,
                "buy_nominal_spend": np.nan,
                "pass_nominal_spend": np.nan,
                "forecast_ev_delta": np.nan,
                "roster_size_before": len(fixed_salary_map),
                "roster_size_after": len(fixed_salary_map),
                "paid_spend_after": float(sum(fixed_salary_map.values())),
                "market_budget_before": market_budget,
                "market_budget_after": market_budget,
                "market_slots_before": market_slots,
                "market_slots_after": market_slots,
            }
        )

    failure_reason = ""
    for source_event in order.itertuples(index=False):
        if len(fixed_salary_map) == base.ROSTER_SIZE:
            break
        if max_events is not None and events_seen >= max_events:
            break
        events_seen += 1
        event = source_event._asdict()
        if bool(event.get("is_synthetic_contract_fill", False)):
            synthetic_contract_events_seen += 1
        if market_slots <= 0:
            failure_reason = "market_exhausted_before_roster_completion"
            append_failure_event(event, failure_reason)
            break

        open_before = base.ROSTER_SIZE - len(fixed_salary_map)
        budget_before = market_budget
        slots_before = market_slots
        observed_clearing_price = float(event["clearing_price"])
        clearing_price = observed_clearing_price
        post_slots = market_slots - 1
        pass_structural = open_before <= post_slots

        if not bool(event["actionable"]):
            if pass_structural:
                market_budget -= clearing_price
                market_slots = post_slots
                sold_keys.add(event["player_key"])
                event_rows.append(
                    {
                        "path_id": path_id,
                        "year": year,
                        "trial": trial,
                        "order_regime": order_regime,
                        "price_rule": price_rule,
                        "nominal_buffer": requested_buffer,
                        "policy_mode": policy_mode,
                        "nomination_number": int(event["nomination_number"]),
                        "nominee": event["player"],
                        "nominee_key": event["player_key"],
                        "nominee_pos": event["pos"],
                        "event_source": event["event_source"],
                        "clearing_price": clearing_price,
                        "observed_clearing_price": observed_clearing_price,
                        "action": "opponent_only_pass",
                        "forced_buy": False,
                        "generic_repair": False,
                        "nominal_stage": requested_buffer,
                        "nominal_relaxed": False,
                        "buy_feasible": False,
                        "pass_feasible": True,
                        "buy_forecast_ev": np.nan,
                        "pass_forecast_ev": np.nan,
                        "buy_sampled_spend": np.nan,
                        "pass_sampled_spend": np.nan,
                        "buy_nominal_spend": np.nan,
                        "pass_nominal_spend": np.nan,
                        "forecast_ev_delta": np.nan,
                        "roster_size_before": base.ROSTER_SIZE - open_before,
                        "roster_size_after": len(fixed_salary_map),
                        "paid_spend_after": float(sum(fixed_salary_map.values())),
                        "market_budget_before": budget_before,
                        "market_budget_after": market_budget,
                        "market_slots_before": slots_before,
                        "market_slots_after": market_slots,
                    }
                )
                continue

            failure_reason = "forced_opponent_only_event"
            append_failure_event(
                event,
                failure_reason,
                pass_structural=pass_structural,
            )
            break

        nominee_key = str(event["player_key"])
        nominee = key_to_name.get(nominee_key)
        if nominee is None:
            failure_reason = "actionable_nominee_missing_from_predictions"
            append_failure_event(event, failure_reason)
            break
        paid_price = clearing_price + (1.0 if price_rule == "plus_one" else 0.0)
        nominee_pos = str(event["pos"])
        counts = position_counts(list(fixed_salary_map), predictions)
        pass_structural = pass_structural and partial_position_feasible(
            counts,
            open_before,
        )
        buy_counts = dict(counts)
        buy_counts[nominee_pos] += 1
        buy_structural = (
            partial_position_feasible(buy_counts, open_before - 1)
            and sum(fixed_salary_map.values())
            + paid_price
            + max(open_before - 1, 0)
            <= base.SALARY_CAP + 1e-8
            and open_before - 1 <= post_slots
            and market_budget - paid_price >= post_slots - 1e-8
        )
        pass_sold = sold_keys | {nominee_key}
        buy_sold = sold_keys | {nominee_key}
        buy_fixed = {**fixed_salary_map, nominee: paid_price}
        strict_buy_feasible = False
        strict_pass_feasible = False
        top_n_relaxed = False
        chosen_stage: float | None = requested_buffer
        buy_result = None
        pass_result = None

        for stage_idx, stage in enumerate(
            nominal_stages(requested_buffer, policy_mode)
        ):
            buy_result = None
            pass_result = None
            if buy_structural:
                buy_result = solve_branch(
                    sim=sim,
                    predictions=predictions,
                    player_keys=player_keys,
                    raw_salary_market=raw_salary_market,
                    raw_nominal_salary=raw_nominal_salary,
                    weekly=weekly,
                    decisions=decisions,
                    played=played,
                    reference_weekly=reference_weekly,
                    reference_decisions=reference_decisions,
                    reference_played=reference_played,
                    evaluation_weekly=evaluation_weekly,
                    evaluation_decisions=evaluation_decisions,
                    evaluation_played=evaluation_played,
                    context_idx=context_idx,
                    waiver_baseline=waiver_baseline,
                    sold_keys_after=buy_sold,
                    fixed_salary_map_after=buy_fixed,
                    post_market_budget=market_budget - paid_price,
                    post_market_slots=post_slots,
                    nominal_buffer=stage,
                    objective_cache=objective_cache,
                    forecast_cache=forecast_cache,
                )
            if pass_structural:
                pass_result = solve_branch(
                    sim=sim,
                    predictions=predictions,
                    player_keys=player_keys,
                    raw_salary_market=raw_salary_market,
                    raw_nominal_salary=raw_nominal_salary,
                    weekly=weekly,
                    decisions=decisions,
                    played=played,
                    reference_weekly=reference_weekly,
                    reference_decisions=reference_decisions,
                    reference_played=reference_played,
                    evaluation_weekly=evaluation_weekly,
                    evaluation_decisions=evaluation_decisions,
                    evaluation_played=evaluation_played,
                    context_idx=context_idx,
                    waiver_baseline=waiver_baseline,
                    sold_keys_after=pass_sold,
                    fixed_salary_map_after=fixed_salary_map,
                    post_market_budget=market_budget - clearing_price,
                    post_market_slots=post_slots,
                    nominal_buffer=stage,
                    objective_cache=objective_cache,
                    forecast_cache=forecast_cache,
                )
            if stage_idx == 0:
                strict_buy_feasible = buy_result is not None
                strict_pass_feasible = pass_result is not None
            if buy_result is not None or pass_result is not None:
                chosen_stage = stage
                break

        if (
            policy_mode == "operational"
            and buy_result is None
            and pass_result is None
        ):
            if buy_structural:
                buy_result = solve_branch(
                    sim=sim,
                    predictions=predictions,
                    player_keys=player_keys,
                    raw_salary_market=raw_salary_market,
                    raw_nominal_salary=raw_nominal_salary,
                    weekly=weekly,
                    decisions=decisions,
                    played=played,
                    reference_weekly=reference_weekly,
                    reference_decisions=reference_decisions,
                    reference_played=reference_played,
                    evaluation_weekly=evaluation_weekly,
                    evaluation_decisions=evaluation_decisions,
                    evaluation_played=evaluation_played,
                    context_idx=context_idx,
                    waiver_baseline=waiver_baseline,
                    sold_keys_after=buy_sold,
                    fixed_salary_map_after=buy_fixed,
                    post_market_budget=market_budget - paid_price,
                    post_market_slots=post_slots,
                    nominal_buffer=None,
                    objective_cache=objective_cache,
                    forecast_cache=forecast_cache,
                    enforce_top_n=False,
                )
            if pass_structural:
                pass_result = solve_branch(
                    sim=sim,
                    predictions=predictions,
                    player_keys=player_keys,
                    raw_salary_market=raw_salary_market,
                    raw_nominal_salary=raw_nominal_salary,
                    weekly=weekly,
                    decisions=decisions,
                    played=played,
                    reference_weekly=reference_weekly,
                    reference_decisions=reference_decisions,
                    reference_played=reference_played,
                    evaluation_weekly=evaluation_weekly,
                    evaluation_decisions=evaluation_decisions,
                    evaluation_played=evaluation_played,
                    context_idx=context_idx,
                    waiver_baseline=waiver_baseline,
                    sold_keys_after=pass_sold,
                    fixed_salary_map_after=fixed_salary_map,
                    post_market_budget=market_budget - clearing_price,
                    post_market_slots=post_slots,
                    nominal_buffer=None,
                    objective_cache=objective_cache,
                    forecast_cache=forecast_cache,
                    enforce_top_n=False,
                )
            if buy_result is not None or pass_result is not None:
                chosen_stage = None
                top_n_relaxed = True

        if buy_result is None and pass_result is None:
            failure_reason = "both_buy_and_pass_continuations_infeasible"
            append_failure_event(
                event,
                failure_reason,
                buy_structural=buy_structural,
                pass_structural=pass_structural,
            )
            break
        nominal_relaxed = chosen_stage != requested_buffer
        if nominal_relaxed:
            relaxation_events += 1
        if top_n_relaxed:
            top_n_relaxation_events += 1

        if buy_result is not None and pass_result is not None:
            choose_buy = buy_result["forecast_ev"] > pass_result["forecast_ev"] + 1e-9
        else:
            choose_buy = buy_result is not None
        forced_buy = bool(choose_buy and pass_result is None)

        if choose_buy:
            fixed_salary_map = buy_fixed
            fixed_clearing_map[nominee] = clearing_price
            market_budget -= paid_price
            action = "buy"
            if forced_buy:
                forced_buys += 1
            if bool(event.get("is_synthetic_contract_fill", False)):
                synthetic_contract_buys += 1
        else:
            market_budget -= clearing_price
            action = "pass"
        market_slots = post_slots
        sold_keys.add(nominee_key)

        buy_ev = np.nan if buy_result is None else buy_result["forecast_ev"]
        pass_ev = np.nan if pass_result is None else pass_result["forecast_ev"]
        event_rows.append(
            {
                "path_id": path_id,
                "year": year,
                "trial": trial,
                "order_regime": order_regime,
                "price_rule": price_rule,
                "nominal_buffer": requested_buffer,
                "policy_mode": policy_mode,
                "nomination_number": int(event["nomination_number"]),
                "nominee": nominee,
                "nominee_key": nominee_key,
                "nominee_pos": nominee_pos,
                "event_source": event["event_source"],
                "clearing_price": clearing_price,
                "observed_clearing_price": observed_clearing_price,
                "decision_price": paid_price,
                "action": action,
                "forced_buy": forced_buy,
                "generic_repair": False,
                "nominal_stage": (
                    "none" if chosen_stage is None else f"{chosen_stage:g}"
                ),
                "nominal_relaxed": nominal_relaxed,
                "top_n_relaxed": top_n_relaxed,
                "buy_feasible": buy_result is not None,
                "pass_feasible": pass_result is not None,
                "strict_buy_feasible": strict_buy_feasible,
                "strict_pass_feasible": strict_pass_feasible,
                "buy_forecast_ev": buy_ev,
                "pass_forecast_ev": pass_ev,
                "buy_sampled_spend": (
                    np.nan if buy_result is None else buy_result["sampled_spend"]
                ),
                "pass_sampled_spend": (
                    np.nan if pass_result is None else pass_result["sampled_spend"]
                ),
                "buy_nominal_spend": (
                    np.nan if buy_result is None else buy_result["nominal_spend"]
                ),
                "pass_nominal_spend": (
                    np.nan if pass_result is None else pass_result["nominal_spend"]
                ),
                "forecast_ev_delta": buy_ev - pass_ev,
                "roster_size_before": base.ROSTER_SIZE - open_before,
                "roster_size_after": len(fixed_salary_map),
                "paid_spend_after": float(sum(fixed_salary_map.values())),
                "market_budget_before": budget_before,
                "market_budget_after": market_budget,
                "market_slots_before": slots_before,
                "market_slots_after": market_slots,
            }
        )

    complete = len(fixed_salary_map) == base.ROSTER_SIZE
    status = "complete" if complete else ("prefix" if max_events is not None else "failed")
    if not complete and status == "failed" and failure_reason == "":
        failure_reason = "tape_exhausted_before_roster_completion"
    roster = tuple(sorted(fixed_salary_map))
    counts = position_counts(list(roster), predictions)
    actual_score: dict[str, Any] = {}
    final_forecast_ev = np.nan
    if complete:
        if sum(fixed_salary_map.values()) > base.SALARY_CAP + 1e-8:
            raise AssertionError("Completed path exceeds the real $298 cap.")
        for pos in base.POSITIONS:
            if not base.POS_MIN[pos] <= counts[pos] <= base.POS_MAX[pos]:
                raise AssertionError(f"Completed path violates the {pos} roster bounds.")
        final_forecast_ev = base.forecast_roster_ev(
            roster,
            CURRENT_WAIVER,
            waiver_baseline,
            predictions,
            evaluation_weekly,
            evaluation_decisions,
            evaluation_played,
            forecast_cache,
        )
        if score_outcomes:
            # This is the sole target-season outcome access in the policy path.
            actual_score = base.score_actual_roster(environment, roster)

    path_row = {
        "path_id": path_id,
        "year": year,
        "trial": trial,
        "order_regime": order_regime,
        "price_rule": price_rule,
        "nominal_buffer": requested_buffer,
        "policy_mode": policy_mode,
        "status": status,
        "failure_reason": failure_reason,
        "complete": complete,
        "clean_policy_path": bool(
            complete
            and relaxation_events == 0
            and top_n_relaxation_events == 0
            and generic_repairs == 0
            and synthetic_contract_buys == 0
            and synthetic_contract_events_seen == 0
        ),
        "roster": "|".join(roster),
        "roster_size": len(roster),
        "paid_spend": float(sum(fixed_salary_map.values())),
        "historical_clearing_spend": float(sum(fixed_clearing_map.values())),
        "unused_personal_budget": float(base.SALARY_CAP - sum(fixed_salary_map.values())),
        "events_seen": events_seen,
        "completion_nomination": (
            int(event_rows[-1]["nomination_number"]) if complete and event_rows else np.nan
        ),
        "forced_buys": forced_buys,
        "nominal_relaxation_events": relaxation_events,
        "top_n_relaxation_events": top_n_relaxation_events,
        "generic_repair_buys": generic_repairs,
        "synthetic_contract_fill_buys": synthetic_contract_buys,
        "synthetic_contract_fill_events_seen": synthetic_contract_events_seen,
        "final_forecast_ev": final_forecast_ev,
        "actual_points": actual_score.get("actual_points", np.nan),
        "failure_penalized_points": actual_score.get("actual_points", 0.0),
        "actual_ppg": actual_score.get("actual_points", np.nan) / 16.0,
        "drafted_only_points": actual_score.get("drafted_only_points", np.nan),
        "actual_waiver_starts": actual_score.get("actual_waiver_starts", np.nan),
        "outcome_salary_missing_players": actual_score.get(
            "actual_salary_missing_players", np.nan
        ),
        "raw_outcome_missing_players": actual_score.get(
            "raw_outcome_missing_players", np.nan
        ),
        **{f"{pos.lower()}_count": counts[pos] for pos in base.POSITIONS},
    }
    return path_row, event_rows


def paired_results(paths: pd.DataFrame) -> pd.DataFrame:
    key = ["year", "trial", "order_regime", "price_rule", "policy_mode"]
    metrics = [
        "complete",
        "clean_policy_path",
        "actual_points",
        "failure_penalized_points",
        "actual_ppg",
        "final_forecast_ev",
        "paid_spend",
        "unused_personal_budget",
        "events_seen",
        "forced_buys",
        "nominal_relaxation_events",
        "top_n_relaxation_events",
        "generic_repair_buys",
        "synthetic_contract_fill_buys",
        "synthetic_contract_fill_events_seen",
    ]
    pieces = []
    for value in DEFAULT_BUFFERS:
        piece = paths[paths.nominal_buffer.eq(value)][key + metrics].copy()
        piece = piece.rename(columns={metric: f"{metric}_{value:g}" for metric in metrics})
        pieces.append(piece)
    if len(pieces) != 2:
        raise AssertionError("Paired summary currently requires the $5 and $10 buffers.")
    paired = pieces[0].merge(pieces[1], on=key, how="outer", validate="one_to_one")
    paired["paired_complete"] = paired["complete_5"] & paired["complete_10"]
    paired["paired_clean"] = (
        paired["clean_policy_path_5"] & paired["clean_policy_path_10"]
    )
    for metric in [
        value for value in metrics if value not in {"complete", "clean_policy_path"}
    ]:
        paired[f"{metric}_diff_5_minus_10"] = (
            paired[f"{metric}_5"] - paired[f"{metric}_10"]
        )
    paired["points_win_5"] = paired.actual_points_diff_5_minus_10.gt(0)
    return paired


def summarize_pairs(paired: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in paired.groupby(groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        complete = group[group.paired_complete]
        clean = group[group.paired_clean]
        diff = complete.actual_points_diff_5_minus_10.dropna()
        clean_diff = clean.actual_points_diff_5_minus_10.dropna()
        penalty_diff = group.failure_penalized_points_diff_5_minus_10
        clean_origin_means = (
            clean.groupby("year").actual_points_diff_5_minus_10.mean()
        )
        required_years = sorted(group.year.unique())
        clean_origin_precision = clean.groupby("year").agg(
            clean_n=("actual_points_diff_5_minus_10", "count"),
            clean_sd=("actual_points_diff_5_minus_10", "std"),
        )
        precision_ready = (
            len(clean_origin_precision) == len(required_years)
            and clean_origin_precision.clean_n.gt(1).all()
        )
        equal_origin_randomization_se = (
            math.sqrt(
                float(
                    (
                        clean_origin_precision.clean_sd.pow(2)
                        / clean_origin_precision.clean_n
                    ).sum()
                )
            )
            / len(required_years)
            if precision_ready
            else np.nan
        )
        row = dict(zip(groups, keys))
        row.update(
            {
                "paired_paths": len(group),
                "completion_rate_5": float(group.complete_5.mean()),
                "completion_rate_10": float(group.complete_10.mean()),
                "completion_rate_diff_5_minus_10": float(
                    group.complete_5.mean() - group.complete_10.mean()
                ),
                "complete_5_only_rate": float(
                    (group.complete_5 & ~group.complete_10).mean()
                ),
                "complete_10_only_rate": float(
                    (~group.complete_5 & group.complete_10).mean()
                ),
                "paired_completion_rate": float(group.paired_complete.mean()),
                "paired_clean_rate": float(group.paired_clean.mean()),
                "mean_points_diff_completed_5_minus_10": (
                    float(diff.mean()) if len(diff) else np.nan
                ),
                "mean_points_diff_clean_5_minus_10": (
                    float(clean_diff.mean()) if len(clean_diff) else np.nan
                ),
                "mean_points_diff_clean_equal_origin_5_minus_10": (
                    float(clean_origin_means.mean())
                    if len(clean_origin_means) == len(required_years)
                    else np.nan
                ),
                "mean_failure_penalized_diff_5_minus_10": float(
                    penalty_diff.mean()
                ),
                "median_points_diff_completed_5_minus_10": (
                    float(diff.median()) if len(diff) else np.nan
                ),
                "pooled_path_se_completed_points_diff": (
                    float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else np.nan
                ),
                "pooled_path_se_clean_points_diff": (
                    float(clean_diff.std(ddof=1) / math.sqrt(len(clean_diff)))
                    if len(clean_diff) > 1
                    else np.nan
                ),
                "randomization_se_clean_equal_origin": (
                    equal_origin_randomization_se
                ),
                "minimum_clean_paths_per_origin": (
                    int(clean_origin_precision.clean_n.min())
                    if len(clean_origin_precision) == len(required_years)
                    else 0
                ),
                "win_rate_5_completed": (
                    float((diff > 0).mean()) if len(diff) else np.nan
                ),
                "mean_ppg_diff_5_minus_10": (
                    float(complete.actual_ppg_diff_5_minus_10.mean())
                    if len(complete)
                    else np.nan
                ),
                "mean_forecast_ev_diff_5_minus_10": (
                    float(complete.final_forecast_ev_diff_5_minus_10.mean())
                    if len(complete)
                    else np.nan
                ),
                "mean_relax_events_5": float(group.nominal_relaxation_events_5.mean()),
                "mean_relax_events_10": float(group.nominal_relaxation_events_10.mean()),
                "mean_top_n_relax_events_5": float(
                    group.top_n_relaxation_events_5.mean()
                ),
                "mean_top_n_relax_events_10": float(
                    group.top_n_relaxation_events_10.mean()
                ),
                "generic_repair_path_rate_5": float(group.generic_repair_buys_5.gt(0).mean()),
                "generic_repair_path_rate_10": float(group.generic_repair_buys_10.gt(0).mean()),
                "mean_paid_spend_diff_5_minus_10": float(
                    complete.paid_spend_diff_5_minus_10.mean()
                ) if len(complete) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_readout(
    output_dir: Path,
    paths: pd.DataFrame,
    paired: pd.DataFrame,
    development: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        def clean(value: Any) -> str:
            if pd.isna(value):
                return ""
            return str(value).replace("|", "\\|")

        headers = [clean(column) for column in frame.columns]
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        rows.extend(
            "| " + " | ".join(clean(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        return "\n".join(rows)

    lines = [
        "# Sequential recourse readout",
        "",
        "This is a non-anticipating fixed-price stress test, not a reconstruction of "
        "the historical nomination room. Future prices and target outcomes were hidden "
        "from every Buy/Pass decision.",
        "",
        f"- Paths: {len(paths):,}",
        f"- Completed legal rosters: {int(paths.complete.sum()):,}/{len(paths):,}",
        "- Runtime generic-repair paths: 0 (the replay never invents a player)",
        f"- Prefix-invariance check: {validation['prefix_invariance']}",
        "",
    ]
    primary_pairs = paired[
        paired.year.le(2024)
        & paired.order_regime.isin(("tier_early", "uniform", "position_run"))
        & paired.price_rule.eq("plus_one")
        & paired.policy_mode.eq("strict")
    ]
    if len(primary_pairs):
        complete_5 = int(primary_pairs.complete_5.sum())
        complete_10 = int(primary_pairs.complete_10.sum())
        paired_complete = int(primary_pairs.paired_complete.sum())
        paired_clean = int(primary_pairs.paired_clean.sum())
        discordant = int(
            primary_pairs.complete_5.ne(primary_pairs.complete_10).sum()
        )
        lines.extend(
            [
                "## Decision",
                "",
                "**No buffer is selected by this replay.** The primary comparison "
                "fails the predeclared completion, discordance, sign-stability, and "
                "randomization-precision requirements.",
                "",
                f"- `$5` completed {complete_5}/{len(primary_pairs)} primary paths; "
                f"`$10` completed {complete_10}/{len(primary_pairs)}.",
                f"- Both completed in {paired_complete}/{len(primary_pairs)} pairs; "
                f"{discordant}/{len(primary_pairs)} pairs had discordant completion.",
                f"- Only {paired_clean}/{len(primary_pairs)} pairs were clean enough "
                "for the prespecified point comparison.",
                "- No primary order family had clean observations in every development "
                "origin, so the equal-origin effect and its randomization error are "
                "undefined.",
                "",
            ]
        )
    lines.extend(
        [
            "## Primary: strict p+1, 2022-2024, by order family",
            "",
            "The table is intentionally separated by order family; no probabilities "
            "are assigned to the synthetic regimes. Tier-early, uniform, and "
            "position-run are the primary families; star-late is adversarial "
            "sensitivity.",
            "",
        ]
    )
    primary = (
        development[
            development.price_rule.eq("plus_one")
            & development.policy_mode.eq("strict")
        ].copy()
        if "price_rule" in development
        else pd.DataFrame()
    )
    if len(primary):
        columns = [
            "order_regime",
            "paired_paths",
            "completion_rate_5",
            "completion_rate_10",
            "completion_rate_diff_5_minus_10",
            "paired_completion_rate",
            "paired_clean_rate",
            "mean_points_diff_clean_equal_origin_5_minus_10",
            "randomization_se_clean_equal_origin",
            "minimum_clean_paths_per_origin",
            "mean_points_diff_completed_5_minus_10",
            "mean_failure_penalized_diff_5_minus_10",
            "mean_relax_events_5",
            "mean_relax_events_10",
            "mean_top_n_relax_events_5",
            "mean_top_n_relax_events_10",
            "generic_repair_path_rate_5",
            "generic_repair_path_rate_10",
        ]
        lines.append(markdown_table(primary[columns].round(3)))
    lines.extend(
        [
            "",
            "Positive point differences favor `$5`; negative differences favor `$10`. "
            "Points are decision-worthy only when paired-clean completion is effectively "
            "complete. The failure-penalized difference assigns an incomplete draft zero "
            "points; it is a deliberately harsh policy-invalid sensitivity, not an "
            "observed season score.",
            "",
            "Operational mode (`+$5 -> +$10 -> no nominal row`, or `+$10 -> no nominal "
            "row`, followed by Top-N relaxation) and recorded-price `p` are sensitivities "
            "in `summary_development.csv`. No runtime fallback relaxes the real `$298` "
            "cap, roster size, or position limits.",
            "",
            "The replay begins with an empty personal roster after all league keepers are "
            "removed. It does not validate a universal buffer for a fixed personal keeper "
            "state.",
            "",
            "If completion, order family, or price convention changes the conclusion, "
            "this study does not identify a buffer choice.",
            "",
            "2025 is a temporal sensitivity rather than a fresh holdout because its "
            "results were inspected during earlier tuning.",
        ]
    )
    (output_dir / "decision_readout.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument(
        "--trial-indices",
        nargs="+",
        type=int,
        help=(
            "Optional zero-based trial subset. The full --trials plan is still "
            "generated, so selected trials preserve their original draws and seeds."
        ),
    )
    parser.add_argument(
        "--order-regimes",
        nargs="+",
        choices=ORDER_REGIMES,
        default=list(ORDER_REGIMES),
    )
    parser.add_argument(
        "--price-rules", nargs="+", choices=PRICE_RULES, default=list(PRICE_RULES)
    )
    parser.add_argument(
        "--policy-modes",
        nargs="+",
        choices=POLICY_MODES,
        default=list(POLICY_MODES),
    )
    parser.add_argument("--buffers", nargs="+", type=float, default=list(DEFAULT_BUFFERS))
    parser.add_argument("--contexts", type=int, default=40)
    parser.add_argument("--evaluation-contexts", type=int, default=80)
    parser.add_argument("--context-draws", type=int, default=5)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--skip-prefix-check", action="store_true")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(STUDY_DIR / "results"),
    )
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(base.FROZEN_SOURCES))
    if invalid:
        parser.error(f"Unsupported replay years: {invalid}")
    if tuple(sorted(args.buffers)) != DEFAULT_BUFFERS:
        parser.error("This paired runner currently requires exactly --buffers 5 10.")
    if min(
        args.trials,
        args.contexts,
        args.evaluation_contexts,
        args.context_draws,
        args.projection_draws,
        args.salary_draws,
    ) <= 0:
        parser.error("Trial and context counts must be positive.")
    if args.context_draws > args.contexts:
        parser.error("--context-draws cannot exceed --contexts.")
    if args.trial_indices:
        if len(set(args.trial_indices)) != len(args.trial_indices):
            parser.error("--trial-indices must be unique.")
        if min(args.trial_indices) < 0 or max(args.trial_indices) >= args.trials:
            parser.error("--trial-indices must be between zero and --trials minus one.")
        if not args.skip_prefix_check:
            parser.error("Trial shards require --skip-prefix-check.")
        if args.checkpoint_dir:
            parser.error("Trial shards write final outputs and cannot write year checkpoints.")
    if args.resume and not args.checkpoint_dir:
        parser.error("--resume requires --checkpoint-dir.")
    return args


CHECKPOINT_FRAME_NAMES = (
    "paths",
    "events",
    "raw_rosters",
    "owner_audit",
    "tape_audit",
    "tape_events",
    "template_audit",
)


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "trials",
        "order_regimes",
        "price_rules",
        "policy_modes",
        "buffers",
        "contexts",
        "evaluation_contexts",
        "context_draws",
        "projection_draws",
        "salary_draws",
        "seed",
        "skip_prefix_check",
    )
    return {key: getattr(args, key) for key in keys}


def checkpoint_files(checkpoint_dir: Path, year: int) -> dict[str, Path]:
    files = {
        name: checkpoint_dir / f"{year}_{name}.pkl"
        for name in CHECKPOINT_FRAME_NAMES
    }
    files["meta"] = checkpoint_dir / f"{year}_meta.json"
    return files


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = ROOT / checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selected_trials = (
        list(range(args.trials))
        if not args.trial_indices
        else list(args.trial_indices)
    )

    if not BASE_MANIFEST.exists():
        raise FileNotFoundError(BASE_MANIFEST)
    prior_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    current_outcome_hashes = {
        "simulation_db_sha256": base.sha256_file(base.SIM_DB),
        "raw_weekly_sha256": base.sha256_file(base.DAILY_DB),
    }
    for key, value in current_outcome_hashes.items():
        if value != prior_manifest["current_outcome_sources"][key]:
            raise AssertionError(f"Outcome source drifted since the base replay ({key}).")

    print("Loading raw outcomes and frozen replay inputs...", flush=True)
    raw_weekly = base.load_raw_weekly(max_year=max(args.years))
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    all_paths: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_rosters: list[pd.DataFrame] = []
    all_owner_audits: list[pd.DataFrame] = []
    all_tape_audits: list[pd.DataFrame] = []
    all_tapes: list[pd.DataFrame] = []
    all_template_audits: list[pd.DataFrame] = []
    prefix_invariance: bool | None = None if args.skip_prefix_check else True
    manifest: dict[str, Any] = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "estimand": (
            "paired $5-$10 receding-horizon policy result conditional on a fixed "
            "historical clearing-price tape, synthetic order regime, and price rule"
        ),
        "method_boundary": {
            "historical_nomination_order_available": False,
            "losing_bids_available": False,
            "opponent_response_modeled": False,
            "future_prices_visible_to_policy": False,
            "target_outcomes_visible_before_completion": False,
            "personal_starting_roster": "empty; all league keepers unavailable",
            "runtime_generic_player_repair": False,
            "strict_policy": "requested nominal guardrail and Top-N remain hard",
            "operational_policy": (
                "nominal hierarchy then Top-N relaxation; never real-cap/position relaxation"
            ),
            "solver_cap_audit_tolerance": SOLVER_CAP_TOLERANCE,
            "salary_law": "frozen replay law; mostly legacy for 2023-2025",
            "price_rules": {
                "clearing": "optimistic first refusal at recorded winning price p",
                "plus_one": "one-dollar outbid stress p+1; not an equilibrium price",
            },
        },
        "base_replay_manifest": {
            "path": str(BASE_MANIFEST),
            "sha256": base.sha256_file(BASE_MANIFEST),
        },
        "runners": {
            "buffer_runner": str(BUFFER_RUNNER),
            "buffer_runner_sha256": base.sha256_file(BUFFER_RUNNER),
            "base_runner": str(buffer_replay.BASE_RUNNER),
            "base_runner_sha256": base.sha256_file(buffer_replay.BASE_RUNNER),
        },
        "simulation_helper": {
            "path": str(base.APP_HELPER),
            "sha256": base.sha256_file(base.APP_HELPER),
        },
        "current_outcome_sources": current_outcome_hashes,
        "origins": {},
    }
    checkpoint_contract = {
        "config": checkpoint_config(args),
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "simulation_helper_sha256": base.sha256_file(base.APP_HELPER),
        "current_outcome_sources": current_outcome_hashes,
    }

    for year in args.years:
        year_started = time.perf_counter()
        print(f"\n=== Origin {year} ===", flush=True)
        files = None if checkpoint_dir is None else checkpoint_files(checkpoint_dir, year)
        if args.resume and files is not None and files["meta"].exists():
            checkpoint_meta = json.loads(files["meta"].read_text(encoding="utf-8"))
            for key, expected in checkpoint_contract.items():
                actual = checkpoint_meta.get(key)
                compatible_runner = (
                    key == "runner_sha256"
                    and actual in COMPATIBLE_CHECKPOINT_RUNNER_SHA256
                )
                if actual != expected and not compatible_runner:
                    raise AssertionError(
                        f"{year}: checkpoint contract mismatch for {key}."
                    )
            missing_files = [
                str(files[name])
                for name in CHECKPOINT_FRAME_NAMES
                if not files[name].exists()
            ]
            if missing_files:
                raise FileNotFoundError(
                    f"{year}: checkpoint commit exists but frames are missing: {missing_files}"
                )
            for name in CHECKPOINT_FRAME_NAMES:
                if (
                    base.sha256_file(files[name])
                    != checkpoint_meta["frame_sha256"][name]
                ):
                    raise AssertionError(
                        f"{year}: checkpoint frame hash mismatch for {name}."
                    )
            all_paths.append(pd.read_pickle(files["paths"]))
            all_events.append(pd.read_pickle(files["events"]))
            all_rosters.append(pd.read_pickle(files["raw_rosters"]))
            all_owner_audits.append(pd.read_pickle(files["owner_audit"]))
            all_tape_audits.append(pd.read_pickle(files["tape_audit"]))
            all_tapes.append(pd.read_pickle(files["tape_events"]))
            all_template_audits.append(pd.read_pickle(files["template_audit"]))
            manifest["origins"][str(year)] = checkpoint_meta["origin_manifest"]
            manifest.setdefault("resumed_checkpoints", {})[str(year)] = {
                "runner_sha256": checkpoint_meta["runner_sha256"],
                "current_runner_sha256": checkpoint_contract["runner_sha256"],
                "execution_control_compatible": bool(
                    checkpoint_meta["runner_sha256"]
                    in COMPATIBLE_CHECKPOINT_RUNNER_SHA256
                ),
            }
            if not args.skip_prefix_check:
                prefix_invariance = bool(
                    prefix_invariance and checkpoint_meta["prefix_invariance"]
                )
            print(f"{year}: resumed from validated checkpoint.", flush=True)
            continue
        target_features = features[features.season.eq(year)].copy()
        conn, source_manifest = base.open_frozen_source(base.FROZEN_SOURCES[year])
        try:
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

        raw_roster, owner_audit = parse_raw_rosters(year, forecast, target_features)
        raw_keeper_keys = set(
            raw_roster.loc[
                raw_roster.pos.isin(base.POSITIONS) & raw_roster.is_keeper,
                "player_key",
            ]
        )
        candidate_mask = ~forecast.player_key.isin(raw_keeper_keys).to_numpy()
        candidate_forecast = forecast.loc[candidate_mask].reset_index(drop=True)
        candidate_ppg = ppg_draws[candidate_mask]
        candidate_salary_draws = salary_draws[candidate_mask]
        predictions = base.build_predictions(candidate_forecast, candidate_ppg)
        player_keys = candidate_forecast.player_key.to_numpy(dtype=object)
        raw_nominal_salary = candidate_forecast.salary.to_numpy(dtype=float)
        if predictions.player.duplicated().any() or len(set(player_keys)) != len(player_keys):
            raise AssertionError(f"{year}: candidate forecast identities are not unique.")

        tape, tape_audit = build_price_tape(
            year,
            raw_roster,
            candidate_forecast,
            target_features,
        )
        environment, _ = base.build_actual_environment(
            year,
            forecast,
            raw_weekly,
            features,
            actual,
        )
        raw_skill_keepers = raw_roster[
            raw_roster.pos.isin(base.POSITIONS) & raw_roster.is_keeper
        ]
        if raw_keeper_keys != set(environment["keeper_keys"]):
            raise AssertionError(
                f"{year}: raw ESPN and flattened keeper identities disagree."
            )
        if environment["keeper_count"] != len(raw_skill_keepers):
            raise AssertionError(f"{year}: raw and flattened keeper counts disagree.")
        if not math.isclose(
            environment["keeper_spend"],
            float(raw_skill_keepers.replay_price.sum()),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise AssertionError(f"{year}: raw and flattened keeper spend disagrees.")
        cache, template_audit = base.build_template_cache(
            year,
            forecast,
            features,
            raw_weekly,
        )
        template_audit["max_donor_is_causal"] = template_audit.max_donor_season.lt(year)
        if not template_audit.max_donor_is_causal.all():
            raise AssertionError("Construction template pool crossed the replay origin.")

        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game", "salary"]
        ].copy()
        sim = base.make_simulation(year, player_data, cache)
        waiver_baseline = sim.estimate_waiver_baselines(
            num_teams=base.NUM_TEAMS,
            roster_size=base.ROSTER_SIZE,
        )
        print(
            f"{year}: {len(tape)} tape events, {int(tape.actionable.sum())} actionable; "
            f"building contexts...",
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
                args.evaluation_contexts,
                args.seed + 100_000 + year,
            )
        )

        skill_keepers = raw_roster[
            raw_roster.pos.isin(base.POSITIONS) & raw_roster.is_keeper
        ]
        initial_market_budget = float(
            base.TOTAL_MARKET_BUDGET - skill_keepers.replay_price.sum()
        )
        initial_market_slots = int(base.TOTAL_MARKET_SLOTS - len(skill_keepers))
        if len(tape) != initial_market_slots:
            raise AssertionError(f"{year}: tape length and modeled open slots disagree.")

        rng = np.random.default_rng(args.seed + year * 101)
        salary_plan = rng.integers(
            0,
            candidate_salary_draws.shape[1],
            size=(args.trials, 5),
        )
        context_plan = rng.integers(
            0,
            args.contexts,
            size=(args.trials, args.context_draws),
        )
        objective_cache: dict[
            tuple[tuple[int, ...], tuple[str, ...]], np.ndarray
        ] = {}
        forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}
        year_paths: list[dict[str, Any]] = []
        year_events: list[dict[str, Any]] = []

        for regime_idx, regime in enumerate(args.order_regimes):
            print(f"{year}: order regime {regime}...", flush=True)
            for trial in selected_trials:
                order_seed = (
                    args.seed + year * 10_000 + ORDER_SEED_OFFSETS[regime] + trial
                )
                order = make_order(tape, regime, order_seed)
                raw_salary_market = candidate_salary_draws[
                    :, salary_plan[trial]
                ].mean(axis=1)
                context_idx = context_plan[trial]

                for price_rule in args.price_rules:
                    for policy_mode in args.policy_modes:
                        for requested_buffer in args.buffers:
                            path_row, event_rows = run_policy_path(
                                year=year,
                                trial=trial,
                                order_regime=regime,
                                price_rule=price_rule,
                                policy_mode=policy_mode,
                                requested_buffer=requested_buffer,
                                order=order,
                                initial_market_budget=initial_market_budget,
                                initial_market_slots=initial_market_slots,
                                sim=sim,
                                predictions=predictions,
                                player_keys=player_keys,
                                raw_salary_market=raw_salary_market,
                                raw_nominal_salary=raw_nominal_salary,
                                weekly=weekly,
                                decisions=decisions,
                                played=played,
                                evaluation_weekly=evaluation_weekly,
                                evaluation_decisions=evaluation_decisions,
                                evaluation_played=evaluation_played,
                                context_idx=context_idx,
                                waiver_baseline=waiver_baseline,
                                environment=environment,
                                objective_cache=objective_cache,
                                forecast_cache=forecast_cache,
                            )
                            year_paths.append(path_row)
                            year_events.extend(event_rows)

                if not args.skip_prefix_check and regime_idx == 0 and trial == 0:
                    prefix_len = min(3, len(order))
                    altered = pd.concat(
                        [
                            order.iloc[:prefix_len],
                            order.iloc[prefix_len:].iloc[::-1],
                        ],
                        ignore_index=True,
                    )
                    altered["nomination_number"] = np.arange(1, len(altered) + 1)
                    first_path, first_events = run_policy_path(
                        year=year,
                        trial=trial,
                        order_regime=regime,
                        price_rule="plus_one",
                        policy_mode="strict",
                        requested_buffer=5.0,
                        order=order,
                        initial_market_budget=initial_market_budget,
                        initial_market_slots=initial_market_slots,
                        sim=sim,
                        predictions=predictions,
                        player_keys=player_keys,
                        raw_salary_market=raw_salary_market,
                        raw_nominal_salary=raw_nominal_salary,
                        weekly=weekly,
                        decisions=decisions,
                        played=played,
                        evaluation_weekly=evaluation_weekly,
                        evaluation_decisions=evaluation_decisions,
                        evaluation_played=evaluation_played,
                        context_idx=context_idx,
                        waiver_baseline=waiver_baseline,
                        environment=environment,
                        objective_cache=objective_cache,
                        forecast_cache=forecast_cache,
                        max_events=prefix_len,
                        score_outcomes=False,
                    )
                    second_path, second_events = run_policy_path(
                        year=year,
                        trial=trial,
                        order_regime=regime,
                        price_rule="plus_one",
                        policy_mode="strict",
                        requested_buffer=5.0,
                        order=altered,
                        initial_market_budget=initial_market_budget,
                        initial_market_slots=initial_market_slots,
                        sim=sim,
                        predictions=predictions,
                        player_keys=player_keys,
                        raw_salary_market=raw_salary_market,
                        raw_nominal_salary=raw_nominal_salary,
                        weekly=weekly,
                        decisions=decisions,
                        played=played,
                        evaluation_weekly=evaluation_weekly,
                        evaluation_decisions=evaluation_decisions,
                        evaluation_played=evaluation_played,
                        context_idx=context_idx,
                        waiver_baseline=waiver_baseline,
                        environment=environment,
                        objective_cache=objective_cache,
                        forecast_cache=forecast_cache,
                        max_events=prefix_len,
                        score_outcomes=False,
                    )
                    del first_path, second_path
                    def prefix_signature(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
                        return [
                            (
                                row["nominee_key"],
                                row["action"],
                                row["roster_size_after"],
                                round(float(row["paid_spend_after"]), 8),
                                round(float(row["market_budget_after"]), 8),
                                (
                                    None
                                    if pd.isna(row["buy_forecast_ev"])
                                    else round(float(row["buy_forecast_ev"]), 8)
                                ),
                                (
                                    None
                                    if pd.isna(row["pass_forecast_ev"])
                                    else round(float(row["pass_forecast_ev"]), 8)
                                ),
                            )
                            for row in rows
                        ]
                    first_signature = prefix_signature(first_events)
                    second_signature = prefix_signature(second_events)
                    prefix_invariance = bool(
                        prefix_invariance and first_signature == second_signature
                    )
                    if not prefix_invariance:
                        raise AssertionError("Policy decisions depend on a hidden tape suffix.")

        year_paths_frame = pd.DataFrame(year_paths)
        year_events_frame = pd.DataFrame(year_events)
        all_paths.append(year_paths_frame)
        all_events.append(year_events_frame)
        roster_export = raw_roster.copy()
        roster_export["raw_source_sha256"] = base.sha256_file(
            RAW_SALARY_DIR / f"beta_{year}_results.csv"
        )
        all_rosters.append(roster_export)
        all_owner_audits.append(owner_audit)
        all_tape_audits.append(tape_audit)
        all_tapes.append(tape)
        all_template_audits.append(template_audit)

        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(
            {
                "raw_roster_source": str(
                    RAW_SALARY_DIR / f"beta_{year}_results.csv"
                ),
                "raw_roster_sha256": base.sha256_file(
                    RAW_SALARY_DIR / f"beta_{year}_results.csv"
                ),
                "raw_skill_keeper_count": int(len(skill_keepers)),
                "raw_skill_keeper_spend": float(skill_keepers.replay_price.sum()),
                "tape_events": int(len(tape)),
                "tape_actionable_events": int(tape.actionable.sum()),
                "projected_waiver_baseline": waiver_baseline,
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        manifest["origins"][str(year)] = source_manifest
        if files is not None:
            checkpoint_frames = {
                "paths": year_paths_frame,
                "events": year_events_frame,
                "raw_rosters": roster_export,
                "owner_audit": owner_audit,
                "tape_audit": tape_audit,
                "tape_events": tape,
                "template_audit": template_audit,
            }
            for name, frame in checkpoint_frames.items():
                frame.to_pickle(files[name])
            checkpoint_meta = {
                **checkpoint_contract,
                "year": year,
                "prefix_invariance": (
                    None if args.skip_prefix_check else bool(prefix_invariance)
                ),
                "origin_manifest": source_manifest,
                "rows": {
                    name: int(len(frame))
                    for name, frame in checkpoint_frames.items()
                },
                "frame_sha256": {
                    name: base.sha256_file(files[name])
                    for name in CHECKPOINT_FRAME_NAMES
                },
            }
            files["meta"].write_text(
                json.dumps(checkpoint_meta, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        print(
            f"{year}: {len(year_paths)} paths complete in "
            f"{time.perf_counter() - year_started:.1f}s.",
            flush=True,
        )
        del objective_cache, forecast_cache, weekly, decisions, played
        del evaluation_weekly, evaluation_decisions, evaluation_played
        gc.collect()

    paths = pd.concat(all_paths, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    raw_rosters = pd.concat(all_rosters, ignore_index=True)
    owner_audit = pd.concat(all_owner_audits, ignore_index=True)
    tape_audit = pd.concat(all_tape_audits, ignore_index=True)
    price_tape_events = pd.concat(all_tapes, ignore_index=True)
    template_audit = pd.concat(all_template_audits, ignore_index=True)
    expected_paths = (
        len(args.years)
        * len(selected_trials)
        * len(args.order_regimes)
        * len(args.price_rules)
        * len(args.policy_modes)
        * len(args.buffers)
    )
    key = [
        "year",
        "trial",
        "order_regime",
        "price_rule",
        "policy_mode",
        "nominal_buffer",
    ]
    if len(paths) != expected_paths or paths.duplicated(key).any():
        raise AssertionError("Sequential replay path grid is incomplete or duplicated.")
    if paths.loc[paths.complete, "paid_spend"].gt(base.SALARY_CAP + 1e-8).any():
        raise AssertionError("A completed path exceeds the real salary cap.")
    if not template_audit.max_donor_is_causal.all():
        raise AssertionError("A template donor crossed its replay origin.")
    ledger_counts = events.groupby("path_id").size()
    expected_ledger_counts = paths.set_index("path_id").events_seen.astype(int)
    if not ledger_counts.reindex(expected_ledger_counts.index).eq(
        expected_ledger_counts
    ).all():
        raise AssertionError("Event ledger does not contain every attempted nomination.")
    if events.market_slots_after.lt(0).any():
        raise AssertionError("An event produced negative remaining market slots.")
    for column in ("buy_sampled_spend", "pass_sampled_spend"):
        if pd.to_numeric(events[column], errors="coerce").gt(
            base.SALARY_CAP + SOLVER_CAP_TOLERANCE
        ).any():
            raise AssertionError(f"{column} exceeds the sampled cap.")
    nominal_stage_numeric = pd.to_numeric(events.nominal_stage, errors="coerce")
    for column in ("buy_nominal_spend", "pass_nominal_spend"):
        spend = pd.to_numeric(events[column], errors="coerce")
        constrained = spend.notna() & nominal_stage_numeric.notna()
        if (
            spend[constrained]
            > base.SALARY_CAP
            + nominal_stage_numeric[constrained]
            + SOLVER_CAP_TOLERANCE
        ).any():
            raise AssertionError(f"{column} exceeds the exported nominal stage.")
    strict_paths = paths.policy_mode.eq("strict")
    if (
        paths.loc[strict_paths, "nominal_relaxation_events"].ne(0).any()
        or paths.loc[strict_paths, "top_n_relaxation_events"].ne(0).any()
    ):
        raise AssertionError("A strict path used an operational fallback.")
    if paths.generic_repair_buys.ne(0).any():
        raise AssertionError("Runtime generic repair must remain disabled.")
    if paths.loc[~paths.complete, "actual_points"].notna().any():
        raise AssertionError("An incomplete path accessed target-season scoring.")

    paired = paired_results(paths)
    summary_by_year = summarize_pairs(
        paired,
        ["year", "order_regime", "price_rule", "policy_mode"],
    )
    development = summarize_pairs(
        paired[paired.year.le(2024)],
        ["order_regime", "price_rule", "policy_mode"],
    )
    temporal_2025 = summarize_pairs(
        paired[paired.year.eq(2025)],
        ["order_regime", "price_rule", "policy_mode"],
    )
    split = paired.assign(
        trial_half=np.where(paired.trial < args.trials / 2, "first", "second")
    )
    split_half = summarize_pairs(
        split,
        ["order_regime", "price_rule", "policy_mode", "trial_half"],
    )
    failure_reasons = (
        paths[~paths.complete]
        .groupby(
            [
                "year",
                "order_regime",
                "price_rule",
                "policy_mode",
                "nominal_buffer",
                "failure_reason",
            ],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "failed_paths"})
    )

    validation = {
        "expected_paths": expected_paths,
        "actual_paths": int(len(paths)),
        "unique_path_cells": True,
        "prefix_invariance": prefix_invariance,
        "all_template_donors_pre_origin": True,
        "k_dst_excluded_from_tape": True,
        "all_completed_paid_spend_within_298": True,
        "all_attempted_events_in_ledger": True,
        "all_branch_sampled_spend_within_298": True,
        "all_branch_nominal_spend_within_exported_stage": True,
        "strict_paths_used_no_fallback": True,
        "runtime_generic_repair_disabled": True,
        "incomplete_paths_not_target_scored": True,
        "completed_paths": int(paths.complete.sum()),
        "completion_rate": float(paths.complete.mean()),
        "generic_repair_paths": int(paths.generic_repair_buys.gt(0).sum()),
        "top_n_relaxation_paths": int(
            paths.top_n_relaxation_events.gt(0).sum()
        ),
        "synthetic_contract_fill_paths": int(
            paths.synthetic_contract_fill_buys.gt(0).sum()
        ),
    }
    outputs = {
        "policy_paths.csv": paths,
        "event_decisions.csv": events,
        "paired_buffer_paths.csv": paired,
        "summary_by_year.csv": summary_by_year,
        "summary_development.csv": development,
        "summary_2025_temporal_sensitivity.csv": temporal_2025,
        "summary_split_half.csv": split_half,
        "failure_reasons.csv": failure_reasons,
        "raw_roster_rows.csv": raw_rosters,
        "raw_owner_audit.csv": owner_audit,
        "price_tape_audit.csv": tape_audit,
        "price_tape_events.csv": price_tape_events,
        "template_pool_audit.csv": template_audit,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["validation"] = validation
    manifest["output_rows"] = {
        filename: int(len(frame)) for filename, frame in outputs.items()
    }
    manifest["runner_sha256"] = base.sha256_file(Path(__file__).resolve())
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_readout(output_dir, paths, paired, development, validation)
    print(
        f"\nSequential recourse replay complete in {time.perf_counter() - started:.1f}s. "
        f"Results: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
