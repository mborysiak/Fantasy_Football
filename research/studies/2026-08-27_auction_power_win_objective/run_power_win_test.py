"""Paired 2025 Auction test of win-frequency and winning-margin objectives."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
BASE_SCRIPT = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-27_auction_championship_waiver_objective"
    / "run_paired_test.py"
)
RESULTS_DIR = STUDY_DIR / "results"
SPEC = importlib.util.spec_from_file_location("auction_objective_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


PRIMARY_GUARDRAIL = 0.005
SENSITIVITY_GUARDRAIL = 0.010
POWER_ALPHA = 0.25
MARGIN_SCALE = 25.0
DOMINANT_MARGIN = 50.0
LCB80_Z = base.LCB80_Z
SUBSET_SIZES = (2, 4, 8, 16)
ARMS = (
    "waiver_control",
    "mean_frontier",
    "win_g005",
    "excess_g005",
    "power_g005",
    "win_g010",
    "excess_g010",
    "power_g010",
    "win_direct_g010",
    "excess_direct_g010",
    "power_direct_g010",
    "win_pure",
    "excess_pure",
    "power_pure",
    "win_half_mean",
    "excess_half_mean",
    "power_half_mean",
)


def json_value(value):
    return base.json_value(value)


def paired_lcb(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        return float("nan")
    return float(values.mean() - LCB80_Z * values.std(ddof=1) / np.sqrt(len(values)))


def field_utility_cells(
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    opponents: int = base.NUM_TEAMS - 1,
    power_alpha: float = POWER_ALPHA,
    margin_scale: float = MARGIN_SCALE,
    dominant_margin: float = DOMINANT_MARGIN,
) -> dict[str, np.ndarray]:
    """Return candidate-by-context field-relative utility cells.

    Opponent rosters are independent draws with replacement from the common
    same-context reference distribution. Expected positive margin and its log
    transform integrate exactly over the maximum order statistic.
    """

    candidates = np.asarray(candidate_scores, dtype=np.float64)
    references = np.asarray(reference_scores, dtype=np.float64)
    if candidates.ndim != 2 or references.ndim != 2:
        raise ValueError("Field utility inputs must be roster-by-context matrices.")
    if candidates.shape[1] != references.shape[1] or references.shape[0] < 2:
        raise ValueError("Field utility references must align and contain two rosters.")
    if opponents < 1 or margin_scale <= 0 or dominant_margin < 0:
        raise ValueError("Field utility parameters are invalid.")

    reference_count = references.shape[0]
    less = (references[None, :, :] < candidates[:, None, :]).sum(axis=1)
    equal = np.isclose(
        references[None, :, :], candidates[:, None, :], rtol=0.0, atol=1e-9
    ).sum(axis=1)
    percentile = (less + 0.5 * equal) / reference_count
    win = np.power(percentile, int(opponents))

    ordered = np.sort(references, axis=0)
    ranks = np.arange(reference_count + 1, dtype=np.float64) / reference_count
    maximum_weights = np.diff(np.power(ranks, int(opponents)))
    margins = np.maximum(candidates[:, None, :] - ordered[None, :, :], 0.0)
    expected_excess = np.sum(
        margins * maximum_weights[None, :, None], axis=1
    )
    log_excess = np.sum(
        np.log1p(margins / float(margin_scale))
        * maximum_weights[None, :, None],
        axis=1,
    )

    dominant_less = (
        references[None, :, :]
        < (candidates[:, None, :] - float(dominant_margin))
    ).sum(axis=1)
    dominant = np.power(dominant_less / reference_count, int(opponents))
    power = win + float(power_alpha) * log_excess
    return {
        "win_probability": win,
        "expected_excess": expected_excess,
        "log_excess": log_excess,
        "power_utility": power,
        "dominant_win_probability": dominant,
    }


def diverse_value_vectors(state, block_idx: int, starts: int, seed: int):
    """Generate mean, coherent scenario, subset, and marginal-tail candidates."""

    bank = state["construction_banks"][block_idx]
    predictions = state["predictions"]
    scores = bank["weekly_scores"]
    decisions = bank["decision_scores"]
    played = bank["played_mask"]
    waivers = state["churn_waivers"]
    context_count = scores.shape[0]
    output = [("production_mean", state["churn_values"][block_idx])]
    single_values = []
    for context_idx in range(context_count):
        values = state["sim"].managed_marginal_values_multi_context_batch(
            scores[context_idx : context_idx + 1],
            predictions.pos.to_numpy(),
            decisions[context_idx : context_idx + 1],
            predictions.player.to_numpy(),
            [[]],
            waiver_baselines=waivers,
            lineup_require=base.LINEUP_REQUIRE,
            played_mask=played[context_idx : context_idx + 1],
        )[0]
        single_values.append(values)
        output.append((f"single_{context_idx:03d}", values))
    single_matrix = np.stack(single_values)
    output.extend([
        ("player_marginal_p75", np.quantile(single_matrix, 0.75, axis=0)),
        ("player_marginal_p90", np.quantile(single_matrix, 0.90, axis=0)),
    ])

    rng = np.random.default_rng(seed)
    random_starts = max(0, int(starts) - context_count)
    for start_idx in range(random_starts):
        subset_size = min(SUBSET_SIZES[start_idx % len(SUBSET_SIZES)], context_count)
        subset = np.sort(rng.choice(context_count, size=subset_size, replace=False))
        values = state["sim"].managed_marginal_values_multi_context_batch(
            scores[subset],
            predictions.pos.to_numpy(),
            decisions[subset],
            predictions.player.to_numpy(),
            [[]],
            waiver_baselines=waivers,
            lineup_require=base.LINEUP_REQUIRE,
            played_mask=played[subset],
        )[0]
        output.append((f"subset_{subset_size:02d}_{start_idx:03d}", values))
    return output


def compile_candidates(state, block_idx: int, starts: int, seed: int, static_cache):
    plans = {}
    sources = {}
    for label, values in diverse_value_vectors(state, block_idx, starts, seed):
        plan = base.solve_plan(state, values, static_cache)
        if plan is None:
            continue
        roster = tuple(sorted(plan["selected"]))
        if roster not in plans:
            plans[roster] = plan
            sources[roster] = []
        sources[roster].append(label)
    if not plans:
        raise RuntimeError(f"No candidates compiled for block {block_idx}.")
    return plans, sources


def score_candidates(state, block_idx: int, plans: dict, sources: dict):
    bank = state["construction_banks"][block_idx]
    predictions = state["predictions"]
    rosters = sorted(plans)
    cache = {}
    score_matrix = np.stack([
        base.sequential._score_roster_bank(
            state["sim"],
            predictions,
            roster,
            bank["weekly_scores"],
            bank["decision_scores"],
            bank["played_mask"],
            base.LINEUP_REQUIRE,
            state["churn_waivers"],
            cache,
        )
        for roster in rosters
    ])
    utility = field_utility_cells(score_matrix, score_matrix)
    production_indices = [
        idx for idx, roster in enumerate(rosters)
        if "production_mean" in sources[roster]
    ]
    if len(production_indices) != 1:
        raise ValueError("Exactly one production-mean roster is required.")
    mean_idx = int(np.argmax(score_matrix.mean(axis=1)))
    events = base.difference_maker_events(
        bank["weekly_scores"],
        bank["played_mask"],
        predictions,
        state["tail_thresholds"],
    )
    players = predictions.player.to_numpy()
    rows = []
    for roster_idx, roster in enumerate(rosters):
        roster_mask = np.isin(players, roster)
        diff = base.roster_difference_metrics(events, roster_mask)
        row = {
            "block": block_idx,
            "roster_key": " | ".join(roster),
            "candidate_sources": " | ".join(sources[roster]),
            "production_candidate": roster_idx == production_indices[0],
            "mean_frontier_candidate": roster_idx == mean_idx,
            "construction_mean": float(score_matrix[roster_idx].mean()),
            "construction_p10": float(np.percentile(score_matrix[roster_idx], 10)),
            "construction_p90": float(np.percentile(score_matrix[roster_idx], 90)),
            "construction_expected_difference_makers": diff["expected_difference_makers"],
            "construction_prob_two_difference_makers": diff["prob_two_difference_makers"],
            "forecast_spend": float(sum(
                plans[roster]["forecast_cost"][player] for player in roster
            )),
        }
        for metric, cells in utility.items():
            candidate_cells = cells[roster_idx]
            delta = candidate_cells - cells[mean_idx]
            row[f"construction_{metric}"] = float(candidate_cells.mean())
            row[f"construction_{metric}_delta_vs_mean"] = float(delta.mean())
            row[f"construction_{metric}_delta_lcb80_vs_mean"] = paired_lcb(delta)
        rows.append(row)
    return pd.DataFrame(rows), rosters, score_matrix


def choose_tail_candidate(metrics: pd.DataFrame, objective: str, guardrail: float):
    best_mean = float(metrics.construction_mean.max())
    eligible = metrics.loc[
        metrics.construction_mean.ge(best_mean * (1.0 - float(guardrail)) - 1e-9)
    ].copy()
    lcb_column = f"construction_{objective}_delta_lcb80_vs_mean"
    value_column = f"construction_{objective}"
    return eligible.sort_values(
        [lcb_column, value_column, "construction_mean", "roster_key"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_direct_candidate(metrics: pd.DataFrame, objective: str, guardrail: float):
    """Choose the largest point-estimate utility inside the mean guardrail."""

    best_mean = float(metrics.construction_mean.max())
    eligible = metrics.loc[
        metrics.construction_mean.ge(best_mean * (1.0 - float(guardrail)) - 1e-9)
    ].copy()
    value_column = f"construction_{objective}"
    return eligible.sort_values(
        [value_column, "construction_mean", "roster_key"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_pure_candidate(metrics: pd.DataFrame, objective: str):
    """Choose the unconstrained maximum point estimate for one tail metric."""

    value_column = f"construction_{objective}"
    return metrics.sort_values(
        [value_column, "construction_mean", "roster_key"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_half_mean_candidate(metrics: pd.DataFrame, objective: str):
    """Blend equally weighted standardized mean and tail utility."""

    value_column = f"construction_{objective}"
    frame = metrics.copy()
    mean_scale = float(frame.construction_mean.std(ddof=0))
    objective_scale = float(frame[value_column].std(ddof=0))
    if not np.isfinite(mean_scale) or mean_scale <= 0:
        mean_z = np.zeros(len(frame), dtype=np.float64)
    else:
        mean_z = (
            frame.construction_mean - frame.construction_mean.mean()
        ) / mean_scale
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        objective_z = np.zeros(len(frame), dtype=np.float64)
    else:
        objective_z = (frame[value_column] - frame[value_column].mean()) / objective_scale
    frame["half_mean_score"] = 0.5 * mean_z + 0.5 * objective_z
    return frame.sort_values(
        ["half_mean_score", value_column, "construction_mean", "roster_key"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def select_arms(metrics: pd.DataFrame, plans: dict):
    production = metrics.loc[metrics.production_candidate]
    mean = metrics.loc[metrics.mean_frontier_candidate]
    if len(production) != 1 or len(mean) != 1:
        raise ValueError("Production and exact-mean candidates must be unique.")
    rows = {
        "waiver_control": production.iloc[0],
        "mean_frontier": mean.iloc[0],
    }
    for suffix, guardrail in (("g005", PRIMARY_GUARDRAIL), ("g010", SENSITIVITY_GUARDRAIL)):
        for label, objective in (
            ("win", "win_probability"),
            ("excess", "expected_excess"),
            ("power", "power_utility"),
        ):
            rows[f"{label}_{suffix}"] = choose_tail_candidate(
                metrics, objective, guardrail
            )
    for label, objective in (
        ("win", "win_probability"),
        ("excess", "expected_excess"),
        ("power", "power_utility"),
    ):
        rows[f"{label}_direct_g010"] = choose_direct_candidate(
            metrics, objective, SENSITIVITY_GUARDRAIL
        )
        rows[f"{label}_pure"] = choose_pure_candidate(metrics, objective)
        rows[f"{label}_half_mean"] = choose_half_mean_candidate(metrics, objective)
    selected = {}
    for arm, row in rows.items():
        roster = tuple(row.roster_key.split(" | "))
        selected[arm] = {"row": row, "roster": roster, "plan": plans[roster]}
    return selected


def score_validation_block(
    state,
    block_idx: int,
    selected: dict,
    rosters: list[tuple[str, ...]],
    validation_bank,
):
    predictions = state["predictions"]
    cache = {}
    score_matrix = np.stack([
        base.sequential._score_roster_bank(
            state["sim"],
            predictions,
            roster,
            *validation_bank,
            base.LINEUP_REQUIRE,
            state["churn_waivers"],
            cache,
        )
        for roster in rosters
    ])
    utility = field_utility_cells(score_matrix, score_matrix)
    roster_index = {roster: idx for idx, roster in enumerate(rosters)}
    events = base.difference_maker_events(
        validation_bank[0],
        validation_bank[2],
        predictions,
        state["tail_thresholds"],
    )
    players = predictions.player.to_numpy()
    rows = []
    cell_rows = []
    for arm, choice in selected.items():
        roster = choice["roster"]
        idx = roster_index[roster]
        scores = score_matrix[idx]
        roster_mask = np.isin(players, roster)
        diff = base.roster_difference_metrics(events, roster_mask)
        counts = Counter(predictions.loc[roster_mask, "pos"])
        rb_experience = predictions.loc[roster_mask & predictions.pos.eq("RB"), "year_exp"]
        position_spend = {
            pos: sum(
                choice["plan"]["forecast_cost"][player]
                for player in roster
                if predictions.set_index("player").at[player, "pos"] == pos
            )
            for pos in ("QB", "RB", "WR", "TE")
        }
        row = {
            "block": block_idx,
            "arm": arm,
            "roster": " | ".join(roster),
            "forecast_spend": float(choice["row"].forecast_spend),
            "holdout_mean": float(scores.mean()),
            "holdout_p10": float(np.percentile(scores, 10)),
            "holdout_p90": float(np.percentile(scores, 90)),
            "holdout_expected_difference_makers": diff["expected_difference_makers"],
            "holdout_prob_two_difference_makers": diff["prob_two_difference_makers"],
            "dead_zone_rb_count": int(len(set(roster) & base.DEAD_ZONE_RBS)),
            "rb_mean_year_exp": float(rb_experience.mean()),
            "rookie_rb_count": int((rb_experience == 0).sum()),
            **{f"count_{pos.lower()}": int(counts.get(pos, 0)) for pos in ("QB", "RB", "WR", "TE")},
            **{f"spend_{pos.lower()}": float(position_spend[pos]) for pos in ("QB", "RB", "WR", "TE")},
        }
        for metric, values in utility.items():
            row[f"holdout_{metric}"] = float(values[idx].mean())
        rows.append(row)
        for context_idx, score in enumerate(scores):
            cell_rows.append({
                "block": block_idx,
                "arm": arm,
                "context": context_idx,
                "managed_score": float(score),
                **{
                    metric: float(values[idx, context_idx])
                    for metric, values in utility.items()
                },
            })
    return pd.DataFrame(rows), pd.DataFrame(cell_rows)


def score_actual(state, plan_rows: pd.DataFrame, all_candidate_rosters: set):
    actual_bank, coverage = base.load_actual_2025_bank(state)
    predictions = state["predictions"]
    rosters = sorted(all_candidate_rosters)
    cache = {}
    score_matrix = np.stack([
        base.sequential._score_roster_bank(
            state["sim"],
            predictions,
            roster,
            *actual_bank,
            base.LINEUP_REQUIRE,
            state["churn_waivers"],
            cache,
        )
        for roster in rosters
    ])
    utility = field_utility_cells(score_matrix, score_matrix)
    roster_index = {roster: idx for idx, roster in enumerate(rosters)}
    events = base.difference_maker_events(
        actual_bank[0], actual_bank[2], predictions, state["tail_thresholds"]
    )
    players = predictions.player.to_numpy()
    rows = []
    for row in plan_rows[["block", "arm", "roster"]].itertuples(index=False):
        roster = tuple(row.roster.split(" | "))
        idx = roster_index[roster]
        mask = np.isin(players, roster)
        actual_difference_makers = sorted(
            predictions.loc[mask & events[0], "player"].tolist()
        )
        record = {
            "block": int(row.block),
            "arm": row.arm,
            "actual_managed_score": float(score_matrix[idx, 0]),
            "actual_difference_maker_count": float(events[0, mask].sum()),
            "actual_difference_makers": " | ".join(actual_difference_makers),
        }
        for metric, values in utility.items():
            record[f"actual_{metric}"] = float(values[idx, 0])
        rows.append(record)
    return pd.DataFrame(rows), coverage


def summarize(plan_rows: pd.DataFrame, actual_rows: pd.DataFrame, control: str):
    merged = plan_rows.merge(actual_rows, on=["block", "arm"], validate="one_to_one")
    metrics = [
        column for column in merged.columns
        if column.startswith("holdout_") or column.startswith("actual_")
    ] + [
        "dead_zone_rb_count", "rb_mean_year_exp", "rookie_rb_count",
        "count_qb", "count_rb", "count_wr", "count_te",
        "spend_qb", "spend_rb", "spend_wr", "spend_te",
    ]
    numeric_metrics = [
        column for column in metrics
        if column in merged and pd.api.types.is_numeric_dtype(merged[column])
    ]
    summary = merged.groupby("arm", as_index=False)[numeric_metrics].mean()
    baseline = merged.loc[merged.arm.eq(control)].set_index("block")
    paired_rows = []
    for arm in ARMS:
        arm_rows = merged.loc[merged.arm.eq(arm)].set_index("block")
        record = {"arm": arm, "control": control, "blocks": len(arm_rows)}
        for metric in numeric_metrics:
            delta = arm_rows[metric] - baseline[metric]
            record[f"{metric}_delta"] = float(delta.mean())
            record[f"{metric}_delta_lcb80"] = base.paired_lcb(delta)
        paired_rows.append(record)
    return merged, summary, pd.DataFrame(paired_rows)


def write_summary(summary, paired_control, paired_mean, frequencies, metadata):
    labels = {
        "waiver_control": "Waiver control",
        "mean_frontier": "Exact-mean frontier",
        "win_g005": "Win probability (0.5%)",
        "excess_g005": "Expected excess (0.5%)",
        "power_g005": "Power win (0.5%)",
        "win_g010": "Win probability (1.0%)",
        "excess_g010": "Expected excess (1.0%)",
        "power_g010": "Power win (1.0%)",
        "win_direct_g010": "Direct win probability (1.0%)",
        "excess_direct_g010": "Direct expected excess (1.0%)",
        "power_direct_g010": "Direct power win (1.0%)",
        "win_pure": "Pure win probability",
        "excess_pure": "Pure expected excess",
        "power_pure": "Pure power win",
        "win_half_mean": "50/50 mean + win",
        "excess_half_mean": "50/50 mean + excess",
        "power_half_mean": "50/50 mean + power",
    }
    primary = ("waiver_control", "mean_frontier", "win_g005", "excess_g005", "power_g005")
    lines = [
        "# Auction Power-Win Results",
        "",
        "All arms use the same best-available waiver proxy. Simulated holdouts are the decision authority; actual 2025 results are descriptive because that season was inspected before this test.",
        "",
        "## Primary 0.5% guardrail",
        "",
        "| Arm | Managed EV | P90 | Win proxy | Expected excess | Power utility | Dominant-win proxy | P(2+ difference-makers) | Dead-zone RBs | Actual 2025 score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    indexed = summary.set_index("arm")
    for arm in primary:
        row = indexed.loc[arm]
        lines.append(
            f"| {labels[arm]} | {row.holdout_mean:.2f} | {row.holdout_p90:.2f} | "
            f"{row.holdout_win_probability:.3%} | {row.holdout_expected_excess:.2f} | "
            f"{row.holdout_power_utility:.4f} | {row.holdout_dominant_win_probability:.3%} | "
            f"{row.holdout_prob_two_difference_makers:.2%} | {row.dead_zone_rb_count:.2f} | "
            f"{row.actual_managed_score:.2f} |"
        )
    lines.extend([
        "",
        "## Paired deltas versus waiver control",
        "",
        "| Arm | EV delta | EV LCB80 | Win delta | Win LCB80 | Excess delta | Excess LCB80 | Power delta | Power LCB80 | P90 delta | Actual-score delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    control_rows = paired_control.set_index("arm")
    for arm in primary:
        row = control_rows.loc[arm]
        lines.append(
            f"| {labels[arm]} | {row.holdout_mean_delta:+.2f} | {row.holdout_mean_delta_lcb80:+.2f} | "
            f"{row.holdout_win_probability_delta:+.3%} | {row.holdout_win_probability_delta_lcb80:+.3%} | "
            f"{row.holdout_expected_excess_delta:+.2f} | {row.holdout_expected_excess_delta_lcb80:+.2f} | "
            f"{row.holdout_power_utility_delta:+.4f} | {row.holdout_power_utility_delta_lcb80:+.4f} | "
            f"{row.holdout_p90_delta:+.2f} | {row.actual_managed_score_delta:+.2f} |"
        )
    lines.extend([
        "",
        "## Aggressive 1.0% sensitivity versus exact-mean frontier",
        "",
        "| Arm | EV delta | Win delta | Excess delta | Power delta | P90 delta | Dead-zone RB delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    mean_rows = paired_mean.set_index("arm")
    for arm in ("win_g010", "excess_g010", "power_g010"):
        row = mean_rows.loc[arm]
        lines.append(
            f"| {labels[arm]} | {row.holdout_mean_delta:+.2f} | "
            f"{row.holdout_win_probability_delta:+.3%} | "
            f"{row.holdout_expected_excess_delta:+.2f} | "
            f"{row.holdout_power_utility_delta:+.4f} | {row.holdout_p90_delta:+.2f} | "
            f"{row.dead_zone_rb_count_delta:+.2f} |"
        )
    lines.extend([
        "",
        "## Exploratory direct-objective sensitivity versus exact-mean frontier",
        "",
        "| Arm | EV delta | Win delta | Win LCB80 | Excess delta | Excess LCB80 | Power delta | Power LCB80 | P90 delta | Actual-score delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ("win_direct_g010", "excess_direct_g010", "power_direct_g010"):
        row = mean_rows.loc[arm]
        lines.append(
            f"| {labels[arm]} | {row.holdout_mean_delta:+.2f} | "
            f"{row.holdout_win_probability_delta:+.3%} | "
            f"{row.holdout_win_probability_delta_lcb80:+.3%} | "
            f"{row.holdout_expected_excess_delta:+.2f} | "
            f"{row.holdout_expected_excess_delta_lcb80:+.2f} | "
            f"{row.holdout_power_utility_delta:+.4f} | "
            f"{row.holdout_power_utility_delta_lcb80:+.4f} | "
            f"{row.holdout_p90_delta:+.2f} | {row.actual_managed_score_delta:+.2f} |"
        )
    lines.extend([
        "",
        "## Unguarded pure objectives versus exact-mean frontier",
        "",
        "| Arm | EV delta | P10 delta | P90 delta | Win delta | Expected-excess delta | Power delta | P(2+) delta | Dead-zone RB delta | Actual-score delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ("win_pure", "excess_pure", "power_pure"):
        row = mean_rows.loc[arm]
        lines.append(
            f"| {labels[arm]} | {row.holdout_mean_delta:+.2f} | "
            f"{row.holdout_p10_delta:+.2f} | {row.holdout_p90_delta:+.2f} | "
            f"{row.holdout_win_probability_delta:+.3%} | "
            f"{row.holdout_expected_excess_delta:+.2f} | "
            f"{row.holdout_power_utility_delta:+.4f} | "
            f"{row.holdout_prob_two_difference_makers_delta:+.2%} | "
            f"{row.dead_zone_rb_count_delta:+.2f} | {row.actual_managed_score_delta:+.2f} |"
        )
    lines.extend([
        "",
        "## Standardized 50/50 mean + tail objectives versus exact-mean frontier",
        "",
        "| Arm | EV delta | P10 delta | P90 delta | Win delta | Expected-excess delta | Power delta | P(2+) delta | Dead-zone RB delta | Actual-score delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ("win_half_mean", "excess_half_mean", "power_half_mean"):
        row = mean_rows.loc[arm]
        lines.append(
            f"| {labels[arm]} | {row.holdout_mean_delta:+.2f} | "
            f"{row.holdout_p10_delta:+.2f} | {row.holdout_p90_delta:+.2f} | "
            f"{row.holdout_win_probability_delta:+.3%} | "
            f"{row.holdout_expected_excess_delta:+.2f} | "
            f"{row.holdout_power_utility_delta:+.4f} | "
            f"{row.holdout_prob_two_difference_makers_delta:+.2%} | "
            f"{row.dead_zone_rb_count_delta:+.2f} | {row.actual_managed_score_delta:+.2f} |"
        )
    lines.extend(["", "## Most frequent RBs", ""])
    for arm in primary:
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
        "## Guardrails",
        "",
        f"- Power utility uses alpha `{metadata['power_alpha']}`, margin scale `{metadata['margin_scale']}`, and dominant margin `{metadata['dominant_margin']}`.",
        "- Selection uses construction-bank paired LCB80; evaluation uses independent common contexts.",
        "- The opponent field is a common feasible-roster empirical bank, not eleven mutually exclusive auction rosters.",
        "- Actual 2025 outcomes are descriptive and cannot confirm this post-hoc objective choice.",
    ])
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(args):
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = base.prepare_state(
        block_count=args.blocks,
        construction_contexts=args.construction_contexts,
        seed=args.seed,
    )
    candidate_frames = []
    plan_frames = []
    cell_frames = []
    roster_rows = []
    all_candidate_rosters = set()
    static_cache = {}
    validation_seeds = np.random.SeedSequence(args.seed + 500).spawn(args.blocks)
    try:
        for block_idx in range(args.blocks):
            plans, sources = compile_candidates(
                state,
                block_idx,
                args.candidate_starts,
                args.seed + 1000 * (block_idx + 1),
                static_cache,
            )
            metrics, rosters, _ = score_candidates(
                state, block_idx, plans, sources
            )
            selected = select_arms(metrics, plans)
            candidate_frames.append(metrics)
            all_candidate_rosters.update(rosters)
            validation_seed = int(
                validation_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
            )
            validation_bank = base.sequential._sample_validation_bank(
                state["sim"],
                state["predictions"],
                args.validation_contexts,
                16,
                6,
                0.65,
                validation_seed,
                canonical_predictions=state["canonical_predictions"],
            )
            plan_frame, cell_frame = score_validation_block(
                state, block_idx, selected, rosters, validation_bank
            )
            plan_frames.append(plan_frame)
            cell_frames.append(cell_frame)
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
        cell_rows = pd.concat(cell_frames, ignore_index=True)
        roster_players = pd.DataFrame(roster_rows)
        actual_rows, actual_coverage = score_actual(
            state, plan_rows, all_candidate_rosters
        )
        merged, summary, paired_control = summarize(
            plan_rows, actual_rows, "waiver_control"
        )
        _, _, paired_mean = summarize(plan_rows, actual_rows, "mean_frontier")
        frequencies = (
            roster_players.groupby(["arm", "player", "pos", "year_exp"], as_index=False)
            .agg(blocks_selected=("block", "nunique"))
        )
        frequencies["selection_rate"] = frequencies.blocks_selected / args.blocks
        metadata = {
            "year": base.YEAR,
            "league": base.LEAGUE,
            "salary_source": "actual",
            "waivers": state["churn_waivers"],
            "primary_guardrail": PRIMARY_GUARDRAIL,
            "sensitivity_guardrail": SENSITIVITY_GUARDRAIL,
            "power_alpha": POWER_ALPHA,
            "margin_scale": MARGIN_SCALE,
            "dominant_margin": DOMINANT_MARGIN,
            "opponents": base.NUM_TEAMS - 1,
            "blocks": args.blocks,
            "construction_contexts_per_block": args.construction_contexts,
            "candidate_start_budget": args.candidate_starts,
            "validation_contexts_per_block": args.validation_contexts,
            "unique_candidate_rosters": len(all_candidate_rosters),
            "seed": args.seed,
            "actual_2025_status": "descriptive_post_hoc",
            "actual_weekly_coverage": actual_coverage,
            "runtime_seconds": time.perf_counter() - started,
            "production_changed": False,
        }
        candidate_metrics.to_csv(RESULTS_DIR / "candidate_rosters.csv", index=False)
        merged.to_csv(RESULTS_DIR / "plan_blocks.csv", index=False)
        cell_rows.to_csv(RESULTS_DIR / "holdout_utility_cells.csv", index=False)
        roster_players.to_csv(RESULTS_DIR / "roster_players.csv", index=False)
        frequencies.to_csv(RESULTS_DIR / "player_frequency.csv", index=False)
        actual_rows.to_csv(RESULTS_DIR / "actual_2025_descriptive.csv", index=False)
        summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
        paired_control.to_csv(RESULTS_DIR / "paired_vs_waiver_control.csv", index=False)
        paired_mean.to_csv(RESULTS_DIR / "paired_vs_mean_frontier.csv", index=False)
        (RESULTS_DIR / "metadata.json").write_text(
            json.dumps(json_value(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_summary(summary, paired_control, paired_mean, frequencies, metadata)
        print("\nSUMMARY")
        print(summary.to_string(index=False))
        print("\nPAIRED VS WAIVER CONTROL")
        print(paired_control.to_string(index=False))
        print("\nRuntime seconds", metadata["runtime_seconds"])
        return summary, paired_control, paired_mean
    finally:
        state["conn"].close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--construction-contexts", type=int, default=64)
    parser.add_argument("--candidate-starts", type=int, default=128)
    parser.add_argument("--validation-contexts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


if __name__ == "__main__":
    run_study(parse_args())
