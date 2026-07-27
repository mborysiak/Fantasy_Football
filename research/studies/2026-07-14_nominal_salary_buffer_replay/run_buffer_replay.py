"""Paired replay of nominal predicted-salary buffers by salary draw count."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
BASE_STUDY_DIR = (
    ROOT / "research" / "studies" / "2026-07-13_managed_auction_rolling_replay"
)
BASE_RUNNER = BASE_STUDY_DIR / "run_replay.py"
BASE_RESULTS = BASE_STUDY_DIR / "results"
BASE_MANIFEST = BASE_RESULTS / "source_manifest.json"
BASE_TRIALS = BASE_RESULTS / "roster_trials.csv"


def load_base_replay() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_managed_auction_replay_base",
        BASE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base replay runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_replay()

BUFFER_SPECS: tuple[tuple[str, float | None], ...] = (
    ("none", None),
    ("0", 0.0),
    ("5", 5.0),
    ("10", 10.0),
    ("15", 15.0),
    ("25", 25.0),
)
BUFFER_LABELS = [label for label, _ in BUFFER_SPECS]
BUFFER_VALUES = dict(BUFFER_SPECS)
CONTROL_LABEL = "none"
CURRENT_WAIVER = "current_projected"
CURRENT_BENCH_WEIGHT = 0.25


def variant_name(draw_count: int, buffer_label: str) -> str:
    return f"d{draw_count}_nominal{buffer_label}"


def attach_buffer_dollars(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a numeric buffer key beside labels that may round-trip through CSV."""
    output = frame.copy()
    output["nominal_buffer_dollars"] = output.nominal_buffer.map(BUFFER_VALUES)
    return output


def run_buffer_trials(
    year: int,
    sim: Any,
    predictions: pd.DataFrame,
    salary_draws: np.ndarray,
    raw_nominal_salaries: np.ndarray,
    normalized_nominal_salaries: np.ndarray,
    salary_source_matched: np.ndarray,
    environment: dict[str, Any],
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    evaluation_weekly: np.ndarray,
    evaluation_decisions: np.ndarray,
    evaluation_played: np.ndarray,
    managed_values: np.ndarray,
    waiver_baseline: dict[str, float],
    trials: int,
    context_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    remaining_budget = base.TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = base.TOTAL_MARKET_SLOTS - environment["keeper_count"]
    top_n = predictions.nlargest(
        min(base.TOP_N, len(predictions)),
        "salary",
    ).player.tolist()
    static = sim.build_managed_ilp_static_matrices(
        predictions,
        {},
        [],
        top_n,
        base.ROSTER_SIZE,
        base.POS_MIN,
        base.POS_MAX,
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
    markets: dict[int, np.ndarray] = {}
    for draw_count in (1, 5):
        raw = np.column_stack(
            [
                salary_draws[:, plan_row[:draw_count]].mean(axis=1)
                for plan_row in salary_plan
            ]
        )
        markets[draw_count] = base.normalize_market_draws(
            sim,
            raw,
            remaining_budget,
            remaining_slots,
        )

    forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trial in range(trials):
        context_idx = context_plan[trial]
        objective = managed_values[:, context_idx].mean(axis=1)
        for draw_count in (1, 5):
            sampled_market = markets[draw_count][:, trial]
            predictions["salary"] = sampled_market
            for buffer_label, buffer_value in BUFFER_SPECS:
                nominal_cap = (
                    None
                    if buffer_value is None
                    else base.SALARY_CAP + float(buffer_value)
                )
                solved = sim._solve_managed_scenario(
                    predictions,
                    objective,
                    ref_weekly,
                    ref_decisions,
                    static,
                    [],
                    {},
                    top_n,
                    base.ROSTER_SIZE,
                    base.POS_MIN,
                    base.POS_MAX,
                    waiver_baseline,
                    base.LINEUP_REQUIRE,
                    True,
                    refine_roster=True,
                    score_roster=False,
                    salary_values=sampled_market,
                    played_mask=ref_played,
                    nominal_salary_values=(
                        None
                        if nominal_cap is None
                        else normalized_nominal_salaries
                    ),
                    nominal_salary_cap=nominal_cap,
                )
                if solved is None:
                    rows.append(
                        {
                            "year": year,
                            "trial": trial,
                            "salary_draw_count": draw_count,
                            "nominal_buffer": buffer_label,
                            "nominal_buffer_dollars": buffer_value,
                            "nominal_cap": nominal_cap,
                            "variant": variant_name(draw_count, buffer_label),
                            "solve_status": "infeasible",
                        }
                    )
                    continue

                selected = np.asarray(solved["selected_mask"], dtype=bool)
                roster = tuple(sorted(solved["selected_players"]))
                actual_score = base.score_actual_roster(environment, roster)
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
                sampled_spend = float(sampled_market[selected].sum())
                nominal_spend = float(normalized_nominal_salaries[selected].sum())
                raw_nominal_spend = float(raw_nominal_salaries[selected].sum())
                nominal_slack = (
                    np.nan if nominal_cap is None else nominal_cap - nominal_spend
                )
                pos_counts = predictions.loc[selected, "pos"].value_counts().to_dict()
                actual_feasible = (
                    actual_score["actual_salary_spend"] <= base.SALARY_CAP + 1e-8
                )
                rows.append(
                    {
                        "year": year,
                        "trial": trial,
                        "salary_draw_count": draw_count,
                        "nominal_buffer": buffer_label,
                        "nominal_buffer_dollars": buffer_value,
                        "nominal_cap": nominal_cap,
                        "variant": variant_name(draw_count, buffer_label),
                        "solve_status": "optimal",
                        "roster": "|".join(roster),
                        "forecast_salary_spend": sampled_spend,
                        "normalized_nominal_salary_spend": nominal_spend,
                        "raw_nominal_salary_spend": raw_nominal_spend,
                        "nominal_slack": nominal_slack,
                        "near_nominal_cap_within_1": bool(
                            nominal_cap is not None and nominal_slack <= 1.0 + 1e-8
                        ),
                        "baseline_nominal_violation": False,
                        "actual_cap_feasible": bool(actual_feasible),
                        "actual_cap_overage": float(
                            max(
                                actual_score["actual_salary_spend"]
                                - base.SALARY_CAP,
                                0.0,
                            )
                        ),
                        "forecast_ev": forecast_ev,
                        "forecast_error": actual_score["actual_points"] - forecast_ev,
                        "contains_top_n": bool(set(roster) & set(top_n)),
                        "salary_source_missing_players": int(
                            (~salary_source_matched[selected]).sum()
                        ),
                        "qb_count": int(pos_counts.get("QB", 0)),
                        "rb_count": int(pos_counts.get("RB", 0)),
                        "wr_count": int(pos_counts.get("WR", 0)),
                        "te_count": int(pos_counts.get("TE", 0)),
                        **actual_score,
                    }
                )
        if (trial + 1) % max(1, min(25, trials)) == 0:
            print(
                f"{year}: completed {trial + 1}/{trials} paired trials "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    controls = frame[frame.nominal_buffer.eq(CONTROL_LABEL)][
        [
            "year",
            "trial",
            "salary_draw_count",
            "normalized_nominal_salary_spend",
        ]
    ].rename(
        columns={"normalized_nominal_salary_spend": "control_nominal_spend"}
    )
    frame = frame.merge(
        controls,
        on=["year", "trial", "salary_draw_count"],
        how="left",
        validate="many_to_one",
    )
    active = frame.nominal_buffer.ne(CONTROL_LABEL)
    frame.loc[active, "baseline_nominal_violation"] = (
        frame.loc[active, "control_nominal_spend"]
        > frame.loc[active, "nominal_cap"] + 1e-8
    )
    return frame, {
        "top_n_players": top_n,
        "remaining_market_budget": remaining_budget,
        "remaining_market_slots": remaining_slots,
        "normalized_nominal_market_total": float(
            np.sort(normalized_nominal_salaries)[-remaining_slots:].sum()
        ),
        "variant_runtime_seconds": time.perf_counter() - started,
    }


PAIR_METRICS = [
    "actual_points",
    "drafted_only_points",
    "actual_cap_feasible",
    "actual_cap_overage",
    "actual_salary_spend",
    "forecast_ev",
    "forecast_error",
    "forecast_salary_spend",
    "normalized_nominal_salary_spend",
    "raw_nominal_salary_spend",
    "actual_waiver_starts",
    "actual_salary_missing_players",
    "salary_source_missing_players",
]


def paired_rows(
    default: pd.DataFrame,
    candidate: pd.DataFrame,
    join_cols: list[str],
    comparison_type: str,
    comparison: str,
    draw_context: str,
    buffer_label: str,
) -> pd.DataFrame:
    keep = [
        *join_cols,
        "roster",
        "near_nominal_cap_within_1",
        "baseline_nominal_violation",
        *PAIR_METRICS,
    ]
    merged = default[keep].merge(
        candidate[keep],
        on=join_cols,
        suffixes=("_default", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        values = row._asdict()
        jaccard, changed = base.roster_jaccard(
            values["roster_default"],
            values["roster_candidate"],
        )
        both_feasible = bool(
            values["actual_cap_feasible_default"]
            and values["actual_cap_feasible_candidate"]
        )
        output: dict[str, Any] = {
            "comparison_type": comparison_type,
            "comparison": comparison,
            "draw_context": draw_context,
            "nominal_buffer": buffer_label,
            "year": int(values["year"]),
            "trial": int(values["trial"]),
            "roster_jaccard": jaccard,
            "roster_slots_changed": changed,
            "roster_changed": changed > 0,
            "both_actual_cap_feasible": both_feasible,
            "candidate_near_nominal_cap_within_1": bool(
                values["near_nominal_cap_within_1_candidate"]
            ),
            "baseline_nominal_violation": bool(
                values["baseline_nominal_violation_candidate"]
            ),
        }
        for metric in PAIR_METRICS:
            default_value = values[f"{metric}_default"]
            candidate_value = values[f"{metric}_candidate"]
            output[f"{metric}_effect"] = float(candidate_value) - float(default_value)
        output["joint_feasible_actual_points_effect"] = (
            values["actual_points_candidate"] - values["actual_points_default"]
            if both_feasible
            else np.nan
        )
        rows.append(output)
    return pd.DataFrame(rows)


def build_pair_tables(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    optimal = trials[trials.solve_status.eq("optimal")].copy()
    buffer_pairs = []
    for draw_count in (1, 5):
        default = optimal[
            optimal.salary_draw_count.eq(draw_count)
            & optimal.nominal_buffer.eq(CONTROL_LABEL)
        ]
        for buffer_label in BUFFER_LABELS[1:]:
            candidate = optimal[
                optimal.salary_draw_count.eq(draw_count)
                & optimal.nominal_buffer.eq(buffer_label)
            ]
            buffer_pairs.append(
                paired_rows(
                    default,
                    candidate,
                    ["year", "trial"],
                    "buffer_vs_none",
                    f"buffer_{buffer_label}_minus_none",
                    str(draw_count),
                    buffer_label,
                )
            )
    buffer_frame = attach_buffer_dollars(pd.concat(buffer_pairs, ignore_index=True))

    draw_pairs = []
    for buffer_label in BUFFER_LABELS:
        default = optimal[
            optimal.salary_draw_count.eq(5)
            & optimal.nominal_buffer.eq(buffer_label)
        ]
        candidate = optimal[
            optimal.salary_draw_count.eq(1)
            & optimal.nominal_buffer.eq(buffer_label)
        ]
        draw_pairs.append(
            paired_rows(
                default,
                candidate,
                ["year", "trial"],
                "one_minus_five",
                f"one_minus_five_at_{buffer_label}",
                "1_minus_5",
                buffer_label,
            )
        )
    draw_frame = attach_buffer_dollars(pd.concat(draw_pairs, ignore_index=True))

    effect_columns = [
        column for column in buffer_frame.columns if column.endswith("_effect")
    ]
    d1 = buffer_frame[buffer_frame.draw_context.eq("1")].copy()
    d5 = buffer_frame[buffer_frame.draw_context.eq("5")].copy()
    did = d5[
        [
            "comparison",
            "nominal_buffer",
            "year",
            "trial",
            "both_actual_cap_feasible",
            *effect_columns,
        ]
    ].merge(
        d1[
            [
                "comparison",
                "nominal_buffer",
                "year",
                "trial",
                "both_actual_cap_feasible",
                *effect_columns,
            ]
        ],
        on=["comparison", "nominal_buffer", "year", "trial"],
        suffixes=("_d5", "_d1"),
        validate="one_to_one",
    )
    did["comparison_type"] = "buffer_effect_d1_minus_d5"
    did["draw_context"] = "d1_minus_d5"
    did["nominal_buffer_dollars"] = did.nominal_buffer.map(BUFFER_VALUES)
    did["all_four_actual_cap_feasible"] = (
        did["both_actual_cap_feasible_d1"]
        & did["both_actual_cap_feasible_d5"]
    )
    for column in effect_columns:
        did[f"{column}_interaction"] = did[f"{column}_d1"] - did[f"{column}_d5"]
    keep = [
        "comparison_type",
        "comparison",
        "draw_context",
        "nominal_buffer",
        "nominal_buffer_dollars",
        "year",
        "trial",
        "all_four_actual_cap_feasible",
        *[f"{column}_interaction" for column in effect_columns],
    ]
    return buffer_frame, draw_frame, did[keep]


def summarize_pairs(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_columns = [column for column in paired.columns if column.endswith("_effect")]
    group_cols = [
        "comparison_type",
        "comparison",
        "draw_context",
        "nominal_buffer",
        "year",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "comparisons": ("trial", "size"),
    }
    for optional in [
        "roster_changed",
        "roster_jaccard",
        "both_actual_cap_feasible",
        "candidate_near_nominal_cap_within_1",
        "baseline_nominal_violation",
    ]:
        if optional in paired:
            aggregations[f"mean_{optional}"] = (optional, "mean")
    aggregations.update(
        {f"mean_{column}": (column, "mean") for column in effect_columns}
    )
    by_year = paired.groupby(group_cols, as_index=False).agg(**aggregations)

    across_rows = []
    for keys, group in by_year.groupby(group_cols[:-1]):
        row = dict(zip(group_cols[:-1], keys))
        row["seasons"] = int(group.year.nunique())
        for column in [c for c in by_year if c.startswith("mean_")]:
            row[column] = float(group[column].mean())
            row[f"development_2022_2024_{column}"] = float(
                group.loc[group.year.le(2024), column].mean()
            )
            check = group.loc[group.year.eq(2025), column]
            row[f"temporal_check_2025_{column}"] = (
                float(check.iloc[0]) if len(check) else np.nan
            )
        across_rows.append(row)

    half = paired.copy()
    trial_count = int(half.trial.max()) + 1
    split_at = max(1, trial_count // 2)
    half["trial_half"] = np.where(
        half.trial.lt(split_at),
        f"0_{split_at - 1}",
        f"{split_at}_{trial_count - 1}",
    )
    half_year = (
        half.groupby([*group_cols[:-1], "year", "trial_half"], as_index=False)[
            effect_columns
        ]
        .mean()
    )
    split_half = (
        half_year.groupby([*group_cols[:-1], "trial_half"], as_index=False)[
            effect_columns
        ]
        .mean()
    )
    return (
        attach_buffer_dollars(by_year),
        attach_buffer_dollars(pd.DataFrame(across_rows)),
        attach_buffer_dollars(split_half),
    )


def summarize_interactions(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        column for column in interactions if column.endswith("_interaction")
    ]
    keys = [
        "comparison_type",
        "comparison",
        "draw_context",
        "nominal_buffer",
    ]
    by_year = interactions.groupby([*keys, "year"], as_index=False).agg(
        all_four_feasible_share=("all_four_actual_cap_feasible", "mean"),
        all_four_feasible_count=("all_four_actual_cap_feasible", "sum"),
        **{column: (column, "mean") for column in metric_columns},
    )
    across = by_year.groupby(keys, as_index=False).agg(
        mean_all_four_feasible_share=("all_four_feasible_share", "mean"),
        total_all_four_feasible_count=("all_four_feasible_count", "sum"),
        **{column: (column, "mean") for column in metric_columns},
    )
    return attach_buffer_dollars(by_year), attach_buffer_dollars(across)


def build_variant_summaries(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    optimal = trials[trials.solve_status.eq("optimal")].copy()
    optimal["absolute_forecast_error"] = optimal.forecast_error.abs()
    by_year = (
        optimal.groupby(
            ["year", "variant", "salary_draw_count", "nominal_buffer"],
            as_index=False,
        )
        .agg(
            trials=("trial", "size"),
            unique_rosters=("roster", "nunique"),
            actual_points=("actual_points", "mean"),
            drafted_only_points=("drafted_only_points", "mean"),
            cap_feasible_rate=("actual_cap_feasible", "mean"),
            feasible_trials=("actual_cap_feasible", "sum"),
            mean_cap_overage=("actual_cap_overage", "mean"),
            median_cap_overage=("actual_cap_overage", "median"),
            p90_cap_overage=("actual_cap_overage", lambda values: values.quantile(0.9)),
            actual_salary_spend=("actual_salary_spend", "mean"),
            forecast_salary_spend=("forecast_salary_spend", "mean"),
            normalized_nominal_salary_spend=(
                "normalized_nominal_salary_spend",
                "mean",
            ),
            raw_nominal_salary_spend=("raw_nominal_salary_spend", "mean"),
            near_nominal_cap_within_1_rate=(
                "near_nominal_cap_within_1",
                "mean",
            ),
            baseline_nominal_violation_rate=("baseline_nominal_violation", "mean"),
            forecast_ev=("forecast_ev", "mean"),
            absolute_forecast_error=("absolute_forecast_error", "mean"),
            actual_waiver_starts=("actual_waiver_starts", "mean"),
            actual_price_missing=("actual_salary_missing_players", "mean"),
            salary_source_missing=("salary_source_missing_players", "mean"),
            actual_points_feasible=(
                "actual_points",
                lambda values: values[
                    optimal.loc[values.index, "actual_cap_feasible"]
                ].mean(),
            ),
        )
    )
    across = (
        by_year.groupby(
            ["variant", "salary_draw_count", "nominal_buffer"],
            as_index=False,
        )
        .agg(
            seasons=("year", "nunique"),
            total_trials=("trials", "sum"),
            feasible_trials=("feasible_trials", "sum"),
            actual_points=("actual_points", "mean"),
            drafted_only_points=("drafted_only_points", "mean"),
            cap_feasible_rate=("cap_feasible_rate", "mean"),
            mean_cap_overage=("mean_cap_overage", "mean"),
            median_cap_overage=("median_cap_overage", "mean"),
            p90_cap_overage=("p90_cap_overage", "mean"),
            actual_salary_spend=("actual_salary_spend", "mean"),
            forecast_salary_spend=("forecast_salary_spend", "mean"),
            normalized_nominal_salary_spend=(
                "normalized_nominal_salary_spend",
                "mean",
            ),
            near_nominal_cap_within_1_rate=(
                "near_nominal_cap_within_1_rate",
                "mean",
            ),
            baseline_nominal_violation_rate=(
                "baseline_nominal_violation_rate",
                "mean",
            ),
            forecast_ev=("forecast_ev", "mean"),
            absolute_forecast_error=("absolute_forecast_error", "mean"),
            actual_waiver_starts=("actual_waiver_starts", "mean"),
            actual_points_feasible=("actual_points_feasible", "mean"),
        )
    )
    period_rows = []
    metric_columns = [
        "actual_points",
        "drafted_only_points",
        "cap_feasible_rate",
        "mean_cap_overage",
        "p90_cap_overage",
        "actual_salary_spend",
        "normalized_nominal_salary_spend",
        "near_nominal_cap_within_1_rate",
        "actual_points_feasible",
        "forecast_ev",
        "absolute_forecast_error",
    ]
    for keys, group in by_year.groupby(
        ["variant", "salary_draw_count", "nominal_buffer"]
    ):
        row = dict(zip(["variant", "salary_draw_count", "nominal_buffer"], keys))
        for column in metric_columns:
            row[f"development_2022_2024_{column}"] = float(
                group.loc[group.year.le(2024), column].mean()
            )
            check = group.loc[group.year.eq(2025), column]
            row[f"temporal_check_2025_{column}"] = (
                float(check.iloc[0]) if len(check) else np.nan
            )
        period_rows.append(row)
    return (
        attach_buffer_dollars(by_year),
        attach_buffer_dollars(across),
        attach_buffer_dollars(pd.DataFrame(period_rows)),
    )


def add_development_pareto(periods: pd.DataFrame) -> pd.DataFrame:
    frame = periods.copy()
    frame["development_pareto"] = pd.Series(
        [True] * len(frame),
        index=frame.index,
        dtype="boolean",
    )
    points_col = "development_2022_2024_actual_points"
    feasible_col = "development_2022_2024_cap_feasible_rate"
    overage_col = "development_2022_2024_mean_cap_overage"
    for idx, row in frame.iterrows():
        if pd.isna(row[[points_col, feasible_col, overage_col]]).any():
            frame.loc[idx, "development_pareto"] = pd.NA
            continue
        dominates = (
            frame[points_col].ge(row[points_col])
            & frame[feasible_col].ge(row[feasible_col])
            & frame[overage_col].le(row[overage_col])
            & (
                frame[points_col].gt(row[points_col])
                | frame[feasible_col].gt(row[feasible_col])
                | frame[overage_col].lt(row[overage_col])
            )
        )
        if dominates.any():
            frame.loc[idx, "development_pareto"] = False
    return frame


def validate_trial_invariants(trials: pd.DataFrame) -> dict[str, bool]:
    roster_players = trials.roster.str.split("|")
    if not roster_players.map(len).eq(base.ROSTER_SIZE).all():
        raise AssertionError("A buffer replay roster does not contain 13 players.")
    if not roster_players.map(lambda players: len(set(players))).eq(
        base.ROSTER_SIZE
    ).all():
        raise AssertionError("A buffer replay roster contains a duplicate player.")

    for pos in base.POSITIONS:
        column = f"{pos.lower()}_count"
        if trials[column].lt(base.POS_MIN[pos]).any():
            raise AssertionError(f"A buffer replay roster is below the {pos} minimum.")
        if trials[column].gt(base.POS_MAX[pos]).any():
            raise AssertionError(f"A buffer replay roster exceeds the {pos} maximum.")

    expected_feasible = trials.actual_salary_spend.le(base.SALARY_CAP + 1e-8)
    if not trials.actual_cap_feasible.astype(bool).eq(expected_feasible).all():
        raise AssertionError("Actual-cap feasibility disagrees with realized spend.")
    expected_overage = np.maximum(
        trials.actual_salary_spend.to_numpy(dtype=float) - base.SALARY_CAP,
        0.0,
    )
    if not np.allclose(
        trials.actual_cap_overage,
        expected_overage,
        atol=1e-8,
        rtol=0.0,
    ):
        raise AssertionError("Actual-cap overage disagrees with realized spend.")

    constrained = trials[trials.nominal_buffer.ne(CONTROL_LABEL)].copy()
    constrained = constrained.sort_values(
        ["year", "trial", "salary_draw_count", "nominal_buffer_dollars"]
    )
    constrained["violation_int"] = constrained.baseline_nominal_violation.astype(int)
    violation_change = constrained.groupby(
        ["year", "trial", "salary_draw_count"]
    ).violation_int.diff()
    if violation_change.fillna(0).gt(0).any():
        raise AssertionError(
            "Baseline nominal violations are not monotone as the buffer increases."
        )

    return {
        "all_rosters_size_13": True,
        "all_rosters_have_unique_players": True,
        "all_position_bounds_satisfied": True,
        "actual_cap_feasibility_matches_spend": True,
        "actual_cap_overage_matches_spend": True,
        "baseline_nominal_violations_monotone": True,
    }


def validate_prior_controls(
    trials: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    is_full_default = (
        args.years == [2022, 2023, 2024, 2025]
        and args.trials == 250
        and args.contexts == 250
        and args.context_draws == 5
        and args.projection_draws == 1000
        and args.salary_draws == 5000
        and args.seed == 20260713
    )
    result: dict[str, Any] = {
        "required": bool(is_full_default),
        "checked": False,
        "matched": None,
    }
    if not is_full_default:
        return result
    if not BASE_TRIALS.exists():
        raise FileNotFoundError(f"Prior replay controls are missing: {BASE_TRIALS}")
    prior = pd.read_csv(BASE_TRIALS)
    prior = prior[
        prior.enforce_top_n
        & prior.waiver_source.eq(CURRENT_WAIVER)
        & prior.bench_upside_weight.eq(CURRENT_BENCH_WEIGHT)
    ].copy()
    current = trials[trials.nominal_buffer.eq(CONTROL_LABEL)].copy()
    joined = prior.merge(
        current,
        on=["year", "trial", "salary_draw_count"],
        suffixes=("_prior", "_current"),
        validate="one_to_one",
    )
    if len(joined) != 2000:
        raise AssertionError(f"Expected 2,000 prior-control rows, found {len(joined)}.")
    exact_columns = ["roster", "solve_status", "contains_top_n"]
    numeric_columns = [
        "forecast_salary_spend",
        "actual_points",
        "drafted_only_points",
        "actual_salary_spend",
        "actual_cap_overage",
        "forecast_ev",
        "forecast_error",
        "actual_waiver_starts",
    ]
    mismatches: dict[str, int] = {}
    for column in exact_columns:
        mismatches[column] = int(
            joined[f"{column}_prior"].ne(joined[f"{column}_current"]).sum()
        )
    for column in numeric_columns:
        mismatches[column] = int(
            (~np.isclose(
                joined[f"{column}_prior"],
                joined[f"{column}_current"],
                atol=1e-6,
                rtol=1e-9,
                equal_nan=True,
            )).sum()
        )
    if any(mismatches.values()):
        raise AssertionError(f"Unconstrained controls drifted from prior replay: {mismatches}")
    result.update(
        {
            "checked": True,
            "matched": True,
            "rows": int(len(joined)),
            "mismatches": mismatches,
        }
    )
    return result


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    variant_periods: pd.DataFrame,
    buffer_across: pd.DataFrame,
    draw_across: pd.DataFrame,
    interaction_across: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    periods = variant_periods.copy()
    order = {label: idx for idx, label in enumerate(BUFFER_LABELS)}
    periods["buffer_order"] = periods.nominal_buffer.map(order)
    periods = periods.sort_values(["salary_draw_count", "buffer_order"])
    buffer_display = buffer_across.copy()
    buffer_display["buffer_order"] = buffer_display.nominal_buffer.map(order)
    buffer_display = buffer_display.sort_values(["draw_context", "buffer_order"])
    draw_display = draw_across.copy()
    draw_display["buffer_order"] = draw_display.nominal_buffer.map(order)
    draw_display = draw_display.sort_values("buffer_order")
    interaction_display = interaction_across.copy()
    interaction_display["buffer_order"] = interaction_display.nominal_buffer.map(order)
    interaction_display = interaction_display.sort_values("buffer_order")

    lines = [
        "# Nominal Salary Buffer Replay Results",
        "",
        f"Run: {args.trials} paired trials across 12 cells per origin, "
        f"{args.contexts} construction plus independent evaluation contexts, "
        f"seed {args.seed}.",
        "",
        "Every cell retains the sampled-price $298 cap, Top-N on, projected waivers, "
        "and bench weight 0.25. Constrained cells add normalized point-price spend "
        "at or below $298 plus the named buffer.",
        "",
        "## Development and temporal-check outcomes",
        "",
        "Unqualified points include rosters that exceed historical final prices. "
        "Read them together with feasibility and overage; 2025 is a temporal check, "
        "not a pristine holdout.",
        "",
        base.markdown_table(
            periods,
            [
                "salary_draw_count",
                "nominal_buffer",
                "development_pareto",
                "development_2022_2024_actual_points",
                "development_2022_2024_cap_feasible_rate",
                "development_2022_2024_mean_cap_overage",
                "temporal_check_2025_actual_points",
                "temporal_check_2025_cap_feasible_rate",
                "temporal_check_2025_mean_cap_overage",
            ],
            digits=3,
        ),
        "",
        "`development_pareto` requires no other tested cell to have at least as many "
        "unqualified points, at least as much realized-price feasibility, and no more "
        "mean overage. It is a descriptive frontier, not an automatic policy choice.",
        "",
        "## Buffer-minus-no-constraint paired effects",
        "",
        "Positive point and feasibility effects favor the named buffer; negative "
        "overage and spend effects favor it. Baseline violation is the share of "
        "unconstrained rosters that the nominal row would reject.",
        "",
        base.markdown_table(
            buffer_display,
            [
                "draw_context",
                "nominal_buffer",
                "mean_actual_points_effect",
                "mean_actual_cap_feasible_effect",
                "mean_actual_cap_overage_effect",
                "mean_actual_salary_spend_effect",
                "mean_roster_changed",
                "mean_baseline_nominal_violation",
                "mean_candidate_near_nominal_cap_within_1",
                "mean_both_actual_cap_feasible",
                "mean_joint_feasible_actual_points_effect",
                "development_2022_2024_mean_actual_points_effect",
                "temporal_check_2025_mean_actual_points_effect",
            ],
            digits=3,
        ),
        "",
        "## One-minus-five draw effects at each buffer",
        "",
        base.markdown_table(
            draw_display,
            [
                "nominal_buffer",
                "mean_actual_points_effect",
                "mean_actual_cap_feasible_effect",
                "mean_actual_cap_overage_effect",
                "mean_actual_salary_spend_effect",
                "mean_roster_changed",
                "development_2022_2024_mean_actual_points_effect",
                "temporal_check_2025_mean_actual_points_effect",
            ],
            digits=3,
        ),
        "",
        "## Buffer-by-draw interaction",
        "",
        "This is (one-draw buffer effect) minus (five-draw buffer effect). Positive "
        "favors one draw for points and feasibility; negative favors one draw for "
        "overage and spend reductions. Joint-feasible interactions require all four "
        "underlying rosters to fit historical prices.",
        "",
        base.markdown_table(
            interaction_display,
            [
                "nominal_buffer",
                "actual_points_effect_interaction",
                "actual_cap_feasible_effect_interaction",
                "actual_cap_overage_effect_interaction",
                "actual_salary_spend_effect_interaction",
                "joint_feasible_actual_points_effect_interaction",
                "mean_all_four_feasible_share",
                "total_all_four_feasible_count",
            ],
            digits=3,
        ),
        "",
        "## Validation and limits",
        "",
        f"Prior unconstrained controls reproduced: {validation['prior_controls']}.",
        "",
        "- Historical final prices are exogenous and missing prices use the intentional "
        "$1 fallback, so realized affordability is diagnostic and optimistic.",
        "- This tests the nominal guardrail on frozen historical salary laws. It does "
        "not rebuild the current empirical residual-quantile method walk-forward.",
        "- Four seasons are four independent outcome units. Split halves measure Monte "
        "Carlo stability only.",
        "- Waiver eligibility remains hindsight availability-filtered and frictionless, "
        "as in the parent replay.",
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
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output-dir",
        default=str(STUDY_DIR / "results"),
    )
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(base.FROZEN_SOURCES))
    if invalid:
        parser.error(f"Unsupported replay years: {invalid}")
    if min(
        args.trials,
        args.contexts,
        args.context_draws,
        args.projection_draws,
        args.salary_draws,
    ) <= 0:
        parser.error("Trial and draw counts must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if not BASE_MANIFEST.exists():
        raise FileNotFoundError(f"Base replay manifest is missing: {BASE_MANIFEST}")
    prior_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    current_outcome_hashes = {
        "simulation_db_sha256": base.sha256_file(base.SIM_DB),
        "raw_weekly_sha256": base.sha256_file(base.DAILY_DB),
    }
    for key, current_hash in current_outcome_hashes.items():
        prior_hash = prior_manifest["current_outcome_sources"][key]
        if current_hash != prior_hash:
            raise AssertionError(
                f"Outcome source drifted since the base replay ({key})."
            )

    print("Loading raw outcomes and frozen replay inputs...", flush=True)
    raw_weekly = base.load_raw_weekly(max_year=max(args.years))
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    all_trials = []
    all_template_audit = []
    all_player_audit = []
    manifest: dict[str, Any] = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "buffers": [
            {"label": label, "dollars": value} for label, value in BUFFER_SPECS
        ],
        "fixed_settings": {
            "enforce_top_n": True,
            "waiver_source": CURRENT_WAIVER,
            "bench_upside_weight": CURRENT_BENCH_WEIGHT,
            "sampled_salary_cap": base.SALARY_CAP,
        },
        "base_replay": {
            "runner": str(BASE_RUNNER),
            "runner_sha256": base.sha256_file(BASE_RUNNER),
            "manifest": str(BASE_MANIFEST),
            "manifest_sha256": base.sha256_file(BASE_MANIFEST),
        },
        "simulation_helper": {
            "path": str(base.APP_HELPER),
            "sha256": base.sha256_file(base.APP_HELPER),
            "prior_replay_sha256": prior_manifest["simulation_helper"]["sha256"],
            "git_head": str(base.git_output(base.APP_ROOT, "rev-parse", "HEAD")),
        },
        "current_outcome_sources": current_outcome_hashes,
        "origins": {},
        "method_boundary": {
            "guardrail_estimand": "normalized point salary <= 298 + buffer",
            "historical_salary_laws": "reused from frozen base replay",
            "current_method_walk_forward": False,
            "target_outcome_source": "raw FastR_Beta weeks 1-16",
            "current_nomination_replay": False,
        },
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
        template_audit["max_donor_is_causal"] = template_audit.max_donor_season.lt(year)
        if not template_audit.max_donor_is_causal.all():
            raise AssertionError("Construction template pool crossed the replay origin.")
        all_template_audit.append(template_audit)

        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game", "salary"]
        ].copy()
        sim = base.make_simulation(year, player_data, cache)
        waiver_baseline = sim.estimate_waiver_baselines(
            num_teams=base.NUM_TEAMS,
            roster_size=base.ROSTER_SIZE,
        )

        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_full_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_full_idx].reset_index(drop=True)
        candidate_ppg = ppg_draws[candidate_full_idx]
        candidate_salary_draws = salary_draws[candidate_full_idx]
        predictions = base.build_predictions(candidate_forecast, candidate_ppg)
        raw_nominal = candidate_forecast.salary.to_numpy(dtype=float)
        remaining_budget = base.TOTAL_MARKET_BUDGET - environment["keeper_spend"]
        remaining_slots = base.TOTAL_MARKET_SLOTS - environment["keeper_count"]
        normalized_nominal = base.normalize_market_draws(
            sim,
            raw_nominal[:, None],
            remaining_budget,
            remaining_slots,
        )[:, 0]
        normalized_market_total = float(
            np.sort(normalized_nominal)[-remaining_slots:].sum()
        )
        if not math.isclose(
            normalized_market_total,
            remaining_budget,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                "Normalized nominal market does not match the keeper-adjusted budget."
            )

        player_audit = candidate_forecast[
            [
                "player",
                "player_key",
                "pos",
                "salary",
                "salary_source_matched",
            ]
        ].copy()
        player_audit = player_audit.rename(columns={"salary": "raw_nominal_salary"})
        player_audit["year"] = year
        player_audit["normalized_nominal_salary"] = normalized_nominal
        player_audit["actual_salary"] = outcome_labels.iloc[
            candidate_full_idx
        ].actual_salary.to_numpy()
        player_audit["actual_salary_matched"] = outcome_labels.iloc[
            candidate_full_idx
        ].actual_salary_matched.to_numpy()
        all_player_audit.append(player_audit)

        print(
            f"{year}: {len(predictions)} selectable players; building "
            f"{args.contexts} construction and evaluation contexts...",
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
        value_banks = base.managed_value_banks(
            weekly,
            decisions,
            played,
            predictions,
            {CURRENT_WAIVER: waiver_baseline},
        )
        trials, run_meta = run_buffer_trials(
            year,
            sim,
            predictions,
            candidate_salary_draws,
            raw_nominal,
            normalized_nominal,
            candidate_forecast.salary_source_matched.to_numpy(dtype=bool),
            environment,
            weekly,
            decisions,
            played,
            evaluation_weekly,
            evaluation_decisions,
            evaluation_played,
            value_banks[(CURRENT_WAIVER, CURRENT_BENCH_WEIGHT)],
            waiver_baseline,
            args.trials,
            args.context_draws,
            args.seed,
        )
        all_trials.append(trials)

        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(run_meta)
        source_manifest.update(
            {
                "keeper_count": environment["keeper_count"],
                "keeper_spend": environment["keeper_spend"],
                "projected_waiver_baseline": waiver_baseline,
                "raw_nominal_market_top_slot_sum": float(
                    np.sort(raw_nominal)[-remaining_slots:].sum()
                ),
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        manifest["origins"][str(year)] = source_manifest
        print(
            f"{year}: complete in {time.perf_counter() - year_started:.1f}s.",
            flush=True,
        )

    trials = pd.concat(all_trials, ignore_index=True)
    template_audit = pd.concat(all_template_audit, ignore_index=True)
    player_audit = pd.concat(all_player_audit, ignore_index=True)
    expected_rows = len(args.years) * args.trials * 12
    if len(trials) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} cells, found {len(trials)}.")
    key = ["year", "trial", "salary_draw_count", "nominal_buffer"]
    if trials.duplicated(key).any():
        raise AssertionError("Replay contains duplicate paired cells.")
    if not trials.groupby(["year", "trial"]).size().eq(12).all():
        raise AssertionError("A trial does not contain all 12 paired cells.")
    if not trials.solve_status.eq("optimal").all():
        failures = trials.loc[
            trials.solve_status.ne("optimal"),
            ["year", "trial", "variant"],
        ]
        raise AssertionError(f"Buffer replay has infeasible cells: {failures.head().to_dict('records')}")
    trial_invariants = validate_trial_invariants(trials)
    if (trials.forecast_salary_spend > base.SALARY_CAP + 1e-4).any():
        raise AssertionError("A buffer roster exceeds the sampled salary cap.")
    constrained = trials.nominal_buffer.ne(CONTROL_LABEL)
    if (
        trials.loc[constrained, "normalized_nominal_salary_spend"]
        > trials.loc[constrained, "nominal_cap"] + 1e-4
    ).any():
        raise AssertionError("A refined buffer roster exceeds its nominal cap.")
    if (~trials.contains_top_n).any():
        raise AssertionError("A buffer roster violates the Top-N constraint.")
    if not template_audit.max_donor_is_causal.all():
        raise AssertionError("A template donor crossed its replay origin.")

    prior_controls = validate_prior_controls(trials, args)
    variant_by_year, variant_across, variant_periods = build_variant_summaries(trials)
    variant_periods = add_development_pareto(variant_periods)
    buffer_pairs, draw_pairs, interactions = build_pair_tables(trials)
    buffer_by_year, buffer_across, buffer_split = summarize_pairs(buffer_pairs)
    draw_by_year, draw_across, draw_split = summarize_pairs(draw_pairs)
    interaction_by_year, interaction_across = summarize_interactions(interactions)

    outputs = {
        "roster_trials.csv": trials,
        "variant_summary_by_year.csv": variant_by_year,
        "variant_summary_across_years.csv": variant_across,
        "variant_summary_development_check.csv": variant_periods,
        "paired_buffer_effects.csv": buffer_pairs,
        "buffer_effects_by_year.csv": buffer_by_year,
        "buffer_effects_across_years.csv": buffer_across,
        "buffer_effects_split_half.csv": buffer_split,
        "paired_draw_effects.csv": draw_pairs,
        "draw_effects_by_year.csv": draw_by_year,
        "draw_effects_across_years.csv": draw_across,
        "draw_effects_split_half.csv": draw_split,
        "paired_buffer_draw_interactions.csv": interactions,
        "buffer_draw_interactions_by_year.csv": interaction_by_year,
        "buffer_draw_interactions_across_years.csv": interaction_across,
        "nominal_salary_player_audit.csv": player_audit,
        "template_pool_audit.csv": template_audit,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    validation = {
        "expected_rows": expected_rows,
        "all_cells_present": True,
        "all_solves_optimal": True,
        "all_sampled_spend_within_cap": True,
        "all_constrained_nominal_spend_within_cap": True,
        "all_top_n_constraints_satisfied": True,
        "all_template_donors_pre_origin": True,
        "all_nominal_markets_match_keeper_adjusted_budget": True,
        **trial_invariants,
        "prior_controls": prior_controls,
    }
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["validation"] = validation
    manifest["output_rows"] = {
        filename: int(len(frame)) for filename, frame in outputs.items()
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_summary(
        output_dir,
        args,
        variant_periods,
        buffer_across,
        draw_across,
        interaction_across,
        validation,
    )
    print(
        f"\nBuffer replay complete in {time.perf_counter() - started:.1f}s. "
        f"Results: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
