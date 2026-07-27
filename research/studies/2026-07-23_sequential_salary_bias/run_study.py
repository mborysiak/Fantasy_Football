"""Paired rolling study of salary bias under static and blind-sequential selection.

The static arm sees an entire sampled salary surface before choosing a roster.
The blind arm uses the current Sequential Target completion policy and sees a
player's historical-price replay only when that player is nominated.
"""

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
APP_ROOT = ROOT.parent / "Fantasy_Football_App"
APP_TARGET = APP_ROOT / "app" / "zSequential_Target.py"
SURCHARGE_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-16_optimizer_selection_surcharge"
    / "run_replay.py"
)

YEARS = (2022, 2023, 2024, 2025)
VARIANTS = (
    ("static_full_surface", "none"),
    ("static_full_surface", "half"),
    ("blind_sequential", "none"),
    ("blind_sequential", "half"),
)
COMPARISONS = (
    ("static_half_minus_none", ("static_full_surface", "half"), ("static_full_surface", "none")),
    ("blind_half_minus_none", ("blind_sequential", "half"), ("blind_sequential", "none")),
    ("blind_none_minus_static_none", ("blind_sequential", "none"), ("static_full_surface", "none")),
    ("blind_half_minus_static_half", ("blind_sequential", "half"), ("static_full_surface", "half")),
)
SUMMARY_METRICS = (
    "complete",
    "sampled_base_spend",
    "scenario_discount_point_minus_sampled",
    "plan_actual_salary_spend",
    "plan_point_salary_spend",
    "plan_actual_minus_point",
    "plan_actual_cap_feasible",
    "plan_recorded_salary_rate",
    "plan_recorded_actual_minus_point",
    "final_actual_salary_spend",
    "final_point_salary_spend",
    "final_actual_minus_point",
    "final_actual_cap_feasible",
    "final_recorded_salary_rate",
    "final_recorded_actual_minus_point",
    "paid_or_decision_spend",
    "unused_paid_budget",
    "actual_points",
    "forecast_ev",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


surcharge = load_module(SURCHARGE_RUNNER, "_sequential_salary_bias_surcharge")
base = surcharge.base
target = load_module(APP_TARGET, "_sequential_salary_bias_target")


def roster_salary_metrics(
    roster: tuple[str, ...],
    predictions: pd.DataFrame,
    point_salary: np.ndarray,
    replay_prices: np.ndarray,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    mask = predictions.player.isin(roster).to_numpy()
    if int(mask.sum()) != len(roster):
        missing = sorted(set(roster) - set(predictions.loc[mask, "player"]))
        raise ValueError(f"Roster players missing from predictions: {missing}")
    actual = labels.actual_salary.to_numpy(dtype=float)[mask]
    matched = labels.actual_salary_matched.to_numpy(dtype=bool)[mask]
    point = np.asarray(point_salary, dtype=float)[mask]
    replay = np.asarray(replay_prices, dtype=float)[mask]
    return {
        "roster": "|".join(sorted(roster)),
        "roster_size": int(len(roster)),
        "actual_salary_spend": float(actual.sum()),
        "point_salary_spend": float(point.sum()),
        "actual_minus_point": float((actual - point).sum()),
        "replay_clearing_spend": float(replay.sum()),
        "actual_cap_feasible": bool(actual.sum() <= base.SALARY_CAP + 1e-8),
        "recorded_salary_players": int(matched.sum()),
        "recorded_salary_rate": float(matched.mean()) if len(matched) else np.nan,
        "recorded_actual_minus_point": (
            float((actual[matched] - point[matched]).sum()) if matched.any() else np.nan
        ),
    }


def add_prefixed(row: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    row.update({f"{prefix}_{key}": value for key, value in values.items()})


def solve_static(
    sim: Any,
    predictions: pd.DataFrame,
    objective: np.ndarray,
    decision_market: np.ndarray,
    reference_weekly: np.ndarray,
    reference_decisions: np.ndarray,
    reference_played: np.ndarray,
    static: Any,
    top_n: list[str],
    waiver_baseline: dict[str, float],
) -> dict[str, Any] | None:
    predictions["salary"] = decision_market
    return sim._solve_managed_scenario(
        predictions,
        objective,
        reference_weekly,
        reference_decisions,
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
        salary_values=decision_market,
        played_mask=reference_played,
    )


def summarize_arms(trials: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "development_2022_2024": (2022, 2023, 2024),
        "check_2025": (2025,),
        "all_2022_2025": YEARS,
    }
    rows: list[dict[str, Any]] = []
    for period, years in periods.items():
        subset = trials[trials.year.isin(years)]
        for (mode, reserve), group in subset.groupby(
            ["information_mode", "reserve"], sort=True
        ):
            row: dict[str, Any] = {
                "period": period,
                "information_mode": mode,
                "reserve": reserve,
                "trials": int(len(group)),
                "origins": int(group.year.nunique()),
            }
            for metric in SUMMARY_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(values.mean())
                origin_means = group.assign(_value=values).groupby("year")._value.mean()
                row[f"equal_origin_mean_{metric}"] = float(origin_means.mean())
                row[f"equal_origin_se_{metric}"] = (
                    float(origin_means.std(ddof=1) / math.sqrt(len(origin_means)))
                    if len(origin_means) > 1
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "plan_actual_minus_point",
        "plan_actual_salary_spend",
        "plan_actual_cap_feasible",
        "final_actual_minus_point",
        "final_actual_salary_spend",
        "final_actual_cap_feasible",
        "complete",
        "actual_points",
        "forecast_ev",
    ]
    indexed = {
        key: trials[
            trials.information_mode.eq(key[0]) & trials.reserve.eq(key[1])
        ][["year", "trial", *metrics]].copy()
        for _, key, _ in COMPARISONS
    }
    for _, _, key in COMPARISONS:
        if key not in indexed:
            indexed[key] = trials[
                trials.information_mode.eq(key[0]) & trials.reserve.eq(key[1])
            ][["year", "trial", *metrics]].copy()
    rows: list[dict[str, Any]] = []
    periods = {
        "development_2022_2024": (2022, 2023, 2024),
        "check_2025": (2025,),
        "all_2022_2025": YEARS,
    }
    for label, candidate_key, reference_key in COMPARISONS:
        merged = indexed[candidate_key].merge(
            indexed[reference_key],
            on=["year", "trial"],
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        for period, years in periods.items():
            group = merged[merged.year.isin(years)]
            row: dict[str, Any] = {
                "comparison": label,
                "period": period,
                "pairs": int(len(group)),
                "origins": int(group.year.nunique()),
            }
            for metric in metrics:
                difference = (
                    pd.to_numeric(
                        group[f"{metric}_candidate"], errors="coerce"
                    ).astype(float)
                    - pd.to_numeric(
                        group[f"{metric}_reference"], errors="coerce"
                    ).astype(float)
                )
                row[f"mean_diff_{metric}"] = float(difference.mean())
                origin_means = (
                    group.assign(_difference=difference)
                    .groupby("year")
                    ._difference.mean()
                )
                row[f"equal_origin_mean_diff_{metric}"] = float(origin_means.mean())
                row[f"equal_origin_se_diff_{metric}"] = (
                    float(origin_means.std(ddof=1) / math.sqrt(len(origin_means)))
                    if len(origin_means) > 1
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def player_selection_summary(
    selections: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    denominators = (
        trials.groupby(["year", "information_mode", "reserve"])
        .size()
        .rename("available_trials")
        .reset_index()
    )
    output = (
        selections.groupby(
            [
                "year",
                "information_mode",
                "reserve",
                "phase",
                "player",
                "player_key",
                "pos",
            ],
            as_index=False,
        )
        .agg(
            selections=("player", "size"),
            point_salary=("point_salary", "first"),
            actual_salary=("actual_salary", "first"),
            actual_minus_point=("actual_minus_point", "first"),
            actual_salary_matched=("actual_salary_matched", "first"),
        )
        .merge(
            denominators,
            on=["year", "information_mode", "reserve"],
            how="left",
            validate="many_to_one",
        )
    )
    output["selection_rate"] = output.selections / output.available_trials
    return output.sort_values(
        ["year", "information_mode", "reserve", "phase", "selection_rate", "player"],
        ascending=[True, True, True, True, False, True],
    ).reset_index(drop=True)


def selection_rows(
    year: int,
    trial: int,
    mode: str,
    reserve: str,
    phase: str,
    roster: tuple[str, ...],
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    point_salary: np.ndarray,
) -> list[dict[str, Any]]:
    frame = predictions[["player", "pos"]].copy()
    frame["player_key"] = labels.player_key.to_numpy()
    frame["actual_salary"] = labels.actual_salary.to_numpy(dtype=float)
    frame["actual_salary_matched"] = labels.actual_salary_matched.to_numpy(dtype=bool)
    frame["point_salary"] = point_salary
    frame = frame[frame.player.isin(roster)]
    rows = []
    for player in frame.itertuples(index=False):
        rows.append(
            {
                "year": year,
                "trial": trial,
                "information_mode": mode,
                "reserve": reserve,
                "phase": phase,
                "player": player.player,
                "player_key": player.player_key,
                "pos": player.pos,
                "point_salary": float(player.point_salary),
                "actual_salary": float(player.actual_salary),
                "actual_minus_point": float(
                    player.actual_salary - player.point_salary
                ),
                "actual_salary_matched": bool(player.actual_salary_matched),
            }
        )
    return rows


def write_readout(
    output_dir: Path,
    arm_summary: pd.DataFrame,
    pairs: pd.DataFrame,
    trials: pd.DataFrame,
) -> None:
    def value(frame: pd.DataFrame, mode: str, reserve: str, column: str) -> float:
        row = frame[
            frame.information_mode.eq(mode) & frame.reserve.eq(reserve)
        ]
        return float(row.iloc[0][column])

    overall = arm_summary[arm_summary.period.eq("all_2022_2025")]
    static_gap = value(
        overall,
        "static_full_surface",
        "none",
        "equal_origin_mean_plan_actual_minus_point",
    )
    static_sampled = value(
        overall,
        "static_full_surface",
        "none",
        "equal_origin_mean_sampled_base_spend",
    )
    static_point = value(
        overall,
        "static_full_surface",
        "none",
        "equal_origin_mean_plan_point_salary_spend",
    )
    static_actual = value(
        overall,
        "static_full_surface",
        "none",
        "equal_origin_mean_plan_actual_salary_spend",
    )
    static_scenario_discount = value(
        overall,
        "static_full_surface",
        "none",
        "equal_origin_mean_scenario_discount_point_minus_sampled",
    )
    blind_plan_gap = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_plan_actual_minus_point",
    )
    blind_final_gap = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_final_actual_minus_point",
    )
    blind_plan_point = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_plan_point_salary_spend",
    )
    blind_plan_actual = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_plan_actual_salary_spend",
    )
    blind_plan_feasible = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_plan_actual_cap_feasible",
    )
    blind_paid = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_paid_or_decision_spend",
    )
    blind_unused = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_unused_paid_budget",
    )
    half_plan_actual = value(
        overall,
        "blind_sequential",
        "half",
        "equal_origin_mean_plan_actual_salary_spend",
    )
    half_plan_feasible = value(
        overall,
        "blind_sequential",
        "half",
        "equal_origin_mean_plan_actual_cap_feasible",
    )
    half_complete = value(
        overall,
        "blind_sequential",
        "half",
        "equal_origin_mean_complete",
    )
    static_feasible = value(
        overall,
        "static_full_surface",
        "none",
        "equal_origin_mean_final_actual_cap_feasible",
    )
    blind_complete = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_complete",
    )
    blind_feasible = value(
        overall,
        "blind_sequential",
        "none",
        "equal_origin_mean_final_actual_cap_feasible",
    )
    lines = [
        "# Blind sequential salary-bias replay",
        "",
        "This rolling replay keeps the construction draw and nomination order paired "
        "within each origin. The static arm sees its entire sampled salary surface; "
        "the Sequential Target arm sees replay prices only as nominations occur.",
        "",
        "## Headline",
        "",
        "**The oracle/scenario-shopping component largely disappears, but the "
        "player-level residual bias does not.**",
        "",
        f"- Static full-surface selection spent **${static_sampled:.1f}** on the sampled surface, "
        f"but those players cost **${static_point:.1f}** at point prices and **${static_actual:.1f}** historically. "
        f"That is a **${static_scenario_discount:.1f}** scenario-shopping discount plus a "
        f"**${static_gap:.1f}** actual-minus-point residual.",
        f"- Blind Sequential Target's initial plan cost **${blind_plan_point:.1f}** at point prices and "
        f"**${blind_plan_actual:.1f}** historically: a **${blind_plan_gap:.1f}** residual. "
        "That residual is essentially the same size as the static arm, but the plan is "
        "not built around an unusually cheap full salary draw.",
        f"- Consequently, initial-plan historical-cap feasibility improved from "
        f"**{100 * static_feasible:.1f}%** static to **{100 * blind_plan_feasible:.1f}%** blind.",
        f"- With the half reserve, the blind initial plan fell to **${half_plan_actual:.1f}** "
        f"historical spend and **{100 * half_plan_feasible:.1f}%** feasibility.",
        f"- After live recourse, blind no-reserve completion was **{100 * blind_complete:.1f}%** "
        f"and every completed roster was legal, but it paid only **${blind_paid:.1f}** and "
        f"left **${blind_unused:.1f}** unused on average. Half-reserve completion was "
        f"**{100 * half_complete:.1f}%**.",
        "",
        "The initial-plan comparison is the clean test of selection concentration. "
        "The acquired-roster comparison includes the benefit of observing prices and "
        "pivoting during the auction. A legal completed sequential roster is also "
        "audited against its actual paid p+1 spend. The acquired-roster gap of "
        f"**${blind_final_gap:.1f}** therefore reflects genuine recourse, but the large "
        "unused budget says the current replay policy overcorrects and should not yet "
        "be read as an optimal spending policy.",
        "",
        "## Design limits",
        "",
        "- Historical nomination order and losing bids are unavailable; orders are current production-style noisy salary orderings.",
        "- Historical clearing prices are treated as exogenous. The policy does not model opponents reacting to our purchases.",
        "- Missing historical salaries retain the established $1 fallback and are reported through recorded-salary coverage.",
        "- Four season origins are the independent time units; trial variation is paired Monte Carlo precision, not four dozen independent seasons.",
        "",
        f"Rows evaluated: {len(trials):,}. See `arm_summary.csv`, `paired_effects.csv`, and `player_selection_rates.csv` for the full decomposition.",
        "",
    ]
    (output_dir / "readout.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--contexts", type=int, default=48)
    parser.add_argument("--context-draws", type=int, default=8)
    parser.add_argument("--projection-draws", type=int, default=256)
    parser.add_argument("--salary-draws", type=int, default=256)
    parser.add_argument("--nomination-noise", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", default=str(STUDY_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw_weekly = base.load_raw_weekly(max_year=max(args.years))
    features = base.load_feature_templates()
    actual = base.load_actual_salaries()
    salary_rows, source_rows = surcharge.current.load_salary_tables()
    diagnostic = surcharge.load_selection_diagnostic()
    all_trials: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_calibration: list[pd.DataFrame] = []
    all_template_audit: list[pd.DataFrame] = []
    origin_manifest: dict[str, Any] = {}

    for year in args.years:
        year_started = time.perf_counter()
        print(f"\n=== Origin {year} ===", flush=True)
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
            raise AssertionError("Construction templates crossed the replay origin.")
        all_template_audit.append(template_audit)

        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game"]
        ].copy()
        # The current v5 salary surface is built immediately below.  The
        # simulation constructor only needs a valid placeholder at this point;
        # every study solve receives the aligned v5 price vector explicitly.
        player_data["salary"] = 1.0
        sim = base.make_simulation(year, player_data, cache)
        waiver_baseline = sim.estimate_waiver_baselines(
            num_teams=base.NUM_TEAMS,
            roster_size=base.ROSTER_SIZE,
        )
        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_idx].reset_index(drop=True)
        candidate_forecast["salary"] = 1.0
        candidate_labels = outcome_labels.iloc[candidate_idx].reset_index(drop=True)
        candidate_ppg = ppg_draws[candidate_idx]
        predictions = base.build_predictions(candidate_forecast, candidate_ppg)
        remaining_budget = base.TOTAL_MARKET_BUDGET - environment["keeper_spend"]
        remaining_slots = base.TOTAL_MARKET_SLOTS - environment["keeper_count"]

        surface, salary_draws, salary_meta = surcharge.current.build_salary_surface(
            year,
            candidate_forecast,
            salary_rows,
            source_rows,
            sim,
            remaining_budget,
            remaining_slots,
            args.salary_draws,
            args.seed,
        )
        if not np.array_equal(surface.player.to_numpy(), predictions.player.to_numpy()):
            raise AssertionError("Salary surface and predictions lost row alignment.")
        calibration, coefficients, calibration_meta = surcharge.fit_origin_surcharge(
            year,
            surface,
            diagnostic,
        )
        calibration["year"] = year
        all_calibration.append(calibration)
        point_salary = surface.point_salary.to_numpy(dtype=float)
        premium_half = calibration.surcharge_half.to_numpy(dtype=float)
        predictions["salary"] = point_salary

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
            {"current_projected": waiver_baseline},
        )
        managed_values = value_banks[("current_projected", 0.25)]

        raw_actual_prices = candidate_labels.actual_salary.to_numpy(dtype=float)
        replay_prices = sim.normalize_salary_market_values(
            raw_actual_prices,
            np.ones(len(predictions), dtype=bool),
            remaining_market_budget=remaining_budget,
            remaining_market_slots=remaining_slots,
        )
        replay_prices = np.maximum(1, np.rint(replay_prices)).astype(np.float64)

        rng = np.random.default_rng(args.seed + year * 101)
        salary_plan = rng.integers(
            0,
            salary_draws.shape[1],
            size=(args.trials, 5),
        )
        context_plan = rng.integers(
            0,
            args.contexts,
            size=(args.trials, args.context_draws),
        )
        raw_markets = np.column_stack(
            [
                salary_draws[:, indices].mean(axis=1)
                for indices in salary_plan
            ]
        )
        sampled_markets = base.normalize_market_draws(
            sim,
            raw_markets,
            remaining_budget,
            remaining_slots,
        )
        orders = target.noisy_salary_orders(
            point_salary,
            args.trials,
            random_seed=args.seed + year * 10_000,
            noise=args.nomination_noise,
        )
        top_n = (
            pd.DataFrame({"player": predictions.player, "salary": point_salary})
            .sort_values(["salary", "player"], ascending=[False, True])
            .head(min(base.TOP_N, len(predictions)))
            .player.tolist()
        )
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
        reference_weekly = weekly.mean(axis=0)
        reference_decisions = decisions.mean(axis=0)
        reference_played = np.any(played > 0, axis=0).astype(np.int8)
        forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}

        for trial in range(args.trials):
            objective = managed_values[:, context_plan[trial]].mean(axis=1)
            sampled_market = sampled_markets[:, trial]
            for mode, reserve in VARIANTS:
                premium = (
                    np.zeros_like(point_salary)
                    if reserve == "none"
                    else premium_half
                )
                row: dict[str, Any] = {
                    "year": year,
                    "trial": trial,
                    "information_mode": mode,
                    "reserve": reserve,
                    "selection_premium_mean": float(premium.mean()),
                    "selection_premium_max": float(premium.max()),
                }

                if mode == "static_full_surface":
                    decision_market = sampled_market + premium
                    solved = solve_static(
                        sim,
                        predictions,
                        objective,
                        decision_market,
                        reference_weekly,
                        reference_decisions,
                        reference_played,
                        static,
                        top_n,
                        waiver_baseline,
                    )
                    if solved is None:
                        row.update(
                            {
                                "status": "infeasible",
                                "complete": False,
                            }
                        )
                        all_trials.append(row)
                        continue
                    selected = np.asarray(solved["selected_mask"], dtype=bool)
                    roster = tuple(sorted(solved["selected_players"]))
                    plan_metrics = roster_salary_metrics(
                        roster,
                        predictions,
                        point_salary,
                        replay_prices,
                        candidate_labels,
                    )
                    add_prefixed(row, "plan", plan_metrics)
                    add_prefixed(row, "final", plan_metrics)
                    row.update(
                        {
                            "status": "optimal",
                            "complete": True,
                            "sampled_base_spend": float(sampled_market[selected].sum()),
                            "paid_or_decision_spend": float(decision_market[selected].sum()),
                            "unused_paid_budget": float(
                                base.SALARY_CAP - decision_market[selected].sum()
                            ),
                            "scenario_discount_point_minus_sampled": float(
                                point_salary[selected].sum()
                                - sampled_market[selected].sum()
                            ),
                            "replans": 0,
                        }
                    )
                else:
                    predictions["salary"] = point_salary
                    plan = target.solve_history_only_plan(
                        sim,
                        predictions,
                        objective,
                        point_salary,
                        premium,
                        {},
                        set(predictions.player),
                        base.ROSTER_SIZE,
                        base.POS_MIN,
                        base.POS_MAX,
                        base.TOP_N,
                        True,
                    )
                    if plan is None:
                        row.update(
                            {
                                "status": "initial_plan_infeasible",
                                "complete": False,
                            }
                        )
                        all_trials.append(row)
                        continue
                    plan_roster = tuple(sorted(plan["selected"]))
                    add_prefixed(
                        row,
                        "plan",
                        roster_salary_metrics(
                            plan_roster,
                            predictions,
                            point_salary,
                            replay_prices,
                            candidate_labels,
                        ),
                    )
                    policy_scores = dict(
                        zip(
                            predictions.player,
                            objective
                            / np.sqrt(np.maximum(point_salary + premium, 1.0)),
                        )
                    )
                    result = target.simulate_history_only_branch(
                        sim=sim,
                        predictions=predictions,
                        managed_values=objective,
                        base_prices=point_salary,
                        selection_premiums=premium,
                        initial_salary_map={},
                        candidate=None,
                        candidate_price=None,
                        force_buy=False,
                        order=orders[trial],
                        revealed_prices=replay_prices,
                        remaining_market_budget=remaining_budget,
                        remaining_market_slots=remaining_slots,
                        roster_size=base.ROSTER_SIZE,
                        pos_min_counts=base.POS_MIN,
                        pos_max_counts=base.POS_MAX,
                        require_top_n=base.TOP_N,
                        enforce_top_n=True,
                        compiled_plan=plan,
                        policy_scores=policy_scores,
                    )
                    row.update(
                        {
                            "status": (
                                "complete"
                                if result["complete"]
                                else f"failed:{result.get('failure_reason')}"
                            ),
                            "complete": bool(result["complete"]),
                            "sampled_base_spend": np.nan,
                            "paid_or_decision_spend": float(
                                result.get("salary_spend", np.nan)
                            ),
                            "unused_paid_budget": float(
                                base.SALARY_CAP
                                - result.get("salary_spend", np.nan)
                            ),
                            "scenario_discount_point_minus_sampled": np.nan,
                            "replans": int(result.get("replans", 0)),
                            "events_seen": int(result.get("events_seen", 0)),
                        }
                    )
                    roster = tuple(result["roster"])
                    if result["complete"]:
                        add_prefixed(
                            row,
                            "final",
                            roster_salary_metrics(
                                roster,
                                predictions,
                                point_salary,
                                replay_prices,
                                candidate_labels,
                            ),
                        )
                    else:
                        for key in roster_salary_metrics(
                            roster,
                            predictions,
                            point_salary,
                            replay_prices,
                            candidate_labels,
                        ):
                            row[f"final_{key}"] = np.nan

                if row["complete"]:
                    actual_score = base.score_actual_roster(environment, roster)
                    row["actual_points"] = float(actual_score["actual_points"])
                    row["forecast_ev"] = base.forecast_roster_ev(
                        roster,
                        "current_projected",
                        waiver_baseline,
                        predictions,
                        evaluation_weekly,
                        evaluation_decisions,
                        evaluation_played,
                        forecast_cache,
                    )
                    all_selections.extend(
                        selection_rows(
                            year,
                            trial,
                            mode,
                            reserve,
                            "plan",
                            plan_roster if mode == "blind_sequential" else roster,
                            predictions,
                            candidate_labels,
                            point_salary,
                        )
                    )
                    all_selections.extend(
                        selection_rows(
                            year,
                            trial,
                            mode,
                            reserve,
                            "final",
                            roster,
                            predictions,
                            candidate_labels,
                            point_salary,
                        )
                    )
                else:
                    row["actual_points"] = np.nan
                    row["forecast_ev"] = np.nan
                all_trials.append(row)

            if (trial + 1) % max(1, min(8, args.trials)) == 0:
                print(f"{year}: completed {trial + 1}/{args.trials} paired trials", flush=True)

        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(calibration_meta)
        source_manifest.update(
            {
                "runtime_seconds": time.perf_counter() - year_started,
                "keeper_count": environment["keeper_count"],
                "keeper_spend": environment["keeper_spend"],
                "remaining_market_budget": remaining_budget,
                "remaining_market_slots": remaining_slots,
                "replay_price_top_slot_total": float(
                    np.sort(replay_prices)[-remaining_slots:].sum()
                ),
            }
        )
        origin_manifest[str(year)] = source_manifest
        print(
            f"{year}: complete in {time.perf_counter() - year_started:.1f}s",
            flush=True,
        )

    trials = pd.DataFrame(all_trials)
    selections = pd.DataFrame(all_selections)
    expected = len(args.years) * args.trials * len(VARIANTS)
    if len(trials) != expected:
        raise AssertionError(f"Expected {expected} trial rows; found {len(trials)}.")
    if trials.duplicated(["year", "trial", "information_mode", "reserve"]).any():
        raise AssertionError("Duplicate paired trial cells.")
    completed_blind = trials[
        trials.information_mode.eq("blind_sequential") & trials.complete
    ]
    if (completed_blind.paid_or_decision_spend > base.SALARY_CAP + 1e-8).any():
        raise AssertionError("A completed blind path exceeded the paid salary cap.")

    arm_summary = summarize_arms(trials)
    pairs = paired_effects(trials)
    player_summary = player_selection_summary(selections, trials)
    calibration = pd.concat(all_calibration, ignore_index=True)
    template_audit = pd.concat(all_template_audit, ignore_index=True)
    trials.to_csv(output_dir / "trial_results.csv", index=False)
    arm_summary.to_csv(output_dir / "arm_summary.csv", index=False)
    pairs.to_csv(output_dir / "paired_effects.csv", index=False)
    player_summary.to_csv(output_dir / "player_selection_rates.csv", index=False)
    calibration.to_csv(output_dir / "selection_premium_calibration.csv", index=False)
    template_audit.to_csv(output_dir / "template_pool_audit.csv", index=False)

    manifest = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "runtime_seconds": time.perf_counter() - started,
        "origins": origin_manifest,
        "method_boundary": {
            "static_future_salary_surface_visible": True,
            "blind_future_replay_prices_visible": False,
            "blind_current_nominee_price_visible": True,
            "historical_nomination_order_available": False,
            "nomination_order": "production noisy salary order",
            "losing_bids_available": False,
            "opponent_response_modeled": False,
            "blind_payment_rule": "recorded-price replay plus one dollar",
            "selection_reserve": "strictly prior-origin half surcharge",
            "target_season_points_visible_to_policy": False,
            "target_season_salaries_visible_to_static_policy": False,
            "target_season_salaries_visible_to_blind_policy": "only at nomination",
        },
        "validation": {
            "expected_cells": expected,
            "actual_cells": int(len(trials)),
            "unique_paired_cells": True,
            "all_template_donors_pre_origin": bool(
                template_audit.max_donor_is_causal.all()
            ),
            "completed_blind_paths_within_paid_cap": True,
        },
        "sources": {
            "target_runner": str(APP_TARGET),
            "target_runner_sha256": base.sha256_file(APP_TARGET),
            "surcharge_runner": str(SURCHARGE_RUNNER),
            "surcharge_runner_sha256": base.sha256_file(SURCHARGE_RUNNER),
        },
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_readout(output_dir, arm_summary, pairs, trials)
    print(
        f"\nStudy complete in {time.perf_counter() - started:.1f}s: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
