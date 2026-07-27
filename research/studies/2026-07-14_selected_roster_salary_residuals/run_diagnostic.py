"""Diagnose salary residual concentration in optimizer-selected rosters."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
CHANCE_STUDY = (
    ROOT / "research" / "studies" / "2026-07-14_salary_chance_frontier"
)
CHANCE_RESULTS = CHANCE_STUDY / "results"
CURRENT_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_current_salary_buffer_replay"
    / "run_replay.py"
)
RESULTS = STUDY_DIR / "results"
PRICE_TIER_ORDER = ["$1-5", "$6-15", "$16-30", "$31-50", "$51+"]
SELECTION_BUCKET_ORDER = ["never", "rare_0-5%", "occasional_5-25%", "frequent_25-50%", "core_>50%"]


def load_current_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_selected_residual_current_runner", CURRENT_RUNNER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import replay helper: {CURRENT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


current = load_current_runner()
base = current.base


def cohort_row(
    data: pd.DataFrame,
    cohort: str,
    period: str,
    weights: pd.Series | None = None,
) -> dict[str, Any]:
    values = data.salary_residual.to_numpy(dtype=float)
    if weights is None:
        weight_values = np.ones(len(data), dtype=float)
        observations = int(len(data))
    else:
        weight_values = weights.loc[data.index].to_numpy(dtype=float)
        observations = int(weight_values.sum())
    if len(data) == 0 or weight_values.sum() <= 0:
        return {
            "period": period,
            "cohort": cohort,
            "player_origins": int(len(data)),
            "weighted_observations": observations,
            "mean_salary_residual": np.nan,
            "median_salary_residual_unique": np.nan,
            "positive_residual_rate": np.nan,
            "mean_predicted_salary": np.nan,
            "mean_actual_salary": np.nan,
            "mean_selection_rate": np.nan,
            "mean_scenario_center_shift": np.nan,
            "mean_actual_minus_scenario": np.nan,
        }
    return {
        "period": period,
        "cohort": cohort,
        "player_origins": int(len(data)),
        "weighted_observations": observations,
        "mean_salary_residual": float(np.average(values, weights=weight_values)),
        "median_salary_residual_unique": float(np.median(values)),
        "positive_residual_rate": float(
            np.average(values > 0, weights=weight_values)
        ),
        "mean_predicted_salary": float(
            np.average(data.point_salary, weights=weight_values)
        ),
        "mean_actual_salary": float(
            np.average(data.actual_salary, weights=weight_values)
        ),
        "mean_selection_rate": float(
            np.average(data.selection_rate, weights=weight_values)
        ),
        "mean_scenario_center_shift": float(
            np.average(data.scenario_center_shift, weights=weight_values)
        ),
        "mean_actual_minus_scenario": float(
            np.average(data.actual_minus_scenario, weights=weight_values)
        ),
    }


def grouped_slot_summary(
    slots: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in slots.groupby(group_columns, observed=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "selected_slots": int(len(group)),
                "player_origins": int(
                    group[["year", "player_key"]].drop_duplicates().shape[0]
                ),
                "mean_salary_residual": float(group.salary_residual.mean()),
                "median_salary_residual": float(group.salary_residual.median()),
                "positive_residual_rate": float(group.salary_residual.gt(0).mean()),
                "mean_predicted_salary": float(group.point_salary.mean()),
                "mean_actual_salary": float(group.actual_salary.mean()),
                "mean_scenario_center_shift": float(
                    group.scenario_center_shift.mean()
                ),
                "mean_actual_minus_scenario": float(
                    group.actual_minus_scenario.mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_scenario_market_means(candidates: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the full normalized five-draw market bank from the replay."""
    manifest = json.loads(
        (CHANCE_RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    config = manifest["config"]
    num_draws = int(config["salary_draws"])
    seed = int(config["seed"])
    draw_count = 5
    if num_draws % draw_count:
        raise ValueError("Chance replay salary draws do not divide into five-draw markets.")
    output = candidates.copy()
    output["scenario_market_mean"] = np.nan
    for year, group in output.groupby("year", sort=True):
        residual_quantiles = group[current.RESID_COLS].to_numpy(dtype=float)
        salary_draws = current.sample_residual_quantiles(
            group.point_salary.to_numpy(dtype=float),
            residual_quantiles,
            num_draws,
            seed + int(year) * 37,
        ).astype(np.float64)
        raw_markets = salary_draws.reshape(
            len(group), num_draws // draw_count, draw_count
        ).mean(axis=2)
        origin = manifest["origins"][str(int(year))]
        available = np.ones(len(group), dtype=bool)
        normalized = np.column_stack(
            [
                base.FootballSimulation.normalize_salary_market_values(
                    raw_markets[:, market_idx],
                    available,
                    remaining_market_budget=float(origin["remaining_budget"]),
                    remaining_market_slots=int(origin["remaining_slots"]),
                )
                for market_idx in range(raw_markets.shape[1])
            ]
        )
        top_count = int(origin["remaining_slots"])
        top_totals = np.partition(
            normalized,
            kth=len(group) - top_count,
            axis=0,
        )[-top_count:, :].sum(axis=0)
        if not np.allclose(
            top_totals,
            float(origin["remaining_budget"]),
            atol=1e-6,
        ):
            raise AssertionError("A reconstructed scenario market does not balance.")
        output.loc[group.index, "scenario_market_mean"] = normalized.mean(axis=1)
    output["scenario_center_shift"] = (
        output.scenario_market_mean - output.point_salary
    )
    output["actual_minus_scenario"] = (
        output.actual_salary - output.scenario_market_mean
    )
    return output


def build_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(CHANCE_RESULTS / "salary_surface_audit.csv")
    rosters = pd.read_csv(CHANCE_RESULTS / "roster_trials.csv")
    actual = base.load_actual_salaries()
    actual = (
        actual[actual.league.eq("beta")]
        .sort_values("actual_salary", ascending=False)
        .drop_duplicates(["year", "player_key"])
    )
    actual["actual_salary_recorded"] = True
    candidates = candidates.merge(
        actual[
            [
                "year",
                "player_key",
                "actual_salary",
                "is_keeper",
                "actual_salary_recorded",
            ]
        ],
        on=["year", "player_key"],
        how="left",
        validate="one_to_one",
    )
    candidates["actual_salary_recorded"] = candidates.actual_salary_recorded.eq(True)
    candidates["actual_salary_used_in_replay"] = candidates.actual_salary.fillna(1.0)
    candidates["salary_residual"] = (
        candidates.actual_salary - candidates.point_salary
    )
    candidates["replay_salary_residual"] = (
        candidates.actual_salary_used_in_replay - candidates.point_salary
    )
    candidates = add_scenario_market_means(candidates)

    slots = rosters[["year", "trial", "chance_level", "roster"]].copy()
    slots["player"] = slots.roster.str.split("|")
    slots = slots.explode("player", ignore_index=True)
    slots = base.add_identity(slots)
    selection = (
        slots.groupby(["year", "player_key"])
        .size()
        .rename("selection_slots")
        .reset_index()
    )
    per_level = (
        slots.groupby(["year", "player_key", "chance_level"])
        .size()
        .unstack("chance_level", fill_value=0)
    )
    per_level.columns = [f"selection_slots_chance_{int(level * 100)}" for level in per_level]
    per_level = per_level.reset_index()
    candidates = candidates.merge(
        selection, on=["year", "player_key"], how="left", validate="one_to_one"
    ).merge(
        per_level, on=["year", "player_key"], how="left", validate="one_to_one"
    )
    selection_columns = [column for column in candidates if column.startswith("selection_slots")]
    candidates[selection_columns] = candidates[selection_columns].fillna(0).astype(int)
    roster_counts = rosters.groupby("year").size().to_dict()
    candidates["selection_rate"] = candidates.apply(
        lambda row: row.selection_slots / roster_counts[int(row.year)], axis=1
    )

    candidates["selection_bucket"] = pd.cut(
        candidates.selection_rate,
        bins=[-1e-12, 0.0, 0.05, 0.25, 0.50, 1.000001],
        labels=SELECTION_BUCKET_ORDER,
        include_lowest=True,
    )
    candidates["predicted_salary_tier"] = pd.cut(
        candidates.point_salary,
        bins=[-np.inf, 5.0, 15.0, 30.0, 50.0, np.inf],
        labels=PRICE_TIER_ORDER,
    )
    candidates["projection_strength_pct"] = candidates.groupby(
        ["year", "pos"]
    ).pred_fp_per_game.rank(method="average", pct=True, ascending=True)
    candidates["price_strength_pct"] = candidates.groupby(
        ["year", "pos"]
    ).point_salary.rank(method="average", pct=True, ascending=True)
    candidates["value_rank_gap"] = (
        candidates.projection_strength_pct - candidates.price_strength_pct
    )
    candidates["value_over_price_quintile"] = candidates.groupby(
        ["year", "pos"]
    ).value_rank_gap.transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            5,
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    candidates["top_projection_quartile"] = candidates.groupby(
        ["year", "pos"]
    ).pred_fp_per_game.rank(method="average", pct=True, ascending=False).le(0.25)
    candidates["salary_center_source"] = np.select(
        [
            candidates.salary_model_matched,
            candidates.espn_source_matched,
        ],
        ["current_salary_model", "espn_fallback"],
        default="minimum_fallback",
    )

    slots = slots.merge(
        candidates.drop(columns=["player"]),
        on=["year", "player_key"],
        how="left",
        validate="many_to_one",
    )
    if slots.point_salary.isna().any():
        raise AssertionError("A selected roster player did not join to its salary surface.")
    return candidates, rosters, slots


def validate_inputs(
    candidates: pd.DataFrame,
    rosters: pd.DataFrame,
    slots: pd.DataFrame,
) -> dict[str, Any]:
    if len(rosters) != 4_000 or not rosters.status.eq("optimal").all():
        raise AssertionError("The chance replay is incomplete.")
    if len(slots) != 52_000:
        raise AssertionError("Expected 13 slots across 4,000 rosters.")
    if candidates.duplicated(["year", "player_key"]).any():
        raise AssertionError("Candidate surfaces contain duplicate player-origins.")
    if candidates.loc[candidates.actual_salary_recorded, "is_keeper"].ne(0).any():
        raise AssertionError("A recorded keeper remained in an auctionable candidate pool.")
    reconstructed = slots.groupby(["year", "trial", "chance_level"], as_index=False).agg(
        joined_actual_salary_spend=("actual_salary_used_in_replay", "sum"),
        joined_point_salary_spend=("point_salary", "sum"),
        joined_missing_actual_players=("actual_salary_recorded", lambda values: int((~values).sum())),
    )
    check = rosters.merge(
        reconstructed,
        on=["year", "trial", "chance_level"],
        validate="one_to_one",
    )
    if not np.allclose(
        check.actual_salary_spend,
        check.joined_actual_salary_spend,
        atol=1e-8,
    ):
        raise AssertionError("Joined actual salaries do not reproduce roster spending.")
    if not np.allclose(
        check.point_salary_spend,
        check.joined_point_salary_spend,
        atol=3e-6,
    ):
        raise AssertionError("Joined point salaries do not reproduce roster spending.")
    if not check.actual_salary_missing_players.eq(
        check.joined_missing_actual_players
    ).all():
        raise AssertionError("Missing actual-price counts do not reproduce the replay.")
    return {
        "chance_rosters_complete": True,
        "all_52000_selected_slots_joined": True,
        "actual_and_point_roster_spend_reproduced": True,
        "missing_actual_price_counts_reproduced": True,
        "auctionable_candidates_exclude_recorded_keepers": True,
    }


def cohort_summaries(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[candidates.actual_salary_recorded].copy()
    periods: list[tuple[str, pd.DataFrame]] = [("all_years", observed)]
    periods.extend((str(year), group) for year, group in observed.groupby("year"))
    rows = []
    for period, data in periods:
        groups = {
            "all_observed_auctionable": data,
            "ever_selected_unique": data[data.selection_slots.gt(0)],
            "never_selected_unique": data[data.selection_slots.eq(0)],
            "top_projection_quartile_ever_selected": data[
                data.top_projection_quartile & data.selection_slots.gt(0)
            ],
            "top_projection_quartile_never_selected": data[
                data.top_projection_quartile & data.selection_slots.eq(0)
            ],
            "top_projection_quartile_rare_le_5pct": data[
                data.top_projection_quartile & data.selection_rate.le(0.05)
            ],
            "top_projection_quartile_frequent_ge_25pct": data[
                data.top_projection_quartile & data.selection_rate.ge(0.25)
            ],
        }
        for label, group in groups.items():
            rows.append(cohort_row(group, label, period))
        selected = data[data.selection_slots.gt(0)]
        rows.append(
            cohort_row(
                selected,
                "selected_roster_slots_weighted",
                period,
                weights=selected.selection_slots,
            )
        )
    return pd.DataFrame(rows)


def selection_bucket_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[candidates.actual_salary_recorded].copy()
    rows = []
    for bucket in SELECTION_BUCKET_ORDER:
        group = observed[observed.selection_bucket.astype(str).eq(bucket)]
        row = cohort_row(group, bucket, "all_years")
        rows.append(row)
    return pd.DataFrame(rows)


def high_projection_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[
        candidates.actual_salary_recorded & candidates.top_projection_quartile
    ].copy()
    observed["high_projection_selection_group"] = pd.cut(
        observed.selection_rate,
        bins=[-1e-12, 0.05, 0.25, 0.50, 1.000001],
        labels=["selected_<=5%", "selected_5-25%", "selected_25-50%", "selected_>50%"],
        include_lowest=True,
    )
    rows = []
    for label, group in observed.groupby(
        "high_projection_selection_group", observed=False
    ):
        row = cohort_row(group, str(label), "all_years")
        rows.append(row)
    return pd.DataFrame(rows)


def roster_gap_decomposition(rosters: pd.DataFrame) -> pd.DataFrame:
    by_year = rosters.groupby(["year", "chance_level"], as_index=False).agg(
        rosters=("trial", "size"),
        scenario_mean_spend=("heldout_salary_spend_mean", "mean"),
        point_predicted_spend=("point_salary_spend", "mean"),
        actual_spend=("actual_salary_spend", "mean"),
    )
    by_year["point_minus_scenario_discount"] = (
        by_year.point_predicted_spend - by_year.scenario_mean_spend
    )
    by_year["actual_minus_point_residual"] = (
        by_year.actual_spend - by_year.point_predicted_spend
    )
    by_year["actual_minus_scenario_total"] = (
        by_year.actual_spend - by_year.scenario_mean_spend
    )
    if not np.allclose(
        by_year.point_minus_scenario_discount + by_year.actual_minus_point_residual,
        by_year.actual_minus_scenario_total,
        atol=1e-10,
    ):
        raise AssertionError("Roster salary-gap decomposition does not reconcile.")
    return by_year


def period_gap_summary(by_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, group in by_year.groupby("chance_level"):
        for period, selected in (
            ("development_2022_2024", group[group.year.le(2024)]),
            ("temporal_check_2025", group[group.year.eq(2025)]),
        ):
            rows.append(
                {
                    "period": period,
                    "chance_level": level,
                    **{
                        column: float(selected[column].mean())
                        for column in [
                            "scenario_mean_spend",
                            "point_predicted_spend",
                            "actual_spend",
                            "point_minus_scenario_discount",
                            "actual_minus_point_residual",
                            "actual_minus_scenario_total",
                        ]
                    },
                }
            )
    return pd.DataFrame(rows)


def correlation_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    observed = candidates[candidates.actual_salary_recorded]
    rows = []
    for period, group in [("all_years", observed), *[
        (str(year), year_group) for year, year_group in observed.groupby("year")
    ]]:
        correlation, p_value = stats.spearmanr(
            group.selection_rate,
            group.salary_residual,
        )
        rows.append(
            {
                "period": period,
                "player_origins": int(len(group)),
                "selection_rate_residual_spearman": float(correlation),
                "two_sided_p_value_descriptive": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def top_contributors(candidates: pd.DataFrame) -> pd.DataFrame:
    output = candidates[
        candidates.actual_salary_recorded & candidates.selection_slots.gt(0)
    ].copy()
    output["residual_contribution_per_roster_within_origin"] = (
        output.selection_slots * output.salary_residual / 1_000.0
    )
    output["gross_direction"] = np.where(
        output.residual_contribution_per_roster_within_origin.ge(0),
        "positive",
        "negative",
    )
    output["direction_rank"] = output.groupby("gross_direction")[
        "residual_contribution_per_roster_within_origin"
    ].rank(
        method="first",
        ascending=False,
    )
    negative = output.gross_direction.eq("negative")
    output.loc[negative, "direction_rank"] = output.loc[negative].groupby(
        "gross_direction"
    )["residual_contribution_per_roster_within_origin"].rank(
        method="first", ascending=True
    )
    keep = [
        "year",
        "player",
        "player_key",
        "pos",
        "pred_fp_per_game",
        "point_salary",
        "actual_salary",
        "salary_residual",
        "selection_slots",
        "selection_rate",
        "predicted_salary_tier",
        "value_over_price_quintile",
        "salary_center_source",
        "residual_contribution_per_roster_within_origin",
        "gross_direction",
        "direction_rank",
    ]
    return output.loc[output.direction_rank.le(30), keep].sort_values(
        ["gross_direction", "direction_rank"]
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    candidates, rosters, slots = build_inputs()
    validation = validate_inputs(candidates, rosters, slots)
    observed_slots = slots[slots.actual_salary_recorded].copy()

    cohorts = cohort_summaries(candidates)
    buckets = selection_bucket_summary(candidates)
    high_projection = high_projection_summary(candidates)
    by_position = grouped_slot_summary(observed_slots, ["pos"])
    by_year_position = grouped_slot_summary(observed_slots, ["year", "pos"])
    by_price_tier = grouped_slot_summary(observed_slots, ["predicted_salary_tier"])
    by_value_quintile = grouped_slot_summary(
        observed_slots, ["value_over_price_quintile"]
    )
    by_year_chance = grouped_slot_summary(
        observed_slots, ["year", "chance_level"]
    )
    by_center_source = grouped_slot_summary(observed_slots, ["salary_center_source"])
    gap_by_year = roster_gap_decomposition(rosters)
    gap_periods = period_gap_summary(gap_by_year)
    correlations = correlation_summary(candidates)
    contributors = top_contributors(candidates)

    all_cohorts = cohorts[cohorts.period.eq("all_years")]
    all_row = all_cohorts[
        all_cohorts.cohort.eq("all_observed_auctionable")
    ].iloc[0]
    weighted_row = all_cohorts[
        all_cohorts.cohort.eq("selected_roster_slots_weighted")
    ].iloc[0]
    validation.update(
        {
            "candidate_player_origins": int(len(candidates)),
            "recorded_actual_player_origins": int(candidates.actual_salary_recorded.sum()),
            "selected_slots": int(len(slots)),
            "selected_slots_with_recorded_actual": int(len(observed_slots)),
            "selected_slot_actual_coverage": float(
                observed_slots.shape[0] / slots.shape[0]
            ),
        }
    )

    outputs = {
        "candidate_diagnostic.csv": candidates,
        "cohort_summary.csv": cohorts,
        "selection_frequency_summary.csv": buckets,
        "high_projection_summary.csv": high_projection,
        "selected_residual_by_position.csv": by_position,
        "selected_residual_by_year_position.csv": by_year_position,
        "selected_residual_by_price_tier.csv": by_price_tier,
        "selected_residual_by_value_quintile.csv": by_value_quintile,
        "selected_residual_by_year_chance.csv": by_year_chance,
        "selected_residual_by_salary_center_source.csv": by_center_source,
        "roster_gap_decomposition_by_year_chance.csv": gap_by_year,
        "roster_gap_decomposition_periods.csv": gap_periods,
        "selection_residual_correlations.csv": correlations,
        "top_residual_contributors.csv": contributors,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS / filename, index=False)

    lines = [
        "# Selected-Roster Salary Residual Diagnostic",
        "",
        (
            f"Residuals use {validation['recorded_actual_player_origins']} recorded "
            f"player-origin prices from {validation['candidate_player_origins']} "
            f"auctionable candidates. Recorded-price coverage is "
            f"{100 * validation['selected_slot_actual_coverage']:.1f}% of the "
            "52,000 selected roster slots."
        ),
        "",
        "## Core cohort comparison",
        "",
        base.markdown_table(
            all_cohorts,
            [
                "cohort",
                "player_origins",
                "weighted_observations",
                "mean_salary_residual",
                "positive_residual_rate",
                "mean_selection_rate",
                "mean_scenario_center_shift",
                "mean_actual_minus_scenario",
            ],
            digits=3,
        ),
        "",
        "## Selection-frequency gradient",
        "",
        base.markdown_table(
            buckets,
            [
                "cohort",
                "player_origins",
                "mean_salary_residual",
                "positive_residual_rate",
                "mean_selection_rate",
                "mean_scenario_center_shift",
            ],
            digits=3,
        ),
        "",
        "## High-projection players by selection frequency",
        "",
        base.markdown_table(
            high_projection,
            [
                "cohort",
                "player_origins",
                "mean_salary_residual",
                "positive_residual_rate",
                "mean_selection_rate",
                "mean_scenario_center_shift",
            ],
            digits=3,
        ),
        "",
        "## Selected roster slots by position",
        "",
        base.markdown_table(
            by_position,
            [
                "pos",
                "selected_slots",
                "player_origins",
                "mean_salary_residual",
                "positive_residual_rate",
                "mean_scenario_center_shift",
            ],
            digits=3,
        ),
        "",
        "## Selected roster slots by predicted-price tier",
        "",
        base.markdown_table(
            by_price_tier,
            [
                "predicted_salary_tier",
                "selected_slots",
                "player_origins",
                "mean_salary_residual",
                "positive_residual_rate",
                "mean_scenario_center_shift",
            ],
            digits=3,
        ),
        "",
        "## Selected roster slots by value-over-price quintile",
        "",
        "Quintile 5 has the strongest projection rank relative to its predicted-price rank within year and position.",
        "",
        base.markdown_table(
            by_value_quintile,
            [
                "value_over_price_quintile",
                "selected_slots",
                "player_origins",
                "mean_salary_residual",
                "positive_residual_rate",
                "mean_scenario_center_shift",
            ],
            digits=3,
        ),
        "",
        "## Roster-gap decomposition",
        "",
        base.markdown_table(
            gap_periods,
            [
                "period",
                "chance_level",
                "scenario_mean_spend",
                "point_predicted_spend",
                "actual_spend",
                "point_minus_scenario_discount",
                "actual_minus_point_residual",
                "actual_minus_scenario_total",
            ],
            digits=3,
        ),
        "",
        "## Main interpretation",
        "",
        (
            f"The all-player mean residual is ${all_row.mean_salary_residual:.2f}, "
            f"while the roster-slot-weighted mean is "
            f"${weighted_row.mean_salary_residual:.2f} per selected player."
        ),
        "",
        (
            f"The all-player mean five-draw scenario shift versus the point salary is "
            f"${candidates.scenario_center_shift.mean():.2f} per player, while the "
            f"roster-slot-weighted shift is "
            f"${np.average(candidates.scenario_center_shift, weights=candidates.selection_slots):.2f}."
        ),
        "",
        "Ever-selected unique players do not have materially higher residuals than the full pool. The bias appears when selection frequency is retained: frequently reused players carry positive residuals and rare selections carry negative residuals.",
        "",
        "The prior roughly $29 actual-minus-scenario gap has two components: actual prices above the point-predicted salary row and normalized five-draw scenario spend below that point row for the selected roster. It should not be attributed entirely to player-level salary residuals.",
        "",
        "## Limits",
        "",
        "- The value-over-price measure is a transparent rank-gap proxy, not the exact context-specific managed ILP coefficient.",
        "- A player selected in many Monte Carlo trials repeats one realized season salary; slot weighting measures roster impact, not independent statistical sample size.",
        "- Only four realized auction markets are available, and the salary method specification is retrospective as of 2026.",
        "- Recorded actual prices cover only part of the candidate pool; player-residual summaries exclude missing prices, while exact roster reconstruction retains the replay's intentional $1 fallback.",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "study": STUDY_DIR.name,
        "residual_definition": "recorded actual salary minus normalized point-predicted salary",
        "selection_weighting": {
            "unique_player_origin": "one row per year/player",
            "roster_slot": "selection count across 250 trials x 4 chance levels per origin",
        },
        "value_proxy": "within-year-position projection strength percentile minus predicted salary strength percentile",
        "scenario_center_definition": "mean of the exact 1,000 normalized five-draw salary markets minus normalized point salary",
        "sources": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": base.sha256_file(Path(__file__).resolve()),
            "chance_rosters": str(CHANCE_RESULTS / "roster_trials.csv"),
            "chance_rosters_sha256": base.sha256_file(CHANCE_RESULTS / "roster_trials.csv"),
            "chance_salary_surface": str(CHANCE_RESULTS / "salary_surface_audit.csv"),
            "chance_salary_surface_sha256": base.sha256_file(CHANCE_RESULTS / "salary_surface_audit.csv"),
            "actual_salary_database": str(base.SIM_DB),
            "actual_salary_database_sha256": base.sha256_file(base.SIM_DB),
            "simulation_helper": str(base.APP_HELPER),
            "simulation_helper_sha256": base.sha256_file(base.APP_HELPER),
        },
        "validation": validation,
        "output_rows": {name: int(len(frame)) for name, frame in outputs.items()},
    }
    (RESULTS / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"Diagnostic complete: {RESULTS}")


if __name__ == "__main__":
    main()
