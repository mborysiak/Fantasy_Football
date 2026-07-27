"""Compare paired v1 and v2 salary chance-frontier replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
RESULTS = STUDY_DIR / "results"
V1_RESULTS = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_salary_chance_frontier"
    / "results"
)
V2_RESULTS = RESULTS / "frontier_v2"
KEYS = ["year", "trial", "chance_level"]
PAIR_METRICS = [
    "managed_forecast_season_points",
    "heldout_cap_probability",
    "heldout_salary_spend_mean",
    "heldout_salary_spend_p90",
    "heldout_salary_spend_p95",
    "actual_cap_feasible",
    "actual_cap_overage",
    "actual_salary_spend",
    "raw_actual_points_audit_only",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def periods(data: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all_years", data
    yield "development_2022_2024", data[data.year.between(2022, 2024)]
    yield "temporal_check_2025", data[data.year.eq(2025)]
    for year, group in data.groupby("year", sort=True):
        yield str(int(year)), group


def validate_manifests(
    v1_manifest: dict[str, Any],
    v2_manifest: dict[str, Any],
) -> dict[str, Any]:
    expected_v1 = "current_locked_spec_v1"
    expected_v2 = "current_locked_spec_v2_ensemble_features"
    if v1_manifest["salary_identity"]["method_version"] != expected_v1:
        raise AssertionError("Unexpected v1 salary method.")
    if v2_manifest["salary_identity"]["method_version"] != expected_v2:
        raise AssertionError("Unexpected v2 salary method.")
    config_v1 = dict(v1_manifest["config"])
    config_v2 = dict(v2_manifest["config"])
    config_v1.pop("output_dir", None)
    config_v2.pop("output_dir", None)
    if config_v1 != config_v2:
        raise AssertionError("v1 and v2 replay configurations differ.")
    if v1_manifest["chance_levels"] != v2_manifest["chance_levels"]:
        raise AssertionError("v1 and v2 chance levels differ.")
    if v1_manifest["fixed_settings"] != v2_manifest["fixed_settings"]:
        raise AssertionError("v1 and v2 fixed settings differ.")
    for year in ["2022", "2023", "2024", "2025"]:
        origin_v1 = v1_manifest["origins"][year]
        origin_v2 = v2_manifest["origins"][year]
        stable = [
            "candidate_players",
            "forecast_players",
            "forecast_draws",
            "keeper_count",
            "keeper_spend",
            "remaining_budget",
            "remaining_slots",
            "projection_distribution",
            "projection_table",
        ]
        for column in stable:
            if origin_v1[column] != origin_v2[column]:
                raise AssertionError(f"{year} origin differs on {column}.")
    return {
        "same_replay_config_except_output_directory": True,
        "same_chance_levels_and_fixed_settings": True,
        "same_frozen_forecast_and_keeper_origin_metadata": True,
        "salary_method_is_only_intended_model_change": True,
    }


def build_pairs(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    if not v1.status.eq("optimal").all() or not v2.status.eq("optimal").all():
        raise AssertionError("A frontier contains non-optimal cells.")
    if v1.duplicated(KEYS).any() or v2.duplicated(KEYS).any():
        raise AssertionError("A frontier contains duplicate paired keys.")
    keep = [*KEYS, "roster", *PAIR_METRICS]
    paired = v1[keep].merge(
        v2[keep],
        on=KEYS,
        suffixes=("_v1", "_v2"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError("The two frontiers do not contain identical paired cells.")
    paired = paired.drop(columns="_merge")
    paired["roster_changed"] = paired.roster_v1.ne(paired.roster_v2)
    for metric in PAIR_METRICS:
        left = paired[f"{metric}_v1"]
        right = paired[f"{metric}_v2"]
        if metric == "actual_cap_feasible":
            left = left.astype(float)
            right = right.astype(float)
        paired[f"{metric}_effect_v2_minus_v1"] = right - left
    paired["both_actual_cap_feasible"] = (
        paired.actual_cap_feasible_v1 & paired.actual_cap_feasible_v2
    )
    paired["v2_only_actual_cap_feasible"] = (
        ~paired.actual_cap_feasible_v1 & paired.actual_cap_feasible_v2
    )
    paired["v1_only_actual_cap_feasible"] = (
        paired.actual_cap_feasible_v1 & ~paired.actual_cap_feasible_v2
    )
    return paired


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, period_rows in periods(paired):
        groups: list[tuple[str, pd.DataFrame]] = [("all", period_rows)]
        groups.extend(
            (f"{chance:.1f}", group)
            for chance, group in period_rows.groupby("chance_level", sort=True)
        )
        for chance, group in groups:
            row: dict[str, Any] = {
                "period": period,
                "chance_level": chance,
                "paired_rosters": int(len(group)),
                "roster_changed_rate": float(group.roster_changed.mean()),
                "v1_actual_cap_feasible_rate": float(
                    group.actual_cap_feasible_v1.mean()
                ),
                "v2_actual_cap_feasible_rate": float(
                    group.actual_cap_feasible_v2.mean()
                ),
                "v2_only_feasible_rosters": int(
                    group.v2_only_actual_cap_feasible.sum()
                ),
                "v1_only_feasible_rosters": int(
                    group.v1_only_actual_cap_feasible.sum()
                ),
            }
            for metric in PAIR_METRICS:
                if metric == "actual_cap_feasible":
                    continue
                row[f"v1_mean_{metric}"] = float(group[f"{metric}_v1"].mean())
                row[f"v2_mean_{metric}"] = float(group[f"{metric}_v2"].mean())
                row[f"mean_{metric}_effect_v2_minus_v1"] = float(
                    group[f"{metric}_effect_v2_minus_v1"].mean()
                )
            row["mean_actual_cap_feasible_effect_v2_minus_v1"] = float(
                group.actual_cap_feasible_effect_v2_minus_v1.mean()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def surface_comparison(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "player_key",
        "player",
        "pos",
        "pred_fp_per_game",
        "point_salary",
        "salary_model_matched",
    ]
    paired = v1[columns].merge(
        v2[columns],
        on=["year", "player_key"],
        suffixes=("_v1", "_v2"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError("The v1 and v2 replay candidate universes differ.")
    paired = paired.drop(columns="_merge")
    if not np.allclose(
        paired.pred_fp_per_game_v1,
        paired.pred_fp_per_game_v2,
        atol=1e-12,
    ):
        raise AssertionError("Frozen point forecasts differ between methods.")
    paired["point_salary_shift_v2_minus_v1"] = (
        paired.point_salary_v2 - paired.point_salary_v1
    )
    return paired


def write_summary(summary: pd.DataFrame) -> None:
    selected = summary[
        summary.period.isin(
            ["development_2022_2024", "temporal_check_2025"]
        )
        & summary.chance_level.ne("all")
    ]
    columns = [
        "period",
        "chance_level",
        "roster_changed_rate",
        "mean_managed_forecast_season_points_effect_v2_minus_v1",
        "mean_heldout_cap_probability_effect_v2_minus_v1",
        "mean_actual_cap_feasible_effect_v2_minus_v1",
        "mean_actual_cap_overage_effect_v2_minus_v1",
        "mean_actual_salary_spend_effect_v2_minus_v1",
        "mean_raw_actual_points_audit_only_effect_v2_minus_v1",
    ]
    lines = [
        "# Paired v1 versus v2 Salary Frontier",
        "",
        "All effects are v2 minus v1. Positive feasibility and heldout-cap effects favor v2; negative overage and spend effects favor v2.",
        "",
        "Managed forecast points are preseason simulated EV. Raw actual points are audit-only because historically unaffordable rosters remain in that column.",
        "",
        "| "
        + " | ".join(columns)
        + " |",
        "|"
        + "|".join("---" for _ in columns)
        + "|",
    ]
    for _, row in selected.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "The replay changes only the rolling salary method. Projection draws, managed-value contexts, optimizer settings, chance thresholds, and random seeds are paired.",
            "",
        ]
    )
    (RESULTS / "frontier_comparison_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_decision_readout(summary: pd.DataFrame) -> None:
    accuracy = pd.read_csv(RESULTS / "paired_accuracy_summary.csv")
    candidate_accuracy = pd.read_csv(RESULTS / "candidate_accuracy_summary.csv")
    value_tail = pd.read_csv(RESULTS / "value_tail_summary.csv")
    point_all = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.eq("all_years")
    ].iloc[0]
    point_2025 = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.eq("temporal_check_2025")
    ].iloc[0]
    candidate = candidate_accuracy[
        candidate_accuracy.period.eq("all_years")
    ].set_index("method")
    strongest = value_tail[
        value_tail.period.eq("all_years") & value_tail.value_quintile.eq(5)
    ].set_index("method")
    period_all = summary[
        summary.chance_level.eq("all")
        & summary.period.isin(
            ["development_2022_2024", "temporal_check_2025"]
        )
    ].set_index("period")
    season_rows = summary[
        summary.chance_level.ne("all")
        & summary.period.isin(["2022", "2023", "2024", "2025"])
    ]
    season_means = season_rows.groupby("period").mean(numeric_only=True)
    dev = period_all.loc["development_2022_2024"]
    check = period_all.loc["temporal_check_2025"]
    lines = [
        "# Decision Readout",
        "",
        "## Finding",
        "",
        (
            "v2 reduces the salary model's average underprediction bias, but it "
            "does not improve ordinary absolute error consistently."
        ),
        "",
        (
            f"Across {int(point_all.player_years)} common observed player-years, "
            f"mean residual moved from {point_all.v1_mean_residual:+.2f} to "
            f"{point_all.v2_mean_residual:+.2f}, while MAE changed from "
            f"{point_all.v1_mae:.2f} to {point_all.v2_mae:.2f} "
            f"({point_all.mae_delta_v2_minus_v1:+.2f})."
        ),
        (
            f"In the 2025 temporal check, MAE changed from {point_2025.v1_mae:.2f} "
            f"to {point_2025.v2_mae:.2f} "
            f"({point_2025.mae_delta_v2_minus_v1:+.2f})."
        ),
        "",
        "## Optimizer-relevant tail",
        "",
        (
            "On the frozen replay candidate universe, the strongest within-position "
            "value quintile's old-v1-selection-weighted residual changed from "
            f"{strongest.loc['v1', 'selection_weighted_mean_residual']:+.2f} to "
            f"{strongest.loc['v2', 'selection_weighted_mean_residual']:+.2f}."
        ),
        (
            "Across every recorded candidate, the old-v1-selection-weighted "
            "residual changed from "
            f"{candidate.loc['v1', 'selection_weighted_mean_residual']:+.2f} to "
            f"{candidate.loc['v2', 'selection_weighted_mean_residual']:+.2f}."
        ),
        "",
        "## Paired optimizer replay",
        "",
        (
            "The identical-seed v2 chance-frontier replay completed all 4,000 "
            f"cells and changed {dev.roster_changed_rate:.1%} of development "
            f"rosters and {check.roster_changed_rate:.1%} of 2025 rosters."
        ),
        (
            f"Across chance thresholds, development managed forecast EV changed "
            f"{dev.mean_managed_forecast_season_points_effect_v2_minus_v1:+.2f} "
            "season points, held-out modeled affordability changed "
            f"{dev.mean_heldout_cap_probability_effect_v2_minus_v1:+.2%}, "
            "historical feasibility changed "
            f"{dev.mean_actual_cap_feasible_effect_v2_minus_v1:+.2%}, and "
            "historical overage changed "
            f"${dev.mean_actual_cap_overage_effect_v2_minus_v1:+.2f}."
        ),
        (
            f"For 2025, managed forecast EV changed "
            f"{check.mean_managed_forecast_season_points_effect_v2_minus_v1:+.2f}, "
            "held-out modeled affordability changed "
            f"{check.mean_heldout_cap_probability_effect_v2_minus_v1:+.2%}, "
            "historical feasibility changed "
            f"{check.mean_actual_cap_feasible_effect_v2_minus_v1:+.2%}, and "
            "historical overage changed "
            f"${check.mean_actual_cap_overage_effect_v2_minus_v1:+.2f}."
        ),
        "",
        (
            "Season directions were unstable. In 2023, v2 changed managed "
            "forecast EV by "
            f"{season_means.loc['2023', 'mean_managed_forecast_season_points_effect_v2_minus_v1']:+.2f} "
            "but changed historical roster spend by "
            f"${season_means.loc['2023', 'mean_actual_salary_spend_effect_v2_minus_v1']:+.2f}; "
            "in 2022 those effects were "
            f"{season_means.loc['2022', 'mean_managed_forecast_season_points_effect_v2_minus_v1']:+.2f} "
            "and "
            f"${season_means.loc['2022', 'mean_actual_salary_spend_effect_v2_minus_v1']:+.2f}."
        ),
        "",
        "## Action",
        "",
        (
            "Do not promote v2 as the production salary method and do not discard "
            "its feature set. It improves mean bias and the apparent strongest-"
            "value residual tail, but worsens ordinary MAE and does not produce a "
            "stable affordability gain after optimizer reselection."
        ),
        "",
        (
            "Keep v1 as the current comparison/default surface. The next "
            "inexpensive test should be a causally evaluated v1/v2 shrinkage blend "
            "or a restricted correction focused on the optimizer-relevant high-"
            "value/high-price tail. Any candidate should pass both point-error and "
            "selected-roster affordability gates before another full frontier."
        ),
        "",
    ]
    (RESULTS / "decision_readout.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    v1_manifest = json.loads(
        (V1_RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    v2_manifest = json.loads(
        (V2_RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    validation = validate_manifests(v1_manifest, v2_manifest)
    v1_trials = pd.read_csv(V1_RESULTS / "roster_trials.csv")
    v2_trials = pd.read_csv(V2_RESULTS / "roster_trials.csv")
    paired = build_pairs(v1_trials, v2_trials)
    summary = summarize_pairs(paired)
    surfaces = surface_comparison(
        pd.read_csv(V1_RESULTS / "salary_surface_audit.csv"),
        pd.read_csv(V2_RESULTS / "salary_surface_audit.csv"),
    )
    paired.to_csv(RESULTS / "frontier_paired_effects.csv", index=False)
    summary.to_csv(RESULTS / "frontier_comparison_by_period.csv", index=False)
    surfaces.to_csv(RESULTS / "frontier_salary_surface_comparison.csv", index=False)
    write_summary(summary)
    write_decision_readout(summary)

    manifest = {
        "study": STUDY_DIR.name,
        "validation": {
            **validation,
            "paired_frontier_cells": int(len(paired)),
            "candidate_player_origins": int(len(surfaces)),
            "all_frontier_cells_optimal": True,
        },
        "sources": {
            "v1_manifest": str(V1_RESULTS / "source_manifest.json"),
            "v1_manifest_sha256": sha256_file(
                V1_RESULTS / "source_manifest.json"
            ),
            "v1_roster_trials_sha256": sha256_file(
                V1_RESULTS / "roster_trials.csv"
            ),
            "v2_manifest": str(V2_RESULTS / "source_manifest.json"),
            "v2_manifest_sha256": sha256_file(
                V2_RESULTS / "source_manifest.json"
            ),
            "v2_roster_trials_sha256": sha256_file(
                V2_RESULTS / "roster_trials.csv"
            ),
            "wrapper": str(STUDY_DIR / "run_frontier_v2.py"),
            "wrapper_sha256": sha256_file(STUDY_DIR / "run_frontier_v2.py"),
        },
        "outputs": {
            "frontier_paired_effects.csv": int(len(paired)),
            "frontier_comparison_by_period.csv": int(len(summary)),
            "frontier_salary_surface_comparison.csv": int(len(surfaces)),
        },
    }
    (RESULTS / "frontier_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        (RESULTS / "frontier_comparison_summary.md").read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
