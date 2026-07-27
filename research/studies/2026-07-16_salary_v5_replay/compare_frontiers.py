"""Compare identical-seed v1 and v5 salary chance-frontier replays."""

from __future__ import annotations

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
V5_RESULTS = RESULTS / "frontier_v5"
KEYS = ["year", "trial", "chance_level"]
METRICS = [
    "managed_forecast_season_points",
    "heldout_cap_probability",
    "heldout_salary_spend_mean",
    "heldout_salary_spend_p90",
    "heldout_salary_spend_p95",
    "actual_cap_feasible",
    "actual_cap_overage",
    "actual_salary_spend",
    "point_salary_spend",
    "raw_actual_points_audit_only",
]


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Format a small DataFrame without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def periods(data: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all_years", data
    yield "development_2022_2024", data[data.year.between(2022, 2024)]
    yield "temporal_check_2025", data[data.year.eq(2025)]
    for year, group in data.groupby("year", sort=True):
        yield str(int(year)), group


def validate_manifests(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
) -> None:
    if baseline["salary_identity"]["method_version"] != "current_locked_spec_v1":
        raise AssertionError("Unexpected baseline method.")
    if (
        challenger["salary_identity"]["method_version"]
        != "current_locked_spec_v5_compact_salary_features"
    ):
        raise AssertionError("Unexpected challenger method.")
    left = dict(baseline["config"])
    right = dict(challenger["config"])
    left.pop("output_dir", None)
    right.pop("output_dir", None)
    if left != right:
        raise AssertionError("Replay configurations differ.")
    if baseline["chance_levels"] != challenger["chance_levels"]:
        raise AssertionError("Chance levels differ.")
    if baseline["fixed_settings"] != challenger["fixed_settings"]:
        raise AssertionError("Fixed settings differ.")
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
    for year in ["2022", "2023", "2024", "2025"]:
        for column in stable:
            if (
                baseline["origins"][year][column]
                != challenger["origins"][year][column]
            ):
                raise AssertionError(
                    f"{year} origin differs on {column}."
                )


def build_pairs(v1: pd.DataFrame, v5: pd.DataFrame) -> pd.DataFrame:
    if not v1.status.eq("optimal").all() or not v5.status.eq("optimal").all():
        raise AssertionError("A frontier has non-optimal cells.")
    keep = [*KEYS, "roster", *METRICS]
    paired = v1[keep].merge(
        v5[keep],
        on=KEYS,
        suffixes=("_v1", "_v5"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError("Frontier paired keys differ.")
    paired = paired.drop(columns="_merge")
    paired["roster_changed"] = paired.roster_v1.ne(paired.roster_v5)
    for metric in METRICS:
        left = paired[f"{metric}_v1"]
        right = paired[f"{metric}_v5"]
        if metric == "actual_cap_feasible":
            left = left.astype(float)
            right = right.astype(float)
        paired[f"{metric}_effect_v5_minus_v1"] = right - left
    paired["v5_only_actual_cap_feasible"] = (
        ~paired.actual_cap_feasible_v1
        & paired.actual_cap_feasible_v5
    )
    paired["v1_only_actual_cap_feasible"] = (
        paired.actual_cap_feasible_v1
        & ~paired.actual_cap_feasible_v5
    )
    return paired


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, period_rows in periods(paired):
        groups: list[tuple[str, pd.DataFrame]] = [("all", period_rows)]
        groups.extend(
            (f"{chance:.1f}", group)
            for chance, group in period_rows.groupby(
                "chance_level",
                sort=True,
            )
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
                "v5_actual_cap_feasible_rate": float(
                    group.actual_cap_feasible_v5.mean()
                ),
                "v5_only_feasible_rosters": int(
                    group.v5_only_actual_cap_feasible.sum()
                ),
                "v1_only_feasible_rosters": int(
                    group.v1_only_actual_cap_feasible.sum()
                ),
            }
            for metric in METRICS:
                if metric == "actual_cap_feasible":
                    continue
                row[f"v1_mean_{metric}"] = float(
                    group[f"{metric}_v1"].mean()
                )
                row[f"v5_mean_{metric}"] = float(
                    group[f"{metric}_v5"].mean()
                )
                row[f"mean_{metric}_effect_v5_minus_v1"] = float(
                    group[
                        f"{metric}_effect_v5_minus_v1"
                    ].mean()
                )
            row["mean_actual_cap_feasible_effect_v5_minus_v1"] = float(
                group.actual_cap_feasible_effect_v5_minus_v1.mean()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def compare_surfaces(
    v1: pd.DataFrame,
    v5: pd.DataFrame,
) -> pd.DataFrame:
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
        v5[columns],
        on=["year", "player_key"],
        suffixes=("_v1", "_v5"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError("Candidate universes differ.")
    paired = paired.drop(columns="_merge")
    if not np.allclose(
        paired.pred_fp_per_game_v1,
        paired.pred_fp_per_game_v5,
        atol=1e-12,
    ):
        raise AssertionError("Frozen projections differ.")
    paired["point_salary_shift_v5_minus_v1"] = (
        paired.point_salary_v5 - paired.point_salary_v1
    )
    return paired


def write_summary(summary: pd.DataFrame) -> None:
    selected = summary[
        summary.period.isin(
            ["development_2022_2024", "temporal_check_2025"]
        )
        & summary.chance_level.eq("all")
    ]
    columns = [
        "period",
        "paired_rosters",
        "roster_changed_rate",
        "v1_actual_cap_feasible_rate",
        "v5_actual_cap_feasible_rate",
        "mean_actual_cap_feasible_effect_v5_minus_v1",
        "mean_actual_cap_overage_effect_v5_minus_v1",
        "mean_actual_salary_spend_effect_v5_minus_v1",
        "mean_managed_forecast_season_points_effect_v5_minus_v1",
        "mean_raw_actual_points_audit_only_effect_v5_minus_v1",
    ]
    lines = [
        "# Paired v1 versus v5 Salary Frontier",
        "",
        (
            "All effects are v5 minus v1. Positive feasibility and managed "
            "forecast effects favor v5; negative overage/spend effects favor v5."
        ),
        "",
        markdown_table(selected[columns]),
        "",
        (
            "Projection draws, managed contexts, thresholds, optimizer settings, "
            "and random seeds are paired. Raw actual points remain audit-only "
            "when the historical roster was unaffordable."
        ),
        "",
    ]
    (RESULTS / "frontier_comparison_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    baseline_manifest = json.loads(
        (V1_RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    challenger_manifest = json.loads(
        (V5_RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    validate_manifests(baseline_manifest, challenger_manifest)
    paired = build_pairs(
        pd.read_csv(V1_RESULTS / "roster_trials.csv"),
        pd.read_csv(V5_RESULTS / "roster_trials.csv"),
    )
    summary = summarize(paired)
    surfaces = compare_surfaces(
        pd.read_csv(V1_RESULTS / "salary_surface_audit.csv"),
        pd.read_csv(V5_RESULTS / "salary_surface_audit.csv"),
    )
    paired.to_csv(RESULTS / "frontier_paired_effects.csv", index=False)
    summary.to_csv(
        RESULTS / "frontier_comparison_by_period.csv",
        index=False,
    )
    surfaces.to_csv(
        RESULTS / "frontier_salary_surface_comparison.csv",
        index=False,
    )
    write_summary(summary)
    manifest = {
        "baseline_method": "current_locked_spec_v1",
        "challenger_method": "current_locked_spec_v5_compact_salary_features",
        "paired_frontier_cells": int(len(paired)),
        "candidate_player_origins": int(len(surfaces)),
        "all_frontier_cells_optimal": True,
    }
    (RESULTS / "frontier_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        (RESULTS / "frontier_comparison_summary.md").read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()
