"""Independent checks for the selected-roster residual diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    candidates = pd.read_csv(RESULTS / "candidate_diagnostic.csv")
    cohorts = pd.read_csv(RESULTS / "cohort_summary.csv")
    buckets = pd.read_csv(RESULTS / "selection_frequency_summary.csv")
    by_year_chance = pd.read_csv(RESULTS / "selected_residual_by_year_chance.csv")
    by_value = pd.read_csv(RESULTS / "selected_residual_by_value_quintile.csv")
    by_source = pd.read_csv(RESULTS / "selected_residual_by_salary_center_source.csv")
    gaps = pd.read_csv(RESULTS / "roster_gap_decomposition_by_year_chance.csv")
    manifest = json.loads((RESULTS / "source_manifest.json").read_text(encoding="utf-8"))

    assert len(candidates) == 930
    assert candidates.actual_salary_recorded.sum() == 518
    assert candidates.selection_slots.sum() == 52_000
    observed = candidates[candidates.actual_salary_recorded]
    selected_observed = observed[observed.selection_slots.gt(0)]
    assert selected_observed.selection_slots.sum() == 49_920

    all_mean = float(observed.salary_residual.mean())
    unique_selected_mean = float(selected_observed.salary_residual.mean())
    weighted_selected_mean = float(
        np.average(
            selected_observed.salary_residual,
            weights=selected_observed.selection_slots,
        )
    )
    all_scenario_shift = float(candidates.scenario_center_shift.mean())
    selected_scenario_shift = float(
        np.average(
            candidates.scenario_center_shift,
            weights=candidates.selection_slots,
        )
    )

    reported = cohorts[cohorts.period.eq("all_years")].set_index("cohort")
    assert np.isclose(
        reported.loc["all_observed_auctionable", "mean_salary_residual"],
        all_mean,
    )
    assert np.isclose(
        reported.loc["ever_selected_unique", "mean_salary_residual"],
        unique_selected_mean,
    )
    assert np.isclose(
        reported.loc["selected_roster_slots_weighted", "mean_salary_residual"],
        weighted_selected_mean,
    )

    bucket_values = buckets.set_index("cohort").mean_salary_residual
    assert bucket_values["frequent_25-50%"] > 0
    assert bucket_values["core_>50%"] > bucket_values["frequent_25-50%"]
    assert bucket_values["rare_0-5%"] < 0
    assert by_year_chance.mean_salary_residual.gt(0).all()
    assert by_year_chance.mean_scenario_center_shift.lt(0).all()
    assert by_value.loc[
        by_value.value_over_price_quintile.eq(5), "mean_salary_residual"
    ].iloc[0] == by_value.mean_salary_residual.max()
    current_source = by_source[
        by_source.salary_center_source.eq("current_salary_model")
    ].iloc[0]
    assert current_source.mean_salary_residual > 0

    assert np.allclose(
        gaps.point_minus_scenario_discount + gaps.actual_minus_point_residual,
        gaps.actual_minus_scenario_total,
        atol=1e-10,
    )
    runner = STUDY_DIR / "run_diagnostic.py"
    assert sha256(runner) == manifest["sources"]["runner_sha256"]
    assert all(manifest["validation"].values())

    audit = {
        "candidate_player_origins": int(len(candidates)),
        "recorded_actual_player_origins": int(candidates.actual_salary_recorded.sum()),
        "selected_slots": int(candidates.selection_slots.sum()),
        "selected_slots_with_recorded_actual": int(
            selected_observed.selection_slots.sum()
        ),
        "all_observed_mean_residual": all_mean,
        "unique_ever_selected_mean_residual": unique_selected_mean,
        "roster_slot_weighted_mean_residual": weighted_selected_mean,
        "all_candidate_mean_scenario_shift": all_scenario_shift,
        "roster_slot_weighted_mean_scenario_shift": selected_scenario_shift,
        "all_year_chance_selected_residuals_positive": True,
        "all_year_chance_selected_scenario_shifts_negative": True,
        "roster_gap_decomposition_reconciles": True,
        "reported_core_cohorts_recomputed": True,
        "runner_hash_matches_manifest": True,
    }
    (RESULTS / "independent_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

