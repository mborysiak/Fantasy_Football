"""Independent consistency checks for the salary ensemble-feature ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"


def main() -> None:
    manifest = json.loads(
        (RESULTS / "source_manifest.json").read_text(encoding="utf-8")
    )
    paired = pd.read_csv(RESULTS / "paired_validation_predictions.csv")
    coverage = pd.read_csv(RESULTS / "validation_coverage.csv")
    candidates = pd.read_csv(RESULTS / "candidate_surface_v1_v2.csv")
    repriced = pd.read_csv(RESULTS / "fixed_v1_rosters_repriced.csv")
    accuracy = pd.read_csv(RESULTS / "paired_accuracy_summary.csv")

    all_coverage = coverage[coverage.period.eq("all_years")].iloc[0]
    assert len(paired) == int(all_coverage.paired_rows)
    assert int(all_coverage.v1_only_rows) == 0
    assert int(all_coverage.v2_only_rows) == 1
    assert not paired.duplicated(["year", "player_key"]).any()
    assert np.allclose(paired.actual_salary_v1, paired.actual_salary_v2)

    residual_v1 = paired.actual_salary_v1 - paired.pred_salary_v1
    residual_v2 = paired.actual_salary_v2 - paired.pred_salary_v2
    overall = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.eq("all_years")
    ].iloc[0]
    assert np.isclose(overall.v1_mean_residual, residual_v1.mean())
    assert np.isclose(overall.v2_mean_residual, residual_v2.mean())
    assert np.isclose(overall.v1_mae, residual_v1.abs().mean())
    assert np.isclose(overall.v2_mae, residual_v2.abs().mean())

    assert len(candidates) == manifest["validation"]["candidate_player_origins"]
    assert (
        np.abs(candidates.point_salary_v1 - candidates.point_salary).max()
        < 1e-5
    )
    assert not candidates.point_salary_v2.isna().any()

    assert len(repriced) == manifest["validation"]["fixed_roster_rows"]
    assert np.abs(
        repriced.point_salary_spend_v1 - repriced.point_salary_spend
    ).max() < 1e-4
    assert np.abs(
        repriced.actual_salary_spend_reconstructed
        - repriced.actual_salary_spend
    ).max() < 1e-8
    assert np.allclose(
        repriced.actual_minus_point_v2 - repriced.actual_minus_point_v1,
        -repriced.point_spend_shift_v2_minus_v1,
    )

    comparison_manifest_path = RESULTS / "frontier_comparison_manifest.json"
    if comparison_manifest_path.exists():
        comparison_manifest = json.loads(
            comparison_manifest_path.read_text(encoding="utf-8")
        )
        frontier = pd.read_csv(RESULTS / "frontier_paired_effects.csv")
        frontier_summary = pd.read_csv(
            RESULTS / "frontier_comparison_by_period.csv"
        )
        assert len(frontier) == 4_000
        assert frontier.groupby(["year", "trial"]).size().eq(4).all()
        assert comparison_manifest["validation"]["paired_frontier_cells"] == 4_000
        for period, mask in [
            ("development_2022_2024", frontier.year.between(2022, 2024)),
            ("temporal_check_2025", frontier.year.eq(2025)),
        ]:
            raw = frontier[mask]
            reported = frontier_summary[
                frontier_summary.period.eq(period)
                & frontier_summary.chance_level.eq("all")
            ].iloc[0]
            assert np.isclose(
                reported.roster_changed_rate,
                raw.roster_changed.mean(),
            )
            assert np.isclose(
                reported.mean_actual_cap_feasible_effect_v2_minus_v1,
                raw.actual_cap_feasible_effect_v2_minus_v1.mean(),
            )
            assert np.isclose(
                reported.mean_managed_forecast_season_points_effect_v2_minus_v1,
                raw.managed_forecast_season_points_effect_v2_minus_v1.mean(),
            )

    audit = {
        "paired_observed_rows_reproduced": True,
        "one_v2_only_observed_row_confirmed": True,
        "overall_accuracy_metrics_reproduced": True,
        "prior_v1_candidate_surface_reproduced": True,
        "all_v2_candidate_prices_present": True,
        "prior_v1_roster_spending_reproduced": True,
        "repricing_gap_identity_reproduced": True,
        "paired_frontier_cells_reproduced": comparison_manifest_path.exists(),
        "frontier_period_summaries_reproduced": comparison_manifest_path.exists(),
    }
    (RESULTS / "independent_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
