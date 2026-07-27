"""Independent consistency checks for the completed salary v5 replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"
FRONTIER = RESULTS / "frontier_v5"
SELECTED = RESULTS / "selected_residuals_v5"


def main() -> None:
    accuracy = pd.read_csv(RESULTS / "point_accuracy_by_period.csv")
    trials = pd.read_csv(FRONTIER / "roster_trials.csv")
    comparison = pd.read_csv(RESULTS / "frontier_paired_effects.csv")
    comparison_summary = pd.read_csv(
        RESULTS / "frontier_comparison_by_period.csv"
    )
    candidates = pd.read_csv(SELECTED / "candidate_diagnostic.csv")
    gap = pd.read_csv(
        SELECTED / "roster_gap_decomposition_periods.csv"
    )

    all_normalized = accuracy[
        accuracy.prediction_scale.eq("normalized")
        & accuracy.period.eq("all_years")
    ].set_index("method")
    assert all_normalized.loc["v5", "mae"] < all_normalized.loc["v1", "mae"]
    assert all_normalized.loc["v5", "mae"] < all_normalized.loc["v3", "mae"]
    assert all_normalized.loc["v5", "rmse"] < all_normalized.loc["v1", "rmse"]
    assert all_normalized.loc["v5", "rmse"] < all_normalized.loc["v3", "rmse"]

    assert len(trials) == 4_000
    assert trials.status.eq("optimal").all()
    assert trials.groupby(["year", "trial"]).size().eq(4).all()
    assert len(comparison) == 4_000
    assert comparison.groupby(["year", "trial"]).size().eq(4).all()

    for period, mask in [
        ("development_2022_2024", comparison.year.between(2022, 2024)),
        ("temporal_check_2025", comparison.year.eq(2025)),
    ]:
        raw = comparison[mask]
        reported = comparison_summary[
            comparison_summary.period.eq(period)
            & comparison_summary.chance_level.eq("all")
        ].iloc[0]
        assert np.isclose(
            reported.roster_changed_rate,
            raw.roster_changed.mean(),
        )
        assert np.isclose(
            reported.mean_actual_cap_feasible_effect_v5_minus_v1,
            raw.actual_cap_feasible_effect_v5_minus_v1.mean(),
        )
        assert np.isclose(
            reported.mean_actual_cap_overage_effect_v5_minus_v1,
            raw.actual_cap_overage_effect_v5_minus_v1.mean(),
        )

    assert len(candidates) == 930
    assert not candidates.duplicated(["year", "player_key"]).any()
    assert candidates.selection_slots.sum() == 52_000
    assert np.allclose(
        gap.point_minus_scenario_discount
        + gap.actual_minus_point_residual,
        gap.actual_minus_scenario_total,
    )

    audit = {
        "v5_best_all_year_normalized_mae_of_v1_v3_v5": True,
        "v5_best_all_year_normalized_rmse_of_v1_v3_v5": True,
        "all_4000_v5_frontier_cells_optimal": True,
        "all_paired_frontier_cells_reconciled": True,
        "frontier_period_effects_reproduced": True,
        "all_52000_v5_selected_slots_reconciled": True,
        "roster_gap_decomposition_reproduced": True,
    }
    (RESULTS / "independent_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
