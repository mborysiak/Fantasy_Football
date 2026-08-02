"""Focused integrity checks for the persisted PFF screen outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline_includes_expert_projection"] is True
    assert manifest["baseline_includes_adp"] is True
    assert manifest["validation_seasons"] == [2017, 2025]
    assert manifest["rolling_test_seasons"] == [2018, 2025]

    summary = pd.read_csv(RESULTS_DIR / "candidate_summary.csv")
    seasons = pd.read_csv(RESULTS_DIR / "season_diagnostics.csv")
    coverage = pd.read_csv(RESULTS_DIR / "coverage.csv")

    assert set(summary["league"]) == {"dk", "beta"}
    assert summary.duplicated(["league", "position", "candidate"]).sum() == 0
    season_key = ["league", "domain", "position", "candidate", "season"]
    assert seasons.duplicated(season_key).sum() == 0
    assert seasons["season"].between(2018, 2025).all()
    required_season_metrics = [
        "production_rmse", "opportunity_control_rmse", "candidate_rmse",
        "ppg_delta_vs_control", "opportunity_control_brier",
        "candidate_brier", "q90_brier_delta_vs_control",
    ]
    assert seasons[required_season_metrics].notna().all().all()
    assert coverage["identity_coverage"].between(0, 1).all()
    assert coverage["positive_opportunity_coverage"].between(0, 1).all()

    leader = summary[
        summary["league"].eq("dk")
        & summary["position"].eq("TE")
        & summary["candidate"].eq("rec_mtf_per_reception")
    ].iloc[0]
    assert leader["season_bootstrap_high"] < 0
    assert leader["candidate_brier_bootstrap_high"] < 0
    print(
        "verified:",
        len(summary),
        "candidate rows;",
        len(seasons),
        "candidate-season diagnostics",
    )


if __name__ == "__main__":
    main()
