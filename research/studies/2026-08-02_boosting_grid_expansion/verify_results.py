"""Verify temporal boundaries and output integrity for the boosting-grid study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
EXPECTED_ROWS = {"dk": 1237, "beta": 1226}
EXPECTED_METHODS = {
    "current_single",
    "expanded_lgbm_replacement_equal3",
    "expanded_catboost_equal4",
    "current_plus_extra_trees_equal4",
    "expanded_lgbm_plus_extra_trees_equal4",
    "expanded_lgbm_plus_catboost_equal4",
}


def main() -> None:
    for league, expected_rows in EXPECTED_ROWS.items():
        directory = STUDY_DIR / f"results_{league}"
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(directory / "holdout_predictions.csv")
        selections = pd.read_csv(directory / "selected_parameters.csv")
        origins = pd.read_csv(directory / "origin_candidate_scores.csv")
        bootstrap = pd.read_csv(directory / "player_cluster_bootstrap.csv")
        assert metadata["training_max_season"] == 2022
        assert metadata["holdout_seasons"] == [2023, 2024, 2025]
        assert metadata["tuning_origins"] == list(range(2013, 2023))
        assert metadata["lightgbm_candidate_count"] == 16
        assert metadata["catboost_candidate_count"] == 16
        assert metadata["production_changed"] is False
        assert set(predictions["season"]) == {2023, 2024, 2025}
        blend = predictions[predictions["model_family"].eq("primary_blend")]
        assert set(blend["method"]) == EXPECTED_METHODS
        assert blend.groupby("method").size().eq(expected_rows).all()
        assert np.isfinite(predictions["prediction"]).all()
        assert selections.groupby("model_family").size().eq(1).all()
        assert origins.groupby("model_family").size().eq(160).all()
        assert origins.groupby(["model_family", "origin"])["candidate_id"].nunique().eq(16).all()
        assert bootstrap["draws"].eq(20_000).all()
    manifest = json.loads(
        (STUDY_DIR / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    selections = pd.read_csv(STUDY_DIR / "results" / "selection_comparisons.csv")
    dk_lgbm = selections[
        selections["league"].eq("dk")
        & selections["model_family"].eq("conditional_ppg_lightgbm_expanded")
    ].iloc[0]
    beta_lgbm = selections[
        selections["league"].eq("beta")
        & selections["model_family"].eq("conditional_ppg_lightgbm_expanded")
    ].iloc[0]
    catboost = selections[
        selections["model_family"].eq("conditional_ppg_catboost_expanded")
    ]
    assert dk_lgbm["candidate_id"] == 7
    assert beta_lgbm["candidate_id"] == 8
    assert beta_lgbm["selection_score_delta_vs_original"] < 0
    assert catboost["candidate_id"].eq(4).all()
    overall = pd.read_csv(STUDY_DIR / "results" / "overall_comparisons.csv")
    beta_lgbm_result = overall[
        overall["league"].eq("beta")
        & overall["method"].eq("expanded_lgbm_replacement_equal3")
        & overall["baseline_method"].eq("current_single")
    ]
    assert beta_lgbm_result["rmse_delta"].gt(0).all()
    assert manifest["decision"] == "retain_existing_boosting_grids_and_parameters"
    assert manifest["production_changed"] is False
    print("verified temporal boundaries, grid coverage, predictions, and decision")


if __name__ == "__main__":
    main()
