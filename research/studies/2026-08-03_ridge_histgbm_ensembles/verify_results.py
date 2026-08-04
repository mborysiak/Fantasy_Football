"""Verify temporal boundaries, grids, predictions, and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
LEAGUES = {"dk": 1237, "beta": 1226}
ENSEMBLE_METHODS = {
    "current_single",
    "ridge_replaces_lasso_equal3",
    "lasso_ridge_split_linear_third",
    "current_plus_ridge_equal4",
    "histgbm_replaces_lightgbm_equal3",
    "current_plus_histgbm_equal4",
}
EXPECTED_COUNTS = {
    "conditional_ppg_ridge": 7,
    "conditional_ppg_hist_gradient_boosting": 10,
}


def main() -> None:
    for league, expected_rows in LEAGUES.items():
        directory = STUDY_DIR / f"results_{league}"
        metadata = json.loads(
            (directory / "run_metadata.json").read_text(encoding="utf-8")
        )
        predictions = pd.read_csv(directory / "holdout_predictions.csv")
        selections = pd.read_csv(directory / "selected_parameters.csv")
        origin_scores = pd.read_csv(directory / "origin_candidate_scores.csv")
        scores = pd.read_csv(directory / "scores.csv")
        bootstrap = pd.read_csv(directory / "player_cluster_bootstrap.csv")

        assert metadata["training_max_season"] == 2022
        assert metadata["holdout_seasons"] == [2023, 2024, 2025]
        assert metadata["tuning_origins"] == list(range(2013, 2023))
        assert metadata["feature_count"] == 40
        assert metadata["candidate_counts"] == EXPECTED_COUNTS
        assert metadata["production_changed"] is False
        assert set(predictions["season"]) == {2023, 2024, 2025}
        assert np.isfinite(predictions["prediction"]).all()

        blend = predictions[predictions["model_family"].eq("primary_blend")]
        assert set(blend["method"]) == ENSEMBLE_METHODS
        assert blend.groupby("method").size().eq(expected_rows).all()
        assert selections.groupby("model_family").size().eq(1).all()
        assert set(selections["model_family"]) == set(EXPECTED_COUNTS)
        assert set(origin_scores["origin"]) == set(range(2013, 2023))
        for model_family, candidate_count in EXPECTED_COUNTS.items():
            family = origin_scores[origin_scores["model_family"].eq(model_family)]
            assert family.groupby("origin").size().eq(candidate_count).all()

        all_scores = scores[
            scores["model_family"].eq("primary_blend")
            & scores["slice_type"].eq("all")
        ].set_index("method")
        baseline_rmse = all_scores.loc["current_single", "rmse"]
        for row in bootstrap.itertuples(index=False):
            expected_delta = all_scores.loc[row.method, "rmse"] - baseline_rmse
            assert np.isclose(row.rmse_delta, expected_delta)

    overall = pd.read_csv(STUDY_DIR / "results" / "overall_comparisons.csv")
    assert set(overall["method"]) == ENSEMBLE_METHODS - {"current_single"}
    manifest = json.loads(
        (STUDY_DIR / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["decision"] == "ridge_histgbm_screen_complete"
    assert manifest["production_changed"] is False
    print("verified temporal boundaries, candidate grids, scores, and summary")


if __name__ == "__main__":
    main()

