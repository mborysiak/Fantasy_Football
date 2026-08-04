"""Verify boundaries, prediction integrity, and summarized study decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
LEAGUES = {"dk": 1237, "beta": 1226}
ENSEMBLE_METHODS = {
    "current_single",
    "current_plus_extra_trees_equal4",
    "current_plus_catboost_equal4",
    "current_plus_both_equal5",
}


def main() -> None:
    for league, expected_rows in LEAGUES.items():
        directory = STUDY_DIR / f"results_{league}"
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(directory / "holdout_predictions.csv")
        selections = pd.read_csv(directory / "selected_parameters.csv")
        scores = pd.read_csv(directory / "scores.csv")
        bootstrap = pd.read_csv(directory / "player_cluster_bootstrap.csv")

        assert metadata["training_max_season"] == 2022
        assert metadata["holdout_seasons"] == [2023, 2024, 2025]
        assert metadata["tuning_origins"] == list(range(2013, 2023))
        assert metadata["candidate_count_per_family"] == 8
        assert metadata["production_changed"] is False
        assert set(predictions["season"]) == {2023, 2024, 2025}
        blend = predictions[predictions["model_family"].eq("primary_blend")]
        assert set(blend["method"]) == ENSEMBLE_METHODS
        assert blend.groupby("method").size().eq(expected_rows).all()
        assert selections.groupby("model_family").size().eq(1).all()
        assert set(selections["candidate_id"]) == {4, 7}
        assert np.isfinite(predictions["prediction"]).all()

        all_scores = scores[
            scores["model_family"].eq("primary_blend")
            & scores["slice_type"].eq("all")
        ].set_index("method")
        baseline_rmse = all_scores.loc["current_single", "rmse"]
        for row in bootstrap.itertuples(index=False):
            expected_delta = all_scores.loc[row.method, "rmse"] - baseline_rmse
            assert np.isclose(row.rmse_delta, expected_delta)

        extra_seasons = scores[
            scores["model_family"].eq("primary_blend")
            & scores["slice_type"].eq("season")
        ].pivot(index="slice_value", columns="method", values="rmse")
        assert (
            extra_seasons["current_plus_extra_trees_equal4"]
            < extra_seasons["current_single"]
        ).all()

    overall = pd.read_csv(STUDY_DIR / "results" / "overall_comparisons.csv")
    extra = overall[overall["method"].eq("current_plus_extra_trees_equal4")]
    cat = overall[overall["method"].eq("current_plus_catboost_equal4")]
    assert extra["rmse_delta"].lt(0).all()
    assert cat.loc[cat["league"].eq("beta"), "rmse_delta"].gt(0).all()
    manifest = json.loads(
        (STUDY_DIR / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["decision"] == "advance_extra_trees_as_research_shadow_candidate"
    assert manifest["production_changed"] is False
    seed_check = pd.read_csv(STUDY_DIR / "results" / "extra_trees_seed_robustness.csv")
    single_seed = seed_check[seed_check["variant"].eq("single_seed")]
    assert single_seed.groupby("league").size().eq(5).all()
    assert single_seed["rmse_delta"].lt(0).all()
    assert seed_check[seed_check["variant"].eq("five_seed_mean")]["rmse_delta"].lt(0).all()
    print("verified temporal boundaries, candidate counts, scores, and decisions")


if __name__ == "__main__":
    main()
