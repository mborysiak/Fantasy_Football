"""Verify sealed-holdout and output integrity for the SKM comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent


def main() -> None:
    for league, expected_rows in (("dk", 1237), ("beta", 1226)):
        directory = STUDY_DIR / f"results_{league}"
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(directory / "holdout_predictions.csv")
        selections = pd.read_csv(directory / "selected_parameters.csv")
        assert metadata["training_max_season"] == 2022
        assert metadata["holdout_seasons"] == [2023, 2024, 2025]
        assert metadata["production_changed"] is False
        assert set(predictions["season"]) == {2023, 2024, 2025}
        assert predictions[
            predictions["model_family"].eq("primary_blend")
        ].groupby("method").size().eq(expected_rows).all()
        assert selections["selection_method"].eq("current_mean_season_rmse").sum() == 3
        assert selections["selection_method"].eq("skm_fold_pooled_rmse").sum() == 15
        lasso = predictions[predictions["model_family"].eq("conditional_ppg_lasso")]
        lasso_wide = lasso.pivot(
            index=["player_key", "season"], columns="method", values="prediction"
        )
        assert np.allclose(lasso_wide["current_single"], lasso_wide["current_seed_bag"])
        lightgbm = predictions[predictions["model_family"].eq("conditional_ppg_lightgbm")]
        lightgbm_wide = lightgbm.pivot(
            index=["player_key", "season"], columns="method", values="prediction"
        )
        assert np.allclose(lightgbm_wide["current_single"], lightgbm_wide["current_seed_bag"])
    manifest = json.loads(
        (STUDY_DIR / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["decision"] == "retain_current_single_fit"
    assert manifest["production_changed"] is False
    print("verified sealed holdout, fold counts, deterministic controls, and decision")


if __name__ == "__main__":
    main()
