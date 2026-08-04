"""Post-selection Extra Trees estimator-seed sensitivity check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_validation import (
    CURRENT_FAMILIES,
    DATABASES,
    HOLDOUT_SEASONS,
    MODEL_GRIDS,
    STUDY_DIR,
    _fit_predict,
    _load_current_predictions,
    _load_target,
)


SEEDS = (1234, 1335, 1436, 1537, 1638)
MODEL_NAME = "conditional_ppg_extra_trees"


def main() -> None:
    rows = []
    for league in DATABASES:
        target = _load_target(league)
        train = target[target["season"].lt(min(HOLDOUT_SEASONS))].copy()
        test = target[target["season"].isin(HOLDOUT_SEASONS)].copy()
        current = _load_current_predictions(league, test)
        current_prediction = current[list(CURRENT_FAMILIES)].mean(axis=1).to_numpy(float)
        actual = current["actual"].to_numpy(float)
        selected = pd.read_csv(STUDY_DIR / f"results_{league}" / "selected_parameters.csv")
        selected = selected[selected["model_family"].eq(MODEL_NAME)].iloc[0]
        candidate_id = int(selected["candidate_id"])
        parameters = MODEL_GRIDS[MODEL_NAME][candidate_id]
        seed_predictions = []
        baseline_rmse = float(np.sqrt(np.mean(np.square(actual - current_prediction))))
        for seed in SEEDS:
            print(f"{league}: Extra Trees robustness seed {seed}", flush=True)
            prediction = _fit_predict(
                train,
                test,
                MODEL_NAME,
                parameters,
                estimator_seed=seed,
            )
            seed_predictions.append(prediction)
            ensemble = (3.0 * current_prediction + prediction) / 4.0
            challenger_rmse = float(np.sqrt(np.mean(np.square(actual - ensemble))))
            rows.append(
                {
                    "league": league,
                    "variant": "single_seed",
                    "seed": seed,
                    "candidate_id": candidate_id,
                    "baseline_rmse": baseline_rmse,
                    "challenger_rmse": challenger_rmse,
                    "rmse_delta": challenger_rmse - baseline_rmse,
                    "parameters_json": json.dumps(parameters, sort_keys=True),
                }
            )
        seed_bag = np.mean(seed_predictions, axis=0)
        ensemble = (3.0 * current_prediction + seed_bag) / 4.0
        challenger_rmse = float(np.sqrt(np.mean(np.square(actual - ensemble))))
        rows.append(
            {
                "league": league,
                "variant": "five_seed_mean",
                "seed": pd.NA,
                "candidate_id": candidate_id,
                "baseline_rmse": baseline_rmse,
                "challenger_rmse": challenger_rmse,
                "rmse_delta": challenger_rmse - baseline_rmse,
                "parameters_json": json.dumps(parameters, sort_keys=True),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(STUDY_DIR / "results" / "extra_trees_seed_robustness.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

