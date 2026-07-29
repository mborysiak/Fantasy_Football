"""Create paired season-clustered comparisons for the active M4A run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPARISONS = (
    (
        "conditional_ppg",
        "rmse",
        "expert_consensus_hybrid",
        "expert_team_game_consensus",
    ),
    (
        "conditional_ppg",
        "rmse",
        "direct_lgbm_shallow",
        "consensus_recalibrated_ridge",
    ),
    (
        "conditional_ppg",
        "rmse",
        "direct_lgbm_shallow",
        "direct_ridge_full",
    ),
    (
        "conditional_ppg",
        "rmse",
        "direct_lgbm_shallow",
        "residual_lgbm_shallow",
    ),
    (
        "conditional_ppg",
        "rmse",
        "residual_ridge_full",
        "residual_ridge_compact",
    ),
    (
        "participation",
        "brier",
        "participation_lgbm_shallow",
        "participation_logistic_full",
    ),
    (
        "participation",
        "brier",
        "participation_logistic_full",
        "participation_logistic_compact",
    ),
)


def main() -> None:
    slices = pd.read_csv(RESULTS_DIR / "model_slice_summary.csv")
    rng = np.random.default_rng(1234)
    rows = []
    for target, metric, model, comparison in COMPARISONS:
        selected = slices[
            slices["target_name"].eq(target)
            & slices["metric"].eq(metric)
            & slices["slice_type"].eq("season")
        ]
        model_seasons = selected[
            selected["model_name"].eq(model)
        ].set_index("slice_value")["value"]
        comparison_seasons = selected[
            selected["model_name"].eq(comparison)
        ].set_index("slice_value")["value"]
        delta = (model_seasons - comparison_seasons).dropna()
        bootstrap = np.array(
            [
                rng.choice(delta.to_numpy(), len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "target_name": target,
                "metric": metric,
                "model_name": model,
                "comparison_model": comparison,
                "mean_season_delta": delta.mean(),
                "median_season_delta": delta.median(),
                "season_wins": int(delta.lt(0).sum()),
                "season_count": len(delta),
                "bootstrap_95_low": np.quantile(bootstrap, 0.025),
                "bootstrap_95_high": np.quantile(bootstrap, 0.975),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(
        RESULTS_DIR / "paired_season_comparisons.csv", index=False
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
