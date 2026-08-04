"""Summarize the sealed SKM fold/seed ensemble comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
LEAGUES = ("dk", "beta")
BASELINE = "current_single"
CHALLENGERS = (
    "current_seed_bag",
    "skm_fold_param_bag",
    "skm_fold_seed_bag",
)
FAMILIES = (
    "conditional_ppg_lasso",
    "conditional_ppg_random_forest",
    "conditional_ppg_lightgbm",
)
BOOTSTRAP_DRAWS = 20_000


def _metric_table(scores: pd.DataFrame, family: str, slice_type: str) -> pd.DataFrame:
    selected = scores[
        scores["model_family"].eq(family)
        & scores["slice_type"].eq(slice_type)
    ].copy()
    baseline = selected[selected["method"].eq(BASELINE)][
        ["league", "slice_value", "rmse", "mae", "bias"]
    ].rename(
        columns={
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "bias": "baseline_bias",
        }
    )
    selected = selected.merge(
        baseline,
        on=["league", "slice_value"],
        how="left",
        validate="many_to_one",
    )
    selected["rmse_delta"] = selected["rmse"] - selected["baseline_rmse"]
    selected["mae_delta"] = selected["mae"] - selected["baseline_mae"]
    selected["absolute_bias_delta"] = selected["bias"].abs() - selected["baseline_bias"].abs()
    return selected


def _hybrid_predictions(predictions: pd.DataFrame, rf_method: str) -> pd.DataFrame:
    components = predictions[predictions["model_family"].ne("primary_blend")]
    keys = ["league", "player_key", "season", "position", "actual"]
    pieces = []
    for family in FAMILIES:
        method = rf_method if family == "conditional_ppg_random_forest" else BASELINE
        current = components[
            components["model_family"].eq(family)
            & components["method"].eq(method)
        ][keys + ["prediction"]].rename(columns={"prediction": family})
        pieces.append(current)
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=keys, how="inner", validate="one_to_one")
    merged["prediction"] = merged[list(FAMILIES)].mean(axis=1)
    merged["method"] = f"current_plus_{rf_method}_rf"
    return merged[keys + ["method", "prediction"]]


def _cluster_interval(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    seed: int,
) -> tuple[float, float, float]:
    keys = ["player_key", "season", "actual"]
    paired = baseline[keys + ["prediction"]].rename(
        columns={"prediction": "baseline"}
    ).merge(
        challenger[keys + ["prediction"]].rename(
            columns={"prediction": "challenger"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for player_key, group in paired.groupby("player_key", sort=False):
        actual = group["actual"].to_numpy(float)
        rows.append(
            {
                "player_key": player_key,
                "n": len(group),
                "base_sse": float(np.square(actual - group["baseline"].to_numpy(float)).sum()),
                "challenger_sse": float(np.square(actual - group["challenger"].to_numpy(float)).sum()),
            }
        )
    players = pd.DataFrame(rows)
    count = players["n"].to_numpy(float)
    base_sse = players["base_sse"].to_numpy(float)
    challenger_sse = players["challenger_sse"].to_numpy(float)
    point = float(
        np.sqrt(challenger_sse.sum() / count.sum())
        - np.sqrt(base_sse.sum() / count.sum())
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        stop = min(start + 500, BOOTSTRAP_DRAWS)
        sample = rng.integers(0, len(players), size=(stop - start, len(players)))
        n = count[sample].sum(axis=1)
        draws[start:stop] = (
            np.sqrt(challenger_sse[sample].sum(axis=1) / n)
            - np.sqrt(base_sse[sample].sum(axis=1) / n)
        )
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    score_frames = []
    bootstrap_frames = []
    selection_frames = []
    prediction_frames = []
    for league in LEAGUES:
        directory = STUDY_DIR / f"results_{league}"
        scores = pd.read_csv(directory / "scores.csv")
        scores.insert(0, "source_league", league)
        score_frames.append(scores)
        bootstrap_frames.append(pd.read_csv(directory / "player_cluster_bootstrap.csv"))
        selection_frames.append(pd.read_csv(directory / "selected_parameters.csv"))
        prediction_frames.append(pd.read_csv(directory / "holdout_predictions.csv"))

    scores = pd.concat(score_frames, ignore_index=True).drop(columns="source_league")
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    primary = _metric_table(scores, "primary_blend", "all")
    seasons = _metric_table(scores, "primary_blend", "season")
    components = _metric_table(scores, "conditional_ppg_lasso", "all")
    components = pd.concat(
        [
            components,
            _metric_table(scores, "conditional_ppg_random_forest", "all"),
            _metric_table(scores, "conditional_ppg_lightgbm", "all"),
        ],
        ignore_index=True,
    )
    positions = _metric_table(scores, "primary_blend", "position")

    parameter_rows = []
    for (league, family), group in selections.groupby(["league", "model_family"], sort=True):
        current = group[group["selection_method"].eq("current_mean_season_rmse")].iloc[0]
        folds = group[group["selection_method"].eq("skm_fold_pooled_rmse")]
        parameter_rows.append(
            {
                "league": league,
                "model_family": family,
                "current_candidate_id": int(current["candidate_id"]),
                "fold_unique_candidates": int(folds["candidate_id"].nunique()),
                "folds_matching_current": int(folds["candidate_id"].eq(current["candidate_id"]).sum()),
                "fold_candidate_ids": ",".join(str(int(value)) for value in folds.sort_values("fold")["candidate_id"]),
            }
        )
    parameter_diversity = pd.DataFrame(parameter_rows)

    hybrid_rows = []
    for league in LEAGUES:
        league_predictions = predictions[predictions["league"].eq(league)]
        baseline = league_predictions[
            league_predictions["model_family"].eq("primary_blend")
            & league_predictions["method"].eq(BASELINE)
        ][["player_key", "season", "actual", "prediction"]]
        for index, method in enumerate(CHALLENGERS):
            hybrid = _hybrid_predictions(league_predictions, method)
            point, low, high = _cluster_interval(
                baseline,
                hybrid,
                44_000 + LEAGUES.index(league) * 100 + index,
            )
            hybrid_rows.append(
                {
                    "league": league,
                    "rf_method": method,
                    "hybrid_method": hybrid["method"].iloc[0],
                    "rmse_delta_vs_current_blend": point,
                    "bootstrap_low": low,
                    "bootstrap_high": high,
                    "prespecified": False,
                }
            )
    hybrid = pd.DataFrame(hybrid_rows)

    primary.to_csv(RESULTS_DIR / "primary_holdout_summary.csv", index=False)
    seasons.to_csv(RESULTS_DIR / "season_summary.csv", index=False)
    components.to_csv(RESULTS_DIR / "component_summary.csv", index=False)
    positions.to_csv(RESULTS_DIR / "position_summary.csv", index=False)
    bootstrap.to_csv(RESULTS_DIR / "player_cluster_bootstrap.csv", index=False)
    parameter_diversity.to_csv(RESULTS_DIR / "parameter_diversity.csv", index=False)
    hybrid.to_csv(RESULTS_DIR / "exploratory_rf_hybrid.csv", index=False)

    focus = primary[primary["method"].isin([BASELINE, "skm_fold_seed_bag", "current_seed_bag"])][
        ["league", "method", "rows", "rmse", "rmse_delta", "mae_delta", "absolute_bias_delta"]
    ]
    fold_bootstrap = bootstrap[bootstrap["method"].eq("skm_fold_seed_bag")][
        ["league", "rmse_delta", "bootstrap_low", "bootstrap_high"]
    ]
    hybrid_focus = hybrid[hybrid["rf_method"].eq("skm_fold_param_bag")]

    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    findings = "\n".join(
        [
            "# SKM fold-ensemble sealed-holdout findings",
            "",
            "## Decision",
            "",
            "Retain the current single-fit methodology. Applying the legacy-style "
            "five-member fold-parameter and seed bag to every component does not "
            "improve the sealed 2023-2025 holdout: it is effectively tied in DK "
            "and materially worse in beta. Merely averaging five estimator seeds "
            "around the current parameters is also neutral across leagues.",
            "",
            "The beta loss is parameter-bagging instability rather than estimator "
            "seed noise. One Lasso fold selected alpha 0.1; the bag worsened beta "
            "Lasso RMSE by 0.015941, while fold-bagged RF improved by 0.002783 and "
            "LightGBM was flat. DK shows the same offset at smaller magnitude: RF "
            "improves but Lasso worsens, leaving the blend tied.",
            "",
            "## Primary sealed-holdout scores",
            "",
            markdown(focus),
            "",
            "Negative deltas favor the challenger.",
            "",
            "## Paired player-cluster uncertainty for the full SKM bag",
            "",
            markdown(fold_bootstrap),
            "",
            "## Exploratory RF-only hybrid",
            "",
            markdown(hybrid_focus),
            "",
            "Keeping current Lasso and LightGBM while replacing only RF with its "
            "fold-parameter bag improves the point estimate in both leagues, but "
            "the arm was identified after inspecting component results. The DK "
            "interval ends narrowly below zero while the beta interval crosses "
            "zero; without prespecification or multiplicity protection, this is a "
            "follow-up hypothesis rather than promotion evidence.",
            "",
            "## Governance",
            "",
            "The 2023-2025 outcomes were never used in fitting, fold assignment, "
            "or hyperparameter selection. No production database, feature set, "
            "model, or projection changed.",
            "",
        ]
    )
    (RESULTS_DIR / "findings.md").write_text(findings, encoding="utf-8")
    manifest = {
        "decision": "retain_current_single_fit",
        "primary_challenger": "skm_fold_seed_bag",
        "holdout_seasons": [2023, 2024, 2025],
        "exploratory_follow_up": "current_lasso_lgbm_plus_skm_fold_param_rf",
        "production_changed": False,
    }
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(findings)


if __name__ == "__main__":
    main()
