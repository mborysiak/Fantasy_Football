"""Summarize expanded boosting grids across DK and beta scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
LEAGUES = ("dk", "beta")


def _load(name: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(STUDY_DIR / f"results_{league}" / name) for league in LEAGUES],
        ignore_index=True,
    )


def _score_comparisons(scores: pd.DataFrame, slice_type: str) -> pd.DataFrame:
    blend = scores[
        scores["model_family"].eq("primary_blend")
        & scores["slice_type"].eq(slice_type)
    ].copy()
    rows = []
    comparisons = (
        ("expanded_lgbm_replacement_equal3", "current_single"),
        ("expanded_catboost_equal4", "current_single"),
        ("current_plus_extra_trees_equal4", "current_single"),
        ("expanded_lgbm_plus_extra_trees_equal4", "current_single"),
        ("expanded_lgbm_plus_extra_trees_equal4", "current_plus_extra_trees_equal4"),
    )
    keys = ["league", "slice_value"]
    metrics = ("rmse", "mae", "bias", "spearman")
    for challenger_method, baseline_method in comparisons:
        baseline = blend[blend["method"].eq(baseline_method)][
            [*keys, "rows", *metrics]
        ].rename(columns={metric: f"baseline_{metric}" for metric in metrics})
        challenger = blend[blend["method"].eq(challenger_method)][
            [*keys, *metrics]
        ].rename(columns={metric: f"challenger_{metric}" for metric in metrics})
        comparison = baseline.merge(challenger, on=keys, validate="one_to_one")
        comparison.insert(1, "method", challenger_method)
        comparison.insert(2, "baseline_method", baseline_method)
        for metric in metrics:
            comparison[f"{metric}_delta"] = (
                comparison[f"challenger_{metric}"]
                - comparison[f"baseline_{metric}"]
            )
        rows.append(comparison)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["league", "method", "baseline_method", "slice_value"]
    )


def _selection_comparison(
    selections: pd.DataFrame,
    origins: pd.DataFrame,
) -> pd.DataFrame:
    ranked = (
        origins.groupby(
            ["league", "model_family", "candidate_id", "candidate_source", "parameters_json"],
            as_index=False,
        )["rmse"]
        .mean()
        .rename(columns={"rmse": "selection_score"})
    )
    original = (
        ranked[ranked["candidate_source"].eq("original")]
        .sort_values(["league", "model_family", "selection_score", "candidate_id"])
        .groupby(["league", "model_family"], as_index=False)
        .first()
        .rename(
            columns={
                "candidate_id": "original_winner_id",
                "selection_score": "original_winner_score",
                "parameters_json": "original_winner_parameters_json",
            }
        )
    )
    output = selections.merge(
        original[
            [
                "league",
                "model_family",
                "original_winner_id",
                "original_winner_score",
                "original_winner_parameters_json",
            ]
        ],
        on=["league", "model_family"],
        validate="one_to_one",
    )
    output["selection_score_delta_vs_original"] = (
        output["selection_score"] - output["original_winner_score"]
    )
    return output.sort_values(["league", "model_family"])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores = _load("scores.csv")
    bootstrap = _load("player_cluster_bootstrap.csv")
    selections = _load("selected_parameters.csv")
    origins = _load("origin_candidate_scores.csv")
    correlations = _load("component_correlations.csv")

    overall = _score_comparisons(scores, "all").drop(columns="slice_value")
    overall = overall.merge(
        bootstrap[
            [
                "league",
                "method",
                "baseline_method",
                "bootstrap_low",
                "bootstrap_high",
                "player_clusters",
                "draws",
            ]
        ],
        on=["league", "method", "baseline_method"],
        how="left",
        validate="one_to_one",
    )
    seasons = _score_comparisons(scores, "season")
    positions = _score_comparisons(scores, "position")
    selection_comparison = _selection_comparison(selections, origins)
    rankings = (
        origins.groupby(
            ["league", "model_family", "candidate_id", "candidate_source", "parameters_json"],
            as_index=False,
        )["rmse"]
        .mean()
        .rename(columns={"rmse": "selection_score"})
        .sort_values(["league", "model_family", "selection_score", "candidate_id"])
    )

    lgbm = overall[
        overall["method"].eq("expanded_lgbm_replacement_equal3")
        & overall["baseline_method"].eq("current_single")
    ]
    catboost = overall[
        overall["method"].eq("expanded_catboost_equal4")
        & overall["baseline_method"].eq("current_single")
    ]
    extra_interaction = overall[
        overall["method"].eq("expanded_lgbm_plus_extra_trees_equal4")
        & overall["baseline_method"].eq("current_plus_extra_trees_equal4")
    ]
    manifest = {
        "decision": "retain_existing_boosting_grids_and_parameters",
        "lightgbm": (
            "DK retained the incumbent 0.05/100 configuration. Beta selected 0.01/500 "
            "pre-2023 but worsened pooled 2023-2025 RMSE, so do not change the beta model."
        ),
        "catboost": (
            "Both scorings retained the original 0.03/300 configuration; expanded tuning "
            "did not rescue CatBoost as a fourth member."
        ),
        "extra_trees": (
            "The prior Extra Trees result remains the model-family finalist. Replacing "
            "beta LightGBM with the expanded selection slightly weakens the Extra Trees blend."
        ),
        "lgbm_overall_rmse_delta": {
            row.league: float(row.rmse_delta) for row in lgbm.itertuples(index=False)
        },
        "catboost_overall_rmse_delta": {
            row.league: float(row.rmse_delta) for row in catboost.itertuples(index=False)
        },
        "expanded_lgbm_extra_interaction_rmse_delta": {
            row.league: float(row.rmse_delta)
            for row in extra_interaction.itertuples(index=False)
        },
        "production_changed": False,
    }

    overall.to_csv(RESULTS_DIR / "overall_comparisons.csv", index=False)
    seasons.to_csv(RESULTS_DIR / "season_comparisons.csv", index=False)
    positions.to_csv(RESULTS_DIR / "position_comparisons.csv", index=False)
    selection_comparison.to_csv(RESULTS_DIR / "selection_comparisons.csv", index=False)
    rankings.to_csv(RESULTS_DIR / "candidate_rankings.csv", index=False)
    correlations.to_csv(RESULTS_DIR / "component_correlations.csv", index=False)
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    primary = overall[
        (
            overall["method"].isin(
                ("expanded_lgbm_replacement_equal3", "expanded_catboost_equal4")
            )
        )
        & overall["baseline_method"].eq("current_single")
    ].sort_values(["method", "league"])
    labels = {
        "expanded_lgbm_replacement_equal3": "Expanded LightGBM replacement",
        "expanded_catboost_equal4": "Expanded CatBoost equal-four",
    }
    lines = [
        "# Boosting grid expansion result",
        "",
        "Negative RMSE deltas favor the challenger.",
        "",
        "| Scoring | Test | Baseline | Challenger | RMSE delta | 95% player-cluster interval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.league} | {labels[row.method]} | {row.baseline_rmse:.6f} | "
            f"{row.challenger_rmse:.6f} | {row.rmse_delta:+.6f} | "
            f"[{row.bootstrap_low:+.6f}, {row.bootstrap_high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Retain the existing LightGBM grid and selected parameters. DK selected the exact incumbent "
            "0.05 / 100-tree model. Beta selected the new 0.01 / 500-tree schedule by 0.001897 mean "
            "pre-2023 seasonal RMSE, but the resulting blend worsened pooled 2023-2025 RMSE by 0.000583. "
            "It slightly improved 2023 and 2024 and worsened 2025; its player-cluster interval crosses zero.",
            "",
            "Retain CatBoost's rejection. Both scorings selected the original 0.03 / 300-iteration "
            "candidate despite the expanded schedules and boundary search, reproducing the earlier "
            "small DK gain and beta loss.",
            "",
            "The expanded beta LightGBM also worsened the Extra Trees blend by 0.000414 RMSE versus "
            "the incumbent-LightGBM Extra Trees blend. Extra Trees therefore remains the only model-family "
            "shadow candidate from these tests. No production files were changed.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

