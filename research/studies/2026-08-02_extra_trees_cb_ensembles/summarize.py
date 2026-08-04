"""Create durable cross-league summaries for the challenger-model study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
LEAGUES = ("dk", "beta")
BASELINE = "current_single"
CHALLENGERS = (
    "current_plus_extra_trees_equal4",
    "current_plus_catboost_equal4",
    "current_plus_both_equal5",
)


def _load(name: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(STUDY_DIR / f"results_{league}" / name) for league in LEAGUES],
        ignore_index=True,
    )


def _comparison_table(scores: pd.DataFrame, slice_type: str) -> pd.DataFrame:
    blend = scores[
        scores["model_family"].eq("primary_blend")
        & scores["slice_type"].eq(slice_type)
    ].copy()
    keys = ["league", "slice_value"]
    baseline = blend[blend["method"].eq(BASELINE)][
        [*keys, "rows", "rmse", "mae", "bias", "spearman"]
    ].rename(
        columns={
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "bias": "baseline_bias",
            "spearman": "baseline_spearman",
        }
    )
    rows = []
    for method in CHALLENGERS:
        challenger = blend[blend["method"].eq(method)][
            [*keys, "rmse", "mae", "bias", "spearman"]
        ].rename(
            columns={
                "rmse": "challenger_rmse",
                "mae": "challenger_mae",
                "bias": "challenger_bias",
                "spearman": "challenger_spearman",
            }
        )
        comparison = baseline.merge(challenger, on=keys, validate="one_to_one")
        comparison.insert(1, "method", method)
        for metric in ("rmse", "mae", "bias", "spearman"):
            comparison[f"{metric}_delta"] = (
                comparison[f"challenger_{metric}"]
                - comparison[f"baseline_{metric}"]
            )
        rows.append(comparison)
    return pd.concat(rows, ignore_index=True).sort_values(["league", "method", "slice_value"])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores = _load("scores.csv")
    bootstrap = _load("player_cluster_bootstrap.csv")
    selections = _load("selected_parameters.csv")
    correlations = _load("component_correlations.csv")
    seed_check = pd.read_csv(RESULTS_DIR / "extra_trees_seed_robustness.csv")

    overall = _comparison_table(scores, "all").drop(columns="slice_value")
    overall = overall.merge(
        bootstrap[
            ["league", "method", "bootstrap_low", "bootstrap_high", "player_clusters", "draws"]
        ],
        on=["league", "method"],
        how="left",
        validate="one_to_one",
    )
    seasons = _comparison_table(scores, "season")
    positions = _comparison_table(scores, "position")

    standalone = scores[
        scores["method"].eq("challenger_standalone")
        & scores["slice_type"].eq("all")
    ].copy()
    standalone = standalone[
        ["league", "model_family", "rows", "rmse", "mae", "bias", "spearman"]
    ].sort_values(["league", "model_family"])

    extra_method = "current_plus_extra_trees_equal4"
    cat_method = "current_plus_catboost_equal4"
    extra_overall = overall[overall["method"].eq(extra_method)]
    cat_overall = overall[overall["method"].eq(cat_method)]
    extra_seasons = seasons[seasons["method"].eq(extra_method)]
    extra_position = positions[positions["method"].eq(extra_method)]
    extra_non_qb_wins = int(
        extra_position[extra_position["slice_value"].ne("QB")]["rmse_delta"].lt(0).sum()
    )

    decision = {
        "decision": "advance_extra_trees_as_research_shadow_candidate",
        "extra_trees": "advance_for_further_causal_validation; do_not_promote_to_production_yet",
        "catboost": "reject_as_a_fourth_equal_weight_member",
        "equal_fifths": "reject_because_both_challengers_did_not_pass_independently",
        "extra_trees_overall_rmse_delta": {
            row.league: float(row.rmse_delta)
            for row in extra_overall.itertuples(index=False)
        },
        "catboost_overall_rmse_delta": {
            row.league: float(row.rmse_delta)
            for row in cat_overall.itertuples(index=False)
        },
        "extra_trees_league_season_wins": int(extra_seasons["rmse_delta"].lt(0).sum()),
        "extra_trees_league_season_cells": len(extra_seasons),
        "extra_trees_non_qb_position_wins": extra_non_qb_wins,
        "extra_trees_non_qb_position_cells": int(
            extra_position["slice_value"].ne("QB").sum()
        ),
        "extra_trees_single_seed_wins": int(
            seed_check[seed_check["variant"].eq("single_seed")]["rmse_delta"].lt(0).sum()
        ),
        "extra_trees_single_seed_cells": int(
            seed_check["variant"].eq("single_seed").sum()
        ),
        "caveat": (
            "DK and beta are alternate scoring views of substantially the same player-seasons, "
            "and 2023-2025 is a reused historical confirmation block."
        ),
        "production_changed": False,
    }

    overall.to_csv(RESULTS_DIR / "overall_comparisons.csv", index=False)
    seasons.to_csv(RESULTS_DIR / "season_comparisons.csv", index=False)
    positions.to_csv(RESULTS_DIR / "position_comparisons.csv", index=False)
    standalone.to_csv(RESULTS_DIR / "standalone_scores.csv", index=False)
    selections.to_csv(RESULTS_DIR / "selected_parameters.csv", index=False)
    correlations.to_csv(RESULTS_DIR / "component_correlations.csv", index=False)
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Extra Trees / CatBoost ensemble result",
        "",
        "## Primary RMSE comparisons",
        "",
        "Negative deltas favor the challenger. Confidence intervals use a paired player-cluster bootstrap.",
        "",
        "| Scoring | Challenger | Baseline | Challenger | Delta | 95% interval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "current_plus_extra_trees_equal4": "Extra Trees equal-fourths",
        "current_plus_catboost_equal4": "CatBoost equal-fourths",
        "current_plus_both_equal5": "Both equal-fifths (secondary)",
    }
    for row in overall.sort_values(["method", "league"]).itertuples(index=False):
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
            "Extra Trees advances as a research shadow candidate, not a production change. "
            "It improved pooled RMSE in DK and beta and won all six scoring-system/season cells. "
            "Its non-QB slices also improved in both systems, so the result is not solely a QB effect. "
            "All ten single-seed robustness cells remained favorable. The gain is small, DK uncertainty "
            "overlaps zero, and the confirmation seasons have been reused.",
            "",
            "CatBoost is rejected as an equal-weight fourth member: its small DK gain reversed in beta. "
            "The equal-fifths blend is also rejected under the prespecified rule that both challengers "
            "must add value independently.",
            "",
            "No production files were changed.",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
