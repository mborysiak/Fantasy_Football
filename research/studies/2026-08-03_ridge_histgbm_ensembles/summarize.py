"""Create durable cross-league summaries for the Ridge/HistGBM screen."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
LEAGUES = ("dk", "beta")
BASELINE = "current_single"
CHALLENGERS = (
    "ridge_replaces_lasso_equal3",
    "lasso_ridge_split_linear_third",
    "current_plus_ridge_equal4",
    "histgbm_replaces_lightgbm_equal3",
    "current_plus_histgbm_equal4",
)
PRIMARY_METHODS = {
    "ridge": "ridge_replaces_lasso_equal3",
    "histgbm": "current_plus_histgbm_equal4",
}


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
    return pd.concat(rows, ignore_index=True).sort_values(
        ["league", "method", "slice_value"]
    )


def _verdict(
    overall: pd.DataFrame,
    seasons: pd.DataFrame,
    method: str,
) -> tuple[str, int]:
    pooled = overall[overall["method"].eq(method)]
    seasonal = seasons[seasons["method"].eq(method)]
    wins = int(seasonal["rmse_delta"].lt(0).sum())
    advances = pooled["rmse_delta"].lt(0).all() and wins >= 4
    return ("research_shadow_candidate" if advances else "reject", wins)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores = _load("scores.csv")
    bootstrap = _load("player_cluster_bootstrap.csv")
    selections = _load("selected_parameters.csv")
    correlations = _load("component_correlations.csv")

    overall = _comparison_table(scores, "all").drop(columns="slice_value")
    overall = overall.merge(
        bootstrap[
            [
                "league",
                "method",
                "bootstrap_low",
                "bootstrap_high",
                "player_clusters",
                "draws",
            ]
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
    ][["league", "model_family", "rows", "rmse", "mae", "bias", "spearman"]]
    standalone = standalone.sort_values(["league", "model_family"])

    ridge_verdict, ridge_wins = _verdict(
        overall, seasons, PRIMARY_METHODS["ridge"]
    )
    hist_verdict, hist_wins = _verdict(
        overall, seasons, PRIMARY_METHODS["histgbm"]
    )
    decision = {
        "decision": "ridge_histgbm_screen_complete",
        "ridge": ridge_verdict,
        "histgbm": hist_verdict,
        "primary_methods": PRIMARY_METHODS,
        "primary_rmse_deltas": {
            family: {
                row.league: float(row.rmse_delta)
                for row in overall[
                    overall["method"].eq(method)
                ].itertuples(index=False)
            }
            for family, method in PRIMARY_METHODS.items()
        },
        "primary_season_wins": {
            "ridge": ridge_wins,
            "histgbm": hist_wins,
        },
        "caveat": (
            "DK and beta share substantially the same player-seasons, and "
            "2023-2025 is a reused historical confirmation block."
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

    labels = {
        "ridge_replaces_lasso_equal3": "Ridge replaces Lasso",
        "lasso_ridge_split_linear_third": "Lasso/Ridge split linear third",
        "current_plus_ridge_equal4": "Ridge equal-fourths diagnostic",
        "histgbm_replaces_lightgbm_equal3": "HistGBM replaces LightGBM",
        "current_plus_histgbm_equal4": "HistGBM equal-fourths",
    }
    lines = [
        "# Ridge / histogram-gradient-boosting result",
        "",
        "Negative RMSE deltas favor the challenger. Intervals use a paired player-cluster bootstrap.",
        "",
        "| Scoring | Comparison | Baseline | Challenger | Delta | 95% interval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in overall.sort_values(["method", "league"]).itertuples(index=False):
        lines.append(
            f"| {row.league} | {labels[row.method]} | {row.baseline_rmse:.6f} | "
            f"{row.challenger_rmse:.6f} | {row.rmse_delta:+.6f} | "
            f"[{row.bootstrap_low:+.6f}, {row.bootstrap_high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Screen decision",
            "",
            f"Ridge primary verdict: `{ridge_verdict}` ({ridge_wins}/6 season cells improved).",
            f"HistGBM primary verdict: `{hist_verdict}` ({hist_wins}/6 season cells improved).",
            "",
            "Both scorings selected Ridge alpha 10, so the expanded regularization "
            "boundaries did not improve the pre-2023 selection score. Replacing Lasso "
            "improves MAE and reduces positive bias in both scorings, but Lasso/Ridge "
            "error correlations exceed 0.996, RB RMSE worsens slightly in both, and "
            "DK's player-cluster interval crosses zero. Ridge remains a shadow only.",
            "",
            "DK selected the shallow HistGBM schedule with zero L2; beta selected the "
            "deeper schedule with L2 10. The prespecified equal-four comparison worsens "
            "both scorings. Replacing LightGBM is mildly favorable in pooled RMSE but "
            "wins only three of six season cells and has intervals crossing zero. Reject "
            "HistGBM on this surface.",
            "",
            "This is a shadow screen on a reused confirmation block. No production files changed.",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
