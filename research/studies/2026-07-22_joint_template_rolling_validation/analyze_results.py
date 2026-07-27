"""Create causal centering policy and season-clustered method comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "results" / "production_oos"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

from run_validation import binary_auc, grouped_summary  # noqa: E402


BOOTSTRAP_REPEATS = 500
BOOTSTRAP_SEED = 20260722


def build_rolling_centering_policy(predictions):
    source = predictions[
        predictions.method.isin(
            ["adaptive_full_centered", "adaptive_full_uncentered"]
        )
    ]
    selected = []
    choices = []
    for season in range(2020, 2026):
        for pos in ["QB", "RB", "WR", "TE"]:
            prior = source[
                source.season.between(2017, season - 1)
                & source.pos.eq(pos)
            ]
            method_crps = prior.groupby("method").contribution_crps.mean()
            chosen_method = method_crps.idxmin()
            choices.append(
                {
                    "season": season,
                    "pos": pos,
                    "chosen_method": chosen_method,
                    "prior_target_rows": len(prior) // 2,
                    "centered_prior_contribution_crps": method_crps[
                        "adaptive_full_centered"
                    ],
                    "uncentered_prior_contribution_crps": method_crps[
                        "adaptive_full_uncentered"
                    ],
                }
            )
            rows = source[
                source.season.eq(season)
                & source.pos.eq(pos)
                & source.method.eq(chosen_method)
            ].copy()
            rows["source_method"] = chosen_method
            rows["method"] = "rolling_position_centering"
            selected.append(rows)
    return pd.concat(selected, ignore_index=True), pd.DataFrame(choices)


def metric_values(frame):
    ppg_bias = (frame.ppg_mean - frame.observed_ppg).mean()
    contribution_bias = (
        frame.contribution_mean - frame.observed_contribution
    ).mean()
    ppg_coverage = frame.ppg_80_covered.mean()
    impact_auc = binary_auc(frame.observed_impact, frame.prob_impact)
    plus3_auc = binary_auc(frame.observed_plus3, frame.prob_plus3)
    plus5_auc = binary_auc(frame.observed_plus5, frame.prob_plus5)
    tail_spearman = spearmanr(
        frame.contribution_tail_lift,
        frame.observed_contribution - frame.contribution_mean,
    ).statistic
    return {
        "ppg_crps": frame.ppg_crps.mean(),
        "contribution_crps": frame.contribution_crps.mean(),
        "absolute_ppg_bias": abs(ppg_bias),
        "absolute_contribution_bias": abs(contribution_bias),
        "ppg_coverage_error": abs(ppg_coverage - 0.80),
        "plus3_brier": np.square(
            frame.prob_plus3 - frame.observed_plus3
        ).mean(),
        "impact_brier": np.square(
            frame.prob_impact - frame.observed_impact
        ).mean(),
        "plus3_auc": plus3_auc,
        "plus5_auc": plus5_auc,
        "impact_auc": impact_auc,
        "tail_lift_surprise_spearman": tail_spearman,
    }


HIGHER_IS_BETTER = {
    "plus3_auc",
    "plus5_auc",
    "impact_auc",
    "tail_lift_surprise_spearman",
}


def improvement(candidate, baseline, metric):
    if metric in HIGHER_IS_BETTER:
        return candidate[metric] - baseline[metric]
    return baseline[metric] - candidate[metric]


def cluster_bootstrap_comparison(
    candidate,
    baseline,
    comparison,
    rng,
):
    seasons = np.array(sorted(set(candidate.season) & set(baseline.season)))
    candidate = candidate[candidate.season.isin(seasons)]
    baseline = baseline[baseline.season.isin(seasons)]
    point_candidate = metric_values(candidate)
    point_baseline = metric_values(baseline)
    draws = {metric: [] for metric in point_candidate}

    for _ in range(BOOTSTRAP_REPEATS):
        sampled_seasons = rng.choice(seasons, size=len(seasons), replace=True)
        candidate_draw = pd.concat(
            [candidate[candidate.season.eq(season)] for season in sampled_seasons],
            ignore_index=True,
        )
        baseline_draw = pd.concat(
            [baseline[baseline.season.eq(season)] for season in sampled_seasons],
            ignore_index=True,
        )
        candidate_metrics = metric_values(candidate_draw)
        baseline_metrics = metric_values(baseline_draw)
        for metric in draws:
            draws[metric].append(
                improvement(candidate_metrics, baseline_metrics, metric)
            )

    rows = []
    for metric, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        point_improvement = improvement(
            point_candidate, point_baseline, metric
        )
        rows.append(
            {
                "comparison": comparison,
                "candidate": candidate.method.iloc[0],
                "baseline": baseline.method.iloc[0],
                "season_count": len(seasons),
                "target_rows": len(candidate),
                "metric": metric,
                "candidate_value": point_candidate[metric],
                "baseline_value": point_baseline[metric],
                "improvement": point_improvement,
                "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "probability_better": np.mean(values > 0),
            }
        )
    return rows


def write_recent_upside_diagnostics(predictions):
    production = predictions[
        predictions.season.ge(2020)
        & predictions.method.eq("adaptive_full_centered")
    ].copy()

    profile = grouped_summary(
        production[production.pos.isin(["RB", "WR", "TE"])],
        ["pos", "experience_group"],
    )
    profile.to_csv(
        RESULTS / "recent_upside_by_position_experience.csv",
        index=False,
    )

    calibration_rows = []
    bins = [-1e-9, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 1.0]
    for event, probability_col, outcome_col in [
        ("+3_ppg", "prob_plus3", "observed_plus3"),
        ("+5_ppg", "prob_plus5", "observed_plus5"),
        ("impact", "prob_impact", "observed_impact"),
    ]:
        event_frame = production.copy()
        event_frame["probability_bin"] = pd.cut(
            event_frame[probability_col],
            bins=bins,
            include_lowest=True,
        )
        calibration = (
            event_frame.groupby("probability_bin", observed=True)
            .agg(
                n=(outcome_col, "size"),
                predicted=(probability_col, "mean"),
                observed=(outcome_col, "mean"),
            )
            .reset_index()
        )
        calibration.insert(0, "event", event)
        calibration_rows.append(calibration)
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        RESULTS / "recent_upside_probability_bins.csv",
        index=False,
    )

    production["impact_probability_quintile"] = pd.qcut(
        production.prob_impact.rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    impact_quintiles = (
        production.groupby("impact_probability_quintile", observed=True)
        .agg(
            n=("player", "size"),
            predicted_impact=("prob_impact", "mean"),
            observed_impact=("observed_impact", "mean"),
            observed_plus3=("observed_plus3", "mean"),
            observed_residual=("observed_residual", "mean"),
            observed_contribution=("observed_contribution", "mean"),
        )
        .reset_index()
    )
    impact_quintiles.to_csv(
        RESULTS / "recent_impact_probability_quintiles.csv",
        index=False,
    )


def main():
    predictions = pd.read_csv(RESULTS / "target_predictions.csv")
    rolling, choices = build_rolling_centering_policy(predictions)
    rolling.to_csv(RESULTS / "rolling_policy_predictions.csv", index=False)
    choices.to_csv(RESULTS / "rolling_centering_choices.csv", index=False)

    recent = predictions[predictions.season.ge(2020)]
    core = predictions[predictions.season.ge(2017)]
    summary = pd.concat(
        [
            grouped_summary(recent, ["method"]).assign(period="2020-2025"),
            grouped_summary(rolling, ["method"]).assign(period="2020-2025"),
            grouped_summary(
                rolling[rolling.pos.isin(["RB", "WR", "TE"])],
                ["method", "pos", "experience_group"],
            ).assign(period="2020-2025_by_experience"),
        ],
        ignore_index=True,
    )
    summary.to_csv(RESULTS / "summary_rolling_policy.csv", index=False)
    write_recent_upside_diagnostics(predictions)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [
        (
            rolling,
            recent[recent.method.eq("adaptive_full_centered")],
            "recent_rolling_vs_production_centered",
        ),
        (
            recent[recent.method.eq("adaptive_full_uncentered")],
            recent[recent.method.eq("adaptive_full_centered")],
            "recent_uncentered_vs_production_centered",
        ),
        (
            recent[recent.method.eq("adaptive_full_centered")],
            recent[recent.method.eq("legacy_2x_centered")],
            "recent_production_vs_legacy",
        ),
        (
            recent[recent.method.eq("adaptive_full_centered")],
            recent[recent.method.eq("adaptive_projection_centered")],
            "recent_full_vs_projection_only",
        ),
        (
            core[core.method.eq("adaptive_full_uncentered")],
            core[core.method.eq("adaptive_full_centered")],
            "core_uncentered_vs_production_centered",
        ),
    ]
    rows = []
    for candidate, baseline, label in comparisons:
        rows.extend(
            cluster_bootstrap_comparison(
                candidate,
                baseline,
                label,
                rng,
            )
        )
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(
        RESULTS / "paired_season_bootstrap.csv",
        index=False,
    )
    print(choices.to_string(index=False))
    print(
        summary[
            [
                "period",
                "method",
                "n",
                "ppg_crps",
                "ppg_bias",
                "ppg_80_coverage",
                "contribution_crps",
                "contribution_bias",
                "impact_auc",
                "tail_lift_surprise_spearman",
            ]
        ].head(8).round(4).to_string(index=False)
    )
    print(
        bootstrap[
            bootstrap.metric.isin(
                [
                    "ppg_crps",
                    "contribution_crps",
                    "absolute_ppg_bias",
                    "ppg_coverage_error",
                    "impact_auc",
                ]
            )
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
