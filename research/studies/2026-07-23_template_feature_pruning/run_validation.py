"""Strict rolling backward ablation of weekly-template match features."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CONTEXT_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-23_template_context_ablation"
    / "run_validation.py"
)
CONTEXT_SPEC = importlib.util.spec_from_file_location(
    "template_context_ablation",
    CONTEXT_STUDY,
)
if CONTEXT_SPEC is None or CONTEXT_SPEC.loader is None:
    raise ImportError(f"Could not import context validation from {CONTEXT_STUDY}")
context = importlib.util.module_from_spec(CONTEXT_SPEC)
sys.modules[CONTEXT_SPEC.name] = context
CONTEXT_SPEC.loader.exec_module(context)
base = context.base
builder = context.builder


RESULTS = Path(__file__).resolve().parent / "results"
BASELINE_METHOD = "full"
RECENCY_HALF_LIFE = 12.0
DEVELOPMENT_END = 2022
META_SELECTION_START = 2021
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260724

PERIODS = {
    "development_2017_2022": (2017, 2022),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}

LOWER_IS_BETTER = [
    "ppg_crps",
    "contribution_crps",
    "played_crps",
    "plus3_brier_row",
    "plus5_brier_row",
    "impact_brier_row",
    "zero_brier_row",
    "extended_absence_brier_row",
]

FEATURE_FAMILIES = {
    "exp_interaction": {
        pos: {"projection_x_exp"} for pos in builder.POSITIONS
    },
    "adp_rank": {
        pos: {"adp_rank_pct"} for pos in builder.POSITIONS
    },
    "market_gap": {
        pos: {"market_projection_gap"} for pos in builder.POSITIONS
    },
    "market_all": {
        pos: {"adp_rank_pct", "market_projection_gap"}
        for pos in builder.POSITIONS
    },
    "disagreement": {
        pos: {
            "projection_disagreement_frac",
            "rank_disagreement_scaled",
        }
        for pos in builder.POSITIONS
    },
    "component_ranks": {
        "QB": {"rush_proj_rank_pct", "pass_proj_rank_pct"},
        "RB": {"rush_proj_rank_pct", "rec_proj_rank_pct"},
        "WR": {"rec_proj_rank_pct"},
        "TE": {"rec_proj_rank_pct"},
    },
    "room_hierarchy": {
        "QB": {"qb_team_rank_distance", "qb1_over_qb2_gap_pct"},
        "RB": {
            "rb_combined_share_of_room",
            "rb_room_rank_scaled",
            "rb_gap_to_next_share",
        },
        "WR": {
            "pass_catcher_rank_scaled",
            "pass_catcher_gap_to_next_share",
        },
        "TE": {
            "pass_catcher_rank_scaled",
            "pass_catcher_gap_to_next_share",
        },
    },
    "concentration": {
        "QB": set(),
        "RB": {"rb_room_concentration"},
        "WR": {"pass_catcher_room_concentration"},
        "TE": {"pass_catcher_room_concentration"},
    },
    "receiver_team_pass": {
        "QB": set(),
        "RB": set(),
        "WR": {"team_qb_pass_proj_rank_pct"},
        "TE": {"team_qb_pass_proj_rank_pct"},
    },
}

WEIGHT_VARIANTS = {
    "full": (),
    "no_exp_interaction": ("exp_interaction",),
    "no_adp_rank": ("adp_rank",),
    "no_market_gap": ("market_gap",),
    "no_market": ("market_all",),
    "no_disagreement": ("disagreement",),
    "no_component_ranks": ("component_ranks",),
    "compact_room_hierarchy": ("room_hierarchy",),
    "no_concentration": ("concentration",),
    "no_receiver_team_pass": ("receiver_team_pass",),
    "compact_adp": (
        "exp_interaction",
        "market_gap",
        "room_hierarchy",
    ),
    "compact_gap": (
        "exp_interaction",
        "adp_rank",
        "room_hierarchy",
    ),
    "compact_adp_no_disagreement": (
        "exp_interaction",
        "market_gap",
        "disagreement",
        "room_hierarchy",
    ),
    "role_core_adp": (
        "exp_interaction",
        "market_gap",
        "disagreement",
        "component_ranks",
        "room_hierarchy",
        "receiver_team_pass",
    ),
    "protected_core": (
        "exp_interaction",
        "market_all",
        "disagreement",
        "component_ranks",
        "room_hierarchy",
        "concentration",
        "receiver_team_pass",
    ),
}


def remove_feature_families(
    family_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    for family_name in family_names:
        family = FEATURE_FAMILIES[family_name]
        for pos in builder.POSITIONS:
            for feature in family[pos]:
                weights[pos].pop(feature, None)
    return weights


def build_methods() -> tuple[dict, pd.DataFrame]:
    methods = {}
    metadata = []
    for variant, removed_families in WEIGHT_VARIANTS.items():
        weights = remove_feature_families(removed_families)
        for recency in [False, True]:
            method = f"{variant}__r12" if recency else variant
            recency_half_life = RECENCY_HALF_LIFE if recency else None
            feature_counts = {
                pos: len(weights[pos]) for pos in builder.POSITIONS
            }
            methods[method] = {
                "weights": weights,
                "recency_half_life": recency_half_life,
                "variant": variant,
                "removed_families": removed_families,
            }
            metadata.append(
                {
                    "method": method,
                    "variant": variant,
                    "recency_half_life": recency_half_life,
                    "uses_recency": int(recency),
                    "removed_families": ",".join(removed_families),
                    **{
                        f"{pos.lower()}_feature_count": feature_counts[pos]
                        for pos in builder.POSITIONS
                    },
                    "feature_count_total": sum(feature_counts.values()),
                    "total_match_weight": sum(
                        sum(position_weights.values())
                        for position_weights in weights.values()
                    ),
                    # The recency half-life was fixed by the separate context
                    # study and adds no fitted feature or free parameter here.
                    "complexity_score": sum(feature_counts.values()),
                }
            )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def selected_pool(target, eligible_donors, specification):
    distances = base.donor_distances(
        target,
        eligible_donors,
        specification["weights"],
    )
    tie_rng = np.random.default_rng(
        builder.stable_seed(
            target.player,
            target.pos,
            target.season,
            "rolling",
        )
    )
    tie_break = tie_rng.random(len(eligible_donors))
    order = np.lexsort((tie_break, distances))
    selected_count = min(builder.MAX_TEMPLATE_POOL_SIZE, len(order))
    selected_index = order[:selected_count]
    selected = eligible_donors.iloc[selected_index].copy()
    selected_distances = distances[selected_index]
    probabilities = context.adaptive_probabilities(
        target,
        selected_distances,
    )

    season_gap = (
        int(target.season) - selected.season.to_numpy(dtype=int)
    )
    if np.any(season_gap <= 0):
        raise AssertionError("A feature-ablation pool contains a non-prior donor.")
    recency_half_life = specification["recency_half_life"]
    if recency_half_life is not None:
        recency_multiplier = np.power(
            0.5,
            season_gap / float(recency_half_life),
        )
        probabilities = probabilities * recency_multiplier
        probabilities = probabilities / probabilities.sum()
        probabilities = builder.cap_probability_vector(
            probabilities,
            builder.TEMPLATE_MAX_SAMPLE_PROBABILITY,
        )
    return {
        "donors": selected,
        "probabilities": np.asarray(probabilities, dtype=float),
        "distances": selected_distances,
        "season_gap": season_gap,
    }


def evaluate_extended(target, pool):
    donors = pool["donors"]
    probabilities = pool["probabilities"]
    evaluation = base.evaluate_distribution(
        target,
        donors,
        probabilities,
        {"center": True},
    )
    played = donors.played_games.to_numpy(dtype=float)
    played_quantiles = {
        quantile: base.weighted_quantile(
            played,
            probabilities,
            quantile,
        )
        for quantile in [0.10, 0.50, 0.90]
    }
    observed_played = float(target.played_games)
    evaluation.update(
        {
            "played_mean": base.weighted_mean(played, probabilities),
            "played_q10": played_quantiles[0.10],
            "played_q50": played_quantiles[0.50],
            "played_q90": played_quantiles[0.90],
            "played_crps": base.weighted_crps(
                played,
                probabilities,
                observed_played,
            ),
            "played_80_covered": int(
                played_quantiles[0.10]
                <= observed_played
                <= played_quantiles[0.90]
            ),
            "prob_extended_absence": float(
                probabilities[played <= 8].sum()
            ),
            "prob_zero_active": float(
                probabilities[
                    donors.active_games.to_numpy(dtype=float) <= 0
                ].sum()
            ),
            "weighted_season_gap": base.weighted_mean(
                pool["season_gap"],
                probabilities,
            ),
            "weight_last3_seasons": float(
                probabilities[pool["season_gap"] <= 3].sum()
            ),
            "weight_10plus_seasons": float(
                probabilities[pool["season_gap"] >= 10].sum()
            ),
        }
    )
    return evaluation


def run_replay(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    grouped_donors = {
        (season, pos): group.reset_index(drop=True)
        for (season, pos), group in templates.groupby(["season", "pos"])
    }
    donors_by_origin_pos = {}
    donor_seasons = sorted(templates.season.unique())
    for season in sorted(targets.season.unique()):
        for pos in builder.POSITIONS:
            donors = pd.concat(
                [
                    grouped_donors[(donor_season, pos)]
                    for donor_season in donor_seasons
                    if donor_season < season
                    and (donor_season, pos) in grouped_donors
                ],
                ignore_index=True,
            )
            donors_by_origin_pos[(season, pos)] = donors[
                donors.template_eligible.eq(1)
            ].reset_index(drop=True)

    records = []
    total_targets = len(targets)
    for target_number, target in enumerate(
        targets.itertuples(index=False),
        start=1,
    ):
        eligible_donors = donors_by_origin_pos[
            (target.season, target.pos)
        ]
        if len(eligible_donors) < builder.MIN_TEMPLATE_POOL_SIZE:
            raise ValueError(
                f"Only {len(eligible_donors)} prior donors for "
                f"{target.pos} {target.season}."
            )
        common = {
            "player": target.player,
            "pos": target.pos,
            "season": int(target.season),
            "year_exp": float(target.year_exp),
            "experience_group": target.experience_group,
            "projection_tier": target.projection_tier,
            "predicted_ppg": float(target.historical_pred_fp_per_game),
            "avg_pick": float(target.avg_pick),
            "observed_ppg": float(target.observed_ppg),
            "observed_residual": float(target.observed_residual),
            "observed_contribution": float(
                target.observed_contribution
            ),
            "observed_total": float(target.observed_total),
            "observed_played": float(target.played_games),
            "observed_active": float(target.active_games),
            "observed_zero_contribution": int(
                target.observed_zero_contribution
            ),
            "observed_plus3": int(target.observed_plus3),
            "observed_plus5": int(target.observed_plus5),
            "observed_impact": int(target.observed_impact),
            "observed_extended_absence": int(
                float(target.played_games) <= 8
            ),
            "observed_zero_active": int(
                float(target.active_games) <= 0
            ),
            "impact_threshold": float(target.impact_threshold),
            "eligible_prior_donors": int(len(eligible_donors)),
        }
        for method, specification in METHODS.items():
            pool = selected_pool(
                target,
                eligible_donors,
                specification,
            )
            probabilities = pool["probabilities"]
            if not np.isclose(probabilities.sum(), 1.0):
                raise AssertionError(
                    f"Template probabilities do not sum to one for {method}."
                )
            evaluation = evaluate_extended(target, pool)
            records.append(
                {
                    **common,
                    "method": method,
                    "pool_size": len(pool["donors"]),
                    "min_template_distance": float(
                        pool["distances"].min()
                    ),
                    "median_template_distance": float(
                        np.median(pool["distances"])
                    ),
                    **evaluation,
                }
            )
        if target_number % 200 == 0 or target_number == total_targets:
            print(
                f"Completed {target_number}/{total_targets} "
                "held-out targets"
            )
    predictions = pd.DataFrame(records)
    predictions["plus3_brier_row"] = np.square(
        predictions.prob_plus3 - predictions.observed_plus3
    )
    predictions["plus5_brier_row"] = np.square(
        predictions.prob_plus5 - predictions.observed_plus5
    )
    predictions["impact_brier_row"] = np.square(
        predictions.prob_impact - predictions.observed_impact
    )
    predictions["zero_brier_row"] = np.square(
        predictions.prob_zero_contribution
        - predictions.observed_zero_contribution
    )
    predictions["extended_absence_brier_row"] = np.square(
        predictions.prob_extended_absence
        - predictions.observed_extended_absence
    )
    return predictions


def summarize_group(group: pd.DataFrame) -> pd.Series:
    summary = base.summarize_group(group).to_dict()
    summary.update(
        {
            "played_crps": group.played_crps.mean(),
            "played_mae_mean": np.abs(
                group.played_mean - group.observed_played
            ).mean(),
            "played_bias": (
                group.played_mean - group.observed_played
            ).mean(),
            "played_80_coverage": group.played_80_covered.mean(),
            "extended_absence_actual_rate": (
                group.observed_extended_absence.mean()
            ),
            "extended_absence_predicted_rate": (
                group.prob_extended_absence.mean()
            ),
            "extended_absence_brier": (
                group.extended_absence_brier_row.mean()
            ),
            "extended_absence_auc": base.binary_auc(
                group.observed_extended_absence,
                group.prob_extended_absence,
            ),
            "weighted_season_gap": group.weighted_season_gap.mean(),
            "weight_last3_seasons": group.weight_last3_seasons.mean(),
            "weight_10plus_seasons": group.weight_10plus_seasons.mean(),
        }
    )
    return pd.Series(summary)


def grouped_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    return (
        frame.groupby(group_cols, observed=True, sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def add_selection_loss(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    baseline = output[output.method.eq(BASELINE_METHOD)]
    denominators = {
        "ppg_crps": float(baseline.ppg_crps.mean()),
        "contribution_crps": float(baseline.contribution_crps.mean()),
        "played_crps": float(baseline.played_crps.mean()),
    }
    output["selection_loss"] = np.mean(
        [
            output.ppg_crps / denominators["ppg_crps"],
            output.contribution_crps
            / denominators["contribution_crps"],
            output.played_crps / denominators["played_crps"],
        ],
        axis=0,
    )
    return output


def selection_table(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    scored = add_selection_loss(frame)
    summary = grouped_summary(scored, ["method"]).merge(
        METHOD_METADATA,
        on="method",
        how="left",
        validate="one_to_one",
    )
    loss = (
        scored.groupby("method", as_index=False)
        .selection_loss.mean()
    )
    summary = summary.merge(
        loss,
        on="method",
        how="left",
        validate="one_to_one",
    )
    position_metrics = (
        frame.groupby(["method", "pos"], observed=True)[
            ["ppg_crps", "contribution_crps", "played_crps"]
        ]
        .mean()
    )
    position_checks = []
    for method in sorted(frame.method.unique()):
        matched_baseline = (
            "full__r12" if method.endswith("__r12") else BASELINE_METHOD
        )
        method_metrics = position_metrics.loc[method]
        baseline_metrics = position_metrics.loc[matched_baseline]
        metric_ratios = method_metrics / baseline_metrics
        composite_delta = metric_ratios.mean(axis=1) - 1.0
        position_checks.append(
            {
                "method": method,
                "max_position_composite_delta": float(
                    composite_delta.max()
                ),
                "max_position_metric_delta": float(
                    (metric_ratios - 1.0).max().max()
                ),
                "position_guardrail_pass": bool(
                    composite_delta.max() <= 0.005
                    and (metric_ratios - 1.0).max().max() <= 0.01
                ),
            }
        )
    summary = summary.merge(
        pd.DataFrame(position_checks),
        on="method",
        how="left",
        validate="one_to_one",
    )
    baseline = summary[summary.method.eq(BASELINE_METHOD)].iloc[0]
    summary["aggregate_guardrail_pass"] = (
        summary.ppg_crps.le(float(baseline.ppg_crps) * 1.0025)
        & summary.contribution_crps.le(
            float(baseline.contribution_crps) * 1.0025
        )
        & summary.played_crps.le(
            float(baseline.played_crps) * 1.0025
        )
        & summary.ppg_80_coverage.ge(
            float(baseline.ppg_80_coverage) - 0.01
        )
        & summary.contribution_80_coverage.ge(
            float(baseline.contribution_80_coverage) - 0.01
        )
        & summary.played_80_coverage.ge(
            float(baseline.played_80_coverage) - 0.01
        )
        & summary.plus3_brier.le(float(baseline.plus3_brier) + 0.001)
        & summary.plus5_brier.le(float(baseline.plus5_brier) + 0.001)
        & summary.impact_brier.le(float(baseline.impact_brier) + 0.001)
        & summary.zero_brier.le(float(baseline.zero_brier) + 0.001)
        & summary.extended_absence_brier.le(
            float(baseline.extended_absence_brier) + 0.001
        )
    )
    summary["guardrail_pass"] = (
        summary.aggregate_guardrail_pass
        & summary.position_guardrail_pass
    )
    guarded = summary[summary.guardrail_pass]
    if len(guarded) == 0:
        raise ValueError("No feature specification passed selection guardrails.")
    best_method = guarded.sort_values(
        ["selection_loss", "complexity_score", "method"]
    ).method.iloc[0]

    key_cols = ["player", "pos", "season"]
    best = scored[scored.method.eq(best_method)][
        key_cols + ["selection_loss"]
    ].rename(columns={"selection_loss": "best_loss"})
    paired = scored.merge(
        best,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    paired["loss_delta_to_best"] = (
        paired.selection_loss - paired.best_loss
    )
    one_se = []
    for method, group in paired.groupby("method"):
        season_deltas = (
            group.groupby("season").loss_delta_to_best.mean()
        )
        standard_error = (
            float(season_deltas.std(ddof=1) / np.sqrt(len(season_deltas)))
            if len(season_deltas) > 1
            else 0.0
        )
        one_se.append(
            {
                "method": method,
                "mean_loss_delta_to_best": float(
                    group.loss_delta_to_best.mean()
                ),
                "paired_se_to_best": standard_error,
                "within_one_se": int(
                    float(group.loss_delta_to_best.mean())
                    <= standard_error + 1e-12
                ),
            }
        )
    summary = summary.merge(
        pd.DataFrame(one_se),
        on="method",
        how="left",
        validate="one_to_one",
    )
    candidates = summary[
        summary.guardrail_pass & summary.within_one_se.eq(1)
    ]
    selected_method = candidates.sort_values(
        [
            "complexity_score",
            "feature_count_total",
            "selection_loss",
            "method",
        ]
    ).method.iloc[0]
    summary["selected"] = summary.method.eq(selected_method).astype(int)
    return summary.sort_values(
        ["guardrail_pass", "within_one_se", "complexity_score", "selection_loss"],
        ascending=[False, False, True, True],
    ), selected_method


def period_summaries(predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for period, (start, end) in PERIODS.items():
        frame = predictions[predictions.season.between(start, end)]
        summary = grouped_summary(frame, ["method"])
        summary.insert(0, "period", period)
        output.append(summary)
    return pd.concat(output, ignore_index=True).merge(
        METHOD_METADATA,
        on="method",
        how="left",
        validate="many_to_one",
    )


def build_nested_selection(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    path = []
    for target_season in sorted(
        predictions[
            predictions.season.ge(META_SELECTION_START)
        ].season.unique()
    ):
        training = predictions[
            predictions.season.lt(target_season)
        ]
        leaderboard, selected_method = selection_table(training)
        selected_row = leaderboard[
            leaderboard.method.eq(selected_method)
        ].iloc[0]
        path.append(
            {
                "target_season": int(target_season),
                "training_start": int(training.season.min()),
                "training_end": int(training.season.max()),
                "training_target_rows": int(
                    training[
                        training.method.eq(BASELINE_METHOD)
                    ].shape[0]
                ),
                "selected_method": selected_method,
                "selected_variant": selected_row.variant,
                "selected_uses_recency": int(
                    selected_row.uses_recency
                ),
                "selected_feature_count_total": int(
                    selected_row.feature_count_total
                ),
                "selected_complexity_score": int(
                    selected_row.complexity_score
                ),
                "selected_development_loss": float(
                    selected_row.selection_loss
                ),
            }
        )
        target = predictions[
            predictions.season.eq(target_season)
            & predictions.method.eq(selected_method)
        ].copy()
        target["selected_method"] = selected_method
        target["method"] = "nested_rolling_selected"
        selected_frames.append(target)
    return pd.concat(selected_frames, ignore_index=True), pd.DataFrame(path)


def bootstrap_comparison(
    predictions: pd.DataFrame,
    candidate_method: str,
    periods: dict[str, tuple[int, int]],
    baseline_method: str = BASELINE_METHOD,
) -> pd.DataFrame:
    key_cols = ["player", "pos", "season"]
    fields = key_cols + LOWER_IS_BETTER
    baseline = predictions[
        predictions.method.eq(baseline_method)
    ][fields]
    candidate = predictions[
        predictions.method.eq(candidate_method)
    ][fields]
    paired = candidate.merge(
        baseline,
        on=key_cols,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for period, (start, end) in periods.items():
        frame = paired[paired.season.between(start, end)]
        seasons = np.sort(frame.season.unique())
        season_frames = {
            season: frame[frame.season.eq(season)]
            for season in seasons
        }
        draws = {metric: [] for metric in LOWER_IS_BETTER}
        for _ in range(BOOTSTRAP_SAMPLES):
            sampled = rng.choice(
                seasons,
                size=len(seasons),
                replace=True,
            )
            sample = pd.concat(
                [season_frames[season] for season in sampled],
                ignore_index=True,
            )
            for metric in LOWER_IS_BETTER:
                draws[metric].append(
                    float(
                        (
                            sample[f"{metric}_candidate"]
                            - sample[f"{metric}_baseline"]
                        ).mean()
                    )
                )
        for metric in LOWER_IS_BETTER:
            observed = float(
                (
                    frame[f"{metric}_candidate"]
                    - frame[f"{metric}_baseline"]
                ).mean()
            )
            values = np.asarray(draws[metric], dtype=float)
            output.append(
                {
                    "candidate_method": candidate_method,
                    "baseline_method": baseline_method,
                    "period": period,
                    "metric": metric,
                    "n": len(frame),
                    "season_clusters": len(seasons),
                    "candidate_minus_baseline": observed,
                    "bootstrap_p025": float(
                        np.quantile(values, 0.025)
                    ),
                    "bootstrap_p975": float(
                        np.quantile(values, 0.975)
                    ),
                    "probability_candidate_better": float(
                        np.mean(values < 0)
                    ),
                }
            )
    return pd.DataFrame(output)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def write_summary(
    development: pd.DataFrame,
    selected_method: str,
    period_summary: pd.DataFrame,
    nested_summary: pd.DataFrame,
    selection_path: pd.DataFrame,
    bootstrap: pd.DataFrame,
    target_rows: int,
    runtime_seconds: float,
) -> None:
    temporal = period_summary[
        period_summary.period.eq("temporal_2023_2025")
    ].set_index("method")
    temporal_baseline = temporal.loc[BASELINE_METHOD]
    temporal_selected = temporal.loc[selected_method]
    component_ranks = development[
        development.method.eq("no_component_ranks__r12")
    ].iloc[0]
    leaderboard = development[
        [
            "method",
            "feature_count_total",
            "complexity_score",
            "selection_loss",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
            "guardrail_pass",
            "position_guardrail_pass",
            "max_position_composite_delta",
            "max_position_metric_delta",
            "within_one_se",
            "selected",
        ]
    ].sort_values(
        ["selected", "guardrail_pass", "selection_loss"],
        ascending=[False, False, True],
    ).head(12)
    selected_periods = period_summary[
        period_summary.method.isin(
            [BASELINE_METHOD, "full__r12", selected_method]
        )
    ][
        [
            "period",
            "method",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
            "plus5_brier",
            "impact_brier",
            "impact_auc",
            "weighted_season_gap",
            "weight_10plus_seasons",
        ]
    ]
    nested = nested_summary[
        [
            "method",
            "n",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
            "plus5_brier",
            "impact_brier",
            "impact_auc",
        ]
    ]
    boot = bootstrap[
        bootstrap.metric.isin(
            ["ppg_crps", "contribution_crps", "played_crps"]
        )
    ][
        [
            "period",
            "baseline_method",
            "metric",
            "candidate_minus_baseline",
            "bootstrap_p025",
            "bootstrap_p975",
            "probability_candidate_better",
        ]
    ]
    rounded_frames = []
    for frame in [leaderboard, selected_periods, nested, boot]:
        frame = frame.copy()
        numeric = frame.select_dtypes(include=[np.number]).columns
        frame[numeric] = frame[numeric].round(5)
        rounded_frames.append(frame)
    leaderboard, selected_periods, nested, boot = rounded_frames
    text = "\n".join(
        [
            "# Weekly Template Feature Pruning",
            "",
            "## Design",
            "",
            f"- Held out {target_rows:,} player-seasons at strict rolling origins.",
            f"- Evaluated {len(METHODS)} paired feature/recency specifications.",
            "- Every donor season is strictly earlier than its target season.",
            "- Development selection uses 2017-2022 only, predeclared guardrails, "
            "position-level safety checks, and a paired season-level "
            "one-standard-error rule.",
            "",
            "## Development selection",
            "",
            f"Selected specification: `{selected_method}`.",
            "",
            markdown_table(leaderboard),
            "",
            "## Fixed-method period checks",
            "",
            markdown_table(selected_periods),
            "",
            "## Nested rolling selection",
            "",
            markdown_table(selection_path),
            "",
            markdown_table(nested),
            "",
            "## Selected-vs-production clustered uncertainty",
            "",
            "Negative score deltas favor the selected specification.",
            "",
            markdown_table(boot),
            "",
            "## Interpretation",
            "",
            f"- Recommend `{selected_method}` for the next production update: "
            "retain direct projected PPG and uncapped experience, but remove "
            "their redundant projection-by-experience interaction.",
            f"- Versus production in untouched 2023-2025, PPG CRPS changed by "
            f"{float(temporal_selected.ppg_crps - temporal_baseline.ppg_crps):+.5f}, "
            f"contribution CRPS by "
            f"{float(temporal_selected.contribution_crps - temporal_baseline.contribution_crps):+.5f}, "
            f"and played-games CRPS by "
            f"{float(temporal_selected.played_crps - temporal_baseline.played_crps):+.5f}; "
            "negative changes are improvements.",
            f"- Dropping component ranks was rejected despite its aggregate "
            f"development score: its worst position composite moved by "
            f"{float(component_ranks.max_position_composite_delta):+.3%} and "
            f"its worst individual position metric by "
            f"{float(component_ranks.max_position_metric_delta):+.3%}.",
            "- Keep ADP/market context, disagreement, component ranks, room "
            "hierarchy, concentration, and pass-catcher team environment. "
            "The aggressive compact variants lost too much downside and "
            "availability precision.",
            "- The gains are small, as expected for pruning a redundant "
            "feature. Production remains unchanged until this recommendation "
            "is explicitly promoted.",
            "",
            f"Runtime: {runtime_seconds:.1f} seconds.",
            "",
        ]
    )
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    max_season = builder.get_daily_max_template_season()
    print(f"Loading historical inputs through {max_season}")
    projections = builder.load_historical_projection_context(max_season)
    weekly = builder.load_weekly_points(max_season)
    templates = builder.build_weekly_templates(projections, weekly)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)
    predictions = run_replay(templates, targets)

    expected_rows = len(targets) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} prediction rows; found "
            f"{len(predictions)}."
        )
    development_frame = predictions[
        predictions.season.le(DEVELOPMENT_END)
    ]
    development, selected_method = selection_table(
        development_frame
    )
    period_summary = period_summaries(predictions)
    position_summary = grouped_summary(
        predictions[predictions.season.ge(2020)],
        ["method", "pos"],
    ).merge(
        METHOD_METADATA,
        on="method",
        how="left",
        validate="many_to_one",
    )
    nested_predictions, selection_path = build_nested_selection(
        predictions
    )
    nested_comparison = pd.concat(
        [
            nested_predictions,
            predictions[
                predictions.season.ge(META_SELECTION_START)
                & predictions.method.isin(
                    [BASELINE_METHOD, "full__r12"]
                )
            ],
        ],
        ignore_index=True,
    )
    nested_summary = grouped_summary(
        nested_comparison,
        ["method"],
    )
    bootstrap = bootstrap_comparison(
        predictions,
        selected_method,
        PERIODS,
    )
    candidate_pairs = [
        ("full__r12", BASELINE_METHOD),
        ("no_exp_interaction__r12", BASELINE_METHOD),
        ("no_exp_interaction__r12", "full__r12"),
        ("no_component_ranks__r12", "full__r12"),
        ("no_disagreement__r12", "full__r12"),
        ("no_receiver_team_pass__r12", "full__r12"),
    ]
    candidate_bootstrap = pd.concat(
        [
            bootstrap_comparison(
                predictions,
                candidate_method,
                PERIODS,
                baseline_method=baseline_method,
            )
            for candidate_method, baseline_method in candidate_pairs
        ],
        ignore_index=True,
    )

    predictions.to_csv(
        RESULTS / "target_predictions.csv",
        index=False,
    )
    METHOD_METADATA.to_csv(
        RESULTS / "method_metadata.csv",
        index=False,
    )
    development.to_csv(
        RESULTS / "development_selection.csv",
        index=False,
    )
    period_summary.to_csv(
        RESULTS / "summary_by_period.csv",
        index=False,
    )
    position_summary.to_csv(
        RESULTS / "summary_by_position.csv",
        index=False,
    )
    nested_predictions.to_csv(
        RESULTS / "nested_selected_predictions.csv",
        index=False,
    )
    selection_path.to_csv(
        RESULTS / "nested_selection_path.csv",
        index=False,
    )
    nested_summary.to_csv(
        RESULTS / "nested_selection_summary.csv",
        index=False,
    )
    bootstrap.to_csv(
        RESULTS / "selected_bootstrap.csv",
        index=False,
    )
    candidate_bootstrap.to_csv(
        RESULTS / "candidate_bootstrap.csv",
        index=False,
    )

    runtime_seconds = time.perf_counter() - started
    metadata = {
        "max_template_season": int(max_season),
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "recency_half_life": RECENCY_HALF_LIFE,
        "development_end": DEVELOPMENT_END,
        "selected_method": selected_method,
        "meta_selection_start": META_SELECTION_START,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "future_donor_rows": 0,
        "runtime_seconds": runtime_seconds,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_summary(
        development,
        selected_method,
        period_summary,
        nested_summary,
        selection_path,
        bootstrap,
        len(targets),
        runtime_seconds,
    )
    print(
        development[
            [
                "method",
                "feature_count_total",
                "selection_loss",
                "ppg_crps",
                "contribution_crps",
                "played_crps",
                "guardrail_pass",
                "within_one_se",
                "selected",
            ]
        ]
        .head(15)
        .round(5)
        .to_string(index=False)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
