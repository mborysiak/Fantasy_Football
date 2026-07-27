"""Strict rolling-origin calibration for joint residual/weekly template draws."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "Scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from Modeling import s4_Best_Ball_Weekly as builder  # noqa: E402


RESULTS = Path(__file__).resolve().parent / "results" / "production_oos"
ORIGIN_START = 2017
CORE_ORIGIN_START = 2017
RECENT_ORIGIN_START = 2020
TARGET_COUNTS = {"QB": 24, "RB": 60, "WR": 72, "TE": 24}
WAIVER_BASELINES = {"QB": 15.0, "RB": 7.0, "WR": 7.0, "TE": 5.0}
WEEK_COLS = [f"managed_week_{week}" for week in builder.WEEKS]


LEGACY_COMMON_WEIGHTS = {
    "match_projection_rank_pct": 2.5,
    "year_exp_scaled": 2.0,
    "projection_x_exp": 1.0,
    "adp_rank_pct": 0.5,
}
LEGACY_POSITION_WEIGHTS = {
    "QB": {
        "qb_team_rank_distance": 1.5,
        "qb_room_share": 1.25,
        "qb1_over_qb2_gap_pct": 0.75,
        "rush_share_of_own_points": 1.25,
        "rush_proj_rank_pct": 1.0,
        "pass_proj_rank_pct": 1.0,
    },
    "RB": {
        "rush_proj_rank_pct": 1.0,
        "rec_proj_rank_pct": 1.0,
        "rec_share_of_own_points": 1.0,
        "rb_rush_share_of_room": 1.25,
        "rb_rec_share_of_room": 0.75,
    },
    "WR": {
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
    "TE": {
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
}
LEGACY_WEIGHTS = {
    pos: {**LEGACY_COMMON_WEIGHTS, **LEGACY_POSITION_WEIGHTS[pos]}
    for pos in builder.POSITIONS
}
PROJECTION_ONLY_WEIGHTS = {
    pos: {
        "match_projection_rank_pct": 2.5,
        "match_projection_ppg_scaled": 1.5,
        "year_exp_scaled": 2.0,
        "projection_x_exp": 1.0,
        "adp_rank_pct": 0.5,
    }
    for pos in builder.POSITIONS
}
METHODS = {
    "adaptive_full_centered": {
        "weights": builder.MATCH_FEATURE_WEIGHTS,
        "probability": "adaptive",
        "center": True,
    },
    "adaptive_full_uncentered": {
        "weights": builder.MATCH_FEATURE_WEIGHTS,
        "probability": "adaptive",
        "center": False,
    },
    "uniform_full_centered": {
        "weights": builder.MATCH_FEATURE_WEIGHTS,
        "probability": "uniform",
        "center": True,
    },
    "legacy_2x_centered": {
        "weights": LEGACY_WEIGHTS,
        "probability": "legacy_2x",
        "center": True,
    },
    "adaptive_projection_centered": {
        "weights": PROJECTION_ONLY_WEIGHTS,
        "probability": "adaptive",
        "center": True,
    },
}


def weighted_quantile(values, probabilities, quantile):
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(probabilities[order])
    index = min(np.searchsorted(cumulative, quantile, side="left"), len(values) - 1)
    return float(ordered_values[index])


def weighted_mid_pit(values, probabilities, observed):
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    tolerance = 1e-10
    below = probabilities[values < observed - tolerance].sum()
    equal = probabilities[np.isclose(values, observed, atol=tolerance)].sum()
    return float(below + 0.5 * equal)


def weighted_crps(values, probabilities, observed):
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    first = np.sum(probabilities * np.abs(values - observed))
    pairwise = np.abs(values[:, None] - values[None, :])
    second = 0.5 * np.sum(probabilities[:, None] * probabilities[None, :] * pairwise)
    return float(first - second)


def weighted_mean(values, probabilities):
    return float(np.sum(np.asarray(values, dtype=float) * probabilities))


def binary_auc(outcome, score):
    outcome = np.asarray(outcome, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = outcome == 1
    negatives = outcome == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - positives.sum() * (positives.sum() + 1) / 2)
        / (positives.sum() * negatives.sum())
    )


def target_observation(row):
    scale = (
        float(row.active_ppg)
        if float(row.active_games) > 0
        else float(row.historical_pred_fp_per_game)
    )
    weekly_scores = row[WEEK_COLS].to_numpy(dtype=float) * scale
    baseline = WAIVER_BASELINES[row.pos]
    contribution = float(np.maximum(weekly_scores - baseline, 0).sum())
    total = float(weekly_scores.sum())
    prediction = float(row.historical_pred_fp_per_game)
    return {
        "observed_ppg": float(row.active_ppg),
        "observed_residual": float(row.active_ppg) - prediction,
        "observed_contribution": contribution,
        "observed_total": total,
        "observed_zero_contribution": int(np.isclose(contribution, 0)),
        "observed_plus3": int(float(row.active_ppg) - prediction >= 3),
        "observed_plus5": int(float(row.active_ppg) - prediction >= 5),
    }


def build_targets(templates):
    target_frames = []
    for (season, pos), group in templates[
        templates.season.ge(ORIGIN_START)
    ].groupby(["season", "pos"], sort=True):
        target_frames.append(
            group.sort_values(
                ["historical_pred_fp_per_game", "avg_pick", "player"],
                ascending=[False, True, True],
            ).head(TARGET_COUNTS[pos])
        )
    targets = pd.concat(target_frames, ignore_index=True)
    observations = [target_observation(row) for _, row in targets.iterrows()]
    targets = pd.concat(
        [targets.reset_index(drop=True), pd.DataFrame(observations)],
        axis=1,
    )
    targets["impact_threshold"] = targets.groupby(
        ["season", "pos"]
    )["observed_contribution"].transform(lambda values: values.quantile(0.80))
    targets["observed_impact"] = (
        targets.observed_plus3.eq(1)
        & targets.observed_contribution.ge(targets.impact_threshold)
    ).astype(int)
    targets["experience_group"] = pd.cut(
        targets.year_exp,
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=["0-2", "3-5", "6-8", "9+"],
    ).astype(str)
    return targets


def load_production_oos_forecasts(max_season):
    """Load the causal final-ensemble point forecasts used by production."""
    forecasts = builder.dm.read(
        f"""
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               pred_fp_per_game production_oos_pred_fp_per_game,
               y_act production_validation_y_act
        FROM Final_Validations_Resid
        WHERE version='{builder.LEAGUE}'
              AND model_spec_asof_year={builder.YEAR}
              AND data_oos=1
              AND season BETWEEN {ORIGIN_START} AND {max_season}
        """,
        "Validations",
    )
    forecasts = builder.clean_player_names(forecasts)
    duplicate_count = int(
        forecasts.duplicated(["player", "season", "pos"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "Final_Validations_Resid contains "
            f"{duplicate_count} duplicate player-season-position rows."
        )
    return forecasts


def build_production_oos_target_templates(templates, forecasts):
    """Put held-out targets on the current final-forecast scale.

    Donor rows intentionally retain the forecasts and residuals persisted in
    the production template bank. This target-only override reproduces the
    transport problem faced by the live app and directly tests whether donor
    residual centering handles that scale difference.
    """
    target_templates = templates[
        templates.season.between(ORIGIN_START, forecasts.season.max())
    ].copy()
    target_templates = target_templates.rename(
        columns={
            "historical_pred_fp_per_game": "builder_historical_pred_fp_per_game",
            "historical_projection_source": "builder_historical_projection_source",
        }
    ).merge(
        forecasts,
        on=["player", "season", "pos"],
        how="inner",
        validate="one_to_one",
    )
    target_templates["historical_pred_fp_per_game"] = target_templates[
        "production_oos_pred_fp_per_game"
    ]
    target_templates["historical_projection_source"] = (
        "final_validations_resid_oos"
    )

    target_templates = builder.add_projection_buckets(
        target_templates,
        value_col="historical_pred_fp_per_game",
        group_cols=["season", "pos"],
    )
    target_templates["match_projection_rank_pct"] = target_templates[
        "projection_rank_pct"
    ]
    target_templates["match_projection_ppg_scaled"] = (
        target_templates["historical_pred_fp_per_game"]
        .clip(lower=0)
        .div(builder.PROJECTION_PPG_SCALE)
    )
    target_templates["projection_x_exp"] = (
        target_templates["match_projection_rank_pct"]
        * target_templates["year_exp_scaled"]
    )
    target_templates["market_projection_gap"] = (
        target_templates["adp_rank_pct"]
        - target_templates["match_projection_rank_pct"]
    )
    return target_templates


def donor_distances(target, donors, weights):
    distances = np.zeros(len(donors), dtype=float)
    target_qb_bucket = getattr(target, "qb_team_rank_bucket", "non_qb")
    target_qb_value = builder.QB_RANK_DISTANCE_ORDER.get(target_qb_bucket, 2)
    qb_distance = (
        donors.qb_team_rank_bucket.map(builder.QB_RANK_DISTANCE_ORDER)
        .fillna(2)
        .sub(target_qb_value)
        .abs()
        .to_numpy(dtype=float)
        if target.pos == "QB"
        else np.zeros(len(donors), dtype=float)
    )
    for feature, weight in weights[target.pos].items():
        if feature == "qb_team_rank_distance":
            feature_distance = qb_distance
        else:
            donor_values = pd.to_numeric(
                donors[feature], errors="coerce"
            ).fillna(builder.MATCH_FILL_VALUE).to_numpy(dtype=float)
            target_value = pd.to_numeric(
                pd.Series([getattr(target, feature, builder.MATCH_FILL_VALUE)]),
                errors="coerce",
            ).fillna(builder.MATCH_FILL_VALUE).iloc[0]
            feature_distance = np.abs(donor_values - float(target_value))
        distances += float(weight) * feature_distance
    return distances


def selected_pool(target, eligible_donors, specification):
    distances = donor_distances(target, eligible_donors, specification["weights"])
    tie_rng = np.random.default_rng(
        builder.stable_seed(target.player, target.pos, target.season, "rolling")
    )
    tie_break = tie_rng.random(len(eligible_donors))
    order = np.lexsort((tie_break, distances))
    selected_count = min(builder.MAX_TEMPLATE_POOL_SIZE, len(order))
    selected_index = order[:selected_count]
    selected = eligible_donors.iloc[selected_index]
    selected_distances = distances[selected_index]

    probability_method = specification["probability"]
    if probability_method == "uniform" or np.ptp(selected_distances) <= 0:
        probabilities = np.repeat(1 / selected_count, selected_count)
    elif probability_method == "legacy_2x":
        weights = np.exp(
            -np.log(2.0)
            * (selected_distances - selected_distances.min())
            / np.ptp(selected_distances)
        )
        probabilities = weights / weights.sum()
    elif probability_method == "adaptive":
        distance_min = float(selected_distances.min())
        bandwidth = builder.TEMPLATE_KERNEL_BANDWIDTH[target.pos]
        local_weights = np.exp(-(selected_distances - distance_min) / bandwidth)
        local_probabilities = local_weights / local_weights.sum()
        local_fraction = max(
            builder.TEMPLATE_MIN_LOCAL_WEIGHT,
            np.exp(-distance_min / builder.TEMPLATE_LOCAL_DISTANCE_SCALE),
        )
        local_fraction = min(float(local_fraction), 1.0)
        probabilities = (
            local_fraction * local_probabilities
            + (1 - local_fraction) / selected_count
        )
        probabilities = builder.cap_probability_vector(
            probabilities,
            builder.TEMPLATE_MAX_SAMPLE_PROBABILITY,
        )
    else:
        raise ValueError(f"Unknown probability method: {probability_method}")
    return selected, np.asarray(probabilities, dtype=float), selected_distances


def evaluate_distribution(target, donors, probabilities, specification):
    prediction = float(target.historical_pred_fp_per_game)
    donor_residuals = donors.active_ppg_resid.to_numpy(dtype=float)
    raw_residual_mean = weighted_mean(donor_residuals, probabilities)
    applied_residuals = donor_residuals.copy()
    if specification["center"]:
        applied_residuals -= raw_residual_mean
    predicted_ppg = np.maximum(prediction + applied_residuals, 0)
    profiles = donors[WEEK_COLS].to_numpy(dtype=float)
    weekly_scores = predicted_ppg[:, None] * profiles
    contribution = np.maximum(
        weekly_scores - WAIVER_BASELINES[target.pos], 0
    ).sum(axis=1)
    total = weekly_scores.sum(axis=1)

    ppg_quantiles = {
        quantile: weighted_quantile(predicted_ppg, probabilities, quantile)
        for quantile in [0.10, 0.50, 0.90]
    }
    contribution_quantiles = {
        quantile: weighted_quantile(contribution, probabilities, quantile)
        for quantile in [0.10, 0.50, 0.90]
    }
    return {
        "raw_residual_mean": raw_residual_mean,
        "effective_sample_size": float(1 / np.square(probabilities).sum()),
        "max_donor_probability": float(probabilities.max()),
        "ppg_mean": weighted_mean(predicted_ppg, probabilities),
        "ppg_q10": ppg_quantiles[0.10],
        "ppg_q50": ppg_quantiles[0.50],
        "ppg_q90": ppg_quantiles[0.90],
        "ppg_crps": weighted_crps(
            predicted_ppg, probabilities, target.observed_ppg
        ),
        "ppg_pit": weighted_mid_pit(
            predicted_ppg, probabilities, target.observed_ppg
        ),
        "ppg_80_covered": int(
            ppg_quantiles[0.10] <= target.observed_ppg <= ppg_quantiles[0.90]
        ),
        "contribution_mean": weighted_mean(contribution, probabilities),
        "contribution_q10": contribution_quantiles[0.10],
        "contribution_q50": contribution_quantiles[0.50],
        "contribution_q90": contribution_quantiles[0.90],
        "contribution_tail_lift": (
            contribution_quantiles[0.90]
            - weighted_mean(contribution, probabilities)
        ),
        "contribution_crps": weighted_crps(
            contribution, probabilities, target.observed_contribution
        ),
        "contribution_pit": weighted_mid_pit(
            contribution, probabilities, target.observed_contribution
        ),
        "contribution_80_covered": int(
            contribution_quantiles[0.10]
            <= target.observed_contribution
            <= contribution_quantiles[0.90]
        ),
        "total_mean": weighted_mean(total, probabilities),
        "total_q90": weighted_quantile(total, probabilities, 0.90),
        "prob_zero_contribution": float(
            probabilities[np.isclose(contribution, 0)].sum()
        ),
        "prob_plus3": float(probabilities[applied_residuals >= 3].sum()),
        "prob_plus5": float(probabilities[applied_residuals >= 5].sum()),
        "prob_impact": float(
            probabilities[
                (applied_residuals >= 3)
                & (contribution >= float(target.impact_threshold))
            ].sum()
        ),
    }


def run_replay(templates, targets):
    records = []
    grouped_donors = {
        (season, pos): group.reset_index(drop=True)
        for (season, pos), group in templates.groupby(["season", "pos"])
    }
    donors_by_origin_pos = {}
    for season in sorted(targets.season.unique()):
        for pos in builder.POSITIONS:
            frames = [
                grouped_donors[(donor_season, pos)]
                for donor_season in sorted(templates.season.unique())
                if donor_season < season and (donor_season, pos) in grouped_donors
            ]
            donors = pd.concat(frames, ignore_index=True)
            donors_by_origin_pos[(season, pos)] = donors[
                donors.template_eligible.eq(1)
            ].reset_index(drop=True)

    total_targets = len(targets)
    for target_number, target in enumerate(targets.itertuples(index=False), start=1):
        eligible_donors = donors_by_origin_pos[(target.season, target.pos)]
        if len(eligible_donors) < builder.MIN_TEMPLATE_POOL_SIZE:
            raise ValueError(
                f"Only {len(eligible_donors)} donors for {target.pos} {target.season}."
            )
        common = {
            "player": target.player,
            "pos": target.pos,
            "season": int(target.season),
            "year_exp": float(target.year_exp),
            "experience_group": target.experience_group,
            "projection_tier": target.projection_tier,
            "historical_projection_source": target.historical_projection_source,
            "predicted_ppg": float(target.historical_pred_fp_per_game),
            "avg_pick": float(target.avg_pick),
            "observed_ppg": float(target.observed_ppg),
            "observed_residual": float(target.observed_residual),
            "observed_contribution": float(target.observed_contribution),
            "observed_total": float(target.observed_total),
            "observed_zero_contribution": int(target.observed_zero_contribution),
            "observed_plus3": int(target.observed_plus3),
            "observed_plus5": int(target.observed_plus5),
            "impact_threshold": float(target.impact_threshold),
            "observed_impact": int(target.observed_impact),
            "eligible_prior_donors": int(len(eligible_donors)),
        }
        for method, specification in METHODS.items():
            selected, probabilities, selected_distances = selected_pool(
                target, eligible_donors, specification
            )
            evaluation = evaluate_distribution(
                target, selected, probabilities, specification
            )
            records.append(
                {
                    **common,
                    "method": method,
                    "pool_size": len(selected),
                    "min_template_distance": float(selected_distances.min()),
                    "median_template_distance": float(
                        np.median(selected_distances)
                    ),
                    **evaluation,
                }
            )
        if target_number % 250 == 0 or target_number == total_targets:
            print(f"Completed {target_number}/{total_targets} held-out targets")
    return pd.DataFrame(records)


def safe_spearman(left, right):
    result = spearmanr(left, right, nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else np.nan


def summarize_group(group):
    ppg_ks = kstest(group.ppg_pit, "uniform")
    contribution_ks = kstest(group.contribution_pit, "uniform")
    return pd.Series(
        {
            "n": len(group),
            "ppg_crps": group.ppg_crps.mean(),
            "ppg_mae_mean": np.abs(group.ppg_mean - group.observed_ppg).mean(),
            "ppg_bias": (group.ppg_mean - group.observed_ppg).mean(),
            "ppg_80_coverage": group.ppg_80_covered.mean(),
            "ppg_pit_mean": group.ppg_pit.mean(),
            "ppg_pit_ks": ppg_ks.statistic,
            "contribution_crps": group.contribution_crps.mean(),
            "contribution_mae_mean": np.abs(
                group.contribution_mean - group.observed_contribution
            ).mean(),
            "contribution_bias": (
                group.contribution_mean - group.observed_contribution
            ).mean(),
            "contribution_80_coverage": group.contribution_80_covered.mean(),
            "contribution_pit_mean": group.contribution_pit.mean(),
            "contribution_pit_ks": contribution_ks.statistic,
            "plus3_actual_rate": group.observed_plus3.mean(),
            "plus3_predicted_rate": group.prob_plus3.mean(),
            "plus3_brier": np.square(
                group.prob_plus3 - group.observed_plus3
            ).mean(),
            "plus3_auc": binary_auc(group.observed_plus3, group.prob_plus3),
            "plus5_actual_rate": group.observed_plus5.mean(),
            "plus5_predicted_rate": group.prob_plus5.mean(),
            "plus5_brier": np.square(
                group.prob_plus5 - group.observed_plus5
            ).mean(),
            "plus5_auc": binary_auc(group.observed_plus5, group.prob_plus5),
            "impact_actual_rate": group.observed_impact.mean(),
            "impact_predicted_rate": group.prob_impact.mean(),
            "impact_brier": np.square(
                group.prob_impact - group.observed_impact
            ).mean(),
            "impact_auc": binary_auc(group.observed_impact, group.prob_impact),
            "zero_actual_rate": group.observed_zero_contribution.mean(),
            "zero_predicted_rate": group.prob_zero_contribution.mean(),
            "zero_brier": np.square(
                group.prob_zero_contribution - group.observed_zero_contribution
            ).mean(),
            "zero_auc": binary_auc(
                group.observed_zero_contribution,
                group.prob_zero_contribution,
            ),
            "contribution_q90_spearman": safe_spearman(
                group.contribution_q90, group.observed_contribution
            ),
            "tail_lift_surprise_spearman": safe_spearman(
                group.contribution_tail_lift,
                group.observed_contribution - group.contribution_mean,
            ),
            "raw_residual_mean": group.raw_residual_mean.mean(),
            "effective_sample_size": group.effective_sample_size.mean(),
            "max_donor_probability": group.max_donor_probability.max(),
        }
    )


def grouped_summary(frame, group_cols):
    return (
        frame.groupby(group_cols, observed=True, sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def probability_calibration(frame, probability_col, outcome_col, label):
    output = []
    for method, method_frame in frame.groupby("method", sort=True):
        method_frame = method_frame.copy()
        method_frame["probability_bin"] = pd.cut(
            method_frame[probability_col],
            bins=[-1e-9, 0.025, 0.05, 0.10, 0.20, 0.35, 1.0],
            labels=["0-2.5%", "2.5-5%", "5-10%", "10-20%", "20-35%", "35%+"],
        )
        calibration = (
            method_frame.groupby("probability_bin", observed=True)
            .agg(
                n=(outcome_col, "size"),
                predicted=(probability_col, "mean"),
                observed=(outcome_col, "mean"),
            )
            .reset_index()
        )
        calibration["method"] = method
        calibration["event"] = label
        output.append(calibration)
    return pd.concat(output, ignore_index=True)


def main():
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    max_season = builder.get_daily_max_template_season()
    print(f"Loading historical inputs through {max_season}")
    projections = builder.load_historical_projection_context(max_season)
    weekly = builder.load_weekly_points(max_season)
    templates = builder.build_weekly_templates(projections, weekly)
    forecasts = load_production_oos_forecasts(max_season)
    target_templates = build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = build_targets(target_templates)
    predictions = run_replay(templates, targets)

    predictions["period"] = np.select(
        [
            predictions.season.ge(RECENT_ORIGIN_START),
            predictions.season.ge(CORE_ORIGIN_START),
        ],
        ["2020-2025", "2017-2019"],
        default="pre_2017",
    )
    core = predictions[predictions.season.ge(CORE_ORIGIN_START)].copy()
    recent = predictions[predictions.season.ge(RECENT_ORIGIN_START)].copy()
    period_frames = []
    for label, frame in [
        ("production_oos_2017_2025", predictions),
        ("validation_ensemble_2017_2025", core),
        ("recent_2020_2025", recent),
    ]:
        summary = grouped_summary(frame, ["method"])
        summary.insert(0, "period", label)
        period_frames.append(summary)
    summary_overall = pd.concat(period_frames, ignore_index=True)
    summary_position = grouped_summary(core, ["method", "pos"])
    summary_experience = grouped_summary(
        core[core.pos.isin(["RB", "WR", "TE"])],
        ["method", "pos", "experience_group"],
    )
    summary_season = grouped_summary(predictions, ["method", "season"])
    calibration = pd.concat(
        [
            probability_calibration(core, "prob_plus3", "observed_plus3", "+3_ppg"),
            probability_calibration(core, "prob_plus5", "observed_plus5", "+5_ppg"),
            probability_calibration(core, "prob_impact", "observed_impact", "impact"),
            probability_calibration(
                core,
                "prob_zero_contribution",
                "observed_zero_contribution",
                "zero_contribution",
            ),
        ],
        ignore_index=True,
    )
    quantile_calibration = (
        core.groupby(["method", "pos"], observed=True)
        .agg(
            n=("player", "size"),
            observed_below_ppg_q10=(
                "ppg_q10",
                lambda values: np.mean(
                    core.loc[values.index, "observed_ppg"].to_numpy()
                    <= values.to_numpy()
                ),
            ),
            observed_below_ppg_q50=(
                "ppg_q50",
                lambda values: np.mean(
                    core.loc[values.index, "observed_ppg"].to_numpy()
                    <= values.to_numpy()
                ),
            ),
            observed_below_ppg_q90=(
                "ppg_q90",
                lambda values: np.mean(
                    core.loc[values.index, "observed_ppg"].to_numpy()
                    <= values.to_numpy()
                ),
            ),
            observed_below_contribution_q10=(
                "contribution_q10",
                lambda values: np.mean(
                    core.loc[values.index, "observed_contribution"].to_numpy()
                    <= values.to_numpy()
                ),
            ),
            observed_below_contribution_q50=(
                "contribution_q50",
                lambda values: np.mean(
                    core.loc[values.index, "observed_contribution"].to_numpy()
                    <= values.to_numpy()
                ),
            ),
            observed_below_contribution_q90=(
                "contribution_q90",
                lambda values: np.mean(
                    core.loc[values.index, "observed_contribution"].to_numpy()
                    <= values.to_numpy()
                ),
            ),
        )
        .reset_index()
    )

    predictions.to_csv(RESULTS / "target_predictions.csv", index=False)
    summary_overall.to_csv(RESULTS / "summary_overall.csv", index=False)
    summary_position.to_csv(RESULTS / "summary_by_position.csv", index=False)
    summary_experience.to_csv(RESULTS / "summary_by_experience.csv", index=False)
    summary_season.to_csv(RESULTS / "summary_by_season.csv", index=False)
    calibration.to_csv(RESULTS / "event_calibration.csv", index=False)
    quantile_calibration.to_csv(
        RESULTS / "quantile_calibration.csv", index=False
    )
    metadata = {
        "origin_start": ORIGIN_START,
        "core_origin_start": CORE_ORIGIN_START,
        "recent_origin_start": RECENT_ORIGIN_START,
        "max_season": int(max_season),
        "target_counts": TARGET_COUNTS,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "methods": list(METHODS),
        "target_forecast_source": (
            "Final_Validations_Resid beta model_spec_asof_year=2026 data_oos=1"
        ),
        "donor_forecast_source": (
            "production Weekly_Outcome_Templates historical_pred_fp_per_game"
        ),
        "forecast_rows": int(len(forecasts)),
        "target_template_rows": int(len(target_templates)),
        "runtime_seconds": time.perf_counter() - started,
        "future_donor_rows": 0,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary_overall.round(4).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
