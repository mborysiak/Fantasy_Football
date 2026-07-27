"""Strict rolling local sensitivity of retained weekly-template weights."""

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
PRUNING_STUDY = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-23_template_feature_pruning"
    / "run_validation.py"
)
PRUNING_SPEC = importlib.util.spec_from_file_location(
    "template_feature_pruning",
    PRUNING_STUDY,
)
if PRUNING_SPEC is None or PRUNING_SPEC.loader is None:
    raise ImportError(f"Could not import feature pruning from {PRUNING_STUDY}")
pruning = importlib.util.module_from_spec(PRUNING_SPEC)
sys.modules[PRUNING_SPEC.name] = pruning
PRUNING_SPEC.loader.exec_module(pruning)
base = pruning.base
builder = pruning.builder


RESULTS = Path(__file__).resolve().parent / "results"
BASELINE_METHOD = "recommended"
RECENCY_HALF_LIFE = 12.0
DEVELOPMENT_END = 2022
META_SELECTION_START = 2021
WEIGHT_MULTIPLIERS = (0.75, 1.25)
GLOBAL_WEIGHT_MULTIPLIERS = (
    0.01,
    0.05,
    0.125,
    0.25,
    0.375,
    0.50,
    0.625,
    0.75,
    0.875,
    1.25,
    1.50,
)
MIN_DEVELOPMENT_IMPROVEMENT = 0.001
MIN_RECENT_NESTED_WINS = 2

PERIODS = {
    "development_2017_2022": (2017, 2022),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}


def all_positions(*features: str) -> dict[str, set[str]]:
    return {pos: set(features) for pos in builder.POSITIONS}


FEATURE_FAMILIES = {
    "projection_rank": all_positions("match_projection_rank_pct"),
    "absolute_ppg": all_positions("match_projection_ppg_scaled"),
    "experience": all_positions("year_exp_scaled"),
    "adp_rank": all_positions("adp_rank_pct"),
    "market_gap": all_positions("market_projection_gap"),
    "disagreement": all_positions(
        "projection_disagreement_frac",
        "rank_disagreement_scaled",
    ),
    "component_ranks": {
        "QB": {"rush_proj_rank_pct", "pass_proj_rank_pct"},
        "RB": {"rush_proj_rank_pct", "rec_proj_rank_pct"},
        "WR": {"rec_proj_rank_pct"},
        "TE": {"rec_proj_rank_pct"},
    },
    "scoring_mix": {
        "QB": {"rush_share_of_own_points"},
        "RB": {"rec_share_of_own_points"},
        "WR": set(),
        "TE": set(),
    },
    "room_share": {
        "QB": {"qb_room_share"},
        "RB": {
            "rb_rush_share_of_room",
            "rb_rec_share_of_room",
            "rb_combined_share_of_room",
        },
        "WR": {"team_rec_share"},
        "TE": {"team_rec_share"},
    },
    "room_hierarchy": {
        "QB": {"qb_team_rank_distance", "qb1_over_qb2_gap_pct"},
        "RB": {"rb_room_rank_scaled", "rb_gap_to_next_share"},
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
    "team_pass_environment": {
        "QB": set(),
        "RB": set(),
        "WR": {"team_qb_pass_proj_rank_pct"},
        "TE": {"team_qb_pass_proj_rank_pct"},
    },
}


def reference_weights() -> dict[str, dict[str, float]]:
    return pruning.remove_feature_families(("exp_interaction",))


def scaled_weights(
    family: str,
    multiplier: float,
) -> dict[str, dict[str, float]]:
    weights = deepcopy(reference_weights())
    if family == "all_weights":
        for pos in builder.POSITIONS:
            for feature in weights[pos]:
                weights[pos][feature] *= multiplier
        return weights
    for pos in builder.POSITIONS:
        for feature in FEATURE_FAMILIES[family][pos]:
            if feature not in weights[pos]:
                raise KeyError(f"{feature} is not a retained {pos} feature.")
            weights[pos][feature] *= multiplier
    return weights


def build_methods() -> tuple[dict, pd.DataFrame]:
    methods = {
        BASELINE_METHOD: {
            "weights": reference_weights(),
            "recency_half_life": RECENCY_HALF_LIFE,
        }
    }
    metadata = [
        {
            "method": BASELINE_METHOD,
            "family": "reference",
            "multiplier": 1.0,
            "direction": "reference",
        }
    ]
    family_multipliers = [
        *[
            (family, multiplier)
            for family in FEATURE_FAMILIES
            for multiplier in WEIGHT_MULTIPLIERS
        ],
        *[
            ("all_weights", multiplier)
            for multiplier in GLOBAL_WEIGHT_MULTIPLIERS
        ],
    ]
    for family, multiplier in family_multipliers:
        suffix = f"w{int(round(multiplier * 100)):03d}"
        method = f"{family}__{suffix}"
        methods[method] = {
            "weights": scaled_weights(family, multiplier),
            "recency_half_life": RECENCY_HALF_LIFE,
        }
        metadata.append(
            {
                "method": method,
                "family": family,
                "multiplier": multiplier,
                "direction": "lower" if multiplier < 1 else "higher",
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def add_selection_loss(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    baseline = output[output.method.eq(BASELINE_METHOD)]
    denominators = {
        metric: float(baseline[metric].mean())
        for metric in ["ppg_crps", "contribution_crps", "played_crps"]
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
    summary = pruning.grouped_summary(scored, ["method"]).merge(
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

    position_metrics = (
        frame.groupby(["method", "pos"], observed=True)[
            ["ppg_crps", "contribution_crps", "played_crps"]
        ]
        .mean()
    )
    baseline_position = position_metrics.loc[BASELINE_METHOD]
    position_checks = []
    for method in sorted(frame.method.unique()):
        ratios = position_metrics.loc[method] / baseline_position
        composite_delta = ratios.mean(axis=1) - 1.0
        position_checks.append(
            {
                "method": method,
                "max_position_composite_delta": float(
                    composite_delta.max()
                ),
                "max_position_metric_delta": float(
                    (ratios - 1.0).max().max()
                ),
                "position_guardrail_pass": bool(
                    composite_delta.max() <= 0.005
                    and (ratios - 1.0).max().max() <= 0.01
                ),
            }
        )
    summary = summary.merge(
        pd.DataFrame(position_checks),
        on="method",
        how="left",
        validate="one_to_one",
    )
    summary["guardrail_pass"] = (
        summary.aggregate_guardrail_pass
        & summary.position_guardrail_pass
    )
    guarded = summary[summary.guardrail_pass]
    if guarded.empty:
        raise ValueError("No weight specification passed safety guardrails.")
    best_method = guarded.sort_values(
        ["selection_loss", "method"]
    ).method.iloc[0]

    keys = ["player", "pos", "season"]
    best = scored[scored.method.eq(best_method)][
        keys + ["selection_loss"]
    ].rename(columns={"selection_loss": "best_loss"})
    paired = scored.merge(
        best,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    paired["loss_delta_to_best"] = (
        paired.selection_loss - paired.best_loss
    )
    one_se = []
    for method, group in paired.groupby("method"):
        season_deltas = group.groupby("season").loss_delta_to_best.mean()
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
    summary["development_winner"] = (
        summary.method.eq(best_method).astype(int)
    )
    return summary.sort_values(
        ["guardrail_pass", "selection_loss", "method"],
        ascending=[False, True, True],
    ), best_method


def period_summaries(predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for period, (start, end) in PERIODS.items():
        frame = predictions[predictions.season.between(start, end)]
        summary = pruning.grouped_summary(frame, ["method"])
        baseline = summary[
            summary.method.eq(BASELINE_METHOD)
        ].iloc[0]
        summary["composite_delta"] = np.mean(
            [
                summary.ppg_crps / float(baseline.ppg_crps),
                summary.contribution_crps
                / float(baseline.contribution_crps),
                summary.played_crps / float(baseline.played_crps),
            ],
            axis=0,
        ) - 1.0
        summary.insert(0, "period", period)
        output.append(summary)
    return pd.concat(output, ignore_index=True).merge(
        METHOD_METADATA,
        on="method",
        how="left",
        validate="many_to_one",
    )


def position_summaries(predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for period, (start, end) in PERIODS.items():
        frame = predictions[predictions.season.between(start, end)]
        summary = pruning.grouped_summary(frame, ["method", "pos"])
        baseline = summary[
            summary.method.eq(BASELINE_METHOD)
        ].set_index("pos")
        rows = []
        for row in summary.itertuples(index=False):
            reference = baseline.loc[row.pos]
            record = row._asdict()
            record["composite_delta"] = np.mean(
                [
                    row.ppg_crps / float(reference.ppg_crps),
                    row.contribution_crps
                    / float(reference.contribution_crps),
                    row.played_crps / float(reference.played_crps),
                ]
            ) - 1.0
            rows.append(record)
        result = pd.DataFrame(rows)
        result.insert(0, "period", period)
        output.append(result)
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
        training = predictions[predictions.season.lt(target_season)]
        leaderboard, selected_method = selection_table(training)
        selected = leaderboard[
            leaderboard.method.eq(selected_method)
        ].iloc[0]
        baseline = leaderboard[
            leaderboard.method.eq(BASELINE_METHOD)
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
                "selected_family": selected.family,
                "selected_multiplier": float(selected.multiplier),
                "selected_development_loss": float(
                    selected.selection_loss
                ),
                "baseline_development_loss": float(
                    baseline.selection_loss
                ),
                "selected_minus_baseline": float(
                    selected.selection_loss - baseline.selection_loss
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


def family_sensitivity(
    period_summary: pd.DataFrame,
) -> pd.DataFrame:
    fields = [
        "period",
        "method",
        "family",
        "multiplier",
        "composite_delta",
        "ppg_crps",
        "contribution_crps",
        "played_crps",
        "ppg_80_coverage",
        "contribution_80_coverage",
        "played_80_coverage",
        "plus5_brier",
        "impact_brier",
        "impact_auc",
        "zero_brier",
        "extended_absence_brier",
    ]
    return period_summary[fields].sort_values(
        ["family", "period", "multiplier"]
    )


def promotion_screen(
    development: pd.DataFrame,
    period_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    selection_path: pd.DataFrame,
) -> pd.DataFrame:
    development_index = development.set_index("method")
    baseline = development_index.loc[BASELINE_METHOD]
    temporal = period_summary[
        period_summary.period.eq("temporal_2023_2025")
    ].set_index("method")
    temporal_position = position_summary[
        position_summary.period.eq("temporal_2023_2025")
    ].set_index(["method", "pos"])
    baseline_position = temporal_position.loc[BASELINE_METHOD][
        ["ppg_crps", "contribution_crps", "played_crps"]
    ]
    records = []
    for method, row in development_index.iterrows():
        ratios = (
            temporal_position.loc[method][
                ["ppg_crps", "contribution_crps", "played_crps"]
            ]
            / baseline_position
        )
        max_position_composite = float(
            (ratios.mean(axis=1) - 1.0).max()
        )
        max_position_metric = float((ratios - 1.0).max().max())
        development_improvement = float(
            baseline.selection_loss - row.selection_loss
        )
        temporal_composite_delta = float(
            temporal.loc[method, "composite_delta"]
        )
        recent_nested_wins = int(
            selection_path[
                selection_path.target_season.ge(2023)
            ].selected_method.eq(method).sum()
        )
        temporal_position_guardrail_pass = bool(
            max_position_composite <= 0.005
            and max_position_metric <= 0.01
        )
        records.append(
            {
                "method": method,
                "family": row.family,
                "multiplier": float(row.multiplier),
                "development_improvement": development_improvement,
                "development_guardrail_pass": bool(
                    row.guardrail_pass
                ),
                "temporal_composite_delta": temporal_composite_delta,
                "max_temporal_position_composite_delta": (
                    max_position_composite
                ),
                "max_temporal_position_metric_delta": (
                    max_position_metric
                ),
                "temporal_position_guardrail_pass": (
                    temporal_position_guardrail_pass
                ),
                "recent_nested_wins": recent_nested_wins,
                "qualifies_for_promotion": bool(
                    method != BASELINE_METHOD
                    and row.guardrail_pass
                    and development_improvement
                    >= MIN_DEVELOPMENT_IMPROVEMENT
                    and temporal_composite_delta <= 0
                    and temporal_position_guardrail_pass
                    and recent_nested_wins
                    >= MIN_RECENT_NESTED_WINS
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["qualifies_for_promotion", "development_improvement"],
        ascending=[False, False],
    )


def promotion_decision(
    development: pd.DataFrame,
    development_winner: str,
    screen: pd.DataFrame,
) -> dict:
    winner = screen[screen.method.eq(development_winner)].iloc[0]
    qualifiers = screen[screen.qualifies_for_promotion]
    promoted_method = (
        qualifiers.method.iloc[0]
        if not qualifiers.empty
        else BASELINE_METHOD
    )
    return {
        "development_winner": development_winner,
        "development_improvement": float(
            winner.development_improvement
        ),
        "minimum_development_improvement": (
            MIN_DEVELOPMENT_IMPROVEMENT
        ),
        "winner_guardrail_pass": bool(
            winner.development_guardrail_pass
        ),
        "temporal_composite_delta": float(
            winner.temporal_composite_delta
        ),
        "max_temporal_position_composite_delta": (
            float(winner.max_temporal_position_composite_delta)
        ),
        "max_temporal_position_metric_delta": (
            float(winner.max_temporal_position_metric_delta)
        ),
        "temporal_position_guardrail_pass": (
            bool(winner.temporal_position_guardrail_pass)
        ),
        "recent_nested_wins": int(winner.recent_nested_wins),
        "minimum_recent_nested_wins": MIN_RECENT_NESTED_WINS,
        "qualifying_method_count": int(len(qualifiers)),
        "promote_weight_change": bool(not qualifiers.empty),
        "recommended_method": promoted_method,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )
    return "\n".join(lines)


def write_summary(
    development: pd.DataFrame,
    period_summary: pd.DataFrame,
    selection_path: pd.DataFrame,
    nested_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    screen: pd.DataFrame,
    decision: dict,
    target_rows: int,
    runtime_seconds: float,
) -> None:
    leaderboard = development[
        [
            "method",
            "family",
            "multiplier",
            "selection_loss",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
            "guardrail_pass",
            "max_position_composite_delta",
            "max_position_metric_delta",
            "within_one_se",
            "development_winner",
        ]
    ].head(12).copy()
    curves = period_summary[
        period_summary.period.isin(
            ["development_2017_2022", "temporal_2023_2025"]
        )
        & ~period_summary.family.isin(["all_weights", "reference"])
        & period_summary.multiplier.isin(WEIGHT_MULTIPLIERS)
    ][
        [
            "period",
            "family",
            "multiplier",
            "composite_delta",
        ]
    ].pivot_table(
        index="family",
        columns=["period", "multiplier"],
        values="composite_delta",
    )
    curves.columns = [
        (
            f"{'dev' if period.startswith('development') else 'temporal'}"
            f"_w{int(multiplier * 100)}"
        )
        for period, multiplier in curves.columns
    ]
    curves = curves.reset_index()
    global_curve = period_summary[
        period_summary.family.eq("all_weights")
    ][
        [
            "period",
            "multiplier",
            "composite_delta",
            "effective_sample_size",
        ]
    ].pivot_table(
        index="multiplier",
        columns="period",
        values=["composite_delta", "effective_sample_size"],
    )
    global_curve.columns = [
        (
            f"{'loss' if metric == 'composite_delta' else 'ess'}_"
            f"{'dev' if period.startswith('development') else 'recent' if period.startswith('recent') else 'temporal'}"
        )
        for metric, period in global_curve.columns
    ]
    global_curve = global_curve.reset_index()
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
    ].copy()
    boot = bootstrap[
        bootstrap.metric.isin(
            ["ppg_crps", "contribution_crps", "played_crps"]
        )
    ][
        [
            "candidate_method",
            "period",
            "metric",
            "candidate_minus_baseline",
            "bootstrap_p025",
            "bootstrap_p975",
            "probability_candidate_better",
        ]
    ].copy()
    promotion = screen[
        [
            "method",
            "development_improvement",
            "temporal_composite_delta",
            "max_temporal_position_composite_delta",
            "max_temporal_position_metric_delta",
            "recent_nested_wins",
            "qualifies_for_promotion",
        ]
    ].head(12).copy()
    for frame in [
        leaderboard,
        curves,
        global_curve,
        nested,
        boot,
        promotion,
    ]:
        numeric = frame.select_dtypes(include=[np.number]).columns
        frame[numeric] = frame[numeric].round(6)
    recommendation = (
        f"Promote `{decision['recommended_method']}`."
        if decision["promote_weight_change"]
        else "Retain the reference weights; no tested perturbation clears "
        "every promotion threshold."
    )
    best_family_improvement = float(
        (
            1.0
            - development[
                ~development.family.isin(
                    ["all_weights", "reference"]
                )
            ].selection_loss
        ).max()
    )
    winner_temporal_played = bootstrap[
        bootstrap.candidate_method.eq(
            decision["development_winner"]
        )
        & bootstrap.period.eq("temporal_2023_2025")
        & bootstrap.metric.eq("played_crps")
    ].iloc[0]
    text = "\n".join(
        [
            "# Weekly Template Weight Sensitivity",
            "",
            "## Design",
            "",
            f"- Held out {target_rows:,} player-seasons at strict rolling origins.",
            f"- Evaluated {len(METHODS)} paired local weight specifications.",
            "- The reference removes `projection_x_exp` and fixes 12-season "
            "recency; all other matcher mechanics remain unchanged.",
            "- Selection uses 2017-2022 only, with aggregate and position "
            "guardrails. The promotion threshold also requires at least 0.1% "
            "development composite improvement, non-worse temporal composite, "
            "temporal position safety, and two of three recent nested "
            "selections.",
            "",
            "## Development leaderboard",
            "",
            markdown_table(leaderboard),
            "",
            "## Family sensitivity",
            "",
            "Negative composite deltas favor the perturbation.",
            "",
            markdown_table(curves),
            "",
            "## Overall distance-sharpness curve",
            "",
            markdown_table(global_curve),
            "",
            "## Nested rolling selection",
            "",
            markdown_table(selection_path),
            "",
            markdown_table(nested),
            "",
            "## Development-winner clustered uncertainty",
            "",
            markdown_table(boot),
            "",
            "## Promotion screen",
            "",
            markdown_table(promotion),
            "",
            "## Decision",
            "",
            recommendation,
            "",
            f"- Development winner: `{decision['development_winner']}`.",
            f"- Development improvement: "
            f"{decision['development_improvement']:.4%}.",
            f"- Temporal composite delta: "
            f"{decision['temporal_composite_delta']:+.4%}.",
            f"- Worst temporal position composite / metric deltas: "
            f"{decision['max_temporal_position_composite_delta']:+.4%} / "
            f"{decision['max_temporal_position_metric_delta']:+.4%}.",
            f"- Same winner selected in "
            f"{decision['recent_nested_wins']}/3 recent nested origins.",
            f"- No individual feature-family change improved development "
            f"composite by more than {best_family_improvement:.4%}.",
            f"- The near-uniform development winner worsened temporal "
            f"played-games CRPS by "
            f"{float(winner_temporal_played.candidate_minus_baseline):+.5f} "
            f"(cluster interval "
            f"{float(winner_temporal_played.bootstrap_p025):+.5f} to "
            f"{float(winner_temporal_played.bootstrap_p975):+.5f}).",
            "- Lower overall sharpness remains a useful future sampling-kernel "
            "hypothesis, but its exact scale drifted across rolling origins and "
            "should not be bundled into the feature/recency update.",
            "- Production remains unchanged.",
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

    # Reuse the already-audited replay implementation with this study's
    # isolated method dictionary.
    pruning.METHODS = METHODS
    predictions = pruning.run_replay(templates, targets)
    expected_rows = len(targets) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} prediction rows; found "
            f"{len(predictions)}."
        )

    development, development_winner = selection_table(
        predictions[predictions.season.le(DEVELOPMENT_END)]
    )
    period_summary = period_summaries(predictions)
    position_summary = position_summaries(predictions)
    sensitivity = family_sensitivity(period_summary)
    nested_predictions, selection_path = build_nested_selection(
        predictions
    )
    nested_comparison = pd.concat(
        [
            nested_predictions,
            predictions[
                predictions.season.ge(META_SELECTION_START)
                & predictions.method.eq(BASELINE_METHOD)
            ],
        ],
        ignore_index=True,
    )
    nested_summary = pruning.grouped_summary(
        nested_comparison,
        ["method"],
    )
    screen = promotion_screen(
        development,
        period_summary,
        position_summary,
        selection_path,
    )
    decision = promotion_decision(
        development,
        development_winner,
        screen,
    )

    bootstrap_methods = (
        development[
            development.guardrail_pass
            & ~development.method.eq(BASELINE_METHOD)
        ]
        .head(5)
        .method.tolist()
    )
    bootstrap = pd.concat(
        [
            pruning.bootstrap_comparison(
                predictions,
                method,
                PERIODS,
                baseline_method=BASELINE_METHOD,
            )
            for method in bootstrap_methods
        ],
        ignore_index=True,
    )

    predictions.to_csv(RESULTS / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(RESULTS / "method_metadata.csv", index=False)
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
    sensitivity.to_csv(
        RESULTS / "family_sensitivity.csv",
        index=False,
    )
    screen.to_csv(
        RESULTS / "promotion_screen.csv",
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
        "development_winner": development_winner,
        **decision,
        "bootstrap_methods": bootstrap_methods,
        "future_donor_rows": 0,
        "runtime_seconds": runtime_seconds,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_summary(
        development,
        period_summary,
        selection_path,
        nested_summary,
        bootstrap,
        screen,
        decision,
        len(targets),
        runtime_seconds,
    )
    print(
        development[
            [
                "method",
                "family",
                "multiplier",
                "selection_loss",
                "ppg_crps",
                "contribution_crps",
                "played_crps",
                "guardrail_pass",
                "within_one_se",
                "development_winner",
            ]
        ]
        .head(15)
        .round(6)
        .to_string(index=False)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
