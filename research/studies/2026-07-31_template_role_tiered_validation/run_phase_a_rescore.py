"""Rescore saved weekly-template experiments under a role-tiered policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
RESULTS = STUDY_DIR / "results"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20260731

PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}
TIER_LIMITS = {
    "core_strict": {"QB": 12, "RB": 24, "WR": 36, "TE": 12},
    "core_main": {"QB": 18, "RB": 36, "WR": 48, "TE": 18},
    "core_broad": {"QB": 24, "RB": 48, "WR": 60, "TE": 24},
}
METRICS = (
    "ppg_crps",
    "contribution_crps",
    "played_crps",
)


@dataclass(frozen=True)
class Source:
    study: str
    result_dir: str
    league: str
    baseline: str
    evidence_class: str

    @property
    def predictions_path(self) -> Path:
        return (
            REPO_ROOT
            / "research"
            / "studies"
            / self.study
            / self.result_dir
            / "target_predictions.csv"
        )


SOURCES = (
    Source(
        "2026-07-23_template_context_ablation",
        "results",
        "dk_legacy",
        "production_baseline",
        "historical_superseded",
    ),
    Source(
        "2026-07-23_template_feature_pruning",
        "results",
        "dk_legacy",
        "full",
        "historical_selection_audit",
    ),
    Source(
        "2026-07-23_template_weight_sensitivity",
        "results",
        "dk_legacy",
        "recommended",
        "historical_nomination_only",
    ),
    Source(
        "2026-07-29_template_projection_weight_bump",
        "results",
        "dk",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-29_template_projection_weight_bump",
        "results_beta",
        "beta",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_template_receiver_rate_ablation",
        "results",
        "dk",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_template_receiver_rate_ablation",
        "results_beta",
        "beta",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_wr_template_ppg_profile_tradeoff",
        "results",
        "dk",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_wr_template_ppg_profile_tradeoff",
        "results_beta",
        "beta",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_template_projection_trajectory",
        "results",
        "dk",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_template_projection_trajectory",
        "results_beta",
        "beta",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_template_height_weight_ablation",
        "results",
        "dk",
        "production",
        "current_direct",
    ),
    Source(
        "2026-07-30_template_height_weight_ablation",
        "results_beta",
        "beta",
        "production",
        "current_direct",
    ),
)


def read_source(source: Source) -> pd.DataFrame:
    frame = pd.read_csv(source.predictions_path)
    required = {
        "player",
        "pos",
        "season",
        "predicted_ppg",
        "avg_pick",
        "method",
        "ppg_mean",
        "observed_ppg",
        "contribution_mean",
        "observed_contribution",
        "played_mean",
        "observed_played",
        "ppg_80_covered",
        "observed_extended_absence",
        "prob_extended_absence",
        "observed_zero_active",
        "prob_zero_active",
        *METRICS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source.predictions_path} missing {sorted(missing)}")
    target_cols = ["player", "pos", "season"]
    target_count = frame[target_cols].drop_duplicates().shape[0]
    methods = sorted(frame.method.unique())
    if len(frame) != target_count * len(methods):
        raise ValueError(f"Incomplete method-target grid: {source.predictions_path}")
    if source.baseline not in methods:
        raise ValueError(f"Missing baseline {source.baseline}: {source.study}")

    target_order = (
        frame.drop_duplicates(target_cols)
        .sort_values(
            ["season", "pos", "predicted_ppg", "avg_pick", "player"],
            ascending=[True, True, False, True, True],
        )
        .copy()
    )
    target_order["preseason_pos_rank"] = (
        target_order.groupby(["season", "pos"]).cumcount() + 1
    )
    reconstructed_rank = target_order[
        target_cols + ["preseason_pos_rank"]
    ]
    if "preseason_pos_rank" in frame.columns:
        check = frame.merge(
            reconstructed_rank.rename(
                columns={"preseason_pos_rank": "reconstructed_rank"}
            ),
            on=target_cols,
            how="left",
            validate="many_to_one",
        )
        if not check.preseason_pos_rank.eq(check.reconstructed_rank).all():
            raise ValueError(
                f"Stored position ranks do not reproduce: {source.study}"
            )
    else:
        frame = frame.merge(
            reconstructed_rank,
            on=target_cols,
            how="left",
            validate="many_to_one",
        )
    frame["study"] = source.study
    frame["result_dir"] = source.result_dir
    frame["league"] = source.league
    frame["baseline_method"] = source.baseline
    frame["evidence_class"] = source.evidence_class
    frame["experiment"] = source.study + "__" + source.result_dir
    return frame


def tier_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    limits = {
        tier: frame.pos.map(position_limits)
        for tier, position_limits in TIER_LIMITS.items()
    }
    output = {
        tier: frame.preseason_pos_rank.le(limit)
        for tier, limit in limits.items()
    }
    output["depth_main"] = ~output["core_main"]
    if "qb_team_rank" in frame.columns:
        output["core_main_qb1"] = output["core_main"] & (
            frame.pos.ne("QB")
            | pd.to_numeric(frame.qb_team_rank, errors="coerce").eq(1)
        )
    output["all_saved"] = pd.Series(True, index=frame.index)
    return output


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "ppg_crps": float(frame.ppg_crps.mean()),
        "contribution_crps": float(frame.contribution_crps.mean()),
        "played_crps": float(frame.played_crps.mean()),
        "ppg_bias": float((frame.ppg_mean - frame.observed_ppg).mean()),
        "contribution_bias": float(
            (frame.contribution_mean - frame.observed_contribution).mean()
        ),
        "played_bias": float(
            (frame.played_mean - frame.observed_played).mean()
        ),
        "ppg_80_coverage": float(frame.ppg_80_covered.mean()),
        "extended_absence_calibration": float(
            frame.prob_extended_absence.mean()
            - frame.observed_extended_absence.mean()
        ),
        "extended_absence_brier": float(
            np.mean(
                (
                    frame.prob_extended_absence
                    - frame.observed_extended_absence
                )
                ** 2
            )
        ),
        "zero_active_calibration": float(
            frame.prob_zero_active.mean() - frame.observed_zero_active.mean()
        ),
        "predicted_missed_games": float(16.0 - frame.played_mean.mean()),
        "observed_missed_games": float(16.0 - frame.observed_played.mean()),
    }


def metric_table(
    predictions: pd.DataFrame,
    sources: tuple[Source, ...] = SOURCES,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        experiment = source.study + "__" + source.result_dir
        source_frame = predictions[predictions.experiment.eq(experiment)]
        masks = tier_masks(source_frame)
        for tier, mask in masks.items():
            tier_frame = source_frame[mask]
            for period, (start, end) in PERIODS.items():
                period_frame = tier_frame[tier_frame.season.between(start, end)]
                for method, method_frame in period_frame.groupby("method"):
                    if method_frame.empty:
                        continue
                    rows.append(
                        {
                            "study": source.study,
                            "result_dir": source.result_dir,
                            "experiment": experiment,
                            "league": source.league,
                            "evidence_class": source.evidence_class,
                            "baseline_method": source.baseline,
                            "method": method,
                            "tier": tier,
                            "period": period,
                            **summarize(method_frame),
                        }
                    )
    return pd.DataFrame(rows)


def add_baseline_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "tier", "period"]
    baseline = metrics[metrics.method.eq(metrics.baseline_method)].copy()
    value_cols = [
        *METRICS,
        "ppg_bias",
        "contribution_bias",
        "played_bias",
        "ppg_80_coverage",
        "extended_absence_calibration",
        "extended_absence_brier",
        "zero_active_calibration",
    ]
    baseline = baseline[keys + value_cols].rename(
        columns={column: f"baseline_{column}" for column in value_cols}
    )
    output = metrics.merge(
        baseline,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    for metric in METRICS:
        output[f"{metric}_delta"] = (
            output[metric] - output[f"baseline_{metric}"]
        )
        output[f"{metric}_relative_delta"] = (
            output[metric] / output[f"baseline_{metric}"] - 1.0
        )
    for metric in (
        "ppg_bias",
        "contribution_bias",
        "played_bias",
        "extended_absence_calibration",
        "zero_active_calibration",
    ):
        output[f"abs_{metric}_degradation"] = (
            output[metric].abs() - output[f"baseline_{metric}"].abs()
        )
    output["ppg_80_coverage_delta"] = (
        output.ppg_80_coverage - output.baseline_ppg_80_coverage
    )
    return output


def bootstrap_intervals(
    predictions: pd.DataFrame,
    sources: tuple[Source, ...] = SOURCES,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for source in sources:
        experiment = source.study + "__" + source.result_dir
        source_frame = predictions[
            predictions.experiment.eq(experiment)
        ].copy()
        methods = sorted(source_frame.method.unique())
        masks = tier_masks(source_frame)
        for tier in (
            "core_strict",
            "core_main",
            "core_broad",
            "depth_main",
        ):
            tier_frame = source_frame[masks[tier]]
            for period in (
                "development_2017_2022",
                "temporal_2023_2025",
            ):
                start, end = PERIODS[period]
                scoped = tier_frame[tier_frame.season.between(start, end)]
                keys = ["player", "pos", "season"]
                fields = keys + list(METRICS)
                baseline = scoped[scoped.method.eq(source.baseline)][fields]
                for method in methods:
                    if method == source.baseline:
                        continue
                    candidate = scoped[scoped.method.eq(method)][fields]
                    paired = candidate.merge(
                        baseline,
                        on=keys,
                        suffixes=("_candidate", "_baseline"),
                        validate="one_to_one",
                    )
                    if paired.empty:
                        continue
                    seasons = np.sort(paired.season.unique())
                    season_frames = {
                        season: paired[paired.season.eq(season)]
                        for season in seasons
                    }
                    for metric in METRICS:
                        season_deltas = np.asarray(
                            [
                                float(
                                    (
                                        season_frames[season][
                                            f"{metric}_candidate"
                                        ]
                                        - season_frames[season][
                                            f"{metric}_baseline"
                                        ]
                                    ).mean()
                                )
                                for season in seasons
                            ]
                        )
                        sampled_indices = rng.integers(
                            0,
                            len(seasons),
                            size=(BOOTSTRAP_SAMPLES, len(seasons)),
                        )
                        values = season_deltas[sampled_indices].mean(axis=1)
                        observed = float(
                            (
                                paired[f"{metric}_candidate"]
                                - paired[f"{metric}_baseline"]
                            ).mean()
                        )
                        output.append(
                            {
                                "study": source.study,
                                "result_dir": source.result_dir,
                                "experiment": experiment,
                                "league": source.league,
                                "evidence_class": source.evidence_class,
                                "candidate_method": method,
                                "baseline_method": source.baseline,
                                "tier": tier,
                                "period": period,
                                "metric": metric,
                                "n": len(paired),
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
                                "season_cluster_se": float(np.std(values)),
                            }
                        )
    return pd.DataFrame(output)


def position_ppg_guardrails(
    predictions: pd.DataFrame,
    sources: tuple[Source, ...] = SOURCES,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        experiment = source.study + "__" + source.result_dir
        scoped = predictions[predictions.experiment.eq(experiment)].copy()
        scoped = scoped[tier_masks(scoped)["core_main"]]
        scoped = scoped[scoped.season.between(*PERIODS["development_2017_2022"])]
        position_metrics = (
            scoped.groupby(["method", "pos"], as_index=False).ppg_crps.mean()
        )
        baseline = position_metrics[
            position_metrics.method.eq(source.baseline)
        ][["pos", "ppg_crps"]].rename(
            columns={"ppg_crps": "baseline_position_ppg_crps"}
        )
        compared = position_metrics.merge(
            baseline, on="pos", how="left", validate="many_to_one"
        )
        compared["position_ppg_relative_delta"] = (
            compared.ppg_crps / compared.baseline_position_ppg_crps - 1.0
        )
        for method, group in compared.groupby("method"):
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "max_position_ppg_relative_delta": float(
                        group.position_ppg_relative_delta.max()
                    ),
                    "worst_ppg_position": str(
                        group.loc[
                            group.position_ppg_relative_delta.idxmax(), "pos"
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def screen_candidates(
    deltas: pd.DataFrame,
    intervals: pd.DataFrame,
    position_guardrails: pd.DataFrame,
) -> pd.DataFrame:
    main = deltas[
        deltas.tier.eq("core_main")
        & deltas.period.eq("development_2017_2022")
        & ~deltas.method.eq(deltas.baseline_method)
    ].copy()
    ppg_intervals = intervals[
        intervals.tier.eq("core_main")
        & intervals.period.eq("development_2017_2022")
        & intervals.metric.eq("ppg_crps")
    ][
        [
            "experiment",
            "candidate_method",
            "bootstrap_p025",
            "bootstrap_p975",
            "probability_candidate_better",
            "season_cluster_se",
        ]
    ].rename(columns={"candidate_method": "method"})
    main = main.merge(
        ppg_intervals,
        on=["experiment", "method"],
        how="left",
        validate="one_to_one",
    )
    main = main.merge(
        position_guardrails,
        on=["experiment", "method"],
        how="left",
        validate="one_to_one",
    )
    main["one_se_near_best"] = False
    for _, indices in main.groupby("experiment").groups.items():
        group = main.loc[indices]
        best_index = group.ppg_crps_delta.idxmin()
        threshold = (
            float(main.loc[best_index, "ppg_crps_delta"])
            + float(main.loc[best_index, "season_cluster_se"])
        )
        main.loc[indices, "one_se_near_best"] = group.ppg_crps_delta.le(
            threshold
        )

    recent = deltas[
        deltas.tier.eq("core_main")
        & deltas.period.eq("temporal_2023_2025")
    ][
        [
            "experiment",
            "method",
            "ppg_crps_relative_delta",
            "contribution_crps_relative_delta",
        ]
    ].rename(
        columns={
            "ppg_crps_relative_delta": "temporal_ppg_relative_delta",
            "contribution_crps_relative_delta": (
                "temporal_contribution_relative_delta"
            ),
        }
    )
    depth = deltas[
        deltas.tier.eq("depth_main")
        & deltas.period.eq("development_2017_2022")
    ][
        [
            "experiment",
            "method",
            *[f"{metric}_relative_delta" for metric in METRICS],
        ]
    ].copy()
    depth["depth_legacy_composite_delta"] = depth[
        [f"{metric}_relative_delta" for metric in METRICS]
    ].mean(axis=1)
    depth = depth.rename(
        columns={
            f"{metric}_relative_delta": f"depth_{metric}_relative_delta"
            for metric in METRICS
        }
    )
    main = main.merge(
        recent,
        on=["experiment", "method"],
        how="left",
        validate="one_to_one",
    ).merge(
        depth,
        on=["experiment", "method"],
        how="left",
        validate="one_to_one",
    )
    main["core_ppg_improves"] = main.ppg_crps_delta.lt(0)
    main["core_contribution_guardrail"] = (
        main.contribution_crps_relative_delta.le(0.0025)
    )
    main["core_played_bias_guardrail"] = (
        main.abs_played_bias_degradation.le(0.15)
    )
    main["core_absence_guardrail"] = (
        main.abs_extended_absence_calibration_degradation.le(0.01)
    )
    main["core_coverage_guardrail"] = main.ppg_80_coverage_delta.ge(-0.01)
    main["core_position_ppg_guardrail"] = (
        main.max_position_ppg_relative_delta.le(0.01)
    )
    main["temporal_ppg_guardrail"] = (
        main.temporal_ppg_relative_delta.le(0.005)
    )
    main["depth_composite_guardrail"] = (
        main.depth_legacy_composite_delta.isna()
        | main.depth_legacy_composite_delta.le(0.005)
    )
    depth_component_cols = [
        f"depth_{metric}_relative_delta" for metric in METRICS
    ]
    main["depth_component_guardrail"] = (
        main[depth_component_cols].isna().all(axis=1)
        | main[depth_component_cols].max(axis=1).le(0.01)
    )
    main["phase_a_core_screen_pass"] = (
        main.core_ppg_improves
        & main.core_contribution_guardrail
        & main.core_played_bias_guardrail
        & main.core_absence_guardrail
        & main.core_coverage_guardrail
        & main.core_position_ppg_guardrail
        & main.temporal_ppg_guardrail
        & main.depth_composite_guardrail
        & main.depth_component_guardrail
    )
    main["direct_fresh_replay_eligible"] = (
        main.evidence_class.isin(["current_direct", "fresh_expanded"])
        & main.phase_a_core_screen_pass
    )
    return main.sort_values(
        [
            "direct_fresh_replay_eligible",
            "phase_a_core_screen_pass",
            "ppg_crps_delta",
        ],
        ascending=[False, False, True],
    )


def scope_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    scoped = predictions.copy()
    scoped["target_key"] = (
        scoped.player.astype(str)
        + "|"
        + scoped.pos.astype(str)
        + "|"
        + scoped.season.astype(str)
    )
    return (
        scoped.groupby(
            [
                "study",
                "result_dir",
                "experiment",
                "league",
                "evidence_class",
                "baseline_method",
            ],
            as_index=False,
        )
        .agg(
            rows=("player", "size"),
            targets=("target_key", "nunique"),
            methods=("method", "nunique"),
            min_season=("season", "min"),
            max_season=("season", "max"),
        )
    )


def cross_league_screen(screen: pd.DataFrame) -> pd.DataFrame:
    direct = screen[
        screen.evidence_class.isin(["current_direct", "fresh_expanded"])
    ].copy()
    rows = []
    for (study, method), group in direct.groupby(["study", "method"]):
        rows.append(
            {
                "study": study,
                "method": method,
                "league_count": int(group.league.nunique()),
                "leagues": ",".join(sorted(group.league.unique())),
                "mean_development_ppg_relative_delta": float(
                    group.ppg_crps_relative_delta.mean()
                ),
                "worst_development_ppg_relative_delta": float(
                    group.ppg_crps_relative_delta.max()
                ),
                "mean_development_contribution_relative_delta": float(
                    group.contribution_crps_relative_delta.mean()
                ),
                "worst_temporal_ppg_relative_delta": float(
                    group.temporal_ppg_relative_delta.max()
                ),
                "all_leagues_ppg_improve": bool(
                    group.ppg_crps_relative_delta.lt(0).all()
                ),
                "all_leagues_screen_pass": bool(
                    group.phase_a_core_screen_pass.all()
                ),
                "all_leagues_one_se_near_best": bool(
                    group.one_se_near_best.all()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["all_leagues_screen_pass", "mean_development_ppg_relative_delta"],
        ascending=[False, True],
    )


def write_findings(
    screen: pd.DataFrame, cross_league: pd.DataFrame
) -> None:
    direct = screen[screen.evidence_class.eq("current_direct")]
    passes = direct[direct.phase_a_core_screen_pass]
    nominations = screen[
        screen.evidence_class.eq("historical_nomination_only")
        & screen.phase_a_core_screen_pass
    ]
    best = direct.nsmallest(12, "ppg_crps_delta")[
        [
            "league",
            "study",
            "method",
            "ppg_crps_delta",
            "contribution_crps_relative_delta",
            "played_crps_relative_delta",
            "abs_played_bias_degradation",
            "abs_extended_absence_calibration_degradation",
            "probability_candidate_better",
            "phase_a_core_screen_pass",
        ]
    ]
    replicated = cross_league[
        cross_league.league_count.eq(2)
        & cross_league.all_leagues_screen_pass
    ]
    lines = [
        "# Phase A findings",
        "",
        "This file is generated by `run_phase_a_rescore.py`.",
        "",
        f"- Current-lineage candidate rows evaluated: {len(direct)}.",
        f"- Current-lineage rows passing the mechanical core screen: {len(passes)}.",
        f"- Historical weight-sensitivity nominations passing: {len(nominations)}.",
        "- A single-league pass is not a promotion decision; paired DK/beta and "
        "fresh expanded-cohort evidence are required.",
        "",
        "## Strongest current-lineage development PPG deltas",
        "",
        markdown_table(best),
        "",
        "## Replicated mechanical passes",
        "",
        markdown_table(replicated),
        "",
    ]
    if not passes.empty:
        lines.extend(
            [
                "## Mechanical Phase A passes",
                "",
                markdown_table(passes[
                    [
                        "league",
                        "study",
                        "method",
                        "ppg_crps_delta",
                        "contribution_crps_relative_delta",
                        "played_crps_relative_delta",
                        "temporal_ppg_relative_delta",
                    ]
                ]),
                "",
            ]
        )
    (RESULTS / "phase_a_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(
        [read_source(source) for source in SOURCES], ignore_index=True
    )
    metrics = metric_table(predictions)
    deltas = add_baseline_deltas(metrics)
    intervals = bootstrap_intervals(predictions)
    position_guardrails = position_ppg_guardrails(predictions)
    screen = screen_candidates(deltas, intervals, position_guardrails)
    cross_league = cross_league_screen(screen)
    scopes = scope_summary(predictions)

    scopes.to_csv(RESULTS / "source_scope.csv", index=False)
    metrics.to_csv(RESULTS / "role_tier_metrics.csv", index=False)
    deltas.to_csv(RESULTS / "candidate_deltas.csv", index=False)
    intervals.to_csv(RESULTS / "bootstrap_intervals.csv", index=False)
    position_guardrails.to_csv(
        RESULTS / "position_ppg_guardrails.csv", index=False
    )
    screen.to_csv(RESULTS / "phase_a_candidate_screen.csv", index=False)
    cross_league.to_csv(RESULTS / "cross_league_screen.csv", index=False)
    write_findings(screen, cross_league)
    metadata = {
        "source_count": len(SOURCES),
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "bootstrap_rows": len(intervals),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "current_direct_candidate_rows": int(
            screen.evidence_class.eq("current_direct").sum()
        ),
        "current_direct_screen_passes": int(
            screen.direct_fresh_replay_eligible.sum()
        ),
        "production_changed": False,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
