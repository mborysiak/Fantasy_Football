"""Combine DK/beta roster replays and apply the frozen Phase-C gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results_phase_c"
LEAGUE_RESULTS = {
    "dk": STUDY_DIR / "results_phase_c_dk",
    "beta": STUDY_DIR / "results_phase_c_beta",
}
PERIODS = ("development_2017_2022", "temporal_2023_2025")
ROSTER_SCORE_NONINFERIORITY = 0.005
MISSED_WEEK_BIAS_PER_PLAYER_MARGIN = 0.15
ROSTER_SIZE = 20


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{float(value):.6f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    bootstraps = []
    for league, directory in LEAGUE_RESULTS.items():
        summary = pd.read_csv(directory / "summary.csv")
        summary.insert(0, "league", league)
        summaries.append(summary)
        bootstrap = pd.read_csv(directory / "season_bootstrap.csv")
        bootstrap.insert(0, "league", league)
        bootstraps.append(bootstrap)
    summaries = pd.concat(summaries, ignore_index=True)
    bootstraps = pd.concat(bootstraps, ignore_index=True)

    scoped = summaries[summaries.period.isin(PERIODS)]
    keys = ["league", "period"]
    candidate = scoped[scoped.matcher.eq("flatter_w025_all")]
    baseline = scoped[scoped.matcher.eq("production")]
    decisions = candidate.merge(
        baseline,
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    decisions["score_crps_relative_delta"] = (
        decisions.score_crps_candidate / decisions.score_crps_baseline - 1.0
    )
    decisions["abs_missed_week_bias_degradation_per_player"] = (
        decisions.zero_player_weeks_bias_candidate.abs()
        - decisions.zero_player_weeks_bias_baseline.abs()
    ) / ROSTER_SIZE
    decisions["zero_player_weeks_crps_relative_delta"] = (
        decisions.zero_player_weeks_crps_candidate
        / decisions.zero_player_weeks_crps_baseline
        - 1.0
    )
    decisions["zero_active_players_crps_relative_delta"] = (
        decisions.zero_active_players_crps_candidate
        / decisions.zero_active_players_crps_baseline
        - 1.0
    )
    decisions["roster_score_guardrail"] = (
        decisions.score_crps_relative_delta.le(
            ROSTER_SCORE_NONINFERIORITY
        )
    )
    decisions["missed_week_bias_guardrail"] = (
        decisions.abs_missed_week_bias_degradation_per_player.le(
            MISSED_WEEK_BIAS_PER_PLAYER_MARGIN
        )
    )
    promote = bool(
        decisions.roster_score_guardrail.all()
        and decisions.missed_week_bias_guardrail.all()
    )
    decisions["phase_c_joint_pass"] = promote

    decision_columns = [
        "league",
        "period",
        "score_crps_candidate",
        "score_crps_baseline",
        "score_crps_relative_delta",
        "abs_missed_week_bias_degradation_per_player",
        "zero_player_weeks_crps_relative_delta",
        "zero_active_players_crps_relative_delta",
        "roster_score_guardrail",
        "missed_week_bias_guardrail",
        "phase_c_joint_pass",
    ]
    decisions[decision_columns].to_csv(
        RESULTS / "decision_table.csv", index=False
    )
    summaries.to_csv(RESULTS / "combined_summary.csv", index=False)
    bootstraps.to_csv(RESULTS / "combined_bootstrap.csv", index=False)

    score_intervals = bootstraps[
        bootstraps.metric.eq("score_crps")
        & bootstraps.period.isin(PERIODS)
    ]
    lines = [
        "# Phase C findings",
        "",
        "The Phase-B finalist was evaluated on 1,296 paired historical "
        "20-player best-ball rosters per league. Each weekly lineup naturally "
        "uses roster depth to replace missed player-weeks.",
        "",
        f"**Promotion decision: {'PROMOTE' if promote else 'DO NOT PROMOTE'}.**",
        "",
        "The frozen roster-score CRPS non-inferiority margin is +0.5%. "
        "Individual played-games CRPS is not a Phase-C gate.",
        "",
        "## Decision table",
        "",
        markdown_table(decisions[decision_columns]),
        "",
        "## Season-cluster roster-score intervals",
        "",
        markdown_table(score_intervals),
        "",
    ]
    (RESULTS / "phase_c_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    metadata = {
        "leagues": list(LEAGUE_RESULTS),
        "rosters_per_league": 1296,
        "scenarios_per_roster": 384,
        "roster_score_noninferiority_margin": ROSTER_SCORE_NONINFERIORITY,
        "missed_week_bias_per_player_margin": (
            MISSED_WEEK_BIAS_PER_PLAYER_MARGIN
        ),
        "promote_flatter_w025_all": promote,
        "production_changed": False,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
