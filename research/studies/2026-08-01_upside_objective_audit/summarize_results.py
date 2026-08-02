"""Aggregate uncertainty and durable findings for the upside-objective audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LEAGUES = ("dk", "beta")
PERIODS = {
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
BASELINE = "production"
PLAYER_METHODS = ("flatter_w025_all", "wr_ppg225_both025")
BOOTSTRAP_REPEATS = 5_000
ROOT_SEED = 20260801
CORE_COUNTS = {"QB": 18, "RB": 36, "WR": 48, "TE": 18}


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).astype(float))


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def player_bootstrap() -> pd.DataFrame:
    output = []
    rng = np.random.default_rng(ROOT_SEED)
    for league in LEAGUES:
        predictions = pd.read_csv(
            STUDY_DIR / f"results_player_{league}" / "target_predictions.csv"
        )
        predictions = predictions[predictions.is_core]
        for period, (start, end) in PERIODS.items():
            period_frame = predictions[predictions.season.between(start, end)]
            seasons = np.sort(period_frame.season.unique())
            for severity in (90, 95):
                outcome_col = f"observed_league_winner_q{severity}"
                probability_col = f"prob_league_winner_q{severity}"
                utility_col = f"tail_utility_q{severity}_crps"
                for method in PLAYER_METHODS:
                    paired = period_frame[
                        period_frame.method.isin([BASELINE, method])
                    ].pivot(
                        index=["season", "player", "pos"],
                        columns="method",
                        values=[
                            outcome_col,
                            probability_col,
                            "ppg_crps",
                            "contribution_crps",
                            utility_col,
                        ],
                    )
                    if paired.isna().any().any():
                        raise ValueError(f"Incomplete player pairing for {league} {method}.")

                    def metrics(frame: pd.DataFrame) -> np.ndarray:
                        outcome = frame[(outcome_col, BASELINE)].to_numpy(int)
                        candidate_probability = frame[
                            (probability_col, method)
                        ].to_numpy(float)
                        baseline_probability = frame[
                            (probability_col, BASELINE)
                        ].to_numpy(float)
                        candidate_ap = average_precision_score(
                            outcome,
                            candidate_probability,
                        )
                        baseline_ap = average_precision_score(
                            outcome,
                            baseline_probability,
                        )
                        return np.asarray(
                            [
                                np.mean(
                                    np.square(candidate_probability - outcome)
                                    - np.square(baseline_probability - outcome)
                                ),
                                candidate_ap - baseline_ap,
                                np.mean(
                                    frame[(utility_col, method)]
                                    - frame[(utility_col, BASELINE)]
                                ),
                                np.mean(
                                    frame[("ppg_crps", method)]
                                    - frame[("ppg_crps", BASELINE)]
                                ),
                                np.mean(
                                    frame[("contribution_crps", method)]
                                    - frame[("contribution_crps", BASELINE)]
                                ),
                            ],
                            dtype=float,
                        )

                    point = metrics(paired)
                    draws = np.empty((BOOTSTRAP_REPEATS, len(point)), dtype=float)
                    season_parts = {
                        int(season): paired[paired.index.get_level_values("season") == season]
                        for season in seasons
                    }
                    for draw in range(BOOTSTRAP_REPEATS):
                        sampled = rng.choice(seasons, size=len(seasons), replace=True)
                        resampled = pd.concat(
                            [season_parts[int(season)] for season in sampled],
                            ignore_index=True,
                        )
                        draws[draw] = metrics(resampled)
                    metric_names = (
                        "league_winner_brier",
                        "average_precision",
                        "tail_utility_crps",
                        "ppg_crps",
                        "contribution_crps",
                    )
                    for metric_idx, metric in enumerate(metric_names):
                        low, high = percentile_interval(draws[:, metric_idx])
                        output.append(
                            {
                                "league": league,
                                "period": period,
                                "scope": "core",
                                "severity": f"q{severity}",
                                "method": method,
                                "metric": metric,
                                "delta": float(point[metric_idx]),
                                "season_bootstrap_low": low,
                                "season_bootstrap_high": high,
                                "seasons": int(len(seasons)),
                            }
                        )
    return pd.DataFrame(output)


def roster_bootstrap() -> pd.DataFrame:
    output = []
    rng = np.random.default_rng(ROOT_SEED + 1)
    for league in LEAGUES:
        predictions = pd.read_csv(
            STUDY_DIR
            / f"results_roster_{league}"
            / "roster_championship_predictions.csv"
        )
        for period, (start, end) in PERIODS.items():
            period_frame = predictions[predictions.season.between(start, end)]
            seasons = np.sort(period_frame.season.unique())
            for method in PLAYER_METHODS:
                paired = period_frame[
                    period_frame.matcher.isin([BASELINE, method])
                ]
                season_parts = {
                    int(season): paired[paired.season.eq(season)]
                    for season in seasons
                }

                def metrics(frame: pd.DataFrame) -> np.ndarray:
                    candidate = frame[frame.matcher.eq(method)]
                    baseline = frame[frame.matcher.eq(BASELINE)]
                    keys = ["season", "room", "team"]
                    aligned = candidate.merge(
                        baseline,
                        on=keys,
                        suffixes=("_candidate", "_baseline"),
                        validate="one_to_one",
                    )
                    winners = aligned.actual_champion_candidate.eq(1)
                    return np.asarray(
                        [
                            np.mean(
                                aligned.score_crps_candidate
                                - aligned.score_crps_baseline
                            ),
                            np.mean(
                                aligned.championship_brier_row_candidate
                                - aligned.championship_brier_row_baseline
                            ),
                            np.mean(
                                -np.log(
                                    np.clip(
                                        aligned.loc[
                                            winners,
                                            "championship_probability_candidate",
                                        ],
                                        1e-9,
                                        1.0,
                                    )
                                )
                                + np.log(
                                    np.clip(
                                        aligned.loc[
                                            winners,
                                            "championship_probability_baseline",
                                        ],
                                        1e-9,
                                        1.0,
                                    )
                                )
                            ),
                        ],
                        dtype=float,
                    )

                point = metrics(paired)
                draws = np.empty((BOOTSTRAP_REPEATS, len(point)), dtype=float)
                for draw in range(BOOTSTRAP_REPEATS):
                    sampled = rng.choice(seasons, size=len(seasons), replace=True)
                    resampled_parts = []
                    for replicate, season in enumerate(sampled):
                        part = season_parts[int(season)].copy()
                        part["season"] = replicate
                        resampled_parts.append(part)
                    draws[draw] = metrics(pd.concat(resampled_parts, ignore_index=True))
                for metric_idx, metric in enumerate(
                    ("score_crps", "championship_brier", "championship_log_loss")
                ):
                    low, high = percentile_interval(draws[:, metric_idx])
                    output.append(
                        {
                            "league": league,
                            "period": period,
                            "method": method,
                            "metric": metric,
                            "delta": float(point[metric_idx]),
                            "season_bootstrap_low": low,
                            "season_bootstrap_high": high,
                            "seasons": int(len(seasons)),
                        }
                    )
    return pd.DataFrame(output)


def saved_fastr_diagnostics() -> pd.DataFrame:
    studies = {
        "receiver_profiles": (
            "2026-07-31_template_fastr_receiver_profiles",
            ("WR", "TE"),
        ),
        "rb_roles": ("2026-07-31_template_fastr_rb_roles", ("RB",)),
    }
    rows = []
    for label, (study, positions) in studies.items():
        for league in LEAGUES:
            predictions = pd.read_csv(
                STUDY_DIR.parent / study / f"results_{league}" / "target_predictions.csv"
            )
            predictions = predictions[
                predictions.pos.isin(positions)
                & predictions.preseason_pos_rank.le(
                    predictions.pos.map(CORE_COUNTS)
                )
            ]
            for period, (start, end) in PERIODS.items():
                period_frame = predictions[predictions.season.between(start, end)]
                baseline = period_frame[period_frame.method.eq(BASELINE)]
                for method, group in period_frame.groupby("method", sort=True):
                    for event, outcome_col, probability_col in (
                        ("plus5", "observed_plus5", "prob_plus5"),
                        ("legacy_impact", "observed_impact", "prob_impact"),
                    ):
                        candidate_brier = np.mean(
                            np.square(group[probability_col] - group[outcome_col])
                        )
                        baseline_brier = np.mean(
                            np.square(
                                baseline[probability_col] - baseline[outcome_col]
                            )
                        )
                        candidate_ap = average_precision_score(
                            group[outcome_col],
                            group[probability_col],
                        )
                        baseline_ap = average_precision_score(
                            baseline[outcome_col],
                            baseline[probability_col],
                        )
                        rows.append(
                            {
                                "study": label,
                                "league": league,
                                "period": period,
                                "positions": "/".join(positions),
                                "method": method,
                                "event": event,
                                "n": int(len(group)),
                                "brier_delta": float(
                                    candidate_brier - baseline_brier
                                ),
                                "average_precision_delta": float(
                                    candidate_ap - baseline_ap
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def write_findings(
    player_uncertainty: pd.DataFrame,
    roster_uncertainty: pd.DataFrame,
) -> None:
    candidate_player = player_uncertainty[
        player_uncertainty.method.eq("wr_ppg225_both025")
        & player_uncertainty.severity.eq("q90")
    ]
    candidate_roster = roster_uncertainty[
        roster_uncertainty.method.eq("wr_ppg225_both025")
    ]
    lines = [
        "# Findings",
        "",
        "## Decision",
        "",
        "Keep the production matcher and both app objectives unchanged. Adopt the "
        "new rare-upside and championship metrics as secondary validation objectives "
        "for future tests.",
        "",
        "## Player level",
        "",
        "`wr_ppg225_both025` is the first challenger exposed by the new objective. "
        "On the primary q90 core event it improves Brier score, log loss, continuous "
        "tail-utility CRPS, and contribution CRPS in all four league-by-period cells. "
        "The absolute gains are small and their season-bootstrap intervals generally "
        "cross zero. DK rare-event probabilities are also materially underpredicted, "
        "so raw absolute tail probabilities should not drive app decisions yet.",
        "",
        "## Roster level",
        "",
        "The player-level tail signal does not transport to 12-team championship "
        "probability. `wr_ppg225_both025` worsens championship Brier/log loss in DK "
        "development and recent periods and beta development, improving only recent "
        "beta. The prior flatter-distance arm also fails joint replication. All "
        "season-bootstrap intervals are wide. This rejects both matcher promotions.",
        "",
        "## Recommended downstream objective",
        "",
        "Use a constrained, lexicographic tilt rather than a weighted or distorted "
        "forecast: retain ordinary calibrated scenario draws; require expected-score "
        "non-inferiority; among candidates within 0.25% of the best expected roster "
        "score, prefer the highest paired championship-probability lower bound. "
        "Auction should compare Buy versus Pass on the same scenario rooms; Snake "
        "should compare forced current-pick candidates on the same future-draft and "
        "weekly scenario banks.",
        "",
        "## Bootstrap evidence",
        "",
        "Player q90 candidate rows:",
        "",
        markdown_table(candidate_player),
        "",
        "Roster candidate rows:",
        "",
        markdown_table(candidate_roster),
        "",
    ]
    (RESULTS_DIR / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    player_uncertainty = player_bootstrap()
    roster_uncertainty = roster_bootstrap()
    fastr = saved_fastr_diagnostics()
    player_uncertainty.to_csv(RESULTS_DIR / "player_bootstrap.csv", index=False)
    roster_uncertainty.to_csv(RESULTS_DIR / "roster_bootstrap.csv", index=False)
    fastr.to_csv(RESULTS_DIR / "saved_fastr_tail_diagnostics.csv", index=False)
    write_findings(player_uncertainty, roster_uncertainty)
    print((RESULTS_DIR / "findings.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
