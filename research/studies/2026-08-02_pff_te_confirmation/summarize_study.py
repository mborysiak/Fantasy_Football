"""Create compact projection, template, and roster confirmation findings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"
DATABASES = {
    "dk": STUDY_DIR.parents[2] / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": STUDY_DIR.parents[2] / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
PERIODS = {
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
PROJECTION_CANDIDATES = {
    "te_pff_mtf": "te_pff_mtf__te_route",
    "te_pff_yac": "te_pff_yac__te_route",
}
CONTROL = "te_pff_opportunity_control__te_route"
BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 1234


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - prediction))))


def _bootstrap(values: pd.Series, seed: int) -> tuple[float, float]:
    array = values.dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        array,
        size=(BOOTSTRAP_DRAWS, len(array)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def projection_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    season_rows = []
    component_rows = []
    for league, database in DATABASES.items():
        result_dir = STUDY_DIR / f"results_projection_{league}"
        ppg = pd.read_csv(result_dir / "ppg_summary.csv")
        q90 = pd.read_csv(result_dir / "q90_summary.csv")
        blends = pd.read_csv(result_dir / "blend_predictions.csv")
        components = pd.read_csv(result_dir / "challenger_component_predictions.csv")
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
            baseline_components = pd.read_sql_query(
                """
                SELECT player_key, season, position, method, prediction, actual
                FROM locked_whole_season_predictions
                WHERE target_name = 'conditional_ppg'
                  AND method IN (
                    'conditional_ppg_lasso',
                    'conditional_ppg_random_forest',
                    'conditional_ppg_lightgbm'
                  )
                """,
                connection,
            )
        for candidate, method in PROJECTION_CANDIDATES.items():
            for scope in ("all", "te"):
                for period in PERIODS:
                    candidate_row = ppg[
                        ppg["scope"].eq(scope)
                        & ppg["period"].eq(period)
                        & ppg["method"].eq(method)
                    ].iloc[0]
                    control_row = ppg[
                        ppg["scope"].eq(scope)
                        & ppg["period"].eq(period)
                        & ppg["method"].eq(CONTROL)
                    ].iloc[0]
                    q_candidate = q90[
                        q90["scope"].eq(scope)
                        & q90["period"].eq(period)
                        & q90["method"].eq(method)
                    ]
                    q_control = q90[
                        q90["scope"].eq(scope)
                        & q90["period"].eq(period)
                        & q90["method"].eq(CONTROL)
                    ]
                    record = {
                        "league": league,
                        "candidate": candidate,
                        "scope": scope,
                        "period": period,
                        "rmse_delta_vs_production": candidate_row["rmse_delta"],
                        "rmse_delta_vs_opportunity_control": (
                            candidate_row["rmse_delta"] - control_row["rmse_delta"]
                        ),
                        "mae_delta_vs_production": candidate_row["mae_delta"],
                    }
                    if not q_candidate.empty and not q_control.empty:
                        record.update(
                            {
                                "q90_brier_delta_vs_production": q_candidate.iloc[0]["brier_delta"],
                                "q90_brier_delta_vs_opportunity_control": (
                                    q_candidate.iloc[0]["brier_delta"]
                                    - q_control.iloc[0]["brier_delta"]
                                ),
                                "q90_ap_delta_vs_production": q_candidate.iloc[0]["average_precision_delta"],
                            }
                        )
                    rows.append(record)

            scoped = blends[blends["position"].eq("TE")]
            for season, season_frame in scoped.groupby("season", sort=True):
                challenger = season_frame[season_frame["method"].eq(method)]
                control = season_frame[season_frame["method"].eq(CONTROL)]
                season_rows.append(
                    {
                        "league": league,
                        "candidate": candidate,
                        "season": int(season),
                        "te_rmse_delta_vs_opportunity_control": (
                            _rmse(challenger["actual"], challenger["prediction"])
                            - _rmse(control["actual"], control["prediction"])
                        ),
                    }
                )

            for component in (
                "conditional_ppg_lasso",
                "conditional_ppg_random_forest",
                "conditional_ppg_lightgbm",
            ):
                challenger = components[
                    components["variant"].eq(candidate)
                    & components["component"].eq(component)
                    & components["position"].eq("TE")
                ].merge(
                    baseline_components[
                        baseline_components["method"].eq(component)
                    ][["player_key", "season", "actual", "prediction"]].rename(
                        columns={"prediction": "production_prediction"}
                    ),
                    on=["player_key", "season"],
                    how="inner",
                    validate="one_to_one",
                )
                for period, (start, end) in PERIODS.items():
                    selected = challenger[challenger["season"].between(start, end)]
                    component_rows.append(
                        {
                            "league": league,
                            "candidate": candidate,
                            "component": component,
                            "period": period,
                            "te_rmse_delta_vs_production": (
                                _rmse(selected["actual"], selected["prediction"])
                                - _rmse(selected["actual"], selected["production_prediction"])
                            ),
                        }
                    )
    seasons = pd.DataFrame(season_rows)
    intervals = []
    for index, (key, group) in enumerate(
        seasons.groupby(["league", "candidate"], sort=True)
    ):
        low, high = _bootstrap(
            group["te_rmse_delta_vs_opportunity_control"],
            RANDOM_SEED + index,
        )
        intervals.append(
            {
                "league": key[0],
                "candidate": key[1],
                "mean_season_te_rmse_delta_vs_control": float(
                    group["te_rmse_delta_vs_opportunity_control"].mean()
                ),
                "season_wins_vs_control": int(
                    group["te_rmse_delta_vs_opportunity_control"].lt(0).sum()
                ),
                "seasons": len(group),
                "bootstrap_low": low,
                "bootstrap_high": high,
            }
        )
    summary = pd.DataFrame(rows).merge(
        pd.DataFrame(intervals),
        on=["league", "candidate"],
        how="left",
    )
    return summary, seasons, pd.DataFrame(component_rows)


def roster_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    bootstrap_rows = []
    for league in ("dk", "beta"):
        result_dir = STUDY_DIR / f"results_roster_{league}"
        summary = pd.read_csv(result_dir / "summary.csv")
        summary.insert(0, "league", league)
        summaries.append(summary)
        predictions = pd.read_csv(result_dir / "roster_championship_predictions.csv")
        for period, (start, end) in PERIODS.items():
            period_frame = predictions[predictions["season"].between(start, end)]
            season_metrics = []
            for season, group in period_frame.groupby("season", sort=True):
                values = {"season": int(season)}
                for matcher, matcher_frame in group.groupby("matcher"):
                    winners = matcher_frame[matcher_frame["actual_champion"].eq(1)]
                    values[f"{matcher}__score_crps"] = float(matcher_frame["score_crps"].mean())
                    values[f"{matcher}__championship_brier"] = float(matcher_frame["championship_brier_row"].mean())
                    values[f"{matcher}__championship_log_loss"] = float(
                        -np.log(
                            np.clip(winners["championship_probability"].to_numpy(float), 1e-9, 1)
                        ).mean()
                    )
                season_metrics.append(values)
            season_metrics = pd.DataFrame(season_metrics)
            for metric in ("score_crps", "championship_brier", "championship_log_loss"):
                delta = (
                    season_metrics[f"te_pff_yac_w025__{metric}"]
                    - season_metrics[f"production__{metric}"]
                )
                low, high = _bootstrap(delta, RANDOM_SEED + len(bootstrap_rows))
                bootstrap_rows.append(
                    {
                        "league": league,
                        "period": period,
                        "matcher": "te_pff_yac_w025",
                        "metric": metric,
                        "mean_season_delta": float(delta.mean()),
                        "season_wins": int(delta.lt(0).sum()),
                        "seasons": len(delta),
                        "bootstrap_low": low,
                        "bootstrap_high": high,
                    }
                )
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(bootstrap_rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[column]) else "---" for column in columns) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    projection, projection_seasons, components = projection_summary()
    template_decision = pd.read_csv(
        STUDY_DIR / "results_template_summary" / "primary_decision.csv"
    )
    template_slices = pd.read_csv(
        STUDY_DIR / "results_template_summary" / "te_slices.csv"
    )
    roster, roster_bootstrap = roster_summary()

    projection.to_csv(RESULTS / "projection_confirmation.csv", index=False)
    projection_seasons.to_csv(RESULTS / "projection_season_diagnostics.csv", index=False)
    components.to_csv(RESULTS / "projection_component_diagnostics.csv", index=False)
    template_decision.to_csv(RESULTS / "template_decisions.csv", index=False)
    template_slices.to_csv(RESULTS / "template_te_slices.csv", index=False)
    roster.to_csv(RESULTS / "roster_summary.csv", index=False)
    roster_bootstrap.to_csv(RESULTS / "roster_season_bootstrap.csv", index=False)

    projection_focus = projection[
        projection["candidate"].eq("te_pff_mtf")
        & projection["scope"].eq("te")
    ][
        [
            "league", "period", "rmse_delta_vs_production",
            "rmse_delta_vs_opportunity_control",
            "q90_brier_delta_vs_production",
            "q90_brier_delta_vs_opportunity_control",
            "season_wins_vs_control", "bootstrap_low", "bootstrap_high",
        ]
    ]
    template_focus = template_decision[
        template_decision["method"].isin(
            ["te_pff_mtf_w025", "te_pff_yac_w025"]
        )
    ][
        [
            "league", "method", "te_development_ppg_delta",
            "te_recent_ppg_delta", "te_development_q90_brier_delta",
            "te_recent_q90_brier_delta", "advance_to_roster",
        ]
    ]
    roster_focus = roster[
        roster["matcher"].eq("te_pff_yac_w025")
        & roster["period"].isin(PERIODS)
    ][
        [
            "league", "period", "score_crps_delta",
            "championship_brier_delta", "championship_log_loss_delta",
        ]
    ]

    findings = "\n".join(
        [
            "# PFF TE confirmation findings",
            "",
            "## Decision",
            "",
            "The two tracks diverge. Prior-season PFF TE efficiency/tackle-breaking "
            "adds a small, repeatable signal to the point-projection model, but it does "
            "not justify changing weekly-template matching.",
            "",
            "Advance `te_pff_mtf` as the primary **projection-only** implementation "
            "candidate, routed to TE predictions so tree-fit spillover cannot change "
            "QB/RB/WR outputs. `te_pff_yac` is a separately passing projection "
            "sensitivity, not a bundle recommendation. Production remains unchanged "
            "because the broad screen and confirmation reuse the same historical "
            "origins; this is strong retrospective evidence, not a genuinely new-origin "
            "confirmation.",
            "",
            "Reject both PFF features for template matching. The primary 0.25 "
            "tackle-breaking arm worsens development TE PPG and q90 Brier in both "
            "leagues. The YAC/route sensitivity clears the mechanical player-level "
            "screen but fails roster transport: all three DK roster metrics worsen in "
            "development and 2023-2025, while beta is mixed.",
            "",
            "## Projection: TE-routed avoided tackles/reception",
            "",
            _markdown_table(projection_focus, list(projection_focus.columns)),
            "",
            "Negative deltas favor the challenger. Bootstrap intervals compare the TE "
            "rate challenger with the prior-PFF-opportunity control across all nine "
            "seasons.",
            "",
            "## Template mechanical screen",
            "",
            _markdown_table(template_focus, list(template_focus.columns)),
            "",
            "## Template roster transport for the YAC finalist",
            "",
            _markdown_table(roster_focus, list(roster_focus.columns)),
            "",
            "## Governance note",
            "",
            "No database, production feature manifest, projection lock, template "
            "weight, or app objective was changed by this study.",
            "",
        ]
    )
    (RESULTS / "findings.md").write_text(findings, encoding="utf-8")
    manifest = {
        "projection_candidate": "te_pff_mtf__te_route",
        "projection_sensitivity": "te_pff_yac__te_route",
        "template_decision": "reject",
        "roster_finalist_tested": "te_pff_yac_w025",
        "production_changed": False,
        "raw_intermediates_retained": False,
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(findings)


if __name__ == "__main__":
    main()
