"""Consolidate the frozen Ridge downstream gates and findings."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
LEAGUES = ("dk", "beta")
CHALLENGER = "ridge_swap"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: f"{value:.6f}" if pd.notna(value) else ""
        )
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def load_outputs() -> dict[str, dict[str, pd.DataFrame]]:
    outputs = {}
    for league in LEAGUES:
        outputs[league] = {
            "point": pd.read_csv(
                STUDY_DIR / f"results_projection_{league}" / "point_summary.csv"
            ),
            "point_bootstrap": pd.read_csv(
                STUDY_DIR
                / f"results_projection_{league}"
                / "point_player_cluster_bootstrap.csv"
            ),
            "distribution": pd.read_csv(
                STUDY_DIR
                / f"results_projection_{league}"
                / "distribution_summary.csv"
            ),
            "shadow": pd.read_csv(
                STUDY_DIR
                / f"results_projection_{league}"
                / "shadow_2026_summary.csv"
            ),
            "template": pd.read_csv(
                STUDY_DIR
                / f"results_template_{league}"
                / "role_tier_deltas.csv"
            ),
            "roster": pd.read_csv(
                STUDY_DIR / f"results_roster_{league}" / "summary.csv"
            ),
            "roster_bootstrap": pd.read_csv(
                STUDY_DIR
                / f"results_roster_{league}"
                / "season_bootstrap.csv"
            ),
        }
    return outputs


def point_table(outputs: dict) -> pd.DataFrame:
    rows = []
    for league, data in outputs.items():
        candidate = data["point"][
            data["point"].method.eq(CHALLENGER)
            & data["point"].slice_type.eq("all")
        ]
        bootstrap = data["point_bootstrap"]
        for period in (
            "all_2017_2025",
            "development_2017_2022",
            "temporal_2023_2025",
        ):
            row = candidate[candidate.period.eq(period)].iloc[0]
            interval = bootstrap[bootstrap.period.eq(period)].iloc[0]
            rows.append(
                {
                    "league": league,
                    "period": period,
                    "production_rmse": row.rmse_baseline,
                    "ridge_swap_rmse": row.rmse,
                    "rmse_delta": row.rmse_delta,
                    "bootstrap_p025": interval.bootstrap_p025,
                    "bootstrap_p975": interval.bootstrap_p975,
                    "mae_delta": row.mae_delta,
                    "bias_delta": row.bias_delta,
                }
            )
    return pd.DataFrame(rows)


def distribution_table(outputs: dict) -> pd.DataFrame:
    rows = []
    for league, data in outputs.items():
        candidate = data["distribution"][
            data["distribution"].method.eq(CHALLENGER)
            & data["distribution"].slice_type.eq("all")
        ]
        for _, row in candidate.iterrows():
            rows.append(
                {
                    "league": league,
                    "period": row.period,
                    "crps_delta": row.crps_delta,
                    "production_coverage_50": row.coverage_50_baseline,
                    "ridge_coverage_50": row.coverage_50,
                    "production_coverage_80": row.coverage_80_baseline,
                    "ridge_coverage_80": row.coverage_80,
                }
            )
    return pd.DataFrame(rows)


def template_table(outputs: dict) -> pd.DataFrame:
    rows = []
    for league, data in outputs.items():
        frame = data["template"]
        selected = frame[
            frame.method.eq(CHALLENGER)
            & frame.tier.isin(("core_main", "depth_main"))
            & frame.period.isin(
                ("development_2017_2022", "temporal_2023_2025")
            )
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "league": league,
                    "tier": row.tier,
                    "period": row.period,
                    "ppg_crps_relative_delta": row.ppg_crps_relative_delta,
                    "contribution_crps_relative_delta": row.contribution_crps_relative_delta,
                    "played_crps_relative_delta": row.played_crps_relative_delta,
                    "coverage_delta": row.ppg_80_coverage_delta,
                }
            )
    return pd.DataFrame(rows)


def roster_table(outputs: dict) -> pd.DataFrame:
    rows = []
    for league, data in outputs.items():
        candidate = data["roster"][data["roster"].method.eq(CHALLENGER)]
        for _, row in candidate.iterrows():
            rows.append(
                {
                    "league": league,
                    "period": row.period,
                    "rosters": int(row.rosters),
                    "score_crps_relative_delta": row.score_crps_relative_delta,
                    "score_bias_delta": row.score_bias_delta,
                    "coverage_delta": row.score_coverage_80_delta,
                    "championship_brier_relative_delta": row.championship_brier_relative_delta,
                    "championship_log_loss_delta": row.championship_log_loss_delta,
                }
            )
    return pd.DataFrame(rows)


def gate_table(
    outputs: dict,
    point: pd.DataFrame,
    distribution: pd.DataFrame,
    template: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    pooled_point = point[point.period.eq("all_2017_2025")]
    recent_wins = 0
    position_guardrail = True
    for data in outputs.values():
        seasons = data["point"]
        season_values = pd.to_numeric(
            seasons.slice_value, errors="coerce"
        )
        seasons = seasons[
            seasons.method.eq(CHALLENGER)
            & seasons.slice_type.eq("season")
            & seasons.period.eq("all_2017_2025")
            & season_values.between(2023, 2025)
        ]
        recent_wins += int(seasons.rmse_delta.lt(0).sum())
        positions = data["point"]
        positions = positions[
            positions.method.eq(CHALLENGER)
            & positions.slice_type.eq("position")
            & positions.period.eq("temporal_2023_2025")
        ]
        position_guardrail &= bool(positions.rmse_delta.le(0.01).all())
    point_pooled_pass = bool(pooled_point.rmse_delta.lt(0).all())
    point_pass = point_pooled_pass and recent_wins >= 5 and position_guardrail

    pooled_distribution = distribution[
        distribution.period.eq("all_2018_2025")
    ]
    crps_pass = bool(pooled_distribution.crps_delta.le(0).all())
    coverage_pass = True
    for _, row in distribution.iterrows():
        for level, target in ((50, 0.50), (80, 0.80)):
            candidate = row[f"ridge_coverage_{level}"]
            baseline = row[f"production_coverage_{level}"]
            coverage_pass &= bool(
                abs(candidate - target) <= 0.02
                or abs(candidate - target) <= abs(baseline - target)
            )
    distribution_pass = crps_pass and coverage_pass

    template_metrics = [
        "ppg_crps_relative_delta",
        "contribution_crps_relative_delta",
        "played_crps_relative_delta",
    ]
    template_pass = bool(template[template_metrics].le(0.0025).all().all())

    roster_cells = roster[
        roster.period.isin(
            ("development_2018_2022", "temporal_2023_2025")
        )
    ]
    roster_margin_pass = bool(
        roster_cells.score_crps_relative_delta.le(0.005).all()
    )
    roster_better_cells = int(
        roster_cells.score_crps_relative_delta.le(0).sum()
    )
    roster_pass = roster_margin_pass and roster_better_cells >= 3

    return pd.DataFrame(
        [
            {
                "gate": "point_ppg",
                "passed": point_pass,
                "detail": (
                    f"pooled_both={point_pooled_pass}; recent_wins={recent_wins}/6; "
                    f"position_guardrail={position_guardrail}"
                ),
            },
            {
                "gate": "player_distribution",
                "passed": distribution_pass,
                "detail": f"crps_both={crps_pass}; coverage={coverage_pass}",
            },
            {
                "gate": "weekly_template_transport",
                "passed": template_pass,
                "detail": (
                    "all core/depth PPG, contribution, and played CRPS relative "
                    "deltas <= +0.25%"
                ),
            },
            {
                "gate": "fixed_roster_snake",
                "passed": roster_pass,
                "detail": (
                    f"margin={roster_margin_pass}; nonworse_cells="
                    f"{roster_better_cells}/4"
                ),
            },
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = load_outputs()
    point = point_table(outputs)
    distribution = distribution_table(outputs)
    template = template_table(outputs)
    roster = roster_table(outputs)
    gates = gate_table(outputs, point, distribution, template, roster)
    promote = bool(gates.passed.all())

    point.to_csv(RESULTS_DIR / "point_summary.csv", index=False)
    distribution.to_csv(
        RESULTS_DIR / "distribution_summary.csv", index=False
    )
    template.to_csv(RESULTS_DIR / "template_summary.csv", index=False)
    roster.to_csv(RESULTS_DIR / "roster_summary.csv", index=False)
    gates.to_csv(RESULTS_DIR / "gate_summary.csv", index=False)

    key_point = point[point.period.eq("all_2017_2025")][
        ["league", "production_rmse", "ridge_swap_rmse", "rmse_delta", "bootstrap_p025", "bootstrap_p975"]
    ]
    key_roster = roster[
        roster.period.isin(
            ("development_2018_2022", "temporal_2023_2025")
        )
    ][
        [
            "league",
            "period",
            "rosters",
            "score_crps_relative_delta",
            "championship_brier_relative_delta",
        ]
    ]
    lines = [
        "# Ridge Swap Downstream Findings",
        "",
        "## Decision",
        "",
        (
            "Do not replace Lasso with Ridge in the 2026 production point-center "
            "ensemble. Keep the active equal-third Lasso/RandomForest/LightGBM "
            "blend. The frozen Ridge swap fails the point-season replication and "
            "fixed-roster gates even though its pooled point and player-distribution "
            "metrics are slightly better."
        ),
        "",
        "## Gates",
        "",
        markdown_table(gates),
        "",
        "## Point forecast",
        "",
        markdown_table(key_point),
        "",
        (
            "The swap lowers pooled RMSE by about 0.0013-0.0014 in both leagues, "
            "but every player-cluster interval crosses zero. It wins 2023 and 2024 "
            "and loses 2025 in both leagues, for only four of six recent season cells. "
            "Recent RB RMSE also worsens slightly in both leagues."
        ),
        "",
        "## Distribution and weekly templates",
        "",
        (
            "Strict-prior player CRPS improves slightly in both leagues and 50%/80% "
            "coverage remains calibrated. All eight core/depth league-period template "
            "cells stay inside the +0.25% PPG/contribution/played CRPS margins."
        ),
        "",
        "## Fixed-roster Snake replay",
        "",
        markdown_table(key_roster),
        "",
        (
            "Only DK development improves roster-score CRPS. DK temporal and both "
            "beta periods worsen; beta temporal is +0.527%, just outside the +0.5% "
            "non-inferiority margin. Championship diagnostics do not rescue the point "
            "center because expected-score calibration is the primary gate."
        ),
        "",
        "Beta 2018 has no QB rows in the active locked whole-season forecast table, "
        "so a legal beta roster room cannot be formed for that origin. Beta player and "
        "template metrics retain 2018; beta roster metrics cover 2019-2025. DK roster "
        "metrics cover 2018-2025.",
        "",
        "## 2026 shadow",
        "",
        (
            "The Ridge swap changes the 2026 center very little: mean PPG is lower by "
            "0.026 in both leagues and rank correlation with production is about 0.997. "
            "Because the historical gates fail, the preregistered Auction shadow was "
            "not run. No production or app database was changed."
        ),
        "",
    ]
    (RESULTS_DIR / "findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    decision = {
        "promote_ridge_swap": promote,
        "recommended_2026_point_center": (
            "production_lasso_random_forest_lightgbm_equal_thirds"
        ),
        "failed_gates": gates.loc[~gates.passed, "gate"].tolist(),
        "production_changed": False,
    }
    (RESULTS_DIR / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(gates.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
