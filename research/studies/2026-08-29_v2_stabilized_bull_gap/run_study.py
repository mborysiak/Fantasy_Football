"""Evaluate denominator-stabilized bullish provider projection gaps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parent
PARENT_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-29_v2_asymmetric_expert_projection"
    / "run_study.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SMOOTH_STABILIZERS = (3.0, 5.0, 8.0)
PRIMARY_POINT_VARIANT = "smooth_k5"
PRIMARY_TAIL_VARIANT = "tail_smooth_k5"
PRIMARY_TEMPLATE_METHOD = "smooth_k5_add_w050"
RANDOM_SEED = 1234

POINT_VARIANT_FEATURES = {
    "incumbent": (),
    "raw_gap": ("expert_ppg_bull_gap", "expert_ppg_bull_gap_available"),
    "smooth_k5": (
        "expert_ppg_bull_gap_smooth_k5",
        "expert_ppg_bull_gap_available",
    ),
    "smooth_k3": (
        "expert_ppg_bull_gap_smooth_k3",
        "expert_ppg_bull_gap_available",
    ),
    "smooth_k8": (
        "expert_ppg_bull_gap_smooth_k8",
        "expert_ppg_bull_gap_available",
    ),
    "hard_floor_k5": (
        "expert_ppg_bull_gap_hard_floor_k5",
        "expert_ppg_bull_gap_available",
    ),
    "additive_k5": (
        "expert_ppg_bull_gap_additive_k5",
        "expert_ppg_bull_gap_available",
    ),
}

TAIL_VARIANT_FEATURES = {
    "tail_symmetric": (),
    "tail_raw": (
        "expert_ppg_bull_gap",
        "expert_ppg_bull_gap_raw_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
    "tail_smooth_k5": (
        "expert_ppg_bull_gap_smooth_k5",
        "expert_ppg_bull_gap_smooth_k5_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
    "tail_smooth_k3": (
        "expert_ppg_bull_gap_smooth_k3",
        "expert_ppg_bull_gap_smooth_k3_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
    "tail_smooth_k8": (
        "expert_ppg_bull_gap_smooth_k8",
        "expert_ppg_bull_gap_smooth_k8_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
    "tail_hard_floor_k5": (
        "expert_ppg_bull_gap_hard_floor_k5",
        "expert_ppg_bull_gap_hard_floor_k5_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
    "tail_additive_k5": (
        "expert_ppg_bull_gap_additive_k5",
        "expert_ppg_bull_gap_additive_k5_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
}

TEMPLATE_FEATURES = {
    "raw_add_w050": "expert_ppg_bull_gap_raw_position_percentile",
    "smooth_k5_add_w050": (
        "expert_ppg_bull_gap_smooth_k5_position_percentile"
    ),
    "smooth_k3_add_w050": (
        "expert_ppg_bull_gap_smooth_k3_position_percentile"
    ),
    "smooth_k8_add_w050": (
        "expert_ppg_bull_gap_smooth_k8_position_percentile"
    ),
    "hard_floor_k5_add_w050": (
        "expert_ppg_bull_gap_hard_floor_k5_position_percentile"
    ),
    "additive_k5_add_w050": (
        "expert_ppg_bull_gap_additive_k5_position_percentile"
    ),
}


def _load_parent():
    spec = importlib.util.spec_from_file_location(
        "v2_asymmetric_parent_for_stabilized_gap", PARENT_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load parent runner: {PARENT_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent = _load_parent()
DATABASES = parent.DATABASES
parent._ORIGINAL_BUILD_ASYMMETRIC_FEATURES = (
    parent.build_asymmetric_projection_features
)
parent._ORIGINAL_RUN_POINT_PREDICTIONS = parent._run_point_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta", "all"), default="all")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--template-bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--skip-templates", action="store_true")
    parser.add_argument("--combine-existing", action="store_true")
    return parser.parse_args()


def build_stabilized_projection_features(
    projection_values: pd.DataFrame,
) -> pd.DataFrame:
    """Add frozen smooth, hard-floor, additive, and raw bull-gap variants."""

    output = parent._ORIGINAL_BUILD_ASYMMETRIC_FEATURES(projection_values)
    available = output["expert_ppg_bull_gap_available"].eq(1)
    median_abs = output["expert_ppg_gap_median"].abs()
    raw_gap = output["expert_ppg_bull_gap"]
    for stabilizer in SMOOTH_STABILIZERS:
        label = int(stabilizer)
        column = f"expert_ppg_bull_gap_smooth_k{label}"
        denominator = np.sqrt(np.square(median_abs) + stabilizer**2)
        output[column] = (raw_gap / denominator).where(available)
    output["expert_ppg_bull_gap_hard_floor_k5"] = (
        raw_gap / median_abs.clip(lower=5.0)
    ).where(available)
    output["expert_ppg_bull_gap_additive_k5"] = (
        raw_gap / (median_abs + 5.0)
    ).where(available)
    value_columns = [
        "expert_ppg_bull_gap",
        *(f"expert_ppg_bull_gap_smooth_k{int(k)}" for k in SMOOTH_STABILIZERS),
        "expert_ppg_bull_gap_hard_floor_k5",
        "expert_ppg_bull_gap_additive_k5",
    ]
    for value_column in value_columns:
        if value_column == "expert_ppg_bull_gap":
            percentile = "expert_ppg_bull_gap_raw_position_percentile"
        else:
            percentile = f"{value_column}_position_percentile"
        output[percentile] = output.groupby(
            ["season", "position"], sort=True
        )[value_column].rank(method="average", pct=True)
    return output


def build_template_methods(builder) -> dict[str, dict[str, object]]:
    incumbent = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    half_life = float(builder.TEMPLATE_RECENCY_HALF_LIFE)
    methods: dict[str, dict[str, object]] = {
        "incumbent": {
            "weights": incumbent,
            "recency_half_life": half_life,
            "variant": "incumbent",
            "removed_families": (),
        }
    }
    for method, feature in TEMPLATE_FEATURES.items():
        weights = deepcopy(incumbent)
        for position in builder.POSITIONS:
            weights[position][feature] = 0.50
            weights[position]["expert_ppg_bull_gap_available"] = 0.25
        methods[method] = {
            "weights": weights,
            "recency_half_life": half_life,
            "variant": method.removesuffix("_add_w050"),
            "removed_families": (),
        }
    return methods


def _extended_point_predictions(locked, features, selected) -> pd.DataFrame:
    output = parent._ORIGINAL_RUN_POINT_PREDICTIONS(locked, features, selected)
    extra_columns = sorted(
        {
            feature
            for variants in (POINT_VARIANT_FEATURES, TAIL_VARIANT_FEATURES)
            for feature_set in variants.values()
            for feature in feature_set
            if feature not in output.columns
        }
    )
    if extra_columns:
        output = output.merge(
            features[["player_key", "season", "position", *extra_columns]],
            on=["player_key", "season", "position"],
            how="left",
            validate="one_to_one",
        )
    return output


def _tail_comparison_summary(
    evaluation: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    challengers = [
        variant for variant in TAIL_VARIANT_FEATURES if variant != "tail_symmetric"
    ]
    for event in parent.TAIL_EVENTS:
        selected = evaluation[evaluation["event"].eq(event)]
        baseline = selected[selected["variant"].eq("tail_symmetric")][
            [
                "player_key",
                "season",
                "position",
                "outcome",
                "probability",
                "brier",
                "log_loss",
            ]
        ].rename(
            columns={
                "probability": "baseline_probability",
                "brier": "baseline_brier",
                "log_loss": "baseline_log_loss",
            }
        )
        for challenger in challengers:
            challenge = selected[selected["variant"].eq(challenger)][
                ["player_key", "season", "probability", "brier", "log_loss"]
            ].rename(
                columns={
                    "probability": "challenger_probability",
                    "brier": "challenger_brier",
                    "log_loss": "challenger_log_loss",
                }
            )
            compared = baseline.merge(
                challenge,
                on=["player_key", "season"],
                how="inner",
                validate="one_to_one",
            )
            if len(compared) != len(baseline) or len(compared) != len(challenge):
                raise ValueError(f"Unmatched tail rows for {event}/{challenger}")
            compared["delta"] = (
                compared["challenger_brier"] - compared["baseline_brier"]
            )
            by_season = compared.groupby("season", sort=True)["delta"].mean()
            recent = compared[compared["season"].ge(2023)]
            season_low, season_high = parent._mean_cluster_interval(
                compared,
                "season",
                iterations,
                np.random.default_rng(RANDOM_SEED + 2),
            )
            player_low, player_high = parent._mean_cluster_interval(
                compared,
                "player_key",
                iterations,
                np.random.default_rng(RANDOM_SEED + 3),
            )
            rows.append(
                {
                    "event": event,
                    "baseline_variant": "tail_symmetric",
                    "challenger_variant": challenger,
                    "n_rows": len(compared),
                    "event_rate": float(compared["outcome"].mean()),
                    "baseline_brier": float(compared["baseline_brier"].mean()),
                    "challenger_brier": float(compared["challenger_brier"].mean()),
                    "brier_delta": float(compared["delta"].mean()),
                    "recent_brier_delta": float(recent["delta"].mean()),
                    "baseline_log_loss": float(
                        compared["baseline_log_loss"].mean()
                    ),
                    "challenger_log_loss": float(
                        compared["challenger_log_loss"].mean()
                    ),
                    "log_loss_delta": float(
                        (
                            compared["challenger_log_loss"]
                            - compared["baseline_log_loss"]
                        ).mean()
                    ),
                    "baseline_auc": parent._safe_auc(
                        compared["outcome"], compared["baseline_probability"]
                    ),
                    "challenger_auc": parent._safe_auc(
                        compared["outcome"], compared["challenger_probability"]
                    ),
                    "baseline_calibration_bias": float(
                        compared["baseline_probability"].mean()
                        - compared["outcome"].mean()
                    ),
                    "challenger_calibration_bias": float(
                        compared["challenger_probability"].mean()
                        - compared["outcome"].mean()
                    ),
                    "season_wins": int(by_season.lt(0).sum()),
                    "season_count": int(len(by_season)),
                    "season_95_low": season_low,
                    "season_95_high": season_high,
                    "player_95_low": player_low,
                    "player_95_high": player_high,
                }
            )
    return pd.DataFrame(rows)


def _tail_gates(summary: pd.DataFrame) -> dict[str, bool]:
    primary = summary[
        summary["challenger_variant"].eq(PRIMARY_TAIL_VARIANT)
    ].set_index("event")
    return {
        "plus3_brier_improves": bool(primary.loc["plus3", "brier_delta"] < 0),
        "plus5_brier_improves": bool(primary.loc["plus5", "brier_delta"] < 0),
        "plus3_recent_brier_nonworse": bool(
            primary.loc["plus3", "recent_brier_delta"] <= 0
        ),
        "plus5_recent_brier_nonworse": bool(
            primary.loc["plus5", "recent_brier_delta"] <= 0
        ),
        "plus3_auc_nonworse": bool(
            primary.loc["plus3", "challenger_auc"]
            >= primary.loc["plus3", "baseline_auc"]
        ),
        "plus5_auc_nonworse": bool(
            primary.loc["plus5", "challenger_auc"]
            >= primary.loc["plus5", "baseline_auc"]
        ),
        "plus3_season_interval_upper_nonpositive": bool(
            primary.loc["plus3", "season_95_high"] <= 0
        ),
        "plus5_season_interval_upper_nonpositive": bool(
            primary.loc["plus5", "season_95_high"] <= 0
        ),
    }


def _gap_outcome_diagnostics(
    point_evaluation: pd.DataFrame,
    gap_features: pd.DataFrame,
) -> pd.DataFrame:
    point = point_evaluation[
        point_evaluation["variant"].eq("incumbent")
        & point_evaluation["method"].eq(parent.CONTROLLED_METHOD)
    ][["player_key", "season", "position", "actual", "prediction"]].copy()
    feature = "expert_ppg_bull_gap_smooth_k5"
    percentile = f"{feature}_position_percentile"
    point = point.merge(
        gap_features[
            [
                "player_key",
                "season",
                "position",
                "expert_ppg_gap_provider_count",
                "expert_ppg_gap_median",
                "expert_ppg_bull_gap",
                feature,
                percentile,
                "expert_ppg_bull_gap_available",
            ]
        ],
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    point["expert_ppg_bull_gap_available"] = point[
        "expert_ppg_bull_gap_available"
    ].fillna(0).astype(int)
    point["residual"] = point["actual"] - point["prediction"]
    point["absolute_error"] = point["residual"].abs()
    point["observed_plus3"] = point["residual"].ge(3).astype(int)
    point["observed_plus5"] = point["residual"].ge(5).astype(int)
    available = point[point["expert_ppg_bull_gap_available"].eq(1)].copy()
    available["bull_gap_quartile"] = pd.cut(
        available[percentile],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=("Q1", "Q2", "Q3", "Q4"),
        include_lowest=True,
    ).astype("string")
    available["projection_band"] = pd.cut(
        available["expert_ppg_gap_median"].abs(),
        bins=[-np.inf, 3.0, 6.0, 10.0, np.inf],
        labels=("lt3", "3to6", "6to10", "10plus"),
    ).astype("string")
    rows: list[dict[str, object]] = []
    for period, period_frame in (
        ("full_2017_2025", available),
        ("recent_2023_2025", available[available["season"].ge(2023)]),
    ):
        groups = [("all", "all", period_frame)]
        groups.extend(
            ("position", str(value), group)
            for value, group in period_frame.groupby("position", sort=True)
        )
        groups.extend(
            ("quartile", str(value), group)
            for value, group in period_frame.groupby(
                "bull_gap_quartile", observed=True, sort=True
            )
        )
        groups.extend(
            ("projection_band", str(value), group)
            for value, group in period_frame.groupby(
                "projection_band", observed=True, sort=True
            )
        )
        for scope, value, group in groups:
            rows.append(
                {
                    "period": period,
                    "scope": scope,
                    "scope_value": value,
                    "n_rows": len(group),
                    "mean_consensus_ppg": float(
                        group["expert_ppg_gap_median"].mean()
                    ),
                    "mean_bull_gap_ppg": float(
                        group["expert_ppg_bull_gap"].mean()
                    ),
                    "mean_smooth_k5": float(group[feature].mean()),
                    "mean_residual": float(group["residual"].mean()),
                    "mean_absolute_error": float(group["absolute_error"].mean()),
                    "plus3_rate": float(group["observed_plus3"].mean()),
                    "plus5_rate": float(group["observed_plus5"].mean()),
                    "spearman_stabilized_residual": float(
                        group[feature].corr(group["residual"], method="spearman")
                    ),
                    "spearman_stabilized_absolute_error": float(
                        group[feature].corr(
                            group["absolute_error"], method="spearman"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_template_replay(
    league: str,
    database: Path,
    gap_features: pd.DataFrame,
    results_dir: Path,
    bootstrap_iterations: int,
) -> dict[str, object]:
    template_runner = parent._load_module(
        parent.TEMPLATE_RUNNER_PATH,
        f"v2_stabilized_gap_template_runner_{league}",
    )
    builder = template_runner.builder
    builder.set_active_league(league)
    methods = build_template_methods(builder)
    template_runner.METHODS = methods
    template_runner.BASELINE_METHOD = "incumbent"
    template_runner.BOOTSTRAP_SAMPLES = bootstrap_iterations
    max_season = min(int(builder.get_daily_max_template_season()), 2025)
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=database,
        scoring_matched_context=(
            league in builder.V2_SCORING_CONTEXT_CAPABLE_LEAGUES
        ),
    )
    template_features = list(dict.fromkeys(TEMPLATE_FEATURES.values()))
    merge_columns = [
        "player_key",
        "season",
        "expert_ppg_gap_provider_count",
        "expert_ppg_bull_gap_available",
        *template_features,
    ]
    template_gaps = gap_features[merge_columns + ["position"]].rename(
        columns={"position": "pos"}
    )
    projections = projections.merge(
        template_gaps,
        on=["player_key", "season", "pos"],
        how="left",
        validate="one_to_one",
    )
    projections["expert_ppg_bull_gap_available"] = projections[
        "expert_ppg_bull_gap_available"
    ].fillna(0).astype(int)
    projections["expert_ppg_gap_provider_count"] = projections[
        "expert_ppg_gap_provider_count"
    ].fillna(0).astype(int)
    for column in merge_columns[2:]:
        if column not in builder.MATCH_OUTPUT_COLS:
            builder.MATCH_OUTPUT_COLS.append(column)
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    forecasts = template_runner.base.load_production_oos_forecasts(max_season)
    target_templates = template_runner.base.build_production_oos_target_templates(
        templates, forecasts
    )
    targets = template_runner.base.build_targets(target_templates)
    predictions = template_runner.run_replay(templates, targets)
    expected_rows = len(targets) * len(methods)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} template predictions; found {len(predictions)}"
        )
    summary = parent._template_period_summary(template_runner, predictions)
    bootstrap_frames = [
        template_runner.bootstrap_comparison(
            predictions,
            candidate,
            parent.TEMPLATE_PERIODS,
            baseline_method="incumbent",
        )
        for candidate in methods
        if candidate != "incumbent"
    ]
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    coverage = parent._template_feature_coverage(templates, targets)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / "template_target_predictions.csv", index=False)
    summary.to_csv(results_dir / "template_period_summary.csv", index=False)
    bootstrap.to_csv(results_dir / "template_bootstrap.csv", index=False)
    coverage.to_csv(results_dir / "template_gap_coverage.csv", index=False)
    method_receipt = {
        method: {
            "recency_half_life": specification["recency_half_life"],
            "position_weights": specification["weights"],
        }
        for method, specification in methods.items()
    }
    (results_dir / "template_method_spec.json").write_text(
        json.dumps(method_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "max_template_season": max_season,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "summary": summary.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records"),
    }


def _configure_parent_runner() -> None:
    parent.STUDY_ROOT = STUDY_ROOT
    parent.POINT_VARIANT_FEATURES = POINT_VARIANT_FEATURES.copy()
    parent.PRIMARY_POINT_VARIANT = PRIMARY_POINT_VARIANT
    parent.TAIL_VARIANT_FEATURES = TAIL_VARIANT_FEATURES.copy()
    parent.PRIMARY_TEMPLATE_METHOD = PRIMARY_TEMPLATE_METHOD
    parent.build_asymmetric_projection_features = build_stabilized_projection_features
    parent._run_point_predictions = _extended_point_predictions
    parent._tail_comparison_summary = _tail_comparison_summary
    parent._tail_gates = _tail_gates
    parent._gap_outcome_diagnostics = _gap_outcome_diagnostics
    parent.build_template_methods = build_template_methods
    parent._run_template_replay = _run_template_replay


def _amend_manifest(results_dir: Path) -> None:
    path = results_dir / "input_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["parent_study"] = str(PARENT_RUNNER_PATH.resolve())
    manifest["feature_spec"] = {
        "source_table": "player_season_projection_values",
        "provider_value": "provider_points_per_team_game",
        "configured_points_complete_required": True,
        "minimum_providers": parent.MIN_ASYMMETRIC_PROVIDERS,
        "primary_formula": (
            "bull_gap / sqrt(abs(provider_ppg_median)^2 + 5^2)"
        ),
        "primary_point_variant": PRIMARY_POINT_VARIANT,
        "primary_tail_variant": PRIMARY_TAIL_VARIANT,
        "primary_template_method": PRIMARY_TEMPLATE_METHOD,
        "incumbent_provider_count_control": "projection_provider_count",
        "sensitivities": [
            "raw_gap",
            "smooth_k3",
            "smooth_k8",
            "hard_floor_k5",
            "additive_k5",
        ],
        "sensitivity_promotion_eligible": False,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_league(
    league: str,
    database: Path,
    results_dir: Path,
    bootstrap_iterations: int,
    template_bootstrap_iterations: int,
    *,
    skip_templates: bool,
) -> dict[str, object]:
    payload = parent._run_league(
        league,
        database,
        results_dir,
        bootstrap_iterations,
        template_bootstrap_iterations,
        skip_templates=skip_templates,
    )
    _amend_manifest(results_dir)
    return payload


def _combined_findings(decision: dict[str, object]) -> str:
    lines = [
        "# Stabilized bullish expert-gap cross-league decision",
        "",
        f"- Smooth-k5 point primary passes both leagues: `{decision['point_both_leagues_pass']}`.",
        f"- Smooth-k5 upper-tail primary passes both leagues: `{decision['tail_both_leagues_pass']}`.",
        f"- Smooth-k5 weekly-template primary passes both leagues: `{decision['template_both_leagues_pass']}`.",
        f"- Overall next action: `{decision['next_action']}`.",
        "",
        "## Point-model RMSE deltas",
        "",
        "| League | Surface | Raw | Smooth k3 | Smooth k5 primary | Smooth k8 | Hard floor k5 | Additive k5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "raw_gap",
        "smooth_k3",
        "smooth_k5",
        "smooth_k8",
        "hard_floor_k5",
        "additive_k5",
    ]
    for league, payload in decision["league_results"].items():
        point = pd.DataFrame(payload["point"])
        for method, label in (
            (parent.CONTROLLED_METHOD, "controlled"),
            (parent.PRODUCTION_METHOD, "production"),
        ):
            indexed = point[point["method"].eq(method)].set_index(
                "challenger_variant"
            )
            values = " | ".join(
                f"{float(indexed.loc[variant, 'pooled_delta']):+.5f}"
                for variant in order
            )
            lines.append(f"| {league.upper()} | {label} | {values} |")
    lines.extend(
        [
            "",
            "## Smooth-k5 upper-tail deltas",
            "",
            "| League | Event | Brier delta | Recent delta | AUC change | Season 95% |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for league, payload in decision["league_results"].items():
        tail = pd.DataFrame(payload["tail"])
        primary = tail[
            tail["challenger_variant"].eq(PRIMARY_TAIL_VARIANT)
        ]
        for row in primary.itertuples(index=False):
            lines.append(
                f"| {league.upper()} | {row.event} | {row.brier_delta:+.6f} | "
                f"{row.recent_brier_delta:+.6f} | "
                f"{row.challenger_auc - row.baseline_auc:+.5f} | "
                f"[{row.season_95_low:+.6f}, {row.season_95_high:+.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Smooth-k5 template deltas",
            "",
            "| League | Period | PPG CRPS | Contribution CRPS | Played CRPS | +3 Brier | +5 Brier | Impact Brier |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for league, payload in decision["league_results"].items():
        if not payload.get("template"):
            continue
        template = pd.DataFrame(payload["template"]["summary"])
        for period in ("full_2017_2025", "temporal_2023_2025"):
            selected = template[template["period"].eq(period)].set_index("method")
            baseline = selected.loc["incumbent"]
            primary = selected.loc[PRIMARY_TEMPLATE_METHOD]
            lines.append(
                f"| {league.upper()} | {period} | "
                f"{primary.ppg_crps - baseline.ppg_crps:+.6f} | "
                f"{primary.contribution_crps - baseline.contribution_crps:+.6f} | "
                f"{primary.played_crps - baseline.played_crps:+.6f} | "
                f"{primary.plus3_brier - baseline.plus3_brier:+.6f} | "
                f"{primary.plus5_brier - baseline.plus5_brier:+.6f} | "
                f"{primary.impact_brier - baseline.impact_brier:+.6f} |"
            )
    lines.extend(
        [
            "",
            "Only smooth k5 was promotion-eligible. The other denominator forms are declared sensitivities and cannot rescue a failed primary.",
            "",
            "The study is read-only. No production table, lock, feature contract, or template was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0 or args.template_bootstrap_iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    if args.league == "all" and args.database is not None:
        raise ValueError("--database cannot be used with --league all")
    _configure_parent_runner()
    parent._combined_findings = _combined_findings
    results_root = args.results_dir or STUDY_ROOT / "results"
    if args.league == "all":
        if args.combine_existing:
            payloads = [
                parent._load_existing_payload(league, results_root)
                for league in ("dk", "beta")
            ]
        else:
            payloads = [
                _run_league(
                    league,
                    DATABASES[league],
                    results_root / league,
                    args.bootstrap_iterations,
                    args.template_bootstrap_iterations,
                    skip_templates=args.skip_templates,
                )
                for league in ("dk", "beta")
            ]
        decision = parent._combine(payloads, results_root)
        print(json.dumps(decision, indent=2))
        return
    database = args.database or DATABASES[args.league]
    payload = _run_league(
        args.league,
        database,
        results_root / args.league,
        args.bootstrap_iterations,
        args.template_bootstrap_iterations,
        skip_templates=args.skip_templates,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
