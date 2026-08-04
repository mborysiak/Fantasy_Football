"""Strict rolling validation of beta scoring-matched template context."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
PHASE_B_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_b_replay.py"
)
PHASE_A_PATH = PHASE_B_PATH.with_name("run_phase_a_rescore.py")
BASELINE = "production_hybrid"
ARMS = {
    BASELINE: {
        "scoring_matched_context": False,
        "scoring_matched_fallback_center": False,
    },
    "beta_context_only": {
        "scoring_matched_context": True,
        "scoring_matched_fallback_center": False,
    },
    "beta_scored_full": {
        "scoring_matched_context": True,
        "scoring_matched_fallback_center": True,
    },
}
V2_ERA_ARM = "beta_scored_v2_era"
V2_ERA_OPTIONS = {
    "scoring_matched_context": True,
    "scoring_matched_fallback_center": False,
    "min_template_season": 2017,
}
DECOUPLED_ARM = "beta_scored_decoupled"
DECOUPLED_OPTIONS = {
    "scoring_matched_context": True,
    "scoring_matched_fallback_center": False,
    "decouple_beta_match_center": True,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase_b = load_module("beta_context_phase_b", PHASE_B_PATH)
phase_a = load_module("beta_context_phase_a", PHASE_A_PATH)
builder = phase_b.builder
base = phase_b.base
pruning = phase_b.pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-db", type=Path, required=True)
    parser.add_argument("--simulation-db", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=STUDY_DIR / "results")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Regenerate findings and metadata from completed result CSVs.",
    )
    return parser.parse_args()


def build_arm_templates(
    arm: str,
    options: dict[str, bool | int],
    *,
    v2_database: Path,
    baseline_projections: pd.DataFrame,
    weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    projections = baseline_projections.copy()
    if options["scoring_matched_context"]:
        regenerated_match_columns = sorted(
            set(builder.MATCH_OUTPUT_COLS)
            - set(builder.PROJECTION_COMPONENT_COLS)
            - set(builder.PROJECTION_UNCERTAINTY_SOURCE_COLS)
        )
        projections = projections.drop(
            columns=[
                *regenerated_match_columns,
                "team_rb_rush_points",
                "team_rb_rec_points",
                "team_rec_points",
                "pass_catcher_share_of_room",
            ],
            errors="ignore",
        )
        projections = builder.apply_v2_scored_projection_context(
            projections,
            v2_database=v2_database,
            season_column="season",
            use_expert_donor_center=False,
            use_expert_fallback_center=options[
                "scoring_matched_fallback_center"
            ],
        )
        projections = builder.add_qb_team_rank_fields(
            projections,
            year_col="season",
            projection_col="avg_proj_points",
        )
        match_projection_ppg_col = "historical_pred_fp_per_game"
        if options.get("decouple_beta_match_center", False):
            fallback = projections.historical_center_policy.eq(
                "preseason_projection_fallback"
            ) & pd.to_numeric(
                projections.get("scoring_context_available", 1),
                errors="coerce",
            ).eq(1)
            projections["beta_match_projection_ppg"] = projections[
                "historical_pred_fp_per_game"
            ]
            projections.loc[
                fallback, "beta_match_projection_ppg"
            ] = (
                pd.to_numeric(
                    projections.loc[fallback, "avg_proj_points"],
                    errors="coerce",
                )
                / builder.projection_schedule_games(
                    projections.loc[fallback, "season"]
                )
            )
            projections["beta_match_projection_source"] = np.where(
                fallback,
                "v2_beta_expert_consensus_match_only",
                projections["historical_projection_source"],
            )
            match_projection_ppg_col = "beta_match_projection_ppg"
        projections = builder.add_projection_buckets(
            projections,
            value_col=match_projection_ppg_col,
            group_cols=["season", "pos"],
        )
        projections = builder.add_template_match_features(
            projections,
            group_cols=["season", "pos"],
            rank_pct_col="projection_rank_pct",
            total_points_col="avg_proj_points",
            projection_ppg_col=match_projection_ppg_col,
            preserve_signed_team_qb_context=True,
        )
    templates = builder.build_weekly_templates(
        projections,
        weekly,
        league="beta",
    )
    min_template_season = options.get("min_template_season")
    if min_template_season is not None:
        templates = templates[
            templates.season.ge(int(min_template_season))
        ].reset_index(drop=True)
    audit = pd.DataFrame(
        [
            {
                "arm": arm,
                "templates": int(len(templates)),
                "projection_context_rows": int(
                    templates.get(
                        "projection_context_source",
                        pd.Series(pd.NA, index=templates.index),
                    ).notna().sum()
                ),
                "legacy_oos_centers": int(
                    templates.historical_center_policy.eq(
                        "legacy_validated_oos"
                    ).sum()
                ),
                "dk_preseason_fallback_centers": int(
                    templates.historical_center_policy.eq(
                        "preseason_projection_fallback"
                    ).sum()
                ),
                "beta_expert_fallback_centers": int(
                    templates.historical_center_policy.eq(
                        "beta_scored_expert_fallback"
                    ).sum()
                ),
                "scoring_context_unavailable_rows": int(
                    pd.to_numeric(
                        templates.get(
                            "scoring_context_available",
                            pd.Series(1, index=templates.index),
                        ),
                        errors="coerce",
                    ).eq(0).sum()
                ),
                "template_eligible_rows": int(
                    templates.template_eligible.eq(1).sum()
                ),
                "missing_active_match_features": int(
                    templates[
                        sorted(
                            {
                                feature
                                for weights in builder.MATCH_FEATURE_WEIGHTS.values()
                                for feature in weights
                                if feature != "qb_team_rank_distance"
                            }
                        )
                    ].isna().sum().sum()
                ),
            }
        ]
    )
    return templates, audit


def build_targets(
    templates: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)
    # Beta 2018 QB projections have no complete sack-aware consensus under the
    # governed source quarantine. They remain auditable historical rows but
    # cannot serve as corrected donors or held-out scoring-context targets.
    targets = targets[
        ~(targets.season.eq(2018) & targets.pos.eq("QB"))
    ].copy()
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = (
        targets.groupby(["season", "pos"]).cumcount() + 1
    )
    return targets


def assert_identical_targets(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    arm: str,
) -> None:
    keys = ["player", "pos", "season"]
    baseline_keys = baseline[keys].sort_values(keys).reset_index(drop=True)
    candidate_keys = candidate[keys].sort_values(keys).reset_index(drop=True)
    if not baseline_keys.equals(candidate_keys):
        raise ValueError(f"Target cohort changed for {arm}")
    observed = [
        "historical_pred_fp_per_game",
        "observed_ppg",
        "observed_contribution",
        "played_games",
        "active_games",
    ]
    paired = baseline[keys + observed].merge(
        candidate[keys + observed],
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    for column in observed:
        if not np.allclose(
            paired[f"{column}_baseline"],
            paired[f"{column}_candidate"],
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"Target outcome/center changed for {arm}: {column}")


def feature_change_audit(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    arm: str,
) -> pd.DataFrame:
    keys = ["player", "season", "pos"]
    features = sorted(
        {
            feature
            for weights in builder.MATCH_FEATURE_WEIGHTS.values()
            for feature in weights
            if feature != "qb_team_rank_distance"
        }
    )
    paired = baseline[keys + features + ["historical_pred_fp_per_game"]].merge(
        candidate[keys + features + ["historical_pred_fp_per_game"]],
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    for feature in [*features, "historical_pred_fp_per_game"]:
        left = pd.to_numeric(paired[f"{feature}_baseline"], errors="coerce")
        right = pd.to_numeric(paired[f"{feature}_candidate"], errors="coerce")
        delta = (right - left).abs()
        rows.append(
            {
                "arm": arm,
                "feature": feature,
                "rows": int(len(paired)),
                "changed_rows": int(delta.gt(1e-12).sum()),
                "mean_absolute_delta": float(delta.mean()),
                "max_absolute_delta": float(delta.max()),
            }
        )
    return pd.DataFrame(rows)


def decision_table(
    deltas: pd.DataFrame,
    position_guardrails: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for arm in sorted(set(deltas.method) - {BASELINE}):
        method = deltas[deltas.method.eq(arm)]

        def one(tier: str, period: str) -> pd.Series:
            selected = method[
                method.tier.eq(tier) & method.period.eq(period)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one metric row for {arm}/{tier}/{period}"
                )
            return selected.iloc[0]

        development = one("core_main", "development_2017_2022")
        temporal = one("core_main", "temporal_2023_2025")
        depth = one("depth_main", "development_2017_2022")
        sensitivities = method[
            method.tier.isin(["core_strict", "core_main", "core_broad"])
            & method.period.isin(
                ["development_2017_2022", "temporal_2023_2025"]
            )
        ]
        position = position_guardrails[
            position_guardrails.method.eq(arm)
        ].iloc[0]
        depth_components = [
            float(depth[f"{metric}_relative_delta"])
            for metric in phase_a.METRICS
        ]
        depth_composite = float(np.mean(depth_components))
        gates = {
            "development_ppg": development.ppg_crps_relative_delta <= 0.0025,
            "development_contribution": (
                development.contribution_crps_relative_delta <= 0.0025
            ),
            "played_bias": development.abs_played_bias_degradation <= 0.15,
            "absence_calibration": (
                development.abs_extended_absence_calibration_degradation <= 0.01
            ),
            "coverage": development.ppg_80_coverage_delta >= -0.01,
            "position_ppg": position.max_position_ppg_relative_delta <= 0.01,
            "temporal_ppg": temporal.ppg_crps_relative_delta <= 0.005,
            "depth_composite": depth_composite <= 0.005,
            "depth_components": max(depth_components) <= 0.01,
            "tier_sensitivity": (
                sensitivities.ppg_crps_relative_delta.max() <= 0.005
                and sensitivities.contribution_crps_relative_delta.max() <= 0.005
            ),
        }
        rows.append(
            {
                "method": arm,
                "development_core_ppg_relative_delta": float(
                    development.ppg_crps_relative_delta
                ),
                "development_core_contribution_relative_delta": float(
                    development.contribution_crps_relative_delta
                ),
                "temporal_core_ppg_relative_delta": float(
                    temporal.ppg_crps_relative_delta
                ),
                "depth_composite_relative_delta": depth_composite,
                "worst_position_ppg_relative_delta": float(
                    position.max_position_ppg_relative_delta
                ),
                "worst_tier_ppg_relative_delta": float(
                    sensitivities.ppg_crps_relative_delta.max()
                ),
                "worst_tier_contribution_relative_delta": float(
                    sensitivities.contribution_crps_relative_delta.max()
                ),
                **{f"gate_{name}": bool(value) for name, value in gates.items()},
                "player_level_pass": bool(all(gates.values())),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    display = display.where(display.notna(), "")

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_findings_and_metadata(
    *,
    results_dir: Path,
    v2_database: Path,
    simulation_database: Path,
    max_season: int,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    template_audit: pd.DataFrame,
    runtime_seconds: float | None,
    recovered_from_completed_csvs: bool = False,
) -> dict[str, object]:
    full = decisions[decisions.method.eq("beta_scored_full")].iloc[0]
    lines = [
        "# Findings",
        "",
        "## Player-level decision",
        "",
        markdown_table(decisions),
        "",
        (
            "`beta_scored_full` advances to roster validation."
            if bool(full.player_level_pass)
            else "`beta_scored_full` does not advance to roster validation."
        ),
        "",
        "## Template policy audit",
        "",
        markdown_table(template_audit),
        "",
    ]
    (results_dir / "findings.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    metadata = {
        "league": "beta",
        "v2_database": str(v2_database),
        "simulation_database": str(simulation_database),
        "max_template_season": int(max_season),
        "arms": ARMS,
        "targets": int(
            predictions[["player", "pos", "season"]]
            .drop_duplicates()
            .shape[0]
        ),
        "prediction_rows": int(len(predictions)),
        "player_level_full_arm_pass": bool(full.player_level_pass),
        "runtime_seconds": runtime_seconds,
        "recovered_from_completed_csvs": recovered_from_completed_csvs,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    v2_database = args.v2_db.resolve()
    simulation_database = args.simulation_db.resolve()
    started = time.perf_counter()

    if args.summarize_existing:
        predictions = pd.read_csv(results_dir / "target_predictions.csv")
        decisions = pd.read_csv(results_dir / "decision_gates.csv")
        template_audit = pd.read_csv(
            results_dir / "template_policy_audit.csv"
        )
        metadata = write_findings_and_metadata(
            results_dir=results_dir,
            v2_database=v2_database,
            simulation_database=simulation_database,
            max_season=int(predictions.season.max()),
            predictions=predictions,
            decisions=decisions,
            template_audit=template_audit,
            runtime_seconds=None,
            recovered_from_completed_csvs=True,
        )
        print(json.dumps(metadata, indent=2), flush=True)
        return

    phase_b.configure_reference_globals()
    builder.set_simulation_db(simulation_database)
    builder.set_active_league("beta")
    base.builder.LEAGUE = "beta"
    max_season = builder.get_daily_max_template_season()
    weekly = builder.load_weekly_points(max_season, league="beta")
    forecasts = base.load_production_oos_forecasts(max_season)
    baseline_projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
        scoring_matched_context=False,
        scoring_matched_fallback_center=False,
    )

    templates_by_arm = {}
    targets_by_arm = {}
    predictions = []
    template_audits = []
    feature_audits = []
    baseline_targets = None
    baseline_templates = None
    production_specification = deepcopy(phase_b.METHODS["production"])

    for arm, options in ARMS.items():
        print(f"Building {arm}", flush=True)
        templates, audit = build_arm_templates(
            arm,
            options,
            v2_database=v2_database,
            baseline_projections=baseline_projections,
            weekly=weekly,
        )
        targets = build_targets(templates, forecasts)
        if baseline_targets is None:
            baseline_targets = targets
            baseline_templates = templates
        else:
            assert_identical_targets(baseline_targets, targets, arm)
            feature_audits.append(
                feature_change_audit(baseline_templates, templates, arm)
            )
        template_audits.append(audit)
        templates_by_arm[arm] = templates
        targets_by_arm[arm] = targets

    # Construct every arm before replay. The replay stack imports several
    # older research modules with mutable globals; separating construction
    # from scoring prevents one completed arm from contaminating the next
    # arm's identity/context build.
    for arm in ARMS:
        print(f"Replaying {arm}", flush=True)
        pruning.METHODS = {arm: deepcopy(production_specification)}
        arm_predictions = pruning.run_replay(
            templates_by_arm[arm],
            targets_by_arm[arm],
        )
        arm_predictions = phase_b.add_target_metadata(
            arm_predictions,
            targets_by_arm[arm],
        )
        predictions.append(arm_predictions)

    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    pd.concat(template_audits, ignore_index=True).to_csv(
        results_dir / "template_policy_audit.csv",
        index=False,
    )
    pd.concat(feature_audits, ignore_index=True).to_csv(
        results_dir / "feature_change_audit.csv",
        index=False,
    )
    baseline_targets.to_csv(results_dir / "target_cohort.csv", index=False)

    source = phase_a.Source(
        STUDY_DIR.name,
        results_dir.name,
        "beta",
        BASELINE,
        "fresh_expanded",
    )
    scored = phase_a.read_source(source)
    sources = (source,)
    metrics = phase_a.metric_table(scored, sources)
    deltas = phase_a.add_baseline_deltas(metrics)
    intervals = phase_a.bootstrap_intervals(scored, sources)
    position_guardrails = phase_a.position_ppg_guardrails(scored, sources)
    decisions = decision_table(deltas, position_guardrails)

    metrics.to_csv(results_dir / "role_tier_metrics.csv", index=False)
    deltas.to_csv(results_dir / "candidate_deltas.csv", index=False)
    intervals.to_csv(results_dir / "season_cluster_bootstrap.csv", index=False)
    position_guardrails.to_csv(
        results_dir / "position_ppg_guardrails.csv",
        index=False,
    )
    decisions.to_csv(results_dir / "decision_gates.csv", index=False)

    metadata = write_findings_and_metadata(
        results_dir=results_dir,
        v2_database=v2_database,
        simulation_database=simulation_database,
        max_season=max_season,
        predictions=predictions,
        decisions=decisions,
        template_audit=pd.concat(template_audits, ignore_index=True),
        runtime_seconds=time.perf_counter() - started,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
