"""Nested point/distribution gate for normalized expert-rank promotion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STUDY_ROOT.parents[2]
RAW_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-30_v2_market_rank_challengers"
    / "run_raw_rank_challenger.py"
)
PROJECTION_HELPER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-03_ridge_swap_downstream"
    / "run_projection_validation.py"
)
HASH_HELPER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-04_v2_logged_rank_disagreement"
    / "run_study.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.contracts import scoring_hash


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
BASELINE = "production"
CHALLENGER = "normalized_rank"
RANK_FEATURE = "scoring_specific_rank_position_percentile_median"
COMPONENTS = (
    "conditional_ppg_lasso",
    "conditional_ppg_random_forest",
    "conditional_ppg_lightgbm",
)
POINT_PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
DISTRIBUTION_PERIODS = {
    "all_2018_2025": (2018, 2025),
    "development_2018_2022": (2018, 2022),
    "temporal_2023_2025": (2023, 2025),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta", "all"), default="all")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--combine-existing", action="store_true")
    return parser.parse_args()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_helpers():
    raw = _load_module(
        RAW_RUNNER_PATH,
        "normalized_rank_promotion_raw_rank_reference",
    )
    projection = _load_module(
        PROJECTION_HELPER_PATH,
        "normalized_rank_promotion_projection_reference",
    )
    hashing = _load_module(
        HASH_HELPER_PATH,
        "normalized_rank_promotion_hash_reference",
    )
    projection.BASELINE = BASELINE
    projection.CHALLENGER = CHALLENGER
    projection.PERIODS = POINT_PERIODS.copy()
    projection.DISTRIBUTION_PERIODS = DISTRIBUTION_PERIODS.copy()
    return raw, projection, hashing


def _load_shadow(database: Path) -> pd.DataFrame:
    with sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True
    ) as connection:
        return pd.read_sql_query(
            """
            SELECT player_key, display_name, season, position, team,
                   conditional_ppg_primary_blend
            FROM locked_2026_shadow_predictions
            """,
            connection,
        )


def _challenger_predictions(
    locked,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    ppg, _, candidates = locked._target_frames(features)
    feature_columns = tuple((*locked.PRIMARY_PPG_FEATURES, RANK_FEATURE))
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Rank feature duplicates the locked feature surface")
    selections = []
    prediction_frames = []
    grids: dict[str, pd.DataFrame] = {}
    for component in COMPONENTS:
        grid = locked.MODEL_GRIDS[component]
        print(
            f"Nested normalized-rank grid: {component} "
            f"({len(grid)} candidates)",
            flush=True,
        )
        grid_predictions = locked._runtime_grid_predictions(
            ppg,
            feature_columns,
            component,
            grid,
            probability=False,
        )
        selected = locked._select_hyperparameters(
            grid_predictions,
            grid,
            component,
            probability=False,
        )
        predictions = locked._runtime_selected_predictions(
            ppg,
            candidates,
            feature_columns,
            fit_model_name=component,
            output_model_name=component,
            selected=selected,
        )
        selections.append(selected)
        prediction_frames.append(predictions)
        grids[component] = grid_predictions
    selected_hyperparameters = pd.concat(selections, ignore_index=True)
    component_predictions = pd.concat(prediction_frames, ignore_index=True)
    wide = component_predictions.pivot(
        index=["player_key", "season", "position"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    wide["prediction"] = wide[list(COMPONENTS)].mean(axis=1, skipna=False)
    if wide["prediction"].isna().any():
        raise ValueError("Challenger blend contains incomplete component rows")
    return wide, selected_hyperparameters, grids


def _assemble_evaluation(
    locked_predictions: pd.DataFrame,
    challenger: pd.DataFrame,
) -> pd.DataFrame:
    baseline = locked_predictions[
        locked_predictions["target_name"].eq("conditional_ppg")
        & locked_predictions["method"].eq(
            "conditional_ppg_primary_blend"
        )
    ].copy()
    baseline["method"] = BASELINE
    keys = ["player_key", "season", "position"]
    challenge = baseline.drop(columns=["prediction", "residual", "method"]).merge(
        challenger[keys + ["prediction"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(challenge) != len(baseline):
        raise ValueError(
            f"Challenger/baseline OOF mismatch: {len(challenge)} vs {len(baseline)}"
        )
    challenge["method"] = CHALLENGER
    challenge["residual"] = challenge["actual"] - challenge["prediction"]
    challenge["target_name"] = "conditional_ppg"
    columns = list(baseline.columns)
    challenge = challenge.loc[:, columns]
    output = pd.concat([baseline, challenge], ignore_index=True)
    output.sort_values(["method", "season", "player_key"], inplace=True)
    return output.reset_index(drop=True)


def _selection_comparison(
    locked_selected: pd.DataFrame,
    challenger_selected: pd.DataFrame,
) -> pd.DataFrame:
    baseline = locked_selected[
        locked_selected["model_name"].isin(COMPONENTS)
    ][
        [
            "model_name",
            "forecast_origin",
            "candidate_id",
            "parameters_json",
            "selection_score",
        ]
    ].copy()
    return baseline.merge(
        challenger_selected[
            [
                "model_name",
                "forecast_origin",
                "candidate_id",
                "parameters_json",
                "selection_score",
            ]
        ],
        on=["model_name", "forecast_origin"],
        how="outer",
        suffixes=("_production", "_normalized_rank"),
        validate="one_to_one",
    ).assign(
        selection_changed=lambda frame: frame[
            "candidate_id_production"
        ].ne(frame["candidate_id_normalized_rank"])
    )


def _coverage_pass(candidate: float, baseline: float, nominal: float) -> bool:
    return bool(
        abs(candidate - nominal) <= 0.02
        or abs(candidate - nominal) <= abs(baseline - nominal) + 1e-12
    )


def _league_gate(
    point: pd.DataFrame,
    distribution: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, object]:
    candidate_point = point[
        point["method"].eq(CHALLENGER)
        & point["slice_type"].eq("all")
        & point["slice_value"].eq("all")
    ].set_index("period")
    pooled = candidate_point.loc["all_2017_2025"]
    recent = candidate_point.loc["temporal_2023_2025"]
    player_interval = bootstrap.set_index("period").loc["all_2017_2025"]
    positions = point[
        point["method"].eq(CHALLENGER)
        & point["slice_type"].eq("position")
        & point["period"].eq("all_2017_2025")
    ]
    candidate_distribution = distribution[
        distribution["method"].eq(CHALLENGER)
        & distribution["slice_type"].eq("all")
        & distribution["slice_value"].eq("all")
        & distribution["period"].eq("all_2018_2025")
    ].iloc[0]
    gates = {
        "pooled_rmse_improves": bool(pooled.rmse_delta < 0),
        "recent_rmse_improves": bool(recent.rmse_delta < 0),
        "player_cluster_interval_upper_nonpositive": bool(
            player_interval.bootstrap_p975 <= 0
        ),
        "no_position_worsens_more_than_0_01": bool(
            positions.rmse_delta.max() <= 0.01
        ),
        "pooled_distribution_crps_nonworse": bool(
            candidate_distribution.crps_delta <= 0
        ),
        "coverage_50_acceptable": _coverage_pass(
            float(candidate_distribution.coverage_50),
            float(candidate_distribution.coverage_50_baseline),
            0.50,
        ),
        "coverage_80_acceptable": _coverage_pass(
            float(candidate_distribution.coverage_80),
            float(candidate_distribution.coverage_80_baseline),
            0.80,
        ),
    }
    return {
        "gates": gates,
        "all_league_gates_pass": all(gates.values()),
        "pooled_rmse_delta": float(pooled.rmse_delta),
        "recent_rmse_delta": float(recent.rmse_delta),
        "player_cluster_95": [
            float(player_interval.bootstrap_p025),
            float(player_interval.bootstrap_p975),
        ],
        "max_position_rmse_delta": float(positions.rmse_delta.max()),
        "pooled_distribution_crps_delta": float(
            candidate_distribution.crps_delta
        ),
        "coverage_50": float(candidate_distribution.coverage_50),
        "coverage_80": float(candidate_distribution.coverage_80),
    }


def _run_league(
    league: str,
    database: Path,
    results_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    raw, projection, hashing = _load_helpers()
    before_hash = hashing._file_sha256(database)
    (
        _,
        locked,
        features,
        locked_selected,
        locked_predictions,
        _,
        source_coverage,
        _,
        feature_run_id,
        _,
        ppr_resolution,
        input_manifest,
        position_audit,
    ) = raw._load_inputs(database, league)
    expected_scoring_hash = scoring_hash(league)
    observed_scoring_hashes = set(features["scoring_hash"].dropna().astype(str))
    if observed_scoring_hashes != {expected_scoring_hash}:
        raise ValueError(
            f"Scoring mismatch: {observed_scoring_hashes} vs {expected_scoring_hash}"
        )
    if RANK_FEATURE not in features or features[RANK_FEATURE].notna().sum() == 0:
        raise ValueError("Normalized expert-rank feature is unavailable")
    challenger, challenger_selected, grids = _challenger_predictions(
        locked, features
    )
    evaluation = _assemble_evaluation(locked_predictions, challenger)
    projection.HISTORICAL_ORIGINS = tuple(locked.OUTER_SEASONS)
    calibrated = projection.strict_prior_residuals(evaluation)
    distribution_rows = projection.add_distribution_rows(calibrated)
    point_summary = projection.point_summary(evaluation)
    distribution_summary = projection.distribution_summary(distribution_rows)
    bootstrap = projection.cluster_bootstrap(evaluation)
    selection_comparison = _selection_comparison(
        locked_selected, challenger_selected
    )
    shadow = _load_shadow(database).merge(
        challenger[
            challenger["season"].eq(locked.CURRENT_SEASON)
        ][["player_key", "season", "position", "prediction"]].rename(
            columns={"prediction": CHALLENGER}
        ),
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    shadow.rename(
        columns={"conditional_ppg_primary_blend": BASELINE}, inplace=True
    )
    shadow["candidate_minus_production"] = shadow[CHALLENGER] - shadow[BASELINE]
    gate = _league_gate(point_summary, distribution_summary, bootstrap)
    after_hash = hashing._file_sha256(database)
    if before_hash != after_hash:
        raise RuntimeError("Production database changed during read-only nested test")

    results_dir.mkdir(parents=True, exist_ok=True)
    for component, grid_predictions in grids.items():
        grid_predictions.to_csv(
            results_dir / f"grid_predictions_{component}.csv", index=False
        )
    challenger.to_csv(results_dir / "component_predictions.csv", index=False)
    evaluation.to_csv(results_dir / "paired_point_predictions.csv", index=False)
    calibrated.to_csv(results_dir / "strict_prior_residuals.csv", index=False)
    distribution_rows.to_csv(results_dir / "distribution_rows.csv", index=False)
    shadow.to_csv(results_dir / "shadow_2026_predictions.csv", index=False)
    challenger_selected.to_csv(
        results_dir / "selected_hyperparameters.csv", index=False
    )
    selection_comparison.to_csv(
        results_dir / "selection_comparison.csv", index=False
    )
    point_summary.to_csv(results_dir / "point_summary.csv", index=False)
    distribution_summary.to_csv(
        results_dir / "distribution_summary.csv", index=False
    )
    bootstrap.to_csv(
        results_dir / "point_player_cluster_bootstrap.csv", index=False
    )
    source_coverage.to_csv(
        results_dir / "rank_source_coverage.csv", index=False
    )
    position_audit.to_csv(results_dir / "rank_position_audit.csv", index=False)
    manifest = {
        **input_manifest,
        "database_sha256_before": before_hash,
        "database_sha256_after": after_hash,
        "feature_run_id": feature_run_id,
        "feature_columns": [*locked.PRIMARY_PPG_FEATURES, RANK_FEATURE],
        "candidate_feature": RANK_FEATURE,
        "ppr_identity_resolution": ppr_resolution,
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "gate_audit.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "league": league,
        "database": str(database.resolve()),
        "database_sha256": before_hash,
        "feature_run_id": feature_run_id,
        "gate": gate,
        "recent_season_deltas": point_summary[
            point_summary["method"].eq(CHALLENGER)
            & point_summary["slice_type"].eq("season")
            & point_summary["period"].eq("all_2017_2025")
            & point_summary["slice_value"].astype(str).isin(("2023", "2024", "2025"))
        ][["slice_value", "rmse_delta"]].to_dict("records"),
        "selection_changes": int(selection_comparison["selection_changed"].sum()),
        "selection_rows": int(len(selection_comparison)),
        "shadow_complete_rows": int(shadow[[BASELINE, CHALLENGER]].notna().all(axis=1).sum()),
        "shadow_median_abs_delta": float(shadow["candidate_minus_production"].abs().median()),
    }
    (results_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _load_existing(league: str) -> dict[str, object]:
    return json.loads(
        (STUDY_ROOT / f"results_projection_{league}" / "result.json").read_text(
            encoding="utf-8"
        )
    )


def _combine(
    payloads: Sequence[dict[str, object]],
    results_dir: Path,
) -> dict[str, object]:
    by_league = {str(payload["league"]): payload for payload in payloads}
    if set(by_league) != {"dk", "beta"}:
        raise ValueError("Stage A combination requires DK and beta")
    recent_deltas = [
        float(row["rmse_delta"])
        for league in ("dk", "beta")
        for row in by_league[league]["recent_season_deltas"]
    ]
    recent_wins = sum(value < 0 for value in recent_deltas)
    league_pass = all(
        bool(by_league[league]["gate"]["all_league_gates_pass"])
        for league in ("dk", "beta")
    )
    recent_gate = recent_wins >= 5
    advance = league_pass and recent_gate
    decision = {
        "stage": "nested_point_and_distribution",
        "league_gates_pass": league_pass,
        "recent_season_wins": recent_wins,
        "recent_season_count": len(recent_deltas),
        "recent_five_of_six_gate": recent_gate,
        "stage_a_pass": advance,
        "next_action": (
            "run_stage_b_template_and_roster_transport"
            if advance
            else "retain_normalized_rank_outside_production"
        ),
        "league_results": by_league,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "stage_a_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Normalized Expert-Rank Promotion - Stage A",
        "",
        f"- Stage A passes: `{advance}`.",
        f"- Recent season wins: `{recent_wins}/{len(recent_deltas)}`.",
        f"- Next action: `{decision['next_action']}`.",
        "",
        "| League | Pooled RMSE delta | Recent RMSE delta | Player 95% | Distribution CRPS delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for league in ("dk", "beta"):
        gate = by_league[league]["gate"]
        interval = gate["player_cluster_95"]
        lines.append(
            f"| {league.upper()} | {gate['pooled_rmse_delta']:+.5f} | "
            f"{gate['recent_rmse_delta']:+.5f} | "
            f"[{interval[0]:+.5f}, {interval[1]:+.5f}] | "
            f"{gate['pooled_distribution_crps_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "Stage B runs only after a Stage A pass. No production table, model lock, parameter cache, template, or app artifact changed.",
            "",
        ]
    )
    (results_dir / "stage_a_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def main() -> None:
    args = parse_args()
    if args.league == "all" and args.database is not None:
        raise ValueError("--database cannot be used with --league all")
    if args.combine_existing and args.league != "all":
        raise ValueError("--combine-existing requires --league all")
    if args.league == "all":
        if args.combine_existing:
            payloads = [_load_existing(league) for league in ("dk", "beta")]
        else:
            payloads = [
                _run_league(
                    league,
                    DATABASES[league],
                    STUDY_ROOT / f"results_projection_{league}",
                )
                for league in ("dk", "beta")
            ]
        _combine(payloads, args.results_dir or STUDY_ROOT / "results")
        return
    _run_league(
        args.league,
        args.database or DATABASES[args.league],
        args.results_dir or STUDY_ROOT / f"results_projection_{args.league}",
    )


if __name__ == "__main__":
    main()
