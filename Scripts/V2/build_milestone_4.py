"""Run the initial V2 shadow-model comparison and publish OOF evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Scripts.V2.build_milestone_3 import build_milestone_3
from Scripts.V2.config import (
    COMPLETED_THROUGH_SEASON,
    LEAGUE,
    OUTPUT_DB_PATH,
    START_SEASON,
    USEFUL_SEASON_MIN_GAMES,
)
from Scripts.V2.contracts import (
    BUILD_RUN_COLUMNS,
    MODEL_FOLD_COLUMNS,
    MODEL_HYPERPARAMETER_COLUMNS,
    MODEL_OOF_COLUMNS,
    MODEL_RUN_COLUMNS,
    MODEL_SCORE_COLUMNS,
    MODEL_SLICE_COLUMNS,
    MODEL_SPECIFICATION_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    align_columns,
    create_run_id,
    publish_tables_atomic,
    read_existing_table,
    scoring_hash,
    utc_now,
)
from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    PARTICIPATION_TARGET,
    build_feature_sets,
    build_score_summary,
    build_slice_summary,
    build_target_frames,
    initial_model_specs,
    make_fold_assignments,
    run_model_spec,
    specification_row,
    validate_oof_predictions,
)


DEFAULT_VALIDATION_START_SEASON = 2017
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_SEED = 1234
DEFAULT_SEARCH_ITERATIONS = 4
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "studies"
    / "2026-07-27_v2_modeling_framework"
    / "results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-db", type=Path, default=OUTPUT_DB_PATH)
    parser.add_argument(
        "--validation-start",
        type=int,
        default=DEFAULT_VALIDATION_START_SEASON,
    )
    parser.add_argument(
        "--validation-end",
        type=int,
        default=COMPLETED_THROUGH_SEASON,
    )
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=DEFAULT_SEARCH_ITERATIONS,
    )
    parser.add_argument(
        "--models",
        help="Optional comma-separated model names for a focused run.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-results-files", action="store_true")
    parser.add_argument("--rebuild-feature-mart", action="store_true")
    parser.add_argument("--verbose-models", action="store_true")
    return parser.parse_args()


def _combined_history(
    existing: pd.DataFrame,
    current: pd.DataFrame,
    columns: tuple[str, ...],
    keys: list[str],
) -> pd.DataFrame:
    if existing.empty:
        combined = current.copy()
    elif current.empty:
        combined = existing.copy()
    else:
        combined = pd.DataFrame.from_records(
            existing.to_dict("records") + current.to_dict("records"),
            columns=columns,
        )
    combined = combined.drop_duplicates(keys, keep="last")
    return align_columns(combined, columns, "history")


def _ensure_feature_mart(
    output_database: Path,
    rebuild_feature_mart: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    features = read_existing_table(output_database, "player_season_features")
    manifests = read_existing_table(output_database, "feature_manifests")
    catalog = read_existing_table(output_database, "feature_catalog")
    if (
        rebuild_feature_mart
        or features.empty
        or manifests.empty
        or catalog.empty
    ):
        build_milestone_3(output_database=output_database)
        features = read_existing_table(
            output_database, "player_season_features"
        )
        manifests = read_existing_table(output_database, "feature_manifests")
        catalog = read_existing_table(output_database, "feature_catalog")
    if features.empty or manifests.empty or catalog.empty:
        raise ValueError("Milestone 3 feature mart is unavailable")
    feature_run_ids = features["run_id"].dropna().astype(str).unique()
    if len(feature_run_ids) != 1:
        raise ValueError(
            "Feature mart must contain exactly one active feature run ID"
        )
    feature_run_id = str(feature_run_ids[0])
    manifest_run_ids = manifests["run_id"].dropna().astype(str).unique()
    if set(manifest_run_ids) != {feature_run_id}:
        raise ValueError("Feature mart and manifests do not share one run ID")
    catalog_run_ids = catalog["run_id"].dropna().astype(str).unique()
    if set(catalog_run_ids) != {feature_run_id}:
        raise ValueError("Feature mart and catalog do not share one run ID")
    return features, manifests, catalog, feature_run_id


def _selected_specs(
    search_iterations: int,
    model_names: str | None,
):
    if search_iterations < 1:
        raise ValueError("search_iterations must be positive")
    specs = initial_model_specs(search_iterations=search_iterations)
    if model_names:
        selected = {
            name.strip() for name in model_names.split(",") if name.strip()
        }
        unknown = selected.difference(spec.model_name for spec in specs)
        if unknown:
            raise ValueError(f"Unknown model names: {sorted(unknown)}")
        specs = tuple(spec for spec in specs if spec.model_name in selected)
    if not specs:
        raise ValueError("No model specifications were selected")
    return specs


def _model_run_history(
    output_database: Path,
    model_run: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    history = _combined_history(
        read_existing_table(output_database, "model_runs", MODEL_RUN_COLUMNS),
        model_run,
        MODEL_RUN_COLUMNS,
        ["run_id"],
    )
    prior = (
        history["status"].eq("complete")
        & history["run_id"].ne(run_id)
    )
    history.loc[prior, "status"] = "superseded"
    return history


def _build_run_history(
    output_database: Path,
    build_run: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    history = _combined_history(
        read_existing_table(output_database, "build_runs", BUILD_RUN_COLUMNS),
        build_run,
        BUILD_RUN_COLUMNS,
        ["run_id"],
    )
    prior = (
        history["component"].eq("milestone_4a")
        & history["status"].eq("complete")
        & history["run_id"].ne(run_id)
    )
    history.loc[prior, "status"] = "superseded"
    return history


def _source_history(
    output_database: Path,
    source: pd.DataFrame,
) -> pd.DataFrame:
    return _combined_history(
        read_existing_table(
            output_database,
            "source_manifest",
            SOURCE_MANIFEST_COLUMNS,
        ),
        source,
        SOURCE_MANIFEST_COLUMNS,
        ["run_id", "source_name"],
    )


def _metric_leader(
    scores: pd.DataFrame,
    target_name: str,
    metric: str,
) -> pd.Series:
    candidates = scores[
        scores["target_name"].eq(target_name)
        & scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq(metric)
    ].sort_values(["value", "model_name"])
    if candidates.empty:
        return pd.Series(dtype=object)
    return candidates.iloc[0]


def _format_summary(
    run_id: str,
    feature_run_id: str,
    validation_start: int,
    validation_end: int,
    folds: pd.DataFrame,
    specifications: pd.DataFrame,
    scores: pd.DataFrame,
) -> str:
    ppg = scores[
        scores["target_name"].eq(CONDITIONAL_PPG_TARGET)
        & scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("rmse")
    ].sort_values(["value", "model_name"])
    participation = scores[
        scores["target_name"].eq(PARTICIPATION_TARGET)
        & scores["aggregation"].eq("pooled_oof")
        & scores["metric"].eq("brier")
    ].sort_values(["value", "model_name"])

    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No models run for this target._"
        rows = [
            "| Model | Value | Baseline | Delta |",
            "|---|---:|---:|---:|",
        ]
        for row in frame.itertuples(index=False):
            rows.append(
                f"| `{row.model_name}` | {row.value:.4f} | "
                f"{row.baseline_value:.4f} | {row.delta:+.4f} |"
            )
        return "\n".join(rows)

    return f"""# V2 M4A Initial OOF Results

This is a shadow-only comparison. It does not replace production projections,
templates, or optimizer inputs.

- Model run: `{run_id}`
- Feature run: `{feature_run_id}`
- OOF window: {validation_start}-{validation_end}
- Conditional-PPG fold rows: {
        len(folds[folds["target_name"].eq(CONDITIONAL_PPG_TARGET)])
    }
- Participation fold rows: {
        len(folds[folds["target_name"].eq(PARTICIPATION_TARGET)])
    }
- Compared specifications: {len(specifications)}

## Conditional PPG

Pooled OOF RMSE; lower is better. Delta is model minus the active-game-when-
available expert hybrid.

{table(ppg)}

## Participation

Pooled OOF Brier score; lower is better. Delta is model minus the leakage-safe
prior-position-rate baseline.

{table(participation)}

## Interpretation Boundary

Five folds cover every validation season. Each OOF prediction is fit only on
seasons strictly earlier than that player-season. Hyperparameters may use the
other four folds' rolling predictions, but never the held fold. PCA,
agglomeration, and univariate selection are isolated pipeline challengers;
they are not stacked together.
"""


def _write_results(
    results_directory: Path,
    run_id: str,
    feature_run_id: str,
    validation_start: int,
    validation_end: int,
    folds: pd.DataFrame,
    specifications: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    scores: pd.DataFrame,
    slices: pd.DataFrame,
) -> None:
    results_directory.mkdir(parents=True, exist_ok=True)
    specifications.to_csv(
        results_directory / "model_specifications.csv", index=False
    )
    hyperparameters.to_csv(
        results_directory / "model_hyperparameter_results.csv", index=False
    )
    scores.to_csv(results_directory / "model_score_summary.csv", index=False)
    slices.to_csv(results_directory / "model_slice_summary.csv", index=False)
    summary = _format_summary(
        run_id,
        feature_run_id,
        validation_start,
        validation_end,
        folds,
        specifications,
        scores,
    )
    (results_directory / "summary.md").write_text(summary, encoding="utf-8")


def build_milestone_4(
    output_database: Path = OUTPUT_DB_PATH,
    validation_start_season: int = DEFAULT_VALIDATION_START_SEASON,
    validation_end_season: int = COMPLETED_THROUGH_SEASON,
    n_splits: int = DEFAULT_N_SPLITS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    model_names: str | None = None,
    results_directory: Path | None = DEFAULT_RESULTS_DIR,
    rebuild_feature_mart: bool = False,
    quiet_models: bool = True,
) -> dict[str, object]:
    if validation_start_season > validation_end_season:
        raise ValueError("Validation start cannot be later than validation end")
    if n_splits < 2:
        raise ValueError("At least two folds are required")
    features, manifests, catalog, feature_run_id = _ensure_feature_mart(
        output_database, rebuild_feature_mart
    )
    if validation_end_season > int(features["season"].max()):
        raise ValueError("Validation end is later than the feature mart")
    specs = _selected_specs(search_iterations, model_names)
    feature_sets = build_feature_sets(manifests)
    target_frames = build_target_frames(features, validation_end_season)
    run_id = create_run_id("milestone_4a")

    target_names = sorted({spec.target_name for spec in specs})
    assignments_by_target = {
        target_name: make_fold_assignments(
            target_frames[target_name],
            target_name=target_name,
            run_id=run_id,
            validation_start_season=validation_start_season,
            n_splits=n_splits,
            random_seed=random_seed,
        )
        for target_name in target_names
    }
    fold_assignments = align_columns(
        pd.concat(assignments_by_target.values(), ignore_index=True),
        MODEL_FOLD_COLUMNS,
        "model_fold_assignments",
    )

    oof_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    specification_rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        feature_columns = feature_sets[spec.target_name][spec.feature_set]
        specification_rows.append(
            specification_row(spec, feature_columns, run_id)
        )
        print(
            f"[{index}/{len(specs)}] {spec.target_name}: "
            f"{spec.model_name}",
            flush=True,
        )
        oof, parameters = run_model_spec(
            target_frames[spec.target_name],
            assignments_by_target[spec.target_name],
            spec,
            feature_columns,
            run_id=run_id,
            feature_run_id=feature_run_id,
            validation_start_season=validation_start_season,
            n_splits=n_splits,
            random_seed=random_seed,
            quiet=quiet_models,
        )
        oof_frames.append(oof)
        if not parameters.empty:
            parameter_frames.append(parameters)
    specifications = align_columns(
        pd.DataFrame(specification_rows),
        MODEL_SPECIFICATION_COLUMNS,
        "model_specifications",
    )
    oof_predictions = align_columns(
        pd.concat(oof_frames, ignore_index=True),
        MODEL_OOF_COLUMNS,
        "model_oof_predictions",
    )
    if parameter_frames:
        hyperparameters = align_columns(
            pd.concat(parameter_frames, ignore_index=True),
            MODEL_HYPERPARAMETER_COLUMNS,
            "model_hyperparameter_results",
        )
    else:
        hyperparameters = pd.DataFrame(
            columns=MODEL_HYPERPARAMETER_COLUMNS
        )
    validate_oof_predictions(oof_predictions, fold_assignments, specs)
    scores = align_columns(
        build_score_summary(oof_predictions, run_id),
        MODEL_SCORE_COLUMNS,
        "model_score_summary",
    )
    slices = align_columns(
        build_slice_summary(oof_predictions, run_id),
        MODEL_SLICE_COLUMNS,
        "model_slice_summary",
    )

    conditional_rows = len(
        fold_assignments[
            fold_assignments["target_name"].eq(CONDITIONAL_PPG_TARGET)
        ]
    )
    participation_rows = len(
        fold_assignments[
            fold_assignments["target_name"].eq(PARTICIPATION_TARGET)
        ]
    )
    model_run = align_columns(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at_utc": utc_now(),
                    "feature_run_id": feature_run_id,
                    "league": LEAGUE,
                    "validation_start_season": validation_start_season,
                    "validation_end_season": validation_end_season,
                    "n_splits": n_splits,
                    "random_seed": random_seed,
                    "conditional_ppg_rows": conditional_rows,
                    "participation_rows": participation_rows,
                    "model_count": len(specifications),
                    "status": "complete",
                }
            ]
        ),
        MODEL_RUN_COLUMNS,
        "model_runs",
    )
    build_run = align_columns(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at_utc": utc_now(),
                    "component": "milestone_4a",
                    "league": LEAGUE,
                    "start_season": START_SEASON,
                    "completed_through_season": validation_end_season,
                    "useful_season_min_games": USEFUL_SEASON_MIN_GAMES,
                    "scoring_hash": scoring_hash(LEAGUE),
                    "identity_rows": features["player_key"].nunique(),
                    "alias_rows": pd.NA,
                    "outcome_rows": pd.NA,
                    "source_observation_rows": pd.NA,
                    "spine_rows": len(features),
                    "projection_value_rows": pd.NA,
                    "market_value_rows": pd.NA,
                    "feature_rows": len(features),
                    "feature_count": catalog["feature_name"].nunique(),
                    "foundation_run_id": feature_run_id,
                    "status": "complete",
                }
            ]
        ),
        BUILD_RUN_COLUMNS,
        "build_runs",
    )
    source = align_columns(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "component": "modeling",
                    "source_name": "v2_player_season_features",
                    "source_uri": (
                        f"sqlite://{output_database.resolve()}"
                        "#player_season_features"
                    ),
                    "source_sha256": pd.NA,
                    "row_count": len(features),
                }
            ]
        ),
        SOURCE_MANIFEST_COLUMNS,
        "source_manifest",
    )

    publish_tables_atomic(
        output_database,
        {
            "model_runs": _model_run_history(
                output_database, model_run, run_id
            ),
            "model_fold_assignments": fold_assignments,
            "model_specifications": specifications,
            "model_hyperparameter_results": hyperparameters,
            "model_oof_predictions": oof_predictions,
            "model_score_summary": scores,
            "model_slice_summary": slices,
            "source_manifest": _source_history(output_database, source),
            "build_runs": _build_run_history(
                output_database, build_run, run_id
            ),
        },
    )
    if results_directory is not None:
        _write_results(
            results_directory,
            run_id,
            feature_run_id,
            validation_start_season,
            validation_end_season,
            fold_assignments,
            specifications,
            hyperparameters,
            scores,
            slices,
        )

    ppg_leader = _metric_leader(
        scores, CONDITIONAL_PPG_TARGET, "rmse"
    )
    participation_leader = _metric_leader(
        scores, PARTICIPATION_TARGET, "brier"
    )
    return {
        "run_id": run_id,
        "feature_run_id": feature_run_id,
        "output_database": str(output_database.resolve()),
        "validation_seasons": [
            validation_start_season,
            validation_end_season,
        ],
        "n_splits": n_splits,
        "conditional_ppg_rows": conditional_rows,
        "participation_rows": participation_rows,
        "model_count": len(specifications),
        "oof_prediction_rows": len(oof_predictions),
        "conditional_ppg_leader": (
            {
                "model_name": ppg_leader["model_name"],
                "rmse": float(ppg_leader["value"]),
                "delta_vs_expert_hybrid": float(ppg_leader["delta"]),
            }
            if not ppg_leader.empty
            else None
        ),
        "participation_leader": (
            {
                "model_name": participation_leader["model_name"],
                "brier": float(participation_leader["value"]),
                "delta_vs_prior_rate": float(participation_leader["delta"]),
            }
            if not participation_leader.empty
            else None
        ),
        "results_directory": (
            str(results_directory.resolve())
            if results_directory is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    result = build_milestone_4(
        output_database=args.output_db,
        validation_start_season=args.validation_start,
        validation_end_season=args.validation_end,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        search_iterations=args.search_iterations,
        model_names=args.models,
        results_directory=(
            None if args.no_results_files else args.results_dir
        ),
        rebuild_feature_mart=args.rebuild_feature_mart,
        quiet_models=not args.verbose_models,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
