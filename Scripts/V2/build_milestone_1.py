"""Build and atomically publish V2 Milestone 1 identity/outcome tables."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from Scripts.V2.build_player_identity import build_player_identity_frames
from Scripts.V2.build_player_outcomes import build_player_outcome_frames
from Scripts.V2.config import (
    COMPLETED_THROUGH_SEASON,
    LEAGUE,
    OUTPUT_DB_PATH,
    SOURCE_DB_PATH,
    START_SEASON,
    USEFUL_SEASON_MIN_GAMES,
)
from Scripts.V2.contracts import (
    BUILD_RUN_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    align_columns,
    create_run_id,
    publish_tables_atomic,
    read_existing_table,
    scoring_hash,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=SOURCE_DB_PATH)
    parser.add_argument("--output-db", type=Path, default=OUTPUT_DB_PATH)
    parser.add_argument("--start-season", type=int, default=START_SEASON)
    parser.add_argument(
        "--completed-through",
        type=int,
        default=COMPLETED_THROUGH_SEASON,
    )
    parser.add_argument("--league", default=LEAGUE)
    parser.add_argument(
        "--useful-season-min-games",
        type=int,
        default=USEFUL_SEASON_MIN_GAMES,
    )
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def _combined_history(
    existing: pd.DataFrame,
    current: pd.DataFrame,
    columns: tuple[str, ...],
    dedupe_columns: list[str],
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined = pd.concat([existing, current], ignore_index=True)
    combined = align_columns(combined, columns, "manifest_history")
    return combined.drop_duplicates(dedupe_columns, keep="last").reset_index(drop=True)


def build_milestone_1(
    source_database: Path = SOURCE_DB_PATH,
    output_database: Path = OUTPUT_DB_PATH,
    start_season: int = START_SEASON,
    completed_through_season: int = COMPLETED_THROUGH_SEASON,
    league: str = LEAGUE,
    useful_season_min_games: int = USEFUL_SEASON_MIN_GAMES,
    max_workers: int = 6,
) -> dict[str, object]:
    if start_season > completed_through_season:
        raise ValueError("start_season cannot be later than completed_through_season")
    if useful_season_min_games < 1:
        raise ValueError("useful_season_min_games must be positive")

    run_id = create_run_id()
    existing_identity = read_existing_table(
        output_database, "player_identity"
    )
    identity, aliases, identity_manifest = build_player_identity_frames(
        run_id=run_id,
        source_database=source_database,
        existing_identity=existing_identity,
    )
    outcomes, outcome_manifest = build_player_outcome_frames(
        player_identity=identity,
        player_aliases=aliases,
        seasons=range(start_season, completed_through_season + 1),
        league=league,
        run_id=run_id,
        completed_through_season=completed_through_season,
        useful_season_min_games=useful_season_min_games,
        max_workers=max_workers,
    )

    source_manifest = pd.concat(
        [identity_manifest, outcome_manifest], ignore_index=True
    )
    source_manifest = align_columns(
        source_manifest, SOURCE_MANIFEST_COLUMNS, "source_manifest"
    )
    build_run = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at_utc": utc_now(),
                "component": "milestone_1",
                "league": league,
                "start_season": start_season,
                "completed_through_season": completed_through_season,
                "useful_season_min_games": useful_season_min_games,
                "scoring_hash": scoring_hash(league),
                "identity_rows": len(identity),
                "alias_rows": len(aliases),
                "outcome_rows": len(outcomes),
                "source_observation_rows": pd.NA,
                "spine_rows": pd.NA,
                "projection_value_rows": pd.NA,
                "market_value_rows": pd.NA,
                "feature_rows": pd.NA,
                "feature_count": pd.NA,
                "foundation_run_id": run_id,
                "status": "complete",
            }
        ]
    )
    build_run = align_columns(build_run, BUILD_RUN_COLUMNS, "build_runs")

    existing_sources = read_existing_table(
        output_database, "source_manifest", SOURCE_MANIFEST_COLUMNS
    )
    existing_runs = read_existing_table(
        output_database, "build_runs", BUILD_RUN_COLUMNS
    )
    downstream_complete = (
        existing_runs["component"].isin(["milestone_2", "milestone_3"])
        & existing_runs["status"].eq("complete")
    )
    existing_runs.loc[downstream_complete, "status"] = "superseded"
    source_history = _combined_history(
        existing_sources,
        source_manifest,
        SOURCE_MANIFEST_COLUMNS,
        ["run_id", "source_name"],
    )
    run_history = _combined_history(
        existing_runs,
        build_run,
        BUILD_RUN_COLUMNS,
        ["run_id"],
    )

    publish_tables_atomic(
        output_database,
        {
            "player_identity": identity,
            "player_aliases": aliases,
            "player_season_outcomes": outcomes,
            "source_manifest": source_history,
            "build_runs": run_history,
        },
        drop_tables=(
            "player_season_sources",
            "player_season_spine",
            "player_season_projection_values",
            "player_season_market_values",
            "player_season_features",
            "feature_catalog",
            "feature_manifests",
            "feature_audit",
            "feature_correlations",
            "feature_source_resolution_audit",
            "model_runs",
            "model_fold_assignments",
            "model_specifications",
            "model_hyperparameter_results",
            "model_oof_predictions",
            "model_score_summary",
            "model_slice_summary",
        ),
    )

    return {
        "run_id": run_id,
        "output_database": str(output_database.resolve()),
        "identity_rows": len(identity),
        "confirmed_identities": int(identity["gsis_id"].notna().sum()),
        "provisional_identities": int(identity["gsis_id"].isna().sum()),
        "alias_rows": len(aliases),
        "outcome_rows": len(outcomes),
        "outcome_seasons": [
            int(outcomes["season"].min()),
            int(outcomes["season"].max()),
        ],
        "league": league,
        "scoring_hash": scoring_hash(league),
    }


def main() -> None:
    args = parse_args()
    result = build_milestone_1(
        source_database=args.source_db,
        output_database=args.output_db,
        start_season=args.start_season,
        completed_through_season=args.completed_through,
        league=args.league,
        useful_season_min_games=args.useful_season_min_games,
        max_workers=args.max_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
