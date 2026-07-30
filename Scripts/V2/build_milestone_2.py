"""Build and publish V2 Milestone 2 identity, outcomes, and projection spine."""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

from Scripts.V2.build_milestone_1 import build_milestone_1
from Scripts.V2.build_projection_spine import (
    build_player_season_sources,
    build_player_season_spine,
    validate_projection_spine,
)
from Scripts.V2.config import (
    CANDIDATE_SOURCE_TABLES,
    COMPLETED_THROUGH_SEASON,
    LEAGUE,
    OUTPUT_DB_PATH,
    PROJECTION_THROUGH_SEASON,
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
    source_row_exclusion_policy_receipt,
    table_exists,
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
    parser.add_argument(
        "--projection-through",
        type=int,
        default=PROJECTION_THROUGH_SEASON,
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
    combined = align_columns(combined, columns, "v2_history")
    return combined.drop_duplicates(dedupe_columns, keep="last").reset_index(
        drop=True
    )


def _source_manifest(
    source_database: Path,
    output_database: Path,
    run_id: str,
    start_season: int,
    projection_through_season: int,
    identity_rows: int,
    alias_rows: int,
    outcome_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with sqlite3.connect(source_database) as connection:
        for table, spec in CANDIDATE_SOURCE_TABLES.items():
            if not table_exists(connection, table):
                continue
            season_column = spec.get("season")
            if isinstance(season_column, str):
                row_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}" '
                    f'WHERE CAST("{season_column}" AS INTEGER) BETWEEN ? AND ?',
                    (start_season, projection_through_season),
                ).fetchone()[0]
            else:
                row_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            rows.append(
                {
                    "run_id": run_id,
                    "component": "projection_spine",
                    "source_name": f"candidate_{table}",
                    "source_uri": (
                        f"sqlite://{source_database.resolve()}#{table}"
                    ),
                    "source_sha256": pd.NA,
                    "row_count": int(row_count),
                }
            )

    for table, row_count in (
        ("player_identity", identity_rows),
        ("player_aliases", alias_rows),
        ("player_season_outcomes", outcome_rows),
    ):
        rows.append(
            {
                "run_id": run_id,
                "component": "projection_spine",
                "source_name": f"foundation_{table}",
                "source_uri": f"sqlite://{output_database.resolve()}#{table}",
                "source_sha256": pd.NA,
                "row_count": row_count,
            }
        )
    rows.append(source_row_exclusion_policy_receipt(run_id))
    return align_columns(
        pd.DataFrame(rows),
        SOURCE_MANIFEST_COLUMNS,
        "source_manifest",
    )


def build_milestone_2(
    source_database: Path = SOURCE_DB_PATH,
    output_database: Path = OUTPUT_DB_PATH,
    start_season: int = START_SEASON,
    completed_through_season: int = COMPLETED_THROUGH_SEASON,
    projection_through_season: int = PROJECTION_THROUGH_SEASON,
    league: str = LEAGUE,
    useful_season_min_games: int = USEFUL_SEASON_MIN_GAMES,
    max_workers: int = 6,
) -> dict[str, object]:
    if start_season > completed_through_season:
        raise ValueError("start_season cannot be later than completed-through")
    if projection_through_season < completed_through_season:
        raise ValueError(
            "projection-through cannot be earlier than completed-through"
        )

    foundation = build_milestone_1(
        source_database=source_database,
        output_database=output_database,
        start_season=start_season,
        completed_through_season=completed_through_season,
        league=league,
        useful_season_min_games=useful_season_min_games,
        max_workers=max_workers,
    )
    foundation_run_id = str(foundation["run_id"])
    run_id = create_run_id("milestone_2")

    identity = read_existing_table(output_database, "player_identity")
    aliases = read_existing_table(output_database, "player_aliases")
    outcomes = read_existing_table(output_database, "player_season_outcomes")
    player_sources = build_player_season_sources(
        aliases,
        identity,
        run_id=run_id,
        start_season=start_season,
        projection_through_season=projection_through_season,
    )
    spine = build_player_season_spine(
        player_sources,
        identity,
        outcomes,
        league=league,
        run_id=run_id,
        foundation_run_id=foundation_run_id,
        completed_through_season=completed_through_season,
    )
    validate_projection_spine(player_sources, spine)

    manifest = _source_manifest(
        source_database,
        output_database,
        run_id,
        start_season,
        projection_through_season,
        len(identity),
        len(aliases),
        len(outcomes),
    )
    build_run = align_columns(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at_utc": utc_now(),
                    "component": "milestone_2",
                    "league": league,
                    "start_season": start_season,
                    "completed_through_season": completed_through_season,
                    "useful_season_min_games": useful_season_min_games,
                    "scoring_hash": scoring_hash(league),
                    "identity_rows": len(identity),
                    "alias_rows": len(aliases),
                    "outcome_rows": len(outcomes),
                    "source_observation_rows": len(player_sources),
                    "spine_rows": len(spine),
                    "projection_value_rows": pd.NA,
                    "market_value_rows": pd.NA,
                    "feature_rows": pd.NA,
                    "feature_count": pd.NA,
                    "foundation_run_id": foundation_run_id,
                    "status": "complete",
                }
            ]
        ),
        BUILD_RUN_COLUMNS,
        "build_runs",
    )

    source_history = _combined_history(
        read_existing_table(
            output_database,
            "source_manifest",
            SOURCE_MANIFEST_COLUMNS,
        ),
        manifest,
        SOURCE_MANIFEST_COLUMNS,
        ["run_id", "source_name"],
    )
    run_history = _combined_history(
        read_existing_table(
            output_database,
            "build_runs",
            BUILD_RUN_COLUMNS,
        ),
        build_run,
        BUILD_RUN_COLUMNS,
        ["run_id"],
    )
    downstream_complete = (
        run_history["component"].eq("milestone_3")
        & run_history["status"].eq("complete")
    )
    run_history.loc[downstream_complete, "status"] = "superseded"
    publish_tables_atomic(
        output_database,
        {
            "player_season_sources": player_sources,
            "player_season_spine": spine,
            "source_manifest": source_history,
            "build_runs": run_history,
        },
        drop_tables=(
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

    completed = spine["outcome_complete"].eq(1)
    return {
        "run_id": run_id,
        "foundation_run_id": foundation_run_id,
        "output_database": str(output_database.resolve()),
        "source_observation_rows": len(player_sources),
        "spine_rows": len(spine),
        "spine_seasons": [
            int(spine["season"].min()),
            int(spine["season"].max()),
        ],
        "completed_candidates": int(completed.sum()),
        "observed_opportunity_candidates": int(
            (completed & spine["outcome_observed"].eq(1)).sum()
        ),
        "no_opportunity_candidates": int(
            spine["outcome_join_status"].eq("no_opportunity").sum()
        ),
        "unresolved_identity_candidates": int(
            spine["outcome_join_status"].eq("unresolved_identity").sum()
        ),
        "participation_label_rows": int(
            spine["active_target_available"].eq(1).sum()
        ),
        "pending_candidates": int(
            spine["outcome_join_status"].eq("pending").sum()
        ),
        "conditional_ppg_training_rows": int(
            spine["conditional_ppg_training_eligible"].sum()
        ),
        "league": league,
        "scoring_hash": scoring_hash(league),
    }


def main() -> None:
    args = parse_args()
    result = build_milestone_2(
        source_database=args.source_db,
        output_database=args.output_db,
        start_season=args.start_season,
        completed_through_season=args.completed_through,
        projection_through_season=args.projection_through,
        league=args.league,
        useful_season_min_games=args.useful_season_min_games,
        max_workers=args.max_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
