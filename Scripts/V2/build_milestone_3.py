"""Build and publish the V2 Milestone 3 preseason feature mart."""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import pandas as pd

from Scripts.V2.build_feature_mart import build_feature_mart
from Scripts.V2.build_feature_sources import (
    build_market_values,
    build_projection_values,
)
from Scripts.V2.build_milestone_2 import build_milestone_2
from Scripts.V2.config import (
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
    FEATURE_SOURCE_AUDIT_COLUMNS,
    MODEL_RUN_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    align_columns,
    assert_no_source_row_exclusions,
    create_run_id,
    publish_tables_atomic,
    read_existing_table,
    scoring_hash,
    source_row_exclusion_policy_receipt,
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
    parser.add_argument(
        "--reuse-foundation",
        action="store_true",
        help="Reuse a validated active Milestone 2 spine instead of rebuilding it.",
    )
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


def _active_foundation(
    output_database: Path,
    start_season: int,
    completed_through_season: int,
    projection_through_season: int,
    league: str,
    useful_season_min_games: int,
) -> dict[str, object]:
    runs = read_existing_table(
        output_database,
        "build_runs",
        BUILD_RUN_COLUMNS,
    )
    eligible = runs[
        runs["component"].eq("milestone_2")
        & runs["status"].eq("complete")
        & runs["league"].eq(league)
        & runs["start_season"].eq(start_season)
        & runs["completed_through_season"].eq(completed_through_season)
        & runs["useful_season_min_games"].eq(useful_season_min_games)
        & runs["scoring_hash"].eq(scoring_hash(league))
    ].sort_values("created_at_utc")
    if eligible.empty:
        raise ValueError(
            "No active Milestone 2 foundation matches the requested build"
        )
    foundation_run_id = str(eligible.iloc[-1]["run_id"])
    source_manifest = read_existing_table(
        output_database,
        "source_manifest",
        SOURCE_MANIFEST_COLUMNS,
    )
    expected_receipt = source_row_exclusion_policy_receipt(foundation_run_id)
    receipts = source_manifest[
        source_manifest["run_id"].eq(foundation_run_id)
        & source_manifest["component"].eq(expected_receipt["component"])
        & source_manifest["source_name"].eq(expected_receipt["source_name"])
    ]
    if len(receipts) != 1:
        raise ValueError(
            "The active Milestone 2 foundation has no unique source-row "
            "exclusion policy receipt and must be rebuilt"
        )
    receipt = receipts.iloc[0]
    if (
        str(receipt["source_sha256"]) != expected_receipt["source_sha256"]
        or pd.isna(receipt["row_count"])
        or int(receipt["row_count"]) != expected_receipt["row_count"]
    ):
        raise ValueError(
            "The active Milestone 2 foundation source-row exclusion policy "
            "does not match the current governed policy"
        )

    aliases = read_existing_table(output_database, "player_aliases")
    spine = read_existing_table(output_database, "player_season_spine")
    sources = read_existing_table(output_database, "player_season_sources")
    if aliases.empty or spine.empty or sources.empty:
        raise ValueError("The active Milestone 2 foundation tables are missing")
    assert_no_source_row_exclusions(
        aliases,
        "active Milestone 2 player_aliases",
    )
    spine_run_ids = spine["run_id"].dropna().astype(str).unique()
    if set(spine_run_ids) != {foundation_run_id}:
        raise ValueError("The active spine does not match its Milestone 2 run")
    if (
        int(spine["season"].min()) != start_season
        or int(spine["season"].max()) != projection_through_season
    ):
        raise ValueError("The active spine does not match the requested seasons")
    return {
        "run_id": foundation_run_id,
        "source_observation_rows": len(sources),
    }


def _validate_feature_sources(
    projection_values: pd.DataFrame,
    market_values: pd.DataFrame,
    spine: pd.DataFrame,
) -> None:
    projection_key = ["player_key", "season", "provider"]
    market_key = ["player_key", "season", "source"]
    if projection_values.duplicated(projection_key).any():
        raise ValueError("Projection values contain duplicate provider rows")
    if market_values.duplicated(market_key).any():
        raise ValueError("Market values contain duplicate source rows")

    spine_keys = set(
        spine[["player_key", "season"]].itertuples(index=False, name=None)
    )
    for name, values in (
        ("projection", projection_values),
        ("market", market_values),
    ):
        value_keys = set(
            values[["player_key", "season"]].itertuples(
                index=False,
                name=None,
            )
        )
        missing = value_keys - spine_keys
        if missing:
            sample = sorted(missing, key=lambda key: (key[1], key[0]))[:5]
            raise ValueError(
                f"{name} values contain player-seasons outside the spine: "
                f"{sample}"
            )
        if values["run_id"].nunique() > 1:
            raise ValueError(f"{name} values contain mixed run IDs")


def _validate_feature_outputs(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    manifests: pd.DataFrame,
    source_audit: pd.DataFrame,
) -> None:
    keys = ["player_key", "season", "league"]
    expected = set(spine[keys].itertuples(index=False, name=None))
    actual = set(features[keys].itertuples(index=False, name=None))
    if expected != actual or len(features) != len(spine):
        raise ValueError("Feature mart keys do not exactly match the spine")

    target_columns = {
        "appeared",
        "opportunity_games",
        "observed_season_points",
        "unconditional_season_points",
        "conditional_ppg",
        "outcome_complete",
        "outcome_observed",
        "outcome_join_status",
        "active_target_available",
        "conditional_ppg_target_available",
        "conditional_ppg_training_eligible",
    }
    manifested = set(manifests["feature_name"])
    if manifested.intersection(target_columns):
        raise ValueError("A target column was included in a feature manifest")
    if not manifested.issubset(set(catalog["feature_name"])):
        raise ValueError("Feature manifest references an uncatalogued feature")
    if not set(catalog["feature_name"]).issubset(set(features.columns)):
        raise ValueError("Feature catalog references a missing mart column")

    duplicate_manifest = manifests.duplicated(
        ["manifest_name", "feature_name"]
    )
    if duplicate_manifest.any():
        raise ValueError("Feature manifests contain duplicate entries")
    template = manifests[
        manifests["manifest_name"].eq("template_challenger_v1")
    ]
    if len(template) > 12:
        raise ValueError("Template challenger exceeds the 12-feature budget")
    unique_budget = template.drop_duplicates("family")[
        "family_weight_budget"
    ]
    if unique_budget.notna().any() and not unique_budget.sum() <= 1.0 + 1e-9:
        raise ValueError("Template family weight budgets exceed one")

    if not source_audit.empty:
        rates = pd.to_numeric(source_audit["resolution_rate"], errors="coerce")
        if ((rates.dropna() < 0) | (rates.dropna() > 1)).any():
            raise ValueError("Source resolution rates must be between zero and one")
        excluded = pd.to_numeric(
            source_audit["excluded_rows"],
            errors="coerce",
        ).fillna(0)
        if excluded.lt(0).any():
            raise ValueError("Source exclusion counts cannot be negative")
        missing_metadata = excluded.gt(0) & source_audit[
            [
                "source_row_exclusion_ids",
                "source_row_exclusion_reasons",
                "source_row_exclusion_references",
            ]
        ].isna().any(axis=1)
        if missing_metadata.any():
            raise ValueError(
                "Excluded source rows require ID, reason, and reference metadata"
            )


def _source_manifest(
    source_database: Path,
    output_database: Path,
    run_id: str,
    source_audit: pd.DataFrame,
    spine_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in source_audit.itertuples(index=False):
        rows.append(
            {
                "run_id": run_id,
                "component": "feature_mart",
                "source_name": f"{row.source_kind}_{row.source_table}",
                "source_uri": (
                    f"sqlite://{source_database.resolve()}#{row.source_table}"
                ),
                "source_sha256": pd.NA,
                "row_count": int(row.input_rows),
            }
        )
        excluded_rows = (
            0 if pd.isna(row.excluded_rows) else int(row.excluded_rows)
        )
        if excluded_rows:
            rows.append(
                {
                    "run_id": run_id,
                    "component": "source_quarantine",
                    "source_name": str(row.source_row_exclusion_ids),
                    "source_uri": str(row.source_row_exclusion_references),
                    "source_sha256": pd.NA,
                    "row_count": excluded_rows,
                }
            )
    rows.append(
        {
            "run_id": run_id,
            "component": "feature_mart",
            "source_name": "foundation_player_season_spine",
            "source_uri": (
                f"sqlite://{output_database.resolve()}#player_season_spine"
            ),
            "source_sha256": pd.NA,
            "row_count": int(spine_rows),
        }
    )
    return align_columns(
        pd.DataFrame(rows),
        SOURCE_MANIFEST_COLUMNS,
        "source_manifest",
    )


def build_milestone_3(
    source_database: Path = SOURCE_DB_PATH,
    output_database: Path = OUTPUT_DB_PATH,
    start_season: int = START_SEASON,
    completed_through_season: int = COMPLETED_THROUGH_SEASON,
    projection_through_season: int = PROJECTION_THROUGH_SEASON,
    league: str = LEAGUE,
    useful_season_min_games: int = USEFUL_SEASON_MIN_GAMES,
    max_workers: int = 6,
    rebuild_foundation: bool = True,
) -> dict[str, object]:
    if start_season > completed_through_season:
        raise ValueError("start_season cannot be later than completed-through")
    if projection_through_season < completed_through_season:
        raise ValueError(
            "projection-through cannot be earlier than completed-through"
        )

    if rebuild_foundation:
        foundation = build_milestone_2(
            source_database=source_database,
            output_database=output_database,
            start_season=start_season,
            completed_through_season=completed_through_season,
            projection_through_season=projection_through_season,
            league=league,
            useful_season_min_games=useful_season_min_games,
            max_workers=max_workers,
        )
    else:
        foundation = _active_foundation(
            output_database,
            start_season,
            completed_through_season,
            projection_through_season,
            league,
            useful_season_min_games,
        )
    gc.collect()
    spine_run_id = str(foundation["run_id"])
    run_id = create_run_id("milestone_3")

    identity = read_existing_table(output_database, "player_identity")
    aliases = read_existing_table(output_database, "player_aliases")
    outcomes = read_existing_table(output_database, "player_season_outcomes")
    spine = read_existing_table(output_database, "player_season_spine")

    projection_values, projection_audit = build_projection_values(
        aliases,
        league=league,
        run_id=run_id,
        source_database=source_database,
        start_season=start_season,
        projection_through_season=projection_through_season,
    )
    # Release transient provider frames before the second wide source pass.
    # Repeated Windows/pandas builds can otherwise terminate inside the market
    # aggregation without a Python exception despite the small final tables.
    gc.collect()
    market_values, market_audit = build_market_values(
        aliases,
        run_id=run_id,
        source_database=source_database,
        start_season=start_season,
        projection_through_season=projection_through_season,
    )
    source_audit = pd.concat(
        [projection_audit, market_audit],
        ignore_index=True,
    )
    for column in FEATURE_SOURCE_AUDIT_COLUMNS:
        if column not in source_audit:
            source_audit[column] = pd.NA
    source_audit["run_id"] = run_id
    source_audit = align_columns(
        source_audit,
        FEATURE_SOURCE_AUDIT_COLUMNS,
        "feature_source_resolution_audit",
    )
    _validate_feature_sources(projection_values, market_values, spine)

    features, catalog, manifests, feature_audit, correlations = (
        build_feature_mart(
            spine,
            identity,
            outcomes,
            projection_values,
            market_values,
            run_id=run_id,
            spine_run_id=spine_run_id,
        )
    )
    _validate_feature_outputs(
        spine,
        features,
        catalog,
        manifests,
        source_audit,
    )

    manifest = _source_manifest(
        source_database,
        output_database,
        run_id,
        source_audit,
        len(spine),
    )
    build_run = align_columns(
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at_utc": utc_now(),
                    "component": "milestone_3",
                    "league": league,
                    "start_season": start_season,
                    "completed_through_season": completed_through_season,
                    "useful_season_min_games": useful_season_min_games,
                    "scoring_hash": scoring_hash(league),
                    "identity_rows": len(identity),
                    "alias_rows": len(aliases),
                    "outcome_rows": len(outcomes),
                    "source_observation_rows": foundation[
                        "source_observation_rows"
                    ],
                    "spine_rows": len(spine),
                    "projection_value_rows": len(projection_values),
                    "market_value_rows": len(market_values),
                    "feature_rows": len(features),
                    "feature_count": len(catalog),
                    "foundation_run_id": spine_run_id,
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
    prior_complete = (
        run_history["component"].eq("milestone_3")
        & run_history["status"].eq("complete")
        & run_history["run_id"].ne(run_id)
    )
    run_history.loc[prior_complete, "status"] = "superseded"

    publish_tables = {
        "player_season_projection_values": projection_values,
        "player_season_market_values": market_values,
        "player_season_features": features,
        "feature_catalog": catalog,
        "feature_manifests": manifests,
        "feature_audit": feature_audit,
        "feature_correlations": correlations,
        "feature_source_resolution_audit": source_audit,
        "source_manifest": source_history,
        "build_runs": run_history,
    }
    model_history = read_existing_table(
        output_database,
        "model_runs",
        MODEL_RUN_COLUMNS,
    )
    if not model_history.empty:
        stale_models = (
            model_history["status"].eq("complete")
            & model_history["feature_run_id"].ne(run_id)
        )
        model_history.loc[stale_models, "status"] = "superseded"
        publish_tables["model_runs"] = model_history

    publish_tables_atomic(output_database, publish_tables)

    manifest_counts = (
        manifests.groupby("manifest_name")["feature_name"].nunique().to_dict()
    )
    current = features["season"].eq(projection_through_season)
    resolved_inputs = int(source_audit["resolved_rows"].sum())
    total_inputs = int(source_audit["input_rows"].sum())
    excluded_inputs = int(source_audit["excluded_rows"].sum())
    return {
        "run_id": run_id,
        "foundation_run_id": spine_run_id,
        "output_database": str(output_database.resolve()),
        "projection_value_rows": len(projection_values),
        "market_value_rows": len(market_values),
        "feature_rows": len(features),
        "feature_count": len(catalog),
        "manifest_feature_counts": {
            str(name): int(count)
            for name, count in manifest_counts.items()
        },
        "high_correlation_pairs": len(correlations),
        "source_resolution_rate": (
            resolved_inputs / total_inputs if total_inputs else None
        ),
        "source_excluded_rows": excluded_inputs,
        "current_season_rows": int(current.sum()),
        "current_expert_consensus_rows": int(
            features.loc[current, "expert_points_median"].notna().sum()
        ),
        "current_adp_rows": int(
            features.loc[current, "adp_median"].notna().sum()
        ),
        "league": league,
        "scoring_hash": scoring_hash(league),
    }


def main() -> None:
    args = parse_args()
    result = build_milestone_3(
        source_database=args.source_db,
        output_database=args.output_db,
        start_season=args.start_season,
        completed_through_season=args.completed_through,
        projection_through_season=args.projection_through,
        league=args.league,
        useful_season_min_games=args.useful_season_min_games,
        max_workers=args.max_workers,
        rebuild_foundation=not args.reuse_foundation,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
