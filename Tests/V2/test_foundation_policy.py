import sqlite3

import pandas as pd
import pytest

from Scripts.V2.build_milestone_2 import _source_manifest
from Scripts.V2.build_milestone_3 import _active_foundation
from Scripts.V2.contracts import (
    scoring_hash,
    source_row_exclusion_policy_receipt,
)


FOUNDATION_RUN_ID = "milestone_2_fixture"


def _foundation_database(path, receipt_mode):
    build_runs = pd.DataFrame(
        [
            {
                "run_id": FOUNDATION_RUN_ID,
                "created_at_utc": "2026-07-29T00:00:00+00:00",
                "component": "milestone_2",
                "league": "dk",
                "start_season": 2015,
                "completed_through_season": 2025,
                "useful_season_min_games": 4,
                "scoring_hash": scoring_hash("dk"),
                "status": "complete",
            }
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "player_key": "clean-player",
                "source_table": "Fixture_Projections",
                "position": "WR",
                "season": 2025,
            }
        ]
    )
    spine = pd.DataFrame(
        [
            {"run_id": FOUNDATION_RUN_ID, "season": 2015},
            {"run_id": FOUNDATION_RUN_ID, "season": 2026},
        ]
    )
    sources = pd.DataFrame(
        [{"run_id": FOUNDATION_RUN_ID, "player_key": "clean-player"}]
    )
    receipt = source_row_exclusion_policy_receipt(FOUNDATION_RUN_ID)
    if receipt_mode == "stale":
        receipt["source_sha256"] = "stale-policy"

    with sqlite3.connect(path) as connection:
        build_runs.to_sql("build_runs", connection, index=False)
        aliases.to_sql("player_aliases", connection, index=False)
        spine.to_sql("player_season_spine", connection, index=False)
        sources.to_sql("player_season_sources", connection, index=False)
        if receipt_mode != "missing":
            pd.DataFrame([receipt]).to_sql(
                "source_manifest",
                connection,
                index=False,
            )


def test_milestone_2_manifest_records_source_exclusion_policy(tmp_path):
    source_database = tmp_path / "source.sqlite3"
    output_database = tmp_path / "output.sqlite3"
    manifest = _source_manifest(
        source_database,
        output_database,
        FOUNDATION_RUN_ID,
        start_season=2015,
        projection_through_season=2026,
        identity_rows=1,
        alias_rows=1,
        outcome_rows=1,
    )
    expected = source_row_exclusion_policy_receipt(FOUNDATION_RUN_ID)
    receipt = manifest[
        manifest["component"].eq(expected["component"])
        & manifest["source_name"].eq(expected["source_name"])
    ]

    assert len(receipt) == 1
    assert receipt.iloc[0]["source_sha256"] == expected["source_sha256"]
    assert receipt.iloc[0]["row_count"] == expected["row_count"]


def test_active_foundation_accepts_matching_source_exclusion_policy(tmp_path):
    database = tmp_path / "foundation.sqlite3"
    _foundation_database(database, "current")

    foundation = _active_foundation(
        database,
        start_season=2015,
        completed_through_season=2025,
        projection_through_season=2026,
        league="dk",
        useful_season_min_games=4,
    )

    assert foundation == {
        "run_id": FOUNDATION_RUN_ID,
        "source_observation_rows": 1,
    }


@pytest.mark.parametrize(
    ("receipt_mode", "message"),
    [
        ("missing", "no unique source-row exclusion policy receipt"),
        ("stale", "does not match the current governed policy"),
    ],
)
def test_active_foundation_rejects_stale_source_exclusion_policy(
    tmp_path,
    receipt_mode,
    message,
):
    database = tmp_path / "foundation.sqlite3"
    _foundation_database(database, receipt_mode)

    with pytest.raises(ValueError, match=message):
        _active_foundation(
            database,
            start_season=2015,
            completed_through_season=2025,
            projection_through_season=2026,
            league="dk",
            useful_season_min_games=4,
        )
