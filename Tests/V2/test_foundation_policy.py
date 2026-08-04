import sqlite3

import pandas as pd
import pytest

import Scripts.V2.contracts as contracts
from Scripts.V2.build_milestone_2 import _source_manifest
from Scripts.V2.build_milestone_3 import _active_foundation
from Scripts.V2.contracts import (
    scoring_hash,
    source_row_exclusion_policy_receipt,
    source_team_trust_policy_receipt,
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
                "team": "BUF",
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
    team_receipt = source_team_trust_policy_receipt(FOUNDATION_RUN_ID)
    if receipt_mode == "stale":
        receipt["source_sha256"] = "stale-policy"
    if receipt_mode == "stale_team":
        team_receipt["source_sha256"] = "stale-team-policy"

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
        if receipt_mode != "missing_team":
            pd.DataFrame([team_receipt]).to_sql(
                "source_manifest",
                connection,
                index=False,
                if_exists="append",
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

    expected_team = source_team_trust_policy_receipt(FOUNDATION_RUN_ID)
    team_receipt = manifest[
        manifest["component"].eq(expected_team["component"])
        & manifest["source_name"].eq(expected_team["source_name"])
    ]
    assert len(team_receipt) == 1
    assert (
        team_receipt.iloc[0]["source_sha256"]
        == expected_team["source_sha256"]
    )
    assert team_receipt.iloc[0]["row_count"] == expected_team["row_count"]


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


@pytest.mark.parametrize(
    ("receipt_mode", "message"),
    [
        ("missing_team", "no unique source-team trust policy receipt"),
        ("stale_team", "source-team trust policy does not match"),
    ],
)
def test_active_foundation_rejects_stale_source_team_policy(
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


def test_active_foundation_rejects_stale_team_alias_map(
    tmp_path,
    monkeypatch,
):
    database = tmp_path / "foundation.sqlite3"
    _foundation_database(database, "current")
    changed_map = {**contracts.TEAM_MAP, "JAX": "JAX"}
    monkeypatch.setattr(contracts, "TEAM_MAP", changed_map)

    with pytest.raises(
        ValueError,
        match="source-team trust policy does not match",
    ):
        _active_foundation(
            database,
            start_season=2015,
            completed_through_season=2025,
            projection_through_season=2026,
            league="dk",
            useful_season_min_games=4,
        )
