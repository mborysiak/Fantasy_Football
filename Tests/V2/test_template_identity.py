import sqlite3

import pandas as pd
import pytest

from Scripts.V2.template_identity import attach_v2_player_keys


def _identity_database(path):
    identity = pd.DataFrame(
        [
            {
                "player_key": "older",
                "normalized_name": "same player",
                "position": "WR",
                "rookie_season": 2001,
                "last_season": 2016,
                "draft_year": 2001,
                "draft_team": "CAR",
                "latest_team": "BAL",
                "identity_status": "confirmed",
            },
            {
                "player_key": "newer",
                "normalized_name": "same player",
                "position": "WR",
                "rookie_season": 2007,
                "last_season": 2012,
                "draft_year": 2007,
                "draft_team": "NYG",
                "latest_team": "LAR",
                "identity_status": "confirmed",
            },
            {
                "player_key": "redundant-provisional",
                "normalized_name": "same player",
                "position": "WR",
                "rookie_season": pd.NA,
                "last_season": pd.NA,
                "draft_year": pd.NA,
                "draft_team": "CAR",
                "latest_team": "CAR",
                "identity_status": "provisional",
            },
            {
                "player_key": "rookie-provisional",
                "normalized_name": "future rookie",
                "position": "RB",
                "rookie_season": 2026,
                "last_season": pd.NA,
                "draft_year": 2026,
                "draft_team": "TB",
                "latest_team": "TB",
                "identity_status": "provisional",
            },
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "player_key": "older",
                "normalized_name": "same player",
                "position": "WR",
                "team": "CAR",
                "season": 2008,
                "source_table": "Fixture_Projections",
            },
            {
                "player_key": "newer",
                "normalized_name": "same player",
                "position": "WR",
                "team": "LAR",
                "season": 2008,
                "source_table": "Fixture_Projections",
            },
            {
                "player_key": "redundant-provisional",
                "normalized_name": "same player",
                "position": "WR",
                "team": "CAR",
                "season": 2008,
                "source_table": "Fixture_Projections",
            },
            {
                "player_key": "rookie-provisional",
                "normalized_name": "future rookie",
                "position": "RB",
                "team": "TB",
                "season": 2026,
                "source_table": "Fixture_Projections",
            },
        ]
    )
    with sqlite3.connect(path) as connection:
        identity.to_sql("player_identity", connection, index=False)
        aliases.to_sql("player_aliases", connection, index=False)


def test_template_identity_prefers_confirmed_and_uses_team(tmp_path):
    database = tmp_path / "identity.sqlite3"
    _identity_database(database)
    frame = pd.DataFrame(
        [
            {
                "player": "Same Player",
                "pos": "WR",
                "team": "CAR",
                "season": 2008,
            },
            {
                "player": "Same Player",
                "pos": "WR",
                "team": "LAR",
                "season": 2008,
            },
        ]
    )
    resolved = attach_v2_player_keys(frame, database)
    assert resolved["player_key"].tolist() == ["older", "newer"]
    assert set(resolved["player_key_match_method"]) == {
        "alias_team_confirmed_unique"
    }


def test_template_identity_retains_preplay_provisional_key(tmp_path):
    database = tmp_path / "identity.sqlite3"
    _identity_database(database)
    frame = pd.DataFrame(
        [
            {
                "player": "Future Rookie",
                "pos": "RB",
                "team": "TB",
                "season": 2026,
            }
        ]
    )
    resolved = attach_v2_player_keys(frame, database)
    assert resolved.loc[0, "player_key"] == "rookie-provisional"
    assert resolved.loc[0, "player_key_match_method"] == "alias_unique"


def test_template_identity_quarantines_stale_fftoday_alias(tmp_path):
    database = tmp_path / "identity.sqlite3"
    _identity_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO player_identity (
                player_key, normalized_name, position, rookie_season,
                last_season, draft_year, draft_team, latest_team,
                identity_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "stale-provisional",
                "stale quarterback",
                "QB",
                None,
                None,
                None,
                "LAR",
                "LAR",
                "provisional",
            ),
        )
        connection.execute(
            """
            INSERT INTO player_aliases (
                player_key, normalized_name, position, team, season,
                source_table
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "stale-provisional",
                "stale quarterback",
                "QB",
                "LAR",
                2018,
                "FFToday_Projections",
            ),
        )

    frame = pd.DataFrame(
        [
            {
                "player": "Stale Quarterback",
                "pos": "QB",
                "team": "LAR",
                "season": 2018,
            }
        ]
    )
    resolved = attach_v2_player_keys(
        frame,
        database,
        require_complete=False,
    )

    assert pd.isna(resolved.loc[0, "player_key"])
    assert resolved.loc[0, "player_key_match_method"] == (
        "unresolved_ambiguous_identity"
    )


def test_template_identity_rejects_aliases_without_source_provenance(tmp_path):
    database = tmp_path / "identity.sqlite3"
    _identity_database(database)
    with sqlite3.connect(database) as connection:
        aliases = pd.read_sql_query(
            """
            SELECT player_key, normalized_name, position, team, season
            FROM player_aliases
            """,
            connection,
        )
        aliases.to_sql(
            "player_aliases",
            connection,
            if_exists="replace",
            index=False,
        )

    frame = pd.DataFrame(
        [
            {
                "player": "Same Player",
                "pos": "WR",
                "team": "CAR",
                "season": 2008,
            }
        ]
    )
    with pytest.raises(ValueError, match="source provenance"):
        attach_v2_player_keys(frame, database)
