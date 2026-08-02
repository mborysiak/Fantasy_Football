import sqlite3

import pandas as pd
import pytest

from Scripts.Data_Generation.adp_rank_ingest import replace_adp_rank_slice


ADP_COLUMNS = ["player", "pick", "pos", "year", "source"]


def _create_database(path, rows, *, player_check=None):
    check = f" CHECK (player != '{player_check}')" if player_check else ""
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"""
            CREATE TABLE ADP_Ranks (
                player TEXT{check},
                pick REAL,
                pos TEXT,
                year INTEGER,
                source TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO ADP_Ranks VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def _stored(path):
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM ADP_Ranks ORDER BY year, source, pos, player, pick",
            connection,
        )


def test_mfl_position_replacement_preserves_fpros_and_other_positions(tmp_path):
    path = tmp_path / "Season_Stats_New.sqlite3"
    _create_database(
        path,
        [
            ("Old MFL QB", 10.0, "QB", 2026, "mfl"),
            ("MFL RB", 20.0, "RB", 2026, "mfl"),
            ("FantasyPros QB", 30.0, "QB", 2026, "fpros"),
        ],
    )
    replacement = pd.DataFrame(
        [("New MFL QB", 11.0, "QB", 2026, "mfl")],
        columns=ADP_COLUMNS,
    )

    replace_adp_rank_slice(
        path,
        replacement,
        year=2026,
        source="mfl",
        position="QB",
        allowed_positions=("QB",),
        minimum_rows_by_position={"QB": 1},
    )

    stored = _stored(path)
    assert set(stored["player"]) == {
        "New MFL QB",
        "MFL RB",
        "FantasyPros QB",
    }


def test_fpros_year_replacement_is_idempotent_and_removes_stale_rows(tmp_path):
    path = tmp_path / "Season_Stats_New.sqlite3"
    _create_database(
        path,
        [
            ("MFL QB", 10.0, "QB", 2026, "mfl"),
            ("Old FantasyPros QB", 20.0, "QB", 2026, "fpros"),
            ("Duplicate DST", 200.0, "DST", 2026, "fpros"),
            ("Duplicate DST", 205.0, "DST", 2026, "fpros"),
            ("Legacy Defense", 210.0, "DS", 2026, "fpros"),
            ("Legacy Kicker", 220.0, "K1", 2026, "fpros"),
        ],
    )
    replacement = pd.DataFrame(
        [
            ("FantasyPros QB", 11.0, "QB", 2026, "fpros"),
            ("FantasyPros RB", 12.0, "RB", 2026, "fpros"),
            ("FantasyPros WR", 13.0, "WR", 2026, "fpros"),
            ("FantasyPros TE", 14.0, "TE", 2026, "fpros"),
            ("FantasyPros DST", 15.0, "DST", 2026, "fpros"),
        ],
        columns=ADP_COLUMNS,
    )
    kwargs = {
        "year": 2026,
        "source": "fpros",
        "allowed_positions": ("QB", "RB", "WR", "TE", "DST"),
        "minimum_rows_by_position": {
            "QB": 1,
            "RB": 1,
            "WR": 1,
            "TE": 1,
            "DST": 1,
        },
    }

    replace_adp_rank_slice(path, replacement, **kwargs)
    replace_adp_rank_slice(path, replacement, **kwargs)

    stored = _stored(path)
    fpros = stored.loc[stored["source"].eq("fpros")]
    assert len(fpros) == 5
    assert set(fpros["pos"]) == {"QB", "RB", "WR", "TE", "DST"}
    assert stored.loc[stored["source"].eq("mfl"), "player"].tolist() == ["MFL QB"]


def test_shallow_slice_is_rejected_before_existing_rows_are_deleted(tmp_path):
    path = tmp_path / "Season_Stats_New.sqlite3"
    initial = [("Existing QB", 10.0, "QB", 2026, "fpros")]
    _create_database(path, initial)
    replacement = pd.DataFrame(
        [("Only QB", 11.0, "QB", 2026, "fpros")],
        columns=ADP_COLUMNS,
    )

    with pytest.raises(ValueError, match="too shallow"):
        replace_adp_rank_slice(
            path,
            replacement,
            year=2026,
            source="fpros",
            allowed_positions=("QB",),
            minimum_rows_by_position={"QB": 2},
        )

    assert _stored(path).itertuples(index=False, name=None).__next__() == initial[0]


def test_failed_insert_rolls_back_the_deleted_slice(tmp_path):
    path = tmp_path / "Season_Stats_New.sqlite3"
    initial = [("Existing QB", 10.0, "QB", 2026, "fpros")]
    _create_database(path, initial, player_check="Rejected QB")
    replacement = pd.DataFrame(
        [("Rejected QB", 11.0, "QB", 2026, "fpros")],
        columns=ADP_COLUMNS,
    )

    with pytest.raises(sqlite3.IntegrityError):
        replace_adp_rank_slice(
            path,
            replacement,
            year=2026,
            source="fpros",
            allowed_positions=("QB",),
            minimum_rows_by_position={"QB": 1},
        )

    assert _stored(path).itertuples(index=False, name=None).__next__() == initial[0]
