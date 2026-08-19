import pandas as pd
import pytest

from Scripts.Data_Generation.ftn_projection_csv import (
    FTN_PROJECTION_COLUMNS,
    ftn_projection_filename,
    normalize_ftn_projection_csv,
)


def _projection_row(**overrides):
    row = {
        "Player": "Example Runner",
        "Position": "RB",
        "Team": "buf",
        "Auction": "$42.50",
        "Opp.": None,
        "PaCom": 0,
        "PaAtt": 0,
        "PaYds": 0,
        "PaTD": 0,
        "PaINT": 0,
        "RuAtt": 250,
        "RuYds": "1,250",
        "RuTD": 10,
        "Fum.": None,
        "Tar": 70,
        "Rec": 55,
        "ReYds": 450,
        "ReTD": 3,
        "FPTS": 260,
    }
    row.update(overrides)
    return row


def test_grouped_header_export_maps_to_ftn_database_schema():
    normalized = normalize_ftn_projection_csv(
        pd.DataFrame([_projection_row()]),
        year=2026,
        minimum_rows=1,
        minimum_position_rows={"RB": 1},
    )

    assert tuple(normalized.columns) == FTN_PROJECTION_COLUMNS
    assert normalized.iloc[0]["player"] == "Example Runner"
    assert normalized.iloc[0]["pos"] == "RB"
    assert normalized.iloc[0]["team"] == "BUF"
    assert normalized.iloc[0]["year"] == 2026
    assert normalized.iloc[0]["ftn_auction_value"] == 42.5
    assert normalized.iloc[0]["ftn_rush_yds"] == 1250
    assert "Opp." not in normalized.columns
    assert "Fum." not in normalized.columns


def test_literal_filename_is_year_specific():
    assert ftn_projection_filename(2026) == (
        "NFL Fantasy Football Player Projections (2026 Season).csv"
    )


def test_missing_projection_field_fails_closed():
    row = _projection_row()
    del row["ReTD"]

    with pytest.raises(ValueError, match="missing columns.*ReTD"):
        normalize_ftn_projection_csv(
            pd.DataFrame([row]),
            year=2026,
            minimum_rows=1,
            minimum_position_rows={"RB": 1},
        )


def test_missing_numeric_projection_fails_closed():
    with pytest.raises(ValueError, match="missing or non-numeric"):
        normalize_ftn_projection_csv(
            pd.DataFrame([_projection_row(RuYds=None)]),
            year=2026,
            minimum_rows=1,
            minimum_position_rows={"RB": 1},
        )


def test_duplicate_player_position_fails_closed():
    frame = pd.DataFrame([_projection_row(), _projection_row()])

    with pytest.raises(ValueError, match="duplicate player-position"):
        normalize_ftn_projection_csv(
            frame,
            year=2026,
            minimum_rows=1,
            minimum_position_rows={"RB": 1},
        )


def test_default_depth_floor_rejects_partial_export():
    with pytest.raises(ValueError, match="only 1 rows"):
        normalize_ftn_projection_csv(
            pd.DataFrame([_projection_row()]),
            year=2026,
        )
