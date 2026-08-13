import pandas as pd
import pytest

from Scripts.Data_Generation.fantasypoints_projection_csv import (
    FANTASYPOINTS_PROJECTION_COLUMNS,
    normalize_fantasypoints_projection_csv,
)


def _projection_values():
    return {
        "ADP": 12.3,
        "FPTS": 250.5,
        "FPTS/G": 16.7,
        "TIER": 2,
        "ATT": 450,
        "CMP": 300,
        "YDS": "3,500",
        "TD": 25,
        "INT": 10,
        "ATT.1": 80,
        "YDS.1": 450,
        "TD.1": 5,
        "REC": 0,
        "YDS.2": 0,
        "TD.2": 0,
    }


def test_current_two_header_export_maps_to_existing_database_schema():
    row = {
        "RANK": 10,
        "NAME": "Example Quarterback",
        "Position": "QB",
        "Team": "buf",
        "GP": 15,
        "POS": 3,
        **_projection_values(),
    }

    normalized = normalize_fantasypoints_projection_csv(
        pd.DataFrame([row]),
        year=2026,
        minimum_rows=1,
        minimum_position_rows={"QB": 1},
    )

    assert tuple(normalized.columns) == FANTASYPOINTS_PROJECTION_COLUMNS
    assert normalized.iloc[0]["player"] == "Example Quarterback"
    assert normalized.iloc[0]["pos"] == "QB"
    assert normalized.iloc[0]["team"] == "BUF"
    assert normalized.iloc[0]["year"] == 2026
    assert normalized.iloc[0]["fpts_games"] == 15
    assert normalized.iloc[0]["fpts_pass_yds"] == 3500
    assert "POS" not in normalized.columns


def test_legacy_export_remains_supported_and_retired_columns_are_ignored():
    row = {
        "RK": 10,
        "Name": "Example Quarterback",
        "POS": "QB",
        "Team": "BUF",
        "G": 15,
        "POS.1": "QB3",
        "UP": "-",
        "DOWN": "-",
        "MOVE": "-",
        "TARGET": "-",
        "WIN": "-",
        **_projection_values(),
    }

    normalized = normalize_fantasypoints_projection_csv(
        pd.DataFrame([row]),
        year=2026,
        minimum_rows=1,
        minimum_position_rows={"QB": 1},
    )

    assert tuple(normalized.columns) == FANTASYPOINTS_PROJECTION_COLUMNS
    assert normalized.iloc[0]["fpts_overall_rank"] == 10
    assert normalized.iloc[0]["fpts_games"] == 15


def test_missing_current_projection_field_fails_closed():
    row = {
        "RANK": 10,
        "NAME": "Example Quarterback",
        "Position": "QB",
        "Team": "BUF",
        "GP": 15,
        **_projection_values(),
    }
    del row["TD.2"]

    with pytest.raises(ValueError, match="missing columns.*TD.2"):
        normalize_fantasypoints_projection_csv(
            pd.DataFrame([row]),
            year=2026,
            minimum_rows=1,
            minimum_position_rows={"QB": 1},
        )


def test_default_depth_floor_rejects_partial_export():
    row = {
        "RANK": 10,
        "NAME": "Example Quarterback",
        "Position": "QB",
        "Team": "BUF",
        "GP": 15,
        **_projection_values(),
    }

    with pytest.raises(ValueError, match="only 1 rows"):
        normalize_fantasypoints_projection_csv(
            pd.DataFrame([row]),
            year=2026,
        )
