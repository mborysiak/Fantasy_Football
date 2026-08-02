import pandas as pd
import pytest

from Scripts.Data_Generation.fantasypros_projection_csv import (
    FANTASYPROS_PROJECTION_COLUMNS,
    build_fantasypros_projection_rows,
    normalize_fantasypros_projection_csv,
)


@pytest.mark.parametrize(
    ("position", "row", "expected"),
    [
        (
            "QB",
            {
                "Player": "Example Quarterback",
                "Team": "ARI",
                "ATT": 500,
                "CMP": 325,
                "YDS": "4,000",
                "TDS": 30,
                "INTS": 10,
                "ATT.1": 80,
                "YDS.1": 450,
                "TDS.1": 5,
                "FL": 2,
                "FPTS": 300,
            },
            {
                "fpros_pass_yds": 4000,
                "fpros_rush_yds": 450,
                "fpros_rec_yds": 0,
            },
        ),
        (
            "RB",
            {
                "Player": "Example Running Back",
                "Team": "ARI",
                "ATT": 220,
                "YDS": "1,050",
                "TDS": 8,
                "REC": 45,
                "YDS.1": 350,
                "TDS.1": 3,
                "FL": 1,
                "FPTS": 210,
            },
            {
                "fpros_rush_yds": 1050,
                "fpros_rec_yds": 350,
                "fpros_pass_yds": 0,
            },
        ),
        (
            "WR",
            {
                "Player": "Example Receiver",
                "Team": "ARI",
                "REC": 80,
                "YDS": "1,100",
                "TDS": 7,
                "ATT": 10,
                "YDS.1": 65,
                "TDS.1": 1,
                "FL": 1,
                "FPTS": 190,
            },
            {
                "fpros_rec_yds": 1100,
                "fpros_rush_yds": 65,
                "fpros_pass_yds": 0,
            },
        ),
        (
            "TE",
            {
                "Player": "Example Tight End",
                "Team": "ARI",
                "REC": 65,
                "YDS": 750,
                "TDS": 6,
                "FL": 1,
                "FPTS": 150,
            },
            {
                "fpros_rec_yds": 750,
                "fpros_rush_yds": 0,
                "fpros_pass_yds": 0,
            },
        ),
    ],
)
def test_position_exports_map_duplicate_headers_by_stat_family(
    position,
    row,
    expected,
):
    blank = {column: pd.NA for column in row}
    blank["Player"] = "\u00a0"
    frame = pd.DataFrame([blank, row])

    normalized = normalize_fantasypros_projection_csv(
        frame,
        position=position,
        year=2026,
        minimum_rows=1,
    )

    assert tuple(normalized.columns) == FANTASYPROS_PROJECTION_COLUMNS
    assert len(normalized) == 1
    assert normalized.iloc[0]["player"] == row["Player"]
    assert normalized.iloc[0]["pos"] == position
    assert normalized.iloc[0]["year"] == 2026
    for column, value in expected.items():
        assert normalized.iloc[0][column] == value


def test_default_position_floor_rejects_partial_export():
    frame = pd.DataFrame(
        [
            {
                "Player": "Partial Tight End",
                "REC": 1,
                "YDS": 10,
                "TDS": 0,
                "FL": 0,
                "FPTS": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="only 1 usable rows"):
        normalize_fantasypros_projection_csv(
            frame,
            position="TE",
            year=2026,
        )


def test_combined_loader_requires_all_four_exports():
    with pytest.raises(ValueError, match="TE"):
        build_fantasypros_projection_rows(
            {position: pd.DataFrame() for position in ("QB", "RB", "WR")},
            year=2026,
        )
