from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from Scripts.V2.refresh_dk_adp import (
    ADP_COLUMNS,
    build_dk_adp_rows,
    parse_dk_payload,
    replace_current_dk_rows,
)


def nffc_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Amon Ra St Brown",
                "pos": "WR",
                "year": 2026,
                "avg_pick": 10.0,
                "min_pick": 5.0,
                "max_pick": 20.0,
                "std_dev": 3.0,
                "league": "nffc",
            },
            {
                "player": "Tet Mcmillan",
                "pos": "WR",
                "year": 2026,
                "avg_pick": 40.0,
                "min_pick": 20.0,
                "max_pick": 60.0,
                "std_dev": 8.0,
                "league": "nffc",
            },
            {
                "player": "BUF",
                "pos": "TDSP",
                "year": 2026,
                "avg_pick": 200.0,
                "min_pick": 180.0,
                "max_pick": 220.0,
                "std_dev": 8.0,
                "league": "nffc",
            },
        ]
    )


def test_parse_and_build_dk_rows_rescales_complete_distribution():
    api = parse_dk_payload(
        {
            "adps": [
                {
                    "player_name": "Amon Ra St. Brown",
                    "pos": "WR",
                    "curr_adp": 8.0,
                },
                {
                    "player_name": "Tetairoa McMillan",
                    "pos": "WR",
                    "curr_adp": 50.0,
                },
            ]
        }
    )

    result = build_dk_adp_rows(
        nffc_rows(),
        api,
        year=2026,
        min_depth=2,
    )

    assert list(result["player"]) == ["Amon Ra St Brown", "Tet Mcmillan"]
    amon = result.loc[result["player"].eq("Amon Ra St Brown")].iloc[0]
    tet = result.loc[result["player"].eq("Tet Mcmillan")].iloc[0]
    assert amon["avg_pick"] == pytest.approx(8.0)
    assert amon["min_pick"] == pytest.approx(4.0)
    assert amon["max_pick"] == pytest.approx(16.0)
    assert amon["std_dev"] == pytest.approx((16.0 - 4.0) / 5.0)
    assert tet["avg_pick"] == pytest.approx(50.0)
    assert tet["min_pick"] == pytest.approx(25.0)
    assert tet["max_pick"] == pytest.approx(75.0)
    assert tet["std_dev"] == pytest.approx(10.0)
    assert set(result["pos"]) == {"WR"}
    assert set(result["league"]) == {"dk"}


def test_build_retains_api_player_without_nffc_distribution():
    api = pd.DataFrame(
        [{"player": "Unknown Player", "pos": "RB", "pick_dk": 100.0}]
    )
    result = build_dk_adp_rows(
        nffc_rows(),
        api,
        year=2026,
        min_depth=1,
    )

    row = result.iloc[0]
    assert row["player"] == "Unknown Player"
    assert row["pos"] == "RB"
    assert row["avg_pick"] == pytest.approx(100.0)
    assert row["min_pick"] == pytest.approx(80.0)
    assert row["max_pick"] == pytest.approx(120.0)
    assert row["std_dev"] == pytest.approx(20.0)
    assert result.attrs["nffc_distribution_fallback_count"] == 1


def test_build_repairs_sparse_nffc_bounds_before_publication():
    sparse = nffc_rows().iloc[[0]].copy()
    sparse.loc[:, "avg_pick"] = 300.0
    sparse.loc[:, "min_pick"] = 320.0
    sparse.loc[:, "max_pick"] = 250.0
    api = pd.DataFrame(
        [{"player": "Amon Ra St Brown", "pos": "WR", "pick_dk": 100.0}]
    )

    result = build_dk_adp_rows(
        sparse,
        api,
        year=2026,
        min_depth=1,
    )

    row = result.iloc[0]
    assert row["min_pick"] == pytest.approx(80.0)
    assert row["avg_pick"] == pytest.approx(100.0)
    assert row["max_pick"] == pytest.approx(120.0)
    assert row["std_dev"] == pytest.approx(8.0)
    assert result.attrs["repaired_lower_bound_count"] == 1
    assert result.attrs["repaired_upper_bound_count"] == 1


def test_replace_current_dk_rows_changes_only_target_slice(tmp_path):
    path = tmp_path / "Season_Stats_New.sqlite3"
    initial = pd.concat(
        [
            nffc_rows(),
            pd.DataFrame(
                [
                    {
                        "player": "Old DK Player",
                        "pos": "WR",
                        "year": 2026,
                        "avg_pick": 1.0,
                        "min_pick": 1.0,
                        "max_pick": 2.0,
                        "std_dev": 0.2,
                        "league": "dk",
                    },
                    {
                        "player": "Historical DK Player",
                        "pos": "WR",
                        "year": 2025,
                        "avg_pick": 2.0,
                        "min_pick": 1.0,
                        "max_pick": 3.0,
                        "std_dev": 0.4,
                        "league": "dk",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    with sqlite3.connect(path) as connection:
        initial.to_sql("ADP_Averages", connection, index=False)

    api = pd.DataFrame(
        [
            {"player": "Amon Ra St Brown", "pos": "WR", "pick_dk": 8.0},
            {"player": "Tet Mcmillan", "pos": "WR", "pick_dk": 50.0},
        ]
    )
    current = build_dk_adp_rows(
        nffc_rows(),
        api,
        year=2026,
        min_depth=2,
    )
    replace_current_dk_rows(path, current, year=2026)

    with sqlite3.connect(path) as connection:
        stored = pd.read_sql_query(
            "SELECT * FROM ADP_Averages ORDER BY year, league, avg_pick",
            connection,
        )

    assert list(stored.columns) == list(ADP_COLUMNS)
    assert len(stored.loc[stored["league"].eq("nffc")]) == 3
    assert (
        stored.loc[
            stored["player"].eq("Historical DK Player"), "year"
        ].squeeze()
        == 2025
    )
    current_stored = stored.loc[
        stored["year"].eq(2026) & stored["league"].eq("dk")
    ]
    assert set(current_stored["player"]) == {
        "Amon Ra St Brown",
        "Tet Mcmillan",
    }
    assert np.allclose(
        current_stored["std_dev"],
        (current_stored["max_pick"] - current_stored["min_pick"]) / 5.0,
    )
