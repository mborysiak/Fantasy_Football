"""Validate and atomically replace provider slices in ``ADP_Ranks``."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


ADP_RANK_COLUMNS = ("player", "pick", "pos", "year", "source")
OFFENSIVE_ADP_POSITIONS = ("QB", "RB", "WR", "TE")
FANTASYPROS_ADP_POSITIONS = (*OFFENSIVE_ADP_POSITIONS, "DST")

MFL_MINIMUM_ROWS = {
    "QB": 25,
    "RB": 60,
    "WR": 70,
    "TE": 25,
}
FANTASYPROS_MINIMUM_ROWS = {
    "QB": 30,
    "RB": 70,
    "WR": 80,
    "TE": 30,
    "DST": 20,
}


def validate_adp_rank_slice(
    rows: pd.DataFrame,
    *,
    year: int,
    source: str,
    allowed_positions: Sequence[str],
    minimum_rows_by_position: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Return a normalized slice after fail-closed schema and depth checks."""

    missing = sorted(set(ADP_RANK_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"ADP_Ranks input is missing columns: {missing}")

    expected_year = int(year)
    expected_source = str(source).strip().lower()
    allowed = tuple(str(position).strip().upper() for position in allowed_positions)
    if not allowed:
        raise ValueError("ADP_Ranks replacement requires allowed positions")

    output = rows.loc[:, list(ADP_RANK_COLUMNS)].copy()
    output["player"] = output["player"].astype("string").str.strip()
    output["pick"] = pd.to_numeric(output["pick"], errors="coerce")
    output["pos"] = output["pos"].astype("string").str.strip().str.upper()
    output["year"] = pd.to_numeric(output["year"], errors="coerce")
    output["source"] = output["source"].astype("string").str.strip().str.lower()

    invalid_values = (
        output["player"].isna()
        | output["player"].eq("")
        | output["pick"].isna()
        | ~np.isfinite(output["pick"])
        | output["pick"].le(0)
        | output["pos"].isna()
        | ~output["pos"].isin(allowed)
        | output["year"].isna()
        | ~output["year"].eq(expected_year)
        | output["source"].isna()
        | ~output["source"].eq(expected_source)
    )
    if invalid_values.any():
        invalid = output.loc[
            invalid_values,
            ["player", "pick", "pos", "year", "source"],
        ]
        raise ValueError(
            "ADP_Ranks replacement contains invalid or out-of-slice rows: "
            f"{invalid.head(20).to_dict('records')}"
        )

    duplicate_keys = output.duplicated(
        ["player", "pos", "year", "source"],
        keep=False,
    )
    if duplicate_keys.any():
        duplicates = output.loc[
            duplicate_keys,
            ["player", "pos", "year", "source"],
        ]
        raise ValueError(
            "ADP_Ranks replacement contains duplicate provider keys: "
            f"{duplicates.head(20).to_dict('records')}"
        )

    minimums = {
        str(position).strip().upper(): int(count)
        for position, count in (minimum_rows_by_position or {}).items()
    }
    unknown_minimums = sorted(set(minimums).difference(allowed))
    if unknown_minimums:
        raise ValueError(
            "ADP_Ranks row floors contain unsupported positions: " f"{unknown_minimums}"
        )
    position_counts = output.groupby("pos").size().to_dict()
    shallow = {
        position: {
            "observed": int(position_counts.get(position, 0)),
            "minimum": minimum,
        }
        for position, minimum in minimums.items()
        if int(position_counts.get(position, 0)) < minimum
    }
    if shallow:
        raise ValueError(f"ADP_Ranks replacement is too shallow: {shallow}")

    output["year"] = output["year"].astype(int)
    return output.sort_values(["pos", "pick", "player"]).reset_index(drop=True)


def replace_adp_rank_slice(
    db_path: str | Path,
    rows: pd.DataFrame,
    *,
    year: int,
    source: str,
    allowed_positions: Sequence[str],
    minimum_rows_by_position: Mapping[str, int] | None = None,
    position: str | None = None,
) -> None:
    """Replace one provider/year slice in a single SQLite transaction."""

    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ADP source database does not exist: {path}")

    expected = validate_adp_rank_slice(
        rows,
        year=year,
        source=source,
        allowed_positions=allowed_positions,
        minimum_rows_by_position=minimum_rows_by_position,
    )
    expected_source = str(source).strip().lower()
    expected_position = str(position).strip().upper() if position is not None else None
    if expected_position is not None:
        observed_positions = set(expected["pos"])
        if observed_positions != {expected_position}:
            raise ValueError(
                "Position-scoped ADP_Ranks replacement contains other positions: "
                f"expected {expected_position}, observed {sorted(observed_positions)}"
            )

    values = [
        tuple(value.item() if isinstance(value, np.generic) else value for value in row)
        for row in expected.itertuples(index=False, name=None)
    ]
    with sqlite3.connect(path, timeout=45) as connection:
        table_info = connection.execute('PRAGMA table_info("ADP_Ranks")').fetchall()
        available = {str(row[1]) for row in table_info}
        missing = sorted(set(ADP_RANK_COLUMNS).difference(available))
        if missing:
            raise ValueError(f"ADP_Ranks is missing required columns: {missing}")

        connection.execute("PRAGMA busy_timeout=45000")
        connection.execute("BEGIN IMMEDIATE")
        try:
            delete_sql = """
                DELETE FROM ADP_Ranks
                WHERE CAST(year AS INTEGER)=?
                  AND LOWER(source)=?
            """
            parameters: tuple[object, ...] = (int(year), expected_source)
            if expected_position is not None:
                delete_sql += " AND UPPER(pos)=?"
                parameters += (expected_position,)
            connection.execute(delete_sql, parameters)
            connection.executemany(
                """
                INSERT INTO ADP_Ranks (player, pick, pos, year, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                values,
            )

            count_sql = """
                SELECT COUNT(*)
                FROM ADP_Ranks
                WHERE CAST(year AS INTEGER)=?
                  AND LOWER(source)=?
            """
            if expected_position is not None:
                count_sql += " AND UPPER(pos)=?"
            stored_count = connection.execute(
                count_sql,
                parameters,
            ).fetchone()[0]
            if int(stored_count) != len(expected):
                raise RuntimeError(
                    "ADP_Ranks replacement count mismatch: "
                    f"expected {len(expected)}, stored {stored_count}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
