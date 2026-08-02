"""Normalize manually exported FantasyPros season projection CSVs."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


FANTASYPROS_PROJECTION_POSITIONS = ("QB", "RB", "WR", "TE")
FANTASYPROS_PROJECTION_METRICS = (
    "fpros_pass_att",
    "fpros_pass_cmp",
    "fpros_pass_yds",
    "fpros_pass_td",
    "fpros_pass_int",
    "fpros_rush_att",
    "fpros_rush_yds",
    "fpros_rush_td",
    "fpros_fum_lost",
    "fpros_proj_pts",
    "fpros_rec",
    "fpros_rec_yds",
    "fpros_rec_td",
)
FANTASYPROS_PROJECTION_COLUMNS = (
    "player",
    "pos",
    "year",
    *FANTASYPROS_PROJECTION_METRICS,
)

_MINIMUM_POSITION_ROWS = {
    "QB": 50,
    "RB": 80,
    "WR": 100,
    "TE": 60,
}

# FantasyPros reuses ATT/YDS/TDS headers for different stat families. Pandas
# suffixes the second occurrence with ``.1``, and its meaning depends on the
# position export, so each file needs an explicit mapping.
_POSITION_COLUMN_MAPS = {
    "QB": {
        "ATT": "fpros_pass_att",
        "CMP": "fpros_pass_cmp",
        "YDS": "fpros_pass_yds",
        "TDS": "fpros_pass_td",
        "INTS": "fpros_pass_int",
        "ATT.1": "fpros_rush_att",
        "YDS.1": "fpros_rush_yds",
        "TDS.1": "fpros_rush_td",
        "FL": "fpros_fum_lost",
        "FPTS": "fpros_proj_pts",
    },
    "RB": {
        "ATT": "fpros_rush_att",
        "YDS": "fpros_rush_yds",
        "TDS": "fpros_rush_td",
        "REC": "fpros_rec",
        "YDS.1": "fpros_rec_yds",
        "TDS.1": "fpros_rec_td",
        "FL": "fpros_fum_lost",
        "FPTS": "fpros_proj_pts",
    },
    "WR": {
        "REC": "fpros_rec",
        "YDS": "fpros_rec_yds",
        "TDS": "fpros_rec_td",
        "ATT": "fpros_rush_att",
        "YDS.1": "fpros_rush_yds",
        "TDS.1": "fpros_rush_td",
        "FL": "fpros_fum_lost",
        "FPTS": "fpros_proj_pts",
    },
    "TE": {
        "REC": "fpros_rec",
        "YDS": "fpros_rec_yds",
        "TDS": "fpros_rec_td",
        "FL": "fpros_fum_lost",
        "FPTS": "fpros_proj_pts",
    },
}


def fantasypros_projection_filename(position: str) -> str:
    position = str(position).upper()
    if position not in FANTASYPROS_PROJECTION_POSITIONS:
        raise ValueError(f"Unsupported FantasyPros position: {position}")
    return f"FantasyPros_Fantasy_Football_Projections_{position}.csv"


def _numeric(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "-": pd.NA, "—": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_fantasypros_projection_csv(
    frame: pd.DataFrame,
    *,
    position: str,
    year: int,
    minimum_rows: int | None = None,
) -> pd.DataFrame:
    """Normalize one position export to the existing SQLite table schema."""
    position = str(position).upper()
    if position not in FANTASYPROS_PROJECTION_POSITIONS:
        raise ValueError(f"Unsupported FantasyPros position: {position}")

    column_map = _POSITION_COLUMN_MAPS[position]
    required = {"Player", *column_map}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"FantasyPros {position} export is missing columns: {missing}"
        )

    players = (
        frame["Player"]
        .astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .replace("", pd.NA)
    )
    output = pd.DataFrame(
        {
            "player": players,
            "pos": position,
            "year": int(year),
        },
        index=frame.index,
    )
    for metric in FANTASYPROS_PROJECTION_METRICS:
        output[metric] = 0.0
    for source_column, target_column in column_map.items():
        output[target_column] = _numeric(frame[source_column])

    output = output.loc[output["player"].notna()].copy()
    output.loc[:, FANTASYPROS_PROJECTION_METRICS] = output.loc[
        :, FANTASYPROS_PROJECTION_METRICS
    ].fillna(0.0)
    output = output.loc[:, FANTASYPROS_PROJECTION_COLUMNS].round(2)

    row_floor = (
        _MINIMUM_POSITION_ROWS[position]
        if minimum_rows is None
        else int(minimum_rows)
    )
    if len(output) < row_floor:
        raise ValueError(
            f"FantasyPros {position} export has only {len(output)} usable rows; "
            f"expected at least {row_floor}"
        )
    if output["player"].duplicated().any():
        duplicates = sorted(
            output.loc[output["player"].duplicated(False), "player"].unique()
        )
        raise ValueError(
            f"FantasyPros {position} export has duplicate players: {duplicates}"
        )
    return output.reset_index(drop=True)


def build_fantasypros_projection_rows(
    position_frames: Mapping[str, pd.DataFrame],
    *,
    year: int,
) -> pd.DataFrame:
    """Combine all four required position exports and fail on partial input."""
    normalized_frames = {
        str(position).upper(): frame for position, frame in position_frames.items()
    }
    missing_positions = sorted(
        set(FANTASYPROS_PROJECTION_POSITIONS).difference(normalized_frames)
    )
    if missing_positions:
        raise ValueError(
            f"Missing FantasyPros projection exports: {missing_positions}"
        )

    output = pd.concat(
        [
            normalize_fantasypros_projection_csv(
                normalized_frames[position],
                position=position,
                year=year,
            )
            for position in FANTASYPROS_PROJECTION_POSITIONS
        ],
        ignore_index=True,
    )
    duplicate_keys = output.duplicated(["player", "pos"], keep=False)
    if duplicate_keys.any():
        duplicates = output.loc[duplicate_keys, ["player", "pos"]].to_dict(
            "records"
        )
        raise ValueError(
            f"FantasyPros combined projections have duplicate keys: {duplicates}"
        )
    return output.loc[:, FANTASYPROS_PROJECTION_COLUMNS].reset_index(drop=True)
