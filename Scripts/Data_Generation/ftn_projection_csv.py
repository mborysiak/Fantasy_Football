"""Normalize FTN season projection CSV exports."""

from __future__ import annotations

import numpy as np
import pandas as pd


FTN_PROJECTION_POSITIONS = ("QB", "RB", "WR", "TE")
FTN_PROJECTION_METRICS = (
    "ftn_auction_value",
    "ftn_proj_pts",
    "ftn_pass_comp",
    "ftn_pass_att",
    "ftn_pass_yds",
    "ftn_pass_td",
    "ftn_pass_int",
    "ftn_rush_att",
    "ftn_rush_yds",
    "ftn_rush_td",
    "ftn_rec_targets",
    "ftn_rec",
    "ftn_rec_yds",
    "ftn_rec_td",
)
FTN_PROJECTION_COLUMNS = (
    "player",
    "pos",
    "team",
    "year",
    *FTN_PROJECTION_METRICS,
)

_MINIMUM_ROWS = 100
_MINIMUM_POSITION_ROWS = {
    "QB": 10,
    "RB": 40,
    "WR": 45,
    "TE": 8,
}

_COLUMN_MAP = {
    "Player": "player",
    "Position": "pos",
    "Team": "team",
    "Auction": "ftn_auction_value",
    "FPTS": "ftn_proj_pts",
    "PaCom": "ftn_pass_comp",
    "PaAtt": "ftn_pass_att",
    "PaYds": "ftn_pass_yds",
    "PaTD": "ftn_pass_td",
    "PaINT": "ftn_pass_int",
    "RuAtt": "ftn_rush_att",
    "RuYds": "ftn_rush_yds",
    "RuTD": "ftn_rush_td",
    "Tar": "ftn_rec_targets",
    "Rec": "ftn_rec",
    "ReYds": "ftn_rec_yds",
    "ReTD": "ftn_rec_td",
}


def ftn_projection_filename(year: int) -> str:
    """Return the literal browser filename for an FTN season export."""

    return f"NFL Fantasy Football Player Projections ({int(year)} Season).csv"


def _numeric(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
        .replace({"-": "0", "": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_ftn_projection_csv(
    frame: pd.DataFrame,
    *,
    year: int,
    minimum_rows: int = _MINIMUM_ROWS,
    minimum_position_rows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Normalize FTN's grouped-header export to the SQLite source schema."""

    missing = sorted(set(_COLUMN_MAP).difference(frame.columns))
    if missing:
        raise ValueError(
            f"FTN season projection export is missing columns: {missing}; "
            f"available columns are {list(frame.columns)}"
        )

    output = frame.loc[:, list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP).copy()
    output["player"] = (
        output["player"]
        .astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .replace("", pd.NA)
    )
    output["pos"] = output["pos"].astype("string").str.strip().str.upper()
    output["team"] = output["team"].astype("string").str.strip().str.upper()

    for column in FTN_PROJECTION_METRICS:
        output[column] = _numeric(output[column])
    output["year"] = int(year)
    output = output.loc[:, FTN_PROJECTION_COLUMNS]

    invalid_identity = (
        output["player"].isna()
        | output["team"].isna()
        | output["team"].eq("")
        | ~output["pos"].isin(FTN_PROJECTION_POSITIONS)
    )
    if invalid_identity.any():
        invalid = output.loc[
            invalid_identity,
            ["player", "pos", "team"],
        ]
        raise ValueError(
            "FTN export contains invalid player identities: "
            f"{invalid.head(20).to_dict('records')}"
        )

    numeric = output.loc[:, FTN_PROJECTION_METRICS]
    invalid_numeric = numeric.isna().any(axis=1) | ~np.isfinite(numeric).all(axis=1)
    if invalid_numeric.any():
        invalid = output.loc[
            invalid_numeric,
            ["player", "pos", *FTN_PROJECTION_METRICS],
        ]
        raise ValueError(
            "FTN export contains missing or non-numeric projections: "
            f"{invalid.head(20).to_dict('records')}"
        )
    if (numeric < 0).any(axis=None):
        raise ValueError("FTN export contains negative projections")

    duplicate_keys = output.duplicated(["player", "pos"], keep=False)
    if duplicate_keys.any():
        duplicates = output.loc[duplicate_keys, ["player", "pos"]]
        raise ValueError(
            "FTN export contains duplicate player-position keys: "
            f"{duplicates.head(20).to_dict('records')}"
        )

    if len(output) < int(minimum_rows):
        raise ValueError(
            f"FTN export has only {len(output)} rows; "
            f"expected at least {int(minimum_rows)}"
        )
    floors = minimum_position_rows or _MINIMUM_POSITION_ROWS
    counts = output["pos"].value_counts()
    shallow = {
        position: {
            "observed": int(counts.get(position, 0)),
            "minimum": int(minimum),
        }
        for position, minimum in floors.items()
        if int(counts.get(position, 0)) < int(minimum)
    }
    if shallow:
        raise ValueError(f"FTN export is too shallow: {shallow}")

    return output.reset_index(drop=True)
