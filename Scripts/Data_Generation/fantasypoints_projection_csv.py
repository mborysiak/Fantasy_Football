"""Normalize FantasyPoints season projection CSV exports."""

from __future__ import annotations

import numpy as np
import pandas as pd


FANTASYPOINTS_PROJECTION_POSITIONS = ("QB", "RB", "WR", "TE")
FANTASYPOINTS_PROJECTION_METRICS = (
    "fpts_overall_rank",
    "fpts_adp",
    "fpts_proj_points",
    "fpts_games",
    "fpts_proj_points_per_game",
    "fpts_tier",
    "fpts_pass_att",
    "fpts_pass_cmp",
    "fpts_pass_yds",
    "fpts_pass_td",
    "fpts_pass_int",
    "fpts_rush_att",
    "fpts_rush_yds",
    "fpts_rush_td",
    "fpts_rec",
    "fpts_rec_yds",
    "fpts_rec_td",
)
FANTASYPOINTS_PROJECTION_COLUMNS = (
    "player",
    "pos",
    "team",
    "year",
    *FANTASYPOINTS_PROJECTION_METRICS,
)

_MINIMUM_ROWS = 400
_MINIMUM_POSITION_ROWS = {
    "QB": 40,
    "RB": 100,
    "WR": 150,
    "TE": 100,
}

# The current export has a grouped first header row. After reading with
# ``header=1``, player position is ``Position`` and ``POS`` is the numeric
# position rank. The older one-row export used ``POS`` for player position and
# ``POS.1`` for position rank. Position rank and the retired movement columns
# were never part of the database contract.
_CURRENT_COLUMN_MAP = {
    "RANK": "fpts_overall_rank",
    "NAME": "player",
    "Position": "pos",
    "Team": "team",
    "ADP": "fpts_adp",
    "FPTS": "fpts_proj_points",
    "GP": "fpts_games",
    "FPTS/G": "fpts_proj_points_per_game",
    "TIER": "fpts_tier",
    "ATT": "fpts_pass_att",
    "CMP": "fpts_pass_cmp",
    "YDS": "fpts_pass_yds",
    "TD": "fpts_pass_td",
    "INT": "fpts_pass_int",
    "ATT.1": "fpts_rush_att",
    "YDS.1": "fpts_rush_yds",
    "TD.1": "fpts_rush_td",
    "REC": "fpts_rec",
    "YDS.2": "fpts_rec_yds",
    "TD.2": "fpts_rec_td",
}

_LEGACY_COLUMN_MAP = {
    "RK": "fpts_overall_rank",
    "Name": "player",
    "POS": "pos",
    "Team": "team",
    "ADP": "fpts_adp",
    "FPTS": "fpts_proj_points",
    "G": "fpts_games",
    "FPTS/G": "fpts_proj_points_per_game",
    "TIER": "fpts_tier",
    "ATT": "fpts_pass_att",
    "CMP": "fpts_pass_cmp",
    "YDS": "fpts_pass_yds",
    "TD": "fpts_pass_td",
    "INT": "fpts_pass_int",
    "ATT.1": "fpts_rush_att",
    "YDS.1": "fpts_rush_yds",
    "TD.1": "fpts_rush_td",
    "REC": "fpts_rec",
    "YDS.2": "fpts_rec_yds",
    "TD.2": "fpts_rec_td",
}


def _select_column_map(frame: pd.DataFrame) -> dict[str, str]:
    if {"RANK", "NAME", "Position", "GP"}.issubset(frame.columns):
        return _CURRENT_COLUMN_MAP
    if {"RK", "Name", "POS", "G"}.issubset(frame.columns):
        return _LEGACY_COLUMN_MAP
    raise ValueError(
        "FantasyPoints export does not match the current or legacy season "
        f"projection schema; available columns are {list(frame.columns)}"
    )


def _numeric(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"-": "0", "": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_fantasypoints_projection_csv(
    frame: pd.DataFrame,
    *,
    year: int,
    minimum_rows: int = _MINIMUM_ROWS,
    minimum_position_rows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Normalize a current or legacy export to the existing SQLite schema."""

    column_map = _select_column_map(frame)
    missing = sorted(set(column_map).difference(frame.columns))
    if missing:
        raise ValueError(
            f"FantasyPoints season projection export is missing columns: {missing}"
        )

    output = frame.loc[:, list(column_map)].rename(columns=column_map).copy()
    output["player"] = (
        output["player"]
        .astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .replace("", pd.NA)
    )
    output["pos"] = output["pos"].astype("string").str.strip().str.upper()
    output["team"] = output["team"].astype("string").str.strip().str.upper()

    for column in FANTASYPOINTS_PROJECTION_METRICS:
        output[column] = _numeric(output[column])
    output["year"] = int(year)
    output = output.loc[:, FANTASYPOINTS_PROJECTION_COLUMNS]

    invalid_identity = (
        output["player"].isna()
        | output["team"].isna()
        | ~output["pos"].isin(FANTASYPOINTS_PROJECTION_POSITIONS)
    )
    if invalid_identity.any():
        invalid = output.loc[
            invalid_identity,
            ["player", "pos", "team"],
        ]
        raise ValueError(
            "FantasyPoints export contains invalid player identities: "
            f"{invalid.head(20).to_dict('records')}"
        )

    numeric = output.loc[:, FANTASYPOINTS_PROJECTION_METRICS]
    invalid_numeric = numeric.isna().any(axis=1) | ~np.isfinite(numeric).all(axis=1)
    if invalid_numeric.any():
        invalid = output.loc[
            invalid_numeric,
            ["player", "pos", *FANTASYPOINTS_PROJECTION_METRICS],
        ]
        raise ValueError(
            "FantasyPoints export contains missing or non-numeric projections: "
            f"{invalid.head(20).to_dict('records')}"
        )
    if (numeric < 0).any(axis=None):
        raise ValueError("FantasyPoints export contains negative projections")
    if output["fpts_overall_rank"].le(0).any() or output["fpts_adp"].le(0).any():
        raise ValueError("FantasyPoints export contains non-positive rank or ADP")

    duplicate_keys = output.duplicated(["player", "pos"], keep=False)
    if duplicate_keys.any():
        duplicates = output.loc[duplicate_keys, ["player", "pos"]]
        raise ValueError(
            "FantasyPoints export contains duplicate player-position keys: "
            f"{duplicates.head(20).to_dict('records')}"
        )

    if len(output) < int(minimum_rows):
        raise ValueError(
            f"FantasyPoints export has only {len(output)} rows; "
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
        raise ValueError(f"FantasyPoints export is too shallow: {shallow}")

    return output.reset_index(drop=True)
