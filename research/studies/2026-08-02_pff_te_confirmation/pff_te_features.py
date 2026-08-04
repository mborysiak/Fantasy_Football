"""Shared leakage-safe prior-season PFF TE feature construction."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


MTF_RAW = "pff_te_mtf_per_reception_raw"
YAC_RAW = "pff_te_yac_per_route_raw"
MTF_CENTERED = "pff_te_mtf_per_reception_centered"
YAC_CENTERED = "pff_te_yac_per_route_centered"
MTF_MATCH = "match_pff_te_mtf_per_reception"
YAC_MATCH = "match_pff_te_yac_per_route"
AVAILABLE = "pff_te_receiver_available"
LOG_ROUTES = "pff_te_log_routes"
LOG_RECEPTIONS = "pff_te_log_receptions"
MTF_RELIABILITY = "pff_te_mtf_reliability"
YAC_RELIABILITY = "pff_te_yac_reliability"

PROJECTION_CONTROL_FEATURES = (AVAILABLE, LOG_ROUTES, LOG_RECEPTIONS)
PROJECTION_MTF_FEATURES = (*PROJECTION_CONTROL_FEATURES, MTF_CENTERED)
PROJECTION_YAC_FEATURES = (*PROJECTION_CONTROL_FEATURES, YAC_CENTERED)
TEMPLATE_FEATURES = (MTF_MATCH, YAC_MATCH)
AUDIT_COLUMNS = (
    "pff_te_source_season",
    "pff_te_routes",
    "pff_te_receptions",
    MTF_RAW,
    YAC_RAW,
    MTF_RELIABILITY,
    YAC_RELIABILITY,
    AVAILABLE,
    MTF_CENTERED,
    YAC_CENTERED,
    MTF_MATCH,
    YAC_MATCH,
)


def _read_sql(database: Path, query: str) -> pd.DataFrame:
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return pd.read_sql_query(query, connection)


def _normalize_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round().astype("Int64")


def _weighted_center(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[pd.Series, float]:
    low = float(values.quantile(0.01))
    high = float(values.quantile(0.99))
    clipped = values.clip(low, high)
    center = float(np.average(clipped, weights=weights))
    return clipped, center


def build_te_profiles(
    v2_database: Path,
    raw_database: Path,
    max_source_season: int,
) -> pd.DataFrame:
    identity = _read_sql(
        v2_database,
        "SELECT player_key, pff_id FROM player_identity WHERE pff_id IS NOT NULL",
    )
    identity["pff_id_num"] = _normalize_id(identity["pff_id"])
    if identity["player_key"].duplicated().any():
        raise ValueError("player_identity is not unique by player_key")
    if identity["pff_id_num"].duplicated().any():
        raise ValueError("player_identity is not unique by numeric PFF ID")

    raw = _read_sql(
        raw_database,
        f"""
        SELECT player_id, player, CAST(year AS INTEGER) AS pff_te_source_season,
               position, routes, receptions, avoided_tackles, yards_after_catch
        FROM PFF_Rec_Stats
        WHERE position = 'TE' AND year <= {int(max_source_season)}
        """,
    )
    raw["pff_id_num"] = _normalize_id(raw["player_id"])
    numeric = ["routes", "receptions", "avoided_tackles", "yards_after_catch"]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw.duplicated(["pff_id_num", "pff_te_source_season"]).any():
        raise ValueError("PFF TE receiving rows are not unique by ID-season")
    raw[MTF_RAW] = (
        raw["avoided_tackles"]
        / raw["receptions"].where(raw["receptions"].gt(0))
    )
    raw[YAC_RAW] = (
        raw["yards_after_catch"]
        / raw["routes"].where(raw["routes"].gt(0))
    )
    raw[MTF_RELIABILITY] = (
        raw["receptions"] / (raw["receptions"] + 20.0)
    ).fillna(0).clip(0, 1)
    raw[YAC_RELIABILITY] = (
        raw["routes"] / (raw["routes"] + 100.0)
    ).fillna(0).clip(0, 1)

    frames = []
    for source_season, group in raw.groupby("pff_te_source_season", sort=True):
        current = group.copy()
        for raw_column, reliability_column, centered_column, match_column, weight_column in (
            (MTF_RAW, MTF_RELIABILITY, MTF_CENTERED, MTF_MATCH, "receptions"),
            (YAC_RAW, YAC_RELIABILITY, YAC_CENTERED, YAC_MATCH, "routes"),
        ):
            valid = current[raw_column].notna() & current[weight_column].gt(0)
            clipped = pd.Series(np.nan, index=current.index, dtype=float)
            center = np.nan
            if valid.any():
                clipped_values, center = _weighted_center(
                    current.loc[valid, raw_column].astype(float),
                    current.loc[valid, weight_column].astype(float),
                )
                clipped.loc[valid] = clipped_values
            reliability = current[reliability_column]
            shrunk = center + reliability * (clipped - center)
            current[centered_column] = (shrunk - center).fillna(0.0)
            percentile = current.loc[valid, raw_column].rank(
                method="average", pct=True
            )
            current[match_column] = 0.5
            current.loc[valid, match_column] = 0.5 + (
                percentile - 0.5
            ) * reliability.loc[valid]
            current[match_column] = current[match_column].clip(0, 1)
        frames.append(current)
    profiles = pd.concat(frames, ignore_index=True)
    profiles = profiles.merge(
        identity[["player_key", "pff_id_num"]],
        on="pff_id_num",
        how="inner",
        validate="many_to_one",
    )
    profiles["season"] = profiles["pff_te_source_season"] + 1
    profiles["pff_te_routes"] = profiles["routes"]
    profiles["pff_te_receptions"] = profiles["receptions"]
    profiles[AVAILABLE] = profiles["routes"].gt(0).astype(float)
    profiles[LOG_ROUTES] = np.log1p(profiles["routes"].fillna(0).clip(lower=0))
    profiles[LOG_RECEPTIONS] = np.log1p(
        profiles["receptions"].fillna(0).clip(lower=0)
    )
    keep = ["player_key", "season", *AUDIT_COLUMNS, LOG_ROUTES, LOG_RECEPTIONS]
    profiles = profiles[keep].sort_values(["season", "player_key"])
    if profiles.duplicated(["player_key", "season"]).any():
        raise ValueError("Mapped PFF TE profiles are not unique by key-season")
    return profiles.reset_index(drop=True)


def attach_projection_features(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.merge(
        profiles,
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    is_te = output["position"].eq("TE")
    for column in (
        *PROJECTION_CONTROL_FEATURES,
        MTF_CENTERED,
        YAC_CENTERED,
        MTF_RELIABILITY,
        YAC_RELIABILITY,
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
        output.loc[~is_te, column] = 0.0
    return output


def attach_template_features(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    overlap = sorted(set(AUDIT_COLUMNS).intersection(frame.columns))
    if overlap:
        raise ValueError("PFF TE columns already exist: " + ", ".join(overlap))
    output = frame.merge(
        profiles,
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    for column in TEMPLATE_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.5)
    output[AVAILABLE] = pd.to_numeric(
        output[AVAILABLE], errors="coerce"
    ).fillna(0).astype(int)
    return output

