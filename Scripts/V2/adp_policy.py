"""Governed ADP-family and NFFC distribution policy.

This module deliberately separates three concepts that were previously mixed:

* raw provider observations retained for audit and identity resolution;
* one family-level ADP observation per player-season for model consensus; and
* the two-feed NFFC distribution published to the draft applications.

The policy is shared by the manual source ingest, legacy Model_Inputs compiler,
V2 feature mart, and production source validator.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ADP_POLICY_VERSION = "canonical_market_adp_family_v2"
MFL_LAST_MODELED_SEASON = 2024

# These are the literal browser download names. The notebook archives them
# under semantic names after moving them out of Downloads.
NFFC_DOWNLOADS: Mapping[str, Mapping[str, str]] = {
    "ADP.tsv": {
        "source": "nffc_best_ball_overall",
        "archive_name": "NFFC_Best_Ball_Overall_ADP.tsv",
    },
    "ADP (1).tsv": {
        "source": "nffc_best_ball_25s50s",
        "archive_name": "NFFC_Best_Ball_25s50s_ADP.tsv",
    },
}
NFFC_MODELED_SOURCES = tuple(
    value["source"] for value in NFFC_DOWNLOADS.values()
)
NFFC_AGGREGATION_POLICY = "nffc_overall_25s50s_equal_mean_v1"
NFFC_BOUNDS_POLICY = "nffc_overall_25s50s_equal_mean_bounds_v1"
NFFC_STD_DEV_POLICY = "nffc_within_between_pooled_sd_v1"
NFFC_RANGE_TO_SD_DIVISOR = 5.0
NFFC_AGREEMENT_TOP_PICK = 240.0
NFFC_AGREEMENT_MIN_COMMON = 180
NFFC_AGREEMENT_MIN_SPEARMAN = 0.95
NFFC_AGREEMENT_MAX_MEDIAN_GAP = 12.0

DK_AGGREGATION_POLICY = "draftkings_direct_center_v1"
DK_BOUNDS_POLICY = "scaled_nffc_two_feed_bounds_v1"
DK_STD_DEV_POLICY = "scaled_nffc_two_feed_pooled_sd_v1"

ADP_PROVENANCE_COLUMNS: Mapping[str, str] = {
    "source_count": "INTEGER",
    "feed_gap": "REAL",
    "aggregation_policy": "TEXT",
    "bounds_policy": "TEXT",
    "std_dev_policy": "TEXT",
    "adp_policy_version": "TEXT",
}

NFFC_RAW_PROVENANCE_COLUMNS: Mapping[str, str] = {
    "source_rank": "REAL",
    "draft_count": "REAL",
    "snapshot_file": "TEXT",
    "snapshot_sha256": "TEXT",
    "ingested_at_utc": "TEXT",
}

CANONICAL_ADP_SOURCE_FAMILIES: Mapping[str, str] = {
    "adp_mfl": "mfl",
    "adp_fpros": "fantasypros_redraft",
    "fantasypros_best_ball_adp": "fantasypros_best_ball",
    "adp_average_dk": "draftkings",
    "draftkings_adp": "draftkings",
    "adp_average_nffc": "nffc",
}

# Prefer the canonical ADP_Averages DK slice over the legacy DraftKings_ADP
# table when both are present for the same player-season.
CANONICAL_ADP_SOURCE_PRIORITY: Mapping[str, int] = {
    "adp_average_dk": 0,
    "draftkings_adp": 1,
}

LEGACY_CANONICAL_PICK_COLUMNS = (
    "pick_mfl",
    "pick_fpros",
    "pick_best_ball",
    "pick_dk",
    "pick_nffc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _policy_payload() -> dict[str, Any]:
    return {
        "version": ADP_POLICY_VERSION,
        "mfl_last_modeled_season": MFL_LAST_MODELED_SEASON,
        "nffc_downloads": NFFC_DOWNLOADS,
        "nffc_aggregation_policy": NFFC_AGGREGATION_POLICY,
        "nffc_bounds_policy": NFFC_BOUNDS_POLICY,
        "nffc_std_dev_policy": NFFC_STD_DEV_POLICY,
        "canonical_source_families": CANONICAL_ADP_SOURCE_FAMILIES,
        "canonical_source_priority": CANONICAL_ADP_SOURCE_PRIORITY,
        "legacy_pick_columns": LEGACY_CANONICAL_PICK_COLUMNS,
    }


def adp_policy_sha256() -> str:
    payload = json.dumps(
        _policy_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def adp_policy_receipt(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "component": "adp_policy",
        "source_name": ADP_POLICY_VERSION,
        "source_uri": "python://Scripts.V2.adp_policy",
        "source_sha256": adp_policy_sha256(),
        "row_count": len(CANONICAL_ADP_SOURCE_FAMILIES),
    }


def mfl_is_modeled(season: object) -> bool:
    try:
        return int(season) <= MFL_LAST_MODELED_SEASON
    except (TypeError, ValueError):
        return False


def mask_disallowed_mfl(values: pd.Series, seasons: pd.Series) -> pd.Series:
    numeric_seasons = pd.to_numeric(seasons, errors="coerce")
    return values.where(numeric_seasons.le(MFL_LAST_MODELED_SEASON))


def canonical_adp_family_values(market_values: pd.DataFrame) -> pd.DataFrame:
    """Return one approved ADP observation per provider family/player-season.

    Raw market values stay intact in ``player_season_market_values``. This
    function is only the model-consensus view, so audit/challenger sources do
    not silently become extra votes.
    """

    required = {"player_key", "season", "source", "adp"}
    missing = sorted(required.difference(market_values.columns))
    if missing:
        raise ValueError(
            f"Market values are missing canonical ADP columns: {missing}"
        )
    work = market_values.loc[:, list(required)].copy()
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["adp"] = pd.to_numeric(work["adp"], errors="coerce")
    work["source"] = work["source"].astype("string").str.strip().str.lower()
    work["adp_family"] = work["source"].map(CANONICAL_ADP_SOURCE_FAMILIES)
    work = work[
        work["adp_family"].notna()
        & work["season"].notna()
        & work["adp"].notna()
        & np.isfinite(work["adp"])
        & work["adp"].gt(0)
    ].copy()
    work = work[
        ~(
            work["adp_family"].eq("mfl")
            & work["season"].gt(MFL_LAST_MODELED_SEASON)
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["player_key", "season", "source", "adp"]
        )

    work["_source_priority"] = (
        work["source"].map(CANONICAL_ADP_SOURCE_PRIORITY).fillna(0).astype(int)
    )
    family_keys = ["player_key", "season", "adp_family"]
    best_priority = work.groupby(family_keys)["_source_priority"].transform(
        "min"
    )
    work = work[work["_source_priority"].eq(best_priority)].copy()
    family = (
        work.groupby(family_keys, as_index=False, sort=True)["adp"]
        .median()
        .rename(columns={"adp_family": "source"})
    )
    family["source"] = "adp_family_" + family["source"].astype(str)
    return family[["player_key", "season", "source", "adp"]]


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], frame_name: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {missing}")


def build_nffc_two_feed_aggregate(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Build the NFFC center/bounds and pooled SD from Overall plus 25/50."""

    required = (
        "player",
        "pos",
        "year",
        "source",
        "pick_nffc",
        "min_pick",
        "max_pick",
    )
    _require_columns(raw_rows, required, "NFFC raw rows")
    raw = raw_rows.loc[:, list(required)].copy()
    raw["source"] = raw["source"].astype("string").str.strip().str.lower()
    raw = raw[raw["source"].isin(NFFC_MODELED_SOURCES)].copy()
    if raw.empty:
        raise ValueError("NFFC raw rows contain neither governed modeled feed")
    for column in ("year", "pick_nffc", "min_pick", "max_pick"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["player"] = raw["player"].astype("string").str.strip()
    raw["pos"] = raw["pos"].astype("string").str.strip().str.upper()
    invalid = (
        raw[["year", "pick_nffc", "min_pick", "max_pick"]]
        .isna()
        .any(axis=1)
        | ~np.isfinite(raw[["pick_nffc", "min_pick", "max_pick"]]).all(
            axis=1
        )
        | raw["player"].isna()
        | raw["player"].eq("")
        | raw["pos"].isna()
        | raw["pos"].eq("")
        | raw["pick_nffc"].le(0)
        | raw["min_pick"].le(0)
        | raw["max_pick"].le(0)
        | raw["max_pick"].lt(raw["min_pick"])
    )
    if invalid.any():
        raise ValueError(
            "NFFC raw rows contain invalid distribution values: "
            f"{raw.loc[invalid, ['player', 'pos', 'year', 'source']].head(20).to_dict('records')}"
        )
    duplicate_keys = ["player", "pos", "year", "source"]
    if raw.duplicated(duplicate_keys).any():
        raise ValueError("NFFC raw rows contain duplicate player/feed rows")

    raw["_within_sd"] = (
        raw["max_pick"] - raw["min_pick"]
    ) / NFFC_RANGE_TO_SD_DIVISOR
    keys = ["player", "pos", "year"]
    grouped = raw.groupby(keys, sort=True, dropna=False)
    aggregate = grouped.agg(
        avg_pick=("pick_nffc", "mean"),
        min_pick=("min_pick", "mean"),
        max_pick=("max_pick", "mean"),
        source_count=("source", "nunique"),
        feed_gap=("pick_nffc", lambda values: float(values.max() - values.min())),
    ).reset_index()
    aggregate.loc[aggregate["source_count"].lt(2), "feed_gap"] = np.nan
    centered = raw.merge(
        aggregate[keys + ["avg_pick"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    centered["_pooled_variance_component"] = np.square(
        centered["_within_sd"]
    ) + np.square(centered["pick_nffc"] - centered["avg_pick"])
    pooled = (
        centered.groupby(keys, sort=True)["_pooled_variance_component"]
        .mean()
        .pow(0.5)
        .rename("std_dev")
        .reset_index()
    )
    aggregate = aggregate.merge(
        pooled, on=keys, how="left", validate="one_to_one"
    )
    aggregate["year"] = aggregate["year"].astype(int)
    aggregate["league"] = "nffc"
    aggregate["aggregation_policy"] = NFFC_AGGREGATION_POLICY
    aggregate["bounds_policy"] = NFFC_BOUNDS_POLICY
    aggregate["std_dev_policy"] = NFFC_STD_DEV_POLICY
    aggregate["adp_policy_version"] = ADP_POLICY_VERSION
    ordered = [
        "player",
        "pos",
        "year",
        "avg_pick",
        "min_pick",
        "max_pick",
        "std_dev",
        "league",
        *ADP_PROVENANCE_COLUMNS,
    ]
    return aggregate[ordered].sort_values(
        ["year", "avg_pick", "player", "pos"]
    ).reset_index(drop=True)


def validate_nffc_pair_agreement(
    raw_rows: pd.DataFrame,
    *,
    top_pick: float = NFFC_AGREEMENT_TOP_PICK,
    minimum_common: int = NFFC_AGREEMENT_MIN_COMMON,
    minimum_spearman: float = NFFC_AGREEMENT_MIN_SPEARMAN,
    maximum_median_gap: float = NFFC_AGREEMENT_MAX_MEDIAN_GAP,
) -> dict[str, float | int]:
    """Fail closed when the two current NFFC feeds materially disagree."""

    _require_columns(
        raw_rows,
        ("player", "pos", "source", "pick_nffc"),
        "NFFC pair agreement rows",
    )
    raw = raw_rows.copy()
    raw["source"] = raw["source"].astype("string").str.lower()
    raw["pick_nffc"] = pd.to_numeric(raw["pick_nffc"], errors="coerce")
    raw = raw[
        raw["source"].isin(NFFC_MODELED_SOURCES)
        & raw["pick_nffc"].notna()
    ].copy()
    pivot = raw.pivot_table(
        index=["player", "pos"],
        columns="source",
        values="pick_nffc",
        aggfunc="first",
    )
    missing = sorted(set(NFFC_MODELED_SOURCES).difference(pivot.columns))
    if missing:
        raise ValueError(f"NFFC pair agreement is missing feeds: {missing}")
    pair = pivot.dropna(subset=list(NFFC_MODELED_SOURCES)).copy()
    first, second = NFFC_MODELED_SOURCES
    top = pair[pair[[first, second]].min(axis=1).le(float(top_pick))].copy()
    if len(top) < int(minimum_common):
        raise ValueError(
            "NFFC pair agreement has only "
            f"{len(top)} common top-{int(top_pick)} players; "
            f"minimum is {int(minimum_common)}"
        )
    first_rank = top[first].rank(method="average")
    second_rank = top[second].rank(method="average")
    spearman = float(first_rank.corr(second_rank, method="pearson"))
    median_gap = float((top[first] - top[second]).abs().median())
    if not math.isfinite(spearman) or spearman < float(minimum_spearman):
        raise ValueError(
            "NFFC pair top-pick Spearman agreement is below policy: "
            f"observed={spearman:.6f}, minimum={minimum_spearman:.6f}"
        )
    if median_gap > float(maximum_median_gap):
        raise ValueError(
            "NFFC pair top-pick median ADP gap is above policy: "
            f"observed={median_gap:.3f}, maximum={maximum_median_gap:.3f}"
        )
    return {
        "common_top_pick_rows": int(len(top)),
        "spearman": spearman,
        "median_abs_gap": median_gap,
    }


def _ensure_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: Mapping[str, str],
) -> None:
    available = {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }
    if not available:
        raise ValueError(f"Source database is missing table {table}")
    for column, sql_type in columns.items():
        if column not in available:
            connection.execute(
                f'ALTER TABLE "{table}" ADD COLUMN "{column}" {sql_type}'
            )


def _sqlite_values(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple]:
    values: list[tuple] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        converted = []
        for value in row:
            if pd.isna(value):
                converted.append(None)
            elif isinstance(value, np.generic):
                converted.append(value.item())
            else:
                converted.append(value)
        values.append(tuple(converted))
    return values


def replace_current_nffc_policy_rows(
    db_path: str | Path,
    current_rows: pd.DataFrame,
    *,
    year: int,
    rebuild_from_season: int = 2025,
) -> pd.DataFrame:
    """Atomically replace current raw feeds and rebuild governed aggregates."""

    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ADP source database does not exist: {path}")
    required_raw = (
        "player",
        "team",
        "pos",
        "pick_nffc",
        "min_pick",
        "max_pick",
        "source",
        "year",
        *NFFC_RAW_PROVENANCE_COLUMNS,
    )
    _require_columns(current_rows, required_raw, "Current NFFC rows")
    current = current_rows.loc[:, list(required_raw)].copy()
    current_sources = set(current["source"].dropna().astype(str))
    if current_sources != set(NFFC_MODELED_SOURCES):
        raise ValueError(
            "Current NFFC rows must contain exactly the two governed feeds: "
            f"observed={sorted(current_sources)}"
        )
    if not pd.to_numeric(current["year"], errors="coerce").eq(int(year)).all():
        raise ValueError("Current NFFC rows contain an unexpected season")
    validate_nffc_pair_agreement(current)

    with sqlite3.connect(path, timeout=45) as connection:
        connection.execute("PRAGMA busy_timeout=45000")
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_columns(
                connection,
                "NFFC_ADP",
                NFFC_RAW_PROVENANCE_COLUMNS,
            )
            _ensure_columns(
                connection,
                "ADP_Averages",
                ADP_PROVENANCE_COLUMNS,
            )
            connection.execute(
                "DELETE FROM NFFC_ADP WHERE CAST(year AS INTEGER)=?",
                (int(year),),
            )
            placeholders = ", ".join("?" for _ in required_raw)
            quoted = ", ".join(f'"{column}"' for column in required_raw)
            connection.executemany(
                f'INSERT INTO NFFC_ADP ({quoted}) VALUES ({placeholders})',
                _sqlite_values(current, required_raw),
            )
            raw_history = pd.read_sql_query(
                """
                SELECT player, pos, year, source, pick_nffc, min_pick, max_pick
                FROM NFFC_ADP
                WHERE CAST(year AS INTEGER) BETWEEN ? AND ?
                  AND source IN (?, ?)
                """,
                connection,
                params=(
                    int(rebuild_from_season),
                    int(year),
                    *NFFC_MODELED_SOURCES,
                ),
            )
            aggregate = build_nffc_two_feed_aggregate(raw_history)
            aggregate_years = sorted(
                pd.to_numeric(aggregate["year"], errors="raise")
                .astype(int)
                .unique()
            )
            for aggregate_year in aggregate_years:
                connection.execute(
                    """
                    DELETE FROM ADP_Averages
                    WHERE CAST(year AS INTEGER)=?
                      AND LOWER(league)='nffc'
                    """,
                    (int(aggregate_year),),
                )
            aggregate_columns = (
                "player",
                "pos",
                "year",
                "avg_pick",
                "min_pick",
                "max_pick",
                "std_dev",
                "league",
                *ADP_PROVENANCE_COLUMNS,
            )
            aggregate_placeholders = ", ".join(
                "?" for _ in aggregate_columns
            )
            aggregate_quoted = ", ".join(
                f'"{column}"' for column in aggregate_columns
            )
            connection.executemany(
                "INSERT INTO ADP_Averages "
                f"({aggregate_quoted}) VALUES ({aggregate_placeholders})",
                _sqlite_values(aggregate, aggregate_columns),
            )
            stored = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM ADP_Averages
                    WHERE CAST(year AS INTEGER)=?
                      AND LOWER(league)='nffc'
                      AND adp_policy_version=?
                    """,
                    (int(year), ADP_POLICY_VERSION),
                ).fetchone()[0]
            )
            expected = int(aggregate["year"].eq(int(year)).sum())
            if stored != expected:
                raise RuntimeError(
                    "NFFC aggregate replacement count mismatch: "
                    f"expected {expected}, stored {stored}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return aggregate


def rebuild_existing_nffc_policy_aggregates(
    db_path: str | Path,
    *,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """Rebuild aggregates from retained raw pair rows without deleting audit rows."""

    path = Path(db_path).expanduser().resolve()
    target_seasons = sorted({int(season) for season in seasons})
    if not target_seasons:
        raise ValueError("At least one NFFC aggregate season is required")
    with sqlite3.connect(path, timeout=45) as connection:
        placeholders = ", ".join("?" for _ in target_seasons)
        raw = pd.read_sql_query(
            f"""
            SELECT player, pos, year, source, pick_nffc, min_pick, max_pick
            FROM NFFC_ADP
            WHERE CAST(year AS INTEGER) IN ({placeholders})
              AND source IN (?, ?)
            """,
            connection,
            params=(*target_seasons, *NFFC_MODELED_SOURCES),
        )
    aggregate = build_nffc_two_feed_aggregate(raw)
    observed = sorted(aggregate["year"].astype(int).unique())
    if observed != target_seasons:
        raise ValueError(
            "NFFC aggregate rebuild seasons differ from request: "
            f"requested={target_seasons}, observed={observed}"
        )
    for season in target_seasons:
        season_raw = raw[raw["year"].eq(season)]
        validate_nffc_pair_agreement(season_raw)

    aggregate_columns = (
        "player",
        "pos",
        "year",
        "avg_pick",
        "min_pick",
        "max_pick",
        "std_dev",
        "league",
        *ADP_PROVENANCE_COLUMNS,
    )
    with sqlite3.connect(path, timeout=45) as connection:
        connection.execute("PRAGMA busy_timeout=45000")
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_columns(
                connection,
                "ADP_Averages",
                ADP_PROVENANCE_COLUMNS,
            )
            for season in target_seasons:
                connection.execute(
                    """
                    DELETE FROM ADP_Averages
                    WHERE CAST(year AS INTEGER)=?
                      AND LOWER(league)='nffc'
                    """,
                    (season,),
                )
            placeholders = ", ".join("?" for _ in aggregate_columns)
            quoted = ", ".join(
                f'"{column}"' for column in aggregate_columns
            )
            connection.executemany(
                f"INSERT INTO ADP_Averages ({quoted}) VALUES ({placeholders})",
                _sqlite_values(aggregate, aggregate_columns),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return aggregate
