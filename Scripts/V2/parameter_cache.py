"""Fingerprint, persist, and validate annual model hyperparameters."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PARAMETER_CACHE_VERSION = "v2_annual_hyperparameters_v1"
PARAMETER_CACHE_TABLE = "annual_model_parameter_cache"
LOCKED_CACHE_RUNNER = "locked_current"
NEXT_YEAR_CACHE_RUNNER = "next_year"
EXPECTED_CACHE_MODELS = {
    LOCKED_CACHE_RUNNER: (
        "expert_recalibrated_ridge",
        "conditional_ppg_lasso",
        "conditional_ppg_random_forest",
        "conditional_ppg_lightgbm",
        "participation_logistic",
        "participation_lightgbm",
        "conditional_ppg_log_adp_lasso",
    ),
    NEXT_YEAR_CACHE_RUNNER: (
        "next_residual_lasso",
        "next_residual_random_forest",
        "next_residual_lightgbm",
        "next_participation_logistic",
        "next_participation_lightgbm",
    ),
}
CACHE_SELECTION_COLUMNS = (
    "model_name",
    "forecast_origin",
    "candidate_id",
    "parameters_json",
    "selection_metric",
    "selection_score",
    "selection_start_season",
    "selection_end_season",
    "selection_start_origin",
    "selection_end_origin",
    "selection_seasons",
    "selection_source",
    "latest_usable_inner_origin",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def dataframe_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """Hash relevant rows independent of their input ordering."""

    ordered_columns = list(dict.fromkeys(str(column) for column in columns))
    missing = [column for column in ordered_columns if column not in frame]
    if missing:
        raise ValueError(
            f"Parameter-cache fingerprint columns are missing: {missing}"
        )
    values = frame.loc[:, ordered_columns].copy()
    row_hashes = pd.util.hash_pandas_object(
        values,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=True)
    row_hashes.sort()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "columns": ordered_columns,
                "dtypes": [str(values[column].dtype) for column in ordered_columns],
                "rows": len(values),
            }
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def parameter_fingerprint(
    *,
    frame: pd.DataFrame,
    data_columns: Sequence[str],
    specification: Mapping[str, Any],
) -> str:
    payload = {
        "cache_version": PARAMETER_CACHE_VERSION,
        "training_data_sha256": dataframe_sha256(frame, data_columns),
        "specification": dict(specification),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PARAMETER_CACHE_TABLE} (
            cache_version TEXT NOT NULL,
            season INTEGER NOT NULL,
            league TEXT NOT NULL,
            runner TEXT NOT NULL,
            model_name TEXT NOT NULL,
            fingerprint_sha256 TEXT NOT NULL,
            selection_sha256 TEXT NOT NULL,
            selection_rows INTEGER NOT NULL,
            selections_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (season, league, runner, model_name)
        )
        """
    )


def _selection_records(selections: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        column for column in CACHE_SELECTION_COLUMNS if column in selections
    ]

    def json_safe(value: Any) -> Any:
        if isinstance(value, bytes):
            return int.from_bytes(value, byteorder="little", signed=True)
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or pd.isna(value):
            return None
        return value

    return [
        {
            column: json_safe(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in selections.loc[:, columns].itertuples(
            index=False,
            name=None,
        )
    ]


def validate_cached_selections(
    selections: pd.DataFrame,
    *,
    model_name: str,
    expected_origins: Sequence[int],
    grid: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    required = {
        "model_name",
        "forecast_origin",
        "candidate_id",
        "parameters_json",
    }
    missing = sorted(required.difference(selections.columns))
    if missing:
        raise ValueError(f"Cached selections lack columns: {missing}")
    normalized = selections.copy()
    normalized["forecast_origin"] = pd.to_numeric(
        normalized["forecast_origin"], errors="raise"
    ).astype(int)
    normalized["candidate_id"] = pd.to_numeric(
        normalized["candidate_id"], errors="raise"
    ).astype(int)
    if set(normalized["model_name"].astype(str)) != {model_name}:
        raise ValueError(
            f"Cached selections do not belong to {model_name}"
        )
    origins = tuple(sorted(normalized["forecast_origin"].tolist()))
    expected = tuple(sorted(int(origin) for origin in expected_origins))
    if origins != expected:
        raise ValueError(
            f"Cached {model_name} origins differ: {origins} != {expected}"
        )
    if normalized.duplicated(["model_name", "forecast_origin"]).any():
        raise ValueError(f"Cached {model_name} selections are duplicated")
    for row in normalized.itertuples(index=False):
        candidate_id = int(row.candidate_id)
        if candidate_id < 0 or candidate_id >= len(grid):
            raise ValueError(
                f"Cached {model_name} candidate is outside its grid: "
                f"{candidate_id}"
            )
        observed_parameters = json.loads(str(row.parameters_json))
        if _canonical_json(observed_parameters) != _canonical_json(
            dict(grid[candidate_id])
        ):
            raise ValueError(
                f"Cached {model_name}/{row.forecast_origin} parameters "
                "do not match its candidate ID"
            )
    return normalized.sort_values("forecast_origin").reset_index(drop=True)


def load_parameter_cache(
    database: Path,
    *,
    season: int,
    league: str,
    runner: str,
    model_name: str,
    fingerprint_sha256: str,
    expected_origins: Sequence[int],
    grid: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    database = Path(database).resolve()
    miss = {
        "cache_version": PARAMETER_CACHE_VERSION,
        "season": int(season),
        "league": str(league),
        "runner": str(runner),
        "model_name": str(model_name),
        "fingerprint_sha256": str(fingerprint_sha256),
        "cache_hit": 0,
    }
    if not database.is_file():
        return None, {**miss, "cache_status": "database_missing"}
    with sqlite3.connect(database) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (PARAMETER_CACHE_TABLE,),
        ).fetchone()
        if table_exists is None:
            return None, {**miss, "cache_status": "table_missing"}
        row = connection.execute(
            f"""
            SELECT cache_version,
                   fingerprint_sha256,
                   selection_sha256,
                   selection_rows,
                   selections_json,
                   created_at_utc,
                   updated_at_utc
            FROM {PARAMETER_CACHE_TABLE}
            WHERE season=? AND league=? AND runner=? AND model_name=?
            """,
            (int(season), str(league), str(runner), str(model_name)),
        ).fetchone()
    if row is None:
        return None, {**miss, "cache_status": "entry_missing"}
    if str(row[0]) != PARAMETER_CACHE_VERSION:
        return None, {**miss, "cache_status": "version_mismatch"}
    if str(row[1]) != str(fingerprint_sha256):
        return None, {
            **miss,
            "cache_status": "fingerprint_mismatch",
            "cached_fingerprint_sha256": str(row[1]),
        }
    selection_json = str(row[4])
    selection_sha256 = hashlib.sha256(
        selection_json.encode("utf-8")
    ).hexdigest()
    if selection_sha256 != str(row[2]):
        return None, {**miss, "cache_status": "selection_hash_mismatch"}
    try:
        selections = validate_cached_selections(
            pd.DataFrame(json.loads(selection_json)),
            model_name=model_name,
            expected_origins=expected_origins,
            grid=grid,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return None, {
            **miss,
            "cache_status": "invalid_selection",
            "cache_error": str(error),
        }
    if len(selections) != int(row[3]):
        return None, {**miss, "cache_status": "selection_count_mismatch"}
    return selections, {
        **miss,
        "cache_hit": 1,
        "cache_status": "hit",
        "selection_rows": len(selections),
        "selection_sha256": selection_sha256,
        "cache_created_at_utc": str(row[5]),
        "cache_updated_at_utc": str(row[6]),
    }


def write_parameter_cache(
    database: Path,
    *,
    season: int,
    league: str,
    runner: str,
    model_name: str,
    fingerprint_sha256: str,
    expected_origins: Sequence[int],
    grid: Sequence[Mapping[str, object]],
    selections: pd.DataFrame,
) -> dict[str, Any]:
    database = Path(database).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_cached_selections(
        selections,
        model_name=model_name,
        expected_origins=expected_origins,
        grid=grid,
    )
    selection_json = _canonical_json(_selection_records(normalized))
    selection_sha256 = hashlib.sha256(
        selection_json.encode("utf-8")
    ).hexdigest()
    now = _utc_now()
    with sqlite3.connect(database) as connection:
        _ensure_schema(connection)
        connection.execute(
            f"""
            INSERT INTO {PARAMETER_CACHE_TABLE} (
                cache_version,
                season,
                league,
                runner,
                model_name,
                fingerprint_sha256,
                selection_sha256,
                selection_rows,
                selections_json,
                created_at_utc,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, league, runner, model_name) DO UPDATE SET
                cache_version=excluded.cache_version,
                fingerprint_sha256=excluded.fingerprint_sha256,
                selection_sha256=excluded.selection_sha256,
                selection_rows=excluded.selection_rows,
                selections_json=excluded.selections_json,
                created_at_utc=CASE
                    WHEN fingerprint_sha256=excluded.fingerprint_sha256
                    THEN created_at_utc
                    ELSE excluded.created_at_utc
                END,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                PARAMETER_CACHE_VERSION,
                int(season),
                str(league),
                str(runner),
                str(model_name),
                str(fingerprint_sha256),
                selection_sha256,
                len(normalized),
                selection_json,
                now,
                now,
            ),
        )
        connection.commit()
    return {
        "cache_version": PARAMETER_CACHE_VERSION,
        "season": int(season),
        "league": str(league),
        "runner": str(runner),
        "model_name": str(model_name),
        "fingerprint_sha256": str(fingerprint_sha256),
        "cache_hit": 0,
        "cache_status": "miss_written",
        "selection_rows": len(normalized),
        "selection_sha256": selection_sha256,
        "cache_created_at_utc": now,
        "cache_updated_at_utc": now,
    }


def validate_parameter_cache_database(
    database: Path,
    *,
    season: int,
    leagues: Sequence[str],
) -> dict[str, Any]:
    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Parameter cache database not found: {database}")
    with sqlite3.connect(database) as connection:
        integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise ValueError(f"Parameter cache integrity failed: {integrity}")
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (PARAMETER_CACHE_TABLE,),
        ).fetchone()
        if table_exists is None:
            raise ValueError("Parameter cache table is missing")
        rows = connection.execute(
            f"""
            SELECT cache_version,
                   league,
                   runner,
                   model_name,
                   fingerprint_sha256,
                   selection_sha256,
                   selection_rows,
                   selections_json
            FROM {PARAMETER_CACHE_TABLE}
            WHERE season=?
            """,
            (int(season),),
        ).fetchall()
    expected = {
        (str(league), runner, model_name)
        for league in leagues
        for runner, model_names in EXPECTED_CACHE_MODELS.items()
        for model_name in model_names
    }
    observed = {(str(row[1]), str(row[2]), str(row[3])) for row in rows}
    if observed != expected:
        raise ValueError(
            "Annual parameter cache coverage differs: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for row in rows:
        if str(row[0]) != PARAMETER_CACHE_VERSION:
            raise ValueError("Annual parameter cache version mismatch")
        if len(str(row[4])) != 64 or len(str(row[5])) != 64:
            raise ValueError("Annual parameter cache has an invalid SHA-256")
        selections_json = str(row[7])
        if hashlib.sha256(selections_json.encode("utf-8")).hexdigest() != str(
            row[5]
        ):
            raise ValueError("Annual parameter cache selection hash mismatch")
        selections = json.loads(selections_json)
        if len(selections) != int(row[6]) or not selections:
            raise ValueError("Annual parameter cache selection count mismatch")
    return {
        "cache_version": PARAMETER_CACHE_VERSION,
        "season": int(season),
        "entry_count": len(rows),
        "league_entry_counts": {
            str(league): sum(str(row[1]) == str(league) for row in rows)
            for league in leagues
        },
    }
