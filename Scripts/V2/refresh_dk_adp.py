"""Safely refresh the current DraftKings ADP source slice.

The Occupy Fantasy endpoint supplies DraftKings average pick but not the
distribution fields consumed by the draft simulator.  The existing preseason
method borrows each player's NFFC min/max ratios and applies those ratios to the
DraftKings average.  This module keeps that method, recomputes the derived
standard deviation from the scaled range, validates the complete slice, and
replaces only ``ADP_Averages(year, league='dk')`` in one transaction.

The command line intentionally requires an explicit database path so a caller
can stage and validate a release without implicitly mutating the live source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
# The project keeps its shared ``ff`` package in a sibling repository during
# local development.  Add both import roots so this file behaves the same when
# run directly as it does under the repository's configured test environment.
for import_root in (REPO_ROOT, REPO_ROOT.parent / "ff"):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import pandas as pd
import requests

from ff import data_clean

from Scripts.V2.adp_policy import (
    ADP_POLICY_VERSION,
    ADP_PROVENANCE_COLUMNS,
    DK_AGGREGATION_POLICY,
    DK_BOUNDS_POLICY,
    DK_STD_DEV_POLICY,
)
from Scripts.V2.contracts import normalize_player_name


DEFAULT_ENDPOINT = (
    "https://www.occupyfantasyapi.com/best_ball/adps"
    "?site=draftkings&contest=all"
)
ADP_COLUMNS = (
    "player",
    "pos",
    "year",
    "avg_pick",
    "min_pick",
    "max_pick",
    "std_dev",
    "league",
)
OFFENSIVE_POSITIONS = ("QB", "RB", "WR", "TE")
NUMERIC_COLUMNS = ("avg_pick", "min_pick", "max_pick", "std_dev")
DK_PLAYER_NAME_ALIASES = {
    "Nick Singleton": "Nicholas Singleton",
}
NFFC_DISTRIBUTION_NAME_ALIASES = {
    "jayden ott": "jaydn ott",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_dk_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Return validated player/average-pick rows from an API payload."""

    rows = payload.get("adps")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "DraftKings best-ball ADP response does not contain a non-empty "
            "'adps' list"
        )

    raw = pd.DataFrame(rows)
    required = {"player_name", "pos", "curr_adp"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(
            "DraftKings best-ball ADP response is missing columns: "
            f"{missing}"
        )

    api = raw.rename(
        columns={"player_name": "player", "curr_adp": "pick_dk"}
    )[["player", "pos", "pick_dk"]].dropna()
    if api.empty:
        raise ValueError("DraftKings best-ball ADP response has no usable rows")

    try:
        api["pick_dk"] = pd.to_numeric(api["pick_dk"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DraftKings best-ball ADP contains a non-numeric curr_adp"
        ) from exc

    api["player"] = (
        api["player"]
        .astype(str)
        .map(data_clean.name_clean)
        .replace(DK_PLAYER_NAME_ALIASES)
    )
    api["pos"] = api["pos"].astype(str).str.upper()
    invalid_pick = (
        ~np.isfinite(api["pick_dk"].astype(float)) | api["pick_dk"].le(0)
    )
    if invalid_pick.any():
        raise ValueError(
            "DraftKings best-ball ADP contains non-positive or non-finite "
            "curr_adp values"
        )
    if api["player"].eq("").any():
        raise ValueError("DraftKings best-ball ADP contains a blank player")
    invalid_positions = sorted(
        set(api["pos"]).difference(OFFENSIVE_POSITIONS)
    )
    if invalid_positions:
        raise ValueError(
            "DraftKings best-ball ADP contains unsupported positions: "
            f"{invalid_positions}"
        )
    if api.duplicated("player").any():
        duplicates = sorted(
            api.loc[api.duplicated("player", keep=False), "player"].unique()
        )
        raise ValueError(
            "DraftKings best-ball ADP contains duplicate cleaned players: "
            f"{duplicates[:20]}"
        )

    return api.sort_values(["pick_dk", "player"]).reset_index(drop=True)


def _distribution_join_name(player: object) -> str:
    normalized = normalize_player_name(player)
    return NFFC_DISTRIBUTION_NAME_ALIASES.get(normalized, normalized)


def _validate_reference_rows(nffc_rows: pd.DataFrame, year: int) -> pd.DataFrame:
    required = set(ADP_COLUMNS)
    missing = sorted(required.difference(nffc_rows.columns))
    if missing:
        raise ValueError(f"NFFC ADP reference is missing columns: {missing}")

    nffc = nffc_rows.loc[:, list(ADP_COLUMNS)].copy()
    nffc["year"] = pd.to_numeric(nffc["year"], errors="coerce")
    nffc["league"] = nffc["league"].astype(str).str.lower()
    nffc["pos"] = nffc["pos"].astype(str).str.upper()
    nffc = nffc.loc[
        nffc["year"].eq(int(year))
        & nffc["league"].eq("nffc")
        & nffc["pos"].isin(OFFENSIVE_POSITIONS)
    ].copy()
    if nffc.empty:
        raise ValueError(f"No offensive NFFC ADP rows found for {year}")

    for column in NUMERIC_COLUMNS:
        nffc[column] = pd.to_numeric(nffc[column], errors="coerce")
    invalid = (
        nffc[list(NUMERIC_COLUMNS)].isna().any(axis=1)
        | ~np.isfinite(nffc[list(NUMERIC_COLUMNS)]).all(axis=1)
        | nffc["avg_pick"].le(0)
        | nffc["min_pick"].le(0)
        | nffc["max_pick"].le(0)
        | nffc["std_dev"].lt(0)
    )
    if invalid.any():
        raise ValueError(
            "NFFC ADP reference contains invalid distribution rows: "
            f"{nffc.loc[invalid, ['player', 'pos']].head(20).to_dict('records')}"
        )
    nffc["_distribution_join_name"] = nffc["player"].map(
        _distribution_join_name
    )
    if nffc.duplicated(["_distribution_join_name", "year"]).any():
        duplicates = nffc.loc[
            nffc.duplicated(
                ["_distribution_join_name", "year"],
                keep=False,
            ),
            ["player", "pos"],
        ]
        raise ValueError(
            "NFFC ADP reference is not unique by canonical player/year: "
            f"{duplicates.head(20).to_dict('records')}"
        )
    return nffc


def build_dk_adp_rows(
    nffc_rows: pd.DataFrame,
    api_rows: pd.DataFrame,
    *,
    year: int,
    min_depth: int = 280,
) -> pd.DataFrame:
    """Build a complete DK slice using NFFC-relative distribution bounds."""

    nffc = _validate_reference_rows(nffc_rows, year)
    required_api = {"player", "pos", "pick_dk"}
    missing_api = sorted(required_api.difference(api_rows.columns))
    if missing_api:
        raise ValueError(f"DraftKings API rows are missing columns: {missing_api}")
    api = api_rows.loc[:, ["player", "pos", "pick_dk"]].copy()
    api["player"] = (
        api["player"]
        .astype(str)
        .map(data_clean.name_clean)
        .replace(DK_PLAYER_NAME_ALIASES)
    )
    api["pos"] = api["pos"].astype(str).str.upper()
    api["pick_dk"] = pd.to_numeric(api["pick_dk"], errors="coerce")
    if api.duplicated("player").any():
        raise ValueError("DraftKings API rows are not unique by cleaned player")
    invalid_api = (
        api["player"].eq("")
        | api["pick_dk"].isna()
        | ~np.isfinite(api["pick_dk"])
        | api["pick_dk"].le(0)
        | ~api["pos"].isin(OFFENSIVE_POSITIONS)
    )
    if invalid_api.any():
        raise ValueError("DraftKings API rows contain invalid values")
    if len(api) < int(min_depth):
        raise ValueError(
            f"DraftKings API produced only {len(api)} usable rows; "
            f"minimum required depth is {min_depth}"
        )

    api["_distribution_join_name"] = api["player"].map(
        _distribution_join_name
    )
    reference = nffc.rename(
        columns={
            "player": "nffc_player",
            "pos": "nffc_pos",
            "avg_pick": "nffc_avg_pick",
            "min_pick": "nffc_min_pick",
            "max_pick": "nffc_max_pick",
            "std_dev": "nffc_std_dev",
            "league": "nffc_league",
            "year": "nffc_year",
        }
    )
    merged = api.merge(
        reference,
        on="_distribution_join_name",
        how="left",
        validate="one_to_one",
    )
    has_nffc_distribution = merged["nffc_avg_pick"].notna()
    mismatched_position = (
        has_nffc_distribution & merged["pos"].ne(merged["nffc_pos"])
    )

    scale = merged["pick_dk"] / merged["nffc_avg_pick"]
    merged["min_pick"] = merged["nffc_min_pick"] * scale
    merged["max_pick"] = merged["nffc_max_pick"] * scale
    merged["avg_pick"] = merged["pick_dk"]
    missing_distribution = ~has_nffc_distribution
    merged.loc[missing_distribution, "min_pick"] = (
        0.8 * merged.loc[missing_distribution, "avg_pick"]
    )
    merged.loc[missing_distribution, "max_pick"] = (
        1.2 * merged.loc[missing_distribution, "avg_pick"]
    )
    # NFFC includes an unselected-player penalty in deep ADP while min/max
    # reflect only actual selections.  Sparse rows can therefore have a lower
    # bound above ADP or an upper bound below it.  Preserve informative sides
    # and use the simulator's governed +/-20% fallback only for invalid sides.
    invalid_lower = merged["min_pick"].gt(merged["avg_pick"])
    invalid_upper = merged["max_pick"].lt(merged["avg_pick"])
    merged.loc[invalid_lower, "min_pick"] = (
        0.8 * merged.loc[invalid_lower, "avg_pick"]
    )
    merged.loc[invalid_upper, "max_pick"] = (
        1.2 * merged.loc[invalid_upper, "avg_pick"]
    )
    # Scale the governed NFFC pooled SD so DK retains both within-feed range
    # and between-feed center disagreement. Bounds remain synthetic ratios;
    # the DraftKings API is authoritative only for the center.
    merged["std_dev"] = merged["nffc_std_dev"] * scale
    merged["std_dev"] = pd.concat(
        [
            merged["std_dev"],
            (merged["max_pick"] - merged["min_pick"]) / 5.0,
        ],
        axis=1,
    ).max(axis=1)
    merged.loc[missing_distribution, "std_dev"] = (
        0.2 * merged.loc[missing_distribution, "avg_pick"]
    )
    small_dispersion = (
        ~missing_distribution & merged["std_dev"].lt(0.1)
    )
    merged.loc[small_dispersion, "std_dev"] = (
        0.2 * merged.loc[small_dispersion, "avg_pick"]
    )
    merged["year"] = int(year)
    merged["league"] = "dk"
    result = merged.loc[:, list(ADP_COLUMNS)].copy()
    result = result.sort_values(["avg_pick", "player"]).reset_index(drop=True)

    if result.duplicated(["player", "pos", "year", "league"]).any():
        raise ValueError("Built DraftKings ADP slice contains duplicate rows")
    invalid = (
        result[list(NUMERIC_COLUMNS)].isna().any(axis=1)
        | ~np.isfinite(result[list(NUMERIC_COLUMNS)]).all(axis=1)
        | result["avg_pick"].le(0)
        | result["min_pick"].le(0)
        | result["max_pick"].le(0)
        | result["std_dev"].lt(0)
        | result["min_pick"].gt(result["avg_pick"])
        | result["avg_pick"].gt(result["max_pick"])
    )
    if invalid.any():
        raise ValueError(
            "Built DraftKings ADP slice contains invalid distributions"
        )
    result.attrs["repaired_lower_bound_count"] = int(invalid_lower.sum())
    result.attrs["repaired_upper_bound_count"] = int(invalid_upper.sum())
    result.attrs["small_dispersion_fallback_count"] = int(
        small_dispersion.sum()
    )
    result.attrs["nffc_distribution_match_count"] = int(
        has_nffc_distribution.sum()
    )
    result.attrs["nffc_distribution_fallback_count"] = int(
        missing_distribution.sum()
    )
    result.attrs["nffc_position_mismatch_count"] = int(
        mismatched_position.sum()
    )
    result.attrs["nffc_position_mismatches"] = (
        merged.loc[
            mismatched_position,
            ["player", "pos", "nffc_pos"],
        ].to_dict("records")
    )
    return result


def _rows_digest(rows: pd.DataFrame) -> str:
    normalized: list[list[Any]] = []
    for row in rows.sort_values(
        ["year", "league", "avg_pick", "player", "pos"]
    ).itertuples(index=False, name=None):
        normalized.append(
            [
                (
                    float(value)
                    if isinstance(value, (float, np.floating))
                    and math.isfinite(float(value))
                    else int(value)
                    if isinstance(value, (int, np.integer))
                    else value
                )
                for value in row
            ]
        )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replace_current_dk_rows(
    db_path: str | Path,
    rows: pd.DataFrame,
    *,
    year: int,
) -> None:
    """Replace one DK season slice atomically after validating its schema."""

    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ADP source database does not exist: {path}")

    expected = rows.loc[:, list(ADP_COLUMNS)].copy()
    with sqlite3.connect(path, timeout=45) as connection:
        table_info = connection.execute(
            'PRAGMA table_info("ADP_Averages")'
        ).fetchall()
        available = {str(row[1]) for row in table_info}
        missing = sorted(set(ADP_COLUMNS).difference(available))
        if missing:
            raise ValueError(
                f"ADP_Averages is missing required columns: {missing}"
            )
        for column, sql_type in ADP_PROVENANCE_COLUMNS.items():
            if column not in available:
                connection.execute(
                    f'ALTER TABLE "ADP_Averages" ADD COLUMN "{column}" {sql_type}'
                )

        values = [
            tuple(
                value.item() if isinstance(value, np.generic) else value
                for value in row
            )
            for row in expected.itertuples(index=False, name=None)
        ]
        connection.execute("PRAGMA busy_timeout=45000")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                DELETE FROM ADP_Averages
                WHERE CAST(year AS INTEGER)=?
                  AND LOWER(league)='dk'
                """,
                (int(year),),
            )
            provenance_values = (
                1,
                None,
                DK_AGGREGATION_POLICY,
                DK_BOUNDS_POLICY,
                DK_STD_DEV_POLICY,
                ADP_POLICY_VERSION,
            )
            connection.executemany(
                """
                INSERT INTO ADP_Averages
                    (player, pos, year, avg_pick, min_pick, max_pick,
                     std_dev, league, source_count, feed_gap,
                     aggregation_policy, bounds_policy, std_dev_policy,
                     adp_policy_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [tuple(value) + provenance_values for value in values],
            )
            stored_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM ADP_Averages
                WHERE CAST(year AS INTEGER)=?
                  AND LOWER(league)='dk'
                """,
                (int(year),),
            ).fetchone()[0]
            if int(stored_count) != len(expected):
                raise RuntimeError(
                    "DraftKings ADP replacement count mismatch: "
                    f"expected {len(expected)}, stored {stored_count}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _change_summary(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    old = previous.loc[:, list(ADP_COLUMNS)].copy()
    new = current.loc[:, list(ADP_COLUMNS)].copy()
    keys = ["player", "pos"]
    joined = old.merge(new, on=keys, how="inner", suffixes=("_old", "_new"))
    movement = (
        joined["avg_pick_new"] - joined["avg_pick_old"]
        if not joined.empty
        else pd.Series(dtype=float)
    )
    changed = (
        ~np.isclose(
            joined["avg_pick_old"],
            joined["avg_pick_new"],
            rtol=0,
            atol=1e-9,
        )
        if not joined.empty
        else np.array([], dtype=bool)
    )
    old_players = set(zip(old["player"], old["pos"]))
    new_players = set(zip(new["player"], new["pos"]))
    return {
        "previous_row_count": int(len(old)),
        "published_row_count": int(len(new)),
        "matched_previous_row_count": int(len(joined)),
        "changed_avg_pick_count": int(changed.sum()),
        "mean_absolute_avg_pick_move": (
            float(movement.abs().mean()) if not movement.empty else None
        ),
        "max_absolute_avg_pick_move": (
            float(movement.abs().max()) if not movement.empty else None
        ),
        "added_players": [
            {"player": player, "pos": pos}
            for player, pos in sorted(new_players.difference(old_players))
        ],
        "removed_players": [
            {"player": player, "pos": pos}
            for player, pos in sorted(old_players.difference(new_players))
        ],
    }


def refresh_dk_adp(
    db_path: str | Path,
    *,
    year: int,
    endpoint: str = DEFAULT_ENDPOINT,
    min_depth: int = 280,
    receipt_path: str | Path | None = None,
    raw_response_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch, validate, transactionally publish, and receipt a DK ADP slice."""

    fetched_at = utc_now()
    response = requests.get(
        endpoint,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    response_bytes = response.content
    try:
        payload = json.loads(response_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("DraftKings best-ball endpoint returned invalid JSON") from exc

    api = parse_dk_payload(payload)
    path = Path(db_path).expanduser().resolve()
    with sqlite3.connect(path) as connection:
        nffc = pd.read_sql_query(
            """
            SELECT player, pos, year, avg_pick, min_pick, max_pick,
                   std_dev, league
            FROM ADP_Averages
            WHERE CAST(year AS INTEGER)=?
              AND LOWER(league)='nffc'
            """,
            connection,
            params=(int(year),),
        )
        previous = pd.read_sql_query(
            """
            SELECT player, pos, year, avg_pick, min_pick, max_pick,
                   std_dev, league
            FROM ADP_Averages
            WHERE CAST(year AS INTEGER)=?
              AND LOWER(league)='dk'
            """,
            connection,
            params=(int(year),),
        )

    current = build_dk_adp_rows(
        nffc,
        api,
        year=year,
        min_depth=min_depth,
    )
    validated_nffc = _validate_reference_rows(nffc, year)
    nffc_keys = set(
        validated_nffc["_distribution_join_name"].astype(str)
    )
    api_with_keys = api.assign(
        _distribution_join_name=api["player"].map(_distribution_join_name)
    )
    api_keys = set(api_with_keys["_distribution_join_name"].astype(str))
    unmatched_api = api_with_keys.loc[
        ~api_with_keys["_distribution_join_name"].isin(nffc_keys),
        "player",
    ]
    unmatched_nffc = validated_nffc.loc[
        ~validated_nffc["_distribution_join_name"].isin(api_keys),
        "player",
    ]
    receipt: dict[str, Any] = {
        "receipt_version": "draftkings_adp_source_v1",
        "fetched_at_utc": fetched_at,
        "published_at_utc": utc_now(),
        "endpoint": endpoint,
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_row_count": int(len(payload["adps"])),
        "usable_api_row_count": int(len(api)),
        "nffc_offensive_reference_row_count": int(len(validated_nffc)),
        "nffc_distribution_match_count": int(
            current.attrs.get("nffc_distribution_match_count", 0)
        ),
        "nffc_distribution_fallback_count": int(
            current.attrs.get("nffc_distribution_fallback_count", 0)
        ),
        "nffc_position_mismatch_count": int(
            current.attrs.get("nffc_position_mismatch_count", 0)
        ),
        "nffc_position_mismatches": current.attrs.get(
            "nffc_position_mismatches",
            [],
        ),
        "unmatched_api_player_count": int(len(unmatched_api)),
        "unmatched_nffc_player_count": int(len(unmatched_nffc)),
        "unmatched_api_players": sorted(unmatched_api.astype(str)),
        "unmatched_nffc_players": sorted(unmatched_nffc.astype(str)),
        "database": str(path),
        "year": int(year),
        "league": "dk",
        "published_snapshot_sha256": _rows_digest(current),
        "distribution_method": (
            "NFFC min/max ratios scaled to DK avg_pick; "
            "unmatched API rows use the governed 20% fallback; "
            "invalid sparse-source bounds use the governed +/-20% side "
            "fallback; std_dev=(scaled max_pick-scaled min_pick)/5 with the "
            "governed 20% fallback below 0.1"
        ),
        "repaired_lower_bound_count": int(
            current.attrs.get("repaired_lower_bound_count", 0)
        ),
        "repaired_upper_bound_count": int(
            current.attrs.get("repaired_upper_bound_count", 0)
        ),
        "small_dispersion_fallback_count": int(
            current.attrs.get("small_dispersion_fallback_count", 0)
        ),
        **_change_summary(previous, current),
    }

    replace_current_dk_rows(path, current, year=year)

    if raw_response_path is not None:
        raw_path = Path(raw_response_path).expanduser().resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(response_bytes)
        receipt["raw_response_path"] = str(raw_path)
    if receipt_path is not None:
        output = Path(receipt_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Explicit Season_Stats_New.sqlite3 path to update.",
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--min-depth", type=int, default=280)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--raw-response", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = refresh_dk_adp(
        args.db,
        year=args.year,
        endpoint=args.endpoint,
        min_depth=args.min_depth,
        receipt_path=args.receipt,
        raw_response_path=args.raw_response,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
