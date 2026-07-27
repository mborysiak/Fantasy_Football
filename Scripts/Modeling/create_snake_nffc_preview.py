"""Build an isolated Snake app database with DK data tagged as NFFC preview data.

This utility is intentionally temporary scaffolding. It copies an existing Snake
database, clones only the runtime tables required by the app, and leaves the
modeling repo's source Simulation.sqlite3 untouched while NFFC projections run.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


SOURCE_LEAGUE = "dk"
TARGET_LEAGUE = "nffc"
SOURCE_TEMPLATE_OFFSET = 2_000_000
TARGET_TEMPLATE_OFFSET = 3_000_000
TEMPLATE_ID_DELTA = TARGET_TEMPLATE_OFFSET - SOURCE_TEMPLATE_OFFSET
RUNTIME_TABLES = (
    "Final_Predictions_Resid",
    "Best_Ball_Weekly_Templates",
    "Best_Ball_Weekly_Template_Pools",
    "Best_Ball_Weekly_Player_Map",
)


def default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    snake_app = repo_root.parent / "Fantasy_Football_Snake" / "app"
    return (
        snake_app / "Simulation.sqlite3",
        snake_app / "Simulation_nffc_preview.sqlite3",
    )


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def canonical_player_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    if not rows:
        raise ValueError(f"Required table is missing: {table}")
    return [row[1] for row in rows]


def clone_rows(
    conn: sqlite3.Connection,
    table: str,
    overrides: dict[str, tuple[str, tuple[object, ...]]],
    where_sql: str,
    where_params: tuple[object, ...],
) -> int:
    columns = table_columns(conn, table)
    select_expressions: list[str] = []
    select_params: list[object] = []
    for column in columns:
        if column in overrides:
            expression, params = overrides[column]
            select_expressions.append(f"{expression} AS {quote_identifier(column)}")
            select_params.extend(params)
        else:
            select_expressions.append(quote_identifier(column))

    column_sql = ", ".join(quote_identifier(column) for column in columns)
    select_sql = ", ".join(select_expressions)
    cursor = conn.execute(
        f"INSERT INTO {quote_identifier(table)} ({column_sql}) "
        f"SELECT {select_sql} FROM {quote_identifier(table)} WHERE {where_sql}",
        (*select_params, *where_params),
    )
    return int(cursor.rowcount)


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def build_preview_database(
    source_path: Path,
    destination_path: Path,
    year: int,
    dataset: str,
) -> dict[str, int]:
    source_path = source_path.resolve()
    destination_path = destination_path.resolve()
    if source_path == destination_path:
        raise ValueError("Source and destination databases must be different files.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination_path.stem}_",
        suffix=".sqlite3",
        dir=destination_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()

    counts: dict[str, int] = {}
    try:
        source_uri = f"file:{source_path.as_posix()}?mode=ro"
        with closing(sqlite3.connect(source_uri, uri=True)) as source_conn, closing(
            sqlite3.connect(temp_path)
        ) as preview_conn:
            source_conn.backup(preview_conn)

        with closing(sqlite3.connect(temp_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            for table in RUNTIME_TABLES:
                table_columns(conn, table)

            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM Final_Predictions_Resid "
                    "WHERE year=? AND dataset=? AND version=?",
                    (year, dataset, TARGET_LEAGUE),
                )
                counts["Final_Predictions_Resid"] = clone_rows(
                    conn,
                    "Final_Predictions_Resid",
                    {"version": ("?", (TARGET_LEAGUE,))},
                    "year=? AND dataset=? AND version=?",
                    (year, dataset, SOURCE_LEAGUE),
                )

                conn.execute(
                    "DELETE FROM Best_Ball_Weekly_Templates WHERE league=?",
                    (TARGET_LEAGUE,),
                )
                counts["Best_Ball_Weekly_Templates"] = clone_rows(
                    conn,
                    "Best_Ball_Weekly_Templates",
                    {
                        "league": ("?", (TARGET_LEAGUE,)),
                        "template_id": (
                            '"template_id" + ?',
                            (TEMPLATE_ID_DELTA,),
                        ),
                    },
                    "league=?",
                    (SOURCE_LEAGUE,),
                )

                conn.execute(
                    "DELETE FROM Best_Ball_Weekly_Template_Pools "
                    "WHERE pool_year=? AND pool_dataset=? AND pool_version=?",
                    (year, dataset, TARGET_LEAGUE),
                )
                counts["Best_Ball_Weekly_Template_Pools"] = clone_rows(
                    conn,
                    "Best_Ball_Weekly_Template_Pools",
                    {
                        "template_pool_key": (
                            'REPLACE("template_pool_key", ?, ?)',
                            (f"|{SOURCE_LEAGUE}|", f"|{TARGET_LEAGUE}|"),
                        ),
                        "league": ("?", (TARGET_LEAGUE,)),
                        "pool_version": ("?", (TARGET_LEAGUE,)),
                        "template_id": (
                            '"template_id" + ?',
                            (TEMPLATE_ID_DELTA,),
                        ),
                        "template_league": ("?", (TARGET_LEAGUE,)),
                    },
                    "pool_year=? AND pool_dataset=? AND pool_version=?",
                    (year, dataset, SOURCE_LEAGUE),
                )

                conn.execute(
                    "DELETE FROM Best_Ball_Weekly_Player_Map "
                    "WHERE year=? AND dataset=? AND version=?",
                    (year, dataset, TARGET_LEAGUE),
                )
                counts["Best_Ball_Weekly_Player_Map"] = clone_rows(
                    conn,
                    "Best_Ball_Weekly_Player_Map",
                    {
                        "version": ("?", (TARGET_LEAGUE,)),
                        "template_pool_key": (
                            'REPLACE("template_pool_key", ?, ?)',
                            (f"|{SOURCE_LEAGUE}|", f"|{TARGET_LEAGUE}|"),
                        ),
                    },
                    "year=? AND dataset=? AND version=?",
                    (year, dataset, SOURCE_LEAGUE),
                )

                conn.execute(
                    "CREATE TABLE IF NOT EXISTS Snake_Preview_Metadata ("
                    "target_league TEXT NOT NULL, source_league TEXT NOT NULL, "
                    "year INTEGER NOT NULL, dataset TEXT NOT NULL, "
                    "created_at_utc TEXT NOT NULL, warning TEXT NOT NULL)"
                )
                conn.execute(
                    "DELETE FROM Snake_Preview_Metadata WHERE target_league=?",
                    (TARGET_LEAGUE,),
                )
                conn.execute(
                    "INSERT INTO Snake_Preview_Metadata VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        TARGET_LEAGUE,
                        SOURCE_LEAGUE,
                        year,
                        dataset,
                        datetime.now(timezone.utc).isoformat(),
                        "Temporary setup-only slice: projections and weekly templates "
                        "are cloned from DK and are not NFFC-calibrated.",
                    ),
                )

                source_counts = {
                    "Final_Predictions_Resid": scalar(
                        conn,
                        "SELECT COUNT(*) FROM Final_Predictions_Resid "
                        "WHERE year=? AND dataset=? AND version=?",
                        (year, dataset, SOURCE_LEAGUE),
                    ),
                    "Best_Ball_Weekly_Templates": scalar(
                        conn,
                        "SELECT COUNT(*) FROM Best_Ball_Weekly_Templates "
                        "WHERE league=?",
                        (SOURCE_LEAGUE,),
                    ),
                    "Best_Ball_Weekly_Template_Pools": scalar(
                        conn,
                        "SELECT COUNT(*) FROM Best_Ball_Weekly_Template_Pools "
                        "WHERE pool_year=? AND pool_dataset=? AND pool_version=?",
                        (year, dataset, SOURCE_LEAGUE),
                    ),
                    "Best_Ball_Weekly_Player_Map": scalar(
                        conn,
                        "SELECT COUNT(*) FROM Best_Ball_Weekly_Player_Map "
                        "WHERE year=? AND dataset=? AND version=?",
                        (year, dataset, SOURCE_LEAGUE),
                    ),
                }
                if counts != source_counts:
                    raise RuntimeError(
                        f"Preview clone row counts differ from DK source: "
                        f"target={counts}, source={source_counts}"
                    )

                adp_count = scalar(
                    conn,
                    "SELECT COUNT(*) FROM Avg_ADPs WHERE year=? AND league=?",
                    (year, TARGET_LEAGUE),
                )
                if adp_count == 0:
                    raise RuntimeError(
                        f"No Avg_ADPs rows exist for year={year}, "
                        f"league={TARGET_LEAGUE}."
                    )
                counts["Avg_ADPs"] = adp_count
                prediction_players = conn.execute(
                    "SELECT player FROM Final_Predictions_Resid "
                    "WHERE year=? AND dataset=? AND version=?",
                    (year, dataset, TARGET_LEAGUE),
                ).fetchall()
                adp_keys = {
                    canonical_player_key(row[0])
                    for row in conn.execute(
                        "SELECT player FROM Avg_ADPs WHERE year=? AND league=?",
                        (year, TARGET_LEAGUE),
                    )
                }
                direct_adp_matches = sum(
                    canonical_player_key(row[0]) in adp_keys
                    for row in prediction_players
                )
                counts["NFFC_ADP_Direct_Matches"] = direct_adp_matches
                counts["NFFC_ADP_Fallbacks"] = (
                    len(prediction_players) - direct_adp_matches
                )

                orphan_count = scalar(
                    conn,
                    "SELECT COUNT(*) FROM Best_Ball_Weekly_Template_Pools p "
                    "LEFT JOIN Best_Ball_Weekly_Templates t "
                    "ON p.template_id=t.template_id "
                    "AND p.template_league=t.league "
                    "WHERE p.pool_year=? AND p.pool_dataset=? "
                    "AND p.pool_version=? AND t.template_id IS NULL",
                    (year, dataset, TARGET_LEAGUE),
                )
                if orphan_count:
                    raise RuntimeError(
                        f"NFFC preview contains {orphan_count} orphaned template links."
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        os.replace(temp_path, destination_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return counts


def parse_args() -> argparse.Namespace:
    default_source, default_destination = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--destination", type=Path, default=default_destination)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--dataset", default="final_ensemble")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_preview_database(
        source_path=args.source,
        destination_path=args.destination,
        year=args.year,
        dataset=args.dataset,
    )
    print(f"Created NFFC preview database: {args.destination.resolve()}")
    for table, count in counts.items():
        print(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    main()
