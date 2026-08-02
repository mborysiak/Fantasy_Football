"""Promote a validated staged cutover while preserving auction app-owned tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


GENERATED_AUCTION_TABLES = [
    "Avg_ADPs",
    "Avg_ADPs_Publication_Audit",
    "Avg_ADPs_Publication_Receipt",
    "Final_Predictions_Resid",
    "V2_Production_Projection_Handoff",
    "V2_Production_Projection_Audit",
    "V2_Production_Eligibility_Audit",
    "V2_Projection_Legacy_Backup",
    "Best_Ball_Weekly_Templates",
    "Best_Ball_Weekly_Template_Pools",
    "Best_Ball_Weekly_Pool_Summary",
    "Best_Ball_Weekly_Player_Map",
    "Best_Ball_Weekly_Template_Audit",
    "Best_Ball_Weekly_Player_Pool_Audit",
    "Best_Ball_Weekly_Bucket_Audit",
    "Best_Ball_ADP_Audit",
    "Salaries",
    "Salaries_Pred",
    "League_Keepers",
    "Salary_Selection_Premium",
]


def database_tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return [
            row[0]
            for row in connection.execute(
                """SELECT name
                     FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name"""
            )
        ]


def table_digest(path: Path, table: str) -> str:
    digest = hashlib.sha256()
    with sqlite3.connect(path) as connection:
        schema = connection.execute(
            """SELECT COALESCE(sql, '')
                 FROM sqlite_master
                WHERE type='table' AND name=?""",
            (table,),
        ).fetchone()
        if schema is None:
            raise ValueError(f"{path} is missing table {table}")
        digest.update(schema[0].encode("utf-8"))
        columns = [
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            )
        ]
        digest.update("\x1f".join(columns).encode("utf-8"))
        # Materialize the canonical row representation while the SQLite
        # connection remains open.  In this Windows runtime, retaining fetched
        # text values past connection teardown made large-table repr output
        # intermittently unstable and could produce a false parity failure.
        rows = sorted(
            repr(tuple(row)).encode("utf-8")
            for row in connection.execute(f'SELECT * FROM "{table}"')
        )
        for row in rows:
            digest.update(row)
            digest.update(b"\n")
    return digest.hexdigest()


def table_digests(path: Path, tables: list[str]) -> dict[str, str]:
    return {table: table_digest(path, table) for table in sorted(tables)}


def table_content_matches(source: Path, destination: Path, table: str) -> bool:
    """Compare one table as a multiset inside a single SQLite connection."""

    with sqlite3.connect(destination) as connection:
        connection.execute("ATTACH DATABASE ? AS source_db", (str(source),))
        source_schema = connection.execute(
            """SELECT COALESCE(sql, '')
                 FROM source_db.sqlite_master
                WHERE type='table' AND name=?""",
            (table,),
        ).fetchone()
        destination_schema = connection.execute(
            """SELECT COALESCE(sql, '')
                 FROM main.sqlite_master
                WHERE type='table' AND name=?""",
            (table,),
        ).fetchone()
        if source_schema is None or source_schema != destination_schema:
            return False

        source_columns = [
            row[1]
            for row in connection.execute(
                f'PRAGMA source_db.table_info("{table}")'
            )
        ]
        destination_columns = [
            row[1]
            for row in connection.execute(
                f'PRAGMA main.table_info("{table}")'
            )
        ]
        if not source_columns or source_columns != destination_columns:
            return False
        source_count = connection.execute(
            f'SELECT COUNT(*) FROM source_db."{table}"'
        ).fetchone()[0]
        destination_count = connection.execute(
            f'SELECT COUNT(*) FROM main."{table}"'
        ).fetchone()[0]
        if source_count != destination_count:
            return False

        quoted_columns = ", ".join(
            '"' + column.replace('"', '""') + '"'
            for column in source_columns
        )
        grouped_values = f"{quoted_columns}, COUNT(*)"
        for left, right in (("source_db", "main"), ("main", "source_db")):
            mismatch = connection.execute(
                "SELECT EXISTS("
                f"SELECT {grouped_values} "
                f'FROM {left}."{table}" '
                f"GROUP BY {quoted_columns} "
                "EXCEPT "
                f"SELECT {grouped_values} "
                f'FROM {right}."{table}" '
                f"GROUP BY {quoted_columns}"
                ")"
            ).fetchone()[0]
            if mismatch:
                return False

        source_indexes = connection.execute(
            """SELECT name, sql
                 FROM source_db.sqlite_master
                WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
                ORDER BY name""",
            (table,),
        ).fetchall()
        destination_indexes = connection.execute(
            """SELECT name, sql
                 FROM main.sqlite_master
                WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
                ORDER BY name""",
            (table,),
        ).fetchall()
        return source_indexes == destination_indexes


def replace_database(source: Path, destination: Path) -> None:
    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(destination) as destination_connection:
            destination_connection.execute("PRAGMA busy_timeout=30000")
            source_connection.backup(destination_connection)
    with sqlite3.connect(destination) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"Promoted database failed integrity: {destination}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise ValueError(
                f"Promoted database failed foreign keys: {destination}"
            )


def sync_generated_tables(source: Path, destination: Path) -> None:
    with sqlite3.connect(destination) as destination_connection:
        destination_connection.execute("PRAGMA busy_timeout=30000")
        destination_connection.execute(
            "ATTACH DATABASE ? AS generated_source",
            (str(source),),
        )
        destination_connection.execute("BEGIN IMMEDIATE")
        try:
            for table in GENERATED_AUCTION_TABLES:
                create_row = destination_connection.execute(
                    """SELECT sql
                         FROM generated_source.sqlite_master
                        WHERE type='table' AND name=?""",
                    (table,),
                ).fetchone()
                if create_row is None or not create_row[0]:
                    raise ValueError(f"Staged source is missing {table}")
                index_sql = [
                    row[0]
                    for row in destination_connection.execute(
                        """SELECT sql
                             FROM generated_source.sqlite_master
                            WHERE type='index' AND tbl_name=?
                                  AND sql IS NOT NULL
                            ORDER BY name""",
                        (table,),
                    )
                ]
                destination_connection.execute(
                    f'DROP TABLE IF EXISTS main."{table}"'
                )
                destination_connection.execute(create_row[0])
                destination_connection.execute(
                    f'INSERT INTO main."{table}" '
                    f'SELECT * FROM generated_source."{table}"'
                )
                for statement in index_sql:
                    destination_connection.execute(statement)
            destination_connection.commit()
        except Exception:
            destination_connection.rollback()
            raise
    with sqlite3.connect(destination) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Auction app database failed integrity after sync")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise ValueError("Auction app database failed foreign keys after sync")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-simulation", type=Path, required=True)
    parser.add_argument("--staged-validations", type=Path, required=True)
    parser.add_argument("--main-simulation", type=Path, required=True)
    parser.add_argument("--main-validations", type=Path, required=True)
    parser.add_argument("--auction-simulation", type=Path, required=True)
    parser.add_argument("--snake-simulation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.staged_simulation,
        args.staged_validations,
        args.main_simulation,
        args.main_validations,
        args.auction_simulation,
        args.snake_simulation,
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())

    auction_tables = database_tables(args.auction_simulation)
    auction_owned_tables = sorted(
        set(auction_tables) - set(GENERATED_AUCTION_TABLES)
    )
    app_owned_before = table_digests(
        args.auction_simulation,
        auction_owned_tables,
    )

    replace_database(
        args.staged_simulation.resolve(),
        args.main_simulation.resolve(),
    )
    replace_database(
        args.staged_validations.resolve(),
        args.main_validations.resolve(),
    )
    sync_generated_tables(
        args.staged_simulation.resolve(),
        args.auction_simulation.resolve(),
    )
    replace_database(
        args.staged_simulation.resolve(),
        args.snake_simulation.resolve(),
    )

    app_owned_after = table_digests(
        args.auction_simulation,
        auction_owned_tables,
    )
    if app_owned_before != app_owned_after:
        changed = sorted(
            table
            for table in auction_owned_tables
            if app_owned_before.get(table) != app_owned_after.get(table)
        )
        raise ValueError(f"Auction app-owned tables changed: {changed}")

    auction_mismatches = [
        table
        for table in GENERATED_AUCTION_TABLES
        if not table_content_matches(
            args.staged_simulation,
            args.auction_simulation,
            table,
        )
    ]
    if auction_mismatches:
        raise ValueError(
            "Auction generated tables differ from staging: "
            f"{auction_mismatches}"
        )

    staged_tables = database_tables(args.staged_simulation)
    snake_tables = database_tables(args.snake_simulation)
    if staged_tables != snake_tables:
        raise ValueError("Snake table inventory differs from staging")
    snake_mismatches = [
        table
        for table in staged_tables
        if not table_content_matches(
            args.staged_simulation,
            args.snake_simulation,
            table,
        )
    ]
    if snake_mismatches:
        raise ValueError(
            f"Snake tables differ from staging: {snake_mismatches}"
        )

    receipt = {
        "auction_generated_table_count": len(GENERATED_AUCTION_TABLES),
        "auction_app_owned_table_count": len(auction_owned_tables),
        "auction_app_owned_tables_unchanged": True,
        "auction_generated_tables_match_staging": True,
        "snake_tables_match_staging": True,
        "main_simulation_integrity": "ok",
        "main_validations_integrity": "ok",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
