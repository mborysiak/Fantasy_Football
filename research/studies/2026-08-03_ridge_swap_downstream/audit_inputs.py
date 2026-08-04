"""Read-only audit of the live V2 inputs used by the Ridge swap study."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
TABLES = (
    "locked_candidate_runs",
    "locked_selected_hyperparameters",
    "locked_whole_season_predictions",
    "locked_2026_shadow_predictions",
)


def main() -> None:
    audit: dict[str, object] = {}
    for league, database in DATABASES.items():
        with sqlite3.connect(
            f"file:{database.resolve().as_posix()}?mode=ro", uri=True
        ) as connection:
            available = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            league_audit: dict[str, object] = {
                "database": str(database),
                "tables": {},
            }
            for table in TABLES:
                if table not in available:
                    league_audit["tables"][table] = {"present": False}
                    continue
                columns = [
                    row[1]
                    for row in connection.execute(f"PRAGMA table_info({table})")
                ]
                entry: dict[str, object] = {
                    "present": True,
                    "rows": connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0],
                    "columns": columns,
                }
                if table in {
                    "locked_whole_season_predictions",
                    "locked_2026_shadow_predictions",
                }:
                    group_columns = [
                        column
                        for column in ("target_name", "method", "model_name")
                        if column in columns
                    ]
                    select_columns = list(group_columns)
                    labels = list(group_columns)
                    if "season" in columns:
                        select_columns.extend(
                            ("MIN(season)", "MAX(season)")
                        )
                        labels.extend(("min_season", "max_season"))
                    select_columns.append("COUNT(*)")
                    labels.append("rows")
                    group_sql = ", ".join(group_columns)
                    order_sql = f" ORDER BY {group_sql}" if group_sql else ""
                    query = (
                        f"SELECT {', '.join(select_columns)} FROM {table}"
                        + (f" GROUP BY {group_sql}" if group_sql else "")
                        + order_sql
                    )
                    entry["summary"] = [
                        dict(zip(labels, row))
                        for row in connection.execute(query)
                    ]
                elif table == "locked_candidate_runs":
                    entry["rows_detail"] = [
                        dict(zip(columns, row))
                        for row in connection.execute(
                            f"SELECT * FROM {table} ORDER BY rowid"
                        )
                    ]
                league_audit["tables"][table] = entry
            audit[league] = league_audit
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
