"""Publish a completed auction's realized prices for Auction app hindsight mode."""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Scripts.V2.production_handoff import (
    load_identity_frames,
    resolve_source_player_keys,
)


SALARY_RESID_COLUMNS = (
    "salary_resid_5",
    "salary_resid_10",
    "salary_resid_25",
    "salary_resid_75",
    "salary_resid_90",
    "salary_resid_95",
)
ACTUAL_METHOD_VERSION = "actual_draft_results_v1"
DEFAULT_EXPECTED_POOL_ROWS = 12 * 13


def build_actual_salary_slice(
    simulation_database: Path,
    v2_database: Path,
    *,
    year: int,
    league: str,
    expected_pool_rows: int = DEFAULT_EXPECTED_POOL_ROWS,
) -> pd.DataFrame:
    """Return canonical, deterministic actual prices for drafted offensive players."""

    league = str(league).strip().lower()
    if league not in {"beta", "nv"}:
        raise ValueError(f"Actual salary publication supports beta/nv, found {league!r}.")

    with closing(sqlite3.connect(simulation_database)) as connection:
        actual = pd.read_sql_query(
            """
            SELECT player, actual_salary
            FROM Actual_Salaries
            WHERE year=? AND league=?
            """,
            connection,
            params=(int(year), league),
        )
        canonical = pd.read_sql_query(
            """
            SELECT player_key, player, pos
            FROM Final_Predictions_Resid
            WHERE year=? AND version=? AND dataset='final_ensemble'
            """,
            connection,
            params=(int(year), league),
        )

    if actual.empty:
        raise ValueError(f"No Actual_Salaries rows exist for {year} {league}.")
    if canonical.empty:
        raise ValueError(
            f"No final_ensemble projection surface exists for {year} {league}."
        )
    if canonical.player_key.isna().any() or canonical.player_key.duplicated().any():
        raise ValueError("Canonical projection rows require unique non-null player keys.")

    aliases, identities = load_identity_frames(v2_database)
    resolved = resolve_source_player_keys(
        actual,
        aliases,
        identities,
        year=int(year),
        source_name=f"Actual_Salaries {year} {league}",
        require_complete=False,
    )
    canonical_keys = set(canonical.player_key.astype(str))
    resolved = resolved[
        resolved.player_key.notna()
        & resolved.player_key.astype(str).isin(canonical_keys)
    ].copy()
    if resolved.player_key.duplicated().any():
        duplicates = resolved.loc[
            resolved.player_key.duplicated(keep=False),
            ["player", "player_key"],
        ].to_dict("records")
        raise ValueError(
            "Actual salary rows resolve multiple times to a canonical player: "
            f"{duplicates[:10]}"
        )
    if len(resolved) != int(expected_pool_rows):
        raise ValueError(
            f"Actual {year} {league} offensive pool has {len(resolved)} canonical "
            f"rows; expected {expected_pool_rows}."
        )

    output = resolved[["player_key", "actual_salary"]].merge(
        canonical[["player_key", "player"]],
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    output["salary"] = pd.to_numeric(output.actual_salary, errors="raise")
    if (
        output.salary.isna().any()
        or not output.salary.map(math.isfinite).all()
        or (output.salary < 0).any()
    ):
        raise ValueError("Actual salaries must be finite non-negative values.")
    output["year"] = int(year)
    output["league"] = f"{league}_actual"
    output["std_dev"] = 0.0
    output["min_score"] = output.salary
    output["max_score"] = output.salary
    for column in SALARY_RESID_COLUMNS:
        output[column] = 0.0
    output["salary_population_source"] = "actual_draft_results"
    output["ensemble_uncertainty_feature_source"] = "not_applicable_actual"
    output["salary_method_version"] = ACTUAL_METHOD_VERSION

    return output[[
        "player",
        "salary",
        "year",
        "league",
        "std_dev",
        "min_score",
        "max_score",
        *SALARY_RESID_COLUMNS,
        "player_key",
        "salary_population_source",
        "ensemble_uncertainty_feature_source",
        "salary_method_version",
    ]].sort_values("player_key").reset_index(drop=True)


def backup_database(database: Path) -> Path:
    backup_dir = database.parent / "DB_Versioning"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_dir / f"{database.stem}_before_actual_{timestamp}.sqlite3"
    with (
        closing(sqlite3.connect(database)) as source,
        closing(sqlite3.connect(backup)) as target,
    ):
        source.backup(target)
    return backup


def publish_actual_salary_slice(
    output: pd.DataFrame,
    simulation_database: Path,
) -> Path:
    """Atomically replace one ``Salaries_Pred`` actual-results slice."""

    table_columns: list[str]
    with closing(sqlite3.connect(simulation_database)) as connection:
        table_columns = [
            str(row[1])
            for row in connection.execute('PRAGMA table_info("Salaries_Pred")')
        ]
    missing = sorted(set(output.columns).difference(table_columns))
    if missing:
        raise ValueError(f"Salaries_Pred lacks actual-publication columns: {missing}")

    backup = backup_database(simulation_database)
    columns = list(output.columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    league = str(output.league.iloc[0])
    year = int(output.year.iloc[0])
    records = [tuple(row) for row in output.itertuples(index=False, name=None)]

    with closing(sqlite3.connect(simulation_database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM Salaries_Pred WHERE year=? AND league=?",
            (year, league),
        )
        connection.executemany(
            f'INSERT INTO Salaries_Pred ({quoted_columns}) VALUES ({placeholders})',
            records,
        )
        saved = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT player_key),
                   SUM(salary), SUM(std_dev),
                   SUM(ABS(min_score - salary)),
                   SUM(ABS(max_score - salary))
            FROM Salaries_Pred
            WHERE year=? AND league=?
            """,
            (year, league),
        ).fetchone()
        expected = (
            len(output),
            output.player_key.nunique(),
            float(output.salary.sum()),
            0.0,
            0.0,
            0.0,
        )
        if tuple(float(value or 0) for value in saved) != tuple(
            float(value) for value in expected
        ):
            raise ValueError(
                f"Saved actual salary slice failed verification: {saved} != {expected}"
            )
        connection.commit()
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish actual auction results as a deterministic app salary slice."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--league", choices=("beta", "nv"), required=True)
    parser.add_argument(
        "--simulation-database",
        type=Path,
        default=Path("Data/Databases/Simulation.sqlite3"),
    )
    parser.add_argument("--v2-database", type=Path)
    parser.add_argument(
        "--expected-pool-rows",
        type=int,
        default=DEFAULT_EXPECTED_POOL_ROWS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation_database = args.simulation_database.resolve()
    v2_database = (
        args.v2_database.resolve()
        if args.v2_database is not None
        else simulation_database.with_name(f"Projection_V2_{args.league}.sqlite3")
    )
    output = build_actual_salary_slice(
        simulation_database,
        v2_database,
        year=args.year,
        league=args.league,
        expected_pool_rows=args.expected_pool_rows,
    )
    print(
        f"Validated {len(output)} {args.year} {args.league} actual salary rows; "
        f"total=${output.salary.sum():,.0f}."
    )
    if args.dry_run:
        print("Dry run: Salaries_Pred was not changed.")
        return
    backup = publish_actual_salary_slice(output, simulation_database)
    print(
        f"Published {args.year} {args.league}_actual to "
        f"{simulation_database}. Backup: {backup}"
    )


if __name__ == "__main__":
    main()
