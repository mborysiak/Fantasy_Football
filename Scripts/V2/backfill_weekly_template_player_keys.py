"""Backfill canonical V2 player keys into generated weekly handoff tables."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from Scripts.V2.config import OUTPUT_DB_PATH, REPO_ROOT
from Scripts.V2.contracts import publish_tables_atomic
from Scripts.V2.template_identity import attach_v2_player_keys


SIMULATION_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
TEMPLATE_TABLE = "Best_Ball_Weekly_Templates"
PLAYER_MAP_TABLE = "Best_Ball_Weekly_Player_Map"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation-db",
        type=Path,
        default=SIMULATION_DB_PATH,
    )
    parser.add_argument(
        "--identity-db",
        type=Path,
        default=OUTPUT_DB_PATH,
    )
    return parser.parse_args()


def backfill_player_keys(
    simulation_database: Path = SIMULATION_DB_PATH,
    identity_database: Path = OUTPUT_DB_PATH,
) -> dict[str, object]:
    with sqlite3.connect(simulation_database) as connection:
        templates = pd.read_sql_query(
            f'SELECT * FROM "{TEMPLATE_TABLE}"',
            connection,
        )
        player_map = pd.read_sql_query(
            f'SELECT * FROM "{PLAYER_MAP_TABLE}"',
            connection,
        )

    templates = attach_v2_player_keys(
        templates,
        identity_database,
        season_column="season",
    )
    player_map = attach_v2_player_keys(
        player_map,
        identity_database,
        season_column="year",
    )
    if templates.duplicated(["league", "player_key", "season"]).any():
        raise ValueError(
            "Template player keys are not unique by league/player/season"
        )
    if player_map.duplicated(
        ["version", "dataset", "player_key", "year"]
    ).any():
        raise ValueError(
            "Current player-map keys are not unique by version/dataset/player/year"
        )

    publish_tables_atomic(
        simulation_database,
        {
            TEMPLATE_TABLE: templates,
            PLAYER_MAP_TABLE: player_map,
        },
    )
    return {
        "simulation_database": str(simulation_database.resolve()),
        "identity_database": str(identity_database.resolve()),
        "template_rows": len(templates),
        "template_key_coverage": float(templates["player_key"].notna().mean()),
        "player_map_rows": len(player_map),
        "player_map_key_coverage": float(
            player_map["player_key"].notna().mean()
        ),
        "template_match_methods": (
            templates["player_key_match_method"].value_counts().to_dict()
        ),
        "player_map_match_methods": (
            player_map["player_key_match_method"].value_counts().to_dict()
        ),
    }


def main() -> None:
    args = parse_args()
    print(
        pd.Series(
            backfill_player_keys(
                simulation_database=args.simulation_db,
                identity_database=args.identity_db,
            )
        ).to_string()
    )


if __name__ == "__main__":
    main()

