"""Active-keeper market state shared by the Sequential validation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_active_keeper_market(
    conn,
    sim,
    *,
    year: int,
    league: str,
    salary_source: str,
    owned_salary_map: dict[str, float],
):
    """Return the live-App keeper exclusions, spend, and canonical labels."""
    if salary_source not in {"predicted", "actual"}:
        raise ValueError(f"Unknown keeper salary source: {salary_source}")
    keepers = pd.read_sql_query(
        """
        SELECT player_key, player AS source_player, keeper_salary
        FROM League_Keepers
        WHERE year = :year AND league = :league
        ORDER BY player_key
        """,
        conn,
        params={"year": int(year), "league": league},
    )
    if keepers.empty:
        raise ValueError(f"No active keepers found for {year} {league}.")
    if keepers.player_key.isna().any() or keepers.player_key.duplicated().any():
        raise ValueError("Active keepers require unique canonical player keys.")

    canonical = sim.player_data[["player_key", "player", "salary"]].copy()
    canonical = canonical.rename(columns={
        "player": "canonical_player",
        "salary": "actual_salary",
    })
    if canonical.player_key.duplicated().any():
        raise ValueError("Simulation player keys must be unique.")
    keepers = keepers.merge(
        canonical,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    if keepers.canonical_player.isna().any():
        labels = keepers.loc[
            keepers.canonical_player.isna(), "source_player"
        ].tolist()
        raise ValueError(
            "Active keepers missing from the simulation player pool: "
            + ", ".join(map(str, labels))
        )

    salary_column = (
        "actual_salary" if salary_source == "actual" else "keeper_salary"
    )
    keeper_salaries = pd.to_numeric(keepers[salary_column], errors="coerce")
    if keeper_salaries.isna().any():
        labels = keepers.loc[
            keeper_salaries.isna(), "canonical_player"
        ].tolist()
        raise ValueError(
            "Active keepers missing market-state salaries: "
            + ", ".join(map(str, labels))
        )
    keepers["market_salary"] = keeper_salaries.astype(float)

    keeper_players = set(keepers.canonical_player)
    owned_players = set(owned_salary_map)
    missing_owned = sorted(owned_players - keeper_players)
    if missing_owned:
        raise ValueError(
            "Owned replay players are not active league keepers: "
            + ", ".join(missing_owned)
        )
    owned_keeper_salaries = keepers.set_index("canonical_player").market_salary
    mismatched_owned = sorted(
        player
        for player, salary in owned_salary_map.items()
        if not np.isclose(float(salary), float(owned_keeper_salaries[player]))
    )
    if mismatched_owned:
        raise ValueError(
            "Owned replay salaries disagree with the active keeper market: "
            + ", ".join(mismatched_owned)
        )

    return {
        "salary_source": salary_source,
        "keeper_count": int(len(keepers)),
        "keeper_spend": float(keepers.market_salary.sum()),
        "keeper_players": tuple(sorted(keeper_players)),
        "owned_keeper_players": tuple(sorted(owned_players)),
        "unavailable_keeper_players": tuple(sorted(
            keeper_players - owned_players
        )),
    }
