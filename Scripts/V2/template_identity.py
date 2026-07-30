"""Attach canonical V2 player keys to weekly-template handoff rows."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from Scripts.V2.config import TEAM_MAP
from Scripts.V2.contracts import (
    SOURCE_STORED_SEASON_COLUMN,
    apply_source_row_exclusions,
    assert_no_source_row_exclusions,
    normalize_player_name,
    require_columns,
)


def _normalize_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    team = str(value).strip().upper()
    return TEAM_MAP.get(team, team) if team else None


def _candidate_maps(
    aliases: pd.DataFrame,
    confirmed_keys: set[str],
) -> tuple[
    dict[tuple[str, str, int], set[str]],
    dict[tuple[str, str, int, str], set[str]],
]:
    by_season: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    by_team: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    eligible = aliases[
        aliases["player_key"].notna()
        & aliases["normalized_name"].notna()
        & aliases["position"].notna()
        & aliases["season"].notna()
    ].copy()
    for row in eligible.itertuples(index=False):
        key = (
            str(row.normalized_name),
            str(row.position).upper(),
            int(row.season),
        )
        player_key = str(row.player_key)
        by_season[key].add(player_key)
        team = _normalize_team(row.team)
        if team:
            by_team[(*key, team)].add(player_key)

    # Retain all candidates. Confirmation priority is applied row by row so a
    # redundant provisional alias cannot make a confirmed match ambiguous.
    return dict(by_season), dict(by_team)


def _prefer_unique(
    candidates: Iterable[str],
    confirmed_keys: set[str],
) -> tuple[str | None, str | None]:
    values = set(candidates)
    confirmed = values.intersection(confirmed_keys)
    if len(confirmed) == 1:
        return next(iter(confirmed)), "confirmed_unique"
    if len(values) == 1:
        return next(iter(values)), "unique"
    return None, None


def attach_v2_player_keys(
    frame: pd.DataFrame,
    identity_database: Path,
    *,
    player_column: str = "player",
    position_column: str = "pos",
    season_column: str = "season",
    team_column: str = "team",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Resolve a template/player-map frame to permanent V2 player keys.

    Exact preseason aliases are preferred. Confirmed identities take
    precedence over redundant provisional aliases, and team is used only when
    a same-name/position/season key is otherwise ambiguous. A final confirmed
    identity career-window lookup handles sparse older aliases.
    """

    required = (
        player_column,
        position_column,
        season_column,
        team_column,
    )
    require_columns(frame, required, "weekly_template_identity_input")
    if not Path(identity_database).exists():
        raise FileNotFoundError(
            f"V2 identity database does not exist: {identity_database}"
        )

    with sqlite3.connect(identity_database) as connection:
        identity = pd.read_sql_query(
            """
            SELECT player_key, normalized_name, position, rookie_season,
                   last_season, draft_year, draft_team, latest_team,
                   identity_status
            FROM player_identity
            """,
            connection,
        )
        alias_columns = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("player_aliases")')
        }
        required_alias_columns = {
            "player_key",
            "normalized_name",
            "position",
            "team",
            "season",
            "source_table",
        }
        missing_alias_columns = sorted(
            required_alias_columns.difference(alias_columns)
        )
        if missing_alias_columns:
            raise ValueError(
                "V2 player_aliases lacks source provenance required for "
                "governed quarantine enforcement: "
                f"{missing_alias_columns}"
            )
        stored_season_select = (
            f", {SOURCE_STORED_SEASON_COLUMN}"
            if SOURCE_STORED_SEASON_COLUMN in alias_columns
            else ""
        )
        aliases = pd.read_sql_query(
            "SELECT player_key, normalized_name, position, team, season, "
            f"source_table{stored_season_select} FROM player_aliases",
            connection,
        )

    identity["position"] = identity["position"].astype("string").str.upper()
    aliases["position"] = aliases["position"].astype("string").str.upper()
    aliases["season"] = pd.to_numeric(
        aliases["season"], errors="coerce"
    ).astype("Int64")
    source_tables = aliases["source_table"].astype("string").str.strip()
    if (source_tables.isna() | source_tables.eq("")).any():
        raise ValueError(
            "V2 player_aliases contains rows without source_table provenance; "
            "governed quarantine enforcement cannot be verified"
        )
    aliases["source_table"] = source_tables
    aliases = apply_source_row_exclusions(
        aliases,
        "weekly template player_aliases",
    )
    assert_no_source_row_exclusions(
        aliases,
        "weekly template player_aliases after quarantine",
    )
    identity["rookie_season"] = pd.to_numeric(
        identity["rookie_season"], errors="coerce"
    )
    identity["last_season"] = pd.to_numeric(
        identity["last_season"], errors="coerce"
    )
    identity["draft_year"] = pd.to_numeric(
        identity["draft_year"], errors="coerce"
    )
    identity["draft_team_normalized"] = identity["draft_team"].map(
        _normalize_team
    )
    identity["latest_team_normalized"] = identity["latest_team"].map(
        _normalize_team
    )
    known_keys = set(identity["player_key"].dropna().astype(str))
    unknown_alias_keys = set(
        aliases["player_key"].dropna().astype(str)
    ).difference(known_keys)
    if unknown_alias_keys:
        raise ValueError(
            "V2 aliases reference unknown canonical player keys: "
            f"{sorted(unknown_alias_keys)[:10]}"
        )
    confirmed_keys = set(
        identity.loc[
            identity["identity_status"].eq("confirmed"), "player_key"
        ].astype(str)
    )
    alias_by_season, alias_by_team = _candidate_maps(
        aliases,
        confirmed_keys,
    )

    confirmed_identity: dict[tuple[str, str], pd.DataFrame] = {
        (str(name), str(position)): group.copy()
        for (name, position), group in identity[
            identity["identity_status"].eq("confirmed")
        ].groupby(["normalized_name", "position"], dropna=False)
    }

    output = frame.copy()
    normalized_names = output[player_column].map(normalize_player_name)
    positions = output[position_column].astype("string").str.upper()
    seasons = pd.to_numeric(output[season_column], errors="coerce")
    teams = output[team_column].map(_normalize_team)
    player_keys: list[object] = []
    methods: list[str] = []

    for name, position, season, team in zip(
        normalized_names,
        positions,
        seasons,
        teams,
    ):
        if not name or pd.isna(position) or pd.isna(season):
            player_keys.append(pd.NA)
            methods.append("unresolved_missing_identity_fields")
            continue
        season_value = int(season)
        base_key = (str(name), str(position), season_value)
        player_key, suffix = _prefer_unique(
            alias_by_season.get(base_key, set()),
            confirmed_keys,
        )
        method = f"alias_{suffix}" if suffix else None
        if player_key is None and team:
            player_key, suffix = _prefer_unique(
                alias_by_team.get((*base_key, team), set()),
                confirmed_keys,
            )
            method = f"alias_team_{suffix}" if suffix else None

        if player_key is None:
            candidates = confirmed_identity.get(
                (str(name), str(position)),
                pd.DataFrame(),
            )
            if not candidates.empty:
                career_start = (
                    candidates["rookie_season"]
                    .combine_first(candidates["draft_year"])
                    .fillna(-1)
                )
                career_end = candidates["last_season"].fillna(season_value)
                active = candidates[
                    career_start.le(season_value)
                    & career_end.ge(season_value)
                ]
                if len(active) == 1:
                    player_key = str(active.iloc[0]["player_key"])
                    method = "identity_active_window"
                elif team:
                    team_match = active[
                        active["draft_team_normalized"].eq(team)
                        | active["latest_team_normalized"].eq(team)
                    ]
                    if len(team_match) == 1:
                        player_key = str(team_match.iloc[0]["player_key"])
                        method = "identity_active_window_team"
                    else:
                        team_match = candidates[
                            candidates["draft_team_normalized"].eq(team)
                            | candidates["latest_team_normalized"].eq(team)
                        ]
                        if len(team_match) == 1:
                            player_key = str(
                                team_match.iloc[0]["player_key"]
                            )
                            method = "identity_team"
                if player_key is None and len(candidates) == 1:
                    player_key = str(candidates.iloc[0]["player_key"])
                    method = "identity_unique"

        player_keys.append(player_key if player_key is not None else pd.NA)
        methods.append(method or "unresolved_ambiguous_identity")

    output["player_key"] = pd.Series(player_keys, index=output.index).astype(
        "string"
    )
    output["player_key_match_method"] = methods
    if require_complete and output["player_key"].isna().any():
        unresolved = output.loc[
            output["player_key"].isna(),
            [
                player_column,
                position_column,
                team_column,
                season_column,
                "player_key_match_method",
            ],
        ].drop_duplicates()
        raise ValueError(
            "Weekly template identities did not resolve to V2 player_key:\n"
            f"{unresolved.head(30).to_string(index=False)}"
        )
    return output
