"""Normalize preseason projection and market source values for V2."""

from __future__ import annotations

import re
import sqlite3
import warnings
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from Scripts.V2.config import (
    CANDIDATE_SOURCE_TABLES,
    MARKET_VALUE_SPECS,
    POSITIONS,
    PROJECTION_THROUGH_SEASON,
    PROJECTION_VALUE_SPECS,
    SOURCE_DB_PATH,
    START_SEASON,
    TEAM_MAP,
)
from Scripts.V2.contracts import (
    MARKET_VALUE_COLUMNS,
    PROJECTION_VALUE_COLUMNS,
    PROJECTION_VALUE_METRICS,
    align_columns,
    configured_scoring,
    normalize_player_name,
    normalize_source_position,
    require_columns,
    table_exists,
)


def _clean_text(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _normalize_source_values(
    raw: pd.DataFrame,
    identity_spec: dict[str, object],
) -> pd.Series:
    source_column = identity_spec.get("source_column")
    if isinstance(source_column, str) and source_column in raw:
        prefix = str(identity_spec.get("source_prefix", ""))
        values = (
            raw[source_column]
            .astype("string")
            .fillna("unknown")
            .str.lower()
            .map(lambda value: re.sub(r"[^a-z0-9]+", "_", str(value)))
            .str.strip("_")
        )
        return prefix + values
    return pd.Series(
        str(identity_spec["source"]),
        index=raw.index,
        dtype="string",
    )


def _standardize_identity_rows(
    raw: pd.DataFrame,
    table: str,
    identity_spec: dict[str, object],
) -> pd.DataFrame:
    standard = pd.DataFrame(index=raw.index)
    standard["source_table"] = table
    standard["source"] = _normalize_source_values(raw, identity_spec)
    for target in (
        "source_player_id",
        "player",
        "position",
        "team",
        "season",
    ):
        source_column = identity_spec.get(target)
        if isinstance(source_column, str) and source_column in raw:
            standard[target] = raw[source_column]
        else:
            standard[target] = pd.NA
    if "position_value" in identity_spec:
        standard["position"] = identity_spec["position_value"]
    standard["source_player_id"] = standard["source_player_id"].map(_clean_text)
    standard["normalized_name"] = standard["player"].map(normalize_player_name)
    standard["position"] = standard["position"].map(normalize_source_position)
    standard["team"] = standard["team"].map(_clean_text)
    standard["season"] = pd.to_numeric(
        standard["season"], errors="coerce"
    ).astype("Int64")
    return standard


def _unique_lookup(
    aliases: pd.DataFrame,
    columns: list[str],
) -> dict[tuple[object, ...], str]:
    valid = aliases.dropna(subset=columns).copy()
    if valid.empty:
        return {}
    candidates: dict[tuple[object, ...], set[str]] = defaultdict(set)
    selected = valid.loc[:, [*columns, "player_key"]]
    for row in selected.itertuples(index=False, name=None):
        key = tuple(row[:-1])
        candidates[key].add(str(row[-1]))
    return {
        key: next(iter(values))
        for key, values in candidates.items()
        if len(values) == 1
    }


def resolve_source_rows(
    identity_rows: pd.DataFrame,
    player_aliases: pd.DataFrame,
) -> pd.Series:
    """Resolve raw table rows back to the exact Milestone 1 alias identity."""
    required = (
        "source_table",
        "source",
        "source_player_id",
        "normalized_name",
        "position",
        "team",
        "season",
    )
    require_columns(identity_rows, required, "source identity rows")
    require_columns(
        player_aliases,
        ("player_key", *required),
        "player_aliases",
    )
    aliases = player_aliases[
        player_aliases["source_table"].isin(
            identity_rows["source_table"].dropna().unique()
        )
    ].copy()
    for column in ("source_player_id", "normalized_name", "position", "team"):
        aliases[column] = aliases[column].map(_clean_text)
    aliases["season"] = pd.to_numeric(
        aliases["season"], errors="coerce"
    ).astype("Int64")

    id_columns = ["source_table", "source", "source_player_id"]
    full_columns = [
        "source_table",
        "source",
        "normalized_name",
        "position",
        "season",
        "team",
    ]
    position_columns = [
        "source_table",
        "source",
        "normalized_name",
        "position",
        "season",
    ]
    name_columns = [
        "source_table",
        "source",
        "normalized_name",
        "season",
    ]
    lookups = [
        (_unique_lookup(aliases, id_columns), id_columns),
        (_unique_lookup(aliases, full_columns), full_columns),
        (_unique_lookup(aliases, position_columns), position_columns),
        (_unique_lookup(aliases, name_columns), name_columns),
    ]

    resolved: list[object] = []
    identity_columns = list(identity_rows.columns)
    for row in identity_rows.itertuples(index=False, name=None):
        record = dict(zip(identity_columns, row))
        player_key: object = pd.NA
        for lookup, columns in lookups:
            values = tuple(record[column] for column in columns)
            if any(pd.isna(value) for value in values):
                continue
            candidate = lookup.get(values)
            if candidate is not None:
                player_key = candidate
                break
        resolved.append(player_key)
    return pd.Series(resolved, index=identity_rows.index, dtype="string")


def _requested_columns(
    identity_spec: dict[str, object],
    value_spec: dict[str, object],
) -> set[str]:
    requested: set[str] = set()
    for target in (
        "source_player_id",
        "player",
        "position",
        "team",
        "season",
    ):
        column = identity_spec.get(target)
        if isinstance(column, str):
            requested.add(column)
    source_column = identity_spec.get("source_column")
    if isinstance(source_column, str):
        requested.add(source_column)
    for column in value_spec.get("metrics", {}).values():
        requested.add(str(column))
    return requested


def _read_resolved_value_rows(
    connection: sqlite3.Connection,
    table: str,
    value_spec: dict[str, object],
    player_aliases: pd.DataFrame,
    start_season: int,
    projection_through_season: int,
) -> tuple[pd.DataFrame, int, int]:
    identity_spec = CANDIDATE_SOURCE_TABLES[table]
    available = {
        row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
    }
    requested = sorted(
        _requested_columns(identity_spec, value_spec).intersection(available)
    )
    query_columns = ", ".join(f'"{column}"' for column in requested)
    raw = pd.read_sql_query(
        f'SELECT {query_columns} FROM "{table}"',
        connection,
    )
    identity_rows = _standardize_identity_rows(raw, table, identity_spec)
    in_window = (
        identity_rows["season"].ge(start_season)
        & identity_rows["season"].le(projection_through_season)
    )
    has_position_column = isinstance(identity_spec.get("position"), str)
    if has_position_column:
        in_window &= identity_rows["position"].isin(POSITIONS)
    elif not identity_spec.get("allow_missing_position"):
        in_window &= identity_rows["position"].isin(POSITIONS)
    raw = raw[in_window].reset_index(drop=True)
    identity_rows = identity_rows[in_window].reset_index(drop=True)
    identity_rows["player_key"] = resolve_source_rows(
        identity_rows,
        player_aliases,
    )
    resolved = identity_rows["player_key"].notna()
    output = identity_rows[resolved].reset_index(drop=True)
    raw = raw[resolved].reset_index(drop=True)
    for target, source_column in value_spec.get("metrics", {}).items():
        source_column = str(source_column)
        if source_column in raw.columns:
            output[target] = pd.to_numeric(
                raw[source_column], errors="coerce"
            )
        else:
            output[target] = np.nan
    return output, len(identity_rows), int(resolved.sum())


def _deterministic_mode(values: Iterable[object]) -> object:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return pd.NA
    counts = pd.Series(cleaned).value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def _season_context_maps(
    player_aliases: pd.DataFrame,
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    aliases = player_aliases.copy()
    aliases["season"] = pd.to_numeric(
        aliases["season"], errors="coerce"
    ).astype("Int64")
    aliases["position"] = aliases["position"].astype("string").str.upper()
    aliases["team"] = aliases["team"].map(_clean_text)
    positions: dict[tuple[str, int], str] = {}
    teams: dict[tuple[str, int], str] = {}
    for (player_key, season), group in aliases[
        aliases["season"].notna()
    ].groupby(["player_key", "season"]):
        position = _deterministic_mode(
            group.loc[group["position"].isin(POSITIONS), "position"]
        )
        team = _deterministic_mode(group["team"])
        key = (str(player_key), int(season))
        if pd.notna(position):
            positions[key] = str(position)
        if pd.notna(team):
            teams[key] = str(team)
    return positions, teams


def _score_projection_values(
    frame: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    scored = frame.copy()
    rules = configured_scoring(league)
    required_by_position = {
        "QB": (
            "passing_yards",
            "passing_tds",
            "interceptions",
            "rushing_yards",
            "rushing_tds",
        ),
        "RB": (
            "rushing_yards",
            "rushing_tds",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        ),
        "WR": ("receptions", "receiving_yards", "receiving_tds"),
        "TE": ("receptions", "receiving_yards", "receiving_tds"),
    }

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(scored[column], errors="coerce")

    # A provider may omit one required component even when its other raw
    # projections are useful (historically, FFA omits receptions). Fill that
    # one component only when at least two other providers supply a
    # player-season value. Rows missing multiple required components remain
    # unscored; provider-published fantasy totals are retained as raw audit
    # fields but never substituted for configured scoring.
    missing_required = pd.Series(0, index=scored.index, dtype=int)
    for position, columns in required_by_position.items():
        position_mask = scored["position"].eq(position)
        missing_required.loc[position_mask] = scored.loc[
            position_mask, list(columns)
        ].isna().sum(axis=1)
    imputed_count = pd.Series(0, index=scored.index, dtype=int)
    keys = ["player_key", "season"]
    for column in sorted(
        {column for columns in required_by_position.values() for column in columns}
    ):
        values = numeric(column)
        donor_median = values.groupby(
            [scored[key] for key in keys], dropna=False
        ).transform("median")
        donor_count = values.notna().groupby(
            [scored[key] for key in keys], dropna=False
        ).transform("sum")
        required_mask = pd.Series(False, index=scored.index)
        for position, columns in required_by_position.items():
            if column in columns:
                required_mask |= scored["position"].eq(position)
        eligible = (
            required_mask
            & values.isna()
            & missing_required.eq(1)
            & donor_count.ge(2)
            & donor_median.notna()
        )
        scored.loc[eligible, column] = donor_median.loc[eligible]
        imputed_count.loc[eligible] = 1
    scored["configured_points_imputed_component_count"] = imputed_count

    scored["passing_points"] = (
        numeric("passing_yards").fillna(0)
        * rules["passing"].get("pass_yards_gained_sum", 0.0)
        + numeric("passing_tds").fillna(0)
        * rules["passing"].get("pass_pass_touchdown_sum", 0.0)
        + numeric("interceptions").fillna(0)
        * rules["passing"].get("pass_interception_sum", 0.0)
        + numeric("sacks").fillna(0)
        * rules["passing"].get("sack_sum", 0.0)
    )
    scored["rushing_points"] = (
        numeric("rushing_yards").fillna(0)
        * rules["rushing"].get("rush_yards_gained_sum", 0.0)
        + numeric("rushing_tds").fillna(0)
        * rules["rushing"].get("rush_rush_touchdown_sum", 0.0)
    )
    scored["receiving_points"] = (
        numeric("receptions").fillna(0)
        * rules["receiving"].get("rec_complete_pass_sum", 0.0)
        + numeric("receiving_yards").fillna(0)
        * rules["receiving"].get("rec_yards_gained_sum", 0.0)
        + numeric("receiving_tds").fillna(0)
        * rules["receiving"].get("rec_pass_touchdown_sum", 0.0)
    )
    scored["configured_projected_points"] = scored[
        ["passing_points", "rushing_points", "receiving_points"]
    ].sum(axis=1)

    complete: list[int] = []
    for row in scored.itertuples(index=False):
        required = required_by_position.get(str(row.position), ())
        complete.append(
            int(
                bool(required)
                and all(pd.notna(getattr(row, column)) for column in required)
            )
        )
    scored["configured_points_complete"] = complete
    use_configured = scored["configured_points_complete"].eq(1)
    incomplete = ~use_configured
    scored.loc[
        incomplete,
        [
            "passing_points",
            "rushing_points",
            "receiving_points",
            "configured_projected_points",
        ],
    ] = np.nan
    scored["provider_projected_points"] = scored[
        "configured_projected_points"
    ].where(use_configured)
    used_imputation = (
        use_configured
        & scored["configured_points_imputed_component_count"].gt(0)
    )
    scored["points_method"] = np.select(
        [used_imputation, use_configured],
        ["configured_components_imputed", "configured_components"],
        default="insufficient",
    )
    schedule_games = np.where(scored["season"].ge(2021), 17.0, 16.0)
    scored["provider_points_per_team_game"] = (
        scored["provider_projected_points"] / schedule_games
    )
    projected_games = pd.to_numeric(
        scored["projected_games"], errors="coerce"
    )
    scored["provider_points_per_projected_game"] = (
        scored["provider_projected_points"] / projected_games.where(
            projected_games.gt(0)
        )
    )
    return scored


def _add_provider_room_context(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["provider_team_points"] = np.nan
    output["provider_room_points"] = np.nan
    output["provider_room_share"] = np.nan
    output["provider_room_rank"] = np.nan
    output["provider_room_gap_to_leader"] = np.nan
    output["provider_room_hhi"] = np.nan
    standardized_points = pd.to_numeric(
        output["configured_projected_points"], errors="coerce"
    ).where(output["configured_points_complete"].eq(1))
    eligible = (
        output["team"].notna()
        & output["position"].isin(POSITIONS)
        & standardized_points.notna()
    )
    context = output[eligible].copy()
    if context.empty:
        return output
    context["_standardized_projected_points"] = standardized_points.loc[
        eligible
    ]
    team_keys = ["season", "provider", "team"]
    room_keys = [*team_keys, "position"]
    context["provider_team_points"] = context.groupby(team_keys)[
        "_standardized_projected_points"
    ].transform("sum")
    context["provider_room_points"] = context.groupby(room_keys)[
        "_standardized_projected_points"
    ].transform("sum")
    context["provider_room_share"] = (
        context["_standardized_projected_points"]
        / context["provider_room_points"].where(
            context["provider_room_points"].gt(0)
        )
    )
    context["provider_room_rank"] = context.groupby(room_keys)[
        "_standardized_projected_points"
    ].rank(method="min", ascending=False)
    room_leader = context.groupby(room_keys)[
        "_standardized_projected_points"
    ].transform("max")
    context["provider_room_gap_to_leader"] = (
        room_leader - context["_standardized_projected_points"]
    )
    context["provider_room_hhi"] = context.groupby(room_keys)[
        "provider_room_share"
    ].transform(lambda values: float(np.square(values).sum()))
    context_columns = [
        "provider_team_points",
        "provider_room_points",
        "provider_room_share",
        "provider_room_rank",
        "provider_room_gap_to_leader",
        "provider_room_hhi",
    ]
    output.loc[context.index, context_columns] = context[context_columns]
    return output


def build_projection_values(
    player_aliases: pd.DataFrame,
    league: str,
    run_id: str,
    source_database: Path = SOURCE_DB_PATH,
    start_season: int = START_SEASON,
    projection_through_season: int = PROJECTION_THROUGH_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    with sqlite3.connect(source_database) as connection:
        for table, spec in PROJECTION_VALUE_SPECS.items():
            if not table_exists(connection, table):
                continue
            resolved, input_rows, resolved_rows = _read_resolved_value_rows(
                connection,
                table,
                spec,
                player_aliases,
                start_season,
                projection_through_season,
            )
            resolved["provider"] = str(spec["provider"])
            raw_rows.append(resolved)
            audit_rows.append(
                {
                    "source_table": table,
                    "source_kind": "projection_values",
                    "input_rows": input_rows,
                    "resolved_rows": resolved_rows,
                    "resolution_rate": (
                        resolved_rows / input_rows if input_rows else np.nan
                    ),
                }
            )
    if not raw_rows:
        return (
            pd.DataFrame(columns=PROJECTION_VALUE_COLUMNS),
            pd.DataFrame(audit_rows),
        )
    raw_values = pd.concat(raw_rows, ignore_index=True)
    for metric in PROJECTION_VALUE_METRICS:
        if metric not in raw_values:
            raw_values[metric] = np.nan

    keys = ["player_key", "season", "provider"]
    grouped = raw_values.groupby(keys, sort=True, dropna=False)
    metadata = grouped.agg(
        sources=("source", lambda values: "|".join(sorted(set(values)))),
        source_tables=(
            "source_table",
            lambda values: "|".join(sorted(set(values))),
        ),
        position=("position", _deterministic_mode),
        team=("team", _deterministic_mode),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        metrics = grouped[list(PROJECTION_VALUE_METRICS)].median()
    values = metadata.join(metrics).reset_index()
    values["season"] = values["season"].astype(int)
    values["position"] = values["position"].astype("string").str.upper()
    position_map, team_map = _season_context_maps(player_aliases)
    context_keys = [
        (str(player_key), int(season))
        for player_key, season in values[
            ["player_key", "season"]
        ].itertuples(index=False, name=None)
    ]
    missing_position = ~values["position"].isin(POSITIONS)
    fallback_position = pd.Series(
        [position_map.get(key, pd.NA) for key in context_keys],
        index=values.index,
    )
    values.loc[missing_position, "position"] = fallback_position[
        missing_position
    ]
    values = values[values["position"].isin(POSITIONS)].copy()
    context_keys = [
        (str(player_key), int(season))
        for player_key, season in values[
            ["player_key", "season"]
        ].itertuples(index=False, name=None)
    ]
    missing_team = values["team"].isna()
    fallback_team = pd.Series(
        [team_map.get(key, pd.NA) for key in context_keys],
        index=values.index,
    )
    values.loc[missing_team, "team"] = fallback_team[missing_team]
    values["team"] = (
        values["team"].astype("string").str.upper().map(TEAM_MAP)
    )
    values = _score_projection_values(values, league)
    values = _add_provider_room_context(values)
    values["metric_count"] = values[list(PROJECTION_VALUE_METRICS)].notna().sum(
        axis=1
    )
    values["run_id"] = run_id
    values = align_columns(
        values,
        PROJECTION_VALUE_COLUMNS,
        "player_season_projection_values",
    )
    return (
        values.sort_values(["season", "player_key", "provider"]).reset_index(
            drop=True
        ),
        pd.DataFrame(audit_rows),
    )


def build_market_values(
    player_aliases: pd.DataFrame,
    run_id: str,
    source_database: Path = SOURCE_DB_PATH,
    start_season: int = START_SEASON,
    projection_through_season: int = PROJECTION_THROUGH_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    with sqlite3.connect(source_database) as connection:
        for table, spec in MARKET_VALUE_SPECS.items():
            if not table_exists(connection, table):
                continue
            resolved, input_rows, resolved_rows = _read_resolved_value_rows(
                connection,
                table,
                spec,
                player_aliases,
                start_season,
                projection_through_season,
            )
            raw_rows.append(resolved)
            audit_rows.append(
                {
                    "source_table": table,
                    "source_kind": "market_values",
                    "input_rows": input_rows,
                    "resolved_rows": resolved_rows,
                    "resolution_rate": (
                        resolved_rows / input_rows if input_rows else np.nan
                    ),
                }
            )
    if not raw_rows:
        return (
            pd.DataFrame(columns=MARKET_VALUE_COLUMNS),
            pd.DataFrame(audit_rows),
        )
    raw_values = pd.concat(raw_rows, ignore_index=True)
    for metric in ("adp", "expert_rank", "source_position_rank"):
        if metric not in raw_values:
            raw_values[metric] = np.nan

    keys = ["player_key", "season", "source", "source_table"]
    grouped = raw_values.groupby(keys, sort=True, dropna=False)
    metadata = grouped.agg(
        position=("position", _deterministic_mode),
        team=("team", _deterministic_mode),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        metrics = grouped[
            ["adp", "expert_rank", "source_position_rank"]
        ].median()
    values = metadata.join(metrics).reset_index()
    values["season"] = values["season"].astype(int)
    values["position"] = values["position"].astype("string").str.upper()
    position_map, team_map = _season_context_maps(player_aliases)
    context_keys = [
        (str(player_key), int(season))
        for player_key, season in values[
            ["player_key", "season"]
        ].itertuples(index=False, name=None)
    ]
    missing_position = ~values["position"].isin(POSITIONS)
    fallback_position = pd.Series(
        [position_map.get(key, pd.NA) for key in context_keys],
        index=values.index,
    )
    values.loc[missing_position, "position"] = fallback_position[
        missing_position
    ]
    values = values[values["position"].isin(POSITIONS)].copy()
    context_keys = [
        (str(player_key), int(season))
        for player_key, season in values[
            ["player_key", "season"]
        ].itertuples(index=False, name=None)
    ]
    missing_team = values["team"].isna()
    fallback_team = pd.Series(
        [team_map.get(key, pd.NA) for key in context_keys],
        index=values.index,
    )
    values.loc[missing_team, "team"] = fallback_team[missing_team]
    values["team"] = (
        values["team"].astype("string").str.upper().map(TEAM_MAP)
    )
    values["metric_count"] = values[
        ["adp", "expert_rank", "source_position_rank"]
    ].notna().sum(axis=1)
    values = values[values["metric_count"].gt(0)].copy()
    values["run_id"] = run_id
    values = align_columns(
        values,
        MARKET_VALUE_COLUMNS,
        "player_season_market_values",
    )
    return (
        values.sort_values(["season", "player_key", "source"]).reset_index(
            drop=True
        ),
        pd.DataFrame(audit_rows),
    )
