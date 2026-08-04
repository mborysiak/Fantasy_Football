"""Build the V2 canonical player identity and provider alias tables."""

from __future__ import annotations

import io
import re
import sqlite3
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from Scripts.V2.config import (
    IDENTITY_SOURCE_TABLES,
    NFLVERSE_PLAYERS_URL,
    POSITIONS,
    SOURCE_DB_PATH,
    START_SEASON,
    TEAM_MAP,
)
from Scripts.V2.contracts import (
    PLAYER_ALIAS_COLUMNS,
    PLAYER_IDENTITY_COLUMNS,
    SOURCE_ROW_EXCLUSION_ID_COLUMN,
    SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN,
    SOURCE_SEASON_OVERRIDE_ID_COLUMN,
    SOURCE_SEASON_OVERRIDE_REASON_COLUMN,
    SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN,
    SOURCE_STORED_SEASON_COLUMN,
    SOURCE_MANIFEST_COLUMNS,
    align_columns,
    apply_source_row_exclusions,
    apply_source_season_overrides,
    apply_source_team_trust_policy,
    assert_no_source_row_exclusions,
    assert_no_untrusted_source_team_labels,
    bytes_sha256,
    normalize_player_name,
    normalize_source_position,
    partition_source_row_exclusions,
    stable_player_key,
    table_exists,
)

# Deliberately narrow, reviewed exceptions for provider names that cannot be
# recovered by punctuation/suffix normalization.  A ``None`` source applies
# across providers; source-specific rules contain known provider truncations.
GOVERNED_NAME_ALIASES: dict[tuple[str | None, str], str] = {
    (None, "tet mcmillan"): "tetairoa mcmillan",
    ("fantasypros", "amon ra st"): "amon ra st brown",
    ("fantasypros", "equanimeous st"): "equanimeous st brown",
    ("adp_mfl", "brown st"): "equanimeous st brown",
    ("fantasydata", "drew ogletree"): "andrew ogletree",
    ("fantasypros", "drew ogletree"): "andrew ogletree",
    ("fantasydata", "irv charles"): "irvin charles",
    ("barret_rank", "jacorey croskey merritt"): (
        "jacory croskey merritt"
    ),
    ("adp_average_nffc", "jayden ott"): "jaydn ott",
    ("nffc_best_ball_overall", "jayden ott"): "jaydn ott",
    ("nffc_rotowire_online", "jayden ott"): "jaydn ott",
    ("adp_fpros", "matt hibner"): "matthew hibner",
    ("fantasydata", "matt hibner"): "matthew hibner",
    ("fantasypros_best_ball_adp", "matt hibner"): "matthew hibner",
    ("adp_average_nffc", "nathan carter"): "nate carter",
    ("fantasydata", "nathan carter"): "nate carter",
    ("fff", "nathan carter"): "nate carter",
    ("nffc_best_ball_25s50s", "nathan carter"): "nate carter",
    ("nffc_best_ball_overall", "nathan carter"): "nate carter",
    ("fantasydata", "scotty miller"): "scott miller",
    ("fantasypros", "scotty miller"): "scott miller",
    ("fantasypros_best_ball_adp", "scotty miller"): "scott miller",
    ("fftoday", "scotty miller"): "scott miller",
}

# Tet entered the production lineage under this provisional key before nflverse
# exposed his GSIS identity.  Keep the production key authoritative when both
# historical rows are present so downstream player-key continuity is preserved.
GOVERNED_STABLE_PLAYER_KEYS: dict[str, str] = {
    "00-0040124": "c16a5e67-fff0-57b9-838c-c8df91df7b9d",
}


def fetch_csv(url: str) -> tuple[pd.DataFrame, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Fantasy-Football-V2/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload)), bytes_sha256(payload)


def _nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _clean_text(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _normalize_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    team = str(value).strip().upper()
    return TEAM_MAP.get(team, team) if team else None


def _governed_match_name(
    normalized_name: object,
    source: object = None,
) -> str:
    """Return the reviewed canonical match name for a provider name."""
    name = normalize_player_name(normalized_name)
    if not name:
        return ""
    source_name = (
        str(source).strip().lower()
        if source is not None and pd.notna(source)
        else None
    )
    return GOVERNED_NAME_ALIASES.get(
        (source_name, name),
        GOVERNED_NAME_ALIASES.get((None, name), name),
    )


def _existing_key_maps(
    existing_identity: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[tuple[str, str, int | None], str],
    dict[tuple[str, int | None], str],
]:
    gsis_to_key: dict[str, str] = {}
    provisional_candidates: dict[
        tuple[str, str, int | None], list[str]
    ] = defaultdict(list)
    cross_position_candidates: dict[
        tuple[str, int | None], list[str]
    ] = defaultdict(list)
    if existing_identity.empty:
        return gsis_to_key, {}, {}

    for row in existing_identity.itertuples(index=False):
        player_key = str(row.player_key)
        if pd.notna(row.gsis_id):
            gsis_to_key[str(row.gsis_id)] = player_key
        if getattr(row, "identity_status", None) == "provisional":
            draft_year = getattr(row, "draft_year", None)
            draft_value = int(draft_year) if pd.notna(draft_year) else None
            raw_name = str(row.normalized_name)
            match_name = _governed_match_name(
                raw_name,
                getattr(row, "identity_source", None),
            )
            for candidate_name in {raw_name, match_name}:
                provisional_candidates[
                    (
                        candidate_name,
                        str(row.position),
                        draft_value,
                    )
                ].append(player_key)
                cross_position_candidates[
                    (candidate_name, draft_value)
                ].append(player_key)
    provisional_to_key = {
        signature: keys[0]
        for signature, keys in provisional_candidates.items()
        if len(set(keys)) == 1
    }
    cross_position_to_key = {
        signature: keys[0]
        for signature, keys in cross_position_candidates.items()
        if len(set(keys)) == 1
    }
    return gsis_to_key, provisional_to_key, cross_position_to_key


def canonicalize_nflverse_players(
    players: pd.DataFrame,
    existing_identity: pd.DataFrame | None = None,
    start_season: int = START_SEASON,
    eligible_source_names: set[str] | None = None,
) -> pd.DataFrame:
    """Convert nflverse players into the stable internal identity contract."""
    required = {
        "gsis_id",
        "display_name",
        "position_group",
        "rookie_season",
        "last_season",
        "draft_year",
    }
    missing = sorted(required.difference(players.columns))
    if missing:
        raise ValueError(f"nflverse players is missing required columns: {missing}")

    frame = players.copy()
    frame["position"] = frame["position_group"].astype("string").str.upper()

    for column in (
        "rookie_season",
        "last_season",
        "draft_year",
        "draft_round",
        "draft_pick",
    ):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = _nullable_int(frame[column])

    relevant = (
        frame["last_season"].fillna(0).ge(start_season)
        | frame["rookie_season"].fillna(0).ge(start_season)
        | frame["draft_year"].fillna(0).ge(start_season)
    )
    frame = frame[relevant & frame["gsis_id"].notna()].copy()
    frame["normalized_name"] = frame["display_name"].map(normalize_player_name)
    frame = frame[frame["normalized_name"].ne("")].copy()
    for column in (
        "common_first_name",
        "first_name",
        "football_name",
        "last_name",
        "short_name",
    ):
        if column not in frame:
            frame[column] = pd.NA

    def match_names(row: pd.Series) -> tuple[str, ...]:
        raw_names = [row.get("display_name"), row.get("short_name")]
        last_name = row.get("last_name")
        if pd.notna(last_name):
            for first_column in (
                "common_first_name",
                "first_name",
                "football_name",
            ):
                first_name = row.get(first_column)
                if pd.notna(first_name):
                    raw_names.append(f"{first_name} {last_name}")
        aliases = {
            normalize_player_name(value)
            for value in raw_names
            if pd.notna(value)
        }
        aliases.discard("")
        return tuple(sorted(aliases))

    frame["_match_names"] = frame.apply(match_names, axis=1)
    source_names = eligible_source_names or set()
    source_match = frame["_match_names"].map(
        lambda values: bool(set(values).intersection(source_names))
    )
    frame = frame[
        frame["position"].isin(POSITIONS) | source_match
    ].copy()

    existing = (
        existing_identity
        if existing_identity is not None
        else pd.DataFrame(columns=PLAYER_IDENTITY_COLUMNS)
    )
    (
        gsis_to_key,
        provisional_to_key,
        _,
    ) = _existing_key_maps(existing)

    keys: list[str] = []
    for row in frame.itertuples(index=False):
        gsis_id = str(row.gsis_id)
        draft_year = int(row.draft_year) if pd.notna(row.draft_year) else None
        promoted_key = provisional_to_key.get(
            (row.normalized_name, row.position, draft_year)
        ) or provisional_to_key.get(
            (row.normalized_name, row.position, None)
        )
        keys.append(
            GOVERNED_STABLE_PLAYER_KEYS.get(gsis_id)
            or gsis_to_key.get(gsis_id)
            or promoted_key
            or stable_player_key(f"gsis:{gsis_id}")
        )
    frame["player_key"] = keys

    rename = {
        "college_name": "college",
    }
    frame = frame.rename(columns=rename)
    for column in (
        "pfr_id",
        "pff_id",
        "espn_id",
        "nfl_id",
        "birth_date",
        "college",
        "draft_team",
        "latest_team",
    ):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = frame[column].map(_clean_text)

    frame["identity_status"] = "confirmed"
    frame["identity_source"] = "nflverse_players"
    match_name_map = frame.set_index("player_key")["_match_names"].to_dict()
    frame = align_columns(frame, PLAYER_IDENTITY_COLUMNS, "player_identity")
    frame["_match_names"] = frame["player_key"].map(match_name_map)
    frame = frame.drop_duplicates("gsis_id", keep="last")

    if frame["player_key"].duplicated().any():
        duplicate = frame.loc[
            frame["player_key"].duplicated(False),
            ["player_key", "gsis_id", "display_name"],
        ]
        raise ValueError(
            "Multiple nflverse players resolved to one player_key:\n"
            f"{duplicate.to_string(index=False)}"
        )
    return frame.reset_index(drop=True)


def load_identity_source_records(
    source_database: Path = SOURCE_DB_PATH,
    table_specs: dict[str, dict[str, object]] = IDENTITY_SOURCE_TABLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load narrow identity records from existing projection/draft tables."""
    records: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    with sqlite3.connect(source_database) as connection:
        for table, spec in table_specs.items():
            if not table_exists(connection, table):
                continue
            available = {
                row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            requested = {
                source_column
                for target, source_column in spec.items()
                if target
                not in {
                    "source",
                    "position_value",
                    "source_kind",
                    "source_column",
                    "source_prefix",
                    "allow_missing_position",
                }
                and isinstance(source_column, str)
                and source_column in available
            }
            dynamic_source_column = spec.get("source_column")
            if (
                isinstance(dynamic_source_column, str)
                and dynamic_source_column in available
            ):
                requested.add(dynamic_source_column)
            if spec.get("player") not in requested:
                continue
            query_columns = ", ".join(f'"{column}"' for column in sorted(requested))
            raw = pd.read_sql_query(
                f'SELECT {query_columns} FROM "{table}"', connection
            )
            standard = pd.DataFrame(index=raw.index)
            standard["source_table"] = table
            for target in (
                "source_player_id",
                "player",
                "position",
                "team",
                "season",
                "draft_year",
                "draft_round",
                "draft_pick",
                "college",
            ):
                source_column = spec.get(target)
                if source_column and source_column in raw:
                    standard[target] = raw[source_column]
                else:
                    standard[target] = pd.NA
            if "position_value" in spec:
                standard["position"] = spec["position_value"]
            if (
                isinstance(dynamic_source_column, str)
                and dynamic_source_column in raw
            ):
                prefix = str(spec.get("source_prefix", ""))
                normalized_source = (
                    raw[dynamic_source_column]
                    .astype("string")
                    .fillna("unknown")
                    .str.lower()
                    .map(lambda value: re.sub(r"[^a-z0-9]+", "_", str(value)))
                    .str.strip("_")
                )
                standard["source"] = prefix + normalized_source
            else:
                standard["source"] = spec["source"]
            standard["_allow_missing_position"] = bool(
                spec.get("allow_missing_position", False)
            )
            records.append(standard)
            manifest_rows.append(
                {
                    "component": "identity",
                    "source_name": table,
                    "source_uri": f"sqlite://{source_database.resolve()}#{table}",
                    "source_sha256": pd.NA,
                    "row_count": len(raw),
                }
            )

    if not records:
        empty = pd.DataFrame(
            columns=[
                "source",
                "source_table",
                "source_player_id",
                "player",
                "position",
                "team",
                "season",
                "draft_year",
                "draft_round",
                "draft_pick",
                "college",
                "_allow_missing_position",
            ]
        )
        return empty, pd.DataFrame(manifest_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        frame = pd.concat(records, ignore_index=True)
    frame["player"] = frame["player"].map(_clean_text)
    frame["normalized_name"] = frame["player"].map(normalize_player_name)
    frame["position"] = frame["position"].map(normalize_source_position)
    frame["team"] = frame["team"].map(_clean_text)
    frame["college"] = frame["college"].map(_clean_text)
    frame["source_player_id"] = frame["source_player_id"].map(_clean_text)
    frame["season"] = _nullable_int(frame["season"])
    frame["draft_year"] = _nullable_int(frame["draft_year"])
    frame["draft_round"] = _nullable_int(frame["draft_round"])
    frame["draft_pick"] = _nullable_int(frame["draft_pick"])
    frame, excluded_records = partition_source_row_exclusions(
        frame,
        "identity source records",
    )
    for exclusion_id, excluded_group in excluded_records.groupby(
        SOURCE_ROW_EXCLUSION_ID_COLUMN,
        dropna=False,
        sort=True,
    ):
        references = excluded_group[
            SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN
        ].dropna().unique()
        manifest_rows.append(
            {
                "component": "identity_quarantine",
                "source_name": str(exclusion_id),
                "source_uri": (
                    str(references[0]) if len(references) == 1 else pd.NA
                ),
                "source_sha256": pd.NA,
                "row_count": len(excluded_group),
            }
        )
    frame = apply_source_season_overrides(
        frame,
        "identity source records",
    )
    frame = apply_source_team_trust_policy(
        frame,
        "identity source records",
    )
    frame["_draft_year_inferred"] = False
    known_drafts = (
        frame[frame["draft_year"].notna()]
        .groupby(["normalized_name", "position"])["draft_year"]
        .agg(lambda values: sorted({int(value) for value in values}))
    )
    unique_drafts = {
        key: values[0] for key, values in known_drafts.items() if len(values) == 1
    }
    missing_draft = frame["draft_year"].isna()
    inferred_draft_years = [
        unique_drafts.get((name, position), pd.NA)
        for name, position in frame.loc[
            missing_draft, ["normalized_name", "position"]
        ].itertuples(index=False, name=None)
    ]
    frame.loc[missing_draft, "draft_year"] = inferred_draft_years
    frame.loc[missing_draft, "_draft_year_inferred"] = pd.Series(
        inferred_draft_years, index=frame.index[missing_draft]
    ).notna()
    frame["draft_year"] = _nullable_int(frame["draft_year"])
    allow_missing_position = frame["_allow_missing_position"].fillna(False)
    frame = frame[
        frame["normalized_name"].ne("")
        & (
            frame["position"].isin(POSITIONS)
            | (allow_missing_position & frame["position"].isna())
        )
    ].copy()
    frame = frame.drop_duplicates(
        [
            "source",
            "source_table",
            "source_player_id",
            "normalized_name",
            "position",
            "team",
            "season",
            "draft_year",
        ]
    ).reset_index(drop=True)
    frame = frame.drop(columns="_allow_missing_position")
    return frame, pd.DataFrame(manifest_rows)


def _candidate_indexes(
    identity: pd.DataFrame,
) -> dict[tuple[str, str], list[int]]:
    lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in identity.iterrows():
        match_names = row.get("_match_names")
        if not isinstance(match_names, (tuple, list, set)):
            match_names = (row["normalized_name"],)
        for name in match_names:
            lookup[(name, row["position"])].append(index)
    return lookup


def _candidate_name_indexes(
    identity: pd.DataFrame,
) -> dict[str, list[int]]:
    lookup: dict[str, list[int]] = defaultdict(list)
    for index, row in identity.iterrows():
        match_names = row.get("_match_names")
        if not isinstance(match_names, (tuple, list, set)):
            match_names = (row["normalized_name"],)
        for name in match_names:
            lookup[str(name)].append(index)
    return lookup


def _candidate_is_compatible(
    record: pd.Series,
    candidate: pd.Series,
) -> bool:
    """Reject confident entry-year contradictions before name uniqueness."""
    draft_year = record.get("draft_year")
    draft_year_inferred = bool(record.get("_draft_year_inferred", False))
    if pd.notna(draft_year) and not draft_year_inferred:
        known_entry_years = {
            int(value)
            for value in (
                candidate.get("draft_year"),
                candidate.get("rookie_season"),
            )
            if pd.notna(value)
        }
        if known_entry_years and int(draft_year) not in known_entry_years:
            return False

    season = record.get("season")
    if pd.notna(season):
        season_value = int(season)
        career_start = candidate.get("rookie_season")
        if pd.isna(career_start):
            career_start = candidate.get("draft_year")
        if pd.notna(career_start) and season_value < int(career_start):
            return False
    return True


def _resolve_candidate(
    record: pd.Series,
    identity: pd.DataFrame,
    candidates: Iterable[int],
) -> tuple[int | None, str]:
    candidate_list = list(candidates)
    if not candidate_list:
        return None, "provisional_unmatched"

    compatible_list = [
        index
        for index in candidate_list
        if _candidate_is_compatible(record, identity.loc[index])
    ]
    if not compatible_list:
        return None, "provisional_incompatible"
    if len(candidate_list) == 1:
        return compatible_list[0], "name_position_unique"

    subset = identity.loc[compatible_list]
    draft_year = record.get("draft_year")
    draft_year_inferred = bool(record.get("_draft_year_inferred", False))
    if pd.notna(draft_year) and not draft_year_inferred:
        exact_draft = subset[
            subset["draft_year"].eq(int(draft_year))
            | subset["rookie_season"].eq(int(draft_year))
        ]
        if len(exact_draft) == 1:
            return int(exact_draft.index[0]), "name_position_draft_year"

    season = record.get("season")
    if pd.notna(season):
        season_value = int(season)
        # Career windows are only a namesake tie-breaker.  Allow the preseason
        # immediately after the last recorded appearance, but never use this
        # metadata as a hard rejection for an otherwise unique returning
        # player.
        active = subset[
            subset["rookie_season"].fillna(subset["draft_year"]).fillna(-1)
            <= season_value
        ]
        active = active[
            active["last_season"]
            .fillna(season_value)
            .add(1)
            .ge(season_value)
        ]
        if len(active) == 1:
            return int(active.index[0]), "name_position_active_window"

    team = record.get("team")
    if pd.notna(team):
        normalized_team = _normalize_team(team)
        draft_teams = subset["draft_team"].map(_normalize_team)
        latest_teams = subset["latest_team"].map(_normalize_team)
        team_match = subset[
            draft_teams.eq(normalized_team) | latest_teams.eq(normalized_team)
        ]
        if len(team_match) == 1:
            return int(team_match.index[0]), "name_position_team"
    if len(compatible_list) == 1:
        return compatible_list[0], "name_position_compatible_unique"
    return None, "provisional_ambiguous"


def _reconcile_provisional_identities(
    identity: pd.DataFrame,
    aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge redundant provisional rows into one unambiguous confirmed player."""
    provisional = identity["identity_status"].eq("provisional")
    if not provisional.any():
        return identity, aliases

    confirmed = identity[identity["identity_status"].eq("confirmed")]
    remap: dict[str, str] = {}
    for row in identity[provisional].itertuples(index=False):
        provisional_key = str(row.player_key)
        match_name = _governed_match_name(
            row.normalized_name,
            row.identity_source,
        )
        candidates = confirmed[
            confirmed["normalized_name"].eq(match_name)
            & confirmed["position"].eq(row.position)
        ]
        record = pd.Series(
            {
                "draft_year": row.draft_year,
                "_draft_year_inferred": False,
                "season": row.rookie_season,
                "team": (
                    row.latest_team
                    if pd.notna(row.latest_team)
                    else row.draft_team
                ),
            }
        )
        candidate_index, _ = _resolve_candidate(
            record,
            confirmed,
            candidates.index,
        )
        if candidate_index is None:
            continue

        candidate = confirmed.loc[candidate_index]
        provisional_aliases = aliases[
            aliases["player_key"].astype(str).eq(provisional_key)
        ]
        if provisional_aliases.empty:
            # Without source evidence, retaining the provisional row is safer
            # than silently merging it into a confirmed player.
            continue

        aliases_compatible = True
        for alias in provisional_aliases.itertuples(index=False):
            match_method = str(alias.match_method)
            if "provisional_incompatible" in match_method:
                aliases_compatible = False
                break
            alias_record = pd.Series(
                {
                    "draft_year": alias.draft_year,
                    # player_aliases does not retain the inference flag.  A
                    # fail-closed reconciliation treats a recorded year as
                    # explicit; ordinary source resolution already handles
                    # inferred years before provisional creation.
                    "_draft_year_inferred": False,
                    "season": alias.season,
                    "team": alias.team,
                }
            )
            if not _candidate_is_compatible(alias_record, candidate):
                aliases_compatible = False
                break
        if aliases_compatible:
            remap[provisional_key] = str(candidate["player_key"])

    if not remap:
        return identity, aliases
    aliases = aliases.copy()
    aliases["player_key"] = aliases["player_key"].replace(remap)
    identity = identity[
        ~identity["player_key"].astype(str).isin(remap)
    ].copy()
    aliases = aliases.drop_duplicates().reset_index(drop=True)
    identity = identity.reset_index(drop=True)
    return identity, aliases


def resolve_source_records(
    canonical_identity: pd.DataFrame,
    source_records: pd.DataFrame,
    existing_identity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve provider rows to canonical players and add provisional identities."""
    identity = canonical_identity.copy().reset_index(drop=True)
    source_records = source_records.copy()
    if "source_table" not in source_records:
        source_records["source_table"] = pd.NA
    source_records = apply_source_row_exclusions(
        source_records,
        "identity source records",
    )
    source_records = apply_source_season_overrides(
        source_records,
        "identity source records",
    )
    source_records = apply_source_team_trust_policy(
        source_records,
        "identity source records",
    )
    assert_no_source_row_exclusions(
        source_records,
        "identity source records after quarantine",
    )
    assert_no_untrusted_source_team_labels(
        source_records,
        "identity source records after team policy",
    )
    lookup = _candidate_indexes(identity)
    name_lookup = _candidate_name_indexes(identity)
    existing = (
        existing_identity
        if existing_identity is not None
        else pd.DataFrame(columns=PLAYER_IDENTITY_COLUMNS)
    )
    _, existing_provisional, _ = _existing_key_maps(existing)

    def candidate_pool(
        record: pd.Series,
    ) -> tuple[list[int], bool, bool]:
        raw_name = str(record["normalized_name"])
        match_name = _governed_match_name(
            raw_name,
            record.get("source"),
        )
        governed_alias = match_name != raw_name
        position = record.get("position")
        exact = (
            lookup.get((match_name, str(position)), [])
            if pd.notna(position)
            else []
        )
        if exact:
            return list(exact), False, governed_alias
        return list(name_lookup.get(match_name, [])), True, governed_alias

    def resolve_record(record: pd.Series) -> tuple[int | None, str]:
        candidates, cross_position, governed_alias = candidate_pool(record)
        candidate_index, method = _resolve_candidate(
            record,
            identity,
            candidates,
        )
        if candidate_index is not None and cross_position:
            method = method.replace(
                "name_position",
                "name_cross_position",
            )
        if governed_alias:
            method = f"governed_alias_{method}"
        return candidate_index, method

    aliases: list[dict[str, object]] = []
    provisional_rows: dict[
        tuple[str, str, int | None], dict[str, object]
    ] = {}
    source_id_to_index: dict[tuple[str, str], int] = {}

    def enrich_draft_identity(index: int, record: pd.Series) -> None:
        if record.get("source") != "nfl_draft":
            return
        draft_year = record.get("draft_year")
        updates = {
            "draft_year": draft_year,
            "draft_round": record.get("draft_round"),
            "draft_pick": record.get("draft_pick"),
            "draft_team": record.get("team"),
            "college": record.get("college"),
            "rookie_season": draft_year,
        }
        for field, value in updates.items():
            if pd.isna(identity.at[index, field]) and pd.notna(value):
                identity.at[index, field] = value

    identified = source_records[source_records["source_player_id"].notna()]
    for (source, source_player_id), group in identified.groupby(
        ["source", "source_player_id"], dropna=False
    ):
        resolved_indexes: set[int] = set()
        for _, candidate_record in group.iterrows():
            candidate_index, _ = resolve_record(candidate_record)
            if candidate_index is not None:
                resolved_indexes.add(candidate_index)
        if len(resolved_indexes) == 1:
            source_id_to_index[(str(source), str(source_player_id))] = next(
                iter(resolved_indexes)
            )

    for _, record in source_records.iterrows():
        source_player_id = record.get("source_player_id")
        source_id_key = (
            (str(record["source"]), str(source_player_id))
            if pd.notna(source_player_id)
            else None
        )
        resolved_index = (
            source_id_to_index.get(source_id_key)
            if source_id_key is not None
            else None
        )
        if resolved_index is not None:
            method = "source_id_consensus"
        else:
            resolved_index, method = resolve_record(record)

        if resolved_index is None:
            if method == "governed_alias_provisional_incompatible":
                # The governed identity is known, but this source row carries
                # an impossible draft/entry season.  Exclude it instead of
                # creating a duplicate provisional identity.
                continue
            if pd.isna(record["position"]):
                # A position-less market row is useful only when it can be tied
                # to an existing identity. Do not invent an untyped player.
                continue
            draft_year = (
                int(record["draft_year"])
                if pd.notna(record["draft_year"])
                else None
            )
            provisional_signature = (
                record["normalized_name"],
                record["position"],
                draft_year,
            )
            provisional = provisional_rows.get(provisional_signature)
            if provisional is None:
                player_key = existing_provisional.get(
                    provisional_signature
                ) or stable_player_key(
                    "provisional:"
                    f"{record['normalized_name']}:{record['position']}:"
                    f"{draft_year if draft_year is not None else 'unknown'}"
                )
                provisional = {
                    "player_key": player_key,
                    "gsis_id": pd.NA,
                    "pfr_id": pd.NA,
                    "pff_id": pd.NA,
                    "espn_id": pd.NA,
                    "nfl_id": pd.NA,
                    "display_name": record["player"],
                    "normalized_name": record["normalized_name"],
                    "position": record["position"],
                    "birth_date": pd.NA,
                    "college": record.get("college", pd.NA),
                    "draft_year": draft_year,
                    "draft_round": record.get("draft_round", pd.NA),
                    "draft_pick": record.get("draft_pick", pd.NA),
                    "draft_team": record.get("team", pd.NA),
                    "rookie_season": draft_year,
                    "last_season": pd.NA,
                    "latest_team": record.get("team", pd.NA),
                    "identity_status": "provisional",
                    "identity_source": record["source"],
                }
                provisional_rows[provisional_signature] = provisional
            player_key = provisional["player_key"]
        else:
            enrich_draft_identity(resolved_index, record)
            player_key = identity.at[resolved_index, "player_key"]
        aliases.append(
            {
                "player_key": player_key,
                "source": record["source"],
                "source_table": record.get("source_table", pd.NA),
                "source_player_id": record.get("source_player_id", pd.NA),
                "source_name": record["player"],
                "normalized_name": record["normalized_name"],
                "position": record["position"],
                "team": record.get("team", pd.NA),
                "season": record.get("season", pd.NA),
                SOURCE_STORED_SEASON_COLUMN: record.get(
                    SOURCE_STORED_SEASON_COLUMN,
                    record.get("season", pd.NA),
                ),
                SOURCE_SEASON_OVERRIDE_ID_COLUMN: record.get(
                    SOURCE_SEASON_OVERRIDE_ID_COLUMN,
                    pd.NA,
                ),
                SOURCE_SEASON_OVERRIDE_REASON_COLUMN: record.get(
                    SOURCE_SEASON_OVERRIDE_REASON_COLUMN,
                    pd.NA,
                ),
                SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN: record.get(
                    SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN,
                    pd.NA,
                ),
                "draft_year": record.get("draft_year", pd.NA),
                "match_method": method,
            }
        )

    if provisional_rows:
        provisional_frame = pd.DataFrame(provisional_rows.values())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            identity = pd.concat(
                [identity, provisional_frame], ignore_index=True
            )
    identity = align_columns(identity, PLAYER_IDENTITY_COLUMNS, "player_identity")
    identity = identity.drop_duplicates("player_key", keep="first").reset_index(
        drop=True
    )
    alias_frame = pd.DataFrame(aliases)
    if alias_frame.empty:
        alias_frame = pd.DataFrame(columns=PLAYER_ALIAS_COLUMNS)
    alias_frame = align_columns(alias_frame, PLAYER_ALIAS_COLUMNS, "player_aliases")
    alias_frame = alias_frame.drop_duplicates().reset_index(drop=True)
    identity, alias_frame = _reconcile_provisional_identities(
        identity, alias_frame
    )
    return identity, alias_frame


def validate_identity(
    identity: pd.DataFrame, aliases: pd.DataFrame
) -> None:
    if identity.empty:
        raise ValueError("player_identity cannot be empty")
    if identity["player_key"].isna().any() or identity["player_key"].duplicated().any():
        raise ValueError("player_identity.player_key must be non-null and unique")
    confirmed = identity[identity["gsis_id"].notna()]
    if confirmed["gsis_id"].duplicated().any():
        raise ValueError("Confirmed gsis_id values must be unique")
    unknown_aliases = set(aliases["player_key"]).difference(identity["player_key"])
    if unknown_aliases:
        raise ValueError(
            f"player_aliases contains unknown player keys: {sorted(unknown_aliases)[:5]}"
        )
    assert_no_source_row_exclusions(aliases, "player_aliases")


def build_player_identity_frames(
    run_id: str,
    source_database: Path = SOURCE_DB_PATH,
    players_url: str = NFLVERSE_PLAYERS_URL,
    existing_identity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players, players_sha = fetch_csv(players_url)
    source_records, local_manifest = load_identity_source_records(source_database)
    eligible_source_names = set(source_records["normalized_name"].dropna())
    canonical = canonicalize_nflverse_players(
        players,
        existing_identity,
        eligible_source_names=eligible_source_names,
    )
    identity, aliases = resolve_source_records(
        canonical, source_records, existing_identity
    )
    validate_identity(identity, aliases)

    manifest = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "component": "identity",
                        "source_name": "nflverse_players",
                        "source_uri": players_url,
                        "source_sha256": players_sha,
                        "row_count": len(players),
                    }
                ]
            ),
            local_manifest,
        ],
        ignore_index=True,
    )
    manifest["run_id"] = run_id
    manifest = align_columns(manifest, SOURCE_MANIFEST_COLUMNS, "source_manifest")
    return identity, aliases, manifest
