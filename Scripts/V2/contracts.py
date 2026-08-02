"""Shared schemas, identifiers, and SQLite publication helpers for V2."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from Scripts.config import (
    PASSING_SCORING,
    RECEIVING_SCORING,
    RUSH_SCORING,
    get_scoring_dict,
)
from Scripts.V2.config import SOURCE_ROW_EXCLUSIONS, SOURCE_SEASON_OVERRIDES


PLAYER_NAMESPACE = uuid.UUID("0da48215-bc2f-4e7e-a71e-ea3f7036f998")

SOURCE_STORED_SEASON_COLUMN = "source_stored_season"
SOURCE_SEASON_OVERRIDE_ID_COLUMN = "source_season_override_id"
SOURCE_SEASON_OVERRIDE_REASON_COLUMN = "source_season_override_reason"
SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN = "source_season_override_reference"
SOURCE_ROW_EXCLUSION_ID_COLUMN = "source_row_exclusion_id"
SOURCE_ROW_EXCLUSION_REASON_COLUMN = "source_row_exclusion_reason"
SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN = "source_row_exclusion_reference"
SOURCE_ROW_EXCLUSION_POLICY_COMPONENT = "source_row_exclusion_policy"
SOURCE_ROW_EXCLUSION_POLICY_NAME = "configured_source_row_exclusions"

PLAYER_IDENTITY_COLUMNS = (
    "player_key",
    "gsis_id",
    "pfr_id",
    "pff_id",
    "espn_id",
    "nfl_id",
    "display_name",
    "normalized_name",
    "position",
    "birth_date",
    "college",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_team",
    "rookie_season",
    "last_season",
    "latest_team",
    "identity_status",
    "identity_source",
)

PLAYER_ALIAS_COLUMNS = (
    "player_key",
    "source",
    "source_table",
    "source_player_id",
    "source_name",
    "normalized_name",
    "position",
    "team",
    "season",
    SOURCE_STORED_SEASON_COLUMN,
    SOURCE_SEASON_OVERRIDE_ID_COLUMN,
    SOURCE_SEASON_OVERRIDE_REASON_COLUMN,
    SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN,
    "draft_year",
    "match_method",
)

PLAYER_OUTCOME_COLUMNS = (
    "player_key",
    "gsis_id",
    "display_name",
    "position",
    "season",
    "teams",
    "opportunity_games",
    "season_points",
    "conditional_ppg",
    "passing_points",
    "rushing_points",
    "receiving_points",
    "fumble_points",
    "two_point_points",
    "special_teams_points",
    "pass_attempts",
    "rush_attempts",
    "targets",
    "receptions",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "appeared",
    "useful_season",
    "target_available",
    "outcome_complete",
    "league",
    "scoring_hash",
    "run_id",
)

PLAYER_SEASON_SOURCE_COLUMNS = (
    "player_key",
    "season",
    "source",
    "source_kind",
    "source_player_name",
    "source_position",
    "source_team",
    "match_method",
    "record_count",
    "run_id",
)

PLAYER_SEASON_SPINE_COLUMNS = (
    "player_key",
    "gsis_id",
    "display_name",
    "season",
    "position",
    "team",
    "identity_status",
    "identity_source",
    "draft_year",
    "rookie_season",
    "year_exp",
    "experience_known",
    "is_rookie",
    "candidate_rule",
    "candidate_source_count",
    "projection_source_count",
    "market_source_count",
    "ranking_source_count",
    "draft_source_count",
    "candidate_sources",
    "position_conflict",
    "team_conflict",
    "feature_cutoff_season",
    "preseason_source_season",
    "outcome_complete",
    "outcome_join_status",
    "outcome_observed",
    "active_target_available",
    "appeared",
    "opportunity_games",
    "useful_season",
    "observed_season_points",
    "unconditional_season_points",
    "conditional_ppg",
    "conditional_ppg_target_available",
    "conditional_ppg_training_eligible",
    "league",
    "scoring_hash",
    "foundation_run_id",
    "run_id",
)

PROJECTION_VALUE_METRICS = (
    "projected_games",
    "raw_projected_points",
    "raw_projected_ppg",
    "source_floor_points",
    "source_ceiling_points",
    "source_uncertainty",
    "pass_completions",
    "pass_attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)

PROJECTION_VALUE_COLUMNS = (
    "player_key",
    "season",
    "provider",
    "sources",
    "source_tables",
    "source_stored_seasons",
    "source_season_override_ids",
    "source_season_override_reasons",
    "source_season_override_references",
    "position",
    "team",
    *PROJECTION_VALUE_METRICS,
    "passing_points",
    "rushing_points",
    "receiving_points",
    "configured_projected_points",
    "configured_points_complete",
    "configured_points_imputed_component_count",
    "configured_points_imputed_components",
    "configured_points_imputation_donor_providers",
    "configured_points_imputation_donor_count",
    "provider_projected_points",
    "provider_points_estimand",
    "points_method",
    "provider_points_per_team_game",
    "provider_points_per_projected_game",
    "provider_team_points",
    "provider_room_points",
    "provider_room_share",
    "provider_room_rank",
    "provider_room_gap_to_leader",
    "provider_room_hhi",
    "metric_count",
    "run_id",
)

MARKET_VALUE_COLUMNS = (
    "player_key",
    "season",
    "source",
    "source_table",
    "position",
    "team",
    "adp",
    "expert_rank",
    "source_position_rank",
    "metric_count",
    "run_id",
)

FEATURE_CATALOG_COLUMNS = (
    "feature_name",
    "family",
    "description",
    "dtype",
    "collinearity_group",
    "residual_eligible",
    "participation_eligible",
    "template_eligible",
    "audit_only",
    "run_id",
)

FEATURE_MANIFEST_COLUMNS = (
    "manifest_name",
    "feature_name",
    "family",
    "status",
    "family_weight_budget",
    "run_id",
)

FEATURE_AUDIT_COLUMNS = (
    "feature_name",
    "family",
    "non_null_count",
    "coverage_rate",
    "training_non_null_count",
    "training_coverage_rate",
    "current_non_null_count",
    "current_coverage_rate",
    "unique_count",
    "zero_variance",
    "run_id",
)

FEATURE_CORRELATION_COLUMNS = (
    "family",
    "feature_a",
    "feature_b",
    "spearman",
    "abs_spearman",
    "shared_rows",
    "run_id",
)

FEATURE_SOURCE_AUDIT_COLUMNS = (
    "source_table",
    "source_kind",
    "input_rows",
    "resolved_rows",
    "resolution_rate",
    "excluded_rows",
    "source_row_exclusion_ids",
    "source_row_exclusion_reasons",
    "source_row_exclusion_references",
    "run_id",
)

MODEL_RUN_COLUMNS = (
    "run_id",
    "created_at_utc",
    "feature_run_id",
    "league",
    "validation_start_season",
    "validation_end_season",
    "n_splits",
    "random_seed",
    "conditional_ppg_rows",
    "participation_rows",
    "model_count",
    "status",
)

MODEL_FOLD_COLUMNS = (
    "run_id",
    "target_name",
    "player_key",
    "season",
    "position",
    "fold",
    "training_start_season",
    "training_through_season",
)

MODEL_SPECIFICATION_COLUMNS = (
    "run_id",
    "target_name",
    "model_name",
    "model_family",
    "prediction_kind",
    "feature_set",
    "pipeline_variant",
    "feature_count",
    "feature_names_json",
    "hyperparameters_json",
    "search_iterations",
    "status",
)

MODEL_HYPERPARAMETER_COLUMNS = (
    "run_id",
    "target_name",
    "model_name",
    "fold",
    "trial",
    "parameters_json",
    "validation_score",
    "selected",
)

MODEL_OOF_COLUMNS = (
    "run_id",
    "feature_run_id",
    "target_name",
    "model_name",
    "model_family",
    "prediction_kind",
    "feature_set",
    "pipeline_variant",
    "player_key",
    "season",
    "position",
    "team",
    "fold",
    "training_through_season",
    "actual",
    "baseline_prediction",
    "model_prediction",
    "final_prediction",
    "residual_actual",
    "residual_prediction",
    "opportunity_games",
    "has_prior_outcome",
    "is_rookie",
    "year_exp",
    "projection_provider_count",
)

MODEL_SCORE_COLUMNS = (
    "run_id",
    "target_name",
    "model_name",
    "aggregation",
    "metric",
    "n_rows",
    "n_seasons",
    "value",
    "baseline_value",
    "delta",
)

MODEL_SLICE_COLUMNS = (
    "run_id",
    "target_name",
    "model_name",
    "slice_type",
    "slice_value",
    "metric",
    "n_rows",
    "n_seasons",
    "value",
    "baseline_value",
    "delta",
)

SOURCE_MANIFEST_COLUMNS = (
    "run_id",
    "component",
    "source_name",
    "source_uri",
    "source_sha256",
    "row_count",
)

BUILD_RUN_COLUMNS = (
    "run_id",
    "created_at_utc",
    "component",
    "league",
    "start_season",
    "completed_through_season",
    "useful_season_min_games",
    "scoring_hash",
    "identity_rows",
    "alias_rows",
    "outcome_rows",
    "source_observation_rows",
    "spine_rows",
    "projection_value_rows",
    "market_value_rows",
    "feature_rows",
    "feature_count",
    "foundation_run_id",
    "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def create_run_id(component: str = "milestone_1") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{component}_{stamp}_{uuid.uuid4().hex[:8]}"


def normalize_player_name(value: object) -> str:
    """Return a provider-neutral matching name while retaining identity elsewhere."""
    if value is None or pd.isna(value):
        return ""
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    text = (
        text.lower()
        .replace("&", " and ")
        .replace("'", "")
        .replace("’", "")
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token]
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    collapsed: list[str] = []
    index = 0
    while index < len(tokens):
        if len(tokens[index]) == 1:
            initials = [tokens[index]]
            while index + 1 < len(tokens) and len(tokens[index + 1]) == 1:
                index += 1
                initials.append(tokens[index])
            collapsed.append("".join(initials))
        else:
            collapsed.append(tokens[index])
        index += 1
    return " ".join(collapsed)


def normalize_source_position(value: object) -> object:
    """Normalize provider position labels such as ``RB-01`` to ``RB``."""
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    match = re.match(r"^(QB|RB|WR|TE)(?:$|[^A-Z])", text)
    return match.group(1) if match else text


def stable_player_key(identity_signature: str) -> str:
    return str(uuid.uuid5(PLAYER_NAMESPACE, identity_signature))


def configured_scoring(league: str) -> dict[str, dict[str, float]]:
    league = str(league).strip().lower()
    valid_leagues = (
        set(PASSING_SCORING)
        & set(RUSH_SCORING)
        & set(RECEIVING_SCORING)
    )
    if league not in valid_leagues:
        raise ValueError(
            f"Unknown scoring league {league!r}; expected one of "
            f"{sorted(valid_leagues)}"
        )
    return {
        "passing": get_scoring_dict("passing", league),
        "rushing": get_scoring_dict("rush", league),
        "receiving": get_scoring_dict("receiving", league),
    }


def scoring_hash(league: str) -> str:
    payload = json.dumps(
        configured_scoring(league), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], frame_name: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _validated_source_row_exclusions() -> tuple[dict[str, object], ...]:
    """Return validated, non-overlapping source-row quarantine rules."""
    required = {
        "exclusion_id",
        "source_table",
        "position",
        "stored_season",
        "reason",
        "reference",
    }
    rules: list[dict[str, object]] = []
    ids: set[str] = set()
    scopes: set[tuple[str, str, int]] = set()
    for raw_rule in SOURCE_ROW_EXCLUSIONS:
        missing = sorted(required.difference(raw_rule))
        if missing:
            raise ValueError(
                f"Source-row exclusion is missing required fields: {missing}"
            )
        exclusion_id = str(raw_rule["exclusion_id"]).strip()
        source_table = str(raw_rule["source_table"]).strip()
        position = str(raw_rule["position"]).strip().upper()
        stored_season = int(raw_rule["stored_season"])
        reason = str(raw_rule["reason"]).strip()
        reference = str(raw_rule["reference"]).strip()
        scope = (
            source_table,
            position,
            stored_season,
        )
        if not exclusion_id:
            raise ValueError("Source-row exclusion IDs cannot be empty")
        if exclusion_id in ids:
            raise ValueError(f"Duplicate source-row exclusion ID: {exclusion_id}")
        if scope in scopes:
            raise ValueError(
                "Duplicate source-row exclusion scope: "
                f"{scope[0]}/{scope[1]}/{scope[2]}"
            )
        if not reason or not reference:
            raise ValueError(
                f"Source-row exclusion {exclusion_id} requires reason and reference"
            )
        ids.add(exclusion_id)
        scopes.add(scope)
        rules.append(
            {
                "exclusion_id": exclusion_id,
                "source_table": source_table,
                "position": position,
                "stored_season": stored_season,
                "reason": reason,
                "reference": reference,
            }
        )
    return tuple(rules)


def source_row_exclusion_policy_hash() -> str:
    """Return a stable fingerprint for the governed source-row exclusions."""
    rules = sorted(
        _validated_source_row_exclusions(),
        key=lambda rule: str(rule["exclusion_id"]),
    )
    payload = json.dumps(rules, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_row_exclusion_policy_receipt(run_id: str) -> dict[str, object]:
    """Return a source-manifest receipt for the active quarantine policy."""
    rules = _validated_source_row_exclusions()
    return {
        "run_id": run_id,
        "component": SOURCE_ROW_EXCLUSION_POLICY_COMPONENT,
        "source_name": SOURCE_ROW_EXCLUSION_POLICY_NAME,
        "source_uri": "python://Scripts.V2.config#SOURCE_ROW_EXCLUSIONS",
        "source_sha256": source_row_exclusion_policy_hash(),
        "row_count": len(rules),
    }


def partition_source_row_exclusions(
    frame: pd.DataFrame,
    frame_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split governed quarantines from usable rows with audit provenance."""
    require_columns(
        frame,
        ("source_table", "position", "season"),
        frame_name,
    )
    output = frame.copy()
    output["season"] = pd.to_numeric(
        output["season"], errors="coerce"
    ).astype("Int64")
    if SOURCE_STORED_SEASON_COLUMN not in output:
        output[SOURCE_STORED_SEASON_COLUMN] = output["season"]
    else:
        output[SOURCE_STORED_SEASON_COLUMN] = pd.to_numeric(
            output[SOURCE_STORED_SEASON_COLUMN],
            errors="coerce",
        ).astype("Int64").fillna(output["season"])
    for column in (
        SOURCE_ROW_EXCLUSION_ID_COLUMN,
        SOURCE_ROW_EXCLUSION_REASON_COLUMN,
        SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN,
    ):
        if column not in output:
            output[column] = pd.Series(pd.NA, index=output.index, dtype="string")
        else:
            output[column] = output[column].astype("string")

    source_tables = output["source_table"].astype("string")
    positions = output["position"].astype("string").str.upper()
    matched = pd.Series(False, index=output.index, dtype=bool)
    for rule in _validated_source_row_exclusions():
        rule_mask = (
            source_tables.eq(str(rule["source_table"]))
            & positions.eq(str(rule["position"]).upper())
            & output[SOURCE_STORED_SEASON_COLUMN].eq(
                int(rule["stored_season"])
            )
        ).fillna(False)
        if not rule_mask.any():
            continue
        if (matched & rule_mask).any():
            raise ValueError(
                f"{frame_name} matches overlapping source-row exclusions"
            )

        expected_metadata = (
            (SOURCE_ROW_EXCLUSION_ID_COLUMN, str(rule["exclusion_id"])),
            (SOURCE_ROW_EXCLUSION_REASON_COLUMN, str(rule["reason"])),
            (SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN, str(rule["reference"])),
        )
        for column, expected in expected_metadata:
            existing = set(
                output.loc[rule_mask, column].dropna().astype(str)
            )
            if existing.difference({expected}):
                raise ValueError(
                    f"{frame_name} contains conflicting source-row exclusion "
                    f"metadata for {rule['exclusion_id']}: {sorted(existing)}"
                )
            output.loc[rule_mask, column] = expected
        matched |= rule_mask

    included = output.loc[~matched].copy()
    excluded = output.loc[matched].copy()
    return included, excluded


def apply_source_row_exclusions(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.DataFrame:
    """Drop every row covered by a governed source quarantine."""
    included, _ = partition_source_row_exclusions(frame, frame_name)
    return included


def assert_no_source_row_exclusions(
    frame: pd.DataFrame,
    frame_name: str,
) -> None:
    """Fail closed if a governed quarantine reaches a downstream boundary."""
    _, excluded = partition_source_row_exclusions(frame, frame_name)
    if excluded.empty:
        return
    counts = (
        excluded[SOURCE_ROW_EXCLUSION_ID_COLUMN]
        .astype(str)
        .value_counts()
        .sort_index()
    )
    details = ", ".join(f"{key}={int(value)}" for key, value in counts.items())
    raise ValueError(
        f"{frame_name} contains governed source rows that must be "
        f"quarantined: {details}"
    )


def apply_source_season_overrides(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.DataFrame:
    """Apply governed effective seasons while retaining source provenance."""
    require_columns(
        frame,
        ("source_table", "position", "season"),
        frame_name,
    )
    output = frame.copy()
    output["season"] = pd.to_numeric(
        output["season"], errors="coerce"
    ).astype("Int64")
    if SOURCE_STORED_SEASON_COLUMN not in output:
        output[SOURCE_STORED_SEASON_COLUMN] = output["season"]
    else:
        output[SOURCE_STORED_SEASON_COLUMN] = pd.to_numeric(
            output[SOURCE_STORED_SEASON_COLUMN],
            errors="coerce",
        ).astype("Int64")
    for column in (
        SOURCE_SEASON_OVERRIDE_ID_COLUMN,
        SOURCE_SEASON_OVERRIDE_REASON_COLUMN,
        SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN,
    ):
        if column not in output:
            output[column] = pd.Series(pd.NA, index=output.index, dtype="string")

    source_tables = output["source_table"].astype("string")
    positions = output["position"].astype("string").str.upper()
    for rule in SOURCE_SEASON_OVERRIDES:
        table_position = source_tables.eq(str(rule["source_table"])) & positions.eq(
            str(rule["position"]).upper()
        )
        override_rows = table_position & output[
            SOURCE_STORED_SEASON_COLUMN
        ].eq(int(rule["stored_season"]))
        if not override_rows.any():
            continue

        existing_ids = set(
            output.loc[
                override_rows,
                SOURCE_SEASON_OVERRIDE_ID_COLUMN,
            ]
            .dropna()
            .astype(str)
        )
        if existing_ids.difference({str(rule["override_id"])}):
            raise ValueError(
                f"{frame_name} contains conflicting source-season override "
                f"metadata for {rule['override_id']}: {sorted(existing_ids)}"
            )

        native_effective_rows = (
            table_position
            & output["season"].eq(int(rule["effective_season"]))
            & ~override_rows
        )
        if native_effective_rows.any():
            raise ValueError(
                f"{frame_name} cannot apply {rule['override_id']}: "
                f"{int(native_effective_rows.sum())} native "
                f"{rule['position']} rows already exist for effective season "
                f"{rule['effective_season']}"
            )

        output.loc[override_rows, "season"] = int(rule["effective_season"])
        output.loc[
            override_rows,
            SOURCE_SEASON_OVERRIDE_ID_COLUMN,
        ] = str(rule["override_id"])
        output.loc[
            override_rows,
            SOURCE_SEASON_OVERRIDE_REASON_COLUMN,
        ] = str(rule["reason"])
        output.loc[
            override_rows,
            SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN,
        ] = str(rule["reference"])
    return output


def align_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], frame_name: str
) -> pd.DataFrame:
    require_columns(frame, columns, frame_name)
    return frame.loc[:, list(columns)].copy()


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def read_existing_table(
    database_path: Path, table: str, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    if not database_path.exists():
        return pd.DataFrame(columns=list(columns or ()))
    with closing(sqlite3.connect(database_path)) as connection:
        if not table_exists(connection, table):
            return pd.DataFrame(columns=list(columns or ()))
        frame = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame.loc[:, list(columns)]
    return frame


def publish_tables_atomic(
    database_path: Path,
    tables: Mapping[str, pd.DataFrame],
    append_tables: Mapping[str, pd.DataFrame] | None = None,
    drop_tables: tuple[str, ...] = (),
) -> None:
    """Stage replacement tables, then publish all names in one transaction."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        staged_names: dict[str, str] = {}
        for table, frame in tables.items():
            staged = f"__v2_stage_{table}"
            frame.to_sql(staged, connection, if_exists="replace", index=False)
            staged_names[table] = staged

        connection.execute("BEGIN IMMEDIATE")
        try:
            for table, staged in staged_names.items():
                connection.execute(f'DROP TABLE IF EXISTS "{table}"')
                connection.execute(f'ALTER TABLE "{staged}" RENAME TO "{table}"')

            for table in drop_tables:
                if table not in staged_names:
                    connection.execute(f'DROP TABLE IF EXISTS "{table}"')

            if append_tables:
                for table, frame in append_tables.items():
                    frame.to_sql(table, connection, if_exists="append", index=False)

            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "player_identity" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_player_identity_key "
                    "ON player_identity(player_key)"
                )
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_player_identity_gsis "
                    "ON player_identity(gsis_id) WHERE gsis_id IS NOT NULL"
                )
            if "player_aliases" in existing_tables:
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS ix_v2_player_alias_lookup "
                    "ON player_aliases(normalized_name, position, season)"
                )
            if "player_season_outcomes" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_outcome_player_season "
                    "ON player_season_outcomes(player_key, season, league)"
                )
            if "player_season_sources" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_source_player_season "
                    "ON player_season_sources(player_key, season, source)"
                )
            if "player_season_spine" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_spine_player_season "
                    "ON player_season_spine(player_key, season, league)"
                )
            if "player_season_projection_values" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_projection_value "
                    "ON player_season_projection_values"
                    "(player_key, season, provider)"
                )
            if "player_season_market_values" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_market_value "
                    "ON player_season_market_values"
                    "(player_key, season, source)"
                )
            if "player_season_features" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_feature_player_season "
                    "ON player_season_features(player_key, season, league)"
                )
            if "feature_catalog" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_feature_catalog "
                    "ON feature_catalog(feature_name)"
                )
            if "feature_manifests" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_feature_manifest "
                    "ON feature_manifests(manifest_name, feature_name)"
                )
            if "feature_source_resolution_audit" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_feature_source_audit "
                    "ON feature_source_resolution_audit"
                    "(source_table, source_kind)"
                )
            if "model_runs" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_run "
                    "ON model_runs(run_id)"
                )
            if "model_fold_assignments" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_fold "
                    "ON model_fold_assignments"
                    "(run_id, target_name, player_key, season)"
                )
            if "model_specifications" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_spec "
                    "ON model_specifications(run_id, target_name, model_name)"
                )
            if "model_hyperparameter_results" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_hyperparameter "
                    "ON model_hyperparameter_results"
                    "(run_id, target_name, model_name, fold, trial)"
                )
            if "model_oof_predictions" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_oof "
                    "ON model_oof_predictions"
                    "(run_id, target_name, model_name, player_key, season)"
                )
            if "model_score_summary" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_score "
                    "ON model_score_summary"
                    "(run_id, target_name, model_name, aggregation, metric)"
                )
            if "model_slice_summary" in existing_tables:
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_v2_model_slice "
                    "ON model_slice_summary"
                    "(run_id, target_name, model_name, slice_type, "
                    "slice_value, metric)"
                )
            if "Avg_ADPs" in existing_tables:
                avg_adp_columns = {
                    str(row[1])
                    for row in connection.execute(
                        'PRAGMA table_info("Avg_ADPs")'
                    )
                }
                if {
                    "player_key",
                    "draft_entity_key",
                    "year",
                    "league",
                }.issubset(avg_adp_columns):
                    connection.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "ux_avg_adps_player_year_league "
                        "ON Avg_ADPs(player_key, year, league) "
                        "WHERE player_key IS NOT NULL"
                    )
                    connection.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "ux_avg_adps_entity_year_league "
                        "ON Avg_ADPs(draft_entity_key, year, league) "
                        "WHERE draft_entity_key IS NOT NULL"
                    )
            if "Avg_ADPs_Publication_Audit" in existing_tables:
                audit_columns = {
                    str(row[1])
                    for row in connection.execute(
                        'PRAGMA table_info("Avg_ADPs_Publication_Audit")'
                    )
                }
                if {
                    "draft_entity_key",
                    "year",
                    "league",
                }.issubset(audit_columns):
                    connection.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "ux_avg_adps_audit_entity_year_league "
                        "ON Avg_ADPs_Publication_Audit"
                        "(draft_entity_key, year, league) "
                        "WHERE draft_entity_key IS NOT NULL"
                    )
            if "Avg_ADPs_Publication_Receipt" in existing_tables:
                receipt_columns = {
                    str(row[1])
                    for row in connection.execute(
                        'PRAGMA table_info("Avg_ADPs_Publication_Receipt")'
                    )
                }
                if {"year", "league"}.issubset(receipt_columns):
                    connection.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "ux_avg_adps_receipt_year_league "
                        "ON Avg_ADPs_Publication_Receipt(year, league)"
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
