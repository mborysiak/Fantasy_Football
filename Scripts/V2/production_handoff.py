"""Publish the governed V2 production population and locked projections.

The V2 current and next-year shadow tables are the projection authority.  The
legacy production table is retained only as a comparison/rollback artifact; it
does not decide which players are eligible for production.

Eligibility is preseason-only:

* every current-year QB/RB/WR/TE ``ProjOnly`` player is in the core;
* DK adds the first 280 canonical players in current DK ADP;
* NFFC considers the first 363 canonical offensive players in current NFFC
  ADP so the three reviewed 2026 protected-market exclusions still leave a
  complete 360-player draft surface;
* beta and NV add the first 180 canonical players in current ETR overall rank
  and every current keeper in their respective league.

All source labels resolve through the governed V2 identity tables before they
can affect eligibility.  Core players and keepers always fail closed when their
current or next-year V2 handoff is incomplete.  New incomplete market-only
players inside the protected portion of a league's draft also fail closed
unless an explicit annual exclusion has been reviewed.  A market-only player
may be excluded automatically only from the final sixth of the draft surface,
the exclusion remains visible in the eligibility audit, and the remaining
eligible population must still cover the full draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from Scripts.V2.config import REPO_ROOT, SOURCE_DB_PATH, TEAM_MAP
from Scripts.V2.production_cycle import (
    DEFAULT_PRODUCTION_YEAR,
    get_production_cycle,
)
from Scripts.V2.contracts import (
    SOURCE_STORED_SEASON_COLUMN,
    apply_source_row_exclusions,
    assert_no_source_row_exclusions,
    normalize_player_name,
    publish_tables_atomic,
    scoring_hash,
    utc_now,
)


SIMULATION_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
MODEL_INPUTS_DB_PATH = (
    REPO_ROOT / "Data" / "Databases" / "Model_Inputs.sqlite3"
)
V2_DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "nffc": REPO_ROOT
    / "Data"
    / "Databases"
    / "Projection_V2_nffc.sqlite3",
    "beta": REPO_ROOT
    / "Data"
    / "Databases"
    / "Projection_V2_beta.sqlite3",
    "nv": REPO_ROOT
    / "Data"
    / "Databases"
    / "Projection_V2_nv.sqlite3",
}
PRODUCTION_YEAR = DEFAULT_PRODUCTION_YEAR
PRODUCTION_DATASET = "final_ensemble"
POSITIONS = ("QB", "RB", "WR", "TE")
POSITION_ORDER = {position: index for index, position in enumerate(POSITIONS)}
MARKET_ELIGIBILITY_RULES = {
    "dk": ("dk", 280, "dk_adp"),
    # A 12-team, 30-round NFFC room requires a 360-player offensive market
    # surface for the current offense-only runtime. The 2026 candidate window
    # reaches three rows deeper so its reviewed protected-market exclusions are
    # replaced instead of truncating availability.
    "nffc": ("nffc", 363, "nffc_adp"),
    # The canonical Avg_ADPs ``etr`` slice mirrors ETR_Ranks.etr_rank in
    # avg_pick while retaining exact etr_rank and etr_pos_rank fields.
    # The internal ``etr_adp`` label predates that contract; selection is
    # intentionally ordered by ETR overall rank.
    "beta": ("etr", 180, "etr_adp"),
    "nv": ("etr", 180, "etr_adp"),
}
# The minimum complete population must still cover the full draft.  Within that
# surface, the first five-sixths of expected picks are protected and fail closed
# unless an explicit annual exclusion has already been reviewed.
# An incomplete row in the final sixth may be omitted only when it is market-only
# (never a ProjOnly/core player or keeper).  This treats sparse tail ADP as a
# discovery source without inventing projection centers for fringe players.
MARKET_HANDOFF_REQUIRED_DEPTH = {
    "dk": 240,
    "nffc": 360,
    "beta": 180,
    "nv": 180,
}
MARKET_HANDOFF_PROTECTED_PICK_DEPTH = {
    league: (required_depth * 5) // 6
    for league, required_depth in MARKET_HANDOFF_REQUIRED_DEPTH.items()
}
AUTOMATIC_MARKET_BUFFER_EXCLUSION_REASON = (
    "market_buffer_only_without_complete_v2_handoff"
)
ELIGIBILITY_SOURCE_PRIORITY = {
    "core_projonly": 0,
    "league_keeper": 1,
    "dk_adp": 2,
    "nffc_adp": 2,
    "etr_adp": 2,
}
CURRENT_RESIDUAL_COLUMNS = (
    "pred_resid_5",
    "pred_resid_10",
    "pred_resid_25",
    "pred_resid_75",
    "pred_resid_90",
    "pred_resid_95",
)
NEXT_RESIDUAL_SOURCE_COLUMNS = {
    "pred_resid_5_ny_shadow": "pred_resid_5_ny",
    "pred_resid_10_ny_shadow": "pred_resid_10_ny",
    "pred_resid_25_ny_shadow": "pred_resid_25_ny",
    "pred_resid_75_ny_shadow": "pred_resid_75_ny",
    "pred_resid_90_ny_shadow": "pred_resid_90_ny",
    "pred_resid_95_ny_shadow": "pred_resid_95_ny",
}
PRODUCTION_HANDOFF_VERSION = "v2_current_next_production_handoff_v2"
PRODUCTION_ELIGIBILITY_VERSION = "v2_preseason_master_eligibility_v3"
PRODUCTION_EXCLUSION_POLICY_VERSION = (
    "v2_market_only_incomplete_buffer_exclusion_v4"
)
PRODUCTION_EXCLUSION_REFERENCE_BY_YEAR = {
    2026: (
        "2026 V2 feature mart + ProjOnly/salary-source coverage + "
        "season-ending injury review"
    ),
}
ELIGIBILITY_AUDIT_TABLE = "V2_Production_Eligibility_Audit"
LEGACY_BACKUP_TABLE = "V2_Projection_Legacy_Backup"
LEGACY_BACKUP_CREATED_AT_COLUMN = "backup_created_at_utc"
AVG_ADP_TABLE = "Avg_ADPs"
AVG_ADP_AUDIT_TABLE = "Avg_ADPs_Publication_Audit"
AVG_ADP_RECEIPT_TABLE = "Avg_ADPs_Publication_Receipt"
AVG_ADP_PUBLICATION_TABLES = (
    AVG_ADP_TABLE,
    AVG_ADP_AUDIT_TABLE,
    AVG_ADP_RECEIPT_TABLE,
)
AVG_ADP_PUBLICATION_VERSION = "canonical_current_market_v2"
AVG_ADP_SOURCE_LEAGUES = ("dk", "nffc", "etr")
AVG_ADP_ALLOWED_POSITIONS = {
    "dk": set(POSITIONS),
    "nffc": {*POSITIONS, "TK", "TDSP"},
    "etr": set(POSITIONS),
}
AVG_ADP_MIN_OFFENSIVE_DEPTH = {
    "dk": 280,
    "nffc": 363,
    "etr": 180,
}
GOVERNED_MARKET_POSITION_MISMATCHES_BY_YEAR: Mapping[
    int,
    Mapping[str, Mapping[str, str]],
] = {
    2026: {
        # Pittsburgh officially lists Heidenreich as RB/WR. DK currently
        # treats him as WR while the V2 feature mart uses RB.
        "ccd22510-248d-5fa0-a292-412d841e3f68": {
            "source_position": "WR",
            "authority_position": "RB",
            "reason": "official_rb_wr_hybrid_dk_wr_feature_mart_rb",
        },
    },
}

# Entries must be canonical player keys with a durable, non-empty reason.  The
# reviewed rows below predate the deterministic late-market rule or require an
# explicit annual decision.  New incomplete market-only rows are excluded
# automatically only when their expected pick is beyond the protected depth
# above; every other new missing-center case still fails closed.
GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR: Mapping[
    int,
    Mapping[str, Mapping[str, str]],
] = {
    2026: {
        "dk": {
            "ad848f28-4066-522c-b352-43abce87fbcb": (
                "season_ending_pcl_injury_adp_lag_without_current_"
                "projection_center"
            ),
            "3f0b675d-ef58-5606-8f9e-73bc2a9b4118": (
                "market_only_without_current_projection_center"
            ),
            "677b8fa5-8879-5913-8a35-9a71859ab8a3": (
                "market_only_without_current_projection_center"
            ),
            "7ae33581-c9ae-51b6-a8d5-fe24f3e5615a": (
                "market_only_without_current_projection_center"
            ),
            "e492c31b-21c9-55b9-b007-4dd0d8fd1ad4": (
                "market_only_without_current_projection_center"
            ),
            "f973b1c8-3470-57f5-bc68-42e35a830411": (
                "market_only_without_current_projection_center"
            ),
            "380d2c7d-99ef-5ddc-a057-fab93f1480ba": (
                "market_only_without_current_projection_center"
            ),
            "ffc8d08e-a9dd-51af-af68-8b032b066512": (
                "market_only_without_current_projection_center"
            ),
        },
        "nffc": {
            "ad848f28-4066-522c-b352-43abce87fbcb": (
                "season_ending_pcl_injury_adp_lag_without_current_"
                "projection_center"
            ),
            "06b12c47-18b2-51ac-ba66-64de763baac2": (
                "market_only_without_current_projection_center"
            ),
            "0fa72b32-393b-5f55-bb48-0f21f5283baf": (
                "market_only_without_current_projection_center"
            ),
            "0c370254-2acd-5345-a009-d0744ed3affe": (
                "market_only_without_current_projection_center"
            ),
            "31c3fcf7-3f74-524e-8b8f-67177f592742": (
                "market_only_without_current_projection_center"
            ),
            "3f0b675d-ef58-5606-8f9e-73bc2a9b4118": (
                "market_only_without_current_projection_center"
            ),
            "49dce437-30ec-5752-9739-75ed09f72042": (
                "market_only_without_current_projection_center"
            ),
            "7ae33581-c9ae-51b6-a8d5-fe24f3e5615a": (
                "market_only_without_current_projection_center"
            ),
            "862eb067-7abb-5156-9cf1-33c3ad11333c": (
                "market_only_without_current_projection_center"
            ),
            "f973b1c8-3470-57f5-bc68-42e35a830411": (
                "market_only_without_current_projection_center"
            ),
        },
        "beta": {},
        "nv": {},
    },
}


def governed_production_exclusions(
    year: int,
) -> Mapping[str, Mapping[str, str]]:
    """Return the reviewed exclusions for one season, never a stale prior set."""

    try:
        exclusions = GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR[int(year)]
    except KeyError as error:
        raise ValueError(
            f"No governed production-exclusion review is registered for "
            f"{year}"
        ) from error
    missing = sorted(set(V2_DATABASES).difference(exclusions))
    extra = sorted(set(exclusions).difference(V2_DATABASES))
    if missing or extra:
        raise ValueError(
            f"{year} governed exclusion leagues do not match V2 databases: "
            f"missing={missing}, extra={extra}"
        )
    return exclusions

HANDOFF_METADATA_COLUMNS = (
    "player_key",
    "current_projection_model_version",
    "next_projection_model_version",
    "v2_scoring_hash",
    "pred_appear_current",
    "pred_appear_ny",
    "production_handoff_version",
    "production_eligibility_version",
    "current_projection_source",
    "current_uncertainty_source",
    "independent_current_residual_draw_allowed",
    "next_projection_source",
    "next_uncertainty_source",
    "production_handoff_created_at_utc",
)


def _read_table(database: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)


def _table_exists(database: Path, table: str) -> bool:
    with sqlite3.connect(database) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            is not None
        )


def build_legacy_projection_backup(
    existing_backup: pd.DataFrame | None,
    prior_target: pd.DataFrame,
    *,
    year: int,
    dataset: str,
) -> pd.DataFrame:
    """Freeze one immutable pre-promotion baseline per season and dataset.

    The entire target-season slice is captured together.  Once that scope
    exists, later handoff reruns must not overwrite rows or backfill a league
    that was absent from the original legacy population.
    """

    context = f"{LEGACY_BACKUP_TABLE} {year}/{dataset}"
    _require_columns(
        prior_target,
        ("year", "dataset", "version"),
        f"{context} prior production slice",
    )
    target_years = pd.to_numeric(prior_target["year"], errors="coerce")
    target_datasets = prior_target["dataset"].astype("string")
    if prior_target.empty:
        target_matches_scope = pd.Series(dtype=bool)
    else:
        target_matches_scope = target_years.eq(int(year)) & (
            target_datasets.eq(str(dataset))
        )
    if not target_matches_scope.fillna(False).all():
        raise ValueError(
            f"{context} prior production slice contains rows outside its "
            "target season/dataset"
        )

    if existing_backup is not None:
        _require_columns(
            existing_backup,
            (
                "year",
                "dataset",
                "version",
                LEGACY_BACKUP_CREATED_AT_COLUMN,
            ),
            f"{context} existing backup",
        )
        backup_years = pd.to_numeric(
            existing_backup["year"],
            errors="coerce",
        )
        backup_datasets = existing_backup["dataset"].astype("string")
        existing_scope = backup_years.eq(int(year)) & (
            backup_datasets.eq(str(dataset))
        )
        if existing_scope.any():
            return existing_backup.copy()
    else:
        existing_backup = pd.DataFrame()

    if prior_target.empty:
        raise ValueError(
            f"{context} cannot initialize an immutable baseline from an "
            "empty prior production slice"
        )

    captured = prior_target.copy()
    captured[LEGACY_BACKUP_CREATED_AT_COLUMN] = utc_now()
    if existing_backup.empty:
        return captured
    return pd.concat(
        [existing_backup, captured],
        ignore_index=True,
        sort=False,
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _single_nonempty_lineage_value(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
) -> str:
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise ValueError(
            f"{context} has missing {column} lineage values"
        )
    observed = sorted(set(values.astype(str)))
    if len(observed) != 1:
        raise ValueError(
            f"{context} has multiple {column} lineage values: {observed}"
        )
    return observed[0]


def _require_lineage_year(
    frame: pd.DataFrame,
    column: str,
    expected: int,
    *,
    context: str,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"{context} has missing or invalid {column} lineage values"
        )
    observed = sorted(set(values.astype(float)))
    if len(observed) != 1 or observed[0] != float(expected):
        raise ValueError(
            f"{context} {column} mismatch: expected {expected}, "
            f"observed {observed}"
        )


def load_validated_shadow_predictions(
    database: Path,
    *,
    league: str,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current/next shadows only after their run lineage is coherent."""

    database = Path(database)
    league = str(league).strip().lower()
    context = f"{league} shadow lineage"
    cycle = get_production_cycle(year)
    if league not in cycle.leagues:
        raise ValueError(
            f"{context} is not registered for production season {year}"
        )
    if not database.is_file():
        raise ValueError(
            f"{context} database does not exist: {database}"
        )

    current_table = cycle.current_shadow_table
    next_table = cycle.next_shadow_table
    table_names = (
        "player_season_features",
        "locked_candidate_runs",
        current_table,
        next_table,
    )
    missing_tables = [
        table
        for table in table_names
        if not _table_exists(database, table)
    ]
    if missing_tables:
        raise ValueError(
            f"{context} is missing required tables: {missing_tables}"
        )

    features = _read_table(database, "player_season_features")
    locked_runs = _read_table(database, "locked_candidate_runs")
    current_shadow = _read_table(database, current_table)
    next_shadow = _read_table(database, next_table)
    _require_columns(
        features,
        ("run_id", "league", "scoring_hash"),
        f"{context} player_season_features",
    )
    _require_columns(
        locked_runs,
        (
            "lock_version",
            "model_run_id",
            "feature_run_id",
            "current_shadow_season",
            "status",
            "metadata_json",
        ),
        f"{context} locked_candidate_runs",
    )
    _require_columns(
        current_shadow,
        (
            "player_key",
            "lock_version",
            "model_run_id",
            "season",
            "publication_status",
        ),
        f"{context} {current_table}",
    )
    _require_columns(
        next_shadow,
        (
            "player_key",
            "run_id",
            "feature_run_id",
            "origin_season",
            "target_season",
            "target_version",
            "league",
            "scoring_hash",
            "publication_status",
        ),
        f"{context} {next_table}",
    )
    for table, frame in (
        ("player_season_features", features),
        (current_table, current_shadow),
        (next_table, next_shadow),
    ):
        if frame.empty:
            raise ValueError(f"{context} {table} is empty")

    active_feature_run = _single_nonempty_lineage_value(
        features,
        "run_id",
        context=f"{context} player_season_features",
    )
    feature_league = _single_nonempty_lineage_value(
        features,
        "league",
        context=f"{context} player_season_features",
    ).lower()
    if feature_league != league:
        raise ValueError(
            f"{context} player_season_features league mismatch: expected "
            f"{league}, observed {feature_league}"
        )
    expected_scoring_hash = scoring_hash(league)
    feature_scoring_hash = _single_nonempty_lineage_value(
        features,
        "scoring_hash",
        context=f"{context} player_season_features",
    )
    if feature_scoring_hash != expected_scoring_hash:
        raise ValueError(
            f"{context} player_season_features scoring_hash mismatch"
        )

    completed_runs = locked_runs[
        locked_runs["status"].astype("string").str.strip().eq(
            "complete_shadow"
        )
    ].copy()
    if len(completed_runs) != 1:
        raise ValueError(
            f"{context} requires exactly one complete_shadow "
            f"locked_candidate_runs row; observed {len(completed_runs)}"
        )
    completed_run = completed_runs.iloc[0]
    locked_feature_run = str(completed_run["feature_run_id"]).strip()
    if not locked_feature_run or pd.isna(completed_run["feature_run_id"]):
        raise ValueError(
            f"{context} complete locked run has missing feature_run_id"
        )
    if locked_feature_run != active_feature_run:
        raise ValueError(
            f"{context} locked feature_run_id is stale: expected "
            f"{active_feature_run}, observed {locked_feature_run}"
        )
    _require_lineage_year(
        completed_runs,
        "current_shadow_season",
        year,
        context=f"{context} locked_candidate_runs",
    )
    lock_version = str(completed_run["lock_version"]).strip()
    model_run_id = str(completed_run["model_run_id"]).strip()
    if (
        pd.isna(completed_run["lock_version"])
        or not lock_version
        or pd.isna(completed_run["model_run_id"])
        or not model_run_id
    ):
        raise ValueError(
            f"{context} complete locked run has missing run identifiers"
        )
    expected_lock_version = cycle.locked_versions[league]
    if lock_version != expected_lock_version:
        raise ValueError(
            f"{context} lock_version is not approved for {year}: expected "
            f"{expected_lock_version}, observed {lock_version}"
        )
    try:
        lock_metadata = json.loads(str(completed_run["metadata_json"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} complete locked run has invalid metadata_json"
        ) from exc
    if not isinstance(lock_metadata, dict):
        raise ValueError(
            f"{context} complete locked run metadata_json must be an object"
        )
    metadata_league = str(
        lock_metadata.get("scoring_objective", "")
    ).strip().lower()
    if metadata_league != league:
        raise ValueError(
            f"{context} locked run scoring objective mismatch: expected "
            f"{league}, observed {metadata_league or '<missing>'}"
        )
    metadata_lock_version = str(
        lock_metadata.get("lock_version", "")
    ).strip()
    if metadata_lock_version and metadata_lock_version != lock_version:
        raise ValueError(
            f"{context} locked run metadata lock_version mismatch"
        )

    current_lock_version = _single_nonempty_lineage_value(
        current_shadow,
        "lock_version",
        context=f"{context} {current_table}",
    )
    if current_lock_version != lock_version:
        raise ValueError(
            f"{context} current shadow lock_version mismatch: expected "
            f"{lock_version}, observed {current_lock_version}"
        )
    current_model_run = _single_nonempty_lineage_value(
        current_shadow,
        "model_run_id",
        context=f"{context} {current_table}",
    )
    if current_model_run != model_run_id:
        raise ValueError(
            f"{context} current shadow model_run_id mismatch: expected "
            f"{model_run_id}, observed {current_model_run}"
        )
    _require_lineage_year(
        current_shadow,
        "season",
        year,
        context=f"{context} {current_table}",
    )
    current_status = _single_nonempty_lineage_value(
        current_shadow,
        "publication_status",
        context=f"{context} {current_table}",
    )
    if current_status != "shadow":
        raise ValueError(
            f"{context} current publication_status mismatch: expected "
            f"shadow, observed {current_status}"
        )
    for column, expected in (
        ("league", league),
        ("scoring_hash", expected_scoring_hash),
    ):
        if column in current_shadow:
            observed = _single_nonempty_lineage_value(
                current_shadow,
                column,
                context=f"{context} {current_table}",
            )
            if (observed.lower() if column == "league" else observed) != (
                expected
            ):
                raise ValueError(
                    f"{context} current shadow {column} mismatch"
                )

    next_feature_run = _single_nonempty_lineage_value(
        next_shadow,
        "feature_run_id",
        context=f"{context} {next_table}",
    )
    if next_feature_run != active_feature_run:
        raise ValueError(
            f"{context} next shadow feature_run_id is stale: expected "
            f"{active_feature_run}, observed {next_feature_run}"
        )
    _single_nonempty_lineage_value(
        next_shadow,
        "run_id",
        context=f"{context} {next_table}",
    )
    next_target_version = _single_nonempty_lineage_value(
        next_shadow,
        "target_version",
        context=f"{context} {next_table}",
    )
    if next_target_version != cycle.next_target_version:
        raise ValueError(
            f"{context} target_version is not approved for {year}: expected "
            f"{cycle.next_target_version}, observed {next_target_version}"
        )
    _require_lineage_year(
        next_shadow,
        "origin_season",
        year,
        context=f"{context} {next_table}",
    )
    _require_lineage_year(
        next_shadow,
        "target_season",
        year + 1,
        context=f"{context} {next_table}",
    )
    next_league = _single_nonempty_lineage_value(
        next_shadow,
        "league",
        context=f"{context} {next_table}",
    ).lower()
    if next_league != league:
        raise ValueError(
            f"{context} next shadow league mismatch: expected {league}, "
            f"observed {next_league}"
        )
    next_scoring_hash = _single_nonempty_lineage_value(
        next_shadow,
        "scoring_hash",
        context=f"{context} {next_table}",
    )
    if next_scoring_hash != expected_scoring_hash:
        raise ValueError(f"{context} next shadow scoring_hash mismatch")
    next_status = _single_nonempty_lineage_value(
        next_shadow,
        "publication_status",
        context=f"{context} {next_table}",
    )
    if next_status != "shadow":
        raise ValueError(
            f"{context} next publication_status mismatch: expected shadow, "
            f"observed {next_status}"
        )

    for table, frame in (
        (current_table, current_shadow),
        (next_table, next_shadow),
    ):
        keys = frame["player_key"].astype("string").str.strip()
        if keys.isna().any() or keys.eq("").any():
            raise ValueError(f"{context} {table} has missing player_key")
        if keys.duplicated().any():
            raise ValueError(f"{context} {table} has duplicate player_key")

    return current_shadow, next_shadow


def _normalize_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    team = str(value).strip().upper()
    return TEAM_MAP.get(team, team) if team else None


def _prefer_unique_identity(
    candidates: pd.DataFrame,
    confirmed_keys: set[str],
) -> tuple[str | None, str | None]:
    if candidates.empty:
        return None, None
    keys = set(candidates["player_key"].dropna().astype(str))
    confirmed = keys.intersection(confirmed_keys)
    if len(confirmed) == 1:
        return next(iter(confirmed)), "confirmed_unique"
    if len(keys) == 1:
        return next(iter(keys)), "unique"
    return None, None


def _prepare_identity_frames(
    aliases: pd.DataFrame,
    identities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    _require_columns(
        aliases,
        ("player_key", "normalized_name", "position", "team", "season"),
        "player_aliases",
    )
    _require_columns(
        identities,
        (
            "player_key",
            "normalized_name",
            "position",
            "identity_status",
        ),
        "player_identity",
    )
    aliases = aliases.copy()
    identities = identities.copy()
    aliases["player_key"] = aliases["player_key"].astype("string")
    aliases["normalized_name"] = aliases["normalized_name"].astype("string")
    aliases["position"] = aliases["position"].astype("string").str.upper()
    aliases["season"] = pd.to_numeric(
        aliases["season"], errors="coerce"
    ).astype("Int64")
    aliases["_team_key"] = aliases["team"].map(_normalize_team)

    identities["player_key"] = identities["player_key"].astype("string")
    identities["normalized_name"] = identities["normalized_name"].astype(
        "string"
    )
    identities["position"] = (
        identities["position"].astype("string").str.upper()
    )
    latest_team = (
        identities["latest_team"]
        if "latest_team" in identities
        else pd.Series(pd.NA, index=identities.index)
    )
    draft_team = (
        identities["draft_team"]
        if "draft_team" in identities
        else pd.Series(pd.NA, index=identities.index)
    )
    identities["_latest_team_key"] = latest_team.map(_normalize_team)
    identities["_draft_team_key"] = draft_team.map(_normalize_team)
    known_keys = set(
        identities["player_key"].dropna().astype(str)
    )
    unknown_alias_keys = set(
        aliases["player_key"].dropna().astype(str)
    ).difference(known_keys)
    if unknown_alias_keys:
        raise ValueError(
            "V2 player aliases reference unknown canonical keys: "
            f"{sorted(unknown_alias_keys)[:10]}"
        )
    confirmed_keys = set(
        identities.loc[
            identities["identity_status"].eq("confirmed"), "player_key"
        ].dropna().astype(str)
    )
    return aliases, identities, confirmed_keys


def resolve_source_player_keys(
    rows: pd.DataFrame,
    aliases: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    year: int,
    source_name: str,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Resolve preseason source labels to canonical keys without fuzzy joins."""

    _require_columns(rows, ("player",), source_name)
    aliases, identities, confirmed_keys = _prepare_identity_frames(
        aliases,
        identities,
    )
    output = rows.copy().reset_index(drop=True)
    output["_normalized_name"] = output["player"].map(normalize_player_name)
    positions = (
        output["pos"].astype("string").str.upper()
        if "pos" in output
        else pd.Series(pd.NA, index=output.index, dtype="string")
    )
    teams = (
        output["team"].map(_normalize_team)
        if "team" in output
        else pd.Series(None, index=output.index, dtype=object)
    )
    aliases_year = aliases[aliases["season"].eq(int(year))].copy()
    resolved_keys: list[object] = []
    methods: list[str] = []

    for name, position, team in zip(
        output["_normalized_name"],
        positions,
        teams,
    ):
        if not name:
            resolved_keys.append(pd.NA)
            methods.append("unresolved_missing_name")
            continue

        alias_name_candidates = aliases_year[
            aliases_year["normalized_name"].eq(str(name))
        ]
        candidates = alias_name_candidates
        has_position = pd.notna(position) and str(position).strip() != ""
        if has_position:
            candidates = candidates[
                candidates["position"].eq(str(position).upper())
            ]
        player_key, suffix = _prefer_unique_identity(
            candidates,
            confirmed_keys,
        )
        method = f"alias_{suffix}" if suffix else None

        if player_key is None and team:
            team_candidates = candidates[candidates["_team_key"].eq(team)]
            player_key, suffix = _prefer_unique_identity(
                team_candidates,
                confirmed_keys,
            )
            method = f"alias_team_{suffix}" if suffix else None

        if player_key is None:
            identity_name_candidates = identities[
                identities["normalized_name"].eq(str(name))
            ]
            identity_candidates = identity_name_candidates
            if has_position:
                identity_candidates = identity_candidates[
                    identity_candidates["position"].eq(
                        str(position).upper()
                    )
                ]
            player_key, suffix = _prefer_unique_identity(
                identity_candidates,
                confirmed_keys,
            )
            method = f"identity_{suffix}" if suffix else None
            if player_key is None and team:
                team_candidates = identity_candidates[
                    identity_candidates["_latest_team_key"].eq(team)
                    | identity_candidates["_draft_team_key"].eq(team)
                ]
                player_key, suffix = _prefer_unique_identity(
                    team_candidates,
                    confirmed_keys,
                )
                method = (
                    f"identity_team_{suffix}" if suffix else None
                )

        # Hybrid players can be classified differently by a market provider
        # and the feature-mart position authority.  If the exact normalized
        # name still identifies one canonical player across positions, resolve
        # that key and retain the cross-position method for audit.  Ambiguous
        # names continue to fail closed.
        if player_key is None and has_position:
            if team:
                team_candidates = alias_name_candidates[
                    alias_name_candidates["_team_key"].eq(team)
                ]
                player_key, suffix = _prefer_unique_identity(
                    team_candidates,
                    confirmed_keys,
                )
                method = (
                    f"alias_cross_position_team_{suffix}"
                    if suffix
                    else None
                )
            if player_key is None:
                player_key, suffix = _prefer_unique_identity(
                    alias_name_candidates,
                    confirmed_keys,
                )
                method = (
                    f"alias_cross_position_{suffix}" if suffix else None
                )
            if player_key is None:
                identity_name_candidates = identities[
                    identities["normalized_name"].eq(str(name))
                ]
                if team:
                    team_candidates = identity_name_candidates[
                        identity_name_candidates["_latest_team_key"].eq(team)
                        | identity_name_candidates["_draft_team_key"].eq(team)
                    ]
                    player_key, suffix = _prefer_unique_identity(
                        team_candidates,
                        confirmed_keys,
                    )
                    method = (
                        f"identity_cross_position_team_{suffix}"
                        if suffix
                        else None
                    )
                if player_key is None:
                    player_key, suffix = _prefer_unique_identity(
                        identity_name_candidates,
                        confirmed_keys,
                    )
                    method = (
                        f"identity_cross_position_{suffix}"
                        if suffix
                        else None
                    )

        resolved_keys.append(
            player_key if player_key is not None else pd.NA
        )
        methods.append(method or "unresolved_ambiguous_identity")

    output["player_key"] = pd.Series(
        resolved_keys,
        index=output.index,
        dtype="string",
    )
    output["eligibility_key_match_method"] = methods
    if require_complete and output["player_key"].isna().any():
        preview_columns = ["player"]
        preview_columns.extend(
            column for column in ("pos", "team") if column in output
        )
        preview_columns.append("eligibility_key_match_method")
        preview = output.loc[
            output["player_key"].isna(), preview_columns
        ].head(20)
        raise ValueError(
            f"{source_name} contains unresolved canonical identities: "
            f"{preview.to_dict('records')}"
        )
    return output.drop(columns="_normalized_name")


def load_identity_frames(
    v2_database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(v2_database) as connection:
        alias_columns = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("player_aliases")'
            )
        }
        required_alias_columns = {
            "player_key",
            "normalized_name",
            "position",
            "team",
            "season",
            "source_table",
        }
        missing = sorted(required_alias_columns.difference(alias_columns))
        if missing:
            raise ValueError(
                "V2 player_aliases lacks governed provenance columns: "
                f"{missing}"
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
        identities = pd.read_sql_query(
            """
            SELECT player_key, display_name, normalized_name, position,
                   identity_status, latest_team, draft_team
            FROM player_identity
            """,
            connection,
        )
    source_tables = aliases["source_table"].astype("string").str.strip()
    if (source_tables.isna() | source_tables.eq("")).any():
        raise ValueError(
            "V2 player_aliases contains rows without source_table "
            "provenance"
        )
    aliases["source_table"] = source_tables
    aliases = apply_source_row_exclusions(
        aliases,
        "production eligibility player_aliases",
    )
    assert_no_source_row_exclusions(
        aliases,
        "production eligibility player_aliases after quarantine",
    )
    return aliases, identities


def _digest_cell(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Publication digests cannot contain non-finite values")
        return number
    if isinstance(value, (int, bool)):
        return value
    return str(value)


def _stable_frame_digest(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> str:
    _require_columns(frame, columns, "publication digest frame")
    serialized_rows = []
    for values in frame.loc[:, list(columns)].itertuples(
        index=False,
        name=None,
    ):
        record = {
            column: _digest_cell(value)
            for column, value in zip(columns, values)
        }
        serialized_rows.append(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
    digest = hashlib.sha256()
    for row in sorted(serialized_rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _row_digests(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    return pd.Series(
        [
            _stable_frame_digest(
                pd.DataFrame([dict(zip(columns, values))]),
                columns,
            )
            for values in frame.loc[:, list(columns)].itertuples(
                index=False,
                name=None,
            )
        ],
        index=frame.index,
        dtype="string",
    )


def _read_optional_table(database: Path, table: str) -> pd.DataFrame:
    if not _table_exists(database, table):
        return pd.DataFrame()
    return _read_table(database, table)


def _invalid_governed_year_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"year", "league"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    years = pd.to_numeric(frame["year"], errors="coerce")
    finite = years.notna() & np.isfinite(years)
    integral = finite & years.mod(1).eq(0)
    governed = (
        frame["league"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(AVG_ADP_SOURCE_LEAGUES)
    )
    return governed & ~integral


def _invalid_governed_year_counts(
    frame: pd.DataFrame | None,
) -> dict[str, int]:
    counts = {league: 0 for league in AVG_ADP_SOURCE_LEAGUES}
    if frame is None or frame.empty:
        return counts
    invalid = _invalid_governed_year_mask(frame)
    observed = (
        frame.loc[invalid, "league"]
        .astype("string")
        .str.strip()
        .str.lower()
        .value_counts()
    )
    for league in counts:
        counts[league] = int(observed.get(league, 0))
    return counts


def _replace_year_league_slices(
    existing: pd.DataFrame | None,
    current: pd.DataFrame,
    *,
    year: int,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return current.copy().reset_index(drop=True)
    prior = existing.copy()
    if {"year", "league"}.issubset(prior.columns):
        prior = prior.loc[~_invalid_governed_year_mask(prior)].copy()
        prior_year = pd.to_numeric(prior["year"], errors="coerce")
        prior_league = prior["league"].astype("string").str.lower()
        replace = prior_year.eq(int(year)) & prior_league.isin(
            AVG_ADP_SOURCE_LEAGUES
        )
        prior = prior.loc[~replace].copy()
    if prior.empty:
        return current.copy().reset_index(drop=True)
    if current.empty:
        return prior.reset_index(drop=True)
    # Pandas' deprecated concat inference ignores an all-NA input column when
    # the other frame contains values. Remove only those empty contributions
    # before concat, then restore the complete schema below. This preserves
    # every row/value while making the intended dtype inference explicit.
    prior_values = prior.dropna(axis=1, how="all")
    current_values = current.dropna(axis=1, how="all")
    combined = pd.concat(
        [prior_values, current_values],
        ignore_index=True,
        sort=False,
    )
    ordered = list(current.columns) + [
        column for column in combined.columns if column not in current.columns
    ]
    ordered.extend(
        column
        for column in prior.columns
        if column not in ordered
    )
    return combined.reindex(columns=ordered)


def _validate_source_market_frames(
    adp_rows: pd.DataFrame,
    etr_rows: pd.DataFrame,
    *,
    year: int,
    minimum_offensive_depth: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adp_required = (
        "player",
        "pos",
        "year",
        "avg_pick",
        "min_pick",
        "max_pick",
        "std_dev",
        "league",
    )
    adp_policy_columns = (
        "source_count",
        "feed_gap",
        "aggregation_policy",
        "bounds_policy",
        "std_dev_policy",
        "adp_policy_version",
    )
    etr_required = (
        "player",
        "team",
        "pos",
        "etr_rank",
        "etr_pos_rank",
        "etr_adp",
        "etr_adp_pos_rank",
        "etr_adp_diff",
        "year",
    )
    _require_columns(adp_rows, adp_required, "ADP_Averages")
    _require_columns(etr_rows, etr_required, "ETR_Ranks")

    adp = adp_rows.copy()
    for column in adp_policy_columns:
        if column not in adp:
            adp[column] = pd.NA
    adp = adp.loc[:, [*adp_required, *adp_policy_columns]].copy()
    adp["year"] = pd.to_numeric(adp["year"], errors="coerce").astype(
        "Int64"
    )
    adp["league"] = adp["league"].astype("string").str.strip().str.lower()
    adp["pos"] = adp["pos"].astype("string").str.strip().str.upper()
    adp = adp[
        adp["year"].eq(int(year))
        & adp["league"].isin(("dk", "nffc"))
    ].copy()

    etr = etr_rows.loc[:, list(etr_required)].copy()
    etr["year"] = pd.to_numeric(etr["year"], errors="coerce").astype(
        "Int64"
    )
    etr["pos"] = etr["pos"].astype("string").str.strip().str.upper()
    etr = etr[etr["year"].eq(int(year))].copy()

    missing_feeds = [
        league
        for league in ("dk", "nffc")
        if not adp["league"].eq(league).any()
    ]
    if etr.empty:
        missing_feeds.append("etr")
    if missing_feeds:
        raise ValueError(
            f"Current market publication is missing feeds: {missing_feeds}"
        )

    for league in ("dk", "nffc"):
        invalid_positions = sorted(
            set(
                adp.loc[
                    adp["league"].eq(league), "pos"
                ].dropna().astype(str)
            ).difference(AVG_ADP_ALLOWED_POSITIONS[league])
        )
        if invalid_positions:
            raise ValueError(
                f"{league} ADP_Averages contains unsupported positions: "
                f"{invalid_positions}"
            )
    invalid_etr_positions = sorted(
        set(etr["pos"].dropna().astype(str)).difference(
            AVG_ADP_ALLOWED_POSITIONS["etr"]
        )
    )
    if invalid_etr_positions:
        raise ValueError(
            "ETR_Ranks contains unsupported positions: "
            f"{invalid_etr_positions}"
        )

    for column in ("avg_pick", "min_pick", "max_pick", "std_dev"):
        adp[column] = pd.to_numeric(adp[column], errors="coerce")
    invalid_adp = (
        adp["avg_pick"].isna()
        | ~np.isfinite(adp["avg_pick"])
        | adp["avg_pick"].le(0)
    )
    if invalid_adp.any():
        raise ValueError(
            "ADP_Averages contains non-positive or non-finite avg_pick rows: "
            f"{adp.loc[invalid_adp, ['player', 'pos', 'league']].head(20).to_dict('records')}"
        )

    for column in (
        "etr_rank",
        "etr_pos_rank",
        "etr_adp",
        "etr_adp_pos_rank",
        "etr_adp_diff",
    ):
        etr[column] = pd.to_numeric(etr[column], errors="coerce")
    invalid_etr_rank = (
        etr["etr_rank"].isna()
        | ~np.isfinite(etr["etr_rank"])
        | etr["etr_rank"].le(0)
        | etr["etr_pos_rank"].isna()
        | ~np.isfinite(etr["etr_pos_rank"])
        | etr["etr_pos_rank"].le(0)
    )
    if invalid_etr_rank.any():
        raise ValueError(
            "ETR_Ranks contains invalid overall/position ranks: "
            f"{etr.loc[invalid_etr_rank, ['player', 'pos']].head(20).to_dict('records')}"
        )

    adp["_normalized_source_player"] = adp["player"].map(
        normalize_player_name
    )
    etr["_normalized_source_player"] = etr["player"].map(
        normalize_player_name
    )
    if (
        adp["_normalized_source_player"].eq("")
        | adp["_normalized_source_player"].isna()
    ).any():
        raise ValueError("ADP_Averages contains blank player/entity labels")
    if (
        etr["_normalized_source_player"].eq("")
        | etr["_normalized_source_player"].isna()
    ).any():
        raise ValueError("ETR_Ranks contains blank player labels")
    if adp.duplicated(
        ["year", "league", "pos", "_normalized_source_player"]
    ).any():
        raise ValueError(
            "ADP_Averages contains duplicate year/league/position entities"
        )
    if etr.duplicated(
        ["year", "pos", "_normalized_source_player"]
    ).any():
        raise ValueError(
            "ETR_Ranks contains duplicate year/position/player entities"
        )
    if etr["etr_rank"].duplicated().any():
        raise ValueError("ETR_Ranks contains duplicate overall ranks")
    if etr.duplicated(["pos", "etr_pos_rank"]).any():
        raise ValueError("ETR_Ranks contains duplicate position ranks")

    for league, minimum in minimum_offensive_depth.items():
        if league in ("dk", "nffc"):
            count = int(
                (
                    adp["league"].eq(league)
                    & adp["pos"].isin(POSITIONS)
                ).sum()
            )
        elif league == "etr":
            count = int(etr["pos"].isin(POSITIONS).sum())
        else:
            raise ValueError(
                f"Unsupported market depth rule for league {league}"
            )
        if count < int(minimum):
            raise ValueError(
                f"{league} current offensive market depth {count} is below "
                f"the required {int(minimum)}"
            )

    return (
        adp.drop(columns="_normalized_source_player").reset_index(drop=True),
        etr.drop(columns="_normalized_source_player").reset_index(drop=True),
    )


def validate_avg_adp_publication(
    frame: pd.DataFrame,
    *,
    year: int,
    minimum_offensive_depth: (
        Mapping[str, int] | None
    ) = None,
) -> None:
    """Fail closed on the canonical current market publication contract."""

    minimum_offensive_depth = (
        AVG_ADP_MIN_OFFENSIVE_DEPTH
        if minimum_offensive_depth is None
        else minimum_offensive_depth
    )
    required = (
        "player_key",
        "draft_entity_key",
        "player",
        "pos",
        "year",
        "league",
        "avg_pick",
        "etr_rank",
        "etr_pos_rank",
        "source_player",
        "source_pos",
        "identity_position",
        "current_position",
        "position_authority",
        "position_authority_source",
        "source_table",
        "source_metric",
        "identity_match_method",
        "source_row_sha256",
        "source_snapshot_sha256",
        "publication_snapshot_id",
        "publication_version",
        "published_at_utc",
        "removed_invalid_year_row_count",
    )
    _require_columns(frame, required, AVG_ADP_TABLE)
    current = frame[
        pd.to_numeric(frame["year"], errors="coerce").eq(int(year))
        & frame["league"].astype("string").str.lower().isin(
            AVG_ADP_SOURCE_LEAGUES
        )
    ].copy()
    observed_feeds = set(current["league"].dropna().astype(str))
    if observed_feeds != set(AVG_ADP_SOURCE_LEAGUES):
        raise ValueError(
            "Avg_ADPs current publication feeds differ from contract: "
            f"{sorted(observed_feeds)}"
        )
    current["pos"] = current["pos"].astype("string").str.upper()
    current["avg_pick"] = pd.to_numeric(
        current["avg_pick"], errors="coerce"
    )
    invalid_pick = (
        current["avg_pick"].isna()
        | ~np.isfinite(current["avg_pick"])
        | current["avg_pick"].le(0)
    )
    if invalid_pick.any():
        raise ValueError("Avg_ADPs current publication has invalid avg_pick")
    removed_invalid_year_count = pd.to_numeric(
        current["removed_invalid_year_row_count"],
        errors="coerce",
    )
    invalid_removed_count = (
        removed_invalid_year_count.isna()
        | removed_invalid_year_count.lt(0)
        | removed_invalid_year_count.mod(1).ne(0)
    )
    if invalid_removed_count.any():
        raise ValueError(
            "Avg_ADPs has invalid removed-invalid-year audit counts"
        )
    if (
        current.assign(
            _removed_invalid_year_count=removed_invalid_year_count
        )
        .groupby("league")["_removed_invalid_year_count"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Avg_ADPs removed-invalid-year audit count varies within feed"
        )

    entity_keys = current["draft_entity_key"].astype("string").str.strip()
    if entity_keys.isna().any() or entity_keys.eq("").any():
        raise ValueError(
            "Avg_ADPs current publication has missing draft_entity_key"
        )
    current["draft_entity_key"] = entity_keys
    if current.duplicated(["year", "league", "draft_entity_key"]).any():
        raise ValueError(
            "Avg_ADPs contains duplicate canonical draft entities"
        )

    offensive = current["pos"].isin(POSITIONS)
    keys = current["player_key"].astype("string").str.strip()
    if keys.loc[offensive].isna().any() or keys.loc[offensive].eq("").any():
        raise ValueError(
            "Avg_ADPs has unresolved offensive canonical player keys"
        )
    if keys.loc[~offensive].notna().any():
        raise ValueError(
            "Avg_ADPs non-player draft units must not claim player_key"
        )
    current["player_key"] = keys
    if current.loc[offensive].duplicated(
        ["year", "league", "player_key"]
    ).any():
        raise ValueError(
            "Avg_ADPs contains duplicate canonical offensive players"
        )
    position_authority = (
        current["position_authority"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    if (
        position_authority.loc[offensive].isna()
        | position_authority.loc[offensive].eq("")
        | position_authority.loc[offensive].ne(
            current.loc[offensive, "pos"]
        )
    ).any():
        raise ValueError(
            "Avg_ADPs offensive positions disagree with the published "
            "position authority"
        )
    authority_source = current[
        "position_authority_source"
    ].astype("string").str.strip()
    if (
        authority_source.loc[offensive].isna()
        | ~authority_source.loc[offensive].isin(
            {"player_season_features", "player_identity"}
        )
    ).any():
        raise ValueError(
            "Avg_ADPs offensive rows have invalid position authority source"
        )
    if (
        position_authority.loc[~offensive].notna().any()
        or authority_source.loc[~offensive].notna().any()
    ):
        raise ValueError(
            "Avg_ADPs non-player draft units cannot claim a player-position "
            "authority"
        )

    for league in AVG_ADP_SOURCE_LEAGUES:
        league_rows = current[current["league"].eq(league)]
        invalid_positions = sorted(
            set(league_rows["pos"].dropna().astype(str)).difference(
                AVG_ADP_ALLOWED_POSITIONS[league]
            )
        )
        if invalid_positions:
            raise ValueError(
                f"Avg_ADPs {league} has unsupported positions: "
                f"{invalid_positions}"
            )
        offensive_depth = int(league_rows["pos"].isin(POSITIONS).sum())
        minimum = int(minimum_offensive_depth[league])
        if offensive_depth < minimum:
            raise ValueError(
                f"Avg_ADPs {league} offensive depth {offensive_depth} is "
                f"below required {minimum}"
            )

    etr = current[current["league"].eq("etr")]
    etr_rank = pd.to_numeric(etr["etr_rank"], errors="coerce")
    etr_pos_rank = pd.to_numeric(etr["etr_pos_rank"], errors="coerce")
    if (
        etr_rank.isna()
        | etr_pos_rank.isna()
        | etr["avg_pick"].ne(etr_rank)
    ).any():
        raise ValueError(
            "Avg_ADPs ETR rows must preserve exact etr_rank/etr_pos_rank "
            "and use etr_rank as avg_pick"
        )
    non_etr = current["league"].ne("etr")
    if current.loc[non_etr, ["etr_rank", "etr_pos_rank"]].notna().any().any():
        raise ValueError(
            "Avg_ADPs non-ETR rows cannot contain ETR rank fields"
        )

    provenance_columns = (
        "source_table",
        "source_metric",
        "identity_match_method",
        "source_row_sha256",
        "source_snapshot_sha256",
        "publication_snapshot_id",
        "publication_version",
        "published_at_utc",
    )
    for column in provenance_columns:
        values = current[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(
                f"Avg_ADPs current publication has missing {column}"
            )


def build_current_avg_adp_publication(
    adp_rows: pd.DataFrame,
    etr_rows: pd.DataFrame,
    aliases: pd.DataFrame,
    identities: pd.DataFrame,
    season_features: pd.DataFrame,
    *,
    year: int,
    existing_avg_adps: pd.DataFrame | None = None,
    existing_audit: pd.DataFrame | None = None,
    existing_receipts: pd.DataFrame | None = None,
    published_at_utc: str | None = None,
    minimum_offensive_depth: (
        Mapping[str, int] | None
    ) = None,
    governed_position_mismatches: (
        Mapping[str, Mapping[str, str]] | None
    ) = None,
) -> dict[str, pd.DataFrame]:
    """Build one keyed DK/NFFC/ETR market snapshot without writing it."""

    minimum_offensive_depth = (
        AVG_ADP_MIN_OFFENSIVE_DEPTH
        if minimum_offensive_depth is None
        else minimum_offensive_depth
    )
    governed_position_mismatches = (
        GOVERNED_MARKET_POSITION_MISMATCHES_BY_YEAR.get(int(year))
        if governed_position_mismatches is None
        else governed_position_mismatches
    )
    if governed_position_mismatches is None:
        raise ValueError(
            f"No governed market-position review is registered for {year}"
        )
    adp, etr = _validate_source_market_frames(
        adp_rows,
        etr_rows,
        year=year,
        minimum_offensive_depth=minimum_offensive_depth,
    )
    newly_removed_invalid_year_counts = _invalid_governed_year_counts(
        existing_avg_adps
    )
    _require_columns(
        identities,
        (
            "player_key",
            "display_name",
            "position",
            "identity_status",
        ),
        "player_identity",
    )
    _require_columns(
        season_features,
        ("player_key", "season", "year_exp", "position"),
        "player_season_features",
    )

    adp_source_columns = (
        "player",
        "pos",
        "year",
        "avg_pick",
        "min_pick",
        "max_pick",
        "std_dev",
        "league",
        "source_count",
        "feed_gap",
        "aggregation_policy",
        "bounds_policy",
        "std_dev_policy",
        "adp_policy_version",
    )
    etr_source_columns = (
        "player",
        "team",
        "pos",
        "etr_rank",
        "etr_pos_rank",
        "etr_adp",
        "etr_adp_pos_rank",
        "etr_adp_diff",
        "year",
    )
    adp["source_row_sha256"] = _row_digests(adp, adp_source_columns)
    etr["source_row_sha256"] = _row_digests(etr, etr_source_columns)

    adp_standard = adp.rename(
        columns={
            "player": "source_player",
            "pos": "source_pos",
        }
    )
    adp_standard["source_team"] = pd.NA
    adp_standard["team"] = pd.NA
    adp_standard["etr_rank"] = np.nan
    adp_standard["etr_pos_rank"] = np.nan
    adp_standard["etr_adp"] = np.nan
    adp_standard["etr_adp_pos_rank"] = np.nan
    adp_standard["etr_adp_diff"] = np.nan
    adp_standard["source_table"] = "ADP_Averages"
    adp_standard["source_metric"] = "avg_pick"

    etr_standard = etr.rename(
        columns={
            "player": "source_player",
            "pos": "source_pos",
            "team": "source_team",
        }
    )
    etr_standard["league"] = "etr"
    etr_standard["avg_pick"] = etr_standard["etr_rank"]
    etr_standard["min_pick"] = np.nan
    etr_standard["max_pick"] = np.nan
    etr_standard["std_dev"] = np.nan
    etr_standard["team"] = etr_standard["source_team"]
    etr_standard["source_table"] = "ETR_Ranks"
    etr_standard["source_metric"] = "etr_rank"
    for column in (
        "source_count",
        "feed_gap",
        "aggregation_policy",
        "bounds_policy",
        "std_dev_policy",
        "adp_policy_version",
    ):
        etr_standard[column] = pd.NA

    standardized_columns = (
        "source_player",
        "source_pos",
        "source_team",
        "team",
        "year",
        "league",
        "avg_pick",
        "std_dev",
        "min_pick",
        "max_pick",
        "etr_rank",
        "etr_pos_rank",
        "etr_adp",
        "etr_adp_pos_rank",
        "etr_adp_diff",
        "source_table",
        "source_metric",
        "source_row_sha256",
        "source_count",
        "feed_gap",
        "aggregation_policy",
        "bounds_policy",
        "std_dev_policy",
        "adp_policy_version",
    )
    source = pd.concat(
        [
            adp_standard.loc[:, list(standardized_columns)],
            etr_standard.loc[:, list(standardized_columns)],
        ],
        ignore_index=True,
    )
    source["pos"] = source["source_pos"].astype("string").str.upper()
    source["player"] = source["source_player"]
    source["player_key"] = pd.Series(
        pd.NA,
        index=source.index,
        dtype="string",
    )
    source["identity_match_method"] = "non_player_draft_unit"

    offensive = source["pos"].isin(POSITIONS)
    resolved_frames = []
    for league in AVG_ADP_SOURCE_LEAGUES:
        league_mask = source["league"].eq(league) & offensive
        league_rows = source.loc[league_mask].copy()
        league_rows["_publication_row_index"] = league_rows.index
        league_rows["player"] = league_rows["source_player"]
        league_rows["pos"] = league_rows["source_pos"]
        league_rows["team"] = league_rows["source_team"]
        league_rows.drop(columns="identity_match_method", inplace=True)
        resolved = resolve_source_player_keys(
            league_rows,
            aliases,
            identities,
            year=year,
            source_name=f"{league}_current_market_publication",
        ).rename(
            columns={
                "eligibility_key_match_method": "identity_match_method"
            }
        )
        resolved_frames.append(
            resolved.set_index("_publication_row_index", drop=True)
        )
    resolved = pd.concat(resolved_frames, ignore_index=False, sort=False)
    source.loc[resolved.index, "player_key"] = resolved["player_key"]
    source.loc[resolved.index, "identity_match_method"] = resolved[
        "identity_match_method"
    ]

    identity_columns = ["player_key", "display_name", "position"]
    if "latest_team" in identities:
        identity_columns.append("latest_team")
    canonical = identities.loc[:, identity_columns].copy()
    canonical["player_key"] = canonical["player_key"].astype("string")
    if canonical["player_key"].duplicated().any():
        raise ValueError("player_identity contains duplicate player_key rows")
    features = season_features.loc[
        pd.to_numeric(season_features["season"], errors="coerce").eq(
            int(year)
        ),
        ["player_key", "year_exp", "position"],
    ].copy()
    features["player_key"] = features["player_key"].astype("string")
    features["position"] = (
        features["position"].astype("string").str.strip().str.upper()
    )
    if features["player_key"].duplicated().any():
        raise ValueError(
            f"player_season_features has duplicate {year} player_key rows"
        )
    features.rename(
        columns={
            "year_exp": "Years_of_Experience",
            "position": "current_position",
        },
        inplace=True,
    )

    keyed_source = source.loc[
        offensive,
        ["player_key", "source_pos"],
    ].copy()
    keyed_source["_source_index"] = keyed_source.index
    keyed = keyed_source.merge(
        canonical,
        on="player_key",
        how="left",
        validate="many_to_one",
        indicator=True,
    ).merge(
        features[["player_key", "current_position"]],
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    if keyed["_merge"].ne("both").any():
        raise ValueError(
            "Current market publication resolved keys absent from "
            "player_identity"
        )
    keyed["position_authority"] = keyed["current_position"].where(
        keyed["current_position"].notna()
        & keyed["current_position"].ne(""),
        keyed["position"].astype("string").str.strip().str.upper(),
    )
    missing_position_authority = (
        keyed["position_authority"].isna()
        | keyed["position_authority"].eq("")
    )
    if missing_position_authority.any():
        raise ValueError(
            "Current market publication has no current/canonical position "
            "authority for resolved offensive rows: "
            f"{keyed.loc[missing_position_authority, ['player_key', 'source_pos']].head(20).to_dict('records')}"
        )
    position_mismatch = (
        keyed["source_pos"].astype("string").str.upper()
        != keyed["position_authority"]
    )
    keyed["position_mismatch_reason"] = pd.Series(
        pd.NA,
        index=keyed.index,
        dtype="string",
    )
    for row_index in keyed.index[position_mismatch]:
        player_key = str(keyed.at[row_index, "player_key"])
        governance = governed_position_mismatches.get(player_key)
        if governance is None:
            continue
        source_position = str(
            governance.get("source_position", "")
        ).upper()
        authority_position = str(
            governance.get("authority_position", "")
        ).upper()
        reason = str(governance.get("reason", "")).strip()
        if (
            source_position
            == str(keyed.at[row_index, "source_pos"]).upper()
            and authority_position
            == str(keyed.at[row_index, "position_authority"]).upper()
            and reason
        ):
            keyed.at[row_index, "position_mismatch_reason"] = reason
    ungoverned_position_mismatch = (
        position_mismatch & keyed["position_mismatch_reason"].isna()
    )
    if ungoverned_position_mismatch.any():
        raise ValueError(
            "Current market source positions disagree with current canonical "
            "position authority: "
            f"{keyed.loc[ungoverned_position_mismatch, ['player_key', 'source_pos', 'current_position', 'position', 'position_authority']].head(20).to_dict('records')}"
        )
    source["position_mismatch_governed"] = 0
    source["position_mismatch_reason"] = pd.Series(
        pd.NA,
        index=source.index,
        dtype="string",
    )
    governed_rows = keyed["position_mismatch_reason"].notna()
    if governed_rows.any():
        governed_source_indexes = keyed.loc[
            governed_rows,
            "_source_index",
        ].astype(int)
        source.loc[
            governed_source_indexes,
            "position_mismatch_governed",
        ] = 1
        source.loc[
            governed_source_indexes,
            "position_mismatch_reason",
        ] = keyed.loc[
            governed_rows,
            "position_mismatch_reason",
        ].to_numpy()
    canonical = canonical.rename(
        columns={
            "display_name": "_canonical_player",
            "position": "identity_position",
            "latest_team": "_canonical_team",
        }
    )
    source = source.merge(
        canonical,
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    source.loc[offensive, "player"] = source.loc[
        offensive, "_canonical_player"
    ]
    if "_canonical_team" in source:
        source.loc[offensive, "team"] = source.loc[
            offensive, "_canonical_team"
        ]
    source.drop(
        columns=[
            column
            for column in (
                "_canonical_player",
                "_canonical_team",
            )
            if column in source
        ],
        inplace=True,
    )

    source = source.merge(
        features,
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    source["position_authority"] = source["current_position"].where(
        source["current_position"].notna()
        & source["current_position"].ne(""),
        source["identity_position"],
    )
    source["position_authority_source"] = np.where(
        source["current_position"].notna()
        & source["current_position"].ne(""),
        "player_season_features",
        "player_identity",
    )
    source.loc[offensive, "pos"] = source.loc[
        offensive,
        "position_authority",
    ]
    source.loc[
        ~source["pos"].isin(POSITIONS),
        ["position_authority", "position_authority_source"],
    ] = pd.NA
    source["draft_entity_key"] = (
        "player:" + source["player_key"].astype("string")
    )
    unit_mask = ~source["pos"].isin(POSITIONS)
    source.loc[unit_mask, "draft_entity_key"] = source.loc[
        unit_mask
    ].apply(
        lambda row: (
            f"market_unit:{int(row['year'])}:{row['league']}:"
            f"{row['pos']}:{hashlib.sha256(str(row['source_player']).encode('utf-8')).hexdigest()[:16]}"
        ),
        axis=1,
    )

    source_digests = {
        "dk": _stable_frame_digest(
            adp[adp["league"].eq("dk")],
            adp_source_columns,
        ),
        "nffc": _stable_frame_digest(
            adp[adp["league"].eq("nffc")],
            adp_source_columns,
        ),
        "etr": _stable_frame_digest(etr, etr_source_columns),
    }
    source_counts = {
        "dk": int(adp["league"].eq("dk").sum()),
        "nffc": int(adp["league"].eq("nffc").sum()),
        "etr": int(len(etr)),
    }
    current_publish_time = published_at_utc or utc_now()
    prior_receipts = (
        pd.DataFrame()
        if existing_receipts is None
        else existing_receipts.copy()
    )
    removed_invalid_year_counts = (
        newly_removed_invalid_year_counts.copy()
    )
    prior_removal_columns = {
        "year",
        "league",
        "publication_version",
        "source_snapshot_sha256",
        "removed_invalid_year_row_count",
    }
    if (
        not prior_receipts.empty
        and prior_removal_columns.issubset(prior_receipts.columns)
    ):
        for league in AVG_ADP_SOURCE_LEAGUES:
            matching = prior_receipts[
                pd.to_numeric(
                    prior_receipts["year"], errors="coerce"
                ).eq(int(year))
                & prior_receipts["league"].astype("string").eq(league)
                & prior_receipts["publication_version"].astype("string").eq(
                    AVG_ADP_PUBLICATION_VERSION
                )
                & prior_receipts[
                    "source_snapshot_sha256"
                ].astype("string").eq(source_digests[league])
            ]
            if len(matching) > 1:
                raise ValueError(
                    f"Duplicate prior Avg_ADPs receipts for {year} {league}"
                )
            if len(matching) == 1:
                prior_count = pd.to_numeric(
                    matching.iloc[0][
                        "removed_invalid_year_row_count"
                    ],
                    errors="coerce",
                )
                if pd.isna(prior_count) or float(prior_count) < 0:
                    raise ValueError(
                        f"Invalid prior removed-year count for {year} "
                        f"{league}"
                    )
                removed_invalid_year_counts[league] = max(
                    removed_invalid_year_counts[league],
                    int(prior_count),
                )
    source["removed_invalid_year_row_count"] = (
        source["league"]
        .map(removed_invalid_year_counts)
        .fillna(0)
        .astype(int)
    )

    publish_columns = (
        "player_key",
        "draft_entity_key",
        "player",
        "pos",
        "team",
        "Years_of_Experience",
        "avg_pick",
        "year",
        "league",
        "std_dev",
        "min_pick",
        "max_pick",
        "etr_rank",
        "etr_pos_rank",
        "etr_adp",
        "etr_adp_pos_rank",
        "etr_adp_diff",
        "source_player",
        "source_pos",
        "source_team",
        "identity_position",
        "current_position",
        "position_authority",
        "position_authority_source",
        "position_mismatch_governed",
        "position_mismatch_reason",
        "identity_match_method",
        "source_table",
        "source_metric",
        "source_row_sha256",
        "source_count",
        "feed_gap",
        "aggregation_policy",
        "bounds_policy",
        "std_dev_policy",
        "adp_policy_version",
        "removed_invalid_year_row_count",
    )
    current = source.loc[:, list(publish_columns)].copy()
    receipt_rows = []
    enriched = []
    digest_columns = tuple(
        column
        for column in publish_columns
        if column not in ("source_row_sha256",)
    )
    for league in AVG_ADP_SOURCE_LEAGUES:
        league_rows = current[current["league"].eq(league)].copy()
        published_digest = _stable_frame_digest(
            league_rows,
            digest_columns,
        )
        source_digest = source_digests[league]
        snapshot_id = (
            f"{year}:{league}:{AVG_ADP_PUBLICATION_VERSION}:"
            f"{source_digest[:16]}"
        )
        league_published_at = current_publish_time
        if not prior_receipts.empty and {
            "year",
            "league",
            "publication_version",
            "source_snapshot_sha256",
            "published_snapshot_sha256",
            "published_at_utc",
        }.issubset(prior_receipts.columns):
            matching = prior_receipts[
                pd.to_numeric(
                    prior_receipts["year"], errors="coerce"
                ).eq(int(year))
                & prior_receipts["league"].astype("string").eq(league)
                & prior_receipts["publication_version"].astype("string").eq(
                    AVG_ADP_PUBLICATION_VERSION
                )
                & prior_receipts[
                    "source_snapshot_sha256"
                ].astype("string").eq(source_digest)
                & prior_receipts[
                    "published_snapshot_sha256"
                ].astype("string").eq(published_digest)
            ]
            if len(matching) > 1:
                raise ValueError(
                    f"Duplicate prior Avg_ADPs receipts for {year} {league}"
                )
            if len(matching) == 1:
                league_published_at = str(
                    matching.iloc[0]["published_at_utc"]
                )
        league_rows["source_snapshot_sha256"] = source_digest
        league_rows["publication_snapshot_id"] = snapshot_id
        league_rows["publication_version"] = AVG_ADP_PUBLICATION_VERSION
        league_rows["published_at_utc"] = league_published_at
        enriched.append(league_rows)
        receipt_rows.append(
            {
                "year": int(year),
                "league": league,
                "source_table": (
                    "ETR_Ranks" if league == "etr" else "ADP_Averages"
                ),
                "source_metric": (
                    "etr_rank" if league == "etr" else "avg_pick"
                ),
                "source_row_count": source_counts[league],
                "published_row_count": int(len(league_rows)),
                "removed_invalid_year_row_count": (
                    removed_invalid_year_counts[league]
                ),
                "source_snapshot_sha256": source_digest,
                "published_snapshot_sha256": published_digest,
                "publication_snapshot_id": snapshot_id,
                "publication_version": AVG_ADP_PUBLICATION_VERSION,
                "published_at_utc": league_published_at,
            }
        )
    current = pd.concat(enriched, ignore_index=True, sort=False)
    current = current.sort_values(
        ["year", "league", "avg_pick", "pos", "draft_entity_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_avg_adp_publication(
        current,
        year=year,
        minimum_offensive_depth=minimum_offensive_depth,
    )
    expected_counts = pd.Series(source_counts, name="expected").sort_index()
    actual_counts = current.groupby("league").size().sort_index()
    if not actual_counts.equals(expected_counts):
        raise ValueError(
            "Current market publication did not preserve every source row: "
            f"expected={expected_counts.to_dict()}, "
            f"actual={actual_counts.to_dict()}"
        )

    audit_columns = [
        "player_key",
        "draft_entity_key",
        "player",
        "pos",
        "year",
        "league",
        "avg_pick",
        "etr_rank",
        "etr_pos_rank",
        "source_player",
        "source_pos",
        "source_team",
        "identity_position",
        "current_position",
        "position_authority",
        "position_authority_source",
        "identity_match_method",
        "source_table",
        "source_metric",
        "source_row_sha256",
        "source_count",
        "feed_gap",
        "aggregation_policy",
        "bounds_policy",
        "std_dev_policy",
        "adp_policy_version",
        "removed_invalid_year_row_count",
        "source_snapshot_sha256",
        "publication_snapshot_id",
        "publication_version",
        "published_at_utc",
    ]
    current_audit = current.loc[:, audit_columns].copy()
    receipts = pd.DataFrame(receipt_rows)
    combined_avg_adps = _replace_year_league_slices(
        existing_avg_adps,
        current,
        year=year,
    )
    combined_audit = _replace_year_league_slices(
        existing_audit,
        current_audit,
        year=year,
    )
    combined_receipts = _replace_year_league_slices(
        existing_receipts,
        receipts,
        year=year,
    )
    return {
        AVG_ADP_TABLE: combined_avg_adps,
        AVG_ADP_AUDIT_TABLE: combined_audit,
        AVG_ADP_RECEIPT_TABLE: combined_receipts,
    }


def load_current_avg_adp_publication(
    *,
    source_db: Path = SOURCE_DB_PATH,
    simulation_db: Path = SIMULATION_DB_PATH,
    v2_database: Path | None = None,
    year: int = PRODUCTION_YEAR,
) -> dict[str, pd.DataFrame]:
    """Load source/identity data and build the current market publication."""

    v2_database = (
        V2_DATABASES["dk"] if v2_database is None else Path(v2_database)
    )
    with sqlite3.connect(source_db) as connection:
        adp_columns = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("ADP_Averages")'
            )
        }
        policy_columns = (
            "source_count",
            "feed_gap",
            "aggregation_policy",
            "bounds_policy",
            "std_dev_policy",
            "adp_policy_version",
        )
        policy_select = ",\n                   ".join(
            column if column in adp_columns else f"NULL AS {column}"
            for column in policy_columns
        )
        adp_rows = pd.read_sql_query(
            f"""
            SELECT player, pos, year, avg_pick, min_pick, max_pick,
                   std_dev, league,
                   {policy_select}
            FROM ADP_Averages
            WHERE CAST(year AS INTEGER)=?
              AND LOWER(league) IN ('dk', 'nffc')
            """,
            connection,
            params=(int(year),),
        )
        etr_rows = pd.read_sql_query(
            """
            SELECT player, team, pos, etr_rank, etr_pos_rank, etr_adp,
                   etr_adp_pos_rank, etr_adp_diff, year
            FROM ETR_Ranks
            WHERE CAST(year AS INTEGER)=?
            """,
            connection,
            params=(int(year),),
        )
    aliases, identities = load_identity_frames(v2_database)
    with sqlite3.connect(v2_database) as connection:
        season_features = pd.read_sql_query(
            """
            SELECT player_key, season, year_exp, position
            FROM player_season_features
            WHERE CAST(season AS INTEGER)=?
            """,
            connection,
            params=(int(year),),
        )
    return build_current_avg_adp_publication(
        adp_rows,
        etr_rows,
        aliases,
        identities,
        season_features,
        year=year,
        existing_avg_adps=_read_optional_table(
            simulation_db,
            AVG_ADP_TABLE,
        ),
        existing_audit=_read_optional_table(
            simulation_db,
            AVG_ADP_AUDIT_TABLE,
        ),
        existing_receipts=_read_optional_table(
            simulation_db,
            AVG_ADP_RECEIPT_TABLE,
        ),
    )


def publish_current_avg_adps(
    *,
    source_db: Path = SOURCE_DB_PATH,
    simulation_db: Path = SIMULATION_DB_PATH,
    v2_database: Path | None = None,
    year: int = PRODUCTION_YEAR,
) -> dict[str, pd.DataFrame]:
    """Atomically publish only the canonical current market tables."""

    tables = load_current_avg_adp_publication(
        source_db=source_db,
        simulation_db=simulation_db,
        v2_database=v2_database,
        year=year,
    )
    publish_tables_atomic(simulation_db, tables)
    return tables


def _load_projonly_core(
    model_inputs_db: Path,
    *,
    year: int,
) -> pd.DataFrame:
    frames = []
    with sqlite3.connect(model_inputs_db) as connection:
        for position in POSITIONS:
            table = f"{position}_{year}_ProjOnly"
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is None:
                raise ValueError(
                    f"Missing current production core table: {table}"
                )
            columns = {
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                )
            }
            required = {"player", "pos", "team", "year"}
            missing = sorted(required.difference(columns))
            if missing:
                raise ValueError(
                    f"{table} is missing production core columns: {missing}"
                )
            frame = pd.read_sql_query(
                f"""
                SELECT player, pos, team, CAST(year AS INTEGER) year
                FROM "{table}"
                WHERE CAST(year AS INTEGER)=?
                  AND UPPER(pos)=?
                """,
                connection,
                params=(int(year), position),
            )
            frames.append(frame)
    core = pd.concat(frames, ignore_index=True)
    if core.empty:
        raise ValueError(f"No current ProjOnly production core for {year}")
    core["eligibility_source"] = "core_projonly"
    core["source_rank"] = np.nan
    core["source_value"] = np.nan
    return core


def _load_market_source(
    simulation_db: Path,
    *,
    year: int,
    market_league: str,
    avg_adps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if avg_adps is None:
        with sqlite3.connect(simulation_db) as connection:
            columns = {
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{AVG_ADP_TABLE}")'
                )
            }
            required = {
                "player_key",
                "player",
                "pos",
                "year",
                "league",
                "avg_pick",
            }
            missing = sorted(required.difference(columns))
            if missing:
                raise ValueError(
                    "Production Avg_ADPs is not the canonical keyed schema: "
                    f"{missing}"
                )
            frame = pd.read_sql_query(
                f"""
                SELECT *
                FROM "{AVG_ADP_TABLE}"
                WHERE CAST(year AS INTEGER)=?
                  AND league=?
                """,
                connection,
                params=(int(year), market_league),
            )
    else:
        _require_columns(
            avg_adps,
            ("player_key", "player", "pos", "year", "league", "avg_pick"),
            AVG_ADP_TABLE,
        )
        frame = avg_adps[
            pd.to_numeric(avg_adps["year"], errors="coerce").eq(int(year))
            & avg_adps["league"].astype("string").eq(market_league)
        ].copy()
    if frame.empty:
        raise ValueError(
            f"No {year} Avg_ADPs rows for market source {market_league}"
        )
    frame["pos"] = (
        frame["pos"].astype("string").str.strip().str.upper()
    )
    frame = frame[frame["pos"].isin(POSITIONS)].copy()
    if frame.empty:
        raise ValueError(
            f"No offensive {year} Avg_ADPs rows for {market_league}"
        )
    frame["avg_pick"] = pd.to_numeric(frame["avg_pick"], errors="coerce")
    frame = frame[
        frame["avg_pick"].notna() & np.isfinite(frame["avg_pick"])
    ].copy()
    if frame.empty:
        raise ValueError(
            f"No finite {year} Avg_ADPs values for {market_league}"
        )
    player_keys = frame["player_key"].astype("string").str.strip()
    if player_keys.isna().any() or player_keys.eq("").any():
        raise ValueError(
            f"{year} {market_league} production Avg_ADPs contains "
            "unkeyed market rows"
        )
    frame["player_key"] = player_keys
    if frame["player_key"].duplicated().any():
        raise ValueError(
            f"{year} {market_league} production Avg_ADPs contains "
            "duplicate player_key rows"
        )
    return frame


def _load_keeper_source(
    simulation_db: Path,
    *,
    year: int,
    league: str,
) -> pd.DataFrame:
    if not _table_exists(simulation_db, "League_Keepers"):
        raise ValueError(
            "League_Keepers is required for auction-league production "
            "eligibility"
        )
    with sqlite3.connect(simulation_db) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("League_Keepers")'
            )
        }
        select_columns = ["player"]
        select_columns.extend(
            column for column in ("pos", "team") if column in columns
        )
        frame = pd.read_sql_query(
            f"""
            SELECT {", ".join(select_columns)}
            FROM League_Keepers
            WHERE CAST(year AS INTEGER)=?
              AND league=?
            """,
            connection,
            params=(int(year), league),
        )
    frame["eligibility_source"] = "league_keeper"
    frame["source_rank"] = np.nan
    frame["source_value"] = np.nan
    return frame


def build_eligibility_membership(
    core_rows: pd.DataFrame,
    market_rows: pd.DataFrame,
    keeper_rows: pd.DataFrame,
    aliases: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    league: str,
    year: int,
    market_limit: int | None = None,
    market_source_name: str | None = None,
) -> pd.DataFrame:
    """Return one deterministic canonical eligibility row per required key."""

    if league not in MARKET_ELIGIBILITY_RULES:
        raise ValueError(f"Unsupported production league: {league}")
    configured_market, configured_limit, configured_source = (
        MARKET_ELIGIBILITY_RULES[league]
    )
    del configured_market
    market_limit = configured_limit if market_limit is None else market_limit
    market_source_name = (
        configured_source
        if market_source_name is None
        else market_source_name
    )

    core = resolve_source_player_keys(
        core_rows,
        aliases,
        identities,
        year=year,
        source_name=f"{league}_current_projonly_core",
    )
    core["eligibility_source"] = "core_projonly"
    core["source_rank"] = np.nan
    core["source_value"] = np.nan

    if "player_key" in market_rows:
        market = market_rows.copy()
        market["player_key"] = (
            market["player_key"].astype("string").str.strip()
        )
        if (
            market["player_key"].isna()
            | market["player_key"].eq("")
        ).any():
            raise ValueError(
                f"{league} keyed market rows contain missing player_key"
            )
        known_keys = set(
            identities["player_key"].dropna().astype(str)
        )
        unknown_keys = sorted(
            set(market["player_key"].astype(str)).difference(known_keys)
        )
        if unknown_keys:
            raise ValueError(
                f"{league} keyed market rows reference unknown identities: "
                f"{unknown_keys[:20]}"
            )
        if market["player_key"].duplicated().any():
            raise ValueError(
                f"{league} keyed market rows contain duplicate player_key"
            )
        if "identity_match_method" in market:
            market["eligibility_key_match_method"] = market[
                "identity_match_method"
            ].astype("string").fillna("published_player_key")
        else:
            market["eligibility_key_match_method"] = (
                "published_player_key"
            )
    else:
        # Legacy fallback for old fixtures/databases only. Production loading
        # requires the keyed schema in _load_market_source.
        market = resolve_source_player_keys(
            market_rows,
            aliases,
            identities,
            year=year,
            source_name=f"{league}_{market_source_name}",
        )
    _require_columns(market, ("avg_pick",), f"{league}_market_rows")
    market["avg_pick"] = pd.to_numeric(
        market["avg_pick"], errors="coerce"
    )
    if (
        market["avg_pick"].isna()
        | ~np.isfinite(market["avg_pick"])
    ).any():
        raise ValueError(f"{league} market eligibility has invalid avg_pick")
    market["_normalized_label"] = market["player"].map(
        normalize_player_name
    )
    market = market.sort_values(
        ["avg_pick", "_normalized_label", "player_key"],
        kind="mergesort",
    )
    # Canonicalize before applying the market limit so duplicate provider
    # aliases cannot consume two eligibility slots.
    market = market.drop_duplicates("player_key", keep="first")
    if market_limit is not None:
        if int(market_limit) <= 0:
            raise ValueError("market_limit must be positive or None")
        market = market.head(int(market_limit)).copy()
    market["eligibility_source"] = market_source_name
    market["source_rank"] = np.arange(1, len(market) + 1)
    market["source_value"] = market["avg_pick"]
    market.drop(columns="_normalized_label", inplace=True)

    source_frames = [core, market]
    if league in {"beta", "nv"} and not keeper_rows.empty:
        keepers = resolve_source_player_keys(
            keeper_rows,
            aliases,
            identities,
            year=year,
            source_name=f"{league}_league_keepers",
        )
        keepers["eligibility_source"] = "league_keeper"
        keepers["source_rank"] = np.nan
        keepers["source_value"] = np.nan
        source_frames.append(keepers)

    members = pd.concat(source_frames, ignore_index=True, sort=False)
    members["source_priority"] = members["eligibility_source"].map(
        ELIGIBILITY_SOURCE_PRIORITY
    )
    if members["source_priority"].isna().any():
        unknown = sorted(
            set(
                members.loc[
                    members["source_priority"].isna(),
                    "eligibility_source",
                ].astype(str)
            )
        )
        raise ValueError(f"Unknown eligibility sources: {unknown}")
    members["_normalized_label"] = members["player"].map(
        normalize_player_name
    )
    members = members.sort_values(
        [
            "source_priority",
            "source_rank",
            "_normalized_label",
            "player_key",
        ],
        na_position="last",
        kind="mergesort",
    )
    preferred = members.drop_duplicates("player_key", keep="first")[
        ["player_key", "player", "eligibility_key_match_method"]
    ].rename(
        columns={
            "player": "production_player_label",
            "eligibility_key_match_method": (
                "production_label_match_method"
            ),
        }
    )

    source_flags = (
        members.assign(_member=1)
        .pivot_table(
            index="player_key",
            columns="eligibility_source",
            values="_member",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    source_flag_names = {
        "core_projonly": "eligible_core_projonly",
        "dk_adp": "eligible_dk_adp",
        "nffc_adp": "eligible_nffc_adp",
        "etr_adp": "eligible_etr_adp",
        "league_keeper": "eligible_league_keeper",
    }
    source_flags.rename(columns=source_flag_names, inplace=True)
    for column in source_flag_names.values():
        if column not in source_flags:
            source_flags[column] = 0

    source_summary = (
        members.groupby("player_key", as_index=False)
        .agg(
            eligibility_sources=(
                "eligibility_source",
                lambda values: ",".join(
                    sorted(
                        set(values),
                        key=lambda value: (
                            ELIGIBILITY_SOURCE_PRIORITY[value],
                            value,
                        ),
                    )
                ),
            ),
            market_eligibility_rank=("source_rank", "min"),
            market_eligibility_pick=("source_value", "min"),
        )
    )
    membership = (
        preferred.merge(source_flags, on="player_key", validate="one_to_one")
        .merge(source_summary, on="player_key", validate="one_to_one")
    )
    membership["production_eligibility_version"] = (
        PRODUCTION_ELIGIBILITY_VERSION
    )
    return membership.sort_values("player_key").reset_index(drop=True)


def load_eligibility_membership(
    simulation_db: Path,
    model_inputs_db: Path,
    v2_database: Path,
    *,
    league: str,
    year: int,
    avg_adps: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, resolve, and combine one league's governed preseason sources."""

    aliases, identities = load_identity_frames(v2_database)
    core = _load_projonly_core(model_inputs_db, year=year)
    market_league, market_limit, market_source = (
        MARKET_ELIGIBILITY_RULES[league]
    )
    market = _load_market_source(
        simulation_db,
        year=year,
        market_league=market_league,
        avg_adps=avg_adps,
    )
    keepers = (
        _load_keeper_source(
            simulation_db,
            year=year,
            league=league,
        )
        if league in {"beta", "nv"}
        else pd.DataFrame(columns=["player"])
    )
    membership = build_eligibility_membership(
        core,
        market,
        keepers,
        aliases,
        identities,
        league=league,
        year=year,
        market_limit=market_limit,
        market_source_name=market_source,
    )
    return membership, aliases, identities


def _canonicalize_audit_slice(
    frame: pd.DataFrame,
    aliases: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    year: int,
    source_name: str,
) -> pd.DataFrame:
    if frame.empty:
        output = frame.copy()
        if "player_key" not in output:
            output["player_key"] = pd.Series(dtype="string")
        return output
    output = frame.copy()
    existing_keys = (
        output["player_key"].astype("string")
        if "player_key" in output
        else pd.Series(pd.NA, index=output.index, dtype="string")
    )
    missing_key = existing_keys.isna() | existing_keys.str.strip().eq("")
    if missing_key.any():
        resolved = resolve_source_player_keys(
            output.loc[missing_key].copy(),
            aliases,
            identities,
            year=year,
            source_name=source_name,
            require_complete=False,
        )
        existing_keys.loc[missing_key] = resolved["player_key"].to_numpy()
    output["player_key"] = existing_keys
    return output


def _nonempty_string(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    values = frame[column].astype("string").str.strip()
    return values.notna() & values.ne("")


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.notna() & np.isfinite(values)


def _validate_exclusions(
    exclusions: Mapping[str, str],
) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for player_key, reason in exclusions.items():
        key = str(player_key).strip()
        clean_reason = str(reason).strip()
        if not key or not clean_reason:
            raise ValueError(
                "Governed production exclusions require a canonical key "
                "and non-empty reason"
            )
        cleaned[key] = clean_reason
    return cleaned


def _prior_timestamps(
    prior_slice: pd.DataFrame,
    output: pd.DataFrame,
) -> pd.Series:
    created_at = utc_now()
    timestamps = pd.Series(created_at, index=output.index, dtype="object")
    required = {
        "player_key",
        "production_handoff_version",
        "production_handoff_created_at_utc",
    }
    if prior_slice.empty or not required.issubset(prior_slice.columns):
        return timestamps
    prior = prior_slice[
        prior_slice["production_handoff_version"].eq(
            PRODUCTION_HANDOFF_VERSION
        )
    ].copy()
    prior = prior.dropna(
        subset=["player_key", "production_handoff_created_at_utc"]
    ).drop_duplicates("player_key", keep="last")
    comparison_columns = [
        column
        for column in (
            "player",
            "pos",
            "pred_fp_per_game",
            "pred_fp_per_game_ny",
            "pred_appear_current",
            "pred_appear_ny",
            *CURRENT_RESIDUAL_COLUMNS,
            *NEXT_RESIDUAL_SOURCE_COLUMNS.values(),
            "current_projection_model_version",
            "next_projection_model_version",
            "v2_scoring_hash",
            "production_eligibility_version",
        )
        if column in output and column in prior
    ]
    prior_compare = prior.set_index("player_key")
    output_compare = output.set_index("player_key")
    shared_keys = output_compare.index.intersection(prior_compare.index)
    unchanged = pd.Series(False, index=output_compare.index)
    if len(shared_keys):
        same_values = np.ones(len(shared_keys), dtype=bool)
        for column in comparison_columns:
            left = output_compare.loc[shared_keys, column]
            right = prior_compare.loc[shared_keys, column]
            if pd.api.types.is_numeric_dtype(left):
                column_same = np.isclose(
                    pd.to_numeric(left, errors="coerce"),
                    pd.to_numeric(right, errors="coerce"),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                same_values &= np.asarray(column_same, dtype=bool)
            else:
                column_same = left.astype("string").fillna("").eq(
                    right.astype("string").fillna("")
                )
                same_values &= column_same.fillna(False).to_numpy(dtype=bool)
        same = pd.Series(same_values, index=shared_keys)
        unchanged.loc[shared_keys] = same
    prior_map = prior_compare[
        "production_handoff_created_at_utc"
    ]
    preserved = output["player_key"].map(prior_map)
    preserved = preserved.where(
        output["player_key"].map(unchanged).fillna(False)
    )
    return preserved.combine_first(timestamps)


def build_production_projection_slice(
    legacy_slice: pd.DataFrame,
    current_shadow: pd.DataFrame,
    next_shadow: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    league: str,
    year: int = PRODUCTION_YEAR,
    dataset: str = PRODUCTION_DATASET,
    governed_exclusions: Mapping[str, str] | None = None,
    prior_slice: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build production rows plus selected and master population audits."""

    _require_columns(
        current_shadow,
        (
            "player_key",
            "display_name",
            "season",
            "position",
            "conditional_ppg_shadow",
            "participation_probability",
            "lock_version",
            "publication_status",
        ),
        "current_v2_shadow",
    )
    _require_columns(
        next_shadow,
        (
            "player_key",
            "predicted_next_year_conditional_ppg",
            "predicted_next_year_appearance_probability",
            "origin_season",
            "target_season",
            "position",
            "publication_status",
            "target_version",
            "scoring_hash",
            *NEXT_RESIDUAL_SOURCE_COLUMNS,
        ),
        "next_v2_shadow",
    )
    _require_columns(
        eligibility,
        (
            "player_key",
            "production_player_label",
            "eligibility_sources",
            "production_eligibility_version",
        ),
        "production_eligibility",
    )
    for name, frame in (
        ("current", current_shadow),
        ("next", next_shadow),
        ("eligibility", eligibility),
    ):
        if frame["player_key"].isna().any():
            raise ValueError(f"{league} {name} handoff contains null keys")
        if frame["player_key"].duplicated().any():
            raise ValueError(
                f"{league} {name} handoff contains duplicate keys"
            )

    current = current_shadow.copy()
    current["player_key"] = current["player_key"].astype(str)
    current.rename(
        columns={
            "display_name": "v2_display_name",
            "season": "current_shadow_season",
            "position": "current_shadow_position",
        },
        inplace=True,
    )
    next_values = next_shadow.copy()
    next_values["player_key"] = next_values["player_key"].astype(str)
    next_rename = {
        column: f"next_shadow_{column}"
        for column in (
            "display_name",
            "origin_season",
            "target_season",
            "position",
            "team",
            "publication_status",
        )
        if column in next_values
    }
    next_values.rename(columns=next_rename, inplace=True)
    eligibility = eligibility.copy()
    eligibility["player_key"] = eligibility["player_key"].astype(str)
    for source_flag in (
        "eligible_core_projonly",
        "eligible_dk_adp",
        "eligible_nffc_adp",
        "eligible_etr_adp",
        "eligible_league_keeper",
    ):
        if source_flag not in eligibility:
            eligibility[source_flag] = 0

    master = current.merge(
        next_values,
        on="player_key",
        how="outer",
        validate="one_to_one",
        indicator="_current_next_join",
    ).merge(
        eligibility,
        on="player_key",
        how="outer",
        validate="one_to_one",
        indicator="_eligibility_join",
    )
    master["league"] = league
    master["year"] = int(year)
    master["dataset"] = dataset
    master["current_shadow_present"] = master[
        "_current_next_join"
    ].isin(["left_only", "both"]).astype(int)
    master["next_shadow_present"] = master[
        "_current_next_join"
    ].isin(["right_only", "both"]).astype(int)
    master["eligibility_required"] = master[
        "_eligibility_join"
    ].isin(["right_only", "both"]).astype(int)

    current_complete = (
        master["current_shadow_present"].eq(1)
        & _nonempty_string(master, "v2_display_name")
        & _nonempty_string(master, "current_shadow_position")
        & _finite_numeric(master, "current_shadow_season")
        & _finite_numeric(master, "conditional_ppg_shadow")
        & _finite_numeric(master, "participation_probability")
        & _nonempty_string(master, "lock_version")
        & master["publication_status"].astype("string").eq("shadow")
    )
    next_complete = (
        master["next_shadow_present"].eq(1)
        & _finite_numeric(master, "next_shadow_origin_season")
        & _finite_numeric(master, "next_shadow_target_season")
        & _nonempty_string(master, "next_shadow_position")
        & master["next_shadow_publication_status"]
        .astype("string")
        .eq("shadow")
        & _finite_numeric(
            master, "predicted_next_year_conditional_ppg"
        )
        & _finite_numeric(
            master, "predicted_next_year_appearance_probability"
        )
        & _nonempty_string(master, "target_version")
        & _nonempty_string(master, "scoring_hash")
    )
    for column in NEXT_RESIDUAL_SOURCE_COLUMNS:
        next_complete &= _finite_numeric(master, column)
    master["current_handoff_complete"] = current_complete.astype(int)
    master["next_handoff_complete"] = next_complete.astype(int)

    exclusions = _validate_exclusions(
        governed_exclusions
        if governed_exclusions is not None
        else governed_production_exclusions(year).get(league, {})
    )
    unknown_exclusions = set(exclusions).difference(
        set(master["player_key"].astype(str))
    )
    if unknown_exclusions:
        raise ValueError(
            f"{league} governed exclusions reference unknown keys: "
            f"{sorted(unknown_exclusions)}"
        )
    stale_exclusions = set(exclusions).difference(
        set(
            master.loc[
                master["eligibility_required"].eq(1), "player_key"
            ].astype(str)
        )
    )
    if stale_exclusions:
        raise ValueError(
            f"{league} governed exclusions are not eligibility-required: "
            f"{sorted(stale_exclusions)}"
        )
    exclusion_rows = master[
        master["player_key"].astype(str).isin(exclusions)
    ]
    market_only_exclusion = (
        exclusion_rows["eligible_core_projonly"].eq(0)
        & exclusion_rows["eligible_league_keeper"].eq(0)
        & (
            exclusion_rows["eligible_dk_adp"].eq(1)
            | exclusion_rows["eligible_nffc_adp"].eq(1)
            | exclusion_rows["eligible_etr_adp"].eq(1)
        )
    )
    incomplete_exclusion = (
        exclusion_rows["current_handoff_complete"].eq(0)
        | exclusion_rows["next_handoff_complete"].eq(0)
    )
    if not (market_only_exclusion & incomplete_exclusion).all():
        invalid = exclusion_rows.loc[
            ~(market_only_exclusion & incomplete_exclusion),
            [
                "player_key",
                "production_player_label",
                "eligible_core_projonly",
                "eligible_dk_adp",
                "eligible_nffc_adp",
                "eligible_etr_adp",
                "eligible_league_keeper",
                "current_handoff_complete",
                "next_handoff_complete",
            ],
        ].to_dict("records")
        raise ValueError(
            f"{league} governed exclusions must be incomplete market-only "
            f"rows, never core or keeper rows: {invalid}"
        )
    master["governed_exclusion_reason"] = master["player_key"].map(
        exclusions
    ).astype("string")
    required_market_depth = MARKET_HANDOFF_REQUIRED_DEPTH[league]
    protected_pick_depth = MARKET_HANDOFF_PROTECTED_PICK_DEPTH[league]
    market_rank = pd.to_numeric(
        master["market_eligibility_rank"], errors="coerce"
    )
    market_pick = pd.to_numeric(
        master["market_eligibility_pick"], errors="coerce"
    )
    market_draft_position = market_pick.where(
        market_pick.notna(), market_rank
    )
    master["market_handoff_required_depth"] = required_market_depth
    master["market_handoff_protected_pick_depth"] = protected_pick_depth
    master["market_handoff_draft_position"] = market_draft_position
    market_only = (
        master["eligible_core_projonly"].eq(0)
        & master["eligible_league_keeper"].eq(0)
        & (
            master["eligible_dk_adp"].eq(1)
            | master["eligible_nffc_adp"].eq(1)
            | master["eligible_etr_adp"].eq(1)
        )
    )
    incomplete_handoff = (
        master["current_handoff_complete"].eq(0)
        | master["next_handoff_complete"].eq(0)
    )
    automatic_market_buffer_exclusion = (
        master["eligibility_required"].eq(1)
        & market_only
        & incomplete_handoff
        & market_draft_position.gt(protected_pick_depth)
        & master["governed_exclusion_reason"].isna()
    )
    master["automatic_market_buffer_exclusion"] = (
        automatic_market_buffer_exclusion.astype(int)
    )
    master.loc[
        automatic_market_buffer_exclusion,
        "governed_exclusion_reason",
    ] = AUTOMATIC_MARKET_BUFFER_EXCLUSION_REASON
    master["governed_exclusion_policy_version"] = np.where(
        master["governed_exclusion_reason"].notna(),
        PRODUCTION_EXCLUSION_POLICY_VERSION,
        pd.NA,
    )
    master["governed_exclusion_reference"] = np.where(
        master["governed_exclusion_reason"].notna(),
        PRODUCTION_EXCLUSION_REFERENCE_BY_YEAR.get(int(year)),
        pd.NA,
    )
    if (
        master["governed_exclusion_reason"].notna().any()
        and int(year) not in PRODUCTION_EXCLUSION_REFERENCE_BY_YEAR
    ):
        raise ValueError(
            f"No governed production-exclusion reference exists for {year}"
        )
    master["governed_excluded"] = master[
        "governed_exclusion_reason"
    ].notna().astype(int)
    master["production_selected"] = (
        master["eligibility_required"].eq(1)
        & master["governed_excluded"].eq(0)
    ).astype(int)

    eligible_population = int(master["eligibility_required"].sum())
    selected_population = int(master["production_selected"].sum())
    if (
        eligible_population >= required_market_depth
        and selected_population < required_market_depth
    ):
        raise ValueError(
            f"{league} governed exclusions leave only "
            f"{selected_population} production players; "
            f"{required_market_depth} are required to cover the draft"
        )

    incomplete = (
        master["production_selected"].eq(1)
        & (
            master["current_handoff_complete"].eq(0)
            | master["next_handoff_complete"].eq(0)
        )
    )
    if incomplete.any():
        preview = master.loc[
            incomplete,
            [
                "player_key",
                "production_player_label",
                "eligibility_sources",
                "current_shadow_present",
                "current_handoff_complete",
                "next_shadow_present",
                "next_handoff_complete",
            ],
        ].head(20)
        raise ValueError(
            f"{league} eligibility-required V2 handoff is incomplete: "
            f"{preview.to_dict('records')}"
        )

    selected = master["production_selected"].eq(1)
    expected_hash = scoring_hash(league)
    observed_hashes = set(
        master.loc[selected, "scoring_hash"].dropna().astype(str)
    )
    if observed_hashes != {expected_hash}:
        raise ValueError(
            f"{league} scoring hash mismatch: {sorted(observed_hashes)}"
        )
    positions = master.loc[
        selected, "current_shadow_position"
    ].astype(str)
    if not positions.isin(POSITIONS).all():
        raise ValueError(
            f"{league} selected V2 rows contain invalid positions"
        )
    seasons = pd.to_numeric(
        master.loc[selected, "current_shadow_season"],
        errors="coerce",
    )
    if not seasons.eq(int(year)).all():
        raise ValueError(
            f"{league} selected V2 rows do not match production year {year}"
        )
    next_origin_seasons = pd.to_numeric(
        master.loc[selected, "next_shadow_origin_season"],
        errors="coerce",
    )
    next_target_seasons = pd.to_numeric(
        master.loc[selected, "next_shadow_target_season"],
        errors="coerce",
    )
    if not (
        next_origin_seasons.eq(int(year)).all()
        and next_target_seasons.eq(int(year) + 1).all()
    ):
        raise ValueError(
            f"{league} next shadow origin/target seasons do not match "
            f"{year}->{year + 1}"
        )
    next_positions = master.loc[
        selected, "next_shadow_position"
    ].astype(str)
    if not next_positions.eq(positions.to_numpy()).all():
        raise ValueError(
            f"{league} next shadow positions disagree with current positions"
        )
    for probability_column in (
        "participation_probability",
        "predicted_next_year_appearance_probability",
    ):
        probabilities = pd.to_numeric(
            master.loc[selected, probability_column],
            errors="coerce",
        )
        if not probabilities.between(0, 1).all():
            raise ValueError(
                f"{league} {probability_column} is outside [0, 1]"
            )

    selected_next_quantiles = master.loc[
        selected, list(NEXT_RESIDUAL_SOURCE_COLUMNS)
    ].apply(pd.to_numeric, errors="coerce")
    if (
        np.diff(selected_next_quantiles.to_numpy(dtype=float), axis=1)
        < -1e-10
    ).any():
        raise ValueError(f"{league} next residual quantiles are not monotone")

    current_centers = pd.to_numeric(
        master.loc[selected, "conditional_ppg_shadow"],
        errors="raise",
    )
    next_centers = pd.to_numeric(
        master.loc[
            selected, "predicted_next_year_conditional_ppg"
        ],
        errors="raise",
    )
    if current_centers.le(0).any() or next_centers.le(0).any():
        raise ValueError(
            f"{league} selected projection centers must be strictly positive"
        )

    output = pd.DataFrame(
        {
            "player": master.loc[selected, "v2_display_name"],
            "pos": master.loc[selected, "current_shadow_position"],
            "pred_fp_per_game": current_centers,
            "pred_fp_per_game_ny": next_centers,
            "dataset": dataset,
            "version": league,
            "year": int(year),
            "player_key": master.loc[selected, "player_key"],
            "current_projection_model_version": master.loc[
                selected, "lock_version"
            ],
            "next_projection_model_version": master.loc[
                selected, "target_version"
            ],
            "v2_scoring_hash": master.loc[selected, "scoring_hash"],
            "pred_appear_current": pd.to_numeric(
                master.loc[selected, "participation_probability"],
                errors="raise",
            ),
            "pred_appear_ny": pd.to_numeric(
                master.loc[
                    selected,
                    "predicted_next_year_appearance_probability",
                ],
                errors="raise",
            ),
        }
    ).reset_index(drop=True)
    for column in CURRENT_RESIDUAL_COLUMNS:
        output[column] = 0.0
    selected_master = master.loc[selected].reset_index(drop=True)
    for source, destination in NEXT_RESIDUAL_SOURCE_COLUMNS.items():
        output[destination] = pd.to_numeric(
            selected_master[source],
            errors="raise",
        )
    output["production_handoff_version"] = PRODUCTION_HANDOFF_VERSION
    output["production_eligibility_version"] = (
        PRODUCTION_ELIGIBILITY_VERSION
    )
    output["current_projection_source"] = "v2_locked_conditional_ppg"
    output["current_uncertainty_source"] = "joint_weekly_template_only"
    output["independent_current_residual_draw_allowed"] = 0
    output["next_projection_source"] = "v2_next_year_conditional_ppg"
    output["next_uncertainty_source"] = (
        "conditional_residual_plus_appearance"
    )
    output["_position_order"] = output["pos"].map(POSITION_ORDER)
    output = output.sort_values(
        ["_position_order", "player_key"],
        kind="mergesort",
    ).drop(columns="_position_order").reset_index(drop=True)
    prior_slice = (
        prior_slice.copy()
        if prior_slice is not None
        else pd.DataFrame()
    )
    output["production_handoff_created_at_utc"] = _prior_timestamps(
        prior_slice,
        output,
    ).to_numpy()

    legacy = legacy_slice.copy()
    if "player_key" not in legacy:
        legacy["player_key"] = pd.Series(pd.NA, index=legacy.index)
    legacy = legacy.dropna(subset=["player_key"]).copy()
    if not legacy.empty:
        legacy["player_key"] = legacy["player_key"].astype(str)
        legacy = legacy.drop_duplicates("player_key", keep="last")
    legacy_columns = ["player_key"]
    legacy_rename: dict[str, str] = {}
    for source, destination in (
        ("player", "legacy_player"),
        ("pred_fp_per_game", "legacy_pred_fp_per_game"),
        ("pred_fp_per_game_ny", "legacy_pred_fp_per_game_ny"),
    ):
        if source in legacy:
            legacy_columns.append(source)
            legacy_rename[source] = destination
    legacy_compare = legacy[legacy_columns].rename(
        columns=legacy_rename
    )
    master = master.merge(
        legacy_compare,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    legacy_keys = set(legacy["player_key"].astype(str))
    master["legacy_population_member"] = master["player_key"].astype(
        str
    ).isin(legacy_keys).astype(int)
    master["population_action"] = np.select(
        [
            master["governed_excluded"].eq(1),
            master["production_selected"].eq(1)
            & master["legacy_population_member"].eq(1),
            master["production_selected"].eq(1)
            & master["legacy_population_member"].eq(0),
            master["production_selected"].eq(0)
            & master["legacy_population_member"].eq(1),
        ],
        ["governed_excluded", "retained", "added", "dropped"],
        default="not_selected",
    )
    master["production_eligibility_version"] = (
        PRODUCTION_ELIGIBILITY_VERSION
    )
    master.drop(
        columns=["_current_next_join", "_eligibility_join"],
        inplace=True,
    )
    master = master.sort_values("player_key", kind="mergesort").reset_index(
        drop=True
    )

    audit = output[
        [
            "player_key",
            "player",
            "pos",
            "year",
            "version",
            "dataset",
            "pred_fp_per_game",
            "pred_fp_per_game_ny",
            "pred_appear_current",
            "pred_appear_ny",
            "current_projection_model_version",
            "next_projection_model_version",
            "v2_scoring_hash",
            "production_handoff_version",
            "production_eligibility_version",
        ]
    ].merge(
        master[
            [
                "player_key",
                "eligibility_sources",
                "legacy_population_member",
                "population_action",
                *[
                    column
                    for column in (
                        "legacy_pred_fp_per_game",
                        "legacy_pred_fp_per_game_ny",
                    )
                    if column in master
                ],
            ]
        ],
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    if "legacy_pred_fp_per_game" not in audit:
        audit["legacy_pred_fp_per_game"] = np.nan
    if "legacy_pred_fp_per_game_ny" not in audit:
        audit["legacy_pred_fp_per_game_ny"] = np.nan
    audit["current_ppg_delta"] = (
        audit["pred_fp_per_game"]
        - pd.to_numeric(audit["legacy_pred_fp_per_game"], errors="coerce")
    )
    audit["next_ppg_delta"] = (
        audit["pred_fp_per_game_ny"]
        - pd.to_numeric(
            audit["legacy_pred_fp_per_game_ny"], errors="coerce"
        )
    )
    return output, audit, master


def publish_production_handoff(
    simulation_db: Path = SIMULATION_DB_PATH,
    v2_databases: Mapping[str, Path] = V2_DATABASES,
    *,
    model_inputs_db: Path = MODEL_INPUTS_DB_PATH,
    market_source_db: Path = SOURCE_DB_PATH,
    year: int = PRODUCTION_YEAR,
    dataset: str = PRODUCTION_DATASET,
    governed_exclusions: (
        Mapping[str, Mapping[str, str]] | None
    ) = None,
) -> dict[str, pd.DataFrame]:
    """Atomically publish authoritative V2 league slices and durable audits."""

    identity_database = (
        v2_databases["dk"]
        if "dk" in v2_databases
        else next(iter(v2_databases.values()))
    )
    avg_adp_tables = load_current_avg_adp_publication(
        source_db=market_source_db,
        simulation_db=simulation_db,
        v2_database=identity_database,
        year=year,
    )
    current_avg_adps = avg_adp_tables[AVG_ADP_TABLE]

    final_predictions = _read_table(
        simulation_db, "Final_Predictions_Resid"
    )
    target_mask = (
        final_predictions["year"].eq(year)
        & final_predictions["dataset"].eq(dataset)
        & final_predictions["version"].isin(v2_databases)
    )
    prior_target = final_predictions[target_mask].copy()

    existing_legacy_backup = (
        _read_table(simulation_db, LEGACY_BACKUP_TABLE)
        if _table_exists(simulation_db, LEGACY_BACKUP_TABLE)
        else None
    )
    legacy_backup = build_legacy_projection_backup(
        existing_legacy_backup,
        prior_target,
        year=year,
        dataset=dataset,
    )

    slices = []
    audits = []
    eligibility_audits = []
    for league, database in v2_databases.items():
        current_shadow, next_shadow = load_validated_shadow_predictions(
            database,
            league=league,
            year=year,
        )
        membership, aliases, identities = load_eligibility_membership(
            simulation_db,
            model_inputs_db,
            database,
            league=league,
            year=year,
            avg_adps=current_avg_adps,
        )
        baseline = legacy_backup[
            legacy_backup["year"].eq(year)
            & legacy_backup["dataset"].eq(dataset)
            & legacy_backup["version"].eq(league)
        ].copy()
        baseline = _canonicalize_audit_slice(
            baseline,
            aliases,
            identities,
            year=year,
            source_name=f"{league}_legacy_backup",
        )
        prior = prior_target[
            prior_target["year"].eq(year)
            & prior_target["dataset"].eq(dataset)
            & prior_target["version"].eq(league)
        ].copy()
        prior = _canonicalize_audit_slice(
            prior,
            aliases,
            identities,
            year=year,
            source_name=f"{league}_prior_production",
        )
        league_exclusions = (
            governed_exclusions.get(league, {})
            if governed_exclusions is not None
            else governed_production_exclusions(year).get(league, {})
        )
        promoted, audit, eligibility_audit = (
            build_production_projection_slice(
                baseline,
                current_shadow,
                next_shadow,
                membership,
                league=league,
                year=year,
                dataset=dataset,
                governed_exclusions=league_exclusions,
                prior_slice=prior,
            )
        )
        slices.append(promoted)
        audits.append(audit)
        eligibility_audits.append(eligibility_audit)

    promoted_target = pd.concat(slices, ignore_index=True, sort=False)
    untouched = final_predictions[~target_mask].copy()
    combined = pd.concat(
        [untouched, promoted_target],
        ignore_index=True,
        sort=False,
    )
    ordered = list(promoted_target.columns) + [
        column
        for column in combined.columns
        if column not in promoted_target
    ]
    combined = combined.loc[:, ordered]
    audit = pd.concat(audits, ignore_index=True, sort=False)
    eligibility_audit = pd.concat(
        eligibility_audits,
        ignore_index=True,
        sort=False,
    )
    keyed_combined = combined[combined["player_key"].notna()]
    if keyed_combined.duplicated(
        ["player_key", "year", "version", "dataset"]
    ).any():
        raise ValueError(
            "Promoted production projections contain duplicate canonical keys"
        )

    tables = {
        **avg_adp_tables,
        "Final_Predictions_Resid": combined,
        "V2_Production_Projection_Handoff": promoted_target,
        "V2_Production_Projection_Audit": audit,
        ELIGIBILITY_AUDIT_TABLE: eligibility_audit,
        LEGACY_BACKUP_TABLE: legacy_backup,
    }
    publish_tables_atomic(simulation_db, tables)
    return tables


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation-db",
        type=Path,
        default=SIMULATION_DB_PATH,
    )
    parser.add_argument(
        "--model-inputs-db",
        type=Path,
        default=MODEL_INPUTS_DB_PATH,
    )
    parser.add_argument(
        "--market-source-db",
        type=Path,
        default=SOURCE_DB_PATH,
    )
    parser.add_argument(
        "--dk-v2-db",
        type=Path,
        default=V2_DATABASES["dk"],
    )
    parser.add_argument(
        "--nffc-v2-db",
        type=Path,
        default=V2_DATABASES["nffc"],
    )
    parser.add_argument(
        "--beta-v2-db",
        type=Path,
        default=V2_DATABASES["beta"],
    )
    parser.add_argument(
        "--nv-v2-db",
        type=Path,
        default=V2_DATABASES["nv"],
    )
    parser.add_argument("--year", type=int, default=PRODUCTION_YEAR)
    parser.add_argument("--dataset", default=PRODUCTION_DATASET)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tables = publish_production_handoff(
        simulation_db=args.simulation_db.resolve(),
        v2_databases={
            "dk": args.dk_v2_db.resolve(),
            "nffc": args.nffc_v2_db.resolve(),
            "beta": args.beta_v2_db.resolve(),
            "nv": args.nv_v2_db.resolve(),
        },
        model_inputs_db=args.model_inputs_db.resolve(),
        market_source_db=args.market_source_db.resolve(),
        year=args.year,
        dataset=args.dataset,
    )
    audit = tables["V2_Production_Projection_Audit"]
    summary = (
        audit.groupby("version", as_index=False)
        .agg(
            players=("player_key", "nunique"),
            added=("population_action", lambda x: int(x.eq("added").sum())),
            retained=(
                "population_action",
                lambda x: int(x.eq("retained").sum()),
            ),
            current_delta_mae=(
                "current_ppg_delta",
                lambda x: x.abs().mean(),
            ),
            next_delta_mae=("next_ppg_delta", lambda x: x.abs().mean()),
            min_next_appearance=("pred_appear_ny", "min"),
            max_next_appearance=("pred_appear_ny", "max"),
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
