"""Publish the locked DK/beta V2 projections to the production handoff.

The current-season point forecast is deterministic inside the app. Its
uncertainty is supplied by one matched weekly donor residual/path. Following-
season keeper uncertainty remains a conditional residual distribution plus a
separate probability of any appearance.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from Scripts.V2.config import REPO_ROOT
from Scripts.V2.contracts import publish_tables_atomic, scoring_hash, utc_now


SIMULATION_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
V2_DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT
    / "Data"
    / "Databases"
    / "Projection_V2_beta.sqlite3",
}
PRODUCTION_YEAR = 2026
PRODUCTION_DATASET = "final_ensemble"
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
PRODUCTION_HANDOFF_VERSION = "v2_current_next_production_handoff_v1"
REFRESHED_HANDOFF_COLUMNS = (
    "player_key",
    "current_projection_model_version",
    "next_projection_model_version",
    "v2_scoring_hash",
    "pred_appear_current",
    "pred_appear_ny",
    "production_handoff_version",
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


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_production_projection_slice(
    legacy_slice: pd.DataFrame,
    player_map_slice: pd.DataFrame,
    current_shadow: pd.DataFrame,
    next_shadow: pd.DataFrame,
    *,
    league: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one league's production rows and an auditable comparison."""

    # A production slice may already have been published by an earlier V2
    # run. Refresh canonical keys from the rebuilt weekly player map and
    # replace prior handoff metadata instead of creating merge suffixes or
    # duplicate columns.
    legacy_slice = legacy_slice.drop(
        columns=[
            column
            for column in REFRESHED_HANDOFF_COLUMNS
            if column in legacy_slice
        ]
    ).copy()
    join_columns = (
        "player",
        "pos",
        "year",
        "version",
        "dataset",
    )
    _require_columns(
        legacy_slice,
        (*join_columns, "pred_fp_per_game", "pred_fp_per_game_ny"),
        "legacy_prediction_slice",
    )
    _require_columns(
        player_map_slice,
        (*join_columns, "player_key"),
        "weekly_player_map_slice",
    )
    _require_columns(
        current_shadow,
        (
            "player_key",
            "conditional_ppg_shadow",
            "participation_probability",
            "lock_version",
        ),
        "current_v2_shadow",
    )
    _require_columns(
        next_shadow,
        (
            "player_key",
            "predicted_next_year_conditional_ppg",
            "predicted_next_year_appearance_probability",
            "target_version",
            "scoring_hash",
            *NEXT_RESIDUAL_SOURCE_COLUMNS,
        ),
        "next_v2_shadow",
    )
    if legacy_slice.empty:
        raise ValueError(f"No production prediction rows for {league}")
    if legacy_slice.duplicated(list(join_columns)).any():
        raise ValueError(f"Legacy {league} production rows are not unique")
    if player_map_slice.duplicated(list(join_columns)).any():
        raise ValueError(f"Weekly {league} player-map rows are not unique")
    if current_shadow["player_key"].duplicated().any():
        raise ValueError(f"Current {league} V2 shadow contains duplicate keys")
    if next_shadow["player_key"].duplicated().any():
        raise ValueError(f"Next {league} V2 shadow contains duplicate keys")

    keys = player_map_slice.loc[:, [*join_columns, "player_key"]]
    output = legacy_slice.merge(
        keys,
        on=list(join_columns),
        how="left",
        validate="one_to_one",
        indicator="_player_key_join",
    )
    if output["player_key"].isna().any():
        missing = output.loc[
            output["player_key"].isna(), ["player", "pos"]
        ].head(10)
        raise ValueError(
            f"{league} production rows lack canonical keys: "
            f"{missing.to_dict('records')}"
        )
    output.drop(columns="_player_key_join", inplace=True)

    current_columns = [
        "player_key",
        "conditional_ppg_shadow",
        "participation_probability",
        "lock_version",
    ]
    next_columns = [
        "player_key",
        "predicted_next_year_conditional_ppg",
        "predicted_next_year_appearance_probability",
        "target_version",
        "scoring_hash",
        *NEXT_RESIDUAL_SOURCE_COLUMNS,
    ]
    output = output.merge(
        current_shadow.loc[:, current_columns],
        on="player_key",
        how="left",
        validate="one_to_one",
    ).merge(
        next_shadow.loc[:, next_columns],
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    required_values = [
        "conditional_ppg_shadow",
        "participation_probability",
        "predicted_next_year_conditional_ppg",
        "predicted_next_year_appearance_probability",
        *NEXT_RESIDUAL_SOURCE_COLUMNS,
    ]
    if output[required_values].isna().any().any():
        missing = output.loc[
            output[required_values].isna().any(axis=1),
            ["player", "pos", "player_key"],
        ].head(10)
        raise ValueError(
            f"{league} V2 production handoff is incomplete: "
            f"{missing.to_dict('records')}"
        )
    expected_hash = scoring_hash(league)
    observed_hashes = set(output["scoring_hash"].astype(str))
    if observed_hashes != {expected_hash}:
        raise ValueError(
            f"{league} scoring hash mismatch: {sorted(observed_hashes)}"
        )
    for probability_column in (
        "participation_probability",
        "predicted_next_year_appearance_probability",
    ):
        if not output[probability_column].between(0, 1).all():
            raise ValueError(
                f"{league} {probability_column} is outside [0, 1]"
            )

    legacy_current = pd.to_numeric(
        output["pred_fp_per_game"], errors="raise"
    ).copy()
    legacy_next = pd.to_numeric(
        output["pred_fp_per_game_ny"], errors="raise"
    ).copy()
    output["pred_fp_per_game"] = pd.to_numeric(
        output.pop("conditional_ppg_shadow"), errors="raise"
    ).clip(lower=0)
    output["pred_fp_per_game_ny"] = pd.to_numeric(
        output.pop("predicted_next_year_conditional_ppg"),
        errors="raise",
    ).clip(lower=0)
    output["pred_appear_current"] = pd.to_numeric(
        output.pop("participation_probability"), errors="raise"
    )
    output["pred_appear_ny"] = pd.to_numeric(
        output.pop("predicted_next_year_appearance_probability"),
        errors="raise",
    )
    for column in CURRENT_RESIDUAL_COLUMNS:
        output[column] = 0.0
    for source, destination in NEXT_RESIDUAL_SOURCE_COLUMNS.items():
        output[destination] = pd.to_numeric(
            output.pop(source), errors="raise"
        )
    next_residual_columns = list(NEXT_RESIDUAL_SOURCE_COLUMNS.values())
    if (
        np.diff(
            output[next_residual_columns].to_numpy(dtype=float),
            axis=1,
        )
        < -1e-10
    ).any():
        raise ValueError(f"{league} next residual quantiles are not monotone")

    output.rename(
        columns={
            "lock_version": "current_projection_model_version",
            "target_version": "next_projection_model_version",
            "scoring_hash": "v2_scoring_hash",
        },
        inplace=True,
    )
    output["production_handoff_version"] = PRODUCTION_HANDOFF_VERSION
    output["current_projection_source"] = "v2_locked_conditional_ppg"
    output["current_uncertainty_source"] = "joint_weekly_template_only"
    output["independent_current_residual_draw_allowed"] = 0
    output["next_projection_source"] = "v2_next_year_conditional_ppg"
    output["next_uncertainty_source"] = (
        "conditional_residual_plus_appearance"
    )
    output["production_handoff_created_at_utc"] = utc_now()

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
        ]
    ].copy()
    audit["legacy_pred_fp_per_game"] = legacy_current.to_numpy()
    audit["legacy_pred_fp_per_game_ny"] = legacy_next.to_numpy()
    audit["current_ppg_delta"] = (
        audit["pred_fp_per_game"] - audit["legacy_pred_fp_per_game"]
    )
    audit["next_ppg_delta"] = (
        audit["pred_fp_per_game_ny"]
        - audit["legacy_pred_fp_per_game_ny"]
    )
    return output, audit


def publish_production_handoff(
    simulation_db: Path = SIMULATION_DB_PATH,
    v2_databases: Mapping[str, Path] = V2_DATABASES,
    *,
    year: int = PRODUCTION_YEAR,
    dataset: str = PRODUCTION_DATASET,
) -> dict[str, pd.DataFrame]:
    """Atomically publish both league slices and retain the legacy backup."""

    final_predictions = _read_table(
        simulation_db, "Final_Predictions_Resid"
    )
    player_map = _read_table(
        simulation_db, "Best_Ball_Weekly_Player_Map"
    )
    target_mask = (
        final_predictions["year"].eq(year)
        & final_predictions["dataset"].eq(dataset)
        & final_predictions["version"].isin(v2_databases)
    )
    legacy_target = final_predictions[target_mask].copy()
    if legacy_target.empty:
        raise ValueError("No DK/beta production projection rows to promote")

    with sqlite3.connect(simulation_db) as connection:
        backup_exists = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='V2_Projection_Legacy_Backup'"
        ).fetchone()
        if backup_exists:
            legacy_backup = pd.read_sql_query(
                "SELECT * FROM V2_Projection_Legacy_Backup", connection
            )
        else:
            legacy_backup = legacy_target.copy()
            legacy_backup["backup_created_at_utc"] = utc_now()

    slices = []
    audits = []
    for league, database in v2_databases.items():
        league_mask = (
            legacy_target["year"].eq(year)
            & legacy_target["dataset"].eq(dataset)
            & legacy_target["version"].eq(league)
        )
        map_mask = (
            player_map["year"].eq(year)
            & player_map["dataset"].eq(dataset)
            & player_map["version"].eq(league)
        )
        current_shadow = _read_table(
            database, "locked_2026_shadow_predictions"
        )
        next_shadow = _read_table(
            database, "next_year_2027_shadow_predictions"
        )
        promoted, audit = build_production_projection_slice(
            legacy_target[league_mask].copy(),
            player_map[map_mask].copy(),
            current_shadow,
            next_shadow,
            league=league,
        )
        slices.append(promoted)
        audits.append(audit)

    promoted_target = pd.concat(slices, ignore_index=True, sort=False)
    untouched = final_predictions[~target_mask].copy()
    combined = pd.concat(
        [untouched, promoted_target], ignore_index=True, sort=False
    )
    ordered = list(promoted_target.columns) + [
        column
        for column in combined.columns
        if column not in promoted_target
    ]
    combined = combined.loc[:, ordered]
    audit = pd.concat(audits, ignore_index=True)
    if combined.duplicated(
        ["player", "pos", "year", "version", "dataset"]
    ).any():
        raise ValueError("Promoted production projections contain duplicates")
    if len(promoted_target) != len(legacy_target):
        raise ValueError("Production promotion changed the target row count")

    publish_tables_atomic(
        simulation_db,
        {
            "Final_Predictions_Resid": combined,
            "V2_Production_Projection_Handoff": promoted_target,
            "V2_Production_Projection_Audit": audit,
            "V2_Projection_Legacy_Backup": legacy_backup,
        },
    )
    return {
        "Final_Predictions_Resid": combined,
        "V2_Production_Projection_Handoff": promoted_target,
        "V2_Production_Projection_Audit": audit,
        "V2_Projection_Legacy_Backup": legacy_backup,
    }


def main() -> None:
    tables = publish_production_handoff()
    audit = tables["V2_Production_Projection_Audit"]
    summary = (
        audit.groupby("version", as_index=False)
        .agg(
            players=("player_key", "nunique"),
            current_delta_mae=("current_ppg_delta", lambda x: x.abs().mean()),
            next_delta_mae=("next_ppg_delta", lambda x: x.abs().mean()),
            min_next_appearance=("pred_appear_ny", "min"),
            max_next_appearance=("pred_appear_ny", "max"),
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
