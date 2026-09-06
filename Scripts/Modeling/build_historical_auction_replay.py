"""Build an isolated, causally bounded Auction app historical replay.

This entry point is intentionally narrower than the production refresh.  It
copies the reviewed source databases into a study staging directory, publishes
one historical projection/salary/keeper context, and then invokes the weekly
template matcher with donor outcomes capped at ``target_year - 1``.  It never
writes the live ``Data/Databases`` artifacts or synchronizes an app database.

The reviewed contracts cover 2022-2025 Beta.  Their projection and salary
methods were selected with the current (2026) research specification, but
every target row is rolling-origin and trained/calibrated only through the
prior season.  The resulting artifacts are current-method historical replays,
not claims that the exact method was chosen before each historical draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Scripts.Modeling.publish_actual_salaries import (
    build_actual_salary_slice,
    publish_actual_salary_slice,
)
from Scripts.V2.production_cycle import (
    get_historical_replay_template_contract,
)
from Scripts.V2.production_handoff import (
    load_identity_frames,
    resolve_source_player_keys,
)


PROJECTION_TABLE = "Final_Predictions_Resid"
SALARY_TABLE = "Salaries_Pred"
KEEPER_TABLE = "League_Keepers"
PREMIUM_TABLE = "Salary_Selection_Premium"
CONTEXT_TABLE = "Auction_Historical_Replay_Context"
ADP_TABLE = "Avg_ADPs"

PREDICTION_DATASET = "final_ensemble"
PROJECTION_METHOD = "rolling_prior_season_empirical_v1"
SALARY_METHOD = "current_locked_spec_v6_v2_population_11f"
REPLAY_CONTEXT_VERSION = "auction_historical_replay_v1"
PROJECTION_MODEL_VERSION = "historical_current_method_replay_v1"
PRODUCTION_HANDOFF_VERSION = "historical_replay_handoff_v1"
PRODUCTION_ELIGIBILITY_VERSION = "historical_replay_exact_key_parity_v1"

CURRENT_RESIDUAL_COLUMNS = tuple(
    f"pred_resid_{quantile}" for quantile in (5, 10, 25, 75, 90, 95)
)
NEXT_RESIDUAL_COLUMNS = tuple(f"{column}_ny" for column in CURRENT_RESIDUAL_COLUMNS)
SALARY_RESIDUAL_COLUMNS = tuple(
    f"salary_resid_{quantile}" for quantile in (5, 10, 25, 75, 90, 95)
)
POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
GOVERNED_CANONICAL_POSITION_MISMATCHES = {
    # The shared identity spine uses Hunter's defensive designation while the
    # 2025 fantasy projection/salary surfaces correctly roster him at WR.
    ("e36034a5-fc78-5b29-90e0-66619310bb0b", "WR", "DB"):
        "two_way_player_fantasy_role:travis_hunter",
    # Historical fantasy roles intentionally retain the position at which the
    # player was drafted/scored even when the current identity spine differs.
    ("2f3a5f36-ad51-527b-8fdc-ca0a5e431ad6", "RB", "WR"):
        "historical_fantasy_role:ty_montgomery",
    ("877cadec-3157-5007-9ed6-10243e581135", "TE", "RB"):
        "historical_fantasy_role:connor_heyward",
}


@dataclass(frozen=True)
class ReplayContract:
    year: int
    league: str
    projection_rows: int
    current_method_projection_rows: int
    projection_training_through_year: int
    projection_model_spec_asof_year: int
    salary_rows: int
    current_method_salary_rows: int
    legacy_fallback_rows: int
    salary_training_through_year: int
    salary_model_spec_asof_year: int
    keeper_count: int
    keeper_spend: float
    available_slots: int
    available_budget: float
    raw_actual_rows: int
    offensive_actual_rows: int
    historical_etr_rows: int


REPLAY_CONTRACTS = {
    (2022, "beta"): ReplayContract(
        year=2022,
        league="beta",
        projection_rows=308,
        current_method_projection_rows=307,
        projection_training_through_year=2021,
        projection_model_spec_asof_year=2026,
        salary_rows=299,
        current_method_salary_rows=299,
        legacy_fallback_rows=1,
        salary_training_through_year=2021,
        salary_model_spec_asof_year=2026,
        keeper_count=20,
        keeper_spend=690.0,
        available_slots=136,
        available_budget=2886.0,
        raw_actual_rows=169,
        offensive_actual_rows=149,
        historical_etr_rows=308,
    ),
    (2023, "beta"): ReplayContract(
        year=2023,
        league="beta",
        projection_rows=315,
        current_method_projection_rows=313,
        projection_training_through_year=2022,
        projection_model_spec_asof_year=2026,
        salary_rows=311,
        current_method_salary_rows=311,
        legacy_fallback_rows=2,
        salary_training_through_year=2022,
        salary_model_spec_asof_year=2026,
        keeper_count=21,
        keeper_spend=871.0,
        available_slots=135,
        available_budget=2705.0,
        raw_actual_rows=180,
        offensive_actual_rows=155,
        historical_etr_rows=315,
    ),
    (2024, "beta"): ReplayContract(
        year=2024,
        league="beta",
        projection_rows=316,
        current_method_projection_rows=314,
        projection_training_through_year=2023,
        projection_model_spec_asof_year=2026,
        salary_rows=304,
        current_method_salary_rows=304,
        legacy_fallback_rows=2,
        salary_training_through_year=2023,
        salary_model_spec_asof_year=2026,
        keeper_count=18,
        keeper_spend=563.0,
        available_slots=138,
        available_budget=3013.0,
        raw_actual_rows=179,
        offensive_actual_rows=155,
        historical_etr_rows=316,
    ),
    (2025, "beta"): ReplayContract(
        year=2025,
        league="beta",
        projection_rows=309,
        current_method_projection_rows=305,
        projection_training_through_year=2024,
        projection_model_spec_asof_year=2026,
        salary_rows=309,
        current_method_salary_rows=305,
        legacy_fallback_rows=4,
        salary_training_through_year=2024,
        salary_model_spec_asof_year=2026,
        keeper_count=15,
        keeper_spend=407.0,
        available_slots=141,
        available_budget=3169.0,
        raw_actual_rows=179,
        offensive_actual_rows=156,
        historical_etr_rows=238,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_replay_contract(year: int, league: str) -> ReplayContract:
    key = (int(year), str(league).strip().lower())
    try:
        contract = REPLAY_CONTRACTS[key]
    except KeyError as error:
        registered = ", ".join(
            f"{registered_year} {registered_league}"
            for registered_year, registered_league in sorted(REPLAY_CONTRACTS)
        )
        raise ValueError(
            f"No reviewed Auction historical replay contract exists for "
            f"{key[0]} {key[1]}. Registered contexts: {registered}."
        ) from error
    template_contract = get_historical_replay_template_contract(contract.year)
    if contract.league not in template_contract.leagues:
        raise ValueError(
            f"The {contract.year} template contract does not register "
            f"{contract.league}."
        )
    return contract


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    context: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def single_value(
    frame: pd.DataFrame,
    column: str,
    context: str,
    *,
    numeric: bool = False,
):
    values = (
        pd.to_numeric(frame[column], errors="coerce")
        if numeric
        else frame[column].astype("string").str.strip()
    )
    if values.isna().any() or (not numeric and values.eq("").any()):
        raise ValueError(f"{context} contains missing {column} values.")
    observed = list(pd.unique(values))
    if len(observed) != 1:
        raise ValueError(
            f"{context} contains multiple {column} values: {observed[:10]}"
        )
    return observed[0]


def canonicalize_rows(
    rows: pd.DataFrame,
    aliases: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    year: int,
    context: str,
) -> pd.DataFrame:
    resolved = resolve_source_player_keys(
        rows,
        aliases,
        identities,
        year=int(year),
        source_name=context,
        require_complete=True,
    )
    if resolved.player_key.duplicated().any():
        duplicates = resolved.loc[
            resolved.player_key.duplicated(keep=False),
            ["player", "pos", "player_key"],
        ].head(20)
        raise ValueError(
            f"{context} resolves duplicate player keys: "
            f"{duplicates.to_dict('records')}"
        )
    canonical = identities[["player_key", "display_name", "position"]].copy()
    canonical["player_key"] = canonical.player_key.astype("string")
    output = resolved.merge(
        canonical,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    if output[["display_name", "position"]].isna().any().any():
        raise ValueError(f"{context} lacks canonical display metadata.")
    if "pos" not in output:
        output["pos"] = output["position"]
    output["pos"] = output.pos.astype("string").str.upper()
    canonical_positions = output.position.astype("string").str.upper()
    position_mismatch = output.pos.ne(canonical_positions)
    if position_mismatch.any():
        mismatch_keys = {
            (str(player_key), str(source_pos), str(canonical_pos))
            for player_key, source_pos, canonical_pos in zip(
                output.loc[position_mismatch, "player_key"],
                output.loc[position_mismatch, "pos"],
                canonical_positions[position_mismatch],
            )
        }
        unsupported = sorted(
            mismatch_keys.difference(GOVERNED_CANONICAL_POSITION_MISMATCHES)
        )
        if unsupported:
            raise ValueError(
                f"{context} has unsupported canonical position mismatches: "
                f"{unsupported[:20]}"
            )
    output["player"] = output.pop("display_name")
    output = output.drop(columns="position")
    if output.player.duplicated().any():
        duplicates = output.loc[
            output.player.duplicated(keep=False),
            ["player", "pos", "player_key"],
        ].head(20)
        raise ValueError(
            f"{context} contains ambiguous canonical display labels: "
            f"{duplicates.to_dict('records')}"
        )
    return output


def build_projection_slice(
    validations_database: Path,
    v2_database: Path,
    contract: ReplayContract,
) -> tuple[pd.DataFrame, dict]:
    with closing(sqlite3.connect(validations_database)) as connection:
        rows = pd.read_sql_query(
            """
            SELECT *
            FROM Final_Validations_Resid
            WHERE season=? AND version=? AND model_spec_asof_year=?
            """,
            connection,
            params=(
                contract.year,
                contract.league,
                contract.projection_model_spec_asof_year,
            ),
        )
    context = f"{contract.year} {contract.league} historical projections"
    require_columns(
        rows,
        (
            "player",
            "pos",
            "pred_fp_per_game",
            "data_oos",
            "method_version",
            "model_spec_asof_year",
            "resid_training_through_season",
        ),
        context,
    )
    if len(rows) != contract.current_method_projection_rows:
        raise ValueError(
            f"{context} has {len(rows)} rows; expected "
            f"{contract.current_method_projection_rows}."
        )
    if not pd.to_numeric(rows.data_oos, errors="coerce").eq(1).all():
        raise ValueError(f"{context} is not fully out-of-sample.")
    if single_value(rows, "method_version", context) != PROJECTION_METHOD:
        raise ValueError(f"{context} method contract changed.")
    training_year = int(
        single_value(
            rows,
            "resid_training_through_season",
            context,
            numeric=True,
        )
    )
    if training_year != contract.projection_training_through_year:
        raise ValueError(
            f"{context} trained through {training_year}; expected "
            f"{contract.projection_training_through_year}."
        )
    model_spec_year = int(
        single_value(
            rows,
            "model_spec_asof_year",
            context,
            numeric=True,
        )
    )
    if model_spec_year != contract.projection_model_spec_asof_year:
        raise ValueError(
            f"{context} model spec is as of {model_spec_year}; expected "
            f"{contract.projection_model_spec_asof_year}."
        )

    aliases, identities = load_identity_frames(v2_database)
    rows = canonicalize_rows(
        rows,
        aliases,
        identities,
        year=contract.year,
        context=context,
    )
    if not set(rows.pos).issubset(POSITIONS):
        raise ValueError(f"{context} contains unsupported positions.")
    point_predictions = pd.to_numeric(rows.pred_fp_per_game, errors="coerce")
    if point_predictions.isna().any() or not point_predictions.map(math.isfinite).all():
        raise ValueError(f"{context} contains invalid point predictions.")

    with closing(sqlite3.connect(v2_database)) as connection:
        feature_lineage = pd.read_sql_query(
            """
            SELECT player_key, feature_cutoff_season, preseason_source_season,
                   scoring_hash, run_id
            FROM player_season_features
            WHERE season=? AND league=?
            """,
            connection,
            params=(contract.year, contract.league),
        )
    if feature_lineage.empty:
        raise ValueError(f"{context} has no V2 feature lineage.")
    feature_cutoff = int(
        single_value(
            feature_lineage,
            "feature_cutoff_season",
            f"{context} V2 features",
            numeric=True,
        )
    )
    preseason_source = int(
        single_value(
            feature_lineage,
            "preseason_source_season",
            f"{context} V2 features",
            numeric=True,
        )
    )
    scoring_hash = str(
        single_value(
            feature_lineage,
            "scoring_hash",
            f"{context} V2 features",
        )
    )
    v2_run_id = str(
        single_value(
            feature_lineage,
            "run_id",
            f"{context} V2 features",
        )
    )
    if feature_cutoff != contract.year - 1 or preseason_source != contract.year:
        raise ValueError(
            f"{context} V2 feature boundary is cutoff={feature_cutoff}, "
            f"preseason={preseason_source}; expected {contract.year - 1}/"
            f"{contract.year}."
        )
    missing_feature_keys = sorted(
        set(rows.player_key.astype(str)).difference(
            set(feature_lineage.player_key.astype(str))
        )
    )
    if missing_feature_keys:
        raise ValueError(
            f"{context} lacks V2 feature lineage for keys: "
            f"{missing_feature_keys[:20]}"
        )

    created_at = utc_now()
    output = pd.DataFrame(
        {
            "player": rows.player,
            "pos": rows.pos,
            "pred_fp_per_game": point_predictions.astype(float),
            "pred_fp_per_game_ny": 0.0,
            "dataset": PREDICTION_DATASET,
            "version": contract.league,
            "year": contract.year,
            "player_key": rows.player_key.astype(str),
            "current_projection_model_version": PROJECTION_MODEL_VERSION,
            "next_projection_model_version": "historical_replay_disabled",
            "v2_scoring_hash": scoring_hash,
            # The validation table defines the forecast population, not an
            # independently estimated appearance model.  Treat its members as
            # draft-available and record that limitation in the context table.
            "pred_appear_current": 1.0,
            "pred_appear_ny": 0.0,
            "production_handoff_version": PRODUCTION_HANDOFF_VERSION,
            "production_eligibility_version": PRODUCTION_ELIGIBILITY_VERSION,
            "current_projection_source": (
                "historical_current_method_rolling_oos_validation"
            ),
            "current_uncertainty_source": "joint_weekly_template_only",
            "independent_current_residual_draw_allowed": 0,
            "next_projection_source": "historical_replay_disabled",
            "next_uncertainty_source": "historical_replay_disabled",
            "production_handoff_created_at_utc": created_at,
        }
    )
    for column in (*CURRENT_RESIDUAL_COLUMNS, *NEXT_RESIDUAL_COLUMNS):
        output[column] = 0.0
    output = output[[
        "player",
        "pos",
        "pred_fp_per_game",
        "pred_fp_per_game_ny",
        "dataset",
        "version",
        "year",
        "player_key",
        "current_projection_model_version",
        "next_projection_model_version",
        "v2_scoring_hash",
        "pred_appear_current",
        "pred_appear_ny",
        *CURRENT_RESIDUAL_COLUMNS,
        *NEXT_RESIDUAL_COLUMNS,
        "production_handoff_version",
        "production_eligibility_version",
        "current_projection_source",
        "current_uncertainty_source",
        "independent_current_residual_draw_allowed",
        "next_projection_source",
        "next_uncertainty_source",
        "production_handoff_created_at_utc",
    ]].sort_values("player_key").reset_index(drop=True)
    lineage = {
        "projection_method_version": PROJECTION_METHOD,
        "projection_training_through_year": training_year,
        "projection_model_spec_asof_year": model_spec_year,
        "v2_feature_cutoff_year": feature_cutoff,
        "v2_preseason_source_year": preseason_source,
        "v2_scoring_hash": scoring_hash,
        "v2_feature_run_id": v2_run_id,
    }
    return output, lineage


def build_salary_slice(
    validations_database: Path,
    v2_database: Path,
    contract: ReplayContract,
    projection: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    with closing(sqlite3.connect(validations_database)) as connection:
        rows = pd.read_sql_query(
            """
            SELECT *
            FROM Salary_Backtest_Predictions
            WHERE year=? AND league=? AND method_version=?
            """,
            connection,
            params=(contract.year, contract.league, SALARY_METHOD),
        )
    context = f"{contract.year} {contract.league} historical salaries"
    require_columns(
        rows,
        (
            "player",
            "pos",
            "pred_salary",
            "is_keeper",
            "training_through_year",
            "model_spec_asof_year",
            "normalization_uses_target_actuals",
            "candidate_pool_rows",
            "keeper_count",
            "keeper_spend",
            "available_slots",
            "available_budget",
            *SALARY_RESIDUAL_COLUMNS,
        ),
        context,
    )
    if len(rows) != contract.current_method_salary_rows:
        raise ValueError(
            f"{context} has {len(rows)} rows; expected "
            f"{contract.current_method_salary_rows}."
        )
    if not pd.to_numeric(
        rows.normalization_uses_target_actuals,
        errors="coerce",
    ).eq(0).all():
        raise ValueError(f"{context} normalization uses target actuals.")
    for column, expected in (
        ("training_through_year", contract.salary_training_through_year),
        ("model_spec_asof_year", contract.salary_model_spec_asof_year),
        ("candidate_pool_rows", contract.current_method_salary_rows),
        ("keeper_count", contract.keeper_count),
        ("keeper_spend", contract.keeper_spend),
        ("available_slots", contract.available_slots),
        ("available_budget", contract.available_budget),
    ):
        observed = float(single_value(rows, column, context, numeric=True))
        if not math.isclose(observed, float(expected), abs_tol=1e-9):
            raise ValueError(
                f"{context} {column}={observed}; expected {expected}."
            )

    aliases, identities = load_identity_frames(v2_database)
    rows = canonicalize_rows(
        rows,
        aliases,
        identities,
        year=contract.year,
        context=context,
    )
    projection_keys = set(projection.player_key.astype(str))
    salary_keys = set(rows.player_key.astype(str))
    if not salary_keys.issubset(projection_keys):
        raise ValueError(
            f"{context} contains salary-only keys: projection-only="
            f"{sorted(projection_keys - salary_keys)[:10]}, salary-only="
            f"{sorted(salary_keys - projection_keys)[:10]}."
        )

    salary = pd.to_numeric(rows.pred_salary, errors="coerce")
    residuals = rows[list(SALARY_RESIDUAL_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        salary.isna().any()
        or not salary.map(math.isfinite).all()
        or residuals.isna().any().any()
        or not np.isfinite(residuals.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"{context} contains invalid salary values.")
    if not np.all(np.diff(residuals.to_numpy(dtype=float), axis=1) >= -1e-9):
        raise ValueError(f"{context} residual quantiles are not monotonic.")
    keeper_mask = pd.to_numeric(rows.is_keeper, errors="coerce").eq(1)
    if int(keeper_mask.sum()) != contract.keeper_count:
        raise ValueError(f"{context} keeper count changed.")
    if not math.isclose(
        float(salary[keeper_mask].sum()),
        contract.keeper_spend,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{context} keeper salary spend changed.")
    top_nonkeepers = (
        pd.DataFrame({"salary": salary[~keeper_mask], "player": rows.player[~keeper_mask]})
        .sort_values(["salary", "player"], ascending=[False, True])
        .head(contract.available_slots)
    )
    if not math.isclose(
        float(top_nonkeepers.salary.sum()),
        contract.available_budget,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{context} top-{contract.available_slots} nonkeeper spend is "
            f"{top_nonkeepers.salary.sum():.6f}; expected "
            f"{contract.available_budget:.6f}."
        )

    output = pd.DataFrame(
        {
            "player": rows.player,
            "salary": salary.astype(float),
            "year": contract.year,
            "league": f"{contract.league}pred",
            "std_dev": (
                residuals.salary_resid_90 - residuals.salary_resid_10
            ).clip(lower=0).div(2.563).clip(lower=0.5),
            "min_score": np.maximum(
                1.0,
                salary + residuals.salary_resid_5,
            ),
            "max_score": np.maximum(
                salary,
                salary + residuals.salary_resid_95,
            ),
            "player_key": rows.player_key.astype(str),
            "salary_population_source": (
                "historical_v2_candidate_population_rolling_origin"
            ),
            "ensemble_uncertainty_feature_source": (
                "strict_prior_salary_backtest_residuals"
            ),
            "salary_method_version": (
                "historical_current_locked_spec_v6_replay_v1"
            ),
        }
    )
    for column in SALARY_RESIDUAL_COLUMNS:
        output[column] = residuals[column].astype(float)
    # A keeper contract is already fixed at the observed draft price.  Its
    # market-price uncertainty must not perturb the locked roster obligation.
    output.loc[keeper_mask, "std_dev"] = 0.0
    output.loc[keeper_mask, "min_score"] = output.loc[keeper_mask, "salary"]
    output.loc[keeper_mask, "max_score"] = output.loc[keeper_mask, "salary"]
    output.loc[keeper_mask, list(SALARY_RESIDUAL_COLUMNS)] = 0.0
    if (
        (output.min_score > output.salary).any()
        or (output.max_score < output.salary).any()
        or (output.std_dev < 0).any()
    ):
        raise ValueError(f"{context} produced invalid uncertainty bounds.")
    output = output[[
        "player",
        "salary",
        "year",
        "league",
        "std_dev",
        "min_score",
        "max_score",
        *SALARY_RESIDUAL_COLUMNS,
        "player_key",
        "salary_population_source",
        "ensemble_uncertainty_feature_source",
        "salary_method_version",
    ]].sort_values("player_key").reset_index(drop=True)
    lineage = {
        "salary_method_version": SALARY_METHOD,
        "salary_training_through_year": contract.salary_training_through_year,
        "salary_model_spec_asof_year": contract.salary_model_spec_asof_year,
        "salary_normalization_uses_target_actuals": 0,
        "salary_available_slots": contract.available_slots,
        "salary_available_budget": contract.available_budget,
    }
    return output, lineage


def build_actual_pool_projection_fallback(
    simulation_database: Path,
    v2_database: Path,
    contract: ReplayContract,
    primary_projection: pd.DataFrame,
    primary_salary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fill drafted offensive players from strict target-year preseason data.

    The current-method validation surface can omit a small number of players
    who were drafted in the historical auction.  Actual participation selects
    the required replay population, but never supplies a point projection: the
    point center comes from the target-year V2 expert consensus built with a
    prior-season feature cutoff.  These rows receive no predicted salary and
    are available only when the replay explicitly uses actual auction prices.
    """

    with closing(sqlite3.connect(simulation_database)) as connection:
        actual = pd.read_sql_query(
            """
            SELECT player
            FROM Actual_Salaries
            WHERE year=? AND league=?
            """,
            connection,
            params=(contract.year, contract.league),
        )
    context = (
        f"{contract.year} {contract.league} strict preseason "
        "actual-pool projection fallback"
    )
    if len(actual) != contract.raw_actual_rows:
        raise ValueError(
            f"{context} raw auction has {len(actual)} rows; expected "
            f"{contract.raw_actual_rows}."
        )
    aliases, identities = load_identity_frames(v2_database)
    resolved = resolve_source_player_keys(
        actual,
        aliases,
        identities,
        year=contract.year,
        source_name=context,
        # Historical auctions also contain kicker/defense rows that are
        # intentionally outside the V2 offensive identity and feature spine.
        # The governed offensive-row count below still fails closed if an
        # eligible QB/RB/WR/TE identity does not resolve.
        require_complete=False,
    )
    with closing(sqlite3.connect(v2_database)) as connection:
        features = pd.read_sql_query(
            """
            SELECT player_key, display_name, position,
                   expert_ppg_team_game_median, projection_provider_count,
                   feature_cutoff_season, preseason_source_season,
                   scoring_hash, run_id
            FROM player_season_features
            WHERE season=? AND league=?
            """,
            connection,
            params=(contract.year, contract.league),
        )
    if features.player_key.duplicated().any():
        raise ValueError(f"{context} V2 features have duplicate keys.")
    features["position"] = features.position.astype("string").str.upper()
    offensive_features = features[features.position.isin(POSITIONS)].copy()
    actual_keys = set(resolved.player_key.astype(str))
    actual_offensive = offensive_features[
        offensive_features.player_key.astype(str).isin(actual_keys)
    ].copy()
    if len(actual_offensive) != contract.offensive_actual_rows:
        raise ValueError(
            f"{context} resolves {len(actual_offensive)} offensive auction "
            f"players; expected {contract.offensive_actual_rows}."
        )
    fallback_keys = set(actual_offensive.player_key.astype(str)).difference(
        set(primary_projection.player_key.astype(str))
    )
    if len(fallback_keys) != contract.legacy_fallback_rows:
        raise ValueError(
            f"{context} has {len(fallback_keys)} missing projection rows; "
            f"expected {contract.legacy_fallback_rows}."
        )
    fallback = actual_offensive[
        actual_offensive.player_key.astype(str).isin(fallback_keys)
    ].copy()
    for column, expected in (
        ("feature_cutoff_season", contract.year - 1),
        ("preseason_source_season", contract.year),
    ):
        observed = pd.to_numeric(fallback[column], errors="coerce")
        if not observed.eq(expected).all():
            raise ValueError(
                f"{context} {column} is not uniformly {expected}."
            )
    centers = pd.to_numeric(
        fallback.expert_ppg_team_game_median,
        errors="coerce",
    )
    provider_counts = pd.to_numeric(
        fallback.projection_provider_count,
        errors="coerce",
    )
    if (
        centers.isna().any()
        or not centers.map(math.isfinite).all()
        or provider_counts.isna().any()
        or provider_counts.le(0).any()
    ):
        raise ValueError(f"{context} lacks governed preseason expert centers.")
    scoring_hash = str(single_value(fallback, "scoring_hash", context))
    if scoring_hash != str(single_value(
        primary_projection,
        "v2_scoring_hash",
        context,
    )):
        raise ValueError(f"{context} scoring hash differs from primary rows.")
    created_at = utc_now()
    projection = pd.DataFrame(
        {
            "player": fallback.display_name,
            "pos": fallback.position,
            "pred_fp_per_game": centers.astype(float),
            "pred_fp_per_game_ny": 0.0,
            "dataset": PREDICTION_DATASET,
            "version": contract.league,
            "year": contract.year,
            "player_key": fallback.player_key.astype(str),
            "current_projection_model_version": (
                "strict_preseason_v2_expert_fallback_v1"
            ),
            "next_projection_model_version": "historical_replay_disabled",
            "v2_scoring_hash": scoring_hash,
            "pred_appear_current": 1.0,
            "pred_appear_ny": 0.0,
            "production_handoff_version": PRODUCTION_HANDOFF_VERSION,
            "production_eligibility_version": (
                "historical_actual_pool_preseason_projection_fallback_v1"
            ),
            "current_projection_source": (
                "strict_target_year_v2_expert_consensus_fallback"
            ),
            "current_uncertainty_source": "joint_weekly_template_only",
            "independent_current_residual_draw_allowed": 0,
            "next_projection_source": "historical_replay_disabled",
            "next_uncertainty_source": "historical_replay_disabled",
            "production_handoff_created_at_utc": created_at,
        }
    )
    for column in (*CURRENT_RESIDUAL_COLUMNS, *NEXT_RESIDUAL_COLUMNS):
        projection[column] = 0.0
    projection = projection[list(primary_projection.columns)]
    fallback_names = sorted(projection.player.astype(str).tolist())
    lineage = {
        "legacy_fallback_rows": len(projection),
        "legacy_fallback_players_json": json.dumps(fallback_names),
        "legacy_fallback_population_rule": (
            "actual_offensive_pool_missing_current_method_projection_uses_"
            "strict_target_year_v2_expert_center_no_predicted_salary"
        ),
    }
    return (
        projection.sort_values("player_key").reset_index(drop=True),
        primary_salary.iloc[0:0].copy(),
        lineage,
    )


def build_legacy_fallback_slices(
    simulation_database: Path,
    v2_database: Path,
    contract: ReplayContract,
    primary_projection: pd.DataFrame,
    primary_salary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load complete saved preseason rows absent from the current-method replay.

    The reviewed rolling current-method tables omit four players who still have
    both a saved 2025 preseason point projection and a saved predicted salary.
    Taking the intersection of those two causal source surfaces is independent
    of who was actually drafted and avoids ex-post population selection.  A
    projection-only or salary-only legacy row remains excluded.
    """

    if contract.year != 2025:
        return build_actual_pool_projection_fallback(
            simulation_database,
            v2_database,
            contract,
            primary_projection,
            primary_salary,
        )

    with closing(sqlite3.connect(simulation_database)) as connection:
        legacy_projection = pd.read_sql_query(
            """
            SELECT player, pos, pred_fp_per_game
            FROM Final_Predictions
            WHERE year=? AND version=? AND dataset=?
            """,
            connection,
            params=(contract.year, contract.league, PREDICTION_DATASET),
        )
        legacy_salary = pd.read_sql_query(
            """
            SELECT player, salary, std_dev, min_score, max_score
            FROM Salaries_Pred
            WHERE year=? AND league=?
            """,
            connection,
            params=(contract.year, f"{contract.league}pred"),
        )
    context = f"{contract.year} {contract.league} saved preseason fallback"
    if legacy_projection.empty or legacy_salary.empty:
        raise ValueError(f"{context} source surfaces are unavailable.")
    aliases, identities = load_identity_frames(v2_database)
    legacy_projection = canonicalize_rows(
        legacy_projection,
        aliases,
        identities,
        year=contract.year,
        context=f"{context} projections",
    )
    legacy_salary = canonicalize_rows(
        legacy_salary,
        aliases,
        identities,
        year=contract.year,
        context=f"{context} salaries",
    )
    primary_keys = set(primary_projection.player_key.astype(str))
    if primary_keys != set(primary_salary.player_key.astype(str)):
        raise ValueError(f"{context} primary surfaces do not have key parity.")
    fallback_keys = (
        set(legacy_projection.player_key.astype(str))
        & set(legacy_salary.player_key.astype(str))
    ).difference(primary_keys)
    if len(fallback_keys) != contract.legacy_fallback_rows:
        raise ValueError(
            f"{context} has {len(fallback_keys)} complete fallback rows; "
            f"expected {contract.legacy_fallback_rows}."
        )
    legacy_projection = legacy_projection[
        legacy_projection.player_key.astype(str).isin(fallback_keys)
    ].copy()
    legacy_salary = legacy_salary[
        legacy_salary.player_key.astype(str).isin(fallback_keys)
    ].copy()
    if set(legacy_projection.player_key.astype(str)) != set(
        legacy_salary.player_key.astype(str)
    ):
        raise ValueError(f"{context} fallback key parity failed.")

    scoring_hash = str(single_value(
        primary_projection,
        "v2_scoring_hash",
        context,
    ))
    created_at = utc_now()
    projection = pd.DataFrame(
        {
            "player": legacy_projection.player,
            "pos": legacy_projection.pos,
            "pred_fp_per_game": pd.to_numeric(
                legacy_projection.pred_fp_per_game,
                errors="coerce",
            ),
            "pred_fp_per_game_ny": 0.0,
            "dataset": PREDICTION_DATASET,
            "version": contract.league,
            "year": contract.year,
            "player_key": legacy_projection.player_key.astype(str),
            "current_projection_model_version": (
                "saved_2025_final_ensemble_fallback_v1"
            ),
            "next_projection_model_version": "historical_replay_disabled",
            "v2_scoring_hash": scoring_hash,
            "pred_appear_current": 1.0,
            "pred_appear_ny": 0.0,
            "production_handoff_version": PRODUCTION_HANDOFF_VERSION,
            "production_eligibility_version": (
                "historical_replay_complete_saved_surface_fallback_v1"
            ),
            "current_projection_source": (
                "saved_2025_preseason_final_ensemble_fallback"
            ),
            "current_uncertainty_source": "joint_weekly_template_only",
            "independent_current_residual_draw_allowed": 0,
            "next_projection_source": "historical_replay_disabled",
            "next_uncertainty_source": "historical_replay_disabled",
            "production_handoff_created_at_utc": created_at,
        }
    )
    if (
        projection.pred_fp_per_game.isna().any()
        or not projection.pred_fp_per_game.map(math.isfinite).all()
    ):
        raise ValueError(f"{context} has invalid point projections.")
    for column in (*CURRENT_RESIDUAL_COLUMNS, *NEXT_RESIDUAL_COLUMNS):
        projection[column] = 0.0
    projection = projection[list(primary_projection.columns)]

    point_salary = pd.to_numeric(legacy_salary.salary, errors="coerce")
    std_dev = pd.to_numeric(legacy_salary.std_dev, errors="coerce")
    lower = pd.to_numeric(legacy_salary.min_score, errors="coerce") - point_salary
    upper = pd.to_numeric(legacy_salary.max_score, errors="coerce") - point_salary
    if (
        point_salary.isna().any()
        or std_dev.isna().any()
        or lower.isna().any()
        or upper.isna().any()
        or not np.isfinite(
            np.column_stack([point_salary, std_dev, lower, upper])
        ).all()
    ):
        raise ValueError(f"{context} has invalid saved salary bounds.")
    probabilities = np.asarray([0.05, 0.10, 0.25, 0.75, 0.90, 0.95])
    interpolated = np.column_stack([
        lower + ((probability - 0.05) / 0.90) * (upper - lower)
        for probability in probabilities
    ])
    interpolated = np.maximum.accumulate(interpolated, axis=1)
    salary = pd.DataFrame(
        {
            "player": legacy_salary.player,
            "salary": point_salary.astype(float),
            "year": contract.year,
            "league": f"{contract.league}pred",
            "std_dev": std_dev.clip(lower=0).astype(float),
            "min_score": point_salary + interpolated[:, 0],
            "max_score": point_salary + interpolated[:, -1],
            "player_key": legacy_salary.player_key.astype(str),
            "salary_population_source": (
                "saved_2025_preseason_complete_surface_fallback"
            ),
            "ensemble_uncertainty_feature_source": (
                "saved_bounds_linear_quantile_fallback"
            ),
            "salary_method_version": "saved_2025_salary_fallback_v1",
        }
    )
    for index, column in enumerate(SALARY_RESIDUAL_COLUMNS):
        salary[column] = interpolated[:, index]
    salary = salary[list(primary_salary.columns)]
    fallback_names = sorted(projection.player.astype(str).tolist())
    lineage = {
        "legacy_fallback_rows": len(projection),
        "legacy_fallback_players_json": json.dumps(fallback_names),
        "legacy_fallback_population_rule": (
            "saved_projection_intersection_saved_salary_minus_current_method_keys"
        ),
    }
    return (
        projection.sort_values("player_key").reset_index(drop=True),
        salary.sort_values("player_key").reset_index(drop=True),
        lineage,
    )


def additive_floor_normalize_market(
    values: pd.Series,
    slots: int,
    budget: float,
    *,
    floor: float = 1.0,
) -> tuple[pd.Series, float, float, float]:
    values = pd.to_numeric(values, errors="coerce").fillna(floor).clip(lower=floor)
    slots = int(slots)
    budget = float(budget)
    if slots <= 0 or len(values) < slots or budget < slots * floor:
        raise ValueError(
            f"Invalid replay salary market: {len(values)} rows, "
            f"{slots} slots, ${budget:.2f} budget."
        )
    top_indices = values.nlargest(slots).index
    top_values = values.loc[top_indices]
    pre_total = float(top_values.sum())
    if math.isclose(pre_total, budget, abs_tol=1e-10):
        shift = 0.0
    elif pre_total < budget:
        shift = (budget - pre_total) / slots
    else:
        lower = float(floor - top_values.max())
        upper = 0.0
        for _ in range(100):
            midpoint = (lower + upper) / 2
            midpoint_total = float(
                np.maximum(floor, top_values + midpoint).sum()
            )
            if midpoint_total > budget:
                upper = midpoint
            else:
                lower = midpoint
        shift = (lower + upper) / 2
    adjusted = (values + shift).clip(lower=floor)
    post_total = float(adjusted.loc[top_indices].sum())
    if not math.isclose(post_total, budget, abs_tol=1e-7):
        raise ValueError(
            f"Replay salary normalization produced ${post_total:.8f}; "
            f"expected ${budget:.8f}."
        )
    return adjusted, float(shift), pre_total, post_total


def augment_replay_slices(
    primary_projection: pd.DataFrame,
    primary_salary: pd.DataFrame,
    fallback_projection: pd.DataFrame,
    fallback_salary: pd.DataFrame,
    keepers: pd.DataFrame,
    contract: ReplayContract,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    projection = pd.concat(
        [primary_projection, fallback_projection],
        ignore_index=True,
    )
    salary = pd.concat([primary_salary, fallback_salary], ignore_index=True)
    for frame, context in (
        (projection, "augmented projection"),
        (salary, "augmented salary"),
    ):
        if frame.player_key.duplicated().any():
            raise ValueError(f"Historical replay {context} has duplicate keys.")
    if len(projection) != contract.projection_rows or len(salary) != contract.salary_rows:
        raise ValueError(
            f"Historical replay augmented rows are {len(projection)}/"
            f"{len(salary)}; expected {contract.projection_rows}/"
            f"{contract.salary_rows}."
        )
    projection_keys = set(projection.player_key.astype(str))
    salary_keys = set(salary.player_key.astype(str))
    if not salary_keys.issubset(projection_keys):
        raise ValueError(
            "Historical replay augmented salary surface contains keys "
            "outside the projection surface."
        )

    keeper_keys = set(keepers.player_key.astype(str))
    keeper_mask = salary.player_key.astype(str).isin(keeper_keys)
    adjusted, shift, pre_total, post_total = additive_floor_normalize_market(
        salary.loc[~keeper_mask, "salary"],
        contract.available_slots,
        contract.available_budget,
    )
    salary.loc[~keeper_mask, "salary"] = adjusted
    residuals = salary[list(SALARY_RESIDUAL_COLUMNS)].apply(
        pd.to_numeric,
        errors="raise",
    )
    salary.loc[~keeper_mask, "min_score"] = np.maximum(
        1.0,
        salary.loc[~keeper_mask, "salary"]
        + residuals.loc[~keeper_mask, "salary_resid_5"],
    )
    salary.loc[~keeper_mask, "max_score"] = np.maximum(
        salary.loc[~keeper_mask, "salary"],
        salary.loc[~keeper_mask, "salary"]
        + residuals.loc[~keeper_mask, "salary_resid_95"],
    )
    salary.loc[keeper_mask, "std_dev"] = 0.0
    salary.loc[keeper_mask, "min_score"] = salary.loc[keeper_mask, "salary"]
    salary.loc[keeper_mask, "max_score"] = salary.loc[keeper_mask, "salary"]
    salary.loc[keeper_mask, list(SALARY_RESIDUAL_COLUMNS)] = 0.0
    if not math.isclose(
        float(
            salary.loc[~keeper_mask]
            .nlargest(contract.available_slots, "salary")
            .salary.sum()
        ),
        contract.available_budget,
        abs_tol=1e-6,
    ):
        raise ValueError("Historical replay augmented salary budget drifted.")
    lineage = {
        "augmented_salary_normalization_method": "additive_floor",
        "augmented_salary_normalization_shift": shift,
        "augmented_salary_pre_normalized_total": pre_total,
        "augmented_salary_post_normalized_total": post_total,
    }
    return (
        projection.sort_values("player_key").reset_index(drop=True),
        salary.sort_values("player_key").reset_index(drop=True),
        lineage,
    )


def build_keeper_slice(
    simulation_database: Path,
    v2_database: Path,
    contract: ReplayContract,
    projection: pd.DataFrame,
) -> pd.DataFrame:
    with closing(sqlite3.connect(simulation_database)) as connection:
        raw_actual = pd.read_sql_query(
            """
            SELECT player, actual_salary, is_keeper
            FROM Actual_Salaries
            WHERE year=? AND league=?
            """,
            connection,
            params=(contract.year, contract.league),
        )
    context = f"{contract.year} {contract.league} historical keepers"
    if len(raw_actual) != contract.raw_actual_rows:
        raise ValueError(
            f"{context} raw auction has {len(raw_actual)} rows; expected "
            f"{contract.raw_actual_rows}."
        )
    keepers = raw_actual[
        pd.to_numeric(raw_actual.is_keeper, errors="coerce").eq(1)
    ].copy()
    aliases, identities = load_identity_frames(v2_database)
    keepers = canonicalize_rows(
        keepers,
        aliases,
        identities,
        year=contract.year,
        context=context,
    )
    keeper_salary = pd.to_numeric(keepers.actual_salary, errors="coerce")
    if len(keepers) != contract.keeper_count or not math.isclose(
        float(keeper_salary.sum()),
        contract.keeper_spend,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{context} expected {contract.keeper_count} / "
            f"${contract.keeper_spend:.0f}, found {len(keepers)} / "
            f"${keeper_salary.sum():.0f}."
        )
    missing_projection = sorted(
        set(keepers.player_key.astype(str)).difference(
            set(projection.player_key.astype(str))
        )
    )
    if missing_projection:
        raise ValueError(
            f"{context} includes players outside the projection pool: "
            f"{missing_projection[:20]}"
        )
    return pd.DataFrame(
        {
            "year": contract.year,
            "league": contract.league,
            "player": keepers.player,
            "keeper_salary": keeper_salary.astype(float),
            "player_key": keepers.player_key.astype(str),
        }
    ).sort_values("player_key").reset_index(drop=True)


def build_historical_etr_slice(
    simulation_database: Path,
    v2_database: Path,
    contract: ReplayContract,
    projection: pd.DataFrame,
) -> pd.DataFrame:
    """Key the saved target-year ETR ranks for app display and matching."""

    with closing(sqlite3.connect(simulation_database)) as connection:
        raw = pd.read_sql_query(
            "SELECT * FROM Avg_ADPs WHERE year=? AND league='etr'",
            connection,
            params=(contract.year,),
        )
    context = f"{contract.year} historical ETR ranks"
    if raw.empty:
        # Earlier replay seasons have no saved ETR feed.  Build a causal rank
        # surface from the target-year V2 preseason ADP consensus, then the
        # target-year expert rank, with projection-only fringe players placed
        # at the back.  No target auction or weekly outcome enters the order.
        _, identities = load_identity_frames(v2_database)
        with closing(sqlite3.connect(v2_database)) as connection:
            features = pd.read_sql_query(
                """
                SELECT player_key, display_name, position, team, year_exp,
                       adp_median, expert_rank_median,
                       feature_cutoff_season, preseason_source_season
                FROM player_season_features
                WHERE season=? AND league=?
                """,
                connection,
                params=(contract.year, contract.league),
            )
        if features.player_key.duplicated().any():
            raise ValueError(f"{context} V2 fallback has duplicate keys.")
        replay = projection[[
            "player_key", "player", "pos", "pred_fp_per_game"
        ]].merge(
            features,
            on="player_key",
            how="left",
            validate="one_to_one",
        )
        if replay[[
            "display_name", "position", "feature_cutoff_season",
            "preseason_source_season",
        ]].isna().any().any():
            raise ValueError(f"{context} V2 fallback lacks feature coverage.")
        if not pd.to_numeric(
            replay.feature_cutoff_season,
            errors="coerce",
        ).eq(contract.year - 1).all():
            raise ValueError(f"{context} V2 fallback cutoff changed.")
        if not pd.to_numeric(
            replay.preseason_source_season,
            errors="coerce",
        ).eq(contract.year).all():
            raise ValueError(f"{context} V2 fallback source season changed.")
        adp = pd.to_numeric(replay.adp_median, errors="coerce")
        expert_rank = pd.to_numeric(
            replay.expert_rank_median,
            errors="coerce",
        )
        ordering_value = adp.combine_first(expert_rank)
        missing_order = ordering_value.isna()
        if missing_order.any():
            projection_rank = (
                replay.loc[missing_order]
                .groupby("pos").pred_fp_per_game
                .rank(method="first", ascending=False)
            )
            position_offset = replay.loc[missing_order, "pos"].map(
                {"QB": 0, "RB": 1000, "WR": 2000, "TE": 3000}
            )
            ordering_value.loc[missing_order] = (
                1_000_000 + position_offset + projection_rank
            )
        order_frame = pd.DataFrame(
            {
                "ordering_value": ordering_value,
                "player_key": replay.player_key.astype(str),
            },
            index=replay.index,
        )
        ordered_indices = order_frame.sort_values(
            ["ordering_value", "player_key"],
            kind="mergesort",
        ).index
        ranks = pd.Series(
            np.arange(1, len(replay) + 1, dtype=float),
            index=ordered_indices,
        ).reindex(replay.index)
        identities = identities.copy()
        identities["player_key"] = identities.player_key.astype(str)
        identity_positions = identities.set_index("player_key").position
        identity_position = replay.player_key.map(identity_positions).astype(
            "string"
        ).str.upper()
        authority_position = replay.pos.astype("string").str.upper()
        mismatch = authority_position.ne(identity_position)
        mismatch_tuples = set(zip(
            replay.loc[mismatch, "player_key"].astype(str),
            authority_position[mismatch].astype(str),
            identity_position[mismatch].astype(str),
        ))
        ungoverned = mismatch_tuples.difference(
            GOVERNED_CANONICAL_POSITION_MISMATCHES
        )
        if ungoverned:
            raise ValueError(
                f"{context} V2 fallback position mismatches changed: "
                f"{sorted(ungoverned)}"
            )
        source_player = replay.display_name.astype(str)
        row_hashes = [
            hashlib.sha256(
                json.dumps(
                    {
                        "player_key": str(player_key),
                        "adp": None if pd.isna(adp_value) else float(adp_value),
                        "expert_rank": (
                            None if pd.isna(expert_value)
                            else float(expert_value)
                        ),
                        "rank": float(rank),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for player_key, adp_value, expert_value, rank in zip(
                replay.player_key,
                adp,
                expert_rank,
                ranks,
            )
        ]
        snapshot_hash = hashlib.sha256(
            "".join(sorted(row_hashes)).encode("utf-8")
        ).hexdigest()
        published_at = utc_now()
        output = pd.DataFrame(index=replay.index, columns=raw.columns)
        output["player_key"] = replay.player_key.astype(str)
        output["draft_entity_key"] = "player:" + output.player_key
        output["player"] = replay.player
        output["pos"] = authority_position
        output["team"] = replay.team
        output["Years_of_Experience"] = pd.to_numeric(
            replay.year_exp,
            errors="coerce",
        )
        output["avg_pick"] = ranks.astype(float)
        output["year"] = contract.year
        output["league"] = "etr"
        output["std_dev"] = 0.0
        output["min_pick"] = ranks.astype(float)
        output["max_pick"] = ranks.astype(float)
        output["etr_rank"] = ranks.astype(float)
        output["etr_pos_rank"] = (
            pd.DataFrame({"pos": authority_position, "rank": ranks})
            .groupby("pos")["rank"]
            .rank(method="first", ascending=True)
            .astype(float)
        )
        output["etr_adp"] = ranks.astype(float)
        output["etr_adp_pos_rank"] = output.etr_pos_rank
        output["etr_adp_diff"] = 0.0
        output["source_player"] = source_player
        output["source_pos"] = replay.position.astype("string").str.upper()
        output["source_team"] = replay.team
        output["identity_position"] = identity_position
        output["current_position"] = authority_position
        output["position_authority"] = authority_position
        output["position_authority_source"] = "historical_replay_projection"
        output["position_mismatch_governed"] = mismatch.astype(int)
        output["position_mismatch_reason"] = pd.NA
        output.loc[mismatch, "position_mismatch_reason"] = [
            GOVERNED_CANONICAL_POSITION_MISMATCHES[item]
            for item in zip(
                replay.loc[mismatch, "player_key"].astype(str),
                authority_position[mismatch].astype(str),
                identity_position[mismatch].astype(str),
            )
        ]
        output["identity_match_method"] = "published_player_key"
        output["source_table"] = "V2_player_season_features"
        output["source_metric"] = (
            "adp_median_then_expert_rank_then_projection_fringe"
        )
        output["source_row_sha256"] = row_hashes
        output["source_count"] = 1
        output["feed_gap"] = np.where(adp.isna(), "missing_saved_etr", pd.NA)
        output["aggregation_policy"] = "causal_preseason_rank_fallback"
        output["bounds_policy"] = "deterministic_rank"
        output["std_dev_policy"] = "deterministic_rank"
        output["adp_policy_version"] = "historical_v2_preseason_rank_v1"
        output["removed_invalid_year_row_count"] = 0
        output["source_snapshot_sha256"] = snapshot_hash
        output["publication_snapshot_id"] = (
            f"{contract.year}:etr:v2_replay:{snapshot_hash[:16]}"
        )
        output["publication_version"] = "historical_v2_preseason_rank_v1"
        output["published_at_utc"] = published_at
        if len(output) != contract.historical_etr_rows:
            raise ValueError(
                f"{context} V2 fallback has {len(output)} rows; expected "
                f"{contract.historical_etr_rows}."
            )
        return output.sort_values("avg_pick").reset_index(drop=True)
    if len(raw) != contract.historical_etr_rows:
        raise ValueError(
            f"{context} have {len(raw)} rows; expected "
            f"{contract.historical_etr_rows}."
        )
    require_columns(raw, ("player", "avg_pick"), context)
    if raw.player.duplicated().any():
        raise ValueError(f"{context} contain duplicate source labels.")
    source_player = raw.player.astype("string").copy()
    aliases, identities = load_identity_frames(v2_database)
    resolved = resolve_source_player_keys(
        raw[["player", "avg_pick"]],
        aliases,
        identities,
        year=contract.year,
        source_name=context,
        require_complete=True,
    )
    if resolved.player_key.duplicated().any():
        raise ValueError(f"{context} resolve duplicate canonical keys.")
    with closing(sqlite3.connect(v2_database)) as connection:
        features = pd.read_sql_query(
            """
            SELECT player_key, display_name, position, team, year_exp
            FROM player_season_features
            WHERE season=? AND league=?
            """,
            connection,
            params=(contract.year, contract.league),
        )
    if features.player_key.duplicated().any():
        raise ValueError(f"{context} feature authority has duplicate keys.")
    identity_positions = identities[["player_key", "position"]].rename(
        columns={"position": "identity_position"}
    )
    enriched = (
        resolved[["player_key", "eligibility_key_match_method"]]
        .merge(features, on="player_key", how="left", validate="one_to_one")
        .merge(
            identity_positions,
            on="player_key",
            how="left",
            validate="one_to_one",
        )
    )
    if enriched[["display_name", "position", "identity_position"]].isna().any().any():
        raise ValueError(f"{context} lack V2 position authority.")
    positions = enriched.position.astype("string").str.upper()
    if not positions.isin(POSITIONS).all():
        raise ValueError(f"{context} contain non-offensive positions.")
    ranks = pd.to_numeric(raw.avg_pick, errors="coerce")
    if (
        ranks.isna().any()
        or not ranks.map(math.isfinite).all()
        or ranks.le(0).any()
        or ranks.duplicated().any()
        or not np.allclose(ranks, ranks.round())
    ):
        raise ValueError(f"{context} contain invalid overall ranks.")

    output = raw.copy().reset_index(drop=True)
    output["player_key"] = enriched.player_key.astype(str)
    output["draft_entity_key"] = "player:" + output.player_key
    output["player"] = enriched.display_name
    output["pos"] = positions
    output["team"] = enriched.team
    output["Years_of_Experience"] = pd.to_numeric(
        enriched.year_exp,
        errors="coerce",
    )
    output["avg_pick"] = ranks.astype(float)
    output["year"] = contract.year
    output["league"] = "etr"
    output["etr_rank"] = ranks.astype(float)
    output["etr_pos_rank"] = (
        pd.DataFrame({"pos": positions, "rank": ranks})
        .groupby("pos")["rank"]
        .rank(method="first", ascending=True)
        .astype(float)
    )
    output["etr_adp"] = ranks.astype(float)
    output["etr_adp_pos_rank"] = output.etr_pos_rank
    output["etr_adp_diff"] = 0.0
    output["source_player"] = source_player
    output["source_pos"] = positions
    output["source_team"] = enriched.team
    output["identity_position"] = (
        enriched.identity_position.astype("string").str.upper()
    )
    output["current_position"] = positions
    output["position_authority"] = positions
    output["position_authority_source"] = "player_season_features"
    mismatch = output.identity_position.ne(output.position_authority)
    output["position_mismatch_governed"] = mismatch.astype(int)
    output["position_mismatch_reason"] = pd.NA
    mismatch_tuples = set(
        zip(
            output.loc[mismatch, "player_key"],
            output.loc[mismatch, "position_authority"],
            output.loc[mismatch, "identity_position"],
        )
    )
    ungoverned_mismatches = mismatch_tuples.difference(
        GOVERNED_CANONICAL_POSITION_MISMATCHES
    )
    if ungoverned_mismatches:
        raise ValueError(
            f"{context} position mismatches changed: "
            f"{sorted(ungoverned_mismatches)}"
        )
    output.loc[mismatch, "position_mismatch_reason"] = [
        GOVERNED_CANONICAL_POSITION_MISMATCHES[item]
        for item in zip(
            output.loc[mismatch, "player_key"],
            output.loc[mismatch, "position_authority"],
            output.loc[mismatch, "identity_position"],
        )
    ]
    output["identity_match_method"] = enriched.eligibility_key_match_method
    output["source_table"] = f"saved_{contract.year}_Avg_ADPs_etr"
    output["source_metric"] = "avg_pick_as_etr_rank"
    output["source_row_sha256"] = [
        hashlib.sha256(
            json.dumps(
                {"player": str(player), "rank": float(rank)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for player, rank in zip(source_player, ranks)
    ]
    snapshot_hash = hashlib.sha256(
        "".join(sorted(output.source_row_sha256)).encode("utf-8")
    ).hexdigest()
    published_at = utc_now()
    output["source_count"] = 1
    output["feed_gap"] = pd.NA
    output["aggregation_policy"] = "saved_rank_identity_publication"
    output["bounds_policy"] = "not_applicable_rank"
    output["std_dev_policy"] = "not_applicable_rank"
    output["adp_policy_version"] = "historical_etr_rank_replay_v1"
    output["removed_invalid_year_row_count"] = 0
    output["source_snapshot_sha256"] = snapshot_hash
    output["publication_snapshot_id"] = (
        f"{contract.year}:etr:historical_replay:{snapshot_hash[:16]}"
    )
    output["publication_version"] = "historical_etr_rank_replay_v1"
    output["published_at_utc"] = published_at
    if output.player_key.isna().any() or output.player_key.duplicated().any():
        raise ValueError(f"{context} publication has invalid canonical keys.")
    return output[list(raw.columns)].sort_values("avg_pick").reset_index(drop=True)


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    ]


def insert_frame(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
) -> None:
    existing_columns = table_columns(connection, table)
    missing = sorted(set(frame.columns).difference(existing_columns))
    if missing:
        raise ValueError(f"{table} lacks publication columns: {missing}")
    columns = list(frame.columns)
    quoted = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    sqlite_frame = frame.astype(object).where(pd.notna(frame), None)
    connection.executemany(
        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
        [
            tuple(row)
            for row in sqlite_frame.itertuples(index=False, name=None)
        ],
    )


def ensure_context_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CONTEXT_TABLE} (
            year INTEGER NOT NULL,
            league TEXT NOT NULL,
            context_version TEXT NOT NULL,
            replay_kind TEXT NOT NULL,
            projection_method_version TEXT NOT NULL,
            projection_training_through_year INTEGER NOT NULL,
            projection_model_spec_asof_year INTEGER NOT NULL,
            salary_method_version TEXT NOT NULL,
            salary_training_through_year INTEGER NOT NULL,
            salary_model_spec_asof_year INTEGER NOT NULL,
            legacy_fallback_rows INTEGER NOT NULL,
            legacy_fallback_players_json TEXT NOT NULL,
            legacy_fallback_population_rule TEXT NOT NULL,
            augmented_salary_normalization_method TEXT NOT NULL,
            augmented_salary_normalization_shift REAL NOT NULL,
            augmented_salary_pre_normalized_total REAL NOT NULL,
            augmented_salary_post_normalized_total REAL NOT NULL,
            v2_feature_cutoff_year INTEGER NOT NULL,
            v2_preseason_source_year INTEGER NOT NULL,
            v2_scoring_hash TEXT NOT NULL,
            v2_feature_run_id TEXT NOT NULL,
            max_template_donor_season INTEGER NOT NULL,
            projection_rows INTEGER NOT NULL,
            salary_rows INTEGER NOT NULL,
            historical_etr_rows INTEGER NOT NULL,
            raw_actual_rows INTEGER NOT NULL,
            offensive_actual_rows INTEGER NOT NULL,
            keeper_count INTEGER NOT NULL,
            keeper_spend REAL NOT NULL,
            current_appearance_policy TEXT NOT NULL,
            next_year_keeper_signal_enabled INTEGER NOT NULL,
            selection_premium_enabled INTEGER NOT NULL,
            created_at_utc TEXT NOT NULL,
            PRIMARY KEY (year, league)
        )
        """
    )


def publish_core_slices(
    simulation_database: Path,
    contract: ReplayContract,
    projection: pd.DataFrame,
    salary: pd.DataFrame,
    keepers: pd.DataFrame,
    historical_etr: pd.DataFrame,
    lineage: dict,
) -> pd.DataFrame:
    context = pd.DataFrame([
        {
            "year": contract.year,
            "league": contract.league,
            "context_version": REPLAY_CONTEXT_VERSION,
            "replay_kind": "current_method_rolling_origin_historical_replay",
            **lineage,
            "max_template_donor_season": contract.year - 1,
            "projection_rows": len(projection),
            "salary_rows": len(salary),
            "historical_etr_rows": len(historical_etr),
            "raw_actual_rows": contract.raw_actual_rows,
            "offensive_actual_rows": contract.offensive_actual_rows,
            "keeper_count": len(keepers),
            "keeper_spend": float(keepers.keeper_salary.sum()),
            "current_appearance_policy": "assume_forecast_population_available",
            "next_year_keeper_signal_enabled": 0,
            "selection_premium_enabled": 0,
            "created_at_utc": utc_now(),
        }
    ])
    # These values are useful in the build receipt but intentionally excluded
    # from the durable DB schema, which focuses on leakage and app behavior.
    context = context.drop(
        columns=[
            "salary_normalization_uses_target_actuals",
            "salary_available_slots",
            "salary_available_budget",
        ],
        errors="ignore",
    )
    with closing(sqlite3.connect(simulation_database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        ensure_context_table(connection)
        connection.execute(
            f"DELETE FROM {PROJECTION_TABLE} "
            "WHERE year=? AND version=? AND dataset=?",
            (contract.year, contract.league, PREDICTION_DATASET),
        )
        connection.execute(
            f"DELETE FROM {SALARY_TABLE} WHERE year=? AND league=?",
            (contract.year, f"{contract.league}pred"),
        )
        connection.execute(
            f"DELETE FROM {KEEPER_TABLE} WHERE year=? AND league=?",
            (contract.year, contract.league),
        )
        connection.execute(
            f"DELETE FROM {ADP_TABLE} WHERE year=? AND league='etr'",
            (contract.year,),
        )
        if PREMIUM_TABLE in {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }:
            connection.execute(
                f"DELETE FROM {PREMIUM_TABLE} WHERE year=? AND league=?",
                (contract.year, contract.league),
            )
        connection.execute(
            f"DELETE FROM {CONTEXT_TABLE} WHERE year=? AND league=?",
            (contract.year, contract.league),
        )
        insert_frame(connection, PROJECTION_TABLE, projection)
        insert_frame(connection, SALARY_TABLE, salary)
        insert_frame(connection, KEEPER_TABLE, keepers)
        insert_frame(connection, ADP_TABLE, historical_etr)
        insert_frame(connection, CONTEXT_TABLE, context)

        saved = {
            "projection_rows": connection.execute(
                f"SELECT COUNT(*) FROM {PROJECTION_TABLE} "
                "WHERE year=? AND version=? AND dataset=?",
                (contract.year, contract.league, PREDICTION_DATASET),
            ).fetchone()[0],
            "salary_rows": connection.execute(
                f"SELECT COUNT(*) FROM {SALARY_TABLE} "
                "WHERE year=? AND league=?",
                (contract.year, f"{contract.league}pred"),
            ).fetchone()[0],
            "keeper_count": connection.execute(
                f"SELECT COUNT(*) FROM {KEEPER_TABLE} "
                "WHERE year=? AND league=?",
                (contract.year, contract.league),
            ).fetchone()[0],
            "historical_etr_rows": connection.execute(
                f"SELECT COUNT(*) FROM {ADP_TABLE} "
                "WHERE year=? AND league='etr'",
                (contract.year,),
            ).fetchone()[0],
        }
        expected = {
            "projection_rows": len(projection),
            "salary_rows": len(salary),
            "keeper_count": len(keepers),
            "historical_etr_rows": len(historical_etr),
        }
        if saved != expected:
            raise ValueError(
                f"Historical core publication failed verification: "
                f"{saved} != {expected}."
            )
        connection.commit()
    return context


def backup_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(source)) as source_connection,
        closing(sqlite3.connect(destination)) as destination_connection,
    ):
        source_connection.backup(destination_connection)
    with closing(sqlite3.connect(destination)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise ValueError(
            f"Staged database failed integrity check: {destination}: {integrity}"
        )


def stage_source_databases(
    source_directory: Path,
    stage_directory: Path,
    contract: ReplayContract,
    *,
    replace_stage: bool,
) -> dict[str, Path]:
    source_names = {
        "simulation": "Simulation.sqlite3",
        "model_inputs": "Model_Inputs.sqlite3",
        "validations": "Validations.sqlite3",
        "season_stats": "Season_Stats_New.sqlite3",
        "v2": f"Projection_V2_{contract.league}.sqlite3",
    }
    sources = {
        label: (source_directory / name).resolve()
        for label, name in source_names.items()
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Historical replay source databases are missing: " + ", ".join(missing)
        )
    stage_directory.mkdir(parents=True, exist_ok=True)
    destinations = {
        label: (stage_directory / path.name).resolve()
        for label, path in sources.items()
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing and not replace_stage:
        raise FileExistsError(
            "Historical replay staging already exists. Pass --replace-stage "
            "to rebuild these derived artifacts: " + ", ".join(existing)
        )
    for label, source in sources.items():
        destination = destinations[label]
        print(f"Staging {label}: {source.name} -> {destination}")
        backup_database(source, destination)
    return destinations


def run_weekly_replay(
    databases: dict[str, Path],
    contract: ReplayContract,
) -> None:
    environment = os.environ.copy()
    environment["FF_CURRENT_SEASON"] = str(contract.year)
    environment["FF_HISTORICAL_AUCTION_REPLAY"] = "1"
    managed_runtime = (
        REPOSITORY_ROOT / ".venv_ff_312" / "Scripts" / "python.exe"
    )
    replay_python = managed_runtime if managed_runtime.is_file() else Path(sys.executable)
    command = [
        str(replay_python),
        str(REPOSITORY_ROOT / "Scripts" / "Modeling" / "s4_Best_Ball_Weekly.py"),
        "--league",
        contract.league,
        "--simulation-db",
        str(databases["simulation"]),
        "--v2-db",
        str(databases["v2"]),
        "--no-app-sync",
        "--historical-replay",
    ]
    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def validate_completed_replay(
    simulation_database: Path,
    contract: ReplayContract,
) -> dict:
    with closing(sqlite3.connect(simulation_database)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        projection_rows, projection_keys = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT player_key) "
            f"FROM {PROJECTION_TABLE} "
            "WHERE year=? AND version=? AND dataset=?",
            (contract.year, contract.league, PREDICTION_DATASET),
        ).fetchone()
        salary_rows, salary_keys = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT player_key) "
            f"FROM {SALARY_TABLE} WHERE year=? AND league=?",
            (contract.year, f"{contract.league}pred"),
        ).fetchone()
        actual_rows, actual_keys = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT player_key) "
            f"FROM {SALARY_TABLE} WHERE year=? AND league=?",
            (contract.year, f"{contract.league}_actual"),
        ).fetchone()
        keeper_count, keeper_spend = connection.execute(
            f"SELECT COUNT(*), SUM(keeper_salary) FROM {KEEPER_TABLE} "
            "WHERE year=? AND league=?",
            (contract.year, contract.league),
        ).fetchone()
        historical_etr_rows, historical_etr_keys = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT player_key) FROM {ADP_TABLE} "
            "WHERE year=? AND league='etr'",
            (contract.year,),
        ).fetchone()
        map_rows, map_keys = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT player_key) "
            "FROM Best_Ball_Weekly_Player_Map "
            "WHERE year=? AND version=? AND dataset=?",
            (contract.year, contract.league, PREDICTION_DATASET),
        ).fetchone()
        max_donor_season = connection.execute(
            """
            SELECT MAX(t.season)
            FROM Best_Ball_Weekly_Template_Pools p
            JOIN Best_Ball_Weekly_Templates t
              ON t.template_id=p.template_id
             AND t.league=p.template_league
            WHERE p.pool_year=? AND p.pool_version=? AND p.pool_dataset=?
            """,
            (contract.year, contract.league, PREDICTION_DATASET),
        ).fetchone()[0]
        premium_rows = connection.execute(
            f"SELECT COUNT(*) FROM {PREMIUM_TABLE} WHERE year=? AND league=?",
            (contract.year, contract.league),
        ).fetchone()[0]
        next_enabled = connection.execute(
            f"SELECT next_year_keeper_signal_enabled FROM {CONTEXT_TABLE} "
            "WHERE year=? AND league=?",
            (contract.year, contract.league),
        ).fetchone()
    receipt = {
        "integrity_check": integrity,
        "projection_rows": int(projection_rows),
        "projection_keys": int(projection_keys),
        "salary_rows": int(salary_rows),
        "salary_keys": int(salary_keys),
        "actual_salary_rows": int(actual_rows),
        "actual_salary_keys": int(actual_keys),
        "keeper_count": int(keeper_count),
        "keeper_spend": float(keeper_spend),
        "historical_etr_rows": int(historical_etr_rows),
        "historical_etr_keys": int(historical_etr_keys),
        "weekly_player_map_rows": int(map_rows),
        "weekly_player_map_keys": int(map_keys),
        "max_template_donor_season": int(max_donor_season),
        "selection_premium_rows": int(premium_rows),
        "next_year_keeper_signal_enabled": int(next_enabled[0]),
    }
    expected = {
        "integrity_check": "ok",
        "projection_rows": contract.projection_rows,
        "projection_keys": contract.projection_rows,
        "salary_rows": contract.salary_rows,
        "salary_keys": contract.salary_rows,
        "actual_salary_rows": contract.offensive_actual_rows,
        "actual_salary_keys": contract.offensive_actual_rows,
        "keeper_count": contract.keeper_count,
        "keeper_spend": contract.keeper_spend,
        "historical_etr_rows": contract.historical_etr_rows,
        "historical_etr_keys": contract.historical_etr_rows,
        "weekly_player_map_rows": contract.projection_rows,
        "weekly_player_map_keys": contract.projection_rows,
        "max_template_donor_season": contract.year - 1,
        "selection_premium_rows": 0,
        "next_year_keeper_signal_enabled": 0,
    }
    if receipt != expected:
        raise ValueError(
            "Completed historical replay failed validation: "
            + json.dumps({"observed": receipt, "expected": expected}, sort_keys=True)
        )
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated historical Auction app replay database."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--league", default="beta")
    parser.add_argument(
        "--source-database-dir",
        type=Path,
        default=REPOSITORY_ROOT / "Data" / "Databases",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--replace-stage",
        action="store_true",
        help="Replace only the derived replay databases in --stage-dir.",
    )
    parser.add_argument(
        "--skip-weekly",
        action="store_true",
        help="Publish core slices but skip the weekly-template player-map build.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    contract = get_replay_contract(args.year, args.league)
    if args.stage_dir is not None:
        stage_dir = args.stage_dir
    elif contract.year == 2025:
        stage_dir = (
            REPOSITORY_ROOT
            / "research"
            / "studies"
            / "2026-08-26_auction_2025_historical_replay"
            / "staging"
            / "databases"
        )
    else:
        stage_dir = (
            REPOSITORY_ROOT
            / "research"
            / "studies"
            / "2026-08-27_auction_excess_multi_origin"
            / "staging"
            / str(contract.year)
            / "databases"
        )
    databases = stage_source_databases(
        args.source_database_dir.resolve(),
        stage_dir.resolve(),
        contract,
        replace_stage=args.replace_stage,
    )
    projection, projection_lineage = build_projection_slice(
        databases["validations"],
        databases["v2"],
        contract,
    )
    salary, salary_lineage = build_salary_slice(
        databases["validations"],
        databases["v2"],
        contract,
        projection,
    )
    keepers = build_keeper_slice(
        databases["simulation"],
        databases["v2"],
        contract,
        projection,
    )
    fallback_projection, fallback_salary, fallback_lineage = (
        build_legacy_fallback_slices(
            databases["simulation"],
            databases["v2"],
            contract,
            projection,
            salary,
        )
    )
    projection, salary, augmentation_lineage = augment_replay_slices(
        projection,
        salary,
        fallback_projection,
        fallback_salary,
        keepers,
        contract,
    )
    historical_etr = build_historical_etr_slice(
        databases["simulation"],
        databases["v2"],
        contract,
        projection,
    )
    context = publish_core_slices(
        databases["simulation"],
        contract,
        projection,
        salary,
        keepers,
        historical_etr,
        {
            **projection_lineage,
            **salary_lineage,
            **fallback_lineage,
            **augmentation_lineage,
        },
    )
    actual = build_actual_salary_slice(
        databases["simulation"],
        databases["v2"],
        year=contract.year,
        league=contract.league,
        expected_pool_rows=contract.offensive_actual_rows,
    )
    publish_actual_salary_slice(actual, databases["simulation"])

    receipt = {
        "contract": asdict(contract),
        "context": context.iloc[0].to_dict(),
        "simulation_database": str(databases["simulation"]),
        "weekly_build_skipped": bool(args.skip_weekly),
    }
    if not args.skip_weekly:
        run_weekly_replay(databases, contract)
        receipt["validation"] = validate_completed_replay(
            databases["simulation"],
            contract,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return receipt


if __name__ == "__main__":
    main()
