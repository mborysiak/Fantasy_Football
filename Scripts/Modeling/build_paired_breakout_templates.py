"""Build paired current-season and following-season breakout templates.

The matching row is an origin-season player observed before season ``N``.
It uses only preseason-``N`` context, including the causal forecast of season
``N + 1``.  The paired outcomes keep the same historical player's managed
weekly path in ``N`` together with appearance and conditional PPG in
``N + 1``.  Salary is intentionally excluded from every match dimension.

These tables are research-only. The input Simulation database is read-only;
--output-db must name a separate research artifact. Production refresh and apps
do not build, publish, or consume these tables.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION_DB = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"

PROFILE_VERSION = "paired_breakout_v1"
DATASET = "final_ensemble"
POSITIONS = ("RB", "WR", "TE")
LEAGUES = ("beta", "nv", "dk", "nffc")
V2_DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
    "nffc": REPO_ROOT / "Data" / "Databases" / "Projection_V2_nffc.sqlite3",
    "nv": REPO_ROOT / "Data" / "Databases" / "Projection_V2_nv.sqlite3",
}

TEMPLATE_TABLE = "Breakout_Paired_Templates"
POOL_TABLE = "Breakout_Paired_Template_Pools"
PLAYER_MAP_TABLE = "Breakout_Paired_Player_Map"
AUDIT_TABLE = "Breakout_Paired_Template_Audit"
GENERATED_TABLES = (
    TEMPLATE_TABLE,
    POOL_TABLE,
    PLAYER_MAP_TABLE,
    AUDIT_TABLE,
)

MAX_POOL_SIZE = 80
MIN_POOL_SIZE = 40
RECENCY_HALF_LIFE = 12.0
MIN_LOCAL_WEIGHT = 0.35
LOCAL_DISTANCE_SCALE = 1.50
MAX_SAMPLE_PROBABILITY = 0.05
MATCH_FILL_VALUE = 0.5
KERNEL_BANDWIDTH = {"RB": 0.45, "WR": 0.35, "TE": 0.40}
DEFAULT_WAIVER_BASELINES = {"RB": 7.0, "WR": 7.0, "TE": 5.0}

# The production matcher remains the calibrated weekly scoring authority.  This
# review-only profile decreases its projection dominance, retains the same
# opportunity surface, and adds the signed following-season trajectory as a
# substantial but not exclusive matching dimension.
COMMON_MATCH_WEIGHTS = {
    "match_projection_rank_pct": 2.00,
    "match_projection_ppg_scaled": 1.00,
    "year_exp_scaled": 1.50,
    "adp_rank_pct": 0.75,
    "market_projection_gap": 1.00,
    "projection_disagreement_frac": 1.00,
    "rank_disagreement_scaled": 0.75,
    "breakout_next_growth_rank_pct": 1.50,
    "breakout_next_appearance": 0.25,
}
POSITION_MATCH_WEIGHTS = {
    "RB": {
        "rush_proj_rank_pct": 1.00,
        "rec_proj_rank_pct": 1.00,
        "rec_share_of_own_points": 1.00,
        "rb_rush_share_of_room": 1.25,
        "rb_rec_share_of_room": 0.75,
        "rb_combined_share_of_room": 1.00,
        "rb_room_rank_scaled": 0.75,
        "rb_gap_to_next_share": 0.75,
        "rb_room_concentration": 0.50,
    },
    "WR": {
        "rec_proj_rank_pct": 1.00,
        "team_rec_share": 1.25,
        "pass_catcher_rank_scaled": 0.75,
        "pass_catcher_gap_to_next_share": 0.75,
        "pass_catcher_room_concentration": 0.50,
        "team_qb_pass_proj_rank_pct": 0.50,
    },
    "TE": {
        "rec_proj_rank_pct": 1.00,
        "team_rec_share": 1.25,
        "pass_catcher_rank_scaled": 0.75,
        "pass_catcher_gap_to_next_share": 0.75,
        "pass_catcher_room_concentration": 0.50,
        "team_qb_pass_proj_rank_pct": 0.50,
    },
}
MATCH_WEIGHTS = {
    position: {**COMMON_MATCH_WEIGHTS, **POSITION_MATCH_WEIGHTS[position]}
    for position in POSITIONS
}


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    ]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _rank_percentile(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.Series:
    numeric = pd.to_numeric(frame[value], errors="coerce")
    return numeric.groupby([frame[column] for column in groups]).rank(
        method="average",
        pct=True,
    )


def _cap_probability_vector(probabilities: np.ndarray, cap: float) -> np.ndarray:
    """Normalize a vector while respecting the production-style 5% cap."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Probability vector must be non-empty and one-dimensional")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        values = np.ones(len(values), dtype=float)
    values = values / values.sum()
    effective_cap = max(float(cap), 1.0 / len(values))
    fixed = np.zeros(len(values), dtype=bool)
    output = np.zeros(len(values), dtype=float)
    while True:
        remaining = ~fixed
        remaining_mass = 1.0 - output[fixed].sum()
        if not remaining.any():
            break
        base = values[remaining]
        if base.sum() <= 0:
            candidate = np.full(base.shape, remaining_mass / len(base))
        else:
            candidate = remaining_mass * base / base.sum()
        above = candidate > effective_cap + 1e-12
        if not above.any():
            output[remaining] = candidate
            break
        remaining_indices = np.flatnonzero(remaining)
        capped_indices = remaining_indices[above]
        output[capped_indices] = effective_cap
        fixed[capped_indices] = True
    return output / output.sum()


def _weekly_columns(connection: sqlite3.Connection) -> list[int]:
    columns = set(_table_columns(connection, "Best_Ball_Weekly_Templates"))
    weeks = []
    for week in range(1, 19):
        if f"managed_week_{week}" in columns:
            weeks.append(week)
    if not weeks:
        raise ValueError("Weekly templates have no managed weekly profile columns")
    return weeks


def load_weekly_donors(
    connection: sqlite3.Connection,
    league: str,
) -> tuple[pd.DataFrame, list[int]]:
    weeks = _weekly_columns(connection)
    required_features = sorted(
        {feature for weights in MATCH_WEIGHTS.values() for feature in weights}
        - {"breakout_next_growth_rank_pct", "breakout_next_appearance"}
    )
    select_columns = [
        "league",
        "template_id",
        "player_key",
        "player",
        "pos",
        "team",
        "season",
        "avg_pick",
        "year_exp",
        "managed_profile_ppg",
        "managed_residual_center_ppg",
        "managed_active_ppg_resid",
        "active_ppg",
        "played_games",
        "active_games",
        *required_features,
        *[f"managed_week_{week}" for week in weeks],
    ]
    columns = set(_table_columns(connection, "Best_Ball_Weekly_Templates"))
    missing = sorted(set(select_columns).difference(columns))
    if missing:
        raise ValueError(f"Weekly donor table is missing breakout fields: {missing}")
    quoted = ", ".join(f'"{column}"' for column in select_columns)
    placeholders = ", ".join("?" for _ in POSITIONS)
    donors = pd.read_sql_query(
        f"""
        SELECT {quoted}
        FROM Best_Ball_Weekly_Templates
        WHERE league=?
          AND template_eligible=1
          AND pos IN ({placeholders})
        """,
        connection,
        params=(league, *POSITIONS),
    )
    if donors.duplicated(["league", "template_id"]).any():
        raise ValueError(f"{league} weekly donors are not unique by template_id")
    return donors, weeks


def load_current_targets(
    connection: sqlite3.Connection,
    league: str,
    year: int,
    dataset: str,
) -> pd.DataFrame:
    placeholders = ", ".join("?" for _ in POSITIONS)
    targets = pd.read_sql_query(
        f"""
        SELECT *
        FROM Best_Ball_Weekly_Player_Map
        WHERE year=? AND version=? AND dataset=?
          AND pos IN ({placeholders})
        """,
        connection,
        params=(int(year), league, dataset, *POSITIONS),
    )
    if targets.duplicated(["player_key"]).any():
        raise ValueError(f"{league} breakout targets are not unique by player_key")
    keepers = pd.read_sql_query(
        """
        SELECT player_key, player keeper_player, keeper_salary
        FROM League_Keepers
        WHERE year=? AND league=?
        """,
        connection,
        params=(int(year), league),
    )
    if keepers.player_key.duplicated().any():
        raise ValueError(f"{league} keepers are not unique by player_key")
    targets = targets.merge(
        keepers,
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    targets["is_keeper"] = targets.keeper_salary.notna().astype(int)
    return targets


def load_next_year_context(v2_database: Path, league: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with closing(sqlite3.connect(v2_database)) as connection:
        handoff = pd.read_sql_query(
            """
            SELECT player_key,
                   origin_season,
                   target_season,
                   position,
                   predicted_next_year_conditional_ppg,
                   predicted_next_year_appearance_probability,
                   training_through_origin,
                   target_outcome_through,
                   forecast_status
            FROM next_year_template_handoff
            WHERE league=?
            """,
            connection,
            params=(league,),
        )
        outcomes = pd.read_sql_query(
            """
            SELECT player_key,
                   origin_season,
                   target_season,
                   position,
                   next_participation_target_available,
                   next_appeared,
                   next_conditional_ppg,
                   next_conditional_ppg_training_eligible,
                   next_target_join_status
            FROM next_year_targets
            WHERE league=?
            """,
            connection,
            params=(league,),
        )
    for label, frame in (("handoff", handoff), ("outcomes", outcomes)):
        if frame.duplicated(["player_key", "origin_season"]).any():
            raise ValueError(f"{league} next-year {label} has duplicate origins")
    bad_embargo = (
        pd.to_numeric(handoff.training_through_origin, errors="coerce")
        >= pd.to_numeric(handoff.origin_season, errors="coerce") - 1
    )
    if bad_embargo.any():
        raise ValueError(f"{league} next-year handoff violates its outcome embargo")
    return handoff, outcomes


def build_paired_donors(
    donors: pd.DataFrame,
    handoff: pd.DataFrame,
    outcomes: pd.DataFrame,
    weeks: list[int],
    league: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows = len(donors)
    handoff_context = handoff.rename(
        columns={
            "position": "next_forecast_position",
            "target_season": "forecast_target_season",
        }
    )
    actual_context = outcomes.rename(
        columns={
            "position": "next_outcome_position",
            "target_season": "actual_target_season",
        }
    )
    paired = donors.merge(
        handoff_context,
        left_on=["player_key", "season"],
        right_on=["player_key", "origin_season"],
        how="left",
        validate="one_to_one",
    ).merge(
        actual_context,
        left_on=["player_key", "season"],
        right_on=["player_key", "origin_season"],
        how="left",
        suffixes=("", "_actual"),
        validate="one_to_one",
    )
    forecast_available = (
        paired.predicted_next_year_conditional_ppg.notna()
        & paired.predicted_next_year_appearance_probability.notna()
    )
    outcome_available = paired.next_participation_target_available.eq(1)
    position_consistent = (
        paired.pos.eq(paired.next_forecast_position)
        & paired.pos.eq(paired.next_outcome_position)
    )
    next_appeared = pd.to_numeric(paired.next_appeared, errors="coerce")
    next_ppg = pd.to_numeric(paired.next_conditional_ppg, errors="coerce")
    appearance_contract = next_appeared.isin([0, 1]) & (
        (next_appeared.eq(0) & next_ppg.isna())
        | (next_appeared.eq(1) & next_ppg.notna())
    )
    eligible = (
        forecast_available
        & outcome_available
        & position_consistent
        & appearance_contract
    )
    paired = paired.loc[eligible].copy()
    if paired.empty:
        raise ValueError(f"{league} has no eligible paired breakout donors")

    paired["profile_version"] = PROFILE_VERSION
    paired["origin_season"] = pd.to_numeric(
        paired.season, errors="raise"
    ).astype(int)
    paired["next_target_season"] = pd.to_numeric(
        paired.actual_target_season, errors="raise"
    ).astype(int)
    if not paired.next_target_season.eq(paired.origin_season + 1).all():
        raise ValueError(f"{league} paired donors have a non-adjacent next season")
    paired["current_pred_ppg"] = pd.to_numeric(
        paired.managed_residual_center_ppg, errors="coerce"
    )
    paired["current_actual_ppg"] = pd.to_numeric(
        paired.active_ppg, errors="coerce"
    )
    paired["current_ppg_residual"] = pd.to_numeric(
        paired.managed_active_ppg_resid, errors="coerce"
    )
    paired["predicted_next_ppg"] = pd.to_numeric(
        paired.predicted_next_year_conditional_ppg, errors="coerce"
    )
    paired["breakout_signed_next_growth"] = (
        paired.predicted_next_ppg - paired.current_pred_ppg
    )
    paired["breakout_next_appearance"] = pd.to_numeric(
        paired.predicted_next_year_appearance_probability,
        errors="coerce",
    )
    paired["breakout_next_growth_rank_pct"] = _rank_percentile(
        paired,
        "breakout_signed_next_growth",
        ["pos", "origin_season"],
    )
    paired["actual_next_appeared"] = pd.to_numeric(
        paired.next_appeared, errors="raise"
    ).astype(int)
    paired["actual_next_conditional_ppg"] = pd.to_numeric(
        paired.next_conditional_ppg, errors="coerce"
    )
    paired["actual_next_unconditional_ppg"] = np.where(
        paired.actual_next_appeared.eq(1),
        paired.actual_next_conditional_ppg,
        0.0,
    )
    paired["actual_next_ppg_residual"] = np.where(
        paired.actual_next_appeared.eq(1),
        paired.actual_next_conditional_ppg - paired.predicted_next_ppg,
        np.nan,
    )

    current_week_columns = []
    for week in weeks:
        column = f"current_week_{week}"
        current_week_columns.append(column)
        paired[column] = (
            pd.to_numeric(paired[f"managed_week_{week}"], errors="coerce")
            .fillna(0.0)
            * pd.to_numeric(paired.managed_profile_ppg, errors="coerce").fillna(0.0)
        )

    def period_points(selected_weeks: list[int]) -> pd.Series:
        columns = [f"current_week_{week}" for week in selected_weeks if week in weeks]
        if not columns:
            return pd.Series(0.0, index=paired.index)
        return paired[columns].sum(axis=1)

    early_weeks = [week for week in weeks if week <= 9]
    late_weeks = [week for week in weeks if week >= 10]
    playoff_weeks = [week for week in weeks if week >= 14]
    paired["current_season_points"] = period_points(weeks)
    paired["current_late_points"] = period_points(late_weeks)
    paired["current_playoff_points"] = period_points(playoff_weeks)
    paired["current_early_calendar_ppg"] = period_points(early_weeks) / max(
        len(early_weeks), 1
    )
    paired["current_late_calendar_ppg"] = paired.current_late_points / max(
        len(late_weeks), 1
    )
    paired["current_playoff_calendar_ppg"] = paired.current_playoff_points / max(
        len(playoff_weeks), 1
    )
    paired["current_late_lift"] = (
        paired.current_late_calendar_ppg - paired.current_early_calendar_ppg
    )

    season_excess = np.zeros(len(paired), dtype=float)
    playoff_excess = np.zeros(len(paired), dtype=float)
    for position, baseline in DEFAULT_WAIVER_BASELINES.items():
        mask = paired.pos.eq(position).to_numpy()
        if not mask.any():
            continue
        weekly_values = paired.loc[mask, current_week_columns].to_numpy(dtype=float)
        season_excess[mask] = np.maximum(weekly_values - baseline, 0.0).sum(axis=1)
        playoff_indices = [weeks.index(week) for week in playoff_weeks]
        playoff_excess[mask] = np.maximum(
            weekly_values[:, playoff_indices] - baseline,
            0.0,
        ).sum(axis=1)
    paired["current_needle_mover_points"] = season_excess
    paired["current_playoff_excess_points"] = playoff_excess
    paired["current_breakout_rank_pct"] = _rank_percentile(
        paired,
        "current_needle_mover_points",
        ["pos", "origin_season"],
    )
    paired["current_playoff_rank_pct"] = _rank_percentile(
        paired,
        "current_playoff_excess_points",
        ["pos", "origin_season"],
    )
    paired["current_late_lift_rank_pct"] = _rank_percentile(
        paired,
        "current_late_lift",
        ["pos", "origin_season"],
    )
    paired["actual_next_performance_rank_pct"] = _rank_percentile(
        paired,
        "actual_next_unconditional_ppg",
        ["pos", "next_target_season"],
    )
    paired["current_breakout_hit"] = paired.current_breakout_rank_pct.ge(0.90).astype(int)
    paired["current_playoff_hit"] = paired.current_playoff_rank_pct.ge(0.90).astype(int)
    paired["current_late_surge_hit"] = paired.current_late_lift_rank_pct.ge(0.90).astype(int)
    paired["future_high_performer_hit"] = (
        paired.actual_next_performance_rank_pct.ge(0.90).astype(int)
    )
    paired["current_and_future_hit"] = (
        paired.current_breakout_hit.eq(1)
        & paired.future_high_performer_hit.eq(1)
    ).astype(int)
    paired["playoff_and_future_hit"] = (
        paired.current_playoff_hit.eq(1)
        & paired.future_high_performer_hit.eq(1)
    ).astype(int)

    donor_output_columns = [
        "profile_version",
        "league",
        "template_id",
        "player_key",
        "player",
        "pos",
        "team",
        "origin_season",
        "next_target_season",
        "avg_pick",
        "year_exp",
        "current_pred_ppg",
        "current_actual_ppg",
        "current_ppg_residual",
        "predicted_next_ppg",
        "breakout_signed_next_growth",
        "breakout_next_growth_rank_pct",
        "breakout_next_appearance",
        "actual_next_appeared",
        "actual_next_conditional_ppg",
        "actual_next_unconditional_ppg",
        "actual_next_ppg_residual",
        "current_season_points",
        "current_early_calendar_ppg",
        "current_late_calendar_ppg",
        "current_playoff_calendar_ppg",
        "current_late_lift",
        "current_needle_mover_points",
        "current_playoff_excess_points",
        "current_breakout_rank_pct",
        "current_playoff_rank_pct",
        "current_late_lift_rank_pct",
        "actual_next_performance_rank_pct",
        "current_breakout_hit",
        "current_playoff_hit",
        "current_late_surge_hit",
        "future_high_performer_hit",
        "current_and_future_hit",
        "playoff_and_future_hit",
        "training_through_origin",
        "target_outcome_through",
        "forecast_status",
        *sorted({feature for weights in MATCH_WEIGHTS.values() for feature in weights}),
        *current_week_columns,
    ]
    donor_output_columns = list(dict.fromkeys(donor_output_columns))
    audit = {
        "weekly_donor_rows": int(original_rows),
        "paired_donor_rows": int(len(paired)),
        "missing_forecast_rows": int((~forecast_available).sum()),
        "missing_outcome_rows": int((~outcome_available).sum()),
        "position_mismatch_rows": int(
            (forecast_available & outcome_available & ~position_consistent).sum()
        ),
        "invalid_appearance_rows": int(
            (outcome_available & ~appearance_contract).sum()
        ),
        "min_origin_season": int(paired.origin_season.min()),
        "max_origin_season": int(paired.origin_season.max()),
    }
    return paired[donor_output_columns].reset_index(drop=True), audit


def attach_current_next_context(targets: pd.DataFrame) -> pd.DataFrame:
    output = targets.copy()
    output["profile_version"] = PROFILE_VERSION
    output["breakout_signed_next_growth"] = (
        pd.to_numeric(output.pred_fp_per_game_ny, errors="coerce")
        - pd.to_numeric(output.pred_fp_per_game, errors="coerce")
    )
    output["breakout_next_appearance"] = pd.to_numeric(
        output.pred_appear_ny, errors="coerce"
    )
    output["breakout_next_growth_rank_pct"] = _rank_percentile(
        output,
        "breakout_signed_next_growth",
        ["pos", "year"],
    )
    return output


def select_breakout_pool(
    target: pd.Series,
    donors: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    position = str(target.pos)
    candidates = donors.loc[donors.pos.eq(position)].copy()
    if len(candidates) < MIN_POOL_SIZE:
        raise ValueError(
            f"{target.version} {position} has only {len(candidates)} paired donors"
        )
    weights = MATCH_WEIGHTS[position]
    candidates["template_distance"] = 0.0
    distance_columns = []
    for feature, weight in weights.items():
        distance_column = f"distance_{feature}"
        donor_values = pd.to_numeric(candidates[feature], errors="coerce").fillna(
            MATCH_FILL_VALUE
        )
        target_value = pd.to_numeric(
            pd.Series([target.get(feature, MATCH_FILL_VALUE)]),
            errors="coerce",
        ).fillna(MATCH_FILL_VALUE).iloc[0]
        candidates[distance_column] = (donor_values - target_value).abs()
        candidates["template_distance"] += (
            float(weight) * candidates[distance_column]
        )
        distance_columns.append(distance_column)
    candidates = candidates.sort_values(
        ["template_distance", "origin_season", "template_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(MAX_POOL_SIZE).copy()
    candidates["match_rank"] = np.arange(1, len(candidates) + 1)
    minimum_distance = float(candidates.template_distance.min())
    kernel = np.exp(
        -(candidates.template_distance.to_numpy(dtype=float) - minimum_distance)
        / KERNEL_BANDWIDTH[position]
    )
    local_probability = kernel / kernel.sum()
    local_fraction = max(
        MIN_LOCAL_WEIGHT,
        np.exp(-minimum_distance / LOCAL_DISTANCE_SCALE),
    )
    local_fraction = min(float(local_fraction), 1.0)
    probability = (
        local_fraction * local_probability
        + (1.0 - local_fraction) * np.full(len(candidates), 1.0 / len(candidates))
    )
    probability = _cap_probability_vector(probability, MAX_SAMPLE_PROBABILITY)
    candidates["template_season_gap"] = (
        int(target.year) - candidates.origin_season.astype(int)
    )
    if candidates.template_season_gap.le(0).any():
        raise ValueError("Breakout donors must precede the current target season")
    candidates["template_recency_multiplier"] = np.power(
        0.5,
        candidates.template_season_gap / RECENCY_HALF_LIFE,
    )
    probability = _cap_probability_vector(
        probability * candidates.template_recency_multiplier.to_numpy(dtype=float),
        MAX_SAMPLE_PROBABILITY,
    )
    candidates["template_sample_prob"] = probability
    candidates["template_sample_weight"] = kernel
    target_key = str(target.player_key)
    candidates["template_pool_key"] = (
        f"{PROFILE_VERSION}|{int(target.year)}|{target.version}|{dataset}|"
        f"{position}|{target_key}"
    )
    candidates["target_player_key"] = target_key
    candidates["target_player"] = str(target.player)
    candidates["target_pos"] = position
    candidates["target_year"] = int(target.year)
    candidates["target_league"] = str(target.version)
    candidates["target_dataset"] = dataset
    output_columns = [
        "profile_version",
        "template_pool_key",
        "target_player_key",
        "target_player",
        "target_pos",
        "target_year",
        "target_league",
        "target_dataset",
        "template_id",
        "league",
        "match_rank",
        "template_distance",
        "template_sample_weight",
        "template_sample_prob",
        "template_season_gap",
        "template_recency_multiplier",
        *distance_columns,
    ]
    return candidates[output_columns].reset_index(drop=True)


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame.template_sample_prob, errors="coerce")
    valid = values.notna() & weights.notna() & weights.ge(0)
    if not valid.any() or weights.loc[valid].sum() <= 0:
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def summarize_target(
    target: pd.Series,
    pool: pd.DataFrame,
    donors: pd.DataFrame,
    dataset: str,
) -> dict[str, object]:
    joined = pool.merge(
        donors,
        on=["profile_version", "league", "template_id"],
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(pool):
        raise ValueError("Breakout target pool lost donor rows during summary")
    summary_columns = (
        "current_ppg_residual",
        "current_needle_mover_points",
        "current_playoff_excess_points",
        "current_late_lift",
        "actual_next_appeared",
        "actual_next_unconditional_ppg",
        "actual_next_ppg_residual",
        "current_breakout_hit",
        "current_playoff_hit",
        "current_late_surge_hit",
        "future_high_performer_hit",
        "current_and_future_hit",
        "playoff_and_future_hit",
    )
    summary = {column: _weighted_mean(joined, column) for column in summary_columns}
    return {
        "profile_version": PROFILE_VERSION,
        "template_pool_key": str(pool.template_pool_key.iloc[0]),
        "player_key": str(target.player_key),
        "player": str(target.player),
        "pos": str(target.pos),
        "year": int(target.year),
        "league": str(target.version),
        "dataset": dataset,
        "is_keeper": int(target.is_keeper),
        "keeper_salary": (
            float(target.keeper_salary) if pd.notna(target.keeper_salary) else np.nan
        ),
        "pred_fp_per_game": float(target.pred_fp_per_game),
        "pred_fp_per_game_ny": float(target.pred_fp_per_game_ny),
        "pred_appear_ny": float(target.pred_appear_ny),
        "avg_pick": float(target.avg_pick) if pd.notna(target.avg_pick) else np.nan,
        "year_exp": float(target.year_exp) if pd.notna(target.year_exp) else np.nan,
        "breakout_signed_next_growth": float(target.breakout_signed_next_growth),
        "breakout_next_growth_rank_pct": float(target.breakout_next_growth_rank_pct),
        "template_pool_size": int(len(pool)),
        "effective_sample_size": float(
            1.0 / np.square(pool.template_sample_prob.to_numpy(dtype=float)).sum()
        ),
        "min_template_distance": float(pool.template_distance.min()),
        "median_template_distance": float(pool.template_distance.median()),
        "expected_current_ppg_residual": summary["current_ppg_residual"],
        "expected_current_needle_mover_points": summary[
            "current_needle_mover_points"
        ],
        "expected_playoff_excess_points": summary[
            "current_playoff_excess_points"
        ],
        "expected_late_season_lift": summary["current_late_lift"],
        "template_current_breakout_rate": summary["current_breakout_hit"],
        "template_playoff_hit_rate": summary["current_playoff_hit"],
        "template_late_surge_rate": summary["current_late_surge_hit"],
        "template_future_appearance_rate": summary["actual_next_appeared"],
        "expected_future_unconditional_ppg": summary[
            "actual_next_unconditional_ppg"
        ],
        "expected_future_ppg_residual_if_appeared": summary[
            "actual_next_ppg_residual"
        ],
        "template_future_high_performer_rate": summary[
            "future_high_performer_hit"
        ],
        "template_current_and_future_rate": summary["current_and_future_hit"],
        "template_playoff_and_future_rate": summary[
            "playoff_and_future_hit"
        ],
    }


def build_league_tables(
    connection: sqlite3.Connection,
    league: str,
    year: int,
    dataset: str,
    v2_database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    donors, weeks = load_weekly_donors(connection, league)
    targets = load_current_targets(connection, league, year, dataset)
    handoff, outcomes = load_next_year_context(v2_database, league)
    paired_donors, donor_audit = build_paired_donors(
        donors,
        handoff,
        outcomes,
        weeks,
        league,
    )
    targets = attach_current_next_context(targets)
    missing_target_context = targets[
        ["pred_fp_per_game", "pred_fp_per_game_ny", "pred_appear_ny"]
    ].isna().any(axis=1)
    if missing_target_context.any():
        labels = targets.loc[missing_target_context, "player"].head(10).tolist()
        raise ValueError(f"{league} current breakout context is incomplete: {labels}")
    pools = []
    summaries = []
    for _, target in targets.sort_values(["pos", "player_key"]).iterrows():
        pool = select_breakout_pool(target, paired_donors, dataset)
        pools.append(pool)
        summaries.append(summarize_target(target, pool, paired_donors, dataset))
    pool_frame = pd.concat(pools, ignore_index=True)
    player_map = pd.DataFrame(summaries)
    audit = {
        "profile_version": PROFILE_VERSION,
        "league": league,
        "year": int(year),
        "dataset": dataset,
        "current_target_rows": int(len(targets)),
        "keeper_target_rows": int(targets.is_keeper.sum()),
        "pool_rows": int(len(pool_frame)),
        "min_pool_size": int(pool_frame.groupby("template_pool_key").size().min()),
        "max_pool_size": int(pool_frame.groupby("template_pool_key").size().max()),
        "min_probability_sum": float(
            pool_frame.groupby("template_pool_key").template_sample_prob.sum().min()
        ),
        "max_probability_sum": float(
            pool_frame.groupby("template_pool_key").template_sample_prob.sum().max()
        ),
        "week_count": int(len(weeks)),
        "salary_match_feature_count": int(
            sum("salary" in column.lower() for column in pool_frame.columns)
        ),
        **donor_audit,
    }
    validate_league_tables(paired_donors, pool_frame, player_map, audit)
    return paired_donors, pool_frame, player_map, audit


def validate_league_tables(
    donors: pd.DataFrame,
    pools: pd.DataFrame,
    player_map: pd.DataFrame,
    audit: dict[str, object],
) -> None:
    if donors.duplicated(["profile_version", "league", "template_id"]).any():
        raise ValueError("Paired breakout donors are not unique")
    if pools.duplicated(["template_pool_key", "match_rank"]).any():
        raise ValueError("Paired breakout pools have duplicate match ranks")
    if player_map.duplicated(["profile_version", "league", "player_key"]).any():
        raise ValueError("Paired breakout player map has duplicate players")
    probability_sums = pools.groupby("template_pool_key").template_sample_prob.sum()
    if not np.allclose(probability_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
        raise ValueError("Paired breakout pool probabilities do not sum to one")
    if pools.template_sample_prob.max() > MAX_SAMPLE_PROBABILITY + 1e-10:
        raise ValueError("Paired breakout pool exceeds the donor probability cap")
    if pools.template_season_gap.le(0).any():
        raise ValueError("Paired breakout pool includes a future donor")
    if int(audit["salary_match_feature_count"]) != 0:
        raise ValueError("Salary leaked into paired breakout matching")
    no_appearance = donors.actual_next_appeared.eq(0)
    if donors.loc[no_appearance, "actual_next_conditional_ppg"].notna().any():
        raise ValueError("No-appearance paired donors contain future conditional PPG")
    if not donors.loc[no_appearance, "actual_next_unconditional_ppg"].eq(0).all():
        raise ValueError("No-appearance paired donors are not zero-valued")


def write_generated_tables(
    simulation_db: Path,
    templates: pd.DataFrame,
    pools: pd.DataFrame,
    player_map: pd.DataFrame,
    audits: pd.DataFrame,
) -> None:
    with closing(sqlite3.connect(simulation_db)) as connection:
        for table, frame in (
            (TEMPLATE_TABLE, templates),
            (POOL_TABLE, pools),
            (PLAYER_MAP_TABLE, player_map),
            (AUDIT_TABLE, audits),
        ):
            frame.to_sql(table, connection, if_exists="replace", index=False)
        connection.executescript(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_breakout_template_key
            ON {TEMPLATE_TABLE}(profile_version, league, template_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_breakout_pool_rank
            ON {POOL_TABLE}(template_pool_key, match_rank);
            CREATE INDEX IF NOT EXISTS idx_breakout_pool_target
            ON {POOL_TABLE}(target_year, target_league, target_dataset, target_player_key);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_breakout_player_key
            ON {PLAYER_MAP_TABLE}(profile_version, year, league, dataset, player_key);
            """
        )
        connection.commit()
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise ValueError(f"Simulation database integrity failed: {integrity}")


def validate_research_output(output: Path, simulation: Path) -> Path:
    """Keep research writes out of the input and production artifact paths."""

    output = output.resolve()
    protected_dirs = (
        REPO_ROOT / "Data" / "Databases",
        REPO_ROOT.parent / "Fantasy_Football_App" / "app",
        REPO_ROOT.parent / "Fantasy_Football_Snake" / "app",
    )
    if (
        output == simulation.resolve()
        or (output.exists() and output.samefile(simulation))
        or any(output.is_relative_to(path.resolve()) for path in protected_dirs)
    ):
        raise ValueError("Breakout output must be a separate research database.")
    return output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-db", type=Path, default=DEFAULT_SIMULATION_DB)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--leagues", nargs="+", choices=LEAGUES, default=list(LEAGUES))
    for league in LEAGUES:
        parser.add_argument(
            f"--{league}-v2-db",
            type=Path,
            default=V2_DATABASES[league],
        )
    parser.add_argument(
        "--output-db", type=Path, required=True,
        help="Separate research output database; never a production database.",
    )
    return parser.parse_args(argv)



def main() -> None:
    args = parse_args()
    simulation_db = args.simulation_db.resolve()
    if not simulation_db.is_file():
        raise FileNotFoundError(simulation_db)
    output_db = validate_research_output(args.output_db, simulation_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    all_templates = []
    all_pools = []
    all_player_maps = []
    audits = []
    v2_databases = {
        league: getattr(args, f"{league}_v2_db").resolve()
        for league in LEAGUES
    }
    with closing(sqlite3.connect(f"file:{simulation_db}?mode=ro", uri=True)) as connection:
        for league in args.leagues:
            templates, pools, player_map, audit = build_league_tables(
                connection,
                league,
                args.year,
                args.dataset,
                v2_databases[league],
            )
            all_templates.append(templates)
            all_pools.append(pools)
            all_player_maps.append(player_map)
            audits.append(audit)
    templates = pd.concat(all_templates, ignore_index=True)
    pools = pd.concat(all_pools, ignore_index=True)
    player_map = pd.concat(all_player_maps, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    write_generated_tables(
        output_db,
        templates,
        pools,
        player_map,
        audit_frame,
    )
    print(audit_frame.to_string(index=False))
    print(f"Research output: {output_db}")


if __name__ == "__main__":
    main()
