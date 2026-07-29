"""Leakage-safe following-season targets and compact V2 feature locks.

An origin-season row represents information available before season ``t``.
The targets represent outcomes from season ``t + 1``.  The conditional point
model predicts a residual around the origin-season expert consensus:

    next conditional PPG - origin expert team-game PPG

No following-season preseason evidence is joined into the origin row.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import pandas as pd

from Scripts.V2.config import POSITIONS
from Scripts.V2.contracts import require_columns
from Scripts.V2.modeling import POSITION_FEATURES


NEXT_YEAR_TARGET_VERSION = "v2_next_year_expert_residual_v1"
NEXT_YEAR_BASELINE_COLUMN = "expert_ppg_team_game_median"

NEXT_YEAR_RESIDUAL_FEATURES = (
    "expert_ppg_team_game_median",
    "expert_ppg_team_game_std",
    "projection_provider_count",
    "proj_games",
    "adp_log",
    "projection_adp_percentile_diff",
    "age",
    "year_exp",
    "is_rookie",
    "draft_pick_log",
    "history_career_ppg_gap",
    "history_prior_year_ppg_gap",
    "history_prior_3year_ppg_gap",
    "history_career_opportunity_games_log",
    "history_prior_year_opportunity_games_log",
    "history_prior_3year_opportunity_games_log",
    "history_prior_year_ppg_available",
    "history_prior_3year_ppg_available",
    "projection_trajectory_change_1year",
    "projection_trajectory_change_3year",
    "projection_trajectory_prior_3year_count",
    "projection_trajectory_prior_3year_std",
    "projection_trajectory_prior_year_available",
    "projected_pass_point_share",
    "projected_rush_point_share",
    "projected_receiving_point_share",
    "consensus_room_share",
    "room_gap_to_leader_median",
    "team_changed_from_prior_candidate",
    *POSITION_FEATURES,
)

NEXT_YEAR_PARTICIPATION_FEATURES = (
    "expert_ppg_team_game_median",
    "projection_provider_count",
    "proj_games",
    "adp_log",
    "age",
    "year_exp",
    "is_rookie",
    "draft_pick_log",
    "career_observed_seasons",
    "career_opportunity_games",
    "career_useful_seasons",
    "last_observed_opportunity_games",
    "prior_year_appeared",
    "prior_year_candidate",
    "prior_year_opportunity_games",
    "seasons_since_observed",
    "team_changed_from_prior_candidate",
    *POSITION_FEATURES,
)


def feature_hash(features: Sequence[str]) -> str:
    payload = json.dumps(list(features), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _confirmed_identity(frame: pd.DataFrame) -> pd.Series:
    if "identity_is_confirmed" in frame:
        return (
            pd.to_numeric(frame["identity_is_confirmed"], errors="coerce")
            .fillna(0)
            .eq(1)
        )
    return frame["identity_status"].astype(str).str.lower().eq("confirmed")


def build_next_year_targets(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    completed_through_season: int,
) -> pd.DataFrame:
    """Attach following-season outcomes to origin-season feature rows.

    A confirmed canonical identity with no following-season outcome row is a
    true participation zero once that target season is complete.  A missing
    outcome for an unresolved/provisional identity remains unlabeled.
    """

    require_columns(
        features,
        (
            "player_key",
            "season",
            "position",
            "team",
            "identity_status",
            NEXT_YEAR_BASELINE_COLUMN,
        ),
        "player_season_features",
    )
    require_columns(
        outcomes,
        (
            "player_key",
            "season",
            "position",
            "appeared",
            "conditional_ppg",
            "outcome_complete",
            "target_available",
        ),
        "player_season_outcomes",
    )
    origin = features.copy()
    origin["season"] = pd.to_numeric(origin["season"], errors="raise").astype(
        int
    )
    origin = origin[origin["position"].isin(POSITIONS)].copy()
    if origin.duplicated(["player_key", "season"]).any():
        raise ValueError("Origin features are not unique by player-season")
    origin.rename(columns={"season": "origin_season"}, inplace=True)
    origin["target_season"] = origin["origin_season"] + 1
    origin["origin_identity_confirmed"] = _confirmed_identity(origin).astype(
        int
    )
    origin["origin_expert_ppg"] = pd.to_numeric(
        origin[NEXT_YEAR_BASELINE_COLUMN], errors="coerce"
    )

    target = outcomes.loc[
        :,
        [
            "player_key",
            "season",
            "position",
            "appeared",
            "conditional_ppg",
            "opportunity_games",
            "outcome_complete",
            "target_available",
        ],
    ].copy()
    target["season"] = pd.to_numeric(
        target["season"], errors="raise"
    ).astype(int)
    if target.duplicated(["player_key", "season"]).any():
        raise ValueError("Following-season outcomes are not unique")
    target.rename(
        columns={
            "season": "target_season",
            "position": "next_observed_position",
            "appeared": "observed_next_appeared",
            "conditional_ppg": "observed_next_conditional_ppg",
            "opportunity_games": "observed_next_opportunity_games",
            "outcome_complete": "observed_next_outcome_complete",
            "target_available": "observed_next_target_available",
        },
        inplace=True,
    )
    frame = origin.merge(
        target,
        on=["player_key", "target_season"],
        how="left",
        validate="one_to_one",
    )

    target_complete = frame["target_season"].le(
        int(completed_through_season)
    )
    observed = (
        pd.to_numeric(frame["observed_next_appeared"], errors="coerce")
        .fillna(0)
        .eq(1)
    )
    labelable_zero = (
        target_complete
        & frame["origin_identity_confirmed"].eq(1)
        & ~observed
    )
    participation_available = target_complete & (
        observed | labelable_zero
    )
    frame["next_participation_target_available"] = (
        participation_available.astype(int)
    )
    frame["next_appeared"] = np.nan
    frame.loc[observed & target_complete, "next_appeared"] = 1.0
    frame.loc[labelable_zero, "next_appeared"] = 0.0

    frame["next_conditional_ppg"] = np.where(
        frame["next_appeared"].eq(1),
        pd.to_numeric(
            frame["observed_next_conditional_ppg"], errors="coerce"
        ),
        np.nan,
    )
    frame["next_conditional_ppg_training_eligible"] = (
        frame["next_appeared"].eq(1)
        & frame["next_conditional_ppg"].notna()
        & frame["origin_expert_ppg"].notna()
    ).astype(int)
    frame["next_residual_vs_expert"] = (
        frame["next_conditional_ppg"] - frame["origin_expert_ppg"]
    )
    frame["next_target_join_status"] = "target_incomplete"
    frame.loc[
        target_complete
        & frame["origin_identity_confirmed"].ne(1)
        & ~observed,
        "next_target_join_status",
    ] = "unresolved_identity"
    frame.loc[labelable_zero, "next_target_join_status"] = "no_appearance"
    frame.loc[
        observed & target_complete, "next_target_join_status"
    ] = "observed_appearance"
    frame["next_target_version"] = NEXT_YEAR_TARGET_VERSION
    frame["next_target_horizon"] = 1

    validate_next_year_targets(frame, completed_through_season)
    frame.sort_values(["origin_season", "player_key"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def validate_next_year_targets(
    frame: pd.DataFrame,
    completed_through_season: int,
) -> None:
    require_columns(
        frame,
        (
            "player_key",
            "origin_season",
            "target_season",
            "origin_expert_ppg",
            "next_appeared",
            "next_conditional_ppg",
            "next_residual_vs_expert",
            "next_participation_target_available",
            "next_conditional_ppg_training_eligible",
            "next_target_join_status",
        ),
        "next_year_targets",
    )
    if frame.duplicated(["player_key", "origin_season"]).any():
        raise ValueError("Next-year targets are not unique by origin key")
    if not (
        frame["target_season"].astype(int)
        == frame["origin_season"].astype(int) + 1
    ).all():
        raise ValueError("Next-year target season is not origin plus one")
    future = frame["target_season"].gt(int(completed_through_season))
    if frame.loc[future, "next_participation_target_available"].ne(0).any():
        raise ValueError("Incomplete following seasons have available targets")
    if frame.loc[future, "next_appeared"].notna().any():
        raise ValueError("Incomplete following seasons have participation labels")
    no_appearance = frame["next_appeared"].eq(0)
    if frame.loc[no_appearance, "next_conditional_ppg"].notna().any():
        raise ValueError("No-appearance rows contain conditional PPG")
    unresolved = frame["next_target_join_status"].eq(
        "unresolved_identity"
    )
    if frame.loc[unresolved, "next_appeared"].notna().any():
        raise ValueError("Unresolved identities were converted to negatives")
    eligible = frame["next_conditional_ppg_training_eligible"].eq(1)
    reconstructed = (
        frame.loc[eligible, "origin_expert_ppg"]
        + frame.loc[eligible, "next_residual_vs_expert"]
    )
    error = (
        reconstructed - frame.loc[eligible, "next_conditional_ppg"]
    ).abs()
    if len(error) and float(error.max()) > 1e-10:
        raise ValueError("Next-year residual does not reconstruct its target")
    invalid_eligible = eligible & (
        frame["next_appeared"].ne(1)
        | frame["origin_expert_ppg"].isna()
        | frame["next_conditional_ppg"].isna()
    )
    if invalid_eligible.any():
        raise ValueError("Invalid conditional next-year training eligibility")


def build_next_year_target_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (season, position), group in frame.groupby(
        ["origin_season", "position"], sort=True
    ):
        eligible = group["next_conditional_ppg_training_eligible"].eq(1)
        rows.append(
            {
                "origin_season": int(season),
                "position": str(position),
                "origin_rows": len(group),
                "expert_baseline_rows": int(
                    group["origin_expert_ppg"].notna().sum()
                ),
                "participation_labeled_rows": int(
                    group["next_participation_target_available"].sum()
                ),
                "next_appearance_rows": int(
                    group["next_appeared"].eq(1).sum()
                ),
                "next_no_appearance_rows": int(
                    group["next_appeared"].eq(0).sum()
                ),
                "unresolved_identity_rows": int(
                    group["next_target_join_status"]
                    .eq("unresolved_identity")
                    .sum()
                ),
                "conditional_ppg_rows": int(eligible.sum()),
                "mean_next_residual": (
                    float(group.loc[eligible, "next_residual_vs_expert"].mean())
                    if eligible.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
