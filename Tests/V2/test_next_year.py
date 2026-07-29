import pandas as pd
import pytest

from Scripts.V2.next_year import (
    build_next_year_target_audit,
    build_next_year_targets,
)


def feature_rows():
    return pd.DataFrame(
        [
            {
                "player_key": "observed",
                "season": 2023,
                "position": "WR",
                "team": "AAA",
                "identity_status": "confirmed",
                "identity_is_confirmed": 1,
                "expert_ppg_team_game_median": 10.0,
            },
            {
                "player_key": "retired",
                "season": 2023,
                "position": "RB",
                "team": "BBB",
                "identity_status": "confirmed",
                "identity_is_confirmed": 1,
                "expert_ppg_team_game_median": 12.0,
            },
            {
                "player_key": "unresolved",
                "season": 2023,
                "position": "TE",
                "team": "CCC",
                "identity_status": "provisional",
                "identity_is_confirmed": 0,
                "expert_ppg_team_game_median": 6.0,
            },
            {
                "player_key": "future",
                "season": 2024,
                "position": "QB",
                "team": "DDD",
                "identity_status": "confirmed",
                "identity_is_confirmed": 1,
                "expert_ppg_team_game_median": 18.0,
            },
        ]
    )


def outcome_rows():
    return pd.DataFrame(
        [
            {
                "player_key": "observed",
                "season": 2024,
                "position": "WR",
                "appeared": 1,
                "conditional_ppg": 13.5,
                "opportunity_games": 12,
                "outcome_complete": 1,
                "target_available": 1,
            }
        ]
    )


def test_build_next_year_targets_preserves_hurdle_semantics():
    frame = build_next_year_targets(
        feature_rows(),
        outcome_rows(),
        completed_through_season=2024,
    ).set_index("player_key")

    assert frame.loc["observed", "next_appeared"] == 1
    assert frame.loc["observed", "next_conditional_ppg"] == 13.5
    assert frame.loc["observed", "next_residual_vs_expert"] == 3.5
    assert (
        frame.loc["observed", "next_conditional_ppg_training_eligible"]
        == 1
    )

    assert frame.loc["retired", "next_appeared"] == 0
    assert pd.isna(frame.loc["retired", "next_conditional_ppg"])
    assert pd.isna(frame.loc["retired", "next_residual_vs_expert"])
    assert frame.loc["retired", "next_target_join_status"] == "no_appearance"

    assert pd.isna(frame.loc["unresolved", "next_appeared"])
    assert (
        frame.loc["unresolved", "next_target_join_status"]
        == "unresolved_identity"
    )

    assert pd.isna(frame.loc["future", "next_appeared"])
    assert (
        frame.loc["future", "next_participation_target_available"] == 0
    )
    assert frame.loc["future", "next_target_join_status"] == "target_incomplete"


def test_next_year_target_audit_counts_disappearances():
    frame = build_next_year_targets(
        feature_rows(),
        outcome_rows(),
        completed_through_season=2024,
    )
    audit = build_next_year_target_audit(frame)
    row = audit[
        audit["origin_season"].eq(2023) & audit["position"].eq("RB")
    ].iloc[0]
    assert row["next_no_appearance_rows"] == 1
    assert row["conditional_ppg_rows"] == 0


def test_duplicate_following_outcomes_fail_closed():
    outcomes = pd.concat([outcome_rows(), outcome_rows()], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        build_next_year_targets(
            feature_rows(),
            outcomes,
            completed_through_season=2024,
        )
