import numpy as np
import pandas as pd
import pytest

from Scripts.V2.build_feature_mart import (
    ADP_TRANSFORM_CHALLENGER_FEATURES,
    FEATURE_MART_FEATURES,
    HISTORY_GAP_CHALLENGER_FEATURES,
    LEGACY_RESIDUAL_CHALLENGER_FEATURES,
    PARTICIPATION_CANDIDATE_FEATURES,
    PROJECTION_RESEARCH_CHALLENGER_FEATURES,
    PROJECTION_TRAJECTORY_CHALLENGER_FEATURES,
    RESIDUAL_CANDIDATE_FEATURES,
    TEAM_ENVIRONMENT_CHALLENGER_FEATURES,
    TEMPLATE_CHALLENGER_FEATURES,
    add_adp_room_features,
    add_consensus_room_features,
    add_experience_context_features,
    add_history_gap_features,
    add_history_features,
    add_projection_trajectory_features,
    add_projection_shape_features,
    add_team_opportunity_share_features,
    build_feature_catalog,
    build_feature_manifests,
    build_projection_consensus,
)


def _outcome(season: int, ppg: float) -> dict[str, object]:
    games = 10
    points = games * ppg
    return {
        "player_key": "player",
        "season": season,
        "opportunity_games": games,
        "season_points": points,
        "conditional_ppg": ppg,
        "passing_points": 0.0,
        "rushing_points": 0.0,
        "receiving_points": points,
        "useful_season": 1,
    }


def _history_frame(seasons: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_key": "player",
                "season": season,
                "appeared": np.nan,
                "opportunity_games": np.nan,
                "team_normalized": "BUF",
                "expert_ppg_team_game_median": 10.0,
                "expert_ppg_active_median": np.nan,
                "conditional_ppg": np.nan,
            }
            for season in seasons
        ]
    )


def test_history_features_use_only_strictly_prior_seasons():
    frame = _history_frame([2024, 2026])
    outcomes = pd.DataFrame([_outcome(2023, 10), _outcome(2025, 50)])
    first = add_history_features(frame, outcomes).set_index("season")

    mutated = outcomes.copy()
    mutated.loc[mutated["season"].eq(2025), "conditional_ppg"] = 500
    mutated.loc[mutated["season"].eq(2025), "season_points"] = 5000
    second = add_history_features(frame, mutated).set_index("season")

    assert first.loc[2024, "career_weighted_ppg"] == 10
    assert second.loc[2024, "career_weighted_ppg"] == 10
    assert first.loc[2024, "last_observed_season"] == 2023
    assert first.loc[2026, "prior_year_ppg"] == 50
    assert second.loc[2026, "prior_year_ppg"] == 500


def test_missing_calendar_year_is_not_filled_as_prior_year():
    features = add_history_features(
        _history_frame([2025]),
        pd.DataFrame([_outcome(2023, 12)]),
    ).iloc[0]
    assert features["prior_year_outcome_observed"] == 0
    assert pd.isna(features["prior_year_ppg"])
    assert features["last_observed_ppg"] == 12
    assert features["seasons_since_observed"] == 2


def test_history_gaps_use_current_projection_as_neutral_missing_anchor():
    frame = _history_frame([2024, 2026])
    frame.loc[frame["season"].eq(2024), "expert_ppg_team_game_median"] = 5.0
    frame.loc[frame["season"].eq(2026), "expert_ppg_active_median"] = 16.0
    history = add_history_features(
        frame,
        pd.DataFrame([_outcome(2025, 12)]),
    )
    features = add_history_gap_features(history).set_index("season")

    assert features.loc[2024, "history_prior_year_ppg_gap"] == 0
    assert features.loc[2024, "history_career_ppg_gap"] == 0
    assert features.loc[2024, "history_prior_year_ppg_available"] == 0
    assert features.loc[2024, "history_prior_year_residual_neutral"] == 0
    assert features.loc[2024, "history_prior_year_opportunity_games_log"] == 0

    assert features.loc[2026, "history_prior_year_ppg_gap"] == -4
    assert features.loc[2026, "history_career_ppg_gap"] == -4
    assert features.loc[2026, "history_prior_year_ppg_available"] == 1
    assert features.loc[
        2026, "history_prior_year_ppg_gap_shrunk"
    ] == pytest.approx(-4 * 10 / 18)
    assert features.loc[
        2026, "history_prior_year_opportunity_games_log"
    ] == pytest.approx(np.log1p(10))


def test_projection_trajectory_uses_only_prior_preseason_projections():
    frame = pd.DataFrame(
        [
            {
                "player_key": "veteran",
                "season": season,
                "expert_ppg_team_game_median": projection,
            }
            for season, projection in (
                (2023, 8.0),
                (2024, 10.0),
                (2025, 16.0),
                (2026, 20.0),
            )
        ]
        + [
            {
                "player_key": "rookie",
                "season": 2026,
                "expert_ppg_team_game_median": 5.0,
            }
        ]
    )
    features = add_projection_trajectory_features(frame).set_index(
        ["player_key", "season"]
    )

    veteran = features.loc[("veteran", 2026)]
    assert veteran["projection_trajectory_change_1year"] == 4
    assert veteran["projection_trajectory_change_3year"] == pytest.approx(
        20 - ((3 * 16 + 2 * 10 + 1 * 8) / 6)
    )
    assert veteran["projection_trajectory_prior_year_available"] == 1
    assert veteran["projection_trajectory_prior_3year_count"] == 3
    assert veteran["projection_trajectory_prior_3year_std"] == pytest.approx(
        np.std([8, 10, 16])
    )

    rookie = features.loc[("rookie", 2026)]
    assert rookie["projection_trajectory_change_1year"] == 0
    assert rookie["projection_trajectory_change_3year"] == 0
    assert rookie["projection_trajectory_prior_year_available"] == 0
    assert rookie["projection_trajectory_prior_3year_count"] == 0
    assert rookie["projection_trajectory_prior_3year_std"] == 0


def test_consensus_room_features_preserve_nonconsecutive_row_indexes():
    frame = pd.DataFrame(
        [
            {
                "player_key": "wr1",
                "season": 2026,
                "team": "BUF",
                "position": "WR",
                "expert_points_median": 200.0,
                "expert_ppg_team_game_median": 12.0,
                "proj_passing_yards": np.nan,
                "proj_passing_tds": 0.0,
                "proj_rushing_yards": 20.0,
                "proj_rushing_tds": 0.0,
                "projected_rush_point_share": 0.02,
            },
            {
                "player_key": "wr2",
                "season": 2026,
                "team": "BUF",
                "position": "WR",
                "expert_points_median": 100.0,
                "expert_ppg_team_game_median": 6.0,
                "proj_passing_yards": np.nan,
                "proj_passing_tds": 0.0,
                "proj_rushing_yards": 10.0,
                "proj_rushing_tds": 0.0,
                "projected_rush_point_share": 0.01,
            },
            {
                "player_key": "qb1",
                "season": 2026,
                "team": "BUF",
                "position": "QB",
                "expert_points_median": 300.0,
                "expert_ppg_team_game_median": 18.0,
                "proj_passing_yards": 4000.0,
                "proj_passing_tds": 30.0,
                "proj_rushing_yards": 500.0,
                "proj_rushing_tds": 5.0,
                "projected_rush_point_share": 0.20,
            },
        ],
        index=[10, 20, 30],
    )
    features = add_consensus_room_features(frame)
    assert features.loc[10, "consensus_room_share"] == pytest.approx(2 / 3)
    assert features.loc[20, "consensus_room_rank"] == 2
    assert features.loc[20, "consensus_room_gap_to_leader"] == 100
    assert features.loc[10, "team_qb1_ppg"] == 18
    assert features.loc[10, "team_qb1_passing_tds"] == 30
    assert features.loc[10, "team_qb1_rushing_yards"] == 500
    assert features.loc[10, "team_qb1_rush_point_share"] == pytest.approx(
        0.20
    )
    assert features.loc[10, "team_core_skill_points"] == 300
    assert features.loc[10, "team_supporting_cast_points"] == 100
    assert features.loc[20, "team_supporting_cast_points"] == 200
    assert features.loc[30, "team_supporting_cast_points"] == 300
    assert features.loc[10, "team_projected_rushing_yards"] == 530
    assert features.loc[10, "team_projected_offensive_tds"] == 35
    assert features.loc[10, "pass_catcher_room_share"] == pytest.approx(2 / 3)


def test_experience_context_uses_self_excluded_same_season_peers():
    frame = pd.DataFrame(
        [
            {
                "player_key": "rookie_a",
                "season": 2026,
                "position": "WR",
                "year_exp": 0,
                "expert_ppg_team_game_median": 12.0,
            },
            {
                "player_key": "rookie_b",
                "season": 2026,
                "position": "WR",
                "year_exp": 0,
                "expert_ppg_team_game_median": 8.0,
            },
            {
                "player_key": "veteran",
                "season": 2026,
                "position": "WR",
                "year_exp": 10,
                "expert_ppg_team_game_median": 6.0,
            },
        ],
        index=[10, 20, 30],
    )
    features = add_experience_context_features(frame)
    assert features.loc[10, "expert_ppg_exp_peer_mean"] == 8
    assert features.loc[10, "expert_ppg_exp_diff"] == 4
    assert features.loc[10, "expert_ppg_exp_percentile"] == 1
    assert features.loc[20, "expert_ppg_exp_percentile"] == 0
    # The single 8+ veteran falls back to other same-position players.
    assert features.loc[30, "expert_ppg_exp_peer_mean"] == 10
    assert features.loc[30, "expert_ppg_exp_percentile"] == 0.5


def test_adp_room_features_exclude_the_player_from_teammates():
    frame = pd.DataFrame(
        [
            {
                "player_key": "wr1",
                "season": 2026,
                "team_normalized": "BUF",
                "position": "WR",
                "adp_median": 20.0,
            },
            {
                "player_key": "wr2",
                "season": 2026,
                "team_normalized": "BUF",
                "position": "WR",
                "adp_median": 50.0,
            },
            {
                "player_key": "wr3",
                "season": 2026,
                "team_normalized": "BUF",
                "position": "WR",
                "adp_median": 100.0,
            },
        ],
        index=[10, 20, 30],
    )
    features = add_adp_room_features(frame)
    assert features.loc[10, "adp_best_teammate_gap"] == -30
    assert features.loc[10, "adp_worst_teammate_gap"] == -80
    assert features.loc[20, "adp_mean_teammate_gap"] == -10
    assert features.loc[30, "adp_teammates_better_count"] == 2
    assert (
        features.loc[[10, 20, 30], "adp_room_strength_share"].sum()
        == pytest.approx(1)
    )


def test_team_opportunity_shares_use_preseason_team_totals():
    frame = pd.DataFrame(
        [
            {
                "team_normalized": "BUF",
                "season": 2026,
                "proj_targets": 100.0,
                "proj_receptions": 60.0,
                "proj_rush_attempts": 20.0,
                "proj_receiving_yards": 900.0,
            },
            {
                "team_normalized": "BUF",
                "season": 2026,
                "proj_targets": 50.0,
                "proj_receptions": 30.0,
                "proj_rush_attempts": 80.0,
                "proj_receiving_yards": 600.0,
            },
        ],
        index=[10, 20],
    )
    features = add_team_opportunity_share_features(frame)
    assert features.loc[10, "team_target_share"] == pytest.approx(2 / 3)
    assert features.loc[10, "team_reception_share"] == pytest.approx(2 / 3)
    assert features.loc[20, "team_rush_attempt_share"] == pytest.approx(0.8)
    assert features.loc[20, "team_receiving_yard_share"] == pytest.approx(0.4)


def test_projection_consensus_never_uses_provider_published_totals():
    rows = [
        {
            "player_key": "configured_player",
            "season": 2026,
            "provider": "configured",
            "configured_points_complete": 1,
            "configured_projected_points": 200.0,
            "provider_projected_points": 200.0,
            "configured_points_imputed_component_count": 0,
            "provider_points_per_team_game": 10.0,
            "provider_points_per_projected_game": 12.5,
        },
        {
            "player_key": "configured_player",
            "season": 2026,
            "provider": "fallback",
            "configured_points_complete": 0,
            "configured_projected_points": 0.0,
            "provider_projected_points": 900.0,
            "configured_points_imputed_component_count": 0,
            "provider_points_per_team_game": 45.0,
            "provider_points_per_projected_game": np.nan,
        },
        {
            "player_key": "fallback_only",
            "season": 2026,
            "provider": "fallback",
            "configured_points_complete": 0,
            "configured_projected_points": 0.0,
            "provider_projected_points": 160.0,
            "configured_points_imputed_component_count": 0,
            "provider_points_per_team_game": 8.0,
            "provider_points_per_projected_game": np.nan,
        },
    ]
    frame = pd.DataFrame(rows)
    metric_columns = {
        "projected_games",
        "source_uncertainty",
        "source_ceiling_points",
        "source_floor_points",
        "provider_room_share",
        "provider_room_rank",
        "provider_room_gap_to_leader",
        "provider_room_hhi",
        "provider_room_points",
        "provider_team_points",
        *{
            source
            for source in (
                "passing_yards",
                "passing_tds",
                "interceptions",
                "pass_attempts",
                "rush_attempts",
                "rushing_yards",
                "rushing_tds",
                "targets",
                "receptions",
                "receiving_yards",
                "receiving_tds",
                "passing_points",
                "rushing_points",
                "receiving_points",
            )
        },
    }
    for column in metric_columns:
        if column not in frame:
            frame[column] = np.nan
    consensus = build_projection_consensus(frame).set_index("player_key")
    assert (
        consensus.loc["configured_player", "expert_points_median"] == 200
    )
    assert (
        consensus.loc["configured_player", "expert_points_count"] == 1
    )
    assert pd.isna(consensus.loc["fallback_only", "expert_points_median"])
    assert consensus.loc["fallback_only", "expert_points_count"] == 0


def test_provider_specific_projection_requires_three_prior_seasons():
    rows = []
    for season in range(2022, 2026):
        rows.append(
            {
                "player_key": f"player_{season}",
                "season": season,
                "provider": "fantasydata",
                "configured_points_complete": 1,
                "configured_points_imputed_component_count": 0,
                "configured_projected_points": 170.0,
                "provider_projected_points": 170.0,
                "provider_points_per_team_game": 10.0,
                "provider_points_per_projected_game": np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    for column in {
        "projected_games",
        "source_uncertainty",
        "source_ceiling_points",
        "source_floor_points",
        "provider_room_share",
        "provider_room_rank",
        "provider_room_gap_to_leader",
        "provider_room_hhi",
        "provider_room_points",
        "provider_team_points",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "pass_attempts",
        "rush_attempts",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "passing_points",
        "rushing_points",
        "receiving_points",
    }:
        frame[column] = np.nan
    consensus = build_projection_consensus(frame).set_index("season")
    feature = "provider_fantasydata_ppg_team_game"
    assert consensus.loc[2022:2024, feature].isna().all()
    assert consensus.loc[2025, feature] == 10


def test_projection_shape_rates_require_observed_denominators():
    frame = pd.DataFrame(
        [
            {
                "proj_pass_attempts": 500.0,
                "proj_passing_yards": 4000.0,
                "proj_passing_tds": 30.0,
                "proj_interceptions": 10.0,
                "proj_rush_attempts": 100.0,
                "proj_rushing_yards": 450.0,
                "proj_rushing_tds": 5.0,
                "proj_targets": 80.0,
                "proj_receptions": 60.0,
                "proj_receiving_yards": 900.0,
                "proj_receiving_tds": 6.0,
            },
            {
                "proj_pass_attempts": 0.0,
                "proj_passing_yards": 0.0,
                "proj_passing_tds": 0.0,
                "proj_interceptions": 0.0,
                "proj_rush_attempts": np.nan,
                "proj_rushing_yards": np.nan,
                "proj_rushing_tds": np.nan,
                "proj_targets": 80.0,
                "proj_receptions": 60.0,
                "proj_receiving_yards": 900.0,
                "proj_receiving_tds": 6.0,
            },
        ]
    )
    features = add_projection_shape_features(frame)
    assert features.loc[0, "proj_total_touches"] == 160
    assert features.loc[0, "proj_total_opportunities"] == 180
    assert features.loc[0, "proj_pass_yards_per_attempt"] == 8
    assert features.loc[0, "proj_catch_rate"] == pytest.approx(0.75)
    assert pd.isna(features.loc[1, "proj_pass_yards_per_attempt"])
    assert pd.isna(features.loc[1, "proj_total_opportunities"])


def test_manifests_are_compact_and_template_budgets_cover_one_vote_per_family():
    assert len(FEATURE_MART_FEATURES) == 160
    candidate_features = sorted(
        RESIDUAL_CANDIDATE_FEATURES
        | LEGACY_RESIDUAL_CHALLENGER_FEATURES
        | PROJECTION_RESEARCH_CHALLENGER_FEATURES
        | HISTORY_GAP_CHALLENGER_FEATURES
        | PROJECTION_TRAJECTORY_CHALLENGER_FEATURES
        | ADP_TRANSFORM_CHALLENGER_FEATURES
        | TEAM_ENVIRONMENT_CHALLENGER_FEATURES
        | PARTICIPATION_CANDIDATE_FEATURES
        | TEMPLATE_CHALLENGER_FEATURES
    )
    frame = pd.DataFrame(
        {feature: [1.0, 2.0] for feature in candidate_features}
    )
    catalog = build_feature_catalog(
        frame,
        candidate_features,
        run_id="m3",
    )
    manifests = build_feature_manifests(catalog, run_id="m3")
    counts = manifests.groupby("manifest_name")["feature_name"].nunique()
    assert counts["residual_candidate_v1"] == 31
    assert counts["residual_legacy_challenger_v1"] == 12
    assert counts["residual_projection_challenger_v1"] == 26
    assert counts["residual_history_gap_challenger_v1"] == 13
    assert counts["residual_projection_trajectory_challenger_v1"] == 5
    assert counts["residual_adp_transform_challenger_v1"] == 1
    assert counts["residual_team_environment_challenger_v1"] == 11
    assert counts["participation_candidate_v1"] == 19
    assert counts["template_challenger_v1"] == 12

    template = manifests[
        manifests["manifest_name"].eq("template_challenger_v1")
    ]
    assert template.groupby("family")["family_weight_budget"].nunique().eq(1).all()
    assert (
        template.drop_duplicates("family")["family_weight_budget"].sum()
        == pytest.approx(1.0)
    )
