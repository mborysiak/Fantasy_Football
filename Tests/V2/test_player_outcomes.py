import numpy as np
import pandas as pd

from Scripts.V2.build_player_outcomes import (
    WEEKLY_REQUIRED_COLUMNS,
    aggregate_player_outcomes,
    score_weekly_stats,
    validate_outcomes,
)
from Scripts.V2.contracts import stable_player_key


def _weekly_row(
    player_id,
    name,
    position,
    season,
    week,
    **overrides,
):
    row = {column: 0 for column in WEEKLY_REQUIRED_COLUMNS}
    row.update(
        {
            "player_id": player_id,
            "player_display_name": name,
            "position_group": position,
            "season": season,
            "week": week,
            "season_type": "REG",
            "game_id": f"{season}_{week:02d}_{player_id}",
            "team": "TST",
        }
    )
    row.update(overrides)
    return row


def _identity(*players):
    return pd.DataFrame(
        [
            {
                "player_key": stable_player_key(f"gsis:{player_id}"),
                "gsis_id": player_id,
            }
            for player_id in players
        ]
    )


def test_component_scoring_reconciles_to_total():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "wr-1",
                "Scoring Receiver",
                "WR",
                2023,
                1,
                carries=1,
                rushing_yards=100,
                rushing_tds=1,
                targets=10,
                receptions=10,
                receiving_yards=100,
                receiving_tds=1,
                fumbles_lost_total=1,
            )
        ]
    )
    scored = score_weekly_stats(weekly, "dk")
    components = scored[
        [
            "passing_points",
            "rushing_points",
            "receiving_points",
            "fumble_points",
            "two_point_points",
            "special_teams_points",
        ]
    ].sum(axis=1)
    assert scored.loc[0, "rushing_points"] == 19
    assert scored.loc[0, "receiving_points"] == 29
    assert scored.loc[0, "fumble_points"] == -1
    assert scored.loc[0, "fantasy_points_configured"] == 47
    assert np.allclose(components, scored["fantasy_points_configured"])


def test_all_valid_trey_mcbride_weeks_are_retained():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "te-1",
                "Trey McBride",
                "TE",
                2023,
                week,
                targets=1,
                receptions=1,
                receiving_yards=10,
            )
            for week in range(1, 10)
        ]
    )
    outcomes = aggregate_player_outcomes(
        weekly,
        _identity("te-1"),
        league="dk",
        run_id="test",
        completed_through_season=2025,
    )
    assert outcomes.loc[0, "opportunity_games"] == 9
    assert outcomes.loc[0, "season_points"] == 18
    assert outcomes.loc[0, "conditional_ppg"] == 2


def test_missing_calendar_season_is_not_filled_or_shifted():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "rb-1",
                "Gap Player",
                "RB",
                2017,
                1,
                carries=10,
                rushing_yards=50,
            ),
            _weekly_row(
                "rb-1",
                "Gap Player",
                "RB",
                2019,
                1,
                carries=10,
                rushing_yards=100,
            ),
        ]
    )
    outcomes = aggregate_player_outcomes(
        weekly,
        _identity("rb-1"),
        league="dk",
        run_id="test",
        completed_through_season=2025,
    )
    assert outcomes["season"].tolist() == [2017, 2019]
    assert 2018 not in set(outcomes["season"])
    assert outcomes.set_index("season").loc[2017, "conditional_ppg"] == 5
    assert outcomes.set_index("season").loc[2019, "conditional_ppg"] == 13


def test_small_sample_is_retained_but_not_marked_useful():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "wr-2",
                "Small Sample",
                "WR",
                2024,
                week,
                targets=1,
                receptions=1,
                receiving_yards=10,
            )
            for week in (1, 2)
        ]
    )
    outcomes = aggregate_player_outcomes(
        weekly,
        _identity("wr-2"),
        league="dk",
        run_id="test",
        completed_through_season=2025,
    )
    assert outcomes.loc[0, "opportunity_games"] == 2
    assert outcomes.loc[0, "appeared"] == 1
    assert outcomes.loc[0, "useful_season"] == 0
    assert outcomes.loc[0, "conditional_ppg"] == 2


def test_incomplete_season_never_exposes_training_target():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "wr-3",
                "Current Player",
                "WR",
                2026,
                1,
                targets=1,
                receptions=1,
                receiving_yards=10,
            )
        ]
    )
    outcomes = aggregate_player_outcomes(
        weekly,
        _identity("wr-3"),
        league="dk",
        run_id="test",
        completed_through_season=2025,
    )
    assert outcomes.loc[0, "target_available"] == 0
    assert pd.isna(outcomes.loc[0, "conditional_ppg"])


def test_qb_opportunity_threshold_is_explicit():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "qb-1",
                "Backup Quarterback",
                "QB",
                2024,
                1,
                attempts=10,
            ),
            _weekly_row(
                "qb-1",
                "Backup Quarterback",
                "QB",
                2024,
                2,
                attempts=16,
                passing_yards=100,
            ),
        ]
    )
    outcomes = aggregate_player_outcomes(
        weekly,
        _identity("qb-1"),
        league="dk",
        run_id="test",
        completed_through_season=2025,
    )
    assert outcomes.loc[0, "opportunity_games"] == 1
    validate_outcomes(outcomes)


def test_preseason_alias_recovers_offensive_stats_for_canonical_db():
    weekly = pd.DataFrame(
        [
            _weekly_row(
                "hybrid-1",
                "Two Way Player",
                "DB",
                2025,
                1,
                targets=5,
                receptions=4,
                receiving_yards=50,
            )
        ]
    )
    identity = _identity("hybrid-1")
    aliases = pd.DataFrame(
        [
            {
                "player_key": identity.loc[0, "player_key"],
                "source": "fantasydata",
                "position": "WR",
                "season": 2025,
            }
        ]
    )
    outcomes = aggregate_player_outcomes(
        weekly,
        identity,
        player_aliases=aliases,
        league="dk",
        run_id="test",
        completed_through_season=2025,
    )
    assert len(outcomes) == 1
    assert outcomes.loc[0, "position"] == "WR"
    assert outcomes.loc[0, "opportunity_games"] == 1
    assert outcomes.loc[0, "season_points"] == 9
