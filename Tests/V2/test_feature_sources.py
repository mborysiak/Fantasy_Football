import numpy as np
import pandas as pd
import pytest

from Scripts.V2.build_feature_sources import (
    _add_provider_room_context,
    _score_projection_values,
    resolve_source_rows,
)
from Scripts.V2.contracts import (
    PROJECTION_VALUE_METRICS,
    configured_scoring,
    normalize_source_position,
)


def _alias(
    player_key: str,
    source_player_id: object,
    name: str,
    position: str,
    team: str,
) -> dict[str, object]:
    return {
        "player_key": player_key,
        "source_table": "Fixture_Projections",
        "source": "fixture",
        "source_player_id": source_player_id,
        "normalized_name": name,
        "position": position,
        "team": team,
        "season": 2026,
    }


def test_rank_style_position_labels_are_normalized():
    assert normalize_source_position("RB-01") == "RB"
    assert normalize_source_position("wr 12") == "WR"
    assert normalize_source_position("DST") == "DST"


def test_source_rows_resolve_exact_aliases_and_reject_ambiguous_names():
    aliases = pd.DataFrame(
        [
            _alias("wr-a", "101", "same player", "WR", "BUF"),
            _alias("rb-a", "102", "same player", "RB", "NYJ"),
            _alias("te-a", pd.NA, "unique player", "TE", "KC"),
        ]
    )
    identity_rows = pd.DataFrame(
        [
            {
                "source_table": "Fixture_Projections",
                "source": "fixture",
                "source_player_id": "101",
                "normalized_name": "wrong display name",
                "position": "WR",
                "team": "BUF",
                "season": 2026,
            },
            {
                "source_table": "Fixture_Projections",
                "source": "fixture",
                "source_player_id": pd.NA,
                "normalized_name": "unique player",
                "position": "TE",
                "team": pd.NA,
                "season": 2026,
            },
            {
                "source_table": "Fixture_Projections",
                "source": "fixture",
                "source_player_id": pd.NA,
                "normalized_name": "same player",
                "position": pd.NA,
                "team": pd.NA,
                "season": 2026,
            },
        ]
    )
    resolved = resolve_source_rows(identity_rows, aliases)
    assert resolved.iloc[0] == "wr-a"
    assert resolved.iloc[1] == "te-a"
    assert pd.isna(resolved.iloc[2])


def _projection_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        metric: np.nan for metric in PROJECTION_VALUE_METRICS
    }
    row.update(
        {
            "player_key": "player",
            "season": 2026,
            "provider": "fixture",
            "position": "WR",
            "team": "BUF",
        }
    )
    row.update(updates)
    return row


def test_only_configured_scoring_is_used_for_provider_points():
    frame = pd.DataFrame(
        [
            _projection_row(
                receptions=80,
                receiving_yards=1000,
                receiving_tds=8,
                raw_projected_points=999,
                projected_games=16,
            ),
            _projection_row(
                player_key="fallback",
                receptions=np.nan,
                receiving_yards=500,
                receiving_tds=4,
                raw_projected_points=123,
                raw_projected_ppg=99,
            ),
        ]
    )
    scored = _score_projection_values(frame, "dk")
    receiving = configured_scoring("dk")["receiving"]
    expected = (
        80 * receiving["rec_complete_pass_sum"]
        + 1000 * receiving["rec_yards_gained_sum"]
        + 8 * receiving["rec_pass_touchdown_sum"]
    )
    assert scored.loc[0, "provider_projected_points"] == pytest.approx(expected)
    assert scored.loc[0, "points_method"] == "configured_components"
    assert pd.isna(scored.loc[1, "provider_projected_points"])
    assert pd.isna(scored.loc[1, "provider_points_per_projected_game"])
    assert scored.loc[1, "points_method"] == "insufficient"


def test_one_missing_required_component_uses_two_provider_median():
    frame = pd.DataFrame(
        [
            _projection_row(
                provider="provider_a",
                receptions=60,
                receiving_yards=900,
                receiving_tds=6,
            ),
            _projection_row(
                provider="provider_b",
                receptions=70,
                receiving_yards=950,
                receiving_tds=7,
            ),
            _projection_row(
                provider="provider_missing_receptions",
                receptions=np.nan,
                receiving_yards=1000,
                receiving_tds=8,
            ),
        ]
    )
    scored = _score_projection_values(frame, "dk")
    assert scored.loc[2, "receptions"] == 65
    assert scored.loc[2, "configured_points_complete"] == 1
    assert scored.loc[2, "configured_points_imputed_component_count"] == 1
    assert scored.loc[2, "points_method"] == "configured_components_imputed"


def test_provider_room_context_is_position_specific():
    frame = pd.DataFrame(
        [
            {
                "player_key": "wr1",
                "season": 2026,
                "provider": "fixture",
                "team": "BUF",
                "position": "WR",
                "configured_projected_points": 200.0,
                "configured_points_complete": 1,
                "provider_projected_points": 200.0,
            },
            {
                "player_key": "wr2",
                "season": 2026,
                "provider": "fixture",
                "team": "BUF",
                "position": "WR",
                "configured_projected_points": 100.0,
                "configured_points_complete": 1,
                "provider_projected_points": 100.0,
            },
            {
                "player_key": "qb1",
                "season": 2026,
                "provider": "fixture",
                "team": "BUF",
                "position": "QB",
                "configured_projected_points": 300.0,
                "configured_points_complete": 1,
                "provider_projected_points": 300.0,
            },
        ]
    )
    context = _add_provider_room_context(frame).set_index("player_key")
    assert context.loc["wr1", "provider_team_points"] == 600
    assert context.loc["wr1", "provider_room_points"] == 300
    assert context.loc["wr1", "provider_room_share"] == pytest.approx(2 / 3)
    assert context.loc["wr2", "provider_room_rank"] == 2
    assert context.loc["wr2", "provider_room_gap_to_leader"] == 100
    assert context.loc["wr1", "provider_room_hhi"] == pytest.approx(5 / 9)


def test_unstandardized_provider_total_is_excluded_from_room_context():
    frame = pd.DataFrame(
        [
            {
                "player_key": "configured",
                "season": 2026,
                "provider": "fixture",
                "team": "BUF",
                "position": "WR",
                "configured_projected_points": 200.0,
                "configured_points_complete": 1,
                "provider_projected_points": 200.0,
            },
            {
                "player_key": "fallback",
                "season": 2026,
                "provider": "fixture",
                "team": "BUF",
                "position": "WR",
                "configured_projected_points": 100.0,
                "configured_points_complete": 0,
                "provider_projected_points": 900.0,
            },
        ]
    )
    context = _add_provider_room_context(frame).set_index("player_key")
    assert context.loc["configured", "provider_team_points"] == 200
    assert context.loc["configured", "provider_room_share"] == 1
    assert pd.isna(context.loc["fallback", "provider_room_share"])
