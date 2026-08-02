import pandas as pd

from Scripts.V2.build_projection_spine import (
    build_player_season_sources,
    build_player_season_spine,
    validate_projection_spine,
)


def _identity() -> pd.DataFrame:
    rows = []
    for key, name, status, draft_year in (
        ("rookie", "Draft Only Rookie", "provisional", 2025),
        ("veteran", "Observed Veteran", "confirmed", 2020),
        ("small", "Small Sample", "confirmed", 2024),
        ("inactive", "Confirmed Inactive", "confirmed", 2022),
        ("future", "Future Rookie", "provisional", 2026),
        ("outcome-only", "Outcome Only", "confirmed", 2021),
        ("unknown", "Unknown Experience", "provisional", None),
    ):
        rows.append(
            {
                "player_key": key,
                "gsis_id": f"gsis-{key}" if status == "confirmed" else pd.NA,
                "display_name": name,
                "position": "WR",
                "identity_status": status,
                "identity_source": "fixture",
                "draft_year": draft_year,
                "rookie_season": draft_year,
            }
        )
    return pd.DataFrame(rows)


def _alias(
    player_key: str,
    source: str,
    season: int,
    name: str,
    team: str = "TST",
) -> dict[str, object]:
    return {
        "player_key": player_key,
        "source": source,
        "source_player_id": pd.NA,
        "source_name": name,
        "normalized_name": name.lower(),
        "position": "WR",
        "team": team,
        "season": season,
        "draft_year": season if source == "nfl_draft" else pd.NA,
        "match_method": "fixture",
    }


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_key": "veteran",
                "season": 2025,
                "league": "dk",
                "opportunity_games": 10,
                "season_points": 120.0,
                "conditional_ppg": 12.0,
                "appeared": 1,
                "useful_season": 1,
                "target_available": 1,
            },
            {
                "player_key": "small",
                "season": 2025,
                "league": "dk",
                "opportunity_games": 2,
                "season_points": 14.0,
                "conditional_ppg": 7.0,
                "appeared": 1,
                "useful_season": 0,
                "target_available": 1,
            },
            {
                "player_key": "outcome-only",
                "season": 2025,
                "league": "dk",
                "opportunity_games": 12,
                "season_points": 100.0,
                "conditional_ppg": 100 / 12,
                "appeared": 1,
                "useful_season": 1,
                "target_available": 1,
            },
        ]
    )


def test_source_spine_quarantines_only_fftoday_2018_qbs():
    identity = pd.DataFrame(
        [
            {"player_key": "excluded-qb", "position": "QB"},
            {"player_key": "included-rb", "position": "RB"},
            {"player_key": "included-qb", "position": "QB"},
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "player_key": player_key,
                "source": "fftoday",
                "source_table": "FFToday_Projections",
                "source_name": player_key,
                "position": position,
                "team": "TST",
                "season": season,
                "match_method": "fixture",
            }
            for player_key, position, season in (
                ("excluded-qb", "QB", 2018),
                ("included-rb", "RB", 2018),
                ("included-qb", "QB", 2019),
            )
        ]
    )

    sources = build_player_season_sources(
        aliases,
        identity,
        run_id="quarantine_fixture",
        start_season=2018,
        projection_through_season=2019,
    )

    assert set(
        sources[["player_key", "source_position", "season"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        ("included-rb", "RB", 2018),
        ("included-qb", "QB", 2019),
    }


def test_source_spine_keeps_fantasydata_historical_only():
    identity = pd.DataFrame(
        [
            {"player_key": "historical", "position": "WR"},
            {"player_key": "current", "position": "WR"},
        ]
    )
    aliases = pd.DataFrame(
        [
            _alias("historical", "fantasydata", 2025, "Historical Receiver"),
            _alias("current", "fantasydata", 2026, "Current Receiver"),
        ]
    )

    sources = build_player_season_sources(
        aliases,
        identity,
        run_id="fantasydata_historical_only_fixture",
        start_season=2025,
        projection_through_season=2026,
    )

    observed = set(
        sources[["player_key", "season", "source"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert observed == {("historical", 2025, "fantasydata")}


def _build() -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = _identity()
    aliases = pd.DataFrame(
        [
            _alias(
                "rookie",
                "nfl_draft",
                2025,
                "Draft Only Rookie",
            ),
            _alias(
                "veteran",
                "fftoday",
                2025,
                "Observed Veteran",
            ),
            _alias(
                "veteran",
                "adp_mfl",
                2025,
                "Observed Veteran",
            ),
            _alias("small", "fantasydata", 2025, "Small Sample"),
            _alias("inactive", "fantasydata", 2025, "Confirmed Inactive"),
            _alias("unknown", "fantasydata", 2025, "Unknown Experience"),
            _alias("future", "nfl_draft", 2026, "Future Rookie"),
            _alias("future", "fantasydata", 2026, "Future Rookie"),
        ]
    )
    sources = build_player_season_sources(
        aliases,
        identity,
        run_id="m2",
        start_season=2025,
        projection_through_season=2026,
    )
    spine = build_player_season_spine(
        sources,
        identity,
        _outcomes(),
        league="dk",
        run_id="m2",
        foundation_run_id="m1",
        completed_through_season=2025,
    )
    return sources, spine


def test_completed_confirmed_candidate_without_outcome_is_explicit_inactive():
    sources, spine = _build()
    inactive = spine.set_index("player_key").loc["inactive"]
    assert inactive["active_target_available"] == 1
    assert inactive["outcome_join_status"] == "no_opportunity"
    assert inactive["appeared"] == 0
    assert inactive["opportunity_games"] == 0
    assert inactive["unconditional_season_points"] == 0
    assert pd.isna(inactive["conditional_ppg"])
    validate_projection_spine(sources, spine)


def test_completed_unresolved_identity_has_unknown_participation():
    sources, spine = _build()
    rookie = spine.set_index("player_key").loc["rookie"]
    assert rookie["candidate_rule"] == "drafted_rookie_only"
    assert rookie["active_target_available"] == 0
    assert rookie["outcome_join_status"] == "unresolved_identity"
    assert pd.isna(rookie["appeared"])
    assert pd.isna(rookie["opportunity_games"])
    assert pd.isna(rookie["unconditional_season_points"])
    assert pd.isna(rookie["conditional_ppg"])
    validate_projection_spine(sources, spine)


def test_pending_candidate_has_no_outcome_labels():
    _, spine = _build()
    future = spine.set_index("player_key").loc["future"]
    assert future["candidate_rule"] == "drafted_rookie_only"
    assert future["is_rookie"] == 1
    assert future["active_target_available"] == 0
    assert future["outcome_join_status"] == "pending"
    assert pd.isna(future["appeared"])
    assert pd.isna(future["unconditional_season_points"])
    assert pd.isna(future["conditional_ppg"])


def test_small_observed_season_is_available_but_not_training_eligible():
    _, spine = _build()
    small = spine.set_index("player_key").loc["small"]
    assert small["conditional_ppg_target_available"] == 1
    assert small["conditional_ppg_training_eligible"] == 0
    assert small["conditional_ppg"] == 7


def test_outcomes_do_not_create_projection_candidates():
    sources, spine = _build()
    assert "outcome-only" not in set(spine["player_key"])
    veteran = spine.set_index("player_key").loc["veteran"]
    assert veteran["candidate_source_count"] == 2
    assert veteran["projection_source_count"] == 1
    assert veteran["market_source_count"] == 1
    assert len(sources[sources["player_key"].eq("veteran")]) == 2


def test_unknown_experience_is_not_encoded_as_veteran():
    _, spine = _build()
    unknown = spine.set_index("player_key").loc["unknown"]
    assert unknown["experience_known"] == 0
    assert pd.isna(unknown["year_exp"])
    assert pd.isna(unknown["is_rookie"])
    assert unknown["outcome_join_status"] == "unresolved_identity"
    assert unknown["active_target_available"] == 0
    assert pd.isna(unknown["appeared"])
