import pandas as pd

from Scripts.V2.build_player_identity import (
    _reconcile_provisional_identities,
    _resolve_candidate,
    canonicalize_nflverse_players,
    resolve_source_records,
)
from Scripts.V2.contracts import (
    PLAYER_ALIAS_COLUMNS,
    PLAYER_IDENTITY_COLUMNS,
    normalize_player_name,
)


def _players_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gsis_id": "00-0000001",
                "display_name": "Adrian Peterson",
                "position_group": "RB",
                "rookie_season": 2002,
                "last_season": 2009,
                "draft_year": 2002,
                "draft_round": 6,
                "draft_pick": 199,
                "draft_team": "CHI",
                "latest_team": "CHI",
            },
            {
                "gsis_id": "00-0000002",
                "display_name": "Adrian Peterson",
                "position_group": "RB",
                "rookie_season": 2007,
                "last_season": 2021,
                "draft_year": 2007,
                "draft_round": 1,
                "draft_pick": 7,
                "draft_team": "MIN",
                "latest_team": "SEA",
            },
            {
                "gsis_id": "00-0000003",
                "display_name": "Marvin Harrison Jr.",
                "position_group": "WR",
                "rookie_season": 2024,
                "last_season": 2026,
                "draft_year": 2024,
                "draft_round": 1,
                "draft_pick": 4,
                "draft_team": "ARI",
                "latest_team": "ARI",
            },
        ]
    )


def test_name_normalization_retains_identity_outside_match_key():
    assert normalize_player_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_player_name("Le'Veon Bell") == "leveon bell"


def test_same_name_players_resolve_by_draft_year():
    canonical = canonicalize_nflverse_players(_players_fixture())
    records = pd.DataFrame(
        [
            {
                "source": "nfl_draft",
                "source_player_id": pd.NA,
                "player": "Adrian Peterson",
                "normalized_name": "adrian peterson",
                "position": "RB",
                "team": "CHI",
                "season": 2002,
                "draft_year": 2002,
                "college": "Georgia Southern",
            },
            {
                "source": "nfl_draft",
                "source_player_id": pd.NA,
                "player": "Adrian Peterson",
                "normalized_name": "adrian peterson",
                "position": "RB",
                "team": "MIN",
                "season": 2007,
                "draft_year": 2007,
                "college": "Oklahoma",
            },
        ]
    )
    identity, aliases = resolve_source_records(canonical, records)
    adrian = identity[identity["display_name"].eq("Adrian Peterson")]
    assert len(adrian) == 2
    assert aliases["player_key"].nunique() == 2
    assert set(aliases["match_method"]) == {"name_position_draft_year"}


def test_provisional_rookie_key_is_promoted_without_changing():
    existing = pd.DataFrame(
        [
            {
                "player_key": "provisional-key",
                "gsis_id": pd.NA,
                "pfr_id": pd.NA,
                "pff_id": pd.NA,
                "espn_id": pd.NA,
                "nfl_id": pd.NA,
                "display_name": "Future Rookie",
                "normalized_name": "future rookie",
                "position": "WR",
                "birth_date": pd.NA,
                "college": "Example State",
                "draft_year": 2026,
                "draft_round": 2,
                "draft_pick": 40,
                "draft_team": "NYJ",
                "rookie_season": 2026,
                "last_season": pd.NA,
                "latest_team": "NYJ",
                "identity_status": "provisional",
                "identity_source": "nfl_draft",
            }
        ],
        columns=PLAYER_IDENTITY_COLUMNS,
    )
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0099999",
                "display_name": "Future Rookie",
                "position_group": "WR",
                "rookie_season": 2026,
                "last_season": 2026,
                "draft_year": 2026,
                "draft_round": 2,
                "draft_pick": 40,
                "draft_team": "NYJ",
                "latest_team": "NYJ",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players, existing)
    assert canonical.loc[0, "player_key"] == "provisional-key"
    assert canonical.loc[0, "identity_status"] == "confirmed"


def test_unmatched_preseason_player_gets_provisional_identity():
    canonical = canonicalize_nflverse_players(_players_fixture())
    records = pd.DataFrame(
        [
            {
                "source": "nfl_draft",
                "source_player_id": pd.NA,
                "player": "Never Played",
                "normalized_name": "never played",
                "position": "TE",
                "team": "BUF",
                "season": 2026,
                "draft_year": 2026,
                "college": "Example Tech",
            }
        ]
    )
    identity, aliases = resolve_source_records(canonical, records)
    provisional = identity[identity["display_name"].eq("Never Played")].iloc[0]
    assert provisional["identity_status"] == "provisional"
    assert pd.isna(provisional["gsis_id"])
    assert aliases.iloc[0]["player_key"] == provisional["player_key"]


def test_source_id_keeps_position_change_on_one_identity():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0011111",
                "display_name": "Hybrid Player",
                "position_group": "WR",
                "rookie_season": 2020,
                "last_season": 2026,
                "draft_year": 2020,
                "draft_round": 3,
                "draft_pick": 80,
                "draft_team": "NO",
                "latest_team": "NO",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "ffa",
                "source_player_id": "123",
                "player": "Hybrid Player",
                "normalized_name": "hybrid player",
                "position": "WR",
                "team": "NO",
                "season": 2021,
                "draft_year": 2020,
                "college": pd.NA,
            },
            {
                "source": "ffa",
                "source_player_id": "123",
                "player": "Hybrid Player",
                "normalized_name": "hybrid player",
                "position": "TE",
                "team": "NO",
                "season": 2022,
                "draft_year": 2020,
                "college": pd.NA,
            },
        ]
    )
    identity, aliases = resolve_source_records(canonical, records)
    assert len(identity) == 1
    assert aliases["player_key"].nunique() == 1
    assert set(aliases["match_method"]) == {"source_id_consensus"}


def test_unique_name_can_resolve_across_canonical_position_groups():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0022222",
                "display_name": "Converted Player",
                "position_group": "FB",
                "rookie_season": 2022,
                "last_season": 2026,
                "draft_year": 2022,
                "draft_round": 6,
                "draft_pick": 190,
                "draft_team": "PIT",
                "latest_team": "PIT",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(
        players,
        eligible_source_names={"converted player"},
    )
    records = pd.DataFrame(
        [
            {
                "source": "fantasydata",
                "source_player_id": pd.NA,
                "player": "Converted Player",
                "normalized_name": "converted player",
                "position": "TE",
                "team": "PIT",
                "season": 2025,
                "draft_year": 2022,
                "college": pd.NA,
            }
        ]
    )
    identity, aliases = resolve_source_records(canonical, records)
    assert len(identity) == 1
    assert identity.loc[0, "gsis_id"] == "00-0022222"
    assert aliases.loc[0, "player_key"] == identity.loc[0, "player_key"]
    assert aliases.loc[0, "match_method"] == "name_cross_position_unique"


def test_draft_source_enriches_confirmed_rookie_metadata():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0033333",
                "display_name": "Confirmed Rookie",
                "position_group": "WR",
                "rookie_season": 2026,
                "last_season": 2026,
                "draft_year": pd.NA,
                "draft_round": pd.NA,
                "draft_pick": pd.NA,
                "draft_team": pd.NA,
                "latest_team": "BUF",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "nfl_draft",
                "source_table": "Draft_Positions",
                "source_player_id": pd.NA,
                "player": "Confirmed Rookie",
                "normalized_name": "confirmed rookie",
                "position": "WR",
                "team": "BUF",
                "season": 2026,
                "draft_year": 2026,
                "draft_round": 2,
                "draft_pick": 45,
                "college": "Example State",
            }
        ]
    )
    identity, aliases = resolve_source_records(canonical, records)
    rookie = identity.iloc[0]
    assert rookie["gsis_id"] == "00-0033333"
    assert rookie["draft_year"] == 2026
    assert rookie["draft_round"] == 2
    assert rookie["draft_pick"] == 45
    assert rookie["draft_team"] == "BUF"
    assert rookie["college"] == "Example State"
    assert aliases.iloc[0]["player_key"] == rookie["player_key"]


def test_inferred_draft_year_does_not_override_active_career_window():
    identity = pd.DataFrame(
        [
            {
                "normalized_name": "frank gore",
                "position": "RB",
                "draft_year": 2005,
                "rookie_season": 2005,
                "last_season": 2020,
                "draft_team": "SF",
                "latest_team": "NYJ",
            },
            {
                "normalized_name": "frank gore",
                "position": "RB",
                "draft_year": pd.NA,
                "rookie_season": 2024,
                "last_season": 2026,
                "draft_team": pd.NA,
                "latest_team": "BUF",
            },
        ]
    )
    inferred_record = pd.Series(
        {
            "draft_year": 2005,
            "_draft_year_inferred": True,
            "season": 2025,
            "team": "BUF",
        }
    )
    actual_draft_record = pd.Series(
        {
            "draft_year": 2005,
            "_draft_year_inferred": False,
            "season": 2005,
            "team": "SF",
        }
    )

    inferred_index, inferred_method = _resolve_candidate(
        inferred_record, identity, identity.index
    )
    draft_index, draft_method = _resolve_candidate(
        actual_draft_record, identity, identity.index
    )

    assert inferred_index == 1
    assert inferred_method == "name_position_active_window"
    assert draft_index == 0
    assert draft_method == "name_position_draft_year"


def test_redundant_provisional_identity_is_merged_into_confirmed_player():
    confirmed = {
        column: pd.NA for column in PLAYER_IDENTITY_COLUMNS
    }
    confirmed.update(
        {
            "player_key": "confirmed",
            "gsis_id": "00-0012345",
            "display_name": "Michael Thomas",
            "normalized_name": "michael thomas",
            "position": "WR",
            "draft_year": 2016,
            "rookie_season": 2016,
            "identity_status": "confirmed",
        }
    )
    provisional = confirmed.copy()
    provisional.update(
        {
            "player_key": "provisional",
            "gsis_id": pd.NA,
            "identity_status": "provisional",
        }
    )
    alias = {column: pd.NA for column in PLAYER_ALIAS_COLUMNS}
    alias.update(
        {
            "player_key": "provisional",
            "source": "adp_mfl",
            "source_name": "Michael Thomas",
            "normalized_name": "michael thomas",
            "position": "WR",
            "season": 2020,
            "draft_year": 2016,
        }
    )

    identity, aliases = _reconcile_provisional_identities(
        pd.DataFrame([confirmed, provisional]),
        pd.DataFrame([alias]),
    )

    assert identity["player_key"].tolist() == ["confirmed"]
    assert aliases["player_key"].tolist() == ["confirmed"]
