import sqlite3

import pandas as pd
import pytest

from Scripts.V2.build_player_identity import (
    _governed_match_name,
    _reconcile_provisional_identities,
    _resolve_candidate,
    canonicalize_nflverse_players,
    load_identity_source_records,
    resolve_source_records,
)
from Scripts.V2.build_projection_spine import build_player_season_sources
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


def test_fftoday_2018_qb_quarantine_reaches_identity_lineage_and_manifest(
    tmp_path,
):
    raw = pd.DataFrame(
        [
            {
                "player": "Corrupt Quarterback",
                "pos": "QB",
                "team": "LAR",
                "year": 2018,
            },
            {
                "player": "Unaffected Running Back",
                "pos": "RB",
                "team": "BUF",
                "year": 2018,
            },
            {
                "player": "Unaffected Quarterback",
                "pos": "QB",
                "team": "KC",
                "year": 2019,
            },
        ]
    )
    source_database = tmp_path / "identity_quarantine.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FFToday_Projections", connection, index=False)

    records, manifest = load_identity_source_records(source_database)

    assert set(
        records[["normalized_name", "position", "season"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        ("unaffected running back", "RB", 2018),
        ("unaffected quarterback", "QB", 2019),
    }

    quarantine = manifest[manifest["component"].eq("identity_quarantine")]
    assert len(quarantine) == 1
    assert quarantine.iloc[0]["source_name"] == (
        "fftoday_qb_stored_2018_2019_vintage_quarantine_v1"
    )
    assert quarantine.iloc[0]["source_uri"] == (
        "https://www.fftoday.com/rankings/playerproj.php"
        "?Season=2019&PosID=10"
    )
    assert quarantine.iloc[0]["row_count"] == 1

    players = pd.DataFrame(
        [
            {
                "gsis_id": f"fixture-{name}",
                "display_name": name,
                "position_group": position,
                "rookie_season": rookie_season,
                "last_season": 2020,
                "draft_year": rookie_season,
                "draft_round": pd.NA,
                "draft_pick": pd.NA,
                "draft_team": team,
                "latest_team": team,
            }
            for name, position, rookie_season, team in (
                ("Corrupt Quarterback", "QB", 2017, "LAR"),
                ("Unaffected Running Back", "RB", 2017, "BUF"),
                ("Unaffected Quarterback", "QB", 2018, "KC"),
            )
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    _, aliases = resolve_source_records(canonical, records)
    assert set(
        aliases[["normalized_name", "position", "season"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        ("unaffected running back", "RB", 2018),
        ("unaffected quarterback", "QB", 2019),
    }

    unfiltered_records = pd.DataFrame(
        [
            {
                "source": "fftoday",
                "source_table": "FFToday_Projections",
                "source_player_id": pd.NA,
                "player": row.player,
                "normalized_name": normalize_player_name(row.player),
                "position": row.pos,
                "team": row.team,
                "season": row.year,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
            for row in raw.itertuples(index=False)
        ]
    )
    _, fail_safe_aliases = resolve_source_records(
        canonical,
        unfiltered_records,
    )
    assert not fail_safe_aliases["normalized_name"].eq(
        "corrupt quarterback"
    ).any()
    assert len(fail_safe_aliases) == 2


def test_identity_loader_collapses_governed_team_only_duplicates(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "ffa_id": "hopkins",
                "player": "DeAndre Hopkins",
                "position": "WR",
                "team": team,
                "year": 2019,
            }
            for team in ("HOU", "ARI")
        ]
    )
    source_database = tmp_path / "identity_team_policy.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FFA_RawStats", connection, index=False)
    table_specs = {
        "FFA_RawStats": {
            "source": "ffa_raw",
            "source_kind": "projection",
            "source_player_id": "ffa_id",
            "player": "player",
            "position": "position",
            "team": "team",
            "season": "year",
        }
    }

    records, manifest = load_identity_source_records(
        source_database,
        table_specs=table_specs,
    )

    assert len(records) == 1
    record = records.iloc[0]
    assert record["source"] == "ffa_raw"
    assert record["normalized_name"] == "deandre hopkins"
    assert record["season"] == 2019
    assert pd.isna(record["team"])
    assert manifest.iloc[0]["row_count"] == 2

    canonical = canonicalize_nflverse_players(
        pd.DataFrame(
            [
                {
                    "gsis_id": "00-hopkins",
                    "display_name": "DeAndre Hopkins",
                    "position_group": "WR",
                    "rookie_season": 2013,
                    "last_season": 2025,
                    "draft_year": 2013,
                    "draft_round": 1,
                    "draft_pick": 27,
                    "draft_team": "HOU",
                    "latest_team": "BAL",
                }
            ]
        )
    )
    _, aliases = resolve_source_records(canonical, records)

    assert len(aliases) == 1
    assert aliases.iloc[0]["player_key"] == canonical.iloc[0]["player_key"]
    assert aliases.iloc[0]["source"] == "ffa_raw"
    assert pd.isna(aliases.iloc[0]["team"])


def test_identity_loader_canonicalizes_trusted_team_aliases(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "player": "Trevor Lawrence",
                "pos": "QB",
                "team": team,
                "year": 2023,
            }
            for team in ("JAC", "JAX")
        ]
    )
    source_database = tmp_path / "identity_team_aliases.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FFToday_Projections", connection, index=False)
    table_specs = {
        "FFToday_Projections": {
            "source": "fftoday",
            "source_kind": "projection",
            "player": "player",
            "position": "pos",
            "team": "team",
            "season": "year",
        }
    }

    records, _ = load_identity_source_records(
        source_database,
        table_specs=table_specs,
    )

    assert len(records) == 1
    assert records.iloc[0]["team"] == "JAC"

    canonical = canonicalize_nflverse_players(
        pd.DataFrame(
            [
                {
                    "gsis_id": "00-trevor",
                    "display_name": "Trevor Lawrence",
                    "position_group": "QB",
                    "rookie_season": 2021,
                    "last_season": 2025,
                    "draft_year": 2021,
                    "draft_round": 1,
                    "draft_pick": 1,
                    "draft_team": "JAX",
                    "latest_team": "JAC",
                }
            ]
        )
    )
    _, aliases = resolve_source_records(canonical, records)

    assert len(aliases) == 1
    assert aliases.iloc[0]["team"] == "JAC"


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


def test_unique_name_rejects_incompatible_draft_and_career_window():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0042001",
                "display_name": "Chris Brazzell II",
                "position_group": "WR",
                "rookie_season": 2026,
                "last_season": 2026,
                "draft_year": pd.NA,
                "draft_round": pd.NA,
                "draft_pick": pd.NA,
                "draft_team": pd.NA,
                "latest_team": "CAR",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "nfl_draft",
                "source_player_id": pd.NA,
                "player": "Chris Brazzell",
                "normalized_name": "chris brazzell",
                "position": "WR",
                "team": "NYJ",
                "season": 1998,
                "draft_year": 1998,
                "_draft_year_inferred": False,
                "draft_round": 6,
                "draft_pick": 174,
                "college": "Angelo State",
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    confirmed = identity[identity["identity_status"].eq("confirmed")].iloc[0]
    old_namesake = identity[identity["identity_status"].eq("provisional")].iloc[0]
    assert pd.isna(confirmed["draft_year"])
    assert old_namesake["draft_year"] == 1998
    assert aliases.loc[0, "player_key"] == old_namesake["player_key"]
    assert aliases.loc[0, "match_method"] == "provisional_incompatible"


def test_team_aliases_are_normalized_before_disambiguation():
    identity = pd.DataFrame(
        [
            {
                "normalized_name": "same player",
                "position": "WR",
                "draft_year": pd.NA,
                "rookie_season": pd.NA,
                "last_season": pd.NA,
                "draft_team": "JAC",
                "latest_team": "JAC",
            },
            {
                "normalized_name": "same player",
                "position": "WR",
                "draft_year": pd.NA,
                "rookie_season": pd.NA,
                "last_season": pd.NA,
                "draft_team": "GNB",
                "latest_team": "GNB",
            },
        ]
    )
    record = pd.Series(
        {
            "draft_year": pd.NA,
            "_draft_year_inferred": False,
            "season": pd.NA,
            "team": "JAX",
        }
    )

    candidate_index, method = _resolve_candidate(
        record,
        identity,
        identity.index,
    )

    assert candidate_index == 0
    assert method == "name_position_team"


def test_single_compatible_candidate_resolves_after_filtering():
    identity = pd.DataFrame(
        [
            {
                "normalized_name": "same player",
                "position": "WR",
                "draft_year": 2000,
                "rookie_season": 2000,
                "last_season": 2010,
                "draft_team": "BUF",
                "latest_team": "BUF",
            },
            {
                "normalized_name": "same player",
                "position": "WR",
                "draft_year": 2001,
                "rookie_season": 2001,
                "last_season": 2011,
                "draft_team": "DET",
                "latest_team": "DET",
            },
        ]
    )
    record = pd.Series(
        {
            "draft_year": pd.NA,
            "_draft_year_inferred": False,
            "season": 2012,
            "team": pd.NA,
        }
    )

    candidate_index, method = _resolve_candidate(
        record,
        identity,
        identity.index,
    )

    assert candidate_index == 1
    assert method == "name_position_active_window"


def test_governed_tet_key_wins_when_confirmed_and_provisional_both_exist():
    production_key = "c16a5e67-fff0-57b9-838c-c8df91df7b9d"
    existing = pd.DataFrame(
        [
            {
                "player_key": "0ed2c7a0-9a0d-5c97-850e-cc1077466d27",
                "gsis_id": "00-0040124",
                "pfr_id": pd.NA,
                "pff_id": pd.NA,
                "espn_id": pd.NA,
                "nfl_id": pd.NA,
                "display_name": "Tetairoa McMillan",
                "normalized_name": "tetairoa mcmillan",
                "position": "WR",
                "birth_date": pd.NA,
                "college": "Arizona",
                "draft_year": 2025,
                "draft_round": 1,
                "draft_pick": 8,
                "draft_team": "CAR",
                "rookie_season": 2025,
                "last_season": 2026,
                "latest_team": "CAR",
                "identity_status": "confirmed",
                "identity_source": "nflverse_players",
            },
            {
                "player_key": production_key,
                "gsis_id": pd.NA,
                "pfr_id": pd.NA,
                "pff_id": pd.NA,
                "espn_id": pd.NA,
                "nfl_id": pd.NA,
                "display_name": "Tet Mcmillan",
                "normalized_name": "tet mcmillan",
                "position": "WR",
                "birth_date": pd.NA,
                "college": "Arizona",
                "draft_year": 2025,
                "draft_round": 1,
                "draft_pick": 8,
                "draft_team": "CAR",
                "rookie_season": 2025,
                "last_season": pd.NA,
                "latest_team": "CAR",
                "identity_status": "provisional",
                "identity_source": "nfl_draft",
            }
        ],
        columns=PLAYER_IDENTITY_COLUMNS,
    )
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0040124",
                "display_name": "Tetairoa McMillan",
                "position_group": "WR",
                "rookie_season": 2025,
                "last_season": 2026,
                "draft_year": 2025,
                "draft_round": 1,
                "draft_pick": 8,
                "draft_team": "CAR",
                "latest_team": "CAR",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players, existing)
    records = pd.DataFrame(
        [
            {
                "source": "nfl_draft",
                "source_player_id": pd.NA,
                "player": "Tet Mcmillan",
                "normalized_name": "tet mcmillan",
                "position": "WR",
                "team": "CAR",
                "season": 2025,
                "draft_year": 2025,
                "_draft_year_inferred": False,
                "draft_round": 1,
                "draft_pick": 8,
                "college": "Arizona",
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records, existing)

    assert identity["player_key"].tolist() == [production_key]
    assert identity.loc[0, "gsis_id"] == "00-0040124"
    assert identity.loc[0, "display_name"] == "Tetairoa McMillan"
    assert aliases["player_key"].tolist() == [production_key]
    assert aliases.loc[0, "match_method"] == (
        "governed_alias_name_position_unique"
    )


def test_source_specific_amon_ra_alias_remaps_stored_2020_to_2021():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0036963",
                "display_name": "Amon-Ra St. Brown",
                "position_group": "WR",
                "rookie_season": 2021,
                "last_season": 2026,
                "draft_year": 2021,
                "draft_round": 4,
                "draft_pick": 112,
                "draft_team": "DET",
                "latest_team": "DET",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "fantasypros",
                "source_table": "FantasyPros_Projections",
                "source_player_id": pd.NA,
                "player": "Amon Ra St",
                "normalized_name": "amon ra st",
                "position": "WR",
                "team": pd.NA,
                "season": season,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
            for season in (2020, 2022)
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    assert len(identity) == 1
    assert identity.loc[0, "display_name"] == "Amon-Ra St. Brown"
    assert aliases["season"].tolist() == [2021, 2022]
    assert aliases["player_key"].eq(identity.loc[0, "player_key"]).all()
    assert set(aliases["match_method"]) == {
        "governed_alias_name_position_unique"
    }
    assert aliases["source_stored_season"].tolist() == [2020, 2022]
    assert aliases.loc[0, "source_season_override_id"] == (
        "fantasypros_wr_2020_to_2021_v1"
    )
    assert aliases.loc[0, "source_season_override_reference"] == (
        "wayback_timestamp=20210728120136"
    )
    assert isinstance(aliases.loc[0, "source_season_override_reason"], str)
    assert pd.isna(aliases.loc[1, "source_season_override_id"])
    assert not identity["identity_status"].eq("provisional").any()
    player_sources = build_player_season_sources(
        aliases,
        identity,
        "season_override_fixture",
        start_season=2020,
        projection_through_season=2022,
    )
    assert player_sources["season"].tolist() == [2021, 2022]
    assert not player_sources["season"].eq(2020).any()


def test_fantasypros_season_override_collision_fails_closed():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0043000",
                "display_name": "Collision Receiver",
                "position_group": "WR",
                "rookie_season": 2021,
                "last_season": 2026,
                "draft_year": 2021,
                "draft_round": 1,
                "draft_pick": 1,
                "draft_team": "DET",
                "latest_team": "DET",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "fantasypros",
                "source_table": "FantasyPros_Projections",
                "source_player_id": pd.NA,
                "player": "Collision Receiver",
                "normalized_name": "collision receiver",
                "position": "WR",
                "team": pd.NA,
                "season": season,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
            for season in (2020, 2021)
        ]
    )

    with pytest.raises(ValueError, match="native WR rows already exist"):
        resolve_source_records(canonical, records)


def test_reconciliation_never_reattaches_incompatible_pre_rookie_alias():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0042999",
                "display_name": "Example Rookie",
                "position_group": "WR",
                "rookie_season": 2021,
                "last_season": 2026,
                "draft_year": 2021,
                "draft_round": 2,
                "draft_pick": 40,
                "draft_team": "DET",
                "latest_team": "DET",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    confirmed_key = canonical.loc[0, "player_key"]
    records = pd.DataFrame(
        [
            {
                "source": "fantasypros",
                "source_player_id": pd.NA,
                "player": "Example Rookie",
                "normalized_name": "example rookie",
                "position": "WR",
                "team": pd.NA,
                "season": 2020,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    assert len(identity) == 2
    provisional = identity[identity["identity_status"].eq("provisional")].iloc[0]
    assert aliases.loc[0, "match_method"] == "provisional_incompatible"
    assert aliases.loc[0, "player_key"] == provisional["player_key"]
    assert aliases.loc[0, "player_key"] != confirmed_key


@pytest.mark.parametrize(
    ("source", "source_name", "canonical_name"),
    [
        ("fantasydata", "drew ogletree", "andrew ogletree"),
        ("fantasypros", "drew ogletree", "andrew ogletree"),
        (
            "fantasypros",
            "equanimeous st",
            "equanimeous st brown",
        ),
        ("adp_mfl", "brown st", "equanimeous st brown"),
        ("fantasydata", "irv charles", "irvin charles"),
        (
            "barret_rank",
            "jacorey croskey merritt",
            "jacory croskey merritt",
        ),
        ("adp_average_nffc", "jayden ott", "jaydn ott"),
        ("nffc_best_ball_overall", "jayden ott", "jaydn ott"),
        ("nffc_rotowire_online", "jayden ott", "jaydn ott"),
        ("adp_fpros", "matt hibner", "matthew hibner"),
        ("fantasydata", "matt hibner", "matthew hibner"),
        (
            "fantasypros_best_ball_adp",
            "matt hibner",
            "matthew hibner",
        ),
        ("adp_average_nffc", "nathan carter", "nate carter"),
        ("fantasydata", "nathan carter", "nate carter"),
        ("fff", "nathan carter", "nate carter"),
        ("nffc_best_ball_25s50s", "nathan carter", "nate carter"),
        ("nffc_best_ball_overall", "nathan carter", "nate carter"),
        ("fantasydata", "scotty miller", "scott miller"),
        ("fantasypros", "scotty miller", "scott miller"),
        (
            "fantasypros_best_ball_adp",
            "scotty miller",
            "scott miller",
        ),
        ("fftoday", "scotty miller", "scott miller"),
    ],
)
def test_recent_governed_alias_ledger_is_source_scoped(
    source,
    source_name,
    canonical_name,
):
    assert _governed_match_name(source_name, source) == canonical_name
    assert _governed_match_name(source_name, "unreviewed_source") == source_name


@pytest.mark.parametrize(
    (
        "source",
        "source_name",
        "display_name",
        "position",
        "rookie_season",
        "draft_year",
        "season",
        "team",
    ),
    [
        (
            "fantasydata",
            "Drew Ogletree",
            "Andrew Ogletree",
            "TE",
            2022,
            2022,
            2024,
            "IND",
        ),
        (
            "fantasydata",
            "Irv Charles",
            "Irvin Charles",
            "WR",
            2022,
            pd.NA,
            2026,
            "NYJ",
        ),
        (
            "barret_rank",
            "Jacorey Croskey Merritt",
            "Jacory Croskey-Merritt",
            "RB",
            2025,
            2025,
            2025,
            "WAS",
        ),
        (
            "adp_average_nffc",
            "Jayden Ott",
            "Jaydn Ott",
            "RB",
            2026,
            pd.NA,
            2026,
            "KC",
        ),
        (
            "adp_fpros",
            "Matt Hibner",
            "Matthew Hibner",
            "TE",
            2026,
            2026,
            2026,
            "BAL",
        ),
        (
            "adp_average_nffc",
            "Nathan Carter",
            "Nate Carter",
            "RB",
            2025,
            pd.NA,
            2026,
            "ATL",
        ),
        (
            "fantasydata",
            "Scotty Miller",
            "Scott Miller",
            "WR",
            2019,
            2019,
            2024,
            "PIT",
        ),
    ],
)
def test_recent_governed_aliases_resolve_without_provisional_identity(
    source,
    source_name,
    display_name,
    position,
    rookie_season,
    draft_year,
    season,
    team,
):
    players = pd.DataFrame(
        [
            {
                "gsis_id": f"test-{source_name}",
                "display_name": display_name,
                "position_group": position,
                "rookie_season": rookie_season,
                "last_season": 2026,
                "draft_year": draft_year,
                "draft_round": pd.NA,
                "draft_pick": pd.NA,
                "draft_team": team,
                "latest_team": team,
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": source,
                "source_player_id": pd.NA,
                "player": source_name,
                "normalized_name": normalize_player_name(source_name),
                "position": position,
                "team": team,
                "season": season,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    assert len(identity) == 1
    assert identity.loc[0, "identity_status"] == "confirmed"
    assert aliases.loc[0, "player_key"] == identity.loc[0, "player_key"]
    assert aliases.loc[0, "match_method"] == (
        "governed_alias_name_position_unique"
    )


def test_returning_player_after_multi_year_gap_keeps_confirmed_identity():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0039065",
                "display_name": "Zack Kuntz",
                "position_group": "TE",
                "rookie_season": 2023,
                "last_season": 2024,
                "draft_year": 2023,
                "draft_round": 7,
                "draft_pick": 220,
                "draft_team": "NYJ",
                "latest_team": "NYJ",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "fantasydata",
                "source_player_id": pd.NA,
                "player": "Zack Kuntz",
                "normalized_name": "zack kuntz",
                "position": "TE",
                "team": "MIA",
                "season": 2026,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    assert len(identity) == 1
    assert identity.loc[0, "identity_status"] == "confirmed"
    assert aliases.loc[0, "player_key"] == identity.loc[0, "player_key"]
    assert aliases.loc[0, "match_method"] == "name_position_unique"


def test_equanimeous_st_brown_truncations_share_confirmed_identity():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0034419",
                "display_name": "Equanimeous St. Brown",
                "position_group": "WR",
                "rookie_season": 2018,
                "last_season": 2023,
                "draft_year": 2018,
                "draft_round": 6,
                "draft_pick": 207,
                "draft_team": "GB",
                "latest_team": "CHI",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "fantasypros",
                "source_table": "FantasyPros_Projections",
                "source_player_id": pd.NA,
                "player": "Equanimeous St",
                "normalized_name": "equanimeous st",
                "position": "WR",
                "team": "GB",
                "season": stored_season,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
            for stored_season in (2016, 2020)
        ]
        + [
            {
                "source": "adp_mfl",
                "source_table": "ADP_MFL",
                "source_player_id": pd.NA,
                "player": "Brown St",
                "normalized_name": "brown st",
                "position": "WR",
                "team": "GB",
                "season": 2019,
                "draft_year": pd.NA,
                "_draft_year_inferred": False,
                "college": pd.NA,
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    confirmed_key = identity.loc[0, "player_key"]
    assert len(identity) == 1
    assert aliases["player_key"].eq(confirmed_key).all()
    fantasypros = aliases[aliases["source"].eq("fantasypros")].sort_values(
        "source_stored_season"
    )
    assert fantasypros["source_stored_season"].tolist() == [2016, 2020]
    assert fantasypros["season"].tolist() == [2018, 2021]
    assert fantasypros["match_method"].eq(
        "governed_alias_name_position_unique"
    ).all()
    assert aliases.loc[aliases["source"].eq("adp_mfl"), "match_method"].eq(
        "governed_alias_name_position_unique"
    ).all()


def test_historical_scott_miller_namesake_remains_separate():
    players = pd.DataFrame(
        [
            {
                "gsis_id": "00-0035298",
                "display_name": "Scott Miller",
                "position_group": "WR",
                "rookie_season": 2019,
                "last_season": 2026,
                "draft_year": 2019,
                "draft_round": 6,
                "draft_pick": 208,
                "draft_team": "TB",
                "latest_team": "CHI",
            }
        ]
    )
    canonical = canonicalize_nflverse_players(players)
    records = pd.DataFrame(
        [
            {
                "source": "nfl_draft",
                "source_player_id": pd.NA,
                "player": "Scott Miller",
                "normalized_name": "scott miller",
                "position": "WR",
                "team": "MIA",
                "season": 1991,
                "draft_year": 1991,
                "_draft_year_inferred": False,
                "draft_round": 9,
                "draft_pick": 246,
                "college": "UCLA",
            }
        ]
    )

    identity, aliases = resolve_source_records(canonical, records)

    confirmed = identity[identity["identity_status"].eq("confirmed")].iloc[0]
    historical = identity[identity["identity_status"].eq("provisional")].iloc[0]
    assert confirmed["draft_year"] == 2019
    assert historical["draft_year"] == 1991
    assert aliases.loc[0, "player_key"] == historical["player_key"]
    assert aliases.loc[0, "match_method"] == "provisional_incompatible"
