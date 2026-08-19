import sqlite3

import numpy as np
import pandas as pd
import pytest

import Scripts.V2.contracts as contracts
from Scripts.V2.build_feature_sources import (
    PROVIDER_POINTS_ESTIMAND_VERSION,
    _add_provider_room_context,
    _read_resolved_value_rows,
    _required_projection_components,
    _score_projection_values,
    _standardize_identity_rows,
    build_market_values,
    build_projection_values,
    resolve_source_rows,
)
from Scripts.V2.build_feature_mart import build_market_consensus
from Scripts.V2.config import (
    CANDIDATE_SOURCE_TABLES,
    PROJECTION_VALUE_SPECS,
)
from Scripts.V2.contracts import (
    PROJECTION_VALUE_METRICS,
    SOURCE_ROW_EXCLUSION_ID_COLUMN,
    SOURCE_ROW_EXCLUSION_REASON_COLUMN,
    SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN,
    SOURCE_SEASON_OVERRIDE_ID_COLUMN,
    SOURCE_SEASON_OVERRIDE_REASON_COLUMN,
    SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN,
    SOURCE_STORED_SEASON_COLUMN,
    apply_source_team_trust_policy,
    apply_source_season_overrides,
    assert_no_ungoverned_identical_team_conflicts,
    assert_no_source_row_exclusions,
    assert_no_untrusted_source_team_labels,
    configured_scoring,
    normalize_source_team_labels,
    normalize_source_position,
    partition_source_row_exclusions,
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


def test_ftn_projection_source_is_resolved_and_scored_from_components(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "player": "Fixture Runner",
                "pos": "RB",
                "team": "BUF",
                "year": 2026,
                "ftn_auction_value": 42.0,
                "ftn_proj_pts": 999.0,
                "ftn_pass_comp": 0.0,
                "ftn_pass_att": 0.0,
                "ftn_pass_yds": 0.0,
                "ftn_pass_td": 0.0,
                "ftn_pass_int": 0.0,
                "ftn_rush_att": 250.0,
                "ftn_rush_yds": 1200.0,
                "ftn_rush_td": 10.0,
                "ftn_rec_targets": 70.0,
                "ftn_rec": 55.0,
                "ftn_rec_yds": 450.0,
                "ftn_rec_td": 3.0,
            }
        ]
    )
    identity_spec = CANDIDATE_SOURCE_TABLES["FTN_Projections"]
    standardized = _standardize_identity_rows(
        raw,
        "FTN_Projections",
        identity_spec,
    )
    aliases = standardized[
        [
            "source_table",
            "source",
            "source_player_id",
            "normalized_name",
            "position",
            "team",
            "season",
            SOURCE_STORED_SEASON_COLUMN,
        ]
    ].copy()
    aliases["player_key"] = "fixture-runner"
    source_database = tmp_path / "ftn_projection.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FTN_Projections", connection, index=False)

    values, audit = build_projection_values(
        aliases,
        "dk",
        "ftn_projection_fixture",
        source_database=source_database,
        start_season=2026,
        projection_through_season=2026,
    )

    assert len(values) == 1
    row = values.iloc[0]
    assert row["provider"] == "ftn"
    assert row["configured_points_complete"] == 1
    assert row["raw_projected_points"] == 999
    assert row["rushing_yards"] == 1200
    assert row["targets"] == 70
    assert row["provider_projected_points"] != 999
    assert audit.loc[audit["source_table"].eq("FTN_Projections")].iloc[0][
        "resolved_rows"
    ] == 1


@pytest.mark.parametrize(
    ("stored_season", "effective_season", "override_id", "reference"),
    [
        (
            2016,
            2018,
            "fantasypros_wr_2016_to_2018_v1",
            "wayback_timestamp=20180808115212",
        ),
        (
            2020,
            2021,
            "fantasypros_wr_2020_to_2021_v1",
            "wayback_timestamp=20210728120136",
        ),
    ],
)
def test_fantasypros_source_season_override_is_wr_only_and_auditable(
    stored_season,
    effective_season,
    override_id,
    reference,
):
    frame = pd.DataFrame(
        [
            {
                "source_table": source_table,
                "position": position,
                "season": stored_season,
            }
            for source_table, position in (
                ("FantasyPros_Projections", "WR"),
                ("FantasyPros_Projections", "QB"),
                ("FantasyPros_Projections", "RB"),
                ("FantasyPros_Projections", "TE"),
                ("Other_Projections", "WR"),
            )
        ]
    )

    corrected = apply_source_season_overrides(frame, "fixture rows")

    assert corrected["season"].tolist() == [
        effective_season,
        stored_season,
        stored_season,
        stored_season,
        stored_season,
    ]
    assert corrected[SOURCE_STORED_SEASON_COLUMN].eq(stored_season).all()
    wr = corrected.iloc[0]
    assert wr[SOURCE_SEASON_OVERRIDE_ID_COLUMN] == override_id
    assert isinstance(wr[SOURCE_SEASON_OVERRIDE_REASON_COLUMN], str)
    assert wr[SOURCE_SEASON_OVERRIDE_REFERENCE_COLUMN] == reference
    assert corrected.loc[1:, SOURCE_SEASON_OVERRIDE_ID_COLUMN].isna().all()


@pytest.mark.parametrize(
    ("stored_season", "effective_season"),
    [(2016, 2018), (2020, 2021)],
)
def test_fantasypros_source_season_override_rejects_native_wr_collision(
    stored_season,
    effective_season,
):
    frame = pd.DataFrame(
        [
            {
                "source_table": "FantasyPros_Projections",
                "position": "WR",
                "season": season,
            }
            for season in (stored_season, effective_season)
        ]
    )

    with pytest.raises(ValueError, match="native WR rows already exist"):
        apply_source_season_overrides(frame, "collision fixture")


def test_fftoday_2018_qb_quarantine_is_scoped_and_auditable():
    frame = pd.DataFrame(
        [
            {
                "row_id": row_id,
                "source_table": source_table,
                "position": position,
                "season": season,
            }
            for row_id, source_table, position, season in (
                ("excluded_qb", "FFToday_Projections", "QB", 2018),
                ("same_season_rb", "FFToday_Projections", "RB", 2018),
                ("next_season_qb", "FFToday_Projections", "QB", 2019),
                ("other_source_qb", "Other_Projections", "QB", 2018),
            )
        ]
    )

    included, excluded = partition_source_row_exclusions(
        frame,
        "source quarantine fixture",
    )

    assert included["row_id"].tolist() == [
        "same_season_rb",
        "next_season_qb",
        "other_source_qb",
    ]
    assert excluded["row_id"].tolist() == ["excluded_qb"]
    quarantined = excluded.iloc[0]
    assert quarantined[SOURCE_STORED_SEASON_COLUMN] == 2018
    assert quarantined[SOURCE_ROW_EXCLUSION_ID_COLUMN] == (
        "fftoday_qb_stored_2018_2019_vintage_quarantine_v1"
    )
    assert "official 2019 projection archive" in quarantined[
        SOURCE_ROW_EXCLUSION_REASON_COLUMN
    ]
    assert quarantined[SOURCE_ROW_EXCLUSION_REFERENCE_COLUMN] == (
        "https://www.fftoday.com/rankings/playerproj.php"
        "?Season=2019&PosID=10"
    )
    assert_no_source_row_exclusions(
        included,
        "included source quarantine fixture",
    )
    with pytest.raises(ValueError, match="must be quarantined"):
        assert_no_source_row_exclusions(
            frame,
            "unfiltered source quarantine fixture",
        )


@pytest.mark.parametrize(
    "source_table",
    [
        "FFA_RawStats",
        "FFA_Projections",
        "FantasyPros_Best_Ball_ADP",
    ],
)
def test_historical_mutable_team_labels_are_discarded(source_table):
    frame = pd.DataFrame(
        [
            {
                "source_table": source_table,
                "season": 2019,
                "team": "ARI",
            },
            {
                "source_table": source_table,
                "season": 2026,
                "team": "ARI",
            },
            {
                "source_table": "Trusted_Projections",
                "season": 2019,
                "team": "HOU",
            },
        ]
    )

    governed = apply_source_team_trust_policy(frame, "team-policy fixture")

    assert pd.isna(governed.iloc[0]["team"])
    assert governed.iloc[1]["team"] == "ARI"
    assert governed.iloc[2]["team"] == "HOU"
    assert_no_untrusted_source_team_labels(
        governed,
        "governed team-policy fixture",
    )
    with pytest.raises(ValueError, match="must be discarded"):
        assert_no_untrusted_source_team_labels(
            frame,
            "ungoverned team-policy fixture",
        )


def test_identical_except_team_conflicts_fail_closed_without_policy():
    frame = pd.DataFrame(
        [
            {
                "source_table": source_table,
                "season": 2019,
                "player_key": "hopkins",
                "position": "WR",
                "team": team,
                "projected_points": 300.0,
            }
            for source_table, team in (
                ("Fixture_Projections", "HOU"),
                ("Fixture_Projections", "ARI"),
            )
        ]
    )

    with pytest.raises(ValueError, match="identical except for team"):
        assert_no_ungoverned_identical_team_conflicts(
            frame,
            "ungoverned team-conflict fixture",
        )

    governed = frame.assign(source_table="FFA_RawStats")
    assert_no_ungoverned_identical_team_conflicts(
        governed,
        "governed team-conflict fixture",
    )


def test_team_aliases_canonicalize_before_conflict_detection():
    base = {
        "source_table": "Fixture_Projections",
        "season": 2023,
        "player_key": "trevor-lawrence",
        "position": "QB",
        "projected_points": 310.0,
    }
    aliases = pd.DataFrame(
        [{**base, "team": team} for team in ("JAC", "JAX")]
    )

    assert_no_ungoverned_identical_team_conflicts(
        aliases,
        "recognized team-alias fixture",
    )
    normalized = normalize_source_team_labels(
        aliases,
        "recognized team-alias fixture",
    )
    assert normalized["team"].tolist() == ["JAC", "JAC"]

    true_conflict = pd.DataFrame(
        [{**base, "team": team} for team in ("JAC", "BUF")]
    )
    with pytest.raises(ValueError, match="identical except for team"):
        assert_no_ungoverned_identical_team_conflicts(
            true_conflict,
            "true team-conflict fixture",
        )


def test_team_policy_rejects_overlapping_rules_for_one_source(monkeypatch):
    base_rule = {
        "policy_id": "fixture_team_policy_v1",
        "source_table": "Fixture_Projections",
        "through_season": 2024,
        "action": "discard_team",
        "reason": "fixture",
        "reference": "fixture",
    }
    overlapping_rule = {
        **base_rule,
        "policy_id": "fixture_team_policy_v2",
        "through_season": 2025,
    }
    monkeypatch.setattr(
        contracts,
        "SOURCE_TEAM_TRUST_POLICIES",
        (base_rule, overlapping_rule),
    )

    with pytest.raises(ValueError, match="Overlapping.*Fixture_Projections"):
        contracts.source_team_trust_policy_hash()


def test_ffa_2019_team_conflict_uses_trusted_same_season_context(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "ffa_id": "hopkins",
                "player": "DeAndre Hopkins",
                "position": "WR",
                "team": team,
                "year": 2019,
                "ffa_rush_yds": 0,
                "ffa_rush_tds": 0,
                "ffa_rec_yds": 1180,
                "ffa_rec_tds": 8,
            }
            for team in ("HOU", "ARI")
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "player_key": "hopkins",
                "source_table": "FFA_RawStats",
                "source": "ffa_raw",
                "source_player_id": "hopkins",
                "normalized_name": "deandre hopkins",
                "position": "WR",
                "team": team,
                "season": 2019,
            }
            for team in ("HOU", "ARI")
        ]
        + [
            {
                "player_key": "hopkins",
                "source_table": "FFToday_Projections",
                "source": "fftoday",
                "source_player_id": pd.NA,
                "normalized_name": "deandre hopkins",
                "position": "WR",
                "team": "HOU",
                "season": 2019,
            }
        ]
    )
    source_database = tmp_path / "ffa_team_policy.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FFA_RawStats", connection, index=False)

    values, _ = build_projection_values(
        aliases,
        "dk",
        "ffa_team_policy_fixture",
        source_database=source_database,
        start_season=2019,
        projection_through_season=2019,
    )

    assert len(values) == 1
    assert values.iloc[0]["player_key"] == "hopkins"
    assert values.iloc[0]["team"] == "HOU"


def test_value_ingestion_rejects_ungoverned_team_only_conflict(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "player": "Fixture Receiver",
                "pos": "WR",
                "team": team,
                "year": 2025,
                "fdta_rec": 80,
                "fdta_rec_yds": 1000,
                "fdta_rec_td": 7,
            }
            for team in ("JAC", "BUF")
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "player_key": "fixture-receiver",
                "source_table": "FantasyData",
                "source": "fantasydata",
                "source_player_id": pd.NA,
                "normalized_name": "fixture receiver",
                "position": "WR",
                "team": team,
                "season": 2025,
            }
            for team in ("JAC", "BUF")
        ]
    )
    source_database = tmp_path / "ungoverned_team_conflict.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FantasyData", connection, index=False)

    with pytest.raises(ValueError, match="identical except for team"):
        build_projection_values(
            aliases,
            "dk",
            "ungoverned_team_conflict_fixture",
            source_database=source_database,
            start_season=2025,
            projection_through_season=2026,
        )


def test_value_ingestion_canonicalizes_team_aliases_before_grouping(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "player": "Trevor Lawrence",
                "pos": "QB",
                "team": team,
                "year": 2023,
                "fdta_pass_yds": 4000,
                "fdta_pass_td": 25,
                "fdta_pass_int": 10,
                "fdta_rush_yds": 300,
                "fdta_rush_td": 3,
            }
            for team in ("JAC", "JAX")
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "player_key": "trevor-lawrence",
                "source_table": "FantasyData",
                "source": "fantasydata",
                "source_player_id": pd.NA,
                "normalized_name": "trevor lawrence",
                "position": "QB",
                "team": team,
                "season": 2023,
            }
            for team in ("JAC", "JAX")
        ]
    )
    source_database = tmp_path / "team_alias_values.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FantasyData", connection, index=False)

    values, _ = build_projection_values(
        aliases,
        "dk",
        "team_alias_value_fixture",
        source_database=source_database,
        start_season=2023,
        projection_through_season=2023,
    )

    assert len(values) == 1
    assert values.iloc[0]["player_key"] == "trevor-lawrence"
    assert values.iloc[0]["team"] == "JAC"


def test_feature_ingestion_uses_effective_season_before_alias_resolution(
    tmp_path,
):
    raw = pd.DataFrame(
        [
            {
                "player": "Amon Ra St",
                "pos": "WR",
                "year": 2020,
                "fpros_rec": 90,
                "fpros_rec_yds": 1100,
                "fpros_rec_td": 8,
            },
            {
                "player": "Example Quarterback",
                "pos": "QB",
                "year": 2020,
                "fpros_pass_yds": 4000,
                "fpros_pass_td": 30,
            },
        ]
    )
    identity_spec = CANDIDATE_SOURCE_TABLES["FantasyPros_Projections"]
    standardized = _standardize_identity_rows(
        raw,
        "FantasyPros_Projections",
        identity_spec,
    )
    aliases = standardized[
        [
            "source_table",
            "source",
            "source_player_id",
            "normalized_name",
            "position",
            "team",
            "season",
        ]
    ].copy()
    aliases["player_key"] = ["amon", "quarterback"]

    resolved = resolve_source_rows(standardized, aliases)

    assert standardized["season"].tolist() == [2021, 2020]
    assert standardized[SOURCE_STORED_SEASON_COLUMN].tolist() == [2020, 2020]
    assert resolved.tolist() == ["amon", "quarterback"]

    with sqlite3.connect(":memory:") as connection:
        raw.to_sql(
            "FantasyPros_Projections",
            connection,
            index=False,
        )
        (
            provider_rows,
            input_rows,
            resolved_rows,
            excluded_rows,
        ) = _read_resolved_value_rows(
            connection,
            "FantasyPros_Projections",
            PROJECTION_VALUE_SPECS["FantasyPros_Projections"],
            aliases,
            start_season=2020,
            projection_through_season=2021,
        )

    seasons = provider_rows.set_index("player_key")["season"].to_dict()
    assert input_rows == 2
    assert resolved_rows == 2
    assert excluded_rows.empty
    assert seasons == {"amon": 2021, "quarterback": 2020}
    assert not (
        provider_rows["player_key"].eq("amon")
        & provider_rows["season"].eq(2020)
    ).any()

    source_database = tmp_path / "source_seasons.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql(
            "FantasyPros_Projections",
            connection,
            index=False,
        )
    values, _ = build_projection_values(
        aliases,
        "dk",
        "source_season_fixture",
        source_database=source_database,
        start_season=2020,
        projection_through_season=2021,
    )
    amon = values[values["player_key"].eq("amon")].iloc[0]
    quarterback = values[values["player_key"].eq("quarterback")].iloc[0]
    assert amon["season"] == 2021
    assert amon["source_stored_seasons"] == "2020"
    assert amon["source_season_override_ids"] == (
        "fantasypros_wr_2020_to_2021_v1"
    )
    assert isinstance(amon["source_season_override_reasons"], str)
    assert amon["source_season_override_references"] == (
        "wayback_timestamp=20210728120136"
    )
    assert quarterback["season"] == 2020
    assert quarterback["source_stored_seasons"] == "2020"
    assert pd.isna(quarterback["source_season_override_ids"])


def test_fantasydata_remains_historical_only_after_2025(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "player": "Historical Running Back",
                "pos": "RB",
                "team": "ARI",
                "year": 2025,
                "fdta_rush_yds": 800,
                "fdta_rush_td": 6,
                "fdta_rec": 40,
                "fdta_rec_yds": 300,
                "fdta_rec_td": 2,
                "fdta_rank": 80,
            },
            {
                "player": "Current Running Back",
                "pos": "RB",
                "team": "ARI",
                "year": 2026,
                "fdta_rush_yds": 900,
                "fdta_rush_td": 7,
                "fdta_rec": 45,
                "fdta_rec_yds": 350,
                "fdta_rec_td": 3,
                "fdta_rank": 60,
            },
        ]
    )
    standardized = _standardize_identity_rows(
        raw,
        "FantasyData",
        CANDIDATE_SOURCE_TABLES["FantasyData"],
    )
    aliases = standardized[
        [
            "source_table",
            "source",
            "source_player_id",
            "normalized_name",
            "position",
            "team",
            "season",
            SOURCE_STORED_SEASON_COLUMN,
        ]
    ].copy()
    aliases["player_key"] = ["historical-rb", "current-rb"]

    source_database = tmp_path / "fantasydata_historical_only.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FantasyData", connection, index=False)

    values, _ = build_projection_values(
        aliases,
        "dk",
        "fantasydata_historical_only_fixture",
        source_database=source_database,
        start_season=2025,
        projection_through_season=2026,
    )

    observed = set(
        values[["player_key", "season", "provider"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert observed == {("historical-rb", 2025, "fantasydata")}

    market_values, _ = build_market_values(
        aliases,
        "fantasydata_historical_only_fixture",
        source_database=source_database,
        start_season=2025,
        projection_through_season=2026,
    )
    observed_market = set(
        market_values[["player_key", "season", "source"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert observed_market == {("historical-rb", 2025, "fantasydata")}


def test_nffc_contributes_only_its_single_composite_market_row(tmp_path):
    nffc_composite = pd.DataFrame(
        {
            "player": ["Example Receiver"],
            "pos": ["WR"],
            "year": [2026],
            "avg_pick": [42.0],
            "league": ["nffc"],
        }
    )
    nffc_contests = pd.DataFrame(
        {
            "player": ["Example Receiver"] * 4,
            "pos": ["WR"] * 4,
            "team": ["BUF"] * 4,
            "year": [2026] * 4,
            "pick_nffc": [38.0, 40.0, 44.0, 46.0],
            "source": [
                "nffc_rotowire_online",
                "nffc_best_ball_overall",
                "nffc_best_ball_25s50s",
                "nffc_cutline",
            ],
        }
    )
    aliases = pd.concat(
        [
            _standardize_identity_rows(
                nffc_composite,
                "ADP_Averages",
                CANDIDATE_SOURCE_TABLES["ADP_Averages"],
            ),
            _standardize_identity_rows(
                nffc_contests,
                "NFFC_ADP",
                CANDIDATE_SOURCE_TABLES["NFFC_ADP"],
            ),
        ],
        ignore_index=True,
    )
    aliases["player_key"] = "example-receiver"

    source_database = tmp_path / "nffc_family_consensus.sqlite3"
    with sqlite3.connect(source_database) as connection:
        nffc_composite.to_sql(
            "ADP_Averages",
            connection,
            index=False,
        )
        nffc_contests.to_sql(
            "NFFC_ADP",
            connection,
            index=False,
        )

    values, audit = build_market_values(
        aliases,
        "nffc_family_fixture",
        source_database=source_database,
        start_season=2026,
        projection_through_season=2026,
    )

    assert values[["player_key", "source", "adp"]].to_dict("records") == [
        {
            "player_key": "example-receiver",
            "source": "adp_average_nffc",
            "adp": 42.0,
        }
    ]
    assert audit["source_table"].tolist() == ["ADP_Averages"]
    consensus = build_market_consensus(values).iloc[0]
    assert consensus["adp_median"] == 42.0
    assert consensus["adp_source_count"] == 1


def test_fftoday_quarantine_reaches_projection_values_and_resolution_audit(
    tmp_path,
):
    raw = pd.DataFrame(
        [
            {
                "player": "Corrupt Quarterback",
                "pos": "QB",
                "team": "LAR",
                "year": 2018,
                "fft_pass_comp": 300,
                "fft_pass_att": 500,
                "fft_pass_yds": 4000,
                "fft_pass_td": 30,
                "fft_pass_int": 10,
                "fft_rush_att": 40,
                "fft_rush_yds": 200,
                "fft_rush_td": 2,
                "fft_rec": 0,
                "fft_rec_yds": 0,
                "fft_rec_td": 0,
                "fft_sacks": 30,
            },
            {
                "player": "Unaffected Running Back",
                "pos": "RB",
                "team": "BUF",
                "year": 2018,
                "fft_pass_comp": 0,
                "fft_pass_att": 0,
                "fft_pass_yds": 0,
                "fft_pass_td": 0,
                "fft_pass_int": 0,
                "fft_rush_att": 200,
                "fft_rush_yds": 900,
                "fft_rush_td": 7,
                "fft_rec": 50,
                "fft_rec_yds": 400,
                "fft_rec_td": 3,
                "fft_sacks": 0,
            },
            {
                "player": "Unaffected Quarterback",
                "pos": "QB",
                "team": "KC",
                "year": 2019,
                "fft_pass_comp": 350,
                "fft_pass_att": 550,
                "fft_pass_yds": 4500,
                "fft_pass_td": 35,
                "fft_pass_int": 8,
                "fft_rush_att": 50,
                "fft_rush_yds": 250,
                "fft_rush_td": 2,
                "fft_rec": 0,
                "fft_rec_yds": 0,
                "fft_rec_td": 0,
                "fft_sacks": 25,
            },
        ]
    )
    identity_spec = CANDIDATE_SOURCE_TABLES["FFToday_Projections"]
    standardized = _standardize_identity_rows(
        raw,
        "FFToday_Projections",
        identity_spec,
    )
    assert set(
        standardized[["normalized_name", "position", "season"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        ("unaffected running back", "RB", 2018),
        ("unaffected quarterback", "QB", 2019),
    }

    aliases = standardized[
        [
            "source_table",
            "source",
            "source_player_id",
            "normalized_name",
            "position",
            "team",
            "season",
            SOURCE_STORED_SEASON_COLUMN,
        ]
    ].copy()
    aliases["player_key"] = ["running-back", "quarterback"]
    source_database = tmp_path / "fftoday_quarantine.sqlite3"
    with sqlite3.connect(source_database) as connection:
        raw.to_sql("FFToday_Projections", connection, index=False)
        (
            provider_rows,
            input_rows,
            resolved_rows,
            excluded_rows,
        ) = _read_resolved_value_rows(
            connection,
            "FFToday_Projections",
            PROJECTION_VALUE_SPECS["FFToday_Projections"],
            aliases,
            start_season=2018,
            projection_through_season=2019,
        )

    assert input_rows == 2
    assert resolved_rows == 2
    assert len(excluded_rows) == 1
    assert excluded_rows.loc[0, SOURCE_ROW_EXCLUSION_ID_COLUMN] == (
        "fftoday_qb_stored_2018_2019_vintage_quarantine_v1"
    )
    assert set(
        provider_rows[["player_key", "position", "season"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        ("running-back", "RB", 2018),
        ("quarterback", "QB", 2019),
    }

    values, audit = build_projection_values(
        aliases,
        "dk",
        "fftoday_quarantine_fixture",
        source_database=source_database,
        start_season=2018,
        projection_through_season=2019,
    )
    assert set(
        values[["player_key", "position", "season"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        ("running-back", "RB", 2018),
        ("quarterback", "QB", 2019),
    }
    source_audit = audit.iloc[0]
    assert source_audit["input_rows"] == 2
    assert source_audit["resolved_rows"] == 2
    assert source_audit["excluded_rows"] == 1
    assert source_audit["source_row_exclusion_ids"] == (
        "fftoday_qb_stored_2018_2019_vintage_quarantine_v1"
    )
    assert "official 2019 projection archive" in source_audit[
        "source_row_exclusion_reasons"
    ]
    assert source_audit["source_row_exclusion_references"] == (
        "https://www.fftoday.com/rankings/playerproj.php"
        "?Season=2019&PosID=10"
    )


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
    assert scored.loc[2, "configured_points_imputed_components"] == (
        "receptions"
    )
    assert scored.loc[
        2, "configured_points_imputation_donor_providers"
    ] == "provider_a|provider_b"
    assert (
        scored.loc[2, "configured_points_imputation_donor_count"] == 2
    )
    assert scored.loc[2, "points_method"] == "configured_components_imputed"


def test_auction_leagues_require_qb_sacks_and_nv_uses_four_point_pass_tds():
    quarterback = {
        "position": "QB",
        "passing_yards": 4000,
        "passing_tds": 30,
        "interceptions": 10,
        "rushing_yards": 200,
        "rushing_tds": 2,
    }
    frame = pd.DataFrame(
        [
            _projection_row(provider="provider_a", sacks=30, **quarterback),
            _projection_row(provider="provider_b", sacks=40, **quarterback),
            _projection_row(
                provider="provider_missing_sacks",
                sacks=np.nan,
                **quarterback,
            ),
        ]
    )

    beta = _score_projection_values(frame, "beta")
    assert "sacks" in _required_projection_components("beta")["QB"]
    assert beta.loc[2, "sacks"] == 35
    assert beta.loc[2, "configured_points_complete"] == 1
    assert beta.loc[2, "configured_points_imputed_component_count"] == 1
    assert beta.loc[2, "configured_points_imputed_components"] == "sacks"
    assert beta.loc[
        2, "configured_points_imputation_donor_providers"
    ] == "provider_a|provider_b"
    assert beta.loc[2, "configured_points_imputation_donor_count"] == 2
    assert beta.loc[2, "points_method"] == "configured_components_imputed"
    assert beta.loc[2, "passing_points"] == pytest.approx(
        4000 * 0.04 + 30 * 5 - 10 * 2 - 35
    )

    nv = _score_projection_values(frame, "nv")
    assert "sacks" in _required_projection_components("nv")["QB"]
    assert nv.loc[2, "sacks"] == 35
    assert nv.loc[2, "configured_points_complete"] == 1
    assert nv.loc[2, "passing_points"] == pytest.approx(
        4000 * 0.04 + 30 * 4 - 10 * 2 - 35
    )

    dk_without_sack_donors = _score_projection_values(frame.iloc[[2]], "dk")
    assert "sacks" not in _required_projection_components("dk")["QB"]
    assert pd.isna(dk_without_sack_donors.loc[2, "sacks"])
    assert dk_without_sack_donors.loc[2, "configured_points_complete"] == 1
    assert (
        dk_without_sack_donors.loc[
            2, "configured_points_imputed_component_count"
        ]
        == 0
    )
    assert pd.isna(
        dk_without_sack_donors.loc[
            2, "configured_points_imputed_components"
        ]
    )
    assert pd.isna(
        dk_without_sack_donors.loc[
            2, "configured_points_imputation_donor_providers"
        ]
    )
    assert (
        dk_without_sack_donors.loc[
            2, "configured_points_imputation_donor_count"
        ]
        == 0
    )
    assert dk_without_sack_donors.loc[2, "points_method"] == (
        "configured_components"
    )


@pytest.mark.parametrize("league", ["beta", "nv"])
def test_auction_qb_sacks_allow_one_donor_without_weakening_other_components(
    league,
):
    quarterback = {
        "position": "QB",
        "passing_yards": 4000,
        "passing_tds": 30,
        "interceptions": 10,
        "rushing_yards": 200,
        "rushing_tds": 2,
    }
    quarterback_rows = pd.DataFrame(
        [
            _projection_row(provider="fftoday", sacks=32, **quarterback),
            _projection_row(
                provider="provider_missing_sacks",
                sacks=np.nan,
                **quarterback,
            ),
        ]
    )
    scored = _score_projection_values(quarterback_rows, league)
    assert scored.loc[1, "sacks"] == 32
    assert scored.loc[1, "configured_points_complete"] == 1
    assert scored.loc[1, "configured_points_imputed_component_count"] == 1
    assert scored.loc[1, "configured_points_imputed_components"] == "sacks"
    assert scored.loc[
        1, "configured_points_imputation_donor_providers"
    ] == "fftoday"
    assert scored.loc[1, "configured_points_imputation_donor_count"] == 1
    assert scored.loc[1, "points_method"] == "configured_components_imputed"

    receiver_rows = pd.DataFrame(
        [
            _projection_row(
                provider="provider_a",
                receptions=60,
                receiving_yards=900,
                receiving_tds=6,
            ),
            _projection_row(
                provider="provider_missing_receptions",
                receptions=np.nan,
                receiving_yards=1000,
                receiving_tds=8,
            ),
        ]
    )
    receiver_scored = _score_projection_values(receiver_rows, "beta")
    assert pd.isna(receiver_scored.loc[1, "receptions"])
    assert receiver_scored.loc[1, "configured_points_complete"] == 0
    assert receiver_scored.loc[1, "points_method"] == "insufficient"


def test_sack_imputation_does_not_cross_hybrid_player_positions():
    quarterback = {
        "position": "QB",
        "passing_yards": 3000,
        "passing_tds": 20,
        "interceptions": 8,
        "rushing_yards": 500,
        "rushing_tds": 5,
    }
    frame = pd.DataFrame(
        [
            _projection_row(
                provider="non_qb_projection",
                position="TE",
                sacks=0,
                receptions=40,
                receiving_yards=450,
                receiving_tds=4,
            ),
            _projection_row(
                provider="qb_missing_sacks",
                sacks=np.nan,
                **quarterback,
            ),
        ]
    )

    scored = _score_projection_values(frame, "beta")

    assert pd.isna(scored.loc[1, "sacks"])
    assert scored.loc[1, "configured_points_complete"] == 0
    assert scored.loc[1, "configured_points_imputed_component_count"] == 0
    assert pd.isna(
        scored.loc[1, "configured_points_imputation_donor_providers"]
    )
    assert scored.loc[1, "points_method"] == "insufficient"


def test_provider_estimand_is_linear_and_excludes_weekly_yardage_bonuses():
    frame = pd.DataFrame(
        [
            _projection_row(
                position="QB",
                passing_yards=5100,
                passing_tds=40,
                interceptions=10,
                sacks=np.nan,
                rushing_yards=300,
                rushing_tds=3,
            )
        ]
    )
    scored = _score_projection_values(frame, "dk")
    rules = configured_scoring("dk")
    expected = (
        5100 * rules["passing"]["pass_yards_gained_sum"]
        + 40 * rules["passing"]["pass_pass_touchdown_sum"]
        + 10 * rules["passing"]["pass_interception_sum"]
        + 300 * rules["rushing"]["rush_yards_gained_sum"]
        + 3 * rules["rushing"]["rush_rush_touchdown_sum"]
    )
    assert PROVIDER_POINTS_ESTIMAND_VERSION == (
        "core_offensive_season_components_v1"
    )
    assert scored.loc[0, "provider_points_estimand"] == (
        PROVIDER_POINTS_ESTIMAND_VERSION
    )
    assert scored.loc[0, "provider_projected_points"] == pytest.approx(expected)


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
