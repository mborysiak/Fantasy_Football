import pandas as pd
import pandas.testing as pdt
import pytest

from Scripts.V2.production_handoff import (
    AVG_ADP_AUDIT_TABLE,
    AVG_ADP_RECEIPT_TABLE,
    AVG_ADP_TABLE,
    AVG_ADP_PUBLICATION_VERSION,
    _replace_year_league_slices,
    build_current_avg_adp_publication,
    build_eligibility_membership,
)


MINIMUM_DEPTH = {"dk": 1, "nffc": 1, "etr": 1}


def test_slice_replacement_ignores_all_na_dtype_contributions():
    prior = pd.DataFrame(
        {
            "player": ["Historical Player"],
            "year": [2025.0],
            "league": ["etr"],
            "player_key": [None],
            "etr_rank": [float("nan")],
        }
    )
    current = pd.DataFrame(
        {
            "player": ["Current Player"],
            "year": pd.Series([2026], dtype="Int64"),
            "league": ["etr"],
            "player_key": pd.Series(["current-key"], dtype="string"),
            "etr_rank": [1.0],
        }
    )

    output = _replace_year_league_slices(prior, current, year=2026)

    assert output["player"].tolist() == [
        "Historical Player",
        "Current Player",
    ]
    assert pd.isna(output.loc[0, "player_key"])
    assert pd.isna(output.loc[0, "etr_rank"])
    assert output.loc[1, "player_key"] == "current-key"
    assert output.loc[1, "etr_rank"] == 1.0


def identity_frames():
    identities = pd.DataFrame(
        {
            "player_key": ["alpha-key", "beta-key", "gamma-key"],
            "display_name": ["Alpha Runner", "Beta Receiver", "Gamma End"],
            "normalized_name": [
                "alpha runner",
                "beta receiver",
                "gamma end",
            ],
            "position": ["RB", "WR", "TE"],
            "identity_status": ["confirmed", "confirmed", "confirmed"],
            "latest_team": ["AAA", "BBB", "CCC"],
            "draft_team": ["AAA", "BBB", "CCC"],
        }
    )
    aliases = pd.DataFrame(
        {
            "player_key": ["alpha-key", "beta-key", "gamma-key"],
            "normalized_name": [
                "alpha runner",
                "beta receiver",
                "gamma end",
            ],
            "position": ["RB", "WR", "TE"],
            "team": ["AAA", "BBB", "CCC"],
            "season": [2026, 2026, 2026],
        }
    )
    features = pd.DataFrame(
        {
            "player_key": ["alpha-key", "beta-key", "gamma-key"],
            "season": [2026, 2026, 2026],
            "year_exp": [2.0, 3.0, 4.0],
            "position": ["RB", "WR", "TE"],
        }
    )
    return aliases, identities, features


def source_frames():
    adp = pd.DataFrame(
        {
            "player": [
                "Alpha Runner",
                "Beta Receiver",
                "Jeff Holder",
                "Ghost",
            ],
            "pos": ["RB", "WR", "TK", "TDSP"],
            "year": [2026] * 4,
            "avg_pick": [10.5, 20.5, 300.5, 301.5],
            "min_pick": [8.0, 18.0, 290.0, 295.0],
            "max_pick": [13.0, 24.0, 310.0, 315.0],
            "std_dev": [1.0, 1.5, 4.0, 5.0],
            "league": ["dk", "nffc", "nffc", "nffc"],
        }
    )
    etr = pd.DataFrame(
        {
            "player": ["Gamma End"],
            "team": ["CCC"],
            "pos": ["TE"],
            "etr_rank": [33],
            "etr_pos_rank": [4],
            "etr_adp": [40.5],
            "etr_adp_pos_rank": [5],
            "etr_adp_diff": [7.5],
            "year": [2026],
        }
    )
    return adp, etr


def build_publication(**overrides):
    aliases, identities, features = identity_frames()
    adp, etr = source_frames()
    arguments = {
        "adp_rows": adp,
        "etr_rows": etr,
        "aliases": aliases,
        "identities": identities,
        "season_features": features,
        "year": 2026,
        "published_at_utc": "2026-07-30T12:00:00+00:00",
        "minimum_offensive_depth": MINIMUM_DEPTH,
    }
    arguments.update(overrides)
    return build_current_avg_adp_publication(**arguments)


def test_publication_preserves_players_units_and_exact_etr_ranks():
    tables = build_publication()
    current = tables[AVG_ADP_TABLE]

    assert current.groupby("league").size().to_dict() == {
        "dk": 1,
        "etr": 1,
        "nffc": 3,
    }
    offensive = current["pos"].isin(["QB", "RB", "WR", "TE"])
    assert current.loc[offensive, "player_key"].notna().all()
    assert current.loc[~offensive, "player_key"].isna().all()
    assert current["draft_entity_key"].notna().all()
    assert not current.duplicated(
        ["year", "league", "draft_entity_key"]
    ).any()

    units = current[current["source_player"].isin(["Ghost", "Jeff Holder"])]
    assert set(units["pos"]) == {"TK", "TDSP"}
    assert units["identity_match_method"].eq(
        "non_player_draft_unit"
    ).all()
    assert units["draft_entity_key"].str.startswith("market_unit:").all()

    etr = current[current["league"].eq("etr")].iloc[0]
    assert etr["avg_pick"] == 33
    assert etr["etr_rank"] == 33
    assert etr["etr_pos_rank"] == 4
    assert etr["etr_adp"] == 40.5
    assert etr["etr_adp_pos_rank"] == 5
    assert etr["etr_adp_diff"] == 7.5

    receipts = tables[AVG_ADP_RECEIPT_TABLE].set_index("league")
    assert receipts["source_row_count"].to_dict() == {
        "dk": 1,
        "nffc": 3,
        "etr": 1,
    }
    assert receipts["published_row_count"].to_dict() == {
        "dk": 1,
        "nffc": 3,
        "etr": 1,
    }
    assert receipts["publication_version"].eq(
        AVG_ADP_PUBLICATION_VERSION
    ).all()
    assert len(tables[AVG_ADP_AUDIT_TABLE]) == 5


def test_unchanged_publication_reuses_receipt_time_and_is_idempotent():
    first = build_publication()
    second = build_publication(
        existing_avg_adps=first[AVG_ADP_TABLE],
        existing_audit=first[AVG_ADP_AUDIT_TABLE],
        existing_receipts=first[AVG_ADP_RECEIPT_TABLE],
        published_at_utc="2026-07-31T12:00:00+00:00",
    )

    for table in (
        AVG_ADP_TABLE,
        AVG_ADP_AUDIT_TABLE,
        AVG_ADP_RECEIPT_TABLE,
    ):
        pdt.assert_frame_equal(first[table], second[table])


def test_publication_preserves_non_target_historical_rows():
    historical = pd.DataFrame(
        {
            "player": ["Historical Player"],
            "avg_pick": [50.0],
            "year": [2025],
            "league": ["dk"],
        }
    )
    tables = build_publication(existing_avg_adps=historical)
    output = tables[AVG_ADP_TABLE]

    retained = output[
        pd.to_numeric(output["year"], errors="coerce").eq(2025)
    ]
    assert retained["player"].tolist() == ["Historical Player"]
    assert retained["player_key"].isna().all()


def test_publication_removes_invalid_year_governed_junk_and_audits_counts():
    existing = pd.DataFrame(
        {
            "player": [
                "Valid Historical ETR",
                "Null Year ETR One",
                "Null Year ETR Two",
                "Fractional Year DK",
                "Unrelated Null Year",
            ],
            "avg_pick": [50.0, 60.0, 60.0, 70.0, 80.0],
            "year": [2025, None, None, 2025.5, None],
            "league": ["etr", "etr", "ETR", "dk", "custom"],
        }
    )

    first = build_publication(existing_avg_adps=existing)
    output = first[AVG_ADP_TABLE]
    retained_labels = set(output["player"].dropna().astype(str))

    assert "Valid Historical ETR" in retained_labels
    assert "Unrelated Null Year" in retained_labels
    assert "Null Year ETR One" not in retained_labels
    assert "Null Year ETR Two" not in retained_labels
    assert "Fractional Year DK" not in retained_labels

    receipts = first[AVG_ADP_RECEIPT_TABLE].set_index("league")
    assert receipts["removed_invalid_year_row_count"].to_dict() == {
        "dk": 1,
        "nffc": 0,
        "etr": 2,
    }
    current_audit = first[AVG_ADP_AUDIT_TABLE]
    audit_counts = (
        current_audit.groupby("league")[
            "removed_invalid_year_row_count"
        ]
        .first()
        .to_dict()
    )
    assert audit_counts == {"dk": 1, "etr": 2, "nffc": 0}

    second = build_publication(
        existing_avg_adps=first[AVG_ADP_TABLE],
        existing_audit=first[AVG_ADP_AUDIT_TABLE],
        existing_receipts=first[AVG_ADP_RECEIPT_TABLE],
        published_at_utc="2026-07-31T12:00:00+00:00",
    )
    for table in (
        AVG_ADP_TABLE,
        AVG_ADP_AUDIT_TABLE,
        AVG_ADP_RECEIPT_TABLE,
    ):
        pdt.assert_frame_equal(first[table], second[table])


def test_unresolved_offensive_market_identity_fails_closed():
    adp, etr = source_frames()
    adp.loc[adp["league"].eq("dk"), "player"] = "Unknown Runner"

    with pytest.raises(ValueError, match="unresolved canonical identities"):
        build_publication(adp_rows=adp, etr_rows=etr)


def test_current_season_position_overrides_identity_position():
    aliases, identities, features = identity_frames()
    identities.loc[
        identities["player_key"].eq("beta-key"), "position"
    ] = "DB"

    tables = build_publication(
        aliases=aliases,
        identities=identities,
        season_features=features,
    )
    row = tables[AVG_ADP_TABLE].loc[
        lambda frame: frame["player_key"].eq("beta-key")
    ].iloc[0]

    assert row["source_pos"] == "WR"
    assert row["identity_position"] == "DB"
    assert row["current_position"] == "WR"
    assert row["position_authority"] == "WR"
    assert row["position_authority_source"] == "player_season_features"


def test_true_current_position_mismatch_fails_closed():
    aliases, identities, features = identity_frames()
    features.loc[
        features["player_key"].eq("beta-key"), "position"
    ] = "RB"

    with pytest.raises(
        ValueError,
        match="positions disagree with current canonical position authority",
    ):
        build_publication(
            aliases=aliases,
            identities=identities,
            season_features=features,
        )


def test_governed_hybrid_position_mismatch_is_published_and_audited():
    aliases, identities, features = identity_frames()
    features.loc[
        features["player_key"].eq("beta-key"), "position"
    ] = "RB"

    tables = build_publication(
        aliases=aliases,
        identities=identities,
        season_features=features,
        governed_position_mismatches={
            "beta-key": {
                "source_position": "WR",
                "authority_position": "RB",
                "reason": "reviewed_hybrid",
            }
        },
    )
    row = tables[AVG_ADP_TABLE].loc[
        lambda frame: frame["player_key"].eq("beta-key")
    ].iloc[0]

    assert row["source_pos"] == "WR"
    assert row["pos"] == "RB"
    assert row["position_mismatch_governed"] == 1
    assert row["position_mismatch_reason"] == "reviewed_hybrid"


def test_eligibility_uses_published_key_without_resolving_display_name():
    aliases, identities, _ = identity_frames()
    core = pd.DataFrame(
        {"player": ["Alpha Runner"], "pos": ["RB"], "team": ["AAA"]}
    )
    market = pd.DataFrame(
        {
            "player_key": ["beta-key"],
            "player": ["Provider Label That Is Not An Alias"],
            "pos": ["WR"],
            "avg_pick": [1.0],
            "identity_match_method": ["alias_confirmed_unique"],
        }
    )

    membership = build_eligibility_membership(
        core,
        market,
        pd.DataFrame(columns=["player"]),
        aliases,
        identities,
        league="dk",
        year=2026,
        market_limit=1,
        market_source_name="dk_adp",
    ).set_index("player_key")

    assert membership.loc["beta-key", "eligible_dk_adp"] == 1
    assert membership.loc[
        "beta-key", "production_label_match_method"
    ] == "alias_confirmed_unique"


def test_partially_keyed_market_rows_fail_closed():
    aliases, identities, _ = identity_frames()
    market = pd.DataFrame(
        {
            "player_key": ["beta-key", pd.NA],
            "player": ["Beta Receiver", "Gamma End"],
            "pos": ["WR", "TE"],
            "avg_pick": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="missing player_key"):
        build_eligibility_membership(
            pd.DataFrame(
                {
                    "player": ["Alpha Runner"],
                    "pos": ["RB"],
                    "team": ["AAA"],
                }
            ),
            market,
            pd.DataFrame(columns=["player"]),
            aliases,
            identities,
            league="dk",
            year=2026,
            market_limit=2,
            market_source_name="dk_adp",
        )
