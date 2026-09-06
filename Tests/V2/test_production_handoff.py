import json
import sqlite3

import pandas as pd
import pandas.testing as pdt
import pytest

from Scripts.V2.contracts import scoring_hash
from Scripts.V2.production_handoff import (
    AUTOMATIC_MARKET_BUFFER_EXCLUSION_REASON,
    CURRENT_RESIDUAL_COLUMNS,
    GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR,
    MARKET_ELIGIBILITY_RULES,
    MARKET_HANDOFF_PROTECTED_PICK_DEPTH,
    MARKET_HANDOFF_REQUIRED_DEPTH,
    NEXT_RESIDUAL_SOURCE_COLUMNS,
    PRODUCTION_ELIGIBILITY_VERSION,
    PRODUCTION_EXCLUSION_POLICY_VERSION,
    build_eligibility_membership,
    build_legacy_projection_backup,
    build_production_projection_slice,
    load_validated_shadow_predictions,
    resolve_source_player_keys,
)


def identity_frames():
    identities = pd.DataFrame(
        {
            "player_key": ["old-key", "new-key", "other-key"],
            "normalized_name": [
                "old player",
                "tetairoa mcmillan",
                "deebo samuel",
            ],
            "position": ["RB", "WR", "WR"],
            "identity_status": ["confirmed", "confirmed", "confirmed"],
            "latest_team": ["OLD", "CAR", "WAS"],
            "draft_team": ["OLD", "CAR", "SF"],
        }
    )
    aliases = pd.DataFrame(
        {
            "player_key": [
                "old-key",
                "new-key",
                "new-key",
                "other-key",
                "other-key",
            ],
            "normalized_name": [
                "old player",
                "tet mcmillan",
                "tetairoa mcmillan",
                "deebo samuel",
                "deebo samuel sr",
            ],
            "position": ["RB", "WR", "WR", "WR", "WR"],
            "team": ["OLD", "CAR", "CAR", "WAS", "WAS"],
            "season": [2026, 2026, 2026, 2026, 2026],
        }
    )
    return aliases, identities


def shadow_frames():
    current = pd.DataFrame(
        {
            "player_key": ["old-key", "new-key", "other-key"],
            "display_name": [
                "Old Player",
                "Tetairoa McMillan",
                "Deebo Samuel Sr.",
            ],
            "season": [2026, 2026, 2026],
            "position": ["RB", "WR", "WR"],
            "team": ["OLD", "CAR", "WAS"],
            "conditional_ppg_shadow": [7.0, 14.0, 9.0],
            "participation_probability": [0.90, 0.98, 0.95],
            "lock_version": ["current-lock"] * 3,
            "publication_status": ["shadow"] * 3,
        }
    )
    next_year = pd.DataFrame(
        {
            "player_key": ["old-key", "new-key", "other-key"],
            "predicted_next_year_conditional_ppg": [6.5, 14.5, 8.5],
            "predicted_next_year_appearance_probability": [0.7, 0.9, 0.8],
            "target_version": ["next-lock"] * 3,
            "scoring_hash": [scoring_hash("dk")] * 3,
            "origin_season": [2026] * 3,
            "target_season": [2027] * 3,
            "position": ["RB", "WR", "WR"],
            "team": ["OLD", "CAR", "WAS"],
            "publication_status": ["shadow"] * 3,
            **{
                source: [value, value, value]
                for source, value in zip(
                    NEXT_RESIDUAL_SOURCE_COLUMNS,
                    (-5.0, -4.0, -2.0, 2.0, 4.0, 6.0),
                )
            },
        }
    )
    return current, next_year


def eligibility_frame(keys=("new-key", "other-key")):
    return pd.DataFrame(
        {
            "player_key": list(keys),
            "production_player_label": [
                {
                    "new-key": "Tet Mcmillan",
                    "other-key": "Deebo Samuel",
                    "old-key": "Old Player",
                }[key]
                for key in keys
            ],
            "production_label_match_method": ["alias_confirmed_unique"]
            * len(keys),
            "eligibility_sources": ["core_projonly"] * len(keys),
            "eligible_core_projonly": [1] * len(keys),
            "eligible_dk_adp": [0] * len(keys),
            "eligible_etr_adp": [0] * len(keys),
            "eligible_league_keeper": [0] * len(keys),
            "market_eligibility_rank": [None] * len(keys),
            "market_eligibility_pick": [None] * len(keys),
            "production_eligibility_version": [
                PRODUCTION_ELIGIBILITY_VERSION
            ]
            * len(keys),
        }
    )


def legacy_frame():
    return pd.DataFrame(
        {
            "player_key": ["old-key"],
            "player": ["Old Player"],
            "pos": ["RB"],
            "year": [2026],
            "version": ["dk"],
            "dataset": ["final_ensemble"],
            "pred_fp_per_game": [7.5],
            "pred_fp_per_game_ny": [7.0],
        }
    )


def test_legacy_backup_appends_a_new_target_season_once():
    existing = legacy_frame().assign(
        backup_created_at_utc="2026-01-01T00:00:00+00:00"
    )
    next_season = legacy_frame().assign(
        year=2027,
        pred_fp_per_game=8.0,
    )

    appended = build_legacy_projection_backup(
        existing,
        next_season,
        year=2027,
        dataset="final_ensemble",
    )

    pdt.assert_frame_equal(
        appended.iloc[[0]].reset_index(drop=True),
        existing,
    )
    added = appended[appended["year"].eq(2027)].reset_index(drop=True)
    assert len(added) == 1
    assert added.loc[0, "pred_fp_per_game"] == 8.0
    assert pd.notna(added.loc[0, "backup_created_at_utc"])

    changed_source = next_season.assign(pred_fp_per_game=99.0)
    rerun = build_legacy_projection_backup(
        appended,
        changed_source,
        year=2027,
        dataset="final_ensemble",
    )
    pdt.assert_frame_equal(rerun, appended)


def test_legacy_backup_does_not_backfill_a_league_after_scope_is_frozen():
    existing = legacy_frame().assign(
        backup_created_at_utc="2026-01-01T00:00:00+00:00"
    )
    promoted_nffc = legacy_frame().assign(
        version="nffc",
        player_key="promoted-key",
        player="Promoted Player",
    )
    current_target = pd.concat(
        [legacy_frame(), promoted_nffc],
        ignore_index=True,
    )

    preserved = build_legacy_projection_backup(
        existing,
        current_target,
        year=2026,
        dataset="final_ensemble",
    )

    pdt.assert_frame_equal(preserved, existing)
    assert "nffc" not in set(preserved["version"])


def test_legacy_backup_rejects_an_empty_new_target_scope():
    existing = legacy_frame().assign(
        backup_created_at_utc="2026-01-01T00:00:00+00:00"
    )
    empty_2027 = legacy_frame().iloc[0:0].assign(year=2027)

    with pytest.raises(
        ValueError,
        match="cannot initialize an immutable baseline.*empty",
    ):
        build_legacy_projection_backup(
            existing,
            empty_2027,
            year=2027,
            dataset="final_ensemble",
        )


def lineage_frames():
    feature_run_id = "feature-run"
    lock_version = "v2_conditional_ppg_2026_candidate_v1"
    model_run_id = "current-model-run"
    expected_hash = scoring_hash("dk")
    return {
        "player_season_features": pd.DataFrame(
            {
                "player_key": ["one", "two"],
                "run_id": [feature_run_id, feature_run_id],
                "league": ["dk", "dk"],
                "scoring_hash": [expected_hash, expected_hash],
            }
        ),
        "locked_candidate_runs": pd.DataFrame(
            {
                "lock_version": [lock_version],
                "model_run_id": [model_run_id],
                "feature_run_id": [feature_run_id],
                "current_shadow_season": [2026],
                "status": ["complete_shadow"],
                "metadata_json": [
                    json.dumps(
                        {
                            "lock_version": lock_version,
                            "scoring_objective": "dk",
                        }
                    )
                ],
            }
        ),
        "locked_2026_shadow_predictions": pd.DataFrame(
            {
                "player_key": ["one", "two"],
                "lock_version": [lock_version, lock_version],
                "model_run_id": [model_run_id, model_run_id],
                "season": [2026, 2026],
                "publication_status": ["shadow", "shadow"],
            }
        ),
        "next_year_2027_shadow_predictions": pd.DataFrame(
            {
                "player_key": ["one", "two"],
                "run_id": ["next-run", "next-run"],
                "feature_run_id": [feature_run_id, feature_run_id],
                "origin_season": [2026, 2026],
                "target_season": [2027, 2027],
                "target_version": [
                    "v2_next_year_expert_residual_v1",
                    "v2_next_year_expert_residual_v1",
                ],
                "league": ["dk", "dk"],
                "scoring_hash": [expected_hash, expected_hash],
                "publication_status": ["shadow", "shadow"],
            }
        ),
    }


def write_lineage_database(path, frames):
    with sqlite3.connect(path) as connection:
        for table, frame in frames.items():
            frame.to_sql(table, connection, index=False, if_exists="replace")


def test_shadow_lineage_guard_accepts_one_coherent_release(tmp_path):
    database = tmp_path / "v2.sqlite3"
    frames = lineage_frames()
    write_lineage_database(database, frames)

    current, next_year = load_validated_shadow_predictions(
        database,
        league="dk",
        year=2026,
    )

    pdt.assert_frame_equal(
        current,
        frames["locked_2026_shadow_predictions"],
        check_dtype=False,
    )
    pdt.assert_frame_equal(
        next_year,
        frames["next_year_2027_shadow_predictions"],
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frames: frames["player_season_features"].__setitem__(
                "run_id", ["feature-run", None]
            ),
            "missing run_id lineage values",
        ),
        (
            lambda frames: frames["player_season_features"].__setitem__(
                "run_id", ["feature-run", "new-feature-run"]
            ),
            "multiple run_id lineage values",
        ),
        (
            lambda frames: frames["locked_candidate_runs"].__setitem__(
                "feature_run_id", ["stale-feature-run"]
            ),
            "locked feature_run_id is stale",
        ),
        (
            lambda frames: frames["player_season_features"].__setitem__(
                "league", ["beta", "beta"]
            ),
            "player_season_features league mismatch",
        ),
        (
            lambda frames: frames["player_season_features"].__setitem__(
                "scoring_hash", ["stale-hash", "stale-hash"]
            ),
            "player_season_features scoring_hash mismatch",
        ),
        (
            lambda frames: frames["locked_candidate_runs"].__setitem__(
                "metadata_json",
                [
                    json.dumps(
                        {
                            "lock_version": "current-lock",
                            "scoring_objective": "beta",
                        }
                    )
                ],
            ),
            "locked run scoring objective mismatch",
        ),
        (
            lambda frames: frames[
                "locked_2026_shadow_predictions"
            ].__setitem__("lock_version", ["other-lock", "other-lock"]),
            "current shadow lock_version mismatch",
        ),
        (
            lambda frames: frames[
                "locked_2026_shadow_predictions"
            ].__setitem__("season", [2025, 2025]),
            "season mismatch",
        ),
        (
            lambda frames: frames[
                "next_year_2027_shadow_predictions"
            ].__setitem__(
                "feature_run_id",
                ["stale-feature-run", "stale-feature-run"],
            ),
            "next shadow feature_run_id is stale",
        ),
        (
            lambda frames: frames[
                "next_year_2027_shadow_predictions"
            ].__setitem__("target_version", ["next-lock", "other-lock"]),
            "multiple target_version lineage values",
        ),
        (
            lambda frames: frames[
                "next_year_2027_shadow_predictions"
            ].__setitem__("origin_season", [2025, 2025]),
            "origin_season mismatch",
        ),
        (
            lambda frames: frames[
                "next_year_2027_shadow_predictions"
            ].__setitem__("target_season", [2028, 2028]),
            "target_season mismatch",
        ),
        (
            lambda frames: frames[
                "next_year_2027_shadow_predictions"
            ].__setitem__("league", ["beta", "beta"]),
            "next shadow league mismatch",
        ),
        (
            lambda frames: frames[
                "next_year_2027_shadow_predictions"
            ].__setitem__("scoring_hash", ["stale-hash", "stale-hash"]),
            "next shadow scoring_hash mismatch",
        ),
    ],
)
def test_shadow_lineage_guard_rejects_stale_or_mixed_artifacts(
    tmp_path,
    mutation,
    message,
):
    database = tmp_path / "v2.sqlite3"
    frames = lineage_frames()
    mutation(frames)
    write_lineage_database(database, frames)

    with pytest.raises(ValueError, match=message):
        load_validated_shadow_predictions(
            database,
            league="dk",
            year=2026,
        )


@pytest.mark.parametrize("completed_rows", [0, 2])
def test_shadow_lineage_guard_requires_exactly_one_complete_lock(
    tmp_path,
    completed_rows,
):
    database = tmp_path / "v2.sqlite3"
    frames = lineage_frames()
    locked = frames["locked_candidate_runs"]
    if completed_rows == 0:
        locked["status"] = "failed"
    else:
        frames["locked_candidate_runs"] = pd.concat(
            [locked, locked],
            ignore_index=True,
        )
    write_lineage_database(database, frames)

    with pytest.raises(
        ValueError,
        match=(
            "requires exactly one complete_shadow "
            "locked_candidate_runs row"
        ),
    ):
        load_validated_shadow_predictions(
            database,
            league="dk",
            year=2026,
        )


def test_shadow_lineage_guard_rejects_missing_lineage_table(tmp_path):
    database = tmp_path / "v2.sqlite3"
    frames = lineage_frames()
    del frames["next_year_2027_shadow_predictions"]
    write_lineage_database(database, frames)

    with pytest.raises(ValueError, match="missing required tables"):
        load_validated_shadow_predictions(
            database,
            league="dk",
            year=2026,
        )


def test_v2_population_adds_required_players_and_drops_legacy_only_rows():
    current, next_year = shadow_frames()
    output, audit, master = build_production_projection_slice(
        legacy_frame(),
        current,
        next_year,
        eligibility_frame(),
        league="dk",
        governed_exclusions={},
    )

    assert output["player_key"].tolist() == ["new-key", "other-key"]
    assert output["player"].tolist() == [
        "Tetairoa McMillan",
        "Deebo Samuel Sr.",
    ]
    assert set(audit["population_action"]) == {"added"}
    actions = master.set_index("player_key")["population_action"]
    assert actions["old-key"] == "dropped"
    assert actions["new-key"] == "added"
    assert output[list(CURRENT_RESIDUAL_COLUMNS)].eq(0).all().all()
    assert output[
        list(NEXT_RESIDUAL_SOURCE_COLUMNS.values())
    ].iloc[0].tolist() == [-5.0, -4.0, -2.0, 2.0, 4.0, 6.0]
    assert (
        output["current_uncertainty_source"]
        == "joint_weekly_template_only"
    ).all()
    assert (
        output["independent_current_residual_draw_allowed"] == 0
    ).all()


def test_aliases_resolve_core_and_market_labels_to_canonical_keys():
    aliases, identities = identity_frames()
    rows = pd.DataFrame(
        {
            "player": ["Tet Mcmillan", "Deebo Samuel"],
            "pos": ["WR", "WR"],
            "team": ["CAR", "WAS"],
        }
    )

    resolved = resolve_source_player_keys(
        rows,
        aliases,
        identities,
        year=2026,
        source_name="test_source",
    )

    assert resolved["player_key"].tolist() == ["new-key", "other-key"]
    assert resolved["eligibility_key_match_method"].tolist() == [
        "alias_confirmed_unique",
        "alias_confirmed_unique",
    ]


def test_unique_hybrid_identity_resolves_across_provider_position():
    aliases, identities = identity_frames()
    rows = pd.DataFrame(
        {
            "player": ["Old Player"],
            "pos": ["WR"],
            "team": ["OLD"],
        }
    )

    resolved = resolve_source_player_keys(
        rows,
        aliases,
        identities,
        year=2026,
        source_name="test_source",
    )

    assert resolved.loc[0, "player_key"] == "old-key"
    assert resolved.loc[0, "eligibility_key_match_method"] == (
        "alias_cross_position_team_confirmed_unique"
    )


def test_eligibility_is_deterministic_and_market_limit_is_canonical():
    aliases, identities = identity_frames()
    core = pd.DataFrame(
        {
            "player": ["Tet Mcmillan"],
            "pos": ["WR"],
            "team": ["CAR"],
        }
    )
    market = pd.DataFrame(
        {
            "player": [
                "Deebo Samuel Sr.",
                "Old Player",
                "Deebo Samuel",
            ],
            "avg_pick": [10.0, 10.0, 11.0],
        }
    )
    empty_keepers = pd.DataFrame(columns=["player"])

    first = build_eligibility_membership(
        core,
        market,
        empty_keepers,
        aliases,
        identities,
        league="dk",
        year=2026,
        market_limit=2,
        market_source_name="dk_adp",
    )
    second = build_eligibility_membership(
        core.sample(frac=1, random_state=1),
        market.sample(frac=1, random_state=2),
        empty_keepers,
        aliases.sample(frac=1, random_state=3),
        identities.sample(frac=1, random_state=4),
        league="dk",
        year=2026,
        market_limit=2,
        market_source_name="dk_adp",
    )

    pdt.assert_frame_equal(first, second)
    keyed = first.set_index("player_key")
    assert set(keyed.index) == {"new-key", "old-key", "other-key"}
    assert keyed.loc["other-key", "market_eligibility_rank"] == 1
    assert keyed.loc["old-key", "market_eligibility_rank"] == 2
    assert keyed.loc["new-key", "eligible_core_projonly"] == 1


@pytest.mark.parametrize("league", ["beta", "nv"])
def test_auction_eligibility_unions_keepers_outside_the_etr_limit(league):
    aliases, identities = identity_frames()
    core = pd.DataFrame(
        {"player": ["Tet Mcmillan"], "pos": ["WR"], "team": ["CAR"]}
    )
    market = pd.DataFrame(
        {"player": ["Old Player"], "avg_pick": [1.0]}
    )
    keepers = pd.DataFrame({"player": ["Deebo Samuel"]})

    membership = build_eligibility_membership(
        core,
        market,
        keepers,
        aliases,
        identities,
        league=league,
        year=2026,
        market_limit=1,
        market_source_name="etr_adp",
    ).set_index("player_key")

    assert set(membership.index) == {"new-key", "old-key", "other-key"}
    assert membership.loc["old-key", "eligible_etr_adp"] == 1
    assert membership.loc["other-key", "eligible_league_keeper"] == 1
    assert membership.loc["other-key", "production_player_label"] == (
        "Deebo Samuel"
    )


def test_nffc_eligibility_uses_its_own_keyed_market_flag():
    aliases, identities = identity_frames()
    core = pd.DataFrame(
        {"player": ["Tet Mcmillan"], "pos": ["WR"], "team": ["CAR"]}
    )
    market = pd.DataFrame(
        {
            "player_key": ["old-key", "other-key"],
            "player": ["Old Player", "Deebo Samuel"],
            "pos": ["RB", "WR"],
            "avg_pick": [1.0, 2.0],
            "identity_match_method": ["published_player_key"] * 2,
        }
    )

    membership = build_eligibility_membership(
        core,
        market,
        pd.DataFrame(columns=["player"]),
        aliases,
        identities,
        league="nffc",
        year=2026,
        market_limit=2,
        market_source_name="nffc_adp",
    ).set_index("player_key")

    assert set(membership.index) == {"new-key", "old-key", "other-key"}
    assert membership.loc["old-key", "eligible_nffc_adp"] == 1
    assert membership.loc["other-key", "eligible_nffc_adp"] == 1
    assert membership["eligible_dk_adp"].eq(0).all()
    assert membership["eligible_etr_adp"].eq(0).all()


def test_reviewed_dk_exclusion_policy_is_exact_and_versioned():
    assert PRODUCTION_EXCLUSION_POLICY_VERSION == (
        "v2_market_only_incomplete_buffer_exclusion_v4"
    )
    assert MARKET_HANDOFF_REQUIRED_DEPTH == {
        "dk": 240,
        "nffc": 360,
        "beta": 180,
        "nv": 180,
    }
    assert MARKET_HANDOFF_PROTECTED_PICK_DEPTH == {
        "dk": 200,
        "nffc": 300,
        "beta": 150,
        "nv": 150,
    }
    assert MARKET_ELIGIBILITY_RULES["nffc"] == (
        "nffc",
        363,
        "nffc_adp",
    )
    assert GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR[2026]["dk"] == {
        "ad848f28-4066-522c-b352-43abce87fbcb": (
            "season_ending_pcl_injury_adp_lag_without_current_"
            "projection_center"
        ),
        "b5862397-0d45-5560-8aa7-44b046621b96": (
            "market_only_without_current_projection_center"
        ),
        "3f0b675d-ef58-5606-8f9e-73bc2a9b4118": (
            "market_only_without_current_projection_center"
        ),
        "7ae33581-c9ae-51b6-a8d5-fe24f3e5615a": (
            "market_only_without_current_projection_center"
        ),
        "e492c31b-21c9-55b9-b007-4dd0d8fd1ad4": (
            "market_only_without_current_projection_center"
        ),
        "f973b1c8-3470-57f5-bc68-42e35a830411": (
            "market_only_without_current_projection_center"
        ),
        "380d2c7d-99ef-5ddc-a057-fab93f1480ba": (
            "market_only_without_current_projection_center"
        ),
    }
    nffc_exclusions = GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR[2026]["nffc"]
    assert len(nffc_exclusions) == 10
    assert set(nffc_exclusions.values()) == {
        "market_only_without_current_projection_center",
        (
            "season_ending_pcl_injury_adp_lag_without_current_"
            "projection_center"
        ),
    }
    assert nffc_exclusions[
        "ad848f28-4066-522c-b352-43abce87fbcb"
    ] == (
        "season_ending_pcl_injury_adp_lag_without_current_"
        "projection_center"
    )
    assert nffc_exclusions[
        "b5862397-0d45-5560-8aa7-44b046621b96"
    ] == "market_only_without_current_projection_center"
    assert "c5e3e9a4-cc91-5fc7-83f6-2367cbd3793b" not in nffc_exclusions
    assert "86efb1f0-e04a-5f4d-b8cb-048353f1d3f5" not in nffc_exclusions
    assert "06b12c47-18b2-51ac-ba66-64de763baac2" in nffc_exclusions
    assert "38e3ae60-9300-500c-8036-46d77358cd97" not in nffc_exclusions
    assert "31c3fcf7-3f74-524e-8b8f-67177f592742" in nffc_exclusions
    assert {
        "0fa72b32-393b-5f55-bb48-0f21f5283baf",
        "49dce437-30ec-5752-9739-75ed09f72042",
        "54c9ddab-abf4-5e3a-9ded-da4265515065",
    }.issubset(nffc_exclusions)
    assert "862eb067-7abb-5156-9cf1-33c3ad11333c" not in nffc_exclusions
    assert "89aacaaa-acba-5185-83b3-7b68130c4465" not in nffc_exclusions
    assert GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR[2026]["beta"] == {}
    assert GOVERNED_PRODUCTION_EXCLUSIONS_BY_YEAR[2026]["nv"] == {}


def test_missing_required_center_fails_unless_explicitly_excluded():
    current, next_year = shadow_frames()
    current.loc[current["player_key"].eq("new-key"), "conditional_ppg_shadow"] = (
        None
    )
    eligibility = eligibility_frame()
    eligibility.loc[
        eligibility["player_key"].eq("new-key"),
        [
            "eligibility_sources",
            "eligible_core_projonly",
            "eligible_dk_adp",
        ],
    ] = ["dk_adp", 0, 1]

    with pytest.raises(ValueError, match="eligibility-required.*incomplete"):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility,
            league="dk",
            governed_exclusions={},
        )

    output, _, master = build_production_projection_slice(
        legacy_frame(),
        current,
        next_year,
        eligibility,
        league="dk",
        governed_exclusions={
            "new-key": "reviewed_missing_current_center_fixture"
        },
    )
    assert output["player_key"].tolist() == ["other-key"]
    excluded = master.set_index("player_key").loc["new-key"]
    assert excluded["governed_excluded"] == 1
    assert (
        excluded["governed_exclusion_reason"]
        == "reviewed_missing_current_center_fixture"
    )
    assert (
        excluded["governed_exclusion_policy_version"]
        == "v2_market_only_incomplete_buffer_exclusion_v4"
    )


def test_incomplete_market_only_buffer_row_is_audited_not_fatal():
    current, next_year = shadow_frames()
    current.loc[current["player_key"].eq("new-key"), "conditional_ppg_shadow"] = (
        None
    )
    eligibility = eligibility_frame()
    eligibility.loc[
        eligibility["player_key"].eq("new-key"),
        [
            "eligibility_sources",
            "eligible_core_projonly",
            "eligible_dk_adp",
            "market_eligibility_rank",
            "market_eligibility_pick",
        ],
    ] = ["dk_adp", 0, 1, 243, 225.0]

    output, _, master = build_production_projection_slice(
        legacy_frame(),
        current,
        next_year,
        eligibility,
        league="dk",
        governed_exclusions={},
    )

    assert output["player_key"].tolist() == ["other-key"]
    excluded = master.set_index("player_key").loc["new-key"]
    assert excluded["governed_excluded"] == 1
    assert (
        excluded["governed_exclusion_reason"]
        == AUTOMATIC_MARKET_BUFFER_EXCLUSION_REASON
    )
    assert excluded["market_handoff_required_depth"] == 240
    assert excluded["market_handoff_protected_pick_depth"] == 200
    assert excluded["market_handoff_draft_position"] == 225.0
    assert excluded["automatic_market_buffer_exclusion"] == 1
    assert excluded["population_action"] == "governed_excluded"


def test_incomplete_market_only_row_inside_protected_depth_still_fails():
    current, next_year = shadow_frames()
    current.loc[current["player_key"].eq("new-key"), "conditional_ppg_shadow"] = (
        None
    )
    eligibility = eligibility_frame()
    eligibility.loc[
        eligibility["player_key"].eq("new-key"),
        [
            "eligibility_sources",
            "eligible_core_projonly",
            "eligible_dk_adp",
            "market_eligibility_rank",
            "market_eligibility_pick",
        ],
    ] = ["dk_adp", 0, 1, 200, 200.0]

    with pytest.raises(ValueError, match="eligibility-required.*incomplete"):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility,
            league="dk",
            governed_exclusions={},
        )


def test_automatic_tail_exclusion_cannot_underfill_draft(
    monkeypatch,
):
    current, next_year = shadow_frames()
    current.loc[current["player_key"].eq("new-key"), "conditional_ppg_shadow"] = (
        None
    )
    eligibility = eligibility_frame()
    eligibility.loc[
        eligibility["player_key"].eq("new-key"),
        [
            "eligibility_sources",
            "eligible_core_projonly",
            "eligible_dk_adp",
            "market_eligibility_rank",
            "market_eligibility_pick",
        ],
    ] = ["dk_adp", 0, 1, 2, 2.0]
    monkeypatch.setitem(MARKET_HANDOFF_REQUIRED_DEPTH, "dk", 2)
    monkeypatch.setitem(MARKET_HANDOFF_PROTECTED_PICK_DEPTH, "dk", 1)

    with pytest.raises(ValueError, match="required to cover the draft"):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility,
            league="dk",
            governed_exclusions={},
        )


def test_governed_exclusion_cannot_remove_complete_core_player():
    current, next_year = shadow_frames()

    with pytest.raises(
        ValueError,
        match="must be incomplete market-only rows",
    ):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility_frame(),
            league="dk",
            governed_exclusions={
                "new-key": "invalid_complete_core_exclusion"
            },
        )


def test_repeated_build_is_idempotent_and_preserves_creation_timestamp():
    current, next_year = shadow_frames()
    legacy = legacy_frame()
    eligibility = eligibility_frame()
    first, first_audit, first_master = build_production_projection_slice(
        legacy,
        current,
        next_year,
        eligibility,
        league="dk",
        governed_exclusions={},
    )
    second, second_audit, second_master = build_production_projection_slice(
        legacy,
        current.sample(frac=1, random_state=3),
        next_year.sample(frac=1, random_state=4),
        eligibility.sample(frac=1, random_state=5),
        league="dk",
        governed_exclusions={},
        prior_slice=first,
    )

    pdt.assert_frame_equal(first, second)
    pdt.assert_frame_equal(first_audit, second_audit)
    pdt.assert_frame_equal(first_master, second_master)


def test_nonmonotone_next_quantiles_still_fail_closed():
    current, next_year = shadow_frames()
    next_year.loc[
        next_year["player_key"].eq("new-key"),
        "pred_resid_25_ny_shadow",
    ] = -10.0

    with pytest.raises(ValueError, match="not monotone"):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility_frame(),
            league="dk",
            governed_exclusions={},
        )


def test_negative_center_and_wrong_next_target_fail_closed():
    current, next_year = shadow_frames()
    current.loc[
        current["player_key"].eq("new-key"),
        "conditional_ppg_shadow",
    ] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility_frame(),
            league="dk",
            governed_exclusions={},
        )

    current, next_year = shadow_frames()
    next_year.loc[
        next_year["player_key"].eq("new-key"),
        "target_season",
    ] = 2028
    with pytest.raises(ValueError, match="origin/target seasons"):
        build_production_projection_slice(
            legacy_frame(),
            current,
            next_year,
            eligibility_frame(),
            league="dk",
            governed_exclusions={},
        )
