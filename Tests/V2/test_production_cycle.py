from __future__ import annotations

from dataclasses import replace

import pytest

from Scripts.V2 import production_cycle as cycle_module
from Scripts.V2.contracts import configured_scoring, scoring_hash
from Scripts.V2.production_cycle import (
    DEFAULT_PRODUCTION_YEAR,
    PRODUCTION_LEAGUES,
    get_historical_replay_template_contract,
    get_production_cycle,
)


def test_2026_cycle_is_explicit_and_complete():
    cycle = get_production_cycle(2026)

    assert DEFAULT_PRODUCTION_YEAR == 2026
    assert cycle.status == "approved"
    assert cycle.current_season == 2026
    assert cycle.next_season == 2027
    assert cycle.leagues == PRODUCTION_LEAGUES == (
        "dk",
        "nffc",
        "beta",
        "nv",
    )
    assert cycle.current_shadow_table == "locked_2026_shadow_predictions"
    assert cycle.next_shadow_table == "next_year_2027_shadow_predictions"
    assert set(cycle.locked_versions) == set(PRODUCTION_LEAGUES)
    assert cycle.next_target_version == "v2_next_year_expert_residual_v1"
    assert cycle.source_market_minimums["nffc"] == 360
    assert cycle.nffc_source_feed_minimums == {
        "nffc_best_ball_overall": 400,
        "nffc_best_ball_25s50s": 400,
    }
    assert cycle.nffc_source_feed_pick_boundaries == {
        "nffc_best_ball_overall": 360,
        "nffc_best_ball_25s50s": 360,
    }
    assert cycle.production_population_minimums["nffc"] == 360
    assert cycle.weekly_horizons == {
        "dk": 16,
        "nffc": 17,
        "beta": 16,
        "nv": 16,
    }
    assert cycle.template_min_seasons["nffc"] == 2021
    assert cycle.template_center_policies["nffc"] == (
        "nffc_scored_expert_consensus",
    )
    assert cycle.template_center_policies["beta"] == (
        "legacy_validated_oos",
        "beta_scored_expert_fallback",
    )
    assert cycle.template_center_policies["nv"] == (
        "nv_scored_expert_consensus",
        "preseason_projection_fallback",
    )
    assert cycle.template_context_sources["nffc"] == (
        "v2_nffc_scoring_matched_preseason"
    )
    assert cycle.template_context_sources["beta"] == (
        "v2_beta_scoring_matched_preseason"
    )
    assert cycle.template_context_sources["nv"] == (
        "v2_nv_scoring_matched_preseason"
    )


def test_cycle_contract_hash_is_deterministic():
    cycle = get_production_cycle(2026)

    assert cycle.receipt()["nffc_source_feed_minimums"] == {
        "nffc_best_ball_overall": 400,
        "nffc_best_ball_25s50s": 400,
    }
    assert cycle.receipt()["nffc_source_feed_pick_boundaries"] == {
        "nffc_best_ball_overall": 360,
        "nffc_best_ball_25s50s": 360,
    }
    assert cycle.contract_sha256() == cycle.contract_sha256()
    assert len(cycle.contract_sha256()) == 64


def test_nv_scoring_matches_beta_except_for_one_point_per_passing_td():
    beta = configured_scoring("beta")
    nv = configured_scoring("nv")

    assert nv["rushing"] == beta["rushing"]
    assert nv["receiving"] == beta["receiving"]
    assert {
        key: value
        for key, value in nv["passing"].items()
        if key != "pass_pass_touchdown_sum"
    } == {
        key: value
        for key, value in beta["passing"].items()
        if key != "pass_pass_touchdown_sum"
    }
    assert beta["passing"]["pass_pass_touchdown_sum"] == 5
    assert nv["passing"]["pass_pass_touchdown_sum"] == 4


def test_cycle_rejects_invalid_nffc_source_feed_floor(monkeypatch):
    cycle = get_production_cycle(2026)
    invalid = replace(
        cycle,
        nffc_source_feed_minimums={
            **cycle.nffc_source_feed_minimums,
            "nffc_best_ball_overall": 0,
        },
    )
    monkeypatch.setitem(
        cycle_module.APPROVED_PRODUCTION_CYCLES,
        2026,
        invalid,
    )

    with pytest.raises(ValueError, match="invalid NFFC source-feed floors"):
        get_production_cycle(2026)


def test_cycle_rejects_invalid_nffc_source_feed_boundary(monkeypatch):
    cycle = get_production_cycle(2026)
    invalid = replace(
        cycle,
        nffc_source_feed_pick_boundaries={
            **cycle.nffc_source_feed_pick_boundaries,
            "nffc_best_ball_overall": 0,
        },
    )
    monkeypatch.setitem(
        cycle_module.APPROVED_PRODUCTION_CYCLES,
        2026,
        invalid,
    )

    with pytest.raises(
        ValueError,
        match="invalid NFFC source-feed pick boundaries",
    ):
        get_production_cycle(2026)


def test_unregistered_year_fails_closed():
    with pytest.raises(ValueError, match="not an approved production cycle"):
        get_production_cycle(2027)


def test_2025_historical_template_contract_is_not_a_production_cycle():
    replay = get_historical_replay_template_contract(2025)

    assert replay.status == "historical_replay_only"
    assert replay.target_year == 2025
    assert replay.leagues == ("beta",)
    assert replay.weekly_horizons == {"beta": 16}
    assert replay.template_min_seasons == {"beta": 2008}
    assert replay.template_center_policies == {
        "beta": (
            "legacy_validated_oos",
            "beta_scored_expert_fallback",
        )
    }
    assert len(replay.contract_sha256()) == 64

    with pytest.raises(ValueError, match="not an approved production cycle"):
        get_production_cycle(2025)


def test_unregistered_historical_replay_fails_closed():
    with pytest.raises(ValueError, match="no historical replay template contract"):
        get_historical_replay_template_contract(2021)


def test_unknown_scoring_league_does_not_fall_back_to_nffc():
    with pytest.raises(ValueError, match="Unknown scoring league"):
        scoring_hash("nfcc")
