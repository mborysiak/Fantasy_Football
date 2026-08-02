from __future__ import annotations

from dataclasses import replace

import pytest

from Scripts.V2 import production_cycle as cycle_module
from Scripts.V2.contracts import scoring_hash
from Scripts.V2.production_cycle import (
    DEFAULT_PRODUCTION_YEAR,
    PRODUCTION_LEAGUES,
    get_production_cycle,
)


def test_2026_cycle_is_explicit_and_complete():
    cycle = get_production_cycle(2026)

    assert DEFAULT_PRODUCTION_YEAR == 2026
    assert cycle.status == "approved"
    assert cycle.current_season == 2026
    assert cycle.next_season == 2027
    assert cycle.leagues == PRODUCTION_LEAGUES == ("dk", "nffc", "beta")
    assert cycle.current_shadow_table == "locked_2026_shadow_predictions"
    assert cycle.next_shadow_table == "next_year_2027_shadow_predictions"
    assert set(cycle.locked_versions) == set(PRODUCTION_LEAGUES)
    assert cycle.next_target_version == "v2_next_year_expert_residual_v1"
    assert cycle.source_market_minimums["nffc"] == 360
    assert cycle.nffc_source_feed_minimums == {
        "nffc_rotowire_online": 400,
        "nffc_best_ball_overall": 400,
        "nffc_best_ball_25s50s": 400,
        "nffc_cutline": 250,
    }
    assert cycle.production_population_minimums["nffc"] == 360
    assert cycle.weekly_horizons == {"dk": 16, "nffc": 17, "beta": 16}
    assert cycle.template_min_seasons["nffc"] == 2021
    assert cycle.template_center_policies["nffc"] == (
        "nffc_scored_expert_consensus",
    )
    assert cycle.template_context_sources["nffc"] == (
        "v2_nffc_scoring_matched_preseason"
    )


def test_cycle_contract_hash_is_deterministic():
    cycle = get_production_cycle(2026)

    assert cycle.receipt()["nffc_source_feed_minimums"] == {
        "nffc_rotowire_online": 400,
        "nffc_best_ball_overall": 400,
        "nffc_best_ball_25s50s": 400,
        "nffc_cutline": 250,
    }
    assert cycle.contract_sha256() == cycle.contract_sha256()
    assert len(cycle.contract_sha256()) == 64


def test_cycle_rejects_invalid_nffc_source_feed_floor(monkeypatch):
    cycle = get_production_cycle(2026)
    invalid = replace(
        cycle,
        nffc_source_feed_minimums={
            **cycle.nffc_source_feed_minimums,
            "nffc_cutline": 0,
        },
    )
    monkeypatch.setitem(
        cycle_module.APPROVED_PRODUCTION_CYCLES,
        2026,
        invalid,
    )

    with pytest.raises(ValueError, match="invalid NFFC source-feed floors"):
        get_production_cycle(2026)


def test_unregistered_year_fails_closed():
    with pytest.raises(ValueError, match="not an approved production cycle"):
        get_production_cycle(2027)


def test_unknown_scoring_league_does_not_fall_back_to_nffc():
    with pytest.raises(ValueError, match="Unknown scoring league"):
        scoring_hash("nfcc")
