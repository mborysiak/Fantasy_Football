from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-29_v2_asymmetric_expert_projection"
    / "run_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "v2_asymmetric_expert_projection_study_test",
    STUDY_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import study from {STUDY_PATH}")
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _rows(values_by_player: dict[str, list[float]]) -> pd.DataFrame:
    records = []
    for player_key, values in values_by_player.items():
        for index, value in enumerate(values, start=1):
            records.append(
                {
                    "player_key": player_key,
                    "season": 2025,
                    "position": "WR",
                    "provider": f"source_{index}",
                    "configured_points_complete": 1,
                    "provider_points_per_team_game": value,
                }
            )
    return pd.DataFrame(records)


def test_builds_one_sided_and_robust_gaps() -> None:
    features = study.build_asymmetric_projection_features(
        _rows({"wide": [8.0, 10.0, 16.0], "tight": [9.0, 10.0, 11.0]})
    ).set_index("player_key")

    wide = features.loc["wide"]
    assert wide.expert_ppg_gap_provider_count == 3
    assert wide.expert_ppg_bull_gap_available == 1
    assert wide.expert_ppg_bull_gap == pytest.approx(6.0)
    assert wide.expert_ppg_bear_gap == pytest.approx(2.0)
    assert wide.expert_ppg_top2_gap == pytest.approx(3.0)
    assert wide.expert_ppg_bull_gap_fraction == pytest.approx(0.6)
    assert wide.expert_ppg_bear_gap_fraction == pytest.approx(0.2)
    assert wide.expert_ppg_top2_gap_fraction == pytest.approx(0.3)
    assert wide.expert_ppg_gap_asymmetry_fraction == pytest.approx(0.4)
    assert wide.expert_ppg_bull_gap_position_percentile == pytest.approx(1.0)
    assert features.loc[
        "tight", "expert_ppg_bull_gap_position_percentile"
    ] == pytest.approx(0.5)


def test_two_provider_row_is_unknown_not_false_agreement() -> None:
    features = study.build_asymmetric_projection_features(
        _rows({"two_sources": [8.0, 12.0]})
    ).iloc[0]

    assert features.expert_ppg_gap_provider_count == 2
    assert features.expert_ppg_bull_gap_available == 0
    assert np.isnan(features.expert_ppg_bull_gap)
    assert np.isnan(features.expert_ppg_bear_gap)
    assert np.isnan(features.expert_ppg_top2_gap)
    assert np.isnan(features.expert_ppg_bull_gap_position_percentile)


def test_rejects_duplicate_player_season_provider() -> None:
    rows = _rows({"duplicate": [8.0, 10.0, 12.0]})
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate keys"):
        study.build_asymmetric_projection_features(rows)


def test_same_player_season_can_have_distinct_positions() -> None:
    rows = _rows({"multi_position": [8.0, 10.0, 12.0]})
    second = rows.copy()
    second["position"] = "RB"
    second["provider_points_per_team_game"] += 1.0

    features = study.build_asymmetric_projection_features(
        pd.concat([rows, second], ignore_index=True)
    )

    assert len(features) == 2
    assert set(features["position"]) == {"RB", "WR"}


class _Builder:
    POSITIONS = ["QB", "RB", "WR", "TE"]
    TEMPLATE_RECENCY_HALF_LIFE = 12.0
    MATCH_FEATURE_WEIGHTS = {
        position: {
            "projection_disagreement_frac": 0.75,
            "match_projection_rank_pct": 2.5,
        }
        for position in POSITIONS
    }


def test_template_variants_preserve_incumbent_and_freeze_gap_weights() -> None:
    methods = study.build_template_methods(_Builder)

    for position in _Builder.POSITIONS:
        assert methods["incumbent"]["weights"][position] == (
            _Builder.MATCH_FEATURE_WEIGHTS[position]
        )
        primary = methods[study.PRIMARY_TEMPLATE_METHOD]["weights"][position]
        assert primary["expert_ppg_bull_gap_position_percentile"] == 0.50
        assert primary["expert_ppg_bull_gap_available"] == 0.25
        replacement = methods["bull_replace_symmetric_w075"]["weights"][position]
        assert "projection_disagreement_frac" not in replacement
        assert replacement["expert_ppg_bull_gap_position_percentile"] == 0.75


def test_point_primary_is_normalized_max_minus_median() -> None:
    assert study.POINT_VARIANT_FEATURES[study.PRIMARY_POINT_VARIANT] == (
        "expert_ppg_bull_gap_fraction",
        "expert_ppg_bull_gap_available",
    )
    assert "projection_provider_count" in study.PRIMARY_PPG_FEATURES
