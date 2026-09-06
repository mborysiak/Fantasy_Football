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
    / "2026-08-29_v2_stabilized_bull_gap"
    / "run_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "v2_stabilized_bull_gap_study_test", STUDY_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import study from {STUDY_PATH}")
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _provider_rows(values: list[float], player_key: str = "player") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": player_key,
            "season": 2025,
            "position": "WR",
            "provider": [f"source_{index}" for index in range(len(values))],
            "configured_points_complete": 1,
            "provider_points_per_team_game": values,
        }
    )


def test_primary_smooth_k5_formula_is_frozen() -> None:
    feature = study.build_stabilized_projection_features(
        _provider_rows([8.0, 10.0, 16.0])
    ).iloc[0]

    assert feature.expert_ppg_bull_gap == pytest.approx(6.0)
    assert feature.expert_ppg_bull_gap_smooth_k5 == pytest.approx(
        6.0 / np.sqrt(10.0**2 + 5.0**2)
    )
    assert feature.expert_ppg_bull_gap_smooth_k3 == pytest.approx(
        6.0 / np.sqrt(10.0**2 + 3.0**2)
    )
    assert feature.expert_ppg_bull_gap_smooth_k8 == pytest.approx(
        6.0 / np.sqrt(10.0**2 + 8.0**2)
    )
    assert feature.expert_ppg_bull_gap_hard_floor_k5 == pytest.approx(0.6)
    assert feature.expert_ppg_bull_gap_additive_k5 == pytest.approx(0.4)


def test_k5_stabilizes_low_consensus_fraction() -> None:
    feature = study.build_stabilized_projection_features(
        _provider_rows([0.2, 0.5, 2.0])
    ).iloc[0]

    assert feature.expert_ppg_bull_gap_fraction == pytest.approx(1.5)
    assert feature.expert_ppg_bull_gap_smooth_k5 == pytest.approx(
        1.5 / np.sqrt(0.5**2 + 5.0**2)
    )
    assert feature.expert_ppg_bull_gap_smooth_k5 < 0.30


def test_primary_variants_use_only_smooth_k5_plus_availability() -> None:
    assert study.POINT_VARIANT_FEATURES[study.PRIMARY_POINT_VARIANT] == (
        "expert_ppg_bull_gap_smooth_k5",
        "expert_ppg_bull_gap_available",
    )
    assert study.TAIL_VARIANT_FEATURES[study.PRIMARY_TAIL_VARIANT] == (
        "expert_ppg_bull_gap_smooth_k5",
        "expert_ppg_bull_gap_smooth_k5_position_percentile",
        "expert_ppg_bull_gap_available",
    )


class _Builder:
    POSITIONS = ["QB", "RB", "WR", "TE"]
    TEMPLATE_RECENCY_HALF_LIFE = 12.0
    MATCH_FEATURE_WEIGHTS = {
        position: {"projection_disagreement_frac": 0.75}
        for position in POSITIONS
    }


def test_template_primary_adds_frozen_weight_without_replacing_incumbent() -> None:
    methods = study.build_template_methods(_Builder)

    assert len(methods) == 1 + len(study.TEMPLATE_FEATURES)
    for position in _Builder.POSITIONS:
        assert methods["incumbent"]["weights"][position] == {
            "projection_disagreement_frac": 0.75
        }
        primary = methods[study.PRIMARY_TEMPLATE_METHOD]["weights"][position]
        assert primary["projection_disagreement_frac"] == 0.75
        assert (
            primary[
                "expert_ppg_bull_gap_smooth_k5_position_percentile"
            ]
            == 0.50
        )
        assert primary["expert_ppg_bull_gap_available"] == 0.25
