from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "research"
    / "studies"
    / "2026-08-27_auction_power_win_objective"
    / "run_power_win_test.py"
)
SPEC = importlib.util.spec_from_file_location("auction_power_win_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_expected_excess_rewards_larger_winning_margin_at_equal_win_rate():
    references = np.full((4, 3), 100.0)
    candidates = np.array([
        [110.0, 110.0, 110.0],
        [200.0, 200.0, 200.0],
    ])
    utility = MODULE.field_utility_cells(candidates, references)
    assert np.allclose(utility["win_probability"][0], utility["win_probability"][1])
    assert np.all(utility["expected_excess"][1] > utility["expected_excess"][0])
    assert np.all(utility["power_utility"][1] > utility["power_utility"][0])


def test_dominant_win_requires_configured_margin():
    references = np.full((4, 2), 100.0)
    candidates = np.array([
        [140.0, 140.0],
        [160.0, 160.0],
    ])
    utility = MODULE.field_utility_cells(
        candidates, references, dominant_margin=50.0
    )
    assert np.allclose(utility["dominant_win_probability"][0], 0.0)
    assert np.allclose(utility["dominant_win_probability"][1], 1.0)


def test_tail_choice_respects_mean_guardrail_and_paired_lcb():
    metrics = pd.DataFrame([
        {
            "roster_key": "mean",
            "construction_mean": 1600.0,
            "construction_win_probability": 0.10,
            "construction_win_probability_delta_lcb80_vs_mean": 0.0,
        },
        {
            "roster_key": "near",
            "construction_mean": 1593.0,
            "construction_win_probability": 0.15,
            "construction_win_probability_delta_lcb80_vs_mean": 0.02,
        },
        {
            "roster_key": "far",
            "construction_mean": 1580.0,
            "construction_win_probability": 0.30,
            "construction_win_probability_delta_lcb80_vs_mean": 0.10,
        },
    ])
    selected = MODULE.choose_tail_candidate(
        metrics, "win_probability", guardrail=0.005
    )
    assert selected.roster_key == "near"


def test_direct_choice_uses_point_estimate_inside_guardrail():
    metrics = pd.DataFrame([
        {
            "roster_key": "mean",
            "construction_mean": 1600.0,
            "construction_expected_excess": 8.0,
        },
        {
            "roster_key": "near",
            "construction_mean": 1586.0,
            "construction_expected_excess": 12.0,
        },
        {
            "roster_key": "far",
            "construction_mean": 1580.0,
            "construction_expected_excess": 20.0,
        },
    ])
    selected = MODULE.choose_direct_candidate(
        metrics, "expected_excess", guardrail=0.01
    )
    assert selected.roster_key == "near"


def test_pure_choice_ignores_mean_and_maximizes_tail_metric():
    metrics = pd.DataFrame([
        {
            "roster_key": "mean",
            "construction_mean": 1600.0,
            "construction_win_probability": 0.10,
        },
        {
            "roster_key": "tail",
            "construction_mean": 1500.0,
            "construction_win_probability": 0.25,
        },
    ])
    selected = MODULE.choose_pure_candidate(metrics, "win_probability")
    assert selected.roster_key == "tail"


def test_half_mean_choice_uses_equal_standardized_components():
    metrics = pd.DataFrame([
        {
            "roster_key": "mean_only",
            "construction_mean": 10.0,
            "construction_expected_excess": 0.0,
        },
        {
            "roster_key": "tail_only",
            "construction_mean": 0.0,
            "construction_expected_excess": 10.0,
        },
        {
            "roster_key": "balanced",
            "construction_mean": 6.0,
            "construction_expected_excess": 6.0,
        },
    ])
    selected = MODULE.choose_half_mean_candidate(metrics, "expected_excess")
    assert selected.roster_key == "balanced"
