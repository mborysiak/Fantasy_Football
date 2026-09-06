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
    / "2026-08-27_auction_championship_waiver_objective"
    / "run_paired_test.py"
)
SPEC = importlib.util.spec_from_file_location("auction_championship_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_empirical_championship_proxy_prefers_dominating_roster():
    reference = np.array([
        [100.0, 110.0, 120.0],
        [110.0, 120.0, 130.0],
        [120.0, 130.0, 140.0],
    ])
    low = MODULE.empirical_championship_proxy(reference[0], reference)
    high = MODULE.empirical_championship_proxy(reference[2], reference)
    assert high > low
    assert 0 <= low <= 1
    assert 0 <= high <= 1


def test_lexicographic_choice_respects_mean_guardrail():
    metrics = pd.DataFrame([
        {
            "roster_key": "mean",
            "construction_mean": 1600.0,
            "construction_championship_proxy": 0.05,
            "construction_prob_two_difference_makers": 0.10,
        },
        {
            "roster_key": "near",
            "construction_mean": 1597.0,
            "construction_championship_proxy": 0.20,
            "construction_prob_two_difference_makers": 0.20,
        },
        {
            "roster_key": "too_far",
            "construction_mean": 1595.0,
            "construction_championship_proxy": 0.90,
            "construction_prob_two_difference_makers": 0.90,
        },
    ])
    selected = MODULE.choose_lexicographic_candidate(metrics)
    assert selected.roster_key == "near"


def test_difference_maker_event_requires_residual_and_contribution():
    predictions = pd.DataFrame({
        "player": ["Upside", "Volume Only"],
        "pos": ["RB", "RB"],
        "pred_fp_per_game": [10.0, 10.0],
    })
    scores = np.array([[
        [20.0, 20.0, 20.0, 20.0],
        [14.0, 14.0, 14.0, 14.0],
    ]])
    played = np.ones_like(scores, dtype=np.int8)
    events = MODULE.difference_maker_events(
        scores,
        played,
        predictions,
        {"QB": 1, "RB": 40, "WR": 1, "TE": 1},
    )
    assert events.tolist() == [[True, False]]
