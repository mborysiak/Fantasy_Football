from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-04_v2_logged_rank_disagreement"
    / "run_study.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("logged_rank_disagreement_study", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_logged_mad_prioritizes_middle_rank_spread() -> None:
    runner = _load_runner()
    assert runner.logged_rank_mad([30, 50]) > runner.logged_rank_mad([130, 170])


def test_logged_mad_zero_for_unanimous_number_one() -> None:
    runner = _load_runner()
    assert runner.logged_rank_mad([1, 1, 1, 1]) == 0


def test_logged_mad_single_source_is_missing() -> None:
    runner = _load_runner()
    assert np.isnan(runner.logged_rank_mad([30]))


def test_expected_mad_is_strictly_prior() -> None:
    runner = _load_runner()
    rows = []
    for season, mad in ((2020, 0.1), (2021, 0.2), (2022, 0.3)):
        for index in range(25):
            rows.append(
                {
                    "player_key": f"{season}_{index}",
                    "season": season,
                    "position": "RB",
                    "scoring_specific_rank_position_percentile_median": 0.75,
                    "expert_rank_logged_source_count": 3,
                    "expert_rank_logged_mad": mad,
                }
            )
    frame = pd.DataFrame(rows)
    original = runner._strictly_prior_expected_mad(frame)
    changed = frame.copy()
    changed.loc[changed["season"].eq(2022), "expert_rank_logged_mad"] = 99.0
    rerun = runner._strictly_prior_expected_mad(changed)
    columns = ["player_key", "expert_rank_logged_mad_expected_prior"]
    earlier_original = original[original["season"].le(2021)][columns].reset_index(drop=True)
    earlier_rerun = rerun[rerun["season"].le(2021)][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier_original, earlier_rerun)
