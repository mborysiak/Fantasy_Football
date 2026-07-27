"""Focused synthetic checks for bench option and hurdle mechanics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RUNNER = STUDY_DIR / "run_replay.py"
spec = importlib.util.spec_from_file_location("bench_option_hurdle_runner", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load runner: {RUNNER}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def option_fixture(paths: list[list[float]]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    weekly = np.asarray(paths, dtype=np.float32)[:, None, :]
    played = np.ones_like(weekly, dtype=np.int8)
    predictions = pd.DataFrame(
        {
            "player": ["Test Rookie"],
            "pos": ["RB"],
            "salary": [2.0],
            "draw_0": [8.0],
            "draw_1": [8.0],
        }
    )
    return weekly, played, predictions


def main() -> None:
    baseline = {"QB": 15.0, "RB": 7.0, "WR": 7.0, "TE": 6.0}
    raised = runner.construction_waiver(baseline, 2.0)
    assert raised == {"QB": 15.0, "RB": 9.0, "WR": 9.0, "TE": 8.0}

    thresholds = {"QB": 20.0, "RB": 12.0, "WR": 12.0, "TE": 10.0}
    steady = [[8.0] * 16]
    weekly, played, predictions = option_fixture(steady)
    steady_value = runner.sustained_option_bank(
        weekly,
        played,
        predictions,
        thresholds,
    )[0, 0]
    assert steady_value == 0.0

    late_breakout = [[5.0] * 7 + [15.0] * 9]
    weekly, played, predictions = option_fixture(late_breakout)
    late_value = runner.sustained_option_bank(
        weekly,
        played,
        predictions,
        thresholds,
    )[0, 0]
    assert late_value > 0.0

    isolated_spike = [[5.0] * 6 + [30.0] + [5.0] * 9]
    weekly, played, predictions = option_fixture(isolated_spike)
    spike_value = runner.sustained_option_bank(
        weekly,
        played,
        predictions,
        thresholds,
    )[0, 0]
    assert spike_value == 0.0

    early_breakout = [[15.0] * 16]
    weekly, played, predictions = option_fixture(early_breakout)
    early_value = runner.sustained_option_bank(
        weekly,
        played,
        predictions,
        thresholds,
    )[0, 0]
    assert early_value > late_value

    print(
        {
            "steady_value": float(steady_value),
            "late_breakout_value": float(late_value),
            "isolated_spike_value": float(spike_value),
            "early_breakout_value": float(early_value),
            "raised_waiver": raised,
        }
    )


if __name__ == "__main__":
    main()

