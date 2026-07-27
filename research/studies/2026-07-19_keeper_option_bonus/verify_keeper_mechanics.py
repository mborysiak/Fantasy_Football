"""Focused contracts for the keeper-option construction helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().with_name("run_replay.py")
spec = importlib.util.spec_from_file_location("keeper_option_replay_verify", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)


def main() -> None:
    future = np.array(
        [
            [25.0, 25.0],
            [25.0, 25.0],
            [8.0, 30.0],
        ]
    )
    prices = np.array([2.0, 15.0, 2.0])
    surplus = replay.keeper_contract_surplus(future, prices)
    assert np.allclose(surplus[0], [16.0, 16.0])
    assert np.allclose(surplus[1], [0.0, 0.0])
    assert np.allclose(surplus[2], [0.0, 26.0])

    options = surplus.mean(axis=1)
    selected = np.array([True, True, True, False])
    padded = np.array([options[0], options[1], options[2], 100.0])
    assert replay.top_keeper_utility(selected, padded) == options[0] + options[2]
    assert replay.top_keeper_indices(selected, padded).tolist() == [0, 2]

    current = np.array([5.0, 8.0, 10.0, 12.0])
    salaries = np.array([1.0, 4.0, 15.0, 40.0])
    positions = np.array(["RB", "RB", "RB", "RB"])
    curves = replay.fit_position_market_curves(
        np.tile(current, 4),
        np.tile(salaries, 4),
        np.repeat(np.array(replay.POSITIONS), 4),
    )
    transformed = curves["RB"].predict(np.array([5.0, 8.0, 10.0, 12.0]))
    assert np.all(np.diff(transformed) >= 0)
    assert transformed[0] >= 1.0

    print(
        {
            "cheap_25_value_surplus": float(surplus[0, 0]),
            "expensive_25_value_surplus": float(surplus[1, 0]),
            "lottery_mean_surplus": float(options[2]),
            "top_two_utility": float(replay.top_keeper_utility(selected, padded)),
        }
    )


if __name__ == "__main__":
    main()
