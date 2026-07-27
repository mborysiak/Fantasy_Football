"""Focused contract checks for the keeper-reinvestment replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).with_name("run_replay.py")
spec = importlib.util.spec_from_file_location("keeper_reinvestment_verify", RUNNER)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> None:
    assert [policy.max_forced_options for policy in module.POLICIES] == [0, 1, 2, 3]

    # Three roster players: one starter and two bench players. Candidate C adds
    # value only in the second draw, so expected-best marginal value is positive.
    selected = np.array([True, True, True, False, False])
    positions = np.array(["QB", "RB", "WR", "RB", "WR"], dtype=object)
    current_ppg = np.array([20.0, 14.0, 13.0, 8.0, 7.0])
    names = np.array(["A", "B", "C", "D", "E"], dtype=object)
    surplus = np.array(
        [
            [0.0, 0.0],
            [4.0, 4.0],
            [3.0, 3.0],
            [4.0, 12.0],
            [5.0, 5.0],
        ]
    )
    market = np.array([20.0, 10.0, 8.0, 2.0, 1.0])
    ranked = module.ranked_option_candidates(
        selected,
        surplus,
        positions,
        market,
        current_ppg,
        names,
        shortlist=2,
    )
    assert ranked.tolist() == [3, 4]

    future_market = np.array([[20.0, 8.0], [30.0, 15.0]])
    prices = np.array([5.0, 12.0])
    actual = module.portfolio.first_year_surplus_draws(future_market, prices)
    expected = np.array([[5.0, 0.0], [8.0, 0.0]])
    np.testing.assert_allclose(actual, expected)
    print("keeper reinvestment mechanics: PASS")


if __name__ == "__main__":
    main()

