"""Focused contracts for one-year, best-one keeper portfolio utility."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "run_replay.py"
SPEC = importlib.util.spec_from_file_location("one_year_keeper_replay", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def main() -> None:
    future_values = np.array(
        [
            [8.0, 30.0, 8.0, 30.0],
            [8.0, 8.0, 25.0, 25.0],
            [40.0, 8.0, 8.0, 8.0],
        ]
    )
    prices = np.array([2.0, 5.0, 35.0])
    surplus = module.first_year_surplus_draws(future_values, prices)
    expected_best = module.portfolio_utility(
        np.array([0, 1, 2]), surplus, "expected_best"
    )
    probability_ten = module.portfolio_utility(
        np.array([0, 1, 2]), surplus, "probability_10"
    )
    expected_top_two = module.expected_top_two_utility(
        np.array([0, 1, 2]), surplus
    )

    # Player 3 costs $45 next year and therefore never pays off.
    assert np.allclose(surplus[2], 0.0)
    # Best draw surplus is [0, 18, 10, 18].
    assert np.isclose(expected_best, 11.5)
    assert np.isclose(probability_ten, 0.75)
    # Top-two draw totals are [0, 18, 10, 28].
    assert np.isclose(expected_top_two, 14.0)

    print(
        {
            "expected_best_surplus": expected_best,
            "probability_any_10": probability_ten,
            "expected_top_two_surplus": expected_top_two,
        }
    )


if __name__ == "__main__":
    main()
