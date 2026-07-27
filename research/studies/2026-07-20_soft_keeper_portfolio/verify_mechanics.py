"""Focused mechanics checks for the soft keeper portfolio."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).with_name("run_replay.py")
spec = importlib.util.spec_from_file_location("soft_keeper_verify", RUNNER)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> None:
    # One dominant player and two complementary winners should produce an
    # effective count between one and three without assigning role labels.
    bench = np.array([True, True, True, False])
    surplus = np.array(
        [
            [12.0, 0.0, 0.0, 0.0],
            [5.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 20.0],
        ]
    )
    players = np.array(["A", "B", "C", "D"], dtype=object)
    metrics = module.option_concentration_metrics(bench, surplus, players)
    assert metrics["option_active_count_5pct"] == 3
    np.testing.assert_allclose(metrics["option_effective_count"], 3.0)
    np.testing.assert_allclose(metrics["option_positive_draw_rate"], 0.75)

    # Expected-best values the complementary portfolio, not the additive sum.
    utility = module.portfolio.portfolio_utility(
        np.flatnonzero(bench), surplus, "expected_best"
    )
    np.testing.assert_allclose(utility, 7.5)
    assert utility < surplus[bench].mean(axis=1).sum()
    assert module.POLICIES == ("control", "soft_portfolio")
    print("soft keeper portfolio mechanics: PASS")


if __name__ == "__main__":
    main()
