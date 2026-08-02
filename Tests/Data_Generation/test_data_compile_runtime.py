import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "Scripts"
    / "Data_Generation"
    / "4_Data_Compile.py"
)


def _load_function(name: str):
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "pd": pd}
    exec(compile(module, SCRIPT_PATH, "exec"), namespace)
    return namespace[name]


def test_lagged_rolling_stats_matches_grouped_rolling_reference():
    rolling_stats = _load_function("rolling_stats")
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "player": np.repeat(["c", "a", "b"], [7, 5, 9]),
            "year": list(range(7)) + list(range(5)) + list(range(9)),
            **{
                f"feature_{index}": rng.normal(size=21)
                for index in range(6)
            },
        }
    ).sort_values(["player", "year"]).reset_index(drop=True)
    columns = [f"feature_{index}" for index in range(6)]
    frame.loc[[0, 2, 5, 8, 14], ["feature_1", "feature_4"]] = np.nan

    for aggregation in ("mean", "max"):
        expected = (
            frame.groupby(["player"])[columns]
            .rolling(3, min_periods=1)
            .agg(aggregation)
            .reset_index(drop=True)
        )
        expected.columns = [
            f"r{aggregation}3_{column}"
            for column in expected.columns
        ]

        actual = rolling_stats(
            frame,
            ["player"],
            columns,
            3,
            agg_type=aggregation,
        )

        assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=1e-14,
            atol=1e-14,
        )


def test_forward_fill_matches_deprecated_grouped_fill_behavior():
    forward_fill = _load_function("forward_fill")
    frame = pd.DataFrame(
        {
            "player": ["b", "a", "a", "b"],
            "year": [2025, 2024, 2025, 2026],
            "numeric": [2.0, 1.0, np.nan, np.nan],
            "label": ["later", "early", None, None],
        }
    )
    expected = pd.DataFrame(
        {
            "player": ["a", "a", "b", "b"],
            "year": [2024, 2025, 2025, 2026],
            "numeric": [1.0, 1.0, 2.0, 2.0],
            "label": ["early", "early", "later", "later"],
        }
    )

    actual = forward_fill(frame)

    assert_frame_equal(actual.reset_index(drop=True), expected)
