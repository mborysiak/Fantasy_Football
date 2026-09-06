import unittest

import numpy as np
import pandas as pd

from Scripts.Modeling import build_paired_breakout_templates as breakout


class PairedBreakoutTemplatesTest(unittest.TestCase):

    @staticmethod
    def donor_fixture(rows=50):
        data = {
            "profile_version": [breakout.PROFILE_VERSION] * rows,
            "league": ["beta"] * rows,
            "template_id": np.arange(1, rows + 1),
            "player_key": [f"donor_{idx}" for idx in range(rows)],
            "player": [f"Donor {idx}" for idx in range(rows)],
            "pos": ["WR"] * rows,
            "team": ["TST"] * rows,
            "origin_season": np.arange(1975, 1975 + rows),
            "next_target_season": np.arange(1976, 1976 + rows),
        }
        for feature in breakout.MATCH_WEIGHTS["WR"]:
            data[feature] = np.linspace(0.0, 1.0, rows)
        for column in (
            "current_ppg_residual",
            "current_needle_mover_points",
            "current_playoff_excess_points",
            "current_late_lift",
            "actual_next_appeared",
            "actual_next_unconditional_ppg",
            "actual_next_ppg_residual",
            "current_breakout_hit",
            "current_playoff_hit",
            "current_late_surge_hit",
            "future_high_performer_hit",
            "current_and_future_hit",
            "playoff_and_future_hit",
        ):
            data[column] = np.linspace(0.0, 1.0, rows)
        return pd.DataFrame(data)

    def test_signed_growth_is_not_capped_or_appearance_multiplied(self):
        targets = pd.DataFrame(
            {
                "player": ["Negative Growth", "Positive Growth"],
                "player_key": ["negative", "positive"],
                "pos": ["WR", "WR"],
                "year": [2026, 2026],
                "pred_fp_per_game": [8.0, 8.0],
                "pred_fp_per_game_ny": [7.5, 9.0],
                "pred_appear_ny": [0.2, 0.9],
            }
        )
        output = breakout.attach_current_next_context(targets)
        self.assertEqual(output.breakout_signed_next_growth.tolist(), [-0.5, 1.0])
        self.assertEqual(output.breakout_next_appearance.tolist(), [0.2, 0.9])
        self.assertEqual(output.breakout_next_growth_rank_pct.tolist(), [0.5, 1.0])

    def test_select_pool_keeps_salary_out_and_probabilities_normalized(self):
        donors = self.donor_fixture()
        target = pd.Series(
            {
                "player_key": "target",
                "player": "Target",
                "pos": "WR",
                "year": 2026,
                "version": "beta",
                **{feature: 0.5 for feature in breakout.MATCH_WEIGHTS["WR"]},
            }
        )
        pool = breakout.select_breakout_pool(target, donors, "final_ensemble")
        self.assertEqual(len(pool), 50)
        self.assertAlmostEqual(pool.template_sample_prob.sum(), 1.0)
        self.assertLessEqual(
            pool.template_sample_prob.max(),
            breakout.MAX_SAMPLE_PROBABILITY + 1e-12,
        )
        self.assertFalse(any("salary" in column.lower() for column in pool.columns))
        self.assertTrue(pool.template_season_gap.gt(0).all())

    def test_no_appearance_is_zero_future_value(self):
        feature_columns = sorted(
            {
                feature
                for weights in breakout.MATCH_WEIGHTS.values()
                for feature in weights
            }
            - {"breakout_next_growth_rank_pct", "breakout_next_appearance"}
        )
        donors = pd.DataFrame(
            {
                "league": ["beta", "beta"],
                "template_id": [1, 2],
                "player_key": ["a", "b"],
                "player": ["A", "B"],
                "pos": ["WR", "WR"],
                "team": ["A", "B"],
                "season": [2023, 2023],
                "avg_pick": [100.0, 110.0],
                "year_exp": [1.0, 2.0],
                "managed_profile_ppg": [10.0, 8.0],
                "managed_residual_center_ppg": [9.0, 8.0],
                "managed_active_ppg_resid": [1.0, 0.0],
                "active_ppg": [10.0, 8.0],
                "played_games": [2, 2],
                "active_games": [2, 2],
                "managed_week_1": [1.0, 1.0],
                "managed_week_14": [2.0, 0.5],
                **{feature: [0.5, 0.5] for feature in feature_columns},
            }
        )
        handoff = pd.DataFrame(
            {
                "player_key": ["a", "b"],
                "origin_season": [2023, 2023],
                "target_season": [2024, 2024],
                "position": ["WR", "WR"],
                "predicted_next_year_conditional_ppg": [11.0, 7.5],
                "predicted_next_year_appearance_probability": [0.9, 0.4],
                "training_through_origin": [2021, 2021],
                "target_outcome_through": [2022, 2022],
                "forecast_status": ["causal", "causal"],
            }
        )
        outcomes = pd.DataFrame(
            {
                "player_key": ["a", "b"],
                "origin_season": [2023, 2023],
                "target_season": [2024, 2024],
                "position": ["WR", "WR"],
                "next_participation_target_available": [1, 1],
                "next_appeared": [1, 0],
                "next_conditional_ppg": [12.0, np.nan],
                "next_conditional_ppg_training_eligible": [1, 0],
                "next_target_join_status": ["observed_appearance", "no_appearance"],
            }
        )
        paired, _ = breakout.build_paired_donors(
            donors,
            handoff,
            outcomes,
            [1, 14],
            "beta",
        )
        no_appearance = paired.loc[paired.player_key.eq("b")].iloc[0]
        self.assertEqual(no_appearance.actual_next_unconditional_ppg, 0.0)
        self.assertTrue(pd.isna(no_appearance.actual_next_conditional_ppg))
        self.assertEqual(no_appearance.breakout_signed_next_growth, -0.5)
        appeared = paired.loc[paired.player_key.eq("a")].iloc[0]
        self.assertEqual(appeared.actual_next_ppg_residual, 1.0)
        self.assertEqual(appeared.current_playoff_calendar_ppg, 20.0)


if __name__ == "__main__":
    unittest.main()


def test_research_builder_requires_separate_output(tmp_path):
    import pytest

    source = tmp_path / "Simulation.sqlite3"
    source.touch()
    with pytest.raises(SystemExit):
        breakout.parse_args([])
    with pytest.raises(ValueError, match="separate research"):
        breakout.validate_research_output(source, source)
    for output in (
        breakout.DEFAULT_SIMULATION_DB,
        breakout.REPO_ROOT.parent / "Fantasy_Football_App/app/Simulation.sqlite3",
        breakout.REPO_ROOT.parent / "Fantasy_Football_Snake/app/Simulation.sqlite3",
    ):
        with pytest.raises(ValueError, match="separate research"):
            breakout.validate_research_output(output, source)
    destination = tmp_path / "research" / "breakout.sqlite3"
    assert breakout.validate_research_output(destination, source) == destination
