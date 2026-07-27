import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
APP_DIR = REPO_ROOT.parent / 'Fantasy_Football_App' / 'app'
RESULTS_DIR = STUDY_DIR / 'results'
sys.path.insert(0, str(APP_DIR))

from zSim_Helper import FootballSimulation, MAX_TARGET_WORKERS  # noqa: E402
from profile_target import run_serial  # noqa: E402


def assert_lineup_result_equal(actual, expected, fixture_idx):
    for key, expected_value in expected.items():
        if isinstance(expected_value, dict):
            for subkey, expected_array in expected_value.items():
                np.testing.assert_array_equal(
                    actual[key][subkey],
                    expected_array,
                    err_msg=(
                        f'fixture={fixture_idx}, key={key}, subkey={subkey}'
                    ),
                )
        else:
            np.testing.assert_array_equal(
                actual[key],
                expected_value,
                err_msg=f'fixture={fixture_idx}, key={key}',
            )


def verify_lineup_fixtures():
    fixtures = np.load(
        RESULTS_DIR / 'managed_base_lineup_fixtures.npy',
        allow_pickle=True,
    )
    for fixture_idx, fixture in enumerate(fixtures):
        actual = FootballSimulation.managed_base_lineup_state(
            fixture['scores'],
            fixture['positions'],
            fixture['decision_scores'],
            fixture['player_names'],
            lineup_require=fixture['lineup_require'],
            waiver_baselines=fixture['waiver_baselines'],
            inactive_score_threshold=fixture['inactive_score_threshold'],
        )
        assert_lineup_result_equal(actual, fixture['result'], fixture_idx)
    return len(fixtures)


def verify_target_board():
    before = pd.read_csv(RESULTS_DIR / 'target_pre_optimization.csv')
    after = pd.read_csv(RESULTS_DIR / 'target_post_optimization.csv')
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    current, _ = run_serial(50)
    pd.testing.assert_frame_equal(
        after,
        current[after.columns],
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )
    if 'ContributionN' not in current:
        raise AssertionError('Target output is missing ContributionN.')
    selected_trials = np.rint(current.SelectionCounts * 50 / 100)
    if not np.all(current.ContributionN <= selected_trials):
        raise AssertionError('ContributionN exceeds selected Target trials.')
    return len(before)


def verify_batch_marginals(cases=200):
    rng = np.random.default_rng(20260711)
    positions = np.array(
        ['QB'] * 8 + ['RB'] * 18 + ['WR'] * 20 + ['TE'] * 10,
        dtype=object,
    )
    names = np.array([f'P{idx}' for idx in range(len(positions))], dtype=object)
    waiver_baselines = {'QB': 14.0, 'RB': 6.5, 'WR': 6.5, 'TE': 5.0}
    lineup_require = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}

    for case_idx in range(cases):
        num_weeks = int(rng.integers(1, 21))
        case_positions = positions.copy()
        rng.shuffle(case_positions)
        scores = np.maximum(
            0,
            rng.normal(8, 7, size=(len(names), num_weeks)),
        ).astype(np.float32)
        scores[rng.random(scores.shape) < 0.18] = 0
        decisions = (
            scores * rng.uniform(0.2, 0.8)
            + rng.normal(4, 3, size=scores.shape)
        ).astype(np.float32)
        if case_idx % 4 == 0:
            decisions = np.round(decisions, 1)

        base_sets = []
        for _ in range(int(rng.integers(1, 15))):
            base_count = int(rng.integers(0, 14))
            base_sets.append(
                names[
                    rng.choice(len(names), size=base_count, replace=False)
                ].tolist()
            )

        batched = FootballSimulation.managed_marginal_values_batch(
            scores,
            case_positions,
            decisions,
            names,
            base_sets,
            waiver_baselines=waiver_baselines,
            lineup_require=lineup_require,
        )
        scalar = np.stack([
            FootballSimulation.managed_marginal_values(
                scores,
                case_positions,
                decisions,
                names,
                base_players=base_players,
                waiver_baselines=waiver_baselines,
                lineup_require=lineup_require,
            )
            for base_players in base_sets
        ])
        np.testing.assert_array_equal(
            batched,
            scalar,
            err_msg=f'batch marginal case {case_idx}',
        )
        np.testing.assert_array_equal(
            FootballSimulation._managed_percentile_90(scores),
            np.percentile(scores, 90, axis=-1),
            err_msg=f'partial percentile case {case_idx}',
        )
    return cases


def verify_salary_workspaces(cases=500):
    rng = np.random.default_rng(20260711)
    for case_idx in range(cases):
        player_count = 40
        players = np.array(
            [f'P{idx}' for idx in range(player_count)],
            dtype=object,
        )
        raw_salaries = np.maximum(
            1,
            rng.normal(18, 12, size=player_count),
        )
        fixed_count = int(rng.integers(0, 7))
        fixed_idx = rng.choice(
            player_count,
            size=fixed_count,
            replace=False,
        )
        fixed_players = players[fixed_idx].tolist()
        rng.shuffle(fixed_players)
        fixed_salaries = rng.integers(
            1,
            40,
            size=fixed_count,
        ).astype(float).tolist()
        candidate_idx = int(rng.choice(np.flatnonzero(
            ~np.isin(players, fixed_players)
        )))
        candidate = players[candidate_idx]
        candidate_price = float(raw_salaries[candidate_idx])
        buy_fixed = fixed_players + [candidate]
        buy_salaries = fixed_salaries + [candidate_price]
        remaining_slots = 20
        remaining_budget = 400.0 - candidate_price

        legacy = pd.DataFrame({
            'player': players,
            'salary': raw_salaries.copy(),
        })
        available_mask = ~legacy.player.isin(buy_fixed)
        available = np.maximum(
            legacy.loc[available_mask, 'salary'].to_numpy(dtype=float),
            1.0,
        )
        top_idx = np.argpartition(
            available,
            len(available) - remaining_slots,
        )[-remaining_slots:]
        scale = (
            remaining_budget - remaining_slots
        ) / float(np.sum(available[top_idx] - 1.0))
        legacy.loc[available_mask, 'salary'] = (
            1.0 + (available - 1.0) * scale
        )
        legacy.loc[legacy.player == candidate, 'salary'] = candidate_price

        legacy_buy = legacy[['player', 'salary']].copy()
        legacy_buy.loc[
            legacy_buy.player.isin(buy_fixed),
            'salary',
        ] = buy_salaries
        legacy_pass = legacy[['player', 'salary']].copy()
        legacy_pass.loc[
            legacy_pass.player.isin(fixed_players),
            'salary',
        ] = fixed_salaries

        fixed_mask = np.isin(players, fixed_players)
        buy_mask = fixed_mask.copy()
        buy_mask[candidate_idx] = True
        salary_values = FootballSimulation.normalize_salary_market_values(
            raw_salaries,
            ~buy_mask,
            remaining_budget,
            remaining_slots,
        )
        salary_values[candidate_idx] = candidate_price
        pass_input = {
            'players': fixed_players,
            'salaries': fixed_salaries,
        }
        buy_input = {
            'players': buy_fixed,
            'salaries': buy_salaries,
        }
        pass_row = FootballSimulation.create_G_salaries_from_values(
            salary_values,
            players,
            pass_input,
        ).ravel()
        fixed_rows = np.flatnonzero(fixed_mask)
        if len(fixed_rows) == 0 or candidate_idx > int(fixed_rows.max()):
            buy_row = pass_row
        else:
            buy_row = FootballSimulation.create_G_salaries_from_values(
                salary_values,
                players,
                buy_input,
            ).ravel()
            if np.array_equal(buy_row, pass_row):
                buy_row = pass_row

        np.testing.assert_array_equal(
            buy_row,
            legacy_buy.salary.to_numpy(dtype=float),
            err_msg=f'buy salary case {case_idx}',
        )
        np.testing.assert_array_equal(
            pass_row,
            legacy_pass.salary.to_numpy(dtype=float),
            err_msg=f'pass salary case {case_idx}',
        )
    return cases


def main():
    if MAX_TARGET_WORKERS != 8:
        raise AssertionError(
            f'Expected eight Target workers, found {MAX_TARGET_WORKERS}.'
        )
    fixture_count = verify_lineup_fixtures()
    board_rows = verify_target_board()
    marginal_cases = verify_batch_marginals()
    salary_cases = verify_salary_workspaces()
    print(
        'Target optimization verification passed: '
        f'{fixture_count} lineup fixtures, '
        f'{marginal_cases} batch marginal cases, '
        f'{salary_cases} salary cases, '
        f'{board_rows} exact board rows.'
    )


if __name__ == '__main__':
    main()
