import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
APP_DIR = REPO_ROOT.parent / 'Fantasy_Football_App' / 'app'
PROFILE_DIR = (
    REPO_ROOT
    / 'research'
    / 'studies'
    / '2026-07-11_target_runtime_profile'
)
RESULTS_DIR = STUDY_DIR / 'results'

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROFILE_DIR))

from zSim_Helper import FootballSimulation  # noqa: E402
from profile_target import (  # noqa: E402
    SEED,
    create_sim,
    load_market_state,
    target_kwargs,
)


LINEUP = {'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1, 'FLEX': 0}
POS_MIN = {'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1}
POS_MAX = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
WAIVERS = {'QB': 0.0, 'RB': 0.0, 'WR': 0.0, 'TE': 0.0}


def synthetic_fixture(cap=70.0):
    players = np.array(
        [
            'KeeperQB',
            'CoreRB',
            'ExtraRB',
            'WeakWR',
            'CoreTE',
            'StrongWR',
            'AltQB',
        ],
        dtype=object,
    )
    positions = np.array(
        ['QB', 'RB', 'RB', 'WR', 'TE', 'WR', 'QB'],
        dtype=object,
    )
    salaries = np.array([100.0, 30.0, 20.0, 5.0, 5.0, 25.0, 5.0])
    weekly = np.repeat(
        np.array([20.0, 15.0, 12.0, 5.0, 8.0, 14.0, 10.0])[:, None],
        4,
        axis=1,
    ).astype(np.float32)
    predictions = pd.DataFrame({
        'player': players,
        'pos': positions,
        'salary': salaries,
    })
    selected = np.isin(
        players,
        ['KeeperQB', 'CoreRB', 'ExtraRB', 'WeakWR', 'CoreTE'],
    )
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.salary_cap = float(cap)
    return sim, predictions, weekly, weekly.copy(), selected


def exact_score(sim, predictions, weekly, decisions, mask):
    return float(
        sim.managed_lineup_weekly_scores(
            weekly[mask],
            predictions.pos.to_numpy()[mask],
            decision_scores=decisions[mask],
            player_names=predictions.player.to_numpy()[mask],
            lineup_require=LINEUP,
            waiver_baselines=WAIVERS,
        ).sum()
    )


def refine_fixture(
    cap=70.0,
    fixed_players=None,
    top_n=None,
    selected_override=None,
):
    sim, predictions, weekly, decisions, selected = synthetic_fixture(cap=cap)
    if selected_override is not None:
        selected = predictions.player.isin(selected_override).to_numpy()
    fixed_players = list(fixed_players or ['KeeperQB'])
    fixed_salary_map = {
        player: 5.0 if player == 'KeeperQB' else float(
            predictions.loc[predictions.player == player, 'salary'].iloc[0]
        )
        for player in fixed_players
    }
    refined, details = sim.refine_managed_roster_one_swap(
        predictions,
        selected,
        weekly,
        decisions,
        fixed_players=fixed_players,
        fixed_salary_map=fixed_salary_map,
        waiver_baselines=WAIVERS,
        lineup_require=LINEUP,
        pos_min_counts=POS_MIN,
        pos_max_counts=POS_MAX,
        top_n=list(top_n or ['CoreRB']),
        enforce_top_n=True,
    )
    return (
        sim,
        predictions,
        weekly,
        decisions,
        selected,
        refined,
        details,
        fixed_salary_map,
    )


def assert_roster_constraints(
    sim,
    predictions,
    mask,
    fixed_players,
    fixed_salary_map,
    top_n,
):
    players = predictions.player.to_numpy()
    positions = predictions.pos.to_numpy()
    selected_players = set(players[mask])
    if len(selected_players) != 5:
        raise AssertionError('Refinement changed roster size.')
    if not set(fixed_players).issubset(selected_players):
        raise AssertionError('Refinement removed a fixed player.')
    if top_n and not selected_players.intersection(top_n):
        raise AssertionError('Refinement violated the top-N constraint.')
    for pos in ('QB', 'RB', 'WR', 'TE'):
        count = int(np.sum(positions[mask] == pos))
        if not POS_MIN[pos] <= count <= POS_MAX[pos]:
            raise AssertionError(f'Refinement violated the {pos} bounds.')
    salaries = predictions.salary.to_numpy(dtype=float, copy=True)
    for player, salary in fixed_salary_map.items():
        salaries[players == player] = salary
    if float(salaries[mask].sum()) > sim.salary_cap + 1e-8:
        raise AssertionError('Refinement exceeded the salary cap.')


def verify_synthetic_constraints():
    (
        sim,
        predictions,
        weekly,
        decisions,
        selected,
        refined,
        details,
        fixed_salary_map,
    ) = refine_fixture()
    refined_players = set(predictions.player.to_numpy()[refined])
    if not details['changed']:
        raise AssertionError('Expected the redundant-RB roster to improve.')
    if details['out_player'] != 'ExtraRB' or details['in_player'] != 'StrongWR':
        raise AssertionError(f'Unexpected primary swap: {details}')
    if abs(details['score_before'] - 192.0) > 1e-8:
        raise AssertionError(f'Unexpected baseline score: {details}')
    if abs(details['score_after'] - 228.0) > 1e-8:
        raise AssertionError(f'Unexpected refined score: {details}')
    if 'KeeperQB' not in refined_players:
        raise AssertionError('Manual-salary keeper was removed.')
    assert_roster_constraints(
        sim,
        predictions,
        refined,
        ['KeeperQB'],
        fixed_salary_map,
        {'CoreRB'},
    )

    *_, cap_details, _ = refine_fixture(cap=69.0)
    if cap_details['changed']:
        raise AssertionError('Refinement accepted a cap-violating swap.')

    top_n_case = refine_fixture(
        fixed_players=['KeeperQB', 'CoreRB', 'WeakWR', 'CoreTE'],
        top_n=['ExtraRB'],
    )
    if top_n_case[6]['changed']:
        raise AssertionError('Refinement removed the sole top-N player.')

    fixed_case = refine_fixture(fixed_players=['KeeperQB', 'ExtraRB'])
    fixed_players_after = set(
        fixed_case[1].player.to_numpy()[fixed_case[5]]
    )
    if 'ExtraRB' not in fixed_players_after:
        raise AssertionError('Refinement removed a fixed non-keeper player.')

    local_case = refine_fixture(
        selected_override=[
            'KeeperQB',
            'CoreRB',
            'StrongWR',
            'WeakWR',
            'CoreTE',
        ]
    )
    if local_case[6]['changed']:
        raise AssertionError('A locally optimal roster should be a no-op.')

    return {
        'primary_score_before': details['score_before'],
        'primary_score_after': details['score_after'],
        'primary_swap': f"{details['out_player']} -> {details['in_player']}",
    }


def verify_tied_decision_bruteforce(cases=30):
    rng = np.random.default_rng(20260713)
    positions = np.array(
        ['QB'] * 4 + ['RB'] * 12 + ['WR'] * 12 + ['TE'] * 6,
        dtype=object,
    )
    players = np.array([f'P{idx}' for idx in range(len(positions))], dtype=object)
    predictions = pd.DataFrame({
        'player': players,
        'pos': positions,
        'salary': np.ones(len(players), dtype=float),
    })
    lineup = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
    pos_min = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
    pos_max = {'QB': 1, 'RB': 7, 'WR': 7, 'TE': 3}
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.salary_cap = 12.0

    for case_idx in range(int(cases)):
        weekly = np.maximum(
            0,
            rng.normal(9, 6, size=(len(players), 12)),
        ).astype(np.float32)
        weekly[rng.random(weekly.shape) < 0.12] = 0
        decisions = (np.round(weekly / 3) * 3).astype(np.float32)
        selected_idx = np.concatenate([
            rng.choice(np.flatnonzero(positions == 'QB'), 1, replace=False),
            rng.choice(np.flatnonzero(positions == 'RB'), 4, replace=False),
            rng.choice(np.flatnonzero(positions == 'WR'), 5, replace=False),
            rng.choice(np.flatnonzero(positions == 'TE'), 2, replace=False),
        ])
        selected = np.zeros(len(players), dtype=bool)
        selected[selected_idx] = True
        refined, details = sim.refine_managed_roster_one_swap(
            predictions,
            selected,
            weekly,
            decisions,
            fixed_players=[],
            fixed_salary_map={},
            waiver_baselines=WAIVERS,
            lineup_require=lineup,
            pos_min_counts=pos_min,
            pos_max_counts=pos_max,
            top_n=[],
            enforce_top_n=False,
        )

        def score(mask):
            return float(
                sim.managed_lineup_weekly_scores(
                    weekly[mask],
                    positions[mask],
                    decision_scores=decisions[mask],
                    player_names=players[mask],
                    lineup_require=lineup,
                    waiver_baselines=WAIVERS,
                ).sum()
            )

        initial_score = score(selected)
        brute_best = initial_score
        for out_idx in np.flatnonzero(selected):
            base = selected.copy()
            base[out_idx] = False
            for in_idx in np.flatnonzero(~selected):
                candidate = base.copy()
                candidate[in_idx] = True
                counts = {
                    pos: int(np.sum(positions[candidate] == pos))
                    for pos in ('QB', 'RB', 'WR', 'TE')
                }
                if any(
                    not pos_min[pos] <= counts[pos] <= pos_max[pos]
                    for pos in counts
                ):
                    continue
                brute_best = max(brute_best, score(candidate))

        refined_score = score(refined)
        if abs(refined_score - brute_best) > 1e-4:
            raise AssertionError(
                f'Tied-decision case {case_idx} missed the best swap: '
                f'{refined_score} versus {brute_best}; details={details}'
            )

    return int(cases)


def run_target_case(iterations, refinement):
    conn, sim = create_sim()
    refinement_details = []
    event_order = []
    try:
        market = load_market_state(conn)
        kwargs = target_kwargs(sim, market)
        kwargs['managed_roster_refinement'] = bool(refinement)
        if refinement:
            original_refine = sim.refine_managed_roster_to_convergence
            original_holdout = sim.sample_managed_holdout_contexts

            def tracked_refine(*args, **call_kwargs):
                event_order.append('refine')
                if np.asarray(args[2]).shape[1] != 16:
                    raise AssertionError(
                        'Target refinement did not use the expected 16-week '
                        'construction-bank profile.'
                    )
                call_kwargs['max_swaps'] = 1
                mask, details = original_refine(*args, **call_kwargs)
                refinement_details.append(details)
                return mask, details

            def tracked_holdout(*args, **call_kwargs):
                event_order.append('holdout')
                return original_holdout(*args, **call_kwargs)

            sim.refine_managed_roster_to_convergence = tracked_refine
            sim.sample_managed_holdout_contexts = tracked_holdout

        start = time.perf_counter()
        with sim.temp_seed(SEED):
            result = sim.run_sim(
                market['to_add'],
                market['to_drop'],
                int(iterations),
                **kwargs,
            )
        elapsed = time.perf_counter() - start
        summary = sim.get_managed_summary()
    finally:
        conn.close()

    if refinement:
        expected_order = ['refine', 'holdout'] * int(summary['trials'])
        if event_order != expected_order:
            raise AssertionError(
                'Target refinement must finish before each holdout draw.'
            )

    return result, summary, elapsed, refinement_details


def benchmark_target(iterations=50):
    baseline, baseline_summary, baseline_seconds, _ = run_target_case(
        iterations,
        False,
    )
    refined, refined_summary, refined_seconds, details = run_target_case(
        iterations,
        True,
    )
    if baseline_summary['trials'] != iterations:
        raise AssertionError('Baseline Target did not complete every trial.')
    if refined_summary['trials'] != iterations:
        raise AssertionError('Refined Target did not complete every trial.')
    changed = [detail for detail in details if detail['changed']]
    mean_gain = float(np.mean([
        detail['improvement']
        for detail in details
    ]))
    if len(changed) == 0 or mean_gain <= 0:
        raise AssertionError('Real Target refinement produced no improvement.')

    return {
        'iterations': int(iterations),
        'seed': int(SEED),
        'baseline_seconds': float(baseline_seconds),
        'refined_seconds': float(refined_seconds),
        'runtime_overhead_pct': float(
            100 * (refined_seconds / baseline_seconds - 1)
        ),
        'improved_rosters': int(len(changed)),
        'mean_construction_gain_points': mean_gain,
        'baseline_season_ev': float(baseline_summary['season_ev']),
        'refined_season_ev': float(refined_summary['season_ev']),
        'season_ev_delta': float(
            refined_summary['season_ev'] - baseline_summary['season_ev']
        ),
        'baseline_top_player': str(baseline.player.iloc[0]),
        'refined_top_player': str(refined.player.iloc[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--tied-cases', type=int, default=30)
    args = parser.parse_args()

    synthetic = verify_synthetic_constraints()
    tied_cases = verify_tied_decision_bruteforce(args.tied_cases)
    benchmark = benchmark_target(args.iterations)
    output = {
        'synthetic': synthetic,
        'tied_decision_cases': tied_cases,
        'benchmark': benchmark,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'benchmark.json').write_text(
        json.dumps(output, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
