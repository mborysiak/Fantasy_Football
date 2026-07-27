from argparse import ArgumentParser
from pathlib import Path
import sqlite3
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = ROOT.parent / 'Fantasy_Football_App'
sys.path.insert(0, str(APP_ROOT / 'app'))

from zSim_Helper import (  # noqa: E402
    DEFAULT_TARGET_CONFIRM_CANDIDATES,
    DEFAULT_TARGET_MARKET_CONFIRM_ANCHORS,
    DEFAULT_TARGET_CONFIRM_TRIALS,
    DEFAULT_TARGET_PILOT_DISCOVERIES,
    DEFAULT_TARGET_PILOT_TRIALS,
    DEFAULT_TARGET_SCREEN_CANDIDATES,
    DEFAULT_TARGET_SCREEN_TRIALS,
    FootballSimulation,
    MANAGED_POS_MAX,
    TARGET_STAGE_LOGICAL_BLOCKS,
)


FIXED_PLAYERS = [
    'Jayden Daniels',
    'Jahmyr Gibbs',
    'Chase Brown',
    'Bhayshul Tuten',
]
FIXED_SALARIES = [15, 108, 34, 11]
LINEUP_REQUIRE = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
WAIVER_BASELINES = {'QB': 15.7, 'RB': 5.5, 'WR': 6.0, 'TE': 6.5}
RANDOM_SEED = 20260717


def run_target(organic_trials, workers):
    database_path = APP_ROOT / 'app' / 'Simulation.sqlite3'
    conn = sqlite3.connect(database_path)
    try:
        keepers = pd.read_sql_query(
            """SELECT player, keeper_salary
                 FROM League_Keepers
                WHERE year=2026 AND league='beta'""",
            conn,
        )
        simulation = FootballSimulation(
            conn,
            2026,
            LINEUP_REQUIRE,
            298,
            'final_ensemble',
            'beta',
            sal_pred_actual='pred',
        )
        fixed_keeper_players = {'Chase Brown', 'Bhayshul Tuten'}
        to_drop = keepers.loc[
            ~keepers.player.isin(fixed_keeper_players),
            'player',
        ].tolist()
        remaining_market_budget = (
            12 * 298 - float(keepers.keeper_salary.sum()) - 15 - 108
        )
        remaining_market_slots = 12 * 13 - len(keepers) - 2

        start = time.perf_counter()
        results = simulation.run_sim_parallel(
            {'players': FIXED_PLAYERS, 'salaries': FIXED_SALARIES},
            to_drop,
            organic_trials,
            max_workers=workers,
            random_seed=RANDOM_SEED,
            require_top_n=12,
            num_avg_pts=5,
            next_year_frac=0.0,
            enforce_top_n=True,
            scoring_mode='managed',
            roster_size=13,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1},
            pos_max_counts=MANAGED_POS_MAX,
            waiver_baselines=WAIVER_BASELINES,
            bench_upside_weight=0.25,
            remaining_market_budget=remaining_market_budget,
            remaining_market_slots=remaining_market_slots,
            use_selection_premium=True,
            target_screen_size=DEFAULT_TARGET_SCREEN_CANDIDATES,
            target_screen_trials=DEFAULT_TARGET_SCREEN_TRIALS,
            target_pilot_trials=DEFAULT_TARGET_PILOT_TRIALS,
            target_pilot_discoveries=DEFAULT_TARGET_PILOT_DISCOVERIES,
            target_confirm_size=DEFAULT_TARGET_CONFIRM_CANDIDATES,
            target_market_confirm_anchors=(
                DEFAULT_TARGET_MARKET_CONFIRM_ANCHORS
            ),
            target_confirm_trials=DEFAULT_TARGET_CONFIRM_TRIALS,
        )
        elapsed = time.perf_counter() - start
        summary = simulation.get_managed_summary()
        screen_candidates = list(simulation.target_screen_candidates)
        pilot_candidates = list(simulation.target_pilot_candidates)
        pilot_discoveries = list(simulation.target_pilot_discoveries)
        confirm_candidates = list(simulation.target_confirm_candidates)
        evidence_confirm_candidates = list(
            simulation.target_evidence_confirm_candidates
        )
        market_confirm_anchors = list(
            simulation.target_market_confirm_anchors
        )
    finally:
        conn.close()

    return {
        'results': results.sort_values('player').reset_index(drop=True),
        'summary': summary,
        'screen_candidates': screen_candidates,
        'pilot_candidates': pilot_candidates,
        'pilot_discoveries': pilot_discoveries,
        'confirm_candidates': confirm_candidates,
        'evidence_confirm_candidates': evidence_confirm_candidates,
        'market_confirm_anchors': market_confirm_anchors,
        'organic_blocks': list(simulation.target_organic_blocks),
        'elapsed': elapsed,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument('--organic-trials', type=int, default=100)
    args = parser.parse_args()

    serial = run_target(args.organic_trials, workers=1)
    parallel = run_target(args.organic_trials, workers=8)

    pd.testing.assert_frame_equal(
        serial['results'],
        parallel['results'],
        check_exact=True,
    )
    assert serial['summary'] == parallel['summary']
    assert serial['screen_candidates'] == parallel['screen_candidates']
    assert serial['pilot_candidates'] == parallel['pilot_candidates']
    assert serial['pilot_discoveries'] == parallel['pilot_discoveries']
    assert serial['confirm_candidates'] == parallel['confirm_candidates']
    assert serial['evidence_confirm_candidates'] == (
        parallel['evidence_confirm_candidates']
    )
    assert serial['market_confirm_anchors'] == (
        parallel['market_confirm_anchors']
    )
    assert serial['organic_blocks'] == parallel['organic_blocks']
    assert len(serial['organic_blocks']) == min(
        TARGET_STAGE_LOGICAL_BLOCKS,
        args.organic_trials,
    )
    assert max(serial['organic_blocks']) - min(serial['organic_blocks']) <= 1

    results = serial['results']
    prelim_blocks = results.loc[results.PrelimN.fillna(0) > 0, 'PrelimBlocks']
    confirm_blocks = results.loc[results.ConfirmN.fillna(0) > 0, 'ConfirmBlocks']
    assert len(prelim_blocks) > 0
    assert len(confirm_blocks) > 0
    assert (prelim_blocks == TARGET_STAGE_LOGICAL_BLOCKS).all()
    assert (confirm_blocks == TARGET_STAGE_LOGICAL_BLOCKS).all()

    print(
        'Worker invariance passed: '
        f"serial={serial['elapsed']:.1f}s, parallel={parallel['elapsed']:.1f}s, "
        f'logical_blocks={TARGET_STAGE_LOGICAL_BLOCKS}, rows={len(results)}'
    )


if __name__ == '__main__':
    main()
