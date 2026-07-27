import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
PROFILE_DIR = REPO_ROOT / 'research' / 'studies' / '2026-07-11_target_runtime_profile'
sys.path.insert(0, str(PROFILE_DIR))

from profile_target import (  # noqa: E402
    SEED,
    create_sim,
    load_market_state,
    target_kwargs,
)


def run_serial(iterations, use_played_mask):
    conn, sim = create_sim()
    try:
        market = load_market_state(conn)
        sim.load_weekly_template_profiles()
        if not use_played_mask:
            template_cols = pd.read_sql_query(
                'SELECT * FROM Best_Ball_Weekly_Templates LIMIT 0',
                conn,
            ).columns
            pool_cols = pd.read_sql_query(
                'SELECT * FROM Best_Ball_Weekly_Template_Pools LIMIT 0',
                conn,
            ).columns
            week_cols = [f'week_{week}' for week in range(1, 17)]
            if 'league' in template_cols and 'template_league' in pool_cols:
                template_join = (
                    'ON p.template_id = t.template_id '
                    'AND p.template_league = t.league'
                )
            elif 'league' in template_cols:
                template_join = (
                    'ON p.template_id = t.template_id '
                    'AND p.pool_version = t.league'
                )
            else:
                template_join = 'ON p.template_id = t.template_id'
            legacy_profiles = pd.read_sql_query(
                f'''
                SELECT m.player, {', '.join([f't.{col}' for col in week_cols])}
                FROM Best_Ball_Weekly_Player_Map m
                INNER JOIN Best_Ball_Weekly_Template_Pools p
                        ON m.template_pool_key = p.template_pool_key
                INNER JOIN Best_Ball_Weekly_Templates t
                        {template_join}
                WHERE m.year = {sim.set_year}
                      AND m.version = '{sim.league}'
                      AND m.dataset = '{sim.pred_vers}'
                ORDER BY m.player, p.match_rank
                ''',
                conn,
            )
            sim.weekly_template_profiles = {
                player: group[week_cols].to_numpy(dtype=np.float32)
                for player, group in legacy_profiles.groupby('player', sort=False)
            }
            if set(sim.weekly_template_profiles) != set(
                sim.weekly_template_played_masks
            ):
                raise RuntimeError('Legacy score profiles do not align with masks.')
            sim.weekly_template_played_masks = {
                player: np.full(mask.shape, -1, dtype=np.int8)
                for player, mask in sim.weekly_template_played_masks.items()
            }
        kwargs = target_kwargs(sim, market)
        start = time.perf_counter()
        with sim.temp_seed(SEED):
            result = sim.run_sim(
                market['to_add'],
                market['to_drop'],
                int(iterations),
                **kwargs,
            )
        elapsed = time.perf_counter() - start
        return {
            'seconds': float(elapsed),
            'season_ev': float(sim.get_managed_summary()['season_ev']),
            'top_player': str(result.player.iloc[0]),
        }
    finally:
        conn.close()


def run_parallel(iterations, workers):
    conn, sim = create_sim()
    try:
        market = load_market_state(conn)
        kwargs = target_kwargs(sim, market)
        start = time.perf_counter()
        result = sim.run_sim_parallel(
            market['to_add'],
            market['to_drop'],
            int(iterations),
            max_workers=int(workers),
            block_size=50,
            random_seed=SEED,
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        return {
            'seconds': float(elapsed),
            'season_ev': float(sim.get_managed_summary()['season_ev']),
            'top_player': str(result.player.iloc[0]),
            'workers_used': int(sim.parallel_workers_used),
            'blocks': int(sim.parallel_blocks),
        }
    finally:
        conn.close()


def summarize_counterbalanced_runs(runs):
    if len(runs) == 0:
        raise ValueError('At least one benchmark run is required.')
    season_evs = {run['season_ev'] for run in runs}
    top_players = {run['top_player'] for run in runs}
    if len(season_evs) != 1 or len(top_players) != 1:
        raise RuntimeError('Seeded benchmark outputs changed across run order.')
    seconds = [float(run['seconds']) for run in runs]
    return {
        'seconds': float(np.mean(seconds)),
        'seconds_by_order': seconds,
        'season_ev': float(runs[0]['season_ev']),
        'top_player': str(runs[0]['top_player']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--parallel-workers', type=int, default=0)
    args = parser.parse_args()

    # Runtime on this workload is sensitive to process order and system state.
    # Give each variant one first and one second execution, then compare their
    # mean timings. Seeded EV must remain identical within each variant.
    serial_runs = {False: [], True: []}
    for run_order in ((False, True), (True, False)):
        for use_played_mask in run_order:
            serial_runs[use_played_mask].append(
                run_serial(args.iterations, use_played_mask)
            )
    legacy = summarize_counterbalanced_runs(serial_runs[False])
    played = summarize_counterbalanced_runs(serial_runs[True])
    output = {
        'iterations': int(args.iterations),
        'seed': int(SEED),
        'legacy_score_threshold': legacy,
        'played_mask': played,
        'played_minus_legacy_season_ev': float(
            played['season_ev'] - legacy['season_ev']
        ),
        'runtime_overhead_pct': float(
            100 * (played['seconds'] / legacy['seconds'] - 1)
        ),
    }
    if args.parallel_workers > 0:
        output['parallel_played_mask'] = run_parallel(
            args.iterations,
            args.parallel_workers,
        )
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
