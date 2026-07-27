import argparse
import copy
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from profile_target import (
    RESULTS_DIR,
    create_sim,
    load_market_state,
    target_kwargs,
    zsim,
)


def initialized_barrier(task):
    barrier, delay_seconds = task
    barrier.wait()
    time.sleep(delay_seconds)
    return os.getpid()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workers',
        type=int,
        nargs='*',
        default=[2, 4, 8, 10, 16],
    )
    parser.add_argument('--task-delay', type=float, default=0.1)
    args = parser.parse_args()

    conn, sim = create_sim()
    try:
        market = load_market_state(conn)
        run_kwargs = target_kwargs(sim, market)
        worker_config = {
            'database_path': sim._target_database_path(),
            'set_year': sim.set_year,
            'pos_require_start': copy.deepcopy(sim.pos_require_start),
            'salary_cap': sim.salary_cap,
            'pred_vers': sim.pred_vers,
            'league': sim.league,
            'sal_pred_actual': sim.sal_pred_actual,
            'player_data': sim.player_data.copy(deep=True),
        }

        rows = []
        mp_context = multiprocessing.get_context('spawn')
        with multiprocessing.Manager() as manager:
            for worker_count in args.workers:
                barrier = manager.Barrier(worker_count)
                start = time.perf_counter()
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp_context,
                    initializer=zsim._initialize_target_worker,
                    initargs=(
                        worker_config,
                        market['to_add'],
                        market['to_drop'],
                        run_kwargs,
                    ),
                ) as executor:
                    pids = list(executor.map(
                        initialized_barrier,
                        [(barrier, args.task_delay)] * worker_count,
                        chunksize=1,
                    ))
                elapsed = time.perf_counter() - start
                row = {
                    'requested_workers': worker_count,
                    'started_workers': len(set(pids)),
                    'elapsed_seconds': elapsed,
                    'task_delay_seconds': args.task_delay,
                    'startup_and_shutdown_seconds': max(
                        0.0,
                        elapsed - args.task_delay,
                    ),
                }
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / 'worker_startup.csv',
            index=False,
        )
    finally:
        conn.close()


if __name__ == '__main__':
    main()
