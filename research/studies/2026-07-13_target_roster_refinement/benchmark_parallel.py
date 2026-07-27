import argparse
import json
import sys
import time
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
PROFILE_DIR = (
    REPO_ROOT
    / 'research'
    / 'studies'
    / '2026-07-11_target_runtime_profile'
)
RESULTS_DIR = STUDY_DIR / 'results'
sys.path.insert(0, str(PROFILE_DIR))

from profile_target import (  # noqa: E402
    SEED,
    create_sim,
    load_market_state,
    target_kwargs,
)


def run_case(iterations, workers, refinement):
    conn, sim = create_sim()
    try:
        market = load_market_state(conn)
        kwargs = target_kwargs(sim, market)
        kwargs['managed_roster_refinement'] = bool(refinement)
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
        summary = sim.get_managed_summary()
        if summary['trials'] != iterations:
            raise AssertionError(
                f'Completed {summary["trials"]}/{iterations} Target trials.'
            )
        return {
            'seconds': float(elapsed),
            'top_player': str(result.player.iloc[0]),
            'season_ev': float(summary['season_ev']),
            'workers_used': int(sim.parallel_workers_used),
            'blocks': int(sim.parallel_blocks),
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    baseline = run_case(args.iterations, args.workers, False)
    refined = run_case(args.iterations, args.workers, True)
    output = {
        'iterations': int(args.iterations),
        'seed': int(SEED),
        'requested_workers': int(args.workers),
        'baseline': baseline,
        'refined': refined,
        'runtime_overhead_pct': float(
            100 * (refined['seconds'] / baseline['seconds'] - 1)
        ),
        'season_ev_delta': float(
            refined['season_ev'] - baseline['season_ev']
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'parallel_benchmark.json').write_text(
        json.dumps(output, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
