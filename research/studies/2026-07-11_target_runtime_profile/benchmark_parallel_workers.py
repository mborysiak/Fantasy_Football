import argparse
import json
import sys
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR))
from profile_target import RESULTS_DIR, SEED, run_parallel  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=800)
    parser.add_argument(
        '--workers',
        type=int,
        nargs='*',
        default=[8, 10, 12, 16],
    )
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for worker_count in args.workers:
        for repeat in range(max(1, int(args.repeats))):
            _, summary = run_parallel(
                args.iterations,
                worker_count,
                seed=SEED + repeat,
            )
            summary.update({
                'requested_workers': worker_count,
                'repeat': repeat + 1,
            })
            rows.append(summary)
            print(json.dumps(summary, indent=2), flush=True)

    label_suffix = f'_{args.label}' if args.label else ''
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / f'parallel_worker_sweep_{args.iterations}{label_suffix}.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
