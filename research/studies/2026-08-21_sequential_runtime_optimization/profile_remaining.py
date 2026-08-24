"""Measure the remaining market-only Sequential stage barriers."""

from __future__ import annotations

import time

import verify_runtime as study
import zSequential_Target as sequential


def main():
    original_collect = sequential._collect_sequential_worker_batches
    stage_timings = []

    def timed_collect(executor, tasks):
        group_count = sum(len(task[2]) for task in tasks)
        spec_count = sum(
            len(group)
            for task in tasks
            for group in task[2]
        )
        started = time.perf_counter()
        result = original_collect(executor, tasks)
        stage_timings.append({
            "groups": group_count,
            "specs": spec_count,
            "seconds": time.perf_counter() - started,
        })
        return result

    sequential._collect_sequential_worker_batches = timed_collect
    try:
        _, summary, curves = study.run_board(profile_curves=False)
    finally:
        sequential._collect_sequential_worker_batches = original_collect

    print(f"Total: {summary['runtime_seconds']:.3f}s")
    for index, timing in enumerate(stage_timings, start=1):
        print(
            f"Barrier {index}: {timing['seconds']:.3f}s / "
            f"{timing['groups']} groups / {timing['specs']} price specs"
        )
    print(
        f"Roster computations: {summary['roster_score_computations']} / "
        f"unique scored: {summary['unique_scored_rosters']} / "
        f"policy refreshes: {summary['policy_refreshes']} / "
        f"curve rows: {len(curves)}"
    )


if __name__ == "__main__":
    main()
