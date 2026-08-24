"""Profile the market-only Sequential board without worker-process opacity."""

from __future__ import annotations

import cProfile
import pstats

import verify_runtime as study


def main():
    profiler = cProfile.Profile()
    profiler.enable()
    _, summary, _ = study.run_board(
        profile_curves=False,
        parallel_workers=1,
    )
    profiler.disable()
    print(f"Total: {summary['runtime_seconds']:.3f}s")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(35)


if __name__ == "__main__":
    main()
