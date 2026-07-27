"""Run the preserved salary chance frontier with the v5 salary method."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
CHANCE_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_salary_chance_frontier"
    / "run_replay.py"
)
V5_METHOD = "current_locked_spec_v5_compact_salary_features"
DEFAULT_OUTPUT = STUDY_DIR / "results" / "frontier_v5"


def load_chance_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_salary_v5_chance_frontier",
        CHANCE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import chance replay: {CHANCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    chance = load_chance_runner()
    chance.current.SALARY_METHOD = V5_METHOD
    if not any(
        argument == "--output-dir" or argument.startswith("--output-dir=")
        for argument in sys.argv[1:]
    ):
        sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT)])
    chance.main()


if __name__ == "__main__":
    main()
