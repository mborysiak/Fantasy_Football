"""Run the selected-roster residual diagnostic on the v5 frontier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
DIAGNOSTIC_RUNNER = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-14_selected_roster_salary_residuals"
    / "run_diagnostic.py"
)
FRONTIER_RESULTS = STUDY_DIR / "results" / "frontier_v5"
OUTPUT_RESULTS = STUDY_DIR / "results" / "selected_residuals_v5"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_salary_v5_selected_residuals",
        DIAGNOSTIC_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import selected residual diagnostic: {DIAGNOSTIC_RUNNER}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    diagnostic = load_runner()
    diagnostic.STUDY_DIR = STUDY_DIR
    diagnostic.CHANCE_STUDY = STUDY_DIR
    diagnostic.CHANCE_RESULTS = FRONTIER_RESULTS
    diagnostic.RESULTS = OUTPUT_RESULTS
    diagnostic.main()


if __name__ == "__main__":
    main()
