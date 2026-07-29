"""Separate exact prior-year and three-year projection trajectory effects."""

from __future__ import annotations

import importlib.util
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "projection_trajectory_validation",
    STUDY_DIR / "run_validation.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Unable to load projection-trajectory validation module")
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)

ONE_YEAR_FEATURES = (
    "projection_trajectory_change_1year",
    "projection_trajectory_prior_year_available",
)
THREE_YEAR_FEATURES = (
    "projection_trajectory_change_3year",
    "projection_trajectory_prior_3year_count",
    "projection_trajectory_prior_3year_std",
)


def _ablation_variants(manifests):
    incumbent = validation._manifest(
        manifests,
        "residual_candidate_v1",
    )
    return {
        "incumbent": tuple(
            dict.fromkeys((*incumbent, *validation.POSITION_FEATURES))
        ),
        "trajectory_1year": tuple(
            dict.fromkeys(
                (
                    *incumbent,
                    *ONE_YEAR_FEATURES,
                    *validation.POSITION_FEATURES,
                )
            )
        ),
        "trajectory_3year": tuple(
            dict.fromkeys(
                (
                    *incumbent,
                    *THREE_YEAR_FEATURES,
                    *validation.POSITION_FEATURES,
                )
            )
        ),
    }


validation.RESULTS_DIR = STUDY_DIR / "ablation_results"
validation.VARIANTS = (
    "incumbent",
    "trajectory_1year",
    "trajectory_3year",
)
validation._feature_variants = _ablation_variants
_base_summary_markdown = validation._summary_markdown


def _ablation_summary_markdown(scores, comparisons):
    return _base_summary_markdown(scores, comparisons).replace(
        "# Projection Trajectory and Logged-ADP Results",
        "# Projection Trajectory Ablation Results",
        1,
    )


validation._summary_markdown = _ablation_summary_markdown


if __name__ == "__main__":
    validation.main()
