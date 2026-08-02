"""Fail-closed checks for native numerical-library runtime conflicts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from threadpoolctl import threadpool_info


def _normalized_path(path: str) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def openmp_runtime_inventory() -> tuple[dict[str, str], ...]:
    """Return distinct loaded OpenMP runtimes keyed by resolved library path."""
    runtimes: dict[str, dict[str, str]] = {}
    for record in threadpool_info():
        if (
            str(record.get("internal_api", "")).casefold() != "openmp"
            and str(record.get("user_api", "")).casefold() != "openmp"
        ):
            continue
        filepath = str(record.get("filepath") or "").strip()
        if not filepath:
            key = f"<unknown:{len(runtimes)}>"
            display_path = "<unknown>"
        else:
            key = _normalized_path(filepath)
            display_path = str(Path(filepath).resolve(strict=False))
        runtimes.setdefault(
            key,
            {
                "filepath": display_path,
                "prefix": str(record.get("prefix") or "").casefold(),
                "internal_api": str(
                    record.get("internal_api") or ""
                ).casefold(),
            },
        )
    return tuple(runtimes[key] for key in sorted(runtimes))


def assert_single_openmp_runtime() -> tuple[dict[str, str], ...]:
    """Reject unsafe duplicate OpenMP runtimes before native model fitting."""
    runtimes = openmp_runtime_inventory()
    paths = [runtime["filepath"] for runtime in runtimes]
    if len(runtimes) > 1:
        raise RuntimeError(
            "Unsafe native runtime state: multiple OpenMP libraries are "
            f"loaded ({paths}). On Windows, import scikit-learn before "
            "LightGBM so both packages use one vcomp runtime."
        )
    if sys.platform == "win32":
        if len(runtimes) != 1:
            raise RuntimeError(
                "Expected exactly one OpenMP runtime on Windows after "
                f"scikit-learn and LightGBM imports; found {paths}."
            )
        runtime = runtimes[0]
        if runtime["prefix"] != "vcomp":
            raise RuntimeError(
                "Expected the single Windows OpenMP runtime to use the "
                f"vcomp prefix; found {runtime}."
            )
        if runtime["filepath"] == "<unknown>":
            raise RuntimeError(
                "The loaded Windows vcomp runtime has no inspectable path."
            )
    return runtimes
