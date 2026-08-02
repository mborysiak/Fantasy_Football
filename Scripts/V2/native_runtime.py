"""Fail-closed checks for native numerical-library runtime conflicts."""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Mapping, Sequence

from threadpoolctl import threadpool_info


MAX_ISOLATED_LIGHTGBM_FITS = 8
RANDOM_FOREST_N_JOBS = 4


def _invoke_module_function(
    module_path: str,
    function_name: str,
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> Any:
    """Import one file in a spawned worker and invoke a named function."""
    resolved_path = Path(module_path).resolve(strict=True)
    module_name = f"_v2_native_worker_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load isolated worker: {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        assert_single_openmp_runtime()
        function = getattr(module, function_name)
        return function(*args, **kwargs)
    finally:
        sys.modules.pop(module_name, None)


def run_module_function_in_fresh_process(
    module_path: Path,
    function_name: str,
    args: Sequence[object] = (),
    kwargs: Mapping[str, object] | None = None,
) -> Any:
    """Run one native-model workload in a fresh spawn-isolated process."""
    context = multiprocessing.get_context("spawn")
    resolved_path = str(Path(module_path).resolve(strict=True))
    for attempt in (1, 2):
        try:
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
            ) as executor:
                future = executor.submit(
                    _invoke_module_function,
                    resolved_path,
                    function_name,
                    tuple(args),
                    dict(kwargs or {}),
                )
                return future.result()
        except BrokenProcessPool:
            if attempt == 2:
                raise
            print(
                f"[{function_name}] native worker exited abruptly; "
                "retrying batch once",
                flush=True,
            )
    raise AssertionError("Unreachable isolated-worker retry state")


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
