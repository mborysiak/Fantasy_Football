from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from Scripts.V2 import native_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
ANNUAL_RUNNERS = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_locked_final_validation"
    / "run_validation.py",
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_next_year_residual"
    / "run_validation.py",
)


def _openmp_record(path: Path, prefix: str = "vcomp") -> dict[str, object]:
    return {
        "filepath": str(path),
        "prefix": prefix,
        "internal_api": "openmp",
        "user_api": "openmp",
    }


def test_guard_rejects_distinct_openmp_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        native_runtime,
        "threadpool_info",
        lambda: [
            _openmp_record(tmp_path / "system" / "vcomp140.dll"),
            _openmp_record(tmp_path / "sklearn" / "vcomp140.dll"),
        ],
    )

    with pytest.raises(RuntimeError, match="multiple OpenMP libraries"):
        native_runtime.assert_single_openmp_runtime()


def test_guard_deduplicates_the_same_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "sklearn" / "vcomp140.dll"
    monkeypatch.setattr(
        native_runtime,
        "threadpool_info",
        lambda: [
            _openmp_record(runtime),
            _openmp_record(runtime),
        ],
    )

    inventory = native_runtime.assert_single_openmp_runtime()

    assert len(inventory) == 1
    assert inventory[0]["prefix"] == "vcomp"


@pytest.mark.parametrize(
    ("records", "message"),
    (
        ([], "Expected exactly one OpenMP runtime"),
        (
            [
                {
                    "filepath": "libgomp.dll",
                    "prefix": "libgomp",
                    "internal_api": "openmp",
                    "user_api": "openmp",
                }
            ],
            "vcomp prefix",
        ),
    ),
)
def test_windows_guard_requires_exactly_one_vcomp_runtime(
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, object]],
    message: str,
) -> None:
    monkeypatch.setattr(native_runtime.sys, "platform", "win32")
    monkeypatch.setattr(native_runtime, "threadpool_info", lambda: records)

    with pytest.raises(RuntimeError, match=message):
        native_runtime.assert_single_openmp_runtime()


@pytest.mark.parametrize("runner", ANNUAL_RUNNERS, ids=("locked", "next_year"))
def test_annual_runner_import_order_loads_one_openmp_runtime(
    runner: Path,
) -> None:
    probe = f"""
import importlib.util
import json
from pathlib import Path

runner = Path({str(runner)!r})
spec = importlib.util.spec_from_file_location("annual_runtime_probe", runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from Scripts.V2.native_runtime import assert_single_openmp_runtime
print(json.dumps(assert_single_openmp_runtime()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    inventory = json.loads(completed.stdout.strip().splitlines()[-1])
    if sys.platform == "win32":
        assert len(inventory) == 1
        assert inventory[0]["prefix"] == "vcomp"
        assert Path(inventory[0]["filepath"]).name.casefold() == "vcomp140.dll"
