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


@pytest.mark.parametrize(
    ("runner", "factory_name", "model_names"),
    (
        (
            ANNUAL_RUNNERS[0],
            "_model_pipeline",
            ("conditional_ppg_lightgbm", "participation_lightgbm"),
        ),
        (
            ANNUAL_RUNNERS[1],
            "model_pipeline",
            ("next_residual_lightgbm", "next_participation_lightgbm"),
        ),
    ),
    ids=("locked", "next_year"),
)
def test_annual_lightgbm_pipelines_preserve_feature_names(
    runner: Path,
    factory_name: str,
    model_names: tuple[str, str],
) -> None:
    probe = f"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

runner = Path({str(runner)!r})
spec = importlib.util.spec_from_file_location("annual_feature_name_probe", runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
factory = getattr(module, {factory_name!r})
X = pd.DataFrame(
    {{
        "feature_a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "feature_b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    }}
)

for model_name in {model_names!r}:
    is_classifier = "participation" in model_name
    target = (
        np.array([0, 1, 0, 1, 0, 1], dtype=int)
        if is_classifier
        else np.arange(len(X), dtype=float)
    )
    model = factory(model_name, {{}})
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="X does not have valid feature names, but LGBM.*",
            category=UserWarning,
        )
        model.fit(X, target)
        if is_classifier:
            model.predict_proba(X)
        else:
            model.predict(X)
    transformed = model[:-1].transform(X)
    assert isinstance(transformed, pd.DataFrame), type(transformed)
    assert list(transformed.columns) == list(
        model.named_steps["model"].feature_names_in_
    )

print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "ok"


@pytest.mark.parametrize(
    ("runner", "factory_name", "model_name"),
    (
        (
            ANNUAL_RUNNERS[0],
            "_model_pipeline",
            "conditional_ppg_lasso",
        ),
        (
            ANNUAL_RUNNERS[1],
            "model_pipeline",
            "next_residual_lasso",
        ),
    ),
    ids=("locked", "next_year"),
)
def test_annual_lasso_pipelines_have_solver_convergence_headroom(
    runner: Path,
    factory_name: str,
    model_name: str,
) -> None:
    probe = f"""
import importlib.util
from pathlib import Path

runner = Path({str(runner)!r})
spec = importlib.util.spec_from_file_location("annual_lasso_probe", runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = getattr(module, {factory_name!r})({model_name!r}, {{"alpha": 0.001}})
estimator = model.named_steps["model"]
assert estimator.max_iter == 50_000
assert estimator.tol == 1e-6
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "ok"


@pytest.mark.parametrize(
    ("runner", "factory_name", "model_name"),
    (
        (
            ANNUAL_RUNNERS[0],
            "_model_pipeline",
            "conditional_ppg_random_forest",
        ),
        (
            ANNUAL_RUNNERS[1],
            "model_pipeline",
            "next_residual_random_forest",
        ),
    ),
    ids=("locked", "next_year"),
)
def test_annual_random_forests_use_four_workers(
    runner: Path,
    factory_name: str,
    model_name: str,
) -> None:
    probe = f"""
import importlib.util
from pathlib import Path

runner = Path({str(runner)!r})
spec = importlib.util.spec_from_file_location("annual_rf_probe", runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = getattr(module, {factory_name!r})({model_name!r}, {{}})
assert model.named_steps["model"].n_jobs == 4
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "ok"
