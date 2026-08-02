from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from Scripts.V2 import native_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
ANNUAL_RUNNERS = (
    (
        REPO_ROOT
        / "research"
        / "studies"
        / "2026-07-29_v2_locked_final_validation"
        / "run_validation.py",
        "locked",
    ),
    (
        REPO_ROOT
        / "research"
        / "studies"
        / "2026-07-29_v2_next_year_residual"
        / "run_validation.py",
        "next_year",
    ),
)


def _load_runner(runner: Path, runner_kind: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"annual_dispatch_probe_{runner_kind}", runner
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load annual runner: {runner}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("runner", "runner_kind"),
    ANNUAL_RUNNERS,
    ids=("locked", "next_year"),
)
def test_runtime_grid_dispatch_isolates_only_lightgbm(
    monkeypatch: pytest.MonkeyPatch,
    runner: Path,
    runner_kind: str,
) -> None:
    module = _load_runner(runner, runner_kind)
    isolated_calls: list[tuple[Path, str, tuple[object, ...], object]] = []
    local_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def isolate(
        module_path: Path,
        function_name: str,
        args: tuple[object, ...] = (),
        kwargs: object = None,
    ) -> pd.DataFrame:
        isolated_calls.append((module_path, function_name, args, kwargs))
        origins = args[-1]
        assert isinstance(origins, tuple)
        return pd.DataFrame({"isolated_origin": [origins[0]]})

    def local(*args: object, **kwargs: object) -> str:
        local_calls.append((args, kwargs))
        return "local"

    monkeypatch.setattr(module, "run_module_function_in_fresh_process", isolate)
    if runner_kind == "locked":
        monkeypatch.setattr(module, "_grid_predictions", local)
        wrapper = module._runtime_grid_predictions
        module.GRID_ORIGINS = (2024, 2025, 2026)
        target = pd.DataFrame({"season": [2023, 2024, 2025]})
        feature_columns = ("feature",)
        grid = ({"num_leaves": 7},)
        isolated_result = wrapper(
            target,
            feature_columns,
            "conditional_ppg_lightgbm",
            grid,
            probability=True,
        )
        assert wrapper(
            target,
            feature_columns,
            "conditional_ppg_lasso",
            grid,
            probability=False,
        ) == "local"
        expected_worker = "_grid_predictions_for_origins"
        expected_kwargs = {"probability": True}
        local_args, local_kwargs = local_calls[0]
        assert local_args[0] is target
        assert local_args[1:] == (
            feature_columns,
            "conditional_ppg_lasso",
            grid,
        )
        assert local_kwargs == {"probability": False}
    else:
        monkeypatch.setattr(module, "grid_predictions", local)
        wrapper = module.runtime_grid_predictions
        module.GRID_ORIGINS = (2024, 2025, 2026)
        target = pd.DataFrame(
            {
                "origin_season": [2022, 2023, 2024, 2025],
                "origin_expert_ppg": [8.0, 9.0, 10.0, 11.0],
                "next_conditional_ppg_training_eligible": [1, 1, 1, 1],
                "next_participation_target_available": [1, 1, 1, 1],
            }
        )
        isolated_result = wrapper(target, "next_residual_lightgbm")
        assert wrapper(target, "next_residual_lasso") == "local"
        expected_worker = "grid_predictions_for_origins"
        expected_kwargs = None
        local_args, local_kwargs = local_calls[0]
        assert local_args[0] is target
        assert local_args[1:] == ("next_residual_lasso",)
        assert local_kwargs == {}

    assert isolated_result["isolated_origin"].tolist() == [2024, 2025]
    assert len(local_calls) == 1
    assert len(isolated_calls) == 2
    for call, origin in zip(isolated_calls, (2024, 2025), strict=True):
        module_path, worker, args, kwargs = call
        assert module_path == runner.resolve()
        assert worker == expected_worker
        assert args[0] is target
        assert args[-1] == (origin,)
        assert kwargs == expected_kwargs
        if runner_kind == "locked":
            assert args[1:-1] == (
                feature_columns,
                "conditional_ppg_lightgbm",
                grid,
            )
        else:
            assert args[1:-1] == ("next_residual_lightgbm",)


@pytest.mark.parametrize(
    ("runner", "runner_kind"),
    ANNUAL_RUNNERS,
    ids=("locked", "next_year"),
)
def test_runtime_selected_dispatch_isolates_only_lightgbm(
    monkeypatch: pytest.MonkeyPatch,
    runner: Path,
    runner_kind: str,
) -> None:
    module = _load_runner(runner, runner_kind)
    isolated_calls: list[tuple[Path, str, tuple[object, ...], object]] = []
    local_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def isolate(
        module_path: Path,
        function_name: str,
        args: tuple[object, ...] = (),
        kwargs: object = None,
    ) -> pd.DataFrame:
        isolated_calls.append((module_path, function_name, args, kwargs))
        origins = args[-1]
        assert isinstance(origins, tuple)
        return pd.DataFrame(
            {
                "chunk_start": [origins[0]],
                "chunk_size": [len(origins)],
            }
        )

    def local(*args: object, **kwargs: object) -> str:
        local_calls.append((args, kwargs))
        return "local"

    monkeypatch.setattr(module, "run_module_function_in_fresh_process", isolate)
    origins = tuple(range(2017, 2027))
    selections = pd.DataFrame({"forecast_origin": origins})

    if runner_kind == "locked":
        monkeypatch.setattr(module, "_selected_predictions", local)
        wrapper = module._runtime_selected_predictions
        module.OUTER_SEASONS = origins[:-1]
        module.CURRENT_SEASON = origins[-1]
        targets = pd.DataFrame({"season": [2016]})
        candidates = pd.DataFrame({"season": origins})
        feature_columns = ("feature",)
        isolated_result = wrapper(
            targets,
            candidates,
            feature_columns,
            "conditional_ppg_lightgbm",
            "lightgbm_output",
            selections,
            probability=True,
            require_expert=False,
        )
        assert wrapper(
            targets,
            candidates,
            feature_columns,
            "conditional_ppg_lasso",
            "lasso_output",
            selections,
            probability=False,
            require_expert=True,
        ) == "local"
        expected_worker = "_selected_predictions_for_origins"
        expected_kwargs = {
            "probability": True,
            "require_expert": False,
        }
        local_args, local_kwargs = local_calls[0]
        assert local_args[0] is targets
        assert local_args[1] is candidates
        assert local_args[2:5] == (
            feature_columns,
            "conditional_ppg_lasso",
            "lasso_output",
        )
        assert local_args[5] is selections
        assert local_kwargs == {
            "probability": False,
            "require_expert": True,
        }
    else:
        monkeypatch.setattr(module, "selected_predictions", local)
        wrapper = module.runtime_selected_predictions
        module.PREDICTION_ORIGINS = origins
        targets = pd.DataFrame({"origin_season": [2016]})
        isolated_result = wrapper(
            targets,
            "next_residual_lightgbm",
            selections,
        )
        assert wrapper(
            targets,
            "next_residual_lasso",
            selections,
        ) == "local"
        expected_worker = "selected_predictions_for_origins"
        expected_kwargs = None
        local_args, local_kwargs = local_calls[0]
        assert local_args[0] is targets
        assert local_args[1] == "next_residual_lasso"
        assert local_args[2] is selections
        assert local_kwargs == {}

    expected_chunks = (origins[:8], origins[8:])
    assert isolated_result["chunk_start"].tolist() == [2017, 2025]
    assert isolated_result["chunk_size"].tolist() == [8, 2]
    assert len(local_calls) == 1
    assert len(isolated_calls) == 2
    for call, chunk in zip(isolated_calls, expected_chunks, strict=True):
        module_path, worker, args, kwargs = call
        assert module_path == runner.resolve()
        assert worker == expected_worker
        assert args[0] is targets
        assert args[-1] == chunk
        assert kwargs == expected_kwargs
        if runner_kind == "locked":
            assert args[1] is candidates
            assert args[2:5] == (
                feature_columns,
                "conditional_ppg_lightgbm",
                "lightgbm_output",
            )
            assert args[5] is selections
        else:
            assert args[1] == "next_residual_lightgbm"
            assert args[2] is selections


def test_fresh_process_helper_runs_real_annual_function() -> None:
    result = native_runtime.run_module_function_in_fresh_process(
        ANNUAL_RUNNERS[0][0],
        "_metric",
        args=([1.0, 2.0], [1.0, 2.0], False),
    )

    assert result == pytest.approx(0.0)


def test_isolated_lightgbm_predictions_match_inline_predictions() -> None:
    runner = ANNUAL_RUNNERS[0][0]
    module = _load_runner(runner, "locked_equivalence")
    target = pd.DataFrame(
        {
            "player_key": [f"player_{index}" for index in range(16)],
            "season": [2024] * 12 + [2025] * 4,
            "position": ["RB"] * 16,
            "actual": [float(index % 7) for index in range(16)],
            "feature_a": [float(index) for index in range(16)],
            "feature_b": [float(index % 3) for index in range(16)],
        }
    )
    grid = (module.LIGHTGBM_GRID[0],)
    args = (
        target,
        ("feature_a", "feature_b"),
        "conditional_ppg_lightgbm",
        grid,
        (2025,),
    )

    inline = module._grid_predictions_for_origins(*args)
    isolated = native_runtime.run_module_function_in_fresh_process(
        runner,
        "_grid_predictions_for_origins",
        args=args,
    )

    pd.testing.assert_frame_equal(inline, isolated, check_exact=True)


def test_fresh_process_helper_retries_one_broken_pool(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spawn_context = object()
    executor_attempts: list[int] = []
    submissions: list[tuple[object, ...]] = []

    class FakeFuture:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        def result(self) -> str:
            if self.attempt == 1:
                raise native_runtime.BrokenProcessPool("native worker died")
            return "recovered"

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            assert max_workers == 1
            assert mp_context is spawn_context
            self.attempt = len(executor_attempts) + 1
            executor_attempts.append(self.attempt)

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> bool:
            return False

        def submit(self, *args: object) -> FakeFuture:
            submissions.append(args)
            return FakeFuture(self.attempt)

    def get_context(method: str) -> object:
        assert method == "spawn"
        return spawn_context

    monkeypatch.setattr(
        native_runtime.multiprocessing,
        "get_context",
        get_context,
    )
    monkeypatch.setattr(
        native_runtime,
        "ProcessPoolExecutor",
        FakeExecutor,
    )

    result = native_runtime.run_module_function_in_fresh_process(
        ANNUAL_RUNNERS[0][0],
        "probe_function",
        args=("arg",),
        kwargs={"flag": True},
    )

    assert result == "recovered"
    assert executor_attempts == [1, 2]
    assert len(submissions) == 2
    expected_submission = (
        native_runtime._invoke_module_function,
        str(ANNUAL_RUNNERS[0][0].resolve()),
        "probe_function",
        ("arg",),
        {"flag": True},
    )
    assert submissions == [expected_submission, expected_submission]
    assert capsys.readouterr().out == (
        "[probe_function] native worker exited abruptly; "
        "retrying batch once\n"
    )


@pytest.mark.parametrize(
    ("runner", "runner_kind"),
    ANNUAL_RUNNERS,
    ids=("locked", "next_year"),
)
def test_repeated_lightgbm_loops_release_before_each_collection(
    runner: Path,
    runner_kind: str,
) -> None:
    """Both grid and selected-fit loops clean up every LightGBM fit."""

    probe = f"""
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

runner = Path({str(runner)!r})
runner_kind = {runner_kind!r}
spec = importlib.util.spec_from_file_location(
    f"annual_cleanup_probe_{{runner_kind}}", runner
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

phase = ""
collections = []


def tracked_collect():
    caller = inspect.currentframe().f_back
    assert "model" not in caller.f_locals, (
        f"{{phase}} retained the fitted model while collecting"
    )
    assert "prediction" not in caller.f_locals, (
        f"{{phase}} retained the prediction while collecting"
    )
    collections.append(phase)
    return 0


module.gc.collect = tracked_collect

if runner_kind == "locked":
    def fake_fit(model_name, parameters, train, feature_columns):
        return object()


    def fake_predict(model, frame, feature_columns, probability):
        return np.full(len(frame), 0.5, dtype=float)


    module._fit = fake_fit
    module._predict = fake_predict
    module.GRID_ORIGINS = (2025,)
    grid_target = pd.DataFrame(
        {{
            "player_key": ["train", "hold"],
            "season": [2024, 2025],
            "position": ["RB", "RB"],
            "actual": [1.0, 2.0],
            "feature": [0.1, 0.2],
        }}
    )

    phase = "grid"
    grid = module._grid_predictions(
        grid_target,
        ("feature",),
        "conditional_ppg_lightgbm",
        ({{"num_leaves": 7}}, {{"num_leaves": 15}}),
    )
    assert len(grid) == 2
    assert collections == ["grid", "grid"], collections

    module.OUTER_SEASONS = (2025,)
    module.CURRENT_SEASON = 2026
    train_target = pd.DataFrame(
        {{
            "player_key": ["train_1", "train_2", "train_3"],
            "season": [2023, 2024, 2025],
            "position": ["RB", "RB", "RB"],
            "actual": [1.0, 2.0, 3.0],
            "feature": [0.1, 0.2, 0.3],
        }}
    )
    candidates = pd.DataFrame(
        {{
            "player_key": ["hold_2025", "hold_2026"],
            "season": [2025, 2026],
            "position": ["RB", "RB"],
            "expert_ppg_team_game_median": [10.0, 11.0],
            "feature": [0.4, 0.5],
        }}
    )
    selections = pd.DataFrame(
        {{
            "forecast_origin": [2025, 2026],
            "model_name": [
                "conditional_ppg_lightgbm",
                "conditional_ppg_lightgbm",
            ],
            "parameters_json": ["{{}}", "{{}}"],
            "candidate_id": [0, 0],
        }}
    )

    phase = "selected"
    selected = module._selected_predictions(
        train_target,
        candidates,
        ("feature",),
        "conditional_ppg_lightgbm",
        "conditional_ppg_lightgbm_selected",
        selections,
    )
    assert len(selected) == 2
    assert collections == [
        "grid",
        "grid",
        "selected",
        "selected",
    ], collections
else:
    def fake_fit(model_name, parameters, train):
        return object()


    def fake_predict(model_name, model, hold):
        return np.full(len(hold), 0.5, dtype=float)


    module.fit_model = fake_fit
    module.predict_model = fake_predict
    module.GRID_ORIGINS = (2024,)
    module.MODEL_GRIDS["next_residual_lightgbm"] = (
        {{"num_leaves": 7}},
        {{"num_leaves": 15}},
    )
    targets = pd.DataFrame(
        {{
            "player_key": ["train_2022", "train_2023", "hold_2024", "hold_2025"],
            "origin_season": [2022, 2023, 2024, 2025],
            "position": ["RB", "RB", "RB", "RB"],
            "origin_expert_ppg": [8.0, 9.0, 10.0, 11.0],
            "next_residual_vs_expert": [0.1, 0.2, 0.3, 0.4],
            "next_conditional_ppg_training_eligible": [1, 1, 1, 1],
            "next_participation_target_available": [1, 1, 1, 1],
        }}
    )

    phase = "grid"
    grid = module.grid_predictions(targets, "next_residual_lightgbm")
    assert len(grid) == 2
    assert collections == ["grid", "grid"], collections

    module.PREDICTION_ORIGINS = (2024, 2025)
    selections = pd.DataFrame(
        {{
            "forecast_origin": [2024, 2025],
            "model_name": [
                "next_residual_lightgbm",
                "next_residual_lightgbm",
            ],
            "parameters_json": ["{{}}", "{{}}"],
            "candidate_id": [0, 0],
        }}
    )

    phase = "selected"
    selected = module.selected_predictions(
        targets,
        "next_residual_lightgbm",
        selections,
    )
    assert len(selected) == 2
    assert collections == [
        "grid",
        "grid",
        "selected",
        "selected",
    ], collections

print(json.dumps(collections))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == (
        '["grid", "grid", "selected", "selected"]'
    )
