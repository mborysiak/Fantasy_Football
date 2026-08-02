from __future__ import annotations

import sqlite3

import pandas as pd

from Scripts.V2.parameter_cache import (
    LOCKED_CACHE_RUNNER,
    load_parameter_cache,
    parameter_fingerprint,
    write_parameter_cache,
)


GRID = ({"alpha": 0.1}, {"alpha": 1.0})
ORIGINS = (2025, 2026)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": ["b", "a"],
            "season": [2024, 2023],
            "actual": [2.0, 1.0],
            "feature": [20.0, 10.0],
        }
    )


def _selections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["model", "model"],
            "forecast_origin": [2025, 2026],
            "candidate_id": [0, 1],
            "parameters_json": [
                '{"alpha": 0.1}',
                '{"alpha": 1.0}',
            ],
            "selection_score": [1.2, 1.1],
        }
    )


def _fingerprint(frame: pd.DataFrame) -> str:
    return parameter_fingerprint(
        frame=frame,
        data_columns=("player_key", "season", "actual", "feature"),
        specification={"grid": list(GRID), "seed": 1234},
    )


def test_parameter_cache_round_trip_and_order_independent_fingerprint(
    tmp_path,
):
    database = tmp_path / "cache.sqlite3"
    frame = _training_frame()
    fingerprint = _fingerprint(frame)
    assert fingerprint == _fingerprint(frame.iloc[::-1].reset_index(drop=True))

    written = write_parameter_cache(
        database,
        season=2026,
        league="dk",
        runner=LOCKED_CACHE_RUNNER,
        model_name="model",
        fingerprint_sha256=fingerprint,
        expected_origins=ORIGINS,
        grid=GRID,
        selections=_selections(),
    )
    loaded, receipt = load_parameter_cache(
        database,
        season=2026,
        league="dk",
        runner=LOCKED_CACHE_RUNNER,
        model_name="model",
        fingerprint_sha256=fingerprint,
        expected_origins=ORIGINS,
        grid=GRID,
    )

    assert written["cache_status"] == "miss_written"
    assert receipt["cache_status"] == "hit"
    assert receipt["cache_hit"] == 1
    assert loaded is not None
    assert loaded["candidate_id"].tolist() == [0, 1]


def test_parameter_cache_invalidates_when_training_data_changes(tmp_path):
    database = tmp_path / "cache.sqlite3"
    frame = _training_frame()
    write_parameter_cache(
        database,
        season=2026,
        league="dk",
        runner=LOCKED_CACHE_RUNNER,
        model_name="model",
        fingerprint_sha256=_fingerprint(frame),
        expected_origins=ORIGINS,
        grid=GRID,
        selections=_selections(),
    )
    changed = frame.copy()
    changed.loc[0, "feature"] += 0.01

    loaded, receipt = load_parameter_cache(
        database,
        season=2026,
        league="dk",
        runner=LOCKED_CACHE_RUNNER,
        model_name="model",
        fingerprint_sha256=_fingerprint(changed),
        expected_origins=ORIGINS,
        grid=GRID,
    )

    assert loaded is None
    assert receipt["cache_status"] == "fingerprint_mismatch"


def test_parameter_cache_rejects_tampered_selection_payload(tmp_path):
    database = tmp_path / "cache.sqlite3"
    frame = _training_frame()
    fingerprint = _fingerprint(frame)
    write_parameter_cache(
        database,
        season=2026,
        league="dk",
        runner=LOCKED_CACHE_RUNNER,
        model_name="model",
        fingerprint_sha256=fingerprint,
        expected_origins=ORIGINS,
        grid=GRID,
        selections=_selections(),
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE annual_model_parameter_cache "
            "SET selections_json='[]'"
        )
        connection.commit()

    loaded, receipt = load_parameter_cache(
        database,
        season=2026,
        league="dk",
        runner=LOCKED_CACHE_RUNNER,
        model_name="model",
        fingerprint_sha256=fingerprint,
        expected_origins=ORIGINS,
        grid=GRID,
    )

    assert loaded is None
    assert receipt["cache_status"] == "selection_hash_mismatch"
