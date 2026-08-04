"""Spawn-isolated LightGBM fitting helpers for the boosting-grid study."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.locked_candidates import PRIMARY_PPG_FEATURES


def _pipeline(parameters: Mapping[str, object], estimator_seed: int) -> Pipeline:
    estimator = LGBMRegressor(
        objective="regression",
        verbosity=-1,
        subsample=1.0,
        colsample_bytree=1.0,
        deterministic=True,
        force_col_wise=True,
        random_state=estimator_seed,
        n_jobs=1,
        **dict(parameters),
    )
    pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("model", estimator),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def _fit_predict(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    parameters: Mapping[str, object],
    estimator_seed: int,
) -> np.ndarray:
    features = list(PRIMARY_PPG_FEATURES)
    model = _pipeline(parameters, estimator_seed)
    model.fit(
        train[features].apply(pd.to_numeric, errors="coerce"),
        train["actual"].to_numpy(float),
    )
    prediction = np.asarray(
        model.predict(predict[features].apply(pd.to_numeric, errors="coerce")),
        dtype=float,
    )
    del model
    gc.collect()
    return prediction


def score_candidate_chunk(
    train: pd.DataFrame,
    hold: pd.DataFrame,
    candidates: Sequence[tuple[int, Mapping[str, object]]],
    estimator_seed: int,
) -> list[dict[str, float | int]]:
    """Score at most eight candidates in one fresh native worker."""
    if len(candidates) > 8:
        raise ValueError("LightGBM worker chunk exceeds the eight-fit ceiling")
    actual = hold["actual"].to_numpy(float)
    rows = []
    for candidate_id, parameters in candidates:
        prediction = _fit_predict(train, hold, parameters, estimator_seed)
        rows.append(
            {
                "candidate_id": int(candidate_id),
                "rows": len(hold),
                "rmse": float(np.sqrt(np.mean(np.square(actual - prediction)))),
            }
        )
    return rows


def fit_selected(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    parameters: Mapping[str, object],
    estimator_seed: int,
) -> list[float]:
    return _fit_predict(train, predict, parameters, estimator_seed).tolist()

