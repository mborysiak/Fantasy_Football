"""Test one-sided provider projection disagreement in V2 and templates.

The study is read-only with respect to production databases.  It replays the
locked conditional-PPG model, fits strictly-prior upper-residual classifiers,
and replays the weekly donor matcher with frozen asymmetric-gap weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parent
LOCKED_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_locked_final_validation"
    / "run_validation.py"
)
TEMPLATE_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-23_template_feature_pruning"
    / "run_validation.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "Scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
REPOSITORY_PARENT = REPO_ROOT.parent
if str(REPOSITORY_PARENT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_PARENT))
FF_PACKAGE_ROOT = REPOSITORY_PARENT / "ff"
if str(FF_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FF_PACKAGE_ROOT))

from Scripts.V2.contracts import scoring_hash
from Scripts.V2.locked_candidates import (
    LOCKED_BLEND_WEIGHTS,
    PRIMARY_PPG_FEATURES,
    lock_version_for_scoring,
)


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
POINT_VARIANT_FEATURES = {
    "incumbent": (),
    "max_minus_median_raw": (
        "expert_ppg_bull_gap",
        "expert_ppg_bull_gap_available",
    ),
    "max_minus_median_fraction": (
        "expert_ppg_bull_gap_fraction",
        "expert_ppg_bull_gap_available",
    ),
    "asymmetric_robust_stack": (
        "expert_ppg_bull_gap_fraction",
        "expert_ppg_bear_gap_fraction",
        "expert_ppg_top2_gap_fraction",
        "expert_ppg_bull_gap_available",
    ),
}
PRIMARY_POINT_VARIANT = "max_minus_median_fraction"
MODEL_COMPONENTS = tuple(LOCKED_BLEND_WEIGHTS)
RANDOM_FOREST_COMPONENT = "conditional_ppg_random_forest"
FULL_COLUMN_RF_METHOD = "random_forest_full_columns"
CONTROLLED_METHOD = "controlled_equal_thirds"
PRODUCTION_METHOD = "equal_thirds"
TAIL_VARIANT_FEATURES = {
    "tail_symmetric": (),
    "tail_bullish": (
        "expert_ppg_bull_gap_fraction",
        "expert_ppg_bull_gap_position_percentile",
        "expert_ppg_bull_gap_available",
    ),
    "tail_asymmetric": (
        "expert_ppg_bull_gap_fraction",
        "expert_ppg_bear_gap_fraction",
        "expert_ppg_top2_gap_fraction",
        "expert_ppg_bull_gap_available",
    ),
}
TAIL_BASE_FEATURES = (
    "point_prediction",
    "expert_ppg_team_game_median",
    "expert_ppg_team_game_std",
    "expert_points_iqr",
    "projection_provider_count",
)
TAIL_EVENTS = {"plus3": 3.0, "plus5": 5.0}
MIN_ASYMMETRIC_PROVIDERS = 3
FRACTION_DENOMINATOR_FLOOR = 1.0
RANDOM_SEED = 1234
TEMPLATE_PERIODS = {
    "full_2017_2025": (2017, 2025),
    "recent_2020_2025": (2020, 2025),
    "temporal_2023_2025": (2023, 2025),
}
PRIMARY_TEMPLATE_METHOD = "bull_add_w050"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta", "all"), default="all")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--template-bootstrap-iterations", type=int, default=2_000)
    parser.add_argument(
        "--skip-templates",
        action="store_true",
        help="Run only the point and upper-residual portions.",
    )
    parser.add_argument(
        "--combine-existing",
        action="store_true",
        help="Combine already completed DK and beta result receipts.",
    )
    return parser.parse_args()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load study runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_only_connection(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _top_two_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if len(numeric) < MIN_ASYMMETRIC_PROVIDERS:
        return np.nan
    return float(numeric.iloc[-2:].mean())


def build_asymmetric_projection_features(
    projection_values: pd.DataFrame,
) -> pd.DataFrame:
    """Build max-minus-median features from configured scored provider PPG."""

    required = {
        "player_key",
        "season",
        "position",
        "provider",
        "configured_points_complete",
        "provider_points_per_team_game",
    }
    missing = sorted(required.difference(projection_values.columns))
    if missing:
        raise ValueError(f"Projection rows are missing columns: {missing}")
    rows = projection_values.copy()
    rows["season"] = pd.to_numeric(rows["season"], errors="raise").astype(int)
    rows["position"] = rows["position"].astype("string").str.strip().str.upper()
    rows["provider"] = rows["provider"].astype("string").str.strip().str.lower()
    rows["provider_points_per_team_game"] = pd.to_numeric(
        rows["provider_points_per_team_game"], errors="coerce"
    )
    rows = rows[
        pd.to_numeric(rows["configured_points_complete"], errors="coerce").eq(1)
        & rows["provider_points_per_team_game"].notna()
        & np.isfinite(rows["provider_points_per_team_game"])
        & rows["position"].isin(("QB", "RB", "WR", "TE"))
    ].copy()
    duplicate = rows.duplicated(
        ["player_key", "season", "position", "provider"], keep=False
    )
    if duplicate.any():
        preview = rows.loc[
            duplicate, ["player_key", "season", "position", "provider"]
        ].head(10)
        raise ValueError(
            "Configured provider PPG rows contain duplicate keys: "
            f"{preview.to_dict('records')}"
        )
    grouped = (
        rows.groupby(["player_key", "season", "position"], sort=True)
        .agg(
            expert_ppg_gap_provider_count=("provider", "nunique"),
            expert_ppg_gap_median=("provider_points_per_team_game", "median"),
            expert_ppg_gap_max=("provider_points_per_team_game", "max"),
            expert_ppg_gap_min=("provider_points_per_team_game", "min"),
            expert_ppg_gap_top2_mean=(
                "provider_points_per_team_game",
                _top_two_mean,
            ),
        )
        .reset_index()
    )
    available = grouped["expert_ppg_gap_provider_count"].ge(
        MIN_ASYMMETRIC_PROVIDERS
    )
    grouped["expert_ppg_bull_gap_available"] = available.astype(int)
    grouped["expert_ppg_bull_gap"] = (
        grouped["expert_ppg_gap_max"] - grouped["expert_ppg_gap_median"]
    ).where(available)
    grouped["expert_ppg_bear_gap"] = (
        grouped["expert_ppg_gap_median"] - grouped["expert_ppg_gap_min"]
    ).where(available)
    grouped["expert_ppg_top2_gap"] = (
        grouped["expert_ppg_gap_top2_mean"] - grouped["expert_ppg_gap_median"]
    ).where(available)
    denominator = grouped["expert_ppg_gap_median"].abs().clip(
        lower=FRACTION_DENOMINATOR_FLOOR
    )
    for stem in ("bull", "bear", "top2"):
        grouped[f"expert_ppg_{stem}_gap_fraction"] = (
            grouped[f"expert_ppg_{stem}_gap"] / denominator
        ).where(available)
    grouped["expert_ppg_gap_asymmetry_fraction"] = (
        grouped["expert_ppg_bull_gap_fraction"]
        - grouped["expert_ppg_bear_gap_fraction"]
    )
    for stem in ("bull", "bear", "top2"):
        source = f"expert_ppg_{stem}_gap_fraction"
        target = f"expert_ppg_{stem}_gap_position_percentile"
        grouped[target] = grouped.groupby(
            ["season", "position"], sort=True
        )[source].rank(method="average", pct=True)
    gap_columns = [
        "expert_ppg_bull_gap",
        "expert_ppg_bear_gap",
        "expert_ppg_top2_gap",
    ]
    if (grouped[gap_columns].dropna() < -1e-12).any().any():
        raise ValueError("A one-sided expert projection gap is negative")
    return grouped


def _load_projection_values(database: Path) -> pd.DataFrame:
    with _read_only_connection(database) as connection:
        return pd.read_sql_query(
            """
            SELECT player_key,
                   CAST(season AS INTEGER) season,
                   position,
                   provider,
                   configured_points_complete,
                   provider_points_per_team_game,
                   run_id
            FROM player_season_projection_values
            """,
            connection,
        )


def _load_point_inputs(database: Path, league: str):
    locked = _load_module(
        LOCKED_RUNNER_PATH,
        f"v2_asymmetric_locked_runner_{league}",
    )
    locked.ACTIVE_OUTPUT_DB_PATH = database
    locked.ACTIVE_RESULTS_DIR = STUDY_ROOT / "artifacts" / "local"
    locked.ACTIVE_SCORING_OBJECTIVE = league
    locked.ACTIVE_LOCK_VERSION = lock_version_for_scoring(league)
    features, _, feature_run_id = locked._load_inputs()
    with _read_only_connection(database) as connection:
        selected = pd.read_sql_query(
            "SELECT * FROM locked_selected_hyperparameters", connection
        )
        locked_predictions = pd.read_sql_query(
            "SELECT * FROM locked_whole_season_predictions", connection
        )
        candidate_runs = pd.read_sql_query(
            "SELECT * FROM locked_candidate_runs", connection
        )
    lock_version = lock_version_for_scoring(league)
    active = candidate_runs[candidate_runs["lock_version"].eq(lock_version)].copy()
    if len(active) != 1:
        raise ValueError(
            f"Expected one locked run for {lock_version}; observed {len(active)}"
        )
    active_run = active.iloc[0]
    if str(active_run.feature_run_id) != str(feature_run_id):
        raise ValueError("Locked point-model feature lineage does not match the mart")
    model_run_id = str(active_run.model_run_id)
    selected = selected[
        selected["model_run_id"].astype(str).eq(model_run_id)
        & selected["lock_version"].eq(lock_version)
    ].copy()
    locked_predictions = locked_predictions[
        locked_predictions["model_run_id"].astype(str).eq(model_run_id)
        & locked_predictions["lock_version"].eq(lock_version)
    ].copy()
    projection_values = _load_projection_values(database)
    gap_features = build_asymmetric_projection_features(projection_values)
    features = features.merge(
        gap_features,
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    features["expert_ppg_bull_gap_available"] = features[
        "expert_ppg_bull_gap_available"
    ].fillna(0).astype(int)
    features["expert_ppg_gap_provider_count"] = features[
        "expert_ppg_gap_provider_count"
    ].fillna(0).astype(int)
    return (
        locked,
        features,
        selected,
        locked_predictions,
        gap_features,
        {
            "lock_version": lock_version,
            "model_run_id": model_run_id,
            "feature_run_id": str(feature_run_id),
            "locked_created_at_utc": str(active_run.created_at_utc),
        },
    )


def _point_feature_columns(variant: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*PRIMARY_PPG_FEATURES, *POINT_VARIANT_FEATURES[variant]))
    )


def _run_point_predictions(locked, features, selected) -> pd.DataFrame:
    ppg, _, candidates = locked._target_frames(features)
    frames: list[pd.DataFrame] = []
    for variant in POINT_VARIANT_FEATURES:
        columns = _point_feature_columns(variant)
        missing = sorted(set(columns).difference(features.columns))
        if missing:
            raise ValueError(f"{variant} is missing point columns: {missing}")
        for component in MODEL_COMPONENTS:
            component_selection = selected[
                selected["model_name"].eq(component)
            ].copy()
            frames.append(
                locked._selected_predictions(
                    ppg,
                    candidates,
                    columns,
                    fit_model_name=component,
                    output_model_name=f"{variant}__{component}",
                    selected=component_selection,
                )
            )
    full_column_selected = selected[
        selected["model_name"].eq(RANDOM_FOREST_COMPONENT)
    ].copy()
    full_column_selected["parameters_json"] = full_column_selected[
        "parameters_json"
    ].map(
        lambda value: json.dumps(
            {**json.loads(value), "max_features": 1.0}, sort_keys=True
        )
    )
    for variant in POINT_VARIANT_FEATURES:
        frames.append(
            locked._selected_predictions(
                ppg,
                candidates,
                _point_feature_columns(variant),
                fit_model_name=RANDOM_FOREST_COMPONENT,
                output_model_name=(
                    f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns"
                ),
                selected=full_column_selected,
            )
        )
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot(
        index=["player_key", "season", "position"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    metadata = [
        "player_key",
        "season",
        "position",
        "conditional_ppg",
        "conditional_ppg_training_eligible",
        "has_prior_outcome",
        "is_rookie",
        "year_exp",
    ]
    output = candidates[metadata].merge(
        wide,
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    output = locked._add_history_depth(output)
    for variant in POINT_VARIANT_FEATURES:
        component_columns = [
            f"{variant}__{component}" for component in MODEL_COMPONENTS
        ]
        component_values = output[component_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        weights = np.asarray(
            [LOCKED_BLEND_WEIGHTS[component] for component in MODEL_COMPONENTS],
            dtype=float,
        )
        output[f"{variant}__{PRODUCTION_METHOD}"] = (
            component_values.to_numpy().dot(weights)
        )
        output.loc[
            component_values.isna().any(axis=1),
            f"{variant}__{PRODUCTION_METHOD}",
        ] = np.nan
        controlled_columns = [
            f"{variant}__conditional_ppg_lasso",
            f"{variant}__conditional_ppg_lightgbm",
            f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns",
        ]
        controlled = output[controlled_columns].apply(pd.to_numeric, errors="coerce")
        output[f"{variant}__{CONTROLLED_METHOD}"] = controlled.mean(axis=1)
        output.loc[
            controlled.isna().any(axis=1), f"{variant}__{CONTROLLED_METHOD}"
        ] = np.nan
    tail_columns = [
        "player_key",
        "season",
        "position",
        "expert_ppg_team_game_median",
        "expert_ppg_team_game_std",
        "expert_points_iqr",
        "projection_provider_count",
        "expert_ppg_bull_gap",
        "expert_ppg_bear_gap",
        "expert_ppg_top2_gap",
        "expert_ppg_bull_gap_fraction",
        "expert_ppg_bear_gap_fraction",
        "expert_ppg_top2_gap_fraction",
        "expert_ppg_bull_gap_position_percentile",
        "expert_ppg_bear_gap_position_percentile",
        "expert_ppg_top2_gap_position_percentile",
        "expert_ppg_bull_gap_available",
        "expert_ppg_gap_provider_count",
    ]
    output = output.merge(
        features[tail_columns],
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    return output


def _assert_incumbent_reproduction(
    predictions: pd.DataFrame,
    locked_predictions: pd.DataFrame,
) -> float:
    checks = {
        **{
            component: f"incumbent__{component}"
            for component in MODEL_COMPONENTS
        },
        "conditional_ppg_primary_blend": f"incumbent__{PRODUCTION_METHOD}",
    }
    differences: list[float] = []
    keys = ["player_key", "season"]
    for locked_method, challenger_column in checks.items():
        expected = locked_predictions[
            locked_predictions["method"].eq(locked_method)
        ][keys + ["prediction"]].dropna(subset=["prediction"])
        universe = (
            predictions["season"].isin(expected["season"].unique())
            & predictions["conditional_ppg_training_eligible"].eq(1)
            & predictions["conditional_ppg"].notna()
        )
        observed = predictions.loc[
            universe, keys + [challenger_column]
        ].rename(columns={challenger_column: "observed"})
        observed = observed.dropna(subset=["observed"])
        key_audit = expected[keys].merge(
            observed[keys], on=keys, how="outer", indicator=True,
            validate="one_to_one",
        )
        if not key_audit["_merge"].eq("both").all():
            raise ValueError(
                f"Incumbent reproduction key mismatch for {locked_method}: "
                f"{key_audit['_merge'].value_counts().to_dict()}"
            )
        compared = expected.merge(observed, on=keys, validate="one_to_one")
        differences.extend(
            (compared["prediction"] - compared["observed"]).abs().tolist()
        )
    if not differences:
        raise ValueError("No incumbent point predictions were reproduced")
    maximum = float(max(differences))
    if maximum > 1e-10:
        raise ValueError(f"Incumbent reproduction failed: {maximum}")
    return maximum


def _point_evaluation(
    predictions: pd.DataFrame,
    outer_seasons: Sequence[int],
) -> pd.DataFrame:
    eligible = (
        predictions["season"].isin(outer_seasons)
        & predictions["conditional_ppg_training_eligible"].eq(1)
        & predictions["conditional_ppg"].notna()
    )
    metadata = [
        "player_key",
        "season",
        "position",
        "history_depth",
        "conditional_ppg",
    ]
    rows: list[pd.DataFrame] = []
    for variant in POINT_VARIANT_FEATURES:
        methods = {
            component.removeprefix("conditional_ppg_"): f"{variant}__{component}"
            for component in MODEL_COMPONENTS
        }
        methods[PRODUCTION_METHOD] = f"{variant}__{PRODUCTION_METHOD}"
        methods[FULL_COLUMN_RF_METHOD] = (
            f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns"
        )
        methods[CONTROLLED_METHOD] = f"{variant}__{CONTROLLED_METHOD}"
        for method, column in methods.items():
            current = predictions.loc[eligible, metadata].copy()
            current["variant"] = variant
            current["method"] = method
            current["actual"] = current.pop("conditional_ppg")
            current["prediction"] = predictions.loc[eligible, column].to_numpy()
            current = current[current["prediction"].notna()].copy()
            current["squared_error"] = np.square(
                current["actual"] - current["prediction"]
            )
            rows.append(current)
    return pd.concat(rows, ignore_index=True)


def _rmse_cluster_interval(
    compared: pd.DataFrame,
    cluster: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    grouped = compared.groupby(cluster, sort=True).agg(
        baseline_sum=("baseline_score", "sum"),
        challenger_sum=("challenger_score", "sum"),
        n_rows=("player_key", "size"),
    )
    values = grouped[["baseline_sum", "challenger_sum", "n_rows"]].to_numpy(
        dtype=float
    )
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        draws[index] = np.sqrt(sampled[1] / sampled[2]) - np.sqrt(
            sampled[0] / sampled[2]
        )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _point_comparison_summary(
    evaluation: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in (CONTROLLED_METHOD, PRODUCTION_METHOD):
        selected = evaluation[evaluation["method"].eq(method)]
        baseline = selected[selected["variant"].eq("incumbent")][
            ["player_key", "season", "position", "squared_error"]
        ].rename(columns={"squared_error": "baseline_score"})
        for challenger in POINT_VARIANT_FEATURES:
            if challenger == "incumbent":
                continue
            challenge = selected[selected["variant"].eq(challenger)][
                ["player_key", "season", "squared_error"]
            ].rename(columns={"squared_error": "challenger_score"})
            compared = baseline.merge(
                challenge,
                on=["player_key", "season"],
                how="inner",
                validate="one_to_one",
            )
            if len(compared) != len(baseline) or len(compared) != len(challenge):
                raise ValueError(f"Unmatched point rows for {challenger}")
            baseline_rmse = float(np.sqrt(compared["baseline_score"].mean()))
            challenger_rmse = float(np.sqrt(compared["challenger_score"].mean()))
            by_season = compared.groupby("season", sort=True).agg(
                baseline=("baseline_score", "mean"),
                challenger=("challenger_score", "mean"),
            )
            by_season["delta"] = np.sqrt(by_season["challenger"]) - np.sqrt(
                by_season["baseline"]
            )
            recent = compared[compared["season"].ge(2023)]
            position_delta = {
                str(position): float(
                    np.sqrt(group["challenger_score"].mean())
                    - np.sqrt(group["baseline_score"].mean())
                )
                for position, group in compared.groupby("position", sort=True)
            }
            season_low, season_high = _rmse_cluster_interval(
                compared,
                "season",
                iterations,
                np.random.default_rng(RANDOM_SEED),
            )
            player_low, player_high = _rmse_cluster_interval(
                compared,
                "player_key",
                iterations,
                np.random.default_rng(RANDOM_SEED + 1),
            )
            rows.append(
                {
                    "method": method,
                    "baseline_variant": "incumbent",
                    "challenger_variant": challenger,
                    "n_rows": len(compared),
                    "baseline_rmse": baseline_rmse,
                    "challenger_rmse": challenger_rmse,
                    "pooled_delta": challenger_rmse - baseline_rmse,
                    "recent_delta": float(
                        np.sqrt(recent["challenger_score"].mean())
                        - np.sqrt(recent["baseline_score"].mean())
                    ),
                    "season_wins": int(by_season["delta"].lt(0).sum()),
                    "season_count": int(len(by_season)),
                    "season_95_low": season_low,
                    "season_95_high": season_high,
                    "player_95_low": player_low,
                    "player_95_high": player_high,
                    "nonworse_positions": int(
                        sum(value <= 0 for value in position_delta.values())
                    ),
                    "position_count": len(position_delta),
                    "position_deltas_json": json.dumps(
                        position_delta, sort_keys=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def _tail_design(frame: pd.DataFrame, extra: Sequence[str]) -> pd.DataFrame:
    design = frame[[*TAIL_BASE_FEATURES, *extra]].copy()
    for position in ("QB", "RB", "WR", "TE"):
        design[f"position_{position}"] = frame["position"].eq(position).astype(int)
    return design


def _run_tail_models(
    predictions: pd.DataFrame,
    outer_seasons: Sequence[int],
) -> pd.DataFrame:
    center = f"incumbent__{CONTROLLED_METHOD}"
    required = [
        "player_key",
        "season",
        "position",
        "conditional_ppg",
        "conditional_ppg_training_eligible",
        center,
        *[column for column in TAIL_BASE_FEATURES if column != "point_prediction"],
        *dict.fromkeys(
            feature
            for features in TAIL_VARIANT_FEATURES.values()
            for feature in features
        ),
    ]
    data = predictions[required].copy()
    eligible = (
        data["season"].isin(outer_seasons)
        & data["conditional_ppg_training_eligible"].eq(1)
        & data["conditional_ppg"].notna()
        & data[center].notna()
    )
    data = data[eligible].copy()
    data.rename(
        columns={"conditional_ppg": "actual", center: "point_prediction"},
        inplace=True,
    )
    data["residual"] = data["actual"] - data["point_prediction"]
    frames: list[pd.DataFrame] = []
    seasons = sorted(data["season"].astype(int).unique())
    for target_season in seasons[1:]:
        train = data[data["season"].lt(target_season)].copy()
        target = data[data["season"].eq(target_season)].copy()
        if train.empty or target.empty:
            continue
        for event, threshold in TAIL_EVENTS.items():
            train_y = train["residual"].ge(threshold).astype(int)
            target_y = target["residual"].ge(threshold).astype(int)
            for variant, extra in TAIL_VARIANT_FEATURES.items():
                if train_y.nunique() < 2:
                    probability = np.repeat(
                        (float(train_y.sum()) + 0.5) / (len(train_y) + 1),
                        len(target),
                    )
                    fit_method = "strict_prior_smoothed_constant"
                else:
                    pipeline = Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value=0.0,
                                    add_indicator=True,
                                ),
                            ),
                            ("scaler", StandardScaler()),
                            (
                                "logistic",
                                LogisticRegression(
                                    C=1.0,
                                    solver="liblinear",
                                    max_iter=2_000,
                                    random_state=RANDOM_SEED,
                                ),
                            ),
                        ]
                    )
                    pipeline.fit(_tail_design(train, extra), train_y)
                    probability = pipeline.predict_proba(
                        _tail_design(target, extra)
                    )[:, 1]
                    fit_method = "strict_prior_logistic_c1"
                probability = np.clip(probability, 1e-6, 1 - 1e-6)
                current = target[
                    [
                        "player_key",
                        "season",
                        "position",
                        "actual",
                        "point_prediction",
                        "residual",
                    ]
                ].copy()
                current["event"] = event
                current["threshold"] = threshold
                current["variant"] = variant
                current["outcome"] = target_y.to_numpy()
                current["probability"] = probability
                current["brier"] = np.square(probability - target_y.to_numpy())
                current["log_loss"] = -(
                    target_y.to_numpy() * np.log(probability)
                    + (1 - target_y.to_numpy()) * np.log(1 - probability)
                )
                current["training_start"] = int(train["season"].min())
                current["training_end"] = int(train["season"].max())
                current["training_rows"] = len(train)
                current["fit_method"] = fit_method
                frames.append(current)
    if not frames:
        raise ValueError("No strictly-prior tail predictions were produced")
    output = pd.concat(frames, ignore_index=True)
    counts = output.groupby(["event", "variant"]).size()
    if counts.groupby(level=0).nunique().ne(1).any():
        raise ValueError(f"Tail variants have unequal rows: {counts.to_dict()}")
    return output


def _mean_cluster_interval(
    compared: pd.DataFrame,
    cluster: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    grouped = compared.groupby(cluster, sort=True).agg(
        delta_sum=("delta", "sum"), n_rows=("player_key", "size")
    )
    values = grouped[["delta_sum", "n_rows"]].to_numpy(dtype=float)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        draws[index] = sampled[0] / sampled[1]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _safe_auc(outcome: pd.Series, probability: pd.Series) -> float:
    if outcome.nunique() < 2:
        return np.nan
    return float(roc_auc_score(outcome, probability))


def _tail_comparison_summary(
    evaluation: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in TAIL_EVENTS:
        selected = evaluation[evaluation["event"].eq(event)]
        baseline = selected[selected["variant"].eq("tail_symmetric")][
            [
                "player_key",
                "season",
                "position",
                "outcome",
                "probability",
                "brier",
                "log_loss",
            ]
        ].rename(
            columns={
                "probability": "baseline_probability",
                "brier": "baseline_brier",
                "log_loss": "baseline_log_loss",
            }
        )
        for challenger in ("tail_bullish", "tail_asymmetric"):
            challenge = selected[selected["variant"].eq(challenger)][
                ["player_key", "season", "probability", "brier", "log_loss"]
            ].rename(
                columns={
                    "probability": "challenger_probability",
                    "brier": "challenger_brier",
                    "log_loss": "challenger_log_loss",
                }
            )
            compared = baseline.merge(
                challenge,
                on=["player_key", "season"],
                how="inner",
                validate="one_to_one",
            )
            if len(compared) != len(baseline) or len(compared) != len(challenge):
                raise ValueError(f"Unmatched tail rows for {event}/{challenger}")
            compared["delta"] = (
                compared["challenger_brier"] - compared["baseline_brier"]
            )
            by_season = compared.groupby("season", sort=True)["delta"].mean()
            recent = compared[compared["season"].ge(2023)]
            season_low, season_high = _mean_cluster_interval(
                compared,
                "season",
                iterations,
                np.random.default_rng(RANDOM_SEED + 2),
            )
            player_low, player_high = _mean_cluster_interval(
                compared,
                "player_key",
                iterations,
                np.random.default_rng(RANDOM_SEED + 3),
            )
            rows.append(
                {
                    "event": event,
                    "baseline_variant": "tail_symmetric",
                    "challenger_variant": challenger,
                    "n_rows": len(compared),
                    "event_rate": float(compared["outcome"].mean()),
                    "baseline_brier": float(compared["baseline_brier"].mean()),
                    "challenger_brier": float(
                        compared["challenger_brier"].mean()
                    ),
                    "brier_delta": float(compared["delta"].mean()),
                    "recent_brier_delta": float(recent["delta"].mean()),
                    "baseline_log_loss": float(
                        compared["baseline_log_loss"].mean()
                    ),
                    "challenger_log_loss": float(
                        compared["challenger_log_loss"].mean()
                    ),
                    "log_loss_delta": float(
                        (
                            compared["challenger_log_loss"]
                            - compared["baseline_log_loss"]
                        ).mean()
                    ),
                    "baseline_auc": _safe_auc(
                        compared["outcome"], compared["baseline_probability"]
                    ),
                    "challenger_auc": _safe_auc(
                        compared["outcome"], compared["challenger_probability"]
                    ),
                    "baseline_calibration_bias": float(
                        compared["baseline_probability"].mean()
                        - compared["outcome"].mean()
                    ),
                    "challenger_calibration_bias": float(
                        compared["challenger_probability"].mean()
                        - compared["outcome"].mean()
                    ),
                    "season_wins": int(by_season.lt(0).sum()),
                    "season_count": int(len(by_season)),
                    "season_95_low": season_low,
                    "season_95_high": season_high,
                    "player_95_low": player_low,
                    "player_95_high": player_high,
                }
            )
    return pd.DataFrame(rows)


def _gap_outcome_diagnostics(
    point_evaluation: pd.DataFrame,
    gap_features: pd.DataFrame,
) -> pd.DataFrame:
    """Describe outcomes by bullish-gap quartile without affecting gates."""

    point = point_evaluation[
        point_evaluation["variant"].eq("incumbent")
        & point_evaluation["method"].eq(CONTROLLED_METHOD)
    ][
        ["player_key", "season", "position", "actual", "prediction"]
    ].copy()
    point = point.merge(
        gap_features[
            [
                "player_key",
                "season",
                "position",
                "expert_ppg_gap_provider_count",
                "expert_ppg_bull_gap",
                "expert_ppg_bull_gap_fraction",
                "expert_ppg_bull_gap_position_percentile",
                "expert_ppg_bull_gap_available",
            ]
        ],
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    point["expert_ppg_bull_gap_available"] = point[
        "expert_ppg_bull_gap_available"
    ].fillna(0).astype(int)
    point["residual"] = point["actual"] - point["prediction"]
    point["absolute_error"] = point["residual"].abs()
    point["observed_plus3"] = point["residual"].ge(3).astype(int)
    point["observed_plus5"] = point["residual"].ge(5).astype(int)
    available = point[point["expert_ppg_bull_gap_available"].eq(1)].copy()
    available["bull_gap_quartile"] = pd.cut(
        available["expert_ppg_bull_gap_position_percentile"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=("Q1", "Q2", "Q3", "Q4"),
        include_lowest=True,
    ).astype("string")
    rows: list[dict[str, object]] = []
    for period, period_frame in (
        ("full_2017_2025", available),
        ("recent_2023_2025", available[available["season"].ge(2023)]),
    ):
        groups = [("all", "all", period_frame)]
        groups.extend(
            ("position", str(position), group)
            for position, group in period_frame.groupby("position", sort=True)
        )
        groups.extend(
            ("quartile", str(quartile), group)
            for quartile, group in period_frame.groupby(
                "bull_gap_quartile", observed=True, sort=True
            )
        )
        for scope, value, group in groups:
            rows.append(
                {
                    "period": period,
                    "scope": scope,
                    "scope_value": value,
                    "n_rows": len(group),
                    "mean_bull_gap_ppg": float(
                        group["expert_ppg_bull_gap"].mean()
                    ),
                    "mean_residual": float(group["residual"].mean()),
                    "mean_absolute_error": float(group["absolute_error"].mean()),
                    "plus3_rate": float(group["observed_plus3"].mean()),
                    "plus5_rate": float(group["observed_plus5"].mean()),
                    "spearman_gap_residual": float(
                        group["expert_ppg_bull_gap"].corr(
                            group["residual"], method="spearman"
                        )
                    ),
                    "spearman_gap_absolute_error": float(
                        group["expert_ppg_bull_gap"].corr(
                            group["absolute_error"], method="spearman"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_template_methods(builder) -> dict[str, dict[str, object]]:
    """Return frozen matcher variants; all use the governed 12-year recency."""

    incumbent = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
    bull = deepcopy(incumbent)
    asymmetric = deepcopy(incumbent)
    replacement = deepcopy(incumbent)
    for position in builder.POSITIONS:
        bull[position]["expert_ppg_bull_gap_position_percentile"] = 0.50
        bull[position]["expert_ppg_bull_gap_available"] = 0.25
        asymmetric[position]["expert_ppg_bull_gap_position_percentile"] = 0.25
        asymmetric[position]["expert_ppg_bear_gap_position_percentile"] = 0.25
        asymmetric[position]["expert_ppg_bull_gap_available"] = 0.25
        replacement[position].pop("projection_disagreement_frac", None)
        replacement[position]["expert_ppg_bull_gap_position_percentile"] = 0.75
        replacement[position]["expert_ppg_bull_gap_available"] = 0.25
    half_life = float(builder.TEMPLATE_RECENCY_HALF_LIFE)
    return {
        "incumbent": {
            "weights": incumbent,
            "recency_half_life": half_life,
            "variant": "incumbent",
            "removed_families": (),
        },
        PRIMARY_TEMPLATE_METHOD: {
            "weights": bull,
            "recency_half_life": half_life,
            "variant": "bull_add",
            "removed_families": (),
        },
        "asymmetric_add_w025": {
            "weights": asymmetric,
            "recency_half_life": half_life,
            "variant": "asymmetric_add",
            "removed_families": (),
        },
        "bull_replace_symmetric_w075": {
            "weights": replacement,
            "recency_half_life": half_life,
            "variant": "bull_replace_symmetric",
            "removed_families": ("projection_disagreement_frac",),
        },
    }


def _template_period_summary(template_runner, predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for period, (start, end) in TEMPLATE_PERIODS.items():
        selected = predictions[predictions["season"].between(start, end)]
        summary = template_runner.grouped_summary(selected, ["method"])
        summary.insert(0, "period", period)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _template_feature_coverage(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in (("all_templates", templates), ("held_out_targets", targets)):
        for (season, position), group in frame.groupby(
            ["season", "pos"], sort=True
        ):
            available = pd.to_numeric(
                group["expert_ppg_bull_gap_available"], errors="coerce"
            ).fillna(0)
            rows.append(
                {
                    "population": population,
                    "season": int(season),
                    "position": str(position),
                    "rows": len(group),
                    "available_rows": int(available.eq(1).sum()),
                    "coverage": float(available.eq(1).mean()),
                    "median_provider_count": float(
                        pd.to_numeric(
                            group["expert_ppg_gap_provider_count"],
                            errors="coerce",
                        ).fillna(0).median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_template_replay(
    league: str,
    database: Path,
    gap_features: pd.DataFrame,
    results_dir: Path,
    bootstrap_iterations: int,
) -> dict[str, object]:
    template_runner = _load_module(
        TEMPLATE_RUNNER_PATH,
        f"v2_asymmetric_template_runner_{league}",
    )
    builder = template_runner.builder
    builder.set_active_league(league)
    methods = build_template_methods(builder)
    template_runner.METHODS = methods
    template_runner.BASELINE_METHOD = "incumbent"
    template_runner.BOOTSTRAP_SAMPLES = bootstrap_iterations
    max_season = min(int(builder.get_daily_max_template_season()), 2025)
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=database,
        scoring_matched_context=(
            league in builder.V2_SCORING_CONTEXT_CAPABLE_LEAGUES
        ),
    )
    merge_columns = [
        "player_key",
        "season",
        "expert_ppg_gap_provider_count",
        "expert_ppg_bull_gap_available",
        "expert_ppg_bull_gap_position_percentile",
        "expert_ppg_bear_gap_position_percentile",
        "expert_ppg_top2_gap_position_percentile",
    ]
    template_gaps = gap_features[merge_columns + ["position"]].rename(
        columns={"position": "pos"}
    )
    projections = projections.merge(
        template_gaps,
        on=["player_key", "season", "pos"],
        how="left",
        validate="one_to_one",
    )
    projections["expert_ppg_bull_gap_available"] = projections[
        "expert_ppg_bull_gap_available"
    ].fillna(0).astype(int)
    projections["expert_ppg_gap_provider_count"] = projections[
        "expert_ppg_gap_provider_count"
    ].fillna(0).astype(int)
    for column in merge_columns[2:]:
        if column not in builder.MATCH_OUTPUT_COLS:
            builder.MATCH_OUTPUT_COLS.append(column)
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    forecasts = template_runner.base.load_production_oos_forecasts(max_season)
    target_templates = template_runner.base.build_production_oos_target_templates(
        templates, forecasts
    )
    targets = template_runner.base.build_targets(target_templates)
    predictions = template_runner.run_replay(templates, targets)
    expected_rows = len(targets) * len(methods)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} template predictions; found {len(predictions)}"
        )
    summary = _template_period_summary(template_runner, predictions)
    bootstrap_frames = []
    for candidate in methods:
        if candidate == "incumbent":
            continue
        bootstrap_frames.append(
            template_runner.bootstrap_comparison(
                predictions,
                candidate,
                TEMPLATE_PERIODS,
                baseline_method="incumbent",
            )
        )
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    coverage = _template_feature_coverage(templates, targets)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / "template_target_predictions.csv", index=False)
    summary.to_csv(results_dir / "template_period_summary.csv", index=False)
    bootstrap.to_csv(results_dir / "template_bootstrap.csv", index=False)
    coverage.to_csv(results_dir / "template_gap_coverage.csv", index=False)
    method_receipt = {
        method: {
            "recency_half_life": specification["recency_half_life"],
            "position_weights": specification["weights"],
        }
        for method, specification in methods.items()
    }
    (results_dir / "template_method_spec.json").write_text(
        json.dumps(method_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "max_template_season": max_season,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "summary": summary.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records"),
    }


def _point_gates(summary: pd.DataFrame) -> dict[str, bool]:
    controlled = summary[
        summary["method"].eq(CONTROLLED_METHOD)
        & summary["challenger_variant"].eq(PRIMARY_POINT_VARIANT)
    ].iloc[0]
    production = summary[
        summary["method"].eq(PRODUCTION_METHOD)
        & summary["challenger_variant"].eq(PRIMARY_POINT_VARIANT)
    ].iloc[0]
    return {
        "controlled_pooled_improvement_at_least_0_001": bool(
            controlled.pooled_delta <= -0.001
        ),
        "controlled_recent_nonworse": bool(controlled.recent_delta <= 0),
        "controlled_at_least_6_season_wins": bool(controlled.season_wins >= 6),
        "controlled_season_interval_upper_nonpositive": bool(
            controlled.season_95_high <= 0
        ),
        "controlled_player_interval_upper_nonpositive": bool(
            controlled.player_95_high <= 0
        ),
        "production_pooled_nonworse": bool(production.pooled_delta <= 0),
        "production_recent_nonworse": bool(production.recent_delta <= 0),
        "controlled_at_least_3_positions_nonworse": bool(
            controlled.nonworse_positions >= 3
        ),
    }


def _tail_gates(summary: pd.DataFrame) -> dict[str, bool]:
    primary = summary[summary["challenger_variant"].eq("tail_bullish")].set_index(
        "event"
    )
    return {
        "plus3_brier_improves": bool(primary.loc["plus3", "brier_delta"] < 0),
        "plus5_brier_improves": bool(primary.loc["plus5", "brier_delta"] < 0),
        "plus3_recent_brier_nonworse": bool(
            primary.loc["plus3", "recent_brier_delta"] <= 0
        ),
        "plus5_recent_brier_nonworse": bool(
            primary.loc["plus5", "recent_brier_delta"] <= 0
        ),
        "plus3_auc_nonworse": bool(
            primary.loc["plus3", "challenger_auc"]
            >= primary.loc["plus3", "baseline_auc"]
        ),
        "plus5_auc_nonworse": bool(
            primary.loc["plus5", "challenger_auc"]
            >= primary.loc["plus5", "baseline_auc"]
        ),
        "plus3_season_interval_upper_nonpositive": bool(
            primary.loc["plus3", "season_95_high"] <= 0
        ),
        "plus5_season_interval_upper_nonpositive": bool(
            primary.loc["plus5", "season_95_high"] <= 0
        ),
    }


def _template_gates(summary: pd.DataFrame) -> dict[str, bool]:
    indexed = summary.set_index(["period", "method"])
    full_base = indexed.loc[("full_2017_2025", "incumbent")]
    full = indexed.loc[("full_2017_2025", PRIMARY_TEMPLATE_METHOD)]
    recent_base = indexed.loc[("temporal_2023_2025", "incumbent")]
    recent = indexed.loc[("temporal_2023_2025", PRIMARY_TEMPLATE_METHOD)]
    return {
        "full_ppg_crps_nonworse": bool(full.ppg_crps <= full_base.ppg_crps),
        "full_contribution_crps_nonworse": bool(
            full.contribution_crps <= full_base.contribution_crps
        ),
        "full_played_crps_within_0_25_percent": bool(
            full.played_crps <= full_base.played_crps * 1.0025
        ),
        "full_plus5_brier_improves": bool(full.plus5_brier < full_base.plus5_brier),
        "full_impact_brier_improves": bool(full.impact_brier < full_base.impact_brier),
        "full_plus5_auc_nonworse": bool(full.plus5_auc >= full_base.plus5_auc),
        "full_impact_auc_nonworse": bool(full.impact_auc >= full_base.impact_auc),
        "recent_plus5_brier_nonworse": bool(
            recent.plus5_brier <= recent_base.plus5_brier
        ),
        "recent_impact_brier_nonworse": bool(
            recent.impact_brier <= recent_base.impact_brier
        ),
    }


def _league_findings(
    league: str,
    point: pd.DataFrame,
    tail: pd.DataFrame,
    template_summary: pd.DataFrame | None,
    gates: dict[str, object],
    lineage: dict[str, object],
) -> str:
    lines = [
        f"# Asymmetric expert projection study - {league.upper()}",
        "",
        "## Conditional-PPG mean",
        "",
        "| Surface | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Pos nonworse |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in point.itertuples(index=False):
        lines.append(
            f"| {row.method} | `{row.challenger_variant}` | {row.pooled_delta:+.5f} | "
            f"{row.recent_delta:+.5f} | {row.season_wins}/{row.season_count} | "
            f"[{row.season_95_low:+.5f}, {row.season_95_high:+.5f}] | "
            f"[{row.player_95_low:+.5f}, {row.player_95_high:+.5f}] | "
            f"{row.nonworse_positions}/{row.position_count} |"
        )
    lines.extend(
        [
            "",
            "## Upper residual events",
            "",
            "| Event | Challenger | Delta Brier | Recent | Delta log loss | AUC | Wins | Season 95% |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tail.itertuples(index=False):
        lines.append(
            f"| {row.event} | `{row.challenger_variant}` | {row.brier_delta:+.6f} | "
            f"{row.recent_brier_delta:+.6f} | {row.log_loss_delta:+.6f} | "
            f"{row.baseline_auc:.4f} -> {row.challenger_auc:.4f} | "
            f"{row.season_wins}/{row.season_count} | "
            f"[{row.season_95_low:+.6f}, {row.season_95_high:+.6f}] |"
        )
    if template_summary is not None:
        lines.extend(
            [
                "",
                "## Weekly-template replay",
                "",
                "| Period | Method | PPG CRPS | Contribution CRPS | +5 Brier | +5 AUC | Impact Brier | Impact AUC |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in template_summary.itertuples(index=False):
            lines.append(
                f"| {row.period} | `{row.method}` | {row.ppg_crps:.5f} | "
                f"{row.contribution_crps:.5f} | {row.plus5_brier:.6f} | "
                f"{row.plus5_auc:.4f} | {row.impact_brier:.6f} | "
                f"{row.impact_auc:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Gates and scope",
            "",
            f"- Point primary passes all gates: `{gates['point_all_pass']}`.",
            f"- Tail primary passes all gates: `{gates['tail_all_pass']}`.",
            f"- Template primary passes all gates: `{gates.get('template_all_pass')}`.",
            "- A point pass advances only to nested retuning; this receipt cannot promote production.",
            "- No production table, model lock, or template contract was changed.",
            "",
            "## Lineage",
            "",
            f"- Lock: `{lineage['lock_version']}`",
            f"- Model run: `{lineage['model_run_id']}`",
            f"- Feature run: `{lineage['feature_run_id']}`",
            f"- Read-only database SHA-256: `{lineage['database_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _run_league(
    league: str,
    database: Path,
    results_dir: Path,
    bootstrap_iterations: int,
    template_bootstrap_iterations: int,
    *,
    skip_templates: bool,
) -> dict[str, object]:
    if not database.is_file():
        raise FileNotFoundError(database)
    started = time.perf_counter()
    before_hash = _file_sha256(database)
    (
        locked,
        features,
        selected,
        locked_predictions,
        gap_features,
        lineage,
    ) = _load_point_inputs(database, league)
    observed_hashes = set(features["scoring_hash"].dropna().astype(str))
    expected_hash = scoring_hash(league)
    if observed_hashes != {expected_hash}:
        raise ValueError(
            f"Scoring mismatch for {league}: observed={observed_hashes}, "
            f"expected={expected_hash}"
        )

    print(f"[{league}] replaying locked point model")
    point_predictions = _run_point_predictions(locked, features, selected)
    reproduction_delta = _assert_incumbent_reproduction(
        point_predictions, locked_predictions
    )
    point_evaluation = _point_evaluation(point_predictions, locked.OUTER_SEASONS)
    point_summary = _point_comparison_summary(
        point_evaluation, bootstrap_iterations
    )
    gap_diagnostics = _gap_outcome_diagnostics(point_evaluation, gap_features)

    print(f"[{league}] fitting strictly-prior upper-residual classifiers")
    tail_predictions = _run_tail_models(point_predictions, locked.OUTER_SEASONS)
    tail_summary = _tail_comparison_summary(
        tail_predictions, bootstrap_iterations
    )

    template_payload = None
    template_summary = None
    if not skip_templates:
        print(f"[{league}] replaying weekly templates")
        template_dir = results_dir / "template"
        template_payload = _run_template_replay(
            league,
            database,
            gap_features,
            template_dir,
            template_bootstrap_iterations,
        )
        template_summary = pd.DataFrame(template_payload["summary"])

    point_gate_values = _point_gates(point_summary)
    tail_gate_values = _tail_gates(tail_summary)
    gates: dict[str, object] = {
        "point": point_gate_values,
        "tail": tail_gate_values,
        "point_all_pass": all(point_gate_values.values()),
        "tail_all_pass": all(tail_gate_values.values()),
    }
    if template_summary is not None:
        template_gate_values = _template_gates(template_summary)
        gates["template"] = template_gate_values
        gates["template_all_pass"] = all(template_gate_values.values())
    else:
        gates["template"] = None
        gates["template_all_pass"] = None

    after_hash = _file_sha256(database)
    if after_hash != before_hash:
        raise RuntimeError(f"{league} V2 database changed during read-only study")
    lineage = {
        **lineage,
        "database": str(database.resolve()),
        "database_sha256": before_hash,
        "incumbent_reproduction_max_abs_delta": reproduction_delta,
    }
    coverage = (
        gap_features.groupby(["season", "position"], sort=True)
        .agg(
            rows=("player_key", "size"),
            available_rows=("expert_ppg_bull_gap_available", "sum"),
            mean_provider_count=("expert_ppg_gap_provider_count", "mean"),
            median_provider_count=("expert_ppg_gap_provider_count", "median"),
        )
        .reset_index()
    )
    coverage["coverage"] = coverage["available_rows"] / coverage["rows"]
    results_dir.mkdir(parents=True, exist_ok=True)
    gap_features.to_csv(results_dir / "asymmetric_projection_features.csv", index=False)
    coverage.to_csv(results_dir / "asymmetric_feature_coverage.csv", index=False)
    point_evaluation.to_csv(results_dir / "point_oof_predictions.csv", index=False)
    point_summary.to_csv(results_dir / "point_comparison_summary.csv", index=False)
    gap_diagnostics.to_csv(results_dir / "gap_outcome_diagnostics.csv", index=False)
    tail_predictions.to_csv(results_dir / "tail_event_predictions.csv", index=False)
    tail_summary.to_csv(results_dir / "tail_comparison_summary.csv", index=False)
    (results_dir / "gate_audit.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **lineage,
        "league": league,
        "scoring_hash": expected_hash,
        "feature_spec": {
            "source_table": "player_season_projection_values",
            "provider_value": "provider_points_per_team_game",
            "configured_points_complete_required": True,
            "minimum_providers": MIN_ASYMMETRIC_PROVIDERS,
            "fraction_denominator": (
                "max(abs(provider_ppg_median), 1.0)"
            ),
            "primary_point_variant": PRIMARY_POINT_VARIANT,
            "primary_tail_variant": "tail_bullish",
            "primary_template_method": PRIMARY_TEMPLATE_METHOD,
            "incumbent_provider_count_control": "projection_provider_count",
        },
        "bootstrap_iterations": bootstrap_iterations,
        "template_bootstrap_iterations": template_bootstrap_iterations,
        "templates_skipped": skip_templates,
        "runtime_seconds": time.perf_counter() - started,
    }
    (results_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "findings.md").write_text(
        _league_findings(
            league,
            point_summary,
            tail_summary,
            template_summary,
            gates,
            lineage,
        ),
        encoding="utf-8",
    )
    return {
        "league": league,
        "lineage": lineage,
        "point": point_summary.to_dict("records"),
        "tail": tail_summary.to_dict("records"),
        "template": template_payload,
        "gates": gates,
    }


def _load_existing_payload(league: str, results_root: Path) -> dict[str, object]:
    league_dir = results_root / league
    manifest = json.loads(
        (league_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    gates = json.loads(
        (league_dir / "gate_audit.json").read_text(encoding="utf-8")
    )
    template_path = league_dir / "template" / "template_period_summary.csv"
    template_payload = None
    if template_path.is_file():
        template_payload = {
            "summary": pd.read_csv(template_path).to_dict("records")
        }
    return {
        "league": league,
        "lineage": manifest,
        "point": pd.read_csv(
            league_dir / "point_comparison_summary.csv"
        ).to_dict("records"),
        "tail": pd.read_csv(
            league_dir / "tail_comparison_summary.csv"
        ).to_dict("records"),
        "template": template_payload,
        "gates": gates,
    }


def _combined_findings(decision: dict[str, object]) -> str:
    lines = [
        "# Asymmetric expert projection cross-league decision",
        "",
        f"- Point primary passes both leagues: `{decision['point_both_leagues_pass']}`.",
        f"- Upper-tail primary passes both leagues: `{decision['tail_both_leagues_pass']}`.",
        f"- Weekly-template primary passes both leagues: `{decision['template_both_leagues_pass']}`.",
        f"- Overall next action: `{decision['next_action']}`.",
        "",
        "## Primary candidate headline deltas",
        "",
        "| League | Point controlled RMSE | Point production RMSE | +3 Brier | +5 Brier | Template +5 Brier | Template impact Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for league, payload in decision["league_results"].items():
        point = pd.DataFrame(payload["point"])
        point = point[point["challenger_variant"].eq(PRIMARY_POINT_VARIANT)].set_index(
            "method"
        )
        tail = pd.DataFrame(payload["tail"])
        tail = tail[tail["challenger_variant"].eq("tail_bullish")].set_index(
            "event"
        )
        template_plus5 = np.nan
        template_impact = np.nan
        if payload.get("template"):
            template = pd.DataFrame(payload["template"]["summary"])
            template = template[
                template["period"].eq("full_2017_2025")
            ].set_index("method")
            template_plus5 = float(
                template.loc[PRIMARY_TEMPLATE_METHOD, "plus5_brier"]
                - template.loc["incumbent", "plus5_brier"]
            )
            template_impact = float(
                template.loc[PRIMARY_TEMPLATE_METHOD, "impact_brier"]
                - template.loc["incumbent", "impact_brier"]
            )
        lines.append(
            f"| {league.upper()} | {point.loc[CONTROLLED_METHOD, 'pooled_delta']:+.5f} | "
            f"{point.loc[PRODUCTION_METHOD, 'pooled_delta']:+.5f} | "
            f"{tail.loc['plus3', 'brier_delta']:+.6f} | "
            f"{tail.loc['plus5', 'brier_delta']:+.6f} | "
            f"{template_plus5:+.6f} | {template_impact:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The normalized bullish-gap primary is rejected: recent point RMSE, +3/+5 residual Brier, and the multi-outcome template gates do not replicate across DK and beta.",
            "- The raw max-minus-median sensitivity improves pooled point RMSE in both leagues, but recent controlled RMSE worsens, clustered intervals cross zero, and the gain is concentrated in the post-slice QB diagnostic.",
            "- High bullish-gap quartiles are not empirical ceiling groups; +3/+5 residual rates are lower than in the bottom quartile in both leagues. Small consensus denominators make many normalized outliers fringe players.",
            "- If revisited, prespecify a projection-floor or projection-tier-controlled raw QB interaction and require nested retuning. Do not promote a generic asymmetric gap or template weight.",
            "",
            "The study is intentionally read-only. Passing the point gates would only justify a leakage-safe nested retune; it would not directly change the production model or templates.",
            "",
        ]
    )
    return "\n".join(lines)


def _combine(
    payloads: Sequence[dict[str, object]],
    results_dir: Path,
) -> dict[str, object]:
    by_league = {str(payload["league"]): payload for payload in payloads}
    if set(by_league) != {"dk", "beta"}:
        raise ValueError("Combined decision requires DK and beta")
    point_pass = all(
        bool(payload["gates"]["point_all_pass"]) for payload in payloads
    )
    tail_pass = all(
        bool(payload["gates"]["tail_all_pass"]) for payload in payloads
    )
    template_values = [
        payload["gates"].get("template_all_pass") for payload in payloads
    ]
    template_pass = (
        all(bool(value) for value in template_values)
        if all(value is not None for value in template_values)
        else None
    )
    all_pass = point_pass and tail_pass and template_pass is True
    decision = {
        "point_both_leagues_pass": point_pass,
        "tail_both_leagues_pass": tail_pass,
        "template_both_leagues_pass": template_pass,
        "all_surfaces_pass": all_pass,
        "next_action": (
            "advance_to_nested_retune_and_independent_template_confirmation"
            if all_pass
            else "retain_outside_production"
        ),
        "league_results": by_league,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "findings.md").write_text(
        _combined_findings(decision), encoding="utf-8"
    )
    return decision


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0 or args.template_bootstrap_iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    if args.league == "all" and args.database is not None:
        raise ValueError("--database cannot be used with --league all")
    results_root = args.results_dir or STUDY_ROOT / "results"
    if args.league == "all":
        if args.combine_existing:
            payloads = [
                _load_existing_payload(league, results_root)
                for league in ("dk", "beta")
            ]
        else:
            payloads = [
                _run_league(
                    league,
                    DATABASES[league],
                    results_root / league,
                    args.bootstrap_iterations,
                    args.template_bootstrap_iterations,
                    skip_templates=args.skip_templates,
                )
                for league in ("dk", "beta")
            ]
        decision = _combine(payloads, results_root)
        print(json.dumps(decision, indent=2))
        return
    database = args.database or DATABASES[args.league]
    payload = _run_league(
        args.league,
        database,
        results_root / args.league,
        args.bootstrap_iterations,
        args.template_bootstrap_iterations,
        skip_templates=args.skip_templates,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
