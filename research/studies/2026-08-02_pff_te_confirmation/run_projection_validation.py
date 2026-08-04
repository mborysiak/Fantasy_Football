"""Selected-grid replay of PFF TE features in the full locked PPG blend."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from Scripts.V2.locked_candidates import (
    LOCKED_VALIDATION_SEASONS,
    PRIMARY_PPG_FEATURES,
    lock_version_for_scoring,
)
from Scripts.V2.native_runtime import (
    MAX_ISOLATED_LIGHTGBM_FITS,
    run_module_function_in_fresh_process,
)
from pff_te_features import (
    PROJECTION_CONTROL_FEATURES,
    PROJECTION_MTF_FEATURES,
    PROJECTION_YAC_FEATURES,
    attach_projection_features,
    build_te_profiles,
)


REFERENCE_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-29_v2_locked_final_validation"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location("pff_te_locked_reference", REFERENCE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import {REFERENCE_PATH}")
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

RAW_DB = REPO_ROOT / "Data" / "Databases" / "Season_Stats_New.sqlite3"
DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
OUTER_SEASONS = tuple(LOCKED_VALIDATION_SEASONS)
BASELINE = "production"
COMPONENTS = tuple(reference.PRIMARY_COMPONENTS)
VARIANTS = {
    "te_pff_opportunity_control": tuple(PROJECTION_CONTROL_FEATURES),
    "te_pff_mtf": tuple(PROJECTION_MTF_FEATURES),
    "te_pff_yac": tuple(PROJECTION_YAC_FEATURES),
}
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def _selected_predictions(
    ppg: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_columns: tuple[str, ...],
    model_name: str,
    output_name: str,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    origins = OUTER_SEASONS
    if not model_name.endswith("lightgbm"):
        return reference._selected_predictions_for_origins(
            ppg,
            candidates,
            feature_columns,
            model_name,
            output_name,
            selected,
            origins,
        )
    frames = []
    for start in range(0, len(origins), MAX_ISOLATED_LIGHTGBM_FITS):
        frames.append(
            run_module_function_in_fresh_process(
                REFERENCE_PATH,
                "_selected_predictions_for_origins",
                args=(
                    ppg,
                    candidates,
                    feature_columns,
                    model_name,
                    output_name,
                    selected,
                    origins[start : start + MAX_ISOLATED_LIGHTGBM_FITS],
                ),
            )
        )
    return pd.concat(frames, ignore_index=True)


def _load_baseline(database: Path) -> pd.DataFrame:
    methods = [*COMPONENTS, "conditional_ppg_primary_blend"]
    placeholders = ",".join("?" for _ in methods)
    import sqlite3

    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            f"""
            SELECT player_key, season, position, actual, method, prediction
            FROM locked_whole_season_predictions
            WHERE target_name = 'conditional_ppg'
              AND method IN ({placeholders})
            """,
            connection,
            params=methods,
        )
    return frame


def _load_selections(database: Path) -> pd.DataFrame:
    import sqlite3

    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        selected = pd.read_sql_query(
            "SELECT * FROM locked_selected_hyperparameters",
            connection,
        )
    selected = selected[selected["model_name"].isin(COMPONENTS)].copy()
    expected = len(COMPONENTS) * (len(OUTER_SEASONS) + 1)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} primary selections; found {len(selected)}")
    return selected


def _blend_predictions(predictions: pd.DataFrame, variant: str) -> pd.DataFrame:
    wide = predictions.pivot(
        index=["player_key", "season", "position"],
        columns="component",
        values="prediction",
    )
    if wide[list(COMPONENTS)].isna().any().any():
        raise ValueError(f"Incomplete component predictions for {variant}")
    output = wide.reset_index()
    output["method"] = variant
    output["prediction"] = output[list(COMPONENTS)].mean(axis=1)
    return output[["player_key", "season", "position", "method", "prediction"]]


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - prediction))))


def _bootstrap(values: pd.Series, seed: int) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_ppg(evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = {
        "all": pd.Series(True, index=evaluation.index),
        "te": evaluation["position"].eq("TE"),
        "non_te": ~evaluation["position"].eq("TE"),
    }
    metric_rows = []
    season_rows = []
    for scope, mask in scopes.items():
        scoped = evaluation[mask]
        for period, (start, end) in PERIODS.items():
            period_frame = scoped[scoped["season"].between(start, end)]
            baseline = period_frame[period_frame["method"].eq(BASELINE)]
            baseline_rmse = _rmse(baseline["actual"], baseline["prediction"])
            baseline_mae = float(np.mean(np.abs(baseline["actual"] - baseline["prediction"])))
            for method, group in period_frame.groupby("method", sort=True):
                rmse = _rmse(group["actual"], group["prediction"])
                mae = float(np.mean(np.abs(group["actual"] - group["prediction"])))
                metric_rows.append(
                    {
                        "league": group["league"].iloc[0],
                        "scope": scope,
                        "period": period,
                        "method": method,
                        "rows": len(group),
                        "rmse": rmse,
                        "rmse_delta": rmse - baseline_rmse,
                        "mae": mae,
                        "mae_delta": mae - baseline_mae,
                    }
                )
        for method in sorted(set(scoped["method"]) - {BASELINE}):
            for season, group in scoped.groupby("season", sort=True):
                baseline = group[group["method"].eq(BASELINE)]
                challenger = group[group["method"].eq(method)]
                season_rows.append(
                    {
                        "league": group["league"].iloc[0],
                        "scope": scope,
                        "method": method,
                        "season": int(season),
                        "rmse_delta": _rmse(challenger["actual"], challenger["prediction"])
                        - _rmse(baseline["actual"], baseline["prediction"]),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    seasons = pd.DataFrame(season_rows)
    intervals = []
    for index, (key, group) in enumerate(
        seasons.groupby(["league", "scope", "method"], sort=True)
    ):
        low, high = _bootstrap(group["rmse_delta"], RANDOM_SEED + index)
        intervals.append(
            {
                "league": key[0],
                "scope": key[1],
                "method": key[2],
                "mean_season_rmse_delta": float(group["rmse_delta"].mean()),
                "season_wins": int(group["rmse_delta"].lt(0).sum()),
                "seasons": len(group),
                "bootstrap_low": low,
                "bootstrap_high": high,
            }
        )
    return metrics.merge(pd.DataFrame(intervals), on=["league", "scope", "method"], how="left"), seasons


def _q90_probabilities(evaluation: pd.DataFrame) -> pd.DataFrame:
    output = evaluation.copy()
    output["q90_threshold"] = output.groupby(["method", "season", "position"])["actual"].transform(
        lambda values: values.quantile(0.90)
    )
    output["q90_event"] = output["actual"].ge(output["q90_threshold"]).astype(int)
    frames = []
    for (method, position), group in output.groupby(["method", "position"], sort=True):
        for season in range(2018, max(OUTER_SEASONS) + 1):
            train = group[group["season"].lt(season)]
            test = group[group["season"].eq(season)].copy()
            if len(train) < 20 or test.empty or train["q90_event"].nunique() < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2_000, random_state=RANDOM_SEED),
            )
            model.fit(train[["prediction"]], train["q90_event"])
            test["q90_probability"] = model.predict_proba(test[["prediction"]])[:, 1]
            frames.append(test)
    return pd.concat(frames, ignore_index=True)


def summarize_q90(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "all": pd.Series(True, index=predictions.index),
        "te": predictions["position"].eq("TE"),
    }
    for scope, mask in scopes.items():
        scoped = predictions[mask]
        for period, (start, end) in PERIODS.items():
            period_frame = scoped[scoped["season"].between(max(start, 2018), end)]
            baseline = period_frame[period_frame["method"].eq(BASELINE)]
            baseline_brier = float(np.mean(np.square(baseline["q90_event"] - baseline["q90_probability"])))
            baseline_ap = float(average_precision_score(baseline["q90_event"], baseline["q90_probability"]))
            for method, group in period_frame.groupby("method", sort=True):
                brier = float(np.mean(np.square(group["q90_event"] - group["q90_probability"])))
                ap = float(average_precision_score(group["q90_event"], group["q90_probability"]))
                rows.append(
                    {
                        "league": group["league"].iloc[0],
                        "scope": scope,
                        "period": period,
                        "method": method,
                        "rows": len(group),
                        "events": int(group["q90_event"].sum()),
                        "brier": brier,
                        "brier_delta": brier - baseline_brier,
                        "average_precision": ap,
                        "average_precision_delta": ap - baseline_ap,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    league = args.league
    database = DATABASES[league]
    results_dir = args.results_dir.resolve() if args.results_dir else STUDY_DIR / f"results_projection_{league}"
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    reference.ACTIVE_OUTPUT_DB_PATH = database
    reference.ACTIVE_SCORING_OBJECTIVE = league
    reference.ACTIVE_LOCK_VERSION = lock_version_for_scoring(league)
    features, _, feature_run_id = reference._load_inputs()
    profiles = build_te_profiles(database, RAW_DB, int(features["season"].max()) - 1)
    features = attach_projection_features(features, profiles)
    ppg, _, candidates = reference._target_frames(features)
    selected = _load_selections(database)

    challenger_components = []
    for variant, additions in VARIANTS.items():
        feature_columns = tuple(dict.fromkeys((*PRIMARY_PPG_FEATURES, *additions)))
        for component in COMPONENTS:
            print(f"{league}: {variant} / {component}", flush=True)
            predictions = _selected_predictions(
                ppg,
                candidates,
                feature_columns,
                component,
                f"{variant}__{component}",
                selected,
            )
            predictions["variant"] = variant
            predictions["component"] = component
            challenger_components.append(predictions)
    challenger_components = pd.concat(challenger_components, ignore_index=True)
    challenger_blends = pd.concat(
        [
            _blend_predictions(
                challenger_components[challenger_components["variant"].eq(variant)],
                variant,
            )
            for variant in VARIANTS
        ],
        ignore_index=True,
    )

    baseline = _load_baseline(database)
    baseline_blend = baseline[baseline["method"].eq("conditional_ppg_primary_blend")].copy()
    baseline_blend["method"] = BASELINE
    baseline_blend = baseline_blend[["player_key", "season", "position", "actual", "method", "prediction"]]
    actual = baseline_blend[["player_key", "season", "position", "actual"]]
    challenger_blends = challenger_blends.merge(
        actual,
        on=["player_key", "season", "position"],
        how="inner",
        validate="many_to_one",
    )
    routed = challenger_blends.merge(
        baseline_blend[
            ["player_key", "season", "position", "prediction"]
        ].rename(columns={"prediction": "production_prediction"}),
        on=["player_key", "season", "position"],
        how="left",
        validate="many_to_one",
    )
    routed["prediction"] = np.where(
        routed["position"].eq("TE"),
        routed["prediction"],
        routed["production_prediction"],
    )
    routed["method"] = routed["method"] + "__te_route"
    routed = routed[
        ["player_key", "season", "position", "actual", "method", "prediction"]
    ]
    evaluation = pd.concat(
        [baseline_blend, challenger_blends, routed],
        ignore_index=True,
    )
    evaluation["league"] = league
    if evaluation.duplicated(["player_key", "season", "method"]).any():
        raise ValueError("Evaluation predictions are not unique")

    ppg_summary, season_diagnostics = summarize_ppg(evaluation)
    q90_predictions = _q90_probabilities(evaluation)
    q90_summary = summarize_q90(q90_predictions)

    ppg_summary.to_csv(results_dir / "ppg_summary.csv", index=False)
    season_diagnostics.to_csv(results_dir / "ppg_season_diagnostics.csv", index=False)
    q90_summary.to_csv(results_dir / "q90_summary.csv", index=False)
    profiles.to_csv(results_dir / "pff_te_profiles.csv", index=False)
    challenger_components.to_csv(results_dir / "challenger_component_predictions.csv", index=False)
    evaluation.to_csv(results_dir / "blend_predictions.csv", index=False)
    metadata = {
        "league": league,
        "database": str(database),
        "feature_run_id": feature_run_id,
        "validation_seasons": [min(OUTER_SEASONS), max(OUTER_SEASONS)],
        "variants": list(VARIANTS),
        "components": list(COMPONENTS),
        "hyperparameter_policy": "reuse exact production per-origin selections",
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ppg_summary[(ppg_summary["scope"].eq("te")) & (ppg_summary["period"].ne("all_2017_2025"))].to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
