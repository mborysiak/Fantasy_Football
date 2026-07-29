"""Compare pooled and independently fitted position PPG models."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import OUTPUT_DB_PATH
from Scripts.V2.contracts import create_run_id
from Scripts.V2.modeling import (
    CONDITIONAL_PPG_TARGET,
    ModelSpec,
    build_target_frames,
    make_fold_assignments,
    run_model_spec,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
POOLED_RESULTS = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_projection_consensus_ladder"
    / "results"
    / "oof_predictions.csv"
)
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234
POSITIONS = ("QB", "RB", "WR", "TE")
COMPONENT_SCHEMES = {
    "separate": {
        "QB": ("QB",),
        "RB": ("RB",),
        "WR": ("WR",),
        "TE": ("TE",),
    },
    "role_group": {
        "QB": ("QB",),
        "RB": ("RB",),
        "REC": ("WR", "TE"),
    },
    "qb_skill": {
        "QB": ("QB",),
        "SKILL": ("RB", "WR", "TE"),
    },
}
POOLED_MODELS = {
    "pooled_projection_core": "projection_only_lightgbm_core",
    "pooled_full": "full_lightgbm_base",
}

PROJECTION_CORE_FEATURES = (
    "expert_ppg_team_game_median",
    "expert_ppg_active_median",
    "expert_ppg_team_game_std",
    "expert_ppg_team_game_iqr",
    "expert_points_iqr",
    "projection_provider_count",
    "configured_projection_provider_count",
    "proj_games",
    "proj_pass_attempts",
    "proj_passing_yards",
    "proj_passing_tds",
    "proj_interceptions",
    "proj_rush_attempts",
    "proj_rushing_yards",
    "proj_rushing_tds",
    "proj_targets",
    "proj_receptions",
    "proj_receiving_yards",
    "proj_receiving_tds",
    "projected_pass_point_share",
    "projected_rush_point_share",
    "projected_receiving_point_share",
)

LIGHTGBM_PARAMETERS = {
    "lgbm__n_estimators": (100, 200),
    "lgbm__learning_rate": (0.03, 0.05),
    "lgbm__num_leaves": (7, 15),
    "lgbm__max_depth": (3, 4),
    "lgbm__min_child_samples": (20, 40),
    "lgbm__reg_lambda": (1.0, 5.0),
    "lgbm__subsample": (1.0,),
    "lgbm__colsample_bytree": (1.0,),
    "lgbm__deterministic": (True,),
    "lgbm__force_col_wise": (True,),
}


def _load_inputs() -> tuple[pd.DataFrame, tuple[str, ...], str]:
    with sqlite3.connect(OUTPUT_DB_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM player_season_features",
            connection,
        )
        manifests = pd.read_sql_query(
            "SELECT * FROM feature_manifests",
            connection,
        )
    run_ids = features["run_id"].dropna().astype(str).unique()
    if len(run_ids) != 1:
        raise ValueError("Expected one active feature run")
    feature_run_id = str(run_ids[0])
    manifest_run_ids = (
        manifests["run_id"].dropna().astype(str).unique()
    )
    if set(manifest_run_ids) != {feature_run_id}:
        raise ValueError("Feature manifest lineage does not match the mart")
    full_features = tuple(
        manifests.loc[
            manifests["manifest_name"].eq("residual_candidate_v1"),
            "feature_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(full_features) != 31:
        raise ValueError(
            f"Expected 31 full features, found {len(full_features)}"
        )
    return features, full_features, feature_run_id


def _spec(
    scheme: str,
    component: str,
    feature_set: str,
) -> ModelSpec:
    return ModelSpec(
        target_name=CONDITIONAL_PPG_TARGET,
        model_name=f"{scheme}_{feature_set}_{component}",
        model_family="lightgbm",
        prediction_kind="direct",
        feature_set=feature_set,
        pipeline_variant="raw",
        model_piece="lgbm",
        parameters=LIGHTGBM_PARAMETERS,
        search_iterations=4,
    )


def _history_depth(frame: pd.DataFrame) -> pd.Series:
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    prior = pd.to_numeric(
        frame["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    result = pd.Series(
        "other_no_history",
        index=frame.index,
        dtype=object,
    )
    result.loc[rookie] = "rookie"
    result.loc[~rookie & year_exp.eq(1)] = "second_year"
    result.loc[~rookie & year_exp.ge(2) & prior] = (
        "veteran_with_history"
    )
    result.loc[year_exp.isna()] = "unknown_experience"
    return result


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    error = (
        pd.to_numeric(prediction, errors="coerce")
        - pd.to_numeric(actual, errors="coerce")
    )
    return float(np.sqrt(np.square(error).mean()))


def _score_rows(
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    slice_type: str,
    slice_column: str | None,
) -> list[dict[str, object]]:
    groups = (
        [("all", frame)]
        if slice_column is None
        else frame.groupby(slice_column, dropna=False)
    )
    rows = []
    for slice_value, group in groups:
        actual = pd.to_numeric(group["actual"], errors="coerce")
        for method in methods:
            prediction = pd.to_numeric(
                group[method],
                errors="coerce",
            )
            rows.extend(
                (
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "rmse",
                        "n_rows": len(group),
                        "value": _rmse(actual, prediction),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "mae",
                        "n_rows": len(group),
                        "value": float(
                            (prediction - actual).abs().mean()
                        ),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "bias",
                        "n_rows": len(group),
                        "value": float((prediction - actual).mean()),
                    },
                    {
                        "method": method,
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        "metric": "spearman",
                        "n_rows": len(group),
                        "value": float(
                            prediction.corr(actual, method="spearman")
                        ),
                    },
                )
            )
    return rows


def _score_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    methods = (
        "pooled_projection_core",
        "separate_projection_core",
        "role_group_projection_core",
        "qb_skill_projection_core",
        "pooled_full",
        "separate_full",
        "role_group_full",
        "qb_skill_full",
    )
    rows = _score_rows(frame, methods, "pooled", None)
    for slice_type, column in (
        ("season", "season"),
        ("position", "position"),
        ("history_depth", "history_depth"),
        ("position_history", "position_history"),
    ):
        rows.extend(
            _score_rows(
                frame,
                methods,
                slice_type,
                column,
            )
        )
    return pd.DataFrame(rows)


def _paired_comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    season = scores[
        scores["slice_type"].eq("season")
        & scores["metric"].eq("rmse")
    ].pivot(index="slice_value", columns="method", values="value")
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].set_index("method")["value"]
    pairs = (
        ("separate_projection_core", "pooled_projection_core"),
        ("role_group_projection_core", "pooled_projection_core"),
        ("qb_skill_projection_core", "pooled_projection_core"),
        ("separate_full", "pooled_full"),
        ("role_group_full", "pooled_full"),
        ("qb_skill_full", "pooled_full"),
        ("role_group_projection_core", "separate_projection_core"),
        ("role_group_full", "separate_full"),
        ("qb_skill_projection_core", "role_group_projection_core"),
        ("qb_skill_full", "role_group_full"),
    )
    rows = []
    for index, (challenger, reference) in enumerate(pairs):
        delta = (
            season[challenger] - season[reference]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + index)
        draws = np.array(
            [
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "challenger": challenger,
                "reference": reference,
                "challenger_rmse": float(pooled[challenger]),
                "reference_rmse": float(pooled[reference]),
                "pooled_delta": float(
                    pooled[challenger] - pooled[reference]
                ),
                "mean_season_delta": float(delta.mean()),
                "season_wins": int((delta < 0).sum()),
                "season_count": len(delta),
                "bootstrap_95_low": float(
                    np.quantile(draws, 0.025)
                ),
                "bootstrap_95_high": float(
                    np.quantile(draws, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].sort_values("value")
    position = scores[
        scores["slice_type"].eq("position")
        & scores["metric"].eq("rmse")
    ].pivot(
        index="slice_value",
        columns="method",
        values="value",
    )
    lines = [
        "# Position-Specific Model Results",
        "",
        "Negative deltas favor the separately fitted challenger.",
        "",
        "## Pooled OOF",
        "",
        "| Method | RMSE |",
        "|---|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| `{row.method}` | {row.value:.4f} |")
    lines.extend(
        (
            "",
            "## Paired season comparisons",
            "",
            "| Challenger | Reference | Delta | 95% interval | Wins |",
            "|---|---|---:|---:|---:|",
        )
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.challenger}` | `{row.reference}` | "
            f"{row.pooled_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.season_wins}/{row.season_count} |"
        )
    lines.extend(
        (
            "",
            "## Position RMSE",
            "",
            "| Position | Pooled projection | Separate projection | "
            "Role-group projection | QB/skill projection | Pooled full | "
            "Separate full | Role-group full | QB/skill full |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for position_name, row in position.iterrows():
        lines.append(
            f"| {position_name} | "
            f"{row['pooled_projection_core']:.4f} | "
            f"{row['separate_projection_core']:.4f} | "
            f"{row['role_group_projection_core']:.4f} | "
            f"{row['qb_skill_projection_core']:.4f} | "
            f"{row['pooled_full']:.4f} | "
            f"{row['separate_full']:.4f} | "
            f"{row['role_group_full']:.4f} | "
            f"{row['qb_skill_full']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _load_pooled_predictions() -> pd.DataFrame:
    pooled = pd.read_csv(POOLED_RESULTS)
    selected = pooled[
        pooled["model_name"].isin(POOLED_MODELS.values())
    ].copy()
    selected["method"] = selected["model_name"].map(
        {value: key for key, value in POOLED_MODELS.items()}
    )
    keys = ["player_key", "season"]
    wide = selected.pivot(
        index=keys,
        columns="method",
        values="final_prediction",
    ).reset_index()
    if wide.duplicated(keys).any():
        raise ValueError("Pooled predictions are not unique")
    expected = set(POOLED_MODELS)
    if not expected.issubset(wide.columns):
        raise ValueError("Missing pooled comparator predictions")
    return wide


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features, full_features, feature_run_id = _load_inputs()
    target = build_target_frames(
        features,
        VALIDATION_END,
    )[CONDITIONAL_PPG_TARGET]
    run_id = create_run_id("m4a_position_specific_models")
    feature_sets = {
        "projection_core": PROJECTION_CORE_FEATURES,
        "full": full_features,
    }
    oof_parts = []
    parameter_parts = []
    assignment_parts = []
    specification_rows = []
    for scheme, components in COMPONENT_SCHEMES.items():
        for feature_set, base_feature_columns in feature_sets.items():
            for component, component_positions in components.items():
                feature_columns = tuple(
                    dict.fromkeys(
                        (
                            *base_feature_columns,
                            *(
                                tuple(
                                    f"position_{position}"
                                    for position in component_positions
                                )
                                if len(component_positions) > 1
                                else ()
                            ),
                        )
                    )
                )
                missing = sorted(
                    set(feature_columns).difference(target.columns)
                )
                if missing:
                    raise ValueError(
                        f"{scheme}/{feature_set}/{component} is "
                        f"missing feature columns: {missing}"
                    )
                component_target = target[
                    target["position"].isin(component_positions)
                ].copy()
                component_spec = _spec(
                    scheme,
                    component,
                    feature_set,
                )
                assignments = make_fold_assignments(
                    component_target,
                    CONDITIONAL_PPG_TARGET,
                    run_id,
                    VALIDATION_START,
                    N_SPLITS,
                    RANDOM_SEED,
                )
                print(
                    f"Fitting {component_spec.model_name}: "
                    f"{len(component_target):,} total rows, "
                    f"{len(assignments):,} OOF rows",
                    flush=True,
                )
                oof, parameters = run_model_spec(
                    component_target,
                    assignments,
                    component_spec,
                    feature_columns,
                    run_id,
                    feature_run_id,
                    VALIDATION_START,
                    N_SPLITS,
                    RANDOM_SEED,
                )
                oof["component_model_name"] = oof["model_name"]
                oof["model_name"] = f"{scheme}_{feature_set}"
                oof_parts.append(oof)
                parameters["scheme"] = scheme
                parameters["component"] = component
                parameters["component_positions"] = ",".join(
                    component_positions
                )
                parameter_parts.append(parameters)
                assignments["scheme"] = scheme
                assignments["component"] = component
                assignments["component_positions"] = ",".join(
                    component_positions
                )
                assignment_parts.append(assignments)
                specification_rows.append(
                    {
                        "run_id": run_id,
                        "feature_run_id": feature_run_id,
                        "model_name": component_spec.model_name,
                        "scheme": scheme,
                        "component": component,
                        "component_positions": ",".join(
                            component_positions
                        ),
                        "feature_set": feature_set,
                        "feature_count": len(feature_columns),
                        "feature_names_json": json.dumps(
                            feature_columns
                        ),
                        "hyperparameters_json": json.dumps(
                            LIGHTGBM_PARAMETERS,
                            default=list,
                            sort_keys=True,
                        ),
                    }
                )

    component_oof = pd.concat(oof_parts, ignore_index=True)
    if component_oof.duplicated(
        ["model_name", "player_key", "season"]
    ).any():
        raise ValueError("Separate models produced duplicate OOF rows")
    counts = component_oof.groupby("model_name").size()
    expected_rows = int(
        target["season"].between(
            VALIDATION_START,
            VALIDATION_END,
        ).sum()
    )
    if not counts.eq(expected_rows).all():
        raise ValueError(
            "Stitched position models do not cover the OOF population: "
            f"{counts.to_dict()} versus {expected_rows}"
        )

    keys = ["player_key", "season"]
    base = component_oof[
        component_oof["model_name"].eq(
            "separate_projection_core"
        )
    ].copy()
    component_wide = component_oof.pivot(
        index=keys,
        columns="model_name",
        values="final_prediction",
    ).reset_index()
    comparison = (
        base.drop(columns=["final_prediction"]).merge(
            component_wide,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        .merge(
            _load_pooled_predictions(),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    )
    if len(comparison) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} paired rows, found {len(comparison)}"
        )
    comparison["history_depth"] = _history_depth(comparison)
    comparison["history_group"] = np.where(
        comparison["history_depth"].isin(
            ("rookie", "second_year", "other_no_history")
        ),
        "limited",
        "veteran",
    )
    comparison["position_history"] = (
        comparison["position"].astype(str)
        + "_"
        + comparison["history_group"]
    )
    scores = _score_predictions(comparison)
    comparisons = _paired_comparisons(scores)

    component_oof.to_csv(
        RESULTS_DIR / "position_model_oof.csv",
        index=False,
    )
    pd.concat(parameter_parts, ignore_index=True).to_csv(
        RESULTS_DIR / "hyperparameters.csv",
        index=False,
    )
    pd.concat(assignment_parts, ignore_index=True).to_csv(
        RESULTS_DIR / "fold_assignments.csv",
        index=False,
    )
    pd.DataFrame(specification_rows).to_csv(
        RESULTS_DIR / "model_specifications.csv",
        index=False,
    )
    comparison[
        [
            "player_key",
            "season",
            "position",
            "actual",
            "history_depth",
            "history_group",
            "position_history",
            "pooled_projection_core",
            "separate_projection_core",
            "role_group_projection_core",
            "qb_skill_projection_core",
            "pooled_full",
            "separate_full",
            "role_group_full",
            "qb_skill_full",
        ]
    ].to_csv(
        RESULTS_DIR / "comparison_predictions.csv",
        index=False,
    )
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    comparisons.to_csv(
        RESULTS_DIR / "model_comparisons.csv",
        index=False,
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(scores, comparisons),
        encoding="utf-8",
    )
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
