"""Test leakage-safe feature families inside position-specific PPG models."""

from __future__ import annotations

import itertools
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
POSITION_RESULTS = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-28_v2_position_specific_models"
    / "results"
    / "comparison_predictions.csv"
)
VALIDATION_START = 2017
VALIDATION_END = 2025
N_SPLITS = 5
RANDOM_SEED = 1234
POSITIONS = ("QB", "RB", "WR", "TE")
PRIMARY_FAMILIES = (
    "experience_context",
    "market_room",
    "opportunity_role",
    "room_clarity",
)

EXPERIENCE_CONTEXT = (
    "expert_ppg_exp_peer_mean",
    "expert_ppg_exp_diff",
    "expert_ppg_exp_percentile",
)
MARKET_ROOM = (
    "adp_best_teammate_gap",
    "adp_worst_teammate_gap",
    "adp_mean_teammate_gap",
    "adp_teammates_better_count",
    "adp_room_strength_share",
)
OPPORTUNITY_ROLE = {
    "QB": ("team_rush_attempt_share",),
    "RB": (
        "team_rush_attempt_share",
        "team_reception_share",
        "team_receiving_yard_share",
    ),
    "WR": (
        "team_reception_share",
        "team_receiving_yard_share",
    ),
    "TE": (
        "team_reception_share",
        "team_receiving_yard_share",
    ),
}
ROOM_CLARITY_COMMON = (
    "consensus_room_gap_to_next",
    "consensus_room_player_count",
    "room_points_median",
    "room_share_std",
)
ROOM_CLARITY = {
    "QB": (
        *ROOM_CLARITY_COMMON,
        "team_qb_projection_gap",
    ),
    "RB": ROOM_CLARITY_COMMON,
    "WR": (
        *ROOM_CLARITY_COMMON,
        "pass_catcher_room_points",
        "pass_catcher_room_share",
        "team_qb1_passing_yards",
    ),
    "TE": (
        *ROOM_CLARITY_COMMON,
        "pass_catcher_room_points",
        "pass_catcher_room_share",
        "team_qb1_passing_yards",
    ),
}

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
    if set(
        manifests["run_id"].dropna().astype(str).unique()
    ) != {feature_run_id}:
        raise ValueError("Feature manifest lineage does not match the mart")
    base_features = tuple(
        manifests.loc[
            manifests["manifest_name"].eq("residual_candidate_v1"),
            "feature_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(base_features) != 31:
        raise ValueError(
            f"Expected 31 base features, found {len(base_features)}"
        )
    return features, base_features, feature_run_id


def _position_families(
    position: str,
) -> dict[str, tuple[str, ...]]:
    families = {
        "experience_context": EXPERIENCE_CONTEXT,
        "market_room": MARKET_ROOM,
        "opportunity_role": OPPORTUNITY_ROLE[position],
        "room_clarity": ROOM_CLARITY[position],
    }
    families["all_targeted"] = tuple(
        dict.fromkeys(
            feature
            for family in PRIMARY_FAMILIES
            for feature in families[family]
        )
    )
    return families


def _spec(position: str, variant: str) -> ModelSpec:
    return ModelSpec(
        target_name=CONDITIONAL_PPG_TARGET,
        model_name=f"{position}_{variant}",
        model_family="lightgbm",
        prediction_kind="direct",
        feature_set=variant,
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
                )
            )
    return rows


def _score_predictions(
    frame: pd.DataFrame,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    rows = _score_rows(frame, methods, "pooled", None)
    for slice_type, column in (
        ("season", "season"),
        ("history_depth", "history_depth"),
        ("history_group", "history_group"),
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


def _sign_flip_pvalue(delta: np.ndarray) -> float:
    observed = abs(float(np.mean(delta)))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta)):
        values.append(
            abs(float(np.mean(delta * np.asarray(signs))))
        )
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    count = len(valid)
    if not count:
        return output
    adjusted = valid.to_numpy(dtype=float) * count / np.arange(
        1,
        count + 1,
    )
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def _comparisons(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        position_scores = scores[scores["position"].eq(position)]
        pooled = position_scores[
            position_scores["slice_type"].eq("pooled")
            & position_scores["metric"].eq("rmse")
        ].set_index("method")["value"]
        season = position_scores[
            position_scores["slice_type"].eq("season")
            & position_scores["metric"].eq("rmse")
        ].pivot(
            index="slice_value",
            columns="method",
            values="value",
        )
        history = position_scores[
            position_scores["slice_type"].eq("history_group")
            & position_scores["metric"].eq("rmse")
        ].pivot(
            index="slice_value",
            columns="method",
            values="value",
        )
        for family_index, variant in enumerate(
            (*PRIMARY_FAMILIES, "all_targeted")
        ):
            delta = (
                season[variant] - season["separate_full_base"]
            ).to_numpy(dtype=float)
            rng = np.random.default_rng(
                RANDOM_SEED
                + POSITIONS.index(position) * 20
                + family_index
            )
            draws = np.array(
                [
                    rng.choice(delta, len(delta), replace=True).mean()
                    for _ in range(20_000)
                ]
            )
            recent = season.index.astype(int) >= 2023
            rows.append(
                {
                    "position": position,
                    "variant": variant,
                    "primary_test": int(
                        variant in PRIMARY_FAMILIES
                    ),
                    "base_rmse": float(
                        pooled["separate_full_base"]
                    ),
                    "variant_rmse": float(pooled[variant]),
                    "pooled_delta": float(
                        pooled[variant]
                        - pooled["separate_full_base"]
                    ),
                    "pooled_full_reference_rmse": float(
                        pooled["pooled_full_reference"]
                    ),
                    "delta_vs_pooled_full": float(
                        pooled[variant]
                        - pooled["pooled_full_reference"]
                    ),
                    "mean_season_delta": float(delta.mean()),
                    "recent_mean_season_delta": float(
                        delta[recent].mean()
                    ),
                    "season_wins": int((delta < 0).sum()),
                    "season_count": len(delta),
                    "bootstrap_95_low": float(
                        np.quantile(draws, 0.025)
                    ),
                    "bootstrap_95_high": float(
                        np.quantile(draws, 0.975)
                    ),
                    "sign_flip_p_value": _sign_flip_pvalue(delta),
                    "limited_history_delta": float(
                        history.loc["limited", variant]
                        - history.loc[
                            "limited", "separate_full_base"
                        ]
                    ),
                    "veteran_delta": float(
                        history.loc["veteran", variant]
                        - history.loc[
                            "veteran", "separate_full_base"
                        ]
                    ),
                }
            )
    comparisons = pd.DataFrame(rows)
    primary = comparisons["primary_test"].eq(1)
    comparisons["bh_q_value"] = np.nan
    comparisons.loc[primary, "bh_q_value"] = (
        _benjamini_hochberg(
            comparisons.loc[primary, "sign_flip_p_value"]
        )
    )
    return comparisons.sort_values(
        ["position", "pooled_delta"]
    ).reset_index(drop=True)


def _coverage(
    target: pd.DataFrame,
) -> pd.DataFrame:
    training = target["conditional_ppg_training_eligible"].eq(1)
    rows = []
    for position in POSITIONS:
        position_rows = target[
            training & target["position"].eq(position)
        ]
        for family, features in _position_families(position).items():
            if family == "all_targeted":
                continue
            for feature in features:
                values = pd.to_numeric(
                    position_rows[feature],
                    errors="coerce",
                )
                rows.append(
                    {
                        "position": position,
                        "family": family,
                        "feature_name": feature,
                        "training_coverage": float(
                            values.notna().mean()
                        ),
                        "first_available_season": (
                            int(
                                position_rows.loc[
                                    values.notna(),
                                    "season",
                                ].min()
                            )
                            if values.notna().any()
                            else pd.NA
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _summary_markdown(comparisons: pd.DataFrame) -> str:
    lines = [
        "# Position-Aware Feature-Family Results",
        "",
        "Negative deltas favor the added family. The BH q-value covers the "
        "16 prespecified position-family tests; `all_targeted` is secondary.",
        "",
        "| Position | Variant | RMSE | Delta vs position base | "
        "Delta vs pooled full | Recent delta | Limited delta | "
        "95% interval | Wins | p | q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.itertuples(index=False):
        q_value = (
            f"{row.bh_q_value:.3f}"
            if np.isfinite(row.bh_q_value)
            else ""
        )
        lines.append(
            f"| {row.position} | `{row.variant}` | "
            f"{row.variant_rmse:.4f} | {row.pooled_delta:+.4f} | "
            f"{row.delta_vs_pooled_full:+.4f} | "
            f"{row.recent_mean_season_delta:+.4f} | "
            f"{row.limited_history_delta:+.4f} | "
            f"[{row.bootstrap_95_low:+.4f}, "
            f"{row.bootstrap_95_high:+.4f}] | "
            f"{row.season_wins}/{row.season_count} | "
            f"{row.sign_flip_p_value:.3f} | {q_value} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features, base_features, feature_run_id = _load_inputs()
    target = build_target_frames(
        features,
        VALIDATION_END,
    )[CONDITIONAL_PPG_TARGET]
    prior_predictions = pd.read_csv(POSITION_RESULTS)
    required_prior = {
        "pooled_full",
        "separate_full",
    }
    if not required_prior.issubset(prior_predictions.columns):
        raise ValueError("Position-model comparators are unavailable")
    run_id = create_run_id("m4a_position_feature_families")

    oof_parts = []
    parameter_parts = []
    assignment_parts = []
    specification_rows = []
    score_parts = []
    prediction_parts = []
    total_experiments = len(POSITIONS) * (
        len(PRIMARY_FAMILIES) + 1
    )
    experiment_index = 0
    for position in POSITIONS:
        component_target = target[
            target["position"].eq(position)
        ].copy()
        families = _position_families(position)
        keys = ["player_key", "season"]
        reference = prior_predictions[
            prior_predictions["position"].eq(position)
        ][
            [
                *keys,
                "actual",
                "pooled_full",
                "separate_full",
            ]
        ].rename(
            columns={
                "pooled_full": "pooled_full_reference",
                "separate_full": "separate_full_base",
            }
        )
        methods = [
            "pooled_full_reference",
            "separate_full_base",
        ]
        position_predictions = reference.copy()
        for variant, added_features in families.items():
            experiment_index += 1
            feature_columns = tuple(
                dict.fromkeys((*base_features, *added_features))
            )
            missing = sorted(
                set(feature_columns).difference(component_target.columns)
            )
            if missing:
                raise ValueError(
                    f"{position}/{variant} missing columns: {missing}"
                )
            spec = _spec(position, variant)
            assignments = make_fold_assignments(
                component_target,
                CONDITIONAL_PPG_TARGET,
                run_id,
                VALIDATION_START,
                N_SPLITS,
                RANDOM_SEED,
            )
            print(
                f"[{experiment_index}/{total_experiments}] "
                f"{position} {variant} (+{len(added_features)})",
                flush=True,
            )
            oof, parameters = run_model_spec(
                component_target,
                assignments,
                spec,
                feature_columns,
                run_id,
                feature_run_id,
                VALIDATION_START,
                N_SPLITS,
                RANDOM_SEED,
            )
            oof_parts.append(oof)
            parameters["position"] = position
            parameters["variant"] = variant
            parameter_parts.append(parameters)
            assignments["position"] = position
            assignments["variant"] = variant
            assignment_parts.append(assignments)
            specification_rows.append(
                {
                    "run_id": run_id,
                    "feature_run_id": feature_run_id,
                    "model_name": spec.model_name,
                    "position": position,
                    "variant": variant,
                    "base_feature_count": len(base_features),
                    "added_feature_count": len(added_features),
                    "feature_count": len(feature_columns),
                    "added_features_json": json.dumps(
                        added_features
                    ),
                    "feature_names_json": json.dumps(
                        feature_columns
                    ),
                }
            )
            variant_predictions = oof[
                [*keys, "final_prediction"]
            ].rename(columns={"final_prediction": variant})
            position_predictions = position_predictions.merge(
                variant_predictions,
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            methods.append(variant)
        position_predictions = position_predictions.merge(
            component_target[
                [
                    *keys,
                    "year_exp",
                    "is_rookie",
                    "has_prior_outcome",
                ]
            ],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        position_predictions["position"] = position
        position_predictions["history_depth"] = _history_depth(
            position_predictions
        )
        position_predictions["history_group"] = np.where(
            position_predictions["history_depth"].isin(
                (
                    "rookie",
                    "second_year",
                    "other_no_history",
                )
            ),
            "limited",
            "veteran",
        )
        position_scores = _score_predictions(
            position_predictions,
            tuple(methods),
        )
        position_scores["position"] = position
        score_parts.append(position_scores)
        prediction_parts.append(position_predictions)

    oof = pd.concat(oof_parts, ignore_index=True)
    parameters = pd.concat(parameter_parts, ignore_index=True)
    assignments = pd.concat(assignment_parts, ignore_index=True)
    scores = pd.concat(score_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    comparisons = _comparisons(scores)
    coverage = _coverage(target)

    if (
        assignments["training_through_season"]
        >= assignments["season"]
    ).any():
        raise ValueError("Position-family assignment leakage detected")
    expected = int(
        target["season"].between(
            VALIDATION_START,
            VALIDATION_END,
        ).sum()
    )
    for variant in (*PRIMARY_FAMILIES, "all_targeted"):
        if predictions[variant].notna().sum() != expected:
            raise ValueError(
                f"{variant} does not cover all {expected} OOF rows"
            )

    oof.to_csv(RESULTS_DIR / "model_oof.csv", index=False)
    parameters.to_csv(
        RESULTS_DIR / "hyperparameters.csv",
        index=False,
    )
    assignments.to_csv(
        RESULTS_DIR / "fold_assignments.csv",
        index=False,
    )
    pd.DataFrame(specification_rows).to_csv(
        RESULTS_DIR / "model_specifications.csv",
        index=False,
    )
    predictions.to_csv(
        RESULTS_DIR / "comparison_predictions.csv",
        index=False,
    )
    scores.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    comparisons.to_csv(
        RESULTS_DIR / "family_comparisons.csv",
        index=False,
    )
    coverage.to_csv(
        RESULTS_DIR / "feature_coverage.csv",
        index=False,
    )
    (RESULTS_DIR / "summary.md").write_text(
        _summary_markdown(comparisons),
        encoding="utf-8",
    )
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
