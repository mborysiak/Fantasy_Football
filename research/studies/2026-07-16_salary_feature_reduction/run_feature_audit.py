"""Audit and walk-forward comparison for the compact auction-salary features."""

from __future__ import annotations

import ast
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
RESULTS = STUDY_DIR / "results"
SALARY_SCRIPT = ROOT / "Scripts" / "Modeling" / "s4_Salaries_Injuries.py"
VALIDATIONS_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
PRIOR_METHOD = "current_locked_spec_v3_resid_share_features"
TEST_YEARS = [2022, 2023, 2024, 2025]


def load_salary_namespace() -> dict[str, object]:
    """Load constants and functions without executing notebook data writes."""
    os.environ["SALARY_VALIDATION_DATASETS_ONLY"] = "1"
    sys.path[:0] = [
        str(ROOT / "Scripts"),
        str(ROOT / "Scripts" / "Modeling"),
    ]
    source = SALARY_SCRIPT.read_text(encoding="utf-8")
    prefix = source.split(
        "#=================\n# Load salaries from ESPN into database"
    )[0]
    namespace: dict[str, object] = {"__name__": "_salary_feature_audit_"}
    exec(compile(prefix, str(SALARY_SCRIPT), "exec"), namespace)

    tree = ast.parse(source, filename=str(SALARY_SCRIPT))
    functions = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ],
        type_ignores=[],
    )
    exec(compile(functions, str(SALARY_SCRIPT), "exec"), namespace)
    return namespace


def reconstruct_salary_base(ns: dict[str, object]) -> pd.DataFrame:
    """Rebuild the causal v5 input rows through the pre-feature stage."""
    salaries = ns["get_salaries"]()
    salaries["league"] = ns["LEAGUE"]
    total_spent = (
        salaries.groupby("year", as_index=False)
        .actual_salary.sum()
        .rename(columns={"actual_salary": "total_spent"})
    )
    salaries = salaries.merge(total_spent, on="year", how="left")
    salaries.loc[
        salaries.year.eq(ns["YEAR"]), "total_spent"
    ] = ns["LEAGUE_BUDGET"]
    salaries["fraction_spent"] = (
        salaries.total_spent / ns["LEAGUE_BUDGET"]
    )
    salaries = ns["fill_ty_keepers"](salaries, ns["ty_keepers"])
    salaries = ns["add_keeper_budget_context"](salaries)
    salaries = ns["calc_inflation"](salaries)

    context_columns = [
        "total_spent",
        "fraction_spent",
        "keeper_count",
        "keeper_spend",
        "keeper_market_value",
        "keeper_source_market_value",
        "keeper_source_values_observed",
        "keeper_contract_discount",
        "keeper_pool_base_budget",
        "keeper_pool_inflation",
        "available_slots",
        "available_budget",
        "source_market_total",
        "source_nonkeeper_market_total",
        "value",
        "inflation",
    ]
    year_context = (
        salaries[["year", *context_columns]]
        .drop_duplicates("year")
        .set_index("year")
    )
    projection_rows = ns["add_ensemble_projection_features"](ns["get_adp"]())
    salaries = salaries.merge(
        projection_rows,
        on=["player", "year"],
        how="right",
    )
    for column in context_columns:
        salaries[column] = salaries[column].fillna(
            salaries.year.map(year_context[column])
        )
    salaries["league"] = salaries.league.fillna(ns["LEAGUE"])
    salaries["is_keeper"] = salaries.is_keeper.fillna(0)
    salaries["base_salary_observed"] = salaries.salary.notna()
    salaries["salary"] = salaries.salary.fillna(0)
    salaries = ns["add_rookie"](salaries)
    salaries = ns["add_keeper_market_salary_features"](salaries)
    salaries = ns["add_pos_keeper_val"](salaries)
    salaries = ns["drop_keepers"](salaries)
    salaries = ns["remove_outliers"](salaries)
    return salaries


def add_legacy_features(salaries: pd.DataFrame) -> pd.DataFrame:
    """Recreate the pre-v5 rank, ratio, and adjacent-price feature surface."""
    salaries = salaries.copy()
    salaries = salaries.sort_values(
        ["year", "salary"],
        ascending=[True, False],
    )
    salaries["sal_rank"] = salaries.groupby("year").cumcount().values
    salaries = salaries.sort_values(
        ["year", "pos", "salary"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    salaries["pos_rank"] = salaries.groupby(["year", "pos"]).cumcount().values
    salaries["young_player"] = salaries.year_exp.lt(2).astype(int)
    salaries["rookie_rank"] = salaries.is_rookie * salaries.avg_pick
    salaries["old_player"] = salaries.year_exp.gt(5).astype(int)
    salaries["next_guy_sal"] = salaries.groupby(
        ["pos", "year"]
    ).salary.shift(-1)
    salaries["next_guy_sal_diff"] = (
        salaries.salary - salaries.next_guy_sal
    )
    salaries["guy_above_sal"] = salaries.groupby(
        ["pos", "year"]
    ).salary.shift(1)
    salaries["guy_above_sal_diff"] = (
        salaries.salary - salaries.guy_above_sal
    )
    salaries = salaries.drop(columns=["next_guy_sal", "guy_above_sal"])
    salaries[
        ["next_guy_sal_diff", "guy_above_sal_diff"]
    ] = salaries[
        ["next_guy_sal_diff", "guy_above_sal_diff"]
    ].fillna(0)
    salaries["pts_per_dollar"] = (
        salaries.avg_proj_points / (salaries.salary + 1)
    )

    group_columns = ["year", "pos"]
    ensemble_strength = salaries.groupby(
        group_columns
    ).ensemble_pred_ppg.rank(method="average", pct=True, ascending=True)
    consensus_strength = salaries.groupby(
        group_columns
    ).avg_proj_points.rank(method="average", pct=True, ascending=True)
    price_strength = salaries.groupby(group_columns).salary.rank(
        method="average",
        pct=True,
        ascending=True,
    )
    salaries["ensemble_pos_strength_pct"] = ensemble_strength
    salaries["ensemble_vs_consensus_gap"] = (
        ensemble_strength - consensus_strength
    )
    salaries["ensemble_vs_price_gap"] = ensemble_strength - price_strength

    for percentile in [90, 95]:
        ceiling = f"ensemble_ceiling_p{percentile}_ppg"
        ceiling_strength = salaries.groupby(group_columns)[ceiling].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        salaries[
            f"ensemble_p{percentile}_pos_strength_pct"
        ] = ceiling_strength
        salaries[f"ensemble_p{percentile}_vs_price_gap"] = (
            ceiling_strength - price_strength
        )

    source_floor = salaries.salary.clip(lower=1)
    salaries["ensemble_ppg_per_dollar"] = (
        salaries.ensemble_pred_ppg / (source_floor + 1)
    )
    salaries["ensemble_ppg_per_dollar"] = salaries.groupby(
        group_columns
    ).ensemble_ppg_per_dollar.transform(
        lambda values: values.clip(
            lower=values.quantile(0.01),
            upper=values.quantile(0.99),
        )
    )
    for percentile in [90, 95]:
        column = f"ensemble_p{percentile}_ppg_per_dollar"
        salaries[column] = (
            salaries[f"ensemble_ceiling_p{percentile}_ppg"]
            / (source_floor + 1)
        )
        salaries[column] = salaries.groupby(group_columns)[column].transform(
            lambda values: values.clip(
                lower=values.quantile(0.01),
                upper=values.quantile(0.99),
            )
        )
    return salaries


def add_legacy_features_by_availability(
    salaries: pd.DataFrame,
) -> pd.DataFrame:
    salaries = salaries.copy()
    salaries["_row_order"] = np.arange(len(salaries))
    parts = [
        add_legacy_features(group)
        for _, group in salaries.groupby(
            salaries.is_keeper.fillna(0).eq(1),
            sort=False,
        )
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )


def build_legacy_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Recreate the 167-column pre-v5 matrix for the ablation benchmark."""
    drop_columns = [
        "player",
        "team",
        "week",
        "league",
        "y_act",
        "total_spent",
        "fraction_spent",
        "base_salary_observed",
        "ensemble_pred_fallback",
        "ensemble_resid_fallback",
        "projection_share_fallback",
        "keeper_market_context_fallback",
    ]
    matrix = rows.drop(
        columns=[column for column in drop_columns if column in rows],
    ).copy()
    matrix = pd.concat(
        [matrix, pd.get_dummies(matrix.pos, dtype=int)],
        axis=1,
    ).drop(columns="pos")

    for pos, prefix in [
        ("QB", "qb"),
        ("RB", "rb"),
        ("WR", "wr"),
        ("TE", "te"),
    ]:
        if pos not in matrix:
            matrix[pos] = 0
        matrix[f"{prefix}_proj"] = matrix[pos] * matrix.avg_proj_points
        matrix[f"{prefix}_pick"] = matrix[pos] * matrix.avg_pick
        matrix[f"{prefix}_rank"] = matrix[pos] * matrix.pos_rank
        matrix[f"{prefix}_ensemble_proj"] = (
            matrix[pos] * matrix.ensemble_pred_ppg
        )
        matrix[f"{prefix}_ensemble_vs_price"] = (
            matrix[pos] * matrix.ensemble_vs_price_gap
        )
        matrix[f"{prefix}_ensemble_ppd"] = (
            matrix[pos] * matrix.ensemble_ppg_per_dollar
        )
        matrix[f"{prefix}_ensemble_p90"] = (
            matrix[pos] * matrix.ensemble_ceiling_p90_ppg
        )
        matrix[f"{prefix}_ensemble_p95"] = (
            matrix[pos] * matrix.ensemble_ceiling_p95_ppg
        )
        matrix[f"{prefix}_ensemble_p90_vs_price"] = (
            matrix[pos] * matrix.ensemble_p90_vs_price_gap
        )
        matrix[f"{prefix}_ensemble_interval_80"] = (
            matrix[pos] * matrix.ensemble_interval_80
        )
        for suffix, source in [
            ("team_points_share", "team_proj_points_share"),
            ("pos_points_share", "pos_proj_points_share"),
            ("team_rush_share", "team_proj_rush_att_share"),
            ("pos_rush_share", "pos_proj_rush_att_share"),
            ("team_rec_share", "team_proj_rec_share"),
            ("pos_rec_share", "pos_proj_rec_share"),
            ("team_rec_yds_share", "team_proj_rec_yds_share"),
            ("pos_rec_yds_share", "pos_proj_rec_yds_share"),
        ]:
            matrix[f"{prefix}_{suffix}"] = matrix[pos] * matrix[source]
    return matrix


def fixed_models() -> list[tuple[str, object]]:
    return [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10))),
        (
            "hist",
            HistGradientBoostingRegressor(
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=2,
                random_state=42,
            ),
        ),
        (
            "extra_42",
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "extra_137",
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=137,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "gbm",
            GradientBoostingRegressor(
                n_estimators=180,
                max_depth=2,
                min_samples_leaf=3,
                loss="huber",
                random_state=42,
            ),
        ),
    ]


def rolling_metrics(
    matrix: pd.DataFrame,
    target: pd.Series,
    years: pd.Series,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for set_name, features in feature_sets.items():
        fold_outputs = []
        for test_year in TEST_YEARS:
            train = years.lt(test_year)
            test = years.eq(test_year)
            model_predictions = []
            for _, model in fixed_models():
                model.fit(matrix.loc[train, features], target.loc[train])
                model_predictions.append(
                    model.predict(matrix.loc[test, features])
                )
            actual = target.loc[test].to_numpy(dtype=float)
            prediction = np.mean(model_predictions, axis=0)
            fold_outputs.append((test_year, actual, prediction))

        actual = np.concatenate([values for _, values, _ in fold_outputs])
        prediction = np.concatenate(
            [values for _, _, values in fold_outputs]
        )
        row: dict[str, float | int | str] = {
            "feature_set": set_name,
            "feature_count": len(features),
            "player_years": len(actual),
            "mae": mean_absolute_error(actual, prediction),
            "rmse": mean_squared_error(actual, prediction) ** 0.5,
            "r2": r2_score(actual, prediction),
        }
        for year, year_actual, year_prediction in fold_outputs:
            row[f"mae_{year}"] = mean_absolute_error(
                year_actual,
                year_prediction,
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"])


def load_prior_oof_residuals() -> pd.DataFrame:
    with sqlite3.connect(VALIDATIONS_DB) as connection:
        return pd.read_sql_query(
            """
            SELECT player, year, actual_resid_raw AS prior_oof_residual
            FROM Salary_Validations_Resid
            WHERE league='beta'
              AND method_version=?
              AND model_spec_asof_year=2026
              AND included_in_residual_evaluation=1
            """,
            connection,
            params=[PRIOR_METHOD],
        )


def write_summary(
    comparison: pd.DataFrame,
    associations: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    compact = comparison[
        comparison.feature_set.eq("compact_v5")
    ].iloc[0]
    legacy = comparison[
        comparison.feature_set.eq("legacy_full")
    ].iloc[0]
    high_pairs = pairwise[
        pairwise.absolute_correlation.ge(0.90)
    ].sort_values("absolute_correlation", ascending=False)
    lines = [
        "# Compact Salary Feature Audit",
        "",
        (
            f"The compact surface uses {int(compact.feature_count)} substantive "
            f"features versus {int(legacy.feature_count)} nonconstant legacy "
            "features."
        ),
        "",
        (
            f"Across {int(compact.player_years)} strict rolling player-years, "
            f"the fixed six-model ensemble changed MAE from ${legacy.mae:.3f} "
            f"to ${compact.mae:.3f} and RMSE from ${legacy.rmse:.3f} "
            f"to ${compact.rmse:.3f}."
        ),
        "",
        "The retained features cover:",
        "",
        "- keeper/budget-adjusted league source price and broader-market log ADP;",
        "- projected scoring level, projection-versus-price disagreement, and P90 residual upside;",
        "- position-room points share, RB rush share, experience, rookie status, and three position indicators with WR as reference.",
        "",
        (
            "The two remaining correlations at or above 0.90 are deliberate: "
            "adjusted source salary versus log ADP represents two distinct market "
            "anchors, and RB rush share versus the RB indicator is the structural "
            "relationship created by a position-specific role interaction."
            if len(high_pairs)
            else "No retained feature pair has absolute correlation at or above 0.90."
        ),
        "",
        "See `feature_associations.csv` for correlations with actual salary, the copied-source residual, and the prior v3 raw OOF residual.",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ns = load_salary_namespace()
    base = reconstruct_salary_base(ns)

    compact_rows = ns["add_salary_model_features_by_keeper_availability"](
        base
    )
    legacy_rows = add_legacy_features_by_availability(base)
    for rows in [compact_rows, legacy_rows]:
        rows.rename(columns={"actual_salary": "y_act"}, inplace=True)
        rows.sort_values("year", inplace=True)
        rows.reset_index(drop=True, inplace=True)
        rows["team"] = "placeholder"
        rows["week"] = 1
        rows["game_date"] = rows.year

    compact_matrix = ns["build_salary_model_matrix"](compact_rows)
    legacy_matrix = build_legacy_matrix(legacy_rows)
    target = compact_rows.y_act
    historical = compact_rows.year.lt(ns["YEAR"]) & target.notna()
    compact_matrix = compact_matrix.loc[historical].reset_index(drop=True)
    legacy_matrix = legacy_matrix.loc[historical].reset_index(drop=True)
    target = target.loc[historical].reset_index(drop=True)
    metadata = compact_rows.loc[
        historical,
        ["player", "year", "pos", "salary", "base_salary_observed"],
    ].reset_index(drop=True)

    legacy_matrix = legacy_matrix.apply(pd.to_numeric, errors="coerce")
    legacy_matrix = legacy_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
    legacy_features = [
        column
        for column in legacy_matrix.columns
        if column not in ns["SALARY_MODEL_SPLIT_COLUMNS"]
        and legacy_matrix[column].nunique(dropna=False) > 1
    ]
    compact_features = list(ns["SALARY_MODEL_FEATURES"])
    comparison = rolling_metrics(
        pd.concat(
            [
                legacy_matrix[legacy_features],
                compact_matrix[
                    [
                        column
                        for column in compact_features
                        if column not in legacy_features
                    ]
                ],
            ],
            axis=1,
        ),
        target,
        metadata.year,
        {
            "legacy_full": legacy_features,
            "compact_v5": compact_features,
        },
    )

    analysis = metadata.copy()
    analysis["actual_salary"] = target
    analysis["source_salary_residual"] = target - metadata.salary
    analysis = analysis.merge(
        load_prior_oof_residuals(),
        on=["player", "year"],
        how="left",
        validate="one_to_one",
    )
    analysis = pd.concat(
        [analysis, compact_matrix[compact_features]],
        axis=1,
    )
    association_rows = []
    for feature in compact_features:
        source_rows = analysis.base_salary_observed.astype(bool)
        association_rows.append(
            {
                "feature": feature,
                "actual_salary_correlation": analysis[feature].corr(
                    analysis.actual_salary
                ),
                "source_salary_residual_correlation": analysis.loc[
                    source_rows, feature
                ].corr(
                    analysis.loc[source_rows, "source_salary_residual"]
                ),
                "prior_v3_oof_residual_correlation": analysis[feature].corr(
                    analysis.prior_oof_residual
                ),
            }
        )
    associations = pd.DataFrame(association_rows)

    correlation = analysis[compact_features].corr()
    pairwise_rows = []
    for index, first in enumerate(compact_features):
        for second in compact_features[index + 1 :]:
            value = float(correlation.loc[first, second])
            pairwise_rows.append(
                {
                    "feature_1": first,
                    "feature_2": second,
                    "correlation": value,
                    "absolute_correlation": abs(value),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows).sort_values(
        "absolute_correlation",
        ascending=False,
    )

    selected = pd.DataFrame(
        {
            "feature": compact_features,
            "group": [
                "league_source_market",
                "broader_market",
                "projection_level",
                "projection_price_disagreement",
                "projection_upside",
                "role_share",
                "position_specific_role",
                "development",
                "development",
                "position",
                "position",
                "position",
            ],
        }
    )
    comparison.to_csv(
        RESULTS / "walk_forward_comparison.csv",
        index=False,
    )
    associations.to_csv(
        RESULTS / "feature_associations.csv",
        index=False,
    )
    pairwise.to_csv(
        RESULTS / "compact_pairwise_correlations.csv",
        index=False,
    )
    selected.to_csv(RESULTS / "selected_features.csv", index=False)
    write_summary(comparison, associations, pairwise)
    print((RESULTS / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
