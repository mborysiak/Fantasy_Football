"""Audit additive versus proportional salary-market reconciliation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
RESULTS = STUDY_DIR / "results"
VALIDATIONS_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
SIMULATION_DB = ROOT / "Data" / "Databases" / "Simulation.sqlite3"
METHOD = "current_locked_spec_v3_resid_share_features"
LEAGUE = "beta"
YEAR = 2026
LEAGUE_BUDGET = 12 * 298
LEAGUE_SLOTS = 12 * 13
FLOOR = 1.0


def additive_floor_normalize(
    values: pd.Series,
    slots: int,
    budget: float,
) -> tuple[pd.Series, float]:
    values = pd.to_numeric(values, errors="coerce").fillna(FLOOR).clip(lower=FLOOR)
    top_idx = values.nlargest(slots).index
    top_values = values.loc[top_idx]
    pre_total = float(top_values.sum())
    if np.isclose(pre_total, budget, atol=1e-10):
        shift = 0.0
    elif pre_total < budget:
        shift = (budget - pre_total) / slots
    else:
        lower = float(FLOOR - top_values.max())
        upper = 0.0
        for _ in range(100):
            midpoint = (lower + upper) / 2
            total = float(np.maximum(FLOOR, top_values + midpoint).sum())
            if total > budget:
                upper = midpoint
            else:
                lower = midpoint
        shift = (lower + upper) / 2
    adjusted = (values + shift).clip(lower=FLOOR)
    if not np.isclose(adjusted.loc[top_idx].sum(), budget, atol=1e-7):
        raise AssertionError("Additive normalization missed the market budget.")
    return adjusted, float(shift)


def metric_row(
    data: pd.DataFrame,
    prediction: str,
    method: str,
    period: str,
) -> dict[str, float | int | str]:
    residual = data.actual_salary - data[prediction]
    return {
        "period": period,
        "normalization": method,
        "player_years": int(len(data)),
        "mean_actual_minus_prediction": float(residual.mean()),
        "mae": float(residual.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def load_backtest() -> pd.DataFrame:
    with sqlite3.connect(VALIDATIONS_DB) as conn:
        frame = pd.read_sql_query(
            """
            SELECT *
            FROM Salary_Backtest_Predictions
            WHERE league=? AND method_version=?
            """,
            conn,
            params=[LEAGUE, METHOD],
        )
    if frame.empty:
        raise ValueError(f"No stored salary rows for {LEAGUE} / {METHOD}.")
    return frame


def add_additive_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pred_salary_additive"] = np.nan
    output["additive_shift"] = np.nan
    for year, group in output.groupby("year", sort=True):
        non_keeper = group[group.is_keeper.fillna(0).eq(0)]
        slots = int(group.available_slots.iloc[0])
        budget = float(group.available_budget.iloc[0])
        adjusted, shift = additive_floor_normalize(
            non_keeper.pred_salary_raw,
            slots,
            budget,
        )
        output.loc[adjusted.index, "pred_salary_additive"] = adjusted
        output.loc[group.index, "additive_shift"] = shift
        keeper = group[group.is_keeper.fillna(0).eq(1)]
        output.loc[keeper.index, "pred_salary_additive"] = keeper.actual_salary
    return output


def current_keeper_context() -> pd.DataFrame:
    with sqlite3.connect(SIMULATION_DB) as conn:
        source = pd.read_sql_query(
            "SELECT player, salary FROM Salaries WHERE year=? AND league=?",
            conn,
            params=[YEAR, LEAGUE],
        )
        keepers = pd.read_sql_query(
            """
            SELECT player, keeper_salary
            FROM League_Keepers
            WHERE year=? AND league=?
            """,
            conn,
            params=[YEAR, LEAGUE],
        )
        candidates = pd.read_sql_query(
            """
            SELECT player
            FROM Salaries_Pred
            WHERE year=? AND league=?
            """,
            conn,
            params=[YEAR, f"{LEAGUE}pred"],
        )

    keeper_source = keepers.merge(source, on="player", how="left")
    keeper_source["market_value"] = keeper_source.salary.fillna(
        keeper_source.keeper_salary
    )
    keeper_spend = float(keepers.keeper_salary.sum())
    keeper_market_value = float(keeper_source.market_value.sum())
    available_slots = LEAGUE_SLOTS - len(keepers)
    available_budget = LEAGUE_BUDGET - keeper_spend
    keeper_pool_base_budget = LEAGUE_BUDGET - keeper_market_value
    keeper_pool_inflation = available_budget / keeper_pool_base_budget

    candidate_source = (
        candidates.merge(source, on="player", how="left")
        .merge(keepers.assign(is_keeper=1), on="player", how="left")
    )
    auctionable = candidate_source[candidate_source.is_keeper.isna()].copy()
    auctionable["source_floor"] = auctionable.salary.fillna(0).clip(lower=FLOOR)
    top_source = auctionable.nlargest(available_slots, "source_floor")
    source_excess = float((top_source.source_floor - FLOOR).sum())
    source_market_scale = (
        available_budget - available_slots * FLOOR
    ) / source_excess

    return pd.DataFrame(
        [
            {
                "year": YEAR,
                "keeper_count": int(len(keepers)),
                "keeper_spend": keeper_spend,
                "keeper_market_value": keeper_market_value,
                "keeper_contract_discount": keeper_market_value - keeper_spend,
                "available_slots": int(available_slots),
                "available_budget": available_budget,
                "keeper_pool_base_budget": keeper_pool_base_budget,
                "keeper_pool_inflation": keeper_pool_inflation,
                "source_market_scale": source_market_scale,
            }
        ]
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions = add_additive_predictions(load_backtest())
    observed = predictions[
        predictions.actual_salary_observed.eq(1)
        & predictions.is_keeper.fillna(0).eq(0)
    ].copy()

    overall = pd.DataFrame(
        [
            metric_row(observed, "pred_salary", "proportional_floor", "all_years"),
            metric_row(
                observed,
                "pred_salary_additive",
                "additive_floor",
                "all_years",
            ),
        ]
    )
    by_year_rows = []
    for year, group in observed.groupby("year", sort=True):
        for prediction, method in [
            ("pred_salary", "proportional_floor"),
            ("pred_salary_additive", "additive_floor"),
        ]:
            row = metric_row(group, prediction, method, str(int(year)))
            row["additive_shift"] = float(group.additive_shift.iloc[0])
            by_year_rows.append(row)
    by_year = pd.DataFrame(by_year_rows)
    context = current_keeper_context()

    overall.to_csv(RESULTS / "normalization_accuracy.csv", index=False)
    by_year.to_csv(RESULTS / "normalization_by_year.csv", index=False)
    context.to_csv(RESULTS / "current_keeper_market_context.csv", index=False)

    prop = overall[overall.normalization.eq("proportional_floor")].iloc[0]
    add = overall[overall.normalization.eq("additive_floor")].iloc[0]
    summary = "\n".join(
        [
            "# Additive Salary Normalization Audit",
            "",
            "This holds the v3 rolling raw salary predictions fixed and changes only",
            "the final known-budget reconciliation rule.",
            "",
            "| Method | Mean actual - prediction | MAE | RMSE |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Proportional above `$1` | ${prop.mean_actual_minus_prediction:.3f} "
                f"| ${prop.mae:.3f} | ${prop.rmse:.3f} |"
            ),
            (
                f"| Additive with `$1` floor | ${add.mean_actual_minus_prediction:.3f} "
                f"| ${add.mae:.3f} | ${add.rmse:.3f} |"
            ),
            "",
            (
                f"On {int(add.player_years)} observed player-years, additive "
                f"normalization changes MAE by ${add.mae - prop.mae:+.3f} and "
                f"RMSE by ${add.rmse - prop.rmse:+.3f}."
            ),
            "",
            "This result supports testing additive normalization in the full v4",
            "optimizer replay, but it does not measure the incremental value of the",
            "new keeper-market input features.",
            "",
        ]
    )
    (RESULTS / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
