"""Build history-gated and provider-stack diagnostics from completed OOF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_SEED = 1234
PROJECTION_MODEL = "projection_only_lightgbm_core"
FULL_MODEL = "full_lightgbm_base"


def _rmse(actual: pd.Series, prediction: pd.Series) -> float:
    return float(np.sqrt(np.square(prediction - actual).mean()))


def _history_depth(frame: pd.DataFrame) -> pd.Series:
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    prior = pd.to_numeric(
        frame["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    result = pd.Series("other_no_history", index=frame.index, dtype=object)
    result.loc[rookie] = "rookie"
    result.loc[~rookie & year_exp.eq(1)] = "second_year"
    result.loc[~rookie & year_exp.ge(2) & prior] = "veteran_with_history"
    result.loc[year_exp.isna()] = "unknown_experience"
    return result


def _metric_rows(
    frame: pd.DataFrame,
    prediction_columns: tuple[str, ...],
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
        for method in prediction_columns:
            prediction = pd.to_numeric(group[method], errors="coerce")
            rows.extend(
                [
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
                        "value": float((prediction - actual).abs().mean()),
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
                ]
            )
    return rows


def history_hybrid_diagnostics() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    oof = pd.read_csv(RESULTS_DIR / "oof_predictions.csv")
    keys = ["player_key", "season"]
    projection = oof[oof["model_name"].eq(PROJECTION_MODEL)].copy()
    full = oof[oof["model_name"].eq(FULL_MODEL)][
        [*keys, "final_prediction"]
    ].rename(columns={"final_prediction": "full_prediction"})
    frame = projection.merge(full, on=keys, validate="one_to_one")
    frame.rename(
        columns={"final_prediction": "projection_prediction"},
        inplace=True,
    )
    year_exp = pd.to_numeric(frame["year_exp"], errors="coerce")
    rookie = pd.to_numeric(frame["is_rookie"], errors="coerce").eq(1)
    prior = pd.to_numeric(
        frame["has_prior_outcome"], errors="coerce"
    ).fillna(0).eq(1)
    frame["history_depth"] = _history_depth(frame)
    frame["no_prior_hybrid"] = np.where(
        ~prior,
        frame["projection_prediction"],
        frame["full_prediction"],
    )
    limited = rookie | year_exp.eq(1) | (~prior)
    frame["limited_history_hybrid"] = np.where(
        limited,
        frame["projection_prediction"],
        frame["full_prediction"],
    )
    frame["history_group"] = np.where(
        frame["history_depth"].isin(
            ("rookie", "second_year", "other_no_history")
        ),
        "limited",
        "veteran",
    )
    route_rows = []
    router_columns = []
    for minimum_rows in (25, 50, 100):
        column = f"causal_router_min_{minimum_rows}"
        router_columns.append(column)
        frame[column] = np.nan
        for season, current in frame.groupby("season", sort=True):
            prior_frame = frame[frame["season"].lt(season)]
            frame.loc[current.index, column] = current[
                "full_prediction"
            ]
            for (position, history_group), indices in current.groupby(
                ["position", "history_group"]
            ).groups.items():
                prior = prior_frame[
                    prior_frame["position"].eq(position)
                    & prior_frame["history_group"].eq(history_group)
                ]
                select_projection = False
                projection_rmse = np.nan
                full_rmse = np.nan
                if (
                    len(prior) >= minimum_rows
                    and prior["season"].nunique() >= 2
                ):
                    projection_rmse = _rmse(
                        prior["actual"],
                        prior["projection_prediction"],
                    )
                    full_rmse = _rmse(
                        prior["actual"],
                        prior["full_prediction"],
                    )
                    select_projection = projection_rmse < full_rmse
                if select_projection:
                    frame.loc[indices, column] = frame.loc[
                        indices, "projection_prediction"
                    ]
                route_rows.append(
                    {
                        "minimum_prior_rows": minimum_rows,
                        "season": int(season),
                        "position": position,
                        "history_group": history_group,
                        "prior_rows": len(prior),
                        "prior_seasons": int(
                            prior["season"].nunique()
                        ),
                        "prior_projection_rmse": projection_rmse,
                        "prior_full_rmse": full_rmse,
                        "selected_model": (
                            "projection"
                            if select_projection
                            else "full"
                        ),
                    }
                )
        for (position, history_group), prior_group in frame.groupby(
            ["position", "history_group"]
        ):
            select_projection = False
            projection_rmse = np.nan
            full_rmse = np.nan
            if (
                len(prior_group) >= minimum_rows
                and prior_group["season"].nunique() >= 2
            ):
                projection_rmse = _rmse(
                    prior_group["actual"],
                    prior_group["projection_prediction"],
                )
                full_rmse = _rmse(
                    prior_group["actual"],
                    prior_group["full_prediction"],
                )
                select_projection = projection_rmse < full_rmse
            route_rows.append(
                {
                    "minimum_prior_rows": minimum_rows,
                    "season": 2026,
                    "position": position,
                    "history_group": history_group,
                    "prior_rows": len(prior_group),
                    "prior_seasons": int(
                        prior_group["season"].nunique()
                    ),
                    "prior_projection_rmse": projection_rmse,
                    "prior_full_rmse": full_rmse,
                    "selected_model": (
                        "projection"
                        if select_projection
                        else "full"
                    ),
                }
            )
    prediction_columns = (
        "projection_prediction",
        "full_prediction",
        "no_prior_hybrid",
        "limited_history_hybrid",
        *router_columns,
    )
    rows = []
    rows.extend(
        _metric_rows(frame, prediction_columns, "pooled", None)
    )
    for slice_type, column in (
        ("season", "season"),
        ("position", "position"),
        ("history_depth", "history_depth"),
    ):
        rows.extend(
            _metric_rows(
                frame, prediction_columns, slice_type, column
            )
        )
    scores = pd.DataFrame(rows)

    season = scores[
        scores["slice_type"].eq("season")
        & scores["metric"].eq("rmse")
    ].pivot(index="slice_value", columns="method", values="value")
    comparisons = []
    for index, method in enumerate(
        (
            "no_prior_hybrid",
            "limited_history_hybrid",
            *router_columns,
        )
    ):
        delta = (
            season[method] - season["full_prediction"]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + index)
        draws = np.array(
            [
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        pooled = scores[
            scores["slice_type"].eq("pooled")
            & scores["metric"].eq("rmse")
        ].set_index("method")["value"]
        comparisons.append(
            {
                "method": method,
                "reference": "full_prediction",
                "pooled_rmse": float(pooled[method]),
                "reference_rmse": float(pooled["full_prediction"]),
                "pooled_delta": float(
                    pooled[method] - pooled["full_prediction"]
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
    prediction_output = frame[
        [
            "player_key",
            "season",
            "position",
            "actual",
            "history_depth",
            *prediction_columns,
        ]
    ]
    return (
        prediction_output,
        pd.DataFrame(comparisons),
        scores,
        pd.DataFrame(route_rows),
    )


def provider_stack_comparisons() -> pd.DataFrame:
    scores = pd.read_csv(RESULTS_DIR / "provider_stack_scores.csv")
    season = scores[
        scores["slice_type"].eq("season")
        & scores["metric"].eq("rmse")
    ].pivot(index="slice_value", columns="method", values="value")
    pooled = scores[
        scores["slice_type"].eq("pooled")
        & scores["metric"].eq("rmse")
    ].set_index("method")["value"]
    rows = []
    for index, method in enumerate(
        (
            "causal_provider_stack_global",
            "causal_provider_stack_position",
        )
    ):
        delta = (
            season[method] - season["configured_median"]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + 100 + index)
        draws = np.array(
            [
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(20_000)
            ]
        )
        rows.append(
            {
                "method": method,
                "reference": "configured_median",
                "pooled_rmse": float(pooled[method]),
                "reference_rmse": float(
                    pooled["configured_median"]
                ),
                "pooled_delta": float(
                    pooled[method] - pooled["configured_median"]
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


def main() -> None:
    predictions, hybrid_comparisons, hybrid_scores, routes = (
        history_hybrid_diagnostics()
    )
    stack_comparisons = provider_stack_comparisons()
    predictions.to_csv(
        RESULTS_DIR / "history_hybrid_predictions.csv", index=False
    )
    hybrid_scores.to_csv(
        RESULTS_DIR / "history_hybrid_scores.csv", index=False
    )
    hybrid_comparisons.to_csv(
        RESULTS_DIR / "history_hybrid_comparisons.csv", index=False
    )
    routes.to_csv(
        RESULTS_DIR / "history_router_selections.csv", index=False
    )
    stack_comparisons.to_csv(
        RESULTS_DIR / "provider_stack_comparisons.csv", index=False
    )
    print(hybrid_comparisons.to_string(index=False))
    print(stack_comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
