"""Independent aggregate audit for the completed chance-frontier replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"
LEVELS = [0.6, 0.7, 0.8, 0.9]


def main() -> None:
    raw = pd.read_csv(RESULTS / "roster_trials.csv")
    reported = pd.read_csv(RESULTS / "frontier_summary_by_year.csv")
    manifest = json.loads((RESULTS / "source_manifest.json").read_text(encoding="utf-8"))
    assert len(raw) == 4_000
    assert not raw.duplicated(["year", "trial", "chance_level"]).any()
    assert raw.status.eq("optimal").all()
    assert set(raw.year) == {2022, 2023, 2024, 2025}
    assert np.allclose(sorted(raw.chance_level.unique()), LEVELS)
    assert raw.groupby(["year", "trial"]).size().eq(4).all()

    roster_sizes = raw.roster.str.split("|").map(len)
    unique_sizes = raw.roster.str.split("|").map(lambda values: len(set(values)))
    assert roster_sizes.eq(13).all() and unique_sizes.eq(13).all()
    assert raw.construction_hit_count.ge(raw.required_construction_hits).all()
    assert raw.contains_top_n.all()
    assert raw.qb_count.between(1, 1).all()
    assert raw.rb_count.between(2, 7).all()
    assert raw.wr_count.between(2, 7).all()
    assert raw.te_count.between(1, 2).all()
    assert raw.actual_cap_feasible.eq(raw.actual_salary_spend.le(298.0 + 1e-8)).all()
    assert raw.loc[~raw.actual_cap_feasible, "actual_points_if_affordable"].isna().all()

    recomputed = raw.groupby(["year", "chance_level"], as_index=False).agg(
        managed_forecast_season_points=("managed_forecast_season_points", "mean"),
        heldout_cap_probability=("heldout_cap_probability", "mean"),
        actual_cap_feasible_rate=("actual_cap_feasible", "mean"),
        actual_cap_overage=("actual_cap_overage", "mean"),
        actual_salary_spend=("actual_salary_spend", "mean"),
        affordable_actual_rosters=("actual_points_if_affordable", "count"),
        actual_points_if_affordable=("actual_points_if_affordable", "mean"),
        point_salary_spend=("point_salary_spend", "mean"),
    )
    check = recomputed.merge(
        reported,
        on=["year", "chance_level"],
        suffixes=("_audit", "_reported"),
        validate="one_to_one",
    )
    for column in recomputed.columns[2:]:
        assert np.allclose(
            check[f"{column}_audit"],
            check[f"{column}_reported"],
            equal_nan=True,
            atol=1e-10,
        ), column

    by_year = recomputed.set_index(["year", "chance_level"])
    monotone = {}
    for year in sorted(raw.year.unique()):
        group = by_year.loc[year]
        monotone[str(year)] = {
            "heldout_probability_increases": bool(
                np.all(np.diff(group.heldout_cap_probability) > 0)
            ),
            "forecast_ev_nonincreasing": bool(
                np.all(np.diff(group.managed_forecast_season_points) <= 1e-10)
            ),
            "mean_actual_overage_decreases": bool(
                np.all(np.diff(group.actual_cap_overage) < 0)
            ),
            "actual_feasibility_nonmonotonic": bool(
                np.any(np.diff(group.actual_cap_feasible_rate) < 0)
            ),
        }

    half_rows = []
    for half, half_data in raw.assign(
        half=np.where(raw.trial < raw.trial.max() / 2, "first", "second")
    ).groupby("half"):
        summary = half_data.groupby("chance_level").agg(
            forecast=("managed_forecast_season_points", "mean"),
            heldout=("heldout_cap_probability", "mean"),
            overage=("actual_cap_overage", "mean"),
        )
        half_rows.append(
            {
                "half": half,
                "heldout_probability_increases": bool(
                    np.all(np.diff(summary.heldout) > 0)
                ),
                "forecast_ev_nonincreasing": bool(
                    np.all(np.diff(summary.forecast) <= 1e-10)
                ),
                "mean_actual_overage_decreases": bool(
                    np.all(np.diff(summary.overage) < 0)
                ),
            }
        )

    raw["model_to_actual_spend_gap"] = (
        raw.actual_salary_spend - raw.heldout_salary_spend_mean
    )
    spend_gap = raw.groupby("chance_level").model_to_actual_spend_gap.mean()
    audit = {
        "raw_rows": int(len(raw)),
        "reported_rows_recomputed_exactly": True,
        "manifest_validation_all_true": all(manifest["validation"].values()),
        "year_level_directions": monotone,
        "trial_half_directions": half_rows,
        "development_model_to_actual_spend_gap": {
            str(level): float(value)
            for level, value in raw[raw.year.le(2024)]
            .groupby("chance_level")
            .model_to_actual_spend_gap.mean()
            .items()
        },
        "all_year_model_to_actual_spend_gap": {
            str(level): float(value) for level, value in spend_gap.items()
        },
    }
    assert audit["manifest_validation_all_true"]
    assert all(row["heldout_probability_increases"] for row in half_rows)
    assert all(row["forecast_ev_nonincreasing"] for row in half_rows)
    assert all(row["mean_actual_overage_decreases"] for row in half_rows)
    (RESULTS / "independent_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
