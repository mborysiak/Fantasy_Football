"""Build same-engine keeper effects and keeper-selection concentration tables."""

from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
BASELINE = "keeper_engine0"
KEEPER_POLICIES = (
    "keeper_tiebreak",
    "keeper_0p01",
    "keeper_1p0",
    "keeper_10p0",
)
METRICS = (
    "forecast_ev",
    "forecast_p10",
    "actual_points",
    "actual_playoff_points",
    "bench_forecast_spend",
    "top3_spend_share",
    "predicted_keeper_option_top2",
    "realized_next_keeper_surplus",
    "realized_next_keeper_hits",
    "actual_keeper_cost_coverage",
)


def paired_effects(
    trials: pd.DataFrame,
    baseline_policy: str = BASELINE,
    policies: tuple[str, ...] = KEEPER_POLICIES,
) -> pd.DataFrame:
    baseline = trials[trials.policy.eq(baseline_policy)][
        ["year", "trial", "roster", *METRICS]
    ]
    rows = []
    for policy in policies:
        candidate = trials[trials.policy.eq(policy)][
            ["year", "trial", "roster", *METRICS]
        ]
        paired = baseline.merge(
            candidate,
            on=["year", "trial"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        paired.insert(2, "policy", policy)
        paired["roster_changed"] = paired.roster_baseline.ne(
            paired.roster_candidate
        )
        for metric in METRICS:
            paired[f"{metric}_effect"] = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            )
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def effect_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    effect_columns = [f"{metric}_effect" for metric in METRICS]
    return (
        frame.groupby(groups, as_index=False)
        .agg(
            comparisons=("trial", "size"),
            roster_changed_rate=("roster_changed", "mean"),
            **{column: (column, "mean") for column in effect_columns},
        )
    )


def selection_tables(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for row in trials.itertuples():
        for player in str(row.identified_keeper_players).split("|"):
            if player:
                records.append(
                    {"year": row.year, "policy": row.policy, "player": player}
                )
    selections = pd.DataFrame(records)
    frequency = (
        selections.groupby(["year", "policy", "player"], as_index=False)
        .size()
        .rename(columns={"size": "selections"})
        .sort_values(
            ["year", "policy", "selections", "player"],
            ascending=[True, True, False, True],
        )
    )
    concentration_rows = []
    for (year, policy), frame in frequency.groupby(["year", "policy"]):
        counts = frame.selections.to_numpy(dtype=float)
        total = float(counts.sum())
        shares = counts / total
        concentration_rows.append(
            {
                "year": year,
                "policy": policy,
                "identified_slots": int(total),
                "unique_identified_players": int(len(counts)),
                "top1_selection_share": float(shares[:1].sum()),
                "top2_selection_share": float(shares[:2].sum()),
                "top3_selection_share": float(shares[:3].sum()),
                "selection_hhi": float(np.square(shares).sum()),
            }
        )
    concentration = pd.DataFrame(concentration_rows).sort_values(["year", "policy"])
    return frequency, concentration


def main() -> None:
    trials = pd.read_csv(RESULTS_DIR / "roster_trials.csv")
    if len(trials) != 7000:
        raise AssertionError(f"Expected 7,000 trial rows, found {len(trials):,}.")
    paired = paired_effects(trials)
    strength_paired = paired_effects(
        trials,
        baseline_policy="keeper_tiebreak",
        policies=("keeper_0p01", "keeper_1p0", "keeper_10p0"),
    )
    by_year = effect_summary(paired, ["year", "policy"])

    period = np.select(
        [paired.year.le(2023), paired.year.eq(2024), paired.year.eq(2025)],
        ["keeper_development_2022_2023", "keeper_temporal_2024", "keeper_unrealized_2025"],
        default="unclassified",
    )
    current_period = np.where(
        paired.year.eq(2025),
        "current_temporal_2025",
        "current_development_2022_2024",
    )
    by_period = pd.concat(
        [
            effect_summary(paired.assign(period=current_period), ["period", "policy"]),
            effect_summary(paired.assign(period=period), ["period", "policy"]),
        ],
        ignore_index=True,
    )

    origin_means = by_year.copy()
    effect_columns = [f"{metric}_effect" for metric in METRICS]
    across_rows = []
    for policy, frame in origin_means.groupby("policy"):
        row = {"policy": policy, "origins": int(frame.year.nunique())}
        for column in effect_columns:
            values = frame[column].dropna()
            row[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
            row[f"positive_origins_{column}"] = int((values > 0).sum())
            row[f"observed_origins_{column}"] = int(len(values))
        across_rows.append(row)
    across = pd.DataFrame(across_rows)

    frequency, concentration = selection_tables(trials)
    outputs = {
        "same_engine_paired_effects.csv": paired,
        "same_engine_effects_by_year.csv": by_year,
        "same_engine_effects_by_period.csv": by_period,
        "same_engine_effects_across_years.csv": across,
        "keeper_strength_effects_vs_tiebreak.csv": effect_summary(
            strength_paired, ["year", "policy"]
        ),
        "keeper_selection_frequency.csv": frequency,
        "keeper_selection_concentration.csv": concentration,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS_DIR / filename, index=False)


if __name__ == "__main__":
    main()
