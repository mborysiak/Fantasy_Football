"""Create player-level bench, prediction, and realized-hit frequency audits."""

from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"
BASELINE = "control"


def explode_frequency(
    trials: pd.DataFrame,
    column: str,
    output_name: str,
) -> pd.DataFrame:
    records = []
    for row in trials.itertuples():
        raw = getattr(row, column)
        for player in str(raw).split("|"):
            if player and player.lower() != "nan":
                records.append(
                    {"year": row.year, "policy": row.policy, "player": player}
                )
    return (
        pd.DataFrame(records)
        .groupby(["year", "policy", "player"], as_index=False)
        .size()
        .rename(columns={"size": output_name})
    )


def main() -> None:
    trials = pd.read_csv(RESULTS_DIR / "roster_trials.csv")
    if len(trials) != 3000:
        raise AssertionError(f"Expected 3,000 rows, found {len(trials):,}.")

    bench_frequency = explode_frequency(trials, "bench_players", "bench_rosters")
    predicted_frequency = explode_frequency(
        trials, "predicted_best_player", "predicted_best_rosters"
    )
    hit_frequency = explode_frequency(
        trials, "actual_keeper_hit_players", "realized_hit_rosters"
    )
    frequency = (
        bench_frequency.merge(
            predicted_frequency,
            on=["year", "policy", "player"],
            how="outer",
        )
        .merge(
            hit_frequency,
            on=["year", "policy", "player"],
            how="outer",
        )
        .fillna(0)
    )
    for column in ["bench_rosters", "predicted_best_rosters", "realized_hit_rosters"]:
        frequency[column] = frequency[column].astype(int)

    baseline = frequency[frequency.policy.eq(BASELINE)].drop(columns="policy")
    effects = frequency[~frequency.policy.eq(BASELINE)].merge(
        baseline,
        on=["year", "player"],
        how="outer",
        suffixes=("_candidate", "_control"),
    )
    effects["policy"] = effects.policy.fillna("not_selected_by_candidate")
    for column in ["bench_rosters", "predicted_best_rosters", "realized_hit_rosters"]:
        candidate = effects[f"{column}_candidate"].fillna(0).astype(int)
        control = effects[f"{column}_control"].fillna(0).astype(int)
        effects[f"{column}_effect"] = candidate - control
    effects = effects.sort_values(
        ["year", "policy", "bench_rosters_effect", "player"],
        ascending=[True, True, False, True],
    )

    named = effects[
        effects.player.isin(
            [
                "Devon Achane",
                "Zay Flowers",
                "Chase Brown",
                "Bucky Irving",
                "Cam Skattebo",
                "Kenneth Walker",
                "Rachaad White",
            ]
        )
    ].copy()
    frequency.to_csv(RESULTS_DIR / "player_frequency.csv", index=False)
    effects.to_csv(RESULTS_DIR / "player_frequency_effects.csv", index=False)
    named.to_csv(RESULTS_DIR / "named_player_audit.csv", index=False)


if __name__ == "__main__":
    main()
