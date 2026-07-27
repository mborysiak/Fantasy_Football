"""Build durable tradeoff and player-audit outputs for the reinvestment replay."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"
TRIALS = RESULTS / "roster_trials.csv"
RUNNER = STUDY_DIR / "run_replay.py"
POLICIES = ("reinvest_k1", "reinvest_k2", "reinvest_k3")
NAMED_PLAYERS = (
    "Kenneth Walker",
    "Rachaad White",
    "Devon Achane",
    "Zay Flowers",
    "Chase Brown",
    "Bucky Irving",
    "Cam Skattebo",
)


def split_players(value: object) -> set[str]:
    if pd.isna(value) or not str(value):
        return set()
    return {player for player in str(value).split("|") if player}


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "keeper_reinvestment_composition", RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load replay module: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trials = pd.read_csv(TRIALS)
    expected = 4 * 250 * 4
    if len(trials) != expected:
        raise AssertionError(f"Expected {expected} trial rows, found {len(trials)}.")
    if trials.duplicated(["year", "trial", "policy"]).any():
        raise AssertionError("Duplicate year/trial/policy key.")
    if trials.current_construction_delta.min() < -1e-5:
        raise AssertionError("A roster violated the construction no-loss gate.")
    if trials.forecast_salary_spend.max() > 298.0 + 1e-8:
        raise AssertionError("A roster exceeded the forecast salary cap.")
    if any(
        not split_players(row.forced_option_players)
        <= split_players(row.bench_players)
        for row in trials.itertuples()
    ):
        raise AssertionError("A forced option was not retained on the bench.")

    forced_wide = trials.pivot(
        index=["year", "trial"], columns="policy", values="forced_option_players"
    )
    nested_violations = sum(
        not (
            split_players(row.reinvest_k1)
            <= split_players(row.reinvest_k2)
            <= split_players(row.reinvest_k3)
        )
        for row in forced_wide.itertuples()
    )
    if nested_violations:
        raise AssertionError(f"Found {nested_violations} non-nested option portfolios.")

    control = trials[trials.policy.eq("control")].set_index(["year", "trial"])
    effect_metrics = (
        "forecast_salary_spend",
        "unspent_budget",
        "starter_forecast_spend",
        "bench_forecast_spend",
        "starter_forecast_ev",
        "starter_forecast_p10",
        "forecast_ev",
        "forecast_p10",
        "actual_points",
        "actual_playoff_points",
        "actual_waiver_starts",
        "predicted_expected_best_surplus",
        "actual_best_keeper_surplus",
        "actual_any_keeper_hit_20",
        "actual_best_future_ppg",
    )
    tradeoff_rows = []
    by_year_rows = []
    for policy in POLICIES:
        candidate = trials[trials.policy.eq(policy)].set_index(["year", "trial"])
        effects = candidate[list(effect_metrics)] - control[list(effect_metrics)]
        year_effects = effects.groupby(level="year").mean()
        for year, row in year_effects.iterrows():
            by_year_rows.append(
                {
                    "year": year,
                    "policy": policy,
                    "forced_options": candidate.loc[year].forced_option_count.mean(),
                    "starter_changes": candidate.loc[
                        year
                    ].starter_changes_vs_control.mean(),
                    **{f"{metric}_effect": row[metric] for metric in effect_metrics},
                }
            )
        counts = candidate.forced_option_count.value_counts(normalize=True)
        forced_total = float(candidate.forced_option_count.sum())
        tradeoff_rows.append(
            {
                "policy": policy,
                "mean_forced_options": candidate.forced_option_count.mean(),
                "cap_filled_rate": float(
                    counts.get(int(policy[-1]), 0.0)
                ),
                "forced_young_share": float(
                    candidate.forced_young_count.sum() / max(forced_total, 1.0)
                ),
                "starter_changes": candidate.starter_changes_vs_control.mean(),
                **{
                    f"{metric}_effect": year_effects[metric].mean()
                    for metric in effect_metrics
                },
            }
        )
    tradeoff = pd.DataFrame(tradeoff_rows)
    by_year = pd.DataFrame(by_year_rows)
    tradeoff.to_csv(RESULTS / "policy_tradeoff_across_years.csv", index=False)
    by_year.to_csv(RESULTS / "policy_tradeoff_by_year.csv", index=False)

    frequency_rows = []
    for (year, policy), frame in trials[
        trials.policy.isin(POLICIES)
    ].groupby(["year", "policy"]):
        counts: Counter[str] = Counter()
        for value in frame.forced_option_players:
            counts.update(split_players(value))
        for player, count in counts.most_common():
            frequency_rows.append(
                {
                    "year": year,
                    "policy": policy,
                    "player": player,
                    "forced_rosters": count,
                    "forced_roster_rate": count / len(frame),
                }
            )
    frequency = pd.DataFrame(frequency_rows)
    frequency.to_csv(RESULTS / "forced_player_frequency.csv", index=False)

    named_rows = []
    for year in sorted(trials.year.unique()):
        for player in NAMED_PLAYERS:
            values: dict[str, object] = {"year": year, "player": player}
            any_selected = False
            for policy in ("control", *POLICIES):
                frame = trials[
                    trials.year.eq(year) & trials.policy.eq(policy)
                ]
                bench_count = int(
                    frame.bench_players.map(
                        lambda value: player in split_players(value)
                    ).sum()
                )
                forced_count = int(
                    frame.forced_option_players.map(
                        lambda value: player in split_players(value)
                    ).sum()
                )
                values[f"{policy}_bench_rosters"] = bench_count
                values[f"{policy}_forced_rosters"] = forced_count
                any_selected |= bool(bench_count or forced_count)
            if any_selected:
                named_rows.append(values)
    named = pd.DataFrame(named_rows)
    named.to_csv(RESULTS / "named_player_audit.csv", index=False)

    runner = load_runner()
    features = runner.base.load_feature_templates()
    composition_rows = []
    for year in sorted(trials.year.unique()):
        target = features[features.season.eq(year)].copy()
        exp_lookup = (
            target.sort_values("preseason_proj_ppg", ascending=False)
            .drop_duplicates("player")
            .set_index("player")
            .year_exp
            .to_dict()
        )
        for row in trials[trials.year.eq(year)].itertuples():
            experience = np.array(
                [
                    float(exp_lookup.get(player, 99.0))
                    for player in split_players(row.bench_players)
                ]
            )
            composition_rows.append(
                {
                    "year": year,
                    "trial": row.trial,
                    "policy": row.policy,
                    "young_le2": int(np.sum(experience <= 2)),
                    "young_le3": int(np.sum(experience <= 3)),
                    "rookie": int(np.sum(experience <= 0)),
                    "veteran_gt3": int(np.sum(experience > 3)),
                    "missing_experience": int(np.sum(experience >= 99)),
                }
            )
    composition = pd.DataFrame(composition_rows)
    composition_by_year = (
        composition.groupby(["year", "policy"], as_index=False)[
            [
                "young_le2",
                "young_le3",
                "rookie",
                "veteran_gt3",
                "missing_experience",
            ]
        ]
        .mean()
    )
    composition_by_year.to_csv(
        RESULTS / "bench_composition_by_year.csv", index=False
    )
    composition_across = (
        composition_by_year.groupby("policy", as_index=False)[
            ["young_le2", "young_le3", "rookie", "veteran_gt3"]
        ]
        .mean()
    )

    effect_display = tradeoff[
        [
            "policy",
            "mean_forced_options",
            "cap_filled_rate",
            "forced_young_share",
            "starter_changes",
            "starter_forecast_spend_effect",
            "bench_forecast_spend_effect",
            "starter_forecast_ev_effect",
            "starter_forecast_p10_effect",
            "forecast_ev_effect",
            "forecast_p10_effect",
            "actual_points_effect",
            "actual_playoff_points_effect",
            "actual_waiver_starts_effect",
            "actual_best_keeper_surplus_effect",
        ]
    ].copy()
    effect_display["cap_filled_rate"] *= 100
    effect_display["forced_young_share"] *= 100

    player_display = frequency[
        frequency.groupby(["year", "policy"]).cumcount().lt(5)
    ].copy()
    player_display["forced_roster_rate"] *= 100

    lines = [
        "# Decision Readout: Full-Roster Keeper Reinvestment",
        "",
        "## Decision",
        "",
        "The budget-transfer mechanism is real, but a broad full-roster keeper ",
        "bonus is not ready for promotion. The k1/k2/k3 labels count incremental ",
        "forced additions, not mutually exclusive bench roles or total lottery ",
        "tickets. K1 is the safest tested forced-addition policy, but the next model ",
        "should use soft portfolio tradeoffs rather than a hard option count.",
        "",
        "The one-option policy moved `$5.7` from the five-player bench to the ",
        "starting core while total spend increased only `$0.1`. Starter-only ",
        "forecast mean/p10 improved `4.6`/`3.6`, proving that cheap options can ",
        "finance stronger starters. Full-roster mean/p10 improved only `0.6`/`1.2`, ",
        "because the sacrificed bench depth offsets most of the starter gain.",
        "",
        "## Across-Origin Tradeoff",
        "",
        markdown_table(effect_display.round(3)),
        "",
        "The second forced option bought only about `$1.0` of additional realized ",
        "best keeper surplus versus k1 while reducing full-roster p10 by about `1.1` ",
        "points and adding about `0.4` waiver starts. The third option was dominated: ",
        "it raised waiver use, reduced p10 and playoff scoring, and erased the ",
        "incremental realized keeper surplus because the 2024 portfolio failed.",
        "",
        "## Total Bench Composition",
        "",
        markdown_table(composition_across.round(3)),
        "",
        "The forced-option count is incremental to the current optimizer's bench. ",
        "Using at most two years of experience as a descriptive upside proxy, the ",
        "control already averaged `3.34` young players among five bench slots. K1 ",
        "raised that to `3.71`, k2 to `3.96`, and k3 to `4.10`. Thus k3 generally ",
        "tested a much more youth-heavy bench, not a clean test of a flexible ",
        "two-fill-in/two-to-three-ticket preference. Youth is not a role: a young ",
        "player can provide both startable current depth and keeper upside.",
        "",
        "## Why Incremental Keeper Value Was Small",
        "",
        "The rebuilt current-only control already put Kenneth Walker on 249/250 ",
        "2022 benches, Achane on 211/250 and Flowers on 152/250 2023 benches, and ",
        "Chase Brown on 226/250 2024 benches. The option policy therefore had little ",
        "room to improve hit probability; it mostly added further bets. K1 did add ",
        "Rachaad White to 15 rosters and Bucky Irving to 13, but its most common ",
        "forced names were Dameon Pierce, Zach Charbonnet, Trey Benson, and Ray Davis.",
        "",
        "Across all forced selections, 97.5% of k1 options and 96.8% of k2 options ",
        "were players with at most two years of experience. The signal is selecting ",
        "the intended young profile; the problem is incremental calibration and ",
        "bench opportunity cost, not a failure to target youth.",
        "",
        "## Most Common Forced Options",
        "",
        markdown_table(player_display.round(3)),
        "",
        "## Selection Frequency",
        "",
        "K1 accepted one new option in 82.8% of rosters. K2 selected two in 66.7%, ",
        "one in 16.1%, and none in 17.2%. K3 reached all three in only 43.6%; its ",
        "mean was 1.93 forced options. The gates therefore stop some additions, but ",
        "the third slot still activates often enough to cause material depth damage.",
        "",
        "## Boundaries",
        "",
        "- There are only three realized next-season keeper origins.",
        "- The greedy eight-candidate shortlist is an explicit search approximation.",
        "- The selection gate uses the full cached-bank expected profile; independent ",
        "  mean/p10 are evaluation outcomes and were not reused for selection.",
        "- The rebuilt control differs from the preceding five-context control and is ",
        "  intentionally the fair same-engine comparator for full-roster reoptimization.",
        "- This remains a frozen historical salary replay, not the final current ",
        "  v5-plus-selection-reserve production test.",
        "",
        "## Recommended Next Step",
        "",
        "Do not add a generic multi-slot keeper bonus to production. Next give every ",
        "bench player both a current fill-in value and a keeper-option value; the same ",
        "player may contribute to both. Optimize expected-best keeper surplus across ",
        "the whole bench only after roster mean/p10 and aggregate bench coverage are ",
        "protected, without age quotas or designated slot counts. Allow the resulting ",
        "number of lottery profiles to emerge. The waiver baseline can cap lineup ",
        "downside while causal drop/claim timing remains a separate enhancement.",
        "",
    ]
    (RESULTS / "decision_readout.md").write_text(
        "\n".join(line.rstrip() for line in lines), encoding="utf-8"
    )
    print("keeper reinvestment analysis: PASS")


if __name__ == "__main__":
    main()
