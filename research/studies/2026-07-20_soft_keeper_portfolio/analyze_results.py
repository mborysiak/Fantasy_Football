"""Build durable diagnostics for the soft whole-bench keeper replay."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
RESULTS = STUDY_DIR / "results"
TRIALS = RESULTS / "roster_trials.csv"
PAIRED = RESULTS / "paired_effects.csv"
POLICIES = ("control", "soft_portfolio")
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
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join("" if pd.isna(value) else str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    trials = pd.read_csv(TRIALS)
    paired = pd.read_csv(PAIRED)
    expected_trials = 4 * 250 * len(POLICIES)
    if len(trials) != expected_trials or len(paired) != 4 * 250:
        raise AssertionError("The full replay row counts are incomplete.")
    if trials.duplicated(["year", "trial", "policy"]).any():
        raise AssertionError("Duplicate replay key.")
    if set(trials.policy) != set(POLICIES):
        raise AssertionError("Unexpected policy label.")
    soft = trials[trials.policy.eq("soft_portfolio")].copy()
    if soft.construction_mean_delta.min() < -1e-5:
        raise AssertionError("A construction-bank mean gate failed.")
    if soft.construction_p10_delta.min() < -1e-5:
        raise AssertionError("A construction-bank p10 gate failed.")
    if any(
        not split_players(row.search_anchor_players)
        <= split_players(row.bench_players)
        for row in soft.itertuples()
    ):
        raise AssertionError("A search anchor did not remain on the final bench.")

    effect_metrics = (
        "option_effective_count",
        "option_active_count_5pct",
        "option_positive_draw_rate",
        "bench_young_le2",
        "bench_young_le3",
        "bench_rookies",
        "bench_fillin_total",
        "bench_fillin_top2",
        "bench_fillin_second",
        "bench_positive_fillin_count",
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
        "predicted_probability_any_hit",
        "actual_best_keeper_surplus",
        "actual_any_keeper_hit_20",
        "actual_best_future_ppg",
    )
    changed = paired.roster_changed.astype(bool)
    effect_rows = []
    for metric in effect_metrics:
        effect_col = f"{metric}_effect"
        by_year = paired.groupby("year")[effect_col].mean()
        valid = paired[effect_col].dropna()
        valid_changed = paired.loc[changed, effect_col].dropna()
        effect_rows.append(
            {
                "metric": metric,
                "across_origin_effect": by_year.mean(),
                "min_origin_effect": by_year.min(),
                "max_origin_effect": by_year.max(),
                "positive_origins": int((by_year > 0).sum()),
                "origins_with_data": int(by_year.notna().sum()),
                "trial_median_effect": valid.median(),
                "changed_roster_nonnegative_rate": (
                    float((valid_changed >= 0).mean())
                    if len(valid_changed)
                    else np.nan
                ),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(RESULTS / "paired_metric_diagnostics.csv", index=False)

    addition_rows = []
    for year, frame in [("all", soft), *soft.groupby("year")]:
        counts = frame.accepted_option_additions.value_counts().to_dict()
        for additions in range(6):
            addition_rows.append(
                {
                    "year": year,
                    "accepted_option_additions": additions,
                    "rosters": int(counts.get(additions, 0)),
                    "roster_rate": counts.get(additions, 0) / len(frame),
                }
            )
    addition_distribution = pd.DataFrame(addition_rows)
    addition_distribution.to_csv(
        RESULTS / "option_addition_distribution.csv", index=False
    )

    protection_rows = []
    changed_paired = paired[changed].copy()
    for year, frame in [("all", changed_paired), *changed_paired.groupby("year")]:
        protection_rows.append(
            {
                "year": year,
                "changed_rosters": len(frame),
                "forecast_ev_effect": frame.forecast_ev_effect.mean(),
                "forecast_p10_effect": frame.forecast_p10_effect.mean(),
                "ev_nonnegative_rate": (frame.forecast_ev_effect >= 0).mean(),
                "p10_nonnegative_rate": (frame.forecast_p10_effect >= 0).mean(),
                "both_nonnegative_rate": (
                    (frame.forecast_ev_effect >= 0)
                    & (frame.forecast_p10_effect >= 0)
                ).mean(),
            }
        )
    protection = pd.DataFrame(protection_rows)
    protection.to_csv(RESULTS / "heldout_protection_audit.csv", index=False)

    concentration = (
        trials.groupby(["year", "policy"], as_index=False)
        .agg(
            rosters=("trial", "size"),
            effective_options=("option_effective_count", "mean"),
            active_options_5pct=("option_active_count_5pct", "mean"),
            positive_draw_rate=("option_positive_draw_rate", "mean"),
            young_le2=("bench_young_le2", "mean"),
            young_le3=("bench_young_le3", "mean"),
            fillin_top2=("bench_fillin_top2", "mean"),
        )
    )
    concentration.to_csv(
        RESULTS / "option_concentration_by_year.csv", index=False
    )

    anchor_rows = []
    for year, frame in soft.groupby("year"):
        counter: Counter[str] = Counter()
        for value in frame.search_anchor_players:
            counter.update(split_players(value))
        for player, count in counter.most_common():
            anchor_rows.append(
                {
                    "year": year,
                    "player": player,
                    "anchor_rosters": count,
                    "anchor_rate": count / len(frame),
                }
            )
    anchors = pd.DataFrame(anchor_rows)
    anchors.to_csv(RESULTS / "anchor_player_frequency.csv", index=False)

    named_rows = []
    for year in sorted(trials.year.unique()):
        origin = trials[trials.year.eq(year)]
        for player in NAMED_PLAYERS:
            values: dict[str, object] = {"year": year, "player": player}
            for policy in POLICIES:
                frame = origin[origin.policy.eq(policy)]
                values[f"{policy}_bench_rosters"] = int(
                    frame.bench_players.map(
                        lambda value: player in split_players(value)
                    ).sum()
                )
            soft_origin = origin[origin.policy.eq("soft_portfolio")]
            values["anchor_rosters"] = int(
                soft_origin.search_anchor_players.map(
                    lambda value: player in split_players(value)
                ).sum()
            )
            if any(
                values[column]
                for column in (
                    "control_bench_rosters",
                    "soft_portfolio_bench_rosters",
                    "anchor_rosters",
                )
            ):
                named_rows.append(values)
    named = pd.DataFrame(named_rows)
    named.to_csv(RESULTS / "named_player_audit.csv", index=False)

    policy_means = (
        trials.groupby(["year", "policy"], as_index=False)
        .agg(
            effective_options=("option_effective_count", "mean"),
            active_options=("option_active_count_5pct", "mean"),
            young_le2=("bench_young_le2", "mean"),
            fillin_top2=("bench_fillin_top2", "mean"),
            starter_spend=("starter_forecast_spend", "mean"),
            bench_spend=("bench_forecast_spend", "mean"),
            forecast_ev=("forecast_ev", "mean"),
            forecast_p10=("forecast_p10", "mean"),
            predicted_keeper_surplus=("predicted_expected_best_surplus", "mean"),
            actual_keeper_surplus=("actual_best_keeper_surplus", "mean"),
        )
    )
    across_policy = (
        policy_means.groupby("policy", as_index=False)
        .agg(
            **{
                column: (column, "mean")
                for column in policy_means.columns
                if column not in {"year", "policy"}
            }
        )
    )
    across_policy.to_csv(RESULTS / "decision_policy_means.csv", index=False)

    effect_lookup = effects.set_index("metric").across_origin_effect.to_dict()
    control = across_policy.set_index("policy").loc["control"]
    option = across_policy.set_index("policy").loc["soft_portfolio"]
    all_additions = addition_distribution[addition_distribution.year.eq("all")]
    zero_rate = float(
        all_additions.loc[
            all_additions.accepted_option_additions.eq(0), "roster_rate"
        ].iloc[0]
    )
    three_plus_rate = float(
        all_additions.loc[
            all_additions.accepted_option_additions.ge(3), "roster_rate"
        ].sum()
    )
    accepted_mean = soft.accepted_option_additions.mean()
    changed_rate = changed.mean()
    changed_both = protection.loc[
        protection.year.eq("all"), "both_nonnegative_rate"
    ].iloc[0]
    realized_years = int(
        paired.groupby("year").actual_best_keeper_surplus_effect.mean().notna().sum()
    )

    decision_effects = effects[
        effects.metric.isin(
            [
                "starter_forecast_spend",
                "bench_forecast_spend",
                "starter_forecast_ev",
                "starter_forecast_p10",
                "forecast_ev",
                "forecast_p10",
                "bench_fillin_top2",
                "actual_points",
                "actual_playoff_points",
                "actual_waiver_starts",
                "predicted_expected_best_surplus",
                "actual_best_keeper_surplus",
                "actual_best_future_ppg",
            ]
        )
    ][
        [
            "metric",
            "across_origin_effect",
            "min_origin_effect",
            "max_origin_effect",
            "positive_origins",
            "origins_with_data",
        ]
    ]
    named_display = named.copy()
    lines = [
        "# Decision Readout: Soft Whole-Bench Keeper Portfolio",
        "",
        "## Decision",
        "",
        "The expected-best whole-bench objective is promising and is preferable ",
        "to hard k1/k2/k3 keeper counts, but the tested 50-context construction ",
        "gate is not stable enough to promote unchanged. Keep the one-year, ",
        "validation-residual keeper objective; strengthen the current-year ",
        "protection before production use.",
        "",
        f"The soft search changed {changed_rate:.1%} of rosters and accepted ",
        f"{accepted_mean:.2f} incremental search anchors per roster. It accepted ",
        f"none in {zero_rate:.1%} and three or more in only {three_plus_rate:.1%}. ",
        "Those anchors are an implementation trace, not a count of final lottery ",
        "tickets: every final bench player receives both current and option value.",
        "",
        "## The Natural Bench Is Broader Than Three Options",
        "",
        f"The control already averaged {control.effective_options:.2f} effective ",
        f"options and {control.active_options:.2f} players with at least a 5% ",
        f"chance of being the draw-level portfolio winner. The soft policy averaged ",
        f"{option.effective_options:.2f} and {option.active_options:.2f}. Thus this ",
        "objective does not naturally label two fill-ins and three tickets. It ",
        "usually treats almost the entire bench as having some option value, while ",
        "the winner-share concentration captures how many distinct bets matter.",
        "",
        "## Across-Origin Effects",
        "",
        markdown_table(decision_effects.round(3)),
        "",
        f"Predicted expected-best keeper surplus improved by ",
        f"${effect_lookup['predicted_expected_best_surplus']:.1f}. Realized best ",
        f"one-year surplus improved by ${effect_lookup['actual_best_keeper_surplus']:.1f} ",
        f"across all {realized_years} origins with realized next-season outcomes. ",
        "The hit-rate metric was already near saturation in the control, so the ",
        "useful gain is the size of the best hit rather than merely finding any hit.",
        "",
        f"On average, ${effect_lookup['starter_forecast_spend']:.1f} moved to ",
        f"starters and ${-effect_lookup['bench_forecast_spend']:.1f} moved out of ",
        f"the bench. Independent whole-roster mean/p10 changed by ",
        f"{effect_lookup['forecast_ev']:.1f}/{effect_lookup['forecast_p10']:.1f}; ",
        f"top-two fill-in value changed by {effect_lookup['bench_fillin_top2']:.1f}. ",
        "This is consistent with the desired studs-and-scrubs direction without ",
        "requiring a veteran/young-player role assignment.",
        "",
        "## Current-Year Protection Caveat",
        "",
        f"All soft rosters passed mean and p10 on the 50-context construction ",
        f"gate. On the separate 250-context evaluation bank, however, only ",
        f"{changed_both:.1%} of changed rosters were nonnegative on both metrics. ",
        "The average effect remained positive because improvements were larger ",
        "than losses, but 2024 was negative on both mean and p10. The gate therefore ",
        "works as an objective constraint, not yet as a dependable out-of-sample ",
        "no-harm guarantee.",
        "",
        markdown_table(protection.round(3)),
        "",
        "## Named Player Audit",
        "",
        markdown_table(named_display),
        "",
        "The control already found most of the intended examples: Kenneth Walker, ",
        "Achane, Flowers, and Chase Brown were frequently present before the soft ",
        "search. The policy added exposure to Rachaad White, Bucky Irving, and ",
        "Achane, while sometimes replacing Flowers. This is portfolio behavior, ",
        "not a simple young-player bonus: a player is valuable when their future ",
        "surplus improves scenarios not already covered by the other four players.",
        "",
        "## Boundaries",
        "",
        f"- Only {realized_years} origins have realized next-season keeper outcomes.",
        "- Salary trials within an origin share projections and realized player outcomes; ",
        "  they are sensitivity draws, not 250 independent historical seasons.",
        "- The search is greedy with a six-candidate shortlist and accumulated anchors.",
        "- Winner shares depend on the calibrated next-year residual draw distribution; ",
        "  very diffuse residual uncertainty can make nearly every bench player appear ",
        "  to have some option value.",
        "- Exact in-season drop/claim timing remains outside this draft replay. Waiver ",
        "  baselines are included in managed-lineup scoring.",
        "",
        "## Recommended Next Test",
        "",
        "Retain this expected-best objective and no-count/no-age-quota formulation. ",
        "Re-run a focused sensitivity with all 250 construction contexts (or ",
        "cross-fitted lower-confidence mean/p10 constraints) and a minimum material ",
        "keeper-utility improvement. Compare the current 50-context gate against the ",
        "stronger gate on held-out mean/p10, keeper surplus, and search-addition ",
        "frequency. Do not add a production bench bonus until that protection is ",
        "stable, especially in the 2024 origin.",
        "",
    ]
    (RESULTS / "decision_readout.md").write_text(
        "\n".join(line.rstrip() for line in lines), encoding="utf-8"
    )
    print("soft keeper portfolio analysis: PASS")


if __name__ == "__main__":
    main()
