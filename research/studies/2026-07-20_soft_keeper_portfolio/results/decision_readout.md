# Decision Readout: Soft Whole-Bench Keeper Portfolio

## Decision

The expected-best whole-bench objective is promising and is preferable
to hard k1/k2/k3 keeper counts, but the tested 50-context construction
gate is not stable enough to promote unchanged. Keep the one-year,
validation-residual keeper objective; strengthen the current-year
protection before production use.

The soft search changed 50.6% of rosters and accepted
0.98 incremental search anchors per roster. It accepted
none in 49.4% and three or more in only 12.9%.
Those anchors are an implementation trace, not a count of final lottery
tickets: every final bench player receives both current and option value.

## The Natural Bench Is Broader Than Three Options

The control already averaged 4.07 effective
options and 4.62 players with at least a 5%
chance of being the draw-level portfolio winner. The soft policy averaged
4.19 and 4.69. Thus this
objective does not naturally label two fill-ins and three tickets. It
usually treats almost the entire bench as having some option value, while
the winner-share concentration captures how many distinct bets matter.

## Across-Origin Effects

| metric | across_origin_effect | min_origin_effect | max_origin_effect | positive_origins | origins_with_data |
| --- | --- | --- | --- | --- | --- |
| bench_fillin_top2 | 0.658 | -0.018 | 1.511 | 3 | 4 |
| starter_forecast_spend | 2.081 | -1.55 | 5.997 | 2 | 4 |
| bench_forecast_spend | -1.878 | -6.157 | 1.849 | 2 | 4 |
| starter_forecast_ev | 4.406 | 0.563 | 10.662 | 4 | 4 |
| starter_forecast_p10 | 2.455 | 0.131 | 5.239 | 4 | 4 |
| forecast_ev | 4.167 | -2.185 | 15.038 | 3 | 4 |
| forecast_p10 | 2.667 | -1.369 | 10.349 | 2 | 4 |
| actual_points | 4.524 | -8.61 | 14.763 | 2 | 4 |
| actual_playoff_points | -0.611 | -8.909 | 3.689 | 2 | 4 |
| actual_waiver_starts | 0.26 | -0.744 | 1.12 | 2 | 4 |
| predicted_expected_best_surplus | 4.268 | 2.078 | 6.09 | 4 | 4 |
| actual_best_keeper_surplus | 4.413 | 2.811 | 6.831 | 3 | 3 |
| actual_best_future_ppg | 0.211 | 0.071 | 0.383 | 3 | 3 |

Predicted expected-best keeper surplus improved by
$4.3. Realized best
one-year surplus improved by $4.4
across all 3 origins with realized next-season outcomes.
The hit-rate metric was already near saturation in the control, so the
useful gain is the size of the best hit rather than merely finding any hit.

On average, $2.1 moved to
starters and $1.9 moved out of
the bench. Independent whole-roster mean/p10 changed by
4.2/2.7;
top-two fill-in value changed by 0.7.
This is consistent with the desired studs-and-scrubs direction without
requiring a veteran/young-player role assignment.

## Current-Year Protection Caveat

All soft rosters passed mean and p10 on the 50-context construction
gate. On the separate 250-context evaluation bank, however, only
46.2% of changed rosters were nonnegative on both metrics.
The average effect remained positive because improvements were larger
than losses, but 2024 was negative on both mean and p10. The gate therefore
works as an objective constraint, not yet as a dependable out-of-sample
no-harm guarantee.

| year | changed_rosters | forecast_ev_effect | forecast_p10_effect | ev_nonnegative_rate | p10_nonnegative_rate | both_nonnegative_rate |
| --- | --- | --- | --- | --- | --- | --- |
| all | 506 | 8.236 | 5.271 | 0.628 | 0.571 | 0.462 |
| 2022 | 133 | 1.36 | -2.542 | 0.549 | 0.398 | 0.308 |
| 2023 | 153 | 24.573 | 16.911 | 0.908 | 0.824 | 0.784 |
| 2024 | 112 | -4.876 | -3.057 | 0.232 | 0.393 | 0.125 |
| 2025 | 108 | 7.158 | 7.041 | 0.741 | 0.611 | 0.546 |

## Named Player Audit

| year | player | control_bench_rosters | soft_portfolio_bench_rosters | anchor_rosters |
| --- | --- | --- | --- | --- |
| 2022 | Kenneth Walker | 249 | 249 | 0 |
| 2022 | Rachaad White | 0 | 19 | 19 |
| 2023 | Devon Achane | 211 | 242 | 19 |
| 2023 | Zay Flowers | 152 | 115 | 6 |
| 2024 | Zay Flowers | 1 | 0 | 0 |
| 2024 | Chase Brown | 226 | 239 | 0 |
| 2024 | Bucky Irving | 0 | 18 | 18 |
| 2025 | Cam Skattebo | 1 | 1 | 0 |

The control already found most of the intended examples: Kenneth Walker,
Achane, Flowers, and Chase Brown were frequently present before the soft
search. The policy added exposure to Rachaad White, Bucky Irving, and
Achane, while sometimes replacing Flowers. This is portfolio behavior,
not a simple young-player bonus: a player is valuable when their future
surplus improves scenarios not already covered by the other four players.

## Boundaries

- Only 3 origins have realized next-season keeper outcomes.
- Salary trials within an origin share projections and realized player outcomes;
  they are sensitivity draws, not 250 independent historical seasons.
- The search is greedy with a six-candidate shortlist and accumulated anchors.
- Winner shares depend on the calibrated next-year residual draw distribution;
  very diffuse residual uncertainty can make nearly every bench player appear
  to have some option value.
- Exact in-season drop/claim timing remains outside this draft replay. Waiver
  baselines are included in managed-lineup scoring.

## Recommended Next Test

Retain this expected-best objective and no-count/no-age-quota formulation.
Re-run a focused sensitivity with all 250 construction contexts (or
cross-fitted lower-confidence mean/p10 constraints) and a minimum material
keeper-utility improvement. Compare the current 50-context gate against the
stronger gate on held-out mean/p10, keeper surplus, and search-addition
frequency. Do not add a production bench bonus until that protection is
stable, especially in the 2024 origin.
