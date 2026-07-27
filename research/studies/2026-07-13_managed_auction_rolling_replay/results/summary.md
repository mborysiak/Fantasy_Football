# Managed Auction Rolling-Origin Replay Results

Run: 250 paired trials per cell, 250 prior-only construction contexts plus an independently seeded evaluation bank per origin, seed 20260713.

This is a replay of empty-roster Target/look-ahead construction. It is not an end-to-end Current Nomination replay because historical nomination order and auction-state logs do not exist.

## Salary calibration

| year | salary_draw_count | coverage_80 | coverage_90 | mean_forecast_sd | mean_crps | mae |
|---|---|---|---|---|---|---|
| 2022 | 1 | 0.769 | 0.876 | 7.708 | 4.445 | 6.140 |
| 2022 | 5 | 0.372 | 0.463 | 3.468 | 4.768 | 6.218 |
| 2023 | 1 | 0.763 | 0.840 | 4.235 | 2.742 | 3.719 |
| 2023 | 5 | 0.359 | 0.481 | 1.911 | 2.991 | 3.731 |
| 2024 | 1 | 0.455 | 0.545 | 2.719 | 3.821 | 5.017 |
| 2024 | 5 | 0.235 | 0.295 | 1.220 | 4.398 | 5.023 |
| 2025 | 1 | 0.627 | 0.709 | 3.124 | 2.944 | 3.911 |
| 2025 | 5 | 0.284 | 0.328 | 1.408 | 3.306 | 3.900 |

Average-five / one-draw forecast-SD ratios by year: 2022: 0.450, 2023: 0.451, 2024: 0.449, 2025: 0.451.

## One-at-a-time changes from the current app profile

Positive point effects favor the candidate setting named in the factor; positive feasibility effects mean more rosters fit the realized $298 cap. The unqualified point effect scores every selected roster even when its realized price exceeded the cap. The joint-feasible point effect uses only pairs where both settings fit; `mean_feasible_pair_share` reports that coverage. Each candidate changes exactly one setting from 5 salary draws, Top-N on, projected waivers, and bench weight 0.25.

| factor | mean_actual_points_effect | mean_joint_feasible_actual_points_effect | development_2022_2024_actual_points_effect | temporal_check_2025_actual_points_effect | mean_actual_cap_feasible_effect | mean_feasible_pair_share | mean_absolute_forecast_error_effect | mean_roster_changed_share |
|---|---|---|---|---|---|---|---|---|
| bench_0_minus_025 | -3.410 | -0.230 | -6.784 | 6.711 | -0.013 | 0.112 | 3.879 | 0.862 |
| prior_waiver_minus_projected | -22.509 | -20.207 | -27.887 | -6.375 | 0.060 | 0.119 | 16.969 | 0.993 |
| salary_draws_1_minus_5 | 29.002 | -21.021 | 35.138 | 10.593 | -0.137 | 0.040 | -5.086 | 0.990 |
| top_n_off_minus_on | -4.887 | -3.379 | 2.709 | -27.675 | 0.013 | 0.167 | 1.032 | 0.338 |

## Factorial marginal effects

These candidate-minus-default effects average over all eight combinations of the other settings. They are a robustness view, not a direct replay of a one-knob change from the current profile.

| factor | mean_actual_points_effect | mean_joint_feasible_actual_points_effect | development_2022_2024_actual_points_effect | temporal_check_2025_actual_points_effect | mean_actual_cap_feasible_effect | mean_feasible_pair_share | mean_absolute_forecast_error_effect | mean_roster_changed_share |
|---|---|---|---|---|---|---|---|---|
| bench_0_minus_025 | -0.763 | 4.251 | -3.936 | 8.756 | -0.006 | 0.089 | 3.912 | 0.895 |
| prior_waiver_minus_projected | -21.858 | -26.783 | -26.391 | -8.258 | 0.048 | 0.079 | 14.867 | 0.993 |
| salary_draws_1_minus_5 | 30.521 | -5.572 | 36.881 | 11.440 | -0.148 | 0.053 | -9.476 | 0.993 |
| top_n_off_minus_on | -5.334 | -3.504 | 2.876 | -29.962 | 0.016 | 0.127 | 0.239 | 0.293 |

## Highest cap-feasibility variants

Variants are ordered first by realized cap-feasible rate. Point totals are the equal-season average of each year's conditional mean among trials that fit the realized cap; `feasible_trials` is the pooled count. These should not be compared without the feasibility rate and count.

| variant | actual_points_feasible | cap_feasible_rate | feasible_trials | drafted_only_points | absolute_forecast_error | hindsight_heuristic_gap |
|---|---|---|---|---|---|---|
| d5_top0_waiverprior_empirical_bench25 | 1551.64 | 0.28 | 278 | 1437.34 | 141.38 | 593.08 |
| d5_top0_waiverprior_empirical_bench00 | 1541.82 | 0.26 | 260 | 1436.54 | 148.77 | 602.89 |
| d5_top1_waiverprior_empirical_bench25 | 1558.66 | 0.26 | 255 | 1443.80 | 141.44 | 586.06 |
| d5_top1_waiverprior_empirical_bench00 | 1547.84 | 0.25 | 247 | 1439.61 | 147.98 | 596.88 |
| d5_top0_waivercurrent_projected_bench25 | 1575.79 | 0.21 | 208 | 1477.68 | 125.51 | 568.92 |

## Join and survivorship audit

| year | forecast_players | salary_forecast_matches | actual_salary_matches | raw_outcome_matches | excluded_keepers |
|---|---|---|---|---|---|
| 2022 | 274 | 129 | 140 | 263 | 19 |
| 2023 | 303 | 166 | 152 | 284 | 21 |
| 2024 | 246 | 178 | 150 | 243 | 18 |
| 2025 | 180 | 165 | 149 | 176 | 15 |

Raw FastR weekly rows, not the survivorship-filtered target template table, supply realized scores and played evidence. Construction donors are capped at origin year minus one and use preseason consensus features only.

## Interpretation limits

- Four seasons are four independent outcome units; Monte Carlo trials measure simulation stability, not additional seasons.
- Recorded keepers are removed from this empty-roster replay. Their recorded prices remain deterministic, but no historical owner mapping exists for a specific keeper-choice replay.
- Historical final auction prices are treated as exogenous. The replay cannot model how a different roster, nomination order, or bidding path would change those prices, so realized-cap feasibility is diagnostic rather than causal.
- The common realized waiver stream is intentionally optimistic: ranking is causal, but eligibility is hindsight availability-filtered using target-week played evidence. It also omits opponent competition, transaction limits, and roster persistence. Because it is shared, paired differences remain useful.
- The hindsight roster is an approximation using the current marginal objective plus one-swap refinement, not a global nonlinear season optimum. The reported gap is therefore not guaranteed to be nonnegative and is not labeled regret.
- The shared one-swap refinement averages weekly scores across contexts but ORs their played masks. That mirrors the app, but can treat missed-game probability as a played zero and understate the value of depth or waiver cover.
- Frozen files precede target result imports, but exact first-nomination timestamps were not recorded and preseason feature revisions are not fully independently timestamped.
