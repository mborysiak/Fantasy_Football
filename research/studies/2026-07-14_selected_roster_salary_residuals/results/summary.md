# Selected-Roster Salary Residual Diagnostic

Residuals use 518 recorded player-origin prices from 930 auctionable candidates. Recorded-price coverage is 96.0% of the 52,000 selected roster slots.

## Core cohort comparison

| cohort | player_origins | weighted_observations | mean_salary_residual | positive_residual_rate | mean_selection_rate | mean_scenario_center_shift | mean_actual_minus_scenario |
|---|---|---|---|---|---|---|---|
| all_observed_auctionable | 518 | 518 | -0.388 | 0.400 | 0.096 | -0.461 | 0.073 |
| ever_selected_unique | 477 | 477 | -0.398 | 0.407 | 0.105 | -0.562 | 0.164 |
| never_selected_unique | 41 | 41 | -0.274 | 0.317 | 0.000 | 0.712 | -0.986 |
| top_projection_quartile_ever_selected | 220 | 220 | -0.987 | 0.432 | 0.140 | -1.266 | 0.279 |
| top_projection_quartile_never_selected | 2 | 2 | -11.720 | 0.000 | 0.000 | 1.343 | -13.063 |
| top_projection_quartile_rare_le_5pct | 56 | 56 | -4.642 | 0.286 | 0.022 | -0.501 | -4.142 |
| top_projection_quartile_frequent_ge_25pct | 38 | 38 | 2.487 | 0.605 | 0.351 | -1.663 | 4.149 |
| selected_roster_slots_weighted | 477 | 49920 | 1.427 | 0.524 | 0.225 | -1.011 | 2.439 |

## Selection-frequency gradient

| cohort | player_origins | mean_salary_residual | positive_residual_rate | mean_selection_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| never | 41 | -0.274 | 0.317 | 0.000 | 0.712 |
| rare_0-5% | 203 | -1.576 | 0.320 | 0.021 | 0.053 |
| occasional_5-25% | 221 | -0.118 | 0.434 | 0.121 | -0.971 |
| frequent_25-50% | 50 | 2.384 | 0.600 | 0.342 | -1.148 |
| core_>50% | 3 | 12.316 | 1.000 | 0.614 | -2.308 |

## High-projection players by selection frequency

| cohort | player_origins | mean_salary_residual | positive_residual_rate | mean_selection_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| selected_<=5% | 56 | -4.642 | 0.286 | 0.022 | -0.501 |
| selected_5-25% | 130 | -0.697 | 0.431 | 0.129 | -1.464 |
| selected_25-50% | 34 | 2.393 | 0.618 | 0.343 | -1.560 |
| selected_>50% | 2 | 14.337 | 1.000 | 0.597 | -2.142 |

## Selected roster slots by position

| pos | selected_slots | player_origins | mean_salary_residual | positive_residual_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| QB | 3869 | 57 | -1.099 | 0.318 | -0.448 |
| RB | 14962 | 164 | 2.708 | 0.575 | 0.858 |
| TE | 6768 | 52 | 0.517 | 0.465 | -0.454 |
| WR | 24321 | 204 | 1.295 | 0.542 | -2.406 |

## Selected roster slots by predicted-price tier

| predicted_salary_tier | selected_slots | player_origins | mean_salary_residual | positive_residual_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| $1-5 | 5987 | 124 | 0.834 | 0.460 | 0.446 |
| $6-15 | 16523 | 119 | 2.286 | 0.523 | -0.889 |
| $16-30 | 14418 | 100 | 1.053 | 0.518 | -1.459 |
| $31-50 | 6393 | 55 | 0.112 | 0.508 | -1.275 |
| $51+ | 6599 | 79 | 1.909 | 0.613 | -1.408 |

## Selected roster slots by value-over-price quintile

Quintile 5 has the strongest projection rank relative to its predicted-price rank within year and position.

| value_over_price_quintile | selected_slots | player_origins | mean_salary_residual | positive_residual_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| 1 | 1263 | 42 | -0.316 | 0.383 | -1.458 |
| 2 | 6073 | 106 | -0.805 | 0.409 | -1.094 |
| 3 | 14866 | 146 | 1.008 | 0.462 | -1.432 |
| 4 | 17232 | 107 | 0.642 | 0.508 | -0.930 |
| 5 | 10486 | 76 | 4.815 | 0.723 | -0.447 |

## Roster-gap decomposition

| period | chance_level | scenario_mean_spend | point_predicted_spend | actual_spend | point_minus_scenario_discount | actual_minus_point_residual | actual_minus_scenario_total |
|---|---|---|---|---|---|---|---|
| development_2022_2024 | 0.600 | 295.474 | 308.365 | 324.776 | 12.891 | 16.411 | 29.302 |
| temporal_check_2025 | 0.600 | 296.123 | 305.678 | 327.276 | 9.555 | 21.598 | 31.153 |
| development_2022_2024 | 0.700 | 293.549 | 306.408 | 322.708 | 12.859 | 16.300 | 29.159 |
| temporal_check_2025 | 0.700 | 293.923 | 303.392 | 324.852 | 9.469 | 21.460 | 30.929 |
| development_2022_2024 | 0.800 | 291.502 | 304.264 | 320.593 | 12.762 | 16.329 | 29.091 |
| temporal_check_2025 | 0.800 | 291.419 | 300.790 | 322.624 | 9.371 | 21.834 | 31.205 |
| development_2022_2024 | 0.900 | 288.821 | 301.579 | 317.969 | 12.758 | 16.390 | 29.148 |
| temporal_check_2025 | 0.900 | 288.791 | 298.146 | 321.152 | 9.356 | 23.006 | 32.361 |

## Main interpretation

The all-player mean residual is $-0.39, while the roster-slot-weighted mean is $1.43 per selected player.

The all-player mean five-draw scenario shift versus the point salary is $0.23 per player, while the roster-slot-weighted shift is $-0.92.

Ever-selected unique players do not have materially higher residuals than the full pool. The bias appears when selection frequency is retained: frequently reused players carry positive residuals and rare selections carry negative residuals.

The prior roughly $29 actual-minus-scenario gap has two components: actual prices above the point-predicted salary row and normalized five-draw scenario spend below that point row for the selected roster. It should not be attributed entirely to player-level salary residuals.

## Limits

- The value-over-price measure is a transparent rank-gap proxy, not the exact context-specific managed ILP coefficient.
- A player selected in many Monte Carlo trials repeats one realized season salary; slot weighting measures roster impact, not independent statistical sample size.
- Only four realized auction markets are available, and the salary method specification is retrospective as of 2026.
- Recorded actual prices cover only part of the candidate pool; player-residual summaries exclude missing prices, while exact roster reconstruction retains the replay's intentional $1 fallback.
