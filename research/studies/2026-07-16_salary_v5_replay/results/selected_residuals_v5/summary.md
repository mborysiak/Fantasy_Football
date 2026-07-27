# Selected-Roster Salary Residual Diagnostic

Residuals use 518 recorded player-origin prices from 930 auctionable candidates. Recorded-price coverage is 96.0% of the 52,000 selected roster slots.

## Core cohort comparison

| cohort | player_origins | weighted_observations | mean_salary_residual | positive_residual_rate | mean_selection_rate | mean_scenario_center_shift | mean_actual_minus_scenario |
|---|---|---|---|---|---|---|---|
| all_observed_auctionable | 518 | 518 | -0.296 | 0.380 | 0.096 | -0.402 | 0.106 |
| ever_selected_unique | 476 | 476 | -0.211 | 0.395 | 0.105 | -0.477 | 0.266 |
| never_selected_unique | 42 | 42 | -1.262 | 0.214 | 0.000 | 0.453 | -1.716 |
| top_projection_quartile_ever_selected | 218 | 218 | -0.156 | 0.468 | 0.140 | -0.794 | 0.638 |
| top_projection_quartile_never_selected | 4 | 4 | -9.931 | 0.000 | 0.000 | 0.555 | -10.486 |
| top_projection_quartile_rare_le_5pct | 54 | 54 | -3.219 | 0.296 | 0.021 | -0.153 | -3.066 |
| top_projection_quartile_frequent_ge_25pct | 36 | 36 | 2.098 | 0.556 | 0.354 | -1.270 | 3.367 |
| selected_roster_slots_weighted | 476 | 49899 | 1.346 | 0.504 | 0.219 | -0.806 | 2.152 |

## Selection-frequency gradient

| cohort | player_origins | mean_salary_residual | positive_residual_rate | mean_selection_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| never | 42 | -1.262 | 0.214 | 0.000 | 0.453 |
| rare_0-5% | 189 | -1.290 | 0.291 | 0.022 | -0.094 |
| occasional_5-25% | 236 | 0.189 | 0.445 | 0.118 | -0.650 |
| frequent_25-50% | 47 | 1.358 | 0.532 | 0.332 | -1.045 |
| core_>50% | 4 | 8.763 | 0.750 | 0.598 | -1.724 |

## High-projection players by selection frequency

| cohort | player_origins | mean_salary_residual | positive_residual_rate | mean_selection_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| selected_<=5% | 54 | -3.219 | 0.296 | 0.021 | -0.153 |
| selected_5-25% | 132 | 0.187 | 0.500 | 0.126 | -0.885 |
| selected_25-50% | 34 | 1.385 | 0.529 | 0.339 | -1.234 |
| selected_>50% | 2 | 14.209 | 1.000 | 0.605 | -1.873 |

## Selected roster slots by position

| pos | selected_slots | player_origins | mean_salary_residual | positive_residual_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| QB | 3886 | 57 | -0.624 | 0.375 | -0.145 |
| RB | 14606 | 162 | 1.900 | 0.507 | 0.224 |
| TE | 6845 | 53 | 0.567 | 0.506 | -0.346 |
| WR | 24562 | 204 | 1.546 | 0.522 | -1.652 |

## Selected roster slots by predicted-price tier

| predicted_salary_tier | selected_slots | player_origins | mean_salary_residual | positive_residual_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| $1-5 | 5395 | 118 | -0.475 | 0.271 | -0.003 |
| $6-15 | 18033 | 129 | 1.654 | 0.507 | -0.961 |
| $16-30 | 13462 | 99 | 0.914 | 0.515 | -1.164 |
| $31-50 | 6764 | 56 | 0.671 | 0.517 | -0.410 |
| $51+ | 6245 | 74 | 3.694 | 0.658 | -0.713 |

## Selected roster slots by value-over-price quintile

Quintile 5 has the strongest projection rank relative to its predicted-price rank within year and position.

| value_over_price_quintile | selected_slots | player_origins | mean_salary_residual | positive_residual_rate | mean_scenario_center_shift |
|---|---|---|---|---|---|
| 1 | 1471 | 44 | -0.230 | 0.320 | -0.782 |
| 2 | 7355 | 112 | 0.127 | 0.441 | -1.027 |
| 3 | 14652 | 145 | 1.458 | 0.481 | -0.885 |
| 4 | 16232 | 112 | 0.160 | 0.456 | -0.784 |
| 5 | 10189 | 63 | 4.183 | 0.685 | -0.574 |

## Roster-gap decomposition

| period | chance_level | scenario_mean_spend | point_predicted_spend | actual_spend | point_minus_scenario_discount | actual_minus_point_residual | actual_minus_scenario_total |
|---|---|---|---|---|---|---|---|
| development_2022_2024 | 0.600 | 295.470 | 305.872 | 321.909 | 10.401 | 16.038 | 26.439 |
| temporal_check_2025 | 0.600 | 296.365 | 303.259 | 321.904 | 6.894 | 18.645 | 25.539 |
| development_2022_2024 | 0.700 | 293.623 | 303.970 | 319.805 | 10.347 | 15.836 | 26.183 |
| temporal_check_2025 | 0.700 | 294.053 | 301.009 | 319.652 | 6.957 | 18.643 | 25.599 |
| development_2022_2024 | 0.800 | 291.508 | 301.868 | 317.315 | 10.360 | 15.447 | 25.807 |
| temporal_check_2025 | 0.800 | 291.779 | 298.654 | 317.140 | 6.875 | 18.486 | 25.361 |
| development_2022_2024 | 0.900 | 288.790 | 299.157 | 315.425 | 10.368 | 16.268 | 26.636 |
| temporal_check_2025 | 0.900 | 288.973 | 296.041 | 314.796 | 7.067 | 18.755 | 25.823 |

## Main interpretation

The all-player mean residual is $-0.30, while the roster-slot-weighted mean is $1.35 per selected player.

The all-player mean five-draw scenario shift versus the point salary is $0.22 per player, while the roster-slot-weighted shift is $-0.73.

Ever-selected unique players do not have materially higher residuals than the full pool. The bias appears when selection frequency is retained: frequently reused players carry positive residuals and rare selections carry negative residuals.

The prior roughly $29 actual-minus-scenario gap has two components: actual prices above the point-predicted salary row and normalized five-draw scenario spend below that point row for the selected roster. It should not be attributed entirely to player-level salary residuals.

## Limits

- The value-over-price measure is a transparent rank-gap proxy, not the exact context-specific managed ILP coefficient.
- A player selected in many Monte Carlo trials repeats one realized season salary; slot weighting measures roster impact, not independent statistical sample size.
- Only four realized auction markets are available, and the salary method specification is retrospective as of 2026.
- Recorded actual prices cover only part of the candidate pool; player-residual summaries exclude missing prices, while exact roster reconstruction retains the replay's intentional $1 fallback.
