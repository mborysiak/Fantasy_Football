# Optimizer Selection Surcharge Replay

250 paired five-draw trials per origin. 2022 seeds the calibration; development is 2023-2024 and 2025 is the temporal check.

## Variant outcomes by year

| year | variant | actual_points | cap_feasible_rate | mean_cap_overage | actual_salary_spend | decision_salary_spend | surcharge_spend | forecast_ev |
|---|---|---|---|---|---|---|---|---|
| 2022 | baseline_298 | 1492.815 | 0.240 | 21.144 | 316.088 | 293.056 | 0.000 | 1573.467 |
| 2022 | blanket_285 | 1470.653 | 0.412 | 12.340 | 303.448 | 279.931 | 0.000 | 1564.045 |
| 2022 | targeted_full | 1492.815 | 0.240 | 21.144 | 316.088 | 293.056 | 0.000 | 1573.467 |
| 2022 | targeted_half | 1492.815 | 0.240 | 21.144 | 316.088 | 293.056 | 0.000 | 1573.467 |
| 2023 | baseline_298 | 1691.690 | 0.036 | 36.548 | 334.236 | 292.715 | 0.000 | 1544.043 |
| 2023 | blanket_285 | 1681.787 | 0.168 | 22.784 | 318.796 | 279.829 | 0.000 | 1531.527 |
| 2023 | targeted_full | 1684.695 | 0.132 | 29.052 | 325.676 | 292.904 | 9.478 | 1529.002 |
| 2023 | targeted_half | 1687.200 | 0.080 | 30.880 | 328.224 | 292.936 | 5.421 | 1535.080 |
| 2024 | baseline_298 | 1597.317 | 0.084 | 32.764 | 329.960 | 293.452 | 0.000 | 1653.763 |
| 2024 | blanket_285 | 1587.179 | 0.224 | 21.704 | 316.828 | 280.677 | 0.000 | 1641.395 |
| 2024 | targeted_full | 1597.315 | 0.300 | 16.876 | 310.864 | 294.106 | 16.404 | 1631.512 |
| 2024 | targeted_half | 1599.196 | 0.176 | 24.388 | 320.288 | 293.305 | 8.850 | 1640.957 |
| 2025 | baseline_298 | 1582.621 | 0.080 | 29.968 | 327.268 | 294.361 | 0.000 | 1637.204 |
| 2025 | blanket_285 | 1578.576 | 0.224 | 20.348 | 316.168 | 281.367 | 0.000 | 1623.385 |
| 2025 | targeted_full | 1577.357 | 0.292 | 15.920 | 310.400 | 294.767 | 17.027 | 1611.824 |
| 2025 | targeted_half | 1582.567 | 0.148 | 21.904 | 318.096 | 294.719 | 9.538 | 1624.575 |

## Paired effects versus baseline

Positive points/feasibility favor the candidate; negative overage favors the candidate.

| comparison | development_2023_2024_mean_actual_points_effect | development_2023_2024_mean_actual_cap_feasible_effect | development_2023_2024_mean_actual_cap_overage_effect | development_2023_2024_mean_forecast_ev_effect | development_2023_2024_roster_changed_rate | temporal_check_2025_mean_actual_points_effect | temporal_check_2025_mean_actual_cap_feasible_effect | temporal_check_2025_mean_actual_cap_overage_effect | temporal_check_2025_mean_forecast_ev_effect |
|---|---|---|---|---|---|---|---|---|---|
| blanket_285_minus_baseline_298 | -10.020 | 0.136 | -12.412 | -12.442 | 1.000 | -4.045 | 0.144 | -9.620 | -13.820 |
| targeted_full_minus_baseline_298 | -3.499 | 0.156 | -11.692 | -18.646 | 0.980 | -5.264 | 0.212 | -14.048 | -25.380 |
| targeted_half_minus_baseline_298 | -1.306 | 0.068 | -7.022 | -10.885 | 0.946 | -0.054 | 0.068 | -8.064 | -12.629 |

## Player-level calibration

| period | variant | mean_error | mae | selection_weighted_mean_error | selection_weighted_mae | selection_weighted_surcharge |
|---|---|---|---|---|---|---|
| seed_2022 | baseline_298 | -0.783 | 4.572 | 0.708 | 5.402 | 0.000 |
| seed_2022 | targeted_half | -0.783 | 4.572 | 0.708 | 5.402 | 0.000 |
| seed_2022 | targeted_full | -0.783 | 4.572 | 0.708 | 5.402 | 0.000 |
| development_2023_2024 | baseline_298 | -0.134 | 4.299 | 1.607 | 5.558 | 0.000 |
| development_2023_2024 | targeted_half | -0.298 | 4.309 | 1.043 | 5.543 | 0.564 |
| development_2023_2024 | targeted_full | -0.462 | 4.365 | 0.479 | 5.732 | 1.129 |
| temporal_check_2025 | baseline_298 | -0.174 | 3.683 | 1.439 | 4.830 | 0.000 |
| temporal_check_2025 | targeted_half | -0.392 | 3.692 | 0.669 | 4.751 | 0.770 |
| temporal_check_2025 | targeted_full | -0.610 | 3.739 | -0.101 | 4.821 | 1.540 |

## Fixed baseline-roster spend gap

| period | variant | mean_actual_minus_price_gap | mean_absolute_gap | mean_modeled_spend | mean_actual_spend |
|---|---|---|---|---|---|
| seed_2022 | baseline_298 | 8.313 | 17.543 | 302.412 | 310.725 |
| seed_2022 | targeted_half | 8.313 | 17.543 | 302.412 | 310.725 |
| seed_2022 | targeted_full | 8.313 | 17.543 | 302.412 | 310.725 |
| development_2023_2024 | baseline_298 | 19.689 | 23.701 | 302.869 | 322.558 |
| development_2023_2024 | targeted_half | 12.676 | 20.090 | 309.882 | 322.558 |
| development_2023_2024 | targeted_full | 5.662 | 18.216 | 316.896 | 322.558 |
| temporal_check_2025 | baseline_298 | 18.632 | 22.888 | 299.741 | 318.373 |
| temporal_check_2025 | targeted_half | 8.652 | 17.556 | 309.721 | 318.373 |
| temporal_check_2025 | targeted_full | -1.329 | 16.129 | 319.702 | 318.373 |

## Feasible-only roster quality

These point means are conditional on each policy producing an actually affordable roster, so they are descriptive rather than a paired causal point comparison.

| year | variant | feasible_trials | cap_feasible_rate | actual_points_feasible_only | forecast_ev_feasible_only | actual_salary_spend_feasible_only |
|---|---|---|---|---|---|---|
| 2022 | baseline_298 | 60 | 0.240 | 1462.236 | 1569.023 | 285.267 |
| 2022 | blanket_285 | 103 | 0.412 | 1469.319 | 1558.995 | 281.272 |
| 2022 | targeted_full | 60 | 0.240 | 1462.236 | 1569.023 | 285.267 |
| 2022 | targeted_half | 60 | 0.240 | 1462.236 | 1569.023 | 285.267 |
| 2023 | baseline_298 | 9 | 0.036 | 1684.780 | 1522.320 | 289.333 |
| 2023 | blanket_285 | 42 | 0.168 | 1695.334 | 1518.231 | 286.167 |
| 2023 | targeted_full | 33 | 0.132 | 1690.862 | 1520.723 | 287.576 |
| 2023 | targeted_half | 20 | 0.080 | 1692.090 | 1525.835 | 289.800 |
| 2024 | baseline_298 | 21 | 0.084 | 1601.828 | 1648.627 | 288.429 |
| 2024 | blanket_285 | 56 | 0.224 | 1616.969 | 1633.734 | 285.161 |
| 2024 | targeted_full | 75 | 0.300 | 1602.663 | 1624.083 | 284.627 |
| 2024 | targeted_half | 44 | 0.176 | 1607.740 | 1631.644 | 286.068 |
| 2025 | baseline_298 | 20 | 0.080 | 1525.397 | 1617.446 | 289.250 |
| 2025 | blanket_285 | 56 | 0.224 | 1518.917 | 1608.510 | 288.268 |
| 2025 | targeted_full | 73 | 0.292 | 1532.410 | 1601.084 | 285.945 |
| 2025 | targeted_half | 37 | 0.148 | 1536.337 | 1610.815 | 285.784 |

## Limits

- Selection frequency is a causal preseason seed-pass feature, but production use requires that initial optimizer pass.
- The v5 method specification is retrospective even though every salary and surcharge fit rolls strictly by data origin.
- Four seasons are four outcome units; trial counts measure Monte Carlo stability rather than additional independent seasons.
- The surcharge is a decision-price reserve, not a claim that the coherent league-wide salary market has a larger total budget.
