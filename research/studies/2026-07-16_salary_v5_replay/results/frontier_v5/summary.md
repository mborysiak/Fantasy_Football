# Salary Chance-Constraint Frontier

250 paired trials per origin; each roster was constructed against 20 normalized five-draw salary markets and evaluated on 200 unseen markets.

## Frontier by year

| year | chance_level | managed_forecast_season_points | heldout_cap_probability | actual_cap_feasible_rate | actual_cap_overage | affordable_actual_rosters |
|---|---|---|---|---|---|---|
| 2022 | 0.600 | 1563.066 | 0.626 | 0.244 | 18.344 | 61 |
| 2022 | 0.700 | 1561.486 | 0.703 | 0.256 | 16.708 | 64 |
| 2022 | 0.800 | 1559.974 | 0.785 | 0.300 | 15.200 | 75 |
| 2022 | 0.900 | 1557.277 | 0.869 | 0.320 | 14.532 | 80 |
| 2023 | 0.600 | 1529.446 | 0.623 | 0.052 | 32.216 | 13 |
| 2023 | 0.700 | 1527.549 | 0.711 | 0.068 | 29.716 | 17 |
| 2023 | 0.800 | 1525.193 | 0.793 | 0.084 | 28.292 | 21 |
| 2023 | 0.900 | 1523.591 | 0.870 | 0.132 | 24.840 | 33 |
| 2024 | 0.600 | 1635.635 | 0.625 | 0.172 | 26.180 | 43 |
| 2024 | 0.700 | 1634.422 | 0.703 | 0.172 | 24.416 | 43 |
| 2024 | 0.800 | 1631.673 | 0.784 | 0.236 | 21.700 | 59 |
| 2024 | 0.900 | 1631.219 | 0.860 | 0.240 | 21.424 | 60 |
| 2025 | 0.600 | 1625.746 | 0.575 | 0.140 | 25.352 | 35 |
| 2025 | 0.700 | 1623.498 | 0.674 | 0.196 | 23.520 | 49 |
| 2025 | 0.800 | 1620.917 | 0.758 | 0.188 | 21.092 | 47 |
| 2025 | 0.900 | 1617.707 | 0.841 | 0.216 | 19.388 | 54 |

## Development and temporal-check frontier

| chance_level | development_2022_2024_managed_forecast_season_points | development_2022_2024_heldout_cap_probability | development_2022_2024_actual_cap_feasible_rate | development_2022_2024_actual_cap_overage | temporal_check_2025_managed_forecast_season_points | temporal_check_2025_heldout_cap_probability | temporal_check_2025_actual_cap_feasible_rate | temporal_check_2025_actual_cap_overage |
|---|---|---|---|---|---|---|---|---|
| 0.600 | 1576.049 | 0.625 | 0.156 | 25.580 | 1625.746 | 0.575 | 0.140 | 25.352 |
| 0.700 | 1574.486 | 0.706 | 0.165 | 23.613 | 1623.498 | 0.674 | 0.196 | 23.520 |
| 0.800 | 1572.280 | 0.788 | 0.207 | 21.731 | 1620.917 | 0.758 | 0.188 | 21.092 |
| 0.900 | 1570.696 | 0.866 | 0.231 | 20.265 | 1617.707 | 0.841 | 0.216 | 19.388 |

## Adjacent-threshold paired effects

Effects are higher threshold minus lower threshold.

| comparison | development_2022_2024_mean_managed_forecast_season_points_effect | development_2022_2024_mean_heldout_cap_probability_effect | development_2022_2024_mean_actual_cap_feasible_effect | development_2022_2024_mean_actual_cap_overage_effect | temporal_check_2025_mean_managed_forecast_season_points_effect | temporal_check_2025_mean_heldout_cap_probability_effect | temporal_check_2025_mean_actual_cap_feasible_effect | temporal_check_2025_mean_actual_cap_overage_effect |
|---|---|---|---|---|---|---|---|---|
| 70_minus_60 | -1.563 | 0.081 | 0.009 | -1.967 | -2.248 | 0.099 | 0.056 | -1.832 |
| 80_minus_70 | -2.206 | 0.082 | 0.041 | -1.883 | -2.581 | 0.084 | -0.008 | -2.428 |
| 90_minus_80 | -1.585 | 0.079 | 0.024 | -1.465 | -3.210 | 0.083 | 0.028 | -1.704 |

## Interpretation limits

- Managed forecast points are independently simulated preseason EV, not realized historical points.
- Raw points for historically unaffordable rosters are audit-only and are excluded from the policy summary.
- Feasible-only historical points select on future realized prices and cannot identify the best policy.
- The one-swap refiner is disabled because it cannot enforce the multi-scenario chance constraint; every threshold uses the same unrefined optimizer.
- Market scenarios reconcile the shared league budget, but player residuals are sampled marginally; cross-player auction-price correlation is not learned.
- Salary training data roll by origin, but the 2026 model specification is retrospective rather than a fresh method holdout.
- Historical final prices are exogenous, and missing actual prices retain the intentional `$1` fallback.
- Four seasons are four outcome units; trial counts measure Monte Carlo stability, not additional independent seasons.
