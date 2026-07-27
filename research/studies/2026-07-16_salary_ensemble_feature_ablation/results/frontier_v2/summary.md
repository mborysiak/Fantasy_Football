# Salary Chance-Constraint Frontier

250 paired trials per origin; each roster was constructed against 20 normalized five-draw salary markets and evaluated on 200 unseen markets.

## Frontier by year

| year | chance_level | managed_forecast_season_points | heldout_cap_probability | actual_cap_feasible_rate | actual_cap_overage | affordable_actual_rosters |
|---|---|---|---|---|---|---|
| 2022 | 0.600 | 1564.838 | 0.608 | 0.220 | 19.196 | 55 |
| 2022 | 0.700 | 1564.521 | 0.693 | 0.244 | 17.364 | 61 |
| 2022 | 0.800 | 1562.498 | 0.778 | 0.308 | 14.756 | 77 |
| 2022 | 0.900 | 1559.639 | 0.860 | 0.344 | 13.236 | 86 |
| 2023 | 0.600 | 1533.073 | 0.622 | 0.056 | 35.740 | 14 |
| 2023 | 0.700 | 1531.029 | 0.700 | 0.072 | 34.916 | 18 |
| 2023 | 0.800 | 1528.611 | 0.782 | 0.088 | 31.988 | 22 |
| 2023 | 0.900 | 1526.816 | 0.870 | 0.100 | 29.248 | 25 |
| 2024 | 0.600 | 1645.521 | 0.620 | 0.076 | 32.848 | 19 |
| 2024 | 0.700 | 1643.488 | 0.703 | 0.104 | 30.944 | 26 |
| 2024 | 0.800 | 1642.704 | 0.782 | 0.124 | 28.916 | 31 |
| 2024 | 0.900 | 1640.129 | 0.863 | 0.124 | 27.200 | 31 |
| 2025 | 0.600 | 1632.466 | 0.584 | 0.108 | 29.136 | 27 |
| 2025 | 0.700 | 1628.530 | 0.673 | 0.124 | 27.516 | 31 |
| 2025 | 0.800 | 1627.737 | 0.754 | 0.144 | 26.472 | 36 |
| 2025 | 0.900 | 1623.178 | 0.838 | 0.152 | 23.444 | 38 |

## Development and temporal-check frontier

| chance_level | development_2022_2024_managed_forecast_season_points | development_2022_2024_heldout_cap_probability | development_2022_2024_actual_cap_feasible_rate | development_2022_2024_actual_cap_overage | temporal_check_2025_managed_forecast_season_points | temporal_check_2025_heldout_cap_probability | temporal_check_2025_actual_cap_feasible_rate | temporal_check_2025_actual_cap_overage |
|---|---|---|---|---|---|---|---|---|
| 0.600 | 1581.144 | 0.617 | 0.117 | 29.261 | 1632.466 | 0.584 | 0.108 | 29.136 |
| 0.700 | 1579.680 | 0.699 | 0.140 | 27.741 | 1628.530 | 0.673 | 0.124 | 27.516 |
| 0.800 | 1577.938 | 0.781 | 0.173 | 25.220 | 1627.737 | 0.754 | 0.144 | 26.472 |
| 0.900 | 1575.528 | 0.864 | 0.189 | 23.228 | 1623.178 | 0.838 | 0.152 | 23.444 |

## Adjacent-threshold paired effects

Effects are higher threshold minus lower threshold.

| comparison | development_2022_2024_mean_managed_forecast_season_points_effect | development_2022_2024_mean_heldout_cap_probability_effect | development_2022_2024_mean_actual_cap_feasible_effect | development_2022_2024_mean_actual_cap_overage_effect | temporal_check_2025_mean_managed_forecast_season_points_effect | temporal_check_2025_mean_heldout_cap_probability_effect | temporal_check_2025_mean_actual_cap_feasible_effect | temporal_check_2025_mean_actual_cap_overage_effect |
|---|---|---|---|---|---|---|---|---|
| 70_minus_60 | -1.464 | 0.082 | 0.023 | -1.520 | -3.937 | 0.089 | 0.016 | -1.620 |
| 80_minus_70 | -1.742 | 0.082 | 0.033 | -2.521 | -0.793 | 0.081 | 0.020 | -1.044 |
| 90_minus_80 | -2.409 | 0.083 | 0.016 | -1.992 | -4.559 | 0.084 | 0.008 | -3.028 |

## Interpretation limits

- Managed forecast points are independently simulated preseason EV, not realized historical points.
- Raw points for historically unaffordable rosters are audit-only and are excluded from the policy summary.
- Feasible-only historical points select on future realized prices and cannot identify the best policy.
- The one-swap refiner is disabled because it cannot enforce the multi-scenario chance constraint; every threshold uses the same unrefined optimizer.
- Market scenarios reconcile the shared league budget, but player residuals are sampled marginally; cross-player auction-price correlation is not learned.
- Salary training data roll by origin, but the 2026 model specification is retrospective rather than a fresh method holdout.
- Historical final prices are exogenous, and missing actual prices retain the intentional `$1` fallback.
- Four seasons are four outcome units; trial counts measure Monte Carlo stability, not additional independent seasons.
