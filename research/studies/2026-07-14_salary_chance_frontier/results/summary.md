# Salary Chance-Constraint Frontier

250 paired trials per origin; each roster was constructed against 20 normalized five-draw salary markets and evaluated on 200 unseen markets.

## Frontier by year

| year | chance_level | managed_forecast_season_points | heldout_cap_probability | actual_cap_feasible_rate | actual_cap_overage | affordable_actual_rosters |
|---|---|---|---|---|---|---|
| 2022 | 0.600 | 1566.388 | 0.640 | 0.188 | 19.544 | 47 |
| 2022 | 0.700 | 1565.328 | 0.721 | 0.244 | 18.620 | 61 |
| 2022 | 0.800 | 1563.385 | 0.805 | 0.280 | 17.152 | 70 |
| 2022 | 0.900 | 1561.357 | 0.873 | 0.280 | 15.972 | 70 |
| 2023 | 0.600 | 1527.904 | 0.618 | 0.080 | 31.808 | 20 |
| 2023 | 0.700 | 1526.516 | 0.714 | 0.100 | 29.092 | 25 |
| 2023 | 0.800 | 1525.345 | 0.785 | 0.096 | 27.772 | 24 |
| 2023 | 0.900 | 1522.955 | 0.873 | 0.112 | 24.800 | 28 |
| 2024 | 0.600 | 1641.998 | 0.617 | 0.096 | 32.732 | 24 |
| 2024 | 0.700 | 1640.886 | 0.704 | 0.104 | 31.216 | 26 |
| 2024 | 0.800 | 1638.518 | 0.788 | 0.128 | 28.552 | 32 |
| 2024 | 0.900 | 1637.336 | 0.868 | 0.152 | 25.972 | 38 |
| 2025 | 0.600 | 1634.121 | 0.583 | 0.084 | 30.076 | 21 |
| 2025 | 0.700 | 1632.019 | 0.673 | 0.128 | 27.972 | 32 |
| 2025 | 0.800 | 1627.730 | 0.767 | 0.120 | 25.888 | 30 |
| 2025 | 0.900 | 1625.184 | 0.842 | 0.148 | 24.456 | 37 |

## Development and temporal-check frontier

| chance_level | development_2022_2024_managed_forecast_season_points | development_2022_2024_heldout_cap_probability | development_2022_2024_actual_cap_feasible_rate | development_2022_2024_actual_cap_overage | temporal_check_2025_managed_forecast_season_points | temporal_check_2025_heldout_cap_probability | temporal_check_2025_actual_cap_feasible_rate | temporal_check_2025_actual_cap_overage |
|---|---|---|---|---|---|---|---|---|
| 0.600 | 1578.763 | 0.625 | 0.121 | 28.028 | 1634.121 | 0.583 | 0.084 | 30.076 |
| 0.700 | 1577.577 | 0.713 | 0.149 | 26.309 | 1632.019 | 0.673 | 0.128 | 27.972 |
| 0.800 | 1575.749 | 0.793 | 0.168 | 24.492 | 1627.730 | 0.767 | 0.120 | 25.888 |
| 0.900 | 1573.882 | 0.871 | 0.181 | 22.248 | 1625.184 | 0.842 | 0.148 | 24.456 |

## Adjacent-threshold paired effects

Effects are higher threshold minus lower threshold.

| comparison | development_2022_2024_mean_managed_forecast_season_points_effect | development_2022_2024_mean_heldout_cap_probability_effect | development_2022_2024_mean_actual_cap_feasible_effect | development_2022_2024_mean_actual_cap_overage_effect | temporal_check_2025_mean_managed_forecast_season_points_effect | temporal_check_2025_mean_heldout_cap_probability_effect | temporal_check_2025_mean_actual_cap_feasible_effect | temporal_check_2025_mean_actual_cap_overage_effect |
|---|---|---|---|---|---|---|---|---|
| 70_minus_60 | -1.186 | 0.088 | 0.028 | -1.719 | -2.102 | 0.090 | 0.044 | -2.104 |
| 80_minus_70 | -1.828 | 0.080 | 0.019 | -1.817 | -4.289 | 0.093 | -0.008 | -2.084 |
| 90_minus_80 | -1.867 | 0.078 | 0.013 | -2.244 | -2.546 | 0.076 | 0.028 | -1.432 |

## Interpretation limits

- Managed forecast points are independently simulated preseason EV, not realized historical points.
- Raw points for historically unaffordable rosters are audit-only and are excluded from the policy summary.
- Feasible-only historical points select on future realized prices and cannot identify the best policy.
- The one-swap refiner is disabled because it cannot enforce the multi-scenario chance constraint; every threshold uses the same unrefined optimizer.
- Market scenarios reconcile the shared league budget, but player residuals are sampled marginally; cross-player auction-price correlation is not learned.
- Salary training data roll by origin, but the 2026 model specification is retrospective rather than a fresh method holdout.
- Historical final prices are exogenous, and missing actual prices retain the intentional `$1` fallback.
- Four seasons are four outcome units; trial counts measure Monte Carlo stability, not additional independent seasons.
