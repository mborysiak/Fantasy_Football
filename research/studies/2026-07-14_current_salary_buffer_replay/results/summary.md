# Current-Method $5 versus $10 Replay

250 paired five-draw trials per origin; development is 2022-2024 and 2025 is a temporal check.

## Variant outcomes by year

| year | buffer | actual_points | cap_feasible_rate | mean_cap_overage | actual_salary_spend | point_salary_spend | salary_model_fallback_players |
|---|---|---|---|---|---|---|---|
| 2022 | 10 | 1481.929 | 0.268 | 18.104 | 312.300 | 300.875 | 0.128 |
| 2022 | 5 | 1474.083 | 0.308 | 16.172 | 309.704 | 296.823 | 0.128 |
| 2022 | none | 1485.239 | 0.232 | 21.456 | 316.424 | 306.446 | 0.120 |
| 2023 | 10 | 1687.668 | 0.140 | 23.632 | 319.916 | 303.219 | 0.368 |
| 2023 | 5 | 1687.332 | 0.228 | 19.812 | 315.276 | 298.380 | 0.360 |
| 2023 | none | 1694.470 | 0.040 | 37.008 | 334.704 | 317.664 | 0.328 |
| 2024 | 10 | 1605.559 | 0.092 | 31.548 | 328.556 | 302.180 | 1.332 |
| 2024 | 5 | 1597.041 | 0.160 | 26.644 | 323.172 | 297.921 | 1.356 |
| 2024 | none | 1604.313 | 0.044 | 37.120 | 334.672 | 309.982 | 1.336 |
| 2025 | 10 | 1552.646 | 0.100 | 26.516 | 323.604 | 302.375 | 0.396 |
| 2025 | 5 | 1549.548 | 0.164 | 23.200 | 319.696 | 297.815 | 0.412 |
| 2025 | none | 1569.589 | 0.044 | 36.760 | 334.440 | 312.958 | 0.372 |

## Direct $5 minus $10 effects

Positive points/feasibility favor $5; negative overage/spend favor $5.

| comparison | development_2022_2024_mean_actual_points_effect | development_2022_2024_mean_actual_cap_feasible_effect | development_2022_2024_mean_actual_cap_overage_effect | development_2022_2024_roster_changed_rate | temporal_check_2025_mean_actual_points_effect | temporal_check_2025_mean_actual_cap_feasible_effect | temporal_check_2025_mean_actual_cap_overage_effect |
|---|---|---|---|---|---|---|---|
| 5_minus_10 | -5.567 | 0.065 | -3.552 | 0.837 | -3.097 | 0.064 | -3.316 |

## Limits

- Salary training data roll by origin, but the 2026 model specification is retrospective rather than a fresh method holdout.
- Historical final prices remain exogenous and missing actual prices retain the intentional $1 scoring fallback.
- Frozen point forecasts and the current salary pool differ for some players; every salary-model and minimum fallback is recorded.
- Four seasons are four outcome units; trial counts measure Monte Carlo stability, not additional independent seasons.
