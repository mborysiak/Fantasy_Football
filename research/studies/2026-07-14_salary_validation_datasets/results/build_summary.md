# Build Summary

Build identity: `beta / current_locked_spec_v1 / model_spec_asof_year=2026`.

| Year | Full pool rows | Copied base values | Observed actuals | Open slots | Normalized market total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 299 | 137 | 146 | 136 | $2,886 |
| 2023 | 311 | 171 | 152 | 135 | $2,705 |
| 2024 | 304 | 182 | 150 | 138 | $3,013 |
| 2025 | 305 | 170 | 152 | 141 | $3,169 |

The 2023 totals include two deterministic keeper rows (J.K. Dobbins and Nick
Chubb) that lacked origin-year projection features. They do not alter the 309
projection-universe predictions or the 135-slot non-keeper normalization.

The observed validation table contains 644 non-keeper rows: 118, 126, 131, 132,
and 137 for 2021 through 2025. Overall normalized residual MAE/RMSE are
`4.307 / 6.276`; raw residual MAE/RMSE are `4.187 / 6.180`. Known-budget
normalization therefore enforces a feasible aggregate market but did not improve
player-level error in this small replay. Both scales must remain available.

Independent checks passed for unique keys, strict training and residual cutoffs,
monotone residual quantiles, full candidate-pool slot coverage, absence of
target-actual normalization, exact keeper-adjusted market totals, and live
Simulation slice preservation.
