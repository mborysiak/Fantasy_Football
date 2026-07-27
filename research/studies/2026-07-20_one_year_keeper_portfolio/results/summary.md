# One-Year Keeper Portfolio Results

250 paired trials per origin; fixed starters and at most two bench swaps.

## Paired effects versus current-only control

| year | policy | roster_changed_rate | forecast_ev_effect | forecast_p10_effect | actual_points_effect | actual_playoff_points_effect | current_construction_delta_effect | predicted_expected_best_surplus_effect | actual_best_keeper_surplus_effect | actual_any_keeper_hit_10_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | best1_lex0 | 0.9 | 4.725 | 4.157 | 8.458 | 3.272 | 6.482 | 9.044 | 17.949 | 0.128 |
| 2022 | best1_lex2 | 0.94 | 4.792 | 4.106 | 7.247 | 2.902 | 5.127 | 9.69 | 18.011 | 0.128 |
| 2023 | best1_lex0 | 0.96 | 0.906 | -0.711 | 4.914 | 2.902 | 7.216 | 9.702 | 12.458 | 0.004 |
| 2023 | best1_lex2 | 0.976 | 1.38 | -0.01 | 4.88 | 1.888 | 6.113 | 10.212 | 14.885 | 0.012 |
| 2024 | best1_lex0 | 0.964 | -2.723 | -2.982 | -19.502 | -0.215 | 5.479 | 10.592 | 6.398 | 0.0 |
| 2024 | best1_lex2 | 0.976 | -2.499 | -2.477 | -21.318 | -0.238 | 4.047 | 11.094 | 6.787 | 0.004 |
| 2025 | best1_lex0 | 0.952 | 1.492 | 1.76 | -3.738 | -5.498 | 8.383 | 7.168 |  |  |
| 2025 | best1_lex2 | 0.968 | 1.178 | 1.566 | -3.399 | -5.347 | 7.071 | 7.405 |  |  |

## Policy means by origin

| year | policy | forecast_ev | forecast_p10 | actual_points | actual_playoff_points | predicted_expected_best_surplus | actual_best_keeper_surplus | actual_any_keeper_hit_10 | actual_best_future_ppg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | best1_lex0 | 1603.197 | 1439.436 | 1450.254 | 361.364 | 55.322 | 59.318 | 0.932 | 13.47 |
| 2022 | best1_lex2 | 1603.263 | 1439.385 | 1449.043 | 360.994 | 55.968 | 59.38 | 0.932 | 13.486 |
| 2022 | control | 1598.472 | 1435.279 | 1441.797 | 358.093 | 46.278 | 41.369 | 0.804 | 12.375 |
| 2023 | best1_lex0 | 1526.228 | 1381.735 | 1734.568 | 494.682 | 47.821 | 62.372 | 0.848 | 14.062 |
| 2023 | best1_lex2 | 1526.702 | 1382.436 | 1734.534 | 493.667 | 48.331 | 64.799 | 0.856 | 14.296 |
| 2023 | control | 1525.322 | 1382.446 | 1729.654 | 491.779 | 38.119 | 49.914 | 0.844 | 13.142 |
| 2024 | best1_lex0 | 1653.35 | 1499.385 | 1567.627 | 428.362 | 45.871 | 47.235 | 0.76 | 13.52 |
| 2024 | best1_lex2 | 1653.575 | 1499.89 | 1565.811 | 428.34 | 46.373 | 47.624 | 0.764 | 13.521 |
| 2024 | control | 1656.073 | 1502.367 | 1587.129 | 428.578 | 35.279 | 40.838 | 0.76 | 13.543 |
| 2025 | best1_lex0 | 1664.464 | 1513.14 | 1584.208 | 364.349 | 34.7 |  |  |  |
| 2025 | best1_lex2 | 1664.15 | 1512.946 | 1584.547 | 364.501 | 34.937 |  |  |  |
| 2025 | control | 1662.972 | 1511.38 | 1587.946 | 369.847 | 27.532 |  |  |  |

## Interpretation boundaries

- One historical season is one realized outcome unit; trial counts measure construction stability.
- Historical point predictions use the current 2026 model specification on OOS origin data, not a frozen old method.
- Players without a dedicated next validation row use an explicit current-projection proxy.
- Counterfactual modeled acquisition cost is primary; observed salary is a coverage-limited audit.
- Current forecast evaluation never includes keeper surplus as fantasy points.
