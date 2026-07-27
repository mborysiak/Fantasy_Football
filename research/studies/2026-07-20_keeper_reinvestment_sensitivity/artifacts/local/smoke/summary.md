# Keeper Reinvestment Sensitivity Results

Full-roster reoptimization around up to one, two, or three newly forced 
keeper-oriented bench players. All accepted portfolios preserve the 
common construction-bank mean; independent outcomes remain evaluation-only.

## Paired effects versus current-only control

| year | policy | roster_changed_rate | forced_option_count | starter_changes | starter_spend_effect | bench_spend_effect | unspent_budget_effect | starter_forecast_ev_effect | forecast_ev_effect | forecast_p10_effect | actual_points_effect | waiver_starts_effect | playoff_effect | predicted_best_surplus_effect | actual_best_surplus_effect | actual_hit20_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | reinvest_k1 | 0.75 | 0.75 | 4.25 | -4.026 | 5.215 | -1.189 | -41.171 | -47.728 | -64.592 | 102.745 | 3.5 | 55.615 | 5.803 | 0.775 | 0.0 |
| 2024 | reinvest_k2 | 0.75 | 1.5 | 3.5 | 4.375 | -4.964 | 0.59 | -40.155 | -51.802 | -84.725 | -8.065 | 9.0 | 17.865 | 8.028 | -32.014 | -0.5 |
| 2024 | reinvest_k3 | 0.75 | 2.25 | 4.0 | 7.541 | -11.886 | 4.344 | -39.174 | -45.575 | -58.447 | 41.645 | 5.75 | 41.64 | 11.537 | -18.461 | -0.25 |

## Policy means by origin

| year | policy | trials | unique_rosters | forced_option_count | forced_young_count | candidate_attempts | reoptimization_refine_swaps | starter_changes_vs_control | bench_changes_vs_control | roster_changes_vs_control | forecast_salary_spend | unspent_budget | starter_forecast_spend | bench_forecast_spend | starter_projected_ppg_sum | starter_raw_actual_points | starter_forecast_ev | starter_forecast_p10 | starter_forecast_p90 | top3_spend_share | current_construction_delta | forecast_ev | forecast_p10 | forecast_p90 | actual_points | drafted_only_points | actual_waiver_starts | actual_playoff_points | predicted_expected_best_surplus | predicted_probability_any_hit | predicted_probability_any_10 | predicted_probability_any_20 | actual_best_keeper_surplus | actual_any_keeper_hit_10 | actual_any_keeper_hit_20 | actual_best_future_ppg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | control | 4 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 295.658 | 2.342 | 245.3 | 50.358 | 106.582 | 1181.545 | 1598.45 | 1476.733 | 1710.089 | 0.53 | 0.0 | 1652.783 | 1531.813 | 1784.227 | 1540.07 | 1435.37 | 33.25 | 396.41 | 35.42 | 0.77 | 0.693 | 0.637 | 35.896 | 0.5 | 0.5 | 12.807 |
| 2024 | reinvest_k1 | 4 | 4 | 0.75 | 0.75 | 6.0 | 0.0 | 4.25 | 3.5 | 7.75 | 296.847 | 1.153 | 241.274 | 55.573 | 104.196 | 1310.78 | 1557.279 | 1426.83 | 1698.362 | 0.563 | 105.928 | 1605.055 | 1467.221 | 1740.391 | 1642.815 | 1597.78 | 36.75 | 452.025 | 41.223 | 0.801 | 0.734 | 0.689 | 36.671 | 0.75 | 0.5 | 12.769 |
| 2024 | reinvest_k2 | 4 | 4 | 1.5 | 1.25 | 12.0 | 0.25 | 3.5 | 3.5 | 7.0 | 295.068 | 2.932 | 249.674 | 45.394 | 104.786 | 1199.245 | 1558.295 | 1400.402 | 1683.79 | 0.547 | 108.363 | 1600.981 | 1447.088 | 1749.585 | 1532.005 | 1456.42 | 42.25 | 414.275 | 43.448 | 0.818 | 0.731 | 0.683 | 3.883 | 0.25 | 0.0 | 10.204 |
| 2024 | reinvest_k3 | 4 | 4 | 2.25 | 2.0 | 18.0 | 1.0 | 4.0 | 3.5 | 7.5 | 291.314 | 6.686 | 252.841 | 38.473 | 104.286 | 1227.995 | 1559.276 | 1434.026 | 1675.206 | 0.593 | 96.868 | 1607.208 | 1473.366 | 1759.023 | 1581.715 | 1526.02 | 39.0 | 438.05 | 46.956 | 0.83 | 0.749 | 0.707 | 17.435 | 0.25 | 0.25 | 10.577 |

## Interpretation boundaries

- The option search is greedy and uses a bounded marginal-utility shortlist.
- A forced player must remain on the nominal preseason bench.
- Other starter and bench slots may change during the conditional full solve.
- Keeper surplus never enters current-season fantasy-point evaluation.
