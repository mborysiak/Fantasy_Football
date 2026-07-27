# Keeper Reinvestment Sensitivity Results

Full-roster reoptimization around up to one, two, or three newly forced 
keeper-oriented bench players. All accepted portfolios preserve the 
full-bank expected reference score; independent outcomes remain evaluation-only.

## Paired effects versus current-only control

| year | policy | roster_changed_rate | forced_option_count | starter_changes | starter_spend_effect | bench_spend_effect | unspent_budget_effect | starter_forecast_ev_effect | forecast_ev_effect | forecast_p10_effect | actual_points_effect | waiver_starts_effect | playoff_effect | predicted_best_surplus_effect | actual_best_surplus_effect | actual_hit20_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | reinvest_k1 | 1.0 | 1.0 | 1.25 | 9.038 | -8.047 | -0.99 | 2.803 | -2.839 | 7.674 | 27.275 | -1.0 | 14.575 | 7.795 | 0.326 | 0.0 |
| 2024 | reinvest_k2 | 1.0 | 1.75 | 1.75 | 4.845 | -5.779 | 0.935 | -7.934 | -9.464 | 1.406 | 15.2 | -2.75 | 17.775 | 12.941 | 0.302 | 0.0 |
| 2024 | reinvest_k3 | 1.0 | 2.5 | 2.0 | 9.728 | -10.421 | 0.693 | -0.489 | -13.313 | -0.676 | -34.55 | 4.0 | 8.675 | 16.249 | 24.164 | -0.25 |

## Policy means by origin

| year | policy | trials | unique_rosters | forced_option_count | forced_young_count | candidate_attempts | reoptimization_refine_swaps | starter_changes_vs_control | bench_changes_vs_control | roster_changes_vs_control | forecast_salary_spend | unspent_budget | starter_forecast_spend | bench_forecast_spend | starter_projected_ppg_sum | starter_raw_actual_points | starter_forecast_ev | starter_forecast_p10 | starter_forecast_p90 | top3_spend_share | current_construction_delta | forecast_ev | forecast_p10 | forecast_p90 | actual_points | drafted_only_points | actual_waiver_starts | actual_playoff_points | predicted_expected_best_surplus | predicted_probability_any_hit | predicted_probability_any_10 | predicted_probability_any_20 | actual_best_keeper_surplus | actual_any_keeper_hit_10 | actual_any_keeper_hit_20 | actual_best_future_ppg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | control | 4 | 4 | 0.0 | 0.0 | 0.0 | 2.25 | 0.0 | 0.0 | 0.0 | 293.381 | 4.619 | 258.671 | 34.71 | 107.12 | 1207.72 | 1622.245 | 1460.224 | 1801.135 | 0.577 | 0.0 | 1663.223 | 1517.076 | 1843.235 | 1515.25 | 1543.145 | 35.25 | 419.03 | 30.37 | 0.751 | 0.62 | 0.572 | 23.729 | 1.0 | 1.0 | 11.936 |
| 2024 | reinvest_k1 | 4 | 4 | 1.0 | 1.0 | 6.0 | 5.0 | 1.25 | 2.0 | 3.0 | 294.371 | 3.629 | 267.709 | 26.663 | 107.808 | 1234.67 | 1625.048 | 1470.599 | 1802.689 | 0.59 | 9.208 | 1660.384 | 1524.75 | 1838.459 | 1542.525 | 1563.42 | 34.25 | 433.605 | 38.165 | 0.791 | 0.694 | 0.631 | 24.056 | 1.0 | 1.0 | 12.268 |
| 2024 | reinvest_k2 | 4 | 3 | 1.75 | 1.75 | 12.0 | 5.75 | 1.75 | 2.5 | 3.25 | 292.446 | 5.554 | 263.516 | 28.93 | 107.386 | 1194.67 | 1614.311 | 1450.41 | 1788.943 | 0.607 | 12.682 | 1653.759 | 1518.481 | 1829.185 | 1530.45 | 1553.095 | 32.5 | 436.805 | 43.311 | 0.814 | 0.735 | 0.671 | 24.032 | 1.0 | 1.0 | 12.6 |
| 2024 | reinvest_k3 | 4 | 4 | 2.5 | 2.5 | 18.0 | 7.5 | 2.0 | 3.25 | 4.5 | 292.688 | 5.312 | 268.399 | 24.289 | 107.804 | 1164.745 | 1621.755 | 1467.032 | 1784.111 | 0.615 | 18.649 | 1649.909 | 1516.4 | 1816.061 | 1480.7 | 1545.17 | 39.25 | 427.705 | 46.619 | 0.822 | 0.753 | 0.675 | 47.893 | 1.0 | 0.75 | 13.565 |

## Interpretation boundaries

- The option search is greedy and uses a bounded marginal-utility shortlist.
- A forced player must remain on the nominal preseason bench.
- Other starter and bench slots may change during the conditional full solve.
- Keeper surplus never enters current-season fantasy-point evaluation.
