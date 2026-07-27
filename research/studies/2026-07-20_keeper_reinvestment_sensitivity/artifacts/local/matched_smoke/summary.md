# Keeper Reinvestment Sensitivity Results

Full-roster reoptimization around up to one, two, or three newly forced 
keeper-oriented bench players. All accepted portfolios preserve the 
common construction-bank mean; independent outcomes remain evaluation-only.

## Paired effects versus current-only control

| year | policy | roster_changed_rate | forced_option_count | starter_changes | starter_spend_effect | bench_spend_effect | unspent_budget_effect | starter_forecast_ev_effect | forecast_ev_effect | forecast_p10_effect | actual_points_effect | waiver_starts_effect | playoff_effect | predicted_best_surplus_effect | actual_best_surplus_effect | actual_hit20_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | reinvest_k1 | 1.0 | 1.0 | 5.0 | 15.407 | -18.172 | 2.765 | -6.64 | -4.846 | -33.145 | 72.86 | -3.0 | 40.14 | 0.041 | 66.134 | 1.0 |
| 2024 | reinvest_k2 | 1.0 | 2.0 | 5.0 | 12.642 | -13.629 | 0.988 | -8.692 | -8.803 | -24.203 | 12.86 | 6.0 | 14.64 | 10.901 | 66.134 | 1.0 |
| 2024 | reinvest_k3 | 1.0 | 3.0 | 5.0 | 21.333 | -24.098 | 2.765 | 52.798 | 47.233 | 19.808 | 3.16 | -1.0 | 48.74 | 12.029 | 66.134 | 1.0 |

## Policy means by origin

| year | policy | trials | unique_rosters | forced_option_count | forced_young_count | candidate_attempts | reoptimization_refine_swaps | starter_changes_vs_control | bench_changes_vs_control | roster_changes_vs_control | forecast_salary_spend | unspent_budget | starter_forecast_spend | bench_forecast_spend | starter_projected_ppg_sum | starter_raw_actual_points | starter_forecast_ev | starter_forecast_p10 | starter_forecast_p90 | top3_spend_share | current_construction_delta | forecast_ev | forecast_p10 | forecast_p90 | actual_points | drafted_only_points | actual_waiver_starts | actual_playoff_points | predicted_expected_best_surplus | predicted_probability_any_hit | predicted_probability_any_10 | predicted_probability_any_20 | actual_best_keeper_surplus | actual_any_keeper_hit_10 | actual_any_keeper_hit_20 | actual_best_future_ppg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | control | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 297.241 | 0.759 | 243.057 | 54.184 | 105.192 | 1078.22 | 1580.454 | 1429.669 | 1737.066 | 0.558 | 0.0 | 1650.025 | 1505.705 | 1803.78 | 1571.84 | 1359.42 | 32.0 | 378.74 | 36.456 | 0.76 | 0.716 | 0.68 | 0.0 | 0.0 | 0.0 | 8.608 |
| 2024 | reinvest_k1 | 1 | 1 | 1.0 | 1.0 | 4.0 | 0.0 | 5.0 | 4.0 | 9.0 | 294.475 | 3.525 | 258.464 | 36.012 | 105.351 | 1133.12 | 1573.815 | 1416.789 | 1757.572 | 0.538 | 205.609 | 1645.179 | 1472.559 | 1824.815 | 1644.7 | 1630.52 | 29.0 | 418.88 | 36.497 | 0.796 | 0.692 | 0.644 | 66.134 | 1.0 | 1.0 | 13.987 |
| 2024 | reinvest_k2 | 1 | 1 | 2.0 | 2.0 | 8.0 | 2.0 | 5.0 | 4.0 | 8.0 | 296.253 | 1.747 | 255.698 | 40.555 | 104.754 | 1190.62 | 1571.762 | 1419.082 | 1729.869 | 0.535 | 208.716 | 1641.222 | 1481.502 | 1804.352 | 1584.7 | 1585.82 | 38.0 | 393.38 | 47.357 | 0.86 | 0.756 | 0.736 | 66.134 | 1.0 | 1.0 | 13.987 |
| 2024 | reinvest_k3 | 1 | 1 | 3.0 | 3.0 | 12.0 | 4.0 | 5.0 | 5.0 | 9.0 | 294.475 | 3.525 | 264.39 | 30.086 | 108.796 | 1197.02 | 1633.252 | 1464.69 | 1801.813 | 0.649 | 240.319 | 1697.257 | 1525.513 | 1853.19 | 1575.0 | 1613.82 | 31.0 | 427.48 | 48.485 | 0.852 | 0.76 | 0.716 | 66.134 | 1.0 | 1.0 | 13.987 |

## Interpretation boundaries

- The option search is greedy and uses a bounded marginal-utility shortlist.
- A forced player must remain on the nominal preseason bench.
- Other starter and bench slots may change during the conditional full solve.
- Keeper surplus never enters current-season fantasy-point evaluation.
