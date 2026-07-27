# Bench Option And Waiver-Hurdle Results

250 paired trials per origin with 250 construction and evaluation contexts.
All forecast evaluation uses the unchanged current projected waiver baseline.

## Policy means across origins

| policy | forecast_ev | forecast_p10 | actual_points | actual_playoff_points | bench_forecast_spend | top3_spend_share | bench_sustained_15_hits | actual_cap_feasible |
|---|---|---|---|---|---|---|---|---|
| bench0 | 1616.909 | 1462.481 | 1581.500 | 412.971 | 35.910 | 0.585 | 1.664 | 0.074 |
| current_bench025 | 1616.042 | 1461.849 | 1573.244 | 412.541 | 37.722 | 0.577 | 1.622 | 0.082 |
| hurdle_plus1 | 1614.530 | 1461.290 | 1571.043 | 412.016 | 40.522 | 0.569 | 1.726 | 0.068 |
| hurdle_plus2 | 1596.026 | 1444.360 | 1580.394 | 410.464 | 38.012 | 0.600 | 1.642 | 0.118 |
| hurdle_plus3 | 1573.113 | 1420.817 | 1578.270 | 407.985 | 31.664 | 0.664 | 1.305 | 0.232 |
| sustained_option025 | 1626.764 | 1472.481 | 1597.996 | 432.877 | 38.405 | 0.577 | 2.075 | 0.042 |
| sustained_option050 | 1624.730 | 1470.770 | 1594.774 | 436.629 | 40.451 | 0.570 | 2.186 | 0.044 |

## Paired effects versus current bench weight 0.25

| policy | period | roster_changed_rate | mean_forecast_ev_effect | mean_forecast_p10_effect | mean_actual_points_effect | mean_actual_playoff_points_effect | mean_bench_forecast_spend_effect | mean_top3_spend_share_effect | mean_bench_sustained_15_hits_effect | mean_actual_cap_feasible_effect |
|---|---|---|---|---|---|---|---|---|---|---|
| bench0 | development_2022_2024 | 0.863 | 0.403 | 0.604 | 6.585 | 0.907 | -1.361 | 0.006 | 0.048 | -0.004 |
| bench0 | temporal_check_2025 | 0.860 | 2.259 | 0.715 | 13.268 | -1.003 | -3.165 | 0.014 | 0.024 | -0.020 |
| hurdle_plus1 | development_2022_2024 | 0.800 | -1.396 | -0.592 | -2.918 | -1.380 | 2.924 | -0.010 | 0.113 | -0.019 |
| hurdle_plus1 | temporal_check_2025 | 0.720 | -1.860 | -0.462 | -0.048 | 2.039 | 2.428 | -0.005 | 0.076 | 0.000 |
| hurdle_plus2 | development_2022_2024 | 0.988 | -25.734 | -23.063 | 10.124 | -2.935 | -0.948 | 0.033 | 0.016 | 0.056 |
| hurdle_plus2 | temporal_check_2025 | 0.924 | -2.866 | -0.768 | -1.770 | 0.497 | 4.005 | -0.009 | 0.032 | -0.024 |
| hurdle_plus3 | development_2022_2024 | 1.000 | -50.848 | -49.493 | 6.910 | -5.364 | -8.685 | 0.112 | -0.412 | 0.211 |
| hurdle_plus3 | temporal_check_2025 | 0.992 | -19.174 | -15.650 | -0.626 | -2.133 | 1.825 | 0.008 | -0.032 | -0.032 |
| sustained_option025 | development_2022_2024 | 1.000 | 13.529 | 13.851 | 32.072 | 25.716 | 1.049 | -0.003 | 0.765 | -0.032 |
| sustained_option025 | temporal_check_2025 | 1.000 | 2.303 | 0.975 | 2.795 | 4.195 | -0.416 | 0.009 | -0.484 | -0.064 |
| sustained_option050 | development_2022_2024 | 1.000 | 11.321 | 12.200 | 27.381 | 30.252 | 3.253 | -0.011 | 0.921 | -0.032 |
| sustained_option050 | temporal_check_2025 | 1.000 | 0.789 | -0.920 | 3.977 | 5.594 | 1.159 | 0.006 | -0.508 | -0.056 |

## Interpretation boundaries

- Four seasons provide four outcome units; trial counts measure Monte Carlo stability.
- Actual point comparisons include historically unaffordable rosters and must be read with cap feasibility.
- The sustained option is a strategy-utility sensitivity, not literal additional lineup points.
- Realized waiver scoring remains optimistic and lacks opponent claim competition or transaction persistence.
- The construction refinement retains the live mean-profile/OR-played-mask approximation.
