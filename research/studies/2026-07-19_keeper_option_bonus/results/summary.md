# Keeper Option Bonus Results

250 paired trials per origin with 250 construction and evaluation contexts.
Keeper rules: two slots, +$10 per keeper year, maximum three keeper years.

## Policy means across origins

| policy | forecast_ev | forecast_p10 | actual_points | actual_playoff_points | bench_forecast_spend | top3_spend_share | predicted_keeper_option_top2 | realized_next_keeper_surplus | realized_next_keeper_hits | actual_cap_feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| bench0 | 1610.685 | 1457.964 | 1586.114 | 411.753 | 35.923 | 0.584 | 85.222 | 32.206 | 0.658 | 0.058 |
| current_bench025 | 1609.726 | 1457.583 | 1583.751 | 410.553 | 37.634 | 0.577 | 84.971 | 33.739 | 0.702 | 0.049 |
| keeper_0p01 | 1618.857 | 1466.339 | 1593.666 | 411.902 | 31.155 | 0.591 | 113.701 | 30.149 | 0.604 | 0.047 |
| keeper_10p0 | 1619.862 | 1467.665 | 1601.364 | 413.889 | 34.754 | 0.587 | 117.044 | 30.687 | 0.685 | 0.023 |
| keeper_1p0 | 1620.221 | 1467.878 | 1596.726 | 412.050 | 32.647 | 0.591 | 115.568 | 31.053 | 0.651 | 0.040 |
| keeper_engine0 | 1610.710 | 1457.868 | 1586.631 | 412.074 | 35.939 | 0.584 | 85.330 | 32.158 | 0.657 | 0.056 |
| keeper_tiebreak | 1618.611 | 1466.161 | 1593.392 | 411.845 | 31.193 | 0.591 | 113.703 | 30.370 | 0.605 | 0.045 |

## Paired effects versus current bench weight 0.25

| policy | period | roster_changed_rate | mean_forecast_ev_effect | mean_forecast_p10_effect | mean_actual_points_effect | mean_actual_playoff_points_effect | mean_bench_forecast_spend_effect | mean_predicted_keeper_option_top2_effect | mean_realized_next_keeper_surplus_effect | mean_realized_next_keeper_hits_effect | mean_actual_cap_feasible_effect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bench0 | current_development_2022_2024 | 0.855 | 0.776 | 0.173 | -0.806 | 1.048 | -1.724 | -0.523 | -1.682 | -0.043 | 0.021 |
| bench0 | current_temporal_2025 | 0.796 | 1.510 | 1.008 | 11.871 | 1.657 | -1.671 | 2.573 |  |  | -0.028 |
| bench0 | keeper_development_2022_2023 | 0.866 | 0.896 | -0.098 | -0.617 | 2.870 | -1.800 | -0.677 | -0.866 | -0.035 | 0.018 |
| bench0 | keeper_temporal_2024 | 0.832 | 0.536 | 0.713 | -1.184 | -2.597 | -1.571 | -0.214 | -3.276 | -0.060 | 0.028 |
| bench0 | keeper_unrealized_2025 | 0.796 | 1.510 | 1.008 | 11.871 | 1.657 | -1.671 | 2.573 |  |  | -0.028 |
| keeper_0p01 | current_development_2022_2024 | 0.997 | 8.204 | 7.455 | 6.446 | 3.672 | -5.099 | 27.581 | -3.460 | -0.093 | 0.005 |
| keeper_0p01 | current_temporal_2025 | 0.996 | 11.914 | 12.662 | 20.322 | -5.619 | -10.619 | 32.181 |  |  | -0.024 |
| keeper_0p01 | keeper_development_2022_2023 | 0.996 | 8.129 | 7.103 | 15.067 | 6.621 | -3.974 | 30.294 | -8.383 | -0.183 | 0.004 |
| keeper_0p01 | keeper_temporal_2024 | 1.000 | 8.354 | 8.157 | -10.798 | -2.224 | -7.351 | 22.155 | 6.208 | 0.084 | 0.008 |
| keeper_0p01 | keeper_unrealized_2025 | 0.996 | 11.914 | 12.662 | 20.322 | -5.619 | -10.619 | 32.181 |  |  | -0.024 |
| keeper_10p0 | current_development_2022_2024 | 1.000 | 9.603 | 9.302 | 13.086 | 4.770 | -0.270 | 32.021 | -3.209 | -0.019 | -0.021 |
| keeper_10p0 | current_temporal_2025 | 0.996 | 11.734 | 12.424 | 31.192 | -0.966 | -10.710 | 32.228 |  |  | -0.040 |
| keeper_10p0 | keeper_development_2022_2023 | 1.000 | 9.894 | 9.624 | 17.330 | 3.921 | 3.547 | 36.633 | -5.587 | -0.033 | -0.032 |
| keeper_10p0 | keeper_temporal_2024 | 1.000 | 9.023 | 8.659 | 4.598 | 6.468 | -7.903 | 22.798 | 1.464 | 0.008 | 0.000 |
| keeper_10p0 | keeper_unrealized_2025 | 0.996 | 11.734 | 12.424 | 31.192 | -0.966 | -10.710 | 32.228 |  |  | -0.040 |
| keeper_1p0 | current_development_2022_2024 | 0.997 | 9.584 | 8.800 | 10.993 | 3.994 | -3.174 | 30.057 | -2.531 | -0.049 | 0.001 |
| keeper_1p0 | current_temporal_2025 | 0.996 | 13.230 | 14.779 | 18.922 | -5.992 | -10.424 | 32.220 |  |  | -0.040 |
| keeper_1p0 | keeper_development_2022_2023 | 0.996 | 9.854 | 8.438 | 19.340 | 6.109 | -1.008 | 33.852 | -6.415 | -0.108 | 0.000 |
| keeper_1p0 | keeper_temporal_2024 | 1.000 | 9.042 | 9.524 | -5.703 | -0.235 | -7.506 | 22.467 | 5.097 | 0.068 | 0.004 |
| keeper_1p0 | keeper_unrealized_2025 | 0.996 | 13.230 | 14.779 | 18.922 | -5.992 | -10.424 | 32.220 |  |  | -0.040 |
| keeper_engine0 | current_development_2022_2024 | 0.868 | 0.933 | 0.245 | -0.014 | 1.391 | -1.726 | -0.445 | -1.758 | -0.045 | 0.017 |
| keeper_engine0 | current_temporal_2025 | 0.820 | 1.136 | 0.406 | 11.564 | 1.912 | -1.600 | 2.771 |  |  | -0.024 |
| keeper_engine0 | keeper_development_2022_2023 | 0.876 | 1.227 | 0.283 | -0.057 | 2.711 | -1.765 | -0.694 | -1.688 | -0.049 | 0.012 |
| keeper_engine0 | keeper_temporal_2024 | 0.852 | 0.345 | 0.169 | 0.073 | -1.247 | -1.648 | 0.053 | -1.894 | -0.036 | 0.028 |
| keeper_engine0 | keeper_unrealized_2025 | 0.820 | 1.136 | 0.406 | 11.564 | 1.912 | -1.600 | 2.771 |  |  | -0.024 |
| keeper_tiebreak | current_development_2022_2024 | 0.997 | 7.991 | 7.282 | 6.107 | 3.615 | -5.015 | 27.583 | -3.237 | -0.092 | 0.003 |
| keeper_tiebreak | current_temporal_2025 | 0.996 | 11.566 | 12.466 | 20.246 | -5.675 | -10.721 | 32.181 |  |  | -0.024 |
| keeper_tiebreak | keeper_development_2022_2023 | 0.996 | 7.743 | 6.778 | 14.654 | 6.494 | -3.839 | 30.283 | -8.405 | -0.187 | 0.004 |
| keeper_tiebreak | keeper_temporal_2024 | 1.000 | 8.487 | 8.291 | -10.988 | -2.143 | -7.367 | 22.183 | 6.912 | 0.096 | 0.000 |
| keeper_tiebreak | keeper_unrealized_2025 | 0.996 | 11.566 | 12.466 | 20.246 | -5.675 | -10.721 | 32.181 |  |  | -0.024 |

## Interpretation boundaries

- Current-season evaluation never includes keeper utility as fantasy points.
- The three-year construction payoff assumes a next-year hit persists; first-year realized surplus is primary.
- Historical keeper cost uses observed acquisition salary when available, so affordability remains a required companion metric.
- Four current seasons and three realized next-season origins are the independent outcome units.
- Frozen legacy salary and next-year uncertainty methods differ by origin and are not the current v5 production surface.
