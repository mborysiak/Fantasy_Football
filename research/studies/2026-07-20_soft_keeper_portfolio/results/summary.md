# Soft Whole-Bench Keeper Portfolio Results

No age, role, or option-count quotas. The policy maximizes expected-best
one-year keeper surplus across all five bench players subject to
construction-bank mean and p10 protection.

## Paired effects versus current-only control

| year | roster_changed_rate | accepted_option_additions | option_effective_count_effect | active_option_count_effect | bench_fillin_top2_effect | starter_spend_effect | bench_spend_effect | starter_ev_effect | forecast_ev_effect | forecast_p10_effect | actual_points_effect | waiver_starts_effect | playoff_effect | predicted_best_surplus_effect | actual_best_surplus_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 0.532 | 1.144 | 0.032 | -0.004 | -0.018 | -1.55 | 1.849 | 0.563 | 0.723 | -1.352 | -1.054 | -0.744 | 2.795 | 5.879 | 3.597 |
| 2023 | 0.612 | 1.32 | 0.43 | 0.272 | 0.077 | -0.067 | 0.537 | 10.662 | 15.038 | 10.349 | 14.763 | 1.12 | 3.689 | 6.09 | 6.831 |
| 2024 | 0.448 | 0.644 | 0.016 | 0.032 | 1.063 | 5.997 | -6.157 | 2.229 | -2.185 | -1.369 | -8.61 | -0.296 | -8.909 | 3.026 | 2.811 |
| 2025 | 0.432 | 0.796 | 0.02 | -0.028 | 1.511 | 3.944 | -3.743 | 4.171 | 3.092 | 3.042 | 12.997 | 0.96 | -0.018 | 2.078 |  |

## Policy means by origin

| year | policy | trials | unique_rosters | accepted_option_additions | candidate_attempts | reoptimization_refine_swaps | bench_young_le2 | bench_young_le3 | bench_rookies | starter_changes_vs_control | bench_changes_vs_control | roster_changes_vs_control | forecast_salary_spend | unspent_budget | starter_forecast_spend | bench_forecast_spend | starter_projected_ppg_sum | starter_raw_actual_points | starter_forecast_ev | starter_forecast_p10 | starter_forecast_p90 | top3_spend_share | construction_mean_delta | construction_p10_delta | bench_fillin_total | bench_fillin_top2 | bench_fillin_second | bench_positive_fillin_count | option_positive_draw_rate | option_effective_count | option_active_count_5pct | option_top_winner_share | forecast_ev | forecast_p10 | forecast_p90 | actual_points | drafted_only_points | actual_waiver_starts | actual_playoff_points | predicted_expected_best_surplus | predicted_probability_any_hit | predicted_probability_any_10 | predicted_probability_any_20 | actual_best_keeper_surplus | actual_any_keeper_hit_10 | actual_any_keeper_hit_20 | actual_best_future_ppg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | control | 250 | 250 | 0.0 | 0.0 | 2.256 | 2.66 | 3.448 | 1.304 | 0.0 | 0.0 | 0.0 | 295.436 | 2.564 | 258.638 | 36.798 | 102.364 | 1240.382 | 1544.303 | 1377.274 | 1710.245 | 0.546 | 0.0 | 0.0 | 59.749 | 37.602 | 15.366 | 4.46 | 0.85 | 4.206 | 4.908 | 0.307 | 1623.518 | 1460.772 | 1787.943 | 1426.121 | 1438.255 | 37.044 | 365.602 | 52.125 | 0.85 | 0.784 | 0.719 | 48.468 | 1.0 | 1.0 | 13.164 |
| 2022 | soft_portfolio | 250 | 250 | 1.144 | 12.864 | 4.26 | 3.468 | 4.04 | 1.82 | 0.988 | 1.272 | 2.144 | 295.735 | 2.265 | 257.087 | 38.647 | 102.399 | 1225.114 | 1544.865 | 1377.405 | 1712.086 | 0.55 | 6.032 | 13.701 | 60.479 | 37.583 | 14.766 | 4.728 | 0.884 | 4.238 | 4.904 | 0.321 | 1624.241 | 1459.419 | 1789.848 | 1425.067 | 1441.41 | 36.3 | 368.397 | 58.004 | 0.884 | 0.829 | 0.775 | 52.065 | 1.0 | 1.0 | 13.342 |
| 2023 | control | 250 | 235 | 0.0 | 0.0 | 4.896 | 1.828 | 2.604 | 1.488 | 0.0 | 0.0 | 0.0 | 295.64 | 2.36 | 252.404 | 43.236 | 95.989 | 1304.51 | 1498.395 | 1355.179 | 1639.566 | 0.549 | 0.0 | 0.0 | 33.324 | 25.854 | 10.055 | 3.748 | 0.833 | 3.465 | 3.912 | 0.392 | 1538.247 | 1395.487 | 1680.473 | 1687.491 | 1478.224 | 51.932 | 462.876 | 41.03 | 0.833 | 0.748 | 0.672 | 82.425 | 1.0 | 1.0 | 15.99 |
| 2023 | soft_portfolio | 250 | 244 | 1.32 | 13.584 | 8.624 | 2.7 | 3.1 | 2.324 | 1.856 | 1.636 | 3.3 | 296.11 | 1.89 | 252.337 | 43.773 | 97.86 | 1313.843 | 1509.057 | 1360.417 | 1654.188 | 0.562 | 17.236 | 21.12 | 33.367 | 25.93 | 10.401 | 3.712 | 0.859 | 3.895 | 4.184 | 0.334 | 1553.286 | 1405.836 | 1699.455 | 1702.254 | 1489.379 | 53.052 | 466.565 | 47.12 | 0.859 | 0.788 | 0.727 | 89.256 | 1.0 | 1.0 | 16.373 |
| 2024 | control | 250 | 84 | 0.0 | 0.0 | 1.028 | 4.744 | 4.796 | 2.22 | 0.0 | 0.0 | 0.0 | 295.777 | 2.223 | 243.546 | 52.231 | 107.325 | 1210.765 | 1605.657 | 1459.278 | 1767.816 | 0.493 | 0.0 | 0.0 | 64.54 | 38.066 | 15.626 | 4.904 | 0.795 | 4.232 | 4.956 | 0.341 | 1682.283 | 1531.991 | 1833.633 | 1604.579 | 1620.757 | 27.568 | 428.388 | 39.263 | 0.795 | 0.742 | 0.67 | 58.526 | 0.996 | 0.984 | 13.846 |
| 2024 | soft_portfolio | 250 | 125 | 0.644 | 9.864 | 2.36 | 4.884 | 4.928 | 2.38 | 0.756 | 0.852 | 1.524 | 295.618 | 2.382 | 249.543 | 46.074 | 107.666 | 1214.974 | 1607.885 | 1459.6 | 1768.146 | 0.495 | 4.837 | 4.984 | 62.022 | 39.129 | 15.969 | 4.684 | 0.81 | 4.248 | 4.988 | 0.326 | 1680.099 | 1530.621 | 1831.334 | 1595.969 | 1626.025 | 27.272 | 419.479 | 42.289 | 0.81 | 0.757 | 0.688 | 61.337 | 0.992 | 0.988 | 13.917 |
| 2025 | control | 250 | 215 | 0.0 | 0.0 | 1.96 | 4.112 | 4.224 | 2.964 | 0.0 | 0.0 | 0.0 | 295.857 | 2.143 | 245.385 | 50.472 | 109.69 | 1289.793 | 1619.036 | 1462.814 | 1776.314 | 0.524 | 0.0 | 0.0 | 45.84 | 30.741 | 13.115 | 4.392 | 0.826 | 4.36 | 4.696 | 0.299 | 1696.215 | 1540.885 | 1852.049 | 1536.971 | 1473.096 | 35.004 | 348.661 | 32.396 | 0.826 | 0.701 | 0.635 |  |  |  |  |
| 2025 | soft_portfolio | 250 | 222 | 0.796 | 10.776 | 3.196 | 4.336 | 4.396 | 3.008 | 0.904 | 0.976 | 1.716 | 296.059 | 1.941 | 249.329 | 46.729 | 110.037 | 1292.28 | 1623.206 | 1466.941 | 1779.28 | 0.539 | 6.863 | 12.672 | 47.876 | 32.252 | 14.005 | 4.208 | 0.838 | 4.38 | 4.668 | 0.293 | 1699.308 | 1543.927 | 1853.938 | 1549.968 | 1475.403 | 35.964 | 348.643 | 34.474 | 0.838 | 0.72 | 0.652 |  |  |  |  |

## Interpretation boundaries

- Search is greedy and uses a bounded candidate shortlist.
- Search anchors coordinate full-roster reoptimization but are not
  designated keeper slots; all five final bench players receive utility.
- Gate contexts are construction data. Independent evaluation contexts
  and realized outcomes never enter selection.
