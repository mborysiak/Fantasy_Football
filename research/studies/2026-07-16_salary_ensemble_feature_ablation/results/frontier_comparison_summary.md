# Paired v1 versus v2 Salary Frontier

All effects are v2 minus v1. Positive feasibility and heldout-cap effects favor v2; negative overage and spend effects favor v2.

Managed forecast points are preseason simulated EV. Raw actual points are audit-only because historically unaffordable rosters remain in that column.

| period | chance_level | roster_changed_rate | mean_managed_forecast_season_points_effect_v2_minus_v1 | mean_heldout_cap_probability_effect_v2_minus_v1 | mean_actual_cap_feasible_effect_v2_minus_v1 | mean_actual_cap_overage_effect_v2_minus_v1 | mean_actual_salary_spend_effect_v2_minus_v1 | mean_raw_actual_points_audit_only_effect_v2_minus_v1 |
|---|---|---|---|---|---|---|---|---|
| development_2022_2024 | 0.6 | 0.792 | 2.381 | -0.008 | -0.004 | 1.233 | 1.247 | 2.777 |
| development_2022_2024 | 0.7 | 0.800 | 2.103 | -0.014 | -0.009 | 1.432 | 1.508 | 2.485 |
| development_2022_2024 | 0.8 | 0.788 | 2.188 | -0.012 | 0.005 | 0.728 | 0.667 | 4.335 |
| development_2022_2024 | 0.9 | 0.783 | 1.646 | -0.007 | 0.008 | 0.980 | 0.840 | 1.842 |
| temporal_check_2025 | 0.6 | 0.784 | -1.655 | 0.001 | 0.024 | -0.940 | -1.060 | 5.903 |
| temporal_check_2025 | 0.7 | 0.820 | -3.490 | -0.001 | -0.004 | -0.456 | -0.328 | 4.896 |
| temporal_check_2025 | 0.8 | 0.836 | 0.007 | -0.013 | 0.024 | 0.584 | 0.740 | 6.726 |
| temporal_check_2025 | 0.9 | 0.764 | -2.006 | -0.005 | 0.004 | -1.012 | -1.584 | 1.278 |

The replay changes only the rolling salary method. Projection draws, managed-value contexts, optimizer settings, chance thresholds, and random seeds are paired.
