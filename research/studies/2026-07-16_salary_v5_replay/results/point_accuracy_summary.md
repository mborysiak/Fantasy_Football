# Salary v5 Point-Accuracy Readout

## Rolling player accuracy

| method | prediction_scale | period | player_years | mean_residual | mae | rmse | r2 |
|---|---|---|---|---|---|---|---|
| v1 | normalized | all_years | 644 | -0.706 | 4.307 | 6.276 | 0.941 |
| v1 | normalized | replay_development_2022_2024 | 389 | -0.480 | 4.388 | 6.346 | 0.936 |
| v1 | normalized | temporal_check_2025 | 137 | -0.474 | 3.726 | 5.850 | 0.955 |
| v1 | raw | all_years | 644 | 0.071 | 4.187 | 6.180 | 0.943 |
| v1 | raw | replay_development_2022_2024 | 389 | 0.341 | 4.212 | 6.242 | 0.938 |
| v1 | raw | temporal_check_2025 | 137 | 0.559 | 3.740 | 5.828 | 0.955 |
| v3 | normalized | all_years | 645 | -0.596 | 4.455 | 6.252 | 0.941 |
| v3 | normalized | replay_development_2022_2024 | 389 | -0.365 | 4.448 | 6.312 | 0.937 |
| v3 | normalized | temporal_check_2025 | 138 | -0.419 | 4.121 | 5.957 | 0.953 |
| v3 | raw | all_years | 645 | -0.748 | 4.462 | 6.235 | 0.942 |
| v3 | raw | replay_development_2022_2024 | 389 | -0.345 | 4.433 | 6.264 | 0.938 |
| v3 | raw | temporal_check_2025 | 138 | -0.789 | 4.132 | 5.910 | 0.954 |
| v5 | normalized | all_years | 645 | -0.612 | 4.271 | 6.197 | 0.942 |
| v5 | normalized | replay_development_2022_2024 | 389 | -0.341 | 4.261 | 6.043 | 0.942 |
| v5 | normalized | temporal_check_2025 | 138 | -0.439 | 3.750 | 5.753 | 0.956 |
| v5 | raw | all_years | 645 | -0.264 | 4.169 | 6.176 | 0.943 |
| v5 | raw | replay_development_2022_2024 | 389 | -0.114 | 4.222 | 6.045 | 0.942 |
| v5 | raw | temporal_check_2025 | 138 | 0.175 | 3.549 | 5.739 | 0.956 |

## v5 raw market coherence

| year | available_budget | pre_normalized_total | raw_minus_budget | pred_salary_shift | post_normalized_total |
|---|---|---|---|---|---|
| 2022.000 | 2886.000 | 2770.024 | -115.976 | 0.853 | 2886.000 |
| 2023.000 | 2705.000 | 2759.518 | 54.518 | -0.405 | 2705.000 |
| 2024.000 | 3013.000 | 2980.475 | -32.525 | 0.236 | 3013.000 |
| 2025.000 | 3169.000 | 3082.502 | -86.498 | 0.613 | 3169.000 |

v5 raw predictions are not forced to match the budget. The additive shift is the final exact reconciliation and does not use realized target-auction spending.
