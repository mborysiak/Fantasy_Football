# Linear and Tree Blend Results

Negative deltas favor the challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `projection_active_lasso_rf_lgbm_equal_thirds` | 3.0997 |
| `full_lasso_rf_lgbm_equal_thirds` | 3.1000 |
| `projection_core_lasso_rf_lgbm_equal_thirds` | 3.1001 |
| `causal_full_lasso_tree_average` | 3.1027 |
| `full_lasso_rf_average` | 3.1028 |
| `projection_active_lasso_rf_average` | 3.1031 |
| `causal_full_lasso_rf_lgbm` | 3.1031 |
| `projection_core_lasso_rf_average` | 3.1038 |
| `causal_full_lasso_rf` | 3.1053 |
| `projection_active_lasso_lgbm_average` | 3.1064 |
| `projection_core_lasso_lgbm_average` | 3.1070 |
| `full_lasso_lgbm_average` | 3.1094 |
| `causal_full_lasso_lgbm` | 3.1100 |
| `full_rf_lgbm_average` | 3.1143 |
| `full_lightgbm` | 3.1230 |
| `full_random_forest` | 3.1242 |
| `projection_active_lasso` | 3.1500 |
| `projection_core_lasso` | 3.1519 |
| `full_lasso` | 3.1584 |

## Paired season comparisons

| Challenger | Reference | Delta | Post-warmup | Recent | 95% interval | Wins |
|---|---|---:|---:|---:|---:|---:|
| `full_lasso_rf_average` | `full_random_forest` | -0.0213 | -0.0250 | -0.0134 | [-0.0363, -0.0095] | 9/9 |
| `projection_active_lasso_rf_average` | `full_random_forest` | -0.0210 | -0.0279 | -0.0089 | [-0.0410, -0.0028] | 6/9 |
| `projection_core_lasso_rf_average` | `full_random_forest` | -0.0204 | -0.0271 | -0.0018 | [-0.0417, -0.0021] | 7/9 |
| `causal_full_lasso_rf` | `full_random_forest` | -0.0189 | -0.0245 | -0.0162 | [-0.0305, -0.0089] | 7/9 |
| `projection_active_lasso_lgbm_average` | `full_lightgbm` | -0.0166 | -0.0236 | -0.0102 | [-0.0322, -0.0023] | 8/9 |
| `projection_core_lasso_lgbm_average` | `full_lightgbm` | -0.0159 | -0.0226 | -0.0025 | [-0.0327, -0.0001] | 7/9 |
| `projection_active_lasso_rf_lgbm_equal_thirds` | `full_rf_lgbm_average` | -0.0146 | -0.0191 | -0.0093 | [-0.0260, -0.0043] | 6/9 |
| `full_lasso_rf_lgbm_equal_thirds` | `full_rf_lgbm_average` | -0.0143 | -0.0161 | -0.0112 | [-0.0225, -0.0075] | 9/9 |
| `projection_core_lasso_rf_lgbm_equal_thirds` | `full_rf_lgbm_average` | -0.0142 | -0.0186 | -0.0048 | [-0.0262, -0.0036] | 7/9 |
| `full_lasso_lgbm_average` | `full_lightgbm` | -0.0136 | -0.0167 | -0.0118 | [-0.0275, -0.0003] | 7/9 |
| `causal_full_lasso_lgbm` | `full_lightgbm` | -0.0130 | -0.0168 | -0.0133 | [-0.0221, -0.0049] | 6/9 |
| `causal_full_lasso_tree_average` | `full_rf_lgbm_average` | -0.0116 | -0.0151 | -0.0108 | [-0.0196, -0.0049] | 7/9 |
| `causal_full_lasso_rf_lgbm` | `full_rf_lgbm_average` | -0.0112 | -0.0145 | -0.0106 | [-0.0189, -0.0045] | 7/9 |
