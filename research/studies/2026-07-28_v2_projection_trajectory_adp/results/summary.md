# Projection Trajectory and Logged-ADP Results

Negative deltas favor the challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `trajectory_log_adp_equal_thirds` | 3.0930 |
| `trajectory_equal_thirds` | 3.0949 |
| `log_adp_equal_thirds` | 3.0963 |
| `incumbent_equal_thirds` | 3.1001 |
| `trajectory_tree_average` | 3.1112 |
| `trajectory_log_adp_tree_average` | 3.1115 |
| `incumbent_tree_average` | 3.1143 |
| `log_adp_tree_average` | 3.1147 |
| `trajectory_log_adp_random_forest` | 3.1203 |
| `trajectory_random_forest` | 3.1203 |
| `trajectory_lightgbm` | 3.1210 |
| `trajectory_log_adp_lightgbm` | 3.1215 |
| `trajectory_log_adp_lasso` | 3.1227 |
| `incumbent_lightgbm` | 3.1230 |
| `log_adp_lightgbm` | 3.1238 |
| `log_adp_random_forest` | 3.1242 |
| `incumbent_random_forest` | 3.1243 |
| `log_adp_lasso` | 3.1265 |
| `trajectory_lasso` | 3.1523 |
| `incumbent_lasso` | 3.1585 |

## Paired season comparisons

| Challenger | Reference | Delta | Recent | 95% interval | Wins | Sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| `trajectory_log_adp_lasso` | `incumbent_lasso` | -0.0358 | -0.0135 | [-0.0523, -0.0199] | 8/9 | 0.0078 |
| `log_adp_lasso` | `incumbent_lasso` | -0.0319 | -0.0102 | [-0.0500, -0.0162] | 8/9 | 0.0078 |
| `trajectory_log_adp_equal_thirds` | `trajectory_log_adp_tree_average` | -0.0185 | -0.0112 | [-0.0292, -0.0098] | 8/9 | 0.0078 |
| `log_adp_equal_thirds` | `log_adp_tree_average` | -0.0184 | -0.0100 | [-0.0286, -0.0100] | 8/9 | 0.0078 |
| `trajectory_equal_thirds` | `trajectory_tree_average` | -0.0163 | -0.0141 | [-0.0259, -0.0079] | 8/9 | 0.0117 |
| `trajectory_log_adp_equal_thirds` | `incumbent_equal_thirds` | -0.0071 | +0.0025 | [-0.0129, -0.0012] | 7/9 | 0.0547 |
| `trajectory_lasso` | `incumbent_lasso` | -0.0062 | -0.0029 | [-0.0155, +0.0014] | 5/9 | 0.2031 |
| `trajectory_equal_thirds` | `incumbent_equal_thirds` | -0.0051 | -0.0013 | [-0.0093, -0.0007] | 7/9 | 0.0586 |
| `trajectory_log_adp_random_forest` | `incumbent_random_forest` | -0.0040 | +0.0016 | [-0.0089, +0.0013] | 6/9 | 0.2109 |
| `trajectory_random_forest` | `incumbent_random_forest` | -0.0040 | +0.0016 | [-0.0088, +0.0012] | 6/9 | 0.2109 |
| `log_adp_equal_thirds` | `incumbent_equal_thirds` | -0.0037 | +0.0020 | [-0.0084, +0.0007] | 7/9 | 0.1406 |
| `trajectory_tree_average` | `incumbent_tree_average` | -0.0031 | +0.0018 | [-0.0072, +0.0009] | 7/9 | 0.1680 |
| `trajectory_log_adp_tree_average` | `incumbent_tree_average` | -0.0028 | +0.0026 | [-0.0070, +0.0013] | 6/9 | 0.2266 |
| `trajectory_lightgbm` | `incumbent_lightgbm` | -0.0020 | +0.0012 | [-0.0067, +0.0027] | 6/9 | 0.4102 |
| `trajectory_log_adp_lightgbm` | `incumbent_lightgbm` | -0.0015 | +0.0027 | [-0.0060, +0.0030] | 7/9 | 0.5039 |
| `log_adp_random_forest` | `incumbent_random_forest` | -0.0000 | +0.0000 | [-0.0001, +0.0000] | 6/9 | 0.2578 |
| `log_adp_tree_average` | `incumbent_tree_average` | +0.0004 | +0.0010 | [-0.0006, +0.0014] | 2/9 | 0.4609 |
| `log_adp_lightgbm` | `incumbent_lightgbm` | +0.0008 | +0.0019 | [-0.0011, +0.0027] | 2/9 | 0.4297 |
