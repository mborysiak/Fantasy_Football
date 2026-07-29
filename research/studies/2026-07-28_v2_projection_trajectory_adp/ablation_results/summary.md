# Projection Trajectory Ablation Results

Negative deltas favor the challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `trajectory_3year_equal_thirds` | 3.0959 |
| `trajectory_1year_equal_thirds` | 3.0985 |
| `incumbent_equal_thirds` | 3.1001 |
| `trajectory_3year_tree_average` | 3.1136 |
| `incumbent_tree_average` | 3.1143 |
| `trajectory_1year_tree_average` | 3.1151 |
| `trajectory_3year_random_forest` | 3.1223 |
| `trajectory_1year_lightgbm` | 3.1230 |
| `incumbent_lightgbm` | 3.1230 |
| `trajectory_3year_lightgbm` | 3.1232 |
| `incumbent_random_forest` | 3.1243 |
| `trajectory_1year_random_forest` | 3.1248 |
| `trajectory_3year_lasso` | 3.1515 |
| `trajectory_1year_lasso` | 3.1539 |
| `incumbent_lasso` | 3.1585 |

## Paired season comparisons

| Challenger | Reference | Delta | Recent | 95% interval | Wins | Sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| `trajectory_3year_equal_thirds` | `trajectory_3year_tree_average` | -0.0177 | -0.0171 | [-0.0265, -0.0097] | 9/9 | 0.0039 |
| `trajectory_1year_equal_thirds` | `trajectory_1year_tree_average` | -0.0166 | -0.0138 | [-0.0261, -0.0081] | 7/9 | 0.0156 |
| `trajectory_3year_lasso` | `incumbent_lasso` | -0.0069 | -0.0051 | [-0.0151, +0.0006] | 6/9 | 0.1328 |
| `trajectory_1year_lasso` | `incumbent_lasso` | -0.0046 | -0.0004 | [-0.0134, +0.0024] | 6/9 | 0.3438 |
| `trajectory_3year_equal_thirds` | `incumbent_equal_thirds` | -0.0041 | +0.0001 | [-0.0099, +0.0020] | 6/9 | 0.2109 |
| `trajectory_3year_random_forest` | `incumbent_random_forest` | -0.0020 | +0.0084 | [-0.0110, +0.0080] | 5/9 | 0.6641 |
| `trajectory_1year_equal_thirds` | `incumbent_equal_thirds` | -0.0016 | +0.0004 | [-0.0046, +0.0017] | 7/9 | 0.3711 |
| `trajectory_3year_tree_average` | `incumbent_tree_average` | -0.0007 | +0.0062 | [-0.0071, +0.0066] | 6/9 | 0.8125 |
| `trajectory_1year_lightgbm` | `incumbent_lightgbm` | -0.0000 | +0.0041 | [-0.0047, +0.0054] | 5/9 | 0.9766 |
| `trajectory_3year_lightgbm` | `incumbent_lightgbm` | +0.0002 | +0.0028 | [-0.0047, +0.0052] | 5/9 | 0.9688 |
| `trajectory_1year_random_forest` | `incumbent_random_forest` | +0.0006 | +0.0014 | [-0.0032, +0.0044] | 4/9 | 0.7891 |
| `trajectory_1year_tree_average` | `incumbent_tree_average` | +0.0008 | +0.0033 | [-0.0026, +0.0043] | 4/9 | 0.7031 |
