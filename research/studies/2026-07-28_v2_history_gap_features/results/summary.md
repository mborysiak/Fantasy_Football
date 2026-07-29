# Projection-Anchored History Gap Results

Negative deltas favor the challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `gap_raw_equal_thirds` | 3.0972 |
| `gap_shrunk_equal_thirds` | 3.0974 |
| `incumbent_equal_thirds` | 3.1001 |
| `gap_shrunk_causal_lasso_tree` | 3.1002 |
| `gap_raw_causal_lasso_tree` | 3.1002 |
| `incumbent_causal_lasso_tree` | 3.1028 |
| `gap_shrunk_tree_average` | 3.1091 |
| `gap_raw_tree_average` | 3.1099 |
| `gap_raw_lightgbm` | 3.1134 |
| `gap_shrunk_lightgbm` | 3.1141 |
| `incumbent_tree_average` | 3.1143 |
| `incumbent_lightgbm` | 3.1230 |
| `gap_shrunk_random_forest` | 3.1233 |
| `incumbent_random_forest` | 3.1243 |
| `gap_raw_random_forest` | 3.1255 |
| `incumbent_lasso` | 3.1585 |
| `gap_raw_lasso` | 3.1602 |
| `gap_shrunk_lasso` | 3.1611 |

## Paired season comparisons

| Challenger | Reference | Delta | Recent | 95% interval | Wins | Sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| `incumbent_equal_thirds` | `incumbent_tree_average` | -0.0143 | -0.0110 | [-0.0225, -0.0074] | 9/9 | 0.0039 |
| `gap_raw_equal_thirds` | `gap_raw_tree_average` | -0.0128 | -0.0105 | [-0.0217, -0.0056] | 8/9 | 0.0117 |
| `gap_shrunk_equal_thirds` | `gap_shrunk_tree_average` | -0.0116 | -0.0090 | [-0.0207, -0.0038] | 7/9 | 0.0273 |
| `gap_raw_lightgbm` | `incumbent_lightgbm` | -0.0096 | +0.0036 | [-0.0212, +0.0010] | 4/9 | 0.1719 |
| `gap_shrunk_lightgbm` | `incumbent_lightgbm` | -0.0088 | +0.0028 | [-0.0278, +0.0096] | 5/9 | 0.4375 |
| `gap_shrunk_tree_average` | `incumbent_tree_average` | -0.0053 | +0.0005 | [-0.0161, +0.0045] | 5/9 | 0.4609 |
| `gap_raw_tree_average` | `incumbent_tree_average` | -0.0044 | +0.0053 | [-0.0129, +0.0032] | 5/9 | 0.4141 |
| `gap_raw_equal_thirds` | `incumbent_equal_thirds` | -0.0029 | +0.0058 | [-0.0091, +0.0035] | 5/9 | 0.4492 |
| `gap_shrunk_equal_thirds` | `incumbent_equal_thirds` | -0.0026 | +0.0026 | [-0.0092, +0.0041] | 6/9 | 0.5391 |
| `gap_shrunk_causal_lasso_tree` | `incumbent_causal_lasso_tree` | -0.0026 | +0.0025 | [-0.0104, +0.0050] | 4/9 | 0.6172 |
| `gap_raw_causal_lasso_tree` | `incumbent_causal_lasso_tree` | -0.0026 | +0.0059 | [-0.0094, +0.0040] | 5/9 | 0.5586 |
| `gap_shrunk_random_forest` | `incumbent_random_forest` | -0.0010 | -0.0008 | [-0.0112, +0.0086] | 3/9 | 0.8086 |
| `gap_raw_random_forest` | `incumbent_random_forest` | +0.0013 | +0.0076 | [-0.0068, +0.0089] | 3/9 | 0.7188 |
| `gap_raw_lasso` | `incumbent_lasso` | +0.0018 | +0.0088 | [-0.0059, +0.0102] | 3/9 | 0.6875 |
| `gap_shrunk_lasso` | `incumbent_lasso` | +0.0027 | +0.0078 | [-0.0044, +0.0114] | 5/9 | 0.6094 |
