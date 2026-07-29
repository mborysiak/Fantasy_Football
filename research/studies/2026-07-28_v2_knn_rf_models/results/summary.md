# KNN and Random-Forest Results

Negative deltas favor the challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `full_rf_lgbm_average` | 3.1143 |
| `full_lightgbm` | 3.1230 |
| `full_random_forest` | 3.1242 |
| `projection_core_rf_lgbm_average` | 3.1296 |
| `projection_core_lightgbm` | 3.1326 |
| `projection_core_random_forest` | 3.1423 |
| `projection_core_knn_lgbm_average` | 3.1518 |
| `full_knn_lgbm_average` | 3.1682 |
| `projection_core_knn` | 3.2438 |
| `full_knn` | 3.3194 |
| `expert_baseline` | 4.1628 |

## Paired season comparisons

| Challenger | Reference | Delta | Recent delta | 95% interval | Wins |
|---|---|---:|---:|---:|---:|
| `projection_core_knn` | `projection_core_lightgbm` | +0.1111 | +0.0863 | [+0.0566, +0.1640] | 1/9 |
| `full_knn` | `full_lightgbm` | +0.1964 | +0.1435 | [+0.1513, +0.2364] | 0/9 |
| `projection_core_random_forest` | `projection_core_lightgbm` | +0.0097 | +0.0026 | [-0.0041, +0.0272] | 4/9 |
| `full_random_forest` | `full_lightgbm` | +0.0012 | -0.0094 | [-0.0181, +0.0198] | 5/9 |
| `projection_core_knn_lgbm_average` | `projection_core_lightgbm` | +0.0191 | +0.0052 | [-0.0023, +0.0408] | 4/9 |
| `full_knn_lgbm_average` | `full_lightgbm` | +0.0452 | +0.0147 | [+0.0222, +0.0664] | 2/9 |
| `projection_core_rf_lgbm_average` | `projection_core_lightgbm` | -0.0031 | -0.0062 | [-0.0102, +0.0058] | 6/9 |
| `full_rf_lgbm_average` | `full_lightgbm` | -0.0086 | -0.0124 | [-0.0182, +0.0005] | 5/9 |
