# Projection Feature Challenger Results

Negative deltas favor the added feature family.

## Pooled OOF

| Model | RMSE |
|---|---:|
| `direct_lightgbm_plus_provider_projection` | 3.1145 |
| `direct_lightgbm_plus_all` | 3.1220 |
| `direct_lightgbm_base` | 3.1230 |
| `direct_lightgbm_plus_projection_shape` | 3.1278 |
| `direct_lightgbm_plus_projection_disagreement` | 3.1287 |
| `direct_lasso_plus_projection_shape` | 3.1509 |
| `direct_lasso_base` | 3.1556 |
| `direct_lasso_plus_projection_disagreement` | 3.1652 |
| `direct_lasso_plus_all` | 3.1765 |
| `direct_lasso_plus_provider_projection` | 3.1831 |

## Fold-identical family comparisons

| Model | Variant | Added | Pooled delta | Mean season delta | 95% interval | Wins |
|---|---|---:|---:|---:|---:|---:|
| `lasso` | `plus_projection_shape` | 10 | -0.0046 | -0.0043 | [-0.0168, +0.0069] | 5/9 |
| `lasso` | `plus_projection_disagreement` | 8 | +0.0096 | +0.0095 | [-0.0002, +0.0212] | 3/9 |
| `lasso` | `plus_all` | 26 | +0.0209 | +0.0197 | [-0.0084, +0.0579] | 4/9 |
| `lasso` | `plus_provider_projection` | 8 | +0.0275 | +0.0255 | [-0.0034, +0.0653] | 2/9 |
| `lightgbm` | `plus_provider_projection` | 8 | -0.0085 | -0.0085 | [-0.0198, +0.0021] | 6/9 |
| `lightgbm` | `plus_all` | 26 | -0.0009 | -0.0010 | [-0.0130, +0.0116] | 5/9 |
| `lightgbm` | `plus_projection_shape` | 10 | +0.0049 | +0.0050 | [-0.0042, +0.0138] | 3/9 |
| `lightgbm` | `plus_projection_disagreement` | 8 | +0.0057 | +0.0059 | [-0.0044, +0.0167] | 4/9 |
