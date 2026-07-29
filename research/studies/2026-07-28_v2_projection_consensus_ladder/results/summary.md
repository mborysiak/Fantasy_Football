# Projection Consensus Ladder Results

Negative deltas favor the challenger.

## Provider stack on realized team-game PPG

| Method | RMSE |
|---|---:|
| `causal_provider_stack_global` | 2.7476 |
| `causal_provider_stack_position` | 2.7481 |
| `configured_median` | 2.7572 |

## Conditional-PPG OOF models

| Model | RMSE |
|---|---:|
| `full_lightgbm_base` | 3.1230 |
| `full_lightgbm_plus_active` | 3.1249 |
| `full_lightgbm_plus_stack` | 3.1266 |
| `full_lightgbm_plus_room` | 3.1273 |
| `projection_only_lightgbm_plus_active` | 3.1307 |
| `projection_only_lightgbm_plus_room` | 3.1316 |
| `full_lightgbm_plus_all_projection` | 3.1321 |
| `projection_only_lightgbm_core` | 3.1326 |
| `full_lightgbm_plus_targeted` | 3.1333 |
| `projection_only_lightgbm_median` | 3.1360 |
| `projection_only_lightgbm_plus_shape` | 3.1361 |
| `projection_only_lightgbm_plus_stack` | 3.1407 |
| `projection_only_lasso_plus_active` | 3.1500 |
| `projection_only_lasso_core` | 3.1519 |
| `projection_only_lightgbm_plus_all` | 3.1530 |
| `projection_only_lasso_plus_shape` | 3.1543 |
| `projection_only_lasso_plus_room` | 3.1543 |
| `projection_only_lasso_plus_all` | 3.1647 |
| `projection_only_lasso_plus_stack` | 3.1714 |
| `projection_only_lasso_median` | 3.3427 |

## Fold-identical comparisons

| Stage | Model | Variant | Reference | Delta | 95% interval | Wins |
|---|---|---|---|---:|---:|---:|
| `full` | `lightgbm` | `plus_active` | `full_lightgbm_base` | +0.0020 | [+0.0000, +0.0061] | 0/9 |
| `full` | `lightgbm` | `plus_stack` | `full_lightgbm_base` | +0.0036 | [-0.0103, +0.0168] | 4/9 |
| `full` | `lightgbm` | `plus_room` | `full_lightgbm_base` | +0.0043 | [-0.0031, +0.0129] | 3/9 |
| `full` | `lightgbm` | `plus_all_projection` | `full_lightgbm_base` | +0.0091 | [-0.0103, +0.0281] | 3/9 |
| `full` | `lightgbm` | `plus_targeted` | `full_lightgbm_base` | +0.0103 | [-0.0044, +0.0266] | 5/9 |
| `projection_only` | `lasso` | `core` | `projection_only_lasso_median` | -0.1907 | [-0.2109, -0.1681] | 9/9 |
| `projection_only` | `lasso` | `plus_active` | `projection_only_lasso_core` | -0.0019 | [-0.0130, +0.0066] | 4/9 |
| `projection_only` | `lasso` | `plus_shape` | `projection_only_lasso_core` | +0.0024 | [-0.0079, +0.0115] | 2/9 |
| `projection_only` | `lasso` | `plus_room` | `projection_only_lasso_core` | +0.0024 | [-0.0016, +0.0072] | 4/9 |
| `projection_only` | `lasso` | `plus_all` | `projection_only_lasso_core` | +0.0128 | [-0.0101, +0.0349] | 2/9 |
| `projection_only` | `lasso` | `plus_stack` | `projection_only_lasso_core` | +0.0195 | [+0.0025, +0.0367] | 1/9 |
| `projection_only` | `lightgbm` | `core` | `projection_only_lightgbm_median` | -0.0034 | [-0.0260, +0.0225] | 6/9 |
| `projection_only` | `lightgbm` | `plus_active` | `projection_only_lightgbm_core` | -0.0019 | [-0.0052, +0.0010] | 5/9 |
| `projection_only` | `lightgbm` | `plus_room` | `projection_only_lightgbm_core` | -0.0011 | [-0.0087, +0.0085] | 7/9 |
| `projection_only` | `lightgbm` | `plus_shape` | `projection_only_lightgbm_core` | +0.0034 | [-0.0131, +0.0196] | 5/9 |
| `projection_only` | `lightgbm` | `plus_stack` | `projection_only_lightgbm_core` | +0.0081 | [-0.0060, +0.0201] | 3/9 |
| `projection_only` | `lightgbm` | `plus_all` | `projection_only_lightgbm_core` | +0.0203 | [+0.0050, +0.0360] | 2/9 |
