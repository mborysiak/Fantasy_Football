# QB Component-Target Findings (dk)

Negative deltas favor independently modeling passing and rushing PPG and summing the predictions.

## Pooled 2017-2025

| Method | RMSE | MAE | Bias |
|---|---:|---:|---:|
| `qb_direct_total_blend` | 3.4581 | 2.7342 | +0.1797 |
| `qb_direct_total_random_forest` | 3.4590 | 2.7469 | +0.1960 |
| `conditional_ppg_primary_blend` | 3.4765 | 2.7348 | -0.1703 |
| `qb_direct_total_lightgbm` | 3.4767 | 2.7526 | +0.2082 |
| `qb_component_sum_plus_prior_other` | 3.4996 | 2.7464 | +0.2466 |
| `qb_component_sum_blend` | 3.5175 | 2.7655 | +0.4397 |
| `qb_component_sum_random_forest` | 3.5244 | 2.7721 | +0.4565 |
| `qb_direct_total_lasso` | 3.5356 | 2.7621 | +0.1349 |
| `qb_component_sum_lightgbm` | 3.5516 | 2.7964 | +0.4426 |
| `qb_component_sum_lasso` | 3.6338 | 2.8368 | +0.4201 |
| `expert_recalibrated` | 3.7175 | 2.9608 | -0.4835 |
| `expert_direct_team_game` | 7.6287 | 5.5701 | -4.5497 |
| `expert_component_sum` | 7.6355 | 5.5755 | -4.5613 |
| `expert_component_sum_plus_prior_other` | 7.7526 | 5.6687 | -4.7544 |

## Key same-model comparison

The component blend plus its strictly-prior other-points adjustment changes RMSE by **+0.0414** versus the QB-only direct-total blend, wins 4/9 seasons, and has player-cluster interval [-0.0442, +0.1265].

Versus the locked pooled production candidate, the same component challenger changes RMSE by **+0.0231**.

## Passing and rushing targets

| Target | Method | RMSE | MAE | Bias |
|---|---|---:|---:|---:|
| pass | `random_forest` | 3.2526 | 2.5542 | +0.4137 |
| pass | `lightgbm` | 3.2578 | 2.5532 | +0.4349 |
| pass | `lasso` | 3.2835 | 2.5617 | +0.3837 |
| pass | `expert_component` | 6.5679 | 4.7775 | -3.8045 |
| rush | `random_forest` | 1.5257 | 1.0913 | -0.1304 |
| rush | `lightgbm` | 1.5599 | 1.1124 | -0.1655 |
| rush | `lasso` | 1.6005 | 1.1121 | -0.1368 |
| rush | `expert_component` | 1.9614 | 1.2959 | -0.9300 |

## Interpretation

This is a target-decomposition test, not a template-weight test. Production remains unchanged. The direct-total and component models use identical QB samples, features, model families, grids, and strictly-prior selection rules.
