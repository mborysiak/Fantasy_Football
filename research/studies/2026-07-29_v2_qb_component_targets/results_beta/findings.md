# QB Component-Target Findings (beta)

Negative deltas favor independently modeling passing and rushing PPG and summing the predictions.

## Pooled 2017-2025

| Method | RMSE | MAE | Bias |
|---|---:|---:|---:|
| `qb_component_sum_plus_prior_other` | 3.9706 | 3.1447 | +0.1501 |
| `qb_direct_total_blend` | 3.9976 | 3.1773 | +0.0753 |
| `conditional_ppg_primary_blend` | 4.0072 | 3.1488 | -0.1860 |
| `qb_component_sum_blend` | 4.0077 | 3.1801 | +0.5722 |
| `qb_direct_total_random_forest` | 4.0086 | 3.1906 | +0.0881 |
| `qb_component_sum_lightgbm` | 4.0199 | 3.1879 | +0.5834 |
| `qb_direct_total_lasso` | 4.0320 | 3.1707 | +0.0625 |
| `qb_direct_total_lightgbm` | 4.0471 | 3.2548 | +0.0752 |
| `qb_component_sum_random_forest` | 4.0555 | 3.2201 | +0.5955 |
| `qb_component_sum_lasso` | 4.1185 | 3.2814 | +0.5376 |
| `expert_recalibrated` | 4.2729 | 3.3826 | -0.5389 |
| `expert_direct_team_game` | 6.7569 | 4.9832 | -2.9206 |
| `expert_component_sum` | 6.7583 | 4.9836 | -2.9330 |
| `expert_component_sum_plus_prior_other` | 6.9523 | 5.0925 | -3.3551 |

## Key same-model comparison

The component blend plus its strictly-prior other-points adjustment changes RMSE by **-0.0270** versus the QB-only direct-total blend, wins 5/9 seasons, and has player-cluster interval [-0.1261, +0.0630].

Versus the locked pooled production candidate, the same component challenger changes RMSE by **-0.0366**.

## Passing and rushing targets

| Target | Method | RMSE | MAE | Bias |
|---|---|---:|---:|---:|
| pass | `lightgbm` | 3.6906 | 2.9218 | +0.3983 |
| pass | `random_forest` | 3.7136 | 2.9338 | +0.3942 |
| pass | `lasso` | 3.7787 | 2.9910 | +0.3229 |
| pass | `expert_component` | 5.8340 | 4.3204 | -2.4511 |
| rush | `lasso` | 1.5971 | 1.1510 | -0.1698 |
| rush | `random_forest` | 1.6077 | 1.1291 | -0.1831 |
| rush | `lightgbm` | 1.6333 | 1.1489 | -0.1993 |
| rush | `expert_component` | 1.9600 | 1.3094 | -0.8664 |

## Interpretation

This is a target-decomposition test, not a template-weight test. Production remains unchanged. The direct-total and component models use identical QB samples, features, model families, grids, and strictly-prior selection rules.
