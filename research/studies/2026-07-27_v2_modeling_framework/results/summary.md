# V2 M4A Initial OOF Results

This is a shadow-only comparison. It does not replace production projections,
templates, or optimizer inputs.

- Model run: `milestone_4a_20260728T050818Z_86b06f1f`
- Feature run: `milestone_3_20260728T044840Z_88f9f8f3`
- OOF window: 2017-2025
- Conditional-PPG fold rows: 3701
- Participation fold rows: 7877
- Compared specifications: 18

## Conditional PPG

Pooled OOF RMSE; lower is better. Delta is model minus the active-game-when-
available expert hybrid.

| Model | Value | Baseline | Delta |
|---|---:|---:|---:|
| `direct_lgbm_shallow` | 3.1443 | 4.2244 | -1.0801 |
| `residual_lgbm_shallow` | 3.1567 | 4.2244 | -1.0677 |
| `direct_ridge_full` | 3.1731 | 4.2244 | -1.0513 |
| `residual_ridge_full` | 3.1827 | 4.2244 | -1.0417 |
| `consensus_recalibrated_ridge` | 3.1996 | 4.2244 | -1.0248 |
| `residual_ridge_compact` | 3.2611 | 4.2244 | -0.9633 |
| `residual_ridge_kbest` | 3.3626 | 4.2244 | -0.8618 |
| `residual_ridge_pca` | 3.4112 | 4.2244 | -0.8132 |
| `residual_ridge_agg` | 3.6231 | 4.2244 | -0.6012 |
| `expert_consensus_hybrid` | 4.2244 | 4.2244 | +0.0000 |
| `expert_team_game_consensus` | 4.2681 | 4.2244 | +0.0437 |

## Participation

Pooled OOF Brier score; lower is better. Delta is model minus the leakage-safe
prior-position-rate baseline.

| Model | Value | Baseline | Delta |
|---|---:|---:|---:|
| `participation_lgbm_shallow` | 0.1222 | 0.2433 | -0.1211 |
| `participation_logistic_full` | 0.1366 | 0.2433 | -0.1067 |
| `participation_logistic_compact` | 0.1384 | 0.2433 | -0.1049 |
| `participation_logistic_kbest` | 0.1445 | 0.2433 | -0.0988 |
| `participation_logistic_agg` | 0.1507 | 0.2433 | -0.0926 |
| `participation_logistic_pca` | 0.1527 | 0.2433 | -0.0906 |
| `prior_position_rate` | 0.2433 | 0.2433 | +0.0000 |

## Interpretation Boundary

Five folds cover every validation season. Each OOF prediction is fit only on
seasons strictly earlier than that player-season. Hyperparameters may use the
other four folds' rolling predictions, but never the held fold. PCA,
agglomeration, and univariate selection are isolated pipeline challengers;
they are not stacked together.
