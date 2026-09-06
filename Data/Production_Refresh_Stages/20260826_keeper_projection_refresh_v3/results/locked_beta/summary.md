# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_beta_v1`
- Run: `v2_locked_final_beta_20260826T211314Z_6d4d8db9`
- Feature run: `milestone_3_20260826T210835Z_29025fcb`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 785
- Current conditional-PPG centers: 616

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_projection_history_router` | 2.8778 |
| `conditional_ppg_history_gap_no_history_route` | 2.8795 |
| `conditional_ppg_log_lasso_tree_blend` | 2.8801 |
| `conditional_ppg_qb_style_wrte_route` | 2.8804 |
| `conditional_ppg_primary_blend` | 2.8813 |
| `conditional_ppg_random_forest` | 2.8962 |
| `conditional_ppg_lightgbm` | 2.9177 |
| `conditional_ppg_lasso` | 2.9287 |
| `expert_recalibrated` | 2.9600 |
| `expert_team_game` | 3.6273 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0782 | 9/9 | -0.0861 | [-0.0995, -0.0609] |
| `log_lasso_vs_primary` | -0.0013 | 4/9 | +0.0021 | [-0.0048, +0.0017] |
| `history_gap_route_vs_primary` | -0.0018 | 7/9 | -0.0011 | [-0.0043, +0.0003] |
| `projection_router_vs_primary` | -0.0036 | 3/9 | -0.0082 | [-0.0103, +0.0007] |
| `qb_style_route_vs_primary` | -0.0009 | 6/9 | -0.0023 | [-0.0024, +0.0007] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1251 |
| `participation_logistic` | 0.1399 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.498
- Strict-prior P10-P90 residual coverage: 0.797
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
