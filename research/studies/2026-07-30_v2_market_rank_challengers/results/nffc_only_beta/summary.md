# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_beta_v1`
- Run: `v2_locked_final_beta_20260730T140449Z_ef55415d`
- Feature run: `milestone_3_20260730T140041Z_8666f6b2`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 745
- Current conditional-PPG centers: 673

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_projection_history_router` | 2.8799 |
| `conditional_ppg_qb_style_wrte_route` | 2.8827 |
| `conditional_ppg_history_gap_no_history_route` | 2.8828 |
| `conditional_ppg_log_lasso_tree_blend` | 2.8832 |
| `conditional_ppg_primary_blend` | 2.8841 |
| `conditional_ppg_random_forest` | 2.8951 |
| `conditional_ppg_lightgbm` | 2.9079 |
| `conditional_ppg_lasso` | 2.9479 |
| `expert_recalibrated` | 2.9600 |
| `expert_team_game` | 3.6273 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0752 | 9/9 | -0.0762 | [-0.0941, -0.0590] |
| `log_lasso_vs_primary` | -0.0011 | 6/9 | +0.0005 | [-0.0062, +0.0038] |
| `history_gap_route_vs_primary` | -0.0013 | 5/9 | -0.0014 | [-0.0039, +0.0008] |
| `projection_router_vs_primary` | -0.0043 | 5/9 | -0.0087 | [-0.0108, +0.0006] |
| `qb_style_route_vs_primary` | -0.0014 | 6/9 | -0.0034 | [-0.0031, +0.0003] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1256 |
| `participation_logistic` | 0.1391 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.492
- Strict-prior P10-P90 residual coverage: 0.794
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
