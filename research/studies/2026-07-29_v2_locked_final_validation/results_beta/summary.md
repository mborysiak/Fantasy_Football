# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_beta_v1`
- Run: `v2_locked_final_beta_20260729T042804Z_4edfad7e`
- Feature run: `milestone_3_20260729T042626Z_54599d2e`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 751
- Current conditional-PPG centers: 720

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_history_gap_no_history_route` | 2.9094 |
| `conditional_ppg_qb_style_wrte_route` | 2.9105 |
| `conditional_ppg_projection_history_router` | 2.9107 |
| `conditional_ppg_primary_blend` | 2.9109 |
| `conditional_ppg_log_lasso_tree_blend` | 2.9113 |
| `conditional_ppg_random_forest` | 2.9307 |
| `conditional_ppg_lightgbm` | 2.9375 |
| `conditional_ppg_lasso` | 2.9736 |
| `expert_recalibrated` | 2.9759 |
| `expert_team_game` | 3.7357 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0649 | 9/9 | -0.0675 | [-0.0796, -0.0525] |
| `log_lasso_vs_primary` | +0.0006 | 5/9 | -0.0008 | [-0.0039, +0.0064] |
| `history_gap_route_vs_primary` | -0.0014 | 5/9 | -0.0003 | [-0.0050, +0.0018] |
| `projection_router_vs_primary` | -0.0002 | 2/9 | -0.0037 | [-0.0049, +0.0038] |
| `qb_style_route_vs_primary` | -0.0004 | 6/9 | -0.0035 | [-0.0026, +0.0019] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1216 |
| `participation_logistic` | 0.1364 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.486
- Strict-prior P10-P90 residual coverage: 0.793
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
