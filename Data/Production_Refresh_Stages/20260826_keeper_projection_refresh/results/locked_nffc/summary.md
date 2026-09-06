# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_nffc_v1`
- Run: `v2_locked_final_nffc_20260826T202835Z_e7f1ae4d`
- Feature run: `milestone_3_20260826T202043Z_1e4e26b9`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 785
- Current conditional-PPG centers: 635

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_history_gap_no_history_route` | 3.0828 |
| `conditional_ppg_log_lasso_tree_blend` | 3.0842 |
| `conditional_ppg_qb_style_wrte_route` | 3.0845 |
| `conditional_ppg_primary_blend` | 3.0857 |
| `conditional_ppg_projection_history_router` | 3.0891 |
| `conditional_ppg_lightgbm` | 3.1119 |
| `conditional_ppg_random_forest` | 3.1131 |
| `conditional_ppg_lasso` | 3.1304 |
| `expert_recalibrated` | 3.1760 |
| `expert_team_game` | 4.4322 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0894 | 9/9 | -0.0746 | [-0.1125, -0.0670] |
| `log_lasso_vs_primary` | -0.0014 | 6/9 | +0.0037 | [-0.0054, +0.0028] |
| `history_gap_route_vs_primary` | -0.0028 | 4/9 | -0.0007 | [-0.0067, +0.0003] |
| `projection_router_vs_primary` | +0.0032 | 2/9 | +0.0051 | [-0.0007, +0.0079] |
| `qb_style_route_vs_primary` | -0.0012 | 7/9 | -0.0030 | [-0.0029, +0.0004] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1219 |
| `participation_logistic` | 0.1373 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.494
- Strict-prior P10-P90 residual coverage: 0.793
- Residual reconstruction maximum absolute error: 3.55e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
