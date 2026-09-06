# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_nffc_v1`
- Run: `v2_locked_final_nffc_20260828T192300Z_ddcb796e`
- Feature run: `milestone_3_20260828T190611Z_c28a39de`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 783
- Current conditional-PPG centers: 634

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_history_gap_no_history_route` | 3.0838 |
| `conditional_ppg_qb_style_wrte_route` | 3.0852 |
| `conditional_ppg_log_lasso_tree_blend` | 3.0855 |
| `conditional_ppg_primary_blend` | 3.0870 |
| `conditional_ppg_projection_history_router` | 3.0905 |
| `conditional_ppg_random_forest` | 3.1134 |
| `conditional_ppg_lightgbm` | 3.1149 |
| `conditional_ppg_lasso` | 3.1303 |
| `expert_recalibrated` | 3.1760 |
| `expert_team_game` | 4.4322 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0882 | 9/9 | -0.0734 | [-0.1112, -0.0656] |
| `log_lasso_vs_primary` | -0.0015 | 6/9 | +0.0039 | [-0.0055, +0.0028] |
| `history_gap_route_vs_primary` | -0.0031 | 6/9 | -0.0016 | [-0.0072, +0.0001] |
| `projection_router_vs_primary` | +0.0033 | 2/9 | +0.0033 | [-0.0003, +0.0076] |
| `qb_style_route_vs_primary` | -0.0019 | 6/9 | -0.0048 | [-0.0040, +0.0002] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1222 |
| `participation_logistic` | 0.1373 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.495
- Strict-prior P10-P90 residual coverage: 0.792
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
