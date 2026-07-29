# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_v1`
- Run: `v2_locked_final_20260729T034400Z_61d266cf`
- Feature run: `milestone_3_20260729T034246Z_ae57edb4`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 751
- Current conditional-PPG centers: 720

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_history_gap_no_history_route` | 3.0905 |
| `conditional_ppg_log_lasso_tree_blend` | 3.0914 |
| `conditional_ppg_projection_history_router` | 3.0915 |
| `conditional_ppg_qb_style_wrte_route` | 3.0927 |
| `conditional_ppg_primary_blend` | 3.0941 |
| `conditional_ppg_random_forest` | 3.1173 |
| `conditional_ppg_lightgbm` | 3.1182 |
| `conditional_ppg_lasso` | 3.1493 |
| `expert_recalibrated` | 3.1793 |
| `expert_team_game` | 4.1991 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0847 | 9/9 | -0.0720 | [-0.1019, -0.0686] |
| `log_lasso_vs_primary` | -0.0027 | 5/9 | +0.0023 | [-0.0072, +0.0017] |
| `history_gap_route_vs_primary` | -0.0036 | 9/9 | -0.0026 | [-0.0050, -0.0022] |
| `projection_router_vs_primary` | -0.0028 | 3/9 | -0.0024 | [-0.0106, +0.0040] |
| `qb_style_route_vs_primary` | -0.0015 | 6/9 | -0.0044 | [-0.0040, +0.0011] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1215 |
| `participation_logistic` | 0.1355 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.497
- Strict-prior P10-P90 residual coverage: 0.793
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
