# V2 Locked Final Validation Findings

> Superseded data lineage: this replay used two mislabeled FantasyPros WR
> seasons and the pre-correction identity/provider-scoring foundation. Retain
> it as historical evidence only. The accepted corrective replay is documented
> in `../../2026-07-29_v2_identity_scoring_revalidation/results/findings.md`.

## Decision

Freeze `v2_conditional_ppg_2026_candidate_v1` as the 2026 **DK shadow**
candidate. It is not yet a production projection or a beta-league point
center.

The locked conditional-PPG primary is the fixed equal-third pooled
Lasso/random-forest/LightGBM blend over the reviewed 31-feature incumbent,
five preseason projection-trajectory fields, and four position controls.
The participation primary is pooled shallow LightGBM over the reviewed
19-feature participation set plus position controls.

## Whole-season evidence

All 2017-2025 forecast seasons were regenerated with fits, hyperparameter
selection, routing, and interval calibration restricted to earlier seasons.

| Comparison | Pooled RMSE | Mean season delta | Season wins | 2023-25 delta |
|---|---:|---:|---:|---:|
| locked primary | 3.0941 | -0.0847 vs expert | 9/9 | -0.0720 |
| no-history gap route | 3.0905 | -0.0036 vs primary | 9/9 | -0.0026 |
| logged-ADP Lasso route | 3.0914 | -0.0027 vs primary | 5/9 | +0.0023 |
| projection/full history router | 3.0915 | -0.0028 vs primary | 3/9 | -0.0024 |
| WR/TE QB-style route | 3.0927 | -0.0015 vs primary | 6/9 | -0.0044 |

The primary improvement over the recalibrated expert baseline has a
season-bootstrap interval of `[-0.1019, -0.0686]`. The genuinely-no-history
gap route also clears the whole-season check with interval
`[-0.0050, -0.0022]`, but the absolute gain is tiny. Preserve it as a locked
secondary component; do not combine secondary routes after seeing these
results or claim material economic alpha from 0.0036 RMSE.

Logged ADP, the general projection/full router, and the WR/TE QB-style route
do not clear the full temporal/stability standard. Keep their component
columns for diagnosis, not as the 2026 primary.

Participation LightGBM records 0.1215 pooled Brier versus 0.1355 for logistic
and 0.2434 for the prior position rate.

## Calibration

Do not apply a point-calibration overlay. The uncalibrated primary has 3.0941
pooled RMSE. Every strictly-prior expanding intercept or affine policy worsens
pooled RMSE by 0.0032-0.0167, even though several improve the 2023-2025 window.
That pattern is recent-only drift evidence, not a stable promotion result.

Strict-prior residual coverage is 0.497 for P25-P75 and 0.793 for P10-P90.
Those intervals are valid shadow diagnostics, but weekly optimizers should
continue to obtain joint downside/upside and missed-game behavior from one
matched donor path.

## 2026 shadow

The final run fits through 2025 and publishes 751 unique 2026 player rows:
720 have a conditional-PPG center and all 751 have a participation
probability. The run and exact feature/model specifications are persisted
under:

- lock `v2_conditional_ppg_2026_candidate_v1`;
- model run `v2_locked_final_20260729T034400Z_61d266cf`; and
- feature run `milestone_3_20260729T034246Z_ae57edb4`.

Production projections, template tables, and optimizer inputs were not
changed.

## Template handoff boundary

The V2 point prediction is only the center for a matched donor's weekly path.
The weekly builder must recompute
`active_ppg_resid = template_active_ppg - V2_point_center` and draw that
residual with the same donor path. Adding an independent model residual is
prohibited: the audit shows it would double residual variance. The
reconstruction error is below `1.8e-15`.

The initial production audit identified missing canonical handoff keys and a
DK-versus-beta scoring mismatch. Both gates were addressed in the subsequent
2026-07-29 follow-up:

1. `Best_Ball_Weekly_Templates` and `Best_Ball_Weekly_Player_Map` now carry
   canonical `player_key` plus match provenance. The key audit joins 100% of
   DK historical templates and current player-map rows with no unmatched row.
2. Beta now has a separate fully rebuilt scoring/model lineage under
   `Projection_V2_beta.sqlite3`. It must remain separate from the DK database
   because scoring-dependent projection-value tables do not include league in
   their keys.

The DK V2 target differs from the weekly template's first-16/activity-semantic
`active_ppg` by 0.233 MAE. This is a target-semantic distinction, not an
identity or fantasy-scoring mismatch.

## Identity defects fixed during the replay

- Inferred draft year no longer overrides the active career window. This
  prevents 2024-2026 Frank Gore Jr. evidence from attaching to retired Frank
  Gore.
- A redundant provisional identity is reconciled to one compatible confirmed
  identity, removing duplicate Michael Thomas WR season rows.
- Ambiguous legitimate same-name players remain separate canonical identities
  and are exposed by downstream name-only audits.

These corrections reduced the shadow spine from 14,075 to 14,068 rows without
dropping canonical player-seasons.
