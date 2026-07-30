# Beta V2 Locked Validation Findings

> Superseded data lineage: this replay used two mislabeled FantasyPros WR
> seasons and omitted sacks from standardized beta QB provider scores. Retain
> it as historical evidence only. The accepted corrective replay is documented
> in `../../2026-07-29_v2_identity_scoring_revalidation/results/findings.md`.

## Decision

Freeze `v2_conditional_ppg_2026_candidate_beta_v1` as the beta-scored shadow
counterpart to the DK lock. Use the same reviewed feature sets and primary
pooled Lasso/random-forest/LightGBM equal-third architecture, but fit every
model and select every hyperparameter independently on beta-scored prior
seasons.

Do not promote any beta secondary route or point-calibration overlay.

## Data lineage

Beta uses a separate `Projection_V2_beta.sqlite3` because projection-value
tables are scoring-dependent but do not include league in their primary keys.
The beta and DK builds have exactly the same 6,559 canonical identities and
14,068 player-season feature keys, but different scoring hashes.

The scoring rebuild is material:

- historical conditional PPG changes by 1.170 points MAE versus DK;
- expert team-game PPG changes by 0.497 points MAE; and
- standardized provider season points change by 10.275 points MAE.

This confirms that reusing or linearly relabeling the DK center would not have
been acceptable.

## Whole-season evidence

Every 2017-2025 forecast, hyperparameter selection, route, and interval uses
only earlier beta seasons.

| Comparison | Pooled RMSE | Mean season delta | Season wins | 2023-25 delta |
|---|---:|---:|---:|---:|
| locked primary | 2.9109 | -0.0649 vs expert | 9/9 | -0.0675 |
| no-history gap route | 2.9094 | -0.0014 vs primary | 5/9 | -0.0003 |
| logged-ADP Lasso route | 2.9113 | +0.0006 vs primary | 5/9 | -0.0008 |
| projection/full history router | 2.9107 | -0.0002 vs primary | 2/9 | -0.0037 |
| WR/TE QB-style route | 2.9105 | -0.0004 vs primary | 6/9 | -0.0035 |

The primary-versus-expert season-bootstrap interval is
`[-0.0796, -0.0525]`. Every secondary-route interval crosses zero. The
no-history gap route that cleared the DK replay therefore remains
scoring-specific DK evidence and is not promoted for beta.

Participation LightGBM scores 0.1216 Brier versus 0.1364 for logistic and
0.2434 for the prior position rate.

## Calibration and intervals

Keep the beta point center uncalibrated. Its 2.9109 pooled RMSE is better than
every strictly-prior intercept or affine policy; those policies worsen pooled
RMSE by 0.0070-0.0155 despite some recent-period improvements.

Strict-prior P25-P75 and P10-P90 residual coverage is 0.486 and 0.793.
Weekly outcome variance should still come from one matched donor residual/path,
not an additional independent model-residual draw.

## 2026 beta shadow and template handoff

The fit-through-2025 run publishes:

- 751 unique 2026 player rows;
- 720 beta conditional-PPG centers; and
- 751 participation probabilities.

Run lineage:

- lock `v2_conditional_ppg_2026_candidate_beta_v1`;
- model run `v2_locked_final_beta_20260729T042804Z_4edfad7e`; and
- feature run `milestone_3_20260729T042626Z_54599d2e`.

Canonical `player_key` joins cover 100% of the 2,696 beta historical template
rows and all 180 current beta player-map rows. Point-center coverage is also
100% on those production populations. The beta V2 target differs from the
weekly template's first-16/activity-semantic `active_ppg` by 0.240 MAE, close
to the DK semantic difference of 0.233 and far below the earlier 1.388
cross-scoring mismatch. Residual reconstruction error is below `1.8e-15`.

Three historical converted-position rows differ between the template and V2
season labels: Cordarrelle Patterson in 2019 and 2021, and Ty Montgomery in
2022. Current beta map position agreement is 100%.

Production point predictions and optimizer inputs remain unchanged. The
canonical key columns are active in the generated weekly template and player
map tables.
