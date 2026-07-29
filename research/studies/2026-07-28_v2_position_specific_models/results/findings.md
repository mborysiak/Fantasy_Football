# Position-Specific Conditional-PPG Findings

## Decision

Do not replace either pooled model with four independent QB/RB/WR/TE models.
The full QB/RB fits are effectively tied with their pooled slices, while the
loss of shared data materially hurts WR and TE. Retain the pooled projection
core. A two-model QB/RB+WR+TE split also ties rather than improves the pooled
full model. Carry the three-role full model (QB, RB, and WR+TE) only as a
whole-season temporal-robustness finalist: it ties the pooled full model over
2017-2025 and has a favorable recent point estimate, but does not establish an
overall improvement.

## Design

The experiment fits the same deterministic shallow LightGBM and feature sets
as the pooled comparators:

- projection core: 22 projection consensus/component features;
- full: the governed 31-feature `residual_candidate_v1` manifest;
- four independent components: QB, RB, WR, and TE; and
- three role components: QB, RB, and a shared WR+TE receiving model; and
- two components: QB and a shared RB+WR+TE skill-position model.

Each component uses five deterministic folds and predicts a 2017-2025 target
season only from earlier seasons. Constant position indicators are removed
from one-position models; the WR+TE component retains its two position
indicators. The component predictions are stitched after fitting and compared
on the exact 3,696-player OOF population used by the pooled models.

## Overall results

| Model | RMSE | Delta versus corresponding pooled model |
|---|---:|---:|
| Pooled full | 3.1230 | reference |
| Three-role full | 3.1231 | +0.0002 |
| Two-group full | 3.1237 | +0.0007 |
| Pooled projection core | 3.1326 | reference |
| Two-group projection core | 3.1384 | +0.0058 |
| Four-position full | 3.1389 | +0.0160 |
| Three-role projection core | 3.1458 | +0.0132 |
| Four-position projection core | 3.1752 | +0.0425 |

Four-position projection core is worse in eight of nine seasons, with a
season-bootstrap interval entirely above zero (+0.0192 to +0.0693). The
four-position full model is less clearly worse, but its +0.0160 point estimate
does not support the added complexity. The three-role full model is an exact
practical tie: +0.0002 RMSE, four of nine season wins, and interval -0.0280 to
+0.0281.

The two-group full model is also a practical tie: +0.0007 RMSE, five of nine
season wins, and interval -0.0108 to +0.0138. It is worse than pooled full in
the 2023-2025 window by a mean 0.0101 RMSE per season, so it does not displace
the three-role finalist. Its projection-core version is 0.0058 worse than
pooled and wins only two of nine seasons.

Pooling WR and TE is materially safer than splitting them. The three-role
model improves on the four-position version by 0.0158 RMSE for the full set
and 0.0294 for projection core; those improvements win eight and nine of nine
seasons, respectively.

## Position results

| Position | Pooled full | Four-position full | Three-role full |
|---|---:|---:|---:|
| QB | 3.5283 | 3.5254 | 3.5254 |
| RB | 3.4987 | 3.4952 | 3.4952 |
| TE | 2.3057 | 2.3242 | 2.3066 |
| WR | 3.1249 | 3.1602 | 3.1284 |

The independent full QB and RB improvements are only 0.0029 and 0.0034 RMSE.
They are too small to establish that complete separation beats the pooled
model. Independent WR and TE fits lose 0.0352 and 0.0184. A shared WR+TE
component recovers nearly all of that loss, which supports the hypothesis
that receiving roles share useful structure even when QB and RB relationships
differ.

Projection core benefits more clearly from global pooling. Every fully
separate position fit is worse than its pooled position slice. The shared
WR+TE projection model improves WR by 0.0122 but still cannot offset the
separate-QB and separate-RB losses.

## Temporal result

The three-role full model scores 3.0792 RMSE in 2023-2025 versus 3.1024 for the
pooled full model. Its season deltas are -0.0677 in 2023, +0.0365 in 2024, and
-0.0386 in 2025. This is directionally interesting, particularly because
recent RB, WR, and TE slices improve, but three seasons are not enough to
override the flat nine-season result.

## Interpretation and next step

The relevant features can differ by position without requiring fully
independent models. Pooled LightGBM can learn position-conditioned splits while
borrowing calibration and nonlinear structure from a much larger sample. The
current evidence says that the data-sharing benefit dominates for projection
core and for WR/TE full models.

Keep the pooled full and pooled projection-core models as primary finalists.
Include the three-role full model, not four independent models, in the pending
whole-season/calibration replay. Promote it only if the recent result survives
prespecified temporal and position guardrails. If it does, inspect
position-specific feature importance and compact feature subsets before
accepting the extra model surface.
