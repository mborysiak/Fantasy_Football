# Findings

## Data contract

- Recalculate provider fantasy points from normalized raw components under the
  configured DK base scoring.
- If exactly one required position-specific component is absent, fill it only
  from the player-season median of at least two other providers. This repairs
  4,010 FFA rows, primarily the historically omitted receptions field.
- If more than one required component is absent, leave the provider unscored.
- Never use provider-published fantasy totals or PPG in consensus, individual
  provider features, or room/team point context.
- Allow every configured provider to enter the equal-weight consensus
  immediately, but mask provider-specific model columns until the provider has
  three prior projection seasons. Thus FantasyPoints, FFF, FanDuel, and PFF
  remain unavailable as learned provider adjustments for 2026.

The final 2026 shadow mart has 720 configured-score consensus rows versus 727
under the earlier published-total fallback. The 2017-2025 conditional-PPG OOF
population has 3,696 rows versus 3,701 previously.

## Fold-identical results

| Model | Added family | RMSE | Delta vs same-model base |
|---|---|---:|---:|
| deterministic LightGBM | provider projection | 3.1145 | -0.0085 |
| deterministic LightGBM | all 26 | 3.1220 | -0.0009 |
| deterministic LightGBM | none | 3.1230 | 0.0000 |
| deterministic LightGBM | projection shape | 3.1278 | +0.0049 |
| deterministic LightGBM | disagreement | 3.1287 | +0.0057 |
| Lasso | projection shape | 3.1509 | -0.0046 |
| Lasso | none | 3.1556 | 0.0000 |
| Lasso | disagreement | 3.1652 | +0.0096 |
| Lasso | all 26 | 3.1765 | +0.0209 |
| Lasso | provider projection | 3.1831 | +0.0275 |

The LightGBM provider-family delta wins six of nine seasons, but its
season-bootstrap interval `[-0.0198, +0.0021]` crosses zero. Projection shape
is neutral for Lasso and worse for LightGBM. Component disagreement is not
helpful in either model. Combining all 26 additions dilutes the provider-only
tree result.

## Provider attribution

With the same deterministic full-column LightGBM:

| Provider column | Delta | 95% season interval | Wins |
|---|---:|---:|---:|
| FFToday | -0.0073 | [-0.0136, -0.0004] | 6/9 |
| FantasyPros | -0.0024 | [-0.0067, +0.0012] | 5/9 |
| FantasyData | -0.0023 | [-0.0098, +0.0053] | 5/9 |
| FFA | -0.0001 | [-0.0075, +0.0078] | 6/9 |
| FantasyPoints / FFF / FanDuel / PFF | 0.0000 | gated | 0/9 |

FFToday accounts for most of the family point estimate. Its improvement is
concentrated in QB (`-0.0452`) and RB (`-0.0113`) and is neutral to slightly
worse for TE/WR. The individual-provider test is exploratory across multiple
columns, so it is a candidate for a nested/static confirmation rather than a
reason to hand-weight FFToday now.

The same-sample provider accuracy diagnostic makes FantasyPoints look strong
in 2025, but that is one season and the consensus includes the provider itself.
It is descriptive only and does not override the three-prior-season gate.

## Rookie and sparse-history boundary

The provider family does not improve rookies (`+0.0021`) or other no-history
players (`+0.0163`) in LightGBM. The full 26-feature addition is also worse for
rookies (`+0.0105`). Projection-shape additions are essentially flat for
rookies in Lasso.

These features therefore do not replace the existing rookie approach:
same-season consensus, ADP, draft capital, age/experience availability, room
context, and explicit missing-history indicators.

## Decision

- Keep the 31-feature incumbent manifest unchanged.
- Retain all 26 additions in `residual_projection_challenger_v1` for governed
  research and future provider history.
- Do not promote rate, opportunity-total, or disagreement features.
- Carry the gated provider family, especially FFToday, into a later
  nested/static confirmation; do not manually upweight a provider.
- Treat configured-only scoring and guarded component imputation as correctness
  fixes independent of model lift.

