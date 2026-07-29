# Position-Aware Feature-Family Findings

## Decision

Promote no new feature family into the conditional-PPG model. None of the 16
prespecified position-family comparisons survives the season-level
false-discovery correction, and no family consistently closes the independent
WR/TE models' gap to pooled full. The strongest overall directional result is
richer QB room clarity, but its interval crosses zero. Preserve three
exploratory young-player slice patterns for the final whole-season/template
analysis rather than tuning the mean model to them.

## Design

The governed 31-feature full LightGBM was fit independently for QB, RB, WR, and
TE. Four leakage-safe families were added one at a time:

- projection relative to same-position experience peers;
- same-position teammate ADP and market-room competition;
- historically available, position-relevant team opportunity shares; and
- richer room size, gap, disagreement, QB competition, and pass-catcher room
  context.

All models use deterministic full-column LightGBM, five within-position folds,
and strictly earlier-season training. Results cover the same 3,696 OOF rows as
the pooled and separate model comparison. Exact season sign-flip tests and
Benjamini-Hochberg q-values jointly cover the 16 primary position-family
comparisons. The combined `all_targeted` variants are secondary.

## Overall results

No favorable primary comparison has a season-bootstrap interval below zero or
a BH q-value below 0.70.

| Position | Best directional family | Delta vs separate base | Delta vs pooled full | Wins | 95% interval |
|---|---|---:|---:|---:|---:|
| QB | Room clarity | -0.0126 | -0.0155 | 5/9 | [-0.0508, +0.0171] |
| RB | Room clarity | -0.0010 | -0.0044 | 5/9 | [-0.0201, +0.0195] |
| WR | Experience context | -0.0060 | +0.0293 | 5/9 | [-0.0296, +0.0154] |
| TE | Teammate ADP | -0.0042 | +0.0143 | 5/9 | [-0.0200, +0.0109] |

The richer QB room family is the only addition that meaningfully beats both
its separate-position base and the pooled full QB slice. It improves further
in 2023-2025 (-0.0301 mean season delta), but is not season-stable enough for
promotion.

The opportunity-share hypothesis does not improve conditional mean PPG.
Relative to each position base, it worsens QB by 0.0529, RB by 0.0099, and WR
by 0.0120 RMSE; TE improves only 0.0036 with a very wide interval. The QB and
WR worsening intervals are above zero. The likely explanation is redundancy:
the incumbent already contains expert PPG, raw projected volume, role point
shares, and core room structure.

Adding every targeted family together worsens QB and RB by 0.0418 and 0.0285
RMSE. It improves the already weak separate WR fit by 0.0127 but remains
0.0226 worse than pooled full.

## Young-player slices

The prespecified family-level tests do not validate a general rookie rule, but
three exploratory patterns match plausible football mechanisms:

- QB room clarity improves rookie QB RMSE from 4.1095 to 4.0121 (-0.0973) on
  44 rows, winning six of nine season slices with at least three rookies.
- WR experience context improves rookie WR RMSE from 3.4500 to 3.4158
  (-0.0342) on 225 rows, including all three 2023-2025 seasons.
- Teammate-ADP context improves rookie TE RMSE from 2.2626 to 2.2350 (-0.0277)
  and second-year TE RMSE by 0.0101, but the combined limited-history gain is
  only 0.0191 and the model remains worse than pooled full overall.

These are slice discoveries within the same OOF evidence, not independent
confirmation. Experience context does not improve rookie QB, RB, or TE mean
PPG, and richer room clarity does not improve young RB mean PPG. Therefore the
results do not support general position-specific feature promotion.

## Interpretation

The user's football intuition remains useful, but much of it is already
encoded in the incumbent projection and room variables. More detailed
opportunity and ambiguity fields may describe the *shape* of outcomes better
than their conditional mean. Rookie development, uncertain rooms, and spread
passing workloads are therefore stronger hypotheses for template matching,
variance, missed-games, and upside calibration than for another point-forecast
feature expansion.

Keep pooled full and pooled projection core as the mean-model primaries. Carry
QB room clarity and the rookie WR/young TE slice patterns as audit hypotheses
into the pending whole-season and joint-template work. Any mean-model use
requires nested or future-season confirmation; do not select the three
patterns directly from this OOF result.

