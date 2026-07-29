# Projection-Anchored History Gap Findings

## Decision

Keep projection-anchored history gaps as the governed representation for
future sparse-history challengers, but do not replace the 31-feature full
incumbent globally from this evidence. The gap construction fixes the semantic
problem—no history now means zero adjustment to the player's own current
projection—but its model gain is not temporally stable.

Carry the raw-gap and reliability-shrunk variants into the final
history-routing and template-distribution replay. Do not select the
eight-game shrinkage constant from this same OOF result.

## Construction

The 144-feature mart adds a separate 13-feature
`residual_history_gap_challenger_v1` manifest. For prior-year, prior-three-year,
and career production:

```text
history gap = historical PPG - current preseason expert PPG
```

The current expert anchor uses active-game consensus when available and
otherwise team-game consensus. Missing history receives a zero gap, not a
pooled player median. Availability, log opportunity games, neutral prior
projection residual, and neutral recency are explicit. The secondary variant
multiplies each gap by `games / (games + 8)`.

All 3,696 validation rows have non-null gap, reliability, availability, and
sample-size fields. The 631 rows with no career opportunity have exactly zero
career gap.

## Pooled results

| Method | Incumbent RMSE | Raw-gap RMSE | Shrunken-gap RMSE |
|---|---:|---:|---:|
| Lasso | 3.1585 | 3.1602 | 3.1611 |
| Random forest | 3.1243 | 3.1255 | 3.1233 |
| LightGBM | 3.1230 | 3.1134 | 3.1141 |
| RF/LightGBM average | 3.1143 | 3.1099 | 3.1091 |
| Lasso/RF/LightGBM equal thirds | 3.1001 | **3.0972** | 3.0974 |
| Causal Lasso/tree blend | 3.1028 | 3.1002 | 3.1002 |

The user's linear-model concern is valid semantically, but the pooled median
plus missingness indicator was not causing a measurable overall Lasso loss.
Replacing the absolute history fields worsens Lasso by 0.0018-0.0027 RMSE,
with both intervals crossing zero. The selected Lasso alpha is unchanged in
all five folds.

Raw-gap LightGBM improves 0.0096 RMSE, but wins only four of nine seasons, has
interval `[-0.0212, +0.0010]`, and is 0.0036 worse on mean 2023-2025 season
RMSE. The raw equal-third blend improves only 0.0029 versus the incumbent
blend, has interval `[-0.0091, +0.0035]`, and is 0.0058 worse recently.
Shrunken gaps have the same instability. Early seasons supply most of the
pooled improvement.

## Intended sparse-history behavior

The raw-gap equal-third blend versus the incumbent equal-third blend changes
RMSE by:

- rookie: -0.0105;
- non-rookie with no career history: -0.0422;
- prior 1-3 games: +0.0132;
- prior 4-7 games: +0.0115;
- returning after a missed calendar year: +0.0176; and
- prior 8+ games: -0.0042.

This fixes the intended no-history representation and has favorable point
estimates for rookies/no-career players. It does not support treating every
limited or injury-shortened history the same way. A two-game observed history
is information, not missing history, and the tested gap variants do not
improve that group.

## Interpretation

For a linear model, the existing missingness indicator can learn an offset
that largely cancels the pooled-median placeholder. Projection anchoring is
cleaner and improves coefficient meaning, but changing the representation of
observed veteran history also changes regularized estimation. The result is a
trade: better no-history behavior without an overall or recent Lasso gain.

For tree models, gap features can expose useful projection-relative splits,
but the apparent LightGBM gain is concentrated in older provider eras. This is
consistent with current expert projections containing more of the relevant
history adjustment than older projections did.

## Next step

Keep the full incumbent and three-way blend as the primary point-model
finalists. In the final whole-season replay, compare:

1. the incumbent equal-third blend;
2. the projection-only/full-model causal history router; and
3. a prespecified sparse-history option that uses projection-anchored gaps
   only for genuinely no-history rows.

Require the history route to use only prior-origin evidence and to improve
current-provider-era calibration or template behavior before promotion.
