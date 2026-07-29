# Projection Trajectory and Logged-ADP Findings

## Decision

Keep the five preseason projection-trajectory fields and logged ADP as
separately governed research challengers. Do not yet replace the 31-feature
incumbent or any production projection.

The full raw-ADP trajectory variant is the more stable global finalist.
Logged ADP is a supported replacement for raw ADP in Lasso, but not in the
tree models. The pooled-best trajectory-plus-log blend does not clear the
recent-period and sparse-history guardrails required for promotion.

## Semantics and coverage

The trajectory features compare preseason projection evidence only:

- current consensus team-game PPG minus exact prior-year consensus team-game
  PPG;
- current consensus team-game PPG minus a 3/2/1 recency-weighted mean of the
  prior three preseason projections;
- exact-prior availability, prior-three-year count, and prior-projection
  volatility.

No realized PPG, games, injury outcome, or current-season actual enters the
construction. Missing projection history receives a zero change plus explicit
zero availability/count. On the 3,696 common OOF rows, 3,116 (84.3%) have an
exact prior-year projection and 3,163 (85.6%) have at least one projection in
the prior three seasons.

## Fold-identical results

Negative RMSE deltas favor the challenger.

| Variant | Challenger RMSE | Incumbent RMSE | Delta | Recent delta | Season wins |
|---|---:|---:|---:|---:|---:|
| Exact one-year trajectory blend | 3.0985 | 3.1001 | -0.0016 | +0.0004 | 7/9 |
| Three-year trajectory blend | 3.0959 | 3.1001 | -0.0041 | +0.0001 | 6/9 |
| Full five-field trajectory blend | 3.0949 | 3.1001 | -0.0051 | -0.0013 | 7/9 |
| Log-ADP blend | 3.0963 | 3.1001 | -0.0037 | +0.0020 | 7/9 |
| Full trajectory plus log-ADP blend | 3.0930 | 3.1001 | -0.0071 | +0.0025 | 7/9 |

The exact current-versus-last-year projection change is directionally useful
for Lasso (-0.0046 RMSE), but is neutral to worse for the two tree models. The
three-year fields are more useful than the exact-year pair, and the combined
five-field family has the best raw-ADP blend result. Its season-bootstrap
interval is `[-0.0093, -0.0007]`, although the exact nine-season sign-flip
value is 0.0586.

Replacing raw ADP with `log1p(ADP)` improves Lasso from 3.1585 to 3.1265,
wins eight of nine seasons, and has interval `[-0.0500, -0.0162]`. It is
effectively neutral for random forest and slightly worse for LightGBM. Logged
ADP therefore belongs in the linear candidate, not as a blanket transform for
all model families.

## Stability

The pooled-best combined blend improves QB, RB, TE, and WR by 0.0036, 0.0095,
0.0054, and 0.0070 RMSE. It also improves rookies and veterans. However, it
worsens 2023 and 2024, the 2023-2025 mean is 0.0025 worse, and it degrades the
small other/no-history, exactly-one-nonconsecutive-projection, and missing-ADP
slices.

The raw-ADP full trajectory blend is slightly weaker pooled but avoids the
recent reversal. Carry it, plus a model-specific logged-ADP Lasso, into the
prespecified whole-season/calibration and joint-template replay. Do not tune a
new blend weight on these same OOF predictions.
