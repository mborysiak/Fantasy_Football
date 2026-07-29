# Projection Consensus Ladder Findings

## Decision

Treat the generic projection-only feature layer as mature. Do not replace the
configured provider median with the learned stack, and do not add the new
provider-stack, room-disagreement, active-PPG-alignment, or combined projection
families to the 31-feature full model. The remaining promising question is
model routing by position and history depth, not another broad projection
feature expansion.

## What was tested

The 2017-2025 rolling OOF ladder compared:

- the configured-score provider median;
- 22 consensus/component fields plus position;
- a strictly causal nonnegative provider stack, globally and by position;
- six compact room/depth-chart disagreement fields;
- four active-PPG and projected-games alignment fields;
- ten previously tested rate/opportunity shape fields; and
- all projection additions together.

Projection-only Lasso and deterministic shallow LightGBM were evaluated on the
same held rows. Each new family was then added to the governed 31-feature
deterministic LightGBM incumbent to test whether any gain transferred to the
full residual context.

## Main results

The governed full LightGBM remains the best ordinary fitted model at 3.1230
RMSE. Projection-only LightGBM using the consensus core reaches 3.1326 RMSE
and nearly identical MAE (2.4470 versus 2.4468), only 0.0097 RMSE behind the
full model. This is strong evidence that the provider projection layer already
contains most of the predictable conditional-PPG signal.

No new family improves the full model:

| Full-model addition | RMSE | Delta versus 3.1230 |
|---|---:|---:|
| Active-PPG alignment | 3.1249 | +0.0020 |
| Causal provider stack | 3.1266 | +0.0036 |
| Room disagreement | 3.1273 | +0.0043 |
| All projection additions | 3.1321 | +0.0091 |
| Targeted stack + room + active | 3.1333 | +0.0103 |

Within projection-only LightGBM, active alignment (-0.0019 RMSE) and room
disagreement (-0.0011) have tiny favorable point estimates, but both intervals
cross zero and neither transfers to the full model. Shape features are neutral
to worse. Adding every projection family is reliably worse (+0.0203,
season-bootstrap interval +0.0050 to +0.0360).

The active fields are too young for a durable conclusion. Their mean historical
training coverage is only 8.2%; projected-games/active-provider disagreement
does not begin until 2025 and the other active fields begin in 2024. Revisit
them after several additional historical seasons rather than promoting them
from the current sample.

## Provider weighting

The causal global provider stack improves realized team-game PPG from 2.7572
to 2.7476 RMSE, a 0.0096 pooled gain. It wins five of nine seasons and its
season-bootstrap interval spans -0.0318 to +0.0117. The position stack is
similar at 2.7481, with an interval spanning -0.0400 to +0.0196. Both slightly
worsen rank correlation.

For 2026 the global solution remains 92.3% configured median, 5.8% FFToday,
and 1.9% FantasyPros. The small move toward FFToday agrees with the earlier
provider-family result, but the weighting gain is unstable and its derived
features worsen the conditional-PPG model. Keep the provider stack as an audit
diagnostic; do not use it as the production consensus.

## History-aware routing

Projection-only and full models make different errors. The projection model is
better in aggregate for rookies and second-year players, while the full model
is better for veterans with history. The position interaction matters:
limited-history QBs, TEs, and WRs favor projection-only estimates, while
limited-history RBs favor the full model.

A causal router that selects between the two models for each
position-by-history group using only earlier OOF errors improves the pooled
RMSE from 3.1230 to:

| Minimum prior group rows | Router RMSE | Delta | Season wins | 95% interval |
|---:|---:|---:|---:|---:|
| 25 | 3.1133 | -0.0097 | 5/9 | [-0.0234, +0.0013] |
| 50 | 3.1158 | -0.0072 | 5/9 | [-0.0199, +0.0028] |
| 100 | 3.1174 | -0.0055 | 4/9 | [-0.0179, +0.0044] |

The intervals still cross zero, so this is a finalist rather than a promotion.
The 25- and 50-row rules both select projection-only for 2026 limited-history
QB/TE/WR groups, and full LightGBM for limited-history RBs and every veteran
group. That routing pattern has been stable since the 2023 origin.

## Next step

Stop adding generic projection ratios or provider disagreement variants. Carry
the consensus-core projection model alongside the full LightGBM into finalist
fitting, then evaluate the prespecified position-by-history router in the final
whole-season/calibration replay and in template integration. Preserve the two
component forecasts so the templates can use history-appropriate centers
without drawing an additional independent residual.
