# Asymmetric expert projection cross-league decision

- Point primary passes both leagues: `False`.
- Upper-tail primary passes both leagues: `False`.
- Weekly-template primary passes both leagues: `False`.
- Overall next action: `retain_outside_production`.

## Primary candidate headline deltas

| League | Point controlled RMSE | Point production RMSE | +3 Brier | +5 Brier | Template +5 Brier | Template impact Brier |
|---|---:|---:|---:|---:|---:|---:|
| DK | +0.00075 | +0.00160 | +0.002176 | +0.000329 | +0.000464 | +0.001550 |
| BETA | -0.00035 | -0.00070 | +0.003075 | +0.000598 | +0.000106 | -0.000145 |

## Interpretation

- The normalized bullish-gap primary is rejected: recent point RMSE, +3/+5 residual Brier, and the multi-outcome template gates do not replicate across DK and beta.
- The raw max-minus-median sensitivity improves pooled point RMSE in both leagues, but recent controlled RMSE worsens, clustered intervals cross zero, and the gain is concentrated in the post-slice QB diagnostic.
- High bullish-gap quartiles are not empirical ceiling groups; +3/+5 residual rates are lower than in the bottom quartile in both leagues. Small consensus denominators make many normalized outliers fringe players.
- If revisited, prespecify a projection-floor or projection-tier-controlled raw QB interaction and require nested retuning. Do not promote a generic asymmetric gap or template weight.

The study is intentionally read-only. Passing the point gates would only justify a leakage-safe nested retune; it would not directly change the production model or templates.
