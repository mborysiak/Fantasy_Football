# Stabilized bullish expert-gap cross-league decision

- Smooth-k5 point primary passes both leagues: `False`.
- Smooth-k5 upper-tail primary passes both leagues: `False`.
- Smooth-k5 weekly-template primary passes both leagues: `False`.
- Overall next action: `retain_outside_production`.

## Point-model RMSE deltas

| League | Surface | Raw | Smooth k3 | Smooth k5 primary | Smooth k8 | Hard floor k5 | Additive k5 |
|---|---|---:|---:|---:|---:|---:|---:|
| DK | controlled | -0.00548 | -0.00205 | -0.00317 | -0.00497 | -0.00177 | -0.00404 |
| DK | production | -0.00526 | -0.00116 | -0.00276 | -0.00455 | -0.00118 | -0.00378 |
| BETA | controlled | -0.00568 | -0.00194 | -0.00409 | -0.00454 | -0.00325 | -0.00436 |
| BETA | production | -0.00565 | -0.00180 | -0.00403 | -0.00435 | -0.00290 | -0.00426 |

## Smooth-k5 upper-tail deltas

| League | Event | Brier delta | Recent delta | AUC change | Season 95% |
|---|---|---:|---:|---:|---:|
| DK | plus3 | +0.001421 | +0.001943 | -0.00381 | [+0.000010, +0.002998] |
| DK | plus5 | +0.000332 | +0.000119 | -0.01903 | [-0.000276, +0.001038] |
| BETA | plus3 | +0.002763 | +0.004218 | -0.00338 | [+0.000103, +0.005898] |
| BETA | plus5 | +0.000294 | +0.000649 | -0.00963 | [-0.000522, +0.000861] |

## Smooth-k5 template deltas

| League | Period | PPG CRPS | Contribution CRPS | Played CRPS | +3 Brier | +5 Brier | Impact Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| DK | full_2017_2025 | -0.003974 | +0.043934 | +0.000979 | +0.000026 | +0.000351 | +0.001387 |
| DK | temporal_2023_2025 | -0.002329 | +0.046032 | +0.001934 | +0.000405 | -0.000078 | +0.001309 |
| BETA | full_2017_2025 | -0.001261 | -0.014819 | +0.001419 | -0.000370 | +0.000087 | -0.000030 |
| BETA | temporal_2023_2025 | +0.001337 | +0.000694 | +0.001443 | -0.000900 | +0.000195 | +0.000562 |

Only smooth k5 was promotion-eligible. The other denominator forms are declared sensitivities and cannot rescue a failed primary.

The study is read-only. No production table, lock, feature contract, or template was changed.
