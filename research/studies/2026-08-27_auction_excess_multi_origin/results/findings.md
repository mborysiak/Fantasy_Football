# Findings

## Decision

Do not promote either unconstrained expected positive winning margin or the
standardized 50/50 mean-plus-excess blend into the live Auction construction
objective. Retain expected excess as a diagnostic.

## Evidence

The policies were copied unchanged from the 2025 experiment. The runner first
selected every roster for all three origins, then loaded 2022-2024 actual weeks
in a separate phase.

| Year | Pure excess: actual delta | 50/50: actual delta | Pure: holdout EV delta | 50/50: holdout EV delta |
| ---: | ---: | ---: | ---: | ---: |
| 2022 | -5.88 | +26.00 | -44.41 | -23.42 |
| 2023 | -48.84 | -59.56 | -27.60 | -12.91 |
| 2024 | -64.33 | -83.23 | -16.34 | -8.62 |

Both alternatives also reduce independent holdout P90 in every season. Pure
excess reduces independent holdout expected excess in 2022 and 2023; its 2024
gain is only `+0.21`. That pattern is evidence of selection noise/overfit in the
construction candidate bank, not merely a volatile policy missing its realized
upside.

Adding the previously inspected 2025 descriptive outcome gives four-season
mean actual deltas of `-7.20` for pure excess (positive 1/4 seasons) and
`-13.97` for 50/50 (positive 2/4). The positive 2025 result therefore does not
transport across origins.

## What this does and does not say

This rejects the current estimator and winner-selection rule. It does not
reject the user's broader thesis that fantasy football is power-law-like and a
few persistent breakout hits can dominate a season. The current expected-
excess metric is field-relative within a noisy empirical candidate set; it does
not directly model breakout persistence, playoff timing, or whether a player's
ordinary production can be replaced through waivers.

The next credible test should build a dense legal local-swap neighborhood
around the mean roster, cross-fit objective selection inside each historical
origin, and distinguish persistent player upside from replaceable weekly
variance. Production remains unchanged.

