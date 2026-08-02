# Receiver-Rate Weekly-Template Findings

## Decision

Do not add projected yards per reception or receiving touchdowns per reception
to the global WR/TE or RB weekly-template matcher.

The rates meaningfully change donor composition and modestly improve some PPG
and managed-contribution point estimates, but the predeclared combined WR/TE
arm does not clear the joint PPG, contribution, played-games, and impact
guardrails. WR availability and impact discrimination weaken, particularly in
recent seasons. The 1.00 weight sensitivity is clearly too strong, and the 0.25
arm does not resolve the tradeoff.

Keep production unchanged. Carry TE-only projected yards per reception, and
possibly the two-rate TE combination, as an independently prespecified future
challenger. Do not promote that same-evidence position slice: its contribution
signal is encouraging across leagues, but DK recent played-games behavior is
not safe.

## Design and coverage

- Replayed 1,620 held-out 2017-2025 player-seasons in each of DK and beta.
- Every donor strictly precedes its target season.
- Held the top-80 pool, adaptive distance kernel, 12-season recency prior, 5%
  donor cap, centered residual, and joint weekly path fixed.
- Joined only preseason V2 projections by canonical player key and season.
  Realized receiving rates never enter matching.
- Converted each rate to a season-position percentile and shrank it toward 0.5
  using `projected_receptions / (projected_receptions + 10)`.
- All 1,404 RB/WR/TE rolling targets per league have both rates. Historical
  coverage is 4,591/4,592 skill-position templates; Chris Ivory's 2011 RB row
  is the sole neutral-filled exception.
- The seven methods produce 11,340 complete prediction rows per league.
- The production baselines reproduce the corrected league-specific replay
  exactly: 2.343290 DK and 1.913075 beta pooled PPG CRPS.

## Donor composition

The primary 0.50 WR/TE arm retains 88.3% of DK and 88.6% of beta top-80
WR/TE donors, changing about nine donors per target. Weighted distance falls by
0.040/0.031 DK and 0.040/0.031 beta on the shrunk yards-per-reception and
TD-rate profiles. Effective sample size remains broad.

The RB extension retains about 93% of its baseline donors but does not improve
outcomes consistently. The rates therefore add real profile separation; the
rejection is an outcome-calibration decision, not a failure to change matches.

## Primary WR/TE result

Candidate minus production is shown below; negative CRPS is favorable.

| League | Period | PPG CRPS | Contribution CRPS | Played CRPS | Impact AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| DK | 2017-2025 | -0.000283 | -0.020697 | +0.002179 | -0.004877 |
| beta | 2017-2025 | -0.001131 | -0.014121 | +0.000979 | -0.012958 |
| DK | 2023-2025 | -0.002839 | -0.038569 | +0.009880 | -0.016795 |
| beta | 2023-2025 | -0.003181 | -0.026038 | -0.001727 | -0.023902 |

Full-period PPG, contribution, and played-games intervals all cross zero in
both leagues. DK recent played-games CRPS is adverse with player-cluster
interval `[+0.001879, +0.017941]`. The beta WR slice also worsens impact Brier
in both season and player bootstraps, while stronger 1.00 weights deepen the
impact and played-games losses.

The isolated fields divide the small gains rather than producing a stable
global winner:

- yards per reception supplies most of the contribution improvement;
- TD rate supplies more of the small PPG and `+3` event improvement;
- neither clears cross-league, cross-period, multi-outcome guardrails alone.

## Position interpretation

### WR

The fields do not earn a WR match weight. Full-period contribution is neutral
to worse in beta, TD-rate matching materially worsens DK played-games CRPS, and
the combined arm worsens beta WR impact Brier with both clustered intervals
above zero. Recent point improvements do not offset those failures.

### TE

TE is the useful same-evidence hypothesis. At weight 0.50, yards per reception
changes full-period metrics by:

| League | PPG CRPS | Contribution CRPS | Played CRPS | Impact Brier |
| --- | ---: | ---: | ---: | ---: |
| DK | -0.000508 | -0.064926 | +0.001548 | +0.000174 |
| beta | -0.003586 | -0.045904 | -0.003310 | -0.000872 |

DK and beta contribution intervals are below zero in both season and player
bootstraps. Beta PPG and impact-Brier intervals are also below zero in both.
The combined TE arm improves DK/beta contribution by 0.096/0.046 and beta PPG,
played-games, and impact point estimates as well.

This is not ready for promotion. The TE route was not the predeclared primary
selection, and DK 2023-2025 yards-per-reception played-games CRPS worsens with
the season interval above zero. Require a future-origin or explicitly nested
TE-only confirmation with played-games safety before changing production.

### RB

Adding both rates to RB changes full-period contribution CRPS by +0.0045 DK and
+0.0120 beta, with every main interval crossing zero. Do not retain the RB
extension.

## Production boundary

No production matcher, database, contract, or app output changed. The active
receiver criteria remain receiving-point rank, team share, room hierarchy and
concentration, QB context, projection/market context, and the existing global
features.
