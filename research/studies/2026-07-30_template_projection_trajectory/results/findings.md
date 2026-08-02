# Weekly-Template Projection-Trajectory Findings

## Decision

Keep projection trajectory out of the production WR template matcher.

The signed one-year and recency-weighted three-year projection gaps materially
improve the face validity of the motivating Ladd McConkey/Terrelle Pryor pool.
However, every tested trajectory arm worsens held-out WR PPG CRPS in both
leagues over both the full and 2023-2025 periods. That fails the stated
preference for tighter PPG calibration.

Retain the trajectory fields as useful diagnostics. Do not use prior-history
availability or depth as match-distance criteria; the history-aware arms are
clearly worse.

## Design

- 648 held-out 2017-2025 WR targets per league.
- 5,832 target-method rows per league.
- Strictly prior-season donors.
- Production top-80 pool, adaptive kernel, 12-season recency prior, 5% donor
  cap, centered residual, and joint weekly path held fixed.
- Signed gaps ranked within position-season among players with relevant
  history.
- Rookies/no-history players assigned the neutral zero-change profile.
- Exact-prior availability and prior-three-year depth tested separately so
  missing history was not silently treated as observed stability.

## Ladd/Pryor Diagnostic

| League | Player | One-year gap | One-year profile | Three-year gap | Three-year profile |
| --- | --- | ---: | ---: | ---: | ---: |
| DK | Ladd McConkey 2026 | -2.028 | 0.207 | -0.356 | 0.558 |
| DK | Terrelle Pryor 2017 | +7.465 | 0.982 | +9.698 | 0.991 |
| beta | Ladd McConkey 2026 | -1.738 | 0.197 | -0.241 | 0.594 |
| beta | Terrelle Pryor 2017 | +6.424 | 0.982 | +8.302 | 0.996 |

The feature therefore captures the intended distinction: Ladd's current
projection is below his earlier projection evidence, while Pryor entered 2017
as an extreme projection riser.

### Current beta pool

| Method | Pryor rank | Pryor weight | Expected played | Residual SD |
| --- | ---: | ---: | ---: | ---: |
| Production | 3 | 2.23% | 13.290 | 3.135 |
| One-year 0.25 | 4 | 1.84% | 13.348 | 3.136 |
| One-year 0.50 | 12 | 1.43% | 13.119 | 2.992 |
| Both gaps 0.25 | 5 | 1.67% | 13.304 | 3.140 |
| Both gaps 0.50 | 19 | 1.20% | 13.204 | 3.080 |
| Full history 0.50 | 26 | 1.12% | 13.194 | 3.146 |

The same pattern holds in DK. One-year 0.50 moves Pryor from rank 2 to 7;
both-gap 0.50 moves him to 13; full-history 0.50 moves him to 14.

## Strict Rolling Results

The table reports candidate-minus-production deltas. Lower CRPS/Brier is
better.

| League | Period | Method | PPG CRPS | Contribution CRPS | Played CRPS | Impact AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DK | 2017-2025 | One-year 0.25 | +0.001435 | +0.004182 | +0.002312 | +0.001066 |
| beta | 2017-2025 | One-year 0.25 | +0.001723 | +0.011607 | +0.001480 | +0.001628 |
| DK | 2023-2025 | One-year 0.25 | +0.001408 | +0.002285 | +0.005600 | +0.006809 |
| beta | 2023-2025 | One-year 0.25 | +0.001747 | -0.003578 | -0.001939 | +0.022962 |
| DK | 2017-2025 | Three-year 0.25 | +0.001737 | -0.007785 | +0.002105 | +0.006006 |
| beta | 2017-2025 | Three-year 0.25 | +0.000631 | -0.008190 | +0.001218 | +0.007787 |
| DK | 2023-2025 | Three-year 0.25 | +0.001246 | -0.028581 | +0.001011 | +0.004925 |
| beta | 2023-2025 | Three-year 0.25 | +0.000270 | -0.041883 | -0.006625 | +0.034911 |
| DK | 2017-2025 | Full history 0.25 | +0.004037 | +0.055173 | +0.004307 | -0.004140 |
| beta | 2017-2025 | Full history 0.25 | +0.003823 | +0.033588 | +0.001741 | -0.012210 |

The three-year 0.25 arm is the strongest secondary result. It improves
contribution and impact point estimates in both leagues, with beta 2023-2025
contribution, played-games, and impact-Brier season-cluster intervals below
zero. It still worsens PPG in every scope; DK's full-period PPG interval is
entirely above zero. It therefore does not satisfy the requested PPG-first
trade-off.

The one-year feature is specifically unsupported as a match criterion. DK
2023-2025 one-year 0.25 worsens both PPG and played-game CRPS with
season-cluster intervals above zero.

Adding history availability/depth is harmful. The predeclared full-history
0.25 arm worsens full-period PPG and contribution in both leagues; beta's
full-period PPG interval is above zero and DK's contribution interval is above
zero. Rookie/no-history status should remain audit metadata, with existing
experience matching handling lifecycle separation.

## Interpretation

Projection trajectory is an excellent explanation field but not a supported
outcome-distance field. It can make a top-comp table look more semantically
coherent without producing a better calibrated donor distribution. The
current top-80 mixture already limits Pryor to 2.23% of Ladd's beta pool, which
helps explain why removing that visible mismatch does not improve aggregate
PPG calibration.

Production code and databases were not changed.
