# WR Template PPG/Profile Trade-off Findings

## Decision

Keep the production WR matcher unchanged.

A 2.25 absolute-PPG weight does make the current Ladd McConkey donor pool
geometrically tighter on preseason PPG, but it does not improve held-out WR PPG
calibration consistently across DK and beta. Combining that weight with
projected yards/reception or touchdowns/reception also fails the cross-league
test.

The proposed rates do not solve the motivating Terrelle Pryor comparison.
Pryor remains a top-three donor and gains probability in most candidate arms.
Do not add height, weight, or other measurables solely to suppress this one
2-4% donor; first test a direct receiving-role signal such as preseason aDOT or
slot/wide alignment if a historically complete source becomes available.

## Motivating Player Check

The preseason projection data sees Ladd and Pryor as much more similar than the
football-role labels imply:

| Player-season | Projected rec | Projected YPR | Projected TD/rec | Shrunk YPR profile | Shrunk TD/rec profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ladd McConkey 2026 | 76.0 | 13.235 | 0.08207 | 0.66995 | 0.69261 |
| Terrelle Pryor 2017 | 68.5 | 14.453 | 0.08128 | 0.69793 | 0.60003 |

The YPR profile distance is only 0.028. Raw TD/rec is virtually identical; its
larger percentile-profile difference comes from the different season
distributions, not a meaningful absolute scoring-rate gap.

For Ladd's current beta pool:

| Method | Weighted absolute donor PPG gap | Expected played | Pryor rank | Pryor weight |
| --- | ---: | ---: | ---: | ---: |
| Production | 0.969 | 13.290 | 3 | 2.23% |
| PPG 2.25 | 0.767 | 13.156 | 3 | 2.26% |
| YPR 0.50 | 1.002 | 13.138 | 2 | 2.54% |
| TD/rec 0.50 | 0.971 | 13.234 | 2 | 2.28% |
| PPG 2.25 + YPR 0.50 | 0.758 | 13.130 | 2 | 2.54% |
| PPG 2.25 + both 0.50 | 0.757 | 13.077 | 2 | 2.57% |

The PPG bump produces the requested matching trade-off for this pool: a 20.8%
tighter donor PPG gap for 0.135 fewer expected games. It does not remove Pryor
because Pryor's historical preseason center is only 0.69 PPG above Ladd's.

The same qualitative result holds in DK. Pryor remains rank 2 in every arm and
his probability rises from 3.26% in production to 3.56% under YPR and 3.96%
under the 2.25-PPG/two-rate arm.

## Strict Rolling Replay

Each league replay contains 648 held-out 2017-2025 WR targets and 5,832
target-method rows. Lower CRPS/Brier is better. The table reports
candidate-minus-production deltas.

| League | Period | Method | PPG CRPS | Contribution CRPS | Played CRPS | Impact AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DK | 2017-2025 | PPG 2.25 | +0.003308 | +0.020491 | +0.003622 | -0.004501 |
| DK | 2023-2025 | PPG 2.25 | +0.006338 | +0.031545 | +0.005541 | -0.011010 |
| beta | 2017-2025 | PPG 2.25 | -0.001497 | -0.001919 | -0.001042 | -0.001791 |
| beta | 2023-2025 | PPG 2.25 | -0.001557 | +0.008506 | -0.004066 | -0.009606 |
| DK | 2017-2025 | PPG 2.25 + YPR 0.50 | +0.002242 | +0.001673 | +0.003156 | -0.008264 |
| DK | 2023-2025 | PPG 2.25 + YPR 0.50 | +0.000352 | -0.005926 | +0.008598 | -0.016225 |
| beta | 2017-2025 | PPG 2.25 + YPR 0.50 | +0.000754 | +0.002355 | +0.001727 | -0.004287 |
| beta | 2023-2025 | PPG 2.25 + YPR 0.50 | -0.000765 | -0.005256 | -0.000327 | -0.016635 |
| DK | 2017-2025 | PPG 2.25 + both 0.50 | +0.001237 | +0.000725 | +0.004328 | -0.007794 |
| DK | 2023-2025 | PPG 2.25 + both 0.50 | +0.000711 | +0.037548 | +0.013522 | -0.023468 |
| beta | 2017-2025 | PPG 2.25 + both 0.50 | -0.001432 | -0.024624 | +0.005797 | -0.012970 |
| beta | 2023-2025 | PPG 2.25 + both 0.50 | -0.003159 | -0.022663 | +0.002760 | -0.021556 |

The combined arms do not offer the desired exchange of better PPG calibration
for worse played-game calibration. In DK they generally worsen both, and impact
ranking weakens across both leagues.

TD/rec at the production PPG weight is the only arm with directionally better
PPG CRPS in both leagues and periods. Its gains are tiny and generally
uncertain; DK 2023-2025 played-game CRPS worsens by 0.01308 with the
season-cluster interval above zero. More importantly, it does not distinguish
Ladd from Pryor, so it does not answer the motivating archetype problem.

## Interpretation

- A closer input-PPG neighborhood is not automatically a better calibrated
  residual distribution. DK is the counterexample.
- Aggregate projected YPR and TD/rec encode scoring shape, not route role,
  alignment, body type, or position-conversion history.
- Projected catch rate would be a more direct slot/deep proxy, but historical
  target coverage is incomplete and Pryor's 2017 target projection is missing.
- The Explorer's rank is not donor dominance. Pryor carries only 2.23% of
  Ladd's beta production pool; the top 12 donors together carry 26.95%.

Production code and databases were not changed.
