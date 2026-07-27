# Decision Readout

## Bottom line

The sampled chance constraint is mechanically effective but is not sufficient
with the current salary scenario model. Do not select a production threshold from
this replay yet, and do not rank `$5` versus `$10` from unconditional historical
points on unaffordable rosters.

Increasing the required construction probability from 60% to 90% produced a clean
preseason risk/EV frontier:

| period | forecast season-point change | held-out modeled cap-probability change | historical cap-feasibility change | mean historical overage change |
|---|---:|---:|---:|---:|
| Development 2022-2024 | -4.881 | +24.603 pp | +6.000 pp | -$5.780 |
| Temporal check 2025 | -8.937 | +25.942 pp | +6.400 pp | -$5.620 |

However, the absolute historical feasibility remained poor:

| period | 60% rule | 90% rule |
|---|---:|---:|
| Development 2022-2024 | 12.1% | 18.1% |
| Temporal check 2025 | 8.4% | 14.8% |

The failure is mainly roster-level salary calibration, not the chance formulation.
Across development origins, actual selected-roster spend exceeded the independent
scenario-bank mean by about `$29.1-$29.3` at every threshold. Tightening the chance
level chose cheaper scenario rosters, but it did not remove that nearly constant
selected-roster price bias.

## What the replay did establish

- All 4,000 cells solved optimally and satisfied the exact sampled chance rule,
  roster size, position maxima including `TE <= 2`, and Top-N constraint.
- The held-out modeled affordability curve rose monotonically in every year and in
  both trial halves.
- Managed forecast EV and historical overage fell monotonically as the chance level
  tightened in every year and in both trial halves.
- The 90% construction rule achieved only 87.1% held-out modeled affordability in
  development and 84.2% in 2025. Twenty construction scenarios therefore leave
  meaningful optimization/sample error even before comparison with actual prices.
- Historical feasibility was not strictly monotone at every adjacent threshold,
  although mean historical overage was.
- The current one-swap refiner was disabled because it cannot enforce the
  multi-scenario constraint. All four thresholds use the same unrefined optimizer,
  so the frontier comparison is internally paired but not a bit-for-bit production
  replay.

## Interpretation

The current marginal residual quantiles can look reasonable player by player while
still missing the joint, conditional outcome of an optimizer-selected roster. The
market normalization enforces shared league dollars, but it does not learn
cross-player price correlation or correct residual drift among the exact player
profiles the optimizer favors.

The chance rule is also heavier than the current replay solve. The 4,000-cell study
took 357.8 seconds versus 83.6 seconds for the prior 3,000-cell simple-buffer study,
roughly 3.2 times as much runtime per solved cell. That is acceptable for offline
validation but material inside live Target/Nomination loops.

## Next methodological step

Before wiring a chance threshold into the app, calibrate the salary surface at the
selected-roster level using strictly prior origins. The most useful next comparison is:

1. a causal roster-level spend correction or conservative robust salary row;
2. market scenarios that preserve empirically observed cross-player residual
   dependence where the data supports it;
3. the same held-out modeled-probability and historical-affordability frontier;
4. a lower-cost CVaR/robust-row approximation if exact chance MILPs remain too slow.

Historical points should remain audit-only unless the roster was affordable, and
feasible-only point averages should remain descriptive rather than a policy ranking.
