# Decision Readout

## Finding

The diagnostic supports optimizer's curse, but the precise mechanism is selection
frequency rather than a simple ever-selected versus never-selected split.

Across 518 auctionable player-origins with recorded actual prices:

| Cohort | Mean `actual - point predicted` residual |
|---|---:|
| All observed auctionable players | -$0.39 |
| Unique players ever selected | -$0.40 |
| Optimized roster slots, selection weighted | +$1.43 |

The salary model is slightly high on the observed pool and on the unique set of
players selected at least once. The roster bias appears because the optimizer
repeatedly uses a much smaller set of players whose realized residuals are positive.

| Selection frequency across 1,000 rosters per origin | Player-origins | Mean residual |
|---|---:|---:|
| Never | 41 | -$0.27 |
| 0-5% | 203 | -$1.58 |
| 5-25% | 221 | -$0.12 |
| 25-50% | 50 | +$2.38 |
| Above 50% | 3 | +$12.32 |

This gradient also appears among top-quartile projected players: those selected at
most 5% of the time averaged `-$4.64`, while the 38 selected at least 25% of the
time averaged `+$2.49`. Only two top-quartile players were literally never selected,
so the rare-versus-frequent comparison is more informative than the requested
unselected split.

## Where the bias concentrates

- RB roster slots averaged `+$2.71`; WR averaged `+$1.29`; TE averaged `+$0.52`;
  QB averaged `-$1.10`.
- The strongest value-over-price quintile averaged `+$4.82` per selected slot and
  had a positive residual 72.3% of the time. The two weakest quintiles were negative.
- The pattern is not confined to one predicted-price tier. The `$6-$15` tier was
  largest at `+$2.29`, while the `$51+` tier was also positive at `+$1.91`.
- Current salary-model rows alone averaged `+$1.26` per selected recorded slot, so
  the finding is not explained by the small fallback subset.

A limited number of repeatedly selected misses materially drive roster totals. For
example, 2024 Rashee Rice was selected in 67.5% of the origin's rosters at a `$9.10`
point salary and cost `$26`, contributing about `$11.41` of residual to an average
2024 roster across the four tested policies. This is a realized diagnostic, not a
claim that the live 2026 player will repeat the same error.

## Corrected decomposition of the prior `$29` gap

The full actual-minus-scenario gap is not entirely player-level point-prediction
error.

| Period | Chance rule | Scenario to point-row discount | Actual minus point row | Total actual minus scenario |
|---|---:|---:|---:|---:|
| Development 2022-2024 | 60% | $12.89 | $16.41 | $29.30 |
| Development 2022-2024 | 90% | $12.76 | $16.39 | $29.15 |
| 2025 | 60% | $9.56 | $21.60 | $31.15 |
| 2025 | 90% | $9.36 | $23.01 | $32.36 |

Across the full candidate pool, the reconstructed normalized five-draw scenario
mean was `$0.23` above the point salary per player. Across selected roster slots,
it was `$0.92` below the point salary. The optimizer therefore concentrates both:

1. players receiving favorable scenario prices relative to the point row; and
2. players whose actual price ultimately exceeded the point row.

## Actions

1. Do not repair this by hardcoding a `$29` haircut. It combines two mechanisms,
   uses only four realized auction markets, and would overfit the diagnostic.
2. Retain the point-salary row as an explicit anchor in the next replay. A scenario
   chance rule by itself lets selected-roster scenario means fall roughly `$10-$13`
   below their point-salary sum.
3. Build a strictly prior-origin, shrinkage salary surcharge using only preseason
   features: selection propensity from a seed optimizer, within-position
   value-over-price rank, position, and predicted-price tier. The strong but sparse
   frequent/core buckets must be pooled rather than assigned their raw residuals.
4. Recenter or blend the residual scenario means as a separate experimental arm;
   compare the current empirical means, zero-mean residual draws, and a
   `max(point row, scenario-risk row)` robust constraint.
5. Rerun the affordability frontier with the correction trained only on earlier
   origins. Evaluate 2025 without using its residuals in the surcharge.
6. Prefer a deterministic robust row for the first production candidate. It is much
   cheaper than exact chance binaries and can preserve deterministic acquired-player
   salaries and the existing one-swap refinement.

The diagnostic supports the user's intuition: positive residuals are concentrated
where the optimizer sees the strongest repeated values. It also narrows the fix:
calibrate selection propensity/value rank and stop the scenario layer from granting
an unvalidated discount relative to the point-salary row.

