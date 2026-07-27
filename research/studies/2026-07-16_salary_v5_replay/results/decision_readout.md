# Salary v5 Decision Readout

## Finding

The compact v5 salary surface passes the two gates that v2 did not: it improves
ordinary rolling player-price accuracy and improves optimizer-selected
historical affordability after full roster reselection.

Across all validation rows, normalized v5 MAE/RMSE are `$4.271`/`$6.197`,
versus `$4.307`/`$6.276` for v1 and `$4.455`/`$6.252` for v3. The v5 raw model
is stronger still at `$4.169` MAE and `$6.176` RMSE. It also materially
outperforms the input anchors: raw ESPN-source MAE is `$6.442`, while the
mechanically budget-adjusted source is `$7.277`. The model is therefore using
the source price as context rather than simply reproducing it.

## Paired optimizer replay

The identical-seed v5 frontier completed all 4,000 cells optimally and changed
89.1% of development rosters and 93.4% of 2025 rosters.

- Development historical feasibility rises from 15.5% to 19.0%, average
  historical overage falls by `$2.47`, and actual roster spend falls by `$2.90`.
- In the 2025 temporal check, feasibility rises from 12.0% to 18.5%, average
  overage falls by `$4.76`, and actual roster spend falls by `$5.60`.
- Held-out modeled affordability is nearly unchanged, declining by only 0.4 to
  0.5 percentage points on average. The historical improvement therefore comes
  from better player pricing/selection rather than merely imposing a much more
  conservative modeled chance rule.

The cost is lower preseason managed forecast EV: `-3.12` season points in
development and `-7.80` in 2025. That is directionally consistent with removing
apparent bargains that were not actually purchasable. Historical point evidence
is mixed: among cells where both methods happened to be affordable, v5 trails by
6.55 points in development but leads by 5.93 in 2025. With four seasons and
future-price conditioning, this cannot identify the better scoring policy.

## Remaining optimizer bias

v5 reduces but does not solve the selected-roster salary gap.

- Selection-weighted actual-minus-point residual improves from `$1.43` to
  `$1.35` per selected player.
- The strongest value-over-price quintile improves from `$4.82` to `$4.18`.
- Core selections above 50% frequency improve from `$12.32` to `$8.76`.
- The total actual-minus-scenario roster gap falls from roughly `$29` to `$26`
  in development and from roughly `$31–$32` to `$25–$26` in 2025.

The residual concentration remains clearest in frequently selected players,
the strongest value quintile, and the `$51+` tier. Marginal five-draw scenario
normalization still prices selected rosters below the point-salary row by about
`$10.4` in development and `$7.0` in 2025.

## Action

Keep v5 as the leading/current salary surface rather than reverting to v1 or
the broader v3 feature set. Preserve additive budget reconciliation.

Do not interpret the improved surface as making unconstrained optimizer rosters
fully purchasable: even at the 90% chance rule, historical feasibility is only
23.1% in development and 21.6% in 2025. The next salary experiment should target
the remaining selection-conditioned error, either through a shrunk residual
correction relative to `budget_adjusted_source_salary` or a strictly prior-origin
selection/value surcharge. Scenario construction should continue anchoring to
the point-salary row rather than introducing an additional selected-roster
discount.
