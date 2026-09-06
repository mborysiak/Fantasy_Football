# Sequential Shared-Opportunity Results

## Corrected outcome

The earlier result is superseded. Its replay used the 2026 NV rules but failed
to remove opponents' active keepers, making players such as Patrick Mahomes and
Brock Bowers incorrectly available. The corrected replay fixes Maye and Achane
to the tested roster, removes the other 14 active keepers from the player pool
and hidden auction tapes, and subtracts all 16 keepers from league money and
slots. Predicted mode therefore starts with `$3,123` across 140 slots; actual
hindsight mode uses the App's salary overlay and starts with `$3,130`.

The corrected result still favors one full-bank conditional swap with the
normal Top-N constraint enabled. A follow-up tested a utilization/add-one
shortlist and, more importantly, separated the broad screen from confirmation.
The shortlist reproduced the static and organic quality but did not materially
beat full exact confirmation latency and changed four actual-board calls,
including promoting Jahmyr Gibbs at `$111` from PASS to TARGET. App v14 therefore
uses the safer full exact swap only during confirmation and leaves the broad
64-player screen additive.

## Paired quality results

Independent organic sequential rollouts produced:

| Price mode | Baseline mean | One-swap mean | Delta | Baseline p10 | One-swap p10 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Predicted | 1694.62 | 1702.00 | +7.38 | 1524.04 | 1531.88 | +7.84 |
| Actual | 1741.07 | 1747.03 | +5.96 | 1556.40 | 1564.61 | +8.21 |

The paired static completion plans agree directionally. Predicted-price
mean/p10 improve `+14.91`/`+27.50`; actual-price mean/p10 improve
`+8.39`/`+7.50`.

The effect is not uniform. Predicted organic mean improves in all four blocks
and p10 improves in three, with the fourth essentially flat. Actual mean and
p10 each improve in three of four blocks, but the aggregate is heavily helped
by block 3 (`+19.70` mean, `+28.27` p10); block 1 mean is `-2.28` and block 0
p10 is `-9.67`. Without Top-N, the static predicted holdout reverses to
`-1.92` mean and `-16.01` p10, while actual remains positive at `+20.91` mean
and `+6.61` p10. The correction remains dependent on the current policy
configuration rather than establishing a universal improvement.

All 128 actual paths complete in both arms. Predicted completion is 124/128 in
both arms, with four `market_slots` failures each. Every completed roster is
cap, position, and Top-N legal. Validation assertions confirm that no opponent
keeper appears in any static plan or organic rollout.

## Lineup changes

Every corrected Top-N static plan makes exactly one swap, always removing an
RB for a third QB or second TE:

| Price mode | Block | Out | In |
|---|---:|---|---|
| Predicted | 0 | Jadarian Price | Tyler Warren |
| Predicted | 1 | TreVeyon Henderson | Kyle Pitts |
| Predicted | 2 | Tony Pollard | Baker Mayfield |
| Predicted | 3 | Derrick Henry | Josh Allen |
| Actual | 0 | Chuba Hubbard | Bryce Young |
| Actual | 1 | TreVeyon Henderson | Cam Ward |
| Actual | 2 | Kyren Williams | Matthew Stafford |
| Actual | 3 | Josh Jacobs | Matthew Stafford |

The actual-price organic paths show the same shape. Matthew Stafford rises
from 0% to 48%, Cam Ward from 2% to 27%, and Bryce Young from 0% to 22%.
Kyren Williams falls from 48% to 23%, Josh Jacobs from 48% to 24%, Chuba
Hubbard from 23% to 0%, and TreVeyon Henderson from 27% to 3%. This is the
intended shared-opportunity correction: redundant RB/FLEX depth loses credit
relative to a distinct QB or TE opportunity. It is not a keeper selection
effect and is not a direct reward for spending.

## Runtime and stability

Mean warm plan compilation with Top-N enabled is:

- predicted: `0.0029s` baseline, `0.1340s` shortlist, and `0.1441s` full exact;
- actual: `0.0023s` baseline, `0.1274s` shortlist, and `0.1409s` full exact.

The shortlist uses exact simulated start utilization plus a small high-cost
escape route for redundant expensive depth, screens incoming players with one
full-roster add-one pass, and exact-confirms the conditional finalists. It
matches the full exact predicted Top-N plan in all four blocks. The actual
organic shortlist scores `1747.98` mean / `1565.37` p10 versus `1747.03` /
`1564.61` full exact, but that small holdout difference is not a reason to use a
heuristic on the live action surface.

On a production-shaped 64-candidate/18-confirmation board at compute budget
120 with four workers:

| Price mode | Additive | Exact both stages | Exact confirm only | Shortlist confirm only |
|---|---:|---:|---:|---:|
| Predicted | 6.72s | 20.08s | 9.24s | 9.60s |
| Actual | 4.84s | 17.58s | 7.58s | 7.39s |

Moving refinement out of the 64-player screen captures nearly all of the
runtime reduction: exact-confirmation-only is `1.37x` baseline predicted and
`1.57x` actual, versus `2.99x`/`3.63x` when both stages refine. The shortlist
is within timing noise of full exact confirmation (`+0.36s` predicted,
`-0.20s` actual), so it does not earn a live heuristic tradeoff.

All eight paired evidence blocks and all eight production-shaped timing arms
completed in fresh processes without fallback. The 55 focused Sequential tests
pass, including exact shared-slot correction, shortlist matching, and locked
flier protection.

## Decision

- Promote one exact full-bank conditional swap for confirmation only; keep the
  broad discovery screen additive and advance the Sequential cache to App v14.
- Treat the prior keeper-contaminated findings as invalid and superseded.
- Keep the utilization/add-one shortlist as a research challenger, not the live
  action authority. It has no material latency win and can change boundary
  recommendations.
- Reject multi-swap convergence; it adds no production decision and worsens
  the unstable runtime surface.
- Owned players are never removable, so intentionally locked rookie/young
  fliers retain their actual residual profiles while the optimizer fixes the
  remaining roster's shared-slot allocation.
