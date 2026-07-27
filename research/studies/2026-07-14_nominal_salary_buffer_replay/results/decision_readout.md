# Nominal Salary Buffer Decision Readout

## Bottom line

Retain the average of five salary draws. The replay identifies a buffer frontier,
not one statistical winner. For the stated purpose of an affordability guardrail,
use `$298 + $5 = $303` as the provisional setting. Use `+$10` when intentionally
favoring a looser, more point-preserving guardrail.

The replay does not support switching to one salary draw. At a `$5` buffer, one
draw gained only 1.2 season points versus five draws but reduced realized-price
feasibility by 3.8 percentage points and added $1.95 of mean overage. With no
nominal guardrail, one draw's larger point total came with a
13.7-point feasibility loss and $26.09 more overage.

`$10` added 7.2 development-period points relative to `$5`, but lost 2.1
feasibility points and added $1.21 of overage. In 2025, `$5` was marginally better
on all three measures, although the differences were small. A zero buffer is too
aggressive as the general default. The nominal row is a stabilizer, not a complete
solution: even the strictest tested five-draw rule fit historical prices in only
27.6% of 2022-2024 trials and 19.2% of 2025 trials.

## Replay contract

- Origins: frozen preseason 2022, 2023, 2024, and 2025 inputs.
- Design: 250 paired trials in each of 12 cells, or 12,000 solved cells.
- Grid: one versus five averaged salary draws crossed with no nominal row or a
  `$0`, `$5`, `$10`, `$15`, or `$25` buffer.
- Every cell retains the sampled-price `$298` cap, Top-N on, projected waiver
  baselines, and bench-upside weight `0.25`.
- Constrained cells also require the roster's normalized point-predicted salaries
  to total no more than `$298 + buffer`.
- Point salaries are normalized once per origin to the keeper-adjusted remaining
  league money and slots. Recorded league keepers are excluded from the pool and
  retain their recorded spend.
- Construction uses only pre-origin weekly donors. Raw target-season outcomes are
  used only for scoring, and forecast EV uses an independently seeded context bank.

## Five-draw buffer frontier

Unqualified points include rosters that exceed the cap at historical final
prices. Feasibility and overage therefore must be read with the point totals.

| Nominal buffer | 2022-24 points | 2022-24 feasible | 2022-24 overage | 2025 points | 2025 feasible | 2025 overage |
|---|---:|---:|---:|---:|---:|---:|
| None | 1593.1 | 20.8% | $27.38 | 1598.7 | 15.6% | $26.37 |
| $0 | 1573.2 | 27.6% | $20.60 | 1598.2 | 19.2% | $22.35 |
| $5 | 1579.8 | 25.3% | $22.66 | 1600.0 | 16.8% | $24.78 |
| $10 | 1587.0 | 23.2% | $23.87 | 1599.9 | 15.6% | $25.31 |
| $15 | 1586.2 | 22.7% | $24.91 | 1599.1 | 15.6% | $26.23 |
| $25 | 1590.7 | 21.2% | $26.24 | 1598.7 | 15.6% | $26.37 |

Relative to no nominal row, the five-draw `$5` rule:

- lost 13.3 points over 2022-2024, or 0.83 points per scored week;
- improved realized-price feasibility by 4.5 percentage points and reduced mean
  overage by $4.72 over 2022-2024;
- gained 1.3 points, improved feasibility by 1.2 points, and reduced overage by
  $1.59 in the 2025 temporal check;
- changed 46.1% of rosters and rejected 35.7% of unconstrained nominal totals
  across all four origins; in 2025 those rates were 35.2% and 25.6%; and
- changed points by -9.2 and -10.1 in the two Monte Carlo trial halves while
  reducing overage by $3.71 and $4.17. The affordability direction was stable.

The five-draw `$10` rule was less active. It lost 6.1 development-period points,
improved feasibility by 2.4 percentage points, and reduced overage by $3.52. In
2025 it gained 1.2 points, left feasibility unchanged, and reduced overage by
$1.06. This is a valid choice when preserving forecast points is weighted more
heavily than the guardrail's affordability effect.

## One draw versus five

With no nominal row, one draw gained 35.1 development-period points and 10.6
points in 2025, but its feasibility rates were only 5.7% and 6.0%, versus 20.8%
and 15.6% for five draws. Its mean overage was $30.09 higher in development and
$14.08 higher in 2025.

The nominal row removed most of the apparent point advantage but not the risk. At
the provisional `$5` buffer, one minus five draws was:

- -0.3 points, -3.6 feasibility points, and +$2.17 overage in 2022-2024; and
- +5.5 points, -4.4 feasibility points, and +$1.26 overage in 2025.

Across all four origins, the point difference was +1.2 while one draw was 3.8
feasibility points worse and $1.95 more over cap. Five draws therefore remains
the better risk-adjusted construction within this tested family.

## Calibration and interpretation boundary

The point cost is concentrated in the oldest origin: `$5` lost 32.1 points and
`$10` lost 20.2 in 2022. Over 2023-2025, their mean effects were -2.1 and +1.0
points, respectively. The 2022 frozen point salary surface represented only
$2,257 of the $2,886 keeper-adjusted market before
normalization, compared with much smaller gaps in 2024-2025. This makes 2022 useful
as a stress case but weak evidence for the exact 2026 tradeoff.
Its point-salary table also missed 145 of 255 selectable players, which received
the intentional low fallback; selected five-draw rosters contained 1.7 such
players on average, so direct exposure was smaller than the pool-level miss rate.

The five-draw jointly feasible point effect was -2.0 across 17.9% of `$5` pairs
and -0.1 across 19.1% of `$10` pairs. These are not causal performance estimates
because joint feasibility is selected using future final prices. Four seasons
remain only four independent football outcomes; split halves measure Monte Carlo stability, not
season-to-season generalization.

Historical final prices are exogenous. A different nomination sequence could have
changed them, and missing final prices intentionally use a `$1` fallback, making
reported feasibility optimistic. The frozen 2023-2025 salary laws are mainly the
legacy truncated-normal method, not a walk-forward rebuild of the current 2026
empirical residual-quantile method.

## Recommended implementation sequence

1. Keep five-draw averaging and expose the nominal guardrail as a separate option,
   with `+$5` as the provisional affordability setting and `+$10` as the explicit
   looser, point-preserving alternative.
2. Count acquired players at deterministic paid prices in both the sampled and
   nominal rows. Do not replace keeper or acquired-player salaries with draws.
3. Re-run the same locked buffer comparison after rebuilding the current salary
   method independently at each historical origin. Tune on 2022-2024 before
   inspecting 2025.
4. Compare the cheap hard row with coherent salary scenarios or a small chance
   constraint. A roster-level upper-price or tail-risk rule targets optimizer's
   curse more directly than a point estimate alone.
5. Add sequential auction-state logging so Current Nomination affordability can be
   replayed from actual nominations, prices, roster, and budget rather than final
   prices alone.

## Validation

The run completed all 12,000 expected cells. All solves were optimal; sampled and
nominal caps, 13 unique players, position bounds, Top-N, causal donor timing, actual
overage identities, and keeper-adjusted market totals passed. The 2,000
unconstrained controls reproduced the parent replay exactly with zero mismatches.

See `summary.md` for generated aggregate tables, `source_manifest.json` for source
hashes and invariants, and the paired CSVs for trial-level comparisons.
