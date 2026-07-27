# Decision Readout: Keeper-Aware Bench Call Option

## Decision

Keep the waiver baseline unchanged. The replacement floor and the keeper call
option are separate concepts; raising the waiver floor remains the wrong way to
force bench upside.

Do not promote a positive keeper weight to the live optimizer yet. The
position-specific PPG-to-market-dollar residual formulation is mechanically
sound, and the tiny tie-break improved independent current-season mean and p10
forecasts in every origin, but it did not improve realized next-year keeper
surplus across the three observable origins. The next-year signal also became
too concentrated in a few players.

If this option is exposed experimentally, use only a lexicographic/tiny
tie-break. A `0.01` scalar weight was effectively identical to `0.0001`;
weights of `1` and `10` changed most rosters without a stable keeper or
current-season gain. Keep the existing weekly-P90 bench bonus out of that
experimental policy so cheap bench opportunity cost remains explicit.

## League And Option Contract

- Two personal keeper slots.
- Salary rises by `$10` in each keeper season.
- Maximum three future keeper seasons.
- Fit separate monotone QB/RB/WR/TE curves from origin-year projected PPG to
  sampled origin-year Market `$`.
- Transform origin-frozen next-year PPG draws through the applicable curve.
- For acquisition price `p` and future value draw `v`, score
  `max(v-p-10,0) + max(v-p-20,0) + max(v-p-30,0)`.
- Restrict positive option value to non-QBs below the current positional impact
  threshold with a real origin-frozen next-year forecast.
- Activate at most two options, and identify them only from the five nominal
  bench slots.
- Evaluate current-season points with the unchanged waiver stream and no keeper
  utility added as fantasy points.

The three-year sum is a persistent-hit proxy. First-year realized surplus is
the primary keeper outcome because later outcomes are unavailable for recent
origins and a next-year hit does not guarantee three years of equal value.

## Same-Engine Results

The clean control is `keeper_engine0`: no weekly-P90 bench bonus, no keeper
bonus, and the same shortlist/exact-swap refiner as every keeper policy. Changes
below are for `keeper_tiebreak` (`lambda = 0.0001`) versus that control.

| Origin | Forecast mean | Forecast p10 | Actual season | Weeks 13-16 | Predicted option | Actual next surplus | Cost coverage: option/control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | +8.6 | +7.5 | +22.9 | +5.4 | +22.5 | -15.1 | 100% / 89% |
| 2023 | +4.4 | +5.5 | +6.5 | +2.2 | +39.4 | +0.9 | 86% / 85% |
| 2024 | +8.1 | +8.1 | -11.1 | -0.9 | +22.1 | +8.8 | 56% / 97% |
| 2025 | +10.4 | +12.1 | +8.7 | -7.6 | +29.4 | unavailable | unavailable |

The current forecast result is encouraging: mean and p10 rose together in all
four origins, so the option did not merely exchange the modeled floor for mean
upside. Actual season points improved in three of four origins, while playoff
points improved in two of four. Those are four historical outcome units, not
1,000 independent seasons.

The keeper result is not a promotion case. Mean realized first-year surplus was
`$30.37` for the tie-break and `$32.16` for the same-engine control. Keeper hits
were `0.605` versus `0.657` per roster. The 2024 improvement is especially
uncertain because observed acquisition salary covered only 56% of identified
options; missing actual prices are not silently treated as `$1`.

## Strength Frontier

- `0.01` changed only 2.0%-5.2% of tie-break rosters by origin and had
  negligible forecast, actual, and predicted-option effects. The choice has
  already saturated at tie-break scale.
- `1.0` changed 73.6%-85.6% of tie-break rosters. Its incremental realized
  next-year surplus was `-$0.08`, `+$3.94`, and `-$1.81` by observable origin.
- `10.0` changed 94.4%-96.4% of tie-break rosters. Its incremental realized
  surplus was `-$0.22`, `+$6.62`, and `-$5.45`.

The stronger weights therefore create a large policy change without a stable
return. They should not be interpreted as calibrated dollars in the
current-season objective.

## Concentration And Calibration Risk

The top two identified names represented 56.8%, 60.0%, 75.8%, and 96.2% of all
tie-break keeper selections from 2022 through 2025. The number of distinct
identified players fell from 11 to 8 to 5 to 4. In 2025, Jaydon Blue and Cam
Skattebo alone occupied 481 of 500 keeper slots; that outcome is not yet
observable. This is too concentrated for a signal whose direct realized keeper
validation consists of only three origins.

The frozen next-year source also changes across history: 2022-2023 reconstruct
a legacy `Model_Predictions` mixture, while 2024-2025 use
`Final_Predictions.pred_fp_per_game_ny`. The poor 2022 result and better 2024
result could reflect model evolution rather than a stable option relationship.

## Recommended Follow-Up

1. Treat `bench0` as the simple live-policy candidate: across both bench studies
   it makes the bench cheaper without reducing independent forecast mean or
   p10. Keep production unchanged until it is replayed with current v5 salaries,
   the selection reserve, and the converged organic path.
2. Rebuild the keeper signal on the current residual-calibrated next-year
   distribution and report both counterfactual modeled acquisition cost and
   observed historical salary. The former gives complete policy coverage; the
   latter remains an external affordability check.
3. Replace scalarized tie-breaking with a true lexicographic/multi-start rule:
   first maximize current managed value, then maximize two-slot keeper option
   among rosters inside a small current-value tolerance. This prevents a tiny
   coefficient from selecting a different local search basin and masquerading
   as calibrated keeper utility.
4. Add shrinkage or a portfolio concentration guard so one or two uncertain
   next-year forecasts cannot dominate nearly every roster.
5. Evaluate mean-EV versus p10/CVaR objectives as a separate factorial. This
   replay evaluates both mean and p10 on a common bank but does not optimize a
   distinct risk objective.

No production app or database behavior changed in this study.
