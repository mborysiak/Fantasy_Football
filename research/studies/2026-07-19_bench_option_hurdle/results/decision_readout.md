# Decision Readout: Bench Call Options Versus A Higher Waiver Hurdle

## Decision

Do not raise the production waiver baseline as a proxy for a studs-and-scrubs
bench. Keep the current evaluation baseline because it represents the steady
replacement points available from waivers; changing it also distorts starter
and FLEX value rather than isolating bench option value.

Do not promote a new bench objective from this study yet. Removing the current
0.25 weekly-P90 bench heuristic (`bench0`) is a plausible simple improvement,
and the 0.25 sustained-breakout utility is the more promising structural
alternative, but the latter needs stricter breakout calibration and the current
salary-risk controls before it is safe to use live.

## Test Design

- Replayed frozen preseason origins for 2022-2025 with 250 paired trials per
  origin, 250 independent construction contexts, and 250 independent forecast
  evaluation contexts.
- Held five salary draws, the Top-12 rule, roster constraints, and the current
  projected waiver baseline constant during evaluation.
- Tested construction-only RB/WR/TE waiver hurdles of `+1`, `+2`, and `+3` PPG.
- Compared the current 0.25 weekly-P90 bench heuristic with no bench heuristic
  and with 0.25/0.50 sustained-breakout utilities.
- Defined a modeled option as a player who is initially below the positional
  impact threshold, becomes detectable from the prior three played weeks,
  remains above the threshold in at least three of the next four played weeks,
  and then supplies positive future points above it. The causal detection rule
  deliberately allows late-season breakouts and does not impose a four-week
  drop rule.
- Scored every roster on a common independent mean, p10, and p90 forecast bank,
  then on raw target-season outcomes. The option bonus is a construction
  preference, not extra fantasy points in evaluation.

All 7,000 solves were optimal, produced 13-player rosters, fit the forecast cap,
and used the locked evaluation waiver baseline.

## Main Results

Paired changes versus the current 0.25 weekly-P90 policy:

| Policy | Period | Forecast mean | Forecast p10 | Actual points | Weeks 13-16 | Bench spend | Sustained 15+ hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No bench heuristic | 2022-2024 | +0.4 | +0.6 | +6.6 | +0.9 | -$1.4 | +0.05 |
| No bench heuristic | 2025 | +2.3 | +0.7 | +13.3 | -1.0 | -$3.2 | +0.02 |
| Waiver hurdle +1 | 2022-2024 | -1.4 | -0.6 | -2.9 | -1.4 | +$2.9 | +0.11 |
| Waiver hurdle +1 | 2025 | -1.9 | -0.5 | -0.0 | +2.0 | +$2.4 | +0.08 |
| Waiver hurdle +2 | 2022-2024 | -25.7 | -23.1 | +10.1 | -2.9 | -$0.9 | +0.02 |
| Waiver hurdle +2 | 2025 | -2.9 | -0.8 | -1.8 | +0.5 | +$4.0 | +0.03 |
| Waiver hurdle +3 | 2022-2024 | -50.8 | -49.5 | +6.9 | -5.4 | -$8.7 | -0.41 |
| Waiver hurdle +3 | 2025 | -19.2 | -15.7 | -0.6 | -2.1 | +$1.8 | -0.03 |
| Sustained option 0.25 | 2022-2024 | +13.5 | +13.9 | +32.1 | +25.7 | +$1.0 | +0.77 |
| Sustained option 0.25 | 2025 | +2.3 | +1.0 | +2.8 | +4.2 | -$0.4 | -0.48 |
| Sustained option 0.50 | 2022-2024 | +11.3 | +12.2 | +27.4 | +30.3 | +$3.3 | +0.92 |
| Sustained option 0.50 | 2025 | +0.8 | -0.9 | +4.0 | +5.6 | +$1.2 | -0.51 |

The higher-hurdle result is decisive enough to reject the proxy. Even `+1`
made the nominal bench about `$2.4-$2.9` more expensive, reduced top-three
spend concentration, and lowered common-bank forecast mean and p10. The `+2`
and `+3` rules increasingly discarded legitimate weekly lineup value; `+3`
eventually produced a cheaper, more concentrated development roster only by
giving up roughly 51 forecast points and reducing literal sustained 15+ hits.

Removing the existing P90 heuristic moved in the desired structural direction:
the bench became `$1.4` cheaper in development and `$3.2` cheaper in 2025,
top-three spend concentration rose, and forecast mean and p10 did not decline.
Its forecast gain is small relative to Monte Carlo uncertainty, however, and it
did not improve 2025 playoff-week points. Treat this as a candidate for the next
live-method replay, not a completed promotion case.

The 0.25 sustained option was the best balanced experimental objective. It
improved independent forecast mean, p10, and p90 in every origin, and its
development effects were stable across trial halves. It also improved raw
weeks 13-16 scoring in development and 2025. The 0.50 weight was less balanced:
it spent more on the bench and slightly reduced 2025 forecast p10.

## Why The Option Result Is Not Yet Production-Ready

The modeled positional impact thresholds were about 11.3-11.9 PPG for RB/WR
and 9.7-10.5 for TE, not a literal 15 PPG threshold. In 2025 the 0.25 policy
added 0.12 sustained 12+ PPG bench hits but lost 0.48 sustained 15+ hits per
roster. It still gained forecast and playoff points, which says it found useful
lineup-level options, but not reliably the rare ceiling outcome the intended
strategy values most.

The option policy also changed every roster and did not inherently enforce a
cheap bench. In development it added about `$1` of modeled bench spend and
slightly reduced top-three concentration. A breakout preference and a
studs-and-scrubs budget preference therefore need to be represented separately.

Finally, historical-price feasibility was poor for every policy. The 0.25
option reduced it by 3.2 percentage points in development and 6.4 points in
2025, with mean historical overage rising about `$8` and `$12`. Those figures
use the frozen legacy salary surface rather than the current v5 plus selection
reserve, so they are a warning rather than a clean estimate of current live
affordability.

## Recommended Next Test

Retain the current live waiver baseline and production bench weight while
testing a price-aware 0.25 call-option objective on the current live method:

1. Separate the replacement floor from the option strike. Keep waivers at the
   steady attainable baseline, but test a stricter absolute/positional breakout
   strike that better targets sustained 15+ PPG outcomes.
2. Add a distinct cheap-bench control, such as an option-value-per-decision-
   dollar preference or soft bench-spend penalty. Do not encode cheapness by
   inflating replacement points.
3. Compare that policy with `bench0` using the current v5 market salary,
   selection reserve, and converged organic construction path.
4. Require independent forecast mean and p10 not to decline, 2025/future
   sustained-ceiling calibration to improve, playoff-week value to remain
   positive, and affordability not to worsen materially before promotion.

Full result tables are in `policy_summary_by_year.csv`,
`policy_summary_across_years.csv`, `paired_effects.csv`, and
`paired_effects_by_period.csv`.
