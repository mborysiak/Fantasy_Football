# Managed Auction Replay Decision Readout

## Bottom line

Keep the current production settings for now: average-five salary construction,
projected waiver baselines, Top-N on, and bench-upside weight `0.25`. The replay
does not support changing the bench weight, the prior-empirical waiver alternative
lost points in every season, and Top-N provided valuable protection in the 2025
temporal check.

This is not an endorsement of average-five salary prices as the final method.
Averaging five draws badly contracts uncertainty, while switching directly to one
draw makes the selected rosters much less affordable. The next experiment should
replace price averaging with coherent salary-market scenarios plus an explicit
roster-level affordability or recourse rule.

## Replay contract

- Origins: frozen preseason 2022, 2023, 2024, and 2025 inputs.
- Design: 250 paired trials in each of 16 factorial cells, or 16,000 solved cells.
- Current profile: five salary draws, Top-N on, projected waiver baseline, bench
  weight `0.25`.
- Primary comparison: one setting at a time from that exact profile.
- Secondary comparison: factorial marginal effects averaged over all eight
  backgrounds of the other settings.
- Construction templates use only seasons before the origin. Target-season raw
  weeks are used only for realized scoring.
- Forecast EV uses a separately seeded 250-context evaluation bank.

## One-at-a-time results

Effects are candidate minus current. Unqualified points score every selected
roster, including rosters that would exceed the cap at historical final prices.

| Candidate change | Season points | 2022-24 | 2025 check | Cap-feasibility | Roster changed |
|---|---:|---:|---:|---:|---:|
| Bench weight `0` | -3.4 | -6.8 | +6.7 | -1.3 pp | 86.2% |
| Prior-empirical waiver | -22.5 | -27.9 | -6.4 | +6.0 pp | 99.3% |
| One salary draw | +29.0 | +35.1 | +10.6 | -13.7 pp | 99.0% |
| Top-N off | -4.9 | +2.7 | -27.7 | +1.3 pp | 33.8% |

The corresponding factorial marginal effects were -0.8, -21.9, +30.5, and
-5.3 points. Salary-draw, waiver, and Top-N directions therefore were not artifacts
of evaluating only the exact-current-profile corner. Bench weight remained unstable.

First-half versus second-half equal-season point effects were:

| Candidate change | Trials 0-124 | Trials 125-249 |
|---|---:|---:|
| Bench weight `0` | -8.9 | +2.1 |
| Prior-empirical waiver | -20.3 | -24.8 |
| One salary draw | +27.6 | +30.4 |
| Top-N off | -2.2 | -7.6 |

This split checks Monte Carlo stability only. There are still just four independent
season outcomes.

## Salary uncertainty and affordability

Across 518 player-origin observations, averaging five independent salary draws
reduced forecast dispersion to about 45% of the one-draw dispersion, close to the
mechanical `1 / sqrt(5)` contraction.

| Salary construction | 80% coverage | 90% coverage | Mean SD | CRPS | WIS |
|---|---:|---:|---:|---:|---:|
| One draw | 65.1% | 73.9% | 4.37 | 3.47 | 2.88 |
| Average of five | 31.1% | 39.0% | 1.97 | 3.85 | 3.46 |

Point MAE barely changed, so average-five did not improve the salary mean; it
mostly removed dispersion. One draw still undercovered, especially in 2024, and
these historical origins mainly use the salary distributions available at those
dates rather than a walk-forward rebuild of the current 2026 residual-quantile
method.

The current profile fit historical final prices in only 195 of 1,000 trial-origin
rosters:

| Origin | Realized-cap feasible |
|---|---:|
| 2022 | 5.2% |
| 2023 | 36.0% |
| 2024 | 21.2% |
| 2025 | 15.6% |

Mean current-profile forecast spend was $292.85, versus $322.89 at historical
final prices: a $30.05 selected-roster gap. Overall player-level salary means were
near unbiased, but selected players had systematically underpredicted prices. This
is the optimizer's curse at roster level. Missing final prices were filled at $1,
so the affordability rate is, if anything, optimistic; selected rosters nevertheless
had 92.9%-98.5% final-price match coverage by year.

One draw's apparent +29-point gain is not an actionable policy win. It added $27.67
of realized spend, increased mean cap overage by $26.09, and cut feasibility from
19.5% to 5.8%. Only 4.0% of exact-profile pairs were jointly feasible, with none in
2022, so feasible-only point comparisons are too selected to resolve that tradeoff.

## Decisions supported now

1. Retain the projected waiver baseline. The prior-empirical alternative lost
   29.1, 21.5, 33.1, and 6.4 points in 2022-2025 and caused more waiver starts.
2. Retain Top-N as a safeguard. Turning it off was nearly neutral in 2022-2023,
   positive in 2024, and strongly negative in 2025. Test softer or differently
   sized elite-exposure rules later instead of removing it now.
3. Leave bench weight at `0.25`. Its sign changed by season and by trial half; the
   replay does not identify a better setting.
4. Do not replace average-five with one draw in production yet. One draw represents
   uncertainty better, but the current optimizer turns that variance into
   unaffordable bargain bundles.

## Highest-priority next study

Run a walk-forward replay of the current salary method with roster-level price risk:

1. Rebuild the current residual-quantile salary method independently at each origin.
2. Preserve each normalized salary market as a coherent scenario; do not average
   player prices before solving.
3. Start with a cheap forecast-cap haircut grid from $250 to $290, then compare a
   price quantile, chance constraint, or CVaR-style penalty. Tune on 2022-2024 and
   lock the rule before checking 2025.
4. Add second-stage recourse: when realized prices differ, repair or re-solve the
   remaining roster without using player outcomes. Report points only for feasible
   policies, rather than conditioning on the small naturally feasible subset.
5. Log live nomination order, price, roster, budget, settings, model version, and
   recommendation state so future Current Nomination replays can use the actual
   sequential auction path.

After salary affordability, the next scoring improvement is realistic waiver and
availability modeling: pre-kickoff participation probabilities, persistent claims,
opponent competition, and a correction to the refinement step that currently
averages context scores but ORs played masks.

The next user-facing validation should force each Target candidate into Buy and Pass
rosters at the historical origin, then compare predicted Fit/Gain ordering with the
realized feasible difference. Current Nomination requires new live event logging;
historical final prices alone cannot reconstruct its sequential decisions.

## Interpretation boundary

- Historical final prices are exogenous diagnostics. Different nominations or bids
  could have changed them.
- The replay validates empty-roster Target/look-ahead construction, not keeper
  choice, Target Board candidate Gain/Fit ordering, or Current Nomination decisions.
- Realized waiver ranking is causal, but eligibility uses target-week played evidence
  and assumes frictionless weekly re-picking without opponent competition.
- The hindsight roster is a local heuristic, not a global oracle.
- 113 of 1,003 forecast players lacked exact target-feature matches. Exposure was
  generally low, but 2024 Rashee Rice used fallback construction features and was
  selected frequently, so that origin merits a feature-fallback sensitivity check.
- Missing raw outcome rows are treated as missed weeks and waiver-covered, not as
  scored zero. This is usually appropriate for nonparticipants but still depends on
  source-row coverage.
- Frozen files precede target results, but original preseason feature revisions are
  not independently timestamped in every origin.

See `summary.md` for the generated tables, `source_manifest.json` for hashes and
invariants, and the paired CSVs for trial-level results.
