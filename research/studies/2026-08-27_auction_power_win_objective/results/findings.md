# Findings

## Verdict

Keep the best-available waiver proxy as the promising construction change, but
do not add win probability, expected winning margin, or blended power-win
utility directly to the live objective yet.

The frozen LCB-selected primary test is a clean null. The additive waiver
control is also the exact full-roster mean frontier in all eight blocks. Every
0.5% and 1.0% LCB tail arm therefore retains the control roster. This is not
caused by a narrow candidate search: each block contains 130-131 unique
candidates generated from coherent single contexts, randomized subsets, the
full mean, and player-level P75/P90 marginal values. The mean guardrails are
still sparse, however: only 1-2 candidates per block survive 0.5%, and 1-4
survive 1.0%.

Removing the guardrail does not reveal a hidden championship frontier. Pure
tail objectives and standardized 50/50 mean-plus-tail objectives both overfit
the 64-context construction bank and make their own field-relative metrics
worse on 256 independent contexts per block.

## Direct-objective sensitivity

Removing the construction-stage LCB penalty and maximizing each point estimate
inside the 1.0% guardrail changes three of eight blocks. On the construction
bank, direct win/power gains average 0.40 percentage points of win proxy and
0.60 expected-winning-margin points while surrendering 3.37 expected points.
Direct expected excess gains 0.81 margin points while surrendering 4.62
expected points.

Those gains reverse on 256 independent contexts per block:

| Direct arm vs exact-mean control | Mean | P10 | P90 | Win proxy | Expected excess | Power utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Win / power | -3.08 | -3.03 | -2.25 | -0.091 pp | -0.121 | -0.00120 |
| Expected excess | -4.23 | -5.40 | -1.48 | -0.340 pp | -0.237 | -0.00429 |

Every corresponding LCB80 is negative. The LCB selector correctly rejects the
apparent construction-bank improvements.

The direct win/power roster spends about `$9.25` more at RB and `$7.88` less at
WR on average. Direct expected excess spends `$9.63` more at RB. Win/power does
not change the average 1.50 dead-zone RB count; expected excess reduces it by
only 0.125. Aaron Jones and James Conner remain the two most common dead-zone
backs. The tail objectives are not acting as a cheap-flier preference.

The direct win/power arm raises independent P(2+ legacy q90 difference-makers)
by 2.05 percentage points (LCB80 +1.18) even while every field-relative tail
metric declines. This confirms that absolute residual-plus-contribution upside
is not interchangeable with roster championship utility.

Actual 2025 scores are descriptive because the season was already reviewed.
Their point estimates are +1.85 for direct win/power and +12.68 for direct
excess, but both block-level LCB80s are negative and cannot override the fresh
simulation validation.

## Pure and 50/50 objective sensitivity

The pure arms maximize the construction-bank tail point estimate with no
expected-score constraint. The 50/50 arms average within-block z-scores for
expected score and the chosen tail metric, so the weights are comparable even
though their native units differ.

| Arm vs exact-mean control | Blocks changed | Mean | P10 | P90 | Win proxy | Expected excess | Power utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure win / power | 6/8 | -25.78 | -27.34 | -22.04 | -3.185 pp | -2.327 | -0.04104 |
| Pure expected excess | 7/8 | -30.17 | -33.35 | -32.72 | -3.437 pp | -2.146 | -0.04347 |
| 50/50 mean + win / power | 2/8 | -7.67 | -10.81 | -2.73 | -1.087 pp | -0.668 | -0.01383 |
| 50/50 mean + expected excess | 5/8 | -16.30 | -17.80 | -15.57 | -2.065 pp | -1.459 | -0.02641 |

Every row loses the field-relative metric it was built to improve. For
example, pure win gains 1.24 percentage points of apparent win probability on
the construction bank but loses 3.19 points independently. Pure excess gains
4.15 apparent expected-margin points but loses 2.15 independently. The 50/50
blend moderates the expected-score cost, but the reversal remains: its win arm
gains 0.62 construction points and loses 1.09 independent points; its excess
arm gains 3.41 construction margin points and loses 1.46 independently.

The roster shifts are large and do not specifically express the desired
cheap-young-upside preference. All arms still average the same 1 QB / 6 RB /
4 WR / 2 TE shape. Pure win/power moves `$34.50` from the rest of the roster
into RB on average, including `$29.88` out of WR. The 50/50 win arm moves
`$13.50` into RB and `$11.88` out of WR. Aaron Jones remains a 75% selection
under pure win and rises to 87.5% under 50/50 win; James Conner remains 62.5%
and 75%, respectively. This is a noisy RB-spend tilt, not a reliable dead-zone
avoidance mechanism.

One block makes the behavior concrete. The exact-mean roster spends `$121` at
RB and `$119` at WR. Pure win instead spends `$196` at RB and `$44` at WR,
adding Ashton Jeanty, Breece Hall, and Trey Benson while dropping Malik Nabers
and Rome Odunze. That looks superficially more explosive, but the complete
roster loses out of sample.

Pure and 50/50 expected-excess arms score much better on the already-observed
2025 realization (`+90.26` and `+60.92`). That split is interesting but cannot
validate the rule: 2025 outcomes motivated the experiment, and the independent
simulation cells point the other way. Treat those realized gains as a roster
case study, not confirmation.

## Implications

- Win probability and expected positive margin remain the right outputs for the
  user's risk preference. The failure is selection noise and frontier density,
  not the interpretation of those metrics.
- Raw point-estimate optimization overfits 64 construction contexts. Retain
  block-paired LCB or use cross-fitted/larger construction banks before a live
  decision rule.
- Neither removing the guardrail nor assigning a standardized 50% tail weight
  solves that problem. Do not tune a smaller weight against this same 2025
  replay; that would turn the sensitivity into another in-sample search.
- Power alpha 0.25 selects the same exploratory rosters as win probability.
  Expected excess supplies the distinct high-dominance sensitivity, so report
  the two-dimensional win/margin frontier instead of tuning alpha post hoc.
- A future confirmation should generate dense legal local swaps around the
  mean roster, use at least several hundred construction contexts or cross-fit
  the frontier, and validate multiple historical origins. Until then, display
  win/excess diagnostically rather than changing action rank.

## Reproducibility

The full study uses eight blocks, 64 construction contexts per block, 130-131
unique candidates per block, and 256 independent validation contexts per
block. Ten substantive generated result files are byte-identical across a
fixed-seed rerun; only the metadata runtime changes. Thirteen focused replay
tests pass, including six power-win-objective tests.
