# Staged Target Evidence Result

## Current method

The live Target Board now separates discovery, forced estimation, and evidence
maturity without letting probability gates replace the ranking metric:

1. 320 organic trials run in eight balanced 40-trial blocks.
2. Four shared pilot scenarios scan the eligible pool for four discovery slots.
3. Twenty protected heuristic candidates plus four pilot discoveries receive 64
   fresh preliminary Buy/Pass pairs.
4. The ten evidence-priority players plus missing members of the four highest
   Market `$` usable preliminary candidates receive 96 fresh pairs, producing
   10-14 unique confirmations.

Pilot outcomes are discarded after cohort selection. Market salary controls
confirmation allocation and visibility only. Neither pilot nor price enters the
forced posterior as evidence.

Forced effects retain a position-neutral leave-one-player-out random-effects
prior and eight fixed logical blocks. Organic Gain remains separate because it
is a selection-weighted normal-policy contribution rather than the forced
Buy-minus-Pass estimand.

Every usable forced screen ranks on posterior LCB80 (`mean - 0.842 * SE`). With
two stages, excess preliminary/confirmation variation is estimated by the
two-study method-of-moments variance and added to each stage before recomputing
the posterior. Consistent confirmation earns precision; disagreement increases
SE continuously. Strong/mixed/negative/conflicting confirmation and low-fit
pivot are evidence or strategy labels only. They do not switch the row back to
Organic Gain or change its rank family.

Forced rows appear first in continuous LCB80 order. Organic-only rows follow in
a separate Organic Gain watchlist. `Target`, `Watch`, and `Pass` labels summarize
whether LCB80, posterior mean only, or neither is positive without changing the
numeric ordering.

## Fixed-roster edge case

The deterministic 20260717 replay fixed Jayden Daniels (`$15`), Jahmyr Gibbs
(`$108`), Chase Brown (`$34`), and Bhayshul Tuten (`$11`). It completed the full
320/4/64/96 process in 19.1 seconds with eight workers.

- All eight organic logical blocks contained 40 trials.
- The pilot discoveries were Drake London, Emeka Egbuka, Keaton Mitchell, and
  Tyjae Spears; all 20 protected heuristic candidates remained.
- Ten evidence-priority candidates plus Bijan Robinson, Jonathan Taylor,
  Ja'Marr Chase, and Devon Achane produced the maximum 14 confirmations.
- Chris Olave ranked first at LCB80 `+9.85`; the top nine rows carried strong
  confirmation in this seed.
- Ja'Marr Chase remained a Target at LCB80 `+1.55` with mixed confirmation; his
  large stage-disagreement estimate raised posterior SE from `3.38` to `4.61`
  instead of causing a categorical demotion.
- Bijan Robinson ranked 16th at LCB80 `-0.69` with mixed confirmation. He stayed
  on the same forced scale rather than reverting to Organic Gain.

The fast synthetic contract passed continuous forced ordering, exact LCB80,
preliminary-versus-confirmation disagreement inflation, market-anchor union,
pilot isolation, balanced organic blocks, and dynamic Top-N replacement.

A fixed-seed 100-organic replay matched exactly across one and eight worker
processes for every result row, summary value, pilot discovery, preliminary and
confirmation cohort, evidence label, disagreement estimate, and rank. The
serial and parallel runs took 33.0 and 14.2 seconds respectively. Hardware now
changes runtime only; fresh production runs still receive new recorded seeds.

## Limits

The empirical prior is still learned from each run's adaptive preliminary
cohort, and the pilot remains a recall mechanism rather than inferential
evidence. The new allocation should be validated across representative draft
states and multiple paired seeds for top-rank stability and missed-regret.
Current Nomination remains the authority for exact live player/price decisions.
