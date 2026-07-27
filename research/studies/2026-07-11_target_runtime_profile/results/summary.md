# Target Runtime Findings

## Test Configuration

- 2026 beta auction market with 14 keepers and 166 available players
- Empty managed roster, so each successful trial evaluates all 13 selected
  players; this is the highest-cost early-draft state
- One production-sized serial block: 50 trials, 50 managed construction
  contexts, five sampled construction contexts per trial, and five holdouts
- Parallel scaling from 50 through 1,000 trials using 50-trial tasks
- Exact GLPK solves and production stochastic settings were retained

## Serial Block

The uninstrumented 50-trial block completed in 5.771 seconds. Phase-aware
instrumentation completed in 5.932 seconds and reproduced every Target output
exactly.

| Phase | Instrumented seconds | Share of block |
| --- | ---: | ---: |
| Candidate Buy/Pass contributions | 4.886 | 82.4% |
| Five-context holdout generation | 0.340 | 5.7% |
| 50-context construction bank | 0.324 | 5.5% |
| Remaining outer solve, summaries, and bookkeeping | 0.382 | 6.4% |

The outer roster ILP itself took only 0.059 seconds across 50 solves.

## Candidate Contribution Breakdown

Each trial selected 13 nonfixed players, producing 650 candidate evaluations,
1,300 contribution ILPs, and 1,300 exact holdout roster scores.

| Component | Calls | Seconds | Share of block |
| --- | ---: | ---: | ---: |
| Candidate-rebased managed marginal values | 1,938 | 1.866 | 31.5% |
| Forced-Buy GLPK | 650 | 0.693 | 11.7% |
| Forced-Pass GLPK | 650 | 0.808 | 13.6% |
| Full prediction DataFrame copies | 650 | 0.695 | 11.7% |
| Exact five-context roster scoring | 1,300 | 0.281 | 4.7% |
| Candidate salary normalization | 650 | 0.125 | 2.1% |
| Salary rows plus CVXOPT conversion | 5,200 operations | 0.098 | 1.7% |

All 1,350 outer and contribution GLPK solves were optimal. Pass solves were
about 17% slower than Buy solves, but no solver timeouts or infeasible branches
occurred.

## Marginal-Value Work

`managed_base_lineup_state` consumed 1.214 of the 1.866 marginal-value seconds,
or 65% of that component and 20.5% of the full block. It repeatedly adds waiver
players and loops through 16 weeks and four positions to sort lineup thresholds.
This is the largest deeper vectorization target.

The existing candidate/context cache is effective and should remain:

- 3,146 candidate-context requests
- 1,938 calculations
- 1,208 cache hits, or 38.4%
- at the observed mean cost, the cache already avoids roughly 1.16 seconds per
  block

## Exact Roster-Score Reuse

Within each trial, Buy and Pass branches share the same five holdout contexts.
Of 1,300 score calls, only 634 trial/roster combinations were unique:

- Buy: 650 calls but only 50 unique rosters, so 92.3% were reusable
- Pass: 650 calls and 584 unique rosters, so 10.2% were reusable
- Combined: 666 reusable calls, or 51.2%

A per-trial exact score cache is behavior-preserving. Based on measured scoring
time, its direct saving is about 0.14 seconds per block; useful, but smaller than
the salary and marginal-value opportunities.

## Salary Workspace

The contribution path copies a 166 x 1,003 prediction frame once per candidate.
Each frame is about 1.35 MB, producing approximately 876 MB of DataFrame copy
traffic per 50-trial block. The full copies plus pandas salary normalization
cost 0.820 seconds. Porting the nomination NumPy salary workspace pattern here
should recover most of that time while preserving salary calibration exactly.

The Buy and Pass salary constraint rows are also identical after fixed salaries
are applied, but are currently generated twice. Removing that duplication is
exact but worth only about 0.03 seconds per block.

## Parallel Scaling

| Trials | Workers | Seconds | Trials/sec | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 1 | 5.914 | 8.5 | 97.6% |
| 100 | 2 | 6.710 | 14.9 | 86.0% |
| 200 | 4 | 7.595 | 26.3 | 76.0% |
| 400 | 8 | 9.797 | 40.8 | 58.9% |
| 800 | 16 | 17.992 | 44.5 | 32.1% |
| 1,000 | 16 | 21.451 | 46.6 | 33.6% |

Spawn, import, template loading, and shutdown overhead rises sharply:

| Workers | Startup/shutdown seconds |
| ---: | ---: |
| 2 | 0.89 |
| 4 | 1.56 |
| 8 | 2.94 |
| 10 | 4.54 |
| 16 | 9.41 |

A fixed-worker sweep produced identical Target outputs at every worker count:

- 800 trials: 8 workers 15.760s, 10 workers 15.775s, 16 workers 16.986s
- 1,000 trials: 8 workers 20.007s, 10 workers 18.574s, 16 workers 19.527s

Ten workers is the best general cap for the current 500-1,000 trial UI range:
500 trials already creates only ten blocks, 800 is effectively tied between
eight and ten, and 1,000 benefits from two balanced waves of ten blocks.

## Recommended Order

1. Reduce the Target worker cap from 16 to 10. This preserves seeded task
   outputs and improved the measured 800-1,000 trial runs by about 5-7%.
2. Replace candidate DataFrame copies and pandas salary normalization with a
   lean NumPy salary workspace, and reuse the paired Buy/Pass salary row.
3. Add a per-trial exact roster-score cache inside contribution evaluation.
4. Vectorize `managed_base_lineup_state`; benchmark exact threshold and marginal
   value equivalence before changing the app.
5. Retain exact GLPK and the current candidate/context cache. Matrix conversion,
   static matrix construction, and global solver tuning are not priority work.

A persistent warm process pool could remove several seconds of startup, but it
would need explicit invalidation for league, player, salary, roster, and UI
state. The lower worker cap and inner-loop changes are substantially safer first
steps.

Later-draft states should be faster because fixed players reduce the number of
candidate contribution pairs below 13 and drafted players shrink the market.
The measured profile therefore represents the expensive early-draft case.

## Implemented Result

All four recommendations were implemented in
`Fantasy_Football_App/app/zSim_Helper.py` on 2026-07-11, with the user-selected
worker default of eight rather than ten.

- Three repeated uninstrumented 50-trial runs averaged 3.783 seconds versus the
  saved 5.560-second baseline, a 32.0% runtime reduction.
- The complete 30-row seeded Target output matched the pre-change board exactly
  after CSV round trip; season EV and every displayed player statistic were
  unchanged.
- Contribution time fell from 4.617 to 2.840 seconds, a 38.5% reduction.
- Vectorized base-lineup state fell from 1.226 to 0.491 seconds across the same
  1,938 calls, about 2.5x faster.
- Candidate DataFrame copies fell from 650 to zero, salary normalization moved
  to 650 lean array calls, and paired salary rows fell from 1,300 to 650 in the
  profiled empty-roster state.
- Exact holdout scoring calls fell from 1,300 to 634 through per-trial roster
  caching.
- With eight workers, 500 trials completed in 8.811 seconds, 800 in 12.043
  seconds, and 1,000 in 14.445 seconds.
- Relative to the pre-change eight-worker sweep, runtime improved 23.6% at 800
  trials and 27.8% at 1,000 trials, with identical seeded outputs.

Verification covers 500 captured pre-change lineup-state calls, 500 randomized
fixed-player salary workspaces, the exact seeded Target board, a fixed-roster
next-year smoke run, Streamlit AppTest, and the live app server.

## Batched Marginal Follow-up

The remaining candidate marginal-value work was batched on 2026-07-11:

- Target now groups every uncached selected candidate by managed context and
  calculates the full candidate-value matrix once per context.
- Current Nomination calculates its Pass and Buy objective rows together.
- The base-roster threshold selector remains exact and is evaluated once per
  base. A fully tensorized threshold prototype was rejected because unstable
  sorting changed displacement behavior when decision scores tied.
- Managed P90 now uses partial selection rather than a full percentile sort.
  Its interpolation mirrors NumPy's input dtype and stable linear interpolation,
  producing bit-for-bit equality with `np.percentile` in randomized checks.

Five repeated 50-trial runs averaged 3.299 seconds versus the prior 3.783-second
implementation, a further 12.8% reduction. Candidate contribution time fell
from 2.840 to 2.343 seconds, while the 650 Buy and 650 Pass GLPK solves remained
flat. The 25-trial cProfile run fell from 2.662 to 2.167 seconds.

The eight-worker 500-trial benchmark fell from 8.811 to 7.726 seconds, a 12.3%
wall-clock reduction. Verification includes 200 randomized batched marginal
cases, exact partial-percentile checks across 1-20 week inputs, the seeded
30-row Target board, syntax, Streamlit AppTest, and live HTTP 200.
