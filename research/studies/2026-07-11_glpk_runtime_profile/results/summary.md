# GLPK Runtime Findings

## Test Configuration

- 2026 beta auction market with 14 keepers and 166 available players
- Current Nomination: Saquon Barkley at $76, 100 paired trials
- Target: 25 serial trials with five contribution holdouts
- GLPK diagnostic time limit: 100 ms; no sampled solve reached the limit
- Current Nomination evaluated 17 unique prices, producing 3,500 ILP calls

Profiling overhead materially inflates Python runtime, so component shares and
call counts are more reliable than the profiled wall time itself. Solver
microbenchmarks were run outside cProfile.

## Current Nomination

| Component | Calls | Profiled seconds |
| --- | ---: | ---: |
| GLPK ILP | 3,500 | 9.66 |
| Pandas salary-row construction | 3,500 | 3.99 |
| Five-context roster scoring | 3,202 | 2.26 |
| Pandas salary normalization | 1,700 | 1.59 |
| Managed marginal-value setup | 500 | 0.73 |
| Weekly-template setup | 250 | 0.61 |
| CVXOPT matrix conversion | 10,506 | 0.23 |
| Static matrix construction | 3 | 0.04 |

Pass solves were harder than Buy solves: mean profiled GLPK time was 3.17 ms
for Pass versus 2.40 ms for Buy. Of 1,700 Buy attempts, 99 were infeasible;
the code still performed their corresponding Pass solve before discarding the
pair.

## Exact Data-Path Opportunities

- `create_G_salaries` took about 0.58 ms per call outside cProfile. An exact
  NumPy equivalent took 0.002 ms, about 288x faster.
- Copying the full 166x1,003 projection frame and creating its Pass subset took
  about 2.09 ms per price/trial. The 1,000 sampled projection columns are not
  used by these non-scoring nomination ILPs.
- NumPy salary-market normalization matched pandas exactly and took 0.029 ms,
  versus 0.477 ms on a lean pandas frame.
- A score cache keyed by scenario, selected roster, and evaluation contexts has
  an observed upper-bound reuse rate of 65.6% for Pass and 23.0% for Buy.
- Skip Pass construction immediately when the paired Buy problem is infeasible;
  this would have avoided 99 Pass solves in the profile.

Together, a lean array-only scenario path plus scoring cache is the most likely
exact route to a 20-35% Current Nomination improvement. It should be benchmarked
after implementation rather than adding the isolated microbenchmark savings,
which overlap.

## Matrix And Solver Experiments

- The current dense `G` matrix is only about 2.3% nonzero, but sparse conversion
  generally improved representative GLPK solves by only 3-12%.
- Removing 165-166 redundant `-x <= 0` rows preserved objectives and selected
  rosters in captured problems, but did not improve the end-to-end nomination
  benchmark. GLPK presolve already handles much of this redundancy.
- Reusing CVXOPT matrices is low priority: all matrix conversions together were
  under 1% of profiled nomination time.
- Global GLPK branching/backtracking options were instance-sensitive. Best-
  projection backtracking helped one captured Pass problem but made the full
  50-trial nomination 27% slower.
- A 0.5% MIP gap was faster but changed selected rosters and objectives in
  representative Buy and Target problems. It is not behavior-preserving.
- SciPy/HiGHS was substantially slower on these small binary problems and had
  problematic process/import behavior in this environment. GLPK remains the
  appropriate backend.

## Target

Target is not primarily GLPK-bound:

| Component | Calls | Profiled seconds |
| --- | ---: | ---: |
| Managed marginal values | 993 | 3.93 |
| GLPK ILP | 675 | 1.60 |
| Pandas salary rows | 675 | 0.89 |
| Weekly-template sampling | 50 | 0.56 |
| Five-context lineup scoring | 650 | 0.48 |
| Salary normalization | 350 | 0.38 |

The same NumPy salary-row path will help Target, but its next major optimization
remains candidate marginal-value reuse or reduction, not GLPK tuning.

## Operational Finding

Two abandoned parallel Target pools remained resident with one worker in each
pool consuming a full CPU core and all siblings idle. The pools held several GB
of memory until explicitly terminated. The profiling sample completed every
solve within 100 ms, so this does not prove that GLPK itself caused those old
workers. Process-pool cancellation/cleanup and a defensive per-solve time limit
should be audited separately before relying on repeated live parallel runs.

## Recommended Order

1. Build a lean NumPy nomination scenario path and NumPy salary rows.
2. Cache paired roster scores across evaluated prices.
3. Skip Pass when Buy is infeasible.
4. Reprofile and then consider parallel chart-price anchors.
5. For Target, focus on managed marginal-value work rather than solver options.

## Implemented Result

The first three recommendations were implemented in
`Fantasy_Football_App/app/zSim_Helper.py` on 2026-07-11.

- The 50-trial exact benchmark fell from 7.406 seconds to 4.326 seconds, a
  41.6% runtime reduction (1.71x speedup).
- All numeric and nonnumeric outputs matched exactly across the full 18-price
  curve, including Buy/Pass EV, Buy Edge, win rate, fit rate, expected starts,
  decision tier, and alternatives.
- A 50-trial call audit avoided 48 Pass solves after infeasible Buy branches.
- Cross-price roster caching reduced exact lineup-scoring calls from 1,704 to
  894, a 47.5% reduction.
- The normal 100-trial nomination completed in 7.402 seconds and reproduced the
  previously profiled result: Buy Edge `1.4055859375`, Roster Max Bid `$77`,
  Fit Rate `23%`, and Buy Win Rate `49%`.
- A managed Target smoke run and a nomination with an existing fixed player and
  nonzero next-year fraction both completed successfully.

The next runtime work should reprofile the updated path before considering
parallel price anchors. Target optimization should still focus on managed
marginal-value calculation rather than GLPK formulation changes.

## Candidate Review Follow-Up

On 2026-07-12, the app separated optional alternative sensitivity from the
primary Candidate Review and reduced the exact local/bridge chart-price grid.
The decision price and roster max bid still come from exact GLPK evaluations;
additional chart points are monotone display-only interpolation with at most a
five-dollar gap.

- The profiled full wrapper previously took 14.80 seconds: 10.91 seconds for
  the primary 100-trial result and 3.89 seconds for two 50-trial sensitivity
  scenarios.
- The updated deterministic 100-trial evaluator took 7.76 seconds and used 14
  exact prices instead of 19.
- A full Streamlit AppTest rendered the primary result in 7.85 seconds with 11
  exact prices. The optional sensitivity action then took 4.86 seconds and
  returned both scenarios without recomputing them during the initial review.
