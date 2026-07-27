# Target Roster Refinement Results

This single-swap production decision was superseded on 2026-07-19 by the
converged organic refinement policy in
`../../2026-07-19_target_roster_convergence/`.

The bounded one-swap correction materially reduced the fixed-base additive
roster bias without making the Target Board too slow for the live app.

## Correctness

- Synthetic redundant-position case: `ExtraRB -> StrongWR` improved exact
  managed points from 192 to 228.
- Fixed/manual salaries, cap, position bounds, roster size, and top-N cases
  passed.
- Thirty deliberately rounded/tied-decision cases matched exhaustive feasible
  one-swap search.
- The swap is selected before independent contribution/summary holdouts.

## Expected-Profile Choice

Selecting a swap on only the five bootstrapped ILP seasons improved those
construction outcomes but reduced paired holdout EV. The retained correction
therefore averages the full cached construction bank into one expected 16-week
profile before evaluating the swap. This retains the starter/FLEX interaction
signal without optimizing a roster against a few unusually favorable realized
seasons.

## Runtime And Outcome

- Seeded serial 50-trial block: 3.24 seconds without refinement and 3.51
  seconds with refinement (`+8.3%`).
- The expected-profile swap improved 48/50 rosters, averaging `+30.6` managed
  points on the construction-bank profile.
- Seeded production-style 500-trial run on eight workers: 8.50 seconds without
  refinement and 9.00 seconds with refinement (`+5.8%`).
- Paired holdout season EV increased from 1655.67 to 1667.40 (`+11.73`).

A 2026-07-19 compatibility rerun pinned `max_swaps=1` against the then-current
app database after the convergence-policy calibration refresh. It completed all
50 trials and 30 tied cases, improving 50/50 rosters; serial runtime was
`7.97s -> 9.09s` and holdout EV was `1520.21 -> 1542.32` (`+22.11`). The
machine-readable `benchmark.json` records this compatibility rerun; the values
above remain the original 2026-07-13 production evidence.

One pass is a bounded bias reduction, not a guarantee of full one-swap local
optimality. Repeated passes were intentionally rejected for the real-time path
because their incremental quality gain did not justify the added latency.
