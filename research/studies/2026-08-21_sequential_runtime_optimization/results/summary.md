# Sequential Runtime Optimization Results

Verified on 2026-08-21 with the current 2026 beta database, compute budget 120,
four spawned workers, and fixed hidden evidence seed `3962081362`.

- Optimized bounded-reinvestment legality is exactly equal to the retained
  full-check path, including the complete sampled path table.
- Reference legality took 0.613 seconds; the pre-ranked/incremental path took
  0.488 seconds, a 20.4% reduction in this direct candidate replay.
- Immediate and deferred price-curve boards are frame-exact after attachment.
- Immediate profiling took 7.569 seconds. The market-only board took 6.639
  seconds, a 12.3% reduction in initial-board latency. The separately requested
  top-10 curves took 4.776 seconds.
- The board contains 64 rows, 18 market anchors, and 38 anchors after deferred
  profiling.
- **Add Evidence** after a prior max-bid calculation is exactly equal to the
  market-only accumulation path, returns max bids to `Deferred`, preserves the
  retained evidence seeds, and took 6.751 seconds for the added batch.
- The full Fantasy_Football_App suite passes: 90 tests in 1.721 seconds. The
  Streamlit bare-mode warnings are expected test-harness output.

The timings are mechanism checks rather than general hardware guarantees. The
parity assertions are the promotion gates.

## Remaining profile

A fresh four-worker market-only run took 6.865 seconds. Candidate barriers
accounted for 5.710 seconds (83.2%): 2.695 seconds for 24 base discoveries,
1.636 seconds for 40 expanded discoveries, and 1.379 seconds for 18 market
confirmations. The run performed 45,381 policy refreshes and 2,414 roster-score
computations.

The transparent one-worker cProfile attributes 13.939 cumulative seconds to
3,200 history-only branch rollouts. Inside that overlapping call tree,
`refresh_targets` accounts for 3.867 seconds, `bounded_upgrade_targets` for
3.285 seconds, full target-set legality for 2.492 seconds, and partial-target
feasibility for 1.997 seconds. Plan compilation accounts for 4.612 seconds and
roster scoring for 2.207 seconds. cProfile instrumentation increases absolute
runtime, so these values identify relative hotspots rather than app latency.

## Incremental ordinary-refresh challenger

An App v14 challenger maintained target spend, position counts, top-N
membership, unresolved position capacity, and minimum-first rebuild state while
ordinary refreshes considered additions or fallback swaps. It retained the v13
full legality validator for every accepted target set.

All complete paths and diagnostics were exact across the three prespecified
early/middle/late states and variations 0, 1, 2, and 14. Performance did not
pass the fresh-process gate:

- early Brown/Tuten/Gibbs averaged 0.437 seconds under v13 and 0.804 seconds
  under the challenger, 84% slower;
- middle Bowers averaged 0.398 versus 0.419 seconds, 5% slower;
- late-cap Pitts averaged 0.377 versus 0.374 seconds, effectively flat.

Warm repeated-process timings and cProfile instrumentation had misleadingly
favored the challenger because the reference's many short Python calls absorb
more profiler overhead and the native runtime degrades during repeated long
processes. Fresh-process state-paired timings are the promotion authority.

Decision: reject and fully revert the incremental ordinary-refresh challenger;
retain App v13. The next speed candidate should target first-barrier worker
startup/context setup or a more structural reduction in branch evaluation,
with fresh-process timing as a first-class gate.
