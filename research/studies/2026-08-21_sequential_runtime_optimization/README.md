# Sequential Runtime Optimization

This study verifies two execution-only changes to the Auction App's Sequential
Target Board:

1. pre-ranked Top-N scans plus incremental legality checks for same-position
   bounded reinvestment swaps;
2. a market-price board that defers the leading ten low/high Approx Max Bid
   anchors until the user requests them.

`verify_runtime.py` checks the optimized rollout against the retained full-check
path and checks an immediate-curve board against a deferred board followed by
the curve-only replay. It also verifies that **Add Evidence** after prior curve
profiling returns to the identical market-only accumulation path. All
comparisons use the same hidden evidence seeds.

The follow-up incremental ordinary-refresh challenger is documented in
`results/summary.md`. It preserved exact paths but failed fresh-process runtime
gates and was reverted; App v13 remains active.
