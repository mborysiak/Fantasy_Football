# Target Roster Convergence Results

## Decision

Use repeated exact-scored, constraint-safe swaps for organic Target roster
construction until no swap improves, capped at 12. Preserve the full
cached-bank expected weekly profile as the construction signal and draw
holdouts only after refinement.

Keep Target pilot, preliminary, and confirmation Buy/Pass construction and
Current Nomination unrefined. Run the premium-free annual `selection_only` seed
with convergence so its selection-rate feature matches the organic policy.

## Final Paired Verification

The final check used the refreshed 2026/beta app database, 64 empty-roster
trials, seed `20260719`, and the published converged selection premium. Old
behavior was emulated with `max_swaps=1`.

- Holdout season EV: `1700.1 -> 1742.1` (`+42.0`).
- Holdout p10: `1511.1 -> 1577.2` (`+66.1`).
- Holdout p90: `1875.3 -> 1930.2` (`+54.9`).
- Accepted swaps: mean `3.78`, maximum `9`, with `0/64` reaching the 12-swap
  cap.
- Serial runtime: `5.08s -> 11.66s` (`2.30x`) in the final paired run.
- The leading organic players remained Jaxon Smith-Njigba, Chase Brown, Cam
  Skattebo, Javonte Williams, and Christian McCaffrey, with higher converged
  selection rates.

The focused 16-trial instrumentation benchmark measured `1.21x` runtime for an
empty roster and `1.20x` for the supplied mid-draft roster, demonstrating the
expected machine/cache variance. Only organic Target construction pays this
cost by default; forced Buy/Pass stages remain unchanged.

## Selection-Premium Refresh

The fresh premium-free seed completed `1000/1000` rosters, trained the unchanged
ridge on 518 observed 2022-2025 rows, and published 166 active premiums to both
source and app databases under
`app_target_selection_only_converged_v2`. The expected roster reserve is
`$6.76`; RJ Harvey has the maximum applied reserve at `$3.79`.

The first eight-worker attempt failed from process memory pressure before any
database write. The validated one-worker run completed successfully using the
same eight logical seed blocks.
