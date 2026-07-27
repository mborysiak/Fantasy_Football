# Results

- `Best_Ball_Weekly_Templates` now contains complete 0/1 played masks for all
  5,298 `beta` and 5,298 `dk` templates; each row's mask sum equals
  `played_games`, active evidence is always a subset of played evidence, and
  RB/WR/TE counts remain equal.
- The `beta` source contains 2,586 exact-zero scores, 653 negative scores, and
  317 short-QB appearances removed by the existing best-ball performance filter.
  All are now treated as played rather than triggering automatic replacement.
- Of the 317 short-QB appearances, 295 had non-zero scores totaling 713.98
  points. Separate `managed_week_*` multipliers preserve those outcomes for the
  auction app while `week_*` leaves the best-ball distribution unchanged.
- Scalar, multi-context, learned-decision, marginal-value, loader-fallback, and
  paired template-sampling fixtures passed.
- Existing legacy fixtures passed for 500 base lineups, 200 batched marginal
  cases, and 500 salary workspaces.
- In a paired 100-trial Target diagnostic, corrected season EV was 1,655.88
  versus 1,678.27 under the legacy score-threshold rule, a reduction of 22.39
  points. This is the expected removal of the hindsight replacement benefit;
  the top Target player remained Chase Brown.
- Runtime was strongly execution-order/system-state sensitive. Four warmed,
  alternating 50-trial pairs had a median overhead of 0.32% and did not show a
  stable material penalty; a two-worker production run completed 100 corrected
  trials in 5.92 seconds. The mask/profile correction does not add optimizer
  solves.
- A 20-trial Saquon Barkley nomination smoke completed both Buy and Pass in all
  20 trials. Streamlit completed a 100-trial/two-worker Target run with no
  exceptions, two dataframes, and eleven metrics.
- Separate from played-week handling, review found 19 historical TE weeks where
  two different Zach Millers are combined by the current cleaned-name join.
  Stable historical player identity/team resolution remains the next template
  integrity issue.
