# Auction Championship and Waiver Objective Test

Paired 2025 Beta Auction study of four construction policies on the isolated
historical replay database:

- `baseline`: current waiver estimates and the current additive expected-score
  plan;
- `waiver_proxy`: a best-available waiver-churn proxy with QB at least 15.5,
  RB/WR at least 9.0, and the current TE estimate;
- `championship_tiebreak`: current waivers, but among candidate rosters within
  0.25% of the best full-roster construction EV, prefer the highest
  same-scenario probability of beating 11 reference rosters;
- `combined`: the waiver proxy plus the championship tie-break.

Every arm reuses the same salary state, weekly construction contexts, and
independent validation contexts. Selection uses only preseason information and
weekly donors through 2024. Actual 2025 weekly results are loaded only after
the policies are frozen and provide a retrospective diagnostic, not a selection
input.

The championship value is explicitly a within-study proxy rather than a
calibrated absolute league-win probability. In each weekly scenario, a roster's
empirical percentile against the common feasible-roster reference bank is
raised to the eleventh power. This preserves shared player outcomes and rewards
rosters that can beat an entire 12-team field.

The upside diagnostic retains the governed q90 difference-maker event from the
August 1 study: active PPG at least five points above the held-out projection
and contribution above the position-specific q90 threshold built from the five
strictly prior seasons.

Run the full study:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-08-27_auction_championship_waiver_objective\run_paired_test.py
```

Durable outputs are written to `results/`. Nothing in the production or app
databases is modified.

The decision readout is in [`results/findings.md`](results/findings.md). The
combined arm improves simulated and actual-score diagnostics but does not
reduce dead-zone RB selection, so no production arm is promoted from this
single 2025 origin.
