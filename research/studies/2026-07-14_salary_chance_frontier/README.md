# Salary Chance-Constraint Frontier

This study replaces the single sampled-price cap and the retrospective `$5` / `$10`
ranking with an explicit pre-auction affordability policy. Candidate rosters must fit
the `$298` cap in at least 60%, 70%, 80%, or 90% of 20 construction salary-market
scenarios.

Each market scenario averages five current-method residual salary draws and then
reconciles the full auctionable market to the keeper-adjusted league budget. The
reported modeled affordability is measured on 200 separately sampled scenarios per
trial, not on the scenarios used to construct the roster.

The primary frontier is:

- independently evaluated managed-season forecast EV;
- held-out modeled cap probability;
- historical final-price cap feasibility and overage.

Historical points from unaffordable rosters are retained only in the raw audit file.
Feasible-only historical points are descriptive and cannot identify a policy effect,
because feasibility is determined by the future realized price outcome.

The app's one-swap refinement is disabled in this study because the current refiner
cannot enforce a multi-scenario chance constraint. All thresholds therefore use the
same unrefined linear managed-value optimizer; this is a policy-method comparison,
not a bit-for-bit replay of the live construction path.

Run the full replay from the repository root:

```powershell
python research/studies/2026-07-14_salary_chance_frontier/run_replay.py
```

Use `--years`, `--trials`, and the scenario-count arguments for focused smoke tests.
