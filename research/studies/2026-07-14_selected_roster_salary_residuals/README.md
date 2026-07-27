# Selected-Roster Salary Residual Diagnostic

This study tests whether the auction optimizer concentrates positive salary
residuals even when the current salary model is reasonably calibrated across the
full candidate pool.

It reuses the exact candidate surfaces and 4,000 optimized rosters from the
salary chance-constraint frontier. Residuals are defined as:

`recorded actual salary - normalized point-predicted salary`

The study distinguishes:

- all auctionable player-origins with recorded prices;
- unique players ever or never selected;
- selection-frequency buckets and roster-slot-weighted selections;
- high-projection players by selection frequency;
- selected players by position, predicted-price tier, and a within-position
  projection-rank-minus-price-rank value proxy;
- the separate contribution of point-prediction error and scenario-market
  discounting to the prior `$29` roster gap.

Recorded prices come from the same `Actual_Salaries` source used by the replay.
Missing actual prices are excluded from player residual summaries and retained
at the replay's intentional `$1` fallback only for exact roster-gap
reconstruction.

Run from the repository root:

```powershell
python research/studies/2026-07-14_selected_roster_salary_residuals/run_diagnostic.py
```

