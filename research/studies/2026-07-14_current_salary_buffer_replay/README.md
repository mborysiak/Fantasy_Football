# Current-Method Salary Buffer Replay

This study compares the two provisional nominal salary guardrails using the new
rolling-origin salary tables:

- no nominal constraint (paired reference)
- normalized point salary `<= $298 + $5`
- normalized point salary `<= $298 + $10`

Every trial retains the live five-draw salary average, draw-level remaining-
market normalization, sampled `$298` cap, Top-N rule, projected waiver baseline,
bench weight `0.25`, and current position maxima including `TE <= 2`.

Historical point forecasts and managed scoring come from the frozen rolling
replay. Salary centers and residual quantiles come from
`Validations.Salary_Backtest_Predictions` under
`beta/current_locked_spec_v1/model_spec_asof_year=2026`. Model-missing frozen
players use their pre-auction ESPN salary when available (otherwise the existing
minimum fill), and receive prior-only residual quantiles interpolated by position
and normalized point price. The player audit records every fallback.

Run a smoke check:

```powershell
python research/studies/2026-07-14_current_salary_buffer_replay/run_replay.py `
  --years 2025 --trials 2 --contexts 8 --context-draws 3 `
  --projection-draws 100 --salary-draws 100 `
  --output-dir research/studies/2026-07-14_current_salary_buffer_replay/artifacts/local/smoke
```

Run the full replay:

```powershell
python research/studies/2026-07-14_current_salary_buffer_replay/run_replay.py
```

Development years are 2022-2024 and 2025 remains a temporal check. The salary
data roll by origin, but the model specification was selected as of 2026, so
this is a current-method retrospective rather than a fresh method holdout.

The full run completed 3,000 optimal cells. `+$5` versus `+$10` traded 5.57
development season points for 6.53 percentage points of historical-price
feasibility and $3.55 less mean overage; the same directions held in 2025 and
both trial halves. `+$10` is the more point-efficient default if one guardrail
must be chosen, while `+$5` remains a conservative affordability mode. See
`results/decision_readout.md`.
