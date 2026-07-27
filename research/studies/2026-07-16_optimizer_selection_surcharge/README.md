# Optimizer Selection Surcharge

This study tests whether a small second-stage salary correction can address the
remaining optimizer-conditioned price bias without replacing the v5 salary
model or globally reducing the auction budget.

The correction is deliberately simple and rolling-origin:

- target residual: `actual_salary - v5_point_salary`;
- predictors: position, v5 point salary, squared point salary, preseason
  optimizer selection frequency, selection-frequency by salary, and
  selection-frequency by position;
- estimator: heavily regularized ridge regression;
- fit window: only completed origins before the target year;
- action: positive predicted residuals only, capped at $10 per player.

The replay compares:

- `baseline_298`: current five-draw market with the full $298 cap;
- `blanket_285`: the same market and objective draws with a $285 personal cap;
- `targeted_half`: half of the rolling predicted surcharge added to decision
  prices, with the full $298 cap;
- `targeted_full`: the full rolling predicted surcharge added to decision
  prices, with the full $298 cap.

2022 is a seed origin and receives no surcharge because no earlier v5 origin is
available. Development comparison therefore uses 2023-2024, with 2025 retained
as the temporal check.

Run:

```powershell
python research/studies/2026-07-16_optimizer_selection_surcharge/run_replay.py
```

