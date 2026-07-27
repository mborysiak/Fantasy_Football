# Salary Validation Datasets

## Purpose

Build reusable rolling-origin salary predictions for auction replay and an
observed non-keeper residual table for calibration review. This removes the need
to reconstruct salary-model joins for every affordability experiment.

## Build

From `Scripts/Modeling/` in PowerShell:

```powershell
$env:SALARY_VALIDATION_DATASETS_ONLY='1'
..\..\.venv_ff_312\Scripts\python.exe s4_Salaries_Injuries.py
```

Validation-only mode writes the two owned slices in
`Data/Databases/Validations.sqlite3` and does not rewrite live Simulation salary,
keeper, or prediction slices or copy the database to the auction app.

## Method

- Candidate pools come from the preseason `QB/RB/WR/TE_2026_ProjOnly` universe.
- Manually copied ESPN base salaries and actual auction results are left-joined;
  missing base values use zero and retain `base_salary_observed = 0`.
- Models for an origin use only prior-year observed non-keeper targets.
- Raw predictions are retained, then normalized to the known keeper-adjusted
  budget over the known open roster slots without target-year actual totals.
- Residual quantiles at an origin use only earlier origin residuals.
- The model data cutoff rolls honestly, but current 2026 model-family and
  hyperparameter selection is retrospective, so rows are not fresh method
  holdouts.

## Outputs

- `Salary_Backtest_Predictions`: full 2022-2025 player pools.
- `Salary_Validations_Resid`: observed non-keeper rows for 2021-2025.

See `docs/data_contracts/auction_salary_tables.md` for the durable schema and
semantics. Build results are summarized in `results/build_summary.md`.
