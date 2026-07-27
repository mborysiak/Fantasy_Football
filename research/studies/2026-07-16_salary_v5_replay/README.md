# Salary v5 Replay

## Purpose

Evaluate `current_locked_spec_v5_compact_salary_features` after the full salary
pipeline was rebuilt. The study repeats the prior salary gates:

1. rolling player-level raw and normalized salary accuracy;
2. raw market-total and additive-normalization diagnostics;
3. the identical-seed 4,000-cell chance-frontier replay versus preserved v1;
4. optimizer-selected residual concentration and roster-gap decomposition.

The player comparison includes v3 as the immediately preceding feature-rich
method. The optimizer comparison uses v1 because its original frozen replay is
the preserved paired baseline.

## Run

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_v5_replay/run_point_accuracy.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_v5_replay/run_frontier_v5.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_v5_replay/compare_frontiers.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_v5_replay/run_selected_residuals.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_v5_replay/audit_results.py
```

These are rolling-data/current-method development results, not a fresh
method-selection holdout.
