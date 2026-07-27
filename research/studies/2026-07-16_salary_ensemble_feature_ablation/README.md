# Salary Ensemble-Feature Ablation

## Purpose

Compare the preserved rolling-origin salary methods after the salary pipeline
was rebuilt:

- `current_locked_spec_v1`
- `current_locked_spec_v2_ensemble_features`

The study first evaluates paired player-year predictions on the common observed
validation rows. It then projects both methods onto the exact frozen candidate
and projection universe used by the prior salary chance-constraint replay.

The fixed-roster repricing is intentionally not a new optimizer replay. It asks
whether v2 prices the exact rosters previously selected under v1 more accurately,
which isolates the salary-surface change from optimizer reselection.

## Run

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_ensemble_feature_ablation/run_ablation.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_ensemble_feature_ablation/audit_results.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_ensemble_feature_ablation/run_frontier_v2.py

.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_ensemble_feature_ablation/compare_frontiers.py
```

The study uses 2021-2024 as the full salary-validation development period,
2022-2024 as the replay-aligned development period, and 2025 as the temporal
check. The model specification remains retrospectively locked as of 2026, so
these are rolling data holdouts rather than a fresh method-selection holdout.

`run_frontier_v2.py` reuses the prior chance-frontier implementation with the
same seeds and defaults, changing only the salary method and output directory.
The prior v1 artifacts remain immutable comparison inputs.
