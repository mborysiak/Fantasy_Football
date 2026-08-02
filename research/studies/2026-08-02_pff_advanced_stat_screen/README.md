# Prior-season PFF advanced-stat screen

This study screens receiving and rushing metrics from
`Season_Stats_New.sqlite3` for leakage-safe, incremental preseason value.
It is an association/prediction study, not a causal-effect estimate.

## Frozen design

- The target is next-season conditional fantasy PPG for the locked 2017-2025
  validation rows.
- Every PFF value comes from season `t-1` when predicting season `t`.
- Players without a usable prior PFF sample receive the applicable
  position-season neutral mean and reliability zero. Missing values are never
  treated as observed zero performance.
- Rates are empirically shrunk using fixed, outcome-independent prior sample
  sizes (100 routes, 25 targets, 20 receptions, or 50 rush attempts).
- The main baseline is the locked production PPG prediction. That prediction
  already includes the expert consensus projection, ADP, experience, prior
  production, projected opportunity, room context, and trajectory features.
- Each rate challenger is compared with a PFF opportunity-only control so the
  rate does not receive credit merely for revealing prior routes, targets,
  receptions, or carries.
- Rolling corrections and upside classifiers are trained only on earlier
  locked out-of-sample seasons. Development is 2018-2022 and the untouched
  temporal slice is 2023-2025.
- DK is primary; beta scoring is a replication check.

## Outputs

Run:

```powershell
.\.venv\Scripts\python.exe research/studies/2026-08-02_pff_advanced_stat_screen/run_screen.py
```

The script writes coverage, persistence, redundancy, residual-association,
rolling PPG, and q90-upside diagnostics to `results/`, together with a compact
`findings.md`. No production table or model configuration is changed.

