# Additive Salary Normalization

This study isolates the final market-reconciliation rule while holding the v3
raw rolling salary predictions fixed.

It compares:

- the stored proportional-above-floor normalization; and
- a common additive shift with a `$1` floor, chosen so the highest
  `available_slots` non-keeper predictions exactly equal `available_budget`.

This is a normalization ablation, not a full v4 model ablation. The new
keeper-market input features require a fresh salary-pipeline run before their
incremental prediction value can be evaluated.

Run from the repository root:

```powershell
python research/studies/2026-07-16_additive_salary_normalization/run_audit.py
```

Outputs are written under `results/`.
