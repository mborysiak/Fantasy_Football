# Salary Feature Reduction

## Purpose

Reduce the v4 auction-salary feature surface to a compact set that retains:

- the strongest causal salary anchors;
- projection and source-price disagreement signals associated with residuals;
- a small amount of role, upside, and breakout context; and
- explicit position controls without a redundant fourth dummy.

The audit compares the compact surface with the archived pre-v5 feature
construction using strict rolling test origins from 2022 through 2025. The
benchmark averages six fixed model specifications so the result measures the
feature surface rather than another hyperparameter search.

## Run

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-07-16_salary_feature_reduction/run_feature_audit.py
```

## Interpretation

This is a rolling-data development comparison, not a fresh method holdout. The
chosen v5 surface must still be rebuilt through the full salary notebook and
evaluated on player error and optimizer-selected roster affordability before
promotion.
