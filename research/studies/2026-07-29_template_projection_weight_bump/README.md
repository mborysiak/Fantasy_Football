# Weekly-Template Projection Weight Bump

This study tests whether the production weekly-template matcher underweights
absolute projected PPG and position-relevant rushing, receiving, and passing
projection components.

The replay holds the target population, strictly prior donor rule, 80-donor
pool, adaptive distance kernel, 12-season recency half-life, and 5% donor cap
fixed. It compares:

- the production matcher;
- a 50% and 100% increase in absolute-PPG weight;
- a 50% increase in component-rank weights by itself;
- scoring-aligned raw component-PPG magnitude by itself;
- a moderate PPG increase plus a 50% increase in component-rank weights;
- that moderate version plus scoring-aligned raw component-PPG magnitude; and
- an aggressive version of all three changes.

The raw component magnitude allocates the scoring-specific current/historical
PPG center using each component's share of the preseason total, then applies
the same `/10` scale as the existing absolute-PPG feature.

Run DK:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-29_template_projection_weight_bump\run_validation.py
```

Run beta:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-29_template_projection_weight_bump\run_validation.py `
  --league beta `
  --results-dir research\studies\2026-07-29_template_projection_weight_bump\results_beta
```

The study is read-only with respect to production databases and matcher
configuration.

## Result

Every global absolute-PPG or raw-component bump worsens pooled PPG,
contribution, and played-games CRPS in both leagues. Component ranks alone are
approximately neutral, and a QB-only component-rank increase is directionally
favorable but has player-cluster intervals crossing zero. Production weights
remain unchanged. See `results/findings.md`.
