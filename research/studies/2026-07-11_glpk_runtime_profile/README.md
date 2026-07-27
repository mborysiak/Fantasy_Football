# GLPK Runtime Profile

This study decomposes auction-app ILP runtime into:

- GLPK integer solve time
- CVXOPT matrix conversion
- dynamic salary-row construction
- pandas copying and salary normalization
- managed context/value generation
- managed lineup scoring

It profiles both Current Nomination and a serial Target block, then replays
captured representative problems with dense, sparse, and redundant-row-reduced
constraint matrices. It also benchmarks the LP relaxation and SciPy/HiGHS as
diagnostics; those comparisons do not change application behavior.

Run from the modeling repository:

```powershell
python research/studies/2026-07-11_glpk_runtime_profile/profile_glpk.py `
  --nomination-iters 100 --target-iters 25 --repeat-solves 100
```

Durable outputs are written to `results/`.

The implemented optimization benchmark is captured in:

- `results/nomination_pre_optimization.json`
- `results/nomination_post_optimization.json`
- the corresponding `*_curve.csv` files used for exact equivalence checks
- `results/nomination_post_optimization_counts.json`
- `results/summary.md`
