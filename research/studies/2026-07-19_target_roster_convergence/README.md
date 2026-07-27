# Target Roster Convergence

This study validates replacing the Target Board organic construction stage's
single exact-scored roster swap with repeated feasible swaps to local
convergence. Production retains a hard cap of 12 accepted-swap attempts.

The initial roster still comes from the fast additive managed-value ILP. Each
refinement step uses the full cached construction-bank mean, preserves fixed
players and entered salaries, and rechecks the salary cap, position bounds,
roster size, and Top-N constraint. Holdout seasons are sampled only after the
roster is finalized.

The scope is deliberately limited:

- organic Target construction uses convergence;
- Target pilot, preliminary, and confirmation Buy/Pass stages keep refinement
  disabled;
- Current Nomination keeps refinement disabled; and
- the annual premium-free `selection_only` Target seed uses convergence because
  its selection-rate feature must match the live organic construction policy.

Run the paired quality check from the modeling repository:

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-19_target_roster_convergence\verify_refine_convergence.py
```

Run the focused empty-roster and mid-draft runtime benchmark with:

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-19_target_roster_convergence\bench_refine_convergence.py
```

Both scripts use the live auction-app `Simulation.sqlite3`, fixed seed
`20260719`, and write machine-readable outputs under `results/`.
