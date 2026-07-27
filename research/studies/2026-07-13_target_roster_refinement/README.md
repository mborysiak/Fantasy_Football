# Target Roster Refinement

> Superseded for live organic Target construction by the converged refinement
> policy validated in `../2026-07-19_target_roster_convergence/`. This study
> remains the evidence for the expected-profile context and single-swap helper.
> Its serial verification explicitly pins `max_swaps=1`; use the 2026-07-19
> study for the live convergence benchmark.

This study verifies the bounded Target Board correction for additive managed-
roster values.

The original production path used the fast linear ILP for its initial roster. It
then averages the full cached construction bank into an expected 16-week
profile and evaluates one best feasible player swap before any holdout outcomes
are drawn. Averaging the bank avoids selecting a swap on five unusually good or
bad realized seasons while retaining the starter/FLEX interaction correction.
The refinement preserves fixed players and entered salaries, the salary cap,
position bounds, roster size, and the top-N constraint.

One pass is intentional. Iterating to local convergence materially increases
Target latency, while the first swap removes the largest observed interaction
error.

Run from the modeling repository:

```powershell
python research/studies/2026-07-13_target_roster_refinement/verify_target_roster_refinement.py
```

The script runs synthetic constraint cases, rounded/tied-decision brute-force
comparisons, and a seeded 50-trial refinement-on/off benchmark. It writes the
durable benchmark summary to `results/benchmark.json`.

To benchmark the production process-pool path at 500 trials:

```powershell
python research/studies/2026-07-13_target_roster_refinement/benchmark_parallel.py
```

That writes `results/parallel_benchmark.json`.
