# Target Runtime Profile

This study profiles the managed auction Target Board after the Current
Nomination NumPy and score-cache optimization.

It separates:

- managed context-bank generation
- candidate-rebased marginal-value calculation and cache behavior
- outer, forced-Buy, and forced-Pass GLPK solves
- holdout weekly-context generation and exact roster scoring
- salary normalization, salary-row construction, and DataFrame copying
- serial 50-trial block work from spawned-process startup and merge overhead

Run from the modeling repository:

```powershell
python research/studies/2026-07-11_target_runtime_profile/profile_target.py
```

Durable outputs are written to `results/`.

To compare worker caps at a fixed 800 trials:

```powershell
python research/studies/2026-07-11_target_runtime_profile/benchmark_parallel_workers.py
```

Worker spawn, import, template-load, and shutdown overhead can be isolated with:

```powershell
python research/studies/2026-07-11_target_runtime_profile/benchmark_worker_startup.py
```

Key outputs:

- `results/summary.md`
- `results/component_timings.csv`
- `results/cache_metrics.json`
- `results/parallel_scaling.csv`
- `results/parallel_worker_sweep_800.csv`
- `results/parallel_worker_sweep_1000.csv`
- `results/worker_startup.csv`
- `results/target_pre_optimization.csv`
- `results/target_post_optimization.csv`
- `results/component_timings_pre_optimization.csv`
- `results/component_timings_post_optimization.csv`
- `results/serial_post_optimization_repeats.csv`
- `results/component_timings_post_batch.csv`
- `results/serial_post_batch_repeats.csv`
- `results/parallel_worker_sweep_500_post_batch.csv`
- `results/target_serial_cprofile_post_batch.txt`

After changing the Target implementation, run the captured equivalence checks:

```powershell
python research/studies/2026-07-11_target_runtime_profile/verify_target_optimization.py
```
