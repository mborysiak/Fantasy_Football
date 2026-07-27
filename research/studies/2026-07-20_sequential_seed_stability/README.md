# Sequential Target Seed Stability

This study diagnoses two separate stability questions in the auction app:

1. Does removing one unavailable player preserve the same hidden auction and
   weekly worlds for every remaining player?
2. How much does Bijan Robinson's forced market-price Buy-minus-Pass estimate
   vary across independent evidence-bank seeds in a fixed draft state?

The representative state uses the complete 2026 beta `League_Keepers` table
from the app database plus the three user players visible in the screenshots:
Jahmyr Gibbs (`$111`), Chase Brown (`$34`), and Bhayshul Tuten (`$11`). The paired
state additionally removes AJ Brown at `$53`. If another off-screen non-keeper
sale was present in the live UI, absolute estimates will differ, but the
random-bank and nesting diagnostics remain valid.

`run_seed_stability.py` compares the current pool-shaped random sampling with a
research-only nested sampler. The nested sampler draws nominations, salary
shocks, construction templates, and validation templates on the full canonical
player universe and masks unavailable players afterward. It then evaluates
Bijan directly on the same 48 auction paths by 64 validation seasons used by
confirmation, without requiring him to survive the adaptive confirmation gate.

Outputs are written to `results/`:

- `bijan_seed_results.csv`: seed-level forced evidence;
- `aj_state_deltas.csv`: AJ-off minus AJ-on effects within seed;
- `panel_stability.csv`: all seed-panel combinations for 1/2/4/8 banks;
- `variance_decomposition.csv`: construction, auction-path, and weekly-season
  seed components with other components fixed;
- `summary.md`: decision-oriented findings.

`run_v4_stability.py` replays the implemented production v4 design: four
independent blocks, 32 balanced mean-PPG construction templates per block, 12
realized auction paths per block, and 64 complete validation seasons per block.
It writes:

- `v4_bijan_seed_results.csv`: 16-root AJ-on/AJ-off v4 evidence;
- `v4_aj_state_deltas.csv`: matched live-state deltas;
- `v4_summary.md`: the post-implementation stability check.

Run from the modeling repository root:

```powershell
python research/studies/2026-07-20_sequential_seed_stability/run_seed_stability.py
python research/studies/2026-07-20_sequential_seed_stability/run_v4_stability.py
```

The first script is diagnostic only. The v4 design validated by the second
script is now the production Sequential Policy evidence method.
