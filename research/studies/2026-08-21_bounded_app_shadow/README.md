# Bounded App Shadow Validation

## Question

Does the production Sequential rollout safely reinvest meaningful projected
salary slack without introducing future-price leakage, a rollout-time solver,
roster illegality, completion loss, or instability in the main decision
anchors?

## Design

The study calls the App's production `simulate_history_only_branch` directly.
For every comparison, baseline (`budget_reinvestment=False`) and bounded
(`True`) share the evidence seed, managed-value block, hidden nomination tape,
price shocks, validation bank, and compiled completion plan. The paired plans
are compiled with the previously parity-checked stable required-roster solver;
the auction rollout itself never invokes a solver.

Four evidence variations (`0`, `1`, `2`, and `14`) were run in fresh processes
at three fixed checkpoints selected before results were inspected:

- early: Chase Brown `$34` plus Bhayshul Tuten `$11`, evaluate Jahmyr Gibbs at
  `$110`;
- middle: Purdy/Gibbs/Brown/Tuten plus KC Concepcion and Makai Lemon, evaluate
  Brock Bowers at `$51`;
- late cap squeeze: the middle roster plus Bijan at `$105`, evaluate Kyle Pitts
  at `$11` with only `$24` left for six slots.

Each action/state pools 16 evidence blocks and 192 auction paths.

## Results

All promotion gates pass:

- every completed bounded roster passes salary-cap, position-minimum/maximum,
  and dynamic top-N checks;
- completion is not materially worse: early Buy/Pass improve to 100%/100%,
  middle remains 100%/100%, late Buy is unchanged at 97.92%, and late Pass
  declines by one path from 100% to 99.48%;
- average unused salary across the three states falls by `$11.57` on Buy paths
  and `$24.81` on Pass paths;
- accumulated anchor calls remain stable: Gibbs stays TARGET (`+8.23` mean,
  `+5.38` LCB80) and late-cap Pitts stays PASS (`-9.63`, `-11.15`).

The middle Bowers-at-`$51` sensitivity changes from baseline TARGET (`+9.99`,
`+7.63`) to bounded PASS (`-0.26`, `-2.36`). This is an intended
opportunity-cost effect: mean Pass unused salary falls from `$45.04` to
`$20.70`, versus `$27.67` to `$15.76` for Buy. It is not a change to Bowers'
point forecast or weekly upside distribution.

The first shadow exposed a minimum-first fallback bug: after upgrades, a lost
target could rebuild a positionally feasible cheap portfolio that omitted the
required current top-N anchor and terminate. The promoted repair reserves the
cheapest feasible current top-N anchor before filling position minimums. A
separate projected-roster check prevents a bargain or urgent final purchase
from replacing the sole top-N anchor on the last roster slot.

## Decision

Promote deterministic bounded same-position reinvestment as the App default:

- trigger only when projected final slack exceeds `max($5, $1 * open slots)`;
- search at most three swaps among 24 incoming and eight outgoing targets;
- require strictly positive managed-value gain and preserve cap, positions,
  and dynamic top-N legality;
- use the repaired minimum-first fallback; never add a direct reward for
  spending and never solve a new optimization problem during the rollout.

The App cache version advances from 11 to 12 so existing session results cannot
silently survive the policy change.

A live post-promotion smoke exposed equal-price top-N sorting across mixed
NumPy float/string player-label scalar types. All new deterministic player
tie-break keys now compare `str(player)` while preserving the original player
object for roster maps and outputs. The regression reproduces the original
NumPy ufunc failure and the final App suite passes 89 tests.

## Reproduction

Run each variation in a fresh App environment process, then aggregate:

```powershell
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe run_shadow.py --variations 0
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe run_shadow.py --variations 1
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe run_shadow.py --variations 2
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe run_shadow.py --variations 14
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe run_shadow.py --variations 0 1 2 14 --aggregate-only
```

Canonical outputs are under `results/`, especially
`accumulated_decision_summary.csv`, `paired_path_deltas.csv`,
`path_summary.csv`, and `metadata.json`.
