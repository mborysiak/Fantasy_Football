# Sequential Shared-Opportunity Correction

## Question

Does a bounded, full-construction-bank conditional swap correction improve the
Sequential optimizer's roster construction enough to justify its runtime cost?

The correction is applied after the additive completion ILP. It evaluates each
incoming player conditional on the other selected players, so two players
competing for the same weekly starting opportunity are no longer both credited
with their standalone value.

## Design

`run_validation.py` compares two paired policy compilers on the 2026 NV
Maye/Achane state:

- additive ILP only (`baseline`);
- one full-bank conditional swap (`one_swap`).

The comparison is run with both predicted prices (next-draft shadow) and the
published 2026 actual prices (hindsight replay). Every arm shares construction
contexts, hidden auction tapes, and independent historical-template validation
seasons. The study records static plan lineups, organic sequential lineups,
managed-season mean/p10, completion/legality, and separate plan-compilation,
rollout, and scoring timings.

The replay mirrors the live App's active-keeper market state. Maye and Achane
are fixed to the tested roster; every other active 2026 NV keeper is removed
from the player pool and hidden auction tapes. League-wide remaining money and
slots subtract all 16 keepers, using configured keeper salaries for predicted
mode and the App's actual-salary overlay for hindsight mode.

The implementation is retained as an inactive App shadow. The validated
candidate uses one swap in both discovery and confirmation/current nomination.
The previously rejected four-swap arm is excluded from the corrected canonical
rerun because it adds no production decision and repeatedly stalls in the
native runtime.

## Reproduction

```powershell
0..3 | ForEach-Object {
  python `
    research\studies\2026-08-24_sequential_shared_opportunity\run_validation.py `
    --sources predicted --block-index $_
}
0..3 | ForEach-Object {
  python `
    research\studies\2026-08-24_sequential_shared_opportunity\run_validation.py `
    --sources actual --block-index $_
}
python `
  research\studies\2026-08-24_sequential_shared_opportunity\run_validation.py `
  --sources predicted actual --combine-blocks
python `
  research\studies\2026-08-24_sequential_shared_opportunity\run_validation.py `
  --aggregate-only
```

Each evidence block runs in a fresh process so retained GLPK/native state cannot
contaminate the comparison or accumulate across rollout solves.

Canonical outputs are written under `results/`.

For a fresh-process end-to-end production-shaped board timing (64 screened
candidates, 18 confirmations) at the minimum production compute budget:

```powershell
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe `
  research\studies\2026-08-24_sequential_shared_opportunity\run_board_timing.py `
  --joint-swaps 0
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe `
  research\studies\2026-08-24_sequential_shared_opportunity\run_board_timing.py `
  --joint-swaps 1
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe `
  research\studies\2026-08-24_sequential_shared_opportunity\run_board_timing.py `
  --source actual --joint-swaps 0
& ..\Fantasy_Football_App\streamlitvenv\Scripts\python.exe `
  research\studies\2026-08-24_sequential_shared_opportunity\run_board_timing.py `
  --source actual --joint-swaps 1
```
