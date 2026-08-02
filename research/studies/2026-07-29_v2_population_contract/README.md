# V2 Production Population Contract

## Purpose

This study records the live expansion from the former 268 DK/180 beta app
population to a governed, key-first V2 population. It also records the aligned
weekly-context rebuild, beta salary v6 surface, annual selection reserve, and
cross-app cutover.

## Contract

- DK: 351 players (55 QB, 100 RB, 143 WR, 53 TE), selected from the
  326-player core plus top-280 DK union after five governed no-center
  exclusions.
- Beta: 328 players (50 QB, 95 RB, 133 WR, 50 TE), selected from the core,
  top-180 ETR overall-rank union, and all keepers.
- Current and next-season production centers come from the locked V2 shadows
  by canonical `(league, player_key)`. Legacy current/next values are
  audit-only.
- Current weekly context is key-first; every player receives 80 donors. Only
  explicit governed ADP fallbacks are permitted.
- The beta salary surface is
  `current_locked_spec_v6_v2_population_11f` on the same 328 canonical keys.

## Evidence

`validate_cutover.py` and the JSON/text receipts under `staging/` and
`results/` verify population, context, salary, reserve, idempotence, database
integrity, and app parity. `promote_cutover.py` performed the governed
promotion after the staging gates passed.

Pre-promotion database copies are retained in
`results/pre_promotion_20260730/`. The user's live
`Data/Databases/Model_Inputs.sqlite3` was preserved.

## Durable output

See `results/findings.md` for the accepted methodology, exact counts, and
promotion decision.
