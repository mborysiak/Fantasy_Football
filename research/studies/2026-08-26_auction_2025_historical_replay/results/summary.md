# Result

The isolated 2025 Beta Auction replay is app-compatible and passes its governed
build, database, and simulation smokes.

## Published surface

- 305 current-method rolling-origin projections and salaries, each trained or
  residual-calibrated only through 2024.
- Four complete saved-preseason fallback rows: Austin Ekeler, James Conner,
  Najee Harris, and Roschon Johnson. The rule uses the saved projection/salary
  intersection before consulting draft results.
- 309 total projection, predicted-salary, and weekly-map player keys.
- One `-$0.2990` additive salary shift after augmentation; the top 141
  nonkeepers total the exact `$3,169` available budget.
- 156 deterministic drafted-offense actual-price rows from 179 raw auction
  purchases. The other 23 are K/DEF rows outside the offensive app.
- 15 keepers spending `$407`.
- 238 canonically keyed saved ETR ranks. Of the 309 projection players, 236 use
  that direct ETR context and 73 deeper players use the V2/model-context ADP
  fallback.
- 4,993 replay-only weekly donors in namespace `beta_2025_replay`, with a
  maximum donor season of 2024, and 80 selected donors per current player.
- Zero 2025 `Salary_Selection_Premium` rows and a disabled next-year keeper
  signal.

## Verification

- SQLite `PRAGMA integrity_check`: `ok`.
- Projection/salary/map key parity: `309 / 309 / 309`.
- Actual salary row/key parity: `156 / 156`.
- ETR row/key parity: `238 / 238`.
- Weekly app smoke loads 16 weeks and profiles for all 309 predicted-salary
  players and all 156 actual-salary players; sampled weekly score matrices are
  finite and correctly shaped.
- Focused model tests: 69 passed.

## Boundaries

The model specification is recorded as of 2026 even though its data are
rolling-origin through 2024. The 73 players outside the saved ETR depth are
explicit fallbacks, not fabricated ranks. This artifact is intentionally
isolated and must not be promoted over the live 2026 `Simulation.sqlite3`.
