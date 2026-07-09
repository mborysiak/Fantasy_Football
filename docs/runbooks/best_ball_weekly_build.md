# Best-Ball Weekly Build Runbook

Last updated: 2026-07-07

## Entry Point

```powershell
.venv_ff_312\Scripts\python.exe Scripts\Modeling\s4_Best_Ball_Weekly.py
```

Use the active repo environment if `.venv_ff_312` is not available.

The build uses `YEAR`, `LEAGUE`, and `PRED_VERSION` from `Scripts/config.py`.
It rewrites only the active league/year/dataset slice for app-facing current
tables and preserves other league slices already present in `Simulation.sqlite3`.
Historical template rows are keyed by `league`.

## Inputs

- `Data/Databases/Simulation.sqlite3`
- `Data/Databases/Model_Inputs.sqlite3`
- `Data/Databases/Validations.sqlite3`
- Daily fantasy data repo databases, especially `FastR_Beta`
- Settings from `Scripts/config.py`

## Outputs

The script writes these tables to the `Simulation` database:

- `Best_Ball_Weekly_Templates`
- `Best_Ball_Weekly_Template_Pools`
- `Best_Ball_Weekly_Pool_Summary`
- `Best_Ball_Weekly_Player_Map`
- `Best_Ball_Weekly_Template_Audit`
- `Best_Ball_Weekly_Player_Pool_Audit`
- `Best_Ball_Weekly_Bucket_Audit`
- `Best_Ball_ADP_Audit`

It also copies `Simulation.sqlite3` to configured app repos when those paths
exist.

## Checks

After a build, review:

- row counts by `version`/`league` across the best-ball tables
- template counts by position
- historical projection source mix
- zero/low-active template exposure
- player pool levels and minimum pool sizes
- current bucket universe sizes
- `Best_Ball_ADP_Audit` rows with `needs_review = 1`
- a few player-level template match queries for high-value rookies, unclear ADP
  joins, and unusual role profiles

## App Coordination

If any app-consumed table or column changes:

1. Update `docs/data_contracts/best_ball_weekly_tables.md`.
2. Update the Snake app contract/runbook.
3. Verify the app can load `Simulation.sqlite3`.
4. Run a small app-side smoke check when practical.
