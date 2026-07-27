# Best-Ball Weekly Build Runbook

Last updated: 2026-07-23

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

It synchronizes the eight generated best-ball tables into the auction app while
preserving app-owned keeper/salary scenarios, and copies the complete source
`Simulation.sqlite3` to Snake. Export is skipped when any retained template
slice has null played/managed-week fields.

## Checks

After a build, review:

- row counts by `version`/`league` across the best-ball tables
- template counts by position
- historical projection source mix
- zero/low-active template exposure
- `played_week_1` through `played_week_16` contain only 0/1, sum to
  `played_games`, preserve source-observed zero/negative outcomes, and retain
  QB appearances removed from performance profiles by the greater-than-15-play
  filter
- `managed_week_1` through `managed_week_16` are complete and retain the
  unfiltered score profile for those short QB appearances
- player pool levels and minimum pool sizes
- declared template exclusions are present in the audit table and have zero
  pool uses; ordinary zero-active seasons remain eligible
- effective sample size, maximum donor probability (5% cap), and local-weight
  fraction by position
- every `template_season_gap` is positive,
  `template_recency_multiplier = 0.5 ** (template_season_gap / 12)`, and pool
  summaries report the 12-season half-life plus weighted donor-age diagnostics
- `projection_x_exp` is absent from the active position weight maps while its
  persisted diagnostic column remains available to existing comp views
- current and historical absolute PPG, market/projection disagreement, and
  workload-room match fields are complete
- current and historical uncapped `year_exp` ranges, `year_exp_source` mix,
  named veteran values, and pool experience ranges for players above ten years
- current bucket universe sizes
- `Best_Ball_ADP_Audit` rows with `needs_review = 1`
- a few player-level template match queries for high-value rookies, unclear ADP
  joins, and unusual role profiles

When introducing the played/managed-week columns into a multi-league database,
rebuild every retained league slice before copying the database to apps. A
slice-preserving rebuild of only one league leaves the new columns null on
older slices; app consumers should retain a legacy fallback until all slices
have been rebuilt.

## App Coordination

If any app-consumed table or column changes:

1. Update `docs/data_contracts/best_ball_weekly_tables.md`.
2. Update the Snake app contract/runbook.
3. Verify the app can load `Simulation.sqlite3`.
4. Run a small app-side smoke check when practical.

For the auction app smoke check, confirm a donor draw uses the current
`pred_fp_per_game` plus that donor's centered `active_ppg_resid`, followed by the
same donor's `managed_week_*` path. Salary and next-year keeper uncertainty stay
on their existing residual paths.

Do not replace the auction app database wholesale during this build: its
`League_Keepers` and salary tables may intentionally represent a local keeper
comparison scenario. Only the generated best-ball tables are source-owned for
this export.

## Temporary NFFC Snake Setup Preview

While NFFC model runs are incomplete, create an isolated app database with:

```powershell
python Scripts\Modeling\create_snake_nffc_preview.py
```

The command copies the stable Snake app database to
`Fantasy_Football_Snake/app/Simulation_nffc_preview.sqlite3`, clones the 2026
DK `Final_Predictions_Resid` and three runtime weekly-template tables under
NFFC-safe league keys/template IDs, and retains the real NFFC `Avg_ADPs` rows.
It does not modify the source `Simulation.sqlite3`.

This database is for app wiring and draft-flow testing only. Its projections
and weekly scores still use DK scoring/calibration and must be replaced by a
normal NFFC s3/s4 build before recommendations are evaluated.
