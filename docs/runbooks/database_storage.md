# Database storage and Git tracking

Last updated: 2026-09-05

## Retention policy

- Keep `Data/Databases/Season_Stats_New.sqlite3` tracked. It contains historical
  expert projections, stats, ranks, and other source evidence, not just model
  outputs. Keep the older `Season_Stats.sqlite3` and
  `Season_Stats_Playoffs.sqlite3` tracked until their historical contents are
  fully accounted for elsewhere.
- Keep provider downloads, salaries, keepers, and other manual inputs under
  `Data/OtherData/` tracked. Their current file coverage is not proof that all
  historical database tables can be reconstructed.
- Exclude compiled `Model_Inputs*.sqlite3`, `Validations.sqlite3`,
  `Projection_V2*.sqlite3`, and `V2_Parameter_Cache.sqlite3` from Git, including
  SQLite sidecars. Keep their local files for normal builds and refreshes.
  Regeneration depends on retained inputs, the relevant code/configuration,
  and upstream weekly data in `Daily_Fantasy_Data`; an ordinary clone alone
  does not establish a complete recovery path or reproduce every old model run.
- Exclude generated SQLite copies in `Data/Production_Refresh_Stages/`, while
  leaving manifests, logs, and other release evidence eligible for tracking.
  Research staging SQLite files are also ignored at every depth, including
  annual replay paths such as `staging/2022/databases/`. Existing rollback
  directories remain ignored.
- Keep source `Simulation.sqlite3` tracked. Its four display-only breakout
  tables were retired on 2026-09-05; after compaction it is about 80.64 MiB,
  below GitHub's 100 MiB ordinary-Git limit. Historical salary/injury inputs,
  all production weekly scoring tables, and other retained content remain.
  App repositories may keep their existing LFS transport.
- Ignored local copies are not off-computer backups. Establish a dated external
  backup with a verified restore for critical source data and selected releases.
  Untracking does not erase previously committed blobs; oversized unpushed
  history needs a separate, reviewed cleanup or LFS migration.

## Installed retirement result

| Database | Before MiB | After MiB | Retained tables |
| --- | ---: | ---: | ---: |
| Source Simulation | 138.54 | 80.64 | 26 |
| Auction app | 138.54 | 80.64 | 26 |
| Snake app | 138.52 | 80.62 | 26 |

All retained schemas and table contents match each database's own snapshot;
existing differences between source and app histories were preserved. Integrity
and foreign-key checks pass, freelist counts are zero, and installed hashes
match the compacted candidates. All 71 focused refresh/builder tests and 120
Auction tests pass. AppTest is clean for Auction Beta/NV and Snake DK/NFFC.
Snapshots and full receipts are under
`Data/Production_Refresh_Backups/20260906_breakout_retirement/`.

## Simulation storage audit before retirement: 2026-09-05

The source database occupied **145,272,832 bytes (138.54 MiB)** with a 4,096-byte
page size and zero freelist pages. A `VACUUM INTO` copy occupied 145,268,736
bytes, saving only **4,096 bytes**. Vacuum is not a material size reduction for
this snapshot.

| Table | Allocated MiB, including indexes | Rows | Current use |
| --- | ---: | ---: | --- |
| Breakout_Paired_Template_Pools | 50.07 | 92,960 | Auction breakout explorer |
| Best_Ball_Weekly_Template_Pools | 35.88 | 109,600 | Auction and Snake weekly scoring; template explorer |
| Best_Ball_Weekly_Templates | 20.50 | 17,403 | Weekly profiles consumed by the donor pools |
| Breakout_Paired_Templates | 7.26 | 11,867 | Auction breakout explorer |
| Best_Ball_Weekly_Template_Audit | 4.67 | 17,403 | Governed template validation evidence |
| Best_Ball_Weekly_Player_Map | 4.52 | 1,370 | Current player mapping and projection context |
| V2_Production_Eligibility_Audit | 3.38 | 3,132 | Governed eligibility evidence |
| Model_Predictions | 2.43 | 16,132 | Historical legacy model outputs |

The four paired-breakout tables occupied about **57.89 MiB**. The user then
approved retiring this display-only feature. The source and both app databases
were backed up before removing exactly those four tables and compacting staged
candidates. Every remaining table's schema and logical content was verified
against its own pre-retirement snapshot. The weekly donor/scoring tables stay.

Production refresh schema 8 no longer generates or requires breakout tables;
it removes inherited copies before compaction and excludes them from preserved
app-owned data. The Auction display and its runtime imports were removed.
Older refresh manifests must be rebuilt rather than resumed.

The full source and app snapshots are retained locally under
`Data/Production_Refresh_Backups/20260906_breakout_retirement/`. These are local
rollback copies, not an external backup. The original data-generation logic
and historical table contract remain available for research. To rebuild donors
without writing to a production database:

```powershell
.venv_ff_312/Scripts/python.exe -m Scripts.Modeling.build_paired_breakout_templates --simulation-db Data/Databases/Simulation.sqlite3 --output-db research/studies/2026-08-27_paired_breakout_templates/artifacts/local/breakout.sqlite3
```

The builder requires a separate output path, rejects the source and live
production database directories, and no longer offers `--sync-app`.

The legacy `Model_Predictions`, `Final_Predictions`, and
`V2_Projection_Legacy_Backup` tables together occupy only about 2.71 MiB; deleting
them would not bring Simulation below 100 MiB and would sacrifice retained
historical evidence. Source salary, actual-salary, injury, and keeper tables
are small and should be retained.

## Verification and local receipt

The Python SQLite build lacks `dbstat`. Table allocation was measured by
backing up the source to an in-memory database and, for each table independently,
measuring newly freed pages after dropping it inside a rolled-back scratch
transaction. Index pages are included. The source connection was read-only.

The vacuumed copy passed `PRAGMA integrity_check`. Its complete schema and all
30 tables' ordered row values matched the source. The source SHA-256 remained
`036a8df778dc2e6e162b9130ecd0f4b1bdec474323b42e9bc24aff66e6c2b226`.

The compacted copy and full per-table JSON receipt are retained locally under:

`Data/Production_Refresh_Backups/simulation_storage_audit_20260906T012145Z/`

SQLite documents `VACUUM INTO` as a compacted snapshot that leaves the original
database unchanged: <https://www.sqlite.org/lang_vacuum.html>.
