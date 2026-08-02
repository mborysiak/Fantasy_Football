# Production Refresh Runbook

Last updated: 2026-07-31

## Scope and Manual Boundary

`Scripts/V2/refresh_production.py` rebuilds every model and app artifact after
raw projection/market ingest. It does not replace the manual ingest workflow.
The requested `--year` must be an approved cycle in
`Scripts/V2/production_cycle.py`; it is not accepted as an unvalidated display
label.

Before launching the notebook kernel, set the same current season that will be
passed to the governed refresh:

```powershell
$env:FF_CURRENT_SEASON = '2026'
.\.venv_ff_312\Scripts\python.exe -c "from Scripts.config import YEAR; assert YEAR == 2026; print(YEAR)"
```

Then complete `Scripts/Data_Generation/1_Update_Projections.py` in its normal
interactive workflow, including any required downloaded exports, and verify
that `Data/Databases/Season_Stats_New.sqlite3` contains the intended current
DK, NFFC, ETR, and projection inputs. For 2026, `NFFC_ADP` must contain exactly
`nffc_rotowire_online`, `nffc_best_ball_overall`,
`nffc_best_ball_25s50s`, and `nffc_cutline`, with minimum depths of 400, 400,
400, and 250 rows respectively. Missing, renamed, unexpected, or shallow feeds
fail during `snapshot`, before model fitting.

An already-running notebook kernel does not inherit a newly set environment
variable; restart it and confirm `YEAR` inside the notebook before any write.
Close notebooks, apps, and other database writers before continuing. The
refresh rejects active SQLite sidecars and, after its initial snapshot, refuses
to resume or promote if the live source database changes.

## Standard Staged Refresh

Run the complete pipeline without changing live model or app databases:

```powershell
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --year 2026
```

The default creates a unique directory below
`%TEMP%\fantasy-football-production-refresh`, prints its path, runs through both
app smoke tests, and stops without promotion. For a durable review directory,
pass a new empty location:

```powershell
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --year 2026 --stage-dir <empty-stage-dir>
```

Omitting `--year` currently selects 2026, the latest approved cycle. Passing it
explicitly is preferred for annual operating records.

Review `refresh_manifest.json`, `logs/`, `results/`, and the staged app
artifacts. Promotion is a separate, explicit command:

```powershell
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --resume <stage-dir> --promote
```

`--promote` requires all staged steps, including both app smoke tests, to be
complete. It cannot be combined with an earlier `--through` target.

## Resume and Diagnostic Stops

The manifest records each step as pending, running, completed, or failed.
`--resume` skips completed steps and retries the first incomplete or failed
step:

```powershell
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --resume <stage-dir>
```

The manifest stores the complete approved-cycle receipt and its SHA-256. Resume
and promotion fail if that annual contract changes after the stage starts.

Use `--through <step>` to stop inclusively at a named step:

```powershell
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --stage-dir <empty-stage-dir> --through handoff
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --resume <stage-dir> --through weekly_nffc
```

Omitting `--through` on the next resume continues through `app_smoke`.
`--dry-run` prints the planned steps without creating files. A resume retains
the manifest's model options; only `--app-timeout` is treated as a retry-time
override. Start a new refresh if any snapshotted live database, external
weekly/salary input, or fingerprinted pipeline/app source file changed.

Every production refresh runs a fresh 1,000-trial Auction selection seed.
Carrying an earlier seed across changed projections, salaries, keepers, weekly
surfaces, or selection policy is methodologically unsupported.

The runner automatically uses a system Python with Streamlit for app smoke
tests when the modeling virtualenv does not include Streamlit. Pass
`--app-python <python.exe>` only when that automatic selection is unavailable.

## Windows Native Runtime Containment

Refresh subprocesses cap native thread pools at one. The annual LightGBM
runners load scikit-learn first and fail before fitting unless Windows exposes
exactly one `vcomp` OpenMP runtime. Only Windows access violation
`0xC0000005` is retried, once; every automatic attempt is retained in the step
receipt. Ordinary errors fail immediately.

A supported resume replaces the current manifest state for the resumed step,
so inspect its append-only step log for failures that preceded the resume. The
guard removes one known multiple-OpenMP failure mode, but it does not certify
the host as stable. Run a production refresh while unrelated heavy Python
workloads are idle. See the
[LightGBM OpenMP guidance](https://lightgbm.readthedocs.io/en/latest/FAQ.html#lightgbm-crashes-randomly-or-operating-system-hangs-during-or-after-running-lightgbm).

## Ordered Steps

The exact pipeline order is:

```text
snapshot
model_inputs
v2_dk
v2_nffc
v2_beta
locked_dk
locked_nffc
locked_beta
next_dk
next_nffc
next_beta
keepers
handoff
weekly_dk
weekly_nffc
weekly_beta
template_audit_dk
template_audit_nffc
template_audit_beta
salary
selection_premium
validate
prepare_apps
app_smoke
```

- `snapshot` copies and hashes the eight source/model databases plus both live
  app databases. It also records the read-only weekly-history database, current
  Auction salary export, and historical selection bootstrap files that remain
  outside the staged database directory.
- `model_inputs` runs the canonical-input portion of
  `4_Data_Compile.py`, producing current and next-year model inputs.
- `v2_*` rebuilds the DK, NFFC, and beta V2 feature/model databases.
- `locked_*` publishes the accepted current-year locked shadows; `next_*`
  publishes next-year residual/appearance shadows.
- `keepers` publishes canonical keeper identities before beta eligibility is
  resolved.
- `handoff` publishes current/next projections and DK/NFFC/ETR market surfaces
  into staged Simulation, then reruns and hashes all eight governed tables to
  prove idempotence.
- `weekly_*` builds DK, NFFC, and beta weekly templates/pools with app sync
  disabled; `template_audit_*` validates all three handoffs.
- `salary` writes staged keeper/salary outputs and salary validation tables.
- `selection_premium` writes a fresh beta reserve surface plus its seed and
  calibrator.
- `validate`, `prepare_apps`, and `app_smoke` gate the candidate release.

## Staged and Promoted Artifacts

Each stage contains:

```text
refresh_manifest.json
databases/
  Season_Stats_New.sqlite3
  Model_Inputs.sqlite3
  Model_Inputs_next.sqlite3
  Projection_V2.sqlite3
  Projection_V2_nffc.sqlite3
  Projection_V2_beta.sqlite3
  Simulation.sqlite3
  Validations.sqlite3
app_bases/
  Auction_Simulation.sqlite3
  Snake_Simulation.sqlite3
app_artifacts/
  Auction_Simulation.sqlite3
  Snake_Simulation.sqlite3
logs/
results/
```

Promotion replaces these seven main-repo databases:

- `Model_Inputs.sqlite3`
- `Model_Inputs_next.sqlite3`
- `Projection_V2.sqlite3`
- `Projection_V2_nffc.sqlite3`
- `Projection_V2_beta.sqlite3`
- `Simulation.sqlite3`
- `Validations.sqlite3`

It also replaces the Auction and Snake app databases. The staged
`Season_Stats_New.sqlite3` is evidence for the run, not a promotion source; the
live file remains owned by the manual ingest boundary.

The first approved NFFC refresh may create
`Data/Databases/Projection_V2_nffc.sqlite3`; its absence at snapshot is recorded
rather than treated as prior production evidence. Promotion installs it only as
part of the same validated rollback set as the other eight destinations.

The downstream Validation outputs are `Salary_Validations_Resid`,
`Salary_Backtest_Predictions`, `Salary_Selection_Seeds`, and
`Salary_Selection_Calibrator`. The first two are written during `salary`; the
fresh seed and calibrator are written during `selection_premium`.

## Auction and Snake Publication

The Auction candidate starts from the snapshotted Auction database and replaces
exactly these 20 source-owned tables, including schemas and explicit indexes:

- Handoff/market:
  `Avg_ADPs`, `Avg_ADPs_Publication_Audit`,
  `Avg_ADPs_Publication_Receipt`, `Final_Predictions_Resid`,
  `V2_Production_Projection_Handoff`, `V2_Production_Projection_Audit`,
  `V2_Production_Eligibility_Audit`, `V2_Projection_Legacy_Backup`
- Weekly:
  `Best_Ball_Weekly_Templates`, `Best_Ball_Weekly_Template_Pools`,
  `Best_Ball_Weekly_Pool_Summary`, `Best_Ball_Weekly_Player_Map`,
  `Best_Ball_Weekly_Template_Audit`,
  `Best_Ball_Weekly_Player_Pool_Audit`,
  `Best_Ball_Weekly_Bucket_Audit`, `Best_Ball_ADP_Audit`
- Auction:
  `Salaries`, `Salaries_Pred`, `League_Keepers`,
  `Salary_Selection_Premium`

Every Auction table outside that registry is preserved from the app snapshot.
The current app-owned set is `Actual_Salaries`, `Final_Predictions`,
`Injuries`, `Injuries_Source`, `Model_Predictions`, and
`Model_Predictions_Resid`. If the live Auction database changes after snapshot,
promotion fails instead of overwriting the newer app state.

Snake receives a complete copy of the staged `Simulation.sqlite3`; it is not a
table-by-table merge.

The 2026 NFFC Snake mode is an offense-only governed surface. It uses NFFC
offensive scoring, canonical NFFC ADP, a 17-week horizon, and QB/RB/WR/TE
templates drawn from the modern 2021-forward donor era. The canonical market
table still retains `TK` and `TDSP` draft units for audit, but the NFFC
projection population and Snake draft pool filter to offensive positions.

The official $150 Best Ball Championship rules specify 12 teams, 30 rounds,
Weeks 1-17, Third Round Reversal, and `TK`/`TDSP` roster slots. The current
adapter supplies independently scored offensive projections, 17-week
templates, canonical ADP, and 3RR, but it has no `TK`/`TDSP` model or roster
slots and does not enforce the official 30-round roster composition. The
agreed rare two-point and return/special-touchdown components also remain
omitted. It is an experimental offense-only decision aid, not a complete
contest simulator. See the
[official NFFC rules](https://nfc.shgn.com/rules/2680).

Weekly template IDs are regenerated when the donor bank changes. Each league
rebuild therefore removes older pool/map rows for that league and dataset, and
the Snake selector lists only prediction year/league pairs backed by the
current player map. This deliberately prevents an older draft surface from
silently joining to a newer season's unversioned templates.

NFFC scoring-sensitive historical and current matcher context comes only from
the NFFC-scored V2 preseason consensus; the DK-scored `Model_Inputs` projection
context is retained for audit and cannot fill NFFC fields. Historical NFFC
donors use `nffc_scored_expert_consensus`. A 540-target 2023-2025 replay
rejected promotion of the locked OOF donor center: locked-minus-expert PPG CRPS
was `+0.002901`, it lost all three seasons, and the player-cluster 95% interval
was `[-0.004914, +0.010748]`. The staged surface has 1,509 2021-2025
templates, 17 populated weeks, and a 385-player map. No NFFC release has been
promoted.

## Release Gates and Promotion Safety

Before preparing app candidates, validation requires:

- SQLite integrity, foreign-key validity, and no active sidecars
- an unchanged approved-cycle receipt and exact registered model versions
- cycle-specific minimum depths for DK, NFFC, and ETR source markets
- the exact four registered NFFC source-feed labels and their individual
  annual depth floors before model fitting
- unique QB/RB/WR/TE current model-input tables above conservative
  cycle-specific population floors
- exact DK/NFFC/beta identity, alias, spine-key, and feature-key agreement
- exact DK/NFFC/beta agreement on every hashed nflverse player/weekly payload
  used by the active feature foundations
- registered current and next-year shadow tables with zero unmatched templates
- current-model improvement over expert recalibration and next-model
  improvement over carry under the registered acceptance gates
- unique keyed production populations above conservative position floors and
  exact projection/weekly-map parity
- NFFC eligibility from the top-360 canonical offensive ADP union, excluding
  `TK` and `TDSP` from model/app player rows
- exact registered historical center policy and scoring-context source for
  every league; NFFC must use `nffc_scored_expert_consensus` plus the
  NFFC-scored V2 context, while DK/beta retain their existing contract
- exactly 80 weekly donors per production player
- populated league-specific weekly horizons: 16 weeks for DK/beta and 17 for
  NFFC in the 2026 cycle; NFFC donors must begin in 2021
- zero weekly ADP review, default-ADP, or high-impact unresolved flags
- complete, unique keyed salary predictions
- a keyed selection-premium surface with the configured trial count and at
  least one successful seed trial
- an idempotent production handoff across all eight governed tables

`prepare_apps` validates the resulting Auction and Snake candidate databases.
`app_smoke` launches each Streamlit app against its explicit candidate database
and requires zero rendered errors or exceptions. The Snake smoke must render
both the DK and NFFC selectors from the staged candidate. This is a
startup/render smoke, not a clicked optimizer recommendation test; run the
repository test suites separately when application or optimizer code changed.

Promotion rechecks every live database against its snapshot hash, revalidates
the staged release, verifies that candidate hashes are unchanged since
validation/AppTest, and requires the pipeline/app source-code fingerprint and
external weekly/salary input hashes to match the start of the run. It creates
durable
`Data/Production_Refresh_Backups/<run-id>/*.pre_refresh.sqlite3` files, and
installs all nine destinations as one rollback set. Each installed file must
match the staged SHA-256. SQLite cannot provide a true transaction across
multiple files, so keep both apps and all database writers closed until the
command reports success.

## Approved Cycle and Annual Rollover

The only approved current-season cycle is 2026. It binds current 2026 and
following-season 2027 table names, exact DK/NFFC/beta model versions,
source/model/production floors, league weekly horizons, template history
minimums, and the accepted locked-current, next-year, and template-audit
runners. Requesting current season 2027 fails before staging; the existence of
a 2027 next-season forecast inside the 2026 cycle does not approve a 2027
current-season refresh.

For annual rollover:

1. set `FF_CURRENT_SEASON` before launching/restarting the notebook kernel,
   confirm `Scripts.config.YEAR`, finish the new raw-source ingest, and verify
   source years;
2. create the season-specific beta salary and keeper files;
3. complete and review the new locked-current and following-season validation
   runners and record their exact table/model versions;
4. review annual market, model-input, production-population, exclusion, weekly
   horizon, and donor-history contracts;
5. add one approved entry to `Scripts/V2/production_cycle.py`; and
6. run focused tests plus a complete non-promoting stage before promotion.

Do not copy the 2026 registry entry or its population/exclusion evidence
forward without that review. The orchestrator passes `FF_CURRENT_SEASON` to
downstream scripts so registered generic work follows the selected cycle, while
the dated validation evidence remains explicitly annual.

## Reproducibility Caveat

The V2 DK, NFFC, and beta builds download the current nflverse player release and
historical weekly releases during the run. Their URLs and SHA-256 values are
recorded in the V2 source manifests, but the orchestrator does not freeze those
payloads before execution. The release is therefore auditable but not fully
hermetic: a changed upstream release can change a later rebuild. The release
gate does require DK, NFFC, and beta to have consumed the exact same hashed
nflverse player and weekly payloads, so a change between their sequential
downloads fails closed. Exact reproducibility across future rebuilds still
requires pinned local inputs. The external `FastR_Beta` data used by weekly
generation remains outside the staged copy, but its starting hash is recorded
and promotion is rejected if it changes. The same immutability check covers the
season-specific Auction salary and keeper files plus historical selection
bootstrap files.
