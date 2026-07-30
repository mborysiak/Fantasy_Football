# Findings

## Accepted corrections

### Weekly league scoring

The weekly loader now resolves league at call time and passes it explicitly to
every position scoring path. It attaches a transient `scoring_league` marker;
template construction derives `league` and the template-ID offset from that
marker and rejects mixed or mismatched league requests. Research runners also
pass explicit league and V2 database paths.

This closes the failure mode in which a beta template slice was labeled beta
after receiving DK weekly scoring. Beta realized `active_ppg`, residuals, and
weekly paths now include beta reception, touchdown, sack, and configured
cumulative yardage-bonus rules. Yardage bonuses were always intended to flow
through weekly upside; they are scored on each realized week rather than
approximated from season totals.

### FFToday stored-2018 QB vintage

The governed rule
`fftoday_qb_stored_2018_2019_vintage_quarantine_v1` excludes exactly 50
`FFToday_Projections` QB rows stored under 2018. They match the official 2019
projection archive, and the source already contains 50 native 2019 QB rows, so
an effective-season override would double count the vintage. The raw source
database is intentionally unchanged.

Quarantine runs before identity resolution, source-season overrides, candidate
construction, projection values, and template-key backfills. The active
feature-source audit records 6,308 eligible/resolved FFToday rows plus 50
excluded rows with rule ID, reason, and reference. The rebuilt league artifacts
contain zero stored-2018 FFToday QB aliases or provider values and retain all 50
native 2019 provider values.

Milestone 2 now stores a deterministic policy-hash receipt in
`source_manifest`; `--reuse-foundation` rejects a missing or stale receipt.
Milestone 3 publishes a separate `source_quarantine` receipt with the excluded
row count.

## Modeling effect

The rebuilt DK and beta foundations remain in exact identity/spine parity:
6,655 identities, 55,914 aliases, 52,476 source observations, 13,909
player-season feature rows, 31,798 provider projection values, and 31,834
market values per league.

The locked primary architecture remains the fixed equal-third
Lasso/RF/LightGBM blend with no calibration overlay:

| League | Primary RMSE | Expert recalibration RMSE | Season wins |
|---|---:|---:|---:|
| DK | 3.10783 | 3.19507 | 9/9 |
| beta | 2.88446 | 2.95997 | 9/9 |

The following-season conditional blend also remains decisively ahead of expert
carry:

| League | Primary RMSE | Expert carry RMSE | Origin wins |
|---|---:|---:|---:|
| DK | 3.91366 | 5.23512 | 8/8 |
| beta | 3.55377 | 4.36534 | 8/8 |

Beta's locked validation population falls from 3,659 to 3,622 rows. The
37-row reduction is concentrated in the 2018 QB slice: where the quarantined
future-vintage FFToday row was the only apparent sack donor, the beta provider
estimate is now correctly incomplete. No zero-sack fill or DK center is used.
The no-history route remains secondary and no calibration policy is promoted;
stale exact secondary-route metrics are superseded rather than carried forward.

## Weekly scoring gates

Both staged league slices rebuild to 5,298 templates with exact paired
identity/season coverage. Of those pairs, 5,120 `active_ppg` values differ;
mean absolute PPG delta is 1.350815 and the maximum is 6.8. Normalized weekly
paths differ for 5,147 pairs, with mean L1 delta 1.140687 and maximum
15.063653. This rejects the prior all-identical DK-routed beta state.

Two player sentinels reconcile the intended mechanisms:

- Amon-Ra St. Brown's 2024 beta season scores 256.7 points/17.113333 PPG
  versus 302.2/20.146667 in DK, confirming that reception and scoring-rule
  differences reach the realized upside path.
- Josh Allen's 2024 beta season scores 378.16 points/25.210667 PPG versus
  367.16/24.477333 in DK. His 14 sacks cost exactly 14 beta points:
  the no-sack beta total is 392.16.

Beta has 2,657/2,696 historical V2 diagnostic centers (98.5534%). The 39
unavailable rows are all 2018 QBs with a joined locked-handoff unavailable
marker and active FFToday-quarantine proof. Their V2 diagnostic remains null,
the active center remains the validated legacy OOS value, and the exact reason
is persisted. Every other missing V2 center still fails closed. Historical
identity and current-map joins remain 100%.

## Corrected strict rolling template replay

The projection-weight study was rerun on 1,620 strict rolling targets per
league using the corrected weekly scoring and league-specific V2 paths.
Production remains the DK PPG-CRPS leader at 2.343290. Raising absolute PPG
weight from 1.50 to 2.25 worsens DK PPG CRPS by 0.003261, with player-cluster
95% interval `[0.000661, 0.006671]`.

In beta, the same `ppg_w225` change improves the PPG point estimate by only
0.000351 versus the 1.913075 production CRPS, with interval
`[-0.001546, 0.000977]`, while worsening played-games CRPS by 0.002868 with
interval `[0.001141, 0.004513]`. The tiny uncertain PPG movement does not clear
the multi-outcome promotion gate. Template match weights remain unchanged.

The following-season fields were also replayed on the same 1,620 targets per
league. DK `next_residual_w100` improves PPG CRPS by 0.002006 with interval
`[-0.003773, -0.000375]` versus the 2.343290 production baseline, but worsens
contribution CRPS by 0.031825 with interval `[0.009475, 0.056981]` versus the
26.755146 baseline. Beta improves PPG CRPS by an uncertain 0.001110 with
interval `[-0.003060, 0.000976]` versus 1.913075, while contribution changes by
an uncertain +0.000771 with interval `[-0.019639, 0.021995]` versus 20.614248.
The fields do not show a joint template-matching benefit and remain outside the
matcher.

## Verification status

- V2 unit suite: 118 passed.
- Weekly modeling unit suite: 22 passed.
- Total automated tests: 140 passed; three unrelated `pkg_resources`
  deprecation warnings remain.
- Touched modules compile and `git diff --check` passes.
- DK and beta staged foundations publish the expected 50-row quarantine
  receipt and no contaminated aliases or values.
- Locked current and following-season primary gates pass in both leagues.
- Prior-only calibration remains rejected in both leagues.
- Both staged weekly slices rebuild to 5,298 historical templates.
- Cross-league weekly scoring, beta sack, yardage-bonus, identity, and governed
  diagnostic-fallback gates pass.
- The corrected projection-weight replay retains production weights.
- The corrected next-year template replay retains those fields outside the
  matcher.
- The second staged production handoff has exactly zero current/next point
  deltas.
- Staged DK V2, beta V2, and Simulation databases were copied byte-for-byte to
  live. All five source/app databases pass SQLite integrity with zero foreign
  key errors.
- All 11 auction generated tables match source row-for-row and hash-for-hash;
  the auction database's non-generated-content hash is unchanged.
- The Snake database SHA-256 equals the source Simulation database exactly.
- Three reviewed hybrid-role center-position mismatches are explicitly audited;
  every other position mismatch and every live/staged V2-path violation fails
  closed.

The governed fallback clears the sparse beta historical diagnostic-center
issue without manufacturing coverage. A quarantined 2019 vintage, a DK-scored
center, or an invented zero-sack estimate remains prohibited.

## Decision

Accept the corrected lineage and production cutover. Keep the fixed current and
next-year model architectures, no point-calibration overlay, current template
weights, no next-year matcher fields, the validated legacy OOS donor center,
and one joint centered donor residual/path. Future source-quarantine or scoring
changes must repeat these gates.
