# V2 Production Handoff and Donor-Center Guardrail

Date: 2026-07-29

## Production policy

- Current 2026 point centers use the locked league-specific V2 models for 268
  DK and 180 beta app players.
- Current residual quantiles are zero. One matched historical donor supplies
  the centered active-PPG residual and the same weekly/played trajectory.
- Next-year PPG and residual quantiles are conditional on an appearance. A
  separate Bernoulli draw from `pred_appear_ny` sets future market value to
  zero when the player does not appear.
- Historical donor residuals retain the validated legacy OOS point centers.
  Strict-OOS V2 donor centers are stored only as diagnostics.

## Rejected historical recentering

Replacing the historical donor centers with V2 strict-OOS centers worsened the
`production_no_next` replay on the same 1,620 player-season targets per league:

| League | PPG CRPS delta, V2 minus legacy | Player-cluster 95% interval | Contribution CRPS delta | Played-games CRPS delta |
|---|---:|---:|---:|---:|
| DK | +0.00570 | [+0.00213, +0.00928] | +0.01385 | +0.00220 |
| beta | +0.00515 | [+0.00139, +0.00903] | +0.04144 | +0.00327 |

Lower CRPS is better. The PPG result is adverse in both leagues and both
intervals are entirely above zero. Recent 2021-2025 rows also worsen
(+0.00765 DK, +0.00583 beta). The candidate is rejected.

After restoring legacy donor centers, the final-policy replay matches the
previously validated baseline exactly, row-for-row, for PPG, contribution, and
played-games CRPS in both leagues.

## Verification

- `Final_Predictions_Resid`: 448 unique league/player keys, complete V2
  provenance/appearance fields, zero current residual quantiles.
- `Best_Ball_Weekly_Player_Map`: 448 unique keys and exact handoff agreement.
- `Best_Ball_Weekly_Templates`: 10,596 non-null canonical keys, complete
  2017-2025 V2 diagnostic-center coverage, zero promoted V2 recenter rows, and
  exact active-residual reconstruction.
- Source, auction, and Snake database copies have identical invariants.
- Automated checks: 60 V2 tests, 38 auction tests, and 3 Snake V2 handoff tests.
- Runtime smokes: managed auction weekly/keeper paths and a two-trial Snake ILP
  run completed with the V2 joint-template method.
