# Findings

## Decision

Promote the expanded V2 population, rebuilt league-specific weekly context,
beta salary v6 surface, and fresh annual selection reserve. The cutover is live
in source, Auction, and Snake databases.

## Population and identity

- DK has 351 players: 55 QB, 100 RB, 143 WR, and 53 TE. It is the union of the
  326-player core and top-280 DK market population after excluding Kareem Hunt,
  Austin Ekeler, Tyreek Hill, Nick Chubb, and Joe Mixon because they have market
  evidence but no current V2 center.
- Beta has 328 players: 50 QB, 95 RB, 133 WR, and 50 TE. It is the union of the
  core population, top-180 ETR overall-rank population, and all governed
  keepers.
- `V2_Production_Eligibility_Audit` retains all 1,490 reviewed rows.
- Tetairoa McMillan and Amon-Ra St. Brown retain their governed canonical
  identities.
- The locked league-specific V2 shadows are authoritative for current and
  following-season point/appearance fields. Legacy current/next fields are
  audit-only. Historical template residuals separately retain the validated
  legacy OOS center.

## Weekly context and scoring

- Required context joins use canonical `player_key`; every live player has
  exactly 80 donors.
- Fourteen DK and 91 beta players use explicit governed context-ADP fallbacks.
  No generic default or review route remains.
- `LA`/`LAR` and `ARZ`/`ARI` are canonicalized only while constructing room
  features. Outward labels are preserved. Free agents receive zero room
  features.
- Yardage bonuses and beta sacks flow through league-scored weekly
  `active_ppg`, donor paths, and optimizer selection. They are not forced into
  the salary point estimate. Two-point conversions and special-teams
  touchdowns remain omitted by decision.

## Salary v6 and reserve

- The live beta salary method is
  `current_locked_spec_v6_v2_population_11f` on exactly 328 canonical keys.
  There are 326 direct `ProjOnly` rows plus governed V2 fallbacks for Stefon
  Diggs and Deebo Samuel.
- All 14 keepers have canonical keys. The highest 142 non-keeper point salaries
  total exactly `$3,071`.
- `ensemble_pred_resid_90` is not a v6 feature. Strict rolling MAE is `$4.2975`
  for the 11-field surface versus `$4.2991` with all 12 fields. Historical p90
  represented projection residuals, while the current centered weekly-donor
  tail is a different simulation object; current QB p90 standard deviation was
  9.35 times the historical value. Centered donor p90 remains diagnostic only.
- The fresh annual reserve completed 1,000/1,000 Target rosters over 314
  non-keepers. Expected 13-player reserve is `$8.5598`.
- Historical calibration remains on v5 and the current seed is v6. The governed
  transfer is labeled
  `historical_v5_selection_surface_to_current_v6_v1`. On common current
  players, v5/v6 point salaries have correlation `0.99957` and MAE `$0.274`.

## Publication verification

- A second handoff left all five governed table hashes unchanged:
  `Final_Predictions_Resid` and `V2_Production_Projection_Handoff` each have
  679 rows; `V2_Production_Projection_Audit` has 679;
  `V2_Production_Eligibility_Audit` has 1,490; and the legacy backup retains
  its original 448 audit rows.
- All 17 generated Auction tables match staging. All seven app-owned tables are
  unchanged. Every Snake table matches staging.
- Six databases pass SQLite integrity with zero foreign-key errors.
- The user's `Model_Inputs.sqlite3` is preserved. Pre-promotion copies are in
  `results/pre_promotion_20260730/`.
- Current full test suites pass: 153 main-repo tests, 43 Auction-app tests,
  and 10 Snake-app tests.
