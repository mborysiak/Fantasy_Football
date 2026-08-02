# V2 NFFC Market and Expert-Rank Findings

## Decision

- Promote one NFFC-family ADP observation per player-season.
- Preserve ETR overall rank as the beta top-180 production-eligibility
  ordering.
- Keep raw expert ranks audit-only.
- Retain normalized expert-rank level as an unpromoted challenger.
- Reject the expert-minus-projection rank gap.

## NFFC contract and replay

`ADP_Averages(league='nffc')` is already the player-level arithmetic mean of
Rotowire Online, Best Ball Overall, Best Ball $25/$50, and Cutline. V2 now
admits only that `adp_average_nffc` observation to
`player_season_market_values`. The four raw `NFFC_ADP` rows remain available
for candidate discovery and identity resolution.

The normalized market/rank table falls from 31,834 to 28,801 rows. The
maximum modeled ADP-source count falls from 12 to eight. The only NFFC rows in
the live long table are 479 2025 and 431 2026 `adp_average_nffc` rows.

The exact locked architecture, feature surface, strict-prior hyperparameter
selection, and calibration policy were replayed:

| League | Previous RMSE | One-vote NFFC RMSE | Delta |
| --- | ---: | ---: | ---: |
| DK | 3.1078268 | 3.1075576 | -0.0002692 |
| beta | 2.8844644 | 2.8841120 | -0.0003524 |

Only the 2025 out-of-fold origin changes because NFFC history begins late.
Every tested prior-only calibration overlay remains worse, so no calibration
policy changes. The following-season replay remains better than expert carry
in all eight origins: 3.9137 versus 5.2351 DK and 3.5538 versus 4.3653 beta.

## Expert-rank challenger

Raw ranks are not directly comparable across providers because their list
depths and ranking objectives differ. For each source, season, and position,
the study orders rank ascending and converts it to a percentile with 1.0 best.
It then takes the cross-source median. All forecast origins fit only earlier
seasons, reuse the incumbent's strictly-prior selected hyperparameters, and
retain fold-local imputation and missing indicators.

The locked random forest uses 50% feature subsampling. Adding a feature changes
which incumbent fields are sampled, so it is not a clean attribution. The
primary comparison therefore uses `max_features=1.0` for both incumbent and
challenger forests while leaving Lasso, LightGBM, training rows, and all other
settings unchanged.

| League | Variant | Controlled incumbent | Variant RMSE | Delta | Recent delta | Wins | Season 95% | Player 95% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DK | normalized rank level | 3.1078932 | 3.1056818 | -0.0022114 | -0.0042541 | 7/9 | [-0.0041782, -0.0003137] | [-0.0038198, -0.0005945] |
| beta | normalized rank level | 2.8854599 | 2.8836036 | -0.0018563 | -0.0024974 | 7/9 | [-0.0037629, -0.0001110] | [-0.0037183, +0.0000300] |
| DK | expert-minus-projection gap | 3.1078932 | 3.1074396 | -0.0004536 | -0.0012525 | 6/9 | [-0.0019064, +0.0011665] | [-0.0027668, +0.0018847] |
| beta | expert-minus-projection gap | 2.8854599 | 2.8853975 | -0.0000624 | -0.0027594 | 4/9 | [-0.0021804, +0.0021419] | [-0.0023919, +0.0022692] |

With the locked 50% forest retained as a production-surface sensitivity, rank
level improves by 0.0017396 DK and 0.0013780 beta, but both clustered
intervals cross zero. The gap worsens by about 0.00041 in both leagues.

The normalized level has a favorable and recent-stable result after the
attribution control, but the absolute gain remains tiny. The controlled forest
is not the selected production architecture, and provider coverage jumps from
mostly FantasyData plus FFA through 2023 to five sources in 2024 and eight in
2025. Retain the feature as a strong challenger, not a production promotion.
ETR has only 2024-2026 half-PPR history and 2025-2026 full-PPR history, so a
dedicated ETR coefficient is not identifiable in the full locked replay.
Instead, ETR contributes one normalized vote to the cross-provider challenger.
For 2026, removing ETR changes the comparable 242-player consensus by a median
absolute 0.03595 position-percentile points. This demonstrates nontrivial
feature-level influence; it is not a leave-ETR-out forecast-performance claim.

## Publication and verification

- The live DK and beta V2 databases pass SQLite integrity and contain only
  `adp_average_nffc` NFFC market rows.
- The live locked runs are
  `v2_locked_final_dk_20260730T140449Z_8a9b4479` and
  `v2_locked_final_beta_20260730T140449Z_ef55415d`.
- The production handoff retains 351 DK and 328 beta players with complete
  canonical-key coverage.
- Both weekly slices retain 5,298 templates and exactly 80 donors per live
  player. Governed context-ADP fallbacks remain 14 DK and 91 beta.
- The Auction app's 17 generated tables match staging while all seven
  app-owned tables remain unchanged. Every Snake table matches staging.
- The live population/weekly/salary/reserve cutover validator passes, including
  328 salary players, 14 keepers, and 314 reserve players.
- All 125 V2 tests pass. Focused source and production-handoff tests pass
  28/28, and the study/config/handoff modules compile.
- Recoverable pre-promotion database backups are stored under
  `results/pre_promotion_20260730/`.
