# Raw Expert-Rank Challenger - DK

## Method

- Raw median uses every observed provider overall rank and ignores missing provider rows.
- The percentile is calculated across all ranked QB/RB/WR/TE players within a season after the raw median is formed.
- Publication coverage is observed rank sources divided by sources publishing any rank for that season-position; it is not a depth-adjusted rank.
- The normalized comparator is rebuilt in-process from the identical scoring-specific provider rows.
- DK replaces half-PPR ETR with full-PPR ETR; beta retains half-PPR ETR.
- Primary attribution uses full-column random forests on both sides. The locked 50% forest remains a separate sensitivity.

## Controlled results

| Variant | RMSE | Delta | Recent | Early era | Expanded era | Wins | Season 95% | Player 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `normalized_scoring_specific` | 3.10478 | -0.00311 | -0.00527 | -0.00208 | -0.00684 | 6/9 | [-0.00528, -0.00081] | [-0.00489, -0.00135] |
| `raw_available_median` | 3.10819 | +0.00029 | -0.00313 | +0.00167 | -0.00466 | 6/9 | [-0.00239, +0.00341] | [-0.00246, +0.00307] |
| `raw_log` | 3.10460 | -0.00329 | -0.00633 | -0.00228 | -0.00696 | 7/9 | [-0.00637, -0.00035] | [-0.00552, -0.00108] |
| `raw_percentile` | 3.10754 | -0.00035 | -0.00472 | +0.00132 | -0.00636 | 6/9 | [-0.00346, +0.00302] | [-0.00233, +0.00159] |
| `raw_percentile_coverage` | 3.10651 | -0.00138 | -0.00660 | +0.00014 | -0.00686 | 6/9 | [-0.00514, +0.00236] | [-0.00363, +0.00083] |

## Production-surface sensitivity

| Variant | RMSE | Delta | Recent | Wins | Season 95% | Player 95% |
|---|---:|---:|---:|---:|---:|---:|
| `normalized_scoring_specific` | 3.10491 | -0.00264 | -0.00498 | 7/9 | [-0.00495, -0.00026] | [-0.00483, -0.00050] |
| `raw_available_median` | 3.10774 | +0.00018 | -0.00296 | 4/9 | [-0.00309, +0.00396] | [-0.00271, +0.00305] |
| `raw_log` | 3.10412 | -0.00344 | -0.00592 | 6/9 | [-0.00758, +0.00003] | [-0.00611, -0.00083] |
| `raw_percentile` | 3.10652 | -0.00103 | -0.00461 | 6/9 | [-0.00391, +0.00199] | [-0.00350, +0.00140] |
| `raw_percentile_coverage` | 3.10526 | -0.00230 | -0.00762 | 6/9 | [-0.00688, +0.00225] | [-0.00501, +0.00033] |

## Governance

- Feature run: `milestone_3_20260730T140041Z_e06ca8aa`
- Staged database: `C:\Users\borys\OneDrive\Documents\GitHub\Fantasy_Football\research\studies\2026-07-30_v2_market_rank_challengers\artifacts\local\Projection_V2_single_nffc.sqlite3`
- Locked-incumbent reproduction max delta: `3.55e-15`
- Existing raw-median reproduction max delta: `0`
- 2026 scoring-specific rank providers: 6
- Prespecified advancement candidate: `raw_percentile_coverage`
- Single-league gates all pass: `False`
- Failed gates: `['season_interval_upper_nonpositive', 'player_interval_upper_nonpositive', 'production_surface_season_interval_upper_nonpositive', 'production_surface_player_interval_upper_nonpositive', 'beats_normalized_rank_by_0_001', 'early_era_nonworse']`
- Passing these gates only advances the candidate to a nested retune; it does not promote the feature to production.
