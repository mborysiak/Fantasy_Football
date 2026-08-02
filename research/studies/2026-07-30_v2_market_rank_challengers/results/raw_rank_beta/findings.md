# Raw Expert-Rank Challenger - BETA

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
| `normalized_scoring_specific` | 2.88318 | -0.00228 | -0.00544 | -0.00127 | -0.00577 | 6/9 | [-0.00431, -0.00025] | [-0.00429, -0.00025] |
| `raw_available_median` | 2.88439 | -0.00107 | -0.00310 | +0.00014 | -0.00525 | 6/9 | [-0.00296, +0.00056] | [-0.00294, +0.00080] |
| `raw_log` | 2.88315 | -0.00231 | -0.00446 | -0.00124 | -0.00597 | 7/9 | [-0.00422, -0.00037] | [-0.00451, -0.00010] |
| `raw_percentile` | 2.88654 | +0.00108 | -0.00099 | +0.00203 | -0.00219 | 4/9 | [-0.00184, +0.00460] | [-0.00095, +0.00307] |
| `raw_percentile_coverage` | 2.88530 | -0.00016 | -0.00404 | +0.00086 | -0.00365 | 5/9 | [-0.00410, +0.00403] | [-0.00248, +0.00219] |

## Production-surface sensitivity

| Variant | RMSE | Delta | Recent | Wins | Season 95% | Player 95% |
|---|---:|---:|---:|---:|---:|---:|
| `normalized_scoring_specific` | 2.88216 | -0.00196 | -0.00499 | 6/9 | [-0.00469, +0.00090] | [-0.00425, +0.00032] |
| `raw_available_median` | 2.88272 | -0.00139 | -0.00172 | 6/9 | [-0.00277, -0.00005] | [-0.00392, +0.00111] |
| `raw_log` | 2.88155 | -0.00256 | -0.00285 | 8/9 | [-0.00375, -0.00139] | [-0.00540, +0.00028] |
| `raw_percentile` | 2.88456 | +0.00045 | -0.00188 | 5/9 | [-0.00260, +0.00355] | [-0.00205, +0.00285] |
| `raw_percentile_coverage` | 2.88387 | -0.00024 | -0.00202 | 5/9 | [-0.00408, +0.00361] | [-0.00311, +0.00260] |

## Governance

- Feature run: `milestone_3_20260730T140041Z_8666f6b2`
- Staged database: `C:\Users\borys\OneDrive\Documents\GitHub\Fantasy_Football\research\studies\2026-07-30_v2_market_rank_challengers\artifacts\local\Projection_V2_beta_single_nffc.sqlite3`
- Locked-incumbent reproduction max delta: `5.33e-15`
- Existing raw-median reproduction max delta: `0`
- 2026 scoring-specific rank providers: 6
- Prespecified advancement candidate: `raw_percentile_coverage`
- Single-league gates all pass: `False`
- Failed gates: `['season_wins_at_least_6', 'season_interval_upper_nonpositive', 'player_interval_upper_nonpositive', 'production_surface_season_wins_at_least_6', 'production_surface_season_interval_upper_nonpositive', 'production_surface_player_interval_upper_nonpositive', 'beats_normalized_rank_by_0_001', 'three_of_four_positions_nonworse', 'early_era_nonworse']`
- Passing these gates only advances the candidate to a nested retune; it does not promote the feature to production.
