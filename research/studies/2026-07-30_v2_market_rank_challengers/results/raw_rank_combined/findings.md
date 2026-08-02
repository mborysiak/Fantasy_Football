# Raw Expert-Rank Cross-League Decision

- Candidate: `raw_percentile_coverage`
- Both leagues pass every gate: `False`
- Next action: `retain_outside_production`
- DK failed gates: `['beats_normalized_rank_by_0_001', 'early_era_nonworse', 'player_interval_upper_nonpositive', 'production_surface_player_interval_upper_nonpositive', 'production_surface_season_interval_upper_nonpositive', 'season_interval_upper_nonpositive']`
- Beta failed gates: `['beats_normalized_rank_by_0_001', 'early_era_nonworse', 'player_interval_upper_nonpositive', 'production_surface_player_interval_upper_nonpositive', 'production_surface_season_interval_upper_nonpositive', 'production_surface_season_wins_at_least_6', 'season_interval_upper_nonpositive', 'season_wins_at_least_6', 'three_of_four_positions_nonworse']`

## Headline RMSE deltas versus incumbent

| League | Surface | Normalized | Raw median | Raw log | Raw percentile | Percentile + coverage |
|---|---|---:|---:|---:|---:|---:|
| DK | controlled | -0.00311 | +0.00029 | -0.00329 | -0.00035 | -0.00138 |
| DK | production | -0.00264 | +0.00018 | -0.00344 | -0.00103 | -0.00230 |
| BETA | controlled | -0.00228 | -0.00107 | -0.00231 | +0.00108 | -0.00016 |
| BETA | production | -0.00196 | -0.00139 | -0.00256 | +0.00045 | -0.00024 |

## Exploratory raw-log diagnostic

`raw_log` was a prespecified scale diagnostic, not the advancement candidate. Its direct difference from the matched normalized comparator is not distinguishable in this study:

| League | Surface | Raw log - normalized | Wins | Season 95% | Player 95% |
|---|---|---:|---:|---:|---:|
| DK | controlled | -0.00018 | 5/9 | [-0.00266, +0.00230] | [-0.00267, +0.00230] |
| DK | production | -0.00080 | 5/9 | [-0.00434, +0.00244] | [-0.00349, +0.00187] |
| BETA | controlled | -0.00002 | 5/9 | [-0.00234, +0.00209] | [-0.00260, +0.00250] |
| BETA | production | -0.00061 | 5/9 | [-0.00354, +0.00232] | [-0.00330, +0.00204] |

Descriptively, the percentile-plus-coverage point gain is larger in the expanded-provider era and in the 45 OOF rows with no rank. That no-rank slice is exploratory and has no interaction interval. Among rank-available rows the candidate changes controlled RMSE by -0.00024 DK and +0.00027 beta.

Passing this receipt would advance the feature only to a strict nested-retune validation; it would not itself change production.
