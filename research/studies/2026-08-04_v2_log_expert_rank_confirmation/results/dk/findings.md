# Logged Expert-Rank Level Confirmation - DK

| Surface | Baseline | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Positions nonworse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| controlled_equal_thirds | `incumbent` | `normalized_rank` | -0.00214 | -0.00508 | 7/9 | [-0.00405, -0.00004] | [-0.00367, -0.00066] | 4/4 |
| controlled_equal_thirds | `normalized_rank` | `raw_log` | +0.00076 | -0.00027 | 5/9 | [-0.00139, +0.00313] | [-0.00154, +0.00313] | 2/4 |
| equal_thirds | `incumbent` | `normalized_rank` | -0.00194 | -0.00487 | 7/9 | [-0.00398, +0.00042] | [-0.00391, +0.00003] | 3/4 |
| equal_thirds | `normalized_rank` | `raw_log` | +0.00174 | +0.00149 | 2/9 | [-0.00033, +0.00410] | [-0.00084, +0.00436] | 1/4 |

## Decision

- Raw log passes every gate: `False`.
- Next action: `retain_normalized_rank_as_challenger`.
- Failed gates: `['controlled_pooled_improvement_at_least_0_001', 'controlled_at_least_6_season_wins', 'controlled_season_interval_upper_nonpositive', 'controlled_player_interval_upper_nonpositive', 'production_pooled_nonworse', 'production_recent_nonworse', 'controlled_at_least_3_positions_nonworse']`.
- A tie retains normalized rank because it is depth- and QB-placement robust.
- No production feature, lock, template, or table changed.

## Lineage

- Feature run: `milestone_3_20260804T193552Z_d67a2d3f`
- Database SHA-256 before/after: `F3F8BAC5D82FC5F00EF8771E90871739570C15133F24D2897258ADEB185FB082`
- Rank representation missingness mismatches: `0`
