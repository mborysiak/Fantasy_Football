# Logged Expert-Rank Level Confirmation - BETA

| Surface | Baseline | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Positions nonworse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| controlled_equal_thirds | `incumbent` | `normalized_rank` | -0.00161 | -0.00310 | 6/9 | [-0.00277, -0.00041] | [-0.00342, +0.00028] | 4/4 |
| controlled_equal_thirds | `normalized_rank` | `raw_log` | -0.00162 | -0.00087 | 5/9 | [-0.00401, +0.00051] | [-0.00408, +0.00077] | 4/4 |
| equal_thirds | `incumbent` | `normalized_rank` | -0.00118 | -0.00029 | 5/9 | [-0.00270, +0.00043] | [-0.00322, +0.00090] | 4/4 |
| equal_thirds | `normalized_rank` | `raw_log` | -0.00249 | -0.00191 | 5/9 | [-0.00525, -0.00012] | [-0.00508, +0.00008] | 4/4 |

## Decision

- Raw log passes every gate: `False`.
- Next action: `retain_normalized_rank_as_challenger`.
- Failed gates: `['controlled_at_least_6_season_wins', 'controlled_season_interval_upper_nonpositive', 'controlled_player_interval_upper_nonpositive']`.
- A tie retains normalized rank because it is depth- and QB-placement robust.
- No production feature, lock, template, or table changed.

## Lineage

- Feature run: `milestone_3_20260804T193947Z_0d110127`
- Database SHA-256 before/after: `23C9F0B43F17B0F6748FD2D7BED0DE1165E48284C810FC8583ACB9B909027FC1`
- Rank representation missingness mismatches: `0`
