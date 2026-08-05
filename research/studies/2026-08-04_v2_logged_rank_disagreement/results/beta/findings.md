# Logged Rank Disagreement - BETA

## Point-model comparisons

| Surface | Baseline | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Pos nonworse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| controlled_equal_thirds | `incumbent` | `rank_level` | -0.00161 | -0.00310 | 6/9 | [-0.00277, -0.00041] | [-0.00342, +0.00028] | 4/4 |
| controlled_equal_thirds | `rank_level` | `rank_level_logged` | +0.00326 | +0.03239 | 4/9 | [-0.02014, +0.03399] | [-0.00676, +0.01336] | 1/4 |
| controlled_equal_thirds | `rank_level` | `rank_level_excess` | +0.00920 | +0.02421 | 4/9 | [-0.00707, +0.03103] | [+0.00001, +0.01849] | 1/4 |
| equal_thirds | `incumbent` | `rank_level` | -0.00118 | -0.00029 | 5/9 | [-0.00270, +0.00043] | [-0.00322, +0.00090] | 4/4 |
| equal_thirds | `rank_level` | `rank_level_logged` | +0.00134 | +0.02977 | 5/9 | [-0.02192, +0.03189] | [-0.00856, +0.01129] | 1/4 |
| equal_thirds | `rank_level` | `rank_level_excess` | +0.00720 | +0.01978 | 4/9 | [-0.00895, +0.02821] | [-0.00186, +0.01630] | 1/4 |

## Residual-scale comparisons

| Baseline | Challenger | Delta CRPS | Relative | Recent | Wins | Season 95% | Player 95% | 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `scale_rank_level` | `scale_logged` | +0.00041 | +0.03% | +0.00037 | 5/8 | [-0.00150, +0.00252] | [-0.00058, +0.00137] | 83.4% -> 83.5% |
| `scale_rank_level` | `scale_excess` | -0.00002 | -0.00% | -0.00051 | 4/8 | [-0.00240, +0.00221] | [-0.00100, +0.00093] | 83.4% -> 83.4% |

## Decision

- Logged point feature passes every gate: `False`.
- Logged scale feature passes every gate: `False`.
- Point next action: `retain_outside_production`.
- Scale next action: `retain_outside_production`.
- The excess-disagreement variant is a sensitivity only.
- No production table, feature contract, or model lock was changed.

## Lineage

- Feature run: `milestone_3_20260804T193947Z_0d110127`
- Read-only database SHA-256: `23C9F0B43F17B0F6748FD2D7BED0DE1165E48284C810FC8583ACB9B909027FC1`
