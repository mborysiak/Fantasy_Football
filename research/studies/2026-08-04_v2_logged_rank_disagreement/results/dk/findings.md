# Logged Rank Disagreement - DK

## Point-model comparisons

| Surface | Baseline | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Pos nonworse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| controlled_equal_thirds | `incumbent` | `rank_level` | -0.00214 | -0.00508 | 7/9 | [-0.00405, -0.00004] | [-0.00367, -0.00066] | 4/4 |
| controlled_equal_thirds | `rank_level` | `rank_level_logged` | +0.00803 | +0.04360 | 4/9 | [-0.01443, +0.04020] | [-0.00126, +0.01762] | 1/4 |
| controlled_equal_thirds | `rank_level` | `rank_level_excess` | +0.00813 | +0.02819 | 3/9 | [-0.00983, +0.03156] | [-0.00064, +0.01733] | 1/4 |
| equal_thirds | `incumbent` | `rank_level` | -0.00194 | -0.00487 | 7/9 | [-0.00398, +0.00042] | [-0.00391, +0.00003] | 3/4 |
| equal_thirds | `rank_level` | `rank_level_logged` | +0.00721 | +0.04215 | 5/9 | [-0.01417, +0.03812] | [-0.00193, +0.01667] | 1/4 |
| equal_thirds | `rank_level` | `rank_level_excess` | +0.00737 | +0.02682 | 3/9 | [-0.00987, +0.02962] | [-0.00100, +0.01625] | 1/4 |

## Residual-scale comparisons

| Baseline | Challenger | Delta CRPS | Relative | Recent | Wins | Season 95% | Player 95% | 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `scale_rank_level` | `scale_logged` | +0.00125 | +0.07% | +0.00158 | 4/8 | [-0.00014, +0.00286] | [+0.00047, +0.00203] | 83.3% -> 83.1% |
| `scale_rank_level` | `scale_excess` | +0.00123 | +0.07% | +0.00166 | 4/8 | [-0.00011, +0.00287] | [+0.00039, +0.00207] | 83.3% -> 83.0% |

## Decision

- Logged point feature passes every gate: `False`.
- Logged scale feature passes every gate: `False`.
- Point next action: `retain_outside_production`.
- Scale next action: `retain_outside_production`.
- The excess-disagreement variant is a sensitivity only.
- No production table, feature contract, or model lock was changed.

## Lineage

- Feature run: `milestone_3_20260804T193552Z_d67a2d3f`
- Read-only database SHA-256: `F3F8BAC5D82FC5F00EF8771E90871739570C15133F24D2897258ADEB185FB082`
