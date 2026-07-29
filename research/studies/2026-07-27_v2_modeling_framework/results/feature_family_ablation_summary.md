# Feature-Family Dropout Results

Positive deltas mean that removing the family made the primary error metric worse, so the family added OOF value in the full linear model.

## conditional_ppg

| Dropped family | Features | Pooled delta | Mean season delta | 95% season bootstrap | Worse seasons |
|---|---:|---:|---:|---:|---:|
| `projection_level` | 5 | +0.1175 | +0.1187 | [+0.0657, +0.1703] | 8/9 |
| `projection_uncertainty` | 3 | +0.0101 | +0.0097 | [-0.0060, +0.0260] | 6/9 |
| `history` | 7 | +0.0046 | +0.0049 | [-0.0090, +0.0185] | 6/9 |
| `room` | 4 | +0.0033 | +0.0032 | [-0.0017, +0.0082] | 5/9 |
| `availability` | 1 | +0.0021 | +0.0022 | [-0.0064, +0.0137] | 6/9 |
| `team` | 2 | +0.0001 | +0.0001 | [-0.0100, +0.0093] | 4/9 |
| `lifecycle` | 4 | -0.0009 | -0.0010 | [-0.0145, +0.0128] | 5/9 |
| `role_composition` | 3 | -0.0039 | -0.0038 | [-0.0134, +0.0065] | 3/9 |
| `market` | 2 | -0.0147 | -0.0147 | [-0.0359, +0.0117] | 2/9 |

## participation

| Dropped family | Features | Pooled delta | Mean season delta | 95% season bootstrap | Worse seasons |
|---|---:|---:|---:|---:|---:|
| `projection_level` | 1 | +0.0096 | +0.0093 | [+0.0054, +0.0139] | 9/9 |
| `projection_uncertainty` | 1 | +0.0049 | +0.0046 | [-0.0005, +0.0091] | 7/9 |
| `history` | 8 | +0.0037 | +0.0036 | [+0.0013, +0.0059] | 6/9 |
| `lifecycle` | 5 | +0.0021 | +0.0021 | [+0.0002, +0.0054] | 7/9 |
| `market` | 2 | +0.0009 | +0.0009 | [-0.0011, +0.0029] | 6/9 |
| `availability` | 1 | +0.0005 | +0.0005 | [-0.0000, +0.0017] | 4/9 |
| `team` | 1 | -0.0000 | -0.0000 | [-0.0002, +0.0001] | 5/9 |
