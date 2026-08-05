# Logged Expert-Rank Level Cross-League Confirmation

- Raw log passes both leagues: `False`.
- Next action: `retain_normalized_rank_as_challenger`.

| League | Controlled raw-log minus normalized | Recent | Wins | Production delta |
|---|---:|---:|---:|---:|
| DK | +0.00076 | -0.00027 | 5/9 | +0.00174 |
| BETA | -0.00162 | -0.00087 | 5/9 | -0.00249 |

A tie retains normalized rank because it is more robust to provider depth and overall QB placement.
No production feature, model lock, template, or SQLite table changed.
