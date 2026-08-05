# Logged Rank Disagreement Cross-League Decision

- Point feature passes both leagues: `False`.
- Scale feature passes both leagues: `False`.
- Point next action: `retain_outside_production`.
- Scale next action: `retain_outside_production`.

## Primary logged-disagreement deltas

| League | Point controlled RMSE | Point production RMSE | Scale CRPS | Scale relative |
|---|---:|---:|---:|---:|
| DK | +0.00803 | +0.00721 | +0.00125 | +0.07% |
| BETA | +0.00326 | +0.00134 | +0.00041 | +0.03% |

No production feature, model lock, template, or SQLite table was changed.
