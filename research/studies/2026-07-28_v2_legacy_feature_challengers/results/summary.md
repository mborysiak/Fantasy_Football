# Legacy-Inspired Feature Challenger Results

Negative deltas mean the added feature family improved conditional-PPG RMSE versus the same model on the original 31-feature manifest.

## lightgbm

| Variant | Added | RMSE | Pooled delta | Mean season delta | 95% season bootstrap | Wins |
|---|---:|---:|---:|---:|---:|---:|
| `plus_opportunity_share` | 4 | 3.1414 | -0.0028 | -0.0030 | [-0.0111, +0.0048] | 6/9 |
| `plus_market_room` | 5 | 3.1461 | +0.0018 | +0.0016 | [-0.0077, +0.0122] | 4/9 |
| `plus_experience_context` | 3 | 3.1484 | +0.0042 | +0.0040 | [-0.0023, +0.0097] | 3/9 |
| `plus_all_legacy` | 12 | 3.1523 | +0.0080 | +0.0075 | [-0.0073, +0.0233] | 4/9 |

## ridge

| Variant | Added | RMSE | Pooled delta | Mean season delta | 95% season bootstrap | Wins |
|---|---:|---:|---:|---:|---:|---:|
| `plus_opportunity_share` | 4 | 3.1729 | -0.0001 | +0.0002 | [-0.0104, +0.0087] | 4/9 |
| `plus_experience_context` | 3 | 3.1737 | +0.0007 | +0.0007 | [-0.0010, +0.0025] | 4/9 |
| `plus_market_room` | 5 | 3.1793 | +0.0062 | +0.0062 | [-0.0112, +0.0277] | 4/9 |
| `plus_all_legacy` | 12 | 3.1862 | +0.0131 | +0.0133 | [-0.0112, +0.0456] | 5/9 |
