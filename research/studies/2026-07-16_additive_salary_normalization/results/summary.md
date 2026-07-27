# Additive Salary Normalization Audit

This holds the v3 rolling raw salary predictions fixed and changes only
the final known-budget reconciliation rule.

| Method | Mean actual - prediction | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Proportional above `$1` | $-0.379 | $4.363 | $6.221 |
| Additive with `$1` floor | $-0.391 | $4.293 | $6.128 |

On 527 observed player-years, additive normalization changes MAE by $-0.070 and RMSE by $-0.093.

This result supports testing additive normalization in the full v4
optimizer replay, but it does not measure the incremental value of the
new keeper-market input features.
