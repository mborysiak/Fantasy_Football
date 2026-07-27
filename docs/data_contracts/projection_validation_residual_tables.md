# Projection Validation Residual Tables

`Scripts/Modeling/s1_Stacking_Model.py`, `s2_RunAll.py`, and
`s3_Model_Ensemble.py` own these tables in
`Data/Databases/Validations.sqlite3`.

## `Model_Validations_Resid`

This table retains each model slice's out-of-sample point prediction and adds
empirical residual quantiles calibrated within that same homogeneous slice.

The row identity is:

```text
(version, year, pos, rush_pass, dataset, filter_data, year_exp,
 current_or_next_year, season, player)
```

Here, `year` is the model-specification as-of year and `season` is the forecast
origin. `current` models target `season`; `next` models target `season + 1`.

| Column | Meaning |
| --- | --- |
| `pred_resid_5` through `pred_resid_95` | Residual quantile offsets to add to `pred_fp_per_game` |
| `resid_target_season` | Realized season targeted by the model row |
| `resid_target_available` | Whether the stored `y_act` is a valid realized target for evaluation |
| `resid_training_rows` | Number of eligible prior residual donors |
| `resid_training_through_origin` | Latest donor forecast origin |
| `resid_training_through_season` | Latest realized donor outcome season |
| `resid_calibration_available` | Whether the row has a complete residual-quantile vector |
| `resid_calibration_mode` | Calibration method identifier |

Calibration is causal relative to the forecast origin:

- `current` origin `t` can use realized outcomes through `t - 1`;
- `next` origin `t` can also use realized outcomes only through `t - 1`,
  creating an extra origin embargo because a next-model donor at origin `d`
  realizes in `d + 1`;
- the target row's own outcome is not needed to assign its interval; and
- insufficient early donor history produces null quantiles rather than future
  borrowing.

Terminal `next` rows remain useful forecasts and can have calibrated intervals,
but their target outcomes are not yet realized. Consumers must not use `y_act`
when `resid_target_available = 0`.

## `Final_Validations_Resid`

This table is the historical counterpart to
`Simulation.Final_Predictions_Resid`. Its unique row key is:

```text
(version, model_spec_asof_year, season, player, pos)
```

The point and residual ensemble mirrors the production family weights:

1. summed rush/pass/receiving component-current models;
2. noncomponent all-current models; and
3. noncomponent all-next models for non-QBs.

Available families receive equal weight, as in `s3_Model_Ensemble.py`.
Component residual magnitudes use a position-specific prior-only residual
correlation, with the existing `0.35` fallback when fewer than 50 complete
component samples are available.

Target columns remain separate because the production ensemble mixes horizons:

| Column | Meaning |
| --- | --- |
| `season` | Forecast origin and historical salary-feature year |
| `y_act` | Realized current-season PPG |
| `y_act_next` | Realized next-season PPG when available |
| `y_act_ensemble_target` | Family-weighted target, populated only when every contributing family target is realized |
| `ensemble_target_available` | Completeness flag for `y_act_ensemble_target` |

Residual provenance includes source coverage, donor counts, donor cutoffs, and
`resid_calibration_status`:

- `complete`: every contributing base row has calibrated residuals;
- `partial`: a usable final quantile vector exists but some base sources are
  uncalibrated; and
- `unavailable`: no complete final quantile vector exists.

Historical rows preserve the full validation universe. Current-app position
rank truncation and manual player downgrades are not applied.

## Backfill

The one-time migration and audits live under
`research/studies/2026-07-16_projection_validation_residual_backfill/`.
It preserved all 42,351 base rows, created 6,006 final ensemble rows, and
verified that final point means reproduce the prior historical reconstruction
to floating-point precision.
