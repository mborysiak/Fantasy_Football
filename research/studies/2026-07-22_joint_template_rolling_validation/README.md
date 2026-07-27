# Joint Template Rolling Validation

This study performs a strict rolling-origin validation of the managed auction
weekly-template outcome model. For every held-out target season, historical
donors are restricted to seasons strictly before the target season. Target
pools use only preseason-known projection, market, experience, role, workload,
and disagreement fields. The primary replay uses the causal, out-of-sample
`Final_Validations_Resid` point forecast for each held-out target while keeping
the historical donor bank exactly as production stores it. That reproduces the
forecast-scale transport problem faced by the live app.

The replay evaluates:

- active-PPG residual calibration and P10-P90 coverage;
- managed weekly contribution above the position replacement baseline;
- zero-contribution, `+3 PPG`, and `+5 PPG` event calibration;
- a joint impact proxy requiring both `+3 PPG` outperformance and a
  position-season top-quintile managed contribution;
- tail discrimination by position, projection source, and experience; and
- the production matcher against legacy, projection-only, uniform-weight, and
  uncentered variants.

Run from the model repository root:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-22_joint_template_rolling_validation\run_validation.py
```

Primary outputs are written to `results/production_oos/`. The CSV files directly
under `results/` are retained as a builder-forecast diagnostic; they should not
be used to judge live-app calibration because the historical builder forecast
has a materially different scale from the current final ensemble at QB and TE.
