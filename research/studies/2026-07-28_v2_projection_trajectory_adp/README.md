# V2 Projection Trajectory and Logged ADP

## Question

Do preseason-to-preseason projection changes add cleaner historical relevance
than realized-stat gaps, especially for injured or zero-game players? Does
replacing raw ADP with `log1p(ADP)` improve the linear component?

## Design

The projection-trajectory family uses only team-game preseason consensus:

- current projected PPG minus exact prior-year projected PPG;
- current projected PPG minus a 3/2/1 recency-weighted prior-three-year
  projected PPG;
- exact-prior availability;
- prior-three-year projection count; and
- prior-three-year projection volatility.

Missing projection history receives zero change plus zero availability/count.
No realized games or statistics enter these features. Team-game PPG is used on
both sides so recent active-game projection coverage cannot create an era
artifact.

Raw ADP is replaced, not supplemented, by `log1p(ADP)` in the log variants.
Incumbent, trajectory, log-ADP, and combined variants are compared with Lasso,
random forest, shallow deterministic LightGBM, tree averages, and equal-third
linear/tree blends on identical 2017-2025 OOF rows.

This is isolated research. No production projection, template, or optimizer
output changes.

```powershell
python research/studies/2026-07-28_v2_projection_trajectory_adp/run_validation.py
python research/studies/2026-07-28_v2_projection_trajectory_adp/run_ablation.py
```

The ablation run separates the exact prior-year projection change from the
three-year projection-history context.

See [`results/findings.md`](results/findings.md) for the decision readout.
