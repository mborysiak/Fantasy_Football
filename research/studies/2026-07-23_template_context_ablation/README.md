# Weekly Template Context Ablation

This study extends the strict rolling-origin joint-template validation from
`2026-07-22_joint_template_rolling_validation`. It keeps the same held-out
targets, causal production-scale point forecasts, earlier-season-only donor
rule, centered residual transport, and managed weekly scoring.

The study tests three preseason-known additions to template matching:

- **Draft capital:** log-scaled NFL draft-chart value, with a base match weight
  of `0.75` that halves every two NFL seasons of target experience.
- **Supporting-cast environment:** a within-season/position percentile built
  from projected QB1, top-two RBs, and top-four WR/TEs. The target player's own
  contribution is removed before ranking. Its match weight is `0.35`.
- **Recency:** a sampling prior with an eight-season half-life. Four- and
  twelve-season half-lives are retained as sensitivity checks. Recency never
  admits a same-season or future donor.

The primary comparison is the predeclared combined specification versus the
unchanged production matcher. All single features and the complete two-feature
factorial are retained as ablations.

Run from the model repository root:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-23_template_context_ablation\run_validation.py
```

Outputs are written to `results/` and include target-level predictions,
calibration summaries, paired deltas from production, candidate-specific
season-clustered bootstrap intervals, feature-coverage audits, and a generated
readout.
