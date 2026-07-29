# V2 Projection Consensus Ladder

## Question

Has the projection-only point layer reached diminishing returns, or can causal
provider aggregation, provider-relative information, room disagreement, and
active-PPG alignment improve it before history, ADP, and lifecycle features
enter the full residual model?

## Prespecified ladder

1. Configured-score provider median.
2. Projection consensus components and uncertainty.
3. Causal constrained provider stack.
4. Compact provider room/depth-chart disagreement.
5. Standardized active-PPG alignment where projected games exist.
6. Previously tested projection shape/rate features.
7. All projection-only additions together.

The provider stack is trained against realized team-game PPG because provider
season totals live on that scale. At every target season it:

- uses only provider projections and outcomes from earlier seasons;
- differentiates a provider only after three prior accuracy seasons;
- retains the configured median as a shrinkage component;
- estimates nonnegative weights that sum to one;
- selects global and position-shrinkage penalties using earlier inner origins;
- leaves current/new providers in the equal-weight median without assigning
  them a learned provider weight.

Projection-only Lasso and deterministic full-column shallow LightGBM use the
same five-fold 2017-2025 OOF framework as M4A. A second deterministic LightGBM
stage adds the new causal features to the governed 31-feature incumbent to
test whether projection-only gains survive the full residual context.

This is isolated shadow research. It does not modify production projections,
templates, optimizers, feature manifests, or databases.

```powershell
python research/studies/2026-07-28_v2_projection_consensus_ladder/run_validation.py
python research/studies/2026-07-28_v2_projection_consensus_ladder/analyze_results.py
```

See [`results/findings.md`](results/findings.md) for the decision readout.
