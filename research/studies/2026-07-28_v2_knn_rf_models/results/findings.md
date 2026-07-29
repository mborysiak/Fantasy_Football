# KNN and Random-Forest Findings

## Decision

Reject KNN as a conditional-PPG model and ensemble component. Retain pooled
random forest as a finalist alongside shallow LightGBM. A prespecified 50/50
full-feature RF/LightGBM average is the best point estimate in this study, but
its season interval narrowly crosses zero, so it requires the pending
whole-season/calibration replay before promotion.

## Design

Both families predict the exact 3,696-row 2017-2025 OOF population used by the
pooled LightGBM comparators:

- KNN: median imputation, missingness indicators, standardization, 15/35/75
  neighbors, uniform/distance weights, and Manhattan/Euclidean distance;
- random forest: 250 trees, depth 6/10, minimum leaf size 5/15, and 50%/100%
  feature sampling;
- projection core: 22 projection fields plus four position indicators; and
- full: the governed 31-feature manifest plus four position indicators.

Every target season trains only on earlier seasons. Hyperparameters use the
same five-fold rolling framework. Fixed 50/50 averages with the corresponding
LightGBM prediction were specified before scoring as diversity diagnostics;
no blend weight was tuned.

## Results

| Model | RMSE | Delta versus matching LightGBM | Season wins | 95% interval |
|---|---:|---:|---:|---:|
| Projection-core KNN | 3.2438 | +0.1111 | 1/9 | [+0.0566, +0.1640] |
| Full KNN | 3.3194 | +0.1964 | 0/9 | [+0.1513, +0.2364] |
| Projection-core RF | 3.1423 | +0.0097 | 4/9 | [-0.0041, +0.0272] |
| Full RF | 3.1242 | +0.0012 | 5/9 | [-0.0181, +0.0198] |
| Projection RF/LGBM average | 3.1296 | -0.0031 | 6/9 | [-0.0102, +0.0058] |
| Full RF/LGBM average | **3.1143** | **-0.0086** | 5/9 | [-0.0182, +0.0005] |

KNN is clearly inferior. Scaling and a complete compact distance/neighbor grid
do not rescue it, and a 50/50 KNN/LightGBM average also worsens RMSE by 0.0191
for projection core and 0.0452 for full. The result does not invalidate the
weekly-template matcher: that workflow uses a curated football distance and
samples joint outcome paths, whereas this test asks generic supervised KNN to
estimate conditional mean PPG in the full fitted feature space.

Full RF is effectively tied with full LightGBM as a standalone model and has a
favorable recent point estimate (-0.0094 mean season RMSE in 2023-2025). Its
error correlation with full LightGBM is high at 0.988, but the remaining
diversity is sufficient for the fixed average to improve the pooled point
estimate.

## Slice behavior

The full RF/LightGBM average versus full LightGBM changes RMSE by:

- QB: -0.0170;
- RB: -0.0061;
- WR: -0.0124;
- TE: +0.0013;
- limited history: -0.0204; and
- veterans: -0.0030.

Within history depth, it improves rookies by 0.0068 and second-year players by
0.0301. Projection-core RF/LGBM is slightly better for rookies and other
no-history rows, while the full blend is stronger for second-year players.
These slice results support carrying both component predictions into the final
history-routing check, but are not separate promotion claims.

## Next step

Keep KNN out of the finalist surface. Fit full RF alongside full LightGBM
through 2025, preserve both 2026 shadow predictions, and compare standalone RF,
the untuned 50/50 average, projection core, and the causal history router in
the whole-season/calibration replay. Promote a blend only if its gain survives
temporal, position, bias, and template-integration guardrails.

