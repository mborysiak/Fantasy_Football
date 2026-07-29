# Linear and Tree Blend Findings

## Decision

Advance the governed full-feature Lasso/random-forest/LightGBM equal-third
average as the primary simple blend finalist. Advance the strictly
prior-season-weighted full Lasso/tree-average blend as its causal robustness
check. Do not promote either to production until the planned whole-season,
calibration, and joint-template replay is complete.

The projection-plus-active Lasso equal-third blend has the lowest pooled point
estimate, but its 0.0003 RMSE edge over the governed full blend is negligible,
was identified on the same OOF evidence, and depends on active-game projection
fields available only in 2024-2025. Keep it as a sparse-history challenger,
not the primary blend.

## Comparable design

The earlier sparse-linear study used an older 3,701-row projection lineage, so
its predictions were not blended with the current tree predictions. This study
refits full Lasso on the same final-scoring feature lineage and exact 3,696-row
2017-2025 OOF population used by full LightGBM and full random forest.

Every component prediction remains strictly prior-season. The fixed 50/50 and
equal-third blends require no fitted weights. Causal blend weights for each
target season are estimated only from earlier OOF seasons; 2017-2018 retain
their declared tree defaults until two prior validation seasons exist.

## Results

| Model or blend | RMSE | Comparison | Season wins | 95% interval |
|---|---:|---:|---:|---:|
| Full LightGBM | 3.1230 | -- | -- | -- |
| Full random forest | 3.1242 | -- | -- | -- |
| Full Lasso | 3.1584 | -- | -- | -- |
| Fixed RF/LightGBM average | 3.1143 | -0.0086 vs LightGBM | 5/9 | [-0.0182, +0.0005] |
| Fixed Lasso/LightGBM average | 3.1094 | -0.0136 vs LightGBM | 7/9 | [-0.0275, -0.0003] |
| Fixed Lasso/RF average | 3.1028 | -0.0213 vs RF | 9/9 | [-0.0363, -0.0095] |
| Fixed Lasso/RF/LightGBM equal thirds | **3.1000** | **-0.0143 vs tree average** | **9/9** | **[-0.0225, -0.0075]** |
| Causal Lasso/tree-average blend | 3.1027 | -0.0116 vs tree average | 7/9 | [-0.0196, -0.0049] |
| Causal Lasso/RF/LightGBM blend | 3.1031 | -0.0112 vs tree average | 7/9 | [-0.0189, -0.0045] |

The linear model is weaker alone but more complementary to either tree. Full
Lasso error correlation is 0.953 with RF and 0.962 with LightGBM, compared
with 0.988 between RF and LightGBM. The simple equal-third blend therefore
adds more useful diversity than the RF/LightGBM average alone.

The fixed full three-way blend also lowers MAE from 2.4381 to 2.4152 and
absolute pooled bias from 0.2064 to 0.1385 versus the tree average. Spearman
correlation is effectively unchanged at 0.8055. These pooled diagnostics are
encouraging but do not replace the pending season-level calibration replay.

The causal result supports rather than creates the blend finding. In the
two-component Lasso/tree-average fit, the Lasso weight rises from 24.5% at the
first estimable 2019 origin to 35.1% in 2025. The causal unconstrained
three-way 2025 weights are 36.9% Lasso, 48.1% RF, and 15.0% LightGBM.

## Position and history stability

The fixed full Lasso/RF/LightGBM blend improves over the tree average at every
position:

- QB: 3.4815 versus 3.5114;
- RB: 3.4848 versus 3.4926;
- TE: 2.2716 versus 2.3071; and
- WR: 3.1057 versus 3.1126.

It also improves second-year players by 0.0152 RMSE and veterans by 0.0202.
Limited-history performance is only 0.0021 better overall because rookie RMSE
is 0.0128 worse than the tree average. The projection-plus-active Lasso blend
is stronger for rookies and limited-history players, but that remains an
exploratory routing hypothesis because the active fields have only two seasons
of coverage.

## Next step

Carry the fixed full equal-third blend and causal full Lasso/tree blend into
the same whole-season rolling-origin, bias/calibration, and joint-template
evaluation as the existing full LightGBM, RF/LightGBM average, projection-core
router, and three-role finalist. Preserve all three component predictions so
calibration and template construction operate on auditable components rather
than only the blended point estimate.
