# V2 Linear and Tree Blends

## Question

Does the best governed linear model add complementary signal to full
LightGBM, full random forest, or their fixed 50/50 average?

## Design

The earlier sparse-linear study cannot be blended directly because its 3,701
OOF rows predate the final provider-scoring lineage. This study therefore
refits governed full Lasso on the current 3,696-row population and also
includes the current-lineage projection-core and projection-plus-active Lasso
predictions.

For each linear candidate, the study reports prespecified:

- 50/50 linear/LightGBM;
- 50/50 linear/random forest; and
- equal-third linear/random-forest/LightGBM averages.

For the governed full Lasso only, causal blend weights are also estimated at
each origin from strictly earlier OOF seasons. Two-way Lasso/LightGBM,
Lasso/RF, Lasso/fixed-tree-average, and unconstrained three-component convex
weights are evaluated. The first two validation seasons retain their declared
tree defaults until two prior OOF seasons exist.

This is isolated research. No production model, database, projection,
template, or optimizer output is changed.

```powershell
python research/studies/2026-07-28_v2_linear_tree_blends/run_validation.py
```

See `results/findings.md` for the decision and `results/summary.md` for the
generated score tables.
