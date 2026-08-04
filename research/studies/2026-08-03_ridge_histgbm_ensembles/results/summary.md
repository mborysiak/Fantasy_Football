# Ridge / histogram-gradient-boosting result

Negative RMSE deltas favor the challenger. Intervals use a paired player-cluster bootstrap.

| Scoring | Comparison | Baseline | Challenger | Delta | 95% interval |
|---|---|---:|---:|---:|---:|
| beta | HistGBM equal-fourths | 2.892611 | 2.893633 | +0.001022 | [-0.004970, +0.007113] |
| dk | HistGBM equal-fourths | 3.090995 | 3.094641 | +0.003646 | [-0.000529, +0.007856] |
| beta | Ridge equal-fourths diagnostic | 2.892611 | 2.888226 | -0.004385 | [-0.011992, +0.003230] |
| dk | Ridge equal-fourths diagnostic | 3.090995 | 3.087797 | -0.003198 | [-0.010615, +0.004222] |
| beta | HistGBM replaces LightGBM | 2.892611 | 2.890254 | -0.002357 | [-0.009915, +0.005049] |
| dk | HistGBM replaces LightGBM | 3.090995 | 3.088459 | -0.002536 | [-0.005938, +0.000731] |
| beta | Lasso/Ridge split linear third | 2.892611 | 2.889094 | -0.003517 | [-0.006014, -0.001093] |
| dk | Lasso/Ridge split linear third | 3.090995 | 3.089642 | -0.001352 | [-0.002870, +0.000181] |
| beta | Ridge replaces Lasso | 2.892611 | 2.886263 | -0.006348 | [-0.011263, -0.001338] |
| dk | Ridge replaces Lasso | 3.090995 | 3.088552 | -0.002443 | [-0.005467, +0.000617] |

## Screen decision

Ridge primary verdict: `research_shadow_candidate` (6/6 season cells improved).
HistGBM primary verdict: `reject` (3/6 season cells improved).

Both scorings selected Ridge alpha 10, so the expanded regularization boundaries did not improve the pre-2023 selection score. Replacing Lasso improves MAE and reduces positive bias in both scorings, but Lasso/Ridge error correlations exceed 0.996, RB RMSE worsens slightly in both, and DK's player-cluster interval crosses zero. Ridge remains a shadow only.

DK selected the shallow HistGBM schedule with zero L2; beta selected the deeper schedule with L2 10. The prespecified equal-four comparison worsens both scorings. Replacing LightGBM is mildly favorable in pooled RMSE but wins only three of six season cells and has intervals crossing zero. Reject HistGBM on this surface.

This is a shadow screen on a reused confirmation block. No production files changed.
