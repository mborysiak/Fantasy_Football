# NFFC Weekly-Template Center Replay

This study is a shadow-only, strict rolling-origin comparison of two historical
donor residual centers for the offense-only NFFC weekly-template path.

Everything except the donor point center is held fixed:

- NFFC scoring and a 17-week horizon;
- scoring-matched preseason context from the NFFC V2 feature mart;
- the current production matching features, weights, 12-season recency prior,
  adaptive kernel, and probability cap;
- held-out target centers from the locked NFFC out-of-fold model;
- target seasons 2023-2025; and
- donor seasons restricted to 2021 through the season immediately before each
  target.

The two arms are:

1. `expert_donor_center`: NFFC-scored preseason expert PPG.
2. `locked_oof_donor_center`: the locked, strictly out-of-fold NFFC V2 PPG.

The old `Model_Inputs.avg_proj_points` fallback is not an eligible arm because
its quarterback totals are on the DK scoring scale. It is retained only in the
context audit.

## Prespecified decision rule

The locked center is recommended only if all of these conditions pass:

- pooled PPG CRPS is strictly lower than the expert-center arm;
- the player-clustered 95% interval for locked-minus-expert PPG CRPS ends at or
  below zero;
- locked wins PPG CRPS in at least two of the three held-out seasons;
- pooled contribution and played-games CRPS worsen by no more than 0.25%;
- PPG, contribution, and played-games 80% coverage each fall by no more than
  one percentage point;
- each event Brier score worsens by no more than 0.001;
- absolute PPG bias worsens by no more than 0.10 PPG;
- no position's three-metric CRPS composite worsens by more than 0.5%; and
- no individual position/metric CRPS worsens by more than 1%.

The contribution metric retains the existing managed-auction replacement
baselines, so it is a safety diagnostic rather than a direct NFFC contest
scoring target. PPG and played-games calibration are primary.

Only three modern 17-week target seasons are available. Player-clustered
uncertainty and per-season/per-position consistency therefore carry more weight
than season-bootstrap precision.

## Run

From the model repository root, with the completed staged NFFC V2 database:

```powershell
$env:FF_CURRENT_SEASON = "2026"
$env:PYTHONPATH = "..\ff;..\Scikit_Model"
python research\studies\2026-07-31_nffc_template_center_replay\run_validation.py `
  --v2-db <stage>\Projection_V2_nffc.sqlite3
```

The runner reads source databases but does not modify production code, live
SQLite databases, or app artifacts. Durable outputs are written under
`results/`.

## Conclusion

Retain the scoring-matched expert donor center. Do not promote the locked OOF
donor center.

Across 540 held-out player seasons, locked-minus-expert PPG CRPS was
`+0.002901` (`+0.139%`), contribution CRPS was `+0.046970` (`+0.180%`), and
played-games CRPS was unchanged. The locked arm lost PPG CRPS in each of
2023, 2024, and 2025. Its player-clustered 95% interval was
`[-0.004914, +0.010748]`, with a 24.9% bootstrap probability of improvement.
It passed six of the ten prespecified gates and failed all three promotion
gates: pooled PPG improvement, a nonpositive player-cluster upper bound, and
wins in at least two target seasons. It also missed the one-point coverage
safety gate.

The production-context audit also confirms why the old
`Model_Inputs.avg_proj_points` fallback is not a valid NFFC donor center. The
largest scoring mismatch is at quarterback, where the mean absolute difference
from the NFFC-scored V2 expert total is roughly 54-72 season points across the
2021-2025 donor years.

Full metrics, uncertainty, calibration, and gate results are in
[`results/findings.md`](results/findings.md).
