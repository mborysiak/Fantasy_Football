# V2 Conditional-PPG and Participation Model Comparison

## Question

Can a compact residual or direct model improve on preseason expert consensus,
and can a separate participation model improve on a leakage-safe prior-position
rate, without reviving the broad S1/S2 feature and pipeline search?

## Design

- Use the reviewed V2 Milestone 3 feature manifests.
- Use the existing `SciKitModel.time_series_cv` five-fold scheme.
- Generate OOF predictions for every 2017-2025 validation row using only prior
  seasons for each prediction.
- Compare raw, compact, KBest, PCA, agglomeration, and shallow nonlinear
  challengers separately.
- Evaluate pooled, season-mean, position, season, experience/history,
  provider-depth, and provider-era results.
- Keep all outputs shadow-only.

The exact data, CV, model, and publication rules are documented in
`docs/data_contracts/v2_modeling_framework.md`.

## Run

```powershell
python -m Scripts.V2.build_milestone_4
```

Durable summary outputs are written under `results/`. Player-level OOF rows are
stored in the ignored V2 SQLite database rather than duplicated as a large CSV.

After the core run, execute the fold-identical family dropouts with:

```powershell
python research/studies/2026-07-27_v2_modeling_framework/run_feature_ablation.py
```

The dropout study holds the simple model family and CV scheme fixed and removes
one reviewed feature family at a time. Positive error deltas mean that removing
the family hurt the full model.

Refresh the lightweight paired season comparisons after any core rerun with:

```powershell
python research/studies/2026-07-27_v2_modeling_framework/summarize_results.py
```

`results/initial_findings.md` records the interpreted initial decision and the
critical participation-target correction.
