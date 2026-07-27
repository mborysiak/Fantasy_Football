# Weekly Template Weight Sensitivity

This study checks whether the retained weekly-template match weights need a
final local adjustment before the recommended feature/recency update is
promoted.

The reference specification is the feature-pruning recommendation:

- current production weights;
- no `projection_x_exp` interaction;
- direct projected PPG and uncapped experience retained; and
- a fixed 12-season recency sampling half-life.

Each retained conceptual feature family is moved independently to 75% and 125%
of its reference weight. A one-dimensional near-uniform-to-150% sweep scales
every retained match weight together to diagnose overall distance-kernel
sharpness. Positive scaling preserves the matched top-80 donor ordering while
changing how strongly the closest members are favored. Donor eligibility, pool
size, adaptive weighting, probability cap, centering, and outcome scoring
remain unchanged.

Selection uses 2017-2022 origins only. Aggregate calibration/downside
guardrails, position-level CRPS safety checks, a paired season-level
one-standard-error diagnostic, untouched 2023-2025 summaries, nested rolling
selection, and season-clustered bootstrap comparisons are reported. Because
this is a multiple-comparison local search, a new weight is promoted only if it
improves development composite CRPS by at least 0.1%, passes every development
safety guardrail, is non-worse in the temporal composite, remains within the
same position-level temporal safety bounds, and repeats in at least two of the
three 2023-2025 nested selections.

Run from the model repository root:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-23_template_weight_sensitivity\run_validation.py
```

Production code and generated template tables are not changed by this study.
