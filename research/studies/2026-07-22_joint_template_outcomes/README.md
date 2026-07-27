# Joint Weekly Template Outcomes

This study verifies the production weekly-template rebuild that:

- adds absolute PPG, projection-disagreement, and workload-room matching;
- uses adaptive distance-kernel donor probabilities with a 5% donor cap;
- retains ordinary zero-active seasons while excluding the declared 2018
  Le'Veon Bell contract holdout; and
- synchronizes the source-owned tables to the managed auction app.

Run from the model repository root:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-22_joint_template_outcomes\verify_build.py
```

The app-side direct joint residual/path behavior is covered by
`Fantasy_Football_App/tests/test_sequential_target.py`.

