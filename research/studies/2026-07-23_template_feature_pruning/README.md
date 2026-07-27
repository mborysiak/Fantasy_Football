# Weekly Template Feature Pruning

This study runs a strict rolling backward ablation of the production weekly
template matcher before any recency change is promoted.

Protected anchors:

- absolute projected PPG;
- position-relative projection rank;
- uncapped NFL experience; and
- each position's primary rushing/receiving role signals.

Candidate removals are evaluated as correlated families rather than isolated
columns:

- projection-by-experience interaction;
- ADP rank and market/projection gap alternatives;
- projection disagreement;
- position-specific component ranks;
- redundant room hierarchy;
- room concentration; and
- pass-catcher team-QB environment.

Every feature set is evaluated both with the current weighting and with a
12-season recency sampling prior. Selection uses 2017-2022 development origins,
predeclared calibration/downside guardrails, position-level CRPS safety checks,
and a paired season-level one-standard-error rule that prefers the simplest
surviving feature set. The externally fixed recency prior is not counted as an
additional feature. A nested rolling selector beginning in 2021 chooses each
season's specification using only earlier validation origins.

Run from the model repository root:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-23_template_feature_pruning\run_validation.py
```

Production code and generated template tables are not changed by this study.
