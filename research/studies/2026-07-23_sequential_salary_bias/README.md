# Sequential salary-bias replay

This study tests whether the selected-roster salary bias found in the static
managed-auction optimizer remains after moving target selection to the blind
Sequential Target policy.

Four paired arms are evaluated at rolling 2022-2025 origins:

1. Static full sampled salary surface, no selection reserve.
2. Static full sampled salary surface, half-strength selection reserve.
3. Blind Sequential Target, no selection reserve.
4. Blind Sequential Target, half-strength selection reserve.

Each cell shares its preseason origin, construction draw, and nomination-order
seed. The static optimizer sees a complete sampled salary surface. The blind
policy compiles a point-price completion plan, then sees the replay price for
only the current nominee and can pivot through the production cached-priority
policy.

The key diagnostics are:

- initial-plan actual salary minus point salary;
- final-roster actual salary minus point salary;
- historical-cap feasibility;
- blind-path completion and paid-price legality;
- the static scenario discount (point spend minus sampled-surface spend);
- player-level selection rates and residual concentration.

Historical nomination order, losing bids, and opponent reactions are not
available. Nomination order therefore uses the production noisy salary-order
generator, and historical prices are treated as an exogenous replay tape.

Run:

```powershell
python research/studies/2026-07-23_sequential_salary_bias/run_study.py
```

A quick smoke run can use:

```powershell
python research/studies/2026-07-23_sequential_salary_bias/run_study.py `
  --years 2025 --trials 2 --contexts 8 --context-draws 4 `
  --projection-draws 32 --salary-draws 32 `
  --output-dir research/studies/2026-07-23_sequential_salary_bias/results_smoke
```
