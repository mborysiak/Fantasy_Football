# Nominal Salary Buffer Replay

This study extends the frozen 2022-2025 managed-auction rolling replay with a
second roster-price constraint. Every roster must still fit the sampled salary
market at `$298`. Constrained cells must also satisfy:

```text
sum(normalized point-predicted salaries) <= 298 + nominal buffer
```

The paired grid holds Top-N on, the projected waiver baseline, and bench-upside
weight `0.25` fixed while crossing:

- salary draws: one versus the current average of five
- nominal buffer: no constraint, `$0`, `$5`, `$10`, `$15`, `$25`

Point-predicted salaries are normalized once per origin to the same
keeper-adjusted remaining league money and slots as every stochastic salary
market. Recorded league keepers remain unavailable and affect market
normalization; this empty-personal-roster replay has no acquired-player spend.
In a live draft, already acquired players must count at deterministic paid
prices in both constraints.

The runner imports the prior study's read-only frozen loaders, pre-origin weekly
construction, raw target-season scorer, and independent forecast-evaluation
bank. It records the prior runner and manifest hashes and requires the two
unconstrained controls to reproduce the prior replay exactly during the full
default run.

Run a smoke check from the model repository:

```powershell
python research/studies/2026-07-14_nominal_salary_buffer_replay/run_buffer_replay.py `
  --years 2025 --trials 2 --contexts 8 --context-draws 3 `
  --projection-draws 100 --salary-draws 100 `
  --output-dir research/studies/2026-07-14_nominal_salary_buffer_replay/artifacts/local/smoke
```

Run the full paired replay:

```powershell
python research/studies/2026-07-14_nominal_salary_buffer_replay/run_buffer_replay.py
```

Durable outputs are written to `results/`. This is a guardrail test on the
historical frozen salary laws--mainly legacy uncertainty for 2023-2025--not yet a
walk-forward rebuild of the current empirical residual-quantile salary method.

The full run completed 12,000 optimal cells and exactly reproduced all 2,000
unconstrained parent controls. Five draws remained more affordable than one draw.
The provisional affordability guardrail is a `$5` nominal buffer; `$10` is the
looser, point-preserving alternative. See `results/decision_readout.md` for the
tradeoff and `results/source_manifest.json` for the validation contract.
