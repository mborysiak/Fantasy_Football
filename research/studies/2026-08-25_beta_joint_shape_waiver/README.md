# Beta Joint, Shape, and Waiver Comparison

Paired 2026 beta construction comparison starting from the active Chase Brown
`$34` and Bhayshul Tuten `$11` keeper state. Other active beta keepers are
removed from the market and their salaries/slots are subtracted.

The four arms reuse identical construction and validation contexts:

- `current_additive`: current QB1/RB4-6/WR4-6/TE1-2 constraints;
- `joint_one_swap`: the same additive solution followed by one full exact
  construction-bank conditional swap;
- `fixed_shape_additive`: exactly QB1 and TE1, with RB/WR counts constrained to
  5/6 or 6/5;
- `waiver_plus_1_5_additive`: current constraints with every positional waiver
  baseline raised by 1.5 PPG during construction.

All arms are scored on a common independent holdout using the current waiver
baselines. The raised-waiver arm also records its own-assumption score only as a
diagnostic; it is not used for cross-arm quality claims.

Run:

```powershell
python research\studies\2026-08-25_beta_joint_shape_waiver\run_comparison.py
```

Durable outputs are written under `results/`.
