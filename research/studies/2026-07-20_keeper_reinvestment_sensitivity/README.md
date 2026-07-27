# Keeper Reinvestment Sensitivity

This study tests the budget-transfer mechanism that the preceding bench-local
keeper replay intentionally could not measure. A cheap keeper option is forced
into a nominal bench slot, then the other roster slots are re-solved under the
same `$298` cap and current-season managed objective. Starters are not locked,
so bench savings can finance a stronger starting core.

The one-year option contract is unchanged:

```text
max(next_year_market_value - (acquisition_price + 10), 0)
```

The portfolio utility remains the expected best positive surplus across all
five bench players. Historical next-year distributions use the causal
`Model_Validations_Resid` path from the preceding study.

Policies are nested sensitivities:

- `control`: a same-engine full current-only roster rebuilt on the complete
  construction-bank expected profile;
- `reinvest_k1`: at most one newly forced keeper-oriented bench player;
- `reinvest_k2`: at most two; and
- `reinvest_k3`: at most three.

At each step, the study ranks outside-roster candidates by their marginal
expected-best portfolio contribution, forces each shortlisted candidate along
with any prior accepted candidates, and fully re-solves the remaining roster.
The candidate must remain on the nominal bench, improve expected-best keeper
utility, and preserve the baseline full-bank reference score. The independent
forecast bank, raw historical season, playoff weeks, waiver use, and realized
keeper result are evaluation outcomes rather than selection gates.

Run the mechanics check:

```powershell
python research/studies/2026-07-20_keeper_reinvestment_sensitivity/verify_mechanics.py
```

Run a smoke replay:

```powershell
python research/studies/2026-07-20_keeper_reinvestment_sensitivity/run_replay.py `
  --years 2024 --trials 4 --contexts 20 --projection-draws 250 `
  --candidate-shortlist 6 `
  --output-dir research/studies/2026-07-20_keeper_reinvestment_sensitivity/artifacts/local/smoke
```

Run or resume the full replay:

```powershell
python research/studies/2026-07-20_keeper_reinvestment_sensitivity/run_replay.py
python research/studies/2026-07-20_keeper_reinvestment_sensitivity/run_replay.py --resume
```
