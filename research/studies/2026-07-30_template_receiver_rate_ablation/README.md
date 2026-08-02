# Weekly-Template Receiver-Rate Ablation

This study tests whether preseason projected receiving efficiency helps the
weekly-template matcher distinguish receiver archetypes that have similar
projected receiving fantasy points.

The two candidate fields are:

- projected receiving yards per reception; and
- projected receiving touchdowns per reception.

Both fields come from the league-specific V2 preseason feature mart and are
joined to template rows by canonical `player_key` and season. Realized receiving
rates are never used. Each rate is converted to a season-position percentile
and lightly shrunk toward the neutral percentile using
`projected_receptions / (projected_receptions + 10)`.

The predeclared primary comparison adds both fields to WR and TE at weight 0.50
each. Diagnostic arms isolate each field, test combined 0.25 and 1.00 weights,
and extend the 0.50 specification to RB.

Every arm retains:

- strictly prior donors;
- the production top-80 donor pool;
- the production adaptive distance kernel;
- the 12-season recency prior;
- the 5% donor probability cap; and
- the centered joint PPG-residual/weekly-path outcome.

Run DK:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_template_receiver_rate_ablation\run_validation.py
```

Run beta:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_template_receiver_rate_ablation\run_validation.py `
  --league beta `
  --v2-db Data\Databases\Projection_V2_beta.sqlite3 `
  --results-dir research\studies\2026-07-30_template_receiver_rate_ablation\results_beta
```

The study is read-only with respect to production databases and matcher
configuration.

## Result

The rates changed roughly nine of each WR/TE target's 80 donors, so they do
differentiate projected profiles. They did not clear the joint outcome gates:
the combined arm made small PPG/contribution gains while weakening WR
played-games or impact behavior. Production remains unchanged.

TE-only yards per reception is the promising same-evidence follow-up, with
cross-league contribution gains, but DK recent played-games behavior prevents
promotion. See `results/findings.md`.
