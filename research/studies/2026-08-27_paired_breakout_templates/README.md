# Paired Breakout Templates

Production retirement (2026-09-05): the Auction display and the four production
`Breakout_Paired_*` tables are retired. This study and its builder remain
research-only. The builder now requires a separate `--output-db` and no longer
supports `--sync-app`; `--simulation-db` is read-only. Historical table details
are preserved in `retired_table_contract.md`. Pre-retirement complete source
and app snapshots are retained in `Data/Production_Refresh_Backups/20260906_breakout_retirement/`.

## Question

Can the current production evidence support an auditable player-level review
surface that pairs season-N breakout/playoff outcomes with the same player's
N+1 performance, without leaking outcomes or allowing keeper salary to define
the archetype?

## Contract

- Current RB/WR/TE targets use causal preseason-N evidence.
- Historical donors pair the same player in N and N+1.
- Signed predicted N+1 growth is retained below zero and stays separate from
  N+1 appearance probability.
- Salary is excluded from every distance dimension.
- Current keepers remain diagnostic rows but are hidden by default in Auction.
- The output is review-only; no optimizer objective changes.

## Run

```powershell
python Scripts/Modeling/build_paired_breakout_templates.py --sync-app
python research/studies/2026-08-27_paired_breakout_templates/run_validation.py
```

Generated validation receipts are written to `results/`.
