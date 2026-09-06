# Beta Non-Keeper Salary Counterfactual

## Purpose

Rebuild the governed 2026 beta auction salary surface while treating Chase
Brown, Bhayshul Tuten, Luther Burden III, and Colston Loveland as non-keepers.
All other players in the active beta keeper file remain fixed at their current
contracts.

Only Chase Brown and Bhayshul Tuten are active keepers in the production input;
Luther Burden III and Colston Loveland are already non-keepers. The
counterfactual therefore removes Brown and Tuten and verifies that all four
players are modeled on the open market.

The runner copies every database read or written by the salary builder into a
temporary directory and passes the counterfactual keeper CSV through the
existing `FF_KEEPERS_FILE` override. It does not modify production databases,
the production keeper CSV, or either app database.

## Run

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-08-26_beta_nonkeeper_salary_counterfactual/run_counterfactual.py
```

Durable outputs are written to `results/`. Temporary database copies are
deleted automatically after the validated result tables are extracted.

## Validation contract

- the counterfactual keeper slice contains 12 players spending $396;
- none of the four candidate players is a keeper;
- the 2026 `betapred` surface has exact player-key parity with the production
  beta projection population;
- the 144 highest-priced non-keepers sum to the remaining $3,180 league budget;
- all four candidate players have model-derived salary uncertainty rather than
  deterministic keeper overrides; and
- the live keeper input and databases retain their pre-run hashes.

