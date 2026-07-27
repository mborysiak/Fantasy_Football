# Managed Auction Rolling-Origin Replay

This study replays the current managed-season **Target** construction logic at
four frozen preseason origins (2022-2025). It compares a paired 2 x 2 x 2 x 2
factorial:

- one versus five salary draws
- the Top-12 salary constraint off versus on
- the current projected waiver baseline versus a prior-season empirical baseline
- bench-upside weight `0` versus `0.25`

The replay uses the forecast database that existed before target-season results
were imported. Construction weekly profiles are rebuilt from preseason features
and raw historical weeks strictly before the origin year. Target-season raw weeks
are reserved for scoring. This prevents the current historical validation
ensemble and target-season templates from leaking outcomes into construction.

Every variant is scored in the same realized environment: causal lineup choices,
explicit played masks, and the same target-season waiver stream. Waiver ranking is
causal, but eligibility is hindsight availability-filtered using target-week played
evidence; the scorer also omits opponent competition and transaction persistence.
Recorded keepers are unavailable to the empty-roster replay and their historical
salaries remain deterministic. The study validates look-ahead roster construction;
it cannot recreate Current Nomination because no nomination order or auction-state
log exists.

Forecast EV is evaluated on an independently seeded context bank rather than on
the contexts sampled by the construction objective. Target-season realized points
remain the primary outcome.

Historical final auction prices are held fixed as an exogenous feasibility check;
the study does not claim that prices would remain unchanged under a different
nomination or bidding path.

Run a smoke check from the model repository:

```powershell
python research/studies/2026-07-13_managed_auction_rolling_replay/run_replay.py `
  --years 2025 --trials 4 --contexts 20 --salary-calibration-draws 100 `
  --output-dir research/studies/2026-07-13_managed_auction_rolling_replay/artifacts/local/smoke
```

Run the full paired replay:

```powershell
python research/studies/2026-07-13_managed_auction_rolling_replay/run_replay.py
```

Durable outputs are written to `results/`. Local smoke-test output belongs under
`artifacts/local/` and is ignored.

Primary frozen forecast sources:

| Origin | Source |
|---|---|
| 2022 | Git snapshot `fea8ab4845dd0c5efb26292cb59007c865a3a003` |
| 2023 | `Data/Databases/DB_Versioning/Simulation__2023_08_28_52.sqlite3` |
| 2024 | `Data/Databases/DB_Versioning/Simulation__2024_08_26_48.sqlite3` |
| 2025 | `Data/Databases/DB_Versioning/Simulation__2025_08_24_55.sqlite3` |

The runner records SHA-256 hashes, row counts, join coverage, configuration, and
the exact current `zSim_Helper.py` hash in `source_manifest.json`.
