# Cross-Repo Context

Last updated: 2026-07-06

## Repos

- `Fantasy_Football`: modeling, source data generation, residual predictions,
  best-ball weekly template tables, and source `Simulation.sqlite3`.
- `Fantasy_Football_Snake`: Streamlit snake-draft app, ILP optimizer, app copy
  of `Simulation.sqlite3`, and app-side runtime sampling.
- `Fantasy_Football_App`: another configured app consumer of `Simulation.sqlite3`
  when present.

## Database Handoff

`Scripts/Modeling/s4_Best_Ball_Weekly.py` writes the source tables to:

```text
Data/Databases/Simulation.sqlite3
```

It then copies that SQLite file to configured app paths, including:

```text
../Fantasy_Football_Snake/app/Simulation.sqlite3
```

## App-Sensitive Tables

- `Best_Ball_Weekly_Templates`
- `Best_Ball_Weekly_Template_Pools`
- `Best_Ball_Weekly_Pool_Summary`
- `Best_Ball_Weekly_Player_Map`
- `Best_Ball_Weekly_Template_Audit`
- `Best_Ball_Weekly_Player_Pool_Audit`
- `Best_Ball_Weekly_Bucket_Audit`
- `Best_Ball_ADP_Audit`

## Coordination Rules

- If a column used by `Fantasy_Football_Snake/app/zSim_Helper.py` changes, update
  the Snake app and its app-side data contract in the same task when possible.
- If a new audit table is intended for the app UI, document its display intent in
  the Snake runbook before adding UI.
- Keep generated DBs out of durable explanations; describe schemas and behavior
  in docs instead.
