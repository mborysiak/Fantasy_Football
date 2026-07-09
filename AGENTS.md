# Working Agreement

## How To Work
- Think first and keep changes narrow.
- Prefer the existing script style and repo conventions over new infrastructure.
- Ask only when ambiguity is consequential and cannot be resolved from the repo.
- Push back early on data leakage, calibration drift, unstable joins, schema drift,
  and unsupported modeling assumptions.

## Startup Reading Order
1. Read `Agent_Notes/SESSION_NOTES.md`.
2. Read the latest relevant monthly note under `Agent_Notes/Session_Notes/`.
3. Check `Agent_Notes/MODULE_TRACKER.md` and `Agent_Notes/DECISION_LOG.md`.
4. For best-ball work, read `docs/data_contracts/best_ball_weekly_tables.md`.
5. For validation or calibration experiments, check `research/README.md`.

## Notes Policy
- `Agent_Notes/SESSION_NOTES.md` is the landing page for active state.
- `Agent_Notes/MODULE_TRACKER.md` stores durable module status and next steps.
- `Agent_Notes/DECISION_LOG.md` stores durable modeling and process decisions.
- `Agent_Notes/Session_Notes/YYYY-MM.md` stores short task receipts.
- If a fact should survive beyond one task, move it out of a session note.

## Research Policy
- Put runnable investigations in `research/studies/YYYY-MM-DD_<slug>/`.
- Put durable outputs in each study's `results/` folder.
- Keep throwaway local artifacts out of root-level app/model folders.
- Promote lasting conclusions into notes or docs after the study.

## Cross-Repo Policy
- This repo owns model inputs, projections, validation tables, and the source
  `Data/Databases/Simulation.sqlite3`.
- `Fantasy_Football_Snake` consumes the copied `Simulation.sqlite3` for the app.
- When changing Simulation tables used by the Snake app, update the relevant
  data contract here and the app-side contract/runbook in the Snake repo.
- Do not treat generated SQLite databases as code review evidence by themselves;
  summarize the schema or behavioral change in notes.

## Verification
- Run focused checks when practical.
- For importable script changes, prefer `python -m py_compile <file>` as a quick
  syntax check when full model runs are too expensive.
- For DB/schema changes, inspect the resulting tables and update data contracts.
