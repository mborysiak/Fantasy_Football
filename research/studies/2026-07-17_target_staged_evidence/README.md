# Target Staged-Evidence Smoke Test

This focused smoke test recreates the live roster edge case with Jayden Daniels,
Jahmyr Gibbs, Chase Brown, and Bhayshul Tuten fixed. It runs the production
320-organic / four-pilot / 64-preliminary / 96-confirmation Target process and
audits:

- protected 20-player heuristic coverage plus four broad-pilot discoveries;
- confirmation coverage that preserves ten evidence leaders and appends up
  to four highest-market-salary usable preliminary candidates;
- separate preliminary and confirmation sample counts;
- continuous disagreement-adjusted LCB80 ranking and evidence labels;
- dynamic Top-N Pass successor handling;
- forced-Buy roster position mix and WR/TE budget composition; and
- full eight-worker execution across all three stages.

Run from the model repository root with:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-07-17_target_staged_evidence\run_edge_smoke.py
```

Use `--market-confirm-anchors 0` for the same-seed eight-confirmation runtime
baseline.

Run the fast synthetic ranking, disagreement, balanced-block, pilot-allocation,
confirmation-allocation, and Top-N contracts with:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-07-17_target_staged_evidence\verify_ranking_and_discovery.py
```

Verify that process count changes runtime only, while the fixed eight logical
evidence blocks and same-seed Target results remain identical:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-07-17_target_staged_evidence\verify_worker_invariance.py
```
