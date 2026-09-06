# Tiered Target Board

Experimental validation for the App worktree
`C:/Users/borys/GitHub/Fantasy_Football_App_target_tiers`.

The experiment intentionally avoids a conditional scenario tree. It solves one
current optimal roster per construction-evidence block, groups accumulated
plans into recurring position-spend/roster-shape families, and preserves exact
paths plus within-position salary/PPG neighborhoods. Add Evidence appends four
fresh paths and rebuilds the accumulated family support. Marking a player out
in the App triggers the existing full board refresh.

Run:

```powershell
python research/studies/2026-08-25_tiered_target_board/run_structure_audit.py --league beta
python research/studies/2026-08-25_tiered_target_board/run_structure_audit.py --league nv
```

Durable outputs are written under `results/`.
