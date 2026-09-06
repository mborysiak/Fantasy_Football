# LCB-Aligned Roster Structures

Validate experimental Sequential App v19 against the active 2026 beta
Brown/Tuten keeper state. The study runs the normal paired Buy-versus-Pass
board, adds an optional second evidence batch, and records the confirmed
positive LCB anchors plus the conditional Buy-rollout families surfaced by the
new Preferred Roster Structure panel.

```powershell
python research/studies/2026-08-25_lcb_aligned_structures/run_validation.py --budget 320 --batches 2 --workers 4
```

The validation intentionally uses the existing sequential rollouts. It does
not run the legacy unforced structure ILPs or add a second optimization layer.

