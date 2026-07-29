# V2 Projection-Anchored History Gaps

## Question

Should absolute historical PPG fields, whose missing values currently receive
a pooled training-fold median, be replaced by projection-anchored history
gaps with explicit availability and opportunity-game reliability?

## Design

The primary challenger replaces absolute prior-year, three-year, and career
PPG with:

- historical PPG minus the current preseason active-game consensus, falling
  back to team-game consensus;
- zero gap when the corresponding history is unavailable;
- explicit availability, opportunity-game, and recency fields; and
- a prior-year projection residual whose unavailable value is neutral zero.

A secondary challenger shrinks each gap by `games / (games + 8)`. The governed
31-feature incumbent remains unchanged. Incumbent, raw-gap, and shrunk-gap
variants are fit with Lasso, random forest, and deterministic shallow LightGBM
on identical 2017-2025 OOF folds. Fixed equal-third and strictly
prior-season-weighted Lasso/tree blends are also compared.

This is isolated research. It changes no production projection, template, or
optimizer output.

```powershell
python research/studies/2026-07-28_v2_history_gap_features/run_validation.py
```

See `results/findings.md` for the decision and `results/summary.md` for the
generated score tables.
