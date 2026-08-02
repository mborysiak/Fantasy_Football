# Weekly-Template Projection Trajectory

This study tests whether preseason projection trajectory improves WR weekly
template matching and distinguishes the motivating Ladd McConkey/Terrelle
Pryor comparison.

The candidate features are:

- current consensus team-game PPG minus exact prior-year consensus PPG;
- current consensus PPG minus the 3/2/1 recency-weighted mean of available
  projections from the prior three years;
- exact-prior availability; and
- prior-three-year projection depth.

Signed gaps are converted to position-season percentiles among players with
the relevant history. Rookies and other no-history players receive the neutral
zero-change profile; explicit availability/depth variants keep that synthetic
zero distinct from a veteran whose projection was genuinely stable.

The strict rolling replay uses 648 held-out 2017-2025 WR targets per league and
only prior-season donors. Every arm retains the production top-80 pool,
adaptive kernel, 12-season recency prior, 5% donor cap, centered residual, and
joint weekly trajectory.

Run DK:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_template_projection_trajectory\run_validation.py
```

Run beta:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_template_projection_trajectory\run_validation.py `
  --league beta `
  --v2-db Data\Databases\Projection_V2_beta.sqlite3 `
  --results-dir research\studies\2026-07-30_template_projection_trajectory\results_beta
```

Production code and databases are not changed.

## Result

No trajectory feature is promoted. The features successfully demote Terrelle
Pryor in Ladd McConkey's current pool, but every arm worsens held-out WR PPG
CRPS in both leagues over both the full and recent periods. Three-year
trajectory at weight 0.25 improves several managed-contribution and impact
diagnostics, especially in recent beta, but DK full-period PPG CRPS worsens
with its season-cluster interval above zero. History availability/depth should
remain audit metadata rather than match-distance inputs. See
`results/findings.md`.
