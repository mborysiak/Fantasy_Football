# V2 Position-Aware Feature Families

## Question

Can feature families that are weak globally improve particular positions
because experience curves, room ambiguity, and opportunity concentration have
different meanings for QB, RB, WR, and TE?

## Prespecified families

Each family is added to the governed 31-feature full LightGBM within an
independently fitted position model:

- `experience_context`: projection versus same-position experience peers;
- `market_room`: self-excluded same-position teammate ADP gaps and room share;
- `opportunity_role`: position-relevant projected team opportunity shares; and
- `room_clarity`: richer positional-room size, gap, disagreement, and
  pass-catcher/QB context.

An `all_targeted` variant combines the four families but is secondary. The
primary family tests are interpreted jointly across 16 position-family cells
using exact season sign-flip p-values and Benjamini-Hochberg false-discovery
rates. Pooled RMSE, season-cluster intervals, 2023-2025 direction, and
limited-history versus veteran slices remain explicit.

`team_target_share` is excluded because it does not begin until 2024 and has
only one learnable OOF season. Opportunity families instead use historically
available rush-attempt, reception, and receiving-yard shares appropriate to
each position.

This is isolated research. It does not modify the V2 database, production
models, projections, templates, or optimizers.

```powershell
python research/studies/2026-07-28_v2_position_feature_families/run_validation.py
```

See [`results/findings.md`](results/findings.md) for the decision readout.

