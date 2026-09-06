# Historical paired-breakout table contract

Retired from production on 2026-09-05; retained for optional research artifacts.

## Paired Breakout Review Tables

`Scripts/Modeling/build_paired_breakout_templates.py` builds a separate,
review-only donor surface for current breakout, late/playoff production, and
following-season performance. It does not replace the production weekly pools
or alter Auction construction scoring.

### `Breakout_Paired_Templates`

Each donor is one historical player-season `N` paired to the same player's
observed season `N+1`. The season-N path is reconstructed in points from the
managed weekly template multipliers and `managed_profile_ppg`. Following-season
appearance remains a separate binary outcome; non-appearance has zero
unconditional N+1 PPG and null conditional N+1 PPG. Breakout, playoff, late
surge, future-high-performer, and joint flags are position-season percentiles.

All matcher evidence must be available before season N. The signed N+1 growth
feature is the causal preseason N forecast of conditional N+1 PPG minus the
season-N preseason PPG center. It remains signed, is not capped at zero, and is
not multiplied by appearance probability. Salary and keeper discount are not
match dimensions.

### `Breakout_Paired_Template_Pools`

Each current RB/WR/TE receives 80 same-position donors when the full production
surface is available. Distance uses current projection strength, experience,
canonical ADP, projection disagreement, role/room context, signed N+1 growth,
and a small separate N+1 appearance term. The production-style recency kernel,
probability normalization, and 5% donor cap apply. Pool probabilities must sum
to one for every current player.

### `Breakout_Paired_Player_Map`

Current player rows retain the match features, weighted historical event
frequencies, pool diagnostics, and current keeper status. Keepers remain in the
table for diagnosis but the Auction review board excludes them by default.
These weighted donor frequencies are not calibrated event probabilities.

### `Breakout_Paired_Template_Audit`

The audit stores target/keeper counts, pool-size and probability checks,
17-week reconstruction coverage, donor availability/embargo exclusions,
position/appearance-contract failures, origin range, and a required zero count
of salary match features.

