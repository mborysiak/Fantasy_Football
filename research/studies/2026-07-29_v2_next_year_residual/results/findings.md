# Findings

> Superseded data lineage: these metrics predate the governed identity,
> source-season, and beta provider-scoring corrections. Retain them as
> historical evidence only. The accepted corrective replay is documented in
> `../../2026-07-29_v2_identity_scoring_revalidation/results/findings.md`.

The leakage-safe next-year residual model is a useful shadow replacement for
the legacy forward-filled target, but it is not a new weekly-template matching
feature.

- The target is `conditional PPG in t+1 - expert team-game PPG in t`.
- A forecast made in preseason `t` trains on origin labels no later than
  `t-2`, whose outcomes end in `t-1`.
- Confirmed players with no `t+1` appearance are participation zeros.
  Conditional PPG remains null. Unresolved identities remain unlabeled.
- DK residual blend RMSE is 3.9003 versus 5.2070 for carrying the origin expert
  projection forward, with eight of eight origin wins.
- DK appearance LightGBM Brier is 0.1604 versus 0.1732 for logistic and 0.2648
  for the position/experience prior.
- The largest conditional-PPG gains occur for QB and limited-history players,
  where the naive carry-forward baseline is especially poor. This is partly
  survivor selection, so it cannot be interpreted without appearance risk.
- The DK template replay improves all-period weekly-PPG CRPS by at most 0.0023,
  while meaningful weights worsen contribution CRPS. No next-year field is
  promoted into production matching.

The 2027 DK output therefore stays as a two-part shadow distribution:
conditional PPG plus appearance probability. Production integration needs an
explicit disappearance/availability mixture and must continue to use one
joint donor residual and weekly path rather than drawing a second independent
PPG residual.
