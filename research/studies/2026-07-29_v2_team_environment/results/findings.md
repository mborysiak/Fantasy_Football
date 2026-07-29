# Team Environment and QB Style Findings

## Decision

Do not add the full 11-feature team-environment family to the global
conditional-PPG model. No prespecified equal-third blend comparison survives
the six-family correction, and the broad supporting-cast and team-scoring
families do not improve pooled RMSE.

Carry one compact field, `team_qb1_rush_point_share`, as a pass-catcher-only
whole-season and template finalist. It has a coherent WR/TE mechanism and
favorable recent evidence, but its same-OOF routed interval still crosses
zero. Do not yet use it as a production adjustment or as an RB penalty.

## Construction

The 11-field research manifest uses preseason projection evidence only:

- QB1 passing and rushing yards;
- QB1 passing and rushing touchdowns;
- QB1 rushing fantasy-point share;
- top-2-RB, top-3-WR, and top-1-TE core projected points;
- within-season team core percentile;
- self-excluded core supporting-cast points;
- core-plus-QB1 rushing yards and touchdowns; and
- offensive touchdowns defined as QB1 passing TDs plus core-plus-QB1 rushing
  TDs.

The offensive-TD feature does not add receiving TDs, avoiding a second count
of the same passing score. All new fields cover 98.0% of the 3,696 common OOF
rows. Projected target aggregates are excluded because their broad coverage
starts only in 2024.

## Primary fold-identical results

The reference is the 31-feature incumbent plus the five-field preseason
projection-trajectory family. Negative deltas favor the challenger.

| Addition | Blend RMSE | Delta | 2023-2025 delta | Wins | BH q-value |
|---|---:|---:|---:|---:|---:|
| QB yardage split | 3.0949 | +0.0000 | -0.0093 | 4/9 | 0.996 |
| QB TD split | 3.0978 | +0.0028 | -0.0051 | 4/9 | 0.852 |
| QB rushing share | 3.0928 | -0.0021 | -0.0089 | 5/9 | 0.852 |
| Core/supporting cast | 3.0969 | +0.0020 | +0.0030 | 4/9 | 0.852 |
| Team rush/scoring | 3.0950 | +0.0001 | -0.0085 | 5/9 | 0.996 |
| All 11 fields | 3.0946 | -0.0003 | -0.0074 | 5/9 | 0.996 |

QB rushing share is the best compact global point estimate. It improves Lasso
by 0.0053 and LightGBM by 0.0045 RMSE, while random forest is flat. Its pooled
season-bootstrap interval crosses zero and it improves only five of nine
seasons, so it does not clear the global promotion boundary.

## Position mechanism

The QB-rushing-share blend changes RMSE relative to the trajectory reference
by:

- QB: `+0.0108`;
- RB: `-0.0002`;
- WR: `-0.0055`; and
- TE: `-0.0068`.

The result is therefore a pass-catcher effect, not a general team-context or
RB effect. WR improves in all three 2023-2025 seasons by 0.0167, 0.0165, and
0.0052 RMSE. TE improves by 0.0229, 0.0237, and 0.0141.

In high-rush QB environments, the trajectory reference overprojects WR/TE
conditional PPG by 0.48 on average. The QB-style model lowers those
predictions by 0.13 PPG. The full QB-style blend improves the high-rush slice
by 0.0131 RMSE and worsens the low-rush slice by 0.0056, which is consistent
with a missing mobile-QB pass-volume adjustment rather than generic QB quality.

## Exploratory role route

Using the QB-style blend for WR/TE and the trajectory reference for QB/RB
scores 3.0916 versus 3.0949, a 0.0033 RMSE improvement. It wins seven of nine
seasons, improves the 2023-2025 mean by 0.0085, and has a season-bootstrap
interval of `[-0.0070, +0.0002]` with exact sign-flip `p=0.1016`.

This route was motivated by the position diagnostic on the same OOF evidence.
Carry it as a prespecified future whole-season/template test, not as an
independently confirmed model.

## Team-quality conclusion

Core supporting-cast strength worsens the global blend and RB RMSE. Team
rushing/scoring is neutral globally and helps pass catchers more than RBs.
Current expert projections and room features appear to absorb most team-quality
mean signal. Keep these governed fields for template variance, upside, and
weekly-path research rather than expanding the point model.
