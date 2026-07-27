# Compact Salary Feature Audit

The compact surface uses 12 substantive features versus 155 nonconstant legacy features.

Across 526 strict rolling player-years, the fixed six-model ensemble changed MAE from $4.561 to $4.312 and RMSE from $6.740 to $6.461.

The retained features cover:

- keeper/budget-adjusted league source price and broader-market log ADP;
- projected scoring level, projection-versus-price disagreement, and P90 residual upside;
- position-room points share, RB rush share, experience, rookie status, and three position indicators with WR as reference.

The two remaining correlations at or above 0.90 are deliberate: adjusted source salary versus log ADP represents two distinct market anchors, and RB rush share versus the RB indicator is the structural relationship created by a position-specific role interaction.

See `feature_associations.csv` for correlations with actual salary, the copied-source residual, and the prior v3 raw OOF residual.
