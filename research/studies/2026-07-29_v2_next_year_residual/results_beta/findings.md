# Beta Findings

The independently scored beta lineage confirms the DK conclusion.

- The equal-third residual blend scores 3.6718 RMSE versus 4.6685 for the
  origin expert carry-forward, winning all eight validation origins.
- Appearance LightGBM scores 0.1623 Brier versus 0.1748 for logistic and
  0.2648 for the position/experience prior.
- The beta weekly-template replay improves all-period weekly-PPG CRPS by at
  most 0.0023, but season-contribution CRPS is worse for every tested addition.
- DK and beta publish the same 751 canonical 2027 candidate keys, while their
  scoring-specific conditional centers are fitted independently.

Keep beta next-year predictions in
`Data/Databases/Projection_V2_beta.sqlite3`. Do not reuse DK conditional PPG
or promote either next-year field into the production weekly matcher.
