# Weekly Template Context Ablation

## Design

- Held out 1,620 player-seasons at strict rolling origins.
- Every donor season is earlier than its target season.
- Target point forecasts use the same production OOS scale as the prior joint-template validation.
- The production baseline reproduces the prior study exactly across all 1,620
  target distributions.
- The primary specification was declared before the replay: experience-decayed draft capital, 0.35 supporting-cast distance, and an eight-season recency half-life.

## Recent 2020-2025

| method | ppg_crps | ppg_bias | ppg_80_coverage | contribution_crps | contribution_bias | played_crps | plus5_brier | impact_brier | extended_absence_brier | impact_auc | weighted_season_gap | weight_10plus_seasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_context_hl8 | 1.9172 | -0.0075 | 0.8046 | 20.5642 | 0.4541 | 1.5327 | 0.0637 | 0.0959 | 0.1097 | 0.6679 | 5.8111 | 0.2032 |
| production_baseline | 1.9183 | -0.0075 | 0.8083 | 20.5763 | 0.8908 | 1.5384 | 0.0639 | 0.0963 | 0.11 | 0.6548 | 7.214 | 0.3234 |

## Season-clustered primary-vs-baseline uncertainty

Negative score deltas favor the primary context specification.

| metric | primary_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_primary_better |
| --- | --- | --- | --- | --- |
| ppg_crps | -0.0011 | -0.0051 | 0.0031 | 0.6715 |
| contribution_crps | -0.012 | -0.0401 | 0.0143 | 0.7885 |
| played_crps | -0.0057 | -0.0102 | -0.0015 | 0.998 |
| plus3_brier_row | 0.0001 | -0.001 | 0.0012 | 0.4045 |
| plus5_brier_row | -0.0002 | -0.0005 | 0.0001 | 0.9425 |
| impact_brier_row | -0.0005 | -0.0012 | 0.0003 | 0.8825 |
| zero_brier_row | 0.0001 | -0.0003 | 0.0004 | 0.315 |
| extended_absence_brier_row | -0.0002 | -0.0006 | 0.0001 | 0.892 |

## Interpretation

- The combined specification is safe but only incrementally better: recent PPG
  and contribution CRPS improve by about 0.06%, while played-games CRPS improves
  by 0.37%. Only the played-games interval excludes zero in the six-season
  cluster bootstrap.
- Recency supplies nearly all of the stable gain. The eight-season prior reduces
  mean donor age by 1.37 seasons and the 10+-year weight from 32.3% to roughly
  20.5% without a meaningful calibration cost. Eight and twelve seasons both
  improve recent played-games CRPS with season-bootstrap intervals below zero.
  Twelve is slightly safer in the 2023-2025 point/event checks; four is too
  aggressive.
- Draft capital materially tightens pedigree distance for young players, but
  its PPG, participation, and residual-tail results are mixed by position. It
  does not earn a global weekly-template weight.
- Supporting-cast matching is also unstable: its temporal contribution result
  is encouraging, but it does not repeat in development seasons and adds little
  beyond recency.
- Do not promote the full combined matcher. The supported production candidate
  is a light recency prior alone in the eight-to-twelve-season range. The data do
  not resolve the exact half-life; twelve is the conservative implementation
  candidate. Keep draft capital for a separately calibrated upside layer and
  supporting-cast context as an auditable diagnostic until stronger evidence
  exists.

Runtime: 60.6 seconds.
