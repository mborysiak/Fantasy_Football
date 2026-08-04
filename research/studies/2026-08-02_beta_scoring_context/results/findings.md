# Findings

## Player-level decision

| method | development_core_ppg_relative_delta | development_core_contribution_relative_delta | temporal_core_ppg_relative_delta | depth_composite_relative_delta | worst_position_ppg_relative_delta | worst_tier_ppg_relative_delta | worst_tier_contribution_relative_delta | gate_development_ppg | gate_development_contribution | gate_played_bias | gate_absence_calibration | gate_coverage | gate_position_ppg | gate_temporal_ppg | gate_depth_composite | gate_depth_components | gate_tier_sensitivity | player_level_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beta_context_only | 0.0002662800862751 | -0.0004002700965159 | 0.0008487871074434 | 0.0016535808650046 | 0.0031105897557532 | 0.0010432309647649 | 0.0017559717665827 | True | True | True | True | True | True | True | True | True | True | True |
| beta_scored_full | 0.0001047711515542 | 0.0003789350174234 | 0.002433882764522 | 0.0028245058828606 | 0.0025940138781359 | 0.004461874857782 | 0.0041439647360776 | True | True | True | True | True | True | True | True | True | True | True |

`beta_scored_full` advances to roster validation.

## Template policy audit

| arm | templates | projection_context_rows | legacy_oos_centers | dk_preseason_fallback_centers | beta_expert_fallback_centers | scoring_context_unavailable_rows | template_eligible_rows | missing_active_match_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_hybrid | 5298 | 0 | 2696 | 2602 | 0 | 0 | 5297 | 0 |
| beta_context_only | 5298 | 5298 | 2696 | 2602 | 0 | 39 | 5258 | 0 |
| beta_scored_full | 5298 | 5298 | 2696 | 0 | 2602 | 39 | 5258 | 0 |
