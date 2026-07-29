# Projection-Weight Template Replay Findings

Date: 2026-07-29

## Decision

Retain the production matcher weights. Do not globally increase absolute PPG,
component-rank, or raw component-magnitude weights.

## Pooled results

Candidate minus production CRPS is shown below; lower is better, so positive
values are adverse.

| Candidate | DK PPG | beta PPG | DK contribution | beta contribution | DK played | beta played |
|---|---:|---:|---:|---:|---:|---:|
| PPG 1.50 -> 2.25 | +0.00326 | +0.00160 | +0.02753 | +0.02139 | +0.00509 | +0.00287 |
| PPG 1.50 -> 3.00 | +0.00499 | +0.00339 | +0.03423 | +0.02513 | +0.00771 | +0.00523 |
| Component ranks 1.00 -> 1.50 | +0.00090 | +0.00011 | +0.00340 | -0.00306 | -0.00087 | -0.00164 |
| Raw component magnitude at 1.00 | +0.00418 | +0.00239 | +0.02220 | +0.01877 | +0.00466 | +0.00392 |
| PPG 2.25 + component ranks 1.50 | +0.00215 | +0.00161 | +0.01224 | +0.02110 | +0.00252 | +0.00181 |
| PPG 2.25 + ranks 1.50 + raw 1.00 | +0.00412 | +0.00396 | +0.03090 | +0.03992 | +0.00792 | +0.00481 |
| PPG 3.00 + ranks 2.00 + raw 1.50 | +0.00654 | +0.00540 | +0.03413 | +0.03790 | +0.00933 | +0.00639 |

The moderate PPG-plus-rank candidate has DK PPG interval
`[+0.00024, +0.00421]`; its beta interval crosses zero
`[-0.00039, +0.00375]`. Its 2023-2025 PPG deltas are also adverse:
`+0.00346` DK and `+0.00303` beta.

Raw component magnitude is clearly redundant or harmful. Its PPG interval is
entirely adverse in both leagues.

## Position follow-up

The global raw-PPG bump is especially harmful for QB. A 2.25 PPG weight
worsens QB PPG CRPS by `+0.01467` DK and `+0.01611` beta. RB shows tiny
full-period PPG gains from higher PPG weight, but those reverse or flatten in
2023-2025 and do not consistently improve contribution or played-games CRPS.

Increasing only QB rushing/passing component-rank weights from 1.00 to 1.50 is
the one exploratory favorable pattern:

| League | PPG delta | Contribution delta | Played delta | PPG player-cluster 95% interval |
|---|---:|---:|---:|---:|
| DK | -0.00144 | -0.03588 | -0.00303 | [-0.00643, +0.00380] |
| beta | -0.00330 | -0.05900 | -0.00602 | [-0.00952, +0.00322] |

All three metrics also improve for QB in 2023-2025, but every player-cluster
interval crosses zero. DK PPG improves in only 4/9 seasons, while beta improves
in 7/9. Because the QB-only route was identified after inspecting the global
position slices, keep it as an exploratory future challenger rather than
promoting it.

## Interpretation

The absolute-PPG coefficient understates the matcher's total projection
emphasis when viewed alone. Projection percentile already carries weight 2.50,
and the matcher also includes component ranks, projection/market gap, provider
disagreement, and room shares. Raising absolute magnitude further makes donor
selection too projection-dominant and weakens transfer of contribution and
availability outcomes.

Production databases and matcher weights were not changed.
