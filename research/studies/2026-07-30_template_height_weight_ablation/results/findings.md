# Height/Weight Template-Matching Findings

## Decision

Do not add height or weight to the production weekly-template distance.
Do not pull combine data merely to replace the player-master height and weight.

The nflverse player master already supplies effectively complete basic
measurements, and the strict replay shows a real but league-unstable matching
effect. A future combine study would need a separate, prespecified hypothesis
for athletic testing such as size-adjusted speed rather than being treated as
the next stage of this size result.

## Design and coverage

- Replayed six fixed methods on 1,620 strict rolling 2017-2025 targets in each
  of DK and beta.
- Joined the existing nflverse player master to V2 identity by exact `gsis_id`,
  with exact `pfr_id` fallback, then joined templates by canonical
  `player_key`.
- Converted height and weight to season-position percentiles.
- The primary arm added both fields to QB/RB/WR/TE at weight 0.25 each.
- Retained the production top-80 pool, kernel, 12-season recency prior, donor
  cap, centered residual, and joint weekly path.
- Covered 1,620/1,620 rolling targets in both leagues.
- Covered 5,291/5,298 historical templates: 100% QB/RB, 99.76% TE, and 99.77%
  WR.

## Primary full-period comparison

Candidate minus production; negative is better except impact AUC.

| League | PPG CRPS | Contribution CRPS | Played CRPS | +3 Brier | Impact Brier | Impact AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DK | +0.000670 | +0.007114 | -0.002254 | +0.000383 | +0.000218 | -0.003298 |
| beta | -0.001102 | -0.018661 | -0.002838 | -0.000049 | -0.000299 | +0.006091 |

The beta point estimates are broadly favorable but small. Their full-period
season- and player-cluster intervals cross zero for every primary metric. DK
moves in the opposite direction on PPG, contribution, +3 surprise, and impact.
The primary arm wins only 4/9 DK seasons for PPG and contribution, versus 6/9
in beta.

## Recent-period comparison

For 2023-2025, the primary arm improves beta PPG CRPS by 0.002545 and DK by
0.001061. Beta's season-cluster PPG interval is below zero; the player-cluster
interval crosses zero. DK intervals cross zero. Beta impact Brier improves by
0.000914 with its player interval narrowly below zero, but DK impact Brier
worsens by 0.000129 and impact AUC worsens by 0.004771. DK played-games CRPS
also reverses slightly in the recent period.

These are three-season diagnostics, not a cross-league promotion result.

## Isolated fields

Height alone is the safest arm but does not establish a durable multi-outcome
gain:

- DK: PPG +0.000037, contribution -0.001551, played -0.000741, impact Brier
  -0.000192, and impact AUC +0.000071.
- beta: PPG -0.000902, contribution -0.010181, played -0.000993, impact Brier
  +0.000008, and impact AUC -0.000979.

Beta's height-only season intervals exclude zero for full-period PPG and
contribution, but both player intervals cross zero. DK intervals cross zero.
Weight alone and the stronger combined arm retain the same cross-league
instability. Removing QB does not repair the primary comparison.

## Pool behavior

The primary fields are active rather than cosmetic:

- mean baseline donor overlap is 91.2% DK and 91.3% beta;
- WR overlap is 87.6% DK and 87.9% beta;
- weighted height and weight profile distance each fall by roughly 0.047
  overall; and
- effective sample size increases slightly rather than collapsing.

The fields successfully find physically closer donors. That change simply
does not improve the joint outcomes reliably enough to earn production weight.

## Combine-data implication

The existing player master is not the limiting factor for basic size: it has
complete target coverage and near-complete donor coverage. A combine source
would reduce coverage for height/weight and would not resolve the DK/beta
outcome disagreement shown here.

Keep combine acquisition deferred unless a separate study is proposed for
combine-only athletic measurements with explicit missingness, non-attendance,
and 2021-era handling.
