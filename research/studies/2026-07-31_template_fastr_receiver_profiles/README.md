# nflfastR receiver-profile template replay

## Question

Can causal prior-season receiver usage and role profiles improve weekly-template
matching beyond the production preseason projection, market, experience, and
team-room features?

The motivating failure case is Ladd McConkey matching closely to Terrelle
Pryor. Their preseason PPG and room standing can look similar even though their
underlying receiver roles differ.

## Frozen design

- Source: the weekly `WR_Stats`, `TE_Stats`, and `Team_Stats` tables in
  `Daily_Fantasy_Data/Databases/FastR_Beta.sqlite3`.
- A target/template season `t` receives only realized statistics from `t-1`.
- Missing prior-year history is neutral (`0.5`), not zero. This makes rookies
  and unresolved histories neutral in the added distance.
- Raw measures are ranked within source season and source position, then shrunk
  toward `0.5` using opportunity reliability.
- Target/air-yard shares use only weeks in which the player registered a
  target, avoiding a new player-specific injury/missed-game signal.
- Exact deep-target share is not available in the stored weekly aggregate.
  Air-yards share plus aDOT is the prespecified stored-data substitute.
- The production matcher, donor pool size, kernel, recency prior, and joint
  weekly outcome transport are otherwise unchanged.

### Raw profiles

| Profile | Definition | Reliability |
|---|---|---|
| Target share | player targets / team attempts in targeted weeks | `targets / (targets + 40)` |
| Air-yards share | player air yards / team air yards in targeted weeks | target reliability |
| aDOT | player air yards / player targets | target reliability |
| Red-zone target share | player red-zone targets / team red-zone attempts in targeted weeks | `player RZ targets / (player RZ targets + 8)` |
| Target-share IQR | weekly target-share P75 minus P25 in targeted weeks | target reliability x `weeks / (weeks + 8)` |
| High-use-week rate | share of targeted weeks with target share >= 20% | dispersion reliability |

### Matching arms

All primary bundles add exactly `1.0` total distance weight to WR and TE so
feature count does not mechanically increase the strength of a richer bundle.

| Method | Added weights |
|---|---|
| `production` | none |
| `usage_depth_w100` | target share 0.50; aDOT 0.50 |
| `usage_air_value_w100` | target share 0.30; air share 0.25; aDOT 0.20; red-zone share 0.25 |
| `usage_air_value_disp_w100` | target share 0.225; air share 0.20; aDOT 0.175; red-zone share 0.20; IQR 0.10; high-use rate 0.10 |
| `usage_air_value_disp_w150` | 1.5x sensitivity of the full bundle |

## Validation

- Corrected current-lineage DK and beta weekly data.
- Expanded rolling target cohorts: QB 48, RB 90, WR 120, TE 48 per season.
- Role-tiered policy from
  `research/studies/2026-07-31_template_role_tiered_validation/README.md`.
- Development seasons 2017-2022; untouched temporal check 2023-2025.
- A mechanical weekly finalist must still pass replacement-aware roster CRPS
  before production promotion.

## Reproduction

```powershell
.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_fastr_receiver_profiles/run_validation.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_fastr_receiver_profiles/run_validation.py --league beta
python research/studies/2026-07-31_template_fastr_receiver_profiles/run_role_tier_rescore.py
```

Production remains unchanged unless the full validation sequence passes.

## Results

The replay produced 2,647 rolling targets and 13,235 prediction rows per
league. Prior-profile coverage was 89-91% among core WR/TE targets. None of the
four candidate bundles improved development core PPG CRPS in both leagues, so
none passed the first role-tier screen and roster replay was correctly skipped.

- `usage_depth_w100` was the closest candidate, but mean cross-league
  development core PPG CRPS worsened by 0.0095%; its season-cluster intervals
  crossed zero in both leagues.
- The primary full bundle worsened mean development core PPG CRPS by 0.0205%.
  It improved both temporal point estimates, but this did not override the
  frozen development gate.
- WR core development PPG was inconsistent and drove the rejection. The full
  bundle worsened DK by 0.1208% and beta by 0.0172%.
- TE core PPG improved modestly in both development and temporal slices. This
  is exploratory position evidence, not a promoted TE-only matcher.

The features did distinguish the motivating players. In beta, Terrelle Pryor
moved from Ladd McConkey's third-ranked donor to ranks 7-10 depending on the
bundle, and his sampling weight fell from 2.23% to 1.65-1.83%. In DK, Pryor
remained rank 2 but his weight fell from 3.26% to 2.33-2.68%. The weighted
absolute projection-PPG gap increased slightly, illustrating the intended
trade-off, but the changed pools did not improve rolling predictive scores.

**Decision: retain production matching.** Prior-season nflfastR receiver
profiles are descriptively useful in the comp explorer, but this specification
does not justify using them as production sampling-distance criteria. A TE-only
usage/depth arm is the clearest follow-up hypothesis and must be labeled
post-hoc unless confirmed on genuinely new origins.
