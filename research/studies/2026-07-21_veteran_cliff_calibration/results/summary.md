# Veteran Cliff Calibration Results

## Coverage

- Current historical player-seasons: 4,206.
- Following-season-observable rows: 3,945.
- Experience source: draft=3,538, debut_fallback=668.

## Unadjusted historical rates

### RB

- `current_ppg_cliff_30`: 20.3% (211/1039) at/below threshold vs 22.8% (21/92) above.
- `extended_absence_8`: 20.4% (267/1306) at/below threshold vs 15.6% (17/109) above.
- `current_any_bust`: 36.6% (478/1306) at/below threshold vs 34.9% (38/109) above.
- `next_no_appearance`: 14.7% (181/1229) at/below threshold vs 30.7% (31/101) above.
- `next_no_useful`: 43.6% (536/1229) at/below threshold vs 58.4% (59/101) above.

### WR

- `current_ppg_cliff_30`: 17.6% (277/1576) at/below threshold vs 25.3% (23/91) above.
- `extended_absence_8`: 17.3% (330/1906) at/below threshold vs 16.5% (18/109) above.
- `current_any_bust`: 31.8% (607/1906) at/below threshold vs 37.6% (41/109) above.
- `next_no_appearance`: 12.8% (228/1787) at/below threshold vs 33.7% (34/101) above.
- `next_no_useful`: 34.8% (622/1787) at/below threshold vs 62.4% (63/101) above.

### TE

- `current_ppg_cliff_30`: 21.3% (126/591) at/below threshold vs 19.0% (12/63) above.
- `extended_absence_8`: 15.9% (112/703) at/below threshold vs 13.7% (10/73) above.
- `current_any_bust`: 33.9% (238/703) at/below threshold vs 30.1% (22/73) above.
- `next_no_appearance`: 9.4% (62/660) at/below threshold vs 26.9% (18/67) above.
- `next_no_useful`: 32.0% (211/660) at/below threshold vs 43.3% (29/67) above.

## Draft-relevant starter/elite rates

This restriction is closer to the veteran targets under discussion and reduces attrition from fringe preseason players.

- RB `current_any_bust`: 28.5% (205/720) vs 31.0% (18/58).
- RB `next_no_appearance`: 8.1% (55/682) vs 22.6% (12/53).
- RB `next_no_useful`: 38.4% (262/682) vs 50.9% (27/53).

- WR `current_any_bust`: 24.7% (256/1037) vs 36.1% (26/72).
- WR `next_no_appearance`: 7.2% (70/978) vs 25.4% (17/67).
- WR `next_no_useful`: 27.7% (271/978) vs 55.2% (37/67).

- TE `current_any_bust`: 29.3% (110/375) vs 22.2% (10/45).
- TE `next_no_appearance`: 5.4% (19/353) vs 21.4% (9/42).
- TE `next_no_useful`: 26.6% (94/353) vs 33.3% (14/42).

## Per-year adjusted estimates above the primary threshold

Piecewise logistic models control for preseason PPG and origin season; standard errors are clustered by player.

- RB `current_ppg_cliff_30`: OR 1.04 (95% CI 0.82-1.31), average +0.64 percentage points per excess year.
- RB `extended_absence_8`: OR 1.08 (95% CI 0.84-1.39), average +1.07 percentage points per excess year.
- RB `current_any_bust`: OR 1.06 (95% CI 0.85-1.32), average +1.23 percentage points per excess year.
- RB `next_no_appearance`: OR 1.09 (95% CI 0.93-1.27), average +1.62 percentage points per excess year.
- RB `next_no_useful`: OR 1.05 (95% CI 0.86-1.28), average +1.10 percentage points per excess year.
- WR `current_ppg_cliff_30`: OR 1.33 (95% CI 1.08-1.64), average +5.07 percentage points per excess year.
- WR `extended_absence_8`: OR 1.02 (95% CI 0.83-1.26), average +0.25 percentage points per excess year.
- WR `current_any_bust`: OR 1.22 (95% CI 0.99-1.50), average +4.30 percentage points per excess year.
- WR `next_no_appearance`: OR 1.29 (95% CI 1.11-1.51), average +5.07 percentage points per excess year.
- WR `next_no_useful`: OR 1.38 (95% CI 1.07-1.77), average +6.54 percentage points per excess year.
- TE `current_ppg_cliff_30`: OR 1.00 (95% CI 0.81-1.23), average -0.06 percentage points per excess year.
- TE `extended_absence_8`: OR 0.95 (95% CI 0.76-1.18), average -0.58 percentage points per excess year.
- TE `current_any_bust`: OR 0.97 (95% CI 0.85-1.11), average -0.63 percentage points per excess year.
- TE `next_no_appearance`: OR 1.26 (95% CI 1.08-1.47), average +4.27 percentage points per excess year.
- TE `next_no_useful`: OR 1.10 (95% CI 0.94-1.29), average +2.28 percentage points per excess year.

## Threshold jump versus further veteran years

The two-part model estimates a one-time step after crossing the primary threshold separately from the slope for every later year. These are adjusted average risk differences; small veteran samples make several estimates noisy.

- RB current any-bust: threshold step +2.0 pp (p=0.767); every further year +1.0 pp (p=0.796).
- RB following-season disappearance: threshold step +1.1 pp (p=0.876); every further year +1.7 pp (p=0.441).
- RB no useful following season: threshold step -1.6 pp (p=0.834); every further year +2.1 pp (p=0.617).
- WR current any-bust: threshold step +11.6 pp (p=0.082); every further year +1.1 pp (p=0.785).
- WR following-season disappearance: threshold step +1.1 pp (p=0.851); every further year +6.3 pp (p=0.018).
- WR no useful following season: threshold step +7.8 pp (p=0.318); every further year +6.5 pp (p=0.140).
- TE current any-bust: threshold step -8.1 pp (p=0.297); every further year +1.4 pp (p=0.521).
- TE following-season disappearance: threshold step -3.3 pp (p=0.653); every further year +6.0 pp (p=0.010).
- TE no useful following season: threshold step -13.2 pp (p=0.174); every further year +6.4 pp (p=0.054).

## Next-year target censoring

- RB above_threshold: true no-useful-next rate 56.8% vs recorded-target rate 35.1% (+21.6 pp understatement); 12/14 no-appearance rows equal the prior current PPG within 0.10.
- RB threshold_or_below: true no-useful-next rate 35.5% vs recorded-target rate 32.8% (+2.7 pp understatement); 59/63 no-appearance rows equal the prior current PPG within 0.10.
- TE above_threshold: true no-useful-next rate 51.5% vs recorded-target rate 27.3% (+24.2 pp understatement); 10/11 no-appearance rows equal the prior current PPG within 0.10.
- TE threshold_or_below: true no-useful-next rate 33.1% vs recorded-target rate 26.2% (+6.9 pp understatement); 29/30 no-appearance rows equal the prior current PPG within 0.10.
- WR above_threshold: true no-useful-next rate 55.0% vs recorded-target rate 40.0% (+15.0 pp understatement); 12/13 no-appearance rows equal the prior current PPG within 0.10.
- WR threshold_or_below: true no-useful-next rate 26.2% vs recorded-target rate 25.7% (+0.5 pp understatement); 80/85 no-appearance rows equal the prior current PPG within 0.10.

## Current 2026 template pools

These are raw matched-template diagnostics, not exact simulated cliff probabilities because the app centers and rescales active-PPG residuals.

- Alvin Kamara: 11.74/16 weighted played weeks, 15.8% extended-absence template probability, weighted raw template experience 7.4.
- Derrick Henry: 13.30/16 weighted played weeks, 8.7% extended-absence template probability, weighted raw template experience 7.1.
- George Kittle: 13.08/16 weighted played weeks, 7.0% extended-absence template probability, weighted raw template experience 8.7.
- Travis Kelce: 13.11/16 weighted played weeks, 7.2% extended-absence template probability, weighted raw template experience 9.3.

## Interpretation

- The point and template pipelines already contain some average aging, so an additional deterministic PPG haircut would double count part of the effect.
- The next-year target audit measures a distinct missing-outcome problem: retirement/disappearance can be recorded as repeated prior PPG. This should be fixed before treating next-year residuals as calibrated cliff risk.
- Any production veteran adjustment should target the incremental extended-absence/no-useful-season probability that remains after current PPG and template matching, retain raw and adjusted scores separately, and use uncapped experience.
- The estimates do not support one broad current-season tax. WR cliff risk is the clearest current-season candidate; RB and TE current-season taxes would be preference overlays rather than well-estimated corrections.
- Keeper scoring should first model the probability of no following season as an explicit zero-value mixture. Only after rebuilding that target should a residual soft veteran adjustment be calibrated.
