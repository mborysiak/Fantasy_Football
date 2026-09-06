# Paired 2025 Auction Objective Results

Eight blocks use current waiver estimates `{'QB': 16.2, 'RB': 7.2, 'WR': 6.5, 'TE': 7.5}` and the churn proxy `{'QB': 16.2, 'RB': 9.0, 'WR': 9.0, 'TE': 7.5}`. All selection evidence uses donors through 2024; actual 2025 weekly results are holdout only.

## Validation summary

| Arm | Managed EV, churn | Championship proxy, churn | P(2+ difference-makers) | Dead-zone RBs | Rookie RBs | Actual 2025 score, churn | Actual difference-makers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1627.11 | 6.682% | 18.95% | 0.75 | 1.00 | 1448.26 | 0.25 |
| Waiver proxy | 1655.85 | 11.807% | 22.75% | 0.88 | 0.88 | 1514.00 | 0.50 |
| Championship tie-break | 1630.40 | 7.792% | 21.09% | 1.00 | 1.38 | 1468.15 | 0.38 |
| Combined | 1659.43 | 12.428% | 22.95% | 0.88 | 1.00 | 1518.07 | 0.50 |

## Paired deltas versus baseline

| Arm | EV delta | EV LCB80 | Championship delta | Championship LCB80 | P(2+) delta | Dead-zone RB delta | Actual score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | +0.00 | +0.00 | +0.000% | +0.000% | +0.00% | +0.00 | +0.00 |
| Waiver proxy | +28.74 | +24.15 | +5.125% | +4.426% | +3.81% | +0.12 | +65.74 |
| Championship tie-break | +3.29 | +1.32 | +1.111% | +0.491% | +2.15% | +0.25 | +19.89 |
| Combined | +32.32 | +28.92 | +5.746% | +5.045% | +4.00% | +0.12 | +69.81 |

## Player movement

The most frequently selected RBs by arm are:

- **Baseline:** Tyrone Tracy Jr. (100%), Kaleb Johnson (62%), Aaron Jones (38%), James Conner (38%), Joe Mixon (38%), Rhamondre Stevenson (38%), Josh Jacobs (25%), Tyjae Spears (25%), Bhayshul Tuten (12%), Bijan Robinson (12%)
- **Waiver proxy:** Aaron Jones (50%), Josh Jacobs (50%), James Conner (38%), Kaleb Johnson (38%), Rhamondre Stevenson (38%), Trey Benson (38%), Tyrone Tracy Jr. (38%), Bhayshul Tuten (25%), Bijan Robinson (25%), Breece Hall (25%)
- **Championship tie-break:** Kaleb Johnson (62%), Tyrone Tracy Jr. (62%), Aaron Jones (50%), James Conner (50%), Joe Mixon (38%), Bhayshul Tuten (25%), Javonte Williams (25%), Josh Jacobs (25%), Rhamondre Stevenson (25%), Bijan Robinson (12%)
- **Combined:** Aaron Jones (50%), Josh Jacobs (50%), James Conner (38%), Kaleb Johnson (38%), Rhamondre Stevenson (38%), Trey Benson (38%), Bhayshul Tuten (25%), Bijan Robinson (25%), Breece Hall (25%), Dylan Sampson (25%)

## Interpretation guardrails

- The championship value is a common-bank relative proxy, not an absolute calibrated league-win probability.
- The churn arm is a frozen best-available PPG sensitivity, not a complete transaction, learning, or waiver-competition model.
- A single 2025 realized season can diagnose behavior but cannot justify production promotion by itself.
- The q90 difference-maker event is position- and history-aware; it does not directly reward youth.
