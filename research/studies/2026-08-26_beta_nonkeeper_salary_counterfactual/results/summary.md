# 2026 Beta Non-Keeper Salary Counterfactual

The governed salary model was rebuilt with Chase Brown, Bhayshul Tuten, Luther Burden III, and Colston Loveland all on the open market.

The active keeper pool falls from 14 keepers spending $441 to 12 keepers spending $396. The modeled open market therefore has 144 slots and $3180.

## Candidate salaries

| Player | Pos | Pred PPG | Current | Non-keeper model | Change | P10-P90 | Min-Max | ESPN source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chase Brown | RB | 15.2 | $34.0 | $72.2 | +38.2 | $63.2-$87.3 | $59.6-$90.6 | $61 |
| Bhayshul Tuten | RB | 11.2 | $11.0 | $30.3 | +19.3 | $24.2-$41.8 | $22.4-$44.8 | $16 |
| Luther Burden III | WR | 10.8 | $25.4 | $25.3 | -0.1 | $17.6-$33.0 | $16.1-$36.8 | missing |
| Colston Loveland | TE | 10.4 | $27.3 | $27.0 | -0.3 | $21.5-$32.3 | $19.5-$33.8 | $23 |

`P10-P90` is the modeled salary center plus the stored 10th/90th residual quantiles, floored at $1. `Min-Max` is the app's legacy uncertainty range.

## Largest whole-market center moves

| Player | Current | Counterfactual | Change |
|---|---:|---:|---:|
| Chase Brown | $34.0 | $72.2 | +38.2 |
| Bhayshul Tuten | $11.0 | $30.3 | +19.3 |
| David Montgomery | $33.6 | $34.3 | +0.6 |
| Kyren Williams | $52.4 | $52.9 | +0.6 |
| D'Andre Swift | $33.3 | $33.6 | +0.2 |
| Josh Jacobs | $56.1 | $56.3 | +0.2 |
| Ladd McConkey | $34.1 | $34.3 | +0.1 |
| DeVonta Smith | $40.7 | $40.8 | +0.1 |
| De'Von Achane | $86.2 | $85.3 | -0.9 |
| Aaron Jones | $7.2 | $6.3 | -0.9 |
| Brock Bowers | $51.1 | $50.1 | -1.0 |
| Jonathan Taylor | $96.2 | $95.1 | -1.1 |
| Bijan Robinson | $106.0 | $104.8 | -1.2 |
| Amon-Ra St. Brown | $82.1 | $80.9 | -1.2 |
| Trey McBride | $53.0 | $51.3 | -1.7 |
| Jahmyr Gibbs | $109.9 | $107.7 | -2.1 |

## Validation

- Projection/salary key parity: `True`.
- Top-144 non-keeper salary total: $3180.000000 versus $3180.000000 available.
- Live input hashes unchanged after the run: `True`.
- Salary method: `current_locked_spec_v6_v2_population_11f`.
- Production keeper CSV and all production/app databases were left unchanged.
