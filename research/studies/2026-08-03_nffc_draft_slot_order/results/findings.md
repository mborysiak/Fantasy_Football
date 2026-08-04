# Findings

## Recommendation

Enter the current model-based NFFC draft-slot preference as:

`3, 4, 2, 5, 6, 7, 1, 8, 11, 9, 10, 12`

Slots 3 and 4 form the clearest top tier. Slot 3 has the highest held-out
expected score (`2355.16`) and a modeled 64.7% first-place probability; slot 4
is only `0.79` points lower with a paired 95% interval of `[-4.30, +2.72]`
relative to slot 3. Slot 2 is `4.35` points lower and its interval barely
includes a tie (`[-8.86, +0.15]`).

Slots 5 and 6 are essentially tied with each other (`0.19` points apart), but
their point estimates trail slots 3/4. Slot 1 ranks seventh and trails slot 3
by `12.62` points (`[-20.75, -4.49]`). Slots 8-12 are the weakest broad tier,
trailing slot 3 by about `19.9` to `28.5` points. The exact ordering within
that bottom tier is less important than preferring slots 2-7 over it.

## Results

| Preference | Slot | Held-out EV | Difference vs. slot 3 | Most common first pick |
|---:|---:|---:|---:|---|
| 1 | 3 | 2355.16 | 0.00 | Puka Nacua (91.8%) |
| 2 | 4 | 2354.37 | -0.79 | Puka Nacua (73.8%) |
| 3 | 2 | 2350.81 | -4.35 | Jahmyr Gibbs (52.3%) |
| 4 | 5 | 2348.68 | -6.47 | Amon-Ra St. Brown (71.1%) |
| 5 | 6 | 2348.50 | -6.66 | Amon-Ra St. Brown (87.9%) |
| 6 | 7 | 2344.17 | -10.98 | Amon-Ra St. Brown (70.7%) |
| 7 | 1 | 2342.54 | -12.62 | Jahmyr Gibbs (100.0%) |
| 8 | 8 | 2335.29 | -19.87 | Saquon Barkley (52.3%) |
| 9 | 11 | 2329.23 | -25.92 | Saquon Barkley (97.7%) |
| 10 | 9 | 2328.93 | -26.23 | Saquon Barkley (82.8%) |
| 11 | 10 | 2327.76 | -27.39 | Saquon Barkley (94.5%) |
| 12 | 12 | 2326.64 | -28.52 | Saquon Barkley (96.1%) |

The likely interpretation is that current NFFC player values do not make the
first overall pick strong enough to offset the harsher 3RR follow-up turns,
while slots 3-4 preserve elite access with a better balance of subsequent
turns. That interpretation is model-based rather than causal evidence.

## Gibbs-tier sensitivity

The current NFFC projection is `19.705` PPG for Jahmyr Gibbs versus `19.311`
for Puka Nacua, only a `0.394` PPG advantage. The ADP market treats Gibbs more
distinctly (`1.53` versus `4.14`) than the point model does. Across the first
three roster selections, the rollout averages `48.49` baseline PPG from slot 1
versus `48.77` from slot 3 because slots 3-4 recover value at their earlier
Round 2/3 turns. Slot 3 also builds a more WR-heavy roster on average (8.43 WR
and 5.57 RB versus 7.99 WR and 6.02 RB from slot 1).

A fixed-policy held-out sensitivity raises only Gibbs's active PPG center:

| Gibbs PPG bump | Gibbs modeled PPG | Slot 1 minus slot 3 | Slot 1 minus slot 4 |
|---:|---:|---:|---:|
| 0.00 | 19.705 | -12.62 | -11.83 |
| +0.75 | 20.455 | -3.64 | -2.12 |
| +1.00 | 20.705 | -0.63 | +1.13 |
| +1.25 | 20.955 | +2.38 | +4.38 |

Thus, a belief that Gibbs is roughly `1.4` PPG better than Puka moves slot 1
into an effective tie for best, while an edge around `1.6` PPG makes slot 1 the
leader in this sensitivity. This diagnostic holds the previously drafted
policy paths fixed; it establishes that the KDS conclusion is projection-tier
sensitive rather than proving a new exact ordering.

## Validity correction

An initial diagnostic was rejected before reporting because it forced the
user's first candidate before simulating the players drafted ahead of slots
2-12. That made top players incorrectly available at late positions and
artificially favored slots 11-12. The final `schema_version=2` study removes
exactly `slot - 1` opponents before the first user decision in every room and
uses the same room's opponent order thereafter. All stale diagnostic outputs
were overwritten.

## Evidence and scope

- 256 matched noisy-ADP rooms per slot; all 3,072 drafts completed legally.
- 512 common held-out 17-week score seasons per completed roster.
- Common construction, audit, and ADP-room random sources across slots.
- Raw best-ball points are the outcome; stack utility affects decisions but is
  not added to the outcome.
- Database SHA-256:
  `47658fab0a2a98a1714890e8c57d45dbfbce63dd62c5455fad4ccc15374065a2`.

This is a current-projection structural simulation, not historical NFFC ROI or
championship-win validation. It covers the app's 20-player offense-only NFFC
3RR adapter, uses composite NFFC ADP, and excludes K/DST and official 30-round
roster rules. Rerun the preference study after material ADP or projection
updates.
