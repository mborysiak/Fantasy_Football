# Shaheed, Coker, and Doubs Audit

## Scope

The audit reconstructs the displayed beta roster with Chase Brown `$34`,
Bhayshul Tuten `$11`, Jordyn Tyson `$7`, and Jonah Coleman `$1`. The other 12
active beta keepers are unavailable. It uses calculation v15, current beta
lineup and roster bounds, estimated waiver baselines, Top-12 on, selection
reserve off, and compute budget 320.

The screenshot does not expose its random-variation number or sidebar waiver
settings, and its Shaheed block range does not exactly match the default root.
The conclusions below therefore use a controlled same-settings comparison over
nine matched evidence roots rather than claiming to reproduce the screenshot's
exact `+6.1` LCB.

## Why the comp cards look better for Coker

The displayed positive and negative residual numbers are conditional averages;
they omit how often each side occurs. Production also subtracts each player's
raw weighted donor-pool residual mean before applying the selected donor's
weekly trajectory.

| Player | PPG | Price | Raw residual mean | P(positive residual) | Centered residual SD | Expected games |
|---|---:|---:|---:|---:|---:|---:|
| Rashid Shaheed | 7.76 | $2 | -0.87 | 43.2% | 3.40 | 12.3 |
| Jalen Coker | 7.93 | $2 | -0.47 | 38.1% | 2.90 | 12.9 |
| Romeo Doubs | 8.04 | $3 | -0.72 | 43.5% | 3.29 | 12.1 |

Coker is the safer profile, but the matched construction contexts give Shaheed
slightly more usable spike production:

| Player | Weeks over 6.9 waiver | 10+ weeks | 15+ weeks | Total points over waiver | Additive managed value |
|---|---:|---:|---:|---:|---:|
| Rashid Shaheed | 5.60 | 4.07 | 2.06 | 40.99 | 16.99 |
| Jalen Coker | 5.85 | 3.77 | 1.97 | 38.51 | 14.81 |
| Romeo Doubs | 5.41 | 3.87 | 2.19 | 41.20 | 19.18 |

Thus the model is not reading Coker's conditional `+2.6/-2.4` as an always-
better distribution. Coker has more availability and mild usable weeks;
Shaheed has a higher positive-donor probability and slightly more lineup-sized
spikes after pool centering. Doubs has the strongest isolated managed value but
costs one more dollar.

## Whole-action comparison

At the default matched root, giving all three the same confirmation-sized bank
produces:

| Player | Buy-Pass mean | LCB80 | Buy season mean | Pass season mean |
|---|---:|---:|---:|---:|
| Jalen Coker | +31.58 | +24.11 | 1640.62 | 1609.03 |
| Rashid Shaheed | +32.34 | +17.66 | 1644.16 | 1611.82 |
| Romeo Doubs | +9.75 | +2.71 | 1635.54 | 1625.79 |

Coker would have the best conservative action bound on this root if granted
equal confirmation. Across nine matched variations:

| Player | Avg Buy-Pass mean | Avg LCB80 | Positive LCB roots | Best LCB roots | Avg Buy-roster EV |
|---|---:|---:|---:|---:|---:|
| Rashid Shaheed | +20.71 | +10.68 | 9/9 | 6/9 | 1648.11 |
| Jalen Coker | +14.64 | +5.88 | 8/9 | 1/9 | 1639.05 |
| Romeo Doubs | +9.93 | +3.35 | 6/9 | 2/9 | 1646.02 |

There is a real average Shaheed tendency, but it is neither invariant nor a
large isolated-player edge over Doubs.

## Fixed-roster substitution

Holding the other 12 players in each Shaheed additive Buy plan constant removes
the whole-roster recourse effect:

| Substitute | Mean delta vs Shaheed | P10 delta vs Shaheed |
|---|---:|---:|
| Romeo Doubs | +2.17 | +0.64 |
| Rashid Shaheed | 0.00 | 0.00 |
| Jalen Coker | -1.70 | -1.34 |

Doubs is marginally better on the same roster, but the `$1` price difference
and alternate roster construction reverse the average action ordering. Coker's
safer-looking comp pool does not improve the held-fixed roster in this root.

## Structural interpretation

The Target Board ranks the complete action `Buy candidate now` versus `Pass and
reoptimize`, not the named player's isolated weekly profile. In all four
default construction blocks, Shaheed's Pass plan chooses neither Coker nor
Doubs. It instead changes one to three players across positions. Therefore a
large Shaheed Buy-Pass number can partly describe the value of the different
QB/RB/WR/TE portfolio unlocked by a `$2` WR, rather than a direct claim that
Shaheed is much better than every `$2-$3` WR.

This is intended whole-roster optimization, not a Shaheed data bug. The
presentation does have an evidence-boundary weakness: a confirmed Shaheed row
can be shown above discovery-only peers even though a peer such as Coker can win
when given the same confirmation bank. A future UI improvement would be an
explicit same-price peer confirmation or fixed-roster substitution view. No
production behavior changed in this audit.
