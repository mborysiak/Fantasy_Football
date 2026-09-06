# Findings

## Verdict

Do not promote any arm directly from this single-origin replay.

The championship tie-break is the cleanest mechanism result: within a 0.25%
construction-mean guardrail, it improved paired holdout championship proxy by
1.11 percentage points (LCB80 +0.49), P(2+ q90 difference-makers) by 2.15
points, and actual 2025 managed score by 19.89 points. It changed only three of
eight baseline rosters and preserved expected-score non-inferiority by design.

The combined waiver/championship arm produced the largest numerical gains:

- paired churn-scored managed EV: +32.32 points (LCB80 +28.92);
- paired churn-scored championship proxy: +5.75 percentage points
  (LCB80 +5.05);
- paired P(2+ q90 difference-makers): +4.00 percentage points;
- actual 2025 churn-scored managed score: +69.81 points
  (LCB80 +35.04).

Those gains do not solve the motivating roster-composition problem. Dead-zone
RB count rose from 0.75 to 0.88 per roster. Aaron Jones selection rose from
37.5% to 50.0%, James Conner remained at 37.5%, and Isiah Pacheco was not
selected in either arm. The combined arm also shifted the average roster from
QB/RB/WR/TE `1/5/6/1` to `1/5.25/4.75/2`, with average position spend moving
from `$21/$97/$173/$6` to `$31/$119/$135/$14`. It did not create a simple
younger-RB or cheap-upside-bench portfolio.

The actual q90 difference-makers captured by selected rosters were Javonte
Williams, Travis Etienne, and Puka Nacua. Aaron Jones and James Conner were not
credited as difference-makers. This shows that the present tail event can
identify realized surprises, but a roster can still buy replaceable veteran
depth alongside them. The event is absolute contribution plus projection
residual, not the player's marginal value over an accessible waiver substitute.

## Interpretation

The best-available waiver proxy is doing more work than the championship
tie-break. Seven of eight combined rosters exactly match the waiver-only arm;
the championship rule changes one churn roster. The large combined gain is
therefore evidence that the current static RB/WR waiver centers are too low for
this 2025 replay, not evidence that the current championship proxy alone fixes
roster construction.

The current championship value is a within-study rank proxy. Reference rosters
share players and are not generated as eleven mutually exclusive auction
opponents, so it should not be interpreted as an absolute title probability.
The actual-season comparison is one realized year and has a negative LCB80 for
the combined championship delta even though its point estimate is positive.

## Recommended next test

Keep expected managed score as the primary calibration gate and retain the
0.25% lexicographic guardrail. Replace the absolute player tail tie-break with
a roster-marginal scarcity event:

1. In each shared weekly scenario, remove the player from the roster and fill
   his contribution with the same-position best-available waiver/churn policy.
2. Define a needle mover from marginal managed points over that replacement,
   with a strictly-prior position threshold and a playoff-week sensitivity.
3. Rank guarded rosters by paired championship LCB, then probability of at
   least two roster-marginal needle movers.
4. Validate across multiple rolling origins before considering production.

Do not add a hard youth bonus or age tax. Youth can remain an explanatory
diagnostic; it should win only if it predicts scarce marginal roster value.

## Reproducibility

The full run uses eight construction blocks, 32 construction contexts per
block, 25 unique candidates per block/waiver mode, and 128 independent
validation contexts per block. All substantive CSV/Markdown outputs matched
SHA-256 across a fixed-seed rerun. Seven focused study/replay tests pass.
