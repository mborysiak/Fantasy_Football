# Tiered Target Board Audit

## Outcome

Experimental App v18 successfully summarizes current unforced optimal
construction plans without changing the existing player-level Buy-versus-Pass
board. It groups complete plans into recurring position-spend/roster-shape
families, highlights the two most-supported Premium/Mid/Value acquisition
flows, and retains exact paths plus within-position salary/PPG buckets in
collapsed detail.

The added four structure solves are inexpensive:

- beta Brown/Tuten/Tyson/Coleman state: `0.112s`;
- NV Maye/Achane state: `0.102s`.

Each fresh board starts with up to four plans. Add Evidence appends four fresh
plans and reclusters the full accumulated pool; a localhost smoke verified the
transition from four to eight exact plans and updated family support from
`2/4` to `6/8`. Sequential Workers changes process scheduling only and does not
change the four plans generated per evidence batch.

## Behavioral examples

In NV, the open second-QB decision separates cleanly:

- Josh Allen is alone in a `$72`, `19.9` PPG bucket with `2/4` plan support;
- Jordan Love anchors a separate `$7-$13`, `13.3-14.7` PPG bucket with `2/4`
  support; Baker Mayfield, Malik Willis, and Sam Darnold appear as nearby
  alternatives.

This supports the intended workflow: target Allen when that unique bucket is
preferred; if he is marked out, the existing draft-state invalidation and full
rerun rebuild the plan from the remaining pool rather than relying on a stored
conditional branch.

In beta, the four plans distinguish a mostly supported `$4-$7` QB bucket from
a less common `$12-$18` bucket. They also separate the occasional elite-RB
construction from more common upper-, middle-, and lower-cost RB allocations.
The UI surfaces buckets present in at least half the plans and keeps one-block
structures in a collapsed section.

## Interpretation boundary

This is a descriptive summary of current optimizer constructions. It does not
estimate the causal value of missing an entire tier, and player appearance
frequency is not a replacement for the existing block-aware LCB80 action
evidence. Salary and projected PPG determine bucket proximity; managed value is
used only to order players after bucket membership is established.

Production App v15 remains unchanged while the v18 panel is reviewed in
`codex/tiered-target-board`.
