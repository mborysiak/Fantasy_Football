# V2 Identity and Scoring Revalidation

## Purpose

This study records the corrective replay after:

- governed player-identity consolidation for Tetairoa McMillan, Amon-Ra
  St. Brown, Equanimeous St. Brown, and other reviewed provider aliases;
- correction of two mislabeled FantasyPros WR snapshot seasons;
- removal of `last_season` as a hard identity endpoint;
- inclusion of beta QB sacks in standardized provider scoring; and
- explicit versioning of the provider-points estimand and imputation lineage.

The original 2026-07-29 locked and next-year study outputs are retained as
historical evidence, but their data-lineage claims are superseded by this
revalidation.

## Reproduction

Run the foundation and feature builds independently by league:

```powershell
python -m Scripts.V2.build_milestone_3 --league dk --output-db Data/Databases/Projection_V2.sqlite3
python -m Scripts.V2.build_milestone_3 --league beta --output-db Data/Databases/Projection_V2_beta.sqlite3
```

Then run, for each league database:

```powershell
python research/studies/2026-07-29_v2_locked_final_validation/run_validation.py --league <league> --output-db <database> --results-dir <scratch-results>
python research/studies/2026-07-29_v2_locked_final_validation/analyze_calibration.py --output-db <database> --results-dir <scratch-results>
python research/studies/2026-07-29_v2_locked_final_validation/audit_template_handoff.py --league <league> --output-db <database> --results-dir <scratch-results>
python research/studies/2026-07-29_v2_next_year_residual/run_validation.py --league <league> --output-db <database> --results-dir <scratch-next-results>
```

Run production handoff work against a copied `Simulation.sqlite3` first.
Backfill weekly canonical keys before publishing, verify a second publish is a
zero-delta no-op, and only then promote and synchronize the verified database.

## Durable outputs

- `results/findings.md`: accepted conclusions and remaining caveats.
- `results/run_metadata.json`: exact corrected build/model lineage.
- `results/validation_metrics.csv`: locked and following-season gates.
- `results/artifact_audit.csv`: corrected artifact counts and parity.
- `results/beta_qb_coverage.csv`: beta QB provider completeness by season.
- `results/scoring_component_audit.csv`: actual-label scoring coverage.
