"""Backfill causal residual intervals and the final validation ensemble."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
MODELING_DIR = ROOT / 'Scripts' / 'Modeling'
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from s1_Stacking_Model import (  # noqa: E402
    RESID_ALPHAS,
    cross_fit_empirical_resid_quantiles,
)
from zProjection_Validation import (  # noqa: E402
    RESID_COLS,
    build_final_validation_residuals,
)


VALIDATION_DB = ROOT / 'Data' / 'Databases' / 'Validations.sqlite3'
BASE_TABLE = 'Model_Validations_Resid'
FINAL_TABLE = 'Final_Validations_Resid'
SLICE_COLS = [
    'version',
    'year',
    'pos',
    'rush_pass',
    'dataset',
    'filter_data',
    'year_exp',
    'current_or_next_year',
]
BASE_IDENTITY = [
    *SLICE_COLS,
    'season',
    'player',
]
BACKFILL_COLUMNS = {
    **{col: 'REAL' for col in RESID_COLS},
    'resid_training_rows': 'INTEGER',
    'resid_training_through_origin': 'INTEGER',
    'resid_training_through_season': 'INTEGER',
    'resid_calibration_mode': 'TEXT',
    'resid_calibration_available': 'INTEGER',
    'resid_target_season': 'INTEGER',
    'resid_target_available': 'INTEGER',
    'resid_calibration_date_modified': 'TEXT',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Write the calibrated rows and final table to Validations.sqlite3.',
    )
    parser.add_argument('--bootstrap-iters', type=int, default=50)
    parser.add_argument('--min-training-rows', type=int, default=30)
    return parser.parse_args()


def load_base_rows():
    with sqlite3.connect(VALIDATION_DB) as connection:
        return pd.read_sql_query(
            f'SELECT rowid AS _rowid, * FROM {BASE_TABLE}',
            connection,
        )


def calibrate_slices(rows, bootstrap_iters, min_training_rows):
    calibrated_frames = []
    audit_frames = []

    for slice_key, slice_rows in rows.groupby(
        SLICE_COLS,
        dropna=False,
        sort=True,
    ):
        slice_meta = dict(zip(SLICE_COLS, slice_key))
        outcome_horizon = int(
            slice_meta['current_or_next_year'] == 'next'
        )
        calibrated, audit = cross_fit_empirical_resid_quantiles(
            slice_rows,
            origin_col='season',
            outcome_horizon=outcome_horizon,
            as_of_year=int(slice_meta['year']),
            alphas=RESID_ALPHAS,
            min_training_rows=min_training_rows,
            n_bins=None,
            min_n=50,
            smooth=True,
            bootstrap_iters=bootstrap_iters,
            bootstrap_replace=True,
        )
        calibrated_frames.append(calibrated)
        for col, value in slice_meta.items():
            audit[col] = value
        audit_frames.append(audit)

    calibrated = pd.concat(calibrated_frames, ignore_index=True, sort=False)
    audit = pd.concat(audit_frames, ignore_index=True, sort=False)
    calibrated['resid_calibration_date_modified'] = (
        dt.datetime.now().strftime('%m-%d-%Y %H:%M')
    )
    return calibrated, audit


def reconstruct_point_ensemble(rows):
    """Independent reconstruction of the pre-existing historical point logic."""
    validation = rows[
        ~rows.dataset.fillna('').str.contains('Rookie', case=False)
    ].copy()
    keys = ['version', 'year', 'player', 'season', 'pos']
    component_mask = validation.rush_pass.isin(['rush', 'pass', 'rec'])

    component = validation[
        component_mask
        & validation.current_or_next_year.eq('current')
    ]
    component = (
        component
        .groupby([*keys, 'rush_pass'], as_index=False)
        .pred_fp_per_game.mean()
        .groupby(keys, as_index=False)
        .pred_fp_per_game.sum()
    )
    component['source'] = 'rush_pass_rec'

    current = (
        validation[
            ~component_mask
            & validation.current_or_next_year.eq('current')
        ]
        .groupby(keys, as_index=False)
        .pred_fp_per_game.mean()
    )
    current['source'] = 'all_current'

    next_year = (
        validation[
            ~component_mask
            & validation.current_or_next_year.eq('next')
            & validation.pos.ne('QB')
        ]
        .groupby(keys, as_index=False)
        .pred_fp_per_game.mean()
    )
    next_year['source'] = 'all_next'

    output = (
        pd.concat([component, current, next_year], ignore_index=True)
        .groupby(keys, as_index=False)
        .pred_fp_per_game.mean()
        .rename(columns={
            'year': 'model_spec_asof_year',
            'pred_fp_per_game': 'expected_pred_fp_per_game',
        })
    )
    return output


def audit_outputs(original, calibrated, slice_audit, final_rows):
    if len(original) != len(calibrated):
        raise ValueError('The base validation row count changed during calibration.')
    if calibrated._rowid.duplicated().any():
        raise ValueError('A SQLite rowid was duplicated during calibration.')
    if calibrated.duplicated(BASE_IDENTITY).any():
        raise ValueError('Duplicate base validation identities are present.')

    available = calibrated.resid_calibration_available.eq(1)
    quantiles = calibrated.loc[available, RESID_COLS].to_numpy(dtype=float)
    if len(quantiles):
        if not np.isfinite(quantiles).all():
            raise ValueError('A calibrated base quantile is non-finite.')
        if not (np.diff(quantiles, axis=1) >= -1e-10).all():
            raise ValueError('Calibrated base quantiles are not monotone.')
    if (
        calibrated.loc[available, 'resid_training_through_season']
        >= calibrated.loc[available, 'season']
    ).any():
        raise ValueError('A base interval used a non-prior realized outcome.')

    final_identity = [
        'version',
        'model_spec_asof_year',
        'season',
        'player',
        'pos',
    ]
    if final_rows.duplicated(final_identity).any():
        raise ValueError('Duplicate final validation identities are present.')

    independent = reconstruct_point_ensemble(original)
    comparison = final_rows.merge(
        independent,
        on=final_identity,
        how='outer',
        indicator=True,
    )
    if not comparison._merge.eq('both').all():
        raise ValueError('Final point ensemble keys differ from reconstruction.')
    comparison['point_abs_diff'] = (
        comparison.pred_fp_per_game
        - comparison.expected_pred_fp_per_game
    ).abs()
    if comparison.point_abs_diff.max() > 1e-10:
        raise ValueError('Final point means differ from reconstruction.')

    coverage = (
        final_rows
        .groupby(
            ['version', 'model_spec_asof_year', 'season'],
            as_index=False,
        )
        .agg(
            rows=('player', 'size'),
            current_targets=('y_act', 'count'),
            next_targets=('y_act_next', 'count'),
            ensemble_targets=('y_act_ensemble_target', 'count'),
            residual_rows=('resid_calibration_available', 'sum'),
            mean_resid_source_coverage=('resid_source_coverage', 'mean'),
        )
    )
    coverage['residual_row_rate'] = (
        coverage.residual_rows / coverage.rows
    )

    status = (
        final_rows
        .groupby(
            [
                'version',
                'model_spec_asof_year',
                'season',
                'resid_calibration_status',
            ],
            as_index=False,
        )
        .size()
        .rename(columns={'size': 'rows'})
    )

    evaluable = final_rows.dropna(
        subset=['y_act_ensemble_target', *RESID_COLS]
    ).copy()
    evaluable['ensemble_resid'] = (
        evaluable.y_act_ensemble_target - evaluable.pred_fp_per_game
    )
    interval_coverage_records = []
    for (version, spec_year, season), group in evaluable.groupby(
        ['version', 'model_spec_asof_year', 'season']
    ):
        record = {
            'version': version,
            'model_spec_asof_year': spec_year,
            'season': season,
            'rows': len(group),
        }
        for percentile, col in zip(
            [5, 10, 25, 75, 90, 95],
            RESID_COLS,
        ):
            record[f'empirical_cdf_at_{percentile}'] = (
                group.ensemble_resid.le(group[col]).mean()
            )
        interval_coverage_records.append(record)
    interval_coverage = pd.DataFrame(interval_coverage_records)

    return {
        'slice_audit': slice_audit,
        'point_comparison': comparison,
        'season_coverage': coverage,
        'status_coverage': status,
        'interval_coverage': interval_coverage,
    }


def sql_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def backup_database():
    backup_dir = STUDY_DIR / 'artifacts' / 'local'
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = (
        backup_dir
        / f'Validations_pre_projection_resid_backfill_{timestamp}.sqlite3'
    )
    shutil.copy2(VALIDATION_DB, backup_path)
    return backup_path


def persist(calibrated, final_rows):
    backup_path = backup_database()
    update_cols = list(BACKFILL_COLUMNS)
    update_sql = f'''
        UPDATE {BASE_TABLE}
        SET {", ".join(f"{col}=?" for col in update_cols)}
        WHERE rowid=?
    '''
    update_records = [
        tuple(sql_value(row[col]) for col in update_cols)
        + (int(row['_rowid']),)
        for _, row in calibrated.iterrows()
    ]

    with sqlite3.connect(VALIDATION_DB) as connection:
        existing_columns = {
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info({BASE_TABLE})'
            ).fetchall()
        }
        for col, sql_type in BACKFILL_COLUMNS.items():
            if col not in existing_columns:
                connection.execute(
                    f'ALTER TABLE {BASE_TABLE} ADD COLUMN {col} {sql_type}'
                )

        connection.executemany(update_sql, update_records)
        temp_table = f'{FINAL_TABLE}__backfill'
        final_rows.to_sql(temp_table, connection, if_exists='replace', index=False)
        connection.execute(f'DROP TABLE IF EXISTS {FINAL_TABLE}')
        connection.execute(f'ALTER TABLE {temp_table} RENAME TO {FINAL_TABLE}')
        connection.execute(f'''
            CREATE UNIQUE INDEX idx_final_validations_resid_identity
            ON {FINAL_TABLE}
               (version, model_spec_asof_year, season, player, pos)
        ''')
        connection.commit()

    return backup_path


def write_results(audits, calibrated, final_rows, backup_path, applied):
    results_dir = STUDY_DIR / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in audits.items():
        frame.to_csv(results_dir / f'{name}.csv', index=False)

    schema = {
        BASE_TABLE: [
            {'name': col, 'dtype': str(dtype)}
            for col, dtype in calibrated.dtypes.items()
            if col != '_rowid'
        ],
        FINAL_TABLE: [
            {'name': col, 'dtype': str(dtype)}
            for col, dtype in final_rows.dtypes.items()
        ],
    }
    (results_dir / 'schema.json').write_text(
        json.dumps(schema, indent=2),
        encoding='utf-8',
    )

    coverage = audits['season_coverage']
    summary = [
        '# Backfill Summary',
        '',
        f'- Applied to database: `{applied}`',
        f'- Base validation rows: `{len(calibrated):,}`',
        f'- Final validation rows: `{len(final_rows):,}`',
        f'- Model slices: `{len(audits["slice_audit"][SLICE_COLS].drop_duplicates()):,}`',
        (
            '- Final rows with calibrated residuals: '
            f'`{int(final_rows.resid_calibration_available.sum()):,}` '
            f'({final_rows.resid_calibration_available.mean():.1%})'
        ),
        (
            '- Earliest/latest forecast origins: '
            f'`{int(final_rows.season.min())}` / `{int(final_rows.season.max())}`'
        ),
        (
            '- Maximum independent point-ensemble difference: '
            f'`{audits["point_comparison"].point_abs_diff.max():.3g}`'
        ),
        f'- Backup: `{backup_path if backup_path else "not created (dry run)"}`',
        '',
        'The first forecast origins intentionally retain unavailable intervals '
        'when fewer than 30 strictly prior realized residual rows exist. '
        '`next` rows use an additional horizon embargo and their terminal raw '
        'targets are flagged unavailable rather than treated as realized data.',
        '',
        '## Season coverage',
        '',
        '```text',
        coverage.to_string(index=False),
        '```',
        '',
    ]
    (results_dir / 'summary.md').write_text(
        '\n'.join(summary),
        encoding='utf-8',
    )


def main():
    args = parse_args()
    original = load_base_rows()
    calibrated, slice_audit = calibrate_slices(
        original,
        bootstrap_iters=args.bootstrap_iters,
        min_training_rows=args.min_training_rows,
    )
    final_rows = build_final_validation_residuals(calibrated)
    audits = audit_outputs(
        original,
        calibrated,
        slice_audit,
        final_rows,
    )

    backup_path = None
    if args.apply:
        backup_path = persist(calibrated, final_rows)
    write_results(
        audits,
        calibrated,
        final_rows,
        backup_path=backup_path,
        applied=args.apply,
    )

    print(f'Base rows: {len(calibrated):,}')
    print(f'Final rows: {len(final_rows):,}')
    print(
        'Calibrated final rows: '
        f'{int(final_rows.resid_calibration_available.sum()):,} '
        f'({final_rows.resid_calibration_available.mean():.1%})'
    )
    print(f'Applied: {args.apply}')
    if backup_path:
        print(f'Backup: {backup_path}')


if __name__ == '__main__':
    main()
