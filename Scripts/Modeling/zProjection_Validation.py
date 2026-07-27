"""Causal historical roll-up for the production projection ensemble."""

import datetime as dt

import numpy as np
import pandas as pd


RESID_PERCENTILES = (5, 10, 25, 75, 90, 95)
RESID_COLS = [f'pred_resid_{p}' for p in RESID_PERCENTILES]
LOWER_RESID_COLS = ['pred_resid_5', 'pred_resid_10', 'pred_resid_25']
UPPER_RESID_COLS = ['pred_resid_75', 'pred_resid_90', 'pred_resid_95']
COMPONENT_NAMES = ('rush', 'pass', 'rec')
METHOD_VERSION = 'rolling_prior_season_empirical_v1'


def enforce_resid_order(df, cols=RESID_COLS):
    """Enforce nondecreasing residual quantiles without filling null rows."""
    output = df.copy()
    use_cols = [col for col in cols if col in output.columns]
    complete = output[use_cols].notna().all(axis=1) if use_cols else pd.Series(False)
    if complete.any():
        output.loc[complete, use_cols] = np.maximum.accumulate(
            output.loc[complete, use_cols].to_numpy(dtype=float),
            axis=1,
        )
    return output


def _prepare_validation_rows(validation_rows):
    required = {
        'player',
        'season',
        'pos',
        'rush_pass',
        'dataset',
        'current_or_next_year',
        'pred_fp_per_game',
        'y_act',
        'version',
        'year',
    }
    missing = sorted(required.difference(validation_rows.columns))
    if missing:
        raise ValueError(
            f'Projection validation roll-up is missing columns: {missing}'
        )

    rows = validation_rows.copy()
    rows = rows[~rows.dataset.fillna('').str.contains('Rookie', case=False)].copy()
    rows['season'] = pd.to_numeric(rows.season).astype(int)
    rows['year'] = pd.to_numeric(rows.year).astype(int)
    rows['outcome_horizon'] = rows.current_or_next_year.eq('next').astype(int)

    for col in RESID_COLS:
        if col not in rows.columns:
            rows[col] = np.nan

    if 'resid_target_season' not in rows.columns:
        rows['resid_target_season'] = rows.season + rows.outcome_horizon
    if 'resid_target_available' not in rows.columns:
        rows['resid_target_available'] = (
            rows.y_act.notna()
            & rows.resid_target_season.lt(rows.year)
        ).astype(int)

    for col in [
        'resid_training_rows',
        'resid_training_through_origin',
        'resid_training_through_season',
    ]:
        if col not in rows.columns:
            rows[col] = np.nan

    rows['_resid_row_available'] = rows[RESID_COLS].notna().all(axis=1)
    rows['_eligible_y_act'] = rows.y_act.where(
        rows.resid_target_available.eq(1)
    )
    rows['_training_rows_available'] = rows.resid_training_rows.where(
        rows._resid_row_available
    )
    rows['_training_origin_available'] = (
        rows.resid_training_through_origin.where(rows._resid_row_available)
    )
    rows['_training_season_available'] = (
        rows.resid_training_through_season.where(rows._resid_row_available)
    )
    return rows


def _estimate_component_rho(
    spec_rows,
    target_origin,
    positions,
    default_rho=0.35,
    min_samples=50,
):
    donor_rows = spec_rows[
        spec_rows.rush_pass.isin(COMPONENT_NAMES)
        & spec_rows.current_or_next_year.eq('current')
        & spec_rows.resid_target_available.eq(1)
        & spec_rows.resid_target_season.lt(int(target_origin))
    ].copy()

    records = []
    for pos in positions:
        pos_rows = donor_rows[donor_rows.pos.eq(pos)]
        rho = float(default_rho)
        samples = 0
        components = []
        training_through = np.nan

        if len(pos_rows):
            grouped = (
                pos_rows
                .groupby(
                    ['player', 'season', 'pos', 'rush_pass'],
                    as_index=False,
                )
                .agg(
                    pred_fp_per_game=('pred_fp_per_game', 'mean'),
                    y_act=('y_act', 'mean'),
                )
            )
            grouped['resid'] = grouped.y_act - grouped.pred_fp_per_game
            pivot = grouped.pivot_table(
                index=['player', 'season', 'pos'],
                columns='rush_pass',
                values='resid',
                aggfunc='mean',
            )
            components = [
                col for col in COMPONENT_NAMES
                if col in pivot.columns and pivot[col].notna().any()
            ]
            complete = pivot[components].dropna() if components else pivot.iloc[0:0]
            samples = int(len(complete))
            training_through = int(pos_rows.resid_target_season.max())

            if len(components) > 1 and samples >= int(min_samples):
                covariance = complete.cov()
                std = complete.std(ddof=1)
                covariance_sum = 0.0
                std_product_sum = 0.0
                for idx, col1 in enumerate(components):
                    for col2 in components[idx + 1:]:
                        covariance_sum += covariance.loc[col1, col2]
                        std_product_sum += std[col1] * std[col2]
                if std_product_sum > 0:
                    rho = float(np.clip(
                        covariance_sum / std_product_sum,
                        0,
                        0.95,
                    ))

        records.append({
            'pos': pos,
            'component_rho': rho,
            'component_rho_samples': samples,
            'component_rho_components': ','.join(components),
            'component_rho_training_through_season': training_through,
        })

    rho_df = pd.DataFrame(records)
    return rho_df.set_index('pos') if len(rho_df) else pd.DataFrame()


def _aggregate_family_rows(rows, source, horizon):
    keys = ['player', 'season', 'pos']
    if not len(rows):
        return pd.DataFrame()

    grouped = rows.groupby(keys, dropna=False)
    output = grouped[
        ['pred_fp_per_game', '_eligible_y_act', *RESID_COLS]
    ].mean().reset_index()
    output = output.rename(columns={'_eligible_y_act': 'family_y_act'})
    output['resid_source_rows'] = grouped.size().to_numpy()
    output['resid_source_rows_available'] = (
        grouped['_resid_row_available'].sum().to_numpy()
    )
    output['resid_training_rows_min'] = (
        grouped['_training_rows_available'].min().to_numpy()
    )
    output['resid_training_rows_max'] = (
        grouped['_training_rows_available'].max().to_numpy()
    )
    output['resid_training_through_origin'] = (
        grouped['_training_origin_available'].max().to_numpy()
    )
    output['resid_training_through_season'] = (
        grouped['_training_season_available'].max().to_numpy()
    )
    output['ensemble_source'] = source
    output['outcome_horizon'] = int(horizon)
    return output


def _combine_component_rows(component_rows, rho_df, default_rho=0.35):
    keys = ['player', 'season', 'pos']
    if not len(component_rows):
        return pd.DataFrame()

    grouped = component_rows.groupby(
        [*keys, 'rush_pass'],
        dropna=False,
    )
    components = grouped[
        ['pred_fp_per_game', '_eligible_y_act', *RESID_COLS]
    ].mean().reset_index()
    components = components.rename(columns={'_eligible_y_act': 'family_y_act'})
    components['resid_source_rows'] = grouped.size().to_numpy()
    components['resid_source_rows_available'] = (
        grouped['_resid_row_available'].sum().to_numpy()
    )
    components['resid_training_rows_min'] = (
        grouped['_training_rows_available'].min().to_numpy()
    )
    components['resid_training_rows_max'] = (
        grouped['_training_rows_available'].max().to_numpy()
    )
    components['resid_training_through_origin'] = (
        grouped['_training_origin_available'].max().to_numpy()
    )
    components['resid_training_through_season'] = (
        grouped['_training_season_available'].max().to_numpy()
    )

    records = []
    for (player, season, pos), group in components.groupby(keys, dropna=False):
        rho = (
            float(rho_df.loc[pos, 'component_rho'])
            if len(rho_df) and pos in rho_df.index
            else float(default_rho)
        )
        record = {
            'player': player,
            'season': int(season),
            'pos': pos,
            'pred_fp_per_game': group.pred_fp_per_game.sum(),
            'family_y_act': (
                group.family_y_act.sum()
                if group.family_y_act.notna().all()
                else np.nan
            ),
            'resid_source_rows': int(group.resid_source_rows.sum()),
            'resid_source_rows_available': int(
                group.resid_source_rows_available.sum()
            ),
            'resid_training_rows_min': group.resid_training_rows_min.min(),
            'resid_training_rows_max': group.resid_training_rows_max.max(),
            'resid_training_through_origin': (
                group.resid_training_through_origin.max()
            ),
            'resid_training_through_season': (
                group.resid_training_through_season.max()
            ),
            'ensemble_source': 'rush_pass_rec',
            'outcome_horizon': 0,
        }

        for col in RESID_COLS:
            values = group[col].dropna().to_numpy(dtype=float)
            if not len(values):
                record[col] = np.nan
                continue
            if len(values) == 1:
                record[col] = float(values[0])
                continue

            sign = -1 if col in LOWER_RESID_COLS else 1
            values = (
                np.minimum(values, 0)
                if sign < 0
                else np.maximum(values, 0)
            )
            magnitudes = np.abs(values)
            resid_var = (
                (1 - rho) * np.sum(magnitudes ** 2)
                + rho * np.sum(magnitudes) ** 2
            )
            record[col] = sign * np.sqrt(max(float(resid_var), 0))

        records.append(record)

    return enforce_resid_order(pd.DataFrame(records))


def _finalize_origin_ensemble(
    family_rows,
    rho_df,
    version,
    model_spec_asof_year,
):
    records = []
    for (player, season, pos), group in family_rows.groupby(
        ['player', 'season', 'pos'],
        dropna=False,
    ):
        current_actuals = group.loc[
            group.outcome_horizon.eq(0),
            'family_y_act',
        ].dropna()
        if len(current_actuals) and (
            current_actuals.max() - current_actuals.min() > 1e-6
        ):
            raise ValueError(
                f'Current-season targets disagree for {player}, {season}, {pos}.'
            )

        next_actuals = group.loc[
            group.outcome_horizon.eq(1),
            'family_y_act',
        ].dropna()
        all_family_actuals = group.family_y_act
        all_targets_available = bool(all_family_actuals.notna().all())
        source_rows = int(group.resid_source_rows.sum())
        available_source_rows = int(group.resid_source_rows_available.sum())
        source_coverage = (
            available_source_rows / source_rows if source_rows else 0.0
        )

        record = {
            'player': player,
            'season': int(season),
            'pos': pos,
            'y_act': current_actuals.mean() if len(current_actuals) else np.nan,
            'y_act_next': next_actuals.mean() if len(next_actuals) else np.nan,
            'y_act_ensemble_target': (
                all_family_actuals.mean() if all_targets_available else np.nan
            ),
            'ensemble_target_available': int(all_targets_available),
            'pred_fp_per_game': group.pred_fp_per_game.mean(),
            'ensemble_source_count': int(len(group)),
            'ensemble_sources': ','.join(sorted(group.ensemble_source.unique())),
            'resid_source_rows': source_rows,
            'resid_source_rows_available': available_source_rows,
            'resid_source_coverage': source_coverage,
            'resid_training_rows_min': group.resid_training_rows_min.min(),
            'resid_training_rows_max': group.resid_training_rows_max.max(),
            'resid_training_through_origin': (
                group.resid_training_through_origin.max()
            ),
            'resid_training_through_season': (
                group.resid_training_through_season.max()
            ),
            'version': version,
            'model_spec_asof_year': int(model_spec_asof_year),
            'data_oos': 1,
            'method_version': METHOD_VERSION,
            'date_modified': dt.datetime.now().strftime('%m-%d-%Y %H:%M'),
        }
        for col in RESID_COLS:
            record[col] = group[col].mean()

        if len(rho_df) and pos in rho_df.index:
            for col in [
                'component_rho',
                'component_rho_samples',
                'component_rho_components',
                'component_rho_training_through_season',
            ]:
                record[col] = rho_df.loc[pos, col]
        else:
            record.update({
                'component_rho': np.nan,
                'component_rho_samples': 0,
                'component_rho_components': '',
                'component_rho_training_through_season': np.nan,
            })
        records.append(record)

    output = enforce_resid_order(pd.DataFrame(records))
    if not len(output):
        return output

    has_quantiles = output[RESID_COLS].notna().all(axis=1)
    output['resid_calibration_available'] = has_quantiles.astype(int)
    output['resid_calibration_status'] = np.where(
        ~has_quantiles,
        'unavailable',
        np.where(
            output.resid_source_coverage.ge(1 - 1e-12),
            'complete',
            'partial',
        ),
    )
    return output


def build_final_validation_residuals(
    validation_rows,
    default_rho=0.35,
    min_component_samples=50,
):
    """Build production-style historical ensembles from base OOS rows.

    The final mean and residual columns mirror ``s3_Model_Ensemble.py``:
    component-current, all-current, and (for non-QBs) all-next families receive
    equal family weight. ``season`` remains the forecast origin. Current and
    next realized targets are stored separately because the next family targets
    ``season + 1``.
    """
    rows = _prepare_validation_rows(validation_rows)
    final_frames = []

    for (version, model_spec_asof_year), spec_rows in rows.groupby(
        ['version', 'year'],
        dropna=False,
    ):
        for target_origin, origin_rows in spec_rows.groupby('season'):
            positions = sorted(origin_rows.pos.dropna().unique())
            rho_df = _estimate_component_rho(
                spec_rows,
                target_origin=int(target_origin),
                positions=positions,
                default_rho=default_rho,
                min_samples=min_component_samples,
            )

            component_mask = origin_rows.rush_pass.isin(COMPONENT_NAMES)
            component_current = _combine_component_rows(
                origin_rows[
                    component_mask
                    & origin_rows.current_or_next_year.eq('current')
                ],
                rho_df,
                default_rho=default_rho,
            )
            all_current = _aggregate_family_rows(
                origin_rows[
                    ~component_mask
                    & origin_rows.current_or_next_year.eq('current')
                ],
                source='all_current',
                horizon=0,
            )
            all_next = _aggregate_family_rows(
                origin_rows[
                    ~component_mask
                    & origin_rows.current_or_next_year.eq('next')
                    & origin_rows.pos.ne('QB')
                ],
                source='all_next',
                horizon=1,
            )

            family_frames = [
                frame
                for frame in [component_current, all_current, all_next]
                if len(frame)
            ]
            if not family_frames:
                continue

            final_frames.append(_finalize_origin_ensemble(
                pd.concat(family_frames, ignore_index=True, sort=False),
                rho_df,
                version=version,
                model_spec_asof_year=model_spec_asof_year,
            ))

    if not final_frames:
        return pd.DataFrame()

    output = pd.concat(final_frames, ignore_index=True, sort=False)
    identity = ['version', 'model_spec_asof_year', 'season', 'player', 'pos']
    if output.duplicated(identity).any():
        duplicates = output.loc[output.duplicated(identity, keep=False), identity]
        raise ValueError(
            'Duplicate final validation ensemble rows remain: '
            f'{duplicates.head(10).to_dict("records")}'
        )

    calibrated = output.resid_calibration_available.eq(1)
    if calibrated.any():
        quantiles = output.loc[calibrated, RESID_COLS].to_numpy(dtype=float)
        if not np.isfinite(quantiles).all():
            raise ValueError('A final validation residual quantile is non-finite.')
        if not (np.diff(quantiles, axis=1) >= -1e-10).all():
            raise ValueError('Final validation residual quantiles are not monotone.')
        invalid_cutoff = (
            output.loc[calibrated, 'resid_training_through_season']
            >= output.loc[calibrated, 'season']
        )
        if invalid_cutoff.any():
            raise ValueError(
                'A final validation interval uses a contemporaneous/future outcome.'
            )

    return output.sort_values(identity).reset_index(drop=True)
