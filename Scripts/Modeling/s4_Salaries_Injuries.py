
#%%

# # Reading in Old Salary Data

import pandas as pd
import numpy as np
import json
import os
import sqlite3
from pathlib import Path

# Staged/automated rebuilds must not pause on notebook diagnostic figures.
# Set this before importing zModel_Functions, which imports pyplot.
if os.getenv('FF_MODEL_DATABASE_DIR'):
    os.environ.setdefault('MPLBACKEND', 'Agg')

import zModel_Functions as mf
import joblib
from sklearn.base import clone
from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc
from skmodel import SciKitModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from Scripts.V2.template_identity import attach_v2_player_keys
from Scripts.V2.production_handoff import (
    load_identity_frames,
    resolve_source_player_keys,
)
from Scripts.Modeling.salary_source_parser import (
    governed_salary_fallback_null_team_keys,
    governed_salary_source_specs,
    parse_espn_salary_records,
    repair_governed_salary_slices,
    validate_salary_records,
    validate_v2_salary_fallback_context,
)
from Scripts.config import YEAR

from sklearn.preprocessing import StandardScaler
from zFix_Standard_Dev import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

#==========
# General Setting
#==========

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = os.getenv(
    'FF_MODEL_DATABASE_DIR',
    f'{root_path}/Data/Databases/',
)
dm = DataManage(db_path)

# set core path
PATH = f'{root_path}/Data/'
LEAGUE = 'beta'
NUM_TEAMS = 12
TEAM_BUDGET = 298
TEAM_ROSTER_SIZE = 13
LEAGUE_BUDGET = NUM_TEAMS * TEAM_BUDGET
LEAGUE_ROSTER_SLOTS = NUM_TEAMS * TEAM_ROSTER_SIZE

SALARY_RESID_ALPHAS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
SALARY_RESID_COLS = [f'salary_resid_{int(round(alpha * 100))}' for alpha in SALARY_RESID_ALPHAS]
SALARY_RESID_SCHEMA = {col: 'REAL' for col in SALARY_RESID_COLS}
SALARY_VALIDATION_TABLE = 'Salary_Validations_Resid'
SALARY_BACKTEST_TABLE = 'Salary_Backtest_Predictions'
SALARY_BACKTEST_START_YEAR = 2022
SALARY_CALIBRATION_START_YEAR = 2021
SALARY_METHOD_VERSION = 'current_locked_spec_v6_v2_population_11f'
VALIDATION_DATASETS_ONLY = os.getenv('SALARY_VALIDATION_DATASETS_ONLY', '0') == '1'
KEEPERS_ONLY = os.getenv('SALARY_KEEPERS_ONLY', '0') == '1'
IS_STAGED_DATABASE_RUN = (
    Path(db_path).resolve()
    != (Path(root_path) / 'Data' / 'Databases').resolve()
)
V2_BETA_DATABASE = Path(
    os.getenv(
        'FF_V2_BETA_DATABASE',
        str(
            Path(root_path)
            / 'Data'
            / 'Databases'
            / 'Projection_V2_beta.sqlite3'
        ),
    )
)
ENSEMBLE_FALLBACK_GAMES = 16.5
SALARY_OPTUNA_ITERATIONS = 15
SALARY_OPTUNA_TIMEOUT = 45
SALARY_NORMALIZATION_FLOOR = 1.0
SALARY_MODEL_FEATURES = [
    # Two complementary preseason market anchors. Their high correlation is
    # intentional: source dollars encode this league's curve, while log ADP
    # retains nonlinear broader-market information that a rank gap lost in the
    # rolling ablation.
    'budget_adjusted_source_salary',
    'avg_pick_log',
    # Projection level and projection/source disagreement. Weekly donor-tail
    # width remains an audited simulation diagnostic, not a salary-model
    # feature: it is a different construct from the historical OOS residual
    # interval and did not improve strict rolling validation.
    'ensemble_pred_ppg',
    'ensemble_vs_price_gap',
    # Role and breakout context.
    'pos_proj_points_share',
    'rb_pos_rush_share',
    'year_exp',
    'is_rookie',
    # WR is the reference position.
    'QB',
    'RB',
    'TE',
]
SALARY_MODEL_SPLIT_COLUMNS = ['year', 'game_date']
SALARY_NORMALIZATION_SCHEMA = {
    'normalization_method': 'TEXT',
    'normalization_floor': 'REAL',
    'pred_salary_shift': 'REAL',
}
SALARY_KEEPER_MARKET_FEATURE_COLS = [
    'keeper_market_value',
    'keeper_source_market_value',
    'keeper_source_values_observed',
    'keeper_contract_discount',
    'keeper_pool_base_budget',
    'keeper_pool_inflation',
    'source_market_total',
    'source_nonkeeper_market_total',
    'source_salary_floor',
    'log_source_salary',
    'keeper_adjusted_source_salary',
    'source_market_scale',
    'budget_adjusted_source_salary',
    'keeper_adjusted_source_diff',
    'budget_adjusted_source_diff',
]
SALARY_VALIDATION_AUDIT_SCHEMA = {
    **SALARY_NORMALIZATION_SCHEMA,
    **{
        col: (
            'INTEGER'
            if col == 'keeper_source_values_observed'
            else 'REAL'
        )
        for col in SALARY_KEEPER_MARKET_FEATURE_COLS
    },
}

ENSEMBLE_RESID_PERCENTILES = (5, 10, 25, 75, 90, 95)
ENSEMBLE_SOURCE_RESID_COLS = [
    f'pred_resid_{percentile}'
    for percentile in ENSEMBLE_RESID_PERCENTILES
]
ENSEMBLE_FEATURE_RESID_COLS = [
    f'ensemble_pred_resid_{percentile}'
    for percentile in ENSEMBLE_RESID_PERCENTILES
]

PROJECTION_SHARE_FAMILIES = {
    'team_proj_points_share': ('team_proj_share_', 'proj_points'),
    'pos_proj_points_share': ('pos_proj_share_', 'proj_points'),
    'team_proj_rush_att_share': ('team_proj_share_', 'rush_att'),
    'pos_proj_rush_att_share': ('pos_proj_share_', 'rush_att'),
    'team_proj_rec_share': ('team_proj_share_', 'rec'),
    'pos_proj_rec_share': ('pos_proj_share_', 'rec'),
    'team_proj_rec_yds_share': ('team_proj_share_', 'rec_yds'),
    'pos_proj_rec_yds_share': ('pos_proj_share_', 'rec_yds'),
}
PROJECTION_SHARE_FEATURES = list(PROJECTION_SHARE_FAMILIES)
PROJECTION_SHARE_STD_FEATURES = [
    f'{feature}_source_std'
    for feature in PROJECTION_SHARE_FEATURES
]


KEEPERS_FILE = Path(
    os.getenv(
        'FF_KEEPERS_FILE',
        str(
            Path(root_path)
            / 'Data'
            / 'OtherData'
            / 'Keepers'
            / f'keepers_{YEAR}_{LEAGUE}.csv'
        ),
    )
).expanduser().resolve()
if not KEEPERS_FILE.is_file():
    raise FileNotFoundError(
        f'Missing required {YEAR} {LEAGUE} keeper input: {KEEPERS_FILE}'
    )
ty_keepers = pd.read_csv(KEEPERS_FILE)
required_keeper_columns = {'player', 'keeper_salary'}
missing_keeper_columns = sorted(
    required_keeper_columns.difference(ty_keepers.columns)
)
if missing_keeper_columns:
    raise ValueError(
        f'{KEEPERS_FILE} is missing keeper columns: '
        f'{missing_keeper_columns}'
    )
ty_keepers = ty_keepers[['player', 'keeper_salary']].rename(
    columns={'keeper_salary': 'ty_keeper_sal'}
)
ty_keepers['ty_keeper_sal'] = pd.to_numeric(
    ty_keepers['ty_keeper_sal'],
    errors='raise',
)
if (
    ty_keepers['player'].isna().any()
    or ty_keepers['player'].astype(str).str.strip().eq('').any()
):
    raise ValueError(f'{KEEPERS_FILE} contains blank keeper names')
ty_keepers['player'] = ty_keepers.player.apply(dc.name_clean)
ty_keepers['year'] = YEAR


def write_league_keepers(
    keepers,
    db_file,
    year,
    league,
    *,
    v2_database=None,
):
    keeper_output = keepers[['player', 'ty_keeper_sal']].copy()
    keeper_output['player'] = keeper_output.player.apply(dc.name_clean)
    keeper_output['keeper_salary'] = pd.to_numeric(
        keeper_output.pop('ty_keeper_sal'),
        errors='raise',
    )
    keeper_output['year'] = int(year)
    keeper_output['league'] = league

    if keeper_output.player.duplicated().any():
        duplicates = keeper_output.loc[keeper_output.player.duplicated(), 'player'].tolist()
        raise ValueError(f'Duplicate keepers after name cleaning: {duplicates}')

    conn = sqlite3.connect(db_file)
    try:
        if v2_database is not None:
            aliases, identities = load_identity_frames(Path(v2_database))
            projection_keys = resolve_source_player_keys(
                keeper_output[['player']].copy(),
                aliases,
                identities,
                year=int(year),
                source_name=f'{league}_league_keepers',
            )
            projection_keys = projection_keys[['player', 'player_key']]
        else:
            projection_keys = pd.read_sql_query(
                '''SELECT player_key, player
                     FROM Final_Predictions_Resid
                    WHERE year=? AND version=? AND dataset='final_ensemble' ''',
                conn,
                params=(int(year), league),
            )
            projection_keys['player'] = projection_keys.player.apply(dc.name_clean)
            if (
                projection_keys.player_key.isna().any()
                or projection_keys.player_key.duplicated().any()
                or projection_keys.player.duplicated().any()
            ):
                raise ValueError(
                    'Active production projections do not provide a unique '
                    'keeper player-key bridge.'
                )
        keeper_output = keeper_output.merge(
            projection_keys,
            on='player',
            how='left',
            validate='one_to_one',
        )
        if keeper_output.player_key.isna().any():
            missing = keeper_output.loc[
                keeper_output.player_key.isna(), 'player'
            ].tolist()
            raise ValueError(
                f'Keepers are outside the canonical production pool: {missing}'
            )
        keeper_output = keeper_output[[
            'year',
            'league',
            'player_key',
            'player',
            'keeper_salary',
        ]]
        with conn:
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS League_Keepers (
                       year INTEGER NOT NULL,
                       league TEXT NOT NULL,
                       player_key TEXT,
                       player TEXT NOT NULL,
                       keeper_salary REAL NOT NULL,
                       PRIMARY KEY (year, league, player)
                   )'''
            )
            keeper_columns = {
                row[1]
                for row in conn.execute('PRAGMA table_info("League_Keepers")')
            }
            if 'player_key' not in keeper_columns:
                conn.execute(
                    'ALTER TABLE League_Keepers ADD COLUMN player_key TEXT'
                )
            conn.execute(
                'DELETE FROM League_Keepers WHERE year=? AND league=?',
                (int(year), league),
            )
            if len(keeper_output) > 0:
                keeper_output.to_sql('League_Keepers', conn, if_exists='append', index=False)
            conn.execute(
                '''CREATE UNIQUE INDEX IF NOT EXISTS
                       ux_league_keepers_slice_player_key
                   ON League_Keepers(year, league, player_key)'''
            )
    finally:
        conn.close()

    print(f'Saved {len(keeper_output)} keepers for {year} {league}.')
    return keeper_output


if VALIDATION_DATASETS_ONLY:
    print('Validation-only run: leaving Simulation.League_Keepers unchanged.')
else:
    league_keepers = write_league_keepers(
        ty_keepers,
        Path(db_path) / 'Simulation.sqlite3',
        YEAR,
        LEAGUE,
        v2_database=V2_BETA_DATABASE,
    )

if KEEPERS_ONLY:
    print('Keeper publication complete; skipping salary-model cells.')
    raise SystemExit(0)


#%%

#=================
# Load salaries from ESPN into database
#=================

if IS_STAGED_DATABASE_RUN and not VALIDATION_DATASETS_ONLY:
    salary_source_repair = repair_governed_salary_slices(
        Path(db_path) / 'Simulation.sqlite3',
        governed_salary_source_specs(Path(PATH), YEAR),
        name_clean=dc.name_clean,
        live_database_path=(
            Path(root_path) / 'Data' / 'Databases' / 'Simulation.sqlite3'
        ),
    )
    print(
        'GOVERNED_SALARY_REPAIR_RECEIPT='
        + json.dumps(salary_source_repair, sort_keys=True)
    )
else:
    # Preserve the interactive/non-refresh behavior: validate and replace only
    # the current beta source slice.
    df = pd.read_csv(
        f'{PATH}/OtherData/Salaries/salaries_{YEAR}_{LEAGUE}.csv',
        header=None,
    )
    salaries = parse_espn_salary_records(df)
    salaries['year'] = YEAR
    salaries['league'] = LEAGUE
    salaries.player = salaries.player.apply(dc.name_clean)
    validate_salary_records(
        salaries,
        source_name='cleaned ESPN salary records',
    )

if VALIDATION_DATASETS_ONLY:
    print('Validation-only run: leaving the current Simulation.Salaries slice unchanged.')
elif not IS_STAGED_DATABASE_RUN:
    dm.delete_from_db(
        'Simulation',
        'Salaries',
        f"year='{YEAR}' AND league='{LEAGUE}'",
        create_backup=not IS_STAGED_DATABASE_RUN,
    )
    dm.write_to_db(salaries, 'Simulation', 'Salaries', 'append')

#%%

#--------------
# Function to Add Results to Dataframe after season for modeling
#--------------

def clean_results(path, fname, year, league, team_split=True):
    
    # read in csv file
    results = pd.read_csv(f'{path}/OtherData/Salaries/{fname}.csv')

    # drop null rows from formatting and nonsense rows
    results = results.dropna(subset=['actual_salary'])
    results = results[results.player!='Player'].reset_index(drop=True)

    # fill in all non-keeper player flags
    results.loc[results.is_keeper.isnull(), 'is_keeper'] = 0

    if team_split:
        results.player = results.player.apply(lambda x: x.split(',')[0])
        results.player = results.player.apply(lambda x: x.split('\xa0')[0])
    
    # convert salary columns to float after stripping $ and remove bad player name formatting
    results.actual_salary = results.actual_salary.apply(lambda x: float(x.replace('$', '')))
    results.player = results.player.apply(dc.name_clean)
    
    results['year'] = year
    results['league'] = league
    
    return results

# FNAME = f'{LEAGUE}_{YEAR}_results'
# results = clean_results(PATH, FNAME, YEAR, LEAGUE)
# dm.delete_from_db('Simulation', 'Actual_Salaries', f"year='{YEAR}' AND league='{LEAGUE}'")
# dm.write_to_db(results, 'Simulation', 'Actual_Salaries', 'append')

# # push the actuals to salary database to re-run simulation
# to_actual = dm.read(f"SELECT * FROM Actual_Salaries WHERE year={YEAR} AND league='{LEAGUE}'", 'Simulation')
# to_actual = to_actual[['player', 'actual_salary', 'year', 'league']].rename(columns={'actual_salary': 'salary'})
# to_actual['league'] = to_actual.league.apply(lambda x: f'{x}_actual')
# to_actual['std_dev'] = 0.1
# to_actual['min_score'] = to_actual.salary - 1
# to_actual['max_score'] = to_actual.salary + 1

# dm.delete_from_db('Simulation', 'Salaries_Pred', f"year={YEAR} AND league='{LEAGUE}_actual'")
# dm.write_to_db(to_actual, 'Simulation', 'Salaries_Pred', 'append')

# import shutil

# src = f'{root_path}/Data/Databases/Simulation.sqlite3'
# dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3'
# shutil.copyfile(src, dst)

#%%

def _projection_share_source_columns(table_columns):
    """Map compact role-share features to their available source columns."""
    family_columns = {}
    for feature, (prefix, suffix) in PROJECTION_SHARE_FAMILIES.items():
        matches = [
            col
            for col in table_columns
            if col.startswith(prefix)
            and col.endswith(f'_{suffix}')
            and not col.startswith(f'{prefix}diff_')
            and '_exp_' not in col
            and not col.startswith('rmean')
            and not col.startswith('rmax')
        ]
        family_columns[feature] = sorted(matches)
    return family_columns


def _aggregate_projection_share_sources(stats, family_columns):
    """Collapse many source-specific shares into robust preseason summaries."""
    stats = stats.copy()
    source_columns = sorted({
        col
        for columns in family_columns.values()
        for col in columns
    })
    for feature, columns in family_columns.items():
        if columns:
            values = stats[columns].apply(pd.to_numeric, errors='coerce')
            stats[feature] = values.median(axis=1, skipna=True)
            stats[f'{feature}_source_std'] = values.std(
                axis=1,
                ddof=0,
                skipna=True,
            )
        else:
            stats[feature] = np.nan
            stats[f'{feature}_source_std'] = np.nan
    return stats.drop(columns=source_columns, errors='ignore')


def get_adp():
    all_stats = pd.DataFrame()
    model_inputs_db = Path(db_path) / 'Model_Inputs.sqlite3'
    for pos in ['QB', 'RB', 'WR', 'TE']:
        print(pos)
        table_name = f'{pos}_{YEAR}_ProjOnly'
        with sqlite3.connect(model_inputs_db) as connection:
            table_columns = [
                row[1]
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                )
            ]

        family_columns = _projection_share_source_columns(table_columns)
        base_columns = [
            'player',
            'year',
            'team',
            'avg_pick',
            'avg_pick_log',
            'avg_proj_points',
            'avg_pos_rank',
            'year_exp',
            'avg_proj_points_exp_diff',
        ]
        optional_columns = [
            col for col in ['std_proj_points']
            if col in table_columns
        ]
        source_columns = sorted({
            col
            for columns in family_columns.values()
            for col in columns
        })
        select_columns = base_columns + optional_columns + source_columns
        select_sql = ', '.join(f'"{col}"' for col in select_columns)
        stats = dm.read(
            f'SELECT {select_sql} FROM "{table_name}"',
            'Model_Inputs',
        )
        stats = _aggregate_projection_share_sources(stats, family_columns)
        stats['pos'] = pos
        stats['player_key'] = pd.NA
        stats['player_key_match_method'] = pd.NA
        stats['salary_population_source'] = 'model_inputs_projonly'
        current_mask = pd.to_numeric(
            stats.year,
            errors='coerce',
        ).eq(YEAR)
        if current_mask.any():
            current = attach_v2_player_keys(
                stats.loc[current_mask].copy(),
                V2_BETA_DATABASE,
                season_column='year',
                require_complete=True,
            )
            stats.loc[current_mask, 'player_key'] = (
                current.player_key.to_numpy()
            )
            stats.loc[current_mask, 'player_key_match_method'] = (
                current.player_key_match_method.to_numpy()
            )
        all_stats = pd.concat([all_stats, stats], axis=0)

    production_population = dm.read(
        f'''SELECT player_key, player, pos, year
              FROM Final_Predictions_Resid
             WHERE version='{LEAGUE}'
                   AND year={YEAR}
                   AND dataset='final_ensemble' ''',
        'Simulation',
    )
    if (
        production_population.player_key.isna().any()
        or production_population.player_key.duplicated().any()
    ):
        raise ValueError(
            'Current production salary population lacks unique player keys.'
        )
    production_labels = (
        production_population.assign(
            internal_player=production_population.player.apply(dc.name_clean)
        )
        .set_index('player_key')
        .internal_player
    )
    current_rows = pd.to_numeric(
        all_stats.year,
        errors='coerce',
    ).eq(YEAR)
    all_stats.loc[current_rows, 'salary_source_player_label'] = (
        all_stats.loc[current_rows, 'player']
    )
    all_stats.loc[current_rows, 'player'] = (
        all_stats.loc[current_rows, 'player_key'].map(production_labels)
    )
    if all_stats.loc[current_rows, 'player'].isna().any():
        missing = all_stats.loc[
            current_rows & all_stats.player.isna(),
            'player_key',
        ].tolist()
        raise ValueError(
            'Current ProjOnly salary rows are outside canonical production: '
            f'{missing[:20]}'
        )
    current_core_keys = set(
        all_stats.loc[
            pd.to_numeric(all_stats.year, errors='coerce').eq(YEAR),
            'player_key',
        ].dropna().astype(str)
    )
    missing_population = production_population.loc[
        ~production_population.player_key.astype(str).isin(current_core_keys)
    ].copy()
    if len(missing_population) > 0:
        with sqlite3.connect(V2_BETA_DATABASE) as connection:
            v2_features = pd.read_sql_query(
                f'''SELECT player_key,
                           display_name,
                           season,
                           position,
                           team,
                           team_conflict,
                           year_exp,
                           adp_median,
                           adp_log,
                           expert_points_median,
                           expert_points_iqr,
                           expert_rank_median,
                           expert_ppg_exp_diff,
                           consensus_room_share
                      FROM player_season_features
                     WHERE season=?
                           AND player_key IN (
                               {', '.join('?' for _ in missing_population.player_key)}
                           )''',
                connection,
                params=(
                    int(YEAR),
                    *missing_population.player_key.astype(str).tolist(),
                ),
            )
        fallback = missing_population.merge(
            v2_features,
            on='player_key',
            how='left',
            validate='one_to_one',
        )
        fallback_context_receipt = validate_v2_salary_fallback_context(
            fallback,
            allowed_unresolved_team_player_keys=(
                governed_salary_fallback_null_team_keys(YEAR)
            ),
        )
        print(
            'V2_SALARY_FALLBACK_CONTEXT_RECEIPT='
            + json.dumps(fallback_context_receipt, sort_keys=True)
        )
        fallback['pos'] = fallback.pop('position')
        fallback['player'] = fallback.player.apply(dc.name_clean)
        fallback['year'] = YEAR
        fallback['avg_pick'] = fallback.pop('adp_median')
        fallback['avg_pick_log'] = fallback.pop('adp_log')
        fallback['avg_proj_points'] = fallback.pop(
            'expert_points_median'
        )
        fallback['std_proj_points'] = (
            pd.to_numeric(
                fallback.pop('expert_points_iqr'),
                errors='coerce',
            ).fillna(0)
            / 1.349
        )
        fallback['avg_pos_rank'] = fallback.pop('expert_rank_median')
        fallback['avg_proj_points_exp_diff'] = (
            pd.to_numeric(
                fallback.pop('expert_ppg_exp_diff'),
                errors='coerce',
            ).fillna(0)
            * 17
        )
        for feature in PROJECTION_SHARE_FEATURES:
            fallback[feature] = np.nan
        fallback['pos_proj_points_share'] = fallback.pop(
            'consensus_room_share'
        )
        for feature in PROJECTION_SHARE_STD_FEATURES:
            fallback[feature] = np.nan
        fallback['player_key_match_method'] = (
            'production_final_player_key'
        )
        fallback['salary_source_player_label'] = fallback[
            'display_name'
        ]
        fallback['salary_population_source'] = (
            'v2_player_season_features_fallback'
        )
        all_stats = pd.concat(
            [all_stats, fallback[all_stats.columns]],
            ignore_index=True,
            sort=False,
        )

    current_salary_keys = set(
        all_stats.loc[
            pd.to_numeric(all_stats.year, errors='coerce').eq(YEAR),
            'player_key',
        ].dropna().astype(str)
    )
    expected_salary_keys = set(
        production_population.player_key.astype(str)
    )
    if current_salary_keys != expected_salary_keys:
        raise ValueError(
            'Salary candidate population does not match canonical production: '
            f'missing={sorted(expected_salary_keys - current_salary_keys)[:10]}, '
            f'extra={sorted(current_salary_keys - expected_salary_keys)[:10]}'
        )

    share_cols = PROJECTION_SHARE_FEATURES + PROJECTION_SHARE_STD_FEATURES
    all_stats['projection_share_fallback'] = all_stats[
        PROJECTION_SHARE_FEATURES
    ].isna().any(axis=1)
    for col in share_cols:
        all_stats[col] = (
            all_stats[col]
            .fillna(all_stats.groupby(['year', 'pos'])[col].transform('median'))
            .fillna(0)
        )
    if 'std_proj_points' not in all_stats.columns:
        all_stats['std_proj_points'] = 0.0
    all_stats['std_proj_points'] = (
        pd.to_numeric(all_stats.std_proj_points, errors='coerce')
        .fillna(0)
    )
    all_stats['projection_source_disagreement'] = (
        all_stats.std_proj_points
        / all_stats.avg_proj_points.clip(lower=1)
    )
    all_stats['log_avg_points'] = np.log(all_stats.avg_proj_points)
    return all_stats


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    order = np.argsort(values, kind='mergesort')
    values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, quantile, side='left')),
        len(values) - 1,
    )
    return float(values[index])


def get_current_weekly_residual_quantiles():
    """Return centered current upside from the app's joint weekly donor pools."""
    with sqlite3.connect(
        Path(db_path) / 'Simulation.sqlite3'
    ) as connection:
        required_tables = {
            row[0]
            for row in connection.execute(
                """SELECT name
                     FROM sqlite_master
                    WHERE type='table'
                          AND name IN (
                              'Best_Ball_Weekly_Template_Pools',
                              'Best_Ball_Weekly_Templates'
                          )"""
            )
        }
        if required_tables != {
            'Best_Ball_Weekly_Template_Pools',
            'Best_Ball_Weekly_Templates',
        }:
            raise ValueError(
                'Joint weekly template pools must be rebuilt before salaries.'
            )
        rows = pd.read_sql_query(
            '''SELECT p.pool_player player,
                      p.pool_year year,
                      p.pos,
                      t.active_ppg_resid,
                      p.template_sample_prob
                 FROM Best_Ball_Weekly_Template_Pools p
                 INNER JOIN Best_Ball_Weekly_Templates t
                    ON t.league = p.template_league
                   AND t.template_id = p.template_id
                WHERE p.pool_year = ?
                      AND p.pool_version = ?
                      AND p.pool_dataset = 'final_ensemble' ''',
            connection,
            params=(int(YEAR), LEAGUE),
        )
    if rows.empty:
        raise ValueError(
            f'No weekly template residual pools exist for {YEAR} {LEAGUE}.'
        )
    invalid = (
        ~np.isfinite(
            pd.to_numeric(rows.active_ppg_resid, errors='coerce')
        )
        | ~np.isfinite(
            pd.to_numeric(rows.template_sample_prob, errors='coerce')
        )
        | pd.to_numeric(
            rows.template_sample_prob,
            errors='coerce',
        ).le(0)
    )
    if invalid.any():
        raise ValueError(
            'Weekly salary uncertainty input contains invalid residual weights.'
        )

    records = []
    for keys, group in rows.groupby(
        ['player', 'year', 'pos'],
        sort=True,
    ):
        record = dict(zip(['player', 'year', 'pos'], keys))
        residuals = pd.to_numeric(
            group.active_ppg_resid,
            errors='coerce',
        ).to_numpy(dtype=float)
        weights = pd.to_numeric(
            group.template_sample_prob,
            errors='coerce',
        ).to_numpy(dtype=float)
        weights = weights / weights.sum()
        # The auction simulation adds a sampled donor residual to the locked
        # V2 point projection only after subtracting this weighted pool mean.
        # Salary upside must use that same centered distribution; otherwise
        # donor-pool level bias is mislabeled as player upside.
        centered_residuals = residuals - float(
            np.sum(weights * residuals)
        )
        for percentile in ENSEMBLE_RESID_PERCENTILES:
            record[f'pred_resid_{percentile}'] = _weighted_quantile(
                centered_residuals,
                weights,
                percentile / 100,
            )
        records.append(record)
    quantiles = pd.DataFrame(records)
    if quantiles[ENSEMBLE_SOURCE_RESID_COLS].isna().any().any():
        raise ValueError(
            'Weekly salary uncertainty quantiles are incomplete.'
        )
    quantiles['ensemble_uncertainty_feature_source'] = (
        'joint_weekly_template_centered_active_ppg_residual'
    )
    return quantiles


def get_ensemble_projection_predictions():
    """Return one optimizer-aligned point projection per player-season.

    Historical rows use the persisted production-style OOS validation ensemble.
    Current rows use the final ensemble that feeds the auction simulation.
    """
    historical = dm.read(f'''
        SELECT player,
               CAST(season AS INTEGER) year,
               pos,
               pred_fp_per_game,
               {', '.join(ENSEMBLE_SOURCE_RESID_COLS)}
        FROM Final_Validations_Resid
        WHERE version='{LEAGUE}'
              AND model_spec_asof_year={YEAR}
              AND season < {YEAR}
    ''', 'Validations')
    historical = historical.dropna(
        subset=['player', 'year', 'pos', 'pred_fp_per_game']
    ).copy()
    if not len(historical):
        raise ValueError('No historical ensemble validation predictions are available.')
    historical['player'] = historical.player.apply(dc.name_clean)
    historical['year'] = historical.year.astype(int)
    keys = ['player', 'year', 'pos']
    projection_columns = ['pred_fp_per_game', *ENSEMBLE_SOURCE_RESID_COLS]
    historical = historical.groupby(
        keys,
        as_index=False,
    )[projection_columns].mean()
    historical['ensemble_uncertainty_feature_source'] = (
        'oos_ensemble_residual_quantiles'
    )

    current = dm.read(f'''
        SELECT player,
               year,
               pos,
               pred_fp_per_game
        FROM Final_Predictions_Resid
        WHERE version='{LEAGUE}'
              AND year={YEAR}
              AND dataset='final_ensemble'
    ''', 'Simulation')
    current = current.dropna(
        subset=['player', 'year', 'pos', 'pred_fp_per_game']
    ).copy()
    current['player'] = current.player.apply(dc.name_clean)
    current['year'] = current.year.astype(int)
    current = current.groupby(
        keys,
        as_index=False,
    )[['pred_fp_per_game']].mean()
    weekly_uncertainty = get_current_weekly_residual_quantiles()
    weekly_uncertainty['player'] = weekly_uncertainty.player.apply(
        dc.name_clean
    )
    current = current.merge(
        weekly_uncertainty,
        on=keys,
        how='left',
        validate='one_to_one',
    )
    if current[
        [
            *ENSEMBLE_SOURCE_RESID_COLS,
            'ensemble_uncertainty_feature_source',
        ]
    ].isna().any().any():
        missing = current.loc[
            current[ENSEMBLE_SOURCE_RESID_COLS].isna().any(axis=1),
            keys,
        ].to_dict('records')
        raise ValueError(
            'Current salary rows lack joint weekly upside features: '
            f'{missing[:20]}'
        )

    ensemble = pd.concat([historical, current], ignore_index=True)
    source_counts = ensemble.groupby(keys)[
        'ensemble_uncertainty_feature_source'
    ].nunique()
    if source_counts.gt(1).any():
        raise ValueError(
            'A salary player-season has multiple uncertainty feature sources.'
        )
    uncertainty_sources = (
        ensemble.groupby(keys, as_index=False)[
            'ensemble_uncertainty_feature_source'
        ].first()
    )
    ensemble = ensemble.groupby(
        keys,
        as_index=False,
    )[projection_columns].mean().merge(
        uncertainty_sources,
        on=keys,
        how='left',
        validate='one_to_one',
    ).rename(columns={
        'pred_fp_per_game': 'ensemble_pred_ppg',
        **{
            source_col: feature_col
            for source_col, feature_col in zip(
                ENSEMBLE_SOURCE_RESID_COLS,
                ENSEMBLE_FEATURE_RESID_COLS,
            )
        },
    })
    if ensemble.duplicated(keys).any():
        raise ValueError('Duplicate ensemble player-season-position rows remain.')
    return ensemble


def _fill_ensemble_residual_fallbacks(projection_rows, ensemble):
    """Fill deep-player intervals from same-year/position projection tiers."""
    projection_rows = projection_rows.copy()
    projection_rows['ensemble_resid_fallback'] = projection_rows[
        ENSEMBLE_FEATURE_RESID_COLS
    ].isna().any(axis=1)

    for (year, pos), idx in projection_rows.groupby(['year', 'pos']).groups.items():
        reference = ensemble[
            ensemble.year.eq(year)
            & ensemble.pos.eq(pos)
        ].dropna(
            subset=['ensemble_pred_ppg', *ENSEMBLE_FEATURE_RESID_COLS]
        )
        if len(reference):
            reference = (
                reference
                .groupby('ensemble_pred_ppg', as_index=False)[
                    ENSEMBLE_FEATURE_RESID_COLS
                ]
                .mean()
                .sort_values('ensemble_pred_ppg')
            )

        for col in ENSEMBLE_FEATURE_RESID_COLS:
            missing_idx = projection_rows.loc[idx].index[
                projection_rows.loc[idx, col].isna()
            ]
            if not len(missing_idx):
                continue
            if not len(reference):
                projection_rows.loc[missing_idx, col] = 0.0
            elif len(reference) == 1:
                projection_rows.loc[missing_idx, col] = reference[col].iloc[0]
            else:
                projection_rows.loc[missing_idx, col] = np.interp(
                    projection_rows.loc[
                        missing_idx,
                        'ensemble_pred_ppg',
                    ].to_numpy(),
                    reference.ensemble_pred_ppg.to_numpy(),
                    reference[col].to_numpy(),
                )

    residual_values = projection_rows[
        ENSEMBLE_FEATURE_RESID_COLS
    ].to_numpy(dtype=float)
    residual_values = np.maximum.accumulate(residual_values, axis=1)
    projection_rows[ENSEMBLE_FEATURE_RESID_COLS] = residual_values
    return projection_rows


def add_ensemble_projection_features(projection_rows):
    """Left-join point projections and preserve the full salary universe."""
    projection_rows = projection_rows.copy()
    ensemble = get_ensemble_projection_predictions()
    keys = ['player', 'year', 'pos']
    projection_rows = pd.merge(
        projection_rows,
        ensemble,
        on=keys,
        how='left',
        validate='many_to_one',
    )

    projection_rows['ensemble_pred_fallback'] = (
        projection_rows.ensemble_pred_ppg.isna()
    )
    fallback_ppg = projection_rows.avg_proj_points / ENSEMBLE_FALLBACK_GAMES
    projection_rows['ensemble_pred_ppg'] = (
        projection_rows.ensemble_pred_ppg.fillna(fallback_ppg)
    )
    if projection_rows.ensemble_pred_ppg.isna().any():
        missing = projection_rows.loc[
            projection_rows.ensemble_pred_ppg.isna(),
            keys,
        ].to_dict('records')
        raise ValueError(f'Unable to fill ensemble projections: {missing[:10]}')

    validation_years = set(
        ensemble.loc[ensemble.year.lt(YEAR), 'year'].astype(int).unique()
    )
    unexpected_fallback = (
        projection_rows.ensemble_pred_fallback
        & projection_rows.year.astype(int).isin(validation_years)
    )
    if unexpected_fallback.any():
        missing = projection_rows.loc[unexpected_fallback, keys].to_dict('records')
        raise ValueError(
            'Historical ensemble projections failed to join covered validation years: '
            f'{missing[:10]}'
        )

    projection_rows = _fill_ensemble_residual_fallbacks(
        projection_rows,
        ensemble,
    )
    projection_rows['ensemble_floor_p10_ppg'] = (
        projection_rows.ensemble_pred_ppg
        + projection_rows.ensemble_pred_resid_10
    )
    for percentile in [75, 90, 95]:
        projection_rows[f'ensemble_ceiling_p{percentile}_ppg'] = (
            projection_rows.ensemble_pred_ppg
            + projection_rows[f'ensemble_pred_resid_{percentile}']
        )
        projection_rows[f'ensemble_upside_p{percentile}'] = (
            projection_rows[f'ensemble_pred_resid_{percentile}']
        )
    projection_rows['ensemble_downside_p10'] = np.maximum(
        -projection_rows.ensemble_pred_resid_10,
        0,
    )
    projection_rows['ensemble_interval_50'] = (
        projection_rows.ensemble_pred_resid_75
        - projection_rows.ensemble_pred_resid_25
    )
    projection_rows['ensemble_interval_80'] = (
        projection_rows.ensemble_pred_resid_90
        - projection_rows.ensemble_pred_resid_10
    )
    projection_rows['ensemble_interval_90'] = (
        projection_rows.ensemble_pred_resid_95
        - projection_rows.ensemble_pred_resid_5
    )

    exact_rows = int((~projection_rows.ensemble_pred_fallback).sum())
    fallback_rows = int(projection_rows.ensemble_pred_fallback.sum())
    residual_fallback_rows = int(projection_rows.ensemble_resid_fallback.sum())
    print(
        'Ensemble projection feature coverage:',
        f'{exact_rows} exact, {fallback_rows} consensus fallbacks, '
        f'{residual_fallback_rows} residual tier fallbacks',
    )
    return projection_rows

def fill_ty_keepers(salaries, ty_keepers):
    salaries = pd.merge(salaries, ty_keepers, on=['player', 'year'], how='left')
    salaries.loc[(salaries.year==YEAR) & ~(salaries.ty_keeper_sal.isnull()), 'actual_salary'] = \
        salaries.loc[(salaries.year==YEAR) & ~(salaries.ty_keeper_sal.isnull()), 'ty_keeper_sal']
    salaries.loc[(salaries.year==YEAR) & ~(salaries.ty_keeper_sal.isnull()), 'is_keeper'] = 1

    return salaries.drop('ty_keeper_sal', axis=1)


def add_keeper_budget_context(salaries):
    salaries = salaries.copy()
    salaries['is_keeper'] = salaries.is_keeper.fillna(0)

    salaries['source_salary_value'] = (
        pd.to_numeric(salaries.salary, errors='coerce')
        .fillna(0)
        .clip(lower=0)
    )
    salaries['keeper_source_salary_observed'] = (
        salaries.is_keeper.eq(1) & salaries.salary.notna()
    )
    salaries['keeper_source_market_value'] = np.where(
        salaries.is_keeper.eq(1),
        salaries.source_salary_value,
        0,
    )
    # A missing copied keeper value should be neutral rather than implying that
    # the keeper created value equal to the full contract. The deterministic
    # contract is the causal fallback for the missing preseason market value.
    salaries['keeper_market_value_proxy'] = np.where(
        salaries.is_keeper.eq(1),
        salaries.salary.fillna(salaries.actual_salary),
        0,
    )

    keeper_context = (
        salaries[salaries.is_keeper == 1]
        .groupby('year')
        .agg(
            keeper_count=('player', 'size'),
            keeper_spend=('actual_salary', 'sum'),
            keeper_market_value=('keeper_market_value_proxy', 'sum'),
            keeper_source_market_value=('keeper_source_market_value', 'sum'),
            keeper_source_values_observed=('keeper_source_salary_observed', 'sum'),
        )
        .reset_index()
    )
    source_context = (
        salaries.groupby('year')
        .agg(source_market_total=('source_salary_value', 'sum'))
        .reset_index()
    )
    salaries = salaries.drop(
        columns=[
            'source_salary_value',
            'keeper_source_salary_observed',
            'keeper_source_market_value',
            'keeper_market_value_proxy',
        ],
    )
    salaries = pd.merge(salaries, keeper_context, on='year', how='left')
    salaries = pd.merge(salaries, source_context, on='year', how='left')
    context_fill_cols = [
        'keeper_count',
        'keeper_spend',
        'keeper_market_value',
        'keeper_source_market_value',
        'keeper_source_values_observed',
    ]
    salaries[context_fill_cols] = salaries[context_fill_cols].fillna(0)
    salaries['keeper_count'] = salaries.keeper_count.astype(int)
    salaries['keeper_source_values_observed'] = (
        salaries.keeper_source_values_observed.astype(int)
    )
    salaries['available_slots'] = LEAGUE_ROSTER_SLOTS - salaries.keeper_count
    salaries['available_budget'] = LEAGUE_BUDGET - salaries.keeper_spend
    salaries['keeper_contract_discount'] = (
        salaries.keeper_market_value - salaries.keeper_spend
    )
    salaries['keeper_pool_base_budget'] = (
        LEAGUE_BUDGET - salaries.keeper_market_value
    )
    salaries['source_nonkeeper_market_total'] = (
        salaries.source_market_total - salaries.keeper_source_market_value
    )

    invalid_context = (
        (salaries.available_slots < 0)
        | (salaries.available_budget < salaries.available_slots)
        | (salaries.keeper_pool_base_budget <= 0)
        | (salaries.source_nonkeeper_market_total <= 0)
    )
    if invalid_context.any():
        invalid_years = sorted(salaries.loc[invalid_context, 'year'].unique())
        raise ValueError(f'Invalid keeper budget context for years: {invalid_years}')

    salaries['keeper_pool_inflation'] = (
        salaries.available_budget / salaries.keeper_pool_base_budget
    )
    return salaries

def _canonicalize_current_salary_source_labels(source, source_name):
    source = source.copy()
    current_mask = pd.to_numeric(
        source.year,
        errors='coerce',
    ).eq(YEAR)
    if not current_mask.any():
        return source
    aliases, identities = load_identity_frames(V2_BETA_DATABASE)
    resolved = resolve_source_player_keys(
        source.loc[current_mask].copy(),
        aliases,
        identities,
        year=YEAR,
        source_name=source_name,
        # Auction source files also contain kickers and defenses, which are
        # deliberately outside the V2 QB/RB/WR/TE identity surface. Resolve
        # every supported skill player, retain the unrelated source labels,
        # and enforce exact V2 population parity on the final candidate output.
        require_complete=False,
    )
    canonical_names = identities.set_index('player_key').display_name
    resolved_names = resolved.player_key.map(canonical_names).map(
        lambda value: dc.name_clean(value)
        if pd.notna(value)
        else pd.NA
    )
    resolved_mask = resolved.player_key.notna()
    if resolved_names.loc[resolved_mask].isna().any():
        missing = resolved.loc[
            resolved_mask & resolved_names.isna(), 'player'
        ].tolist()
        raise ValueError(
            f'{source_name} lacks canonical V2 display labels: {missing}'
        )
    current_names = (
        source.loc[current_mask, 'player']
        .reset_index(drop=True)
    )
    current_names.loc[resolved_mask] = resolved_names.loc[
        resolved_mask
    ].to_numpy()
    source.loc[current_mask, 'player'] = current_names.to_numpy()
    unresolved_count = int((~resolved_mask).sum())
    if unresolved_count:
        print(
            f'{source_name}: retained {unresolved_count} current non-V2 '
            'source rows (for example kickers/defenses); final salary '
            'candidates remain key-gated.'
        )
    return source


def get_salaries():
    actual_sal = dm.read(f'''SELECT *
                            FROM Actual_Salaries 
                            WHERE League='{LEAGUE}'
                                  AND year <= {YEAR} ''', 'Simulation')
    base_sal = dm.read(f'''SELECT player, salary, year
                                FROM Salaries 
                                WHERE League='{LEAGUE}'
                                 AND year <= {YEAR} ''', 'Simulation')
    # Historical salary sources were not always stored under the same cleaned
    # identity (for example ``Mohamed Sanu Sr`` versus ``Mohamed Sanu``).
    # Normalize both sides before joining so keeper contracts can reuse the
    # projection universe's position history without one-off aliases.
    for source_name, source in [('Actual_Salaries', actual_sal), ('Salaries', base_sal)]:
        source['player'] = source.player.apply(dc.name_clean)
        canonicalized = _canonicalize_current_salary_source_labels(
            source,
            source_name,
        )
        source.loc[:, 'player'] = canonicalized.player
        duplicate_keys = source.duplicated(['player', 'year'], keep=False)
        if duplicate_keys.any():
            duplicates = (
                source.loc[duplicate_keys, ['player', 'year']]
                .drop_duplicates()
                .to_dict('records')
            )
            raise ValueError(
                f'{source_name} has duplicate player-year rows after name cleaning: '
                f'{duplicates[:10]}'
            )
    # Preserve actual-only purchases long enough to join them to the pre-auction
    # projection universe. The old right join silently discarded deep players
    # whose ESPN values were not copied into ``Salaries``.
    salaries = pd.merge(
        actual_sal,
        base_sal,
        on=['player', 'year'],
        how='outer',
        validate='one_to_one',
    )
    return salaries

def add_rookie(salaries):
    rookies = dm.read('''SELECT player, year 
                         FROM Draft_Positions
                         WHERE pos IN ('RB', 'WR', 'TE', 'QB')
                      ''', 'Season_Stats_New')
    rookies['is_rookie'] = 1
    salaries = pd.merge(salaries, rookies, on=['player', 'year'], how='left')
    salaries.is_rookie = salaries.is_rookie.fillna(0)

    return salaries

def calc_inflation(salaries):
    keepers = salaries.loc[salaries.is_keeper==1, ['player', 'salary', 'actual_salary', 'year']].copy()
    keepers['value'] = keepers.salary - keepers.actual_salary
    inflation = keepers.groupby('year').agg({'value': 'sum'}).reset_index()
    inflation['inflation'] = 1 + (inflation.value / LEAGUE_BUDGET)

    salaries = pd.merge(salaries, inflation, on='year', how='left')
    salaries.loc[salaries.inflation.isnull(), 'inflation'] = 1
    salaries.loc[salaries.value.isnull(), 'value'] = 0
    salaries.is_keeper = salaries.is_keeper.fillna(0)

    return salaries

def add_pos_keeper_val(salaries):
    keeper_val = salaries.loc[salaries.is_keeper==1].groupby(['year', 'pos']).agg({'salary': 'sum',
                                                                                 'actual_salary': 'sum'})
    keeper_val.columns = ['keeper_salary', 'pos_keeper_actual_salary']
    keeper_val['pos_keeper_value'] = keeper_val.keeper_salary - keeper_val.pos_keeper_actual_salary
    keeper_val['pos_keeper_inflation'] = keeper_val.pos_keeper_value / (keeper_val.keeper_salary+1)
    keeper_cols = keeper_val.columns
    keeper_val = keeper_val.reset_index()
   
    salaries = pd.merge(salaries, keeper_val, on=['year', 'pos'], how='left')
    salaries.loc[salaries.pos_keeper_value.isnull(), keeper_cols] = 0
    return salaries


def add_keeper_market_salary_features(salaries):
    """Add causal source-price features after accounting for keeper contracts.

    ``keeper_adjusted_source_salary`` applies only the keeper-created remaining
    pool multiplier. ``budget_adjusted_source_salary`` additionally reconciles
    the copied source-price curve to the known open-slot budget. Both are model
    inputs; neither replaces the final prediction-market reconciliation.
    """
    salaries = salaries.copy()
    salaries['keeper_market_context_fallback'] = (
        salaries.keeper_pool_inflation.isna()
        | salaries.available_slots.isna()
        | salaries.available_budget.isna()
    )
    salaries['keeper_pool_inflation'] = salaries.keeper_pool_inflation.fillna(1.0)
    salaries['source_salary_floor'] = (
        pd.to_numeric(salaries.salary, errors='coerce')
        .fillna(0)
        .clip(lower=SALARY_NORMALIZATION_FLOOR)
    )
    salaries['log_source_salary'] = np.log1p(salaries.source_salary_floor)
    salaries['keeper_adjusted_source_salary'] = (
        SALARY_NORMALIZATION_FLOOR
        + salaries.keeper_pool_inflation
        * (salaries.source_salary_floor - SALARY_NORMALIZATION_FLOOR)
    )
    salaries['source_market_scale'] = np.nan
    salaries['budget_adjusted_source_salary'] = np.nan

    for year, year_rows in salaries.groupby('year', sort=False):
        context = year_rows[
            ['available_slots', 'available_budget']
        ].drop_duplicates()
        if len(context) != 1 or context.isna().any(axis=None):
            continue
        available_slots = int(context.available_slots.iloc[0])
        available_budget = float(context.available_budget.iloc[0])
        non_keeper_mask = (
            salaries.index.isin(year_rows.index)
            & salaries.is_keeper.fillna(0).eq(0)
        )
        non_keeper_source = salaries.loc[
            non_keeper_mask,
            'source_salary_floor',
        ]
        if len(non_keeper_source) < available_slots:
            raise ValueError(
                f'{year} has {len(non_keeper_source)} source salary rows for '
                f'{available_slots} open roster slots.'
            )

        top_source = non_keeper_source.nlargest(available_slots)
        source_excess = float(
            (top_source - SALARY_NORMALIZATION_FLOOR).sum()
        )
        target_excess = (
            available_budget
            - available_slots * SALARY_NORMALIZATION_FLOOR
        )
        if source_excess <= 0 or target_excess < 0:
            raise ValueError(
                f'Cannot construct keeper-adjusted source salaries for {year}.'
            )
        source_market_scale = target_excess / source_excess
        salaries.loc[year_rows.index, 'source_market_scale'] = source_market_scale
        salaries.loc[non_keeper_mask, 'budget_adjusted_source_salary'] = (
            SALARY_NORMALIZATION_FLOOR
            + source_market_scale
            * (
                salaries.loc[non_keeper_mask, 'source_salary_floor']
                - SALARY_NORMALIZATION_FLOOR
            )
        )

        keeper_mask = (
            salaries.index.isin(year_rows.index)
            & salaries.is_keeper.fillna(0).eq(1)
        )
        keeper_contracts = pd.to_numeric(
            salaries.loc[keeper_mask, 'actual_salary'],
            errors='coerce',
        )
        salaries.loc[keeper_mask, 'budget_adjusted_source_salary'] = (
            keeper_contracts.fillna(
                salaries.loc[keeper_mask, 'source_salary_floor']
            )
        )

    # Projection-only archival seasons can predate the salary/keeper source.
    # They are never labeled salary rows, so retain them with neutral features
    # rather than allowing their absent context to contaminate later origins.
    salaries['source_market_scale'] = salaries.source_market_scale.fillna(1.0)
    salaries['budget_adjusted_source_salary'] = (
        salaries.budget_adjusted_source_salary.fillna(
            salaries.source_salary_floor
        )
    )

    salaries['keeper_adjusted_source_diff'] = (
        salaries.keeper_adjusted_source_salary - salaries.source_salary_floor
    )
    salaries['budget_adjusted_source_diff'] = (
        salaries.budget_adjusted_source_salary - salaries.source_salary_floor
    )
    return salaries


def drop_keepers(salaries):
    salaries = salaries[(salaries.is_keeper==0) | (salaries.year==YEAR)].reset_index(drop=True)
    salaries = salaries[(salaries.year==YEAR) | (~salaries.actual_salary.isnull())].reset_index(drop=True)
    return salaries

def add_salary_model_features(salaries):
    """Add the one cross-sectional rank feature retained by the compact model."""
    salaries = salaries.copy()
    group_cols = ['year', 'pos']
    ensemble_strength = salaries.groupby(group_cols).ensemble_pred_ppg.rank(
        method='average',
        pct=True,
        ascending=True,
    )
    price_strength = salaries.groupby(
        group_cols
    ).budget_adjusted_source_salary.rank(
        method='average',
        pct=True,
        ascending=True,
    )
    salaries['ensemble_vs_price_gap'] = ensemble_strength - price_strength
    return salaries


def add_salary_model_features_by_keeper_availability(salaries):
    """Calculate rank/gap features within the players who can enter bidding.

    Keeper rows remain in the prediction pool so their deterministic contract
    salaries can be emitted, but they must not alter the projection-versus-price
    rank gap seen by auctionable players. Keeper model predictions are discarded
    by ``finalize_salary_predictions``.
    """
    salaries = salaries.copy()
    salaries['_salary_row_order'] = np.arange(len(salaries))
    featured = []
    for _, availability_rows in salaries.groupby(
        salaries.is_keeper.fillna(0).eq(1),
        sort=False,
    ):
        featured.append(add_salary_model_features(availability_rows))

    return (
        pd.concat(featured, ignore_index=True)
        .sort_values('_salary_row_order')
        .drop(columns='_salary_row_order')
        .reset_index(drop=True)
    )


def build_salary_model_matrix(salary_rows, feature_columns=None):
    """Return the compact causal salary feature surface plus split columns."""
    matrix = salary_rows.copy()
    if 'game_date' not in matrix.columns:
        matrix['game_date'] = matrix.year

    position_dummies = pd.get_dummies(matrix.pos, dtype=int)
    for pos in ['QB', 'RB', 'TE']:
        matrix[pos] = position_dummies[pos] if pos in position_dummies else 0
    matrix['rb_pos_rush_share'] = (
        matrix.RB * matrix.pos_proj_rush_att_share
    )

    required_columns = [*SALARY_MODEL_SPLIT_COLUMNS, *SALARY_MODEL_FEATURES]
    missing_columns = [
        col for col in required_columns if col not in matrix.columns
    ]
    if missing_columns:
        raise ValueError(
            f'Salary model matrix is missing features: {missing_columns}'
        )

    matrix = matrix[required_columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError('Salary model matrix contains non-finite values.')

    if feature_columns is not None:
        missing_columns = [
            col for col in feature_columns if col not in matrix.columns
        ]
        if missing_columns:
            raise ValueError(
                f'Salary prediction pool is missing features: {missing_columns}'
            )
        matrix = matrix.reindex(columns=feature_columns)
    return matrix

def remove_outliers(salaries):
    outlier_list = [
                    ['Jk Dobbins', 2021], #injured
                    ['Leonard Fournette', 2020], #waived
                    ['Ronald Jones', 2020], #fournette came
                    ['Derrius Guice', 2019], #injured
                    ['Brian Robinson', 2022], #shot
                  #  ['Jonathan Taylor', 2023] # pup / holdout
                    ]
    for p, y in outlier_list:
        salaries = salaries[~((salaries.player==p) & (salaries.year==y))].reset_index(drop=True)
    return salaries


def ensure_table_columns(db_file, table_name, column_types):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    existing_cols = {row[1] for row in cur.execute(f"PRAGMA table_info({table_name})")}

    if existing_cols:
        for col, col_type in column_types.items():
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
        conn.commit()

    conn.close()


def additive_floor_normalize_market(
    values,
    slots,
    budget,
    floor=SALARY_NORMALIZATION_FLOOR,
):
    """Project one salary curve onto an exact market budget with a common shift."""
    values = (
        pd.to_numeric(values, errors='coerce')
        .fillna(floor)
        .clip(lower=floor)
        .astype(float)
    )
    slots = int(slots)
    budget = float(budget)
    if slots <= 0 or len(values) < slots or budget < slots * floor:
        raise ValueError(
            f'Invalid additive salary market: {len(values)} rows, '
            f'{slots} slots, ${budget:.2f} budget.'
        )

    top_idx = values.nlargest(slots).index
    top_values = values.loc[top_idx]
    pre_total = float(top_values.sum())

    if np.isclose(pre_total, budget, atol=1e-10):
        shift = 0.0
    elif pre_total < budget:
        shift = (budget - pre_total) / slots
    else:
        lower = float(floor - top_values.max())
        upper = 0.0
        for _ in range(100):
            midpoint = (lower + upper) / 2
            midpoint_total = float(
                np.maximum(floor, top_values + midpoint).sum()
            )
            if midpoint_total > budget:
                upper = midpoint
            else:
                lower = midpoint
        shift = (lower + upper) / 2

    adjusted = (values + shift).clip(lower=floor)
    post_total = float(adjusted.loc[top_idx].sum())
    if not np.isclose(post_total, budget, atol=1e-7):
        raise ValueError(
            f'Additive salary normalization missed its target: '
            f'${post_total:.8f} versus ${budget:.8f}.'
        )
    return adjusted, top_idx, float(shift), pre_total, post_total


def finalize_salary_predictions(
    pred_results,
    show_results=False,
    normalization_mode='auto',
):
    """Apply aggregate market calibration without mixing keeper and non-keeper dollars.

    Historical OOF rows are reconciled to the realized spend of the exact
    represented non-keeper rows by default. Current predictions and explicit
    backtest rows use the known remaining league budget and roster slots after
    keepers. Raw model predictions and normalization metadata are retained so
    callers cannot silently treat ex-post reconciliation as causal.
    """
    pred_results = pred_results.copy()
    pred_results['is_keeper'] = pred_results.is_keeper.fillna(0)
    pred_results['pred_salary'] = pd.to_numeric(pred_results.pred_salary, errors='coerce')
    pred_results['salary'] = pd.to_numeric(pred_results.salary, errors='coerce')
    pred_results.loc[pred_results.pred_salary < 1, 'pred_salary'] = 1
    pred_results['pred_salary'] = pred_results.pred_salary.fillna(1).astype(float)
    pred_results['pred_salary_raw'] = pred_results.pred_salary

    context_cols = ['keeper_count', 'keeper_spend', 'available_slots', 'available_budget']
    missing_context = [col for col in context_cols if col not in pred_results.columns]
    if missing_context:
        raise ValueError(f'Missing salary budget context columns: {missing_context}')

    processed = []
    for year, year_results in pred_results.groupby('year', sort=False):
        year_results = year_results.copy().reset_index(drop=True)

        context = year_results[context_cols].drop_duplicates()
        if len(context) != 1:
            raise ValueError(f'Inconsistent keeper budget context for {year}.')
        context = context.iloc[0]
        available_slots = int(context.available_slots)
        available_budget = float(context.available_budget)

        keeper_mask = year_results.is_keeper == 1
        if year_results.loc[keeper_mask, 'actual_salary'].isna().any():
            missing_keepers = year_results.loc[
                keeper_mask & year_results.actual_salary.isna(),
                'player',
            ].tolist()
            raise ValueError(f'{year} keepers missing actual salaries: {missing_keepers}')
        year_results.loc[keeper_mask, 'pred_salary'] = year_results.loc[keeper_mask, 'actual_salary']
        year_results.loc[keeper_mask, 'pred_salary_raw'] = year_results.loc[
            keeper_mask,
            'actual_salary',
        ]
        non_keeper_mask = ~keeper_mask
        non_keeper_predictions = year_results.loc[non_keeper_mask, 'pred_salary']

        if normalization_mode not in ('auto', 'known_budget', 'represented_actual'):
            raise ValueError(f'Unknown salary normalization mode: {normalization_mode}')

        use_known_budget = (
            normalization_mode == 'known_budget'
            or (normalization_mode == 'auto' and int(year) == int(YEAR))
        )
        if use_known_budget:
            if len(non_keeper_predictions) < available_slots:
                raise ValueError(
                    f'{year} has {len(non_keeper_predictions)} non-keeper predictions '
                    f'for {available_slots} available slots.'
                )
            top_idx = non_keeper_predictions.nlargest(available_slots).index
            normalization_slots = available_slots
            normalization_budget = available_budget
            normalization_source = 'keeper-adjusted league budget'
        else:
            historical_actual = year_results.loc[non_keeper_mask, 'actual_salary']
            if historical_actual.isna().any():
                raise ValueError(f'{year} OOF rows are missing actual salaries.')
            top_idx = historical_actual.index
            normalization_slots = len(top_idx)
            normalization_budget = float(historical_actual.sum())
            normalization_source = 'represented non-keeper actual spend'

        (
            normalized_predictions,
            normalized_top_idx,
            salary_shift,
            pre_normalized_total,
            post_normalized_total,
        ) = additive_floor_normalize_market(
            year_results.loc[non_keeper_mask, 'pred_salary'],
            normalization_slots,
            normalization_budget,
            floor=SALARY_NORMALIZATION_FLOOR,
        )
        if set(normalized_top_idx) != set(top_idx):
            raise ValueError(f'{year} additive normalization changed the market top rows.')
        year_results.loc[non_keeper_mask, 'pred_salary'] = normalized_predictions
        year_results['pred_salary'] = year_results.pred_salary.clip(
            lower=SALARY_NORMALIZATION_FLOOR
        )
        year_results['normalization_mode'] = normalization_mode
        year_results['normalization_method'] = 'additive_floor'
        year_results['normalization_source'] = normalization_source
        year_results['normalization_slots'] = normalization_slots
        year_results['normalization_budget'] = normalization_budget
        # Retain the legacy scale column for schema compatibility. Additive
        # normalization has a unit scale and is fully described by the shift.
        year_results['pred_salary_scale'] = 1.0
        year_results['pred_salary_shift'] = salary_shift
        year_results['normalization_floor'] = SALARY_NORMALIZATION_FLOOR
        year_results['pre_normalized_total'] = pre_normalized_total
        year_results['post_normalized_total'] = post_normalized_total
        year_results['pred_diff'] = year_results.pred_salary - year_results.salary

        if show_results:
            print(
                f'{year} Keeper Count/Spend:',
                int(context.keeper_count),
                round(float(context.keeper_spend), 1),
            )
            print(
                f'{year} Available Slots/Budget:',
                available_slots,
                round(available_budget, 1),
            )
            print(
                f'{year} Normalization Target ({normalization_source}):',
                normalization_slots,
                round(normalization_budget, 1),
            )
            print(
                f'{year} Pred Total Before/After/Shift:',
                round(pre_normalized_total, 1),
                round(post_normalized_total, 1),
                round(salary_shift, 4),
            )
            display(year_results.sort_values('pred_salary', ascending=False).iloc[:50])
            display(
                year_results[np.abs(year_results.pred_diff) > 4]
                .sort_values(by='pred_diff', ascending=False)
            )

        processed.append(year_results)

    return pd.concat(processed, axis=0).reset_index(drop=True)


def _quantile_series(values, alphas=SALARY_RESID_ALPHAS, q_cols=SALARY_RESID_COLS):
    if len(values) == 0:
        return pd.Series(np.zeros(len(q_cols)), index=q_cols)

    q = values.quantile(list(alphas))
    return pd.Series(q.to_numpy(), index=q_cols)


def _single_bucket_table(cur_val, fallback_q, q_cols=SALARY_RESID_COLS, pred_col='pred_salary'):
    bucket_table = pd.DataFrame([fallback_q.to_numpy()], columns=q_cols)
    bucket_table.index.name = '_salary_resid_bucket'
    bucket_table['resid_bucket_n'] = len(cur_val)
    bucket_table['resid_bucket_mean_pred'] = cur_val[pred_col].mean()
    bucket_table[q_cols] = np.maximum.accumulate(bucket_table[q_cols].to_numpy(), axis=1)
    return bucket_table, np.array([])


def _fit_salary_resid_bucket_table(
    cur_val,
    fallback_q,
    q_cols=SALARY_RESID_COLS,
    pred_col='pred_salary',
    resid_col='actual_resid',
    min_n=30,
    min_bins=2,
    max_bins=8,
):
    cur_val = cur_val[[pred_col, resid_col]].dropna().copy()
    unique_pred = cur_val[pred_col].nunique()

    if len(cur_val) == 0 or unique_pred < min_bins:
        return _single_bucket_table(cur_val, fallback_q, q_cols, pred_col)

    n_bins = max(min_bins, int(len(cur_val) // min_n))
    n_bins = min(n_bins, max_bins, unique_pred, len(cur_val))
    if n_bins < min_bins:
        return _single_bucket_table(cur_val, fallback_q, q_cols, pred_col)

    _, edges = pd.qcut(cur_val[pred_col], n_bins, retbins=True, duplicates='drop')
    inner_edges = np.unique(edges)[1:-1]
    cur_val['_salary_resid_bucket'] = np.searchsorted(
        inner_edges,
        cur_val[pred_col].to_numpy(),
        side='right',
    )

    bucket_grp = cur_val.groupby('_salary_resid_bucket')
    bucket_table = bucket_grp[resid_col].quantile(list(SALARY_RESID_ALPHAS)).unstack()
    bucket_table.columns = q_cols
    bucket_table['resid_bucket_n'] = bucket_grp[resid_col].size()
    bucket_table['resid_bucket_mean_pred'] = bucket_grp[pred_col].mean()

    small_buckets = bucket_table.resid_bucket_n < min_n
    for col in q_cols:
        bucket_table.loc[small_buckets, col] = fallback_q[col]

    bucket_table[q_cols] = np.maximum.accumulate(bucket_table[q_cols].to_numpy(), axis=1)
    return bucket_table, inner_edges


def _predict_salary_resid_quantiles(
    bucket_table,
    inner_edges,
    pred_values,
    fallback_q,
    q_cols=SALARY_RESID_COLS,
    smooth=True,
):
    pred_values = np.asarray(pred_values)

    if len(bucket_table) == 0:
        return np.repeat(fallback_q.to_numpy().reshape(1, -1), len(pred_values), axis=0)

    if smooth and bucket_table.resid_bucket_mean_pred.nunique() > 1:
        interp_table = (
            bucket_table
            .sort_values('resid_bucket_mean_pred')
            .drop_duplicates('resid_bucket_mean_pred')
        )
        x = interp_table.resid_bucket_mean_pred.to_numpy()
        pred_q = np.column_stack([
            np.interp(pred_values, x, interp_table[col].to_numpy())
            for col in q_cols
        ])
        return np.maximum.accumulate(pred_q, axis=1)

    bucket_idx = np.searchsorted(inner_edges, pred_values, side='right')
    pred_q = pd.DataFrame({'_salary_resid_bucket': bucket_idx}).join(
        bucket_table[q_cols],
        on='_salary_resid_bucket',
    )
    pred_q = pred_q[q_cols].fillna(fallback_q).to_numpy()
    return np.maximum.accumulate(pred_q, axis=1)


def apply_salary_resid_quantiles(
    val_data,
    output,
    q_cols=SALARY_RESID_COLS,
    pred_col='pred_salary',
    actual_col='actual_salary',
    min_n=30,
    min_bins=2,
    max_bins=8,
    smooth=True,
    bootstrap_iters=50,
    bootstrap_frac=1.0,
    bootstrap_replace=True,
    random_state=42,
):
    val = val_data[['player', 'year', 'pos', pred_col, actual_col]].dropna(subset=[pred_col, actual_col]).copy()
    val['actual_resid'] = val[actual_col] - val[pred_col]
    if len(val) == 0:
        raise ValueError('No validation salary rows are available for residual calibration.')

    global_q = _quantile_series(val.actual_resid)
    output = output.copy()
    for col in q_cols:
        if col not in output.columns:
            output[col] = np.nan
    bucket_records = []
    rng = np.random.default_rng(random_state)
    n_iter = max(1, int(bootstrap_iters))

    for pos, idx in output.groupby('pos').groups.items():
        pos_val = val[val.pos == pos].copy()
        if len(pos_val) >= min_n:
            fallback_q = _quantile_series(pos_val.actual_resid)
        else:
            fallback_q = global_q

        bucket_table, inner_edges = _fit_salary_resid_bucket_table(
            pos_val,
            fallback_q,
            q_cols=q_cols,
            pred_col=pred_col,
            min_n=min_n,
            min_bins=min_bins,
            max_bins=max_bins,
        )
        pred_values = output.loc[idx, pred_col].to_numpy()
        pred_q = np.zeros((len(pred_values), len(q_cols)))

        for i in range(n_iter):
            if i == 0 or len(pos_val) == 0:
                cur_val = pos_val
                cur_fallback_q = fallback_q
            else:
                sample_size = max(min_n, int(round(len(pos_val) * bootstrap_frac)))
                if not bootstrap_replace:
                    sample_size = min(sample_size, len(pos_val))
                sample_idx = rng.choice(len(pos_val), size=sample_size, replace=bootstrap_replace)
                cur_val = pos_val.iloc[sample_idx]
                cur_fallback_q = _quantile_series(cur_val.actual_resid) if len(pos_val) >= min_n else global_q

            cur_table, cur_edges = _fit_salary_resid_bucket_table(
                cur_val,
                cur_fallback_q,
                q_cols=q_cols,
                pred_col=pred_col,
                min_n=min_n,
                min_bins=min_bins,
                max_bins=max_bins,
            )
            pred_q += _predict_salary_resid_quantiles(
                cur_table,
                cur_edges,
                pred_values,
                cur_fallback_q,
                q_cols=q_cols,
                smooth=smooth,
            )

        pred_q = pred_q / n_iter
        pred_q = np.maximum.accumulate(pred_q, axis=1)
        output.loc[idx, q_cols] = pred_q

        bucket_record = bucket_table.reset_index().copy()
        bucket_record['pos'] = pos
        bucket_record['smooth'] = smooth
        bucket_record['bootstrap_iters'] = n_iter
        bucket_record['bootstrap_frac'] = bootstrap_frac
        bucket_record['bootstrap_replace'] = bootstrap_replace
        bucket_records.append(bucket_record)

    output[q_cols] = np.maximum.accumulate(output[q_cols].to_numpy(), axis=1)

    if 'is_keeper' in output.columns:
        keeper_mask = output.is_keeper.fillna(0) == 1
        output.loc[keeper_mask, q_cols] = 0

    bucket_table = pd.concat(bucket_records, axis=0).reset_index(drop=True)
    return output, bucket_table


def add_salary_legacy_uncertainty(pred_results, q_cols=SALARY_RESID_COLS):
    pred_results = pred_results.copy()
    pred_results[q_cols] = pred_results[q_cols].fillna(0)
    pred_results[q_cols] = np.maximum.accumulate(pred_results[q_cols].to_numpy(), axis=1)

    pred_results['min_score'] = np.maximum(1, pred_results.pred_salary + pred_results.salary_resid_5)
    pred_results['min_score'] = np.minimum(pred_results.pred_salary, pred_results.min_score)
    pred_results['max_score'] = np.maximum(pred_results.pred_salary, pred_results.pred_salary + pred_results.salary_resid_95)
    pred_results['std_dev'] = np.maximum(
        (pred_results.salary_resid_90 - pred_results.salary_resid_10) / 2.563,
        0.5,
    )

    if 'is_keeper' in pred_results.columns:
        keeper_mask = pred_results.is_keeper.fillna(0) == 1
        pred_results.loc[keeper_mask, q_cols] = 0
        pred_results.loc[keeper_mask, 'std_dev'] = 0.1
        pred_results.loc[keeper_mask, 'min_score'] = pred_results.loc[keeper_mask, 'pred_salary']
        pred_results.loc[keeper_mask, 'max_score'] = pred_results.loc[keeper_mask, 'pred_salary']

    return pred_results


def salary_resid_coverage(val_data, q_cols=SALARY_RESID_COLS):
    checks = [
        ('p5_p95', 'salary_resid_5', 'salary_resid_95', 0.90),
        ('p10_p90', 'salary_resid_10', 'salary_resid_90', 0.80),
        ('p25_p75', 'salary_resid_25', 'salary_resid_75', 0.50),
    ]
    records = []
    val_data = val_data.dropna(subset=['actual_salary', 'pred_salary']).copy()

    for pos, pos_val in val_data.groupby('pos'):
        for label, lower_col, upper_col, target in checks:
            lower = pos_val.pred_salary + pos_val[lower_col]
            upper = pos_val.pred_salary + pos_val[upper_col]
            records.append({
                'pos': pos,
                'interval': label,
                'target': target,
                'coverage': ((pos_val.actual_salary >= lower) & (pos_val.actual_salary <= upper)).mean(),
                'n': len(pos_val),
            })

    for label, lower_col, upper_col, target in checks:
        lower = val_data.pred_salary + val_data[lower_col]
        upper = val_data.pred_salary + val_data[upper_col]
        records.append({
            'pos': 'ALL',
            'interval': label,
            'target': target,
            'coverage': ((val_data.actual_salary >= lower) & (val_data.actual_salary <= upper)).mean(),
            'n': len(val_data),
        })

    return pd.DataFrame(records)


def cross_fit_salary_resid_quantiles(val_data, q_cols=SALARY_RESID_COLS, **kwargs):
    result = val_data.copy()
    result[q_cols] = np.nan

    for holdout_year, holdout_idx in val_data.groupby('year').groups.items():
        train_data = val_data[val_data.year != holdout_year]
        holdout_data = val_data.loc[holdout_idx].copy()
        if len(train_data) == 0:
            continue

        holdout_pred, _ = apply_salary_resid_quantiles(
            train_data,
            holdout_data,
            q_cols=q_cols,
            **kwargs,
        )
        result.loc[holdout_idx, q_cols] = holdout_pred[q_cols].to_numpy()

    return result


def salary_resid_bias_by_band(val_data):
    audit = val_data.dropna(subset=['actual_salary', 'pred_salary']).copy()
    audit['actual_resid'] = audit.actual_salary - audit.pred_salary
    audit['salary_band'] = pd.qcut(
        audit.pred_salary.rank(method='first'),
        q=min(4, len(audit)),
        labels=False,
        duplicates='drop',
    )
    return (
        audit.groupby(['pos', 'salary_band'])
        .agg(
            n=('player', 'size'),
            mean_pred_salary=('pred_salary', 'mean'),
            mean_actual_salary=('actual_salary', 'mean'),
            mean_resid=('actual_resid', 'mean'),
            median_resid=('actual_resid', 'median'),
        )
        .reset_index()
    )


def salary_quantile_sample_mean(pred_results, q_cols=SALARY_RESID_COLS, grid_size=1001):
    probs = np.array([0.00, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 1.00])
    resid_vals = pred_results[q_cols].to_numpy(dtype=float)
    q5, q10, q25, q75, q90, q95 = resid_vals.T
    q0 = (2 * q5) - q10
    q100 = (2 * q95) - q90
    knots = np.maximum.accumulate(
        np.column_stack([q0, q5, q10, q25, q75, q90, q95, q100]),
        axis=1,
    )

    sample_probs = (np.arange(grid_size) + 0.5) / grid_size
    sampled_resids = np.column_stack([
        np.interp(sample_probs, probs, player_knots)
        for player_knots in knots
    ])
    sampled_salaries = np.maximum(
        1,
        pred_results.pred_salary.to_numpy(dtype=float).reshape(1, -1)
        + sampled_resids,
    )
    return sampled_salaries.mean(axis=0)


def salary_market_budget_audit(pred_results, q_cols=SALARY_RESID_COLS):
    audit = pred_results.copy()
    audit['sample_mean_salary'] = salary_quantile_sample_mean(audit, q_cols=q_cols)
    records = []

    for year, year_data in audit.groupby('year'):
        context = year_data[
            ['keeper_count', 'keeper_spend', 'available_slots', 'available_budget']
        ].drop_duplicates()
        if len(context) != 1:
            raise ValueError(f'Inconsistent keeper budget context for {year}.')
        context = context.iloc[0]
        non_keepers = year_data[year_data.is_keeper.fillna(0) == 0]
        available_slots = int(context.available_slots)

        point_total = non_keepers.nlargest(available_slots, 'pred_salary').pred_salary.sum()
        sample_total = non_keepers.nlargest(
            available_slots,
            'sample_mean_salary',
        ).sample_mean_salary.sum()
        records.append({
            'year': year,
            'keeper_count': int(context.keeper_count),
            'keeper_spend': float(context.keeper_spend),
            'available_slots': available_slots,
            'available_budget': float(context.available_budget),
            'point_total': float(point_total),
            'point_gap': float(point_total - context.available_budget),
            'sample_mean_total': float(sample_total),
            'sample_mean_gap': float(sample_total - context.available_budget),
        })

    return pd.DataFrame(records)


def predict_locked_salary_ensemble(
    best_models,
    model_names,
    X_train,
    y_train,
    X_predict,
):
    predictions = []
    for model_name in model_names:
        for model in best_models[model_name]:
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            predictions.append(np.asarray(fitted.predict(X_predict), dtype=float))

    if not predictions:
        raise ValueError('No salary models were selected for backtest prediction.')
    return np.column_stack(predictions).mean(axis=1)


def build_salary_backtest_datasets(
    salary_pool,
    salary_features,
    training_rows,
    training_features,
    best_models,
    model_names,
    league,
    keeper_contract_rows=None,
    method_version=SALARY_METHOD_VERSION,
    calibration_start_year=SALARY_CALIBRATION_START_YEAR,
    backtest_start_year=SALARY_BACKTEST_START_YEAR,
):
    """Build rolling-data salary predictions under the locked current model spec.

    Each origin is fit only on labeled non-keeper rows from earlier seasons and
    normalized to the keeper-adjusted budget known before that auction. Model
    families and hyperparameters were selected by the current run, so this is a
    current-method retrospective, not a fresh historical method holdout.
    """
    prediction_frames = []
    available_years = sorted(
        int(year)
        for year in salary_pool.year.dropna().unique()
        if calibration_start_year <= int(year) < int(YEAR)
    )
    if not available_years:
        raise ValueError('No historical salary-pool years are available for backtesting.')

    for origin_year in available_years:
        train_mask = (
            training_rows.year.lt(origin_year)
            & training_rows.y_act.notna()
            & training_rows.is_keeper.fillna(0).eq(0)
        )
        pool_mask = salary_pool.year.eq(origin_year)
        if not train_mask.any() or not pool_mask.any():
            continue

        origin_predictions = predict_locked_salary_ensemble(
            best_models,
            model_names,
            training_features.loc[train_mask],
            training_rows.loc[train_mask, 'y_act'],
            salary_features.loc[pool_mask],
        )
        origin = salary_pool.loc[pool_mask, [
            'player',
            'pos',
            'year',
            'league',
            'salary',
            'base_salary_observed',
            'is_keeper',
            'y_act',
            'keeper_count',
            'keeper_spend',
            'available_slots',
            'available_budget',
            *SALARY_KEEPER_MARKET_FEATURE_COLS,
        ]].copy()
        origin = origin.rename(columns={'y_act': 'actual_salary'})
        origin['pred_salary'] = origin_predictions
        origin = finalize_salary_predictions(
            origin,
            normalization_mode='known_budget',
        )

        if keeper_contract_rows is not None:
            missing_keepers = keeper_contract_rows[
                keeper_contract_rows.year.eq(origin_year)
                & ~keeper_contract_rows.player.isin(origin.player)
            ].copy()
            if len(missing_keepers):
                missing_keepers = missing_keepers[[
                    'player',
                    'pos',
                    'year',
                    'league',
                    'salary',
                    'base_salary_observed',
                    'is_keeper',
                    'y_act',
                    'keeper_count',
                    'keeper_spend',
                    'available_slots',
                    'available_budget',
                    *SALARY_KEEPER_MARKET_FEATURE_COLS,
                ]].rename(columns={'y_act': 'actual_salary'})
                if missing_keepers.actual_salary.isna().any():
                    raise ValueError(
                        f'{origin_year} missing keeper contract salaries: '
                        f'{missing_keepers.loc[missing_keepers.actual_salary.isna(), "player"].tolist()}'
                    )
                missing_keepers['pred_salary'] = missing_keepers.actual_salary
                missing_keepers['pred_salary_raw'] = missing_keepers.actual_salary
                missing_keepers['pred_diff'] = (
                    missing_keepers.pred_salary - missing_keepers.salary
                )
                normalization_cols = [
                    'normalization_mode',
                    'normalization_method',
                    'normalization_source',
                    'normalization_slots',
                    'normalization_budget',
                    'pred_salary_scale',
                    'pred_salary_shift',
                    'normalization_floor',
                    'pre_normalized_total',
                    'post_normalized_total',
                ]
                for col in normalization_cols:
                    missing_keepers[col] = origin[col].iloc[0]
                origin = pd.concat([origin, missing_keepers], ignore_index=True)

        origin['training_through_year'] = origin_year - 1
        origin['model_spec_asof_year'] = int(YEAR)
        origin['method_version'] = method_version
        origin['prediction_mode'] = 'rolling_data_locked_current_spec'
        origin['data_rolling_origin'] = True
        origin['fresh_method_holdout'] = False
        origin['normalization_uses_target_actuals'] = False
        origin['candidate_pool_source'] = (
            'Model_Inputs ProjOnly universe + Simulation salary/actual left join'
        )
        origin['candidate_pool_rows'] = len(origin)
        origin['candidate_pool_covers_slots'] = (
            origin.loc[origin.is_keeper.fillna(0).eq(0)].shape[0]
            >= int(origin.available_slots.iloc[0])
        )
        origin['actual_salary_observed'] = origin.actual_salary.notna()
        prediction_frames.append(origin)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    quantified = []
    for origin_year in sorted(predictions.year.unique()):
        origin = predictions[predictions.year.eq(origin_year)].copy()
        residual_history = predictions[
            predictions.year.lt(origin_year)
            & predictions.actual_salary_observed
            & predictions.is_keeper.fillna(0).eq(0)
        ].copy()
        if len(residual_history):
            origin, _ = apply_salary_resid_quantiles(
                residual_history,
                origin,
                min_n=30,
                max_bins=15,
                smooth=True,
                bootstrap_iters=20,
                bootstrap_frac=1.0,
                bootstrap_replace=True,
                random_state=42 + int(origin_year),
            )
        else:
            origin[SALARY_RESID_COLS] = 0.0
        origin['resid_training_rows'] = len(residual_history)
        origin['resid_training_through_year'] = int(origin_year) - 1
        quantified.append(origin)

    predictions = pd.concat(quantified, ignore_index=True)
    predictions['actual_resid'] = predictions.actual_salary - predictions.pred_salary
    predictions['actual_resid_raw'] = (
        predictions.actual_salary - predictions.pred_salary_raw
    )
    predictions['base_salary'] = predictions.pop('salary')
    predictions['date_modified'] = pd.Timestamp.now(tz='UTC').isoformat()

    key = ['league', 'method_version', 'model_spec_asof_year', 'year', 'player']
    if predictions.duplicated(key).any():
        duplicates = predictions.loc[predictions.duplicated(key, keep=False), key]
        raise ValueError(f'Duplicate salary backtest keys:\n{duplicates}')
    if predictions.groupby('year').candidate_pool_covers_slots.first().eq(False).any():
        bad_years = predictions.groupby('year').candidate_pool_covers_slots.first()
        raise ValueError(
            f'Salary backtest pool cannot cover league slots: '
            f'{bad_years[~bad_years].index.tolist()}'
        )

    backtest = predictions[predictions.year.ge(backtest_start_year)].copy()
    validation = predictions[
        predictions.actual_salary_observed
        & predictions.is_keeper.fillna(0).eq(0)
    ].copy()
    validation['included_in_residual_evaluation'] = True
    return validation.reset_index(drop=True), backtest.reset_index(drop=True)


def write_salary_validation_datasets(
    validation,
    backtest,
    db_file,
    validation_table=SALARY_VALIDATION_TABLE,
    backtest_table=SALARY_BACKTEST_TABLE,
):
    """Replace one league/method/spec slice in both validation datasets."""
    slice_cols = ['league', 'method_version', 'model_spec_asof_year']
    validation_slice = validation[slice_cols].drop_duplicates()
    backtest_slice = backtest[slice_cols].drop_duplicates()
    if len(validation_slice) != 1 or not validation_slice.equals(backtest_slice):
        raise ValueError('Validation and backtest slices do not have one matching identity.')
    league, method_version, spec_year = validation_slice.iloc[0]

    for table_name in (validation_table, backtest_table):
        ensure_table_columns(
            db_file,
            table_name,
            SALARY_VALIDATION_AUDIT_SCHEMA,
        )

    conn = sqlite3.connect(db_file)
    try:
        with conn:
            for table_name, frame in (
                (validation_table, validation),
                (backtest_table, backtest),
            ):
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()
                if exists:
                    conn.execute(
                        f'''DELETE FROM "{table_name}"
                            WHERE league=? AND method_version=?
                              AND model_spec_asof_year=?''',
                        (league, method_version, int(spec_year)),
                    )
                    existing_cols = {
                        row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')
                    }
                    missing_cols = set(frame.columns) - existing_cols
                    if missing_cols:
                        raise ValueError(
                            f'{table_name} is missing schema columns: {sorted(missing_cols)}'
                        )
                frame.to_sql(table_name, conn, if_exists='append', index=False)
                index_name = f'ux_{table_name.lower()}_slice_player'
                conn.execute(
                    f'''CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}"
                        ON "{table_name}"
                        (league, method_version, model_spec_asof_year, year, player)'''
                )
    finally:
        conn.close()

    print(
        f'Saved {len(validation)} {validation_table} rows and '
        f'{len(backtest)} {backtest_table} rows.'
    )


salaries = get_salaries()
salaries['league'] = LEAGUE

total_spent = salaries.groupby('year').agg({'actual_salary': 'sum'}).reset_index().rename(columns={'actual_salary': 'total_spent'})
salaries = pd.merge(salaries, total_spent, on=['year'], how='left')
salaries.loc[salaries.year==YEAR, 'total_spent'] = LEAGUE_BUDGET
salaries['fraction_spent'] = salaries.total_spent / LEAGUE_BUDGET

salaries = fill_ty_keepers(salaries, ty_keepers)
salaries = add_keeper_budget_context(salaries)
salaries = calc_inflation(salaries)

# Model_Inputs is the pre-auction candidate universe. Joining from it prevents
# a manually truncated ESPN salary copy (or the realized auction roster) from
# defining who was considered available in a historical replay.
year_context_cols = [
    'total_spent',
    'fraction_spent',
    'keeper_count',
    'keeper_spend',
    'keeper_market_value',
    'keeper_source_market_value',
    'keeper_source_values_observed',
    'keeper_contract_discount',
    'keeper_pool_base_budget',
    'keeper_pool_inflation',
    'available_slots',
    'available_budget',
    'source_market_total',
    'source_nonkeeper_market_total',
    'value',
    'inflation',
]
year_context = (
    salaries[['year', *year_context_cols]]
    .drop_duplicates('year')
    .set_index('year')
)
adp_stats = add_ensemble_projection_features(get_adp())
keeper_position_lookup = (
    adp_stats.groupby('player').pos
    .agg(lambda values: values.mode().iloc[0])
    .rename('pos')
    .reset_index()
)
keeper_contract_rows = salaries[salaries.is_keeper.eq(1)].copy()
keeper_contract_rows = pd.merge(
    keeper_contract_rows,
    keeper_position_lookup,
    on='player',
    how='left',
)
if keeper_contract_rows.pos.isna().any():
    raise ValueError(
        'Unable to infer positions for keeper contracts: '
        f'{keeper_contract_rows.loc[keeper_contract_rows.pos.isna(), "player"].tolist()}'
    )
keeper_contract_rows['base_salary_observed'] = keeper_contract_rows.salary.notna()
keeper_contract_rows['salary'] = keeper_contract_rows.salary.fillna(0)
keeper_contract_rows = keeper_contract_rows.rename(columns={'actual_salary': 'y_act'})
salaries = pd.merge(salaries, adp_stats, on=['player', 'year'], how='right')
for col in year_context_cols:
    salaries[col] = salaries[col].fillna(salaries.year.map(year_context[col]))
salaries['league'] = salaries.league.fillna(LEAGUE)
salaries['is_keeper'] = salaries.is_keeper.fillna(0)
salaries['base_salary_observed'] = salaries.salary.notna()
salaries['salary'] = salaries.salary.fillna(0)
salaries = add_rookie(salaries)
salaries = add_keeper_market_salary_features(salaries)

# Some historical keeper contracts are not present in that origin's projection
# universe. Their model prediction is discarded, but they still need complete
# keeper-market audit fields when appended to the backtest output.
keeper_contract_rows['source_salary_floor'] = (
    keeper_contract_rows.salary.clip(lower=SALARY_NORMALIZATION_FLOOR)
)
keeper_contract_rows['log_source_salary'] = np.log1p(
    keeper_contract_rows.source_salary_floor
)
keeper_contract_rows['keeper_adjusted_source_salary'] = (
    SALARY_NORMALIZATION_FLOOR
    + keeper_contract_rows.keeper_pool_inflation
    * (
        keeper_contract_rows.source_salary_floor
        - SALARY_NORMALIZATION_FLOOR
    )
)
keeper_contract_rows['source_market_scale'] = keeper_contract_rows.year.map(
    salaries.groupby('year').source_market_scale.first()
)
keeper_contract_rows['budget_adjusted_source_salary'] = (
    keeper_contract_rows.y_act.fillna(keeper_contract_rows.source_salary_floor)
)
keeper_contract_rows['keeper_adjusted_source_diff'] = (
    keeper_contract_rows.keeper_adjusted_source_salary
    - keeper_contract_rows.source_salary_floor
)
keeper_contract_rows['budget_adjusted_source_diff'] = (
    keeper_contract_rows.budget_adjusted_source_salary
    - keeper_contract_rows.source_salary_floor
)

salaries = add_pos_keeper_val(salaries)

salary_pool = add_salary_model_features_by_keeper_availability(salaries)
salaries = drop_keepers(salaries)
salaries = add_salary_model_features_by_keeper_availability(salaries)
salaries = remove_outliers(salaries)

#%%
salaries = salaries.rename(columns={'actual_salary': 'y_act'})
salaries = salaries.sort_values(by='year').reset_index(drop=True)
salaries['team'] = 'placeholder'
salaries['week'] = 1
salaries['game_date'] = salaries.year
salary_pool = salary_pool.rename(columns={'actual_salary': 'y_act'})
salary_pool = salary_pool.sort_values(by=['year', 'player']).reset_index(drop=True)
salary_pool['team'] = 'placeholder'
salary_pool['week'] = 1
salary_pool['game_date'] = salary_pool.year

skm = SciKitModel(salaries, r2_wt=0, mse_wt=1)
X = build_salary_model_matrix(salaries)
y = salaries.y_act
X_pool = build_salary_model_matrix(salary_pool, feature_columns=X.columns)


X_train = X[X.year != YEAR]
y_train = y.iloc[X_train.index].reset_index(drop=True); X_train.reset_index(drop=True, inplace=True)   

X_test = X[X.year == YEAR].reset_index(drop=True)
y_test = y.iloc[X_test.index].reset_index(drop=True); X_test.reset_index(drop=True, inplace=True)

print(pd.concat([X,y], axis=1).corr()['y_act'].sort_values().head(20))
pd.concat([X,y], axis=1).corr()['y_act'].sort_values(ascending=False).head(20)

#%%
corr_df = salaries.copy()
corr_df['base_salary_resid'] = corr_df.y_act - corr_df.salary
display(
    corr_df[
        [
            'base_salary_resid',
            'budget_adjusted_source_salary',
            'avg_pick_log',
            'ensemble_pred_ppg',
            'ensemble_vs_price_gap',
            'ensemble_pred_resid_90',
            'pos_proj_points_share',
            'pos_proj_rush_att_share',
            'year_exp',
            'is_rookie',
        ]
    ].corr()['base_salary_resid'].sort_values()
)


#%%
from sklearn.ensemble import VotingRegressor

baseline_data = salaries[salaries.year!=YEAR]

inf_baseline = mean_squared_error(baseline_data.y_act*baseline_data.inflation, baseline_data.salary)
inf_baseline_r2 = r2_score(baseline_data.y_act*baseline_data.inflation, baseline_data.salary)
baseline = mean_squared_error(baseline_data.y_act, baseline_data.salary)
baseline_r2 = r2_score(baseline_data.y_act, baseline_data.salary)

print('Inflation Baseline',  round(inf_baseline, 3), round(inf_baseline_r2, 3))
print('Baseline',  round(baseline, 3), round(baseline_r2, 3))

import optuna

# loop through each potential model
optuna.logging.set_verbosity(optuna.logging.WARNING)
best_models = {}
scores = {}
model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 
              'rf', 'gbmh', 'huber', 'cb', 'mlp', 'et']
i = 0

for m in model_list:
    i += 1
    print('\n============\n')
    print(m)
    

    pipe = skm.model_pipe([ skm.piece('feature_drop'),
                            skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece('k_best'),
                            skm.piece(m)
                            ])
    pipe.set_params(
        feature_drop__drop_cols=SALARY_MODEL_SPLIT_COLUMNS,
    )

    params = skm.default_params(pipe, 'optuna')
    # ``skmodel`` treats FeatureDrop as a tunable step by default and includes
    # ``None`` as an option. This drop is structural, not a hyperparameter:
    # year/game_date must remain available for rolling splits but must never
    # reach the fitted estimator.
    params.pop('feature_drop__drop_cols', None)
    params['random_sample__frac'] = [
        'cat', [0.6, 0.75, 0.9, 1.0]
    ]
    params['random_sample__seed'] = ['int', 0, 20]
    params['k_best__k'] = [
        'cat', [6, 8, 10, 'all']
    ]

    study = optuna.create_study(direction='minimize')
    best_models_cur, oof_data, _, _ = skm.time_series_cv(
                                                        pipe, X_train, y_train, params,
                                                        n_iter=SALARY_OPTUNA_ITERATIONS,
                                                        n_splits=5, alpha='',
                                                        col_split='game_date', time_split=2022,
                                                        bayes_rand='optuna', proba=False, trials=study,
                                                        random_seed=(i+7)*19+(i*12)+6,
                                                        optuna_timeout=SALARY_OPTUNA_TIMEOUT)

    if i == 1:
        all_pred = oof_data['full_hold'][['player', 'year', 'pred', 'y_act']]
        all_pred = all_pred.rename(columns={'pred': m})
    else:
        all_pred = pd.merge(all_pred, oof_data['full_hold'][['player', 'year', 'pred']], on=['player', 'year'])
        all_pred = all_pred.rename(columns={'pred': m})

    scores[m] = (2*oof_data['scores'][-1]+oof_data['scores'][0])/3
    best_models[m] = best_models_cur


#%%


scores_df = pd.DataFrame(scores, index=[0]).T.rename(columns={0: 'mse'})
scores_df = scores_df.sort_values(by='mse', ascending=True)

all_models = []
all_scores_mse = []
all_scores_r2 = []
for m in scores_df.index:
    all_models.append(m)
    
    preds = all_pred[[c for c in all_pred.columns if c in all_models]].mean(axis=1)
    preds = pd.Series(preds, name='pred_salary')
    val_data = pd.concat([all_pred[['player', 'year', 'y_act']], preds], axis=1)
 
    mf.show_scatter_plot(val_data.y_act, val_data.pred_salary)
    ens_mse = np.round(mean_squared_error(val_data.y_act, val_data.pred_salary), 3)
    ens_r2 = np.round(r2_score(val_data.y_act, val_data.pred_salary), 4)
    all_scores_mse.append(ens_mse)
    all_scores_r2.append(ens_r2)
    print('MSE:', ens_mse, 'R2:', ens_r2)

best_score_mse = np.argmin(all_scores_mse)
best_score_r2 = np.argmax(all_scores_r2)

best_models_names_r2 = scores_df.iloc[:best_score_r2+1].index
print(best_models_names_r2)
print(all_scores_r2[best_score_r2])

best_models_names_mse = scores_df.iloc[:best_score_mse+1].index
print(best_models_names_mse)
print(all_scores_mse[best_score_mse])

preds = all_pred[[c for c in all_pred.columns if c in best_models_names_mse]].mean(axis=1)
preds = pd.Series(preds, name='pred_salary')
val_data = pd.concat([all_pred[['player', 'year', 'y_act']], preds], axis=1)
val_data = pd.merge(
    salaries[[
        'player', 'year', 'pos', 'salary', 'is_keeper',
        'keeper_count', 'keeper_spend', 'available_slots', 'available_budget',
        *SALARY_KEEPER_MARKET_FEATURE_COLS,
    ]],
    val_data,
    on=['player', 'year'],
)
val_data = val_data.rename(columns={'y_act': 'actual_salary'})
val_data = finalize_salary_predictions(val_data)

#%%

final_pred = pd.DataFrame()
for m in best_models_names_mse:
    for m_sub in best_models[m]:
        m_sub.fit(X_train, y_train)
        cur_pred =  m_sub.predict(X_test)
        final_pred = pd.concat([final_pred, pd.Series(cur_pred, name=m)], axis=1)
        final_pred = pd.concat([final_pred,], axis=1)

pred_sal = final_pred.mean(axis=1)


#%%

pred_results = pd.concat([salaries.loc[salaries.year==YEAR,[
                              'player_key', 'player', 'pos', 'year', 'salary',
                              'is_keeper', 'y_act',
                              'salary_population_source',
                              'ensemble_uncertainty_feature_source',
                              'keeper_count', 'keeper_spend', 'available_slots', 'available_budget',
                              *SALARY_KEEPER_MARKET_FEATURE_COLS,
                          ]].reset_index(drop=True),
                          pd.Series(pred_sal, name='pred_salary')], axis=1)
pred_results = pred_results.rename(columns={'y_act': 'actual_salary'})

pred_results = finalize_salary_predictions(pred_results, show_results=True)
pred_results.iloc[:50]

#%%

pred_results, salary_resid_bucket_table = apply_salary_resid_quantiles(
    val_data,
    pred_results,
    min_n=30,
    max_bins=15,
    smooth=True,
    bootstrap_iters=50,
    bootstrap_frac=1.0,
    bootstrap_replace=True,
    random_state=42,
)

val_data_resid_cv = cross_fit_salary_resid_quantiles(
    val_data,
    min_n=30,
    max_bins=15,
    smooth=True,
    bootstrap_iters=20,
    bootstrap_frac=1.0,
    bootstrap_replace=True,
    random_state=42,
)

salary_validation_rows, salary_backtest_rows = build_salary_backtest_datasets(
    salary_pool=salary_pool,
    salary_features=X_pool,
    training_rows=salaries,
    training_features=X,
    best_models=best_models,
    model_names=best_models_names_mse,
    league=LEAGUE,
    keeper_contract_rows=keeper_contract_rows,
)
write_salary_validation_datasets(
    salary_validation_rows,
    salary_backtest_rows,
    Path(db_path) / 'Validations.sqlite3',
)

if VALIDATION_DATASETS_ONLY:
    print('Validation-only run complete; skipping live-output diagnostics.')
    raise SystemExit(0)

print('OOF salary residual bias by position and predicted-salary band')
display(salary_resid_bias_by_band(val_data))
print('Leave-one-year-out salary residual interval coverage')
display(salary_resid_coverage(val_data_resid_cv))
print('Current salary market budget audit')
display(salary_market_budget_audit(pred_results))
print('Rolling salary backtest pool coverage')
display(
    salary_backtest_rows.groupby('year')
    .agg(
        candidate_rows=('player', 'size'),
        observed_actual_rows=('actual_salary_observed', 'sum'),
        available_slots=('available_slots', 'first'),
        available_budget=('available_budget', 'first'),
        pred_market_total=('post_normalized_total', 'first'),
        residual_training_rows=('resid_training_rows', 'first'),
    )
    .reset_index()
)
display(salary_resid_bucket_table)

pred_results = add_salary_legacy_uncertainty(pred_results)
pred_results.sort_values(by='std_dev', ascending=False).iloc[:25]

#%%
pred_results['league'] = LEAGUE + 'pred'
pred_results['salary_method_version'] = SALARY_METHOD_VERSION
output = pred_results[[
    'player_key',
    'player',
    'pred_salary',
    'year',
    'league',
    'std_dev',
    'min_score',
    'max_score',
    *SALARY_RESID_COLS,
    'salary_population_source',
    'ensemble_uncertainty_feature_source',
    'salary_method_version',
]]
output = output.rename(columns={'pred_salary': 'salary'})
canonical_salary_labels = dm.read(
    f'''SELECT player_key, player
          FROM Final_Predictions_Resid
         WHERE version='{LEAGUE}'
               AND year={YEAR}
               AND dataset='final_ensemble' ''',
    'Simulation',
).set_index('player_key').player
output['player'] = output.player_key.map(canonical_salary_labels)
if output.player.isna().any():
    raise ValueError(
        'Salary output could not restore canonical production labels.'
    )

if VALIDATION_DATASETS_ONLY:
    print('Validation-only run: leaving Simulation.Salaries_Pred unchanged.')
else:
    ensure_table_columns(
        Path(db_path) / 'Simulation.sqlite3',
        'Salaries_Pred',
        {
            **SALARY_RESID_SCHEMA,
            'player_key': 'TEXT',
            'salary_population_source': 'TEXT',
            'ensemble_uncertainty_feature_source': 'TEXT',
            'salary_method_version': 'TEXT',
        },
    )
    if output.player_key.isna().any() or output.player_key.duplicated().any():
        raise ValueError(
            'Current salary output lacks unique canonical player keys.'
        )
    dm.delete_from_db(
        'Simulation',
        'Salaries_Pred',
        f"year={YEAR} AND league='{LEAGUE}pred'",
        create_backup=not IS_STAGED_DATABASE_RUN,
    )
    dm.write_to_db(output, 'Simulation', 'Salaries_Pred', 'append')

#%%

if VALIDATION_DATASETS_ONLY or IS_STAGED_DATABASE_RUN:
    print('Validation-only run: skipping the Fantasy_Football_App database copy.')
else:
    src = Path(root_path) / 'Data' / 'Databases' / 'Simulation.sqlite3'
    dst = Path(root_path).parent / 'Fantasy_Football_App' / 'app' / 'Simulation.sqlite3'
    generated_salary_tables = [
        'Salaries',
        'Salaries_Pred',
        'League_Keepers',
    ]
    if not dst.parent.exists():
        print(f'Skipping app DB copy; destination folder does not exist: {dst.parent}')
    elif not dst.exists():
        raise FileNotFoundError(
            'Auction app Simulation database does not exist; refusing to '
            f'create it from the modeling database: {dst}'
        )
    else:
        # The auction database owns UI/evidence/cache tables that do not belong
        # to this repository. Synchronize only salary-owned generated tables.
        with sqlite3.connect(dst) as app_connection:
            app_connection.execute(
                'ATTACH DATABASE ? AS salary_source',
                (str(src),),
            )
            app_connection.execute('BEGIN IMMEDIATE')
            try:
                for table_name in generated_salary_tables:
                    create_sql = app_connection.execute(
                        '''SELECT sql
                             FROM salary_source.sqlite_master
                            WHERE type='table' AND name=?''',
                        (table_name,),
                    ).fetchone()
                    if create_sql is None:
                        raise ValueError(
                            f'Missing generated salary table: {table_name}'
                        )
                    app_connection.execute(
                        f'DROP TABLE IF EXISTS main."{table_name}"'
                    )
                    app_connection.execute(create_sql[0])
                    app_connection.execute(
                        f'INSERT INTO main."{table_name}" '
                        f'SELECT * FROM salary_source."{table_name}"'
                    )
                app_connection.commit()
            except Exception:
                app_connection.rollback()
                raise
        print(
            f'Synchronized {len(generated_salary_tables)} generated salary '
            f'tables to {dst}'
        )

# %%

if IS_STAGED_DATABASE_RUN:
    print('Staged salary build complete; skipping notebook diagnostics.')
    raise SystemExit(0)

pred = dm.read("SELECT * FROM Salaries_Pred WHERE year=2025 AND league='betapred'", 'Simulation')
actual = dm.read("SELECT * FROM Actual_Salaries WHERE year=2025 AND league='beta' AND is_keeper=0", 'Simulation')
combined = pd.merge(pred[['player', 'salary']], actual[['player', 'actual_salary']], on='player')
print(r2_score(combined.actual_salary, combined.salary))
print(mean_squared_error(combined.actual_salary, combined.salary))
combined.plot.scatter(x='salary', y='actual_salary')
combined['error'] = combined.actual_salary - combined.salary
display(combined.sort_values(by='error').iloc[:40])
display(combined.sort_values(by='error').iloc[-40:])

# %%
pred = dm.read("SELECT * FROM Salaries WHERE year=2025 AND league='nv'", 'Simulation')
actual = dm.read("SELECT * FROM Actual_Salaries WHERE year=2025 AND league='nv' AND is_keeper=0", 'Simulation')
combined = pd.merge(pred[['player', 'salary']], actual[['player', 'actual_salary']], on='player')
print(r2_score(combined.actual_salary, combined.salary))
print(mean_squared_error(combined.actual_salary, combined.salary))
combined.plot.scatter(x='salary', y='actual_salary')
combined['error'] = combined.actual_salary - combined.salary
display(combined.sort_values(by='error').iloc[:40])
display(combined.sort_values(by='error').iloc[-40:])
# %%
