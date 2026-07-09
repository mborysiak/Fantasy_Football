
#%%

# # Reading in Old Salary Data

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import zModel_Functions as mf
import joblib
from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc
from skmodel import SciKitModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import StandardScaler
from zFix_Standard_Dev import *

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
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set core path
PATH = f'{root_path}/Data/'
YEAR = 2026
LEAGUE = 'beta'

SALARY_RESID_ALPHAS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
SALARY_RESID_COLS = [f'salary_resid_{int(round(alpha * 100))}' for alpha in SALARY_RESID_ALPHAS]
SALARY_RESID_SCHEMA = {col: 'REAL' for col in SALARY_RESID_COLS}


ty_keepers = {
    # 'Bucky Irving': [12],
    'Chase Brown': [34],

    # 'Brock Bowers': [21],
    # 'Kyren Williams': [26],

    # 'James Cook': [54],
    # 'Jayden Daniels': [13],

    # 'Nico Collins': [27],
    # 'Chuba Hubbard': [13],

    # 'Ladd Mcconkey': [19],

    # 'Josh Allen': [46],
    # 'Josh Jacobs': [57],

    # 'Jaxon Smith Njigba': [27],
    # 'Christian Mccaffrey': [57],

    # 'Brian Thomas': [13],
    # 'Derrick Henry': [88],

    # 'Stefon Diggs': [11],
    # 'Davante Adams': [49],

    # 'Jerry Jeudy': [11],   
    # 'Courtland Sutton': [14]
}

# ty_keepers = {
#     "Devon Achane": [27],
#     'Jayden Daniels': [36],

#     'Baker Mayfield': [13],
#     'Bucky Irving': [20],

#     'Bo Nix': [11],
#     'Tee Higgins': [27],

#     'Rashee Rice': [10],
#     'Chris Olave': [13],

#     'Bijan Robinson': [107],
#     'Brock Bowers': [11],
    
#     'Brian Thomas': [12],
#     'Joe Burrow': [29],

#     "Ja'Marr Chase": [71],
#     'Ladd Mcconkey': [11],
    
#     'Chase Brown': [23],

#     'Josh Jacobs': [42]
# }

ty_keepers = pd.DataFrame(ty_keepers)
ty_keepers = ty_keepers.T.reset_index()
ty_keepers.columns = ['player', 'ty_keeper_sal']
ty_keepers['year'] = YEAR


#%%

#=================
# Load salaries from ESPN into database
#=================

# read in csv file of raw copy-pasted data with bad formatting from ESPN
df = pd.read_csv(f'{PATH}/OtherData/Salaries/salaries_{YEAR}_{LEAGUE}.csv', header=None)

def scrape_values(df):
    '''
    This function will scrape a copy-paste of the ESPN salary information (paste special->text)
    into a CSV when the data is in a single long row
    '''
    is_dollar = False
    names = []
    values = []
    for _, v in df.iterrows():
        
        # get the value in the row
        v = v[0]

        # names are longer than other stats in the sheet, so filter based on length
        if len(v) > 7:
            names.append(v)

        # the code below is a trigger for a dollar sign, which
        # signals salary is coming up. if trigger is active, append salary
        if is_dollar:
            values.append(int(v))

        # set the dollar sign trigger based on the current value for next iteration
        if v == '$': is_dollar=True
        else: is_dollar=False
    
    # create a dataframe of the resultant lists
    df = pd.DataFrame([names, values]).T
    df.columns = ['player', 'salary']

    return df

salaries = scrape_values(df)
salaries['year'] = YEAR
salaries['league'] = LEAGUE
salaries = salaries.dropna().reset_index(drop=True)
salaries.player = salaries.player.apply(dc.name_clean)

dm.delete_from_db('Simulation', 'Salaries', f"year='{YEAR}' AND league='{LEAGUE}'")
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

def get_adp():
    all_stats = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE']:
        print(pos)

        stats = dm.read(f'''SELECT player, year, avg_pick, avg_pick_log, avg_proj_points,
                                   avg_pos_rank, year_exp, avg_proj_points_exp_diff
                            FROM {pos}_{YEAR}_ProjOnly
                         ''', 'Model_Inputs')
        stats['pos'] = pos
        all_stats = pd.concat([all_stats, stats], axis=0)

    all_stats['log_avg_points'] = np.log(all_stats.avg_proj_points)
    return all_stats

def fill_ty_keepers(salaries, ty_keepers):
    salaries = pd.merge(salaries, ty_keepers, on=['player', 'year'], how='left')
    salaries.loc[(salaries.year==YEAR) & ~(salaries.ty_keeper_sal.isnull()), 'actual_salary'] = \
        salaries.loc[(salaries.year==YEAR) & ~(salaries.ty_keeper_sal.isnull()), 'ty_keeper_sal']
    salaries.loc[(salaries.year==YEAR) & ~(salaries.ty_keeper_sal.isnull()), 'is_keeper'] = 1

    return salaries.drop('ty_keeper_sal', axis=1)

def get_salaries():
    actual_sal = dm.read(f'''SELECT *
                            FROM Actual_Salaries 
                            WHERE League='{LEAGUE}'
                                  AND year <= {YEAR} ''', 'Simulation')
    base_sal = dm.read(f'''SELECT player, salary, year
                                FROM Salaries 
                                WHERE League='{LEAGUE}'
                                 AND year <= {YEAR} ''', 'Simulation')
    salaries = pd.merge(actual_sal, base_sal, on=['player', 'year'], how='right')
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
    inflation['inflation'] = 1 + (inflation.value / 3600)

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

def drop_keepers(salaries):
    salaries = salaries[(salaries.is_keeper==0) | (salaries.year==YEAR)].reset_index(drop=True)
    salaries = salaries[(salaries.year==YEAR) | (~salaries.actual_salary.isnull())].reset_index(drop=True)
    return salaries

def add_salary_pos_rank(salaries):
    salaries = salaries.sort_values(by=['year', 'salary'], ascending=[True, False])
    salaries['sal_rank'] = salaries.groupby('year').cumcount().values

    salaries = salaries.sort_values(by=['year', 'pos', 'salary'],
                                ascending=[True, True, False]).reset_index(drop=True)
    salaries['pos_rank'] = salaries.groupby(['year', 'pos']).cumcount().values

    return salaries

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


def finalize_salary_predictions(pred_results, avg_spent, top_n=156, show_results=False):
    pred_results = pred_results.copy()
    pred_results['is_keeper'] = pred_results.is_keeper.fillna(0)
    pred_results['pred_salary'] = pd.to_numeric(pred_results.pred_salary, errors='coerce')
    pred_results['salary'] = pd.to_numeric(pred_results.salary, errors='coerce')
    pred_results.loc[pred_results.pred_salary < 1, 'pred_salary'] = 1
    pred_results['pred_salary'] = pred_results.pred_salary.fillna(1).astype(int)

    processed = []
    for year, year_results in pred_results.groupby('year', sort=False):
        year_results = year_results.sort_values(by='salary', ascending=False).reset_index(drop=True)

        keeper_mask = (year_results.is_keeper == 1) & year_results.actual_salary.notna()
        year_results.loc[keeper_mask, 'pred_salary'] = year_results.loc[keeper_mask, 'actual_salary']
        year_results['pred_salary'] = year_results.pred_salary.astype(int)
        year_results['pred_diff'] = year_results.pred_salary - year_results.salary

        total_diff = year_results.pred_diff.sum()
        total_from_available = year_results.iloc[:top_n].pred_salary.sum() - avg_spent
        total_off = np.max([0, -total_from_available])

        if show_results:
            print(f'{year} Total Diff:', total_diff)
            print(f'{year} Total from available:', total_from_available)
            display(year_results.iloc[:50])
            display(year_results[np.abs(year_results.pred_diff) > 4].sort_values(by='pred_diff', ascending=False))

        non_keeper_count = len(year_results[year_results.is_keeper == 0])
        extra_per_player = np.ceil(total_off / non_keeper_count) if non_keeper_count > 0 else 0
        year_results.loc[year_results.is_keeper == 0, 'pred_salary'] += extra_per_player
        year_results['pred_salary'] = year_results.pred_salary.astype(int)
        year_results['pred_diff'] = year_results.pred_salary - year_results.salary

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


salaries = get_salaries()

total_spent = salaries.groupby('year').agg({'actual_salary': 'sum'}).reset_index().rename(columns={'actual_salary': 'total_spent'})
salaries = pd.merge(salaries, total_spent, on=['year'], how='left')
salaries.loc[salaries.year==YEAR, 'total_spent'] = 3576
salaries['fraction_spent'] = salaries.total_spent / 3576

salaries = add_rookie(salaries)
adp_stats = get_adp()
salaries = pd.merge(salaries, adp_stats, on=['player', 'year'])
salaries = fill_ty_keepers(salaries, ty_keepers)
salaries = calc_inflation(salaries)
 
salaries = add_pos_keeper_val(salaries)

salaries = drop_keepers(salaries)
salaries = add_salary_pos_rank(salaries)

salaries = remove_outliers(salaries)

# salaries = pd.concat([salaries, pd.get_dummies(salaries.year, prefix='year')], axis=1)
salaries['young_player'] = (salaries.year_exp < 2).astype('int')
salaries['rookie_rank'] = salaries.is_rookie * salaries.avg_pick
salaries['old_player'] = (salaries.year_exp > 5).astype('int')

salaries = salaries.sort_values(by=['year', 'pos', 'salary'], ascending=[True, True, False]).reset_index(drop=True)
salaries['next_guy_sal'] = salaries.groupby(['pos', 'year']).salary.shift(-1)
salaries['next_guy_sal_diff'] = salaries.salary - salaries.next_guy_sal

salaries['guy_above_sal'] = salaries.groupby(['pos', 'year']).salary.shift(1)
salaries['guy_above_sal_diff'] = salaries.salary - salaries.guy_above_sal
salaries = salaries.drop(['next_guy_sal', 'guy_above_sal'], axis=1)

salaries.loc[salaries.next_guy_sal_diff.isnull(), ['next_guy_sal_diff']] = 0
salaries.loc[salaries.guy_above_sal_diff.isnull(), [ 'guy_above_sal_diff']] = 0

salaries['pts_per_dollar'] = salaries.avg_proj_points / (salaries.salary+1)


#%%
salaries = salaries.rename(columns={'actual_salary': 'y_act'})
salaries = salaries.sort_values(by='year').reset_index(drop=True)
salaries['team'] = 'placeholder'
salaries['week'] = 1
salaries['game_date'] = salaries.year
skm = SciKitModel(salaries, r2_wt=0, mse_wt=1)

X, y = skm.Xy_split('y_act', to_drop=['player', 'team', 'week', 'league'])
X = pd.concat([X, pd.get_dummies(X.pos)], axis=1).drop('pos', axis=1)

X['qb_proj'] = X.QB * X.avg_proj_points
X['rb_proj'] = X.RB * X.avg_proj_points
X['wr_proj'] = X.WR * X.avg_proj_points
X['te_proj'] = X.TE * X.avg_proj_points

X['qb_pick'] = X.QB * X.avg_pick
X['rb_pick'] = X.RB * X.avg_pick
X['wr_pick'] = X.WR * X.avg_pick
X['te_pick'] = X.TE * X.avg_pick


X['qb_rank'] = X.QB * X.pos_rank
X['rb_rank'] = X.RB * X.pos_rank
X['wr_rank'] = X.WR * X.pos_rank
X['te_rank'] = X.TE * X.pos_rank


X_train = X[X.year != YEAR]
y_train = y.iloc[X_train.index].reset_index(drop=True); X_train.reset_index(drop=True, inplace=True)   

X_test = X[X.year == YEAR].reset_index(drop=True)
y_test = y.iloc[X_test.index].reset_index(drop=True); X_test.reset_index(drop=True, inplace=True)

print(pd.concat([X,y], axis=1).corr()['y_act'].sort_values().head(20))
pd.concat([X,y], axis=1).corr()['y_act'].sort_values(ascending=False).head(20)


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
    

    pipe = skm.model_pipe([ skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece('k_best'),
                            skm.piece(m)
                            ])

    params = skm.default_params(pipe, 'optuna')
    params['random_sample__frac'] = ['real', 0.2, 1]
    params['k_best__k'] = ['int', 1, X_train.shape[1]]

    study = optuna.create_study(direction='minimize')
    best_models_cur, oof_data, _, _ = skm.time_series_cv(pipe, X_train, y_train, params, n_iter=25, 
                                                        n_splits=5, alpha='',
                                                        col_split='game_date', time_split=2022,
                                                        bayes_rand='optuna', proba=False, trials=study,
                                                        random_seed=(i+7)*19+(i*12)+6, optuna_timeout=60)

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
    salaries[['player', 'year', 'pos', 'salary', 'is_keeper']],
    val_data,
    on=['player', 'year'],
)
val_data = val_data.rename(columns={'y_act': 'actual_salary'})
avg_spent = salaries.groupby('year').total_spent.mean().mean()
val_data = finalize_salary_predictions(val_data, avg_spent)

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

pred_results = pd.concat([salaries.loc[salaries.year==YEAR,['player', 'pos', 'year', 'salary', 'is_keeper', 'y_act']].reset_index(drop=True), 
                          pd.Series(pred_sal, name='pred_salary')], axis=1)
pred_results = pred_results.rename(columns={'y_act': 'actual_salary'})

pred_results = finalize_salary_predictions(pred_results, avg_spent, show_results=True)
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

val_data_resid, _ = apply_salary_resid_quantiles(
    val_data,
    val_data,
    min_n=30,
    max_bins=15,
    smooth=True,
    bootstrap_iters=50,
    bootstrap_frac=1.0,
    bootstrap_replace=True,
    random_state=42,
)
display(salary_resid_coverage(val_data_resid))
display(salary_resid_bucket_table)

pred_results = add_salary_legacy_uncertainty(pred_results)
pred_results.sort_values(by='std_dev', ascending=False).iloc[:25]

#%%
pred_results['league'] = LEAGUE + 'pred'
output = pred_results[['player', 'pred_salary', 'year', 'league', 'std_dev', 'min_score', 'max_score', *SALARY_RESID_COLS]]
output = output.rename(columns={'pred_salary': 'salary'})

ensure_table_columns(Path(db_path) / 'Simulation.sqlite3', 'Salaries_Pred', SALARY_RESID_SCHEMA)
dm.delete_from_db('Simulation', 'Salaries_Pred', f"year={YEAR} AND league='{LEAGUE}pred'", create_backup=True)
dm.write_to_db(output, 'Simulation', 'Salaries_Pred', 'append')

#%%

import shutil

src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = Path(root_path).parent / 'Fantasy_Football_App' / 'app' / 'Simulation.sqlite3'
if dst.parent.exists():
    shutil.copyfile(src, dst)
else:
    print(f'Skipping app DB copy; destination folder does not exist: {dst.parent}')

# %%

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
