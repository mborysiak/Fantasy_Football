
#%%

# # Reading in Old Salary Data

import pandas as pd
import numpy as np
import sqlite3
import zModel_Functions as mf
import joblib
from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc
from skmodel import SciKitModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

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
YEAR = 2024
LEAGUE = 'beta'


ty_keepers = {
    'Jahmyr Gibbs': [68],
    'Trey Mcbride': [11],

    'Kyren Williams': [11],
    "Ja'Marr Chase": [62],

    'Raheem Mostert': [13],
    'Nico Collins': [12],

    'Amon Ra St Brown': [46],
    'Rhamondre Stevenson': [11],

    'Garrett Wilson': [34],
    'James Conner': [48],

    'Brandon Aiyuk': [24],
    'George Pickens': [12],

    'Travis Etienne': [79],
    'Kenneth Walker': [42],
    
    'Sam Laporta': [11],
    'Michael Pittman': [20],

    'Breece Hall': [44],
    'Dalton Kincaid': [15],
}

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
    results = results[results.player!='PLAYER'].reset_index(drop=True)

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
# to_actual = dm.read(f"SELECT * FROM Actual_Salaries WHERE year={YEAR}", 'Simulation')
# to_actual = to_actual[['player', 'actual_salary', 'year', 'league']].rename(columns={'actual_salary': 'salary'})
# to_actual['league'] = to_actual.league.apply(lambda x: f'{x}_actual')
# to_actual['std_dev'] = 0.1
# to_actual['min_score'] = to_actual.salary - 1
# to_actual['max_score'] = to_actual.salary + 1

# dm.delete_from_db('Simulation', 'Salaries', f"year={YEAR} AND league='{LEAGUE}_actual'")
# dm.write_to_db(to_actual, 'Simulation', 'Salaries', 'append')

#%%

def get_adp():
    all_stats = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE']:
        print(pos)
        stats = dm.read(f'''SELECT player, year, avg_pick, avg_proj_points, avg_proj_rank, year_exp,
                                   avg_proj_points_exp_diff, avg_pick_exp_diff
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

def add_osu(salaries):
    osu = dm.read('''SELECT DISTINCT player, 1 as is_OSU 
                    FROM college_stats
                    where team='Ohio State' ''', 'Season_Stats_New')
    salaries = pd.merge(salaries, osu, on=['player'], how='left')
    salaries.is_OSU = salaries.is_OSU.fillna(0)

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
    salaries.is_keeper = salaries.is_keeper.fillna(0)

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


salaries = get_salaries()

total_spent = salaries.groupby('year').agg({'actual_salary': 'sum'}).reset_index().rename(columns={'actual_salary': 'total_spent'})
salaries = pd.merge(salaries, total_spent, on=['year'], how='left')
salaries.loc[salaries.year==YEAR, 'total_spent'] = 3576
salaries['fraction_spent'] = salaries.total_spent / 3576

salaries = add_osu(salaries)
salaries = add_rookie(salaries)
adp_stats = get_adp()
salaries = pd.merge(salaries, adp_stats, on=['player', 'year'])
salaries = fill_ty_keepers(salaries, ty_keepers)
salaries = calc_inflation(salaries)

salaries = drop_keepers(salaries)
salaries = add_salary_pos_rank(salaries)

salaries = remove_outliers(salaries)
salaries = salaries.sample(frac=1, random_state=1234).reset_index(drop=True)

# salaries = pd.concat([salaries, pd.get_dummies(salaries.year, prefix='year')], axis=1)
salaries['young_player'] = (salaries.year_exp < 2).astype('int')
salaries['young_osu'] = salaries.young_player * salaries.is_OSU
salaries['rookie_osu'] = salaries.is_rookie * salaries.is_OSU
salaries['rookie_rank'] = salaries.is_rookie * salaries.avg_pick
salaries['old_player'] = (salaries.year_exp > 5).astype('int')

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
skm = SciKitModel(salaries)

X, y = skm.Xy_split('y_act', to_drop=['player', 'team', 'week', 'league'])
X = pd.concat([X, pd.get_dummies(X.pos)], axis=1).drop('pos', axis=1)
X['qb_proj'] = X.QB * X.avg_proj_points
X['rb_proj'] = X.RB * X.avg_proj_points
X['wr_proj'] = X.WR * X.avg_proj_points
X['te_proj'] = X.TE * X.avg_proj_points

X['qb_rank'] = X.QB * X.avg_pick
X['rb_rank'] = X.RB * X.avg_pick
X['wr_rank'] = X.WR * X.avg_pick
X['te_rank'] = X.TE * X.avg_pick


X_train = X[X.year != YEAR]
y_train = y[X_train.index].reset_index(drop=True); X_train.reset_index(drop=True, inplace=True)   

X_test = X[X.year == YEAR].reset_index(drop=True)
y_test = y[X_test.index].reset_index(drop=True); X_test.reset_index(drop=True, inplace=True)    

pd.concat([X,y], axis=1).corr()['y_act'].sort_values()


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
model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf', 'gbmh', 'huber', 'cb', 'mlp']
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
                                                        col_split='game_date', time_split=2020,
                                                        bayes_rand='optuna', proba=False, trials=study,
                                                        random_seed=(i+7)*19+(i*12)+6, optuna_timeout=60)

    if i == 1:
        all_pred = oof_data['full_hold'][['player', 'year', 'pred', 'y_act']]
        all_pred = all_pred.rename(columns={'pred': m})
    else:
        all_pred = pd.merge(all_pred, oof_data['full_hold'][['player', 'year', 'pred']], on=['player', 'year'])
        all_pred = all_pred.rename(columns={'pred': m})

    scores[m] = oof_data['scores'][-1]
    best_models[m] = best_models_cur


#%%

from sklearn.preprocessing import StandardScaler
from zFix_Standard_Dev import *

scores_df = pd.DataFrame(scores, index=[0]).T.rename(columns={0: 'mse'})
scores_df = scores_df.sort_values(by='mse', ascending=True)

all_models = []
all_scores = []
for m in scores_df.index:
    all_models.append(m)
    
    preds = all_pred[[c for c in all_pred.columns if c in all_models]].mean(axis=1)
    preds = pd.Series(preds, name='pred_salary')
    val_data = pd.concat([all_pred[['player', 'year', 'y_act']], preds], axis=1)
 
    mf.show_scatter_plot(val_data.y_act, val_data.pred_salary)
    ens_mse = np.round(mean_squared_error(val_data.y_act, val_data.pred_salary), 3)
    ens_r2 = np.round(r2_score(val_data.y_act, val_data.pred_salary), 4)
    all_scores.append(ens_mse)
    print('MSE:', ens_mse)

best_score = np.argmin(all_scores[1:])
best_models_names = scores_df.iloc[:best_score+2].index
print(best_models_names)
print(all_scores[best_score+2])

preds = all_pred[[c for c in all_pred.columns if c in best_models_names]].mean(axis=1)
preds = pd.Series(preds, name='pred_salary')
val_data = pd.concat([all_pred[['player', 'year', 'y_act']], preds], axis=1)
val_data = pd.merge(salaries[['player', 'year', 'pos']], val_data, on=['player', 'year'])

#%%

final_pred = pd.DataFrame()
for m in best_models_names:
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

pred_results = pred_results.sort_values(by='salary', ascending=False)
pred_results.loc[pred_results.pred_salary<1, 'pred_salary'] = 1
pred_results.pred_salary = pred_results.pred_salary.astype('int')

# remove keepers and calculate extra dollars available to inflate predictions
pred_results.loc[pred_results.is_keeper==1, 'pred_salary'] = pred_results.loc[pred_results.is_keeper==1, 'actual_salary']
pred_results['pred_diff'] = pred_results.pred_salary - pred_results.salary

# use predictions minus available dollars to inflate predictions
total_diff = pred_results.pred_diff.sum()
print('Total Diff:', total_diff)

avg_spent = salaries.groupby('year').total_spent.mean().mean()
total_from_available = pred_results.iloc[:156].pred_salary.sum() - avg_spent
print('Total from available:', total_from_available)

# total_off = np.max([0,-(total_diff + total_from_available)/2])
total_off = np.max([0, -total_from_available])

# display the results
display(pred_results.iloc[:50])
display(pred_results[np.abs(pred_results.pred_diff) > 4].sort_values(by='pred_diff', ascending=False))

pred_results.loc[pred_results.is_keeper==0, 'pred_salary'] = (
                                                              pred_results.loc[pred_results.is_keeper==0, 'pred_salary'] 
                                                              * (1 + (total_off / 3600))
                                                              ).astype('int')
pred_results.iloc[:50]

#%%

for p in ['QB', 'RB', 'WR', 'TE']:
    print(f"\n{p}")
    val_data_tmp = val_data.copy()[val_data.pos==p].reset_index(drop=True).copy()
    sd_max_met = StandardScaler().fit(val_data_tmp[['pred_salary']]).transform(pred_results.loc[pred_results.pos==p, ['pred_salary']])

    sd_m, max_m, min_m = get_std_splines(val_data_tmp, {'pred_salary': 1}, show_plot=True, k=2, 
                                        min_grps_den=int(val_data_tmp.shape[0]*0.06), 
                                        max_grps_den=int(val_data_tmp.shape[0]*0.03),
                                        iso_spline='spline')

    pred_results.loc[pred_results.pos==p, 'std_dev'] = sd_m(sd_max_met)
    pred_results.loc[pred_results.pos==p, 'max_score'] = max_m(sd_max_met)
    pred_results.loc[pred_results.pos==p,'min_score'] = min_m(sd_max_met)


pred_results.sort_values(by='std_dev', ascending=False).iloc[:25]

#%%

pred_results.loc[pred_results.player.isin(['Josh Allen', 'Jalen Hurts']), 'std_dev'] = 8
pred_results.loc[pred_results.player.isin(['Josh Allen', 'Jalen Hurts']), 'pred_salary'] = 38
pred_results.loc[pred_results.player.isin(['Marvin Harrison', 'Malik Nabers']), 'pred_salary'] = \
    pred_results.loc[pred_results.player.isin(['Marvin Harrison', 'Malik Nabers']), 'pred_salary'] + 8

pred_results.loc[pred_results.std_dev <= 0, 'std_dev'] = pred_results.loc[pred_results.std_dev <= 0, 'pred_salary'] / 10

pred_results.loc[pred_results.max_score < pred_results.pred_salary, 'max_score'] = \
    pred_results.loc[pred_results.max_score < pred_results.pred_salary, 'pred_salary'] + \
        2 * pred_results.loc[pred_results.max_score < pred_results.pred_salary, 'std_dev']

pred_results.loc[pred_results.min_score > pred_results.pred_salary, 'min_score'] = \
    pred_results.loc[pred_results.min_score > pred_results.pred_salary, 'pred_salary'] - \
        2 * pred_results.loc[pred_results.min_score > pred_results.pred_salary, 'std_dev']

pred_results.loc[pred_results.min_score < (pred_results.pred_salary - pred_results.std_dev), 'min_score'] = \
        pred_results.loc[pred_results.min_score < (pred_results.pred_salary - pred_results.std_dev), 'pred_salary'] - \
            pred_results.loc[pred_results.min_score < (pred_results.pred_salary - pred_results.std_dev), 'std_dev']

pred_results.iloc[:50]

#%%
pred_results['league'] = LEAGUE + 'pred'
output = pred_results[['player', 'pred_salary', 'year', 'league', 'std_dev', 'min_score', 'max_score']]
output = output.rename(columns={'pred_salary': 'salary'})

dm.delete_from_db('Simulation', 'Salaries_Pred', f"year={YEAR} AND league='{LEAGUE}pred'")
dm.write_to_db(output, 'Simulation', 'Salaries_Pred', 'append')

#%%

import shutil

src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3'
shutil.copyfile(src, dst)

# %%

pred = dm.read("SELECT * FROM Salaries WHERE year=2023 AND league='nvpred'", 'Simulation')
actual = dm.read("SELECT * FROM Actual_Salaries WHERE year=2023 AND league='nv' AND is_keeper=0", 'Simulation')
combined = pd.merge(pred[['player', 'salary']], actual[['player', 'actual_salary']], on='player')
print(r2_score(combined.actual_salary, combined.salary))
combined.plot.scatter(x='salary', y='actual_salary')
combined['error'] = combined.actual_salary - combined.salary
display(combined.sort_values(by='error').iloc[:40])
display(combined.sort_values(by='error').iloc[-40:])
# %%
