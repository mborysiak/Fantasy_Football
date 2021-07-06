

#%%

# # Reading in Old Salary Data

import pandas as pd
import numpy as np
import sqlite3
from zData_Functions import *

from ff.db_operations import DataManage
from ff import general
from skmodel import SciKitModel

import pandas_bokeh
pandas_bokeh.output_notebook()

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

#%%
# set core path
PATH = f'{root_path}/Data/'
YEAR = 2020
LEAGUE = 'beta'
FNAME = f'{LEAGUE}_{YEAR}_results'

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
    results.player = results.player.apply(name_clean)
    
    results['year'] = year
    results['league'] = league
    
    return results

results = clean_results(PATH, FNAME, YEAR, LEAGUE)
dm.delete_from_db('Simulation', 'Actual_Salaries', f"year='{YEAR}' AND league='{LEAGUE}'")
dm.write_to_db(results, 'Simulation', 'Actual_Salaries', 'append')


#%%

actual_sal = dm.read(f'''SELECT *
                         FROM Actual_Salaries 
                         WHERE League='{LEAGUE}' ''', 'Simulation')
base_sal = dm.read(f'''SELECT player, salary, year
                              FROM Salaries 
                              WHERE League='{LEAGUE}' ''', 'Simulation')
player_age = dm.read('''SELECT * FROM player_birthdays''', 'Season_Stats')   
osu = dm.read('''SELECT DISTINCT player, 1 as is_OSU 
                           FROM college_stats
                           where team='Ohio State' ''', 'Season_Stats')
rookies = dm.read('''SELECT player, draft_year year FROM rookie_adp''', 'Season_Stats')
rookies['is_rookie'] = 1

# 
salaries = pd.merge(actual_sal, base_sal, on=['player', 'year'])
salaries = pd.merge(salaries, player_age, on=['player'])
salaries = pd.merge(salaries, rookies, on=['player', 'year'], how='left')
salaries = pd.merge(salaries, osu, on=['player'], how='left')
salaries.is_rookie = salaries.is_rookie.fillna(0)
salaries.is_OSU = salaries.is_OSU.fillna(0)

keepers = salaries.loc[salaries.is_keeper==1, ['player', 'salary', 'actual_salary', 'year']].copy()
keepers['value'] = keepers.salary - keepers.actual_salary
inflation = keepers.groupby('year').agg({'value': 'sum'}).reset_index()
inflation['inflation'] = 1 + (inflation.value / 3600)

salaries = pd.merge(salaries, inflation, on='year')
salaries = salaries[salaries.is_keeper==0].reset_index(drop=True)

#%%

skm = SciKitModel(salaries)
X, y = skm.Xy_split_list('actual_salary', ['inflation', 'salary', 'pos',  'is_rookie', 'is_OSU'])
X = pd.concat([X, pd.get_dummies(X.pos, drop_first=True)], axis=1).drop('pos', axis=1)
X['rookie_rb'] = X.RB * X.is_rookie
# X = X.drop('is_rookie', axis=1)

# loop through each potential model
best_models = {}
model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf']
for m in model_list:

    print('\n============\n')
    print(m)

    # set up the model pipe and get the default search parameters
    pipe = skm.model_pipe([
                            skm.piece('std_scale'), 
                            skm.piece('k_best'),
                            skm.piece(m)
                            ])
    params = skm.default_params(pipe, 'rand')
    params['k_best__k'] = range(1,X.shape[1])

    best_model = skm.random_search(pipe, X, y, params)
    mse, r2 = skm.val_scores(best_model, X, y, cv=5)
    best_models[m] = best_model

#%%
from sklearn.metrics import r2_score, mean_squared_error

inf_baseline = mean_squared_error(salaries.actual_salary*salaries.inflation, salaries.salary)
inf_baseline_r2 = r2_score(salaries.actual_salary*salaries.inflation, salaries.salary)
baseline = mean_squared_error(salaries.actual_salary, salaries.salary)
baseline_r2 = r2_score(salaries.actual_salary, salaries.salary)

print('Inflation Baseline',  round(inf_baseline, 3), round(inf_baseline_r2, 3))
print('Baseline',  round(baseline, 3), round(baseline_r2, 3))

#%%

pred_sal = skm.cv_predict(best_models['lasso'], X, y, cv=10)
pred_results = pd.concat([salaries[['player', 'year', 'actual_salary', 'salary']], 
                      pd.Series(pred_sal, name='pred_salary')], axis=1)
pred_results['sal_diff'] = abs(pred_results.pred_salary - pred_results.actual_salary)
pred_results.sort_values(by='sal_diff', ascending=False).iloc[:50]

#%%

output = pred_results[['player', 'pred_salary', 'year']]
output = output.rename(columns={'pred_salary': 'salary'})
output.salary = output.salary.astype('int')
output = output[output.year==2020]
keepers = keepers.loc[keepers.year==2020, ['player', 'actual_salary', 'year']]
keepers = keepers.rename(columns={'actual_salary': 'salary'})
output = pd.concat([output, keepers], axis=0)
output['league'] = LEAGUE + 'pred'
append_to_db(output,'Simulation', 'Salaries', 'append')

#%%

YEAR=2021

salary_final = pd.read_csv(f'{PATH}/OtherData/Salaries/salaries_{YEAR}_{LEAGUE}.csv')
salary_final['year'] = YEAR
salary_final['league'] = LEAGUE
salary_final.player = salary_final.player.apply(name_clean)

if LEAGUE=='snake':
    salary_final['salary'] =1

dm.delete_from_db('Simulation', 'Salaries', f"year='{YEAR}' AND league='{LEAGUE}'")
dm.write_to_db(salary_final, 'Simulation', 'Salaries', 'append')
# -

#%%

# # Pushing Injuries to Season_Stats Database

# +
from sklearn.preprocessing import StandardScaler
year = 2021

# read in the injury predictor file
inj = pd.read_csv(f'{PATH}/OtherData/InjuryPredictor/injury_predictor_{YEAR}.csv', 
                  header=None)

# fix the columns and formatting
inj.columns = ['player', 'pct_miss_one', 'proj_games_missed', 'inj_pct_per_game', 'inj_risk', 'points']
inj.player = inj.player.apply(lambda x: x.split(',')[0])
inj.pct_miss_one = inj.pct_miss_one.apply(lambda x: float(x.strip('%')))
inj.inj_pct_per_game = inj.inj_pct_per_game.apply(lambda x: float(x.strip('%')))
inj = inj.drop(['points', 'inj_risk'], axis=1)

# scale the data and set minimum to 0
X = StandardScaler().fit_transform(inj.iloc[:, 1:])
inj = pd.concat([pd.DataFrame(inj.player),
                 pd.DataFrame(X, columns=['pct_miss_one', 'proj_games_missed', 'pct_per_game'])], 
                axis=1)
for col in ['pct_miss_one', 'proj_games_missed', 'pct_per_game']:
    inj[col] = inj[col] + abs(inj[col].min())
    
# take the mean risk value
inj['mean_risk'] = inj.iloc[:, 1:].mean(axis=1)
inj = inj[['player', 'mean_risk']].sort_values(by='mean_risk').reset_index(drop=True)

inj.player = inj.player.apply(name_clean)

# # adjust specific players if needed
# pts = [1, 1, 1, 1, 1]
# players = ['Kerryon Johnson', 'Tyreek Hill', 'Todd Gurley', 'Deebo Samuel', 'Miles Sanders']
# for pt, pl in zip(pts, players):
#     inj.loc[inj.player==pl, 'mean_risk'] = inj.loc[inj.player==pl, 'mean_risk'] + pt
    
inj['year'] = YEAR

dm.delete_from_db('Simulation', 'Injuries', f"year='{YEAR}'")
dm.write_to_db(inj, 'Simulation', 'Injuries', 'append')



# %%
