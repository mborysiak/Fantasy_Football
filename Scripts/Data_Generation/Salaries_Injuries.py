# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#%%

# # Reading in Old Salary Data

import pandas as pd
import numpy as np
import os
import sqlite3
from zData_Functions import *

# set core path
PATH = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/Data/'
YEAR = 2020
LEAGUE = 'beta'
FNAME = f'{LEAGUE}_{YEAR}_results'


conn_sim = sqlite3.connect(f'{PATH}/Databases/Simulation.sqlite3')
conn_stats = sqlite3.connect(f'{PATH}/Databases/Season_Stats.sqlite3')



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
# append_to_db(results, 'Simulation', 'Actual_Salaries', 'append')


##########################################

#%%

actual_sal = pd.read_sql_query(f'''SELECT *
                                FROM Actual_Salaries 
                                WHERE League='{LEAGUE}' ''', conn_sim)
base_sal = pd.read_sql_query(f'''SELECT player, salary, year
                                 FROM Salaries 
                                 WHERE League='{LEAGUE}' ''', conn_sim)
player_age = pd.read_sql_query('''SELECT * FROM player_birthdays''', conn_stats)   
osu = pd.read_sql_query('''SELECT DISTINCT player, 1 as is_OSU 
                           FROM college_stats
                           where team='Ohio State' ''', conn_stats)
rookies = pd.read_sql_query('''SELECT player, draft_year year FROM rookie_adp''', conn_stats)
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

X = salaries[['inflation', 'salary', 'pos',  'is_rookie', 'is_OSU']] 
X = pd.concat([X, pd.get_dummies(X.pos, drop_first=True)], axis=1).drop('pos', axis=1)
X['rookie_rb'] = X.RB * X.is_rookie
X = X.drop('is_rookie', axis=1)

# X['stud'] = 0
# X.loc[X.salary > 80, 'stud'] = 1 

# X['filler'] = 0
# X.loc[X.salary < 10, 'filler'] = 1 

# X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
y = salaries.actual_salary

# m = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=1)
# m = LGBMRegressor(n_estimators=50, max_depth=5, reg_lambda=10)
m = Ridge(alpha=1)
mse = np.mean(np.sqrt(-cross_val_score(m, X, y, cv=10, scoring='neg_mean_squared_error')))
inf_baseline = np.sqrt(mean_squared_error(salaries.actual_salary*salaries.inflation, salaries.salary))
baseline = np.sqrt(mean_squared_error(salaries.actual_salary, salaries.salary))
print('Model',  round(mse, 3))
print('Inflation Baseline',  round(inf_baseline, 3))
print('Baseline',  round(baseline, 3))

m.fit(X, y)
pred_sal = cross_val_predict(m, X, y, cv=10)
#########################################
salaries = pd.concat([salaries, pd.Series(pred_sal, name='pred_salary')], axis=1)
pred_results = salaries[['player', 'year', 'actual_salary', 'pred_salary']]
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



salary_final = pd.read_csv(f'{path}/OtherData/Salaries/salaries_{year}_{league}.csv')
salary_final['year'] = year
salary_final['league'] = league
salary_final.player = salary_final.player.apply(name_clean)

if league=='snake':
    salary_final['salary'] =1

# +
conn_sim = sqlite3.connect(f'{path}/Databases/Simulation.sqlite3')
conn_sim.cursor().execute(f'''DELETE FROM Salaries WHERE year={year} AND league='{league}' ''')
conn_sim.commit()
    
append_to_db(salary_final,'Simulation', 'Salaries', 'append')
# -

# # Pushing Injuries to Season_Stats Database

# +
from sklearn.preprocessing import StandardScaler
year = 2020

# read in the injury predictor file
inj = pd.read_csv(f'{path}/OtherData/InjuryPredictor/injury_predictor_{year}.csv', header=None)

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

# adjust specific players if needed
pts = [1, 1, 1, 1, 1]
players = ['Kerryon Johnson', 'Tyreek Hill', 'Todd Gurley', 'Deebo Samuel', 'Miles Sanders']
for pt, pl in zip(pts, players):
    inj.loc[inj.player==pl, 'mean_risk'] = inj.loc[inj.player==pl, 'mean_risk'] + pt
    
inj['year'] = year
# -

conn_sim = sqlite3.connect(f'{path}/Databases/Simulation.sqlite3')
conn_sim.cursor().execute(f'''DELETE FROM Injuries WHERE year={year}''')
conn_sim.commit()
append_to_db(inj, 'Simulation', 'Injuries', 'append')



# %%
