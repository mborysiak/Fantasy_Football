

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

# set core path
PATH = f'{root_path}/Data/'
YEAR = 2021
LEAGUE = 'nv'

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
salaries.player = salaries.player.apply(name_clean)

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
    results.player = results.player.apply(name_clean)
    
    results['year'] = year
    results['league'] = league
    
    return results

FNAME = f'{LEAGUE}_{YEAR}_results'
results = clean_results(PATH, FNAME, YEAR, LEAGUE)
# dm.delete_from_db('Simulation', 'Actual_Salaries', f"year='{YEAR}' AND league='{LEAGUE}'")
# dm.write_to_db(results, 'Simulation', 'Actual_Salaries', 'append')


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

salaries = pd.merge(actual_sal, base_sal, on=['player', 'year'], how='right')
salaries = pd.merge(salaries, player_age, on=['player'])
salaries = pd.merge(salaries, rookies, on=['player', 'year'], how='left')
salaries = pd.merge(salaries, osu, on=['player'], how='left')
salaries.is_rookie = salaries.is_rookie.fillna(0)
salaries.is_OSU = salaries.is_OSU.fillna(0)

keepers = salaries.loc[salaries.is_keeper==1, ['player', 'salary', 'actual_salary', 'year']].copy()
keepers['value'] = keepers.salary - keepers.actual_salary
inflation = keepers.groupby('year').agg({'value': 'sum'}).reset_index()
inflation['inflation'] = 1 + (inflation.value / 3600)

salaries = pd.merge(salaries, inflation, on='year', how='left')
salaries.loc[salaries.inflation.isnull(), 'inflation'] = 1

salaries.is_keeper = salaries.is_keeper.fillna(0)
salaries = salaries[salaries.is_keeper==0].reset_index(drop=True)
salaries = salaries[(salaries.year==YEAR) | (~salaries.actual_salary.isnull())].reset_index(drop=True)
salaries = salaries.sort_values(by=['year', 'salary'], ascending=[True, False])
salaries['sal_rank'] = salaries.groupby('year').cumcount()

salaries = salaries.sample(frac=1, random_state=1234)

#%%

skm = SciKitModel(salaries)
X, y = skm.Xy_split_list('actual_salary', ['year', 'sal_rank', 'inflation', 'salary', 'pos',  'is_rookie', 'is_OSU'])
X = pd.concat([X, pd.get_dummies(X.pos, drop_first=True)], axis=1).drop('pos', axis=1)
X['rookie_rb'] = X.RB * X.is_rookie

X_train = X[X.year != YEAR]
y_train = y[X_train.index].reset_index(drop=True); X_train.reset_index(drop=True, inplace=True)   

X_test = X[X.year == YEAR].reset_index(drop=True)
y_test = y[X_test.index].reset_index(drop=True); X_test.reset_index(drop=True, inplace=True)    


#%%
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
    params['k_best__k'] = range(1,X_train.shape[1])

    best_model = skm.random_search(pipe, X_train, y_train, params)
    mse, r2 = skm.val_scores(best_model, X_train, y_train, cv=5)
    best_models[m] = best_model

#%%
from sklearn.metrics import r2_score, mean_squared_error

baseline_data = salaries[salaries.year!=YEAR]

inf_baseline = mean_squared_error(baseline_data.actual_salary*baseline_data.inflation, baseline_data.salary)
inf_baseline_r2 = r2_score(baseline_data.actual_salary*baseline_data.inflation, baseline_data.salary)
baseline = mean_squared_error(baseline_data.actual_salary, baseline_data.salary)
baseline_r2 = r2_score(baseline_data.actual_salary, baseline_data.salary)

print('Inflation Baseline',  round(inf_baseline, 3), round(inf_baseline_r2, 3))
print('Baseline',  round(baseline, 3), round(baseline_r2, 3))

pred_sal = skm.cv_predict(best_models['svr'], X_train, y_train, cv=10)
pred_results = pd.concat([salaries.loc[salaries.year!=YEAR, ['player', 'year', 'salary', 'actual_salary']].reset_index(drop=True), 
                          pd.Series(pred_sal, name='pred_salary')], axis=1)
pred_results['dollar_diff'] = (pred_results.pred_salary - pred_results.actual_salary)
pred_results.dollar_diff.plot.hist()

#%%

X_test.inflation = X_train.inflation.mean()
pred_sal = np.mean([best_models['gbm'].predict(X_test),
                    best_models['rf'].predict(X_test),
                    best_models['svr'].predict(X_test)], axis=0)

pred_results = pd.concat([salaries.loc[salaries.year==YEAR,['player', 'year', 'salary']].reset_index(drop=True), 
                          pd.Series(pred_sal, name='pred_salary')], axis=1)

pred_results = pred_results.sort_values(by='salary', ascending=False)
pred_results.loc[pred_results.pred_salary<1, 'pred_salary'] = 1
pred_results.pred_salary = pred_results.pred_salary.astype('int')

pred_results['pred_diff'] = pred_results.pred_salary - pred_results.salary
pred_results.sort_values(by='pred_diff', ascending=False).iloc[:25]

#%%

output = pred_results[['player', 'pred_salary', 'year']]
output = output.rename(columns={'pred_salary': 'salary'})
output['league'] = LEAGUE + 'pred'
dm.delete_from_db('Simulation', 'Salaries', f"year={YEAR} AND league='{LEAGUE}pred'")
dm.write_to_db(output, 'Simulation', 'Salaries', 'append')

#%%

# # Pushing Injuries to Season_Stats Database

# +
from sklearn.preprocessing import StandardScaler
YEAR = 2021

# read in the injury predictor file
inj = pd.read_csv(f'{PATH}/OtherData/InjuryPredictor/injury_predictor_{YEAR}.csv', 
                  header=None)

# fix the columns and formatting
inj.columns = ['player', 'pct_miss_one', 'proj_games_missed', 'inj_pct_per_game', 'inj_risk', 'points']
inj.player = inj.player.apply(lambda x: x.split(',')[0])
inj.pct_miss_one = inj.pct_miss_one.apply(lambda x: float(x.strip('%')))
inj.inj_pct_per_game = inj.inj_pct_per_game.apply(lambda x: float(x.strip('%')))
inj = inj.drop('points', axis=1)
inj['year'] = YEAR

dm.delete_from_db('Simulation', 'Injuries_Source', f"year='{YEAR}'")
dm.write_to_db(inj, 'Simulation', 'Injuries_Source', 'append')

#%%

inj = dm.read(f'''SELECT * FROM Injuries_Source WHERE year={YEAR}''', 'Simulation')
inj = inj.drop(['inj_risk', 'year'], axis=1)
inj.player = inj.player.apply(name_clean)

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


# adjust specific players if needed
pts = []
players = []
for pt, pl in zip(pts, players):
    inj.loc[inj.player==pl, 'mean_risk'] = inj.loc[inj.player==pl, 'mean_risk'] + pt
    
inj['year'] = YEAR

dm.delete_from_db('Simulation', 'Injuries', f"year='{YEAR}'")
dm.write_to_db(inj, 'Simulation', 'Injuries', 'append')
# %%

# inj = dm.read('''SELECT * FROM Injuries_Source''', 'Simulation')
# inj.player = inj.player.apply(name_clean)

# games = dm.read('''SELECT player, year, pos, games 
#                    FROM RB_Stats
#                    UNION
#                    SELECT player, year, pos, games 
#                    FROM WR_Stats
#                    UNION
#                    SELECT player, year, pos, games 
#                    FROM TE_Stats
#                    UNION 
#                    SELECT player, year, pos, qb_games games
#                    FROM QB_Stats''', 'Season_Stats')

# X_test = inj[inj.year == YEAR].drop('player', axis=1).reset_index(drop=True)
# train = pd.merge(inj, games, on=['player', 'year'])
# train['games'] = 16-train['games']

# skm = SciKitModel(train)
# X_train, y_train = skm.Xy_split('games', to_drop=['player'])

# pipe = skm.model_pipe([skm.column_transform(num_pipe = [skm.piece('std_scale')],
#                                             cat_pipe = [skm.piece('one_hot')]),
#                        skm.piece('ridge')])

# # pipe.fit(X_train, y_train)
# # pipe.predict(X_test)
# print(skm.cv_score(pipe, X_train, y_train))
# mean_squared_error(np.full(len(y_train), y_train.mean()), y_train)