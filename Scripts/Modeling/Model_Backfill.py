#%%
# core packages
from email.mime import message
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from skmodel import SciKitModel

import zHelper_Functions as hf
pos = hf.pos

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)

from sklearn import set_config
set_config(display='diagram')

#==========
# General Setting
#==========


pred_version = 'beta'
sim_version = 'beta'

#############

# set the root path and database management object
root_path = ffgeneral.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
np.random.seed(1234)

# set year to analyze
set_year = 2022

# set the earliest date to begin the validation set
val_year_min = 2012

#==========
# Fantasy Point Values
#==========

# define point values for all statistical categories
pass_yd_per_pt = 0.04 
pass_td_pt = 4
int_pts = -2
sacks = -1
rush_yd_per_pt = 0.1 
rec_yd_per_pt = 0.1
rush_td = 7
rec_td = 7
ppr = 0.5
fumble = -2

# creating dictionary containing point values for each position
pts_dict = {}
pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_td, int_pts, sacks]
pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_td, rec_td]
pts_dict['WR'] = [rec_yd_per_pt, ppr, rec_td]
pts_dict['TE'] = [rec_yd_per_pt, ppr, rec_td]

pos['QB']['req_touch'] = 250
pos['RB']['req_touch'] = 60
pos['WR']['req_touch'] = 50
pos['TE']['req_touch'] = 40

pos['QB']['req_games'] = 8
pos['RB']['req_games'] = 8
pos['WR']['req_games'] = 8
pos['TE']['req_games'] = 8


#-----------------
# Run Baseline Model
#-----------------


def get_non_rookies(set_pos):
    df =  dm.read(f'''SELECT *
                      FROM {set_pos}_{set_year}
                            ''', 'Model_Inputs')

    # apply the specified touches and game filters
    df, _, _ = hf.touch_game_filter(df, pos, set_pos, set_year)

    # calculate FP for a given set of scoring values
    df = hf.calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)
    df['rules_change'] = np.where(df.year>=2010, 1, 0)
    df['is_rookie'] = 0

    pos[set_pos]['use_ay'] = False

    # get the train and prediction dataframes for FP per game
    train, predict = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, set_year-1, 1998, 'v1')

    train = train[['player', 'pos', 'year', 'avg_pick', 'is_rookie', 'y_act']]
    predict = predict[['player', 'pos', 'year', 'avg_pick', 'is_rookie']]

    return train, predict

def get_skm(skm_df, model_obj, to_drop=['player', 'team', 'pos']):
    
    skm = SciKitModel(skm_df, model_obj=model_obj)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y



non_rookies_train = pd.DataFrame()
non_rookies_predict = pd.DataFrame()

for p in ['QB', 'RB', 'WR', 'TE']:

    pos_train, pos_predict = get_non_rookies(p)
    non_rookies_train = pd.concat([non_rookies_train, pos_train])
    non_rookies_predict = pd.concat([non_rookies_predict, pos_predict])



rookies_rb = dm.read(f'''SELECT player, 
                                pos,
                                year, 
                                avg_pick, 
                                1 as is_rookie, 
                                fp_per_game y_act
                            FROM Rookie_WR_{set_year}
                            WHERE year < {set_year}-1
                        ''', 'Model_Inputs')

rookies_wr = dm.read(f'''SELECT player, 
                                pos,
                                year, 
                                avg_pick, 
                                1 as is_rookie, 
                                fp_per_game y_act
                            FROM Rookie_WR_{set_year}
                            WHERE year < {set_year}-1
                        ''', 'Model_Inputs')

rookies_predict = dm.read(f'''SELECT player, 
                                     pos,
                                     draft_year-1 year, 
                                     avg_pick, 
                                     1 as is_rookie
                             FROM Rookie_ADP
                             WHERE draft_year = {set_year}
                             ''', 'Season_Stats')


df_train = pd.concat([non_rookies_train, rookies_rb, rookies_wr], axis=0)
df_train['week'] = 1
df_train['game_date'] = df_train.year

rookies_predict.avg_pick = np.log(rookies_predict.avg_pick)
df_predict = pd.concat([rookies_predict, non_rookies_predict], axis=0)
df_predict['week'] = 1
df_predict['game_date'] = set_year

to_fill = dm.read(f'''SELECT DISTINCT player 
                      FROM Model_Predictions
                      WHERE year_exp != 'Backfill'
                            AND year = {set_year} ''', 'Simulation')

df_predict = df_predict[~df_predict.player.isin(list(to_fill.player))].reset_index(drop=True)
output_start = df_predict[['player', 'pos', 'avg_pick']].copy()

# get the minimum number of training samples for the initial datasets
print('Shape of Train Set', df_train.shape)

#%%

skm = SciKitModel(df_train)
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player'])
cv_time_base = skm.cv_time_splits(X, 'year', 2012)

predictions = pd.DataFrame()
for m in [
          'ridge', 
          'lasso', 
          'gbm', 
          'svr', 
          'knn',  
          'xgb', 
          'rf', 
          'bridge'
          ]:
    
    print(m)
    
    model_base = skm.model_pipe([skm.column_transform(num_pipe = [skm.piece('std_scale')],
                                                      cat_pipe = [skm.piece('one_hot')]),
                                 skm.piece('k_best'),
                                 skm.piece(m)])

    params = skm.default_params(model_base)
    params['k_best__k'] = range(1, 10)

    best_model = skm.random_search(model_base, X, y, params, cv=cv_time_base, n_iter=25)
    mse = skm.cv_score(best_model, X, y, cv_time_base)
    print(np.sqrt(-mse))
    X_predict = df_predict[X.columns]
    predictions = pd.concat([predictions, pd.Series(best_model.predict(X_predict), name=m)], axis=1)
    
#%%

from Fix_Standard_Dev import *

output = output_start.copy()
output['pred_fp_per_game'] = predictions.mean(axis=1)
output['std_dev'] = 0
for p in ['QB', 'RB', 'WR', 'TE']:

    cur_train = df_train[df_train.pos==p].copy().reset_index(drop=True)
    cur_predict = output[output.pos==p].copy().reset_index(drop=True)

    sd_spline, max_spline, min_spline = get_std_splines(cur_train, {'avg_pick': 1},
                                            show_plot=True, k=1, 
                                            min_grps_den=int(cur_train.shape[0]*0.25), 
                                            max_grps_den=int(cur_train.shape[0]*0.15))

    sc = MinMaxScaler()
    sc.fit(cur_train.avg_pick.values.reshape(-1,1))
    

    pred_sd_max = pd.DataFrame(sc.transform(cur_predict.avg_pick.values.reshape(-1,1)))
    output.loc[output.pos==p, 'std_dev'] = sd_spline(pred_sd_max)
    output.loc[output.pos==p, 'max_score']  = max_spline(pred_sd_max)

output.iloc[:50]

output['filter_data'] = 'Backfill'
output['year_exp'] = 'Backfill'
output['version'] = pred_version
output['adp_rank'] = None
output['year']= set_year
output['rush_pass'] = 'both'
output['current_or_next_year'] = 'current'
output['date_modified'] = dt.datetime.now().strftime('%m-%d-%Y %H:%M')

cols = dm.read("SELECT * FROM Model_Predictions", 'Simulation').columns
output = output[cols]
output

#%%
delete_players = tuple(output.player)
dm.delete_from_db('Simulation', 'Model_Predictions', f"player in {delete_players} AND version='{pred_version}'")
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

# %%

rp = dm.read(f'''SELECT player,
                        pos,
                        avg_pick,
                        SUM(pred_fp_per_game) pred_fp_per_game,
                        SUM(std_dev) / 1.4 std_dev, 
                        SUM(max_score) / 1.3 max_score
                FROM Model_Predictions
                WHERE rush_pass IN ('rush', 'pass', 'rec')
                      AND version='{pred_version}'
                      AND year = {set_year}
                GROUP BY player, pos, year
             ''', 'Simulation').sort_values(by='avg_pick')

both = dm.read(f'''SELECT player, 
                          pos,
                          rush_pass,
                          pred_fp_per_game pred_fp_per_game,
                          std_dev,
                          max_score
                FROM Model_Predictions
                WHERE (rush_pass NOT IN ('rush', 'pass', 'rec') OR rush_pass IS NULL)
                      AND version='{pred_version}'
                      AND pos!='QB'
                      AND year = {set_year}
             ''', 'Simulation')

preds = pd.concat([rp, both], axis=0)
preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'mean', 
                                                              'std_dev': 'mean',
                                                              'max_score': 'mean'})
preds.pos = preds.pos.apply(lambda x: x.replace('Rookie_', ''))
preds = preds[preds.pred_fp_per_game > 0].reset_index(drop=True)
display(preds[((preds.pos=='QB'))].sort_values(by='pred_fp_per_game', ascending=False).iloc[:15])
display(preds[((preds.pos!='QB'))].sort_values(by='pred_fp_per_game', ascending=False).iloc[:50])

#%%

def plot_distribution(estimates):

    from IPython.core.pylabtools import figsize
    import seaborn as sns
    import matplotlib.pyplot as plt

    print('\n', estimates.player)
    estimates = estimates.iloc[2:]

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')

    # Plot the mean estimate
    plt.vlines(x = estimates.mean(), ymin = 0, ymax = 0.01, 
                linestyles = '--', colors = 'red',
                label = 'Pred Estimate',
                linewidth = 2.5)

    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');

    # Prediction information
    sum_stats = (np.percentile(estimates, 5), np.percentile(estimates, 95), estimates.std() /estimates.mean())
    print('Average Estimate = %0.4f' % estimates.mean())
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f    Std Error = %0.4f' % sum_stats) 

def create_distribution(player_data, num_samples=1000):
    
    import scipy.stats as stats

    # create truncated distribution
    lower, upper = 0,  player_data.max_score
    lower_bound = (lower - player_data.pred_fp_per_game) / player_data.std_dev, 
    upper_bound = (upper - player_data.pred_fp_per_game) / player_data.std_dev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def create_sim_output(output, num_samples=1000):
    sim_out = pd.DataFrame()
    for _, row in output.iterrows():
        cur_out = pd.DataFrame([row.player, row.pos]).T
        cur_out.columns=['player', 'pos']
        dists = 16*pd.DataFrame(create_distribution(row, num_samples)).T
        cur_out = pd.concat([cur_out, dists], axis=1)
        sim_out = pd.concat([sim_out, cur_out], axis=0)
    
    return sim_out

sim_output = create_sim_output(preds).reset_index(drop=True)

#%%

idx = sim_output[sim_output.player=="Antonio Gibson"].index[0]
plot_distribution(sim_output.iloc[idx])


# %%

dm.write_to_db(sim_output, 'Simulation', f'Version{sim_version}_{set_year}', 'replace')

# %%
