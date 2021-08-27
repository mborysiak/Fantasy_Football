#%%
# core packages
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

# set the root path and database management object
root_path = ffgeneral.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
np.random.seed(1234)

# set year to analyze
set_year = 2021

# set the earliest date to begin the validation set
val_year_min = 2012

#==========
# Fantasy Point Values
#==========

# define point values for all statistical categories
pass_yd_per_pt = 0.04 
pass_td_pt = 5
int_pts = -2
sacks = -1
rush_yd_per_pt = 0.1 
rec_yd_per_pt = 0.1
rush_rec_td = 7
ppr = 0.5
fumble = -2

# creating dictionary containing point values for each position
pts_dict = {}
pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_rec_td, int_pts, sacks]
pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['WR'] = [rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['TE'] = [rec_yd_per_pt, ppr, rush_rec_td]

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

set_pos = 'RB'
pred_version = 1
sim_version = 3

pos[set_pos]['use_ay'] = False
non_rookies =  dm.read(f'''SELECT *
                            FROM {set_pos}_{set_year}
                                ''', 'Model_Inputs')

# apply the specified touches and game filters
non_rookies, _, _ = hf.touch_game_filter(non_rookies, pos, set_pos, set_year)

# calculate FP for a given set of scoring values
non_rookies = hf.calculate_fp(non_rookies, pts_dict, pos=set_pos).reset_index(drop=True)
non_rookies['rules_change'] = np.where(non_rookies.year>=2010, 1, 0)
non_rookies['is_rookie'] = 0

# get the train and prediction dataframes for FP per game
non_rookies_train, non_rookies_predict = hf.get_train_predict(non_rookies, 'fp_per_game', pos, set_pos, 
                                                              set_year-1, 1998, 'v1')

non_rookies_train = non_rookies_train[['player', 'year', 'avg_pick', 'is_rookie', 'y_act']]
non_rookies_predict = non_rookies_predict[['player', 'year', 'avg_pick', 'is_rookie']]


if pos in ('RB', 'WR'):
    rookies_train = dm.read(f'''SELECT player, 
                                    year, 
                                    avg_pick, 
                                    1 as is_rookie, 
                                    fp_per_game y_act
                                FROM Rookie_{set_pos}_{set_year}
                                WHERE year < {set_year}-1
                            ''', 'Model_Inputs')
    df_train = pd.concat([rookies_train, non_rookies_train], axis=0)

else: df_train = non_rookies_train.copy()

rookies_predict = dm.read(f'''SELECT player, 
                                     draft_year-1 year, 
                                     avg_pick, 
                                     1 as is_rookie
                             FROM Rookie_ADP
                             WHERE draft_year = {set_year}
                                   AND pos = '{set_pos}'
                             ''', 'Season_Stats')

rookies_predict.avg_pick = np.log(rookies_predict.avg_pick)
df_predict = pd.concat([rookies_predict, non_rookies_predict], axis=0)

to_fill = dm.read(f'''SELECT DISTINCT player FROM Version{sim_version}_{set_year}''', 'Simulation')
df_predict = df_predict[~df_predict.player.isin(list(to_fill.player))].reset_index(drop=True)
output_start = df_predict[['player', 'avg_pick']].copy()

# get the minimum number of training samples for the initial datasets
print('Shape of Train Set', df_train.shape)

#%%

skm = SciKitModel(df_train)
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player'])
cv_time_base = skm.cv_time_splits('year', X, 2012)

model_base = skm.model_pipe([skm.piece('std_scale'), 
                            skm.piece('k_best'),
                             skm.piece('bridge')])

params = skm.default_params(model_base)
#params['k_best__k'] = range(1, X_base.shape[1])

best_model_base = skm.random_search(model_base, X, y, params, cv=cv_time_base, n_iter=25)
_, _ = skm.val_scores(best_model_base, X, y, cv_time_base)

imp_cols = X.columns[best_model_base['k_best'].get_support()]
skm.print_coef(best_model_base, imp_cols)

X_predict = df_predict[X.columns]
pred, pred_std = best_model_base.predict(X_predict, return_std=True)
output = output_start.copy()
output['pred_fp_per_game'] = pred
output['std_dev'] = pred_std

output['pos'] = set_pos
output['filter_data'] = 'Backfill'
output['year_exp'] = 'Backfill'
output['version'] = 'v1'
output['max_score'] = np.percentile(y, 95)

#%%
delete_players = tuple(output.player)
dm.delete_from_db('Simulation', 'Model_Predictions', f"player in {delete_players} AND version={pred_version}")
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

# %%

preds = dm.read(f'''SELECT * FROM Model_Predictions WHERE version='v{pred_version}' ''', 'Simulation')
preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'mean', 
                                                              'std_dev': 'mean',
                                                              'max_score': 'mean'})

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

idx = sim_output[sim_output.player=='TJ Hockenson'].index[0]
plot_distribution(sim_output.iloc[idx])


# %%


dm.write_to_db(sim_output, 'Simulation', f'Version{sim_version}_{set_year}', 'replace')

# %%
