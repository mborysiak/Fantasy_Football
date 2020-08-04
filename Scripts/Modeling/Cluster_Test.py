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

# +
# core packages
import pandas as pd
import numpy as np
import os
import sqlite3
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, matthews_corrcoef, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import random
import time
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize
import pickle
import gzip
from sklearn.svm import SVR, LinearSVR, LinearSVC

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

from zHelper_Functions import *

import pandas_bokeh
pandas_bokeh.output_notebook()
# +
#==========
# General Setting
#==========

# set core path
path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'QB'

# set year to analyze
set_year = 2020

# specify database name with model data
db_name = 'Model_Inputs.sqlite3'
conn = sqlite3.connect(path + 'Data/Databases/' + db_name)

# set path to param database
param_conn = sqlite3.connect(path + 'Data/Databases/ParamTracking.sqlite3')

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

#==========
# Model Settings
#==========

pos['QB']['req_touch'] = 50
pos['RB']['req_touch'] = 40
pos['WR']['req_touch'] = 35
pos['TE']['req_touch'] = 30

pos['QB']['req_games'] = 8
pos['RB']['req_games'] = 8
pos['WR']['req_games'] = 8
pos['TE']['req_games'] = 8

pos['QB']['earliest_year'] = 1998
pos['RB']['earliest_year'] = 1998
pos['WR']['earliest_year'] = 1998
pos['TE']['earliest_year'] = 1998

pos['QB']['skip_years'] = 5
pos['RB']['skip_years'] = 7
pos['WR']['skip_years'] = 2
pos['TE']['skip_years'] = 4

pos['QB']['features'] = 'v2'
pos['RB']['features'] = 'v1'
pos['WR']['features'] = 'v1'
pos['TE']['features'] = 'v2'

pos['QB']['minimizer'] = 'forest_minimize'
pos['RB']['minimizer'] = 'gp_minimize'
pos['WR']['minimizer'] = 'gp_minimize'
pos['TE']['minimizer'] = 'gp_minimize'

pos['QB']['test_years'] = 4
pos['RB']['test_years'] = 4
pos['WR']['test_years'] = 4
pos['TE']['test_years'] = 4

pos['QB']['use_ay'] = False
pos['RB']['use_ay'] = False
pos['WR']['use_ay'] = True
pos['TE']['use_ay'] = True

output = pd.DataFrame({
    'pkey': [None],
    'year': [set_year],
    'pos': [set_pos],
    'metric': [None],
    'req_touch': [pos[set_pos]['req_touch']],
    'req_games': [pos[set_pos]['req_games']],
    'earliest_year': [None],
    'skip_years': [pos[set_pos]['skip_years']],
    'use_ay': [pos[set_pos]['use_ay']],
    'features': [pos[set_pos]['features']],
    'minimizer': [pos[set_pos]['minimizer']],
    'adp_ppg_low': [None],
    'adp_ppg_high': [None],
    'test_years': [pos[set_pos]['test_years']],
    'model': [None],
    'rmse_validation': [None],
    'rmse_validation_adp': [None],
    'rmse_test': [None],
    'rmse_test_adp': [None],
    'r2_test': [None],
    'r2_test_adp': [None]
})

output_class = pd.DataFrame({
    'pkey': [None],
    'year': [set_year],
    'pos': [set_pos],
    'breakout_metric': [None],
    'req_touch': [pos[set_pos]['req_touch']],
    'req_games': [pos[set_pos]['req_games']],
    'earliest_year': [None],
    'skip_years': [pos[set_pos]['skip_years']],
    'features': [pos[set_pos]['features']],
    'minimizer': pos[set_pos]['minimizer'],
    'use_ay': [pos[set_pos]['use_ay']],
    'act_ppg': [None],
    'pct_off': [None], 
    'adp_ppg_low': [None],
    'adp_ppg_high': [None],
    'test_years': [pos[set_pos]['test_years']],
    'model': [None],
    'val_score': [None],
    'test_score': [None]
})

def save_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
# +
#==========
# Pull and clean compiled data
#==========

# connect to database and pull in positional data
df = pd.read_sql_query('SELECT * FROM ' + set_pos + '_' + str(set_year), con=conn)

# append air yards for specified positions
if pos[set_pos]['use_ay']:
    ay = pd.read_sql_query('SELECT * FROM AirYards', con=sqlite3.connect(path + 'Data/Databases/Season_Stats.sqlite3'))
    df = pd.merge(df, ay, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
        
# apply the specified touches and game filters
df, df_train_results, df_test_results = touch_game_filter(df, pos, set_pos, set_year)

# calculate FP for a given set of scoring values
df = calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)

# +
#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
adp_ppg_high = '<100'
adp_ppg_low = '>=0'

# connect to database and pull in positional data
df = pd.read_sql_query('SELECT * FROM ' + set_pos + '_' + str(set_year), con=conn)

# append air yards for specified positions
if pos[set_pos]['use_ay']:
    ay = pd.read_sql_query('SELECT * FROM AirYards', con=sqlite3.connect(path + 'Data/Databases/Season_Stats.sqlite3'))
    df = pd.merge(df, ay, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])

# apply the specified touches and game filters
df, df_train_results, df_test_results = touch_game_filter(df, pos, set_pos, set_year)

# calculate FP for a given set of scoring values
df = calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)

#==============
# Create Break-out Probability Features
#==============

# get the train and prediction dataframes for FP per game
df_train_orig, df_predict_orig = get_train_predict(df, breakout_metric, pos, set_pos, set_year-pos[set_pos]['test_years'], 
                                                   pos[set_pos]['earliest_year'], pos[set_pos]['features'])

# get the train and prediction dataframes for FP per game
df_train_orig, df_predict_orig = get_train_predict(df, breakout_metric, pos, set_pos, set_year-pos[set_pos]['test_years'], 
                                              pos[set_pos]['earliest_year'], pos[set_pos]['features'])

df_predict_orig = df_predict_orig.dropna(subset=['y_act']).reset_index(drop=True)

# get the adp predictions and merge back to the training dataframe
df_train_orig,df_predict_orig, lr = get_adp_predictions(df_train_orig, df_predict_orig, 1)

# filter to adp cutoffs
df_train_orig = adp_filter(df_train_orig, adp_ppg_low, adp_ppg_high)
df_predict_orig = adp_filter(df_predict_orig, adp_ppg_low, adp_ppg_high)

# print the value-counts
print(f'Number of Samples / Features: {df_train_orig.shape[0]} / {df_train_orig.shape[1]}')
print(f'Number of Samples / Features: {df_predict_orig.shape[0]} / {df_predict_orig.shape[1]}')

print('Min Train Year:', df_train_orig.year.min())
print('Max Train Year:', df_train_orig.year.min() + pos[set_pos]['skip_years'])
print('Min Val Year:', df_train_orig.year.min() + pos[set_pos]['skip_years']+1)
print('Max Val Year:', df_train_orig.year.max())
print('Min Test Year:', df_predict_orig.year.min())
print('Max Test Year:', df_predict_orig.year.max())
# -

df_train_orig.corr()['y_act'].sort_values()

# +
corr_cut = 0.2
col_cut = 0.5

year_start = int(df_train_orig.year.min() + pos[set_pos]['skip_years'])
year_end = int(df_train_orig.year.max()+1)

preds = []
y_vals = []

adp_preds = []
y_adp = []

for y in range(year_start, year_end):
    
    train_split = df_train_orig[df_train_orig.year < y].drop(['pct_off', 'avg_pick_pred'], axis=1)
    val_split = df_train_orig[df_train_orig.year == y].drop(['pct_off', 'avg_pick_pred'], axis=1)
    
    train_split = corr_collinear_removal(train_split, corr_cut, col_cut).drop(['year'], axis=1)
    val_split = val_split[train_split.columns]
    
    print(train_split.columns)
    
    X_train, X_test, y_train, y_test = X_y_split(train_split, val_split, scale=True, pca=False, n_components=0.5)
    lr.fit(X_train, y_train)
    
    preds.extend(lr.predict(X_test))
    y_vals.extend(y_test)
    
    lr.fit(X_train[['avg_pick']], y_train)
    adp_preds.extend(lr.predict(X_test[['avg_pick']]))
    y_adp.extend(y_test)


train_split = corr_collinear_removal(df_train_orig.drop(['pct_off', 'avg_pick_pred'], axis=1), 
                                     corr_cut, col_cut).drop(['year'], axis=1)

val_split = df_predict_orig[train_split.columns]

X_train, X_test, y_train, y_test = X_y_split(train_split, val_split, scale=True, pca=False, n_components=0.5)
lr.fit(X_train, y_train)

lr.fit(X_train, y_train)
pred_test = lr.predict(X_test)

preds.extend(pred_test)
y_vals.extend(y_test)
print(X_train.columns)

lr.fit(X_train[['avg_pick']], y_train)
pred_test_adp = lr.predict(X_test[['avg_pick']])
adp_preds.extend(pred_test_adp)
y_adp.extend(y_test)


print('Model Validation')
print('RMSE: ', round(np.sqrt(mean_squared_error(preds, y_vals)), 3))
print('R2: ', round(np.sqrt(r2_score(y_vals, preds)), 3))

print('ADP Validation')
print('RMSE: ', round(np.sqrt(mean_squared_error(adp_preds, y_vals)), 3))
print('R2: ', round(np.sqrt(r2_score(y_vals, adp_preds)), 3))

print('===============')

print('Model Test')
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, pred_test)), 3))
print('R2: ', round(np.sqrt(r2_score(y_test, pred_test)), 3))

print('ADP Test')
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, pred_test_adp)), 3))
print('R2: ', round(np.sqrt(r2_score(y_test, pred_test_adp)), 3))
# -

r2_score(y_test, pred_test_adp)

pd.concat([df_predict_orig[['player', 'year', 'y_act']], pd.Series(pred_test, name='pred')], axis=1)

plt.scatter(preds,y_vals)

plt.scatter(adp_preds, y_vals)

# +
train_split = corr_collinear_removal(df_train_orig.drop('pct_off', axis=1), 0.2, 0.6).drop(['year', 'avg_pick'], axis=1)
test_split = df_predict_orig[train_split.columns]

X_train, X_test, y_train, y_test = X_y_split(train_split, test_split, scale=True, pca=False, n_components=0.5)
X_train, X_test = create_clusters(X_train, X_test, n_start=10, n_end=30)

c_pred = pd.concat([X_train.cluster, y_train], axis=1).groupby('cluster').agg(grp_predict = ('y_act', 'mean'))
c_pred = c_pred.reset_index()

X_test = pd.merge(X_test, c_pred, on='cluster')
# -

print('Grp Only: ', round(np.sqrt(mean_squared_error(y_test, X_test.grp_predict)), 3))
grp_adp =  np.mean([df_predict_orig.avg_pick_pred, X_test.grp_predict], axis=0)
print('Grp + ADP: ',  round(np.sqrt(mean_squared_error(y_test, grp_adp )), 3))
print('ADP Only:', 3.938)

# +
X_test = df_predict_orig[cols]

c_test = df_predict_orig[all_cols].copy()
c_test['Cluster'] = X_test_labels

grp_mean = c.groupby('Cluster').agg(grp_predict = ('y_act', 'mean')).reset_index()

c_test = pd.merge(c_test, grp_mean, on='Cluster')
print('Grp Only: ', round(np.sqrt(mean_squared_error(c_test.y_act, c_test.grp_predict)), 3))
grp_adp =  np.mean([c_test.avg_pick_pred, c_test.grp_predict], axis=0)
print('Grp + ADP: ',  round(np.sqrt(mean_squared_error(c_test.y_act, grp_adp )), 3))
print('ADP Only:', 3.938)
# -

plt.scatter(grp_adp, c_test.y_act)

plt.scatter(c_test.avg_pick_pred, c_test.y_act)

# # Cluster Predicting

# +
# def create_clusters(X_train, X_predict, n_start=5, n_end=30):
    
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score

#     scores = []
#     for i in range(n_start, n_end):

#         km = KMeans(n_clusters=i, random_state=1234)
#         km.fit(X_train)
#         labels = km.labels_
#         scores.append( round(silhouette_score(X_train, labels, metric='euclidean'), 3))
     
#     # get the best N clusters
#     best_n = n_start + np.argmax(scores) + 1
#     print('Best N Clusters:', best_n)
    
#     # fit the model with best number of clusters
#     km = KMeans(n_clusters = best_n, random_state=1234)
#     km.fit(X_train)
    
#     # get the train labels
#     train_labels = km.labels_
#     X_train['cluster'] = train_labels
    
#     # get the test labels
#     test_labels = km.predict(X_predict)
#     X_predict['cluster'] = test_labels
    
#     return X_train, X_predict

# year_start = int(df_train_orig.year.min() + pos[set_pos]['skip_years'])
# year_end = int(df_train_orig.year.max()+1)

# preds = []
# y_vals = []

# for y in range(year_start, year_end):
    
#     train_split = df_train_orig[df_train_orig.year < y].drop('pct_off', axis=1)
#     val_split = df_train_orig[df_train_orig.year == y].drop('pct_off', axis=1)
    
#     train_split = corr_collinear_removal(train_split, 0.2, 0.5).drop(['year', 'avg_pick'], axis=1)
#     val_split = val_split[train_split.columns]
    
#     X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale=True, pca=False, n_components=0.5)
#     X_train, X_val = create_clusters(X_train, X_val, n_start=5, n_end=30)
    
#     c_pred = pd.concat([X_train.cluster, y_train], axis=1).groupby('cluster').agg(grp_predict = ('y_act', 'mean'))
#     c_pred = c_pred.reset_index()
    
#     X_val = pd.merge(X_val, c_pred, on='cluster')
#     preds.extend(X_val.grp_predict)
#     y_vals.extend(y_val)
# -


