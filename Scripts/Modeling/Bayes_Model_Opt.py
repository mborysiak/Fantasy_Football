# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, matthews_corrcoef
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

from zHelper_Functions import *

import pandas_bokeh
pandas_bokeh.output_notebook()
# +
#==========
# General Setting
#==========

# set core path
path = '/Users/Mark/Documents/Github/Fantasy_Football/'

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'RB'

# set year to analyze
set_year = 2019

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

pos['QB']['earliest_year'] = 2005
pos['RB']['earliest_year'] = 1998
pos['WR']['earliest_year'] = 1998
pos['TE']['earliest_year'] = 1998

pos['QB']['skip_years'] = 8
pos['RB']['skip_years'] = 10
pos['WR']['skip_years'] = 10
pos['TE']['skip_years'] = 4

pos['QB']['features'] = 'v2'
pos['RB']['features'] = 'v2'
pos['WR']['features'] = 'v2'
pos['TE']['features'] = 'v2'

pos['QB']['use_ay'] = False
pos['RB']['use_ay'] = False
pos['WR']['use_ay'] = False
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
    'model': [None],
    'rmse_validation': [None],
    'rmse_validation_adp': [None],
    'r2_validation': [None],
    'r2_validation_adp': [None]
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
    'use_ay': [pos[set_pos]['use_ay']],
    'act_ppg': [None],
    'pct_off': [None], 
    'adp_ppg': [None],
    'model': [None],
    'score': [None]
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

# add features based for a players performance relative to their experience
df = add_exp_metrics(df, set_pos, pos[set_pos]['use_ay'])
# -

# # Breakout Models

# - Do I need to split high vs low draft expectations?
# - Clean up process by writing out predict
# - Add in 2 or 3 year rolling metrics
# - Add in metrics related to games missed + over / under performning recent year vs prior
# - Convert everything to Mathews Coefficient

# +
#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
act_ppg = '>=11'
pct_off = '>0.2'
adp_ppg = '<11'

# get the train and prediction dataframes for FP per game
df_train_orig, df_predict = get_train_predict(df, breakout_metric, pos, set_pos, set_year, pos[set_pos]['earliest_year'], pos[set_pos]['features'])

# get the adp predictions and merge back to the training dataframe
df_train_adp, lr = get_adp_predictions(df_train_orig, 1)
df_train_orig = pd.merge(df_train_orig, df_train_adp, on=['player', 'year'])

# create the label and filter based on inputs
df_train_orig['label'] = 0
df_train_orig.loc[(eval(f'df_train_orig.pct_off{pct_off}')) & (eval(f'df_train_orig.y_act{act_ppg}')), 'label'] = 1
df_train_orig = df_train_orig[(eval(f'df_train_orig.avg_pick_pred{adp_ppg}'))].reset_index(drop=True)
df_train_orig = df_train_orig.drop(['y_act', 'pct_off'], axis=1).rename(columns={'label': 'y_act'})

# get the minimum number of training samples for the initial datasets
min_samples = int(0.5*df_train_orig[df_train_orig.year <= df_train_orig.year.min() + pos[set_pos]['skip_years']].shape[0])

# print the value-counts
print(df_train_orig.y_act.value_counts())
print(df_train_orig.shape)
print('Min Year:', df_train_orig.year.min())
print('Max Year:', df_train_orig.year.max())


# -

@ignore_warnings(category=ConvergenceWarning)
def calc_f1_score(**args):
    
    from sklearn.metrics import f1_score
    from imblearn.over_sampling import SMOTE

    print('\n', args)

    # remove the extra args not needed for modeling
    scale = True if args['scale'] == 1 else False
    pca = True if args['pca'] == 1 else False   
    use_smote = True if args['use_smote'] == 1 else False
    
    n_components = args['n_components']
    collinear_cutoff = args['collinear_cutoff']
    zero_weight = args['zero_weight']
    
    for arg in ['scale', 'pca', 'collinear_cutoff', 'use_smote', 'zero_weight', 'n_components']:
        del args[arg]

    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------

    # remove collinear variables based on difference of means between the 0 and 1 labeled groups
    df_train = remove_classification_collinear(df_train_orig, collinear_cutoff, ['player', 'avg_pick', 'year', 'y_act'])

    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]

    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        # fit training data and creating prediction based on validation data
        estimator.set_params(**args)
        estimator.class_weight = {0: zero_weight, 1: 1}

        if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)
            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_val = X_val.values
            
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

        # skip over the first two year of predictions due to high error for xgb / lgbm
        val_predictions = np.append(val_predictions, val_predict, axis=0)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    # store the current predictions in the results tracker
    val_df = df_train.loc[df_train.year.isin(years), 
                          ['player', 'year', 'avg_pick', 'y_act']].reset_index(drop=True)
    val_pred = pd.concat([val_df, pd.Series(val_predictions, name='pred')], axis=1)

    # calculate the RMSE and MAE of the ensemble predictions
    f1_score = round(matthews_corrcoef(val_pred.pred, val_df.y_act), 3)

    return -f1_score


# +
class_models = {
    'lr': LogisticRegression(random_state=1234, solver='liblinear', tol=.001),
    'lgbm': LGBMClassifier(random_state=1234, n_jobs=-1),
    'xgb': XGBClassifier(random_state=1234, nthread=-1),
    'rf': RandomForestClassifier(random_state=1234, n_jobs=-1),
    'gbm': GradientBoostingClassifier(random_state=1234),
    'knn': KNeighborsClassifier(n_jobs=-1),
    'svr': LinearSVC()
}


class_search_space = {
    'lgbm': [
        Integer(1, 500, name='n_estimators'),
        Integer(2, 50, name='max_depth'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Real(0.01, 1, name='colsample_bytree'),
        Real(0.01, 1, name='subsample'),
        Real(0.001, 100, 'log_uniform', name='min_child_weight'),
        Real(0.001, 10000, "log_uniform", name='reg_lambda'),
        Real(0.001, 100, 'log_uniform', name='reg_alpha'),
        Integer(1, min_samples, name='min_data_in_leaf'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ],
    
    'xgb': [
        Integer(1, 500, 'log_uniform', name='n_estimators'),
        Integer(2, 50, name='max_depth'),
        Real(10**-6, 10**0, "log-uniform", name='learning_rate'),
        Real(0.01, 1, name='colsample_bytree'),
        Real(0.01, 1, name='subsample'),
        Real(0.001, 100, 'log_uniform', name='min_child_weight'),
        Real(0.0001, 1, 'log_uniform', name='gamma'),
        Real(0.001, 10000, "log_uniform", name='reg_lambda'),
        Real(0.001, 100, 'log_uniform', name='reg_alpha'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ],
        
    'lr': [
        Real(0.00001, 100000000, 'log_uniform', name='C'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ],
        
    'rf': [
        Integer(1, 500, name='n_estimators'),
        Integer(2, 100, name='max_depth'),
        Integer(1, min_samples, name='min_samples_leaf'),
        Real(0.01, 1, name='max_features'),
        Integer(2, 50, name='min_samples_split'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ],
    
    'gbm': [
        Integer(1, 500, name='n_estimators'),
        Integer(2, 50, name='max_depth'),
        Integer(1, min_samples, name='min_samples_leaf'),
        Real(0.01, 1, name='max_features'),
        Integer(2, 50, name='min_samples_split'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ],
    
    'knn': [
        Integer(1, min_samples, name='n_neighbors'),
        Categorical(['distance', 'uniform'], name='weights'),
        Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ],
    
    'svr': [
        Real(0.0001, 1000, 'log_uniform', name='C'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 1, name='n_components')
    ]
}
# +
x0=None
skip_years = pos[set_pos]['skip_years']

for model in class_models.keys():

    estimator = class_models[model]
    space = class_search_space[model]
    
    @use_named_args(space)
    def objective_class(**args):
        return calc_f1_score(**args)
    
    @use_named_args(space)
    def space_keys(**args):
        return list(args.keys())

    bayes_seed = 123456
    kappa = 3
    res_gp = gp_minimize(objective_class, space, n_calls=100, n_random_starts=25, x0=x0,
                        random_state=bayes_seed, verbose=True, kappa=kappa, n_jobs=-1)

    output_class.loc[0, 'breakout_metric'] = breakout_metric
    output_class.loc[0, 'act_ppg'] = act_ppg
    output_class.loc[0, 'pct_off'] = pct_off
    output_class.loc[0, 'adp_ppg'] = adp_ppg
    output_class.loc[0, 'model'] = model
    output_class.loc[0, 'earliest_year'] = df_train_orig.year.min()
    output_class.loc[0, 'score'] = -res_gp.fun
    
    params_output = dict(zip(space_keys(space), res_gp.x))

    append_to_db(output_class, db_name='ParamTracking.sqlite3', table_name='ClassParamTracking', if_exist='append')
    max_pkey = pd.read_sql_query("SELECT max(pkey) FROM ClassParamTracking", param_conn).values[0][0]

    save_pickle(params_output, path + f'Data/Model_Params_Class/{max_pkey}.p')
    save_pickle(df_train_orig, path + f'Data/Model_Datasets_Class/{max_pkey}.p')    
    save_pickle(class_search_space[model], path + f'Data/Bayes_Space_Class/{max_pkey}.p')
# -

# # Regression Models

# ### Define Models And Parameters

models = {
    'ridge': Ridge(random_state=1234),
    'lasso': Lasso(random_state=1234),
    'lgbm': LGBMRegressor(random_state=1234, n_jobs=-1),
    'xgb': XGBRegressor(random_state=1234, nthread=-1),
    'rf': RandomForestRegressor(random_state=1234, n_jobs=-1),
    'enet': ElasticNet(random_state=1234),
    'gbm': GradientBoostingRegressor(random_state=1234),
    'knn': KNeighborsRegressor(n_jobs=-1),
    'svr': LinearSVR()
}

# +
df_train, df_predict = get_train_predict(df, pos[set_pos]['metrics'][0], 
                                         pos, set_pos, set_year, pos[set_pos]['earliest_year'], pos[set_pos]['features'])
min_samples = int(0.5*df_train[df_train.year <= df_train.year.min() + pos[set_pos]['skip_years']].shape[0])

search_space = {
    'lgbm': [
        Integer(1, 500, name='n_estimators'),
        Integer(2, 100, name='max_depth'),
        Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
        Real(0.01, 1, name='colsample_bytree'),
        Real(0.01, 1, name='subsample'),
        Real(0.001, 100, 'log_uniform', name='min_child_weight'),
        Real(0.001, 10000, "log_uniform", name='reg_lambda'),
        Real(0.001, 100, 'log_uniform', name='reg_alpha'),
        Integer(1, min_samples, name='min_data_in_leaf'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
    
    'xgb': [
        Integer(1, 500, 'log_uniform', name='n_estimators'),
        Integer(2, 100, name='max_depth'),
        Real(10**-6, 10**0, "log-uniform", name='learning_rate'),
        Real(0.01, 1, name='colsample_bytree'),
        Real(0.01, 1, name='subsample'),
        Real(0.001, 100, 'log_uniform', name='min_child_weight'),
        Real(0.0001, 1, 'log_uniform', name='gamma'),
        Real(0.001, 10000, "log_uniform", name='reg_lambda'),
        Real(0.001, 100, 'log_uniform', name='reg_alpha'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
        
    
    'ridge': [
        Real(0.001, 100000, 'log_uniform', name='alpha'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
    
    'lasso': [
        Real(0.001, 1000, 'log_uniform', name='alpha'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
        
    'rf': [
        Integer(1, 500, name='n_estimators'),
        Integer(2, 100, name='max_depth'),
        Integer(1, min_samples, name='min_samples_leaf'),
        Real(0.01, 1, name='max_features'),
        Integer(2, 50, name='min_samples_split'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
    
    'enet': [
        Real(0.001, 10000, 'log_uniform', name='alpha'),
        Real(0.02, 0.98, name='l1_ratio'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
    
    'gbm': [
        Integer(1, 500, name='n_estimators'),
        Integer(2, 100, name='max_depth'),
        Integer(1, min_samples, name='min_samples_leaf'),
        Real(0.01, 1, name='max_features'),
        Integer(2, 50, name='min_samples_split'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
    
    'knn': [
        Integer(1, min_samples, name='n_neighbors'),
        Categorical(['distance', 'uniform'], name='weights'),
        Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ],
    
    'svr': [
        Real(0.0001, 100, 'log_uniform', name='C'),
        Real(0.005, 100, 'log_uniform', name='epsilon'),
        Real(0, .6, name='corr_cutoff'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Real(0.01, 0.5, name='n_components')
    ]
}


# -

# ### Define Functions

# +
@ignore_warnings(category=ConvergenceWarning)
def calc_rmse(**args):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    print('\n', args)

    # remove the extra args not needed for modeling
    scale = True if args['scale'] == 1 else False
    pca = True if args['pca'] == 1 else False
    
    n_components = args['n_components']
    corr_cutoff = args['corr_cutoff']
    collinear_cutoff = args['collinear_cutoff']

    for arg in ['scale', 'pca', 'corr_cutoff', 'collinear_cutoff', 'n_components']:
        del args[arg]

    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------

    # return the df_train with only relevant features remaining
    df_train = corr_collinear_removal(df_train_orig, corr_cutoff, collinear_cutoff)

    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]

    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        # fit training data and creating prediction based on validation data
        estimator.set_params(**args)
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

        # skip over the first two year of predictions due to high error for xgb / lgbm
        val_predictions = np.append(val_predictions, val_predict, axis=0)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    # store the current predictions in the results tracker
    val_df = df_train.loc[df_train.year.isin(years), 
                          ['player', 'year', 'avg_pick', 'y_act']].reset_index(drop=True)
    val_pred = pd.concat([val_df, pd.Series(val_predictions, name='pred')], axis=1)

    # calculate the RMSE and MAE of the ensemble predictions
    rmse = round(np.sqrt(mean_squared_error(val_pred.pred, val_df.y_act)), 3)

    return rmse

@ignore_warnings(category=ConvergenceWarning)
def run_best_model(model, p, skip_years):

    corr_cut = p['corr_cutoff']
    col_cut= p['collinear_cutoff']
    scale = p['scale']
    pca = p['pca']
    n_components = p['n_components']
    
    model_p = {}
    for k, v in p.items():
        if k not in ['corr_cutoff', 'collinear_cutoff', 'scale', 'pca', 'n_components']:
            model_p[k] = v

    est = models[model]
    est.set_params(**model_p)

    result, val_pred, ty_pred, trained_est, cols = validation(est, df_train_orig, df_predict, corr_cut, 
                                                              col_cut, skip_years=skip_years, scale=scale, pca=pca, n_components=n_components)

    return result, val_pred, ty_pred, trained_est, cols


# -

# ### Get Optimal Parameters

for metric in pos[set_pos]['metrics']:
    
    print(metric)
    
    df_train, df_predict = get_train_predict(df, metric, pos, set_pos, set_year, pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    print(f'Number of Training Samples: {df_train.shape[0]}')
    print(f'Number of Training Features: {df_train.shape[1]}\n')

    for c in df_train.columns:
        if len(df_train[df_train[c]==np.inf]) > 0:
            print(c)
            df_train = df_train.drop(c, axis=1)

    for model in list(models.keys()):

        print(model)
        estimator = models[model]
        space = search_space[model]

        x0=None
        df_train_orig = df_train.copy()
        skip_years = pos[set_pos]['skip_years']

        @use_named_args(space)
        def objective(**args):
            return calc_rmse(**args)

        @use_named_args(space)
        def space_keys(**args):
            return list(args.keys())

        res_gp = gp_minimize(objective, space, n_calls=100, n_random_starts=25, x0=x0,
                            random_state=1234, verbose=True, kappa=2., n_jobs=-1)

        params_output = dict(zip(space_keys(space), res_gp.x))
        result, _, _, _, _ = run_best_model(model, params_output, skip_years)

        output.loc[0, 'metric'] = metric
        output.loc[0, 'model'] = model
        output.loc[0, 'rmse_validation'] = result[2]
        output.loc[0, 'rmse_validation_adp'] = result[3]
        output.loc[0, 'r2_validation'] = result[0]
        output.loc[0, 'r2_validation_adp'] = result[1]
        output.loc[0, 'earliest_year'] = df_train_orig.year.min()

        output.to_sql('RegParamTracking', param_conn, if_exists='append', index=False)
        max_pkey = pd.read_sql_query("SELECT max(pkey) FROM RegParamTracking", param_conn).values[0][0]

        save_pickle(params_output, path + f'Data/Model_Params/{max_pkey}.p')
        save_pickle(df_train_orig, path + f'Data/Model_Datasets/{max_pkey}.p')    
        save_pickle(search_space[model], path + f'Data/Bayes_Space/{max_pkey}.p')
