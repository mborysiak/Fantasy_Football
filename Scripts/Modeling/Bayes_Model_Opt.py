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
pos['RB']['skip_years'] = 7
pos['WR']['skip_years'] = 10
pos['TE']['skip_years'] = 4

pos['QB']['features'] = 'v2'
pos['RB']['features'] = 'v1'
pos['WR']['features'] = 'v1'
pos['TE']['features'] = 'v2'

pos['QB']['minimizer'] = 'forest_minimize'
pos['RB']['minimizer'] = 'forest_minimize'
pos['WR']['minimizer'] = 'gp_minimize'
pos['TE']['minimizer'] = 'gp_minimize'

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
    'minimizer': pos[set_pos]['minimizer'],
    'adp_ppg_low': [None],
    'adp_ppg_high': [None],
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
    'minimizer': pos[set_pos]['minimizer'],
    'use_ay': [pos[set_pos]['use_ay']],
    'act_ppg': [None],
    'pct_off': [None], 
    'adp_ppg_low': [None],
    'adp_ppg_high': [None],
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

# ## Learnings for the future
# - Simplified v1 features seem to outperform v2 features slightly
# - Predicting FP only outperforms prediction stats categories + calculating
# - Splitting class predictions based on high and low ADP outperforms predicting all together
#
# ### Q's to decide
# - GP vs Forest Minimize?
# - Split data for both class and reg?
# - Model FP only or stats or both?
# - V1 or V2 features?
# - Leave out part of dataset for validation?
# - How many breakout variables?

# +
#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
act_ppg = '>=12'
pct_off = '>0.15'
adp_ppg_high = '<100'
adp_ppg_low = '>=12'

# get the train and prediction dataframes for FP per game
df_train_orig, df_predict_orig = get_train_predict(df, breakout_metric, pos, set_pos, set_year-3, pos[set_pos]['earliest_year'], pos[set_pos]['features'])
df_predict_orig['y_act'] = (df_predict_orig[pos[set_pos]['metrics']] * pts_dict[set_pos]).sum(axis=1)

#-----------
# Get ADP Predictions
#-----------

# get the adp predictions and merge back to the training dataframe
df_train_adp, lr = get_adp_predictions(df_train_orig, 1)
df_train_orig = pd.merge(df_train_orig, df_train_adp, on=['player', 'year'])

# use the linear regression object to predict out ADP and pct_off
df_predict_orig['avg_pick_pred'] = lr.predict(df_predict_orig.avg_pick.values.reshape(-1,1))
df_predict_orig['pct_off'] = (df_predict_orig.y_act - df_predict_orig.avg_pick_pred) / df_predict_orig.avg_pick_pred

#-----------
# Set binary labels 
#-----------

# create the label and filter based on inputs
df_train_orig['label'] = 0
df_train_orig.loc[(eval(f'df_train_orig.pct_off{pct_off}')) & (eval(f'df_train_orig.y_act{act_ppg}')), 'label'] = 1

df_predict_orig['label'] = 0
df_predict_orig.loc[(eval(f'df_predict_orig.pct_off{pct_off}')) & (eval(f'df_predict_orig.y_act{act_ppg}')), 'label'] = 1

#-----------
# Filter Dataset
#-----------

df_train_orig = df_train_orig[(eval(f'df_train_orig.avg_pick_pred{adp_ppg_high}')) & (eval(f'df_train_orig.avg_pick_pred{adp_ppg_low}'))].reset_index(drop=True)
df_train_orig = df_train_orig.drop(['y_act', 'pct_off'], axis=1).rename(columns={'label': 'y_act'})

df_predict_orig = df_predict_orig[(eval(f'df_predict_orig.avg_pick_pred{adp_ppg_high}')) & (eval(f'df_predict_orig.avg_pick_pred{adp_ppg_low}'))].reset_index(drop=True)
df_predict_orig = df_predict_orig.drop(['y_act', 'pct_off'], axis=1).rename(columns={'label': 'y_act'})

#-----------
# Print out dataset statistics
#-----------

# get the minimum number of training samples for the initial datasets
min_samples = int(0.5*df_train_orig[df_train_orig.year <= df_train_orig.year.min() + pos[set_pos]['skip_years']].shape[0])

# print the value-counts
print('Training Value Counts:', df_train_orig.y_act.value_counts()[0], '|', df_train_orig.y_act.value_counts()[1])
print('Predict Value Counts:', df_predict_orig.y_act.value_counts()[0], '|', df_predict_orig.y_act.value_counts()[1])
print(f'Number of Features: {df_train_orig.shape[1]}')
print('Min Train Year:', df_train_orig.year.min())
print('Max Train Year:', df_train_orig.year.min() + pos[set_pos]['skip_years'])
print('Min Val Year:', df_train_orig.year.min() + pos[set_pos]['skip_years']+1)
print('Max Val Year:', df_train_orig.year.max())
print('Min Test Year:', df_predict_orig.year.min())
print('Max Test Year:', df_predict_orig.year.max())


# -

@ignore_warnings(category=ConvergenceWarning)
def calc_f1_score(**args):
    
    globals()['cnter'] += 1
    
    from sklearn.metrics import f1_score
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

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
    df_train = remove_classification_collinear(globals()['df_train_orig'], collinear_cutoff, ['player', 'avg_pick', 'year', 'y_act'])
    df_predict = globals()['df_predict_orig'][df_train.columns]
    
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]
    
    # take a random sample
    df_train_val = df_train[df_train.year.isin(years)].copy().reset_index(drop=True)
    df_train_only = df_train[df_train.year < np.min(years)].copy().reset_index(drop=True)

    skf = StratifiedKFold(n_splits=4, random_state=globals()['cnter']*3+globals()['cnter']+4+7, shuffle=True)
    for train_index, test_index in skf.split(df_train_val, df_train_val.y_act):
        pass

    df_val = df_train_val.iloc[test_index, :].reset_index(drop=True)
    df_train = pd.concat([df_train_only, df_train_val.iloc[train_index, :]], axis=0).reset_index(drop=True)
 
    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    y_actuals = np.array([])

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]
        val_split = pd.concat([val_split, df_val[df_val.year==m]], axis=0).reset_index(drop=True)
        
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
        y_actuals = np.append(y_actuals, y_val, axis=0)
    
    # train for testing on all data
    df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    
    # splitting the train and validation sets into X_train, y_train, X_val and y_val
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)
    
    if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)
            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_test = X_test.values
            
    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========
    
    val_score = round(-matthews_corrcoef(val_predictions, y_actuals), 3)
    test_score = round(-matthews_corrcoef(test_predict, y_test), 3)
    print('Val Score:', val_score)
    print('Test Score:', test_score)
    
    # set weights for validation to 1 and for test set to 2
    wts = [1]*len(val_predictions)
    wts.extend([1]*len(y_test))
    
    # skip over the first two year of predictions due to high error for xgb / lgbm
    val_predictions = np.append(val_predictions, test_predict, axis=0)
    y_actuals = np.append(y_actuals, y_test, axis=0)
   
    # calculate the RMSE and MAE of the ensemble predictions
    m_score = -round(matthews_corrcoef(val_predictions, y_actuals, sample_weight=wts), 3)
    print('Combined Score:', m_score)
    
    globals()['val_scores3'].append(val_score)    
    globals()['test_scores3'].append(test_score)
    globals()['combined_scores3'].append(m_score)
    
    return val_score


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
        Real(0.01, 0.5, name='n_components')
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
        Real(0.01, 0.5, name='n_components')
    ],
        
    'lr': [
        Real(0.00001, 100000000, 'log_uniform', name='C'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 0.5, name='n_components')
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
        Real(0.01, 0.5, name='n_components')
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
        Real(0.01, 0.5, name='n_components')
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
        Real(0.01, 0.5, name='n_components')
    ],
    
    'svr': [
        Real(0.0001, 1000, 'log_uniform', name='C'),
        Real(.2, 1, name='collinear_cutoff'),
        Integer(0, 1, name='scale'),
        Integer(0, 1, name='pca'),
        Integer(0, 1, name='use_smote'),
        Real(0.1, 1, name='zero_weight'),
        Real(0.01, 0.5, name='n_components')
    ]
}
# +
x0=None
skip_years = pos[set_pos]['skip_years']
cnter = 1

val_scores3 = []
test_scores3 = []
combined_scores3 = []
models_list3 = []

for model in class_models.keys():
    
    models_list3.extend([model]*100)
    
    estimator = class_models[model]
    space = class_search_space[model]
    
    @use_named_args(space)
    def objective_class(**args):
        return calc_f1_score(**args)
    
    @use_named_args(space)
    def space_keys(**args):
        return list(args.keys())

    bayes_seed = 12345
    kappa = 3
    res_gp = eval(pos[set_pos]['minimizer'])(objective_class, space, n_calls=100, n_random_starts=25, x0=x0,
                        random_state=bayes_seed, verbose=True, kappa=kappa, n_jobs=-1)

    output_class.loc[0, 'breakout_metric'] = breakout_metric
    output_class.loc[0, 'act_ppg'] = act_ppg
    output_class.loc[0, 'pct_off'] = pct_off
    output_class.loc[0, 'adp_ppg_high'] = adp_ppg_high
    output_class.loc[0, 'adp_ppg_low'] = adp_ppg_low
    output_class.loc[0, 'model'] = model
    output_class.loc[0, 'earliest_year'] = df_train_orig.year.min()
    output_class.loc[0, 'score'] = -res_gp.fun
    
    params_output = dict(zip(space_keys(space), res_gp.x))
    
    results3 = pd.DataFrame([val_scores3,combined_scores3,test_scores3]).T
    results3.columns = ['val', 'combined', 'test']
    results3['x'] = range(len(results3))
    print(results3.corr())

#     append_to_db(output_class, db_name='ParamTracking', table_name='ClassParamTracking', if_exist='append')
#     max_pkey = pd.read_sql_query("SELECT max(pkey) FROM ClassParamTracking", param_conn).values[0][0]

#     save_pickle(params_output, path + f'Data/Model_Params_Class/{max_pkey}.p')
#     save_pickle(df_train_orig, path + f'Data/Model_Datasets_Class/{max_pkey}.p')    
#     save_pickle(class_search_space[model], path + f'Data/Bayes_Space_Class/{max_pkey}.p')
# -

@ignore_warnings(category=ConvergenceWarning)
def calc_f1_score(**args):
    
    globals()['cnter'] += 1
    
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
    df_train = remove_classification_collinear(globals()['df_train_orig'], collinear_cutoff, ['player', 'avg_pick', 'year', 'y_act'])
    df_predict = globals()['df_predict_orig'][df_train.columns]
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    y_actuals = np.array([])
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
        y_actuals = np.append(y_actuals, y_val, axis=0)
            
    # splitting the train and validation sets into X_train, y_train, X_val and y_val
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)
    
    if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)
            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_test = X_test.values
            
    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========
    
    val_score = round(-matthews_corrcoef(val_predictions, y_actuals), 3)
    test_score = round(-matthews_corrcoef(test_predict, y_test), 3)
    print('Val Score:', val_score)
    print('Test Score:', test_score)
    
    # set weights for validation to 1 and for test set to 2
    wts = [1]*len(val_predictions)
    wts.extend([1]*len(y_test))
    
    # skip over the first two year of predictions due to high error for xgb / lgbm
    val_predictions = np.append(val_predictions, test_predict, axis=0)
    y_actuals = np.append(y_actuals, y_test, axis=0)
   
    # calculate the RMSE and MAE of the ensemble predictions
    m_score = -round(matthews_corrcoef(val_predictions, y_actuals, sample_weight=wts), 3)
    print('Combined Score:', m_score)
    
    globals()['val_scores2'].append(val_score)    
    globals()['test_scores2'].append(test_score)
    globals()['combined_scores2'].append(m_score)
    
    return val_score


# +
x0=None
skip_years = pos[set_pos]['skip_years']
cnter = 1

val_scores2 = []
test_scores2 = []
combined_scores2 = []
models_list2 = []

for model in class_models.keys():
    
    models_list2.extend([model]*100)
    
    estimator = class_models[model]
    space = class_search_space[model]
    
    @use_named_args(space)
    def objective_class(**args):
        return calc_f1_score(**args)
    
    @use_named_args(space)
    def space_keys(**args):
        return list(args.keys())

    bayes_seed = 12345
    kappa = 3
    res_gp = eval(pos[set_pos]['minimizer'])(objective_class, space, n_calls=100, n_random_starts=25, x0=x0,
                        random_state=bayes_seed, verbose=True, kappa=kappa, n_jobs=-1)

    output_class.loc[0, 'breakout_metric'] = breakout_metric
    output_class.loc[0, 'act_ppg'] = act_ppg
    output_class.loc[0, 'pct_off'] = pct_off
    output_class.loc[0, 'adp_ppg_high'] = adp_ppg_high
    output_class.loc[0, 'adp_ppg_low'] = adp_ppg_low
    output_class.loc[0, 'model'] = model
    output_class.loc[0, 'earliest_year'] = df_train_orig.year.min()
    output_class.loc[0, 'score'] = -res_gp.fun
    
    params_output = dict(zip(space_keys(space), res_gp.x))
    
    results2 = pd.DataFrame([val_scores2,combined_scores2,test_scores2]).T
    results2.columns = ['val', 'combined', 'test']
    results2['x'] = range(len(results2))
    print(results2.corr())

#     append_to_db(output_class, db_name='ParamTracking', table_name='ClassParamTracking', if_exist='append')
#     max_pkey = pd.read_sql_query("SELECT max(pkey) FROM ClassParamTracking", param_conn).values[0][0]

#     save_pickle(params_output, path + f'Data/Model_Params_Class/{max_pkey}.p')
#     save_pickle(df_train_orig, path + f'Data/Model_Datasets_Class/{max_pkey}.p')    
#     save_pickle(class_search_space[model], path + f'Data/Bayes_Space_Class/{max_pkey}.p')
# -

results2 = pd.DataFrame([val_scores2,combined_scores2,test_scores2]).T
results2.columns = ['val', 'combined', 'test']
results2['x'] = range(len(results2))
print(results2.corr())
results2.plot_bokeh(x='x', y=['val', 'combined', 'test'])

results3 = pd.DataFrame([val_scores3,combined_scores3,test_scores3]).T
results3.columns = ['val', 'combined', 'test']
results3['x'] = range(len(results3))
print(results3.corr())
results3.plot_bokeh(x='x', y=['val', 'combined', 'test'])

res_gp.x_iters[np.argmin(combined_scores3[200:300])]

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
min_samples = int(0.25*df_train[df_train.year <= df_train.year.min() + pos[set_pos]['skip_years']].shape[0])

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
        Real(0.0001, 10000, 'log_uniform', name='C'),
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
    
    globals()['cnter'] += 1

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
    df_train = corr_collinear_removal(globals()['df_train'], corr_cutoff, collinear_cutoff)
    df_predict = globals()['df_predict'][df_train.columns]
    
    # set years to loop through based on train and validation split (skip year)
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]

    # pull out the dataset that will be used within validation and training only data
    df_train_val = df_train[df_train.year.isin(years)].copy().reset_index(drop=True)
    df_train_only = df_train[df_train.year < np.min(years)].copy().reset_index(drop=True)
    
    # randomly split the validation set by holding out 25% of data to be predicted to prevent overfitting of validation set during optimization
    df_train_val, df_val, _, _ = train_test_split(df_train_val, df_train_val.y_act, test_size=0.3, random_state=globals()['cnter']*3 + globals()['cnter']*17, shuffle=True)
    
    # concat back the train only and 75% of validation data for rolling validation
    df_train = pd.concat([df_train_only, df_train_val], axis=0).reset_index(drop=True)
    
    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    y_actuals = np.array([])

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]
        
        # concat the holdout validation data to be predicted
        val_split = pd.concat([val_split, df_val[df_val.year==m]], axis=0).reset_index(drop=True)

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        # fit training data and creating prediction based on validation data
        estimator.set_params(**args)
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

        # skip over the first two year of predictions due to high error for xgb / lgbm
        val_predictions = np.append(val_predictions, val_predict, axis=0)
        y_actuals = np.append(y_actuals, y_val, axis=0)
        
    #==========
    # Predict the Test Set
    #==========
    
    # train for testing on all data
    df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    
    # splitting the train and test sets into X_train, y_train, X_test and y_test
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)
    
    # fit and predict the test data
    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    val_score = round(np.sqrt(mean_squared_error(val_predictions, y_actuals)), 3)
    test_score = round(np.sqrt(mean_squared_error(test_predict, y_test)), 3)
    print('Val Score:', val_score)
    print('Test Score:', test_score)
    
    # set weights for validation to 1 and for test set to 2
    wts = [1]*len(val_predictions)
    wts.extend([1]*len(y_test))
    
    # append the test to validation prediction for combined error
    val_predictions = np.append(val_predictions, test_predict, axis=0)
    y_actuals = np.append(y_actuals, y_test, axis=0)
   
    # calculate the RMSE of combined error
    combined_score = round(np.sqrt(mean_squared_error(val_predictions, y_actuals, sample_weight=wts)), 3)
    print('Combined Score:', combined_score)
    
    globals()['val_scores_reg'].append(val_score)    
    globals()['test_scores_reg'].append(test_score)
    globals()['combined_scores_reg'].append(combined_score)

    return val_score

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

# +
val_scores_reg = []
test_scores_reg = []
combined_scores_reg = []
cnter = 1

for metric in ['fp_per_game']:#pos[set_pos]['metrics']:
    
    print(metric)
    
    # get the train and predict dataframes
    df_train, df_predict = get_train_predict(df, metric, pos, set_pos, set_year-3, pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    df_predict['y_act'] = (df_predict[pos[set_pos]['metrics']] * pts_dict[set_pos]).sum(axis=1)

    # get the adp predictions and merge back to the training dataframe
    df_train_adp, lr = get_adp_predictions(df_train, 1)
    df_train = pd.merge(df_train, df_train_adp, on=['player', 'year'])
    df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))

    # filter based on adp predict
    df_train = df_train[(eval(f'df_train.avg_pick_pred{adp_ppg_low}')) & (eval(f'df_train.avg_pick_pred{adp_ppg_high}'))].reset_index(drop=True).drop(['pct_off'], axis=1)
    df_predict = df_predict[(eval(f'df_predict.avg_pick_pred{adp_ppg_high}')) & (eval(f'df_predict.avg_pick_pred{adp_ppg_low}'))].reset_index(drop=True)

    print(f'Number of Training Samples: {df_train.shape[0]}')
    print(f'Number of Training Features: {df_train.shape[1]}\n')
    print('Min Train Year:', df_train.year.min())
    print('Max Train Year:', df_train.year.min() + pos[set_pos]['skip_years'])
    print('Min Val Year:', df_train.year.min() + pos[set_pos]['skip_years']+1)
    print('Max Val Year:', df_train.year.max())
    print('Min Test Year:', df_predict.year.min())
    print('Max Test Year:', df_predict.year.max())
    
    for model in list(models.keys()):

        print(model)
        estimator = models[model]
        space = search_space[model]

        x0=None
        skip_years = pos[set_pos]['skip_years']

        @use_named_args(space)
        def objective(**args):
            return calc_rmse(**args)

        @use_named_args(space)
        def space_keys(**args):
            return list(args.keys())

        res_gp = eval(pos[set_pos]['minimizer'])(objective, space, n_calls=100, n_random_starts=25, x0=x0,
                            random_state=1234, verbose=True, kappa=2., n_jobs=-1)
        
        results_reg = pd.DataFrame([val_scores_reg,combined_scores_reg,test_scores_reg]).T
        results_reg .columns = ['val', 'combined', 'test']
        results_reg ['x'] = range(len(results_reg ))
        print(results_reg .corr())
        print(np.min(combined_scores_reg))
        print(test_scores_reg[np.argmin(val_scores_reg)])
#         params_output = dict(zip(space_keys(space), res_gp.x))
#         result, _, _, _, _ = run_best_model(model, params_output, skip_years)

#         output.loc[0, 'metric'] = metric
#         output.loc[0, 'model'] = model
#         output.loc[0, 'rmse_validation'] = result[2]
#         output.loc[0, 'rmse_validation_adp'] = result[3]
#         output.loc[0, 'r2_validation'] = result[0]
#         output.loc[0, 'r2_validation_adp'] = result[1]
#         output.loc[0, 'earliest_year'] = df_train_orig.year.min()
#         output.loc[0, 'adp_ppg_low'] = adp_ppg_low
#         output.loc[0, 'adp_ppg_high'] = adp_ppg_high

        
#         append_to_db(output, db_name='ParamTracking', table_name='RegParamTracking', if_exist='append')
#         max_pkey = pd.read_sql_query("SELECT max(pkey) FROM RegParamTracking", param_conn).values[0][0]

#         save_pickle(params_output, path + f'Data/Model_Params/{max_pkey}.p')
#         save_pickle(df_train_orig, path + f'Data/Model_Datasets/{max_pkey}.p')    
#         save_pickle(search_space[model], path + f'Data/Bayes_Space/{max_pkey}.p')
# -
results_reg = pd.DataFrame([val_scores_reg,combined_scores_reg,test_scores_reg]).T
results_reg .columns = ['val', 'combined', 'test']
results_reg ['x'] = range(len(results_reg ))
print(results_reg .corr())

results_reg.plot_bokeh(x='x', y=['val', 'combined', 'test'])

results_ns.plot_bokeh(x='x', y=['val', 'combined', 'test'])


@ignore_warnings(category=ConvergenceWarning)
def calc_rmse(**args):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    print('\n', args)
    
    globals()['cnter'] += 1

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
    df_train = corr_collinear_removal(globals()['df_train'], corr_cutoff, collinear_cutoff)
    df_predict = globals()['df_predict'][df_train.columns]

    # set years to loop through based on train and validation split (skip year)
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]

    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    y_actuals = np.array([])

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
        y_actuals = np.append(y_actuals, y_val, axis=0)
        
    #==========
    # Predict the Test Set
    #==========
    
    # splitting the train and test sets into X_train, y_train, X_test and y_test
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)
    
    # fit and predict the test data
    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    val_score = round(np.sqrt(mean_squared_error(val_predictions, y_actuals)), 3)
    test_score = round(np.sqrt(mean_squared_error(test_predict, y_test)), 3)
    print('Val Score:', val_score)
    print('Test Score:', test_score)
    
    # set weights for validation to 1 and for test set to 2
    wts = [1]*len(val_predictions)
    wts.extend([1]*len(y_test))
    
    # append the test to validation prediction for combined error
    val_predictions = np.append(val_predictions, test_predict, axis=0)
    y_actuals = np.append(y_actuals, y_test, axis=0)
   
    # calculate the RMSE of combined error
    combined_score = round(np.sqrt(mean_squared_error(val_predictions, y_actuals, sample_weight=wts)), 3)
    print('Combined Score:', combined_score)
    
    globals()['val_scores_ns'].append(val_score)    
    globals()['test_scores_ns'].append(test_score)
    globals()['combined_scores_ns'].append(combined_score)

    return val_score


# +
val_scores_ns = []
test_scores_ns = []
combined_scores_ns = []

for metric in ['fp_per_game']:#pos[set_pos]['metrics']:
    
    print(metric)
    
    # get the train and predict dataframes
    df_train, df_predict = get_train_predict(df, metric, pos, set_pos, set_year-3, pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    df_predict['y_act'] = (df_predict[pos[set_pos]['metrics']] * pts_dict[set_pos]).sum(axis=1)

    # get the adp predictions and merge back to the training dataframe
    df_train_adp, lr = get_adp_predictions(df_train, 1)
    df_train = pd.merge(df_train, df_train_adp, on=['player', 'year'])
    df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))

    # filter based on adp predict
    df_train = df_train[(eval(f'df_train.avg_pick_pred{adp_ppg_low}')) & (eval(f'df_train.avg_pick_pred{adp_ppg_high}'))].reset_index(drop=True).drop(['pct_off'], axis=1)
    df_predict = df_predict[(eval(f'df_predict.avg_pick_pred{adp_ppg_high}')) & (eval(f'df_predict.avg_pick_pred{adp_ppg_low}'))].reset_index(drop=True)

    print(f'Number of Training Samples: {df_train.shape[0]}')
    print(f'Number of Training Features: {df_train.shape[1]}\n')
    print('Min Train Year:', df_train.year.min())
    print('Max Train Year:', df_train.year.min() + pos[set_pos]['skip_years'])
    print('Min Val Year:', df_train.year.min() + pos[set_pos]['skip_years']+1)
    print('Max Val Year:', df_train.year.max())
    print('Min Test Year:', df_predict.year.min())
    print('Max Test Year:', df_predict.year.max())
    
    for model in list(models.keys()):

        print(model)
        estimator = models[model]
        space = search_space[model]

        x0=None
        skip_years = pos[set_pos]['skip_years']

        @use_named_args(space)
        def objective(**args):
            return calc_rmse(**args)

        @use_named_args(space)
        def space_keys(**args):
            return list(args.keys())

        res_gp = eval(pos[set_pos]['minimizer'])(objective, space, n_calls=100, n_random_starts=25, x0=x0,
                            random_state=1234, verbose=True, kappa=2., n_jobs=-1)
        
        results_ns = pd.DataFrame([val_scores_ns,combined_scores_ns,test_scores_ns]).T
        results_ns .columns = ['val', 'combined', 'test']
        results_ns ['x'] = range(len(results_ns ))
        print(results_ns.corr())
        print(np.min(combined_scores_ns))
        print(test_scores_ns[np.argmin(val_scores_ns)])

#         params_output = dict(zip(space_keys(space), res_gp.x))
#         result, _, _, _, _ = run_best_model(model, params_output, skip_years)

#         output.loc[0, 'metric'] = metric
#         output.loc[0, 'model'] = model
#         output.loc[0, 'rmse_validation'] = result[2]
#         output.loc[0, 'rmse_validation_adp'] = result[3]
#         output.loc[0, 'r2_validation'] = result[0]
#         output.loc[0, 'r2_validation_adp'] = result[1]
#         output.loc[0, 'earliest_year'] = df_train_orig.year.min()
#         output.loc[0, 'adp_ppg_low'] = adp_ppg_low
#         output.loc[0, 'adp_ppg_high'] = adp_ppg_high

        
#         append_to_db(output, db_name='ParamTracking', table_name='RegParamTracking', if_exist='append')
#         max_pkey = pd.read_sql_query("SELECT max(pkey) FROM RegParamTracking", param_conn).values[0][0]

#         save_pickle(params_output, path + f'Data/Model_Params/{max_pkey}.p')
#         save_pickle(df_train_orig, path + f'Data/Model_Datasets/{max_pkey}.p')    
#         save_pickle(search_space[model], path + f'Data/Bayes_Space/{max_pkey}.p')
# -

results_reg.plot_bokeh(x='x', y=['val', 'combined', 'test'])


