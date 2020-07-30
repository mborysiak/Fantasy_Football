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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

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
pos['RB']['features'] = 'v2'
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
# -

# # Breakout Models

# ## Learnings for the future
# - Model FP only or stats or both?: Predicting FP only outperforms prediction stats categories + calculating
# - Split data for both class and reg?: Splitting predictions based on high and low ADP outperforms predicting all together
# - Splitting out samples of validation training set doesn't necessarily improve overall model tuning performance
# - GP vs Forest Minimize?--it depends, but Forest is slightly better
# - V1 or V2 features?: Varies by position and subset
# - Leave out part of dataset for validation?: Yes, leave out testing dataset and use to determine final model params
# - How many breakout variables?

# +
#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
act_ppg = '>=0'
pct_off = '>0.2'
adp_ppg_high = '<100'
adp_ppg_low = '>=10'

# get the train and prediction dataframes for FP per game
df_train_orig, df_predict_orig = get_train_predict(df, breakout_metric, pos, set_pos, set_year-2, pos[set_pos]['earliest_year'], pos[set_pos]['features'])
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

import seaborn as sns
idx = abs(df_train_orig.corr()['y_act'].dropna()).sort_values(ascending=False).index[:25]
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df_train_orig.corr().loc[idx, idx], cmap='coolwarm', robust=True, annot=True, fmt=".2f",annot_kws={'size':10})


def val_results(models, val, combined, test):
    
    # create dataframe out of results lists
    results = pd.DataFrame([models, val, combined, test]).T
    results.columns = ['model', 'val', 'combined', 'test']
    
    # convert all numeric columns to float
    for c in ['val', 'combined', 'test']:
        results[c] = results[c].astype('float')
        
    # create an x column to track improvements over time
    results['x'] = [x for x in range(100)]*int(len(results)/100)
    
    # print out correlation information
    print(results.corr(), '\n')
    
    # determine optimal model location
    min_models = results.groupby('model').agg({'combined': np.argmin})
    min_models = min_models.reset_index().rename(columns={'combined': 'x'})
    
    # print out optimum model data
    print(pd.merge(results, min_models, on=['model', 'x']))
    
    return results


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
@ignore_warnings(category=ConvergenceWarning)
def calc_f1_score(**args):
    
    globals()['cnter'] += 1
    i = globals()['cnter']
    if i % 25 == 0:
        print(f'\n---------- Run {i} Completed\n')
        
    from sklearn.metrics import f1_score
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

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
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    opt_predictions = np.array([]) 
    y_vals = np.array([])
    y_opts = np.array([])
    
#     #==============
#     # K-Fold Holdout Validation Loop for Optimization
#     #==============
    
#     skf = StratifiedKFold(n_splits=5, random_state=i*3+i*13+12, shuffle=True)
#     for train_index, test_index in skf.split(df_train_orig, df_train_orig.year):
        
#         train_fold, holdout = df_train.iloc[train_index, :], df_train.iloc[test_index, :]
#         train_fold = train_fold.sort_values(by='year').reset_index(drop=True)

#         for m in years:

#             # create training set for all previous years and validation set for current year
#             train_split = train_fold[train_fold.year < m]
#             opt_split = train_fold[train_fold.year == m]

#             # set up the estimator
#             estimator.set_params(**args)
#             estimator.class_weight = {0: zero_weight, 1: 1}

#             # splitting the train and validation sets into X_train, y_train, X_val and y_val
#             X_train, X_opt, y_train, y_opt = X_y_split(train_split, opt_split, scale, pca, n_components)

#             if use_smote:
#                 knn = int(len(y_train[y_train==1])*0.5)
#                 smt = SMOTE(k_neighbors=knn, random_state=1234)

#                 X_train, y_train = smt.fit_resample(X_train.values, y_train)
#                 X_opt = X_opt.values

#             # train the estimator and get predictions
#             estimator.fit(X_train, y_train)
#             opt_predict = estimator.predict(X_opt)

#             # append the predictions
#             opt_predictions = np.append(opt_predictions, opt_predict, axis=0)
#             y_opts = np.append(y_opts, y_opt, axis=0)

    #-=============
    # Full Validation Loop Train and Predict
    #==============
    
    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]

        # set up the estimator
        estimator.set_params(**args)
        estimator.class_weight = {0: zero_weight, 1: 1}

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)

            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_val = X_val.values

        # train the estimator and get predictions
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

        # append the predictions
        val_predictions = np.append(val_predictions, val_predict, axis=0)
        y_vals = np.append(y_vals, y_val, axis=0)

        
#     #==========
#     # Full Model Train + Test Set Prediction
#     #==========
    
#     # splitting the train and validation sets into X_train, y_train, X_val and y_val
#     X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)

#     if use_smote:
#             knn = int(len(y_train[y_train==1])*0.5)
#             smt = SMOTE(k_neighbors=knn, random_state=1234)
#             X_train, y_train = smt.fit_resample(X_train.values, y_train)
#             X_test = X_test.values

#     estimator.fit(X_train, y_train)
#     test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========
    
    val_score = round(-matthews_corrcoef(val_predictions, y_vals), 3)
    opt_score = val_score#round(-matthews_corrcoef(opt_predictions, y_opts), 3)
#     test_score = round(-matthews_corrcoef(test_predict, y_test), 3)
    
#     m_score = round(np.mean([opt_score, val_score, test_score]), 3)
    m_score = round(np.mean([opt_score]), 3)
    test_score = None
    
    if i == 25:
        print('\nBest Random Scores:')
        print('OptScore:',globals()['best_opt'])
        print('CombinedScore:', globals()['best_combined'])
    
    if opt_score < globals()['best_opt']:
        globals()['best_opt'] = opt_score
        
        if i > 25:
            print('\nNew Best Opt Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
        
    if m_score < globals()['best_combined']:
        globals()['best_combined'] = m_score
        
        if i > 25:
            print('\nNew Best Combined Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
    
    globals()['opt_scores'].append(opt_score) 
    globals()['val_scores'].append(val_score)    
    globals()['test_scores'].append(test_score)
    globals()['combined_scores'].append(m_score)
    
    return opt_score


#================
# Run Optimization
#================
    
opt_scores = []
val_scores = []
test_scores = []
combined_scores = []
models_list = []

skip_years = pos[set_pos]['skip_years']

for m_num, model in enumerate(list(class_models.keys())):
    
    cnter = 1

    print(f'\n============= Running {model} =============\n')

    best_opt = 100
    best_combined = 100
    models_list.extend([model]*100)

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
    res_gp = eval(pos[set_pos]['minimizer'])(objective_class, space, n_calls=100, n_random_starts=25,
                                             random_state=bayes_seed, verbose=False, kappa=kappa, n_jobs=-1)

    output_class.loc[0, 'breakout_metric'] = breakout_metric
    output_class.loc[0, 'act_ppg'] = act_ppg
    output_class.loc[0, 'pct_off'] = pct_off
    output_class.loc[0, 'adp_ppg_high'] = adp_ppg_high
    output_class.loc[0, 'adp_ppg_low'] = adp_ppg_low
    output_class.loc[0, 'model'] = model
    output_class.loc[0, 'earliest_year'] = df_train_orig.year.min()
    output_class.loc[0, 'score'] = -best_combined
        
    _ = val_results(models_list, val_scores, combined_scores, test_scores)

    best_params_model = res_gp.x_iters[np.argmin(combined_scores[m_num*100:(m_num+1)*100])]
    
    print('Best Combined Score:', best_combined)
    params_output = dict(zip(space_keys(space), best_params_model))
    print(params_output)
    
    append_to_db(output_class, db_name='ParamTracking', table_name='ClassParamTracking', if_exist='append')
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
    
    globals()['cnter'] += 1
    i = globals()['cnter']
    if i % 25 == 0:
        print(f'\n---------- Run {i} Completed\n')
        
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import StratifiedKFold
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
    
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    opt_predictions = np.array([]) 
    y_vals = np.array([])
    y_opts = np.array([])
    
#     #==============
#     # K-Fold Holdout Validation Loop for Optimization
#     #==============
    
#     skf = StratifiedKFold(n_splits=5, random_state=i*3+i*13+12, shuffle=True)
#     for train_index, test_index in skf.split(df_train_orig, df_train_orig.year):
        
#         train_fold, holdout = df_train.iloc[train_index, :], df_train.iloc[test_index, :]
#         train_fold = train_fold.sort_values(by='year').reset_index(drop=True)

#         for m in years:

#             # create training set for all previous years and validation set for current year
#             train_split = train_fold[train_fold.year < m]
#             opt_split = train_fold[train_fold.year == m]

#             # set up the estimator
#             estimator.set_params(**args)

#             # splitting the train and validation sets into X_train, y_train, X_val and y_val
#             X_train, X_opt, y_train, y_opt = X_y_split(train_split, opt_split, scale, pca, n_components)

#             # train the estimator and get predictions
#             estimator.fit(X_train, y_train)
#             opt_predict = estimator.predict(X_opt)

#             # append the predictions
#             opt_predictions = np.append(opt_predictions, opt_predict, axis=0)
#             y_opts = np.append(y_opts, y_opt, axis=0)

    #-=============
    # Full Validation Loop Train and Predict
    #==============
    
    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        # train the estimator and get predictions
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

        # append the predictions
        val_predictions = np.append(val_predictions, val_predict, axis=0)
        y_vals = np.append(y_vals, y_val, axis=0)

        
#     #==========
#     # Full Model Train + Test Set Prediction
#     #==========
    
#     # splitting the train and validation sets into X_train, y_train, X_val and y_val
#     X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)

#     estimator.fit(X_train, y_train)
#     test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========
    
    val_score = round(np.sqrt(mean_squared_error(val_predictions, y_vals)), 3)
    opt_score = val_score#round(np.sqrt(mean_squared_error(opt_predictions, y_opts)), 3)
    test_score = None#round(np.sqrt(mean_squared_error(test_predict, y_test)), 3)
    
#     m_score = round(np.mean([opt_score, val_score, test_score]), 3)
    m_score = round(np.mean([opt_score, val_score]), 3)

    if i == 25:
        print('\nBest Random Scores:')
        print('OptScore:',globals()['best_opt'])
        print('CombinedScore:', globals()['best_combined'])
    
    if opt_score < globals()['best_opt']:
        globals()['best_opt'] = opt_score
        
        if i > 25:
            print('\nNew Best Opt Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
        
    if m_score < globals()['best_combined']:
        globals()['best_combined'] = m_score
        
        if i > 25:
            print('\nNew Best Combined Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
    
    globals()['opt_scores'].append(opt_score) 
    globals()['val_scores'].append(val_score)    
    globals()['test_scores'].append(test_score)
    globals()['combined_scores'].append(m_score)
    
    return opt_score


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
    print(est)
    print(corr_cut)
    print(col_cut)
    print(scale)
    print(pca)
    print(n_components)
    result, val_pred, ty_pred, trained_est, cols = validation(est,  df_train, df_predict, corr_cut, col_cut, 
                                                              skip_years=skip_years,  scale=scale, pca=pca, 
                                                              n_components=n_components)

    return result, val_pred, ty_pred, trained_est, cols


# -

# ### Get Optimal Parameters

# +
#================
# Run Optimization
#================
    
opt_scores = []
val_scores = []
test_scores = []
combined_scores = []
models_list = []

for metric in ['fp_per_game']:#pos[set_pos]['metrics']:
    
    if metric == 'pct_off':
        met = 'fp_per_game'
    else:
        met = metric
    
    # get the train and predict dataframes
    df_train, df_predict = get_train_predict(df, met, pos, set_pos, set_year-2, 
                                             pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    df_predict['y_act'] = (df_predict[pos[set_pos]['metrics']] * pts_dict[set_pos]).sum(axis=1)

    # get the adp predictions and merge back to the training dataframe
    df_train_adp, lr = get_adp_predictions(df_train, 1)
    df_train = pd.merge(df_train, df_train_adp, on=['player', 'year'])
    df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))

    if metric == 'pct_off':
        df_train['y_act'] = df_train.pct_off
        
    print(df_train.y_act)
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
    
    for m_num, model in enumerate(list(models.keys())):

        cnter = 1

        print(f'\n============= Running {model} =============\n')

        best_opt = 100
        best_combined = 100
        models_list.extend([model]*100)
        
        # get the estimator and search params
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
                            random_state=1234, verbose=False, kappa=2., n_jobs=-1)
        
        params_output = dict(zip(space_keys(space), res_gp.x))
        result, _, _, _, _ = run_best_model(model, params_output, skip_years)

        output.loc[0, 'metric'] = metric
        output.loc[0, 'model'] = model
        output.loc[0, 'rmse_validation'] = result[2]
        output.loc[0, 'rmse_validation_adp'] = result[3]
        output.loc[0, 'r2_validation'] = result[0]
        output.loc[0, 'r2_validation_adp'] = result[1]
        output.loc[0, 'earliest_year'] = df_train_orig.year.min()
        output.loc[0, 'adp_ppg_low'] = adp_ppg_low
        output.loc[0, 'adp_ppg_high'] = adp_ppg_high
        
        # print out the validation results
        _ = val_results(models_list, val_scores, combined_scores, test_scores)

        # get the best model params
        best_params_model = res_gp.x_iters[np.argmin(combined_scores[m_num*100:(m_num+1)*100])]

        print('Best Combined Score:', best_combined)
        params_output = dict(zip(space_keys(space), best_params_model))
        print(params_output)
        
        append_to_db(output, db_name='ParamTracking', table_name='RegParamTracking', if_exist='append')
        max_pkey = pd.read_sql_query("SELECT max(pkey) FROM RegParamTracking", param_conn).values[0][0]

        save_pickle(params_output, path + f'Data/Model_Params/{max_pkey}.p')
        save_pickle(df_train_orig, path + f'Data/Model_Datasets/{max_pkey}.p')    
        save_pickle(search_space[model], path + f'Data/Bayes_Space/{max_pkey}.p')
# -

