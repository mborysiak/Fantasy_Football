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

import pandas_bokeh
pandas_bokeh.output_notebook()

# +
#==========
# General Setting
#==========

# set core path
path = '/Users/Mark/Documents/Github/Fantasy_Football/'

# load the helper functions
os.chdir(path + 'Scripts/Analysis/Helper_Scripts')
from helper_functions import *
os.chdir(path)

# specify database name with model data
db_name = 'Model_Inputs.sqlite3'

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'WR'

# set year to analyze
set_year = 2019

# set path to param database
param_conn = sqlite3.connect(path + 'Data/ParamTracking.sqlite3')

#==========
# Postgres Database
#==========

# postgres login information
pg_log = {
    'USER': 'postgres',
    'PASSWORD': 'Ctdim#1bf!!!!!',
    'HOST': 'localhost',
    'PORT': '5432', 
    'DATABASE_NAME': 'fantasyfootball'
}

# # create engine for connecting to database
# engine = create_engine('postgres+psycopg2://{}:{}@{}:{}/{}'.format(pg_log['USER'], pg_log['PASSWORD'], pg_log['HOST'],
#                                                                    pg_log['PORT'], pg_log['DATABASE_NAME']))

# # specify schema and table to write out intermediate results
# table_info = {
#     'engine': engine,
#     'schema': 'websitedev',
# }

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

pos['QB']['earliest_year'] = 2003
pos['RB']['earliest_year'] = 1998
pos['WR']['earliest_year'] = 1998
pos['TE']['earliest_year'] = 1998

pos['QB']['skip_years'] = 10
pos['RB']['skip_years'] = 10
pos['WR']['skip_years'] = 10
pos['TE']['skip_years'] = 4

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
    'earliest_year': [pos[set_pos]['earliest_year']],
    'skip_years': [pos[set_pos]['skip_years']],
    'use_ay': [pos[set_pos]['use_ay']],
    'breakout_class_id': [None],
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
    'use_ay': [pos[set_pos]['use_ay']],
    'act_ppg': [None],
    'pct_off': [None], 
    'model': [None],
    'f1_score': [None],
    'baseline_f1': [None]
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
conn = sqlite3.connect(path + 'Data/' + db_name)
df = pd.read_sql_query('SELECT * FROM ' + set_pos + '_' + str(set_year), con=conn)

# append air yards for specified positions
if pos[set_pos]['use_ay']:
    ay = pd.read_sql_query('SELECT * FROM AirYards', con=sqlite3.connect(path + 'Data/Season_Stats.sqlite3'))
    df = pd.merge(df, ay, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
    
# apply the specified touches and game filters
df, df_train_results, df_test_results = touch_game_filter(df, pos, set_pos, set_year)

# calculate FP for a given set of scoring values
df = calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)

# add features based for a players performance relative to their experience
df = add_exp_metrics(df, set_pos, pos[set_pos]['use_ay'])
# -

# # Breakout Models

# +
#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
act_ppg = 0
pct_off = -0.2
gorl = 'less'
adp_ppg = 11

# get the train and prediction dataframes for FP per game
df_train, df_predict = get_train_predict(df, breakout_metric, pos, set_pos, set_year, pos[set_pos]['earliest_year'])

# get the outlier training dataset with outlier players labeled
df_train_orig, df_predict = get_outliers(df_train, df_predict, act_ppg=act_ppg, pct_off=pct_off, 
                                         year_min_int=1, gorl=gorl)



df_train_orig = df_train_orig[df_train_orig.avg_pick_pred > adp_ppg].drop('avg_pick_pred', axis=1).reset_index(drop=True)



# get the minimum number of training samples for the initial datasets
min_samples = int(0.5*df_train_orig[df_train_orig.year <= df_train_orig.year.min() + pos[set_pos]['skip_years']].shape[0])

# print the value-counts
print(df_train_orig.y_act.value_counts())

val_y = df_train_orig.loc[df_train_orig.year > df_train_orig.year.min() + pos[set_pos]['skip_years'], 'y_act']
baseline = round(matthews_corrcoef(val_y, np.repeat(1, len(val_y))), 3)
print('Baseline F1-score:', baseline)
print('Min Year:', df_train_orig.year.min())
print('Max Year:', df_train_orig.year.max())

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

@ignore_warnings(category=ConvergenceWarning)
def class_run_best_model(model, p, skip_years):

    use_smote = True if p['use_smote']==1 else False
    zero_weight = p['zero_weight']
    col_cut = p['collinear_cutoff']
    scale = True if p['scale']==1 else False
    pca = True if p['pca']==1 else False
    
    model_p = {}
    for k, v in p.items():
        if k not in ['use_smote', 'zero_weight', 'collinear_cutoff', 'scale', 'pca']:
            model_p[k] = v

    est = class_models[model]
    est.set_params(**model_p)
    est.class_weight = {0: zero_weight, 1: 1}

    result, val_pred, ty_pred, ty_proba, trained_est, cols = class_validation(est, df_train_orig, df_predict, col_cut,
                                                                              use_smote, skip_years, scale, pca)

    return result, val_pred, ty_pred, ty_proba, trained_est, cols


# -

def class_ensemble(summary, results, n, iters, agg_type):
    '''
    Function that accepts multiple model results and averages them together to create an ensemble prediction.
    It then calculates metrics to validate that the ensemble is more accurate than individual models.
    '''

    from sklearn.metrics import f1_score

    # initialize Series to store results
    ensemble = pd.Series(dtype='float')
    ty_ensemble = pd.Series(dtype='float')
    ty_proba = pd.Series(dtype='float')

    # for each row in the summary dataframe, append the result
    for i, row in summary.iterrows():
        if i == 0:
            ensemble = pd.concat([ensemble, results['val_pred'][row.Iteration]], axis=1)
            ty_ensemble = pd.concat([ty_ensemble, results['ty_pred'][row.Iteration]], axis=1)
            if row.Model != 'svr':
                ty_proba = pd.concat([ty_proba, results['ty_proba'][row.Iteration]], axis=1)
        else:
            ensemble = pd.concat([ensemble, results['val_pred'][row.Iteration]['pred' + str(row.Iteration)]], axis=1)
            ty_ensemble = pd.concat([ty_ensemble, results['ty_pred'][row.Iteration]['pred' + str(row.Iteration)]], axis=1)
            if row.Model != 'svr':
                ty_proba = pd.concat([ty_proba, results['ty_proba'][row.Iteration]['proba' + str(row.Iteration)]], axis=1)


    # get the median prediction from each of the models
    ensemble = ensemble.drop(0, axis=1)
    ty_ensemble = ty_ensemble.drop(0, axis=1)
    ty_proba = ty_proba.drop(0, axis=1)

    if agg_type=='mean':
        ensemble['pred'] = ensemble.iloc[:, 4:].mean(axis=1)
        ty_ensemble['pred'] = ty_ensemble.iloc[:, 3:].mean(axis=1)
        ty_proba['proba'] = ty_proba.iloc[:, 3:].mean(axis=1)
    elif agg_type=='median':
        ensemble['pred'] = ensemble.iloc[:, 4:].median(axis=1)
        ty_ensemble['pred'] = ty_ensemble.iloc[:, 3:].median(axis=1)
        ty_proba['proba'] = ty_proba.iloc[:, 3:].median(axis=1)

    ensemble.loc[ensemble.pred >= 0.5, 'pred'] = 1
    ensemble.loc[ensemble.pred < 0.5, 'pred'] = 0

    # store the predictions in the results dictionary
    results['val_pred'][int(n)] = ensemble
    results['ty_pred'][int(n)] = ty_ensemble
    results['ty_proba'][int(n)] = ty_proba
    
    # remove the +4 amount and iteration number from the n to properly
    # name the ensemble model with number of included models
    if agg_type=='median': j=str(int(n)-4-iters) 
    else: j=str(int(n) - iters)

    # create the output list to append error metrics
    ens_error = [n, 'Ensemble'+j]
    ens_error.append(f1_score(ensemble.y_act, ensemble.pred))
    
    # create dataframe of results
    ens_error = pd.DataFrame(ens_error).T
    ens_error.columns = summary.columns
    
    return results, ens_error

# +
best_models = min_rowid = 99
max_rowid = 105

skip_year = pos[set_pos]['skip_years']

print('Running ' + breakout_metric)
rows = tuple([c for c in range(min_rowid, max_rowid+1)])
best_models = pd.read_sql_query(f'''SELECT pkey, model
                                    FROM ClassParamTracking
                                    WHERE rowid in {rows}
                                          ''', param_conn)
print(best_models)

results = {'summary': {}, 'val_pred': {}, 'ty_pred': {}, 'ty_proba': {}, 'trained_model': {}, 'cols': {}}
num_models = best_models.index.max() + 1
for i in range(num_models-1):

    m = best_models.loc[i, 'model']
    pkey = best_models.loc[i, 'pkey']
    par = load_pickle(path + f'Data/Model_Params_Class/{pkey}.p')
    print(f'Running {m}')

    results[i] = {}
    results['summary'][i] = [m]

    result, val_pred, ty_pred, ty_proba, trained_est, cols = class_run_best_model(m, par, skip_year)

    # save out the predictions to the dictionary
    results['summary'][i].extend([result])
    results['trained_model'][i] = trained_est
    results['cols'][i] = cols

    val_pred = val_pred.rename(columns={'pred': 'pred' + str(i)})
    results['val_pred'][i] = val_pred

    ty_pred = ty_pred.rename(columns={'pred': 'pred' + str(i)})
    results['ty_pred'][i] = ty_pred
    
    if ty_proba is not None:
        ty_proba = ty_proba.rename(columns={'proba': 'proba' + str(i)})
    results['ty_proba'][i] = ty_proba

#============
# Create Ensembles and Provide Summary of Best Models
#============

# create the summary dataframe
summary = class_create_summary(results, keep_first=False)

iters = num_models
for n in range(2, num_models+2):

    # create the ensemble dataframe based on the top results for each model type
    to_ens = class_create_summary(results, keep_first=True).head(n)
    results, ens_result = class_ensemble(to_ens, results, str(n+iters), iters, 'mean')
    summary =  pd.concat([summary, ens_result], axis=0)

    # create the ensemble dataframe based on the top results for each model type
    to_ens = class_create_summary(results, keep_first=True).head(n)
    results, ens_result = class_ensemble(to_ens, results, str(n+iters+4), iters, 'median')
    summary =  pd.concat([summary, ens_result], axis=0)

summary = summary.sort_values(by='F1Score', ascending=False).reset_index(drop=True)
print(summary.head(20))

#----------
# Store Prediction Results
#----------

# get the best model and pull out the results
best_iter = int(summary[summary.Model.str.contains('Ensemble')].iloc[0].Iteration)

best_ty = results['ty_pred'][best_iter][['player', 'year', 'pred']]
best_ty = best_ty.rename(columns={'pred': 'pred_' + breakout_metric})

best_train = results['val_pred'][best_iter][['player', 'year', 'pred', 'y_act']]
best_train = best_train.rename(columns={'pred': 'pred_' + breakout_metric, 'y_act': 'act_' + breakout_metric})

best_ty_proba = results['ty_proba'][best_iter][['player', 'year', 'proba']]
best_ty_proba = best_ty_proba.rename(columns={'proba': 'proba_' + breakout_metric})

# apply the specified touches and game filters
_, df_train_results, df_test_results = touch_game_filter(df, pos, set_pos, set_year)

# merge the train results for the given metric with all other metric outputs
df_train_results = pd.merge(df_train_results, best_train, on=['player', 'year'])

# merge the test results for the given metric with all other metric outputs
df_test_results = pd.merge(df_test_results, best_ty, on=['player', 'year'])
df_test_results = pd.merge(df_test_results, best_ty_proba, on=['player', 'year'])

#----------
# Format for Next Step
#----------

# reorder the results of the output to have predicted before actual
col_order = ['player', 'year']
col_order.extend([c for c in df_train_results.columns if 'pred' in c])
col_order.extend([c for c in df_train_results.columns if 'act' in c])
df_train_results = df_train_results[col_order]

# +
# get the train and prediction dataframes for FP per game
compare_train, compare_test = get_train_predict(df, breakout_metric, pos, set_pos, 
                                                set_year, pos[set_pos]['earliest_year'])
compare, lr = get_adp_predictions(compare_train, year_min_int=1, pct_off=pct_off, act_ppg=act_ppg, gorl=gorl)
compare = compare.drop('label', axis=1)
compare_test['avg_pick_pred'] = lr.predict(compare_test.avg_pick.values.reshape(-1,1))

df_train_results = pd.merge(df_train_results, compare, on=['player', 'year'])
df_test_results = pd.merge(df_test_results, compare_test[['player', 'year', 'avg_pick_pred']], on=['player', 'year'])
# -

df_test_results[df_test_results.avg_pick_pred > 12].sort_values(by=['pred_fp_per_game', 'proba_fp_per_game'],
                                                                ascending=[False, False])

df_train_results.groupby('pred_fp_per_game').agg({'y_act': 'mean',
                                                  'pct_off': 'mean',
                                                  'avg_pick_pred': 'mean'})

# +
# Create Bokeh-Table with DataFrame:
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource

data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in df_train_results.columns],
    source=ColumnDataSource(df_train_results),
    height=500,
    width=500
)

# Create Scatterplot:
p_scatter = df_train_results.plot_bokeh.scatter(
   y="pct_off",
    x='y_act',
    category="pred_fp_per_game",
    title="Iris DataSet Visualization",
    show_figure=False
)

# Combine Table and Scatterplot via grid layout:
pandas_bokeh.plot_grid([[data_table, p_scatter]], plot_width=500, plot_height=500)
# -

# # Regression Models

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


@ignore_warnings(category=ConvergenceWarning)
def run_best_model(model, p, skip_years):

    corr_cut = p['corr_cutoff']
    col_cut= p['collinear_cutoff']
    scale = p['scale']
    try:
        pca = p['pca']
    except:
        pca = False
    model_p = {}
    for k, v in p.items():
        if k not in ['corr_cutoff', 'collinear_cutoff', 'scale', 'pca']:
            model_p[k] = v

    est = models[model]
    est.set_params(**model_p)

    result, val_pred, ty_pred, trained_est, cols = validation(est, df_train_orig, df_predict, corr_cut, 
                                                              col_cut, skip_years=skip_years, scale=scale, pca=pca)

    return result, val_pred, ty_pred, trained_est, cols


# +
for metric in pos[set_pos]['metrics']:
    print('Running ' + metric)
    ay = 1 if pos[set_pos]['use_ay'] else 0

    best_models = pd.read_sql_query(f'''SELECT pkey, model
                                        FROM RegParamTracking
                                        WHERE pos='{set_pos}'
                                              AND metric='{metric}'
                                           --   AND req_touch='{pos[set_pos]['req_touch']}'
                                           --   AND req_games='{pos[set_pos]['req_games']}'
                                           --   AND earliest_year='{pos[set_pos]['earliest_year']}'
                                           --   AND skip_years='{pos[set_pos]['skip_years']}'
                                           --   AND use_ay={ay}
                                           --   AND year={set_year}
                                              ''', param_conn)
    
    df_train, df_predict = get_train_predict(df, metric, pos, set_pos, set_year, pos[set_pos]['earliest_year'])
    for c in df_train.columns:
        if len(df_train[df_train[c]==np.inf]) > 0:
            df_train = df_train.drop(c, axis=1)
    df_train_orig = df_train.copy()

    results = {'summary': {}, 'val_pred': {}, 'ty_pred': {}, 'trained_model': {}, 'cols': {}}
    num_models = best_models.index.max() + 1
    for i in range(num_models):

        m = best_models.loc[i, 'model']
        pkey = best_models.loc[i, 'pkey']
        par = load_pickle(path + f'Data/Model_Params/{pkey}.p')
        print(f'Running {m}')

        results[i] = {}
        results['summary'][i] = [m]

        result, val_pred, ty_pred, trained_est, cols = run_best_model(m, par, pos[set_pos]['skip_years'])

        # save out the predictions to the dictionary
        results['summary'][i].extend(result)
        results['trained_model'][i] = trained_est
        results['cols'][i] = cols

        val_pred = val_pred.rename(columns={'pred': 'pred' + str(i)})
        results['val_pred'][i] = val_pred

        ty_pred = ty_pred.rename(columns={'pred': 'pred' + str(i)})
        results['ty_pred'][i] = ty_pred

    #============
    # Create Ensembles and Provide Summary of Best Models
    #============

    # Specify metric for sorting by
    sort_metric = 'PredRMSE'
    if sort_metric=='PredRMSE': ascend=True 
    else: ascend=False

    # create the summary dataframe
    summary = create_summary(results, sort_metric, keep_first=False)

    iters = num_models
    for n in range(2, num_models+2):

        # create the ensemble dataframe based on the top results for each model type
        to_ens = create_summary(results, sort_metric, keep_first=True).head(n)
        results, ens_result = test_ensemble(to_ens, results, str(n+iters), iters, 'mean')
        summary =  pd.concat([summary, ens_result], axis=0)

        # create the ensemble dataframe based on the top results for each model type
        to_ens = create_summary(results, sort_metric, keep_first=True).head(n)
        results, ens_result = test_ensemble(to_ens, results, str(n+iters+4), iters, 'median')
        summary =  pd.concat([summary, ens_result], axis=0)

    summary = summary.sort_values(by=sort_metric, ascending=ascend).reset_index(drop=True)
    print(summary.head(20))

    # save the output results to the output dataframe for parameter tracking
    output_tmp = pd.DataFrame(summary.loc[0, ['PredR2', 'AvgPickR2', 'PredRMSE']]).T
    output_tmp.columns = [c+metric for c in output_tmp.columns]
    output = pd.concat([output, output_tmp], axis=1)

    #----------
    # Store Prediction Results
    #----------

    # get the best model and pull out the results
    best_ty = results['ty_pred'][int(summary[summary.Model.str.contains('Ensemble')].iloc[0].Iteration)][['player', 'year', 'pred']]
    best_ty = best_ty.rename(columns={'pred': 'pred_' + metric})
    best_train = results['val_pred'][int(summary[summary.Model.str.contains('Ensemble')].iloc[0].Iteration)][['player', 'year', 'pred', 'y_act']]
    best_train = best_train.rename(columns={'pred': 'pred_' + metric, 'y_act': 'act_' + metric})

    # merge the train results for the given metric with all other metric outputs
    df_train_results = pd.merge(df_train_results, best_train, on=['player', 'year'])

    # merge the test results for the given metric with all other metric outputs
    df_test_results = pd.merge(df_test_results, best_ty, on=['player', 'year'])
    
#----------
# Format for Next Step
#----------

# reorder the results of the output to have predicted before actual
col_order = ['player', 'year']
col_order.extend([c for c in df_train_results.columns if 'pred' in c])
col_order.extend([c for c in df_train_results.columns if 'act' in c])
df_train_results = df_train_results[col_order]

# +
#--------
# Calculate Fantasy Points for Given Scoring System
#-------- 

# extract points list and get the idx of point attributes based on length of list
pts_list = pts_dict[set_pos]
c_idx = len(pts_list) + 2

train_plot = df_train_results.copy()
test_plot = df_test_results.copy()

# multiply stat categories by corresponding point values
train_plot.iloc[:, 2:c_idx] = train_plot.iloc[:, 2:c_idx] * pts_list
test_plot.iloc[:, 2:c_idx] = test_plot.iloc[:, 2:c_idx] * pts_list

# add a total predicted points stat category
train_plot.loc[:, 'pred'] = train_plot.iloc[:, 2:c_idx].sum(axis=1)
test_plot.loc[:, 'pred'] = test_plot.iloc[:, 2:c_idx].sum(axis=1)

#==========
# Plot Predictions for Each Player
#==========

# set length of plot based on number of results
plot_length = int(test_plot.shape[0] / 3.5)

# plot results from highest predicted FP to lowest predicted FP
test_plot.sort_values('pred').plot.barh(x='player', y='pred', figsize=(5, plot_length))
# -

# # Compare Fantasy Pros

# +
#==============
# Create Training and Prediction Dataframes
#==============
fantasy_points = []

# for each metric, load in the dataset with y-target
for m in pos[set_pos]['metrics']:
    df_train_full, df_predict_full = features_target(df,
                                                     pos[set_pos]['earliest_year'], set_year-1,
                                                     pos[set_pos]['med_features'],
                                                     pos[set_pos]['sum_features'],
                                                     pos[set_pos]['max_features'],
                                                     pos[set_pos]['age_features'],
                                                     target_feature=m)
    fantasy_points.append(list(df_train_full.y_act.values))
    
# convert the metrics to fantasy total fantasy points 
fantasy_pts = pd.DataFrame(fantasy_points).T.reset_index(drop=True)
fantasy_pts = (fantasy_pts * pts_dict[set_pos]).sum(axis=1)
fantasy_pts.name = 'fantasy_pts'

# +
stats_fp = []
stats_all = []
min_year = int(max(df_train_results.year.min(), 2012))
    
for metric in pos[set_pos]['metrics']:

    df_train_full, df_predict_full = features_target(df,
                                                     pos[set_pos]['earliest_year'], set_year-1,
                                                     pos[set_pos]['med_features'],
                                                     pos[set_pos]['sum_features'],
                                                     pos[set_pos]['max_features'],
                                                     pos[set_pos]['age_features'],
                                                     target_feature=metric)

    df_train_full = pd.concat([df_train_full, fantasy_pts], axis=1)

    fp = pd.read_sql_query('SELECT * FROM FantasyPros', con=sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Season_Stats.sqlite3'))
    fp.year = fp.year-1
    df_train_full = pd.merge(df_train_full, fp, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
    df_train_full = df_train_full[df_train_full.fp > 5]


    from sklearn.linear_model import Lasso
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import Ridge


    df_train_full = df_train_full.dropna()
    lass = Lasso(alpha=250)
#     y = 'fantasy_pts'
#     y_other = 'y_act'
    
    y_other = 'fantasy_pts'
    y = 'y_act'


    stat_all = []
    stat_fp = []
    results_all = []
    results_fp = []
    for i in range(min_year, (set_year-1)):

        print(i)
        X_train = df_train_full.loc[df_train_full.year < i].drop([y_other, y], axis=1)
        y_train = df_train_full.loc[df_train_full.year < i, y]

        X_fp = X_train[['year', 'rank', 'adp', 'best', 'worst', 'avg', 'std_dev']]
        X_all = X_train.drop(['player', 'team', 'pos','rank', 'adp', 'best', 'worst', 'avg', 'std_dev' ], axis=1)

        X_predict = df_train_full.loc[df_train_full.year== i].drop([y_other, y], axis=1)
        X_pred_fp = X_predict[['year', 'rank', 'adp', 'best', 'worst', 'avg', 'std_dev']]
        X_pred_all = X_predict.drop(['player', 'team', 'pos', 'rank', 'adp', 'best', 'worst', 'avg', 'std_dev'], axis=1)

        y_pred = df_train_full.loc[df_train_full.year == i, y]

        lass.fit(X_fp, y_train)
        fp_pred = lass.predict(X_pred_fp)
        stat_fp.extend(list(fp_pred))
        print('FP error:', round(np.mean(np.sqrt(abs(mean_squared_error(fp_pred, y_pred)))), 3))
        results_fp.append(round(np.mean(np.sqrt(abs(mean_squared_error(fp_pred, y_pred)))), 3))

        lass.fit(X_all.replace([np.inf, -np.inf], np.nan).fillna(0), y_train)
        all_pred = lass.predict(X_pred_all.replace([np.inf, -np.inf], np.nan).fillna(0))
        stat_all.extend(list(all_pred))
        print('All error:', round(np.mean(np.sqrt(abs(mean_squared_error(all_pred, y_pred)))), 3))
        results_all.append(round(np.mean(np.sqrt(abs(mean_squared_error(all_pred, y_pred)))), 3))
    
    stats_fp.append(stat_fp)
    stats_all.append(stat_all)
    
    if y == 'fantasy_pts':
        print('--------------')
        print('Fantasy Pros straight FP Error:', round(np.mean(results_fp), 3))
        print('All straight FP Error:', round(np.mean(results_all), 3))
        break

# +
#----------------
# Convert Fantasy Pros and Lasso Stat Results to Points
#----------------

df_all = pd.DataFrame(stats_all).T
df_fp = pd.DataFrame(stats_fp).T
    
df_all = (df_all * pts_dict[set_pos]).sum(axis=1)
df_fp = (df_fp * pts_dict[set_pos]).sum(axis=1)

y_test = df_train_full.loc[(df_train_full.year <= i) & (df_train_full.year >= min_year), y_other]

lasso_error = round(np.mean(np.sqrt(abs(mean_squared_error(df_all, y_test)))), 2)
fantasy_pros_error = round(np.mean(np.sqrt(abs(mean_squared_error(df_fp, y_test)))), 2)
print(f'Lasso error: {lasso_error}' )
print(f'FantasyPros error: {fantasy_pros_error}')

#----------------
# Merge Fantasy Pros Data with Full Model Results to get Matching Player Sets
#----------------

full_models = pd.merge(
              df_train_full.loc[(df_train_full.year <= i) & (df_train_full.year >= min_year), ['player', 'year']],
              df_train_results, on=['player', 'year']).reset_index(drop=True)

y_test = df_train_full.loc[(df_train_full.year <= i) & (df_train_full.year >= min_year), y_other].reset_index(drop=True)

if set_pos == 'RB':
    pts = pts_dict[set_pos]
    full_models['fantasy_pts'] = (full_models.iloc[:,2:(len(pts)+2)]* pts).sum(axis=1)

elif set_pos == 'WR' or set_pos =='TE':
    pts = pts_dict[set_pos]
    full_models['fantasy_pts'] = (full_models.iloc[:,2:(len(pts)+2)]*pts).sum(axis=1)
    
elif set_pos == 'QB':
    pts = pts_dict[set_pos]
    full_models['fantasy_pts'] = (full_models.iloc[:,2:(len(pts)+2)]* pts).sum(axis=1)

ensemble_error = round(np.mean(np.sqrt(abs(mean_squared_error(full_models.fantasy_pts, y_test)))), 2)
print(f'Ensemble Error: {ensemble_error}')

# save the results to the output dataframe
output = pd.concat([output, pd.DataFrame({'FantasyProsRMSE': [fantasy_pros_error], 'EnsembleRMSE': [ensemble_error]})], axis=1)
# -


