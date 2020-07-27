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
from sklearn.model_selection import cross_val_score
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
path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'

# specify database name with model data
db_name = 'Model_Inputs.sqlite3'
conn = sqlite3.connect(path + 'Data/Databases/' + db_name)

# set path to param database
param_conn = sqlite3.connect(path + 'Data/Databases/ParamTracking.sqlite3')

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

def save_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


# -

# # Classification Ensemble

# ## Specifying Parameters

# +
#==========
# Extract parameters from database results
#==========

def pull_class_params(min_rowid, max_rowid):
    
    rows = tuple([c for c in range(min_rowid, max_rowid+1)])

    # pull in the results for the given primary keys and extract best model coordinates
    res = pd.read_sql_query(f'''SELECT * FROM ClassParamTracking WHERE rowid in {rows}''', param_conn)
    best_models = res[['pkey', 'model']]

    # convert the dataframe to to dictionary to extract data
    res = res.drop(['pkey', 'model', 'score'], axis=1).drop_duplicates()
    res = res.rename(columns = {'year': 'set_year', 'pos': 'set_pos'})
    res = res.T.to_dict()[0]
    
    # set variables for all items in results
    for k, v in res.items():
        globals()[k] = v

    # set all of the data preparation filters 
    pos[set_pos]['req_touch'] = res['req_touch']
    pos[set_pos]['req_games'] = res['req_games']
    pos[set_pos]['earliest_year'] = res['earliest_year'] - 2
    pos[set_pos]['skip_years'] = res['skip_years']
    pos[set_pos]['use_ay'] = True if res['use_ay']==1 else False
    pos[set_pos]['features'] = res['features']
    
    return best_models


# -

# ## Pulling in Data

def pull_class_data(breakout_metric, adp_ppg_low, adp_ppg_high, pct_off, act_ppg, set_pos, set_year, pts_dict=pts_dict, pos=pos):

    #==========
    # Pull and clean compiled data
    #==========

    # connect to database and pull in positional data
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

    #==============
    # Create Break-out Probability Features
    #==============

    # get the train and prediction dataframes for FP per game
    df_train_orig, df_predict = get_train_predict(df, breakout_metric, pos, set_pos, set_year-1, pos[set_pos]['earliest_year'], pos[set_pos]['features'])

    # get the adp predictions and merge back to the training dataframe
    df_train_adp, lr = get_adp_predictions(df_train_orig, 1)
    df_train_orig = pd.merge(df_train_orig, df_train_adp, on=['player', 'year'])
    df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))

    # create the label and filter based on inputs
    df_train_orig['label'] = 0
    df_train_orig.loc[(eval(f'df_train_orig.pct_off{pct_off}')) & (eval(f'df_train_orig.y_act{act_ppg}')), 'label'] = 1
    df_train_orig = df_train_orig[(eval(f'df_train_orig.avg_pick_pred{adp_ppg_low}')) & (eval(f'df_train_orig.avg_pick_pred{adp_ppg_high}'))].reset_index(drop=True)
    df_predict = df_predict[(eval(f'df_predict.avg_pick_pred{adp_ppg_low}')) & (eval(f'df_predict.avg_pick_pred{adp_ppg_high}'))].reset_index(drop=True)

    # add in extra columns to the results dataframes
    df_train_results = pd.merge(df_train_results, df_train_orig[['player', 'year', 'y_act', 'pct_off', 'avg_pick_pred']], on=['player', 'year'])
    df_test_results = pd.merge(df_test_results, df_predict[['player', 'year', 'avg_pick_pred']], on=['player', 'year'])
    df_train_orig = df_train_orig.drop(['y_act', 'pct_off'], axis=1).rename(columns={'label': 'y_act'})

    # get the minimum number of training samples for the initial datasets
    min_samples = int(0.5*df_train_orig[df_train_orig.year <= df_train_orig.year.min() + pos[set_pos]['skip_years']].shape[0])

    # print the value-counts
    print(df_train_orig.y_act.value_counts())
    print('Min Year:', df_train_orig.year.min())
    print('Max Year:', df_train_orig.year.max())
    print('Compiled Dataset Equals Stored Dataset?: ', df_train_orig.equals(load_pickle(f'{path}Data/Model_Datasets_Class/{str(min_rowid)}.p')))
    
    return df_train_orig, df_predict, df_train_results, df_test_results

# ## Ensemble Model Helper Functions


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
    n_components = p['n_components'] 
    
    model_p = {}
    for k, v in p.items():
        if k not in ['use_smote', 'zero_weight', 'collinear_cutoff', 'scale', 'pca', 'n_components']:
            model_p[k] = v
            
    est = class_models[model]
    est.set_params(**model_p)
    est.class_weight = {0: zero_weight, 1: 1}

    result, val_pred, ty_pred, ty_proba, trained_est, cols = class_validation(est, df_train_orig, df_predict, col_cut,
                                                                              use_smote, skip_years, scale, pca, n_components)

    return result, val_pred, ty_pred, ty_proba, trained_est, cols


# -

# ## Running Ensembles

def run_class_ensemble(best_models, df_train_results, df_test_results, pos=pos):

    skip_year = pos[set_pos]['skip_years']

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
    best_ty = best_ty.rename(columns={'pred': 'class_' + breakout_metric})

    best_train = results['val_pred'][best_iter][['player', 'year', 'pred', 'y_act']]
    best_train = best_train.rename(columns={'pred': 'class_' + breakout_metric, 'y_act': 'class_act_' + breakout_metric})

    best_ty_proba = results['ty_proba'][best_iter][['player', 'year', 'proba']]
    best_ty_proba = best_ty_proba.rename(columns={'proba': 'proba_' + breakout_metric})

    # merge the train results for the given metric with all other metric outputs
    df_train_results = pd.merge(df_train_results, best_train, on=['player', 'year'])

    # merge the test results for the given metric with all other metric outputs
    df_test_results = pd.merge(df_test_results, best_ty, on=['player', 'year'])
    df_test_results = pd.merge(df_test_results, best_ty_proba, on=['player', 'year'])
    
    return df_train_results, df_test_results, results


# # Classification Run + Results

# +
# specify the minimum and maximum rows from the database

df_train_results = pd.DataFrame()
df_test_results = pd.DataFrame()

for i in [(216, 222)]:#, (113, 140), (141, 168)]:
# for i in [(1, 28), (29, 56), (57, 84)]:
    
    # set all the variables
    min_rowid=i[0]
    max_rowid=i[1]
    best_models = pull_class_params(min_rowid, max_rowid)
    
    # pull in the classification data
    df_train_orig, df_predict, df_train_results_subset, df_test_results_subset = pull_class_data(breakout_metric, adp_ppg_low, adp_ppg_high, pct_off, act_ppg, set_pos, set_year)
    
    # run the ensembling
    df_train_results_subset, df_test_results_subset, _ = run_class_ensemble(best_models, df_train_results_subset, df_test_results_subset)
    
    # combine all data cuts into single dataframe
    df_train_results = pd.concat([df_train_results, df_train_results_subset], axis=0)
    df_test_results = pd.concat([df_test_results, df_test_results_subset], axis=0)
    
# print out Matthew Score of all data combined
print('Low+High Matthews Score %0.3f' % matthews_corrcoef(df_train_results.class_act_fp_per_game, df_train_results.class_fp_per_game))
# -

df_train_results.groupby('class_fp_per_game').agg({'y_act': 'median',
                                                       'pct_off': 'median',
                                                       'avg_pick_pred': 'mean'})

df_test_results.sort_values(by='proba_fp_per_game', ascending=False).iloc[:60]

# +
# Create Bokeh-Table with DataFrame:
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, Span

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
    category="class_fp_per_game",
    title="Iris DataSet Visualization",
    show_figure=False
)

hline = Span(location=eval(pct_off.replace('>=', '').replace('<=', '').replace('<', '').replace('>', '')), dimension='width', line_color='green', line_width=3)
vline = Span(location=eval(act_ppg.replace('>=', '').replace('<=', '').replace('<', '').replace('>', '')), dimension='height', line_color='blue', line_width=3)
p_scatter.renderers.extend([hline, vline])


# Combine Table and Scatterplot via grid layout:
pandas_bokeh.plot_grid([[data_table, p_scatter]], plot_width=500, plot_height=500)

# + jupyter={"source_hidden": true}
ind = 0
try:
    imp = pd.DataFrame(results['trained_model'][ind].coef_, columns=results['cols'][ind]).T
    imp = imp[abs(imp) > 0.2*imp.max()].sort_values(by=0).reset_index()
except:
    imp = pd.DataFrame([results['trained_model'][ind].feature_importances_], columns=results['cols'][ind]).T
    imp = imp[abs(imp) > 0.1*imp.max()].sort_values(by=0).reset_index()
imp.plot_bokeh(x='index', y=0, kind='barh', figsize=(600, 1000))


# -

# # Regression Models

# +
#==========
# Extract parameters from database results
#==========

def pull_reg_params(min_rowid, max_rowid):

    rows = tuple([c for c in range(min_rowid, max_rowid+1)])

    # pull in the results for the given primary keys and extract best model coordinates
    res = pd.read_sql_query(f'''SELECT * FROM RegParamTracking WHERE rowid in {rows}''', param_conn)
    best_models = res[['pkey', 'metric', 'model']]
    
    # convert the dataframe to to dictionary to extract data
    res = res.drop(['pkey', 'metric', 'model', 'rmse_validation', 'rmse_validation_adp', 'r2_validation', 'r2_validation_adp'], axis=1).drop_duplicates()
    res = res.rename(columns = {'year': 'set_year', 'pos': 'set_pos'})
    res = res.T.to_dict()[0]

    # set variables for all items in results
    for k, v in res.items():
        globals()[k] = v
    
    # set all of the data preparation filters 
    pos[set_pos]['req_touch'] = res['req_touch']
    pos[set_pos]['req_games'] = res['req_games']
    pos[set_pos]['earliest_year'] = res['earliest_year'] - 1
    pos[set_pos]['skip_years'] = res['skip_years']
    pos[set_pos]['use_ay'] = True if res['use_ay']==1 else False
    pos[set_pos]['features'] = res['features']

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
    df, _, _ = touch_game_filter(df, pos, set_pos, set_year)

    # calculate FP for a given set of scoring values
    df = calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)

    # add features based for a players performance relative to their experience
    df = add_exp_metrics(df, set_pos, pos[set_pos]['use_ay'])
    
    return best_models, df


# +
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
def run_best_model(model, p, df_train_orig, df_predict, skip_years):

    corr_cut = p['corr_cutoff']
    col_cut= p['collinear_cutoff']
    scale = p['scale']
    pca = True if p['pca']==1 else False
    n_components = p['n_components'] 

    model_p = {}
    for k, v in p.items():
        if k not in ['corr_cutoff', 'collinear_cutoff', 'scale', 'pca', 'n_components']:
            model_p[k] = v

    est = models[model]
    est.set_params(**model_p)
    
    result, val_pred, ty_pred, trained_est, cols = validation(est, df_train_orig, df_predict, corr_cut, col_cut, skip_years=skip_years, 
                                                              scale=scale, pca=pca, n_components=n_components)
    
    return result, val_pred, ty_pred, trained_est, cols


# -

def run_reg_ensemble(df, best_models, df_train_results, df_test_results, adp_ppg_low, adp_ppg_high, pos=pos):

    output = pd.DataFrame()
    metrics = list(best_models.metric.unique())
    for metric in metrics:            
        
        # extract the models for the current metric
        metric_models = best_models[best_models.metric==metric].copy().reset_index(drop=True)
        
        print('Running ' + metric)
        ay = 1 if pos[set_pos]['use_ay'] else 0
        
        if metric == 'avg_pick_pct_off':
            df_train, df_predict = get_train_predict(df, 'fp_per_game', pos, set_pos, set_year, pos[set_pos]['earliest_year']-1, pos[set_pos]['features'])
        else:
            df_train, df_predict = get_train_predict(df, metric, pos, set_pos, set_year, pos[set_pos]['earliest_year']-1, pos[set_pos]['features'])
            
        # get the adp predictions and merge back to the training dataframe
        df_train_adp, lr = get_adp_predictions(df_train, 1)
        df_train = pd.merge(df_train, df_train_adp, on=['player', 'year'])
        if metric == 'avg_pick_pct_off':
            df_train['y_act'] = (df_train.y_act - df_train.avg_pick_pred) / df_train.y_act
        
        df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))

        # filter based on adp predict
        df_train = df_train[(eval(f'df_train.avg_pick_pred{adp_ppg_low}')) & ((eval(f'df_train.avg_pick_pred{adp_ppg_high}')))].reset_index(drop=True).drop(['avg_pick_pred', 'pct_off'], axis=1)
        df_predict = df_predict[(eval(f'df_predict.avg_pick_pred{adp_ppg_low}')) & ((eval(f'df_predict.avg_pick_pred{adp_ppg_high}')))].reset_index(drop=True).drop('avg_pick_pred', axis=1)
        
        print(f'Number of rows and features: {df_train.shape}')
        print('Dataset equals training?: ' + str(df_train.equals(load_pickle(f'{path}Data/Model_Datasets/{str(min_rowid)}.p'))))
        df_train_orig = df_train.copy()
        
        results = {'summary': {}, 'val_pred': {}, 'ty_pred': {}, 'trained_model': {}, 'cols': {}}
        num_models = metric_models.index.max() + 1
        for i in range(num_models):

            m = metric_models.loc[i, 'model']
            pkey = metric_models.loc[i, 'pkey']
            par = load_pickle(path + f'Data/Model_Params/{pkey}.p')
            print(f'Running {m}')

            results[i] = {}
            results['summary'][i] = [m]
            
            result, val_pred, ty_pred, trained_est, cols = run_best_model(m, par, df_train_orig, df_predict,  pos[set_pos]['skip_years'])

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
    col_order.extend([c for c in df_train_results.columns if 'pred' in c or 'class' in c])
    col_order.extend([c for c in df_train_results.columns if 'act' in c])
    df_train_results = df_train_results[col_order]
    
    return df_train_results, df_test_results, results

# +
best_models, df = pull_reg_params(1, 2)

output = pd.DataFrame()
metrics = list(best_models.metric.unique())
for metric in metrics:            

    # extract the models for the current metric
    metric_models = best_models[best_models.metric==metric].copy().reset_index(drop=True)

    print('Running ' + metric)
    ay = 1 if pos[set_pos]['use_ay'] else 0

    if metric == 'avg_pick_pct_off':
        df_train, df_predict = get_train_predict(df, 'fp_per_game', pos, set_pos, set_year, pos[set_pos]['earliest_year']-1, pos[set_pos]['features'])
    else:
        df_train, df_predict = get_train_predict(df, metric, pos, set_pos, set_year, pos[set_pos]['earliest_year']-1, pos[set_pos]['features'])

    # get the adp predictions and merge back to the training dataframe
    df_train_adp, lr = get_adp_predictions(df_train, 1)
    df_train = pd.merge(df_train, df_train_adp, on=['player', 'year'])
    if metric == 'avg_pick_pct_off':
        df_train['y_act'] = (df_train.y_act - df_train.avg_pick_pred) / df_train.y_act

    df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))

    # filter based on adp predict
    df_train = df_train[(eval(f'df_train.avg_pick_pred{adp_ppg_low}')) & ((eval(f'df_train.avg_pick_pred{adp_ppg_high}')))].reset_index(drop=True).drop(['avg_pick_pred', 'pct_off'], axis=1)
    df_predict = df_predict[(eval(f'df_predict.avg_pick_pred{adp_ppg_low}')) & ((eval(f'df_predict.avg_pick_pred{adp_ppg_high}')))].reset_index(drop=True).drop('avg_pick_pred', axis=1)

    print(f'Number of rows and features: {df_train.shape}')
    print('Dataset equals training?: ' + str(df_train.equals(load_pickle(f'{path}Data/Model_Datasets/{str(min_rowid)}.p'))))
    df_train_orig = df_train.copy()



# +
# take a random sample
from sklearn.model_selection import train_test_split

cnter=1

years = df_train_orig.year.unique()
years = years[years > np.min(years) + skip_years]

df_train_val = df_train[df_train.year.isin(years)].copy().reset_index(drop=True)
df_train_only = df_train[df_train.year < np.min(years)].copy().reset_index(drop=True)
df_train_val, df_val, _, _ = train_test_split(df_train_val, df_train_val.y_act, test_size=0.25, random_state=globals()['cnter']*3 + globals()['cnter']*17, shuffle=True)

df_train = pd.concat([df_train_only, df_train_val], axis=0)
# -

df_val

# +
df_train_results_reg = pd.DataFrame()
df_test_results_reg = pd.DataFrame()

for i in [(109, 144), (145, 180), (181, 216)]:
# for i in [(1, 36), (37, 72), (73, 108)]:
    
    # specify the minimum and maximum rows from the database
    min_rowid = i[0]
    max_rowid = i[1]

    # set all the variables
    best_models, df = pull_reg_params(min_rowid, max_rowid)
    df_train_results_subset, df_test_results_subset, results = run_reg_ensemble(df, best_models, df_train_results, df_test_results, adp_ppg_low, adp_ppg_high)
    
    df_train_results_reg = pd.concat([df_train_results_reg, df_train_results_subset], axis=0)
    df_test_results_reg = pd.concat([df_test_results_reg, df_test_results_subset], axis=0)
    
df_train_results_reg = df_train_results_reg.reset_index(drop=True)
df_test_results_reg = df_test_results_reg.reset_index(drop=True)

# append actual points for the past season "Test"
test_fp = pd.read_sql_query(f"SELECT * FROM {set_pos}_Stats WHERE year={set_year}", sqlite3.connect(path + 'Data/Databases/Season_Stats.sqlite3'))
test_fp = test_fp[['player', 'rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game']]
test_fp['y_act'] = (test_fp[pos[set_pos]['metrics']] * pts_dict['RB']).sum(axis=1)
test_fp = test_fp[['player', 'y_act']]
df_test_results_reg = pd.merge(df_test_results_reg, test_fp, on='player')

# + jupyter={"source_hidden": true}
df_train_results_off = pd.DataFrame()
df_test_results_off = pd.DataFrame()

for i in [(248, 281), (282, 295)]:

    # specify the minimum and maximum rows from the database
    min_rowid = i[0]
    max_rowid = i[1]

    # set all the variables
    best_models, df = pull_reg_params(min_rowid, max_rowid)
    df_train_results_subset, df_test_results_subset, results = run_reg_ensemble(df, best_models, df_train_results_reg, df_test_results_reg, adp_ppg_low, adp_ppg_high)
    
    df_train_results_off = pd.concat([df_train_results_off, df_train_results_subset], axis=0)
    df_test_results_off = pd.concat([df_test_results_off, df_test_results_subset], axis=0)
    
df_train_results_off = df_train_results_off.reset_index(drop=True)
df_test_results_off = df_test_results_off.reset_index(drop=True)
# -

df_train_results = df_train_results_reg.copy()
df_test_results = df_test_results_reg.copy()
df_test_results = df_test_results[~df_test_results.player.isin(['Todd Gurley', 'Damien Williams'])].reset_index(drop=True)
df_test_results = df_test_results.drop('class_fp_per_game', axis=1)
df_test_results = df_test_results.rename(columns={'proba_fp_per_game': 'class_fp_per_game'})

# + jupyter={"source_hidden": true}
#--------
# Calculate Fantasy Points for Given Scoring System
#-------- 

if len(list(best_models.metric.unique())) > 1:

    # extract points list and get the idx of point attributes based on length of list
    pts_list = pts_dict[set_pos]
    pred_metrics = ['pred_' + m for m in pos[set_pos]['metrics']]
    df_train_results['pred_fp_per_game'] = (df_train_results[pred_metrics] * pts_list).sum(axis=1)
    df_test_results['pred_fp_per_game'] = (df_test_results[pred_metrics] * pts_list).sum(axis=1)

# +
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X_train = df_train_results[['pred_fp_per_game', 'class_fp_per_game']]
X_test = df_test_results[['pred_fp_per_game', 'class_fp_per_game']]
y = df_train_results.y_act

scale = StandardScaler()
X_train = pd.DataFrame(scale.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scale.transform(X_test), columns=X_test.columns)

lr = LinearRegression()
lr.fit(X_train, y)
test_pred_plus_class = lr.predict(X_test)

# print out coefficients of fit
print('Coefficients:', lr.coef_)

pred_plus_class_err = np.mean(np.sqrt(-1*(cross_val_score(lr, X_train, y, cv=10, scoring='neg_mean_squared_error'))))
pred_only_err = np.sqrt(mean_squared_error(df_train_results.y_act, df_train_results.pred_fp_per_game))
adp_pred_err = np.sqrt(mean_squared_error(df_train_results.y_act, df_train_results.avg_pick_pred))

test_adp_err = np.sqrt(mean_squared_error(df_test_results.y_act, df_test_results.avg_pick_pred))
test_pred_only_err = np.sqrt(mean_squared_error(df_test_results.y_act, df_test_results.pred_fp_per_game))
test_pred_class_err = np.sqrt(mean_squared_error(df_test_results.y_act, test_pred_plus_class))

test_y = df_test_results.y_act
test_y.name='y_act'
test_adp = df_test_results.avg_pick_pred
X_test = pd.concat([X_test, test_y, test_adp], axis=1)

print('Predicted ADP Error: %0.3f' % adp_pred_err)
print('Predicted Only Error: %0.3f' % pred_only_err )
print('Predicted Plus Class Error: %0.3f' % pred_plus_class_err)
print('')
print('Predicted ADP Error: %0.3f' % test_adp_err)
print('Predicted Only Error: %0.3f' % test_pred_only_err )
print('Predicted Plus Class Error: %0.3f' % test_pred_class_err)

# +
import pymc3 as pm

data = pd.concat([X_train, y], axis=1)
formula = 'y_act ~ pred_fp_per_game + class_fp_per_game'

# Context for the model
with pm.Model() as normal_model:
    
    # The prior for the data likelihood is a Normal Distribution
    family = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data = data, family = family)
    
    # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 500)

# +
from IPython.core.pylabtools import figsize
import seaborn as sns

# Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation):
    
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation)
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)
    
    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()

    # Actual Value
    actual = test_observation['y_act']
    adp = test_observation['avg_pick_pred']
    
    # Add in intercept term
    test_observation['Intercept'] = 1
    test_observation = test_observation.drop(['y_act', 'avg_pick_pred'])
    
    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)
    
    # Estimates of grade
    estimates = np.random.normal(loc = mean_loc, scale = sd_value,
                                 size = 1000)

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                kde_kws = {'linewidth' : 4},
                label = 'Estimated Dist.')
    
    # Plot the actual points
    plt.vlines(x = actual, ymin = 0, ymax = .1, 
               linestyles = '-', colors = 'black',
               label = 'True Points',
              linewidth = 2.5)
    
    # Plot the mean estimate
    plt.vlines(x = mean_loc, ymin = 0, ymax = 0.1, 
               linestyles = '--', colors = 'red',
               label = 'Pred Estimate',
               linewidth = 2.5)
    
    # Plot the mean estimate
    plt.vlines(x = adp, ymin = 0, ymax = 0.1, 
               linestyles = '--', colors = 'orange',
               label = 'ADP Estimate',
               linewidth = 2.5)
    
    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');
    
    # Prediction information
    print('True Grade = %d' % actual)
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                       np.percentile(estimates, 95)))
    
    plt.show()


# -

for i in range(40):
    print(df_test_results.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True).loc[i, 'player'])
    test_model(normal_trace, X_test.sort_values(by='pred_fp_per_game', ascending=False).iloc[i])

# + jupyter={"source_hidden": true}
#==========
# Plot Predictions for Each Player
#==========

# set length of plot based on number of results
plot_length = int(20*df_train_results.shape[0])

# plot results from highest predicted FP to lowest predicted FP
df_train_results.sort_values('avg_pick_pred').plot_bokeh(x='player', y=['pred_fp_per_game_y', 'avg_pick_pred', 'y_act'], kind='barh', figsize=(1500, plot_length))
# -

# # Compare Fantasy Pros

df_train_results = df_train_results_reg.copy()
df_testresults = df_test_results_reg.copy()

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

    fp = pd.read_sql_query('SELECT * FROM FantasyPros', con=sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Databases/Season_Stats.sqlite3'))
    fp.year = fp.year-1
    df_train_full = pd.merge(df_train_full, fp, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
    df_train_full = df_train_full[df_train_full.fp > 5]


    from sklearn.linear_model import Lasso
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import Ridge


    df_train_full = df_train_full.dropna()
    lass = Lasso(alpha=50)
    y = 'fantasy_pts'
    y_other = 'y_act'
    
#     y_other = 'fantasy_pts'
#     y = 'y_act'


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
    
# df_all = (df_all * pts_dict[set_pos]).sum(axis=1)
# df_fp = (df_fp * pts_dict[set_pos]).sum(axis=1)
# df_train_full = pd.concat([df_train_full, fantasy_pts], axis=1)
y_test = df_train_full.loc[(df_train_full.year <= i) & (df_train_full.year >= min_year), y]

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

y_test = df_train_full.loc[(df_train_full.year <= i) & (df_train_full.year >= min_year), y].reset_index(drop=True)


ensemble_error = round(np.mean(np.sqrt(abs(mean_squared_error(full_models.pred_fp_per_game, y_test)))), 2)
print(f'Ensemble Error: {ensemble_error}')

# save the results to the output dataframe
# output = pd.concat([output, pd.DataFrame({'FantasyProsRMSE': [fantasy_pros_error], 'EnsembleRMSE': [ensemble_error]})], axis=1)
# -
(3.99-3.36) / 3.99


