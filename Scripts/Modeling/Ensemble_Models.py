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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, matthews_corrcoef, r2_score
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
# RowId
#==========

# set for true if last year in dataset is validation or false if final predict
val_run = False

# # QB <= 3 year
# class_id = (29, 35)
# reg_id_stat = (163, 216)
# reg_id_fp = (217, 225)

# # QB > 3 year
# class_id = (36, 42)
# reg_id_stat = (226, 279)
# reg_id_fp = (280, 288)

# # RB <= 2 year
# class_id = (15, 21)
# reg_id_stat = (73, 108)
# reg_id_fp = (109, 117)

# # RB > 2 year
# class_id = (22, 28)
# reg_id_stat = (118, 153)
# reg_id_fp = (154, 162)

# # WR <= 2 years
# class_id = (1, 7)
# reg_id_stat = (1, 27)
# reg_id_fp = (28, 36)

# # WR > 2 years
# class_id = (8, 14)
# reg_id_stat = (37, 63)
# reg_id_fp = (64, 72)

# # TE <= 3 year
# class_id = (43, 49)
# reg_id_stat = (289, 315)
# reg_id_fp = (316, 324)

# TE > 3 year
class_id = (50, 56)
reg_id_stat = (325, 351)
reg_id_fp = (352, 360)

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
    res = res.drop(['pkey', 'model', 'val_score', 'test_score'], axis=1).drop_duplicates()
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
    
    if val_run:
        pos[set_pos]['test_years'] = res['test_years']
    else: 
        pos[set_pos]['test_years'] = 1
    
    return best_models


# -

# ## Pulling in Data

def pull_class_data(breakout_metric, adp_ppg_low, adp_ppg_high, pct_off, act_ppg, set_pos, set_year, pts_dict=pts_dict, pos=pos):

    #==========
    # Pull and clean compiled data
    #==========

    # connect to database and pull in positional data
    df = pd.read_sql_query('SELECT * FROM ' + set_pos + '_' + str(set_year), con=conn)
    
    ay_feat = ['ay_per_game', 'yac_per_game', 'racr', 'ay_per_tar', 'ay_per_rec',
                   'ay_converted','yac_per_ay', 'air_yd_mkt_share', 'wopr', 'rec_yds_per_ay',
                   'yac_plus_ay', 'team_yac', 'tm_air_per_att', 'tm_ay_converted', 'tm_rec_yds_per_ay',
                   'tm_yac_per_ay', 'yac_mkt_share', 'yac_wopr', 'total_tgt_mkt_share']
            
    # append air yards for specified positions
    if pos[set_pos]['use_ay']:
        ay = pd.read_sql_query('SELECT * FROM AirYards', con=sqlite3.connect(path + 'Data/Databases/Season_Stats.sqlite3'))
        df = pd.merge(df, ay, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
        
        for f in ['med_features', 'max_features', 'age_features']:
            pos[set_pos][f].extend(ay_feat)
            pos[set_pos][f] = list(set(pos[set_pos][f]))
            
    else:
        for f in ['med_features', 'max_features', 'age_features']:
            pos[set_pos][f] = [x for x in pos[set_pos][f] if x not in ay_feat]

    # apply the specified touches and game filters
    df, _, _ = touch_game_filter(df, pos, set_pos, set_year)

    # calculate FP for a given set of scoring values
    df = calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)

    #==============
    # Create Break-out Probability Features
    #==============

    # get the train and prediction dataframes for FP per game
    df_train_orig, df_predict = get_train_predict(df, breakout_metric, pos, set_pos, set_year-pos[set_pos]['test_years'], 
                                              pos[set_pos]['earliest_year'], pos[set_pos]['features'])

    # get the train and prediction dataframes for FP per game
    df_train_orig, df_predict = get_train_predict(df, breakout_metric, pos, set_pos, set_year-pos[set_pos]['test_years'], 
                                                  pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    
    if val_run:
        df_predict = df_predict.dropna(subset=['y_act']).reset_index(drop=True)

    # get the adp predictions and merge back to the training dataframe
    df_train_orig, df_predict, lr = get_adp_predictions(df_train_orig, df_predict, 1)

    # filter to adp cutoffs
    df_train_orig = adp_filter(df_train_orig, adp_ppg_low, adp_ppg_high)
    df_predict = adp_filter(df_predict, adp_ppg_low, adp_ppg_high)
    
    # filter to year_exp cutoff
    df_train_orig = df_train_orig[eval(f'df_train_orig.year_exp{year_exp_cut}')].reset_index(drop=True)
    df_predict = df_predict[eval(f'df_predict.year_exp{year_exp_cut}')].reset_index(drop=True)

    # create the results dataframes
    df_train_results = df_train_orig[['player', 'year', 'y_act', 'pct_off', 'avg_pick_pred']].copy()
    df_test_results = df_predict[['player', 'year', 'avg_pick_pred', 'y_act']].copy()

    # generate labels for prediction based on cutoffs
    df_train_orig = class_label(df_train_orig, pct_off, act_ppg)
    df_predict = class_label(df_predict, pct_off, act_ppg)

    # get the minimum number of training samples for the initial datasets
    min_samples = int(0.5*df_train_orig[df_train_orig.year <= df_train_orig.year.min() + pos[set_pos]['skip_years']].shape[0])

    # print the value-counts
    print(df_train_orig.y_act.value_counts())
    print('Min Year:', df_train_orig.year.min())
    print('Max Year:', df_train_orig.year.max())
    print('Compiled Dataset Equals Stored Dataset?: ', df_train_orig.equals(load_pickle(f'{path}Model_Outputs/Model_Datasets_Class/{str(min_rowid)}.p')))
  
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
    corr_cut = p['corr_cut']
    scale = True if p['scale']==1 else False
    pca = True if p['pca']==1 else False
    n_components = p['n_components'] 
    
    model_p = {}
    for k, v in p.items():
        if k not in ['use_smote', 'zero_weight', 'collinear_cutoff', 'scale', 'pca', 'n_components', 'corr_cut']:
            model_p[k] = v
            
    est = class_models[model]
    est.set_params(**model_p)
    est.class_weight = {0: zero_weight, 1: 1}

    result, val_pred, ty_pred, ty_proba, trained_est, cols = class_validation(est, df_train_orig, df_predict, corr_cut, col_cut,
                                                                              use_smote, skip_years, scale, pca, n_components)

    return result, val_pred, ty_pred, ty_proba, trained_est, cols


# -

# ## Running Ensembles

def run_class_ensemble(best_models, df_train_results, df_test_results, pos=pos):

    skip_year = pos[set_pos]['skip_years']

    print(best_models)

    results = {'summary': {}, 'val_pred': {}, 'ty_pred': {}, 'ty_proba': {}, 'trained_model': {}, 'cols': {}}
    num_models = best_models.index.max() + 1
    for i in range(num_models):

        m = best_models.loc[i, 'model']
        pkey = best_models.loc[i, 'pkey']
        par = load_pickle(path + f'Model_Outputs/Model_Params_Class/{pkey}.p')
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

        ty_proba = ty_proba.rename(columns={'proba': 'proba' + str(i)})
        results['ty_proba'][i] = ty_proba

    #============
    # Create Ensembles and Provide Summary of Best Models
    #============

    # create the summary dataframe
    summary = class_create_summary(results, keep_first=False)

    best_results = summary[summary.F1Score >= 0.1*summary.F1Score.max()].copy()
    best_results.F1Score = best_results.F1Score - best_results.F1Score.min()
    sum_score = best_results.F1Score.sum()
    best_results['Weighting'] = best_results.F1Score / sum_score
    print(best_results)
    for i, row in best_results.iterrows():

        val_pred = results['val_pred'][row.Iteration]
        val_pred.iloc[:, -1] = val_pred.iloc[:, -1] * row.Weighting

        ty_pred = results['ty_pred'][row.Iteration]
        ty_pred.iloc[:, -1] = ty_pred.iloc[:, -1] * row.Weighting

        ty_proba = results['ty_proba'][row.Iteration]
        ty_proba.iloc[:, -1] = ty_proba.iloc[:, -1] * row.Weighting

        if i == 0:
            val_ensemble = val_pred
            ty_ensemble = ty_pred
            ty_proba_ensemble = ty_proba
        else:
            val_ensemble = pd.merge(val_ensemble, val_pred.drop(['avg_pick','y_act'], axis=1), on=['player', 'year'])
            ty_ensemble = pd.merge(ty_ensemble, ty_pred.drop(['avg_pick'], axis=1), on=['player', 'year'])
            ty_proba_ensemble = pd.merge(ty_proba_ensemble, ty_proba.drop(['avg_pick'], axis=1), on=['player', 'year'])

    for ens in [val_ensemble, ty_ensemble, ty_proba_ensemble]:
        ens_cols = [c for c in ens.columns if 'pred' in c or 'proba' in c]
        ens['pred'] = ens.loc[:, ens_cols].sum(axis=1)

    num_rows = len(summary)
    results['val_pred'][num_rows+1] = val_ensemble
    results['ty_pred'][num_rows+1] = ty_ensemble
    results['ty_proba'][num_rows+1] = ty_proba_ensemble

    val_ensemble['pred_check'] = 0
    val_ensemble.loc[val_ensemble.pred >= 0.25, 'pred_check'] = 1
    acc_score = round(matthews_corrcoef(val_ensemble.y_act, val_ensemble.pred_check), 3)
    new_result = pd.DataFrame({'Iteration': [num_rows+1], 'Model': 'Ensemble', 'F1Score': acc_score})
    summary = pd.concat([summary, new_result], axis=0)

    summary = summary.sort_values(by='F1Score', ascending=False).reset_index(drop=True)
    print(summary.head(20))

    #----------
    # Store Prediction Results
    #----------

    # get the best model and pull out the results
    best_iter = int(summary[summary.Model.str.contains('Ensemble')].iloc[0].Iteration)
    #     best_iter = int(summary.iloc[0].Iteration)
    print(best_iter)
    best_ty = results['ty_pred'][best_iter][['player', 'year', 'pred']]
    best_ty = best_ty.rename(columns={'pred': 'class_' + breakout_metric})

    best_train = results['val_pred'][best_iter][['player', 'year', 'pred', 'y_act']]
    best_train = best_train.rename(columns={'pred': 'class_' + breakout_metric, 'y_act': 'class_act_' + breakout_metric})

    best_ty_proba = results['ty_proba'][best_iter][['player', 'year', 'pred']]
    best_ty_proba = best_ty_proba.rename(columns={'pred': 'proba_' + breakout_metric})

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

for i in [class_id]:
    
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
df_train_results['pred_check'] = 0
df_train_results.loc[df_train_results.class_fp_per_game >= 0.5, 'pred_check'] = 1
print('Val Low+High Matthews Score %0.3f' % matthews_corrcoef(df_train_results.class_act_fp_per_game, df_train_results.pred_check))
# -

df_train_results.groupby('pred_check').agg({'y_act': 'mean',
                                           'pct_off': 'mean',
                                           'avg_pick_pred': 'mean'})

if val_run:
    df_test_results['pct_off' ] = (df_test_results.y_act - df_test_results.avg_pick_pred) / df_test_results.avg_pick_pred
    df_test_results['class_act_fp_per_game'] = 0
    df_test_results.loc[(df_test_results.pct_off >= 0.15) & (eval(f'df_test_results.y_act{act_ppg}')), 'class_act_fp_per_game'] = 1

    df_test_results.loc[df_test_results.class_fp_per_game >= 0.5, 'check'] = 1
    df_test_results.loc[df_test_results.class_fp_per_game < 0.5, 'check'] = 0

    print('Test Low+High Matthews Score %0.3f' % matthews_corrcoef(df_test_results.class_act_fp_per_game, df_test_results.check))

    print(df_test_results.groupby(['year','check']).agg({'y_act': 'mean',
                                                   'pct_off': 'mean',
                                                   'avg_pick_pred': 'mean'}))

df_test_results.sort_values(by='class_fp_per_game', ascending=False).iloc[:60]

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


# + [markdown] jupyter={"source_hidden": true}
# ind = 0
# try:
#     imp = pd.DataFrame(results['trained_model'][ind].coef_, columns=results['cols'][ind]).T
#     imp = imp[abs(imp) > 0.2*imp.max()].sort_values(by=0).reset_index()
# except:
#     imp = pd.DataFrame([results['trained_model'][ind].feature_importances_], columns=results['cols'][ind]).T
#     imp = imp[abs(imp) > 0.1*imp.max()].sort_values(by=0).reset_index()
# imp.plot_bokeh(x='index', y=0, kind='barh', figsize=(600, 1000))
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
    res = res.drop(['pkey', 'metric', 'model', 'rmse_validation', 'rmse_validation_adp', 
                    'rmse_test', 'rmse_test_adp', 'r2_test', 'r2_test_adp'], axis=1).drop_duplicates()
    res = res.rename(columns = {'year': 'set_year', 'pos': 'set_pos'})
    res = res.T.to_dict()[0]

    # set variables for all items in results
    for k, v in res.items():
        globals()[k] = v
    
    if globals()['set_year'] == 2018:
        globals()['set_year'] = 2019
    
    # set all of the data preparation filters 
    pos[set_pos]['req_touch'] = res['req_touch']
    pos[set_pos]['req_games'] = res['req_games']
    pos[set_pos]['earliest_year'] = res['earliest_year'] - 1
    pos[set_pos]['skip_years'] = res['skip_years']
    pos[set_pos]['use_ay'] = True if res['use_ay']==1 else False
    pos[set_pos]['features'] = res['features']
    
    
    if val_run:
        pos[set_pos]['test_years'] = res['test_years']
    else:
        pos[set_pos]['test_years'] = 1

    #==========
    # Pull and clean compiled data
    #==========

    # connect to database and pull in positional data
    df = pd.read_sql_query('SELECT * FROM ' + set_pos + '_' + str(set_year), con=conn)

    ay_feat = ['ay_per_game', 'yac_per_game', 'racr', 'ay_per_tar', 'ay_per_rec',
                   'ay_converted','yac_per_ay', 'air_yd_mkt_share', 'wopr', 'rec_yds_per_ay',
                   'yac_plus_ay', 'team_yac', 'tm_air_per_att', 'tm_ay_converted', 'tm_rec_yds_per_ay',
                   'tm_yac_per_ay', 'yac_mkt_share', 'yac_wopr', 'total_tgt_mkt_share']
            
    # append air yards for specified positions
    if pos[set_pos]['use_ay']:
        ay = pd.read_sql_query('SELECT * FROM AirYards', con=sqlite3.connect(path + 'Data/Databases/Season_Stats.sqlite3'))
        df = pd.merge(df, ay, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
        
        for f in ['med_features', 'max_features', 'age_features']:
            pos[set_pos][f].extend(ay_feat)
            pos[set_pos][f] = list(set(pos[set_pos][f]))
            
    else:
        for f in ['med_features', 'max_features', 'age_features']:
            pos[set_pos][f] = [x for x in pos[set_pos][f] if x not in ay_feat]
            

    # apply the specified touches and game filters
    df, _, _ = touch_game_filter(df, pos, set_pos, set_year)

    # calculate FP for a given set of scoring values
    df = calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)
    
    return best_models, df
# -

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
def validation(m, models, args, df_train_orig, df_predict_orig, skip_years):

    lr = LinearRegression()
    
    # remove the extra args not needed for modeling
    scale = True if args['scale'] == 1 else False
    pca = True if args['pca'] == 1 else False

    n_components = args['n_components']
    corr_cutoff = args['corr_cutoff']
    collinear_cutoff = args['collinear_cutoff']

    for arg in ['scale', 'pca', 'corr_cutoff', 'collinear_cutoff', 'n_components']:
        del args[arg]

    # set up the estimator
    estimator = models[m]
    estimator.set_params(**args)

    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------

    # return the df_train with only relevant features remaining
    df_train = corr_collinear_removal(df_train_orig, corr_cutoff, collinear_cutoff)
    df_predict = df_predict_orig[df_train.columns]

    years = df_train.year.unique()
    years = years[years > np.min(years) + skip_years]

    # set up array to save predictions and years to iterate through
    roll_predictions = np.array([]) 
    adp_predictions = np.array([])
    y_rolls = np.array([])

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        roll_split = df_train[df_train.year == m]

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_roll, y_train, y_roll = X_y_split(train_split, roll_split, scale, pca, n_components)

        # train the estimator and get predictions
        estimator.fit(X_train, y_train)
        roll_predict = estimator.predict(X_roll)

        # append the predictions
        roll_predictions = np.append(roll_predictions, roll_predict, axis=0)

        #--------------
        # Create ADP rolling predictions
        #--------------

        # append the y rolling values
        y_rolls = np.append(y_rolls, y_roll, axis=0)

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_adp, y_train, y_adp = X_y_split(train_split, roll_split, False, False, n_components)
        lr.fit(X_train.avg_pick.values.reshape(-1,1), y_train)
        adp_predict = lr.predict(X_adp.avg_pick.values.reshape(-1,1))
        adp_predictions = np.append(adp_predictions, adp_predict, axis=0)

    # splitting the train and validation sets into X_train, y_train, X_val and y_val
    lr = LinearRegression()
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, False, False, n_components)
    lr.fit(X_train.avg_pick.values.reshape(-1,1), y_train)
    test_adp = lr.predict(X_test.avg_pick.values.reshape(-1,1))

    # splitting the train and validation sets into X_train, y_train, X_val and y_val
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)
    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    results = error_compare(roll_predictions, adp_predictions, y_rolls)

    val_pred = df_train.loc[df_train.year.isin(years), ['player', 'year', 'avg_pick', 'y_act']].reset_index(drop=True)
    val_pred = pd.concat([val_pred, pd.Series(roll_predictions, name='pred')], axis=1)
    val_pred = pd.concat([val_pred, pd.Series(adp_predictions, name='pred_adp')], axis=1)

    ty_pred = df_predict[['player', 'year', 'avg_pick']].copy()
    ty_pred = pd.concat([ty_pred[['player', 'year', 'avg_pick']], pd.Series(test_predict, name='pred')], axis=1)
    
    return results, val_pred, ty_pred


def run_reg_ensemble(df, best_models, df_train_results, df_test_results, adp_ppg_low, adp_ppg_high, pos=pos):

    output = pd.DataFrame()
    metrics = list(best_models.metric.unique())
    for metric in metrics:            
        
        #==========
        # Pull and clean compiled data
        #==========

        # extract the models for the current metric
        print('Running ' + metric)
        metric_models = best_models[best_models.metric==metric].copy().reset_index(drop=True)

        # get the train and predict dataframes
        df_train_orig, df_predict_orig = get_train_predict(df, metric, pos, set_pos, set_year-pos[set_pos]['test_years'], 
                                                           pos[set_pos]['earliest_year']-1, pos[set_pos]['features'])
        
        if val_run:
            df_predict_orig = df_predict_orig.dropna(subset=['y_act']).reset_index(drop=True)
            
        df_predict_orig = df_predict_orig.fillna(0)
        df_train_orig = df_train_orig.fillna(0)
        
        # get the adp predictions and merge back to the training dataframe
        df_train_orig, df_predict_orig, lr = get_adp_predictions(df_train_orig, df_predict_orig, 1)
        df_train_orig = df_train_orig.drop('pct_off', axis=1)
        df_predict_orig = df_predict_orig.drop('pct_off', axis=1)
        
         # filter to year_exp cutoff
        df_train_orig = df_train_orig[eval(f'df_train_orig.year_exp{year_exp_cut}')].reset_index(drop=True)
        df_predict_orig = df_predict_orig[eval(f'df_predict_orig.year_exp{year_exp_cut}')].reset_index(drop=True)
        
        # print out relevant stats about the dataframes
        print(f'Number of training rows and features: {df_train_orig.shape}')
        print(f'Number of test rows and features: {df_predict_orig.shape}')
        print('Dataset equals training?: ' + str(df_train_orig.equals(load_pickle(f'{path}Model_Outputs/Model_Datasets/{str(min_rowid)}.p'))))

        #==========
        # Run the Models
        #==========

        results = {'summary': {}, 'val_pred': {}, 'ty_pred': {}, 'trained_model': {}, 'cols': {}}
        num_models = metric_models.index.max() + 1

        results = {'summary': {}, 'val_pred': {}, 'ty_pred': {}, 'trained_model': {}, 'cols': {}}
        num_models = int(metric_models.index.max() + 1)

        for i in range(num_models):

            m = metric_models.loc[i, 'model']
            pkey = metric_models.loc[i, 'pkey']
            args = load_pickle(path + f'Model_Outputs/Model_Params/{pkey}.p')

            print(f'Running {m}')

            results[i] = {}
            results['summary'][i] = [m]

            result, val_pred, ty_pred = validation(m, models, args, df_train_orig, df_predict_orig, pos[set_pos]['skip_years'])

            # save out the predictions to the dictionary
            results['summary'][i].extend(result)

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
        
        summary['PredRMSEScale'] = summary['PredRMSE'].max() - summary['PredRMSE']
        best_results = summary[summary.PredRMSEScale >= 0.25*summary.PredRMSEScale.max()].copy()
        sum_score = best_results.PredRMSEScale.sum()
        best_results['Weighting'] = best_results.PredRMSEScale / sum_score
        print(best_results)
        for i, row in best_results.iterrows():

            val_pred = results['val_pred'][row.Iteration]
            val_pred.iloc[:, -2] = val_pred.iloc[:, -2] * row.Weighting

            ty_pred = results['ty_pred'][row.Iteration]
            ty_pred.iloc[:, -1] = ty_pred.iloc[:, -1] * row.Weighting

            if i == 0:
                val_ensemble = val_pred
                ty_ensemble = ty_pred
            else:
                val_ensemble = pd.merge(val_ensemble, val_pred.drop(['avg_pick','y_act', 'pred_adp'], axis=1), on=['player', 'year'])
                ty_ensemble = pd.merge(ty_ensemble, ty_pred.drop(['avg_pick'], axis=1), on=['player', 'year'])

        for ens in [val_ensemble, ty_ensemble]:
            ens_cols = [c for c in ens.columns if 'pred' in c and c != 'pred_adp']
            ens['pred'] = ens.loc[:, ens_cols].sum(axis=1)
        
        num_rows = len(summary)
        results['val_pred'][num_rows+1] = val_ensemble
        results['ty_pred'][num_rows+1] = ty_ensemble
        
        rmse = round(np.sqrt(mean_squared_error(val_ensemble.y_act, val_ensemble.pred)), 3)
        r2= round(r2_score(val_ensemble.y_act, val_ensemble.pred), 3)

        new_result = pd.DataFrame({'Iteration': [num_rows+1], 'Model': 'Ensemble', 'PredRMSE': rmse, 'PredR2': r2})
        summary = pd.concat([summary, new_result], axis=0)

        summary = summary.sort_values(by='PredRMSE', ascending=True).reset_index(drop=True)
        print(summary.head(20))

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
df_train_results_reg = pd.DataFrame()
df_test_results_reg = pd.DataFrame()

for i in [reg_id_stat]:
    
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
# -

if len(list(best_models.metric.unique())) > 1:

    # extract points list and get the idx of point attributes based on length of list
    pts_list = pts_dict[set_pos]
    pred_metrics = ['pred_' + m for m in pos[set_pos]['metrics']]
    df_train_results_reg['pred_fp_per_game_stat'] = (df_train_results_reg[pred_metrics] * pts_list).sum(axis=1)
    df_test_results_reg['pred_fp_per_game_stat'] = (df_test_results_reg[pred_metrics] * pts_list).sum(axis=1)

# +
for i in [reg_id_fp]:
    
    # specify the minimum and maximum rows from the database
    min_rowid = i[0]
    max_rowid = i[1]

    # set all the variables
    best_models, df = pull_reg_params(min_rowid, max_rowid)
    df_train_results_reg, df_test_results_reg, results = run_reg_ensemble(df, best_models, 
                                                                                df_train_results_reg, 
                                                                                df_test_results_reg, adp_ppg_low, adp_ppg_high)
    
df_train_results_reg = df_train_results_reg.reset_index(drop=True)
df_test_results_reg = df_test_results_reg.reset_index(drop=True)

# +
df_train_results = df_train_results_reg.copy()
df_test_results = df_test_results_reg.copy()

# remove duplicated columns
df_train_results = df_train_results.loc[:, ~df_train_results.columns.duplicated()]
df_test_results = df_test_results.loc[:, ~df_test_results.columns.duplicated()]

# +
df_test_results['avg_pred'] = df_test_results[[ 'pred_fp_per_game', 'pred_fp_per_game_stat']].mean(axis=1)

if val_run: 
    print('Test R2: %0.3f' % r2_score(df_test_results.y_act, df_test_results.avg_pred))
    print('Test RMSE: %0.3f' % np.sqrt(mean_squared_error(df_test_results.y_act, df_test_results.avg_pred)))
    print('ADP R2: %0.3f' % r2_score(df_test_results.y_act, df_test_results.avg_pick_pred))
    print('ADP RMSE: %0.3f' % np.sqrt(mean_squared_error(df_test_results.y_act, df_test_results.avg_pick_pred)))
# -

df_train_results['avg_pred'] = df_train_results[['pred_fp_per_game', 'pred_fp_per_game_stat']].mean(axis=1)
print('Val R2: %0.3f' % r2_score(df_train_results.y_act, df_train_results.avg_pred))
print('Val RMSE: %0.3f' % np.sqrt(mean_squared_error(df_train_results.y_act, df_train_results.avg_pred)))
print('Val R2: %0.3f' % r2_score(df_train_results.y_act, df_train_results.avg_pick_pred))
print('val RMSE: %0.3f' % np.sqrt(mean_squared_error(df_train_results.y_act, df_train_results.avg_pick_pred)))

df_test_results = df_test_results.sort_values(by='avg_pred', ascending=False).reset_index(drop=True)
df_test_results.loc[:17, ['player', 'avg_pick_pred', 'avg_pred', 'pred_fp_per_game', 'pred_fp_per_game_stat', 'class_fp_per_game', 'y_act']]

# +
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X_train = df_train_results[['class_fp_per_game',  'avg_pred']]
X_test = df_test_results[['class_fp_per_game', 'avg_pred']]
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

if val_run:
    test_adp_err = np.sqrt(mean_squared_error(df_test_results.y_act, df_test_results.avg_pick_pred))
    test_pred_only_err = np.sqrt(mean_squared_error(df_test_results.y_act, df_test_results.pred_fp_per_game))
    test_pred_class_err = np.sqrt(mean_squared_error(df_test_results.y_act, test_pred_plus_class))

test_y = df_test_results.y_act
test_y.name='y_act'
test_adp = df_test_results.avg_pick_pred
# X_test = pd.concat([X_test, test_y, test_adp], axis=1)

print('Predicted ADP Error: %0.3f' % adp_pred_err)
print('Predicted Only Error: %0.3f' % pred_only_err )
print('Predicted Plus Class Error: %0.3f' % pred_plus_class_err)
print('')
if val_run:
    print('Predicted ADP Error: %0.3f' % test_adp_err)
    print('Predicted Only Error: %0.3f' % test_pred_only_err )
    print('Predicted Plus Class Error: %0.3f' % test_pred_class_err)
# -

# # Bayes Regression

# +
import pymc3 as pm

data = pd.concat([X_train, y], axis=1)
formula = 'y_act ~ class_fp_per_game + avg_pred'

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
import scipy.stats as stats

# Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation, max_bound):
    
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
    
    # Add in intercept term
    test_observation['Intercept'] = 1
    
    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)
    
    # create truncated distribution
    lower, upper = 0,  max_bound * 1.2
    trunc_dist = stats.truncnorm((lower - mean_loc) / sd_value, (upper - mean_loc) / sd_value, 
                                  loc=mean_loc, scale=sd_value)
    estimates = trunc_dist.rvs(1000)

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                kde_kws = {'linewidth' : 4},
                label = 'Estimated Dist.')
    
    # Plot the mean estimate
    plt.vlines(x = mean_loc, ymin = 0, ymax = 0.1, 
               linestyles = '--', colors = 'red',
               label = 'Pred Estimate',
               linewidth = 2.5)
    
    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');
    
    # Prediction information
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                       np.percentile(estimates, 95)))
    
    plt.show()
    
    return estimates


# -

dists = []
for i in range(len(df_test_results)):
    print(df_test_results.loc[i, ['player', 'avg_pick_pred', 'y_act']])
    estimates = test_model(normal_trace, 
                           X_test.iloc[i, :].copy(), 
                           df_train_results.y_act.max())
    dists.append(estimates)

# +
output = pd.concat([df_test_results.player, pd.DataFrame(dists)], axis=1)
output = output.assign(pos=set_pos)
order_cols = ['player', 'pos']
order_cols.extend([i for i in range(1000)])

output = output[order_cols]
output.iloc[:, 2:] = np.uint32(output.iloc[:, 2:] * 16)
players = tuple(output.player)
# -

# # 2019

# conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
# conn_sim.cursor().execute(f'''DELETE FROM Version1_{set_year-1} WHERE player in {players}''')
# conn_sim.commit()
#
# append_to_db(output, 'Simulation', f'Version1_{set_year-1}', 'append')

# conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
# pd.read_sql_query('''SELECT * FROM Version1_2019''', conn_sim)

# # 2020

vers = 'Version3'

# +
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
conn_sim.cursor().execute(f'''DELETE FROM {vers}_{set_year} WHERE player in {players}''')
conn_sim.commit()

append_to_db(output, 'Simulation', f'{vers}_{set_year}', 'append')
# -

conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
chk = pd.read_sql_query(f'''SELECT * FROM {vers}_2020''', conn_sim)
chk

# ## Create New Vers

# conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
# new_vers = pd.read_sql_query(f'''SELECT * FROM Version2_2020''', conn_sim)
# append_to_db(new_vers, 'Simulation', 'Version3_2020')


