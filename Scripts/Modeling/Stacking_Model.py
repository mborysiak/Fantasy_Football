#%%
# core packages
from random import random
import pandas as pd
import numpy as np
import os
import gzip
import pickle
from joblib import Parallel, delayed
from ff.db_operations import DataManage
from ff import general
from skmodel import SciKitModel
from Fix_Standard_Dev import *
import zModel_Functions as mf

import pandas_bokeh
pandas_bokeh.output_notebook()
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import zHelper_Functions as hf
pos = hf.pos

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
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'Rookie_RB'

# set year to analyze
set_year = 2023

# set the version
vers = 'beta'

# set with this year or next
current_or_next_year = 'current'

mse_wt = 1
sera_wt = 0
r2_wt = 0
brier_wt = 1
matt_wt = 0

# determine whether to do run/pass/rec separate or together
pos['QB']['rush_pass'] = 'both'
pos['RB']['rush_pass'] = 'both'
pos['WR']['rush_pass'] = ''
pos['TE']['rush_pass'] = ''
pos['Rookie_RB']['rush_pass'] = ''
pos['Rookie_WR']['rush_pass'] = ''

#==========
# Model Settings
#==========

pos['QB']['req_touch'] = 250
pos['RB']['req_touch'] = 60
pos['WR']['req_touch'] = 50
pos['TE']['req_touch'] = 40
pos['Rookie_RB']['req_touch'] = 1
pos['Rookie_WR']['req_touch'] = 1

pos['QB']['req_games'] = 8
pos['RB']['req_games'] = 8
pos['WR']['req_games'] = 8
pos['TE']['req_games'] = 8
pos['Rookie_RB']['req_games'] = 5
pos['Rookie_WR']['req_games'] = 5

pos['QB']['earliest_year'] = 2004 # first year QBR available
pos['RB']['earliest_year'] = 1998
pos['WR']['earliest_year'] = 1998
pos['TE']['earliest_year'] = 1998
pos['Rookie_RB']['earliest_year'] = 1998
pos['Rookie_WR']['earliest_year'] = 1998

pos['QB']['val_start'] = 2012
pos['RB']['val_start'] = 2012
pos['WR']['val_start'] = 2012
pos['TE']['val_start'] = 2012
pos['Rookie_RB']['val_start'] = 2012
pos['Rookie_WR']['val_start'] = 2012

pos['QB']['features'] = 'v2'
pos['RB']['features'] = 'v2'
pos['WR']['features'] = 'v2'
pos['TE']['features'] = 'v2'
pos['Rookie_RB']['features'] = 'v1'
pos['Rookie_WR']['features'] = 'v1'

pos['QB']['test_years'] = 1
pos['RB']['test_years'] = 1
pos['WR']['test_years'] = 1
pos['TE']['test_years'] = 1
pos['Rookie_RB']['test_years'] = 1
pos['Rookie_WR']['test_years'] = 1

pos['QB']['use_ay'] = False
pos['RB']['use_ay'] = False
pos['WR']['use_ay'] = False
pos['TE']['use_ay'] = False
pos['Rookie_RB']['use_ay'] = False
pos['Rookie_WR']['use_ay'] = False

pos['QB']['filter_data'] = 'greater_equal'
pos['RB']['filter_data'] = 'greater_equal'
pos['WR']['filter_data'] = 'greater_equal'
pos['TE']['filter_data'] = 'greater_equal'
pos['Rookie_RB']['filter_data'] = 'greater_equal'
pos['Rookie_WR']['filter_data'] = 'greater_equal'

pos['QB']['year_exp'] = 0
pos['RB']['year_exp'] = 0
pos['WR']['year_exp'] = 0
pos['TE']['year_exp'] = 0
pos['Rookie_RB']['year_exp'] = 0
pos['Rookie_WR']['year_exp'] = 0

pos['QB']['pct_off'] = 0
pos['RB']['pct_off'] = 0
pos['WR']['pct_off'] = 0
pos['TE']['pct_off'] = 0
pos['Rookie_RB']['pct_off'] = 0
pos['Rookie_WR']['pct_off'] = 0

pos['QB']['iters'] = 25
pos['RB']['iters'] = 25
pos['WR']['iters'] = 25
pos['TE']['iters'] = 25
pos['Rookie_RB']['iters'] = 25
pos['Rookie_WR']['iters'] = 25

pos['QB']['n_splits'] = 5
pos['RB']['n_splits'] = 5
pos['WR']['n_splits'] = 5
pos['TE']['n_splits'] = 5
pos['Rookie_RB']['n_splits'] = 4
pos['Rookie_WR']['n_splits'] = 4


def create_pts_dict(pos, set_pos):

    # define point values for all statistical categories
    pass_yd_per_pt = 0.04 
    pass_td_pt = 5
    int_pts = -2
    sacks = -1
    rush_yd_per_pt = 0.1 
    rec_yd_per_pt = 0.1
    rush_td = 7
    rec_td = 7
    ppr = 0.5

    if set_pos=='QB' and pos['QB']['rush_pass'] == 'rush':
        pass_yd_per_pt = 0 
        pass_td_pt = 0
        int_pts = 0
        sacks = 0

    elif set_pos=='QB' and pos['QB']['rush_pass'] == 'pass':
        rush_yd_per_pt = 0
        rush_td = 0

    elif set_pos == 'RB' and pos['RB']['rush_pass'] == 'rush':
        rec_yd_per_pt = 0
        rec_td = 0
        ppr = 0

    elif set_pos=='RB' and pos['RB']['rush_pass'] == 'rec':
        rush_yd_per_pt = 0
        rush_td = 0

    else:
        pass

    # creating dictionary containing point values for each position
    pts_dict = {}
    pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_td, int_pts, sacks]
    pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_td, rec_td]
    pts_dict['WR'] = [rec_yd_per_pt, ppr, rec_td]
    pts_dict['TE'] = [rec_yd_per_pt, ppr, rec_td]
    pts_dict['Rookie_RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_td, rec_td]
    pts_dict['Rookie_WR'] = [rec_yd_per_pt, ppr, rec_td]

    return pts_dict

def class_cutoff(pos):

    if pos['QB']['rush_pass'] == 'rush': pos['QB']['act_ppg'] = 4
    elif pos['QB']['rush_pass'] == 'pass': pos['QB']['act_ppg'] = 17
    else: pos['QB']['act_ppg'] = 19

    if pos['RB']['rush_pass'] == 'rush': pos['RB']['act_ppg'] = 13
    elif pos['RB']['rush_pass'] == 'rec': pos['RB']['act_ppg'] = 6
    else: pos['RB']['act_ppg'] = 17

    pos['WR']['act_ppg'] = 15
    pos['TE']['act_ppg'] = 12
    pos['Rookie_RB']['act_ppg'] = 13
    pos['Rookie_WR']['act_ppg'] = 11

    return pos


def create_pkey(pos, set_pos, cur_next):

    all_vars = ['req_touch', 'req_games', 'earliest_year', 'val_start', 
                'features', 'test_years', 'use_ay', 'filter_data', 'year_exp', 
                'act_ppg', 'pct_off', 'iters', 'rush_pass']

    pkey = str(set_pos)
    for var in all_vars:
        v = str(pos[set_pos][var])
        pkey = pkey + '_' + v

    pkey = pkey + '_' + str(cur_next)

    model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    
    return model_output_path

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def filter_less_equal(df, pos, set_pos):
    if pos[set_pos]['filter_data']=='less_equal':
        df = df.loc[df.year_exp != pos[set_pos]['year_exp']+1, :].reset_index(drop=True)
    return df

def update_output_dict(label, m, suffix, out_dict, oof_data, best_models):

    # append all of the metric outputs
    lbl = f'{label}_{m}{suffix}'
    out_dict['pred'][lbl] = oof_data['hold']
    out_dict['actual'][lbl] = oof_data['actual']
    out_dict['scores'][lbl] = oof_data['scores']
    out_dict['models'][lbl] = best_models
    out_dict['full_hold'][lbl] = oof_data['full_hold']

    return out_dict

def save_output_dict(out_dict, model_output_path, label):

    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')

#==========
# Pull and clean compiled data
#==========

def pull_data(pts_dict, set_pos, set_year):

    # load data and filter down
    df = dm.read(f'''SELECT * FROM {set_pos}_{set_year}''', 'Model_Inputs')

    # calculate FP for a given set of scoring values
    df = hf.calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)
    df['rules_change'] = np.where(df.year>=2010, 1, 0)
    
    # add in data to match up with Daily code
    df['week'] = 1
    df['game_date'] = df.year

    return df


def filter_df(df, pos, set_pos, set_year): 

    # apply the specified touches and game filters
    df, _, _ = hf.touch_game_filter(df, pos, set_pos, set_year)

    # # filter dataset
    if pos[set_pos]['filter_data']=='greater_equal':
        df = df.loc[df.year_exp >= pos[set_pos]['year_exp']].reset_index(drop=True)

    elif pos[set_pos]['filter_data']=='less_equal':
        df = df.loc[df.year_exp <= pos[set_pos]['year_exp']+1].reset_index(drop=True)

    # get the train and prediction dataframes for FP per game
    _, output_start = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                           set_year-pos[set_pos]['test_years'], 
                                           pos[set_pos]['earliest_year'], pos[set_pos]['features'])

    output_start = filter_less_equal(output_start, pos, set_pos)
    output_start = output_start[['player', 'avg_pick']]

    return df, output_start


def prepare_rookie_data(df, set_pos, current_or_next):

    df = df.sort_values(by='year').reset_index(drop=True)
    
    if current_or_next == 'current':
        df = df[(df.games > 6) | (df.year==set_year-1)].reset_index(drop=True)
        df = df.rename(columns={'fp_per_game': 'y_act'})
        df = df.drop(['fp_per_game_next', 'games', 'games_next'], axis=1)

    elif current_or_next == 'next':
        df = df[(df.games_next > 6) | (df.year==set_year-1)].reset_index(drop=True)
        df = df.rename(columns={'fp_per_game_next': 'y_act'})
        df = df.drop(['fp_per_game', 'games', 'games_next'], axis=1)

    df_train = df[(df.year < set_year-1) & (df.y_act > 0)].reset_index(drop=True)
    df_predict = df[df.year == set_year-1].reset_index(drop=True)

    output_start = df_predict[['player', 'avg_pick']].copy()
    print(set_pos, df_train.shape[0], df_predict.shape[0])

    df_predict = df_predict.fillna(0)
    for c in df_predict.columns:
        if len(df_predict[df_predict[c]==np.inf]) > 0:
            df_predict.loc[df_predict[c]==np.inf, c] = 0
        if len(df_predict[df_predict[c]==-np.inf]) > 0:
            df_predict.loc[df_predict[c]==-np.inf, c] = 0

    df_train_class = df_train.copy()
    df_predict_class = df_predict.copy()

    if set_pos == 'Rookie_WR':
        df_train_class['y_act'] = np.where(df_train_class.y_act >= 11, 1, 0)

    elif set_pos == 'Rookie_RB':
        df_train_class['y_act'] = np.where(df_train_class.y_act >= 13, 1, 0)

    min_samples = 3
    return df_train, df_predict, df_train_class, df_predict_class, output_start,min_samples

#==============
# Create Datasets
#==============

def get_reg_data(df, pos, set_pos):

    # get the train and prediction dataframes for FP per game
    df_train, df_predict = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                                set_year-pos[set_pos]['test_years'], 
                                                pos[set_pos]['earliest_year'], 
                                                pos[set_pos]['features'])
    df_predict = filter_less_equal(df_predict, pos, set_pos)

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, min_samples


def get_class_data(df, pos, set_pos):

    breakout_metric = 'fp_per_game'
    adp_ppg_high = '<100'
    adp_ppg_low = '>=0'
    act_ppg = '>=' + str(pos[set_pos]['act_ppg'])
    pct_off = '>=' +  str(pos[set_pos]['pct_off'])

    # get the train and prediction dataframes for FP per game
    df_train_class, df_predict_class = hf.get_train_predict(df, breakout_metric, pos, set_pos, 
                                                        set_year-pos[set_pos]['test_years'], 
                                                        pos[set_pos]['earliest_year'], 
                                                        pos[set_pos]['features'])

    # get the adp predictions and merge back to the training dataframe
    df_train_class, df_predict_class, _ = hf.get_adp_predictions(df_train_class, df_predict_class, 1)
    df_predict_class = filter_less_equal(df_predict_class, pos, set_pos)

    # filter to adp cutoffs
    df_train_class = hf.adp_filter(df_train_class, adp_ppg_low, adp_ppg_high)
    df_predict_class = hf.adp_filter(df_predict_class, adp_ppg_low, adp_ppg_high)

    # generate labels for prediction based on cutoffs
    df_train_class = hf.class_label(df_train_class, pct_off, act_ppg)
    df_predict_class = hf.class_label(df_predict_class, pct_off, act_ppg)
    df_predict_class = df_predict_class.drop('y_act', axis=1).fillna(0)

    # print the value-counts
    print('Training Value Counts:', df_train_class.y_act.value_counts()[0], '|', df_train_class.y_act.value_counts()[1])
    print(f'Number of Features: {df_train_class.shape[1]}')
    
    return  df_train_class, df_predict_class


def adjust_current_or_next(df_train, df_train_class):

    df_train = df_train.sort_values(by=['player', 'year']).reset_index(drop=True)
    df_train['y_act'] = df_train.groupby('player').y_act.shift(-1)
    df_train = df_train.dropna().sort_values(by='year').reset_index(drop=True)

    df_train_class = df_train_class.sort_values(by=['player', 'year']).reset_index(drop=True)
    df_train_class['y_act'] = df_train_class.groupby('player').y_act.shift(-1)
    df_train_class = df_train_class.dropna().sort_values(by='year').reset_index(drop=True)

    return df_train, df_train_class

#=================
# Model Functions
#=================

def output_dict():
    return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

def get_skm(skm_df, model_obj, to_drop=['player', 'team', 'pos']):
    
    skm = SciKitModel(skm_df, model_obj=model_obj, sera_wt=sera_wt, mse_wt=mse_wt, 
                      r2_wt=r2_wt, brier_wt=brier_wt, matt_wt=matt_wt)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10):

    if m == 'adp':
        
        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('feature_select'), 
                               skm.piece('std_scale'), 
                               skm.piece('lr')])

    elif stack_model:
        pipe = skm.model_pipe([
                            skm.piece('random_sample'),
                            skm.piece('k_best'),
                            skm.piece('std_scale'), 
                            skm.piece(m)
                        ])

    elif skm.model_obj == 'reg':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                                skm.piece('pca')
                                                ]),
                                skm.piece('k_best'),
                                skm.piece(m)])

    elif skm.model_obj == 'class':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                               skm.piece('std_scale'), 
                               skm.piece('select_perc_c'),
                               skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best_c'),
                                                ]),
                               skm.piece('k_best_c'),
                               skm.piece(m)])
    
    

    elif skm.model_obj == 'quantile':
        pipe = skm.model_pipe([
                                skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.piece(m)
                                ])
        pipe.steps[-1][-1].alpha = alpha


    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    if skm.model_obj =='reg' and m!='adp' and not stack_model: params['feature_union__pca__n_components'] = range(2,20)
    if m=='adp': params['feature_select__cols'] = [['avg_pick'], ['avg_pick', 'year']]
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if stack_model: 
        params['random_sample__frac'] = np.arange(0.05, 1, 0.05)
        params['k_best__k'] = range(2, 30, 1)

    if skm.model_obj == 'quantile':
        if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha

    return pipe, params


def get_model_output(model_name, cur_df, model_obj, out_dict, pos, set_pos, i, min_samples=10, alpha=''):

    print(f'\n{model_name}\n============\n')

    skm, X, y = get_skm(cur_df, model_obj)
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples)

    if model_obj == 'class': proba = True 
    else: proba = False

    # fit and append the ADP model
    best_models, oof_data, _, _ = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'], n_splits=pos[set_pos]['n_splits'],
                                                  col_split='year', time_split=pos[set_pos]['val_start'],
                                                  bayes_rand='rand', proba=proba, random_seed=(i+7)*19+(i*12)+6)
    out_dict = update_output_dict(model_obj, model_name, str(alpha), out_dict, oof_data, best_models)

    return out_dict, best_models, oof_data


#====================
# Stacking Functions
#====================

def load_all_stack_pred(model_output_path):

    # load the regregression predictions
    pred, actual, models_reg, _, full_hold_reg = mf.load_all_pickles(model_output_path, 'reg')
    X_stack, y_stack = mf.X_y_stack('reg', full_hold_reg, pred, actual)

    # load the class predictions
    pred_class, actual_class, models_class, _, full_hold_class = mf.load_all_pickles(model_output_path, 'class')
    X_stack_class, _ = mf.X_y_stack('class', full_hold_class, pred_class, actual_class)

    # load the quantile predictions
    pred_quant, actual_quant, models_quant, _, full_hold_quant = mf.load_all_pickles(model_output_path, 'quant')
    X_stack_quant, _ = mf.X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)

    # concat all the predictions together
    X_stack = pd.concat([X_stack, X_stack_class, X_stack_quant], axis=1)

    return X_stack, y_stack, models_reg, models_class, models_quant


def run_stack_models(final_m, i, X_stack, y_stack, best_models, scores, stack_val_pred, show_plots=True):

    print(f'\n{final_m}')

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), 'reg', to_drop=[])
    pipe, params = get_full_pipe(skm, final_m, stack_model=True)

    if 'gbmh' in final_m or 'knn' in final_m: print_coef = False
    else: print_coef = True

    best_model, stack_scores, stack_pred, _ = skm.best_stack(pipe, params,
                                                          X_stack, y_stack, n_iter=100, 
                                                          run_adp=True, print_coef=print_coef,
                                                           random_state=(i*12)+(i*17),
                                                           num_k_folds=3)
    best_models.append(best_model)
    scores.append(stack_scores['stack_score'])
    stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)

    if show_plots:
        mf.show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)
        mf.top_predictions(stack_pred['stack_pred'], stack_pred['y'], r2=True)

    return best_models, scores, stack_val_pred
    

# def get_stack_predict_data(df_train, df_predict, df_train_class, df_predict_class, 
#                            models_reg, models_class, models_quant):

#     _, X, y = get_skm(df_train, 'reg')
#     X_predict = mf.get_reg_predict_features(df_predict, models_reg, X, y)
#     X_predict_class = mf.get_class_predictions(models_class, df_train_class, df_predict_class)
#     X_predict_quant = mf.get_quant_predictions(df_train, df_predict, models_quant)

#     X_predict = pd.concat([X_predict, X_predict_class, X_predict_quant], axis=1)

#     return X_predict

def fit_and_predict(m, df_predict, X, y, proba):
    try:
        m.fit(X,y)

        if proba: cur_predict = m.predict_proba(df_predict[X.columns])[:,1]
        else: cur_predict = m.predict(df_predict[X.columns])
    except:
        cur_predict = []

    return cur_predict

def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(m, df_predict, X, y, proba) for m in ind_models)
        predictions = [p for p in predictions if len(p) > 0]
        predictions = pd.Series(pd.DataFrame(predictions).T.mean(axis=1), name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def get_stack_predict_data(df_train, df_predict, df_train_class, df_predict_class,
                           models_reg, models_class, models_quant):

    _, X, y = get_skm(df_train, 'reg', to_drop=['player', 'team', 'pos'])
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)

    print('Predicting Class Models')
    _, X, y = get_skm(df_train_class, 'class', to_drop=['player', 'team', 'pos'])
    X_predict_class = create_stack_predict(df_predict_class, models_class, X, y, proba=True)
    X_predict = pd.concat([X_predict, X_predict_class], axis=1)

    print('Predicting Quant Models')
    _, X, y = get_skm(df_train, 'quantile', to_drop=['player', 'team', 'pos'])
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    X_predict_player = pd.concat([df_predict[['player', 'team', 'week', 'year']], X_predict], axis=1)

    return X_predict_player, X_predict


def std_dev_features(cur_df, model_name, set_pos, show_plot=True):
    skm, X, y = get_skm(cur_df, 'reg')
    pipe, params = get_full_pipe(skm, model_name, stack_model=True)

    # fit and append the ADP model
    best_models, _, _ = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'], n_splits=pos[set_pos]['n_splits'],
                                           col_split='year', time_split=pos[set_pos]['val_start'],
                                           bayes_rand='custom_rand', random_seed=1234)

    for bm in best_models: bm.fit(X, y)
    if show_plot:
        mf.shap_plot(best_models, X, model_num=0)
        plt.show()

    return best_models, X


def add_std_dev_max(df_train, df_predict, output, set_pos, num_cols=5):

    std_dev_models, X = std_dev_features(df_train, 'enet', set_pos, show_plot=True)
    sd_cols, df_train, df_predict = mf.get_sd_cols(df_train, df_predict, X, std_dev_models, num_cols=num_cols)
    sd_spline, max_spline = get_std_splines(df_train, sd_cols, show_plot=True, k=1, 
                                            min_grps_den=int(df_train.shape[0]*0.3), 
                                            max_grps_den=int(df_train.shape[0]*0.1))

    output = mf.assign_sd_max(output, df_predict, df_train, sd_cols, sd_spline, max_spline)

    return output


def val_std_dev(model_output_path, output, best_val, iso_spline, show_plot=True):

    val_data = validation_compare_df(model_output_path, best_val)
    sd_max_met = StandardScaler().fit(val_data[['pred_fp_per_game']]).transform(output[['pred_fp_per_game']])

    if iso_spline=='iso':
        sd_m, max_m, min_m = get_std_splines(val_data, {'pred_fp_per_game': 1}, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.2), 
                                            max_grps_den=int(val_data.shape[0]*0.06),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m.predict(sd_max_met)
        output['max_score'] = max_m.predict(sd_max_met)

    elif iso_spline=='spline':
        sd_m, max_m, min_m = get_std_splines(val_data, {'pred_fp_per_game': 1}, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.2), 
                                            max_grps_den=int(val_data.shape[0]*0.05),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m(sd_max_met)
        output['max_score'] = max_m(sd_max_met)
 
    return output


def average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, show_plot=True, min_include=3):
    
    skm_stack, _, _ = get_skm(df_train, 'reg')
    best_val, best_predictions, best_score = best_average_models(skm_stack, scores, final_models, y_stack, 
                                                                 stack_val_pred, predictions, min_include)
    print(best_val.columns)
    if show_plot:
        mf.show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
    return best_val, best_predictions, best_score

def best_average_models(skm_stack, scores, final_models, y_stack, stack_val_pred, predictions, min_include = 3):

    n_scores = []
    models_included = []
    for i in range(len(scores)-min_include+1):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:i+min_include]
        models_included.append(top_n)
        model_idx = np.array(final_models)[top_n]
        
        n_score = skm_stack.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        n_scores.append(n_score)
        
    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores)
    best_score = n_scores[best_n]
    top_models = models_included[best_n]

    model_idx = np.array(final_models)[top_models]
    best_val = stack_val_pred[model_idx]
    best_predictions = predictions[model_idx]

    return best_val, best_predictions, best_score

#====================
# Outputs
#====================

def validation_compare_df(model_output_path, best_val):

    _, _, _, _, oof_data = mf.load_all_pickles(model_output_path, 'reg')
    oof_data = oof_data['reg_adp'][['player', 'team', 'year', 'y_act']].reset_index(drop=True)
    best_val = pd.Series(best_val.mean(axis=1), name='pred_fp_per_game')
    val_compare = pd.concat([oof_data, best_val], axis=1).rename(columns={'year': 'season'})
    
    return val_compare

def save_out_results(df, db_name, table_name, pos, set_year, set_pos, current_or_next_year):

    import datetime as dt

    df['pos'] = set_pos
    df['rush_pass'] = pos[set_pos]['rush_pass']
    df['filter_data'] = pos[set_pos]['filter_data']
    df['year_exp'] = pos[set_pos]['year_exp']
    df['current_or_next_year'] = current_or_next_year
    df['version'] = vers
    df['year'] = set_year
    df['date_modified'] = dt.datetime.now().strftime('%m-%d-%Y %H:%M')

    del_str = f'''pos='{set_pos}' 
                  AND rush_pass='{pos[set_pos]['rush_pass']}'
                  AND filter_data='{pos[set_pos]['filter_data']}'
                  AND year_exp={pos[set_pos]['year_exp']}
                  AND current_or_next_year = '{current_or_next_year}'
                  AND version = '{vers}'
                  AND year={set_year}'''

    dm.delete_from_db(db_name, table_name, del_str)
    dm.write_to_db(df, db_name, table_name, if_exist='append')

#%%

#------------
# Pull in the data and create train and predict sets
#------------

# pts_dict = create_pts_dict(pos, set_pos)
# pos = class_cutoff(pos)
# model_output_path = create_pkey(pos, set_pos,current_or_next_year)
# df = pull_data(pts_dict, set_pos, set_year)

# if 'Rookie' not in set_pos:
#     df, output_start = filter_df(df, pos, set_pos, set_year)
#     df_train, df_predict, min_samples = get_reg_data(df, pos, set_pos)
#     df_train_class, df_predict_class = get_class_data(df, pos, set_pos)
# else:
#     df_train, df_predict, df_train_class, df_predict_class, output_start, min_samples = prepare_rookie_data(df, set_pos, current_or_next_year)

# if current_or_next_year == 'next' and 'Rookie' not in set_pos: 
#     df_train, df_train_class = adjust_current_or_next(df_train, df_train_class)

# #%%
# #------------
# # Run the Regression, Classification, and Quantiles
# #------------

# # set up blank dictionaries for all metrics
# out_dict_reg, out_dict_class, out_dict_quant = output_dict(), output_dict(), output_dict()

# # run all other models
# model_list = ['adp', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'huber', 'bridge', 'gbmh', 'xgb', 'knn', 'gbm', 'rf']
# for i, m in enumerate(model_list):
#     out_dict_reg, _, _ = get_model_output(m, df_train, 'reg', out_dict_reg, pos, set_pos, i, min_samples)
# save_output_dict(out_dict_reg, model_output_path, 'reg')

# # run all other models
# model_list = ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c','lgbm_c', 'knn_c', 'lr_c']
# for i, m in enumerate(model_list):
#     out_dict_class, _, _= get_model_output(m, df_train_class, 'class', out_dict_class, pos, set_pos, i, min_samples)
# save_output_dict(out_dict_class, model_output_path, 'class')

# # run all other models
# for m in ['qr_q', 'gbm_q', 'rf_q', 'lgbm_q', 'knn_q']:
#     for alph in [0.8, 0.95]:
#         out_dict_quant, _, _ = get_model_output(m, df_train, 'quantile', out_dict_quant, pos, set_pos, i, alpha=alph)
# save_output_dict(out_dict_quant, model_output_path, 'quant')

# #%%
# #------------
# # Run the Stacking Models and Generate Output
# #------------

# # get the training data for stacking and prediction data after stacking
# X_stack, y_stack, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)
# _, X_predict = get_stack_predict_data(df_train, df_predict, df_train_class, df_predict_class, 
#                                    models_reg, models_class, models_quant)

# X_predict = X_predict.dropna(axis=1)

# # create the stacking models
# final_models = ['rf', 'gbm', 'gbmh', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 'bridge']
# stack_val_pred = pd.DataFrame(); scores = []; best_models = []
# for i, fm in enumerate(final_models):
#     best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models, scores, stack_val_pred, show_plots=True)

# # get the best stack predictions and average
# predictions = mf.stack_predictions(X_predict, best_models, final_models)
# best_val, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, show_plot=True, min_include=2)

# # create the output and add standard devations / max scores
# output = mf.create_output(output_start, best_predictions)
# output = val_std_dev(model_output_path, output, best_val, iso_spline='iso', show_plot=True)
# output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]

# #%%

# skm, _, _ = get_skm(df_train, 'reg', to_drop=['player', 'team', 'pos'])
# skm.print_coef(models_reg['reg_ridge'][0])

# #%%
# # save out final results
# val_compare = validation_compare_df(model_output_path, best_val)
# save_out_results(val_compare, 'Simulation', 'Model_Validations', pos, set_year, set_pos, current_or_next_year)
# save_out_results(output, 'Simulation', 'Model_Predictions', pos, set_year, set_pos, current_or_next_year)


# #%%

# set_pos = 'QB'
# version = 'beta'

# from sklearn.metrics import mean_squared_error

# rp = dm.read(f'''SELECT player, 
#                         season, 
#                         SUM(pred_fp_per_game) rp_pred, 
#                         SUM(y_act) rp_y_act
#                 FROM Model_Validations
#                 WHERE rush_pass != 'both'
#                       AND pos = '{set_pos}'
#                       AND year_exp='{pos[set_pos]['year_exp']}'
#                       AND filter_data = '{pos[set_pos]['filter_data']}'
#                       AND current_or_next_year = '{current_or_next_year}'
#                       AND year = '{set_year}'
#                       AND version = '{vers}'
#                 GROUP BY player, season
#              ''', 'Simulation')

# both = dm.read(f'''SELECT player, 
#                          season, 
#                          pred_fp_per_game both_pred, 
#                          y_act both_y_act
#                 FROM Model_Validations
#                 WHERE rush_pass = 'both'
#                       AND pos = '{set_pos}'
#                       AND year_exp='{pos[set_pos]['year_exp']}'
#                       AND filter_data = '{pos[set_pos]['filter_data']}'
#                       AND current_or_next_year = '{current_or_next_year}'
#                       AND year = '{set_year}'
#                     AND version = '{vers}'
#                 ''', 'Simulation')



# # rp = rp[rp.rp_pred < 22].reset_index(drop=True)
# rp = pd.merge(rp, both, on=['player', 'season'])

# mf.show_scatter_plot(rp.rp_pred, rp.rp_y_act, r2=True)
# mf.show_scatter_plot(rp.both_pred, rp.both_y_act, r2=True)

# print('\nRP MSE:', np.sqrt(mean_squared_error(rp.rp_pred, rp.rp_y_act)))
# print('Both MSE:', np.sqrt(mean_squared_error(rp.both_pred, rp.both_y_act)))
# print(rp[abs(rp.both_y_act - rp.rp_y_act) > 0.001])


# #%%

# rp = dm.read(f'''SELECT player, 
#                         year, 
#                         SUM(pred_fp_per_game) rp_pred,
#                         AVG(avg_pick) avg_pick,
#                         SUM(std_dev)/1.4 std_dev,
#                         SUM(max_score)/1.3 max_score
#                 FROM Model_Predictions
#                 WHERE rush_pass != 'both'
#                       AND pos = '{set_pos}'
#                       AND year_exp='{pos[set_pos]['year_exp']}'
#                       AND filter_data = '{pos[set_pos]['filter_data']}'
#                       AND current_or_next_year = '{current_or_next_year}'
#                       AND year = '{set_year}'
#                       AND version = '{vers}'
#                 GROUP BY player, year
#              ''', 'Simulation').sort_values(by='rp_pred', ascending=False)
# rp.iloc[:50]
# # %%
