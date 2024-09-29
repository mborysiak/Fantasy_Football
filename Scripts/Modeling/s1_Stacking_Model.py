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
from zFix_Standard_Dev import *
import zModel_Functions as mf
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from hyperopt import Trials
from hyperopt import hp
import optuna

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
set_pos = 'WR'

# set year to analyze
set_year = 2024

# set the version
vers = 'beta'

# set with this year or next
current_or_next_year = 'next'

mse_wt = 1
sera_wt = 0
r2_wt = 0
brier_wt = 1
matt_wt = 0

# determine whether to do run/pass/rec separate or together
pos['QB']['rush_pass'] = 'pass'
pos['RB']['rush_pass'] = 'rec'
pos['WR']['rush_pass'] = ''
pos['TE']['rush_pass'] = ''
pos['Rookie_RB']['rush_pass'] = ''
pos['Rookie_WR']['rush_pass'] = ''

#==========
# Model Settings
#==========

pos['QB']['val_start'] = 2017
pos['RB']['val_start'] = 2017
pos['WR']['val_start'] = 2017
pos['TE']['val_start'] = 2017

pos['QB']['test_years'] = 1
pos['RB']['test_years'] = 1
pos['WR']['test_years'] = 1
pos['TE']['test_years'] = 1

pos['QB']['filter_data'] = 'greater_equal'
pos['RB']['filter_data'] = 'greater_equal'
pos['WR']['filter_data'] = 'less_equal'
pos['TE']['filter_data'] = 'greater_equal'

pos['QB']['year_exp'] = 0
pos['RB']['year_exp'] = 0
pos['WR']['year_exp'] = 3
pos['TE']['year_exp'] = 0

pos['QB']['iters'] = 20
pos['RB']['iters'] = 20
pos['WR']['iters'] = 20
pos['TE']['iters'] = 20

pos['QB']['n_splits'] = 5
pos['RB']['n_splits'] = 5
pos['WR']['n_splits'] = 5
pos['TE']['n_splits'] = 5



def create_pkey(pos, dataset, set_pos, cur_next, bayes_rand, hp_algo):

    all_vars = ['val_start', 'test_years', 'filter_data', 'year_exp', 'iters', 'rush_pass']

    pkey = str(set_pos)
    pkey = pkey + '_' + dataset
    pkey = pkey + '_' + bayes_rand
    pkey = pkey + '_' + hp_algo
    for var in all_vars:
        v = str(pos[set_pos][var])
        pkey = pkey + '_' + v

    pkey = pkey + '_' + str(cur_next)

    model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    
    pkey = '_'.join(model_output_path.split('/')[-2:])[:-1]

    return model_output_path, pkey

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def update_output_dict(label, m, suffix, out_dict, oof_data, best_models):

    # append all of the metric outputs
    lbl = f'{label}_{m}{suffix}'
    out_dict['pred'][lbl] = oof_data['hold']
    out_dict['actual'][lbl] = oof_data['actual']
    out_dict['scores'][lbl] = oof_data['scores']
    out_dict['models'][lbl] = best_models
    out_dict['full_hold'][lbl] = oof_data['full_hold']

    return lbl, out_dict

def save_output_dict(out_dict, model_output_path, label):

    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')

#==========
# Pull and clean compiled data
#==========

def pull_data(set_pos, set_year, dataset, current_or_next_year):

    if current_or_next_year=='next': lbl = '_next'
    else: lbl = ''

    # load data and filter down
    df = dm.read(f'''SELECT * FROM {set_pos}_{set_year}_{dataset}''', f'Model_Inputs{lbl}')
    if df.shape[1]==2000:
        df = pd.concat([df, dm.read(f"SELECT * FROM {set_pos}_{set_year}_{dataset}_V2 ", f'Model_Inputs{lbl}')], axis=1)
    if dataset=='Rookie': df = df.assign(year_exp=0, team='team', pos=set_pos)

    # add in data to match up with Daily code
    df['week'] = 1
    df['game_date'] = df.year

    df = df.sort_values(by='year', ascending=True).reset_index(drop=True)
    try: df = df.drop(['season', 'games', 'games_next'], axis=1)
    except: df = df.drop(['games', 'games_next'], axis=1)

    return df


def filter_df(df, pos, set_pos, set_year): 

    # # filter dataset
    if pos[set_pos]['filter_data']=='greater_equal':
        df = df.loc[df.year_exp >= pos[set_pos]['year_exp']].reset_index(drop=True)

    elif pos[set_pos]['filter_data']=='less_equal':
        df = df.loc[df.year_exp <= pos[set_pos]['year_exp']].reset_index(drop=True)

    output_start = df.loc[df.year==set_year, ['player', 'team', 'pos', 'avg_pick', 'year', 'year_exp']].copy()
    output_start = output_start[['player', 'avg_pick']].reset_index(drop=True)

    return df, output_start


#==============
# Create Datasets
#==============

def get_train_predict(df, set_year, rush_pass):

    if rush_pass in ('rush', 'pass', 'rec'):
        rush_pass = f'_{rush_pass}'
        df = df.drop('y_act', axis=1).rename(columns={f'y_act{rush_pass}': 'y_act'})

    df_train = df.loc[df.year < set_year, :].reset_index(drop=True).drop([y for y in df.columns if 'y_act_' in y], axis=1)
    df_predict = df.loc[df.year == set_year, :].reset_index(drop=True).drop([y for y in df.columns if 'y_act_' in y], axis=1)

    df_train_upside = df.loc[df.year < set_year, :].copy()
    df_train_upside = df_train_upside.drop(['y_act'], axis=1).rename(columns={f'y_act_class_upside{rush_pass}': 'y_act'})
    df_train_upside = df_train_upside.drop([y for y in df_train_upside.columns if 'y_act_' in y], axis=1)
    
    df_predict_upside = df.loc[df.year == set_year, :].copy()
    df_predict_upside = df_predict_upside.drop(['y_act'], axis=1).rename(columns={f'y_act_class_upside{rush_pass}': 'y_act'})
    df_predict_upside = df_predict_upside.drop([y for y in df_train_upside.columns if 'y_act_' in y], axis=1)


    df_train_top = df.loc[df.year < set_year, :].copy()
    df_train_top = df_train_top.drop(['y_act'], axis=1).rename(columns={f'y_act_class_top{rush_pass}': 'y_act'})
    df_train_top = df_train_top.drop([y for y in df_train_top.columns if 'y_act_' in y], axis=1)
    
    df_predict_top = df.loc[df.year == set_year, :].copy()
    df_predict_top = df_predict_top.drop(['y_act'], axis=1).rename(columns={f'y_act_class_top{rush_pass}': 'y_act'})
    df_predict_top = df_predict_top.drop([y for y in df_train_top.columns if 'y_act_' in y], axis=1)

    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top

#=================
# Model Functions
#=================

def output_dict():
    return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

def get_skm(skm_df, model_obj, hp_algo='atpe'):
    
    to_drop = list(skm_df.dtypes[skm_df.dtypes=='object'].index)
    skm = SciKitModel(skm_df, model_obj=model_obj, sera_wt=sera_wt, mse_wt=mse_wt, 
                      r2_wt=r2_wt, brier_wt=brier_wt, matt_wt=matt_wt, hp_algo=hp_algo)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, bayes_rand, alpha=None, stack_model=False, min_samples=10):

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
    params = skm.default_params(pipe, bayes_rand, min_samples=min_samples)

    if m=='adp': params['feature_select__cols'] = hp.choice('cols', [['avg_pick'], ['avg_pick', 'year']]) 
    
    if skm.model_obj == 'quantile':
        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        elif m == 'cb_q': pipe.set_params(**{f'{m}__loss_function': f'Quantile:alpha={alpha}'})
        else: pipe.set_params(**{f'{m}__alpha': alpha})

    return pipe, params


def get_full_pipe_stack(skm, m, bayes_rand,  alpha=None, stack_model=False, min_samples=10):

    if skm.model_obj=='class': 
        kb = 'k_best_c'
        sp = 'select_perc_c'
    else: 
        kb = 'k_best'
        sp = 'select_perc'

    stack_models = {

        'full_stack': skm.model_pipe([
                                      skm.piece('std_scale'), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(kb),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'random_full_stack': skm.model_pipe([
                                      skm.piece('random_sample'),
                                      skm.piece('std_scale'), 
                                    #   skm.piece('select_perc'),
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(f'{kb}_fu'),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'kbest': skm.model_pipe([
                                 skm.piece('std_scale'),
                                 skm.piece(kb),
                                 skm.piece(m)
                                 ]),

        'random' : skm.model_pipe([
                                    skm.piece('random_sample'),
                                    skm.piece('std_scale'),
                                    skm.piece(m)
                                    ]),

        'random_kbest': skm.model_pipe([
                                        skm.piece('random_sample'),
                                        skm.piece('std_scale'),
                                        skm.piece(kb),
                                        skm.piece(m)
                                        ])
    }

    pipe = stack_models[stack_model]
    params = skm.default_params(pipe, bayes_rand=bayes_rand, min_samples=min_samples)
    
    if skm.model_obj == 'quantile':
        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        elif m == 'cb_q': pipe.set_params(**{f'{m}__loss_function': f'Quantile:alpha={alpha}'})
        else: pipe.set_params(**{f'{m}__alpha': alpha})

    if stack_model=='random_full_stack' and bayes_rand=='optuna':
        params['random_sample__frac'] = ['real', 0.2, 1]
        params['feature_union__agglomeration__n_clusters'] = ['int', 3, 15]
        params['feature_union__pca__n_components'] = ['int', 3, 15]
        params[f'feature_union__{kb}_fu__k'] = ['int', 3, 50]
        params[f'{kb}__k'] = ['int', 5, 50]

    return pipe, params




# def rename_existing(study_db, study_name):
#     import datetime as dt
#     new_study_name = study_name + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
#     optuna.copy_study(from_study_name=study_name, from_storage=study_db, to_storage=study_db, to_study_name=new_study_name)
#     optuna.delete_study(study_name=study_name, storage=study_db)


# def get_new_study(db, old_name, new_name, num_trials):

#     storage = optuna.storages.RDBStorage(
#                                 url=db,
#                                 engine_kwargs={"pool_size": 64, 
#                                             "connect_args": {"timeout": 10},
#                                             },
#                                 )
    
#     if old_name is not None:
#         old_study = optuna.create_study(
#             study_name=old_name,
#             storage=storage,
#             load_if_exists=True
#         )
    
#     try:
#         next_study = optuna.create_study(
#             study_name=new_name, 
#             storage=storage, 
#             load_if_exists=False
#         )

#     except:
#         rename_existing(storage, new_name)
#         next_study = optuna.create_study(
#             study_name=new_name, 
#             storage=storage, 
#             load_if_exists=False
#         )
    
#     if old_name is not None and len(old_study.trials) > 0:
#         print(f"Loaded {new_name} study with {old_name} {len(old_study.trials)} trials")
#         next_study.add_trials(old_study.trials[-num_trials:])

#     return next_study
    
# def get_optuna_study(pkey, model_name, model_obj, alpha):
#     old_name = f"{final_m}_{model_obj}{alpha}_{pkey}"
#     new_name =f"{final_m}_{model_obj}{alpha}_{pkey}"
#     next_study = get_new_study(run_params['study_db'], old_name, new_name, run_params['num_recent_trials'])
#     return next_study

def get_new_study(model_name, model_obj, pos):

    storage = optuna.storages.RDBStorage(
                                url=f"sqlite:///optuna/weekly_train_{pos}_{model_name}_{model_obj}_{int(10000*random())}.sqlite3",
                                engine_kwargs={"pool_size": 64, 
                                            "connect_args": {"timeout": 10},
                                            },
                                )

    study = optuna.create_study(
            storage=storage,
        )

    return study


def get_model_output(model_name, cur_df, model_obj, out_dict, pos, set_pos, hp_algo, bayes_rand, i, alpha='', optuna_timeout=60):

    print(f'\n{model_name}\n============\n')

    skm, X, y = get_skm(cur_df, model_obj, hp_algo)
    pipe, params = get_full_pipe(skm, model_name, bayes_rand, alpha)

    if model_obj == 'class': proba = True 
    else: proba = False

    if bayes_rand == 'bayes': trials = Trials()
    elif bayes_rand == 'optuna': 
        trials = get_new_study(model_name, model_obj, set_pos)

    # fit and append the ADP model
    best_models, oof_data, _, _ = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'], 
                                                     n_splits=pos[set_pos]['n_splits'], alpha=alpha,
                                                     col_split='year', time_split=pos[set_pos]['val_start'],
                                                     bayes_rand=bayes_rand, proba=proba, trials=trials,
                                                     random_seed=(i+7)*19+(i*12)+6, optuna_timeout=optuna_timeout)
    lbl, out_dict = update_output_dict(model_obj, model_name, str(alpha), out_dict, oof_data, best_models)

    return out_dict#, best_models, oof_data


def extract_par_results(results, out_dict_reg):
    for k in out_dict_reg.keys():
        for r in results:
            model_lbl = list(r['pred'].keys())[0]
            out_dict_reg[k][model_lbl] = r[k][model_lbl]
    return out_dict_reg


#====================
# Stacking Functions
#====================

def load_all_stack_pred(model_output_path):

    # load the regregression predictions
    pred, actual, models_reg, _, full_hold_reg = mf.load_all_pickles(model_output_path, 'reg')
    X_stack, y_stack = mf.X_y_stack('reg', full_hold_reg, pred, actual)

    # load the class predictions
    pred_top, actual_top, models_top, _, full_hold_top = mf.load_all_pickles(model_output_path, 'class_top')
    X_stack_top, y_stack_top = mf.X_y_stack('top', full_hold_top, pred_top, actual_top)

    pred_upside, actual_upside, models_upside, _, full_hold_upside = mf.load_all_pickles(model_output_path, 'class_upside')
    X_stack_upside, y_stack_upside = mf.X_y_stack('upside', full_hold_upside, pred_upside, actual_upside)

    # load the quantile predictions
    pred_quant, actual_quant, models_quant, _, full_hold_quant = mf.load_all_pickles(model_output_path, 'quantile')
    X_stack_quant, _ = mf.X_y_stack('quantile', full_hold_quant, pred_quant, actual_quant)

    # concat all the predictions together
    X_stack = pd.concat([X_stack, X_stack_upside, X_stack_top, X_stack_quant], axis=1)
    X_stack_player = full_hold_reg['reg_adp'][['player', 'year']].reset_index(drop=True)

    return X_stack_player, X_stack, y_stack, y_stack_upside, y_stack_top, models_reg, models_upside, models_top, models_quant


def get_proba_adp_coef(model_obj, final_m, run_params):
    if model_obj == 'class': proba = True
    else: proba = False

    if model_obj in ('class', 'quantile'): run_adp = False
    else: run_adp = True

    if 'gbmh' in final_m or 'knn' in final_m  or 'mlp' in final_m or \
        'cb' in final_m or 'full_stack' in run_params['stack_model']: print_coef = False
    else: print_coef = run_params['print_coef']

    return proba, run_adp, print_coef


def run_stack_models(final_m, X_stack, y_stack, i, model_obj, alpha, run_params):

    print(f'\n{final_m}')

    min_samples = int(len(y_stack)/10)
    proba, run_adp, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, hp_algo=run_params['hp_algo'])
    pipe, params = get_full_pipe_stack(skm, final_m, stack_model=run_params['stack_model'],
                                       bayes_rand=run_params['opt_type'], alpha=alpha, 
                                       min_samples=min_samples, )
    
    if run_params['opt_type'] == 'bayes': trials = Trials()
    elif run_params['opt_type'] == 'optuna': 
        trials = get_new_study(final_m, model_obj, '')
    
    best_model, stack_scores, stack_pred, _ = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                n_iter=run_params['n_iter'], alpha=alpha,
                                                                bayes_rand=run_params['opt_type'],trials=trials,
                                                                run_adp=run_adp, print_coef=print_coef,
                                                                proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                random_state=(i*2)+(i*7), optuna_timeout=run_params['optuna_timeout'])
    stack_val_pred =  pd.Series(stack_pred['stack_pred'], name=final_m)

    return best_model, stack_scores['stack_score'], stack_val_pred
    

def fit_and_predict(m, df_predict, X, y, proba):
    # try:
    m.fit(X,y)

    if proba: cur_predict = m.predict_proba(df_predict[X.columns])[:,1]
    else: cur_predict = m.predict(df_predict[X.columns])
    # except:
        # cur_predict = []

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

def get_stack_predict_data(df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top,
                          models_reg, models_upside, models_top, models_quant):

    _, X, y = get_skm(df_train, 'reg')
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)

    print('Predicting Upside Models')
    _, X, y = get_skm(df_train_upside, 'class')
    X_predict_upside = create_stack_predict(df_predict_upside, models_upside, X, y, proba=True)
    X_predict = pd.concat([X_predict, X_predict_upside], axis=1)

    print('Predicting Top Models')
    _, X, y = get_skm(df_train_top, 'class')
    X_predict_top = create_stack_predict(df_predict_top, models_top, X, y, proba=True)
    X_predict = pd.concat([X_predict, X_predict_top], axis=1)

    print('Predicting Quant Models')
    _, X, y = get_skm(df_train, 'quantile')
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    X_predict_player = pd.concat([df_predict[['player', 'team', 'week', 'year']], X_predict], axis=1)

    return X_predict_player, X_predict


def show_calibration_curve(y_true, y_pred, n_bins=10):

    from sklearn.calibration import calibration_curve

    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    plt.plot(y, x, marker = '.', label = 'Quantile')

    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    plt.plot(y, x, marker = '+', label = 'Uniform')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()

    print('Brier Score:', brier_score_loss(y_true, y_pred))


def val_std_dev(output, val_data, metrics={'pred_fp_per_game': 1}, iso_spline='iso', show_plot=True, max_grps_den=0.08, min_grps_den=0.12):
        
    sd_max_met = StandardScaler().fit(val_data[list(metrics.keys())]).transform(output[list(metrics.keys())])
    sd_max_met = np.mean(sd_max_met, axis=1)

    if iso_spline=='iso':
        sd_m, max_m, min_m = get_std_splines(val_data, metrics, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.12), 
                                            max_grps_den=int(val_data.shape[0]*0.08),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m.predict(sd_max_met)
        output['max_score'] = max_m.predict(sd_max_met)
        output['min_score'] = min_m.predict(sd_max_met)

    elif iso_spline=='spline':
        sd_m, max_m, min_m = get_std_splines(val_data, metrics, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*min_grps_den), 
                                            max_grps_den=int(val_data.shape[0]*max_grps_den),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m(sd_max_met)
        output['max_score'] = max_m(sd_max_met)
        output['min_score'] = min_m(sd_max_met)
 
    return output

def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):

        start_cols = bm.steps[0][1].start_columns
        X_predict = X_predict[start_cols]
        
        if model_obj in ('reg', 'quantile'): cur_prediction = np.round(bm.predict(X_predict), 2)
        elif model_obj=='class': cur_prediction = np.round(bm.predict_proba(X_predict)[:,1], 3)
        
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions

def best_average_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, model_obj, min_include = 3):
    
    skm, _, _ = get_skm(df_train, model_obj=model_obj)
    
    n_scores = []
    models_included = []
    for i in range(len(scores)-min_include+1):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:i+min_include]
        models_included.append(top_n)
        model_idx = np.array(final_models)[top_n]
        
        n_score = skm.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        n_scores.append(n_score)
        
    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores)
    best_score = n_scores[best_n]
    top_models = models_included[best_n]

    model_idx = np.array(final_models)[top_models]

    print('Top Models:', model_idx)
    best_val = stack_val_pred[model_idx]
    best_predictions = predictions[model_idx]

    return best_val, best_predictions, best_score

def average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, model_obj, show_plot=True, min_include=3):
    
    best_val, best_predictions, best_score = best_average_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 
                                                                 model_obj=model_obj, min_include=min_include)
    
    if show_plot:
        mf.show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
        if model_obj == 'class':
            show_calibration_curve(y_stack, best_val.mean(axis=1), n_bins=8)

    
    return best_val, best_predictions, best_score

def unpack_stack_results(results):
    best_models = []
    scores = []
    stack_val = pd.DataFrame()
    for r in results:
        best_models.append(r[0])
        scores.append(r[1])
        stack_val = pd.concat([stack_val, r[2]], axis=1)
    return best_models, scores, stack_val


def create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_upside, best_val_top, best_val_quant):
    df_val_final = pd.concat([X_stack_player[['player', 'year']], 
                              pd.Series(best_val_reg.mean(axis=1), name='pred_fp_per_game'),
                              pd.Series(best_val_upside.mean(axis=1), name='pred_fp_per_game_upside'),
                              pd.Series(best_val_top.mean(axis=1), name='pred_fp_per_game_top'),
                              pd.Series(best_val_quant.mean(axis=1), name='pred_fp_per_game_quantile'),
                              y_stack], axis=1)
    # df_val_final = pd.merge(df_val_final, y_stack, on=['player', 'team', 'week', 'year'])
    return df_val_final


def create_output(output_start, predictions, predictions_upside=None, predictions_top=None, predictions_quantile=None):

    output = output_start.copy()
    output['pred_fp_per_game'] = predictions.mean(axis=1)

    if predictions_upside is not None: 
        output['pred_fp_per_game_upside'] = predictions_upside.mean(axis=1)
    
    if predictions_top is not None: 
        output['pred_fp_per_game_top'] = predictions_top.mean(axis=1)

    if predictions_quantile is not None:
        output['pred_fp_per_game_quantile'] = predictions_quantile.mean(axis=1)

    output = output.sort_values(by='avg_pick', ascending=True)
    output['adp_rank'] = range(len(output))
    output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

    return output


#====================
# Outputs
#====================

def validation_compare_df(model_output_path, best_val):

    _, _, _, _, oof_data = mf.load_all_pickles(model_output_path, 'reg')
    oof_data = oof_data['reg_adp'][['player', 'team', 'year', 'y_act']].reset_index(drop=True)
    best_val = pd.Series(best_val.mean(axis=1), name='pred_fp_per_game')
    val_compare = pd.concat([oof_data, best_val], axis=1).rename(columns={'year': 'season'})
    
    return val_compare

def save_out_results(df, db_name, table_name, vers, pos, set_year, set_pos, dataset, current_or_next_year):

    import datetime as dt

    df['pos'] = set_pos
    df['rush_pass'] = pos[set_pos]['rush_pass']
    df['dataset'] = dataset
    df['filter_data'] = pos[set_pos]['filter_data']
    df['year_exp'] = pos[set_pos]['year_exp']
    df['current_or_next_year'] = current_or_next_year
    df['version'] = vers
    df['year'] = set_year

    df['date_modified'] = dt.datetime.now().strftime('%m-%d-%Y %H:%M')

    del_str = f'''pos='{set_pos}' 
                  AND rush_pass='{pos[set_pos]['rush_pass']}'
                  AND dataset='{dataset}'
                  AND filter_data='{pos[set_pos]['filter_data']}'
                  AND year_exp={pos[set_pos]['year_exp']}
                  AND current_or_next_year = '{current_or_next_year}'
                  AND version = '{vers}'
                  AND year={set_year}'''

    dm.delete_from_db(db_name, table_name, del_str)
    dm.write_to_db(df, db_name, table_name, if_exist='append')


#%%

# #------------
# # Pull in the data and create train and predict sets
# #------------

# dataset = 'ProjOnly'
# hp_algo = 'tpe'
# bayes_rand = 'optuna'
# optuna_timeout = 30

# model_output_path, pkey = create_pkey(pos, dataset, set_pos,current_or_next_year, bayes_rand, hp_algo)
# df = pull_data(set_pos, set_year, dataset, current_or_next_year)

# obj_cols = list(df.dtypes[df.dtypes=='object'].index)
# obj_cols = [c for c in obj_cols if c not in ['player', 'team', 'pos']]
# df= df.drop(obj_cols, axis=1)

# df, output_start = filter_df(df, pos, set_pos, set_year)
# df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top = get_train_predict(df, set_year, pos[set_pos]['rush_pass'])

#%%
#------------
# Run the Regression, Classification, and Quantiles
#------------

# # set up blank dictionaries for all metrics
# out_dict_reg, out_dict_class, out_dict_quant = output_dict(), output_dict(), output_dict()

# # run all other models
# model_list = ['adp', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 'bridge', 'enet']
# model_list = ['lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', 'lgbm_c', 'knn_c',  ]
# model_list = ['qr_q', 'gbm_q', 'gbmh_q', 'lgbm_q', 'rf_q', 'cb_q']
# model_list = ['cb_q']
# for i, m in enumerate(model_list):
#     out_dict_class = get_model_output(m, df_train_top, 'class', out_dict_class, pos, set_pos, hp_algo, bayes_rand, i, '_top')

#%%

# # set up blank dictionaries for all metrics
# out_dict_reg, out_dict_top, out_dict_upside, out_dict_quant = output_dict(), output_dict(), output_dict(), output_dict()

# model_list = ['adp', 'lasso', 'lgbm', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'knn', 'ridge', 'bridge', 'enet']
# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(get_model_output)
#                 (m, df_train, 'reg', out_dict_reg, pos, set_pos, hp_algo, bayes_rand, i) \
#                 for i, m in enumerate(model_list) 
#                 )

# out_dict_reg = extract_par_results(results, out_dict_reg)
# save_output_dict(out_dict_reg, model_output_path, 'reg')

# # run all other models
# model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(get_model_output)
#                 (m, df_train_top, 'class', out_dict_top, pos, set_pos, hp_algo, bayes_rand, i, '_top') \
#                 for i, m in enumerate(model_list) 
#                 )
# out_dict_top = extract_par_results(results, out_dict_top)
# save_output_dict(out_dict_top, model_output_path, 'class_top')

# # run all other models
# model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(get_model_output)
#                 (m, df_train_upside, 'class', out_dict_upside, pos, set_pos, hp_algo, bayes_rand, i, '_upside') \
#                 for i, m in enumerate(model_list) 
#                 )
# out_dict_upside = extract_par_results(results, out_dict_upside)
# save_output_dict(out_dict_upside, model_output_path, 'class_upside')

# # run all other models
# model_list = ['qr_q','lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']
# models_q = [[alph, m] for alph in [0.6, 0.8] for m in model_list]
# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(get_model_output)
#                 (m[1], df_train, 'quantile', out_dict_quant, pos, set_pos, hp_algo, bayes_rand, i, alpha=m[0]) \
#                 for i, m in enumerate(models_q) 
#                 )

# out_dict_quant = extract_par_results(results, out_dict_quant)
# save_output_dict(out_dict_quant, model_output_path, 'quantile')

#%%

# #------------
# # Run the Stacking Models and Generate Output
# #------------
# run_params = {
#     'stack_model': 'random_full_stack',
#     'print_coef': False,
#     'opt_type': 'optuna',
#     'hp_algo': 'tpe',
#     'num_k_folds': 3,
#     'n_iter': 50,
#     'optuna_timeout': 60,

#     'sd_metrics': {'pred_fp_per_game': 1, 'pred_fp_per_game_class': 1, 'pred_fp_per_game_quantile': 0.5}
# }

# # get the training data for stacking and prediction data after stacking
# X_stack_player, X_stack, y_stack, y_stack_upside, y_stack_top, \
#  models_reg, models_upside, models_top, models_quant = load_all_stack_pred(model_output_path)

# _, X_predict = get_stack_predict_data(df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top,
#                                       models_reg, models_upside, models_top, models_quant)

# #---------------
# # Regression
# #---------------
# final_models = ['bridge', 'enet', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'lgbm', 'knn', 'ridge', 'lasso', 'xgb']
# stack_val_pred = pd.DataFrame(); scores = []; best_models = []

# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(run_stack_models)
#                 (fm, X_stack, y_stack, i, 'reg', None, run_params) \
#                 for i, fm in enumerate(final_models) 
#                 )

# best_models, scores, stack_val_pred = unpack_stack_results(results)

# # get the best stack predictions and average
# predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
# best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 'reg', show_plot=True, min_include=3)

# #---------------
# # Classification Top
# #---------------
# final_models_top = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', ]
# stack_val_top = pd.DataFrame(); scores_top = []; best_models_top = []
# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(run_stack_models)
#                 (fm, X_stack, y_stack_top, i, 'class', None, run_params) \
#                 for i, fm in enumerate(final_models_top) 
#                 )

# best_models_top, scores_top, stack_val_top = unpack_stack_results(results)

# # get the best stack predictions and average
# predictions_top = stack_predictions(X_predict, best_models_top, final_models_top, 'class')
# best_val_top, best_predictions_top, _ = average_stack_models(df_train, scores_top, final_models_top, y_stack_top, 
#                                                              stack_val_top, predictions_top, 'class', show_plot=True, min_include=2)

# #---------------
# # Classification Upside
# #---------------
# final_models_upside = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', ]
# stack_val_upside = pd.DataFrame(); scores_upside = []; best_models_upside = []
# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(run_stack_models)
#                 (fm, X_stack, y_stack_upside, i, 'class', None, run_params) \
#                 for i, fm in enumerate(final_models_upside) 
#                 )

# best_models_upside, scores_upside, stack_val_upside = unpack_stack_results(results)

# # get the best stack predictions and average
# predictions_upside = stack_predictions(X_predict, best_models_upside, final_models_upside, 'class')
# best_val_upside, best_predictions_upside, _ = average_stack_models(df_train, scores_upside, final_models_upside, y_stack_upside, 
#                                                                   stack_val_upside, predictions_upside, 'class', show_plot=True, min_include=2)


# #------------
# # Quantile
# #---------------

# final_models_quant = ['qr_q', 'gbm_q', 'gbmh_q', 'rf_q', 'lgbm_q', 'cb_q']
# stack_val_quant = pd.DataFrame(); scores_quant = []; best_models_quant = []

# results = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(run_stack_models)
#                 (fm, X_stack, y_stack, i, 'quantile', 0.8, run_params) \
#                 for i, fm in enumerate(final_models_quant) 
# )

# best_models_quant, scores_quant, stack_val_quant = unpack_stack_results(results)

# # get the best stack predictions and average
# predictions_quant = stack_predictions(X_predict, best_models_quant, final_models_quant, 'quantile')
# best_val_quant, best_predictions_quant, _ = average_stack_models(df_train, scores_quant, final_models_quant, y_stack, stack_val_quant, predictions_quant, 'quantile', show_plot=True, min_include=2)

# #---------------
# # Create Output
# #---------------
# #%%
# if X_stack.shape[0] < 200: iso_spline = 'iso'
# else: iso_spline = 'spline'
# output = create_output(output_start, best_predictions, best_predictions_upside, best_predictions_top, best_predictions_quant)
# df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_upside, best_val_top, best_val_quant)
# output = val_std_dev(output, df_val_stack, metrics=run_params['sd_metrics'], iso_spline=iso_spline, show_plot=True, max_grps_den=0.04, min_grps_den=0.08)
# output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]

# output.loc[output.std_dev < 1, 'std_dev'] = output.loc[output.std_dev < 1, 'pred_fp_per_game'] * 0.15
# output.loc[output.max_score < (output.pred_fp_per_game + output.std_dev), 'max_score'] = output.pred_fp_per_game + output.std_dev*1.5
# y_max = df_train.y_act.max()
# output.loc[output.max_score > y_max, 'max_score'] = y_max + output.std_dev / 3
# output = output.round(3)

# # save out final results
# val_compare = validation_compare_df(model_output_path, best_val_reg)
# save_out_results(val_compare, 'Validations', 'Model_Validations', pos, set_year, set_pos, dataset, current_or_next_year)
# save_out_results(output, 'Simulation', 'Model_Predictions', pos, set_year, set_pos, dataset, current_or_next_year)

