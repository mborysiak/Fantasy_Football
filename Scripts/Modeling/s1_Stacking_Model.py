#%%
# core packages
from random import random
from contextlib import nullcontext
import builtins
import warnings

def install_warning_filters(include_pandas=False):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=PendingDeprecationWarning)
    warnings.filterwarnings(
        'ignore',
        message=r'.*CatBoostRegressor.*__sklearn_tags__.*',
        category=DeprecationWarning,
    )
    warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*deprecated.*', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    if include_pandas:
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


install_warning_filters()

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import sys
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
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

install_warning_filters(include_pandas=True)

import zHelper_Functions as hf
pos = hf.pos

pd.set_option('display.max_columns', 999)

from sklearn import set_config
set_config(display='diagram')


def set_optuna_logging(level=None):
    level_name = (level or os.environ.get('FF_OPTUNA_LOG_LEVEL', 'WARNING')).upper()
    optuna.logging.set_verbosity(getattr(optuna.logging, level_name, optuna.logging.WARNING))


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def install_model_print_filter():
    original_print = getattr(builtins.print, '_ff_original_print', builtins.print)

    def filtered_print(*args, **kwargs):
        if (
            _env_int('FF_SUPPRESS_SAMPLING_TREES', 1)
            and len(args) == 1
            and str(args[0]).strip() == 'Sampling trees'
        ):
            return None

        if _env_int('FF_SUPPRESS_SKMODEL_OUTPUT', _env_int('FF_SUPPRESS_SKMODEL_CV', 1)):
            msg = ' '.join(str(arg) for arg in args).strip()
            suppress_exact = {
                '---',
                '-----------------',
                'Overall\n==============',
                'ADP Score\n--------',
                'Stack Score\n--------',
            }
            suppress_prefixes = (
                'ADP MSE:',
                'ADP R2:',
                'ADP Sera',
                'Val MSE:',
                'Val R2:',
                'Val Sera',
                'Test MSE:',
                'Test R2:',
                'Test Sera',
                'Test MC:',
                'Test Brier:',
                'Test Pinball Loss:',
                'Feature Importances',
                'Iter ',
            )

            if msg in suppress_exact or msg.startswith(suppress_prefixes):
                return None

        return original_print(*args, **kwargs)

    filtered_print._ff_original_print = original_print
    builtins.print = filtered_print


class _LineFilteringStdout:
    def __init__(self, stream, blocked_lines):
        self.stream = stream
        self.blocked_lines = set(blocked_lines)
        self.buffer = ''

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip() not in self.blocked_lines:
                self.stream.write(line + '\n')
        return len(text)

    def flush(self):
        if self.buffer:
            if self.buffer.strip() not in self.blocked_lines:
                self.stream.write(self.buffer)
            self.buffer = ''
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


set_optuna_logging()
install_model_print_filter()

JOBLIB_BACKEND = os.environ.get('FF_JOBLIB_BACKEND', 'threading')
MAX_PARALLEL_JOBS = _env_int('FF_MAX_PARALLEL_JOBS', 16)
INNER_THREADS = _env_int('FF_INNER_THREADS', 1)


def _resolve_n_jobs(n_jobs):
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    max_parallel_jobs = _env_int('FF_MAX_PARALLEL_JOBS', MAX_PARALLEL_JOBS)
    return max(1, min(int(n_jobs), max_parallel_jobs))


def run_parallel(delayed_calls, n_jobs, verbose=0):
    install_warning_filters(include_pandas=True)
    backend = os.environ.get('FF_JOBLIB_BACKEND', JOBLIB_BACKEND)
    inner_threads = _env_int('FF_INNER_THREADS', INNER_THREADS)
    joblib_verbose = _env_int('FF_JOBLIB_VERBOSE', verbose)
    parallel = Parallel(
        n_jobs=_resolve_n_jobs(n_jobs),
        verbose=joblib_verbose,
        backend=backend,
    )
    ctx = threadpool_limits(limits=inner_threads) if inner_threads > 0 else nullcontext()
    original_stdout = sys.stdout
    if _env_int('FF_SUPPRESS_SAMPLING_TREES', 1):
        sys.stdout = _LineFilteringStdout(original_stdout, {'Sampling trees'})

    try:
        with ctx:
            return parallel(delayed_calls)
    finally:
        if sys.stdout is not original_stdout:
            sys.stdout.flush()
            sys.stdout = original_stdout

#==========
# General Setting
#==========

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'RB'

# set year to analyze
set_year = 2025

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
pos['QB']['rush_pass'] = ''
pos['RB']['rush_pass'] = ''
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
pos['WR']['filter_data'] = 'greater_equal'
pos['TE']['filter_data'] = 'greater_equal'

pos['QB']['year_exp'] = 0
pos['RB']['year_exp'] = 4
pos['WR']['year_exp'] = 0
pos['TE']['year_exp'] = 0

pos['QB']['iters'] = 20
pos['RB']['iters'] = 20
pos['WR']['iters'] = 20
pos['TE']['iters'] = 20

pos['QB']['n_splits'] = 5
pos['RB']['n_splits'] = 5
pos['WR']['n_splits'] = 5
pos['TE']['n_splits'] = 5



def create_pkey(pos, dataset, set_pos, set_year, cur_next, bayes_rand, hp_algo):

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

    df = df.sort_values(by=['year', 'avg_pick'], ascending=[True, True]).reset_index(drop=True)
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

    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict

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

    if m=='adp': params['feature_select__cols'] = hp.choice('cols', [['avg_pick'], ['avg_pick', 'year'], ['avg_pick', 'year', 'avg_proj_points', 'avg_proj_points_exp_diff']])

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

    print(f'Running {model_name}{alpha}')

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

    # load the quantile predictions
    pred_quant, actual_quant, models_quant, _, full_hold_quant = mf.load_all_pickles(model_output_path, 'quantile')
    X_stack_quant, _ = mf.X_y_stack('quantile', full_hold_quant, pred_quant, actual_quant)

    # concat all the predictions together
    X_stack = pd.concat([X_stack, X_stack_quant], axis=1)
    X_stack_player = full_hold_reg['reg_adp'][['player', 'year']].reset_index(drop=True)

    return X_stack_player, X_stack, y_stack, models_reg, models_quant


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

    print(f'Running {final_m}')

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

    m.fit(X, y)

    X_predict = df_predict[X.columns]
    if proba: cur_predict = m.predict_proba(X_predict)[:,1]
    else: cur_predict = m.predict(X_predict)

    return cur_predict

def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        predictions = run_parallel(
            (delayed(fit_and_predict)(m, df_predict, X, y, proba) for m in ind_models),
            n_jobs=8,
            verbose=0,
        )
        predictions = [p for p in predictions if len(p) > 0]
        predictions = pd.Series(pd.DataFrame(predictions).T.mean(axis=1), name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def get_stack_predict_data(df_train, df_train_quant, df_predict, models_reg, models_quant):

    _, X, y = get_skm(df_train, 'reg')
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)

    print('Predicting Quant Models')
    _, X, y = get_skm(df_train_quant, 'quantile')
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


def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):

    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):

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


RESID_ALPHAS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)


def eligible_empirical_resid_donors(
    validation_rows,
    forecast_origin,
    origin_col='season',
    outcome_horizon=0,
    pred_col='pred_fp_per_game',
    actual_col='y_act',
):
    """Return residual rows whose outcomes predate a forecast origin.

    ``current`` projection rows have a zero-season outcome horizon, while
    ``next`` rows have a one-season horizon. The latter therefore require an
    extra gap: an origin-t next-year outcome is not known when an origin-t+1
    preseason forecast is made.
    """
    required = [origin_col, pred_col, actual_col]
    missing = [col for col in required if col not in validation_rows.columns]
    if missing:
        raise ValueError(
            f'Empirical residual donors are missing columns: {missing}'
        )

    donors = validation_rows.dropna(subset=required).copy()
    donors['resid_target_season'] = (
        pd.to_numeric(donors[origin_col]) + int(outcome_horizon)
    )
    donors = donors[
        donors.resid_target_season.lt(int(forecast_origin))
    ].copy()
    return donors


def create_final_val_df(X_stack_player, y_stack, best_val_reg, all_alphas=None):
    df_val_final = pd.concat([X_stack_player[['player', 'year']],
                              pd.Series(best_val_reg.mean(axis=1), name='pred_fp_per_game')], axis=1)

    if all_alphas is not None:
        for alph, pred in all_alphas['val'].items():
            df_val_final[f'pred_resid_{int(round(alph*100))}'] = pred.mean(axis=1)

    df_val_final = pd.concat([df_val_final, y_stack], axis=1)
    return df_val_final


def apply_empirical_resid_quantiles(
    df_val_stack,
    output,
    alphas=RESID_ALPHAS,
    pred_col='pred_fp_per_game',
    actual_col='y_act',
    n_bins=None,
    min_n=75,
    min_bins=2,
    max_bins=None,
    smooth=True,
    bootstrap_iters=50,
    bootstrap_frac=1.0,
    bootstrap_replace=True,
    random_state=42,
):
    val = df_val_stack[[pred_col, actual_col]].dropna().copy()
    val['actual_resid'] = val[actual_col] - val[pred_col]

    if n_bins is None:
        n_bins = max(min_bins, int(len(val) // min_n))
        if max_bins is not None:
            n_bins = min(n_bins, max_bins)
    n_bins = min(n_bins, len(val))

    q_cols = [f'pred_resid_{int(round(alpha * 100))}' for alpha in alphas]

    def fit_bucket_table(cur_val):
        _, edges = pd.qcut(cur_val[pred_col], n_bins, retbins=True, duplicates='drop')
        inner_edges = np.unique(edges)[1:-1]

        cur_val = cur_val.copy()
        cur_val['_resid_bucket'] = np.searchsorted(inner_edges, cur_val[pred_col].to_numpy(), side='right')
        bucket_grp = cur_val.groupby('_resid_bucket')
        bucket_table = bucket_grp['actual_resid'].quantile(list(alphas)).unstack()
        bucket_table.columns = q_cols
        bucket_table['resid_bucket_n'] = bucket_grp['actual_resid'].size()
        bucket_table['resid_bucket_mean_pred'] = bucket_grp[pred_col].mean()

        global_q = cur_val['actual_resid'].quantile(list(alphas))
        small_buckets = bucket_table['resid_bucket_n'] < min_n
        for alpha, col in zip(alphas, q_cols):
            bucket_table.loc[small_buckets, col] = global_q.loc[alpha]

        bucket_table[q_cols] = np.maximum.accumulate(bucket_table[q_cols].to_numpy(), axis=1)
        return bucket_table, inner_edges

    def predict_from_bucket_table(bucket_table, inner_edges, pred_values):
        if smooth and len(bucket_table) > 1:
            interp_table = (
                bucket_table
                .sort_values('resid_bucket_mean_pred')
                .drop_duplicates('resid_bucket_mean_pred')
            )
            if len(interp_table) == 1:
                return np.repeat(interp_table[q_cols].to_numpy(), len(pred_values), axis=0)

            x = interp_table['resid_bucket_mean_pred'].to_numpy()
            return np.column_stack([
                np.interp(pred_values, x, interp_table[col].to_numpy())
                for col in q_cols
            ])

        bucket_idx = np.searchsorted(inner_edges, pred_values, side='right')
        pred_q = pd.DataFrame({'_resid_bucket': bucket_idx}).join(bucket_table[q_cols], on='_resid_bucket')
        return pred_q[q_cols].to_numpy()

    bucket_table, inner_edges = fit_bucket_table(val)
    pred_values = output[pred_col].to_numpy()
    n_iter = max(1, int(bootstrap_iters))
    pred_q = np.zeros((len(output), len(q_cols)))
    rng = np.random.default_rng(random_state)

    for i in range(n_iter):
        if i == 0:
            cur_val = val
        else:
            sample_size = max(min_n, int(round(len(val) * bootstrap_frac)))
            if not bootstrap_replace:
                sample_size = min(sample_size, len(val))
            sample_idx = rng.choice(len(val), size=sample_size, replace=bootstrap_replace)
            cur_val = val.iloc[sample_idx]

        cur_table, cur_edges = fit_bucket_table(cur_val)
        pred_q += predict_from_bucket_table(cur_table, cur_edges, pred_values)

    pred_q = pred_q / n_iter
    pred_q = np.maximum.accumulate(pred_q, axis=1)

    out = output.copy()
    replace_cols = q_cols + [f'q{int(round(alpha * 100))}_fp_per_game' for alpha in alphas]
    replace_cols += ['resid_bucket_n', 'resid_bucket_mean_pred']
    out = out.drop(columns=[c for c in replace_cols if c in out.columns])
    out[q_cols] = pred_q

    out_bucket_idx = np.searchsorted(inner_edges, pred_values, side='right')
    out['_resid_bucket'] = out_bucket_idx
    out = out.join(bucket_table[['resid_bucket_n', 'resid_bucket_mean_pred']], on='_resid_bucket')

    out = out.drop(columns=['_resid_bucket', 'resid_bucket_n', 'resid_bucket_mean_pred'])
    bucket_table['smooth'] = smooth
    bucket_table['bootstrap_iters'] = n_iter

    return out, bucket_table.reset_index()


def cross_fit_empirical_resid_quantiles(
    validation_rows,
    origin_col='season',
    outcome_horizon=0,
    as_of_year=None,
    alphas=RESID_ALPHAS,
    pred_col='pred_fp_per_game',
    actual_col='y_act',
    min_training_rows=30,
    n_bins=None,
    min_n=75,
    min_bins=2,
    max_bins=None,
    smooth=True,
    bootstrap_iters=50,
    bootstrap_frac=1.0,
    bootstrap_replace=True,
    random_state=42,
):
    """Attach strict-prior-origin empirical residual quantiles to OOS rows.

    This helper is intended for one homogeneous validation slice, such as one
    position/dataset/experience/current-next model emitted by ``s2_RunAll``.
    Point predictions are already out of sample. For every target origin, its
    residual distribution is fitted only from validation outcomes known before
    that target origin, so the target row and its contemporaries cannot
    calibrate their own intervals. A one-season horizon therefore creates a
    one-season embargo between donor outcomes and the target origin.

    Rows without enough earlier donors retain null residual quantiles and
    explicit unavailable provenance rather than borrowing future outcomes.
    """
    output = validation_rows.copy()
    q_cols = [f'pred_resid_{int(round(alpha * 100))}' for alpha in alphas]
    for col in q_cols:
        output[col] = np.nan
    output['resid_training_rows'] = 0
    output['resid_training_through_origin'] = np.nan
    output['resid_training_through_season'] = np.nan
    output['resid_calibration_mode'] = 'strict_prior_season_empirical_ppg_bucket'
    output['resid_calibration_available'] = 0
    output['resid_calibration_date_modified'] = (
        pd.Timestamp.now().strftime('%m-%d-%Y %H:%M')
    )

    required = [origin_col, pred_col, actual_col]
    missing = [col for col in required if col not in output.columns]
    if missing:
        raise ValueError(
            f'Cross-fitted residual calibration is missing columns: {missing}'
        )

    output['resid_target_season'] = (
        pd.to_numeric(output[origin_col]) + int(outcome_horizon)
    )
    output['resid_target_available'] = output[actual_col].notna().astype(int)
    if as_of_year is not None:
        output['resid_target_available'] = (
            output.resid_target_available.eq(1)
            & output.resid_target_season.lt(int(as_of_year))
        ).astype(int)

    valid_donors = output[
        output.resid_target_available.eq(1)
    ].dropna(subset=required).copy()
    valid_targets = output.dropna(subset=[origin_col, pred_col]).copy()
    audit_rows = []
    for target_origin in sorted(valid_targets[origin_col].unique()):
        target_idx = valid_targets.index[
            valid_targets[origin_col].eq(target_origin)
        ]
        donors = valid_donors[
            valid_donors.resid_target_season.lt(target_origin)
        ].copy()
        donor_count = int(len(donors))
        donor_origin_through = (
            int(donors[origin_col].max()) if donor_count else np.nan
        )
        donor_season_through = (
            int(donors.resid_target_season.max()) if donor_count else np.nan
        )
        available = donor_count >= int(min_training_rows)

        output.loc[target_idx, 'resid_training_rows'] = donor_count
        output.loc[
            target_idx,
            'resid_training_through_origin',
        ] = donor_origin_through
        output.loc[
            target_idx,
            'resid_training_through_season',
        ] = donor_season_through

        if available:
            calibrated, _ = apply_empirical_resid_quantiles(
                donors,
                output.loc[target_idx].copy(),
                alphas=alphas,
                pred_col=pred_col,
                actual_col=actual_col,
                n_bins=n_bins,
                min_n=min_n,
                min_bins=min_bins,
                max_bins=max_bins,
                smooth=smooth,
                bootstrap_iters=bootstrap_iters,
                bootstrap_frac=bootstrap_frac,
                bootstrap_replace=bootstrap_replace,
                random_state=int(random_state) + int(target_origin) * 1009,
            )
            output.loc[target_idx, q_cols] = calibrated[q_cols].to_numpy()
            output.loc[target_idx, 'resid_calibration_available'] = 1

        audit_rows.append({
            'target_origin': int(target_origin),
            'target_season': int(target_origin) + int(outcome_horizon),
            'target_rows': int(len(target_idx)),
            'target_actual_rows': int(
                output.loc[target_idx, 'resid_target_available'].sum()
            ),
            'resid_training_rows': donor_count,
            'resid_training_through_origin': donor_origin_through,
            'resid_training_through_season': donor_season_through,
            'resid_calibration_available': int(available),
        })

    output['resid_training_rows'] = (
        output.resid_training_rows.fillna(0).astype(int)
    )
    output['resid_calibration_available'] = (
        output.resid_calibration_available.fillna(0).astype(int)
    )

    calibrated = output.resid_calibration_available.eq(1)
    if calibrated.any():
        quantiles = output.loc[calibrated, q_cols].to_numpy(dtype=float)
        if not np.isfinite(quantiles).all():
            raise ValueError('A calibrated validation residual quantile is non-finite.')
        if not (np.diff(quantiles, axis=1) >= -1e-10).all():
            raise ValueError('Validation residual quantiles are not monotone.')
        invalid_cutoff = (
            output.loc[calibrated, 'resid_training_through_season']
            >= output.loc[calibrated, origin_col]
        )
        if invalid_cutoff.any():
            raise ValueError(
                'A validation residual calibration used its target or a future season.'
            )

    return output, pd.DataFrame(audit_rows)


def create_output(output_start, predictions, predictions_upside=None, predictions_top=None, predictions_quantile=None):

    output = output_start.copy()
    output['pred_fp_per_game'] = predictions.mean(axis=1)

    if predictions_upside is not None:
        output['pred_fp_per_game_upside'] = predictions_upside.mean(axis=1)

    if predictions_top is not None:
        output['pred_fp_per_game_top'] = predictions_top.mean(axis=1)

    if predictions_quantile is not None:
        for alph, pred in predictions_quantile.items():
            output[f'pred_resid_{int(alph*100)}'] = pred.mean(axis=1)

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
# optuna_timeout = 20

# model_output_path, pkey = create_pkey(pos, dataset, set_pos,set_year,current_or_next_year, bayes_rand, hp_algo)
# df = pull_data(set_pos, set_year, dataset, current_or_next_year)

# obj_cols = list(df.dtypes[df.dtypes=='object'].index)
# obj_cols = [c for c in obj_cols if c not in ['player', 'team', 'pos']]
# df= df.drop(obj_cols, axis=1)

# df, output_start = filter_df(df, pos, set_pos, set_year)
# df_train, df_predict = get_train_predict(df, set_year, pos[set_pos]['rush_pass'])
# df_train_residual = df_train.copy()
# df_train_residual['y_act'] = df_train_residual.y_act - df_train_residual.avg_proj_points_per_game

# #%%

# # set up blank dictionaries for all metrics
# out_dict_reg, out_dict_quant = output_dict(), output_dict()

# model_list = ['adp', 'lasso', 'lgbm', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'knn', 'ridge', 'bridge', 'enet']
# results = run_parallel(
#     (delayed(get_model_output)
#         (m, df_train, 'reg', out_dict_reg, pos, set_pos, hp_algo, bayes_rand, i, optuna_timeout=optuna_timeout)
#         for i, m in enumerate(model_list)),
#     n_jobs=8,
#     verbose=1,
# )

# out_dict_reg = extract_par_results(results, out_dict_reg)
# save_output_dict(out_dict_reg, model_output_path, 'reg')

# model_list = ['qr_q','lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']
# models_q = [[alph, m] for alph in [0.75, 0.9] for m in model_list]
# results = run_parallel(
#     (delayed(get_model_output)
#         (m[1], df_train_residual, 'quantile', out_dict_quant, pos, set_pos, hp_algo, bayes_rand, i, alpha=m[0], optuna_timeout=optuna_timeout)
#         for i, m in enumerate(models_q)),
#     n_jobs=8,
#     verbose=1,
# )

# out_dict_quant = extract_par_results(results, out_dict_quant)
# save_output_dict(out_dict_quant, model_output_path, 'quantile')

# #%%

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
#     'optuna_timeout': 30,
# }

# # get the training data for stacking and prediction data after stacking
# X_stack_player, X_stack, y_stack, models_reg, models_quant = load_all_stack_pred(model_output_path)
# _, X_predict = get_stack_predict_data(df_train, df_train_residual, df_predict, models_reg, models_quant)

# #---------------
# # Regression
# #---------------
# final_models = ['bridge', 'enet', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'lgbm', 'knn', 'ridge', 'lasso', 'xgb']
# stack_val_pred = pd.DataFrame(); scores = []; best_models = []

# results = run_parallel(
#                 (delayed(run_stack_models)
#                 (fm, X_stack, y_stack, i, 'reg', None, run_params) \
#                 for i, fm in enumerate(final_models)),
#                 n_jobs=8,
#                 verbose=1,
#                 )

# best_models, scores, stack_val_pred = unpack_stack_results(results)

# # get the best stack predictions and average
# predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
# best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 'reg', show_plot=True, min_include=3)

# #%%
# # #---------------
# # # Create Output
# # #---------------

# output = create_output(output_start, best_predictions)
# df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg)
# output, resid_bucket_table = apply_empirical_resid_quantiles(
#     df_val_stack,
#     output,
#     alphas=RESID_ALPHAS,
#     n_bins=None,
#     min_n=50,
#     smooth=True,
#     bootstrap_iters=50,
#     bootstrap_replace=True,
# )
# output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]

# #%%
# # # save out final results
# val_compare = validation_compare_df(model_output_path, best_val_reg)
# save_out_results(val_compare, 'Validations', 'Model_Validations_Resid', vers, pos, set_year, set_pos, dataset, current_or_next_year)
# save_out_results(output, 'Simulation', 'Model_Predictions_Resid', vers, pos, set_year, set_pos, dataset, current_or_next_year)


# # %%
