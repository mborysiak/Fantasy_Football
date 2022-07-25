#%%
# core packages
from random import random
import pandas as pd
import numpy as np
import os
import gzip
import pickle

from ff.db_operations import DataManage
from ff import general
from skmodel import SciKitModel

import pandas_bokeh
pandas_bokeh.output_notebook()
import matplotlib.pyplot as plt

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
set_pos = 'RB'

# set year to analyze
set_year = 2022

#==========
# Fantasy Point Values
#==========

# define point values for all statistical categories
pass_yd_per_pt = 0.04 
pass_td_pt = 4
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

pos['QB']['req_touch'] = 250
pos['RB']['req_touch'] = 60
pos['WR']['req_touch'] = 50
pos['TE']['req_touch'] = 40

pos['QB']['req_games'] = 8
pos['RB']['req_games'] = 8
pos['WR']['req_games'] = 8
pos['TE']['req_games'] = 8

pos['QB']['earliest_year'] = 2004 # first year QBR available
pos['RB']['earliest_year'] = 1998
pos['WR']['earliest_year'] = 1998
pos['TE']['earliest_year'] = 1998

pos['QB']['val_start'] = 2012
pos['RB']['val_start'] = 2012
pos['WR']['val_start'] = 2012
pos['TE']['val_start'] = 2012

pos['QB']['features'] = 'v2'
pos['RB']['features'] = 'v2'
pos['WR']['features'] = 'v2'
pos['TE']['features'] = 'v2'

pos['QB']['test_years'] = 1
pos['RB']['test_years'] = 1
pos['WR']['test_years'] = 1
pos['TE']['test_years'] = 1

pos['QB']['use_ay'] = False
pos['RB']['use_ay'] = False
pos['WR']['use_ay'] = False
pos['TE']['use_ay'] = False

pos['QB']['filter_data'] = 'greater_equal'
pos['RB']['filter_data'] = 'greater_equal'
pos['WR']['filter_data'] = 'less_equal'
pos['TE']['filter_data'] = 'greater_equal'

pos['QB']['year_exp'] = 0
pos['RB']['year_exp'] = 0
pos['WR']['year_exp'] = 3
pos['TE']['year_exp'] = 0

pos['QB']['act_ppg'] = 20
pos['RB']['act_ppg'] = 17
pos['WR']['act_ppg'] = 15
pos['TE']['act_ppg'] = 12

pos['QB']['pct_off'] = 0
pos['RB']['pct_off'] = 0
pos['WR']['pct_off'] = 0
pos['TE']['pct_off'] = 0

pos['QB']['iters'] = 25
pos['RB']['iters'] = 25
pos['WR']['iters'] = 25
pos['TE']['iters'] = 25

pos['QB']['all_stats'] = False
pos['RB']['all_stats'] = False
pos['WR']['all_stats'] = False
pos['TE']['all_stats'] = False

all_vars = ['req_touch', 'req_games', 'earliest_year', 'val_start', 
            'features', 'test_years', 'use_ay', 'filter_data', 'year_exp', 
            'act_ppg', 'pct_off', 'iters', 'all_stats']

pkey = str(set_pos)
db_output = {'set_pos': set_pos, 'set_year': set_year}
for var in all_vars:
    v = str(pos[set_pos][var])
    pkey = pkey + '_' + v
    db_output[var] = pos[set_pos][var]
db_output['pkey'] = pkey

model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
if not os.path.exists(model_output_path): os.makedirs(model_output_path)


def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def filter_less_equal(df, pos):
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

def save_output_dict(out_dict, label):

    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')

#==========
# Pull and clean compiled data
#==========

# load data and filter down
df = dm.read(f'''SELECT * FROM {set_pos}_{set_year}''', 'Model_Inputs')

# apply the specified touches and game filters
df, df_train_results, df_test_results = hf.touch_game_filter(df, pos, set_pos, set_year)

# calculate FP for a given set of scoring values
df = hf.calculate_fp(df, pts_dict, pos=set_pos).reset_index(drop=True)
df['rules_change'] = np.where(df.year>=2010, 1, 0)

# # filter dataset
if pos[set_pos]['filter_data']=='greater_equal':
    df = df.loc[df.year_exp >= pos[set_pos]['year_exp']].reset_index(drop=True)

elif pos[set_pos]['filter_data']=='less_equal':
    df = df.loc[df.year_exp <= pos[set_pos]['year_exp']+1].reset_index(drop=True)

# get the train and prediction dataframes for FP per game
_, output_start = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                 set_year-pos[set_pos]['test_years'], 
                                 pos[set_pos]['earliest_year'], pos[set_pos]['features'])

output_start = filter_less_equal(output_start, pos)
output_start = output_start[['player', 'avg_pick']]

# append fp_per_game to the metrics and ensure unique values
pos[set_pos]['metrics'].append('fp_per_game')
pos[set_pos]['metrics'] = list(dict.fromkeys(pos[set_pos]['metrics']))

# append fp_per_game to the metrics and ensure unique values
pts_dict[set_pos].append(1)
# pts_dict[set_pos] = list(dict.fromkeys(pts_dict[set_pos]))

#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
adp_ppg_high = '<100'
adp_ppg_low = '>=0'
act_ppg = '>=' + str(pos[set_pos]['act_ppg'])
pct_off = '>=' +  str(pos[set_pos]['pct_off'])

#==============
# Create Break-out Probability Features
#==============

# add in data to match up with Daily code
df['week'] = 1
df['game_date'] = df.year

# get the train and prediction dataframes for FP per game
df_train_class, df_predict_class = hf.get_train_predict(df, breakout_metric, pos, set_pos, 
                                                      set_year-pos[set_pos]['test_years'], 
                                                      pos[set_pos]['earliest_year'], 
                                                      pos[set_pos]['features'])

# get the adp predictions and merge back to the training dataframe
df_train_class, df_predict_class, lr = hf.get_adp_predictions(df_train_class, df_predict_class, 1)
df_predict_class = filter_less_equal(df_predict_class, pos)

# filter to adp cutoffs
df_train_class = hf.adp_filter(df_train_class, adp_ppg_low, adp_ppg_high)
df_predict_class = hf.adp_filter(df_predict_class, adp_ppg_low, adp_ppg_high)

# generate labels for prediction based on cutoffs
df_train_class = hf.class_label(df_train_class, pct_off, act_ppg)
df_predict_class = hf.class_label(df_predict_class, pct_off, act_ppg)

df_predict_class = df_predict_class.drop('y_act', axis=1).fillna(0)

# get the minimum number of training samples for the initial datasets
min_samples = int(df_train_class[df_train_class.year <= df_train_class.year.min()].shape[0])

# print the value-counts
print('Training Value Counts:', df_train_class.y_act.value_counts()[0], '|', df_train_class.y_act.value_counts()[1])
print(f'Number of Features: {df_train_class.shape[1]}')
print('Min Train Year:', df_train_class.year.min())
print('Max Train Year:', df_train_class.year.max())
print('Min Val Year:', pos[set_pos]['val_start'])
print('Max Val Year:', df_train_class.year.max())
print('Min Test Year:', df_predict_class.year.min())
print('Max Test Year:', df_predict_class.year.max())

#%%

# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

if pos[set_pos]['all_stats']:
    mets = pos[set_pos]['metrics']
else:
    mets = pos[set_pos]['metrics'][-1:]

# set up blank dictionaries for all metrics
out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}
for met in mets:

    print(f'\nRunning Metric {met}\n=========================\n')
    print('ADP only\n============\n')
    
    # get the train and prediction dataframes for FP per game
    df_train, df_predict = hf.get_train_predict(df, met, pos, set_pos, 
                                                set_year-pos[set_pos]['test_years'], 
                                                pos[set_pos]['earliest_year'], 
                                                pos[set_pos]['features'])
    df_predict = filter_less_equal(df_predict, pos)

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])  
    
    print('Shape of Train Set', df_train.shape)
    skm = SciKitModel(df_train)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

    # set up the ADP model pipe
    pipe = skm.model_pipe([skm.piece('feature_select'), 
                           skm.piece('std_scale'), 
                           skm.piece('lr')])
    params = skm.default_params(pipe)
    params['feature_select__cols'] = [['avg_pick'], ['avg_pick', 'year']]

    # fit and append the ADP model
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'],
                                                  col_split='year', time_split=pos[set_pos]['val_start'],
                                                  bayes_rand='custom_rand')

    # append all of the metric outputs
    out_dict = update_output_dict('reg', 'adp', '', out_dict, oof_data, best_models)

    #---------------
    # Model Training loop
    #---------------

    # loop through each potential model
    model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf']
    for i, m in enumerate(model_list):

        print('\n============\n')
        print(m)

        # set up the model pipe and get the default search parameters
        pipe = skm.model_pipe([skm.piece('random_sample'),
                               skm.piece('std_scale'), 
                               #skm.piece('select_perc'),
                               skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                                skm.piece('pca')
                                                ]),
                               skm.piece('k_best'),
                               skm.piece(m)])
        params = skm.default_params(pipe, 'rand')
        if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

        # run the model with parameter search
        best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'],
                                                       col_split='year', time_split=pos[set_pos]['val_start'],
                                                       bayes_rand='custom_rand', random_seed=(i+7)*19+(i*12)+6)
        # append all of the metric outputs
        out_dict = update_output_dict('reg', m, '', out_dict, oof_data, best_models)
    save_output_dict(out_dict, 'reg')

#%%

# set up blank dictionaries for all metrics
out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

skm_class = SciKitModel(df_train_class, model_obj='class')
X_class, y_class = skm_class.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

# loop through each potential model
model_list = [
    'lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c'
]
for i, m in enumerate(model_list):

    print('\n============\n')
    print(m)

     # set up the model pipe and get the default search parameters
    pipe = skm_class.model_pipe([skm_class.piece('random_sample'),
                                 skm_class.piece('std_scale'), 
                                 #skm_class.piece('select_perc_c'),
                                 skm_class.feature_union([
                                                skm_class.piece('agglomeration'), 
                                                skm_class.piece('k_best_c'),
                                                ]),
                                 skm_class.piece('k_best_c'),
                                 skm_class.piece(m)])
    
    params = skm_class.default_params(pipe, 'rand')
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)

    # run the model with parameter search
    best_models, oof_data, _ = skm_class.time_series_cv(pipe, X_class, y_class, 
                                                        params, n_iter=pos[set_pos]['iters'],
                                                        col_split='year', time_split=pos[set_pos]['val_start'],
                                                        bayes_rand='custom_rand', proba=True, random_seed=(4*i)+(7*i)+i)

# append all of the metric outputs
    out_dict = update_output_dict('class', m, '', out_dict, oof_data, best_models)
save_output_dict(out_dict, 'class')

#%%

# set up blank dictionaries for all metrics
out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

for alpha in [0.8, 0.95]:

    print(f"\n--------------\nAlpha {alpha}\n--------------\n")

    skm_quantile = SciKitModel(df_train, model_obj='quantile')
    X_quant, y_quant = skm_quantile.Xy_split(y_metric='y_act',  to_drop=['player', 'team', 'pos'])

    # loop through each potential model
    model_list = ['gbm_q']
    for m in model_list:

        print('\n============\n')
        print(m)

        # set up the model pipe and get the default search parameters
        pipe = skm_quantile.model_pipe([
                                        skm_quantile.piece('random_sample'),
                                        skm_quantile.piece(m)
                                        ])
        
        # set params
        pipe.steps[-1][-1].alpha = alpha
        params = skm_quantile.default_params(pipe, 'rand')
        params['random_sample__frac'] = np.arange(0.02, 0.1, 0.01)

        # run the model with parameter search
        best_models, oof_data, param_scores = skm_quantile.time_series_cv(pipe, X_quant, y_quant, 
                                                                            params, n_iter=50,
                                                                            bayes_rand='custom_rand',
                                                                            col_split='year',
                                                                            time_split=pos[set_pos]['val_start'])

        # append the results and the best models for each fold
        out_dict = update_output_dict('quant', m, f'_{alpha}', out_dict, oof_data, best_models)

save_output_dict(out_dict, 'quant')

# %%
def load_all_pickles(model_output_path, label):
    pred = load_pickle(model_output_path, f'{label}_pred')
    actual = load_pickle(model_output_path, f'{label}_actual')
    models = load_pickle(model_output_path, f'{label}_models')
    scores = load_pickle(model_output_path, f'{label}_scores')
    try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    except: full_hold = None
    return pred, actual, models, scores, full_hold

def get_class_predictions(models_class, df_train_class, df_predict_class):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_class = pd.DataFrame()

    skm_class_final = SciKitModel(df_train_class, model_obj='class')
    X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])
    
    for k, v in models_class.items():
        m = skm_class_final.ensemble_pipe(v)
        m.fit(X_class_final, y_class_final)
        cur_pred = pd.Series(m.predict_proba(df_predict_class[X_class_final.columns])[:,1], name=k)
        X_predict_class = pd.concat([X_predict_class, cur_pred], axis=1)

    return X_predict_class


def X_y_stack_old(met, pred, actual):

    X = pd.DataFrame([v for k,v in pred.items() if met in k]).T
    X.columns = [k for k,_ in pred.items() if met in k]
    y = pd.Series(actual[X.columns[0]], name='y_act')

    return X, y


def X_y_stack_full(met, full_hold):
    i = 0
    for k, v in full_hold.items():
        if i == 0:
            df = v.copy()
            df = df.rename(columns={'pred': k})
        else:
            df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
            df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
        i+=1

    X = df[[c for c in df.columns if met in c or 'y_act_' in c]].reset_index(drop=True)
    y = df['y_act'].reset_index(drop=True)
    return X, y, df


def X_y_stack(met, full_hold, pred, actual):
    if full_hold is not None:
        X_stack, y_stack, _ = X_y_stack_full(met, full_hold)
    else:
        X_stack, y_stack = X_y_stack(met, pred, actual)

    return X_stack, y_stack


def get_quant_predictions(df_train, df_predict, models_quant):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_quant = pd.DataFrame()
    for alpha in [0.8, 0.95]:

        print(f"\n--------------\nAlpha {alpha}\n--------------\n")

        skm_quant = SciKitModel(df_train, model_obj='quantile')
        X_quant_final, y_quant_final = skm_quant.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])
        
        for k, v in models_quant.items():
            if str(alpha) in k:
                m = skm_quant.ensemble_pipe(v)
                m.fit(X_quant_final, y_quant_final)
                cur_pred = m.predict(df_predict[X_quant_final.columns])
                cur_pred = pd.Series(cur_pred, name=k)
                X_predict_quant = pd.concat([X_predict_quant, cur_pred], axis=1)
    
    return X_predict_quant


def get_reg_predict_features(models, X, y):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        
        predictions = pd.DataFrame()

        for m in ind_models:

            m.fit(X, y)
            reg_predict = m.predict(df_predict[X.columns])
            predictions = pd.concat([predictions, pd.Series(reg_predict)], axis=1)
            
        predictions = predictions.mean(axis=1)
        predictions = pd.Series(predictions, name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def stack_predictions(X_predict, best_models, final_models):
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        cur_prediction = np.round(bm.predict(X_predict), 2)
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions

def best_average_models(scores, final_models, stack_val_pred, predictions):

    n_scores = []
    for i in range(len(scores)):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:i+1]
        model_idx = np.array(final_models)[top_n]
        
        n_score = skm_stack.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        n_scores.append(n_score)

    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores[1:])
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:best_n+2]
    
    model_idx = np.array(final_models)[top_n]
    best_val = stack_val_pred[model_idx]

    best_predictions = predictions[model_idx]

    return best_val, best_predictions


def show_scatter_plot(y_pred, y, label='Total', r2=True):
    plt.scatter(y_pred, y)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()

    from sklearn.metrics import r2_score
    if r2: print(f'{label} R2:', r2_score(y, y_pred))
    else: print(f'{label} Corr:', np.corrcoef(y, y_pred)[0][1])


def top_predictions(y_pred, y, r2=False):

    val_high_score = pd.concat([pd.Series(y_pred), pd.Series(y)], axis=1)
    val_high_score.columns = ['predictions','y_act']
    val_high_score = val_high_score[val_high_score.predictions >= \
                                    np.percentile(val_high_score.predictions, 75)]
    show_scatter_plot(val_high_score.predictions, val_high_score.y_act, label='Top', r2=r2)

#------------
# Make the Class Predictions
#------------

# load the class predictions
pred_class, actual_class, models_class, _, full_hold_class = load_all_pickles(model_output_path, 'class')
X_predict_class = get_class_predictions(models_class, df_train_class, df_predict_class)
X_stack_class, y_stack_class = X_y_stack('class', full_hold_class, pred_class, actual_class)

#------------
# Make the Reg Predictions
#------------

# get the X and y values for stack trainin for the current metric
pred, actual, models, _, full_hold_reg = load_all_pickles(model_output_path, 'reg')
X_stack, y_stack = X_y_stack('reg', full_hold_reg, pred, actual)
X_stack = pd.concat([X_stack, X_stack_class], axis=1)

#------------
# Make the Quantile Predictions
#------------
try:
    pred_quant, actual_quant, models_quant, scores_quant, full_hold_quant = load_all_pickles(model_output_path, 'quant')
    X_predict_quant = get_quant_predictions(df_train, df_predict, models_quant)
    X_stack_quant, _ = X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)
    X_stack = pd.concat([X_stack, X_stack_quant], axis=1)
except:
    print('No Quantile Data Available')
    X_stack_quant = None
    X_predict_quant = None

best_models = []
final_models = [
            'ridge',
            'lasso',
            'lgbm', 
            'xgb', 
            'rf', 
            'bridge',
            'gbm'
            ]
skm_stack = SciKitModel(df_train, model_obj='reg')
stack_val_pred = pd.DataFrame()
scores = []
for i, final_m in enumerate(final_models):

    print(f'\n{final_m}')

    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm_stack.model_pipe([
                            skm_stack.piece('k_best'), 
                            skm_stack.piece(final_m)
                        ])

    stack_params = skm_stack.default_params(stack_pipe)
    stack_params['k_best__k'] = range(1, X_stack.shape[1])

    best_model, stack_scores, stack_pred = skm_stack.best_stack(stack_pipe, stack_params,
                                                                X_stack, y_stack, n_iter=50, 
                                                                run_adp=True, print_coef=True,
                                                                sample_weight=False, random_state=(i*12)+(i*17))
    best_models.append(best_model)
    scores.append(stack_scores['stack_score'])
    stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)

    show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)
    top_predictions(stack_pred['stack_pred'], stack_pred['y'], r2=True)

# get the final output:
X_full, y_full = skm_stack.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

X_predict = get_reg_predict_features(models, X_full, y_full)
X_predict = pd.concat([X_predict, X_predict_class], axis=1)
try: X_predict = pd.concat([X_predict, X_predict_quant], axis=1)
except: print('No Quantile Data Available')
predictions = stack_predictions(X_predict, best_models, final_models)


best_val, best_predictions = best_average_models(scores, final_models, stack_val_pred, predictions)
output = output_start.copy()
output['pred_fp_per_game'] = predictions.mean(axis=1)
output = output.sort_values(by='avg_pick', ascending=True)
output['adp_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output

# %%

from sklearn.pipeline import Pipeline
met='fp_per_game'
# get the train and prediction dataframes for FP per game
df_train, _ = hf.get_train_predict(df, met, pos, set_pos, 
                                            set_year-pos[set_pos]['test_years'], 
                                            pos[set_pos]['earliest_year'], 
                                            pos[set_pos]['features'])

skm = SciKitModel(df_train, model_obj='reg')
X_all, y = skm.Xy_split('y_act', ['player', 'team', 'pos'])

# X_all['min_teammate_minus_ap'] = X_all.min_teammate - X_all.avg_pick
X = X_all.sample(frac=1, axis=1, random_state=123456)
X['year'] = X_all.year
X['game_date'] = X_all.game_date
# X['min_teammate_minus_ap'] = X_all.min_teammate_minus_ap

# set up the model pipe and get the default search parameters
pipe = skm.model_pipe([skm.piece('std_scale'), 
                       skm.piece('k_best'),
                       skm.piece('ridge')])

# set params
params = skm.default_params(pipe, 'rand')
params['k_best__k'] = range(20, 200, 5)


# run the model with parameter search
best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=50,
                                                        col_split='year', 
                                                        time_split=pos[set_pos]['val_start'],
                                                        bayes_rand='custom_rand',
                                                        sample_weight=False,
                                                        random_seed=1234)

m = best_models[1]
transformer = Pipeline(m.steps[:-1])
X_shap = transformer.transform(X)
cols = X.columns[transformer['k_best'].get_support()]
X_shap = pd.DataFrame(X_shap, columns=cols)

import shap
try: shap_values = shap.TreeExplainer(m.steps[-1][1]).shap_values(X_shap)
except: shap_values = shap.explainers.Linear(m.steps[-1][1], X_shap).shap_values(X_shap)
shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns, plot_size=(8,15), max_display=30, show=False)

coef_results = pd.Series(m.steps[-1][1].coef_, cols)
best_cols = abs(coef_results).sort_values().iloc[-8:].index

# %%
from Fix_Standard_Dev import *

met='fp_per_game'
# get the train and prediction dataframes for FP per game
df_train, _ = hf.get_train_predict(df, met, pos, set_pos, 
                                            set_year-pos[set_pos]['test_years'], 
                                            pos[set_pos]['earliest_year'], 
                                            pos[set_pos]['features'])

sd_cols = []
for c in best_cols:
    if coef_results[coef_results.index==c].values[0] < 0:
        df_train['minus_' + c] = np.log(df_train[c].max() - df_train[c] + 1)
        df_predict['minus_' + c] = np.log(df_predict[c].max() - df_predict[c] + 1)
        sd_cols.append('minus_' + c)
    else:
        sd_cols.append(c)

sd_spline, max_spline = get_std_splines(df_train, sd_cols, sd_cols, show_plot=True, k=1, 
                                            min_grps_den=int(df_train.shape[0]*0.25), 
                                            max_grps_den=int(df_train.shape[0]*0.15))
   
# %%

sc = MinMaxScaler()
sc.fit(df_train[sd_cols])

df_predict = df_predict.set_index('player')
df_predict = df_predict.reindex(index=output['player'])
df_predict = df_predict.reset_index()

pred_sd_max = pd.DataFrame(sc.transform(df_predict[sd_cols])).mean(axis=1)
output['std_dev'] = sd_spline(pred_sd_max)
output['max_score'] = max_spline(pred_sd_max)
output.iloc[:50]
#%%

vers = 'beta'

output['pos'] = set_pos
output['filter_data'] = pos[set_pos]['filter_data']
output['year_exp'] = pos[set_pos]['year_exp']
output['version'] = vers
output['year'] = set_year

del_str = f'''pos='{set_pos}' 
              AND version='{vers}'
              AND filter_data='{pos[set_pos]['filter_data']}'
              AND year_exp={pos[set_pos]['year_exp']}'''

dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

# %%
