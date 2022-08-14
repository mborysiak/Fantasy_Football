#%%
# core packages
import pandas as pd
import numpy as np
import os
import gzip
import pickle

from ff.db_operations import DataManage
from ff import general
from skmodel import SciKitModel
import matplotlib.pyplot as plt
import zModel_Functions as mf
from Fix_Standard_Dev import *

import pandas_bokeh
pandas_bokeh.output_notebook()

import zHelper_Functions as hf
pos = hf.pos

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)

#==========
# General Setting
#==========

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'Rookie_WR'
rb_wr = set_pos.split('_')[1]

# set year to analyze
set_year = 2022

vers = 'beta'
current_or_next_year = 'current'

model_output_path = f'{root_path}/Model_Outputs/{set_year}/{set_pos}/'
if not os.path.exists(model_output_path): os.makedirs(model_output_path)


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
df = df.sort_values(by='year').reset_index(drop=True)
df['week'] = 1
df['game_date'] = df.year
df = df.rename(columns={'fp_per_game': 'y_act'})

df_train = df[df.year < set_year-1].reset_index(drop=True)
df_predict = df[df.year == set_year-1].reset_index(drop=True)

output_start = df_predict[['player', 'avg_pick']].copy()
print(set_pos, df_train.shape[0], df_predict.shape[0])

df_predict = df_predict.fillna(0)
for c in df_predict.columns:
    if len(df_predict[df_predict[c]==np.inf]) > 0:
        df_predict.loc[df_predict[c]==np.inf, c] = 0
    if len(df_predict[df_predict[c]==-np.inf]) > 0:
        df_predict.loc[df_predict[c]==-np.inf, c] = 0

df_train = df_train[(df_train.games > 8)].reset_index(drop=True)

df_train_class = df_train.copy()
df_predict_class = df_predict.copy()

if set_pos == 'Rookie_WR':
    drop_cols = ['player', 'pos', 'team', 'rec_yd_per_game', 
                 'rec_per_game', 'td_per_game', 'games']
    df_train_class['y_act'] = np.where(df_train_class.y_act >= 11, 1, 0)

elif set_pos == 'Rookie_RB':
    drop_cols = ['player', 'pos', 'team', 'rec_yd_per_game',  
                 'rush_yd_per_game', 'rec_per_game', 'td_per_game', 'games']
    df_train_class['y_act'] = np.where(df_train_class.y_act >= 13, 1, 0)

# %%

out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

print('ADP only\n============\n')

print('Shape of Train Set', df_train.shape)
skm = SciKitModel(df_train)
X, y = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)

min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])  

# set up the ADP model pipe
pipe = skm.model_pipe([skm.piece('feature_select'), skm.piece('std_scale'), skm.piece('lr')])
params = skm.default_params(pipe)
params['feature_select__cols'] = [['avg_pick'], ['avg_pick', 'year']]

# fit and append the ADP model
best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, params, n_iter=50,
                                                col_split='year', n_splits=4,
                                                time_split=2013, bayes_rand='custom_rand')

 # append all of the metric outputs
out_dict = update_output_dict('reg', 'adp', '', out_dict, oof_data, best_models)

#---------------
# Model Training loop
#---------------

# loop through each potential model
model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'gbm', 'rf']
for i, m in enumerate(model_list):

    print('\n============\n')
    print(m)

    # set up the model pipe and get the default search parameters
    pipe = skm.model_pipe([#skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece('select_perc'),
                            skm.feature_union([
                                            skm.piece('agglomeration'), 
                                            skm.piece('k_best'),
                                            ]),
                            skm.piece('k_best'),
                            skm.piece(m)])
    params = skm.default_params(pipe, 'rand')

    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

    # run the model with parameter search
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                    col_split='year', n_splits=4, 
                                                    time_split=2013, bayes_rand='custom_rand',
                                                    random_seed=(i*13)+(i*7))

     # append all of the metric outputs
    out_dict = update_output_dict('reg', m, '', out_dict, oof_data, best_models)

save_output_dict(out_dict, 'reg')


#%%


# set up blank dictionaries for all metrics
out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

skm_class = SciKitModel(df_train_class, model_obj='class')
X_class, y_class = skm_class.Xy_split(y_metric='y_act', to_drop=drop_cols)

# loop through each potential model
model_list = [
    'lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c'
]
for i, m in enumerate(model_list):

    print('\n============\n')
    print(m)

     # set up the model pipe and get the default search parameters
    pipe = skm_class.model_pipe([#skm_class.piece('random_sample'),
                                skm_class.piece('std_scale'), 
                                skm_class.piece('select_perc_c'),
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
                                                        params, n_iter=25, n_splits=4,
                                                        col_split='year', time_split=2013,
                                                        bayes_rand='custom_rand', proba=True, 
                                                        random_seed=(4*i)+(7*i)+i)

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
                                                                            time_split=2013)

        # append the results and the best models for each fold
        out_dict = update_output_dict('quant', m, f'_{alpha}', out_dict, oof_data, best_models)

save_output_dict(out_dict, 'quant')


# %%

#------------
# Make the Class Predictions
#------------

# load the class predictions
pred_class, actual_class, models_class, _, full_hold_class = mf.load_all_pickles(model_output_path, 'class')
X_predict_class = mf.get_class_predictions(models_class, df_train_class, df_predict_class)
X_stack_class, y_stack_class = mf.X_y_stack('class', full_hold_class, pred_class, actual_class)

#------------
# Make the Reg Predictions
#------------

# get the X and y values for stack trainin for the current metric
pred, actual, models, _, full_hold_reg = mf.load_all_pickles(model_output_path, 'reg')
X_stack, y_stack = mf.X_y_stack('reg', full_hold_reg, pred, actual)
X_stack = pd.concat([X_stack, X_stack_class], axis=1)

#------------
# Make the Quantile Predictions
#------------

pred_quant, actual_quant, models_quant, scores_quant, full_hold_quant = mf.load_all_pickles(model_output_path, 'quant')
X_predict_quant = mf.get_quant_predictions(df_train, df_predict, models_quant)
X_stack_quant, _ = mf.X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)
X_stack = pd.concat([X_stack, X_stack_quant], axis=1)


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

    mf.show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)
    mf.top_predictions(stack_pred['stack_pred'], stack_pred['y'], r2=True)


X_predict = mf.get_reg_predict_features(df_predict, models, X, y)
X_predict = pd.concat([X_predict, X_predict_class], axis=1)
X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

predictions = mf.stack_predictions(X_predict, best_models, final_models)
best_val, best_predictions = mf.best_average_models(skm_stack, scores, final_models, y_stack, stack_val_pred, predictions)
mf.show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
output = mf.create_output(output_start, predictions)


#%%

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
                                                        time_split=2013,
                                                        bayes_rand='custom_rand',
                                                        sample_weight=False,
                                                        random_seed=1234)

mf.shap_plot(best_models, X)
   
# %%


sd_cols, df_train, df_predict = mf.get_sd_cols(df_train, df_predict, X, best_models)
sd_spline, max_spline = get_std_splines(df_train, sd_cols, sd_cols, show_plot=True, k=1, 
                                        min_grps_den=int(df_train.shape[0]*0.25), 
                                        max_grps_den=int(df_train.shape[0]*0.15))

output = mf.assign_sd_max(output, df_predict, df_train, sd_cols, sd_spline, max_spline)
output.iloc[:50]

# %%



output['pos'] = set_pos
output['filter_data'] = 'Rookie_Model'
output['year_exp'] = 0
output['current_or_next_year'] = current_or_next_year
output['version'] = vers
output['year'] = set_year

del_str = f'''pos='{set_pos}' 
              AND version='{vers}'
              AND filter_data='Rookie_Model'
              AND year_exp=0
              AND current_or_next_year='{current_or_next_year}' '''

dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')
# %%
current_or_next_year
# %%
