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
set_year = 2020

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

#==========
# Pull and clean compiled data
#==========

# load data and filter down
df = dm.read(f'''SELECT * FROM {set_pos}_{set_year}''', 'Model_Inputs')
df_train = df[df.year < set_year-1].reset_index(drop=True)
df_predict = df[df.year == set_year-1].reset_index(drop=True)

# %%

# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

mets = pos[rb_wr]['metrics'][-1:]
for met in ['fp_per_game']:# mets:

    print(f'\nRunning Metric {met}\n=========================\n')
    print('ADP only\n============\n')
    
    print('Shape of Train Set', df_train.shape)
    skm = SciKitModel(df_train)
    X, y = skm.Xy_split(y_metric=met, to_drop=['player', 'pos', 'team', 
                                                'adjust_line_yds', 'pass_block_rank', 'last_year',
                                                 'rec_yd_per_game',
                                                'rec_per_game', 'td_per_game', 'fp_per_game'])

    min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])  


    # set up the ADP model pipe
    pipe = skm.model_pipe([skm.piece('feature_select'), skm.piece('std_scale'), skm.piece('lr')])
    params = skm.default_params(pipe)
    params['feature_select__cols'] = [['avg_pick'], ['avg_pick', 'year']]

    # fit and append the ADP model
    best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=50,
                                                    col_split='year', 
                                                    time_split=2014)

    # append all of the metric outputs
    pred[f'{met}_adp'] = oof_data['combined']; actual[f'{met}_adp'] = oof_data['actual']
    scores[f'{met}_adp'] = r2; models[f'{met}_adp'] = best_models

    #---------------
    # Model Training loop
    #---------------

    # loop through each potential model
    model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', #'knn', 
    'gbm', 'rf']
    for m in model_list:

        print('\n============\n')
        print(m)

        # set up the model pipe and get the default search parameters
        pipe = skm.model_pipe([skm.piece('std_scale'), 
                               skm.piece('select_perc'),
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
        best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=10,
                                                       col_split='year', time_split=2014)

        # append the results and the best models for each fold
        pred[f'{met}_{m}'] = oof_data['combined']; actual[f'{met}_{m}'] = oof_data['actual']
        scores[f'{met}_{m}'] = r2; models[f'{met}_{m}'] = best_models

# %%
output = df_predict[['player', 'avg_pick']].copy()

# get the X and y values for stack trainin for the current metric
X_stack, y_stack = skm.X_y_stack(met, pred, actual)

# get the model pipe for stacking setup and train it on meta features
stack_pipe = skm.model_pipe([skm.piece('k_best'), skm.piece('ridge')])
best_model, stack_score, adp_score = skm.best_stack(stack_pipe, X_stack, y_stack, n_iter=50, run_adp=True)

# create the full stack pipe with meta estimators followed by stacked model
stacked_models = [(k, skm.ensemble_pipe(v)) for k,v in models.items()]
stack = skm.stack_pipe(stacked_models, best_model)

stack.fit(X, y)
prediction = pd.Series(np.round(stack.predict(df_predict[X.columns]), 2), name=f'pred_{met}')
output = pd.concat([output, prediction], axis=1)

output = output.sort_values(by='avg_pick')
output['adp_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output.iloc[:50]
# %%
