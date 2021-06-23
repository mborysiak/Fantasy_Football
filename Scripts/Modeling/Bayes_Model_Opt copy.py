#%%
# core packages
import pandas as pd
import numpy as np

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
set_pos = 'RB'

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
pos['WR']['features'] = 'v1'
pos['TE']['features'] = 'v1'

pos['QB']['test_years'] = 1
pos['RB']['test_years'] = 1
pos['WR']['test_years'] = 1
pos['TE']['test_years'] = 1

pos['QB']['use_ay'] = False
pos['RB']['use_ay'] = False
pos['WR']['use_ay'] = False
pos['TE']['use_ay'] = False

np.random.seed(1234)

# +
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
# high_rush_players = df.loc[df.rush_yd_per_game >= 15, 'player'].unique()
# df = df[df.player.isin(high_rush_players)]
# df = df.loc[df.year_exp > 6].reset_index(drop=True)


# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

#%%

# append fp_per_game to the metrics and ensure unique values
pos[set_pos]['metrics'].append('fp_per_game')
pos[set_pos]['metrics'] = list(dict.fromkeys(pos[set_pos]['metrics']))

for met in pos[set_pos]['metrics']:

    print(f'\nRunning Metric {met}\n=========================\n')
    print('ADP only\n============\n')
    
    # get the train and prediction dataframes for FP per game
    df_train, df_predict = hf.get_train_predict(df, met, pos, set_pos, 
                                            set_year-pos[set_pos]['test_years'], 
                                            pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    
    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])  
    
    print('Shape of Train Set', df_train.shape)
    skm = SciKitModel(df_train)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos', 'last_year'])

    # set up the ADP model pipe
    pipe = skm.model_pipe([skm.piece('feature_select'), skm.piece('std_scale'), skm.piece('lr')])
    params = skm.default_params(pipe)
    params['feature_select__cols'] = [['avg_pick'], ['avg_pick', 'year']]

    # fit and append the ADP model
    best_model, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=5,
                                                               col_split='year', 
                                                               time_split=pos[set_pos]['val_start'])

    # append all of the metric outputs
    pred[f'{met}_adp'] = oof_data['combined']; actual[f'{met}_adp'] = oof_data['actual']
    scores[f'{met}_adp'] = r2; models[f'{met}_adp'] = best_model

    #---------------
    # Model Training loop
    #---------------

    # loop through each potential model
    model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf']
    for m in model_list:

        print('\n============\n')
        print(m)

        # set up the model pipe and get the default search parameters
        pipe = skm.model_pipe([skm.piece('feature_drop'),
                               skm.piece('std_scale'), 
                               skm.piece('select_perc'),
                               skm.feature_union([skm.piece('agglomeration'), skm.piece('k_best'), skm.piece('pca')]),
                               skm.piece(m)])
        params = skm.default_params(pipe, 'rand')
        params['feature_drop__col'] = [['avg_pick', 'avg_pick_median']]
        if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

        # run the model with parameter search
        best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=50,
                                                       col_split='year', time_split=pos[set_pos]['val_start'])

        # append the results and the best models for each fold
        pred[f'{met}_{m}'] = oof_data['combined']; actual[f'{met}_{m}'] = oof_data['actual']
        scores[f'{met}_{m}'] = r2; models[f'{met}_{m}'] = best_models

#%%

output = df_predict[['player', 'avg_pick']]
for met in pos[set_pos]['metrics'][1:]:

    print(f'\nRunning Metric {met}')

    # get the X and y values for stack trainin for the current metric
    X_stack, y_stack = skm.X_y_stack(met, pred, actual)

    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm.model_pipe([skm.piece('k_best'), skm.piece('impute'), skm.piece('ridge')])
    best_model = skm.best_stack(stack_pipe, X_stack, y_stack, n_iter=50, run_adp=True)

    # create the full stack pipe with meta estimators followed by stacked model
    stacked_models = [(k, skm.ensemble_pipe(v)) for k,v in models.items() if met in k]
    stack = skm.stack_pipe(stacked_models, best_model)
    
    # fit and predict for the current metric
    df_train, df_predict = hf.get_train_predict(df, met, pos, set_pos, 
                                                set_year-pos[set_pos]['test_years'], 
                                                pos[set_pos]['earliest_year'], pos[set_pos]['features'])
    
    X, y = SciKitModel(df_train).Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos', 'last_year'])
    stack.fit(X,y)
    prediction = pd.Series(np.round(stack.predict(df_predict[X.columns]), 2), name=f'pred_{met}')
    output = pd.concat([output, prediction], axis=1)

pts_cols = [c for c in output.columns if 'fp' not in c and 'pred' in c]
output['pred_fp_per_game_stat'] = (output[pts_cols] * pts_dict[set_pos]).sum(axis=1)
output['pred_fp_per_game_avg'] = (output.pred_fp_per_game + output.pred_fp_per_game_stat) / 2
output = output.sort_values(by='pred_fp_per_game_avg', ascending=False)
output


#%%
#==============
# Create Break-out Probability Features
#==============

breakout_metric = 'fp_per_game'
act_ppg = '>=12'
pct_off = '>0.15'
adp_ppg_high = '<100'
adp_ppg_low = '>=0'

#==============
# Create Break-out Probability Features
#==============

# get the train and prediction dataframes for FP per game
df_train, df_predict = hf.get_train_predict(df, breakout_metric, pos, set_pos, 
                                                      set_year-pos[set_pos]['test_years'], 
                                                      pos[set_pos]['earliest_year'], 
                                                      pos[set_pos]['features'])

# get the adp predictions and merge back to the training dataframe
df_train, df_predict, lr = hf.get_adp_predictions(df_train, df_predict, 1)

# filter to adp cutoffs
df_train = hf.adp_filter(df_train, adp_ppg_low, adp_ppg_high)
df_predict = hf.adp_filter(df_predict, adp_ppg_low, adp_ppg_high)

# generate labels for prediction based on cutoffs
df_train = hf.class_label(df_train, pct_off, act_ppg)
df_predict = hf.class_label(df_predict, pct_off, act_ppg)

# get the minimum number of training samples for the initial datasets
min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])

# print the value-counts
print('Training Value Counts:', df_train.y_act.value_counts()[0], '|', df_train.y_act.value_counts()[1])
print(f'Number of Features: {df_train.shape[1]}')
print('Min Train Year:', df_train.year.min())
print('Max Train Year:', df_train.year.min())
print('Min Val Year:', df_train.year.min()+1)
print('Max Val Year:', df_train.year.max())
print('Min Test Year:', df_predict.year.min())
print('Max Test Year:', df_predict.year.max())


# %%

skm = SciKitModel(df_train, model_obj='class')
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos', 'last_year'])

# loop through each potential model
model_list = [
    'lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c'
]
for m in model_list:

    print('\n============\n')
    print(m)

     # set up the model pipe and get the default search parameters
    pipe = skm.model_pipe([skm.piece('std_scale'), 
                            skm.piece('select_perc_c'),
                            skm.feature_union([
                                            skm.piece('agglomeration'), 
                                            skm.piece('k_best_c'), 
                                            ]),
                            skm.piece(m)])
    
    params = skm.default_params(pipe, 'rand')
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)

    # run the model with parameter search
    best_models, score_results, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                col_split='year', time_split=pos[set_pos]['val_start'])

    # append the results and the best models for each fold
    pred[f'class_{m}'] = oof_data['combined']; actual[f'class_{m}'] = oof_data['actual']
    scores[f'class_{m}'] = score_results; models[f'class_{m}'] = best_models


#%%


output = df_predict[['player', 'avg_pick']]

# get the X and y values for stack trainin for the current metric
X_stack, y_stack = skm.X_y_stack('class', pred, actual)

# get the model pipe for stacking setup and train it on meta features
stack_pipe = skm.model_pipe([skm.piece('k_best'), skm.piece('lr_c')])
best_model = skm.best_stack(stack_pipe, X_stack, y_stack, n_iter=50, print_coef=True)

# create the full stack pipe with meta estimators followed by stacked model
stacked_models = [(k, skm.ensemble_pipe(v)) for k,v in models.items() if 'class' in k]
stack = skm.stack_pipe(stacked_models, best_model)


X, y = SciKitModel(df_train).Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos', 'last_year'])
stack.fit(X,y)

from sklearn.preprocessing import MinMaxScaler

prediction = stack.predict_proba(df_predict[X.columns])[:,1].reshape(-1,1)
prediction = MinMaxScaler().fit_transform(prediction)
prediction = pd.Series(prediction.reshape(1,-1)[0], name='pred_class')
prediction = round(prediction,2)
output = pd.concat([output, prediction], axis=1)
output = output.sort_values(by='avg_pick')
output.iloc[:50]

# %%

output = df_predict[['player', 'avg_pick']]

# get the X and y values for stack trainin for the current metric
X_stack, y_stack = skm.X_y_stack('class', pred, actual)

# get the model pipe for stacking setup and train it on meta features
stack_pipe = skm.model_pipe([ skm.piece('k_best'), skm.piece('rf_c')])
best_model = skm.best_stack(stack_pipe, X_stack, y_stack, n_iter=50, print_coef=True)

# create the full stack pipe with meta estimators followed by stacked model
stacked_models = [(k, skm.ensemble_pipe(v)) for k,v in models.items() if 'class' in k]
stack = skm.stack_pipe(stacked_models, best_model)


X, y = SciKitModel(df_train).Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos', 'last_year'])
stack.fit(X,y)

from sklearn.preprocessing import MinMaxScaler

prediction = stack.predict_proba(df_predict[X.columns])[:,1].reshape(-1,1)
prediction = MinMaxScaler().fit_transform(prediction)
prediction = pd.Series(prediction.reshape(1,-1)[0], name='pred_class')
prediction = round(prediction,2)
output = pd.concat([output, prediction], axis=1)
output = output.sort_values(by='avg_pick')
output.iloc[:50]
# %%
