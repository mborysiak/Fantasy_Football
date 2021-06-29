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
set_pos = 'QB'

# set year to analyze
set_year = 2021

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
pos['WR']['filter_data'] = 'greater_equal'
pos['TE']['filter_data'] = 'greater_equal'

pos['QB']['year_exp'] = 0
pos['RB']['year_exp'] = 4
pos['WR']['year_exp'] = 0
pos['TE']['year_exp'] = 0

pos['QB']['act_ppg'] = 20
pos['RB']['act_ppg'] = 12
pos['WR']['act_ppg'] = 11
pos['TE']['act_ppg'] = 10

pos['QB']['pct_off'] = 0.05
pos['RB']['pct_off'] = 0.15
pos['WR']['pct_off'] = 0.15
pos['TE']['pct_off'] = 0.15

pos['QB']['iters'] = 25
pos['RB']['iters'] = 25
pos['WR']['iters'] = 25
pos['TE']['iters'] = 25

pos['QB']['all_stats'] = True
pos['RB']['all_stats'] = False
pos['WR']['all_stats'] = False
pos['TE']['all_stats'] = False

pos['QB']['reg_stack_model'] = 'ridge'
pos['RB']['reg_stack_model'] = 'ridge'
pos['WR']['reg_stack_model'] = 'ridge'
pos['TE']['reg_stack_model'] = 'ridge'

pos['QB']['class_stack_model'] = 'rf_c'
pos['RB']['class_stack_model'] = 'rf_c'
pos['WR']['class_stack_model'] = 'rf_c'
pos['TE']['class_stack_model'] = 'rf_c'

np.random.seed(1234)


all_vars = ['req_touch', 'req_games', 'earliest_year', 'val_start', 
            'features', 'test_years', 'use_ay', 'filter_data', 'year_exp', 
            'act_ppg', 'pct_off', 'iters', 'all_stats', 'reg_stack_model',
            'class_stack_model']

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
    df = df.loc[df.year_exp <= pos[set_pos]['year_exp']].reset_index(drop=True)

# get the train and prediction dataframes for FP per game
_, output_start = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                 set_year-pos[set_pos]['test_years'], 
                                 pos[set_pos]['earliest_year'], pos[set_pos]['features'])
output = output_start[['player', 'avg_pick']].copy()

# append fp_per_game to the metrics and ensure unique values
pos[set_pos]['metrics'].append('fp_per_game')
pos[set_pos]['metrics'] = list(dict.fromkeys(pos[set_pos]['metrics']))

# append fp_per_game to the metrics and ensure unique values
pts_dict[set_pos].append(1)
# pts_dict[set_pos] = list(dict.fromkeys(pts_dict[set_pos]))

#%%

# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

if pos[set_pos]['all_stats']:
    mets = pos[set_pos]['metrics']
else:
    mets = pos[set_pos]['metrics'][-1:]

for met in mets:

    print(f'\nRunning Metric {met}\n=========================\n')
    print('ADP only\n============\n')
    
    # get the train and prediction dataframes for FP per game
    df_train, df_predict = hf.get_train_predict(df, met, pos, set_pos, 
                                                set_year-pos[set_pos]['test_years'], 
                                                pos[set_pos]['earliest_year'], 
                                                pos[set_pos]['features'])
    
    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.year <= df_train.year.min()].shape[0])  
    
    print('Shape of Train Set', df_train.shape)
    skm = SciKitModel(df_train)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

    # set up the ADP model pipe
    pipe = skm.model_pipe([skm.piece('feature_select'), skm.piece('std_scale'), skm.piece('lr')])
    params = skm.default_params(pipe)
    params['feature_select__cols'] = [['avg_pick'], ['avg_pick', 'year']]

    # fit and append the ADP model
    best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'],
                                                    col_split='year', 
                                                    time_split=pos[set_pos]['val_start'])

    # append all of the metric outputs
    pred[f'{met}_adp'] = oof_data['combined']; actual[f'{met}_adp'] = oof_data['actual']
    scores[f'{met}_adp'] = r2; models[f'{met}_adp'] = best_models

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
                               skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                                skm.piece('pca')
                                                ]),
                               skm.piece('k_best'),
                               skm.piece(m)])
        params = skm.default_params(pipe, 'rand')
        params['feature_drop__col'] = [

            ['avg_pick', 'avg_pick_exp','avg_pick_exp_diff',
             'avg_pick_exp_div', 'avg_pick_median', 'avg_pick_exp_median',
             'avg_pick_exp_diff_median', 'avg_pick_exp_div_median', 'avg_pick_sum',
             'avg_pick_over_median', 'avg_pick_exp_over_median', 'avg_pick_exp_diff_over_median',
             'avg_pick_exp_div_over_median', 'avg_pick / age', 'avg_pick_exp / age',
             'avg_pick_exp_diff / age','avg_pick_exp_div / age','avg_pick_rolling',
             'avg_pick_exp_rolling', 'avg_pick_exp_diff_rolling', 'avg_pick_exp_div_rolling'],
            
            ['avg_pick_exp_diff_median',  'avg_pick_exp_div_median', 'avg_pick_median',             
             'avg_pick_exp_diff / age', 'avg_pick_exp_diff', 'avg_pick_exp_div / age', 
             'avg_pick_exp_div', 'avg_pick / age', 'avg_pick'],

            ['avg_pick', 'avg_pick / age'],

            ]
        if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

        # run the model with parameter search
        best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=pos[set_pos]['iters'],
                                                       col_split='year', time_split=pos[set_pos]['val_start'])

        # append the results and the best models for each fold
        pred[f'{met}_{m}'] = oof_data['combined']; actual[f'{met}_{m}'] = oof_data['actual']
        scores[f'{met}_{m}'] = r2; models[f'{met}_{m}'] = best_models

save_pickle(pred, model_output_path, 'reg_pred')
save_pickle(actual, model_output_path, 'reg_actual')
save_pickle(models, model_output_path, 'reg_models')
save_pickle(scores, model_output_path, 'reg_scores')

#%%

pred = load_pickle(model_output_path, 'reg_pred')
actual = load_pickle(model_output_path, 'reg_actual')
models = load_pickle(model_output_path, 'reg_models')
scores = load_pickle(model_output_path, 'reg_scores')

output = output_start[['player', 'avg_pick']].copy()

# set up the stacking training + prediction dataset
df_train, df_predict = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                            set_year-pos[set_pos]['test_years'], 
                                            pos[set_pos]['earliest_year'], pos[set_pos]['features'])
df_predict = df_predict.drop('y_act', axis=1).fillna(0)
skm = SciKitModel(df_train)

# get the X and y values for stack trainin for the current metric
X_stack = pd.DataFrame()
for met, pt in zip(pos[set_pos]['metrics'], pts_dict[set_pos]):
    
    if pos[set_pos]['all_stats']:
        X_s, y_s = skm.X_y_stack(met, pred, actual)
        X_stack = pd.concat([X_stack, X_s*pt], axis=1)
        if met=='fp_per_game': y_stack = y_s
    
    elif met == 'fp_per_game':
        X_s, y_s = skm.X_y_stack(met, pred, actual)
        X_stack = pd.concat([X_stack, X_s*pt], axis=1)
        y_stack = y_s

# get the model pipe for stacking setup and train it on meta features
stack_pipe = skm.model_pipe([skm.piece('k_best'), skm.piece(pos[set_pos]['reg_stack_model'])])
best_model, stack_score, adp_score = skm.best_stack(stack_pipe, X_stack, y_stack, n_iter=pos[set_pos]['iters']*3, run_adp=True)
db_output['reg_stack_score'] = stack_score
db_output['adp_stack_score'] = adp_score

# create the full stack pipe with meta estimators followed by stacked model
stacked_models = [(k, skm.ensemble_pipe(v)) for k,v in models.items()]
stack = skm.stack_pipe(stacked_models, best_model)

X_fp, y_fp = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos',])
stack.fit(X_fp,y_fp)
prediction = pd.Series(np.round(stack.predict(df_predict[X_fp.columns]), 2), name=f'pred_{met}')
output = pd.concat([output, prediction], axis=1)

output = output.sort_values(by='avg_pick')
output['adp_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output.iloc[:50]


#%%
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

# get the train and prediction dataframes for FP per game
df_train_class, df_predict_class = hf.get_train_predict(df, breakout_metric, pos, set_pos, 
                                                      set_year-pos[set_pos]['test_years'], 
                                                      pos[set_pos]['earliest_year'], 
                                                      pos[set_pos]['features'])

# get the adp predictions and merge back to the training dataframe
df_train_class, df_predict_class, lr = hf.get_adp_predictions(df_train_class, df_predict_class, 1)

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
print('Min Val Year:', df_train_class.year.min()+1)
print('Max Val Year:', df_train_class.year.max())
print('Min Test Year:', df_predict_class.year.min())
print('Max Test Year:', df_predict_class.year.max())


# %%

# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

skm = SciKitModel(df_train_class, model_obj='class')
X_class, y_class = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

# loop through each potential model
model_list = [
    'lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c', 'svc'
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
                            skm.piece('k_best_c'),
                            skm.piece(m)])
    
    params = skm.default_params(pipe, 'rand')
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)

    # run the model with parameter search
    best_models, score_results, oof_data = skm.time_series_cv(pipe, X_class, y_class, 
                                                              params, n_iter=pos[set_pos]['iters'],
                                                              col_split='year', time_split=pos[set_pos]['val_start'])

    # append the results and the best models for each fold
    pred[f'class_{m}'] = oof_data['combined']; actual[f'class_{m}'] = oof_data['actual']
    scores[f'class_{m}'] = score_results; models[f'class_{m}'] = best_models

save_pickle(pred, model_output_path, 'class_pred')
save_pickle(actual, model_output_path, 'class_actual')
save_pickle(models, model_output_path, 'class_models')
save_pickle(scores, model_output_path, 'class_scores')

# %%

pred_class = load_pickle(model_output_path, 'class_pred')
actual_class = load_pickle(model_output_path, 'class_actual')
models_class = load_pickle(model_output_path, 'class_models')
scores_class = load_pickle(model_output_path, 'class_scores')

output = output_start[['player', 'avg_pick']].copy()

# get the X and y values for stack trainin for the current metric
skm = SciKitModel(df_train_class, model_obj='class')
X_stack_class, y_stack_class = skm.X_y_stack('class', pred_class, actual_class)

# get the model pipe for stacking setup and train it on meta features
stack_pipe_class = skm.model_pipe([ skm.piece('k_best'), skm.piece(pos[set_pos]['class_stack_model'])])
best_model_class, stack_score_class, _ = skm.best_stack(stack_pipe_class, X_stack_class, y_stack_class, 
                                                        n_iter=pos[set_pos]['iters']*3, print_coef=True)
db_output['class_stack_score'] = stack_score_class

# create the full stack pipe with meta estimators followed by stacked model
stacked_models = [(k, skm.ensemble_pipe(v)) for k,v in models_class.items() if 'class' in k]
stack_class = skm.stack_pipe(stacked_models, best_model_class)

skm_class_final = SciKitModel(df_train_class)
X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', 
                                                        to_drop=['player', 'team', 'pos'])
stack_class.fit(X_class_final, y_class_final)

from sklearn.preprocessing import MinMaxScaler

prediction = stack_class.predict_proba(df_predict_class[X_class_final.columns])[:,1].reshape(-1,1)
prediction = MinMaxScaler().fit_transform(prediction)
prediction = pd.Series(prediction.reshape(1,-1)[0], name='pred_class')
prediction = round(prediction,2)
output = pd.concat([output, prediction], axis=1)
output = output.sort_values(by='avg_pick')
output.iloc[:50]
# %%


df_output = pd.DataFrame(db_output, index=[0])
df_output['reg_pct_above'] = round((df_output.reg_stack_score - df_output.adp_stack_score) / df_output.adp_stack_score, 3)
output_cols = ['pkey', 'set_pos', 'set_year']
output_cols.extend(all_vars)
output_cols.extend(['reg_stack_score', 'adp_stack_score', 'reg_pct_above', 'class_stack_score'])
df_output = df_output[output_cols]
dm.write_to_db(df_output, 'ParamTracking', 'StackResults', if_exist='append')
# %%