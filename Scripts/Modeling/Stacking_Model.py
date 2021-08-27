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

pos['QB']['val_start'] = 2010
pos['RB']['val_start'] = 2010
pos['WR']['val_start'] = 2010
pos['TE']['val_start'] = 2010

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
pos['RB']['filter_data'] = 'less_equal'
pos['WR']['filter_data'] = 'less_equal'
pos['TE']['filter_data'] = 'greater_equal'

pos['QB']['year_exp'] = 0
pos['RB']['year_exp'] = 3
pos['WR']['year_exp'] = 3
pos['TE']['year_exp'] = 0

pos['QB']['act_ppg'] = 20
pos['RB']['act_ppg'] = 13
pos['WR']['act_ppg'] = 12
pos['TE']['act_ppg'] = 11

pos['QB']['pct_off'] = 0.05
pos['RB']['pct_off'] = 0.1
pos['WR']['pct_off'] = 0.1
pos['TE']['pct_off'] = 0.1

pos['QB']['iters'] = 25
pos['RB']['iters'] = 25
pos['WR']['iters'] = 25
pos['TE']['iters'] = 25

pos['QB']['all_stats'] = False
pos['RB']['all_stats'] = False
pos['WR']['all_stats'] = False
pos['TE']['all_stats'] = False

np.random.seed(1234)

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
print('Min Val Year:', df_train_class.year.min()+1)
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

            ['avg_pick', 'avg_pick / age', 'avg_pick_median'],

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



# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

skm_class = SciKitModel(df_train_class, model_obj='class')
X_class, y_class = skm_class.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

# loop through each potential model
model_list = [
    'lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c'#, 'svc'
]
for m in model_list:

    print('\n============\n')
    print(m)

     # set up the model pipe and get the default search parameters
    pipe = skm_class.model_pipe([skm_class.piece('std_scale'), 
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
    best_models, score_results, oof_data = skm_class.time_series_cv(pipe, X_class, y_class, 
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

output_class = output_start[['player', 'avg_pick']].copy()

# get the X and y values for stack trainin for the current metric
skm_class = SciKitModel(df_train_class, model_obj='class')
X_stack_class, y_stack_class = skm_class.X_y_stack('class', pred_class, actual_class)

try: X_stack_class = X_stack_class.drop('class_svc', axis=1)
except: pass

skm_class_final = SciKitModel(df_train_class, model_obj='class')
X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', 
                                                        to_drop=['player', 'team', 'pos'])

# create the full stack pipe with meta estimators followed by stacked model
X_predict_class = pd.DataFrame()
for k, v in models_class.items():
    if 'svc' not in k:
        m = skm_class.ensemble_pipe(v)
        m.fit(X_class_final, y_class_final)
        cur_pred = pd.Series(m.predict_proba(df_predict_class[X_class_final.columns])[:,1], name=k)
        X_predict_class = pd.concat([X_predict_class, cur_pred], axis=1)


pred = load_pickle(model_output_path, 'reg_pred')
actual = load_pickle(model_output_path, 'reg_actual')
models = load_pickle(model_output_path, 'reg_models')
scores = load_pickle(model_output_path, 'reg_scores')

output = output_start[['player', 'avg_pick']].copy()

# set up the stacking training + prediction dataset
df_train, df_predict = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                            set_year-pos[set_pos]['test_years'], 
                                            pos[set_pos]['earliest_year'], pos[set_pos]['features'])
df_predict = filter_less_equal(df_predict, pos)
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

X_stack = pd.concat([X_stack, X_stack_class], axis=1)

best_models = []
final_models = ['ridge', 'lgbm', 'xgb', 'rf', 'bridge']
for final_m in final_models:

    print(f'\n{final_m}')
    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm.model_pipe([
                            skm.piece('std_scale'), 
                            skm.piece('k_best'), 
                            skm.piece(final_m)
                        ])
    best_model, stack_score, adp_score = skm.best_stack(stack_pipe, X_stack, 
                                                        y_stack, n_iter=50, 
                                                        run_adp=True, print_coef=True)
    best_models.append(best_model)

# get the final output:
X_fp, y_fp = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

# create the full stack pipe with meta estimators followed by stacked model
X_predict = pd.DataFrame()
for k, v in models.items():
    m = skm.ensemble_pipe(v)
    m.fit(X_fp, y_fp)
    X_predict = pd.concat([X_predict, pd.Series(m.predict(df_predict[X_fp.columns]), name=k)], axis=1)

X_predict = pd.concat([X_predict, X_predict_class], axis=1)

predictions = pd.DataFrame()
for bm, fm in zip(best_models, final_m):
    prediction = pd.Series(np.round(bm.predict(X_predict), 2), name=f'pred_{met}_{fm}')
    predictions = pd.concat([predictions, prediction], axis=1)

db_output['reg_stack_score'] = stack_score
db_output['adp_stack_score'] = adp_score


output['pred_fp_per_game'] = predictions.mean(axis=1)
std_models = predictions.std(axis=1)
std_bridge = bm.predict(X_predict, return_std=True)[1]
output['std_dev'] = (std_models/2)+ std_bridge
output = output.sort_values(by='avg_pick')
output['adp_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output.iloc[:50]

#%%

vers = 'v1'

output['pos'] = set_pos
output['filter_data'] = pos[set_pos]['filter_data']
output['year_exp'] = pos[set_pos]['year_exp']
output['version'] = vers
output['max_score'] = 1.05*np.percentile(df_train.y_act.max(), 99)

del_str = f'''pos='{set_pos}' 
              AND version='{vers}'
              AND filter_data='{pos[set_pos]['filter_data']}'
              AND year_exp={pos[set_pos]['year_exp']}'''

dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

# %%

# %%
