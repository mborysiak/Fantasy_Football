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
import numpy as np
import pandas as pd
import sqlite3
import os
from zHelper_Functions import *

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
# -

# # User Inputs

# +
#==========
# General Setting
#==========

# set core path
path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'
db_name = 'Model_Inputs.sqlite3'

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'Rookie_WR'

# set the year
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

# -

# # Pull In Data

# connect to database and pull in positional data
conn = sqlite3.connect(path + 'Data/Databases/' + db_name)
df = pd.read_sql_query('SELECT * FROM ' + set_pos + '_' + str(set_year), con=conn)

# +
#=============
# Create parameter dictionaries for each algorithm
#=============

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

params = {
    'lgbm': {
        'n_estimators':[10, 20, 30, 40, 50, 60, 75, 100, 150],
        'max_depth':[2, 3, 4, 5, 6, 7, 8, 10, 15],
        'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'min_child_weight': [0.1, 1, 5, 10, 15, 20, 25, 30, 40]
    },

    'xgb': {
        'n_estimators': [20, 30, 40, 50, 60, 75, 100, 150], 
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 15], 
        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'min_child_weight': [0.1, 1, 5, 10, 15, 20, 25, 30, 40],
        'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },

    'rf': {
        'n_estimators': [50, 75, 100, 125, 150, 250], 
        'max_depth': [3, 4, 5, 6, 7, 10, 12, 15, 20], 
        'min_samples_leaf': [1, 2, 3, 5, 7, 10, 15],
        'max_features':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },

    'ridge': {
        'alpha': np.arange(0.1, 1000, 0.5)
    },

    'lasso': {
        'alpha': np.arange(0.1, 1000, 0.5)
    },

     'enet': {
        'alpha': np.arange(0.1, 1000, 0.5),
        'l1_ratio': np.arange(0.02, 0.98, 0.1)
     },
    
    'gbm': {
         'n_estimators': [50, 75, 100, 125, 150, 250], 
        'max_depth': [3, 4, 5, 6, 7, 10, 12, 15, 20], 
        'min_samples_leaf': [1, 2, 3, 5, 7, 10, 15],
        'max_features':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },
    
    'knn': {
        'n_neighbors': range(1, 50), 
        'weights': ['distance', 'uniform'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    
    'svr': {
        'C': np.arange(0.01, 1000, 10), 
        'epsilon': np.arange(0.01, 100, 10)
        }

}


# -

def cv_estimate(est, X_train, y_train, n_folds=5):
    
    # set up KFolds and empty lists to store results
    cv = KFold(n_folds)
    rmse_scores = []
    r2_scores = []
    out_of_sample_pred = []
    all_y = []
    ind = []
    
    # loop through each split and train / predict in and out of sample 
    for train, test in cv.split(X_train):
        
        # fit model
        est.fit(X_train.iloc[train, :], y_train[train])
        
        # predict the out of sample data points and append to list
        pred = est.predict(X_train.iloc[test, :])
        out_of_sample_pred.extend(pred)
    
        # get the rmse and r2 score of out of sample predictions
        rmse_scores.append(np.sqrt(mean_squared_error(pred, y_train[test])))
        r2_scores.append(r2_score(y_train[test], pred))
        
        # add in the y_act values
        all_y.extend(y_train[test])
        
        # append all the indexes in order
        ind.extend(test)

    # take average of out of sample predictions
    rmse_result = np.mean(rmse_scores)
    r2_result = np.mean(r2_scores)

    # print results
    print('RMSE Score:', round(np.mean(rmse_result), 2))
    print('R2 Score:', round(np.mean(r2_result), 2))
    
    return out_of_sample_pred, all_y, rmse_result, r2_result, ind


# +
if set_pos == 'Rookie_RB':
    metrics = ['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game', 'fp_per_game']
elif set_pos == 'Rookie_WR':
    metrics = ['rec_yd_per_game', 'rec_per_game', 'td_per_game', 'fp_per_game']

model_labels = ['lgbm', 'xgb', 'ridge', 'lasso', 'enet', 'rf', 'gbm', 'knn']

results = {}
for y_label in metrics:
    print(f'\n==============\nRunning {y_label}\n==============')
    results[y_label] = {}
    results[y_label]['test_pred'] = {}
    results[y_label]['val_pred'] = {}
    results[y_label]['val_y'] = {}
    results[y_label]['val_error'] = []
    
    # extract the train and predict dataframes
    predict = df[df.year==(set_year - 1)].reset_index(drop=True)
    train = df[df.year!=(set_year - 1)].reset_index(drop=True)
    
    # remove unnecessary columns
    to_drop = [m for m in metrics if m != y_label]
    to_drop.extend(['team'])
    
    # drop unnecessary columns
    Xy = train.rename(columns={y_label: 'y_act'}).drop(to_drop, axis=1)

    # remove low correlation features
    Xy = corr_collinear_removal(Xy, corr_cutoff=0.05, collinear_cutoff=0.8, good_cols_ext=[])
    predict = predict[Xy.drop('y_act', axis=1).columns]

    y = Xy.y_act
    X = Xy.drop(['y_act'], axis=1)
    
    val = []
    all_y = []
    for m in model_labels:
        print(f'\n---------\nRunning {m}\n---------')        
        
        if m in ('ridge', 'lasso', 'enet', 'svr', 'knn'):
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
            
        # set up and run grid search model
        grid = RandomizedSearchCV(models[m], params[m], cv=5, n_iter=30, scoring='neg_mean_squared_error', random_state=1234)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        
        # get the out of sample predictions for the train set with best model
        val_pred, val_y, rmse, r2, ind = cv_estimate(best_model, X, y)
        
        # predict the test set
        test_pred = best_model.predict(predict)
        
        # append the results to dictioary
        results[y_label]['val_pred'][m] = val_pred
        results[y_label]['val_y'][m] = val_y
        results[y_label]['val_error'].append([m, rmse, r2])
        results[y_label]['test_pred'][m] = test_pred
    
    print(f'\n---------\nRunning LR with ADP\n---------')      
    # run and append results using only ADP of the player
    val_pred_adp, val_y, rmse, r2, ind = cv_estimate(LinearRegression(), train[['avg_pick']], y)
    results[y_label]['val_pred']['lr_adp'] = val_pred_adp
    results[y_label]['val_y']['lr_adp'] = val_y
    results[y_label]['val_error'].append(['lr_adp', rmse, r2])
    results[y_label]['test_pred']['lr_adp'] = test_pred
# -

val_results = pd.DataFrame()
test_results= pd.DataFrame()
for met in metrics:
    df_met = pd.DataFrame(results['rec_yd_per_game']['val_error'], columns=['model', 'rmse', 'r2_score']).sort_values(by='rmse')
    df_met = df_met[df_met.r2_score > 0]
    df_met['total_r2'] = df_met.r2_score.sum()
    df_met['wt'] = df_met.r2_score / df_met.total_r2

    for i, row in df_met.iterrows():
        if i == 0:
            val_pred = np.array(results[met]['val_pred'][row.model])*row.wt
            test_pred = np.array(results[met]['test_pred'][row.model])*row.wt
        else:
            val_pred = np.vstack([val_pred, np.array(results[met]['val_pred'][row.model])*row.wt])
            test_pred = np.vstack([test_pred, np.array(results[met]['test_pred'][row.model])*row.wt])
            
    val_results = pd.concat([val_results, pd.Series(np.sum(val_pred, axis=0), name=met+'_pred')], axis=1)
    test_results = pd.concat([test_results, pd.Series(np.sum(test_pred, axis=0), name=met+'_pred')], axis=1)

val_results

np.sqrt(np.mean((val_df - y_df)**2))

# +
if set_pos == 'Rookie_RB':
    pts = [.1, .1, .5, 7]
elif set_pos == 'Rookie_WR':
    pts = [.1, .5, 7]

historical = pd.Series((val_df).sum(axis=1).values, index=df[df.year!=set_year-1].player)
historical.sort_values().plot.barh(figsize=(5,20))
# -



df['year_exp'] = 0
train_out = pd.concat([df.loc[df.year!=set_year-1, ['player', 'year']].reset_index(drop=True),
                       val_df,
                       df.loc[df.year!=set_year-1, ['pp_age', 'year_exp', 'avg_pick']].reset_index(drop=True), 
                       (y_df*pts).sum(axis=1)], axis=1)
cols = ['player', 'year']
m = ['pred_' + m for m in metrics]
cols.extend(m)
cols.extend(['age', 'year_exp', 'avg_pick', 'y_act'])
train_out.columns = cols
train_out['avg_pick'] = np.log(train_out.avg_pick)
train_out['age'] = np.log(train_out.age)

# +
sum_results = pd.DataFrame()
for i in range(0, results.shape[1], 4):
    sum_results = pd.concat([sum_results, results.iloc[:, i:(i+4)].mean(axis=1)], axis=1)
    
sum_results.columns = metrics
sum_results['fp_per_game'] = (sum_results*pts).sum(axis=1)
# -

pd.concat([df.loc[df.year==2018, ['player']].reset_index(drop=True), sum_results], axis=1).sort_values(by='fp_per_game', ascending=False).reset_index(drop=True)

test_out = pd.concat([df.loc[df.year==2018, ['player']].reset_index(drop=True),
                      set_year-1,
                      sum_results,
                      df.loc[df.year==2018, ['pp_age', 'year_exp', 'avg_pick']].reset_index(drop=True)], axis=1)
cols = ['player']
cols.extend(m)
cols.extend(['age', 'year_exp', 'avg_pick'])
test_out.columns = cols
test_out['avg_pick'] = np.log(test_out['avg_pick'])

train_out.to_sql(set_pos.replace('Rookie_', '') + '_Train_' + str(set_year), 
                 engine, schema='websitedev', index=False, if_exists='append')
test_out.to_sql(set_pos.replace('Rookie_', '') + '_Test_' + str(set_year), 
                engine, schema='websitedev', index=False, if_exists='append')


