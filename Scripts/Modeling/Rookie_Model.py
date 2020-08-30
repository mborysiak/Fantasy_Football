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
set_pos = 'Rookie_RB'

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
        'n_estimators':[10, 20, 30, 40, 50, 60, 75, 100, 150, 200],
        'max_depth':[2, 3, 4, 5, 6, 7, 8, 10, 15, 20],
        'colsample_bytree':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'min_child_weight': [0.1, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    },

    'xgb': {
        'n_estimators': [20, 30, 40, 50, 60, 75, 100, 150, 200], 
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 15, 20], 
        'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'min_child_weight': [0.1, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60],
        'colsample_bytree':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },

    'rf': {
        'n_estimators': [50, 75, 100, 125, 150, 250, 300, 350], 
        'max_depth': [3, 4, 5, 6, 7, 10, 12, 15, 20], 
        'min_samples_leaf': [1, 2, 3, 5, 7, 10, 15],
        'max_features':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },

    'ridge': {
        'alpha': np.arange(0.1, 1000, 0.5)
    },

    'lasso': {
        'alpha': np.arange(0.001, 100, 0.5)
    },

     'enet': {
        'alpha': np.arange(0.001, 100, 0.5),
        'l1_ratio': np.arange(0.01, 0.99, 0.05)
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

def cv_estimate(est, m, cor, collin, X, y, n_folds=5, print_result=True):
    
    # set up KFolds and empty lists to store results
    cv = KFold(n_folds)
    rmse_scores = []
    r2_scores = []
    out_of_sample_pred = []
    all_y = []
    ind = []
    
    # loop through each split and train / predict in and out of sample         
    for tr, te in cv.split(X):

        # get the train and test dataset splits
        (X_train, X_test, y_train, y_test) = (X.iloc[tr, :].reset_index(drop=True), 
                                             X.iloc[te, :].reset_index(drop=True), 
                                             y[tr].reset_index(drop=True), y[te].reset_index(drop=True))

        if m in ('ridge', 'lasso', 'enet', 'svr', 'knn'):
            sc = StandardScaler()
            X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)

        # run corr_collinear removal
        X_train = corr_collinear_removal(pd.concat([X_train, y_train], axis=1), 
                                         cor, collin, good_cols_ext=[]).drop('y_act', axis=1)
        X_test = X_test[X_train.columns]

        # fit and predict train/test
        est.fit(X_train, y_train)
        
        # predict
        pred = est.predict(X_test)
        out_of_sample_pred.extend(pred)
    
        # get the rmse and r2 score of out of sample predictions
        rmse_scores.append(np.sqrt(mean_squared_error(pred, y_test)))
        r2_scores.append(r2_score(y_test, pred))
        
        # add in the y_act values
        all_y.extend(y_test)
        
        # append all the indexes in order
        ind.extend(te)

    # take average of out of sample predictions
    rmse_result = np.mean(rmse_scores)
    r2_result = np.mean(r2_scores)

    # print results
    if print_result:
        print('RMSE Score:', round(np.mean(rmse_result), 2))
        print('R2 Score:', round(np.mean(r2_result), 2))
    
    return out_of_sample_pred, all_y, rmse_result, r2_result, ind


def random_param(m, i):
    out_p = {}
    np.random.seed(i)
    p = params[m]
    for k, v in p.items():
        out_p[k] = np.random.choice(v)
    
    return out_p


# +
if set_pos == 'Rookie_RB':
    metrics = ['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game', 'fp_per_game']
elif set_pos == 'Rookie_WR':
    metrics = ['rec_yd_per_game', 'rec_per_game', 'td_per_game', 'fp_per_game']

# set up the different models to run
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
    train = df[df.year < (set_year - 1)].reset_index(drop=True)
    
    # remove unnecessary columns
    to_drop = [m for m in metrics if m != y_label]
    to_drop.extend(['team', 'player', 'pos', 'games'])
    
    # drop unnecessary columns
    Xy = train.rename(columns={y_label: 'y_act'}).drop(to_drop, axis=1)
    y = Xy.y_act
    X = Xy.drop(['y_act'], axis=1)
    
    val = []
    all_y = []
    
    for j, m in enumerate(model_labels):
        print(f'\n---------\nRunning {m}\n---------')   
        est = models[m]
        (best_pred, best_model, best_cor, best_collin) = 100, None, None, None
        
        for i in range(0, 100):

            preds = []

            # set the random parameters
            rp = random_param(m, i*3+i*13+j*27)
            est.set_params(**rp)
            
            # set the cor and collin cutoff
            np.random.seed(i*3+i*17+j*27)
            
            if set_pos == 'Rookie_WR':
                if y_label == 'td_per_game':
                    cor = np.random.choice(np.arange(0.02, 0.12, 0.01))
                else:
                    cor = np.random.choice(np.arange(0.05, 0.25, 0.01))
            
            elif set_pos == 'Rookie_RB':
                if y_label == 'td_per_game':
                    cor = np.random.choice(np.arange(0.1, 0.25, 0.01))
                else:
                    cor = np.random.choice(np.arange(0.05, 0.35, 0.01))
                
            collin = np.random.choice(np.arange(0.5, 0.95, 0.02))
        
            # loop through each split and train / predict in and out of sample 
            _, _, rmse, _, _ = cv_estimate(est, m, cor, collin, X, y, print_result=False)
            
            # if current scores best best score, save out the model and corr/collin values
            if rmse < best_pred:
                best_pred = rmse
                best_model = rp
                best_cor = cor
                best_collin = collin
                print('Found new best score:', round(rmse, 2))
                
        best_model = est.set_params(**best_model)
        print( best_cor, best_collin, best_model)
        
        # get the out of sample predictions for the train set with best model
        val_pred, val_y, rmse, r2, ind = cv_estimate(best_model, m, best_cor, best_collin, X, y)
        
        # predict the test set
        y_train = y.copy()
        X_train = corr_collinear_removal(pd.concat([X, y], axis=1), best_cor, best_collin, []).drop('y_act', axis=1)
        X_predict = predict[X_train.columns].copy()
        
        # scale all X and predict
        if m in ('ridge', 'lasso', 'enet', 'svr', 'knn'):
            sc = StandardScaler()
            X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
            X_predict = pd.DataFrame(sc.transform(X_predict), columns=X_predict.columns)
            
        best_model.fit(X_train, y_train)
        test_pred = best_model.predict(X_predict)
        
        # append the results to dictioary
        results[y_label]['val_pred'][m] = val_pred
        results[y_label]['val_y'][m] = val_y
        results[y_label]['val_error'].append([m, rmse, r2])
        results[y_label]['test_pred'][m] = test_pred
    
    print(f'\n---------\nRunning LR with ADP\n---------')      
    # run and append results using only ADP of the player
    val_pred_adp, val_y, rmse, r2, ind = cv_estimate(LinearRegression(), 'None', 0, 1, X_train[['avg_pick']], y)
    results[y_label]['val_pred']['lr_adp'] = val_pred_adp
    results[y_label]['val_y']['lr_adp'] = val_y
    results[y_label]['val_error'].append(['lr_adp', rmse, r2])
    
    lr = LinearRegression()
    lr.fit(X_train[['avg_pick']], y)
    results[y_label]['test_pred']['lr_adp'] = lr.predict(X_predict[['avg_pick']])

# +
val_results = pd.DataFrame()
test_results= pd.DataFrame()
for met in metrics:
    df_met = pd.DataFrame(results[met]['val_error'], columns=['model', 'rmse', 'r2_score']).sort_values(by='rmse')
    
    if df_met[df_met.r2_score > 0].shape[0] > 0:
        df_met = df_met[df_met.r2_score > 0].reset_index(drop=True)
        df_met['r2_score'] = df_met.r2_score - df_met.r2_score.min()
    else:
        df_met = df_met.sort_values(by='r2_score', ascending=False).iloc[:3].reset_index(drop=True)
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

# pull out the stats columns
stat_col = [m+'_pred' for m in metrics if m != 'fp_per_game']
if len(stat_col) == 3:
    pts = [0.1, 0.5, 7]
else:
    pts = [0.1, 0.1, 0.5, 7]
# predict fantasy points based on stats only
val_results['fp_per_game_pred_stat'] = (val_results[stat_col] * pts).sum(axis=1)
test_results['fp_per_game_pred_stat'] = (test_results[stat_col] * pts).sum(axis=1)

# concat player name and ultimate points scored
val_results = pd.concat([train.player, val_results, 
                         pd.Series(results['fp_per_game']['val_y']['lgbm'], name='fp_act')], axis=1)
test_results = pd.concat([predict.player, test_results], axis=1).sort_values(by='fp_per_game_pred', ascending=False).reset_index(drop=True)
# -

r2_score(val_results.fp_act, np.mean([val_results.fp_per_game_pred_stat, val_results.fp_per_game_pred], axis=0))

val_results.sort_values(by='fp_per_game_pred', ascending=False).iloc[:30]

test_results

# +
import pymc3 as pm

formula = 'fp_act ~ fp_per_game_pred + fp_per_game_pred_stat'

# Context for the model
with pm.Model() as normal_model:
    
    # The prior for the data likelihood is a Normal Distribution
    family = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data = val_results, family = family)
    
    # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 500)
# +
from IPython.core.pylabtools import figsize
import seaborn as sns
import scipy.stats as stats

# Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation, max_bound):
    
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation)
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)
    
    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()
    
    # Add in intercept term
    test_observation['Intercept'] = 1
    
    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)
    
    # create truncated distribution
    lower, upper = 0,  max_bound * 1.2
    trunc_dist = stats.truncnorm((lower - mean_loc) / sd_value, (upper - mean_loc) / sd_value, 
                                  loc=mean_loc, scale=sd_value)
    estimates = trunc_dist.rvs(1000)

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                kde_kws = {'linewidth' : 4},
                label = 'Estimated Dist.')
    
    # Plot the mean estimate
    plt.vlines(x = mean_loc, ymin = 0, ymax = 0.1, 
               linestyles = '--', colors = 'red',
               label = 'Pred Estimate',
               linewidth = 2.5)
    
    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');
    
    # Prediction information
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                       np.percentile(estimates, 95)))
    
    plt.show()
    
    return estimates


# -


dists = []
for i in range(len(test_results)):
    print(test_results.loc[i, 'player'])
    estimates = test_model(normal_trace, 
                           test_results.loc[i,['fp_per_game_pred', 'fp_per_game_pred_stat']], 
                           val_results.fp_act.max())
    dists.append(estimates)

output = pd.concat([test_results.player, pd.DataFrame(dists)], axis=1)
output = output.assign(pos=set_pos.split('_')[1])
order_cols = ['player', 'pos']
order_cols.extend([i for i in range(1000)])
output = output[order_cols]
output.iloc[:, 2:] = np.uint32(output.iloc[:, 2:] * 16)
players = tuple(output.player)


vers = 'Version3'

# +
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
conn_sim.cursor().execute(f'''DELETE FROM {vers}_{set_year} WHERE player in {players}''')
conn_sim.commit()

append_to_db(output, 'Simulation', f'{vers}_{set_year}', 'append')
# -


