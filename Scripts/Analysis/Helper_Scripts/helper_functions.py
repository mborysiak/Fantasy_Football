
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#==========
# Dictionary for position relevant metrics
#==========

# initialize full position dictionary
pos = {}

#---------
# RB dictionary
#---------
 
# initilize RB dictionary
pos['RB'] = {}

# total touch filter name
pos['RB']['touch_filter'] = 'total_touches'

# median feature categories
pos['RB']['med_features'] = ['fp', 'tgt', 'receptions', 'total_touches', 'rush_yds', 'rec_yds', 
                   'rush_yd_per_game', 'rec_yd_per_game', 'rush_td', 'games_started', 
                   'qb_rating', 'qb_yds', 'pass_off', 'tm_rush_td', 'tm_rush_yds', 
                   'tm_rush_att', 'adjust_line_yds', 'ms_rush_yd', 'ms_rec_yd', 'ms_rush_td',
                   'avg_pick', 'fp_per_touch', 'team_rush_avg_att']

# sum feature categories
pos['RB']['sum_features'] = ['total_touches', 'att', 'scrimmage_yds']

# max feature categories
pos['RB']['max_features'] = ['fp', 'rush_td', 'tgt', 'rush_yds', 'rec_yds', 'scrimmage_yds']

# age feature categories
pos['RB']['age_features'] = ['fp', 'rush_yd_per_game', 'rec_yd_per_game', 'total_touches', 'receptions', 'tgt',
                             'ms_rush_yd', 'ms_rec_yd', 'available_rush_att', 'available_tgt', 'total_touches_sum',
                             'scrimmage_yds_sum', 'avg_pick', 'fp_per_touch', 'ms_rush_yd_per_att', 'ms_tgts']


# In[ ]:


def calculate_fp(df, pts, pos):
    
    # calculate fantasy points for QB's associated with a given RB or WR
    if pos == 'RB' or 'WR':
        df['qb_fp'] =         pts['pass_yd_pts']*df['qb_yds'] +         pts['pass_td_pts']*df['qb_tds'] -         pts['int_pts']*df['int'] -         pts['sack_pts']*df['qb_sacks']
    
    # calculate fantasy points for RB's
    if pos == 'RB':
        df['fp'] =         pts['yd_pts']*df['rush_yds'] +         pts['yd_pts']*df['rec_yds'] +         pts['td_pts']*df['rush_td'] +         pts['td_pts']*df['rec_td'] +         pts['rec_pts']*df['receptions'] +         pts['fmb_pts']*df['fmb']
        
        # calculate fantasy points per touch
        df['fp_per_touch'] = df['fp'] / df['total_touches']
        
        # calculate fantasy points per target
        df['yd_per_tgt'] = df['rec_yds'] / df['tgt']
        
    # calculate fantasy points per game
    df['fp_per_game'] = df['fp'] / df['games']
    
    return df


# In[ ]:


def features_target(df, year_start, year_end, median_features, sum_features, max_features, 
                    age_features, target_feature):
    
    import pandas as pd

    new_df = pd.DataFrame()
    years = range(year_start+1, year_end+1)

    for year in years:
        
        # adding the median features
        past = df[df['year'] <= year]
        for metric in median_features:
            past = past.join(past.groupby('player')[metric].median(),on='player', rsuffix='_median')

        for metric in max_features:
            past = past.join(past.groupby('player')[metric].max(),on='player', rsuffix='_max')
            
        for metric in sum_features:
            past = past.join(past.groupby('player')[metric].sum(),on='player', rsuffix='_sum')
            
        # adding the age features
        suffix = '/ age'
        for feature in age_features:
            feature_label = ' '.join([feature, suffix])
            past[feature_label] = past[feature] / past['age']
        
        # adding the values for target feature
        year_n = past[past["year"] == year]
        year_n_plus_one = df[df['year'] == year+1][['player', target_feature]].rename(columns={target_feature: 'y_act'})
        year_n = pd.merge(year_n, year_n_plus_one, how='left', left_on='player', right_on='player')
        new_df = new_df.append(year_n)
    
    # creating dataframes to export
    new_df = new_df.sort_values(by=['year', 'fp'], ascending=[False, False])
    new_df = pd.concat([new_df, pd.get_dummies(new_df.year)], axis=1)
    
    df_train = new_df[new_df.year < year_end].reset_index(drop=True)
    df_predict = new_df[new_df.year == year_end].drop('y_act', axis=1).reset_index(drop=True)
    
    df_train['year'] = df_train.year.astype('int')
    df_train = df_train.sort_values(['year', 'fp_per_game'], ascending=True).reset_index(drop=True)

    df_predict['year'] = df_predict.year.astype('int')
    
    return df_train, df_predict


# In[ ]:


def visualize_features(df_train):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from my_plot import PrettyPlot
    
    plt.figure(figsize=(17,12))
    k = 25
    corrmat = abs(df_train.corr())
    cols_large = corrmat.nlargest(k, 'y_act').index
    hm_large = corrmat.nlargest(k,'y_act')[cols_large]
    sns.set(font_scale=1.2)
    sns_plot = sns.heatmap(hm_large, cmap="YlGnBu", cbar=True, annot=True, square=False, fmt='.2f', 
                 annot_kws={'size': 12});

    fig = sns_plot.get_figure();
    PrettyPlot(plt);


# In[ ]:


def corr_removal(df_train, df_predict, corr_cutoff=0.025):

    corr = df_train.corr()['y_act']
    good_cols = list(corr[abs(corr) > corr_cutoff].index)

    good_cols.extend(['player', 'year'])
    df_train = df_train[good_cols]
    df_train = df_train.loc[:,~df_train.columns.duplicated()]

    good_cols.remove('y_act')
    df_predict = df_predict[good_cols]
    df_predict = df_predict.loc[:,~df_predict.columns.duplicated()]
    
    return df_train, df_predict


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, scale=True, impute=True, impute_strategy='median', print_progress=False):
       
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        self.print_progress = print_progress
       
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)
            
        if scale:
            self.scale = StandardScaler()
 
    def fit(self, X, y=None):
        
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
            
        if hasattr(self, 'scale'):
            self.scale.fit(X)
        return self
 
    def transform(self, X, y=None):
        
        columns = X.columns.tolist()
        
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
            
        if hasattr(self, 'scale'):
            X = pd.DataFrame(self.scale.transform(X), columns=columns)
        
        return ReduceVIF.calculate_vif(self, X, self.thresh)
 
    @staticmethod
    def calculate_vif(self, X, thresh=5.0):
        
        print('Running VIF Feature Reduction')
        
        # filter out warnings during run
        import warnings
        warnings.filterwarnings("ignore")
        
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            
            variables = X.columns
            
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
           
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                if self.print_progress:
                    print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
                        
        return X


# In[ ]:


#=============
# Create parameter dictionaries for each algorithm
#=============

lgbm_params = {
    'n_estimators':[30, 40, 50, 60, 75],
    'max_depth':[2, 3, 4, 5, 6, 7],
    'freature_fraction':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'min_child_weight': [5, 10, 15, 20, 25],
}

xgb_params = {
    'n_estimators': [30, 40, 50, 60, 75], 
    'max_depth': [2, 3, 4, 5, 6, 7], 
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'min_child_weight': [10, 15, 20, 25, 30],
    'freature_fraction':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

rf_params = {
    'n_estimators': [30, 40, 50, 60, 75, 100, 125, 150], 
    'max_depth': [3, 4, 5, 6, 7], 
    'min_samples_leaf': [1, 2, 3, 5, 7, 10],
    'max_features':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

catboost_params = {
    'iterations': [10, 25, 50], 
    'depth': [1, 2, 3, 4, 5, 10]
}

ridge_params = {
    'alpha': [50, 100, 150, 200, 250, 300, 400, 500]
}

lasso_params = {
    'alpha': [0.5, 0.75, 0.8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
}

lasso_pca_params = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2]
}

lr_params = {}


# In[ ]:


def get_estimator(name, params, rand=True, random_state=None):
    
    import random
    from numpy import random
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from catboost import CatBoostRegressor
    from sklearn.linear_model import LinearRegression
    
    state = random.RandomState(random_state)
    
    rnd_params = {}
    tmp_params = params[name]
    if rand == True:
        for line in tmp_params.items():
            rnd_params[line[0]] = state.choice(line[1])
    else:
        rnd_params = tmp_params
    
    if name == 'lgbm':
        estimator = LGBMRegressor(random_state=1234, **rnd_params, min_data=1)
        
    if name == 'xgb':
        estimator = XGBRegressor(random_state=1234, **rnd_params)
        
    if name == 'rf':
        estimator = RandomForestRegressor(random_state=1234, **rnd_params)
        
    if name == 'ridge':
        estimator = Ridge(random_state=1234, **rnd_params)
        
    if name == 'lasso':
        estimator = Lasso(random_state=1234, **rnd_params)
        
    if name == 'catboost':
        estimator = CatBoostRegressor(random_state=1234, logging_level='Silent', **rnd_params)
        
    if name == 'lasso_pca':
        estimator = Lasso(random_state=1234, **rnd_params)
        
    if name == 'lr_pca':
        estimator = LinearRegression()

    return estimator, rnd_params


# In[ ]:


def X_y_split(train, val, scale=True, pca=False):
    '''
    input: train and validation or test datasets
    output: datasets split into X features and y response for train / validation or test
    '''
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X_train = train.select_dtypes(include=['float', 'int', 'uint8']).drop('y_act', axis=1)
    y_train = train.y_act
    
    try:    
        X_val = val.select_dtypes(include=['float', 'int', 'uint8']).drop('y_act', axis=1)
        y_val = val.y_act
    except:
        X_val = val.select_dtypes(include=['float', 'int', 'uint8'])
        y_val = None
    
    if scale == True:
        X_train = StandardScaler().fit_transform(X_train)
        X_val = StandardScaler().fit_transform(X_val)
    else:
        pass
    
    if pca == True:
        pca = PCA(n_components=10)
        pca.fit(X_train)
        
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
    else:
        pass
        
    return X_train, X_val, y_train, y_val


# In[ ]:


def calc_residuals(estimator, X_train, y_train, X_val, y_val, train_error, val_error):
    '''
    input: estimator, feature set to be predicted, ground truth
    output: sum of residuals for train and validation predictions
    '''
    predict_train = estimator.predict(X_train)
    train_error.append(np.sum((predict_train-y_train)**2))
    
    predict_val = estimator.predict(X_val)
    val_error.append(np.sum(abs(predict_val-y_val)**2))
    
    return train_error, val_error


# In[ ]:


def error_compare(df, skip_years):
    
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df[df.year > df.year.min() + skip_years+1].dropna().reset_index(drop=True)

    lr = LinearRegression().fit(df.pred.values.reshape(-1,1), df.y_act)
    r_sq_pred = round(lr.score(df.pred.values.reshape(-1,1), df.y_act), 3)
    corr_pred = round(pearsonr(df.pred, df.y_act)[0], 3)
    
    lr = LinearRegression().fit(df.avg_pick.values.reshape(-1,1), df.y_act)
    r_sq_avg_pick = round(lr.score(df.avg_pick.values.reshape(-1,1), df.y_act), 3)
    corr_avg_pick = abs(round(pearsonr(df.avg_pick, df.y_act)[0], 3))

    return [r_sq_pred, corr_pred, r_sq_avg_pick, corr_avg_pick]


# In[ ]:


def validation(estimators, params, df_train, iterations=50, random_state=None, scale=False, pca=False, skip_years=2):
    '''
    input: training dataset, estimator
    output: out-of-sample errors and predictions for 5 timeframes
    '''
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    import numpy as np
    import datetime
    
    # initialize a parameter tracker dictionary and summary output dataframe
    param_tracker = {}
    summary = pd.DataFrame()
    
    #==========
    # Complete a Random Hyperparameter search for the given models and parameters
    #==========
    
    for i in range(0, iterations):
        
        # update random state to pull new params, but keep consistency based on starting state
        random_state = random_state + 1
        
        # print update on progress
        if (i+1) % 10 == 0:
            print(str(datetime.datetime.now())[:-7])
            print('Completed ' + str(i+1) + '/' + str(iterations) + ' iterations')
            
        # create empty sub-dictionary for current iteration storage
        param_tracker[i] = {}
        
        # create empty lists to store predictions and errors for each estimator
        est_predictions=pd.DataFrame()
        est_errors=pd.DataFrame()
        
        #==========
        # Loop through estimators with a running time series based training and validation method
        #==========
        
        for est in estimators:

            # grab estimator and random parameters for estimator type
            estimator, param_tracker[i][est] = get_estimator(est, params, rand=True, random_state=random_state)
        
            # run through all years for given estimator and save errors and predictions
            val_error = []    
            train_error = [] 
            val_predictions = np.array([]) 
            years = df_train.year.unique()[1:]

            #==========
            # Loop through years and complete a time-series validation of the given model
            #==========
            
            for m in years:
                
                # create training set for all previous years and validation set for current year
                train_split = df_train[df_train.year < m]
                val_split = df_train[df_train.year == m]
        
                # setting the scale parameter based on the given model
                if est == 'ridge' or est == 'lasso' or est == 'knn' or pca == True:
                    scale = True
    
                # splitting the train and validation sets into X_train, y_train, X_val and y_val
                X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca)
        
                # fit training data and creating prediction based on validation data
                estimator.fit(X_train, y_train)
                val_predict = estimator.predict(X_val)
                
                # skip over the first two year of predictions due to high error for xgb / lgbm
                if m > years.min() + skip_years:
                    val_predictions = np.append(val_predictions, val_predict, axis=0)
                    
                    # calculate and append training and validation errors
                    train_error, val_error = calc_residuals(estimator, X_train, y_train, X_val, y_val, train_error, val_error)
                else:
                    pass
                
            # append predictions for all validation samples / models (n_samples x m_models)
            # and all errors (n_years x m_models) to dataframes 
            est_predictions = pd.concat([est_predictions, pd.Series(val_predictions, name=est)], axis=1)
            est_errors = pd.concat([est_errors, pd.Series(val_error, name=est)], axis=1)
        
        #==========
        # Create an ensemble model based on residual errors from each individual model
        #==========
        
        # create weights based on the mean errors across all years for each model
        est_errors = est_errors.iloc[1:, :]
        frac = 1 - (est_errors.mean() / (est_errors.mean().sum()))
        weights = round(frac / frac.sum(), 3)          
        
        # multiply the outputs from each model by their weights and sum to get final prediction
        wt_results = pd.concat([df_train[df_train.year > years.min() + skip_years].reset_index(drop=True),
                                pd.Series((est_predictions*weights).sum(axis=1), name='pred')],
                                axis=1)
        
        #==========
        # Calculate Error Metrics and Prepare Export
        #==========
        
        # calculate r_squared and correlation for n+1 results using predictions and avg_pick
        compare_metrics = error_compare(wt_results, skip_years)

        # calculate the RMSE and MAE of the ensemble predictions
        wt_rmse = round(np.sqrt(mean_squared_error(wt_results.pred, df_train[df_train.year > years.min() + skip_years].reset_index(drop=True).y_act)), 2)
        wt_mae = round(mean_absolute_error(wt_results.pred, df_train[df_train.year > years.min() + skip_years].reset_index(drop=True).y_act), 2)
        
        #--------
        # create a list of model weights based on residuals, as well s the average RMSE & MAE 
        # for the ensemble predictions to append to the output dataframe for export
        #--------
        
        # generate a list of weights used for models
        wt_list = list(weights.values)
        
        # append rmse and mae metrics for a given ensemble
        wt_list.append(wt_rmse)
        wt_list.append(wt_mae)
        
        # extend the results with the r2 and correlation metrics for the ensemble and adp
        wt_list.extend(compare_metrics)
        summary = summary.append([(wt_list)])
    
    #==========
    # Update Summary Table of Weights and Error Metric Results
    #==========
        
    summary = summary.reset_index(drop=True)
    estimators.extend(['rmse', 'mae', 'r2_pred', 'c_pred', 'r2_adp', 'c_adp'])
    summary.columns = estimators
    summary = summary.sort_values(by=['rmse', 'r2_pred'], ascending=[True, False])
    
    return param_tracker, summary, wt_results, est_errors


# In[ ]:


def generate_predictions(best_result, param_list, summary, df_train, df_predict, figsize=(6,15)):
    
    param_list = param_list[best_result]
    weights = summary.iloc[best_result, :len(param_list)]
    est_names = summary.columns[:len(param_list)]
    
    X_train, X_val, y_train, _ = X_y_split(df_train, df_predict)
    
    predictions = pd.DataFrame()
    
    models = []
    for est in est_names[0:len(param_list)]:
        estimator, _ = get_estimator(est, param_list, rand=False)
        
        estimator.fit(X_train, y_train)
        test_predictions = pd.Series(estimator.predict(X_val), name=est)
        
        predictions = pd.concat([predictions, test_predictions], axis=1)
        models.append(estimator)
        
    wt_predictions = pd.Series((predictions*weights).sum(axis=1), name='pred')
    wt_predictions = pd.concat([df_predict.reset_index(drop=True), wt_predictions], axis=1)
    
    to_plot = wt_predictions.pred
    to_plot.index = wt_predictions.player
    to_plot.sort_values().plot.barh(figsize=figsize);
    plt.show()
    
    return wt_predictions, models, 


# In[ ]:


def predict_roc(estimator, X_train, y_train, X_val, y_val, train_rmses, val_rmses, avg='macro'):
    '''
    input: estimator, feature set to be predicted, ground truth
    output: RMSE value for out-of-sample predctions and list of predictions
    '''
    from sklearn.metrics import roc_auc_score
        
    predict_train = estimator.predict(X_train)
    train_rmses.append(roc_auc_score(y_train, predict_train, average=avg))
    
    predict_val = estimator.predict(X_val)
    val_rmses.append(roc_auc_score(y_val, predict_val, average='weighted'))
    
    return train_rmses, val_rmses


# In[ ]:


def validation_class(df_train, estimator, df_predict, scale=True, proba=False, avg='macro', pca=False):
    '''
    input: training dataset, estimator
    output: out-of-sample errors and predictions for 5 timeframes
    '''
    import pandas as pd
    import numpy as np
    
    val_rmses = []    
    train_rmses = []
    train_predictions = np.array([])
    years = df_train.year.unique()[1:]
    
    for i in years:
        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < i]
        val_split = df_train[df_train.year == i]
        
        # splitting in X and y
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca)
        
        # fit training data and predict validation data
        estimator.fit(X_train, y_train)
        if proba == True:
            val_predict = estimator.predict_proba(X_val)[:,1]
        else:
            val_predict = estimator.predict(X_val)
        train_predictions = np.append(train_predictions, val_predict, axis=0)
        
        # determine training and validation errors
        train_rmses, val_rmses = predict_roc(estimator, X_train, y_train, X_val, y_val, train_rmses, val_rmses, avg)
        
    # create predictions for upcoming, unknown year
    X_train, X_val, y_train, _ = X_y_split(df_train, df_predict)
    estimator.fit(X_train, y_train)
    if proba == True:
        test_predictions = estimator.predict_proba(X_val)[:,1]
    else:
        test_predictions = estimator.predict(X_val)
    
    # append training and validation erros
    train_rmses.append(np.mean(train_rmses[3:]))
    val_rmses.append(np.mean(val_rmses[3:]))
    
    # printing results
    labels = [str(year) for year in years]
    labels.append('MEAN')
    
    results = pd.DataFrame([train_rmses, val_rmses]).T
    results.columns = ['Train Error', 'Test Error']
    results.index = labels
    print(results)
    
    return estimator, train_predictions, test_predictions


# In[ ]:


def plot_results(results, col_names, asc=True, barh=True, figsize=(6,16), fontsize=12):
    '''
    Input:  The feature importance or coefficient weights from a trained model.
    Return: A plot of the ordered weights, demonstrating relative importance of each feature.
    '''
    
    import pandas as pd
    import matplotlib.pyplot as plt

    #cols = df_predict.select_dtypes(include=['float', 'int', 'uint8']).columns
    series = pd.Series(results, index=col_names).sort_values(ascending=asc)
    
    if barh == True:
        ax = series.plot.barh(figsize=figsize, fontsize=fontsize)
        #ax.set_xlabel(label, fontsize=fontsize+1)
    else:
        ax = series.plot.bar(figsize=figsize, fontsize=fontsize)
        #ax.set_ylabel(label, fontsize=fontsize+1)
        
    return ax


# In[ ]:


def add_outcomes(df, outcomes, year_cutoff):
    
    import pandas as pd

    tmp = df[df.year > 0].reset_index(drop=True)
    ind = tmp[tmp.year > year_cutoff].index
    outcomes = outcomes.iloc[ind]
    
    df = df[df.year > year_cutoff]
    
    df['pred'] = outcomes.values
    try:
        df['error']  = df.y_act - df.pred
    except:
        pass
    
    df = df.reset_index(drop=True)
    
    return df


# In[ ]:


def show_error(df, year_cutoff = 0):
    
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df[df.year > year_cutoff].reset_index(drop=True)

    lr = LinearRegression().fit(df.pred.values.reshape(-1,1), df.y_act)
    r_sq = lr.score(df.pred.values.reshape(-1,1), df.y_act)
    print('R-Squared: ', round(r_sq, 3))
    
    plt.scatter(df.pred, df.y_act)
    plt.plot(range(6,20), range(6,20))
    print('Prediction vs. Actual Correlation:', 
          round(pearsonr(df.pred, df.y_act)[0], 3))
    plt.scatter(df.pred, df.y_act);


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

class clustering():

    def __init__(self, df_train, df_test, model_weights, pred_weight=2):
    
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        import matplotlib.pyplot as plt
        
        self.df_train = df_train
        self.df_test = df_test
        
        # scale feature importances and take mean for weighting clustering
        all_feature_wts = []
        for model in model_weights:
            try:
                features = MinMaxScaler().fit_transform(abs(model.feature_importances_).reshape(-1,1))
            except:
                features = MinMaxScaler().fit_transform(abs(model.coef_).reshape(-1,1))
                
            all_feature_wts.append(features)

        mean_features = np.mean(all_feature_wts, axis=0)[:,0]
    
        # create df for clustering by selecting numeric value and dropping y_act
        self.X_train = df_train.select_dtypes(include=['float', 'int', 'uint8']).drop(['y_act', 'error'], axis=1)
        self.X_test = df_test.select_dtypes(include=['float', 'int', 'uint8'])

        # scale all columns
        scale = StandardScaler().fit(self.X_train)
        
        self.X_train = scale.transform(self.X_train)
        self.X_test = scale.transform(self.X_test)
    
        # weight the columns according to mean coefficients and add weight for predictions
        self.X_train = pd.DataFrame(self.X_train * np.append(mean_features, pred_weight))
        self.X_test = pd.DataFrame(self.X_test * np.append(mean_features, pred_weight))
        
        
    def explore_k(self, k=15):
        
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
        from sklearn import metrics

        # k means determine k
        X_train = self.X_train
        distortions = []
        silhouettes = []
        K = range(2,k)
        for k in K:
            kmeanModel = KMeans(n_clusters=k, random_state=1).fit(X_train)
            kmeanModel.fit(X_train);
            distortions.append(sum(np.min(cdist(X_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0]);
            
            silhouettes.append(metrics.silhouette_score(X_train, kmeanModel.labels_))
                               
        # create elbow plot
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(K, distortions, 'bx-')
        ax1.set_title("Distortion");
        
        ax2 = fig.add_subplot(122)
        ax2.plot(K, silhouettes, 'x-')
        ax2.set_title("Silhouette Score");
        
        
    def fit_and_predict(self, k=10):
        
        from sklearn.cluster import KMeans

        # retrain with optimal cluster 
        self.k = k
        self.kmeans = KMeans(n_clusters=k, random_state=1).fit(self.X_train)
        self.train_results = self.kmeans.predict(self.X_train)
        self.test_results = self.kmeans.predict(self.X_test) 
    
    
    def add_clusters(self):
        
        self.df_train['cluster'] = self.train_results
        self.df_test['cluster'] = self.test_results
        
        return self.df_train, self.df_test
            
    
    def show_results(self, j):
        from scipy.stats import pearsonr
    
        # calculate and print all percentiles for players in group
        percentile = np.percentile(self.df_train[self.df_train.cluster == j].y_act, q=[5, 25, 50, 75, 95])
        print('Fantasy PPG for Various Percentiles')
        print('-----------------------------------')
        print('5th percentile: ', round(percentile[0], 2))
        print('25th percentile:', round(percentile[1], 2))
        print('50th percentile:', round(percentile[2], 2))
        print('75th percentile:', round(percentile[3], 2))
        print('95th percentile:', round(percentile[4], 2))
    
        # show plot of historical actual results for cluster
        ax = self.df_train[self.df_train.cluster == j].y_act.plot.hist()
        ax.set_xlabel('Fantasy PPG Actual')
    
        # show plot of predicted vs actual points for cluster
        ax = self.df_train[self.df_train.cluster == j].plot.scatter('pred', 'y_act')
        ax.set_xlabel('Predicted Fantasy PPG')
        ax.set_ylabel('Actual Fantasy PPG')
        
        # show correlation coefficient between actual and predicted points for cluster
        print('')
        print('Pred to Actual Correlation')
        print(round(pearsonr(self.df_train[self.df_train.cluster==j].pred, self.df_train[self.df_train.cluster==j].y_act)[0], 3))
        
        # show examples of past players in cluster
        current = self.df_test[self.df_test.cluster == j].sort_values(by='pred', ascending=True)[['player', 'avg_pick', 'pred']]
        
        return current
    
    
    def create_distributions(self, wt=2.5):
        from scipy.stats import skewnorm
        from scipy.stats import pearsonr
        import numpy as np

        distributions = pd.DataFrame()

        for i in range(0,self.k):
            tmp = self.df_train[self.df_train.cluster == i]
            
            # caclculate mean / median, standard error, and skew of actual distribution
            baseline = max(tmp.y_act.median(), tmp.y_act.mean())
            std_err = tmp.y_act.std() / np.sqrt(len(tmp.y_act))
            skew = tmp.y_act.skew()
            
            # calculate the difference between the center of the distribution and each player in the cluster
            diff = self.df_test[self.df_test.cluster==i].pred - baseline

            # shift players to / from the mean based on difference weighted by standard error
            new_pred = baseline + std_err*diff
            
            # adjust predictions with the shifted predictions
            self.df_test.loc[self.df_test.cluster == i, 'pred'] = new_pred
    
            # create error distribution based on skew, and weighted standard error
            error_dist = skewnorm.rvs(skew, size=1000)*std_err*wt
            distributions = distributions.append([list(error_dist)])
    
        distributions['cluster'] = range(0,self.k)
        
        rb_sampling = self.df_test[['player', 'cluster', 'pred']]
        rb_sampling = pd.merge(rb_sampling, distributions, how='left', left_on='cluster', right_on='cluster')
        
        return rb_sampling


# In[ ]:


def view_projections(data, player):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = data[data.player == player]
    ax = pd.Series(16*(df.iloc[0][3:] + df.pred.values)).plot.hist()
    ax.set_xlabel('Fantasy Points');
    ax.set_title(player + ' Projections');

