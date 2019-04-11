#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Initializing Parameters

# In[2]:


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

# metrics to be predicted for fantasy point generation
pos['RB']['metrics'] = ['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['RB']['med_features'] = ['fp', 'tgt', 'receptions', 'total_touches', 'rush_yds', 'rec_yds', 
                           'rush_yd_per_game', 'rec_yd_per_game', 'rush_td', 'games_started', 
                           'qb_rating', 'qb_yds', 'pass_off', 'tm_rush_td', 'tm_rush_yds', 
                           'tm_rush_att', 'adjust_line_yds', 'ms_rush_yd', 'ms_rec_yd', 'ms_rush_td',
                           'avg_pick', 'fp_per_touch', 'team_rush_avg_att']

# sum feature categories
pos['RB']['sum_features'] = ['total_touches', 'att', 'total_yds']

# max feature categories
pos['RB']['max_features'] = ['fp', 'rush_td', 'tgt', 'rush_yds', 'rec_yds', 'total_yds']

# age feature categories
pos['RB']['age_features'] = ['fp', 'rush_yd_per_game', 'rec_yd_per_game', 'total_touches', 'receptions', 'tgt',
                             'ms_rush_yd', 'ms_rec_yd', 'available_rush_att', 'available_tgt', 'total_touches_sum',
                             'total_yds_sum', 'avg_pick', 'fp_per_touch', 'ms_rush_yd_per_att', 'ms_tgts']

#---------
# WR dictionary
#---------
 
# initilize RB dictionary
pos['WR'] = {}

# total touch filter name
pos['WR']['touch_filter'] = 'tgt'

# metrics to calculate stats for
pos['WR']['metrics'] = ['rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['WR']['med_features'] = ['fp', 'tgt', 'receptions', 'rec_yds', 'rec_yd_per_game', 'rec_td', 'games_started', 
                             'qb_rating', 'qb_yds', 'pass_off', 'ms_tgts', 'ms_rec_yd', 
                             'tm_net_pass_yds', 'avg_pick']
# sum feature categories
pos['WR']['sum_features'] = ['receptions', 'rec_yds', 'tgt']

# max feature categories
pos['WR']['max_features'] = ['fp', 'rec_td', 'tgt', 'ms_tgts', 'ms_rec_yd']

# age feature categories
pos['WR']['age_features'] = ['fp', 'rec_yd_per_game', 'receptions', 'tgt', 'ms_tgts', 'ms_rec_yd', 
                             'avg_pick', 'ms_yds_per_tgts']


#---------
# QB dictionary
#---------
 
# initilize RB dictionary
pos['QB'] = {}

# total touch filter name
pos['QB']['touch_filter'] = 'qb_att'

# metrics to calculate stats for
pos['QB']['metrics'] = ['qb_yd_per_game', 'pass_td_per_game','rush_yd_per_game', 
                        'rush_td_per_game' ,'int_per_game', 'sacks_per_game' ]

pos['QB']['med_features'] = ['fp', 'qb_tds','qb_rating', 'qb_yds', 'pass_off', 'qb_complete_pct', 'qb_td_pct', 
                                'sack_pct', 'avg_pick', 'sacks_allowed']
pos['QB']['max_features'] = ['fp', 'qb_rating', 'qb_yds', 'qb_tds']
pos['QB']['age_features'] = ['fp', 'qb_rating', 'qb_yds', 'qb_complete_pct', 'qb_td_pct', 'sack_pct', 'avg_pick']
pos['QB']['sum_features'] = ['qb_tds', 'qb_yds']

#---------
# WR dictionary
#---------
 
# initilize RB dictionary
pos['TE'] = {}

# total touch filter name
pos['TE']['touch_filter'] = 'tgt'

# metrics to calculate stats for
pos['TE']['metrics'] = ['rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['TE']['med_features'] = ['fp', 'tgt', 'receptions', 'rec_yds', 'rec_yd_per_game', 'rec_td', 'games_started', 
                             'qb_rating', 'qb_yds', 'pass_off', 'tm_net_pass_yds', 'avg_pick']
# sum feature categories
pos['TE']['sum_features'] = ['receptions', 'rec_yds', 'tgt']

# max feature categories
pos['TE']['max_features'] = ['fp', 'rec_td', 'tgt']

# age feature categories
pos['TE']['age_features'] = ['fp', 'rec_yd_per_game', 'receptions', 'tgt', 'avg_pick']


# In[ ]:


#=========
# Set the RF search params for each position
#=========

pos['QB']['tree_params'] = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 18, 21, 25, 30],
    'splitter': ['random']
}

pos['RB']['tree_params'] = {
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 18, 21, 25],
    'splitter': ['random']
}

pos['WR']['tree_params'] = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 20, 25, 30, 35],
    'splitter': ['random']
}


pos['TE']['tree_params'] = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 18, 22, 25, 30],
    'splitter': ['random']
}


# # Calculating Fantasy Points

# In[ ]:


def calculate_fp(df, pts, pos):
    
    # calculate fantasy points for QB's associated with a given RB or WR
    if pos == 'RB' or 'WR':
        df['qb_fp'] =         pts['pass_yd_pts']*df['qb_yds'] +         pts['pass_td_pts']*df['qb_tds'] +         pts['int_pts']*df['int'] +         pts['sack_pts']*df['qb_sacks']
    
    # calculate fantasy points for RB's
    if pos == 'RB':
        
        # create the points list corresponding to metrics calculated
        pts_list = [pts['yd_pts'], pts['yd_pts'], pts['rec_pts'], pts['td_pts']]
        
        df['fp'] =         pts['yd_pts']*df['rush_yds'] +         pts['yd_pts']*df['rec_yds'] +         pts['td_pts']*df['rush_td'] +         pts['td_pts']*df['rec_td'] +         pts['rec_pts']*df['receptions'] +         pts['fmb_pts']*df['fmb']
        
        # calculate fantasy points per touch
        df['fp_per_touch'] = df['fp'] / df['total_touches']
        
        # calculate fantasy points per target
        df['fp_per_tgt'] = df['fp'] / df['tgt']
    
    if pos == 'WR':
        
        # create the points list corresponding to metrics calculated
        pts_list = [pts['yd_pts'], pts['rec_pts'], pts['td_pts']]
        
        df['fp'] =         pts['yd_pts']*df['rec_yds'] +         pts['td_pts']*df['rec_td'] +         pts['rec_pts']*df['receptions']
        
        # calculate fantasy points per touch
        df['fp_per_tgt'] = df['fp'] / df['tgt']
        
    if pos == 'TE':
        
        # create the points list corresponding to metrics calculated
        pts_list = [pts['yd_pts'], pts['rec_pts'], pts['td_pts']]
        
        df['fp'] =         pts['yd_pts']*df['rec_yds'] +         pts['td_pts']*df['rec_td'] +         pts['rec_pts']*df['receptions']
        
        # calculate fantasy points per touch
        df['fp_per_tgt'] = df['fp'] / df['tgt']
        
    if pos == 'QB':
        
        # create the points list corresponding to metrics calculated
        pts_list = [pts['pass_yd_pts'], pts['pass_td_pts'], pts['yd_pts'],
                    pts['td_pts'], pts['int_pts'], pts['sack_pts']]
        
        df['fp'] =         pts['pass_yd_pts']*df['qb_yds'] +         pts['pass_td_pts']*df['qb_tds'] +         pts['yd_pts']*df['rush_yds'] +         pts['td_pts']*df['rush_td'] +         pts['int_pts']*df['int'] +         pts['sack_pts']*df['qb_sacks']
        
    # calculate fantasy points per game
    df['fp_per_game'] = df['fp'] / df['games']
    
    return df, pts_list


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


# # Visualization

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


def plot_results(results, col_names, asc=True, barh=True, fontsize=12):
    '''
    Input:  The feature importance or coefficient weights from a trained model.
    Return: A plot of the ordered weights, demonstrating relative importance of each feature.
    '''
    
    import pandas as pd
    import matplotlib.pyplot as plt

    # create series for plotting feature importance
    series = pd.Series(results, index=col_names, name='feature_rank').sort_values(ascending=asc)
    
    # find the max value and filter out any coefficients that less than 10% of the max
    max_val = abs(series).max()
    series = series[abs(series) > max_val*0.1]
    
    # auto determine the proper length of the figure
    figsize_length = int(round(len(series) / 5, 0))
    
    if barh == True:
        ax = series.plot.barh(figsize=(6, figsize_length), fontsize=fontsize)
        #ax.set_xlabel(label, fontsize=fontsize+1)
    else:
        ax = series.plot.bar(figsize=(6, figsize_length), fontsize=fontsize)
        #ax.set_ylabel(label, fontsize=fontsize+1)
        
    return ax


# # Pre-Model Feature Engineering

# In[ ]:


def corr_removal(df_train, df_predict, corr_cutoff=0.025):

    init_features = df_train.shape[1]
    
    corr = df_train.corr()['y_act']
    good_cols = list(corr[abs(corr) > corr_cutoff].index)

    good_cols.extend(['player', 'avg_pick', 'year'])
    df_train = df_train[good_cols]
    df_train = df_train.loc[:,~df_train.columns.duplicated()]

    good_cols.remove('y_act')
    df_predict = df_predict[good_cols]
    df_predict = df_predict.loc[:,~df_predict.columns.duplicated()]
    
    print('Corr removed ', init_features - df_train.shape[1], '/', init_features, ' features')
    
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
        
        num_cols = X.shape[1]
        
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
                
        print('Dropped ', num_cols - X.shape[1], '/', num_cols, ' columns')
                        
        return X


# # Ensemble Model

# In[ ]:


#=============
# Create parameter dictionaries for each algorithm
#=============

lgbm_params = {
    'n_estimators':[30, 40, 50, 60, 75],
    'max_depth':[2, 3, 4, 5, 6, 7],
    'feature_fraction':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'min_child_weight': [5, 10, 15, 20, 25],
}

xgb_params = {
    'n_estimators': [30, 40, 50, 60, 75], 
    'max_depth': [2, 3, 4, 5, 6, 7], 
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'min_child_weight': [10, 15, 20, 25, 30],
    'feature_fraction':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
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
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    
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
    figsize_length = int(round(len(to_plot) / 5, 0))
    to_plot.sort_values().plot.barh(figsize=(5, figsize_length));
    plt.show()
    
    return wt_predictions, models, 


# In[ ]:





# # Post-Model and Clustering

# In[ ]:


#==========
# Calculate fantasy points based on predictions and point values
#==========

def format_results(df_train_results, df_test_results, df_train, df_predict, pts_list):

    # calculate fantasy points for the train set
    df_train_results.iloc[:, 2:] = df_train_results.iloc[:, 2:] * pts_list
    df_train_results.loc[:, 'pred'] = df_train_results.iloc[:, 2:].sum(axis=1)

    # calculate fantasy points for the test set
    df_test_results.iloc[:, 1:] = df_test_results.iloc[:, 1:] * pts_list
    df_test_results.loc[:, 'pred'] = df_test_results.iloc[:, 1:].sum(axis=1)

    # add actual results and adp to the train df
    df_train_results = pd.merge(df_train_results, df_train[['player', 'year', 'age', 'avg_pick', 'y_act']],
                               how='inner', left_on=['player', 'year'], right_on=['player', 'year'])

    # add adp to the test df
    df_test_results = pd.merge(df_test_results, df_predict[['player', 'age', 'avg_pick']],
                               how='inner', left_on='player', right_on='player')

    # calculate the residual between the predictions and results
    df_train_results['error'] = df_train_results.pred - df_train_results.y_act
    
    return df_train_results, df_test_results


# In[ ]:


def searching(est, params, X_grid, y_grid, n_jobs=3, print_results=True):
    '''
    Function to perform GridSearchCV and return the test RMSE, as well as the 
    optimized and fitted model
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    
    Search = GridSearchCV(estimator=est,
                          param_grid=params,
                          scoring='neg_mean_squared_error',
                          n_jobs=n_jobs,
                          cv=5,
                          return_train_score=True,
                          iid=False)
   
    search_results = Search.fit(X_grid, y_grid)
   
    best_params = search_results.cv_results_['params'][search_results.best_index_]
    est.set_params(**best_params)
    
    test_rmse = cross_val_score(est, X_grid, y_grid, scoring='neg_mean_squared_error', cv=5)
    test_rmse = np.mean(np.sqrt(np.abs(test_rmse)))
    
    if print_results==True:
        print(best_params)
        print('Best RMSE: ', round(test_rmse, 3))
   
    est.fit(X_grid, y_grid)
       
    return est


# # Clustering Functions

# In[ ]:


class Clustering():

    def __init__(self, df_train, df_test):
    
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # create self versions of train and test
        self.df_train = df_train
        self.df_test = df_test
    
        # create df for clustering by selecting numeric values and dropping y_act
        self.X_train = df_train.select_dtypes(include=['float', 'int', 'uint8']).drop(['y_act', 'error'], axis=1)
        self.X_test = df_test.select_dtypes(include=['float', 'int', 'uint8']).drop([], axis=1)
        self.y = df_train.y_act
        
    def fit_and_predict_tree(self, print_results=True):
        
        from sklearn.tree import DecisionTreeRegressor
        
        self.tree = searching(DecisionTreeRegressor(random_state=1), pos['RB']['tree_params'], 
                              self.X_train, self.y, print_results=print_results)
        
        self.df_train = pd.merge(self.df_train, pd.DataFrame(self.tree.apply(self.X_train), columns=['cluster']), 
                                    how='inner', left_index=True, right_index=True)

        self.df_test = pd.merge(self.df_test, pd.DataFrame(self.tree.apply(self.X_test), columns=['cluster']),
                                how='inner', left_index=True, right_index=True)
        
        if print_results == True:
            print('Cluster List: ', list(self.df_test.cluster.unique()))
        
        
    def tree_plot(self):
 
        from sklearn.externals.six import StringIO 
        from IPython.display import Image 
        from sklearn.tree import export_graphviz
        import pydotplus

        dot_data = StringIO()

        export_graphviz(self.tree, out_file=dot_data, 
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names=self.X_train.columns)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
        nodes = graph.get_node_list()

        for node in nodes:
            if node.get_name() not in ('node', 'edge'):
                values = self.tree.tree_.value[int(node.get_name())][0]
                #color only nodes where only one class is present
                if values > 20 :   
                    node.set_fillcolor('#d74401')
                elif values > 16:   
                    node.set_fillcolor('#f06511')
                elif values > 12:   
                    node.set_fillcolor('#fdab67')
                elif values > 8:   
                    node.set_fillcolor('#b7cde2')
                else:
                    node.set_fillcolor('#3679a8')

        return Image(graph.create_png())
        
    
    def add_fit_metrics(self):
        
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        
        final_train = pd.DataFrame()
        final_test = pd.DataFrame()
        
        for j in list(self.df_train.cluster.unique()):
        
            test = self.df_test[self.df_test.cluster == j]
            train = self.df_train[self.df_train.cluster==j]
            
            try:
                # create a linear regression instance
                lr = LinearRegression()
                X=train.pred.values.reshape(-1,1)
                y=train.y_act

                # calculate rmse of predicted values vs. actual score
                scores = cross_val_score(lr, X, y, scoring='neg_mean_absolute_error', cv=X.shape[0])
                mae = np.mean(abs(scores))

                # update predictions based on actual score
                lr.fit(X, y)
                X_test = test.pred.values.reshape(-1,1)
                predictions = lr.predict(X_test)
                predictions_train = lr.predict(X)

                # output accuracy metrics and predictions
                rsq = round(lr.score(X,y), 3)
                test['cluster_pred'] = predictions
                test['rsq'] = rsq
                test['mae'] = mae

                rsq = round(lr.score(X,y), 3)
                train['cluster_pred'] = predictions_train
                train['rsq'] = rsq
                train['mae'] = mae

                final_test = pd.concat([final_test, test], axis=0)
                final_train = pd.concat([final_train, train], axis=0)
                
            except:
                pass
            
        self.df_test = final_test
        self.df_train = final_train
        
        return self.df_train, self.df_test
            
    def return_data(self):
        return self.df_train
    
    def show_results(self, j):
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        
        test = self.df_test[self.df_test.cluster == j]
        train = self.df_train[self.df_train.cluster==j]
        
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
        ax = train.y_act.plot.hist()
        ax.set_xlabel('Fantasy PPG Actual')
    
        # show plot of predicted vs actual points for cluster
        ax = train.plot.scatter('pred', 'y_act')
        ax.set_xlabel('Predicted Fantasy PPG')
        ax.set_ylabel('Actual Fantasy PPG')
        
        # show correlation coefficient between actual and predicted points for cluster
        print('')
        print('Pred to Actual Correlation')
        print(round(train.rsq.mean(), 3))
        
        # show examples of past players in cluster
        current = test.sort_values(by='pred', ascending=False)[['player', 'pred', 'cluster_pred']]
        
        return current
    
    
    def create_distributions(self, prior_repeats=15, show_plots=True):
        
        # historical standard deviation and mean for actual results
        hist_std = self.df_train.groupby('player').agg('std').dropna()
        hist_mean = self.df_train.groupby('player').agg('mean').dropna()
        
        # merge historicaly mean and standard deviations
        hist_mean_std = pd.merge(hist_std, hist_mean, how='inner', left_index=True, right_index=True)
        
        # calculate global coefficient of variance for players that don't have enough historical results
        global_cv = (hist_mean_std.y_act_x / hist_mean_std.y_act_y).mean()
        
        #==========
        # Loop to Create Prior and Posterior Distributions
        #==========

        self.df_test = self.df_test.sort_values(by='pred', ascending=False)

        results = pd.DataFrame()

        for player in self.df_test.player[0:]:

            # set seed
            np.random.seed(1234)

            # create list for results
            results_list = [player]

            #==========
            # Pull Out Predictions and Actual Results for Given Player to Create Prior
            #==========

            #--------
            # Extract this year's results and multiply by prior_repeats
            #--------

            # extract predictions from ensemble and updated predictions based on cluster fit
            ty = self.df_test.loc[self.df_test.player == player, ['player', 'pred']]
            #ty_c = self.df_test.loc[self.df_test.player == player, ['player', 'cluster_pred']]

            # replicate the predictions to increase n_0 for the prior
            ty = pd.concat([ty]*prior_repeats, ignore_index=True)
            #ty_c = pd.concat([ty_c]*prior_repeats, ignore_index=True)

            # rename the prediction columns to 'points'
            ty = ty.rename(columns={'pred': 'points'})
            #ty_c = ty_c.rename(columns={'cluster_pred': 'points'})

            #--------
            # Extract previous year's results, if available
            #--------

            # pull out the most recent 5 years worth of performance, if available
            py = self.df_train.loc[self.df_train.player == player, ['player', 'y_act']].reset_index(drop=True)[0:5]

            # convert y_act to points name
            py = py.rename(columns={'y_act': 'points'})

            #--------
            # Create Prior Distribution and Conjugant Hyperparameters
            #--------

            # combine this year's prediction, the cluster prediction, and previous year actual, if available
            priors = pd.concat([ty, py], axis=0)

            # set m_0 to the priors mean
            m_0 = priors.points.mean()

            # Create the prior variance through a weighted average of the actual previous year
            # performance and a global coefficient of variance multiple by the prior mean.
            # If there is not at least 3 years of previous data, simply use the global cv.
            if py.shape[0] >= 3:
                s2_0 = ((py.shape[0]*py.points.std()**2) + (2*prior_repeats*(m_0 * global_cv)**2)) / (py.shape[0] + 2*prior_repeats)
            else:
                s2_0 = (m_0 * global_cv)**2

            # set the prior sample size and degrees of freedom
            n_0 = priors.shape[0]
            v_0 = n_0 - 1

            # calculate the prior distribution
            prior_y = np.random.normal(loc=m_0, scale=np.sqrt(s2_0), size=10000)

            #--------
            # Create the Data and Data Hyperparameters
            #--------

            # pull out the cluster for the current player
            ty_cluster = self.df_test[self.df_test.player == player].cluster.values[0]

            # create a list of the actual points scored to be used as updating data
            update_data = self.df_train[self.df_train.cluster == ty_cluster].y_act

            # set ybar to the mean of the update data
            ybar = update_data.mean()

            # calculate the standard deviation based on the 5th and 95th percentiles
            s2 = ((np.percentile(update_data, q=95)-np.percentile(update_data, q=5)) / 4.0)**2

            # determine the n as the number of data points
            n = len(update_data)

            #--------
            # Create the Posterior Distribution 
            #--------

            # set the poster n samples
            n_n = n_0 + n 

            # update the poster mean
            m_n = (n*ybar + n_0*m_0) / n_n 

            # update the posterior degrees of freedom
            v_n = v_0 + n 

            # update the posterior variance
            s2_n = ((n-1)*s2 + v_0*s2_0 + (n_0*n*(m_0 - ybar)**2)/n_n)/v_n

            # calculate the gamma distribution and convert to sigma
            phi = np.random.gamma(shape=v_n/2, scale=2/(s2_n*v_n), size=10000)
            sigma = 1/np.sqrt(phi)

            # calculate the posterior mean
            post_mu = np.random.normal(loc=m_n, scale=sigma/(np.sqrt(n_n)), size=10000)

            # create the posterior distribution
            pred_y =  np.random.normal(loc=post_mu, scale=sigma, size=10000)

            results_list.extend(pred_y*16)
            results = pd.concat([results, pd.DataFrame(results_list).T], axis=0)

            if show_plots == True:

                # set plot bins based on the width of the distribution
                bins = int((np.percentile(pred_y, 97.5)*16 - np.percentile(pred_y, 2.55)*16) / 10)

                # print the player name
                print(player)

                # create a plot of the prior distribution and a line for the mean
                pd.Series(prior_y*16, name='Prior').plot.hist(alpha=0.4, color='grey', bins=bins, legend=True, 
                                                              xlim=[0, self.df_test.pred.max()*16*1.75])
                plt.axvline(x=prior_y.mean()*16, alpha=0.8, linestyle='--', color='grey')

                # create a plot of the posterior distribution and a line for the mean
                pd.Series(pred_y*16, name='Posterior').plot.hist(alpha=0.4, color='teal', bins=bins, legend=True,
                                                             edgecolor='black', linewidth=1)
                plt.axvline(x=pred_y.mean()*16, alpha=1, linestyle='--', color='black')

                # show the plots
                plt.show();

        return results.reset_index(drop=True)


# In[ ]:


class KMeansClustering():

    def __init__(self, df_train, df_test):
    
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # create self versions of train and test
        self.df_train = df_train
        self.df_test = df_test
    
        # create df for clustering by selecting numeric values and dropping y_act
        self.X_train = df_train.select_dtypes(include=['float', 'int', 'uint8']).drop(['y_act', 'error'], axis=1)
        self.X_test = df_test.select_dtypes(include=['float', 'int', 'uint8']).drop([], axis=1)
        self.y = df_train.y_act
        
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
    
        # add cluster results to the df_train and df_test
        self.df_train['cluster'] = self.train_results
        self.df_test['cluster'] = self.test_results


# In[ ]:


#===========
# Function to append distributions results to the database
#===========

def append_to_db(df, db_name='Season_Stats.sqlite3', table_name='NA', if_exist='append'):

    import sqlite3
    import os
    import datetime as dt
    
    #--------
    # Append pandas df to database in Github
    #--------

    os.chdir('/Users/Mark/Documents/Github/Fantasy_Football/Data/')

    conn = sqlite3.connect(db_name)

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )

    #--------
    # Append pandas df to database in OneDrive
    #--------

    os.chdir('/Users/Mark/OneDrive/FF/DataBase/')

    conn = sqlite3.connect(db_name)
    
    today = dt.datetime.today().strftime('%Y%m%d%H%M')

    df.to_sql(
    name=table_name + '_' + today,
    con=conn,
    if_exists=if_exist,
    index=False
    )

