
# coding: utf-8

# # User Inputs

# In[1]:


# set core path
path = '/Users/Mark/Documents/Github/Fantasy_Football/'

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_position = 'RB'

# set year to analyze
set_year = 2018
earliest_year = 2002

# set required touches (or pass thrown) and games for consideration
req_games = 8
req_touch = 50

# settings for fantasy points
pts = {}
pts['yd_pts'] = 0.1
pts['pass_yd_pts'] = 0.04
pts['td_pts'] = 7
pts['pass_td_pts'] = 5
pts['rec_pts'] = .5
pts['fmb_pts'] = -2.0
pts['int_pts'] = -2
pts['sack_pts'] = -1

# VIF threshold to include a feature
vif_thresh = 1000

# number of hypersearch rounds for training ensemble
iter_rounds = 50


# # Load Libraries

# In[2]:


# core packages
import pandas as pd
import numpy as np
import os

# jupyter specifications
pd.options.mode.chained_assignment = None
from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# plotting functions
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# change directory temporarily to helper scripts
os.chdir(path + 'Scripts/Analysis/Helper_Scripts')

# load custom plot functions
from my_plot import PrettyPlot
PrettyPlot(plt)

# load custom helper functions
from helper_functions import *;


# # Merge and Clean Data Files

# In[4]:


# load prepared data
df = pd.read_csv(path + 'Data/' + str(set_year) + '/' + set_position + '_Input.csv').iloc[:, 1:]

######
### tmp--need to add td per game to data generation ###
df['td_per_game'] = (df.rush_td + df.rec_td) / df.games
jeff_fisher = df[(df.player == 'Todd Gurley') & (df.year == 2016.0)].index
df = df.drop(jeff_fisher, axis=0).reset_index(drop=True)

df.loc[:, 'avg_pick'] = np.log(df.avg_pick)
df.loc[:, 'age'] = np.log(df.age)
######


# split old and new to filter past years based on touches.
# leave all new players in to ensure everyone gets a prediction
old = df[(df[pos['RB']['touch_filter']] > req_touch) & (df.games > req_games) & (df.year < set_year-1)].reset_index(drop=True)
this_year = df[df.year==set_year-1]

# merge old and new back together after filtering
df = pd.concat([old, this_year], axis=0)

# create dataframes to store results
df_train_results = pd.DataFrame([old.player, old.year]).T
df_test_results = pd.DataFrame([this_year.player]).T

# calculate FP
df = calculate_fp(df, pts, pos='RB')


# In[6]:


#==========
# Loop to create statistical predictions
#==========

metrics = ['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game']
output = {}

for metric in metrics:
    
    print('Running Models for ' + metric)
    print('----------------------------------')
    
    #--------
    # Create train and predict dataframes
    #--------
    df_train, df_predict = features_target(df, 
                                           earliest_year, set_year-1, 
                                           pos['RB']['med_features'], 
                                           pos['RB']['sum_features'],
                                           pos['RB']['max_features'], 
                                           pos['RB']['age_features'],
                                           target_feature=metric)

    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    df_train = df_train.fillna(df_train.mean())
    df_predict = df_predict.dropna().reset_index(drop=True)

    #--------
    # Remove low correlation features and high VIF features
    #--------

    # remove low correlation features
    df_train, df_predict = corr_removal(df_train, df_predict, corr_cutoff=0.05)

    # select only features with low vif for modeling
    transformer = ReduceVIF(thresh=vif_thresh, scale=True, print_progress=False)
    df_train_ = transformer.fit_transform(df_train.drop(['y_act', 'player'], axis=1), df_train.y_act)

    # extract best columns and filter down df_predict
    best_cols = list(df_train_.columns)
    best_cols.extend(['player', 'avg_pick'])
    df_predict = df_predict[best_cols]
    df_predict = df_predict.loc[:,~df_predict.columns.duplicated()]

    # add target and filter down df_train
    best_cols.extend(['y_act', 'year', 'avg_pick'])
    df_train = df_train[best_cols]
    df_train = df_train.loc[:,~df_train.columns.duplicated()]

    #--------
    # Run ensemble model with parameter optimization
    #--------

    # generate a master dictionary of parameters (must match the)
    param_list = [lgbm_params, xgb_params, lasso_params, ridge_params]
    est_names = ['lgbm', 'xgb', 'lasso', 'ridge']

    params = {}
    for i, param in enumerate(param_list):
        params[est_names[i]] = param
    
    print('Training Ensemble Model')
    param_results, summary, df_train_results_, errors = validation(est_names, params, df_train, iterations=iter_rounds, random_state=1234)
    
    #--------
    # Print best results
    #--------
    
    # print a summary of error metrics, weightings of various models, and a comparison to 
    # using straight adp as as a prediction for next year's stats
    print(summary.head(10))
    
    # pull out the best result for the random hyperparameter search of models
    best_result = summary.index[0]
    
    # pass the best hyperparameters into the generation_prediction function, which
    # will return the test results for the current year and the trained best models
    df_test_results_, models = generate_predictions(best_result, param_results, summary, df_train, df_predict)
    
    #--------
    # Aggregate all results through merging
    #--------
    
    # add models to output dictionary
    output[metric] = {}
    output[metric]['models'] = models
    
    # add params to output dictionary
    output[metric]['params'] = param_results
    
    # add columns to output dictionary
    cols = list(df_train.columns)
    cols.remove('y_act')
    cols.remove('player')
    output[metric]['cols'] = cols
    
    # merge the train results for the given metric with all other metric outputs
    df_train_results_ = df_train_results_.rename(columns={'pred': 'pred_' + metric})
    df_train_results = pd.merge(df_train_results, df_train_results_[['player', 'year','pred_' + metric]], 
                                how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
    
    # merge the test results for the given metric with all other metric outputs
    df_test_results_ = df_test_results_.rename(columns={'pred': 'pred_' + metric})
    df_test_results = pd.merge(df_test_results, df_test_results_[['player', 'pred_' + metric]], 
                               how='inner', left_on='player', right_on='player')
    
# after loop, set the df_train to have the y_act as fp_per_game
df_train, df_predict = features_target(df, earliest_year, set_year-1, 
                                           pos['RB']['med_features'], 
                                           pos['RB']['sum_features'],
                                           pos['RB']['max_features'], 
                                           pos['RB']['age_features'],
                                           target_feature='fp_per_game')


# In[ ]:


#==========
# If desired, plot feature importances for a given metric / model
#==========

metric = 'td_per_game'
i = 2
try:
    plot_results(output[metric]['models'][i].feature_importances_, col_names=output[metric]['cols']);
except:
    plot_results(output[metric]['models'][i].coef_, col_names=output[metric]['cols']);


# In[7]:


#==========
# Calculate fantasy points based on predictions and point values
#==========

# create list of fantasy point metrics for each stat category
pts_list = [pts['yd_pts'], pts['yd_pts'], pts['rec_pts'], pts['td_pts']]

# calculate fantasy points for the train set
df_train_results.iloc[:, 2:] = df_train_results.iloc[:, 2:] * pts_list
df_train_results.loc[:, 'pred'] = df_train_results.iloc[:, 2:].sum(axis=1)

# calculate fantasy points for the test set
df_test_results.iloc[:, 1:] = df_test_results.iloc[:, 1:] * pts_list
df_test_results.loc[:, 'pred'] = df_test_results.iloc[:, 1:].sum(axis=1)

# calculate projected FP per game based on individual statistcal predictions
df = calculate_fp(df, pts, pos='RB')

# add actual results and adp to the train df
df_train_results = pd.merge(df_train_results, df_train[['player', 'year', 'avg_pick', 'y_act']],
                           how='inner', left_on=['player', 'year'], right_on=['player', 'year'])

# add adp to the test df
df_test_results = pd.merge(df_test_results, df_predict[['player', 'avg_pick']],
                           how='inner', left_on='player', right_on='player')

# calculate the residual between the predictions and results
df_train_results['error'] = df_train_results.pred - df_train_results.y_act


# In[8]:


# plot the predictions in descending order
df_test_results.sort_values('pred').plot.barh(x='player', y='pred', figsize=(5,15));


# # Clustering Players into Tiers

# In[19]:


from sklearn.preprocessing import MinMaxScaler

class clustering():

    def __init__(self, df_train, df_test):
    
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # create self versions of train and test
        self.df_train = df_train
        self.df_test = df_test
    
        # create df for clustering by selecting numeric values and dropping y_act
        self.X_train = df_train.select_dtypes(include=['float', 'int', 'uint8']).drop(['y_act', 'avg_pick', 'error'], axis=1)
        self.X_test = df_test.select_dtypes(include=['float', 'int', 'uint8']).drop(['avg_pick'], axis=1)
        
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
        
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        
        # add cluster results to the df_train and df_test
        self.df_train['cluster'] = self.train_results
        self.df_test['cluster'] = self.test_results
        
        final_train = pd.DataFrame()
        final_test = pd.DataFrame()
        
        for j in range(0, self.k):
        
            test = self.df_test[self.df_test.cluster == j]
            train = self.df_train[self.df_train.cluster==j]
        
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
            
        self.df_test = final_test
        self.df_train = final_train
        
        return self.df_train, self.df_test
            
    
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
        print(train.rsq.mean())
        
        # show examples of past players in cluster
        current = test.sort_values(by='pred', ascending=False)[['player', 'pred', 'cluster_pred']]
        
        return current
    
    
    def create_distributions(self, wt=2.5):
        from scipy.stats import skewnorm
        from scipy.stats import pearsonr
        import numpy as np
        
        #==========
        # Calculate Global Mean and Stand Deviation for Actual Results
        #==========
        
        # historical standard deviation and mean for actual results
        hist_std = df_train_results.groupby('player').agg('std').dropna()
        hist_mean = df_train_results.groupby('player').agg('mean').dropna()
        
        # merge historicaly mean and standard deviations
        hist_mean_std = pd.merge(hist_std, hist_mean, how='inner', left_index=True, right_index=True)
        
        # calculate global coefficient of variance for players that don't have enough historical results
        global_cv = (hist_mean_std.y_act_x / hist_mean_std.y_act_y).mean()
        
        #==========
        # Calculate Global Mean and Stand Deviation for Actual Results
        #==========
        
        return rb_sampling


# In[20]:


cluster = clustering(df_train_results, df_test_results)
cluster.explore_k(k=15)


# In[21]:


cluster.fit_and_predict(k=8)
c_train, c_test = cluster.add_clusters()


# # Tier 1

# In[29]:


cluster.show_results(j=6)


# In[30]:


# historical standard deviation and mean for actual results
hist_std = df_train_results.groupby('player').agg('std').dropna()
hist_mean = df_train_results.groupby('player').agg('mean').dropna()
        
# merge historicaly mean and standard deviations
hist_mean_std = pd.merge(hist_std, hist_mean, how='inner', left_index=True, right_index=True)
        
# calculate global coefficient of variance for players that don't have enough historical results
global_cv = (hist_mean_std.y_act_x / hist_mean_std.y_act_y).mean()


# In[62]:


#==========
# Loop to Create Prior and Posterior Distributions
#==========

prior_repeats = 5

for player in df_test_results.player[0:]:
    
    # set seed
    np.random.seed(1234)
    
    #==========
    # Pull Out Predictions and Actual Results for Given Player to Create Prior
    #==========
    
    #--------
    # Extract this year's results and multiply by prior_repeats
    #--------
    
    # extract predictions from ensemble and updated predictions based on cluster fit
    ty = c_test.loc[c_test.player == player, ['player', 'pred']]
    ty_c = c_test.loc[c_test.player == player, ['player', 'cluster_pred']]
    
    # replicate the predictions to increase n_0 for the prior
    ty = pd.concat([ty]*prior_repeats, ignore_index=True)
    ty_c = pd.concat([ty_c]*prior_repeats, ignore_index=True)
    
    # rename the prediction columns to 'points'
    ty = ty.rename(columns={'pred': 'points'})
    ty_c = ty_c.rename(columns={'cluster_pred': 'points'})
    
    #--------
    # Extract previous year's results, if available
    #--------
    
    py = c_train.loc[c_train.player == player, ['player', 'y_act']].reset_index(drop=True)[0:5]
    py = py.rename(columns={'y_act': 'points'})
    priors = pd.concat([ty, ty_c, py], axis=0)
    
    m_0 = priors.points.mean()
    if py.shape[0] >= 3:
        s2_0 = ((py.shape[0]*py.points.std()**2) + (2*prior_repeats*(m_0 * global_cv)**2)) / (py.shape[0] + 2*prior_repeats)
    else:
        s2_0 = (m_0 * global_cv)**2
    n_0 = priors.shape[0]
    v_0 = n_0 - 1
    
    
    ty_cluster = df_test_results[df_test_results.player ==player].cluster.values[0]
    update_data = df_train_results[df_train_results.cluster == ty_cluster].y_act
    ybar = update_data.mean()
    s2 = ((np.percentile(update_data, q=95)-np.percentile(update_data, q=5)) / 4.0)**2
    n = len(update_data)
    
    # posterior hyperparamters 
    n_n = n_0 + n 
    m_n = (n*ybar + n_0*m_0) / n_n 
    v_n = v_0 + n 
   
    s2_n = ((n-1)*s2 + v_0*s2_0 + (n_0*n*(m_0 - ybar)**2)/n_n)/v_n
    
    phi = np.random.gamma(shape=v_n/2, scale=2/(s2_n*v_n), size=10000)
    sigma = 1/np.sqrt(phi)
    post_mu = np.random.normal(loc=m_n, scale=sigma/(np.sqrt(n_n)), size=10000)
    
    prior_y = np.random.normal(loc=m_0, scale=np.sqrt(s2_0), size=10000)
    pred_y =  np.random.normal(loc=post_mu, scale=sigma, size=10000)
    
    print(player)
    print(s2_0)
    print(s2_n)
    pd.Series(prior_y, name='Prior').plot.hist(alpha=0.5, color='grey', bins=20, legend=True, xlim=[0, 30])
    plt.axvline(x=prior_y.mean(), alpha=0.7, linestyle='--', color='grey')
    pd.Series(pred_y, name='Posterior').plot.hist(alpha=0.4, color='teal', bins=20, legend=True)
    plt.axvline(x=pred_y.mean(), alpha=0.5, linestyle='--', color='teal')

    plt.show();


# In[ ]:


rb_sampling.to_csv('/Users/Mark/Desktop/Jupyter Projects/Fantasy Football/Projections/rb_sampling.csv')

