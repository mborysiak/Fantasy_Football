import pandas as pd
import numpy as np
import sqlite3

#=========
# Set the RF search params for each position
#=========

pos = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {}}

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
    
    
    def create_distributions(self, prior_repeats=15, dist_size=1000, show_plots=False):
        
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
            prior_y = np.random.normal(loc=m_0, scale=np.sqrt(s2_0), size=dist_size)

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
            phi = np.random.gamma(shape=v_n/2, scale=2/(s2_n*v_n), size=dist_size)
            sigma = 1/np.sqrt(phi)

            # calculate the posterior mean
            post_mu = np.random.normal(loc=m_n, scale=sigma/(np.sqrt(n_n)), size=dist_size)

            # create the posterior distribution
            pred_y =  np.random.normal(loc=post_mu, scale=sigma, size=dist_size)

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

def custom_data(db_name, set_year, pts_dict, user_id, prior_repeats=5, dist_size=1000, show_plots=False):

    '''
    The initialization of this Class reads in all of the statistical projection data and
    translates it into clusters and projection distributions given a particular scoring schema.
    The data is then stored in the self.data object, which will be accessed through the analysis.

    Input: A database that contains statistical projections, a dictionary that contains the points
           for each category, and number of prior repeats to use for Bayesian updating.
    Return: Stores all the player projection distributions in that self.data object.
    '''

    # create empty dataframe to store all player distributions
    data = pd.DataFrame()

    # ==========
    # Loop through each position and pull / analyze the data
    # ==========

    for pos in ['aQB', 'bRB', 'cWR', 'dTE']:
        # print current position update
        print('Loading and Preparing ' + pos[1:] + ' Data')

        # --------
        # Connect to Database and Pull Player Data
        # --------

        conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Website/' + db_name)

        df_train_results = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Train_Results_' + str(set_year),
                                             con=conn)
        df_test_results = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Test_Results_' + str(set_year), con=conn)
        df_train = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Train_' + str(set_year), con=conn)
        df_predict = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Predict_' + str(set_year), con=conn)

        # --------
        # Calculate Fantasy Points for Given Scoring System and Cluster
        # --------

        # pull in data results from dataframe
        df_train_results, df_test_results = format_results(df_train_results, df_test_results,
                                                           df_train, df_predict,
                                                           pts_dict[pos[1:]])
        df_train_results = df_train_results.drop('year', axis=1)

        # initialize cluster with train and test results
        cluster = Clustering(df_train_results, df_test_results)

        # fit decision tree and apply nodes to players
        cluster.fit_and_predict_tree(print_results=False)

        # --------
        # Use Bayesian Updating to Create Points Distributions
        # --------

        # create distributions of data
        distributions = cluster.create_distributions(prior_repeats=prior_repeats,
                                                     dist_size=1000,
                                                     show_plots=show_plots)

        # add position to the distributions
        distributions['pos'] = pos

        # append each position of data to master dataset
        data = pd.concat([data, distributions], axis=0)

    # add flex data
    flex = data[data.pos.isin(['bRB', 'cWR', 'dTE'])]
    flex['pos'] = 'eFLEX'
    data = pd.concat([data, flex])

    # format the self.data for later use
    data = data.reset_index(drop=True)
    data = data.rename(columns={0: 'player'})

    data['user_id'] = user_id
    print(data.shape)

    data.to_sql('Session_Data', con=conn, if_exists='replace')
