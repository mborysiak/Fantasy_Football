# core packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas.io.formats.style
import copy

# sql packages
import sqlalchemy
import psycopg2
from sqlalchemy import create_engine

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp
from scipy.stats import skewnorm

class FootballSimulation():

    #==========
    # Creating Player Distributions for Given Settings
    #==========
    
    def __init__(self, pts_dict, table_info, set_year, iterations):
        
        # create empty dataframe to store player point distributions
        tree_output = pd.DataFrame()
        for pos in ['aQB', 'bRB', 'cWR', 'dTE']:

            # extract the train and test data for passing into the tree algorithm
            train, test = self._data_pull(pts_dict, pos, table_info, set_year)

            # obtain the cluster standard deviation + the mixed prediction / cluster mean
            results = self._tree_cluster(train, test, pos)

            # append the results for each position into single dataframe
            tree_output = pd.concat([tree_output, results], axis=0)

        tree_output = tree_output.reset_index(drop=True)

        # loop through each row in tree output and create a normal distribution
        data_list = []
        for row in tree_output.iterrows():
            dist = list(np.uint16(np.random.normal(loc=row[1]['PredMean'], 
                                                   scale=row[1]['ClusterStd'], size=iterations*2).clip(min=0)*16))
            data_list.append(dist)

        # create the player, position, point distribution dataframe
        self.data = pd.concat([tree_output.player, pd.DataFrame(data_list), tree_output.pos], axis=1)

        # add salaries to the dataframe and set index to player
        salaries = pd.read_sql_query('SELECT * FROM {}."salaries_{}"'.format(table_info['schema'], str(set_year)), table_info['engine'])
        self.sal = salaries
        self.data = pd.merge(self.data, salaries, how='inner', left_on='player', right_on='player')
        
        # pull in injury risk information
        self.inj = pd.read_sql_query('SELECT * FROM {}."injuries_{}"'.format(table_info['schema'], str(set_year)), table_info['engine'])
        
        # add flex data
        flex = self.data[self.data.pos.isin(['bRB', 'cWR', 'dTE'])]
        flex.loc[:, 'pos'] = 'eFLEX'
        self.data = pd.concat([self.data, flex])
        
        # reset index
        self.data = self.data.set_index('player')


    @staticmethod
    def _data_pull(pts_dict, pos, table_info, set_year=2018):

        '''
        This function reads in all raw statistical predictions from the ensemble model for a given
        position group and then converts it into predicted points scored based on a given scoring system.

        Input: Database connection to pull stored raw statistical data, a dictionary containing points
               per statistical category, and a position to pull.
        Return: A dataframe with a player, their raw statistical projections and the predicted points
                scored for a given scoring system.
        '''

        import pandas as pd

        #--------
        # Connect to Database and Pull Player Data
        #--------

        train = pd.read_sql_query('SELECT * FROM {}."{}_Train_{}"' \
                                             .format(table_info['schema'], pos[1:], str(set_year)), table_info['engine'])
        test = pd.read_sql_query('SELECT * FROM {}."{}_Test_{}"' \
                                            .format(table_info['schema'], pos[1:], str(set_year)), table_info['engine'])

        #--------
        # Calculate Fantasy Points for Given Scoring System
        #-------- 

        # extract points list and get the idx of point attributes based on length of list
        pts_list = pts_dict[pos[1:]]
        c_idx = len(pts_list) + 2

        # calculate the y_act metrics based on actual stat categories and provided pts list
        act_cols = [c for c in train.columns if 'act' in c]
        train['y_act'] = (train[act_cols] * pts_list).sum(axis=1)
        train = train.drop(act_cols, axis=1)
        
        # multiply stat categories by corresponding point values
        pred_cols = [c for c in train.columns if 'pred' in c]
        train.loc[:, 'pred'] = (train[pred_cols] * pts_list).sum(axis=1)
        test.loc[:, 'pred'] = (test[pred_cols] * pts_list).sum(axis=1)
        
        return train, test

    @staticmethod
    def _searching(est, pos, X_grid, y_grid, n_jobs=1):
        '''
        Function to perform GridSearchCV and return the test RMSE, as well as the 
        optimized and fitted model
        '''
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import cross_val_score

         #=========
        # Set the RF search params for each position
        #=========

        params = {}

        params['QB'] = {
            'max_depth': [4, 5],
            'min_samples_split': [2],
            'min_samples_leaf': [15, 20, 25],
            'splitter': ['random']
        }

        params['RB'] = {
            'max_depth': [5, 6, 7],
            'min_samples_split': [2],
            'min_samples_leaf': [15, 20, 25],
            'splitter': ['random']
        }

        params['WR'] = {
            'max_depth': [4, 5, 6],
            'min_samples_split': [2],
            'min_samples_leaf': [20, 25, 30],
            'splitter': ['random']
        }


        params['TE'] = {
            'max_depth': [4, 5],
            'min_samples_split': [2],
            'min_samples_leaf': [15, 20, 25],
            'splitter': ['random']
        }

        # set up GridSearch object
        Search = GridSearchCV(estimator=est,
                              param_grid=params[pos[1:]],
                              scoring='neg_mean_squared_error',
                              n_jobs=n_jobs,
                              cv=3,
                              return_train_score=False,
                              iid=False)

        # try all combination of parameters with the fit
        search_results = Search.fit(X_grid, y_grid)

        # extract best estimator parameters and create model object with them
        best_params = search_results.cv_results_['params'][search_results.best_index_]
        est.set_params(**best_params)

        # fit the optimal estimator with the data
        est.fit(X_grid, y_grid)

        return est


    def _tree_cluster(self, train, test, pos):

        # create df for clustering by selecting numeric values and dropping y_act
        X_train = train.select_dtypes(include=['float', 'int', 'uint8']).drop('y_act', axis=1)
        X_test = test.select_dtypes(include=['float', 'int', 'uint8'])
        y = train.y_act

        #----------
        # Train the Decision Tree with GridSearch optimization
        #----------

        from sklearn.tree import DecisionTreeRegressor

        # train decision tree with _searching method
        dtree = self._searching(DecisionTreeRegressor(random_state=1), pos, X_train, y)

        #----------
        # Calculate each cluster's mean and standard deviation
        #----------

        # pull out the training clusters and cbind with the actual points scored
        train_results = pd.concat([pd.Series(dtree.apply(X_train), name='Cluster'), y], axis=1)

        # calculate the average and standard deviation of points scored by cluster
        train_results = train_results.groupby('Cluster', as_index=False).agg({'y_act': ['mean', 'std']})
        train_results.columns = ['Cluster', 'ClusterMean', 'ClusterStd']

        #----------
        # Add the cluster to test results and resulting group mean / std: Player | Pred | StdDev
        #----------

        # grab the player, prediction, and add cluster to dataset
        test_results = pd.concat([test[['player', 'pred']], 
                                  pd.Series(dtree.apply(X_test), name='Cluster')], axis=1)

        # merge the test results with the train result on cluster to add mean cluster and std
        test_results = pd.merge(test_results, train_results, how='inner', left_on='Cluster', right_on='Cluster')

        # calculate an overall prediction mean and add position to dataset
        test_results.loc[:, 'PredMean'] = (0.75*test_results.pred + 0.25*test_results.ClusterMean)
        test_results.loc[:, 'pos'] = pos

        # pull out relevant results for creating distributions
        test_results = test_results[['player', 'pos', 'PredMean', 'ClusterStd']]
        
        return test_results

    
    def return_data(self):
        '''
        Returns self.data if necessary.
        '''
        return self.data
    
    
    #==========
    # Running the Simulation for Given League Settings and Keepers
    #==========
    
    def run_simulation(self, league_info, to_drop, to_add, iterations=500):        
        '''
        Method that runs the actual simulation and returns the results.
        
        Input: Projected player data and salaries with variance, the league 
               information (e.g. position requirements, and salary caps), and 
               information about players selected by your team and other teams.
        Returns: The top team results (players selected and salaries), as well
                 as counts of players selected, their salary they were selected at,
                 and the points the scored when selected.
        '''
        #--------
        # Pull Out Data, Salaries, and Inflation for Current Simulation
        #--------
        
        # create a copy of self.data for current iteration settings
        data = self.data.copy()
        league_info = copy.deepcopy(league_info)
        inj = self.inj.copy()
        
        # pull out min salary from data
        min_salary = data.salary.min()
        
        # if the total + the minimum salary 
        if np.sum(to_add['salaries']) + min_salary > league_info['salary_cap']:
            print('''The selected salaries equal {}, the cheapest projected
                     player costs {}, and the max salary cap is {}.  
                     
                     As a result, no player is able to be selected from the optimization.
                     Select lower salaries or increase the salary cap to continue.'''.format(np.sum(to_add['salaries']), 
                                                                                             min_salary,
                                                                                             league_info['salary_cap']))
            return [], []
        
        # give an extra dollar to prevent errors with minimum salaried players
        league_info['salary_cap'] = league_info['salary_cap'] + 1
       
        # drop other selected players + calculate inflation metrics
        data, drop_proj_sal, drop_act_sal = self._drop_players(data, league_info, to_drop)
        
        # drop your selected players + calculate inflation metrics
        data, league_info, to_add, add_proj_sal, add_act_sal = self._add_players(data, league_info, to_add)
        
        pos_require = list(league_info['pos_require'].values())
        
        # calculate inflation based on drafted players and their salaries
        inflation, total_cap = self._calc_inflation(league_info, drop_proj_sal, drop_act_sal, add_proj_sal, add_act_sal)
        print('Current Inflation:', round(inflation, 2))

        # determine salaries, skew distributions, and number of players for each position
        data, salaries, salary_skews, pos_counts = self._pull_salary_poscounts(data, inflation, iterations)
        
        # calculate the injury distributions
        data, inj_dist, replace_val = self._injury_replacement(data, league_info, inj, iterations)

        #--------
        # Initialize Matrix and Results Dictionary for Simulation
        #--------
        
        # generate the A matrix for the simulation constraints
        A = self._Amatrix(pos_counts, league_info['pos_require'])

        # pull out the names of all players and set to names
        names = data.index
        dict_names = list(data.index)
        dict_names.extend(to_add['players'])
        
        # create empty matrices
        results = {}
        results['names'] = []
        results['points'] = []
        results['salary'] = []

        # create empty dictionaries
        counts = {}
        counts['names'] = pd.Series(0, index=dict_names).to_dict()
        counts['points'] = pd.Series(0, index=dict_names).to_dict()
        counts['salary'] = pd.Series(0, index=dict_names).to_dict()
        
        # shuffle the random data--both salary skews and the point projections
        _ = [np.random.shuffle(row) for row in salary_skews]
        inj_dist = self._df_shuffle(inj_dist)
        data = self._df_shuffle(data)
                
        #--------
        # Run the Simulation Loop
        #--------
            
        trial_counts = 0
        for i in range(0, iterations):
    
            # every N trials, randomly shuffle each run in salary skews and data
            if i % (iterations / 10) == 0:
                _ = [np.random.shuffle(row) for row in salary_skews]
                inj_dist = self._df_shuffle(inj_dist)
                data = self._df_shuffle(data)
            
            # pull out a random selection of points and salaries
            points, salaries_tmp = self._random_select(data, salaries, salary_skews, 
                                                       inj_dist, replace_val, iterations)
            
            # adjust salaries to ensure they don't go over the total cap
            # first sum of the salaries of unique players (exclude flex)
            sum_salaries = np.sum(salaries_tmp[:len(data.index.unique())])
            
            # sum up the salaries with total already spent
            all_money = add_act_sal + drop_act_sal + sum_salaries
            
            # calculate an inflation or deflation metric and multiply by salaries
            factor = total_cap / all_money
            salaries_tmp = salaries_tmp * factor

            # run linear integer optimization
            x = self._run_opt(A, points, salaries_tmp, league_info['salary_cap'], pos_require)

            # pull out and store the selected names, points, and salaries
            results, self.counts, trial_counts = self._pull_results(x, names, points, salaries_tmp, 
                                                                    to_add, results, counts, trial_counts)
        
        # format the results after the simulation loop is finished
        self.results = self._format_results(results)

        # add self version of variable for output calculations
        self._inflation = inflation
        self._sal = self.data.reset_index()[['player', 'salary']].drop_duplicates().set_index('player')

        return self.counts
    
    #==========
    # Helper Functions for the Simulation Loop
    #==========

    #--------
    # Salary (+Inflation) and Keeper Setup
    #--------
    
    def add_salaries(self, salaries):
        '''
        Input: Salaries for all players in the dataset.
        Return: The self.data dataframe that has salaries appended to it.
        '''
        #--------
        # Merge salaries and points on names to ensure matches
        #--------
        
        # merge the salary and prediction data together on player
        self.data = pd.merge(self.data, salaries, how='inner', left_on='player', right_on='player')
        
        # sort values and move player to the index of the dataframe
        self.data = self.data.sort_values(by=['pos', 'salary'], ascending=[True, False]).set_index('player')
    
    
    @staticmethod
    def _drop_players(data, league_info, to_drop):
        '''
        Drops a list of players that are chosen as by other teams and calculates actual 
        salary vs. expected salary for inflation calculation.
        
        Input: Data for a given simulation run, league information (e.g. total salary cap), 
               and a dictionary of players with their salaries to drop. 
        Return: The players that remain available for the simulation, along with metrics
                for salary inflation.
        '''
        
        #--------
        # Dropping Other Team's Players
        #--------
        
        # find players from data that will be dropped and remove them from other data
        drop_data = data[data.index.isin(to_drop['players'])]
        other_data = data.drop(drop_data.index, axis=0)
        
        # pull out the projected and actual salaries for the players that are being kept
        drop_proj_salary = drop_data.salary.reset_index().drop_duplicates().sum().salary
        drop_act_salary = np.sum(to_drop['salaries'])

        return other_data, drop_proj_salary, drop_act_salary
    
    
    @staticmethod
    def _add_players(data, league_info, to_add):
        '''
        Removes a list of players that are chosen as to_add and calculates inflation based
        on their added salary vs. expected salary.
        
        Input: Data for a given simulation run, league information (e.g. total salary cap), 
               and a dictionary of players with their salaries to keep. 
        Return: The players that remain available for the simulation, the players to be kept,
                and metrics to calculate salary inflation.
        '''
        
        # pull data for players that have been added to your team and split out other players
        add_data = data[data.index.isin(to_add['players'])]
        other_data = data.drop(add_data.index, axis=0)

        # pull out the salaries of your players and sum
        add_proj_salary =  add_data.salary.reset_index().drop_duplicates().sum().salary
        add_act_salary = np.sum(to_add['salaries'])

        # update the salary for your team to subtract out drafted player's salaries
        league_info['salary_cap'] = float(league_info['salary_cap'] - add_act_salary)
        print('Remaining Salary:', league_info['salary_cap'])
        
        # add the mean points scored by the players who have been added
        to_add['points'] = -1.0*(add_data.drop(['pos', 'salary'],axis=1).mean(axis=1).values)
        
        # create list of letters to append to position for proper indexing
        letters = ['a', 'b', 'c', 'd', 'e']

        # loop through each position in the pos_require dictionary
        for i, pos in enumerate(league_info['pos_require'].keys()):

            # create a unique label based on letter and position
            pos_label = letters[i]+pos

            # loop through each player that has been selected  
            for player in list(add_data[add_data.pos==pos_label].index):

                # if the position is still required to be filled:
                if league_info['pos_require'][pos] > 0:

                    # subtract out the current player from the position count
                    league_info['pos_require'][pos] = league_info['pos_require'][pos] - 1

                    # and remove that player from consideration for filling other positions
                    add_data = add_data[add_data.index != player]
        
        print('Remaining Positions Required:', league_info['pos_require'])
        return other_data, league_info, to_add, add_proj_salary, add_act_salary
    
    
    @staticmethod
    def _calc_inflation(league_info, drop_proj_sal, drop_act_sal, add_proj_sal, add_act_sal):
        '''
        Method to calculate inflation based on players selected and the expected salaries.
        '''
        # add up the total actual and projected salaries for all keepers
        projected_salary = drop_proj_sal + add_proj_sal
        actual_salary = drop_act_sal + add_act_sal
        
        # calculate the salary inflation due to the keepers
        total_cap = league_info['num_teams'] * league_info['initial_cap']
        inflation = (total_cap-actual_salary) / (total_cap-projected_salary)
        
        return inflation, total_cap
        
        
    def _pull_salary_poscounts(self, data, inflation, iterations):
        '''
        Method to pull salaries from the data dataframe, create salary skews, and determine
        the position counts for the A matrix in the simulation
        
        Input: Data for current simulation and inflation metric
        Return: The data without salary column, the inflated salary numpy array, a dataframe of salaru
                skews for current simulation, and a count of positions in the dataframe 
        '''
        #--------
        # Extract salaries into numpy array and drop salary from points data
        #--------

        # set salaries to numpy array and multiply by inflation
        salaries = data.salary.values*inflation

        # calculate salary skews for each player's salary
        salary_skews = self._skews(salaries, iterations*2)

        # extract the number of counts for each position for later creating A matrix
        pos_counts = list(data.pos.value_counts().sort_index())
        
        return data, salaries, salary_skews, pos_counts
        
        
    @staticmethod
    def _skews(salaries, iterations):
        '''
        Input: Internal method that accepts the salaries input for each player in the dataset.
        Return: Right skewed salary uncertainties, scaled to the actual salary of the player.
        '''
        # pull out the salary column and convert to numpy array
        _salaries = salaries.reshape(-1,1)

        # create a skews normal distribution of uncertainty for salaries
        skews = (skewnorm.rvs(5, size=iterations*2)*.1).reshape(1, -1)

        # create a p x m matrix with dot product, where p is the number of players
        # and m is the number of skewed uncertainties, e.g. 320 players x 10000 skewed errors
        salary_skews = np.dot(_salaries, skews)

        return salary_skews
    
    
    @staticmethod
    def _injury_replacement(data, league_info, inj, iterations):
        '''
        Input: Points dataset, league information, and injury risk, which are used to
               calculate the replacement values for each position, as well as the injury
               risk distribution for each player.
        Output: The points dataset with replacement level points attached and a separate
                poisson distribution of expected games missed for each player.
        '''
        #----------
        # Calculate Replacement Level Value
        #----------
        
        # calculate the mean points predicted for each player
        mean_pts = data.iloc[:, 0:iterations*2].mean(axis=1).reset_index()
        mean_pts = pd.concat([data.pos.reset_index(drop=True), mean_pts], axis=1)
        mean_pts = mean_pts.sort_values(by=['pos', 0], ascending=[True, False])

        # loop through each position and calculate the replacement level value
        _pos = ['aQB', 'bRB', 'cWR', 'dTE', 'eFLEX']
        replacement = []

        for i, key_val in enumerate(league_info['pos_require'].items()):

            # use the positional requirement to determine replacement level
            num = key_val[1] * league_info['num_teams']
            num = int(num + (num/2))

            # calculate the replacement level for each position
            replace = mean_pts[mean_pts.pos==_pos[i]].iloc[num:, 2].median() / 16
            replacement.append([_pos[i], replace])

        # convert replace values to dataframe and merge with all data to add replacement column
        replacement = pd.DataFrame(replacement, columns=['pos', 'replace_val'])
        data = pd.merge(data.reset_index(), replacement, how='left', left_on='pos', right_on='pos').set_index('player')
        replace_val = data.replace_val
        
        # drop salary from the points dataframe and reset the columns from 0 to N-1, leaving the replacement columns
        data = data.drop(['pos', 'salary', 'replace_val'], axis=1)
        cols = [i for i in range(0, len(data.columns))]
        data.columns = cols
        
        
        #----------
        # Create the Injury Distribtions
        #----------

        # merge with the points dataset to ensure every player is added with null mean risk filled by median
        inj = pd.merge(pd.DataFrame(data.index).drop_duplicates(), inj, how='left', left_on='player', right_on='player')
        inj.mean_risk = inj.mean_risk.fillna(inj.mean_risk.mean())

        # create a poisson distribution of injury risk for each player (clip at 16 games)
        pois = []
        for val in inj.mean_risk:
            pois.append(np.random.poisson(val, iterations*2))
        inj_dist = pd.concat([inj.player, pd.DataFrame(pois).astype('uint8').clip(upper=16, axis=0)], axis=1)
        
        # ensure that the injury distributions are in the same row order as the points dataset
        inj_dist = pd.merge(pd.DataFrame(data.index), inj_dist, how='left', left_on='player', right_on='player')
        inj_dist = inj_dist.drop('player', axis=1)

        # return the updated dataset with replacement points and injury distribution dataset
        return data, inj_dist, replace_val

        
    #--------
    # Setting up and Running the Simulation
    #--------
    
    @staticmethod
    def _Amatrix(pos_counts, pos_require):
        '''
        This function creates the A matrix that is critical for the ILP solution being equal
        to the positional constraints specified. I identified the given pattern empirically:
        1. Repeat the vector [1, 0, 0, 0, ...] N times for each player for a given position.
           The number of trailing zeros is equal to p-1 positions to draft.
        2. After the above vector is repeated N times for a given player, append a 0 before
           repeating the same pattern for the next player. Repeat for all players up until the 
           last position.
        3. for the last poition, repeat the pattern N-1 times and append a 1 at the end.
        This pattern allows the b vector, e.g. [1, 2, 2, 1] to set the constraints on the positions
        selected by the ILP solution.
        '''
        
        #--------
        # Initialize the Vector Pattern and Matrix
        #--------
        
        # create A matrix
        vec = [1]
        vec.extend([0]*(len(pos_require)-1))
        
        # intialize A matrix by multiplying length one by vec and appending 0 to start pattern
        A = pos_counts[0]*vec
        A.append(0)

        #--------
        # Repeat the Pattern Until Last Position
        #--------
        
        # repeat the same pattern for the inner position requirements
        for i in range(1, len(pos_counts)-1):

            A.extend(pos_counts[i]*vec)
            A.append(0)

        #--------
        # Finish the Pattern for the Last Position
        #--------
        
        # adjust the pattern slightly for the final position requirement
        A.extend((pos_counts[-1]-1)*vec)
        A.append(1)

        # convert A into a matrix for integer optimization
        A = matrix(A, size=(len(vec), np.sum(pos_counts)), tc='d')

        return A
    
    
    @staticmethod
    def _df_shuffle(df):
        '''
        Input: A dataframe to be shuffled, row-by-row indepedently.
        Return: The same dataframe whose columns have been shuffled for each row.
        '''
        # store the index before converting to numpy
        idx = df.index
        df = df.values

        # shuffle each row separately, inplace, and convert o df
        _ = [np.random.shuffle(i) for i in df]

        return pd.DataFrame(df, index=idx)
    
    
    @staticmethod
    def _run_opt(A, points, salaries, salary_cap, pos_require):
        '''
        This function sets up and solves the integer Linear Programming problem 
        c = n x 1 -- c is the vector of points to be optimized
        G = m x n -- G is the salaries of the corresponding players / points (m=1 in this case)
        h = m x 1 -- h is the salary cap (m=1 in this case)
        A = p x n -- A sparse binary matrix that must be developed so b equals player constraints
        b = p x 1 -- b is a vector with player requirements, e.g. [QB, RB, WR] = [1, 2, 2]

        Solve:
        c'*n -- minimize

        Subject to:
        G*x <= h
        A*x = b
        '''
        
        # generate the c matrix with the point values to be optimized
        c = matrix(points, tc='d')

        # generate the G matrix that contains the salary values for constraining
        G = matrix(salaries, tc='d').T

        # generate the h matrix with the salary cap constraint
        h = matrix(salary_cap, size=(1,1), tc='d')

        # generate the b matrix with the number of position constraints
        b = matrix(pos_require, size=(len(pos_require), 1), tc='d')

        # solve the integer LP problem
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(points))))

        return x
    
    @staticmethod
    def _random_select(data, salaries, salary_skews, inj_dist, replace_val, iterations):
        '''
        Random column selection for trial in simulation
        
        Input: Data, salaries, and salary skews
        Return: Randomly selected array of points and salaries + skews for a given trial
        '''
        # select random number between 0 and sise of distributions
        ran_num = random.randint(0, iterations*2-1)

        # pull out a random column of points and convert to points per game
        ppg = data.iloc[:, ran_num] / 16
                
        # pull a random selection of expected games missed
        inj_adjust = inj_dist.iloc[:, ran_num]
        
        # calculate total points scored as the number of games played by expected ppg
        # plus the number of games missed times replacement level value
        points = -1.0*((ppg.values) * (16-inj_adjust.values) + \
                       (inj_adjust.values * pd.concat([replace_val, ppg], axis=1).min(axis=1).values))

        # pull out a random skew and add to the original salaries
        salaries_tmp = salaries + salary_skews[:, ran_num]
        salaries_tmp = salaries_tmp.astype('double')
        
        return points, salaries_tmp
    
    
    #==========
    # Formatting and Displaying All Results
    #==========
    
    @staticmethod
    def _pull_results(x, names, points, salaries, to_add, results, counts, trial_counts):
        '''
        This method pulls in each individual result from the simulation loop and stores it in dictionaries.
        
        Input: Names, points, and salaries for the current simulation lineup.
        Return: Dictionaries with full results and overall simulation counts, continuously updated.
        '''
        # find all LP results chosen and equal to 1
        x = np.array(x)[:, 0]==1

        if len(names[x]) != len(np.unique(names[x])):
            return results, counts, trial_counts
        
        trial_counts += 1
        
        names_ = list(names[x])
        names_.extend(to_add['players'])
        
        points_ = list(points[x])
        points_.extend(to_add['points'])
        
        salaries_ = list(salaries[x])
        salaries_.extend(to_add['salaries'])
        
        for i, p in enumerate(names_):

            counts['names'][p] += 1

            if counts['points'][p] == 0:
                counts['points'][p] = []
            counts['points'][p].append(points_[i])

            if counts['salary'][p] == 0:
                counts['salary'][p] = []
            counts['salary'][p].append(salaries_[i])

        # pull out the corresponding names, points, and salaries for chosen players
        # to append to the higher level results dataframes
        results['names'].append(names_)
        results['points'].append(points_)
        results['salary'].append(salaries_)

        return results, counts, trial_counts
    
    
    @staticmethod
    def _format_results(results):
        '''
        After the simulation loop, this method pulls results from the dictionary and formats
        into dataframes.
        
        Input: The results dictionary with all results
        Return: A formatted dataframe with all results
        '''
        
        # create dataframes for the names of selected players, their points scored, and salaries
        name_results = pd.DataFrame(results['names'])
        point_results = pd.DataFrame(results['points'])*-1
        total_points = point_results.sum(axis=1)
        salary_results = pd.DataFrame(results['salary'])
        total_salary = salary_results.sum(axis=1)
        
        # concatenate names, points, and salaries altogether
        results_df = pd.concat([name_results, total_points, total_salary, point_results, salary_results], axis=1)
        
        # rename columns to numbers
        results_df.columns = range(0, results_df.shape[1])
        
        # find the first numeric column that corresponds to points scored and sort by that column
        first_num_col = results_df.dtypes[results_df.dtypes=='float64'].index[0]
        results_df = results_df.sort_values(by=first_num_col, ascending=False)

        return results_df

    #===============
    # Creating Output Visualizations
    #===============
    
    
    def density_plot(self, player):
        '''
        Creates density player showing points scored and salary selected for a given player
        '''
        
        # pull out points and salary for a given player
        sal = np.array(self.counts['salary'][player])
        
        # create and displayjoint distribution plot
        sns.distplot(sal)
        plt.show()

                
    def show_most_selected(self, to_add, iterations, num_show=20):
        '''
        Input: Dictionary containing all the values (counts, salaries, points) of the various
               iterations from the simulation.
        Output: Formatted dataframe that can be printed with pandas bar charts to visualize results.
        '''
        #----------
        # Create data frame of percent drafted and average salary
        #----------

        # create a dataframe of counts drafted for each player
        counts_df = pd.DataFrame.from_dict(self.counts['names'], orient='index').rename(columns={0: 'Percent Drafted'})
        counts_df = counts_df.sort_values(by='Percent Drafted', 
                                    ascending=False)[len(to_add['players']):].head(30) / iterations

        # pull out the salary that each player was drafted at from dictionary
        avg_sal = {}
        for key, value in self.counts['salary'].items():
            avg_sal[key] = np.mean(value)

        # pass the average salaries into datframe and merge with the counts dataframe
        avg_sal = pd.DataFrame.from_dict(avg_sal, orient='index').rename(columns={0: 'Average Salary'})
        avg_sal = pd.merge(counts_df, avg_sal, how='inner', left_index=True, 
                        right_index=True).sort_values(by='Percent Drafted', ascending=False)
        
        # pull in the list salary + inflation to calculate drafted salary minus expected
        avg_sal = pd.merge(avg_sal, self._sal, how='inner', left_index=True, right_index=True)
        avg_sal.salary = (avg_sal['Average Salary'] - avg_sal.salary * self._inflation)
        avg_sal = avg_sal.rename(columns={'salary': 'Expected Salary Diff'})

        # format the result with rounding
        avg_sal.loc[:, 'Percent Drafted'] = round(avg_sal.loc[:, 'Percent Drafted'] * 100, 1)
        avg_sal.loc[:, 'Average Salary'] = round(avg_sal.loc[:, 'Average Salary'], 1)
        avg_sal.loc[:, 'Expected Salary Diff'] = round(avg_sal.loc[:, 'Expected Salary Diff'], 1)

        return avg_sal
            