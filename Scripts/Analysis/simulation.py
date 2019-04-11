
class FF_Simulation():

    #==========
    # Creating Player Distributions for Given Settings
    #==========
    
    def __init__(self, db_name, set_year, pts_dict, prior_repeats=5, show_plots=False):
        '''
        The initialization of this Class reads in all of the statistical projection data and
        translates it into clusters and projection distributions given a particular scoring schema.
        The data is then stored in the self.data object, which will be accessed through the analysis.
        
        Input: A database that contains statistical projections, a dictionary that contains the points
               for each category, and number of prior repeats to use for Bayesian updating.
        Return: Stores all the player projection distributions in that self.data object.
        '''
        # create empty dataframe to store all player distributions
        self.data = pd.DataFrame()
        
        #==========
        # Loop through each position and pull / analyze the data
        #==========
        
        for pos in ['aQB', 'bRB', 'cWR', 'dTE']:
                
            # print current position update
            print('Loading and Preparing ' + pos[1:] + ' Data')
            
            #--------
            # Connect to Database and Pull Player Data
            #--------
            
            conn = sqlite3.connect(path + 'Data/' + db_name)

            df_train_results = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Train_Results_' + str(set_year), con=conn)
            df_test_results = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Test_Results_' + str(set_year), con=conn)
            df_train = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Train_' + str(set_year), con=conn)
            df_predict = pd.read_sql_query('SELECT * FROM ' + pos[1:] + '_Predict_' + str(set_year), con=conn)

            #--------
            # Calculate Fantasy Points for Given Scoring System and Cluster
            #--------
            df_train_results, df_test_results = format_results(df_train_results, df_test_results, 
                                                               df_train, df_predict, 
                                                               pts_dict[pos[1:]])
            df_train_results = df_train_results.drop('year', axis=1)
            
            # initialize cluster with train and test results
            cluster = Clustering(df_train_results, df_test_results)

            # fit decision tree and apply nodes to players
            cluster.fit_and_predict_tree(print_results=False)

            # add linear regression of predicted vs actual for cluster predictions
            c_train, c_test = cluster.add_fit_metrics()
            
            #--------
            # Use Bayesian Updating to Create Points Distributions
            #--------
            
            # create distributions of data
            distributions = cluster.create_distributions(prior_repeats=prior_repeats, show_plots=show_plots)
            
            # add position to the distributions
            distributions['pos'] = pos
            
            # append each position of data to master dataset
            self.data = pd.concat([self.data, distributions], axis=0)
            
        # add flex data
        flex = self.data[self.data.pos.isin(['bRB', 'cWR', 'dTE'])]
        flex['pos'] = 'eFLEX'
        self.data = pd.concat([self.data, flex])
        
        # format the self.data for later use
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.rename(columns={0: 'player'})
    
    
    def return_data(self):
        '''
        Returns self.data if necessary.
        '''
        return self.data
    
    
    #==========
    # Running the Simulation for Given League Settings and Keepers
    #==========
    
    def run_simulation(self, league_info, to_drop, to_add, iterations=500):        
        
        #--------
        # Pull Out Data, Salaries, and Inflation for Current Simulation
        #--------
        
        # create a copy of self.data for current iteration settings
        data = self.data.copy()
        
        # drop other selected players + calculate inflation metrics
        data, drop_proj_sal, drop_act_sal = self._drop_players(data, league_info, to_drop)
        
        # drop your selected players + calculate inflation metrics
        data, league_info, to_add, add_proj_sal, add_act_sal = self._add_players(data, league_info, to_add)
        
        pos_require = list(league_info['pos_require'].values())
        
        # calculate inflation based on drafted players and their salaries
        inflation = self._calc_inflation(league_info, drop_proj_sal, drop_act_sal, add_proj_sal, add_act_sal)

        # determine salaries, skew distributions, and number of players for each position
        data, salaries, salary_skews, pos_counts = self._pull_salary_poscounts(data, inflation)
        
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
        data = self._df_shuffle(data)
                
        #--------
        # Run the Simulation Loop
        #--------
            
        trial_counts = 0
        for i in range(0, iterations):
    
            # every 250 trials, randomly shuffle each run in salary skews and data
            if i % 250 == 0:
                _ = [np.random.shuffle(row) for row in salary_skews]
                data = self._df_shuffle(data)

            # pull out a random selection of points and salaries
            points, salaries_tmp = self._random_select(data, salaries, salary_skews)
            
            # run linear integer optimization
            x = self._run_opt(A, points, salaries_tmp, league_info['salary_cap'], pos_require)

            # pull out and store the selected names, points, and salaries
            results, self.counts, trial_counts = self._pull_results(x, names, points, salaries_tmp, 
                                                                    to_add, results, counts, trial_counts)
        
        # format the results after the simulation loop is finished
        self.results = self._format_results(results)
        
        return self.results, self.counts
    
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
        drop_proj_salary = drop_data.salary.drop_duplicates().sum()
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

        add_data = data[data.index.isin(to_add['players'])]
        add_data = add_data[~add_data.index.duplicated(keep='first')]
        other_data = data.drop(add_data.index, axis=0)

        add_proj_salary = add_data.salary.drop_duplicates().sum()
        add_act_salary = np.sum(to_add['salaries'])
        
        # update the salary for your team to subtract out drafted player's salaries
        league_info['salary_cap'] = float(league_info['salary_cap'] - add_act_salary)
        
        to_add['points'] = -1.0*(add_data.drop(['pos', 'salary'],axis=1).mean(axis=1).values)
        
        
        for pos in league_info['pos_require'].keys():
            league_info['pos_require'][pos] = league_info['pos_require'][pos] - to_add['positions'][pos]
        
        return other_data, league_info, to_add, add_proj_salary, add_act_salary
    
    
    @staticmethod
    def _calc_inflation(league_info, drop_proj_sal, drop_act_sal, add_proj_sal, add_act_sal):
        
        # add up the total actual and projected salaries for all keepers
        projected_salary = drop_proj_sal + add_proj_sal
        actual_salary = drop_act_sal + add_act_sal
        
        # calculate the salary inflation due to the keepers
        total_cap = league_info['num_teams'] * league_info['salary_cap']
        inflation = (total_cap-actual_salary) / (total_cap-projected_salary)
        
        return inflation
        
        
    def _pull_salary_poscounts(self, data, inflation):
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
        salary_skews = self._skews(salaries)

        # extract the number of counts for each position for later creating A matrix
        pos_counts = list(data.pos.value_counts().sort_index())

        # drop salary from the points dataframe and reset the columns from 0 to N
        data = data.drop(['pos', 'salary'], axis=1)
        data.columns = [i for i in range(0, 10000)]
        
        return data, salaries, salary_skews, pos_counts
        
        
    @staticmethod
    def _skews(salaries):
        '''
        Input: Internal method that accepts the salaries input for each player in the dataset.
        Return: Right skewed salary uncertainties, scaled to the actual salary of the player.
        '''
        # pull out the salary column and convert to numpy array
        _salaries = salaries.reshape(-1,1)

        # create a skews normal distribution of uncertainty for salaries
        skews = (skewnorm.rvs(10, size=10000)*.07).reshape(1, -1)

        # create a p x m matrix with dot product, where p is the number of players
        # and m is the number of skewed uncertainties, e.g. 320 players x 10000 skewed errors
        salary_skews = np.dot(_salaries, skews)

        return salary_skews
        
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
    def _random_select(data, salaries, salary_skews):
        '''
        Random column selection for trial in simulation
        
        Input: Data, salaries, and salary skews
        Return: Randomly selected array of points and salaries + skews for a given trial
        '''
        # select random number between 0-10000
        ran_num = random.randint(0, 9999)

        # pull out a random column of points 
        points = data.iloc[:, ran_num].values.astype('double')*-1.0

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
    
    
    def density_plot(self, player):
        '''
        Creates density player showing points scored and salary selected for a given player
        '''
        
        # pull out points and salary for a given player
        pts = np.array(self.counts['points'][player])*-1
        sal = np.array(self.counts['salary'][player])
        
        # create and displayjoint distribution plot
        sns.jointplot(x=pts, y=sal, kind="kde", ratio=4, size=5, space=0)
        ax = plt.gca()
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
    def show_most_selected(self, to_add, iterations, num_show=20):
        
        counts = pd.DataFrame.from_dict(self.counts['names'], orient='index')
        
        counts = counts.sort_values(by=0, ascending=False)[len(to_add['players']):].head(num_show) / iterations
        to_plot = counts.plot.bar(legend=False, figsize=(15, 4))
