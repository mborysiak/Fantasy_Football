#%%

# core packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import os
import sqlite3

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

class FootballSimulation():

    #==========
    # Creating Player Distributions for Given Settings
    #==========
    
    def __init__(self, conn_sim, table_vers, set_year, league):
        
        # create empty dataframe to store player point distributions
        pos_update = {'QB': 'aQB', 'RB': 'bRB', 'WR': 'cWR', 'TE': 'dTE'}
        self.data = pd.read_sql_query(f'''SELECT * FROM {table_vers}_{set_year}''', conn_sim)
        self.data['pos'] = self.data['pos'].map(pos_update)

        # add salaries to the dataframe and set index to player
        self.salaries = pd.read_sql_query(f'''SELECT player, salary, std_dev, min_score, max_score
                                              FROM Salaries
                                              WHERE year={set_year}
                                                    AND league='{league}pred' ''', conn_sim)

        self.data = pd.merge(self.data, self.salaries[['player', 'salary']], on='player', how='left')
        self.data.salary = self.data.salary.fillna(1)

        # pull in injury risk information
        self.inj = pd.read_sql_query(f'''SELECT player, mean_risk 
                                         FROM Injuries 
                                         WHERE year={set_year} ''', conn_sim)
        
        # add flex data
        flex = self.data[self.data.pos.isin(['bRB', 'cWR', 'dTE'])]
        flex.loc[:, 'pos'] = 'eFLEX'
        self.data = pd.concat([self.data, flex], axis=0)
        
        # reset index
        self.data = self.data.sort_values(by=['pos', 'salary'], ascending=[True, False])
        self.data = self.data.set_index('player')

    
    def return_data(self):
        '''
        Returns self.data if necessary.
        '''
        return self.data
    
    

    #==========
    # Running the Simulation for Given League Settings and Keepers
    #==========
    
    def run_simulation(self, league_info, to_drop, to_add, iterations=500, num_avg_pts=3):        
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
        print('Current Inflation:', round(1, 2))

        # determine salaries, skew distributions, and number of players for each position
        data, salaries, pos_counts = self._pull_salary_poscounts(data)
        
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
        _ = [np.random.shuffle(row) for row in salaries]
        inj_dist = self._df_shuffle(inj_dist)
        data = self._df_shuffle(data)
                
        #--------
        # Run the Simulation Loop
        #--------
            
        trial_counts = 0
        for i in range(0, iterations):
    
            # every N trials, randomly shuffle each run in salary skews and data
            if i % (iterations / 10) == 0:
                _ = [np.random.shuffle(row) for row in salaries]
                inj_dist = self._df_shuffle(inj_dist)
                data = self._df_shuffle(data)
            
            # pull out a random selection of points and salaries
            points, salaries_tmp = self._random_select(data, salaries, inj_dist, replace_val, num_avg_pts)

            # run linear integer optimization
            x = self._run_opt(A, points, salaries_tmp, league_info['salary_cap'], pos_require)

            # pull out and store the selected names, points, and salaries
            results, self.counts, trial_counts = self._pull_results(x, names, points, salaries_tmp, 
                                                                    to_add, results, counts, trial_counts)
        
        # format the results after the simulation loop is finished
        self.results = self._format_results(results)

        # add self version of variable for output calculations
        self._sal = self.data.reset_index()[['player', 'salary']].drop_duplicates().set_index('player')

        return self.counts
    
    #==========
    # Helper Functions for the Simulation Loop
    #==========

    #--------
    # Salary (+Inflation) and Keeper Setup
    #--------
    
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
    def trunc_normal(row, num_samples=1000):

        import scipy.stats as stats

        # create truncated distribution
        lower, upper = row.min_score,  row.max_score
        lower_bound = (lower - row.salary) / row.std_dev, 
        upper_bound = (upper - row.salary) / row.std_dev
        trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=row.salary, scale=row.std_dev)
        
        estimates = trunc_dist.rvs(num_samples)

        return estimates


    def trunc_normal_dist(self, df, num_options=1000):
        predictions = pd.DataFrame()
        for _, row in df.iterrows():
            dists = pd.DataFrame(self.trunc_normal(row, num_options)).T
            predictions = pd.concat([predictions, dists], axis=0)
        
        return predictions.reset_index(drop=True)


    def _pull_salary_poscounts(self, data):
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
        salaries = pd.merge(data.reset_index()[['player']], self.salaries, how='left', left_on='player', right_on='player')
        salaries.salary = salaries.salary.fillna(1)
        salaries.min_score = salaries.min_score.fillna(1)
        salaries.max_score = salaries.max_score.fillna(10)
        salaries.std_dev = salaries.std_dev.fillna(5)

        # calculate salary dist for each player's salary
        salaries = self.trunc_normal_dist(salaries, 1000).values

        # extract the number of counts for each position for later creating A matrix
        pos_counts = list(data.pos.value_counts().sort_index())
        
        return data, salaries, pos_counts
    
    
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
        mean_pts = data.iloc[:, 0:1000].mean(axis=1).reset_index()
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
            pois.append(np.random.poisson(val, 1000))
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
    def _random_select(data, salaries, inj_dist, replace_val, num_avg_pts):
        '''
        Random column selection for trial in simulation
        
        Input: Data, salaries, and salary skews
        Return: Randomly selected array of points and salaries + skews for a given trial
        '''
        # select random number between 0 and sise of distributions
        # ran_num = random.randint(0, 1000-1)
        ran_num = np.random.choice(range(999), size=num_avg_pts)

        # pull out a random column of points and convert to points per game
        # ppg = data.iloc[:, ran_num] / 16
        ppg = data.iloc[:, ran_num].mean(axis=1) / 16
        ppg = pd.DataFrame(ppg, index=data.index)

        # pull a random selection of expected games missed
        inj_adjust = inj_dist.iloc[:, ran_num[0]]#.mean(axis=1)
        
        # calculate total points scored as the number of games played by expected ppg
        # plus the number of games missed times replacement level value
        points = -1.0*((ppg.values) * (16-inj_adjust.values) + \
                       (inj_adjust.values * pd.concat([replace_val, ppg], axis=1).min(axis=1).values))
        
        # pull out a random skew and add to the original salaries
        salaries_tmp = salaries[:, ran_num[0]]#.mean(axis=1)
        salaries_tmp = salaries_tmp.astype('double')
        salaries_tmp[salaries_tmp < 2] = 2
        
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
                                    ascending=False)[len(to_add['players']):].head(num_show) / iterations

        # pull out the salary that each player was drafted at from dictionary
        avg_sal = {}
        for key, value in self.counts['salary'].items():
            avg_sal[key] = [np.mean(value), np.percentile(value, 90)]

        # pass the average salaries into datframe and merge with the counts dataframe
        avg_sal = pd.DataFrame.from_dict(avg_sal, orient='index').rename(columns={0: 'Average Salary', 1: 'Salary90'})
        avg_sal = pd.merge(counts_df, avg_sal, how='inner', left_index=True, 
                        right_index=True).sort_values(by='Percent Drafted', ascending=False)
        
        # pull in the list salary + inflation to calculate drafted salary minus expected
        avg_sal = pd.merge(avg_sal, self._sal, how='inner', left_index=True, right_index=True)
        avg_sal.salary = (avg_sal['Average Salary'] - avg_sal.salary * 1)
        avg_sal = avg_sal.rename(columns={'salary': 'Expected Salary Diff'})

        # format the result with rounding
        avg_sal.loc[:, 'Percent Drafted'] = round(avg_sal.loc[:, 'Percent Drafted'] * 100, 1)
        avg_sal.loc[:, 'Average Salary'] = round(avg_sal.loc[:, 'Average Salary'], 1)
        avg_sal.loc[:, 'Expected Salary Diff'] = round(avg_sal.loc[:, 'Expected Salary Diff'], 1)

        return avg_sal


# %%

# # connection for simulation and specific table
# path = f'c:/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'
# conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')

# set_year = 2023
# league='beta'
# table_vers = 'Version' + league

# # number of iteration to run
# iterations = 500

# # instantiate simulation class and add salary information to data
# sim = FootballSimulation(conn_sim, table_vers, set_year, league)

# # set league information, included position requirements, number of teams, and salary cap
# league_info = {}
# league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
# league_info['num_teams'] = 12
# league_info['initial_cap'] = 293
# league_info['salary_cap'] = 293

# to_drop = {}
# to_drop['players'] = []
# to_drop['salaries'] = []

# # input information for players and their associated salaries selected by your team
# to_add = {}
# to_add['players'] = []
# to_add['salaries'] = []

# sim.run_simulation(league_info, to_drop, to_add, iterations=iterations, num_avg_pts=3)
# sim.show_most_selected(to_add, iterations)
# %%
