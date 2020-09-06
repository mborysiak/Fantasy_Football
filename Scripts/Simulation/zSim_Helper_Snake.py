#%%

# core packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas.io.formats.style
import copy
import os 

# sql packages
import sqlite3

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp
from scipy.stats import skewnorm

class FootballSimulation():

    #==========
    # Creating Player Distributions for Given Settings
    #==========
    
    def __init__(self, pts_dict, conn_sim, table_vers, set_year, league, iterations, initial_pick=None):
        
        # create empty dataframe to store player point distributions
        pos_update = {'QB': 'aQB', 'RB': 'bRB', 'WR': 'cWR', 'TE': 'dTE'}
        self.data = pd.read_sql_query(f'''SELECT * FROM {table_vers}_{set_year}''', conn_sim)
        self.data['pos'] = self.data['pos'].map(pos_update)

        if initial_pick is not None:
            self.is_snake = True
            self.pick_prob = pd.read_sql_query('''SELECT * FROM PickProb''', conn_sim)
            self.my_picks = self._get_picks(initial_pick)

        # pull in injury risk information
        self.inj = pd.read_sql_query(f'''SELECT player, mean_risk 
                                         FROM Injuries 
                                         WHERE year={set_year} ''', conn_sim)
        
        # add flex data
        flex = self.data[self.data.pos.isin(['bRB', 'cWR', 'dTE'])]
        flex.loc[:, 'pos'] = 'eFLEX'
        self.data = pd.concat([self.data, flex], axis=0)
    
    
    #================
    # Initialize Datasets
    #================

    def return_data(self):
        '''
        Returns self.data if necessary.
        '''
        return self.data

    
    def return_picks(self):
        '''
        Returns self.data if necessary.
        '''
        return self.my_picks

    @staticmethod
    def _get_picks(initial_pick, num_teams=12, third_reverse=True):
        '''
        INPUT: Initial pick. number of teams in the draft, and whether the third
               round is reversed or not

        OUTPUT: A list of the team's selections throughout the draft based on the parameters
        '''

        # create a list of the team integer values and also reverse it for back-tracking selection
        teams = [i for i in range(1, num_teams+1)]
        teams_rev=list(reversed(teams))
            
        # initialize the draft order and extend first back-word selection
        draft_order = teams.copy()
        draft_order.extend(teams_rev)

        # if third round reverse is used, extend another reverse selection
        if third_reverse:
            draft_order.extend(teams_rev)
        
        # loop through all 10 iterations of alternating forward and backwards selections
        for i in range(10):
            draft_order.extend(teams)
            draft_order.extend(teams_rev)
        
        # create running total of picks assigned to each team
        draft_order = pd.DataFrame(draft_order, columns=['team_num'])
        draft_order['pick_num'] = range(1, len(draft_order)+1)

        # select relevant picks for the initial pick selection
        return list(draft_order.loc[draft_order.team_num==initial_pick, 'pick_num'].values)

    
    #==========
    # Running the Simulation for Given League Settings and Keepers
    #==========
    
    def run_simulation(self, league_info, to_drop, to_add, pick_num, prob_filter, iterations=500):        
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
        my_picks = [p for p in self.my_picks if p >= pick_num]
       
        # drop other selected players + calculate inflation metrics
        data = self._drop_players(data)
        
        # drop your selected players + calculate inflation metrics
        data, league_info, to_add = self._add_players(data, league_info, to_add)
        
        remain_pos = [k for k, v in league_info['pos_require'].items() if v != 0]
        pos_update = {'QB': 'aQB', 'RB': 'bRB', 'WR': 'cWR', 'TE': 'dTE', 'FLEX': 'eFLEX'}
        remain_pos = [pos_update[k] for k in remain_pos]
        pos_require = [v for k, v in league_info['pos_require'].items() if v != 0]
        pos_counts = list(data.pos.value_counts().sort_index())

        data = data[data.pos.isin(remain_pos)].reset_index(drop=True)
        pick_prob_df = self.pick_prob[self.pick_prob.adp >= 0.5*my_picks[0]].copy().reset_index(drop=True)
        pick_prob_df, data_unique, player_prob_df = self._get_relevant_player_prob(pick_prob_df, data, my_picks, 
                                                                                    prob_filter)

        # use the data unique to bring down data projections to only relevant players and properly order
        data = pd.merge(data, data_unique, on=['player', 'pos']).set_index('player')
        data = data.sort_values(by=['pos', 'adp'], ascending=[True, True])
        
        # get the counts of players for each position
        pos_counts = list(data.pos.value_counts().sort_index())
        
        # calculate the injury distributions
        data, inj_dist, replace_val = self._injury_replacement(data, league_info, inj, iterations)

        #--------
        # Initialize Matrix and Results Dictionary for Simulation
        #--------

        # get the pick array matrix for Ax=B
        pick_array = self._create_pick_matrix(pick_prob_df, data_unique, my_picks)
        
        # generate the A matrix for the simulation constraints
        A = self._Amatrix(pos_counts, pos_require, self.is_snake, pick_array)

        # pull out the names of all players and set to names
        names = data.index
        dict_names = list(data.index)
        dict_names.extend(to_add['players'])
        
        # create empty matrices
        results = {}
        results['names'] = []
        results['points'] = []

        # create empty dictionaries
        counts = {}
        counts['names'] = pd.Series(0, index=dict_names).to_dict()
        counts['points'] = pd.Series(0, index=dict_names).to_dict()
        
        inj_dist = self._df_shuffle(inj_dist)
        data = self._df_shuffle(data)
                
        #--------
        # Run the Simulation Loop
        #--------
            
        trial_counts = 0
        for i in range(0, iterations):
    
            # every N trials, randomly shuffle each run in salary skews and data
            if i % (iterations / 10) == 0:
                inj_dist = self._df_shuffle(inj_dist)
                data = self._df_shuffle(data)
            
            # pull out a random selection of points and salaries
            points = self._random_select(data, inj_dist, replace_val, iterations)
            salaries_tmp = np.array([1 for p in points])

            # run linear integer optimization
            x = self._run_opt(A, points, salaries_tmp, league_info['salary_cap'], pos_require, my_picks)

            # pull out and store the selected names, points, and salaries
            results, self.counts, trial_counts = self._pull_results(x, names, points, salaries_tmp, 
                                                                    to_add, results, counts, trial_counts)
        
        # format the results after the simulation loop is finished
        self.results = self._format_results(results)


        close_picks = list(data_unique.loc[data_unique.adp <= my_picks[0]*1.5 + 10, 'player'].values)
        top_picks = {k: v for k, v in sorted(self.counts['names'].items(), key=lambda item: item[1]) if k in close_picks}

        top_picks = pd.DataFrame([top_picks]).T
        top_picks = pd.merge(top_picks, player_prob_df, left_index=True, right_index=True)
        top_picks = pd.merge(top_picks, data_unique[['player', 'adp']].drop_duplicates().set_index('player'),
                             left_index=True, right_index=True)
        return top_picks
    
    #==========
    # Helper Functions for the Simulation Loop
    #==========

    #--------
    # Salary (+Inflation) and Keeper Setup
    #--------
    
    
    @staticmethod
    def _drop_players(data):
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

        return other_data
    
    
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
        add_data = data[data.player.isin(to_add['players'])].sort_values(by='pos')
        other_data = data[~data.player.isin(to_add['players'])].sort_values(by='pos')

        # add the mean points scored by the players who have been added
        to_add['points'] = -1.0*(add_data.drop(['pos'],axis=1).mean(axis=1).values)

        # create list of letters to append to position for proper indexing
        letters = ['a', 'b', 'c', 'd', 'e']

        # loop through each position in the pos_require dictionary
        for i, pos in enumerate(league_info['pos_require'].keys()):

            # create a unique label based on letter and position
            pos_label = letters[i]+pos

            # loop through each player that has been selected  
            for player in list(add_data.loc[add_data.pos==pos_label, 'player']):

                # if the position is still required to be filled:
                if league_info['pos_require'][pos] > 0:

                    # subtract out the current player from the position count
                    league_info['pos_require'][pos] = league_info['pos_require'][pos] - 1

                    # and remove that player from consideration for filling other positions
                    add_data = add_data[add_data.player != player]
        
        return other_data, league_info, to_add
    
    
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
        try:
            data = data.drop(['pos', 'salary', 'replace_val'], axis=1)
        except:
            data = data.drop(['pos', 'adp', 'replace_val'], axis=1)
            
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


    @staticmethod
    def _get_relevant_player_prob(pick_prob_df, data, my_picks, prob_filter):
        '''
        INPUT: Dataframe containing every player-pick-probability combination
               based on ADP for the league, the player-points dataframe,
               and all picks in the draft based on initial pick

        OUTPUT: Dataframe that contains players able to be selected for each
                pick in the draft based on given probability cutoff and a dataframe
                containing unique player-position-adp data.
        '''
        # create a dataframe of player probability for being drafted in the current slot
        player_prob_df = pick_prob_df[(pick_prob_df.pick >= my_picks[0]) & \
            (pick_prob_df.adp <= my_picks[0]*3)].groupby('player').agg({'pick_prob': 'sum'})

        # narrow down the player list for each potential selection based on their probability of being 
        # drafted near a specific slot. note the filter is determined empirically to prevent infeasible ilp solutions.
        # prob_filter = 7#(np.sum(pos_require) * len(remain_pos))-6
        pick_prob_df = pick_prob_df.loc[(pick_prob_df.pick.isin(my_picks)) & \
            (pick_prob_df.pick_prob >=  prob_filter), 
            ['player', 'adp', 'pick']].reset_index(drop=True)

        # mrege with data to get the position for each player and sort by position / adp
        pick_prob_df = pd.merge(data[['player', 'pos']], pick_prob_df, on='player')
        pick_prob_df = pick_prob_df.sort_values(by=['pos', 'adp'], ascending=[True, True])

        # extract a dataframe of each player | pos| adp without duplicates for pick prob
        data_unique = pick_prob_df[['player', 'pos', 'adp']].drop_duplicates()

        # set a flag for creating the pick array sparse matrix to calculate Ax=b
        pick_prob_df['flag'] = 1
        pick_prob_df = pick_prob_df[['player', 'pos', 'pick', 'flag']]

        return pick_prob_df, data_unique, player_prob_df

        
    #--------
    # Setting up and Running the Simulation
    #--------

    @staticmethod
    def _create_pick_matrix(pick_prob_df, data_unique, my_picks):
        '''
        INPUT: Pick probability dataframe, unique player-adp dataframe, and list containing
               all relevant picks for a given team.

        OUTPUT: A numpy sparse matrix that indicates whether a player is available to be selected
                for at each point in the draft (m_picks x n_players). This matrix will be appended
                to the A matrix that will also contain positional equality constraints (Ax=b)
        '''
        # intialize empty array for pick flags
        pick_array = np.empty(shape=(0,len(data_unique)))

        # loop through each pick and merge the unique player dataset
        # with players who have the potential to be selected at each pick
        for i in my_picks:

            # perform left join with flag and fill 0's to indicate if a player is able to be selected
            tmp = pd.merge(data_unique, pick_prob_df.loc[pick_prob_df.pick==i, ['player', 'pos', 'flag']],
                        how='left', on=['player', 'pos']).fillna(0)
            
            # append flag values to array vertically
            to_add = tmp.flag.values
            pick_array = np.vstack([pick_array, to_add])

        # convert all to binary flags
        pick_array = pick_array.astype('int')

        return pick_array
        
    @staticmethod
    def _Amatrix(pos_counts, pos_require, is_snake=False, pick_array=None):
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

        # if completing a snake draft, extend the dataset to have the pick array
        # that contains flags for each player for each round
        if is_snake:
            A = np.array(A)
            A = np.vstack([A, pick_array])

            A = matrix(A, tc='d')

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
    def _run_opt(A, points, salaries, salary_cap, pos_require, my_picks):
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

        # generate the b matrix with the number of position constraints + pick constraints
        b_vec = pos_require.copy()
        num_picks = len(my_picks)*[1]
        b_vec.extend(num_picks)
        b = matrix(b_vec, size=(len(b_vec), 1), tc='d')

        # solve the integer LP problem
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(points))))

        return x
    
    @staticmethod
    def _random_select(data, inj_dist, replace_val, iterations):
        '''
        Random column selection for trial in simulation
        
        Input: Data, salaries, and salary skews
        Return: Randomly selected array of points and salaries + skews for a given trial
        '''
        # select random number between 0 and sise of distributions
        ran_num = random.randint(0, 1000-1)

        # pull out a random column of points and convert to points per game
        ppg = data.iloc[:, ran_num] / 16
                
        # pull a random selection of expected games missed
        inj_adjust = inj_dist.iloc[:, ran_num]
        
        # calculate total points scored as the number of games played by expected ppg
        # plus the number of games missed times replacement level value
        points = -1.0*((ppg.values) * (16-inj_adjust.values) + \
                       (inj_adjust.values * pd.concat([replace_val, ppg], axis=1).min(axis=1).values))
        
        return points
    
    
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
        
        for i, p in enumerate(names_):

            counts['names'][p] += 1

            if counts['points'][p] == 0:
                counts['points'][p] = []
            counts['points'][p].append(points_[i])

        # pull out the corresponding names, points, and salaries for chosen players
        # to append to the higher level results dataframes
        results['names'].append(names_)
        results['points'].append(points_)

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
        
        # concatenate names, points, and salaries altogether
        results_df = pd.concat([name_results, total_points, point_results], axis=1)
        
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


# # %%

# connection for simulation and specific table
path = f'c:/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
table_vers = 'Version4'
set_year = 2020
initial_pick = 1
league='nffc'

# number of iteration to run
iterations = 500

# define point values for all statistical categories
pass_yd_per_pt = 0.04 
pass_td_pt = 4
int_pts = -2
sacks = -1
rush_yd_per_pt = 0.1 
rec_yd_per_pt = 0.1
rush_rec_td = 7
ppr = .5

# creating dictionary containing point values for each position
pts_dict = {}
pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_rec_td, int_pts, sacks]
pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['WR'] = [rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['TE'] = [rec_yd_per_pt, ppr, rush_rec_td]

# instantiate simulation class and add salary information to data
sim = FootballSimulation(pts_dict, conn_sim, table_vers, set_year, league, iterations, initial_pick)

# set league information, included position requirements, number of teams, and salary cap
league_info = {}
league_info['pos_require'] = {'QB': 3, 'RB': 6, 'WR': 9, 'TE': 3, 'FLEX': 0}
league_info['num_teams'] = 12
league_info['initial_cap'] = 293
league_info['salary_cap'] = 293

to_drop = {}
to_drop['players'] = []

# input information for players and their associated salaries selected by your team
to_add = {}
to_add['players'] = ['Christian McCaffrey', 'Lamar Jackson', 'Jonathan Taylor',
                     'Cooper Kupp', 'Matt Ryan', 'Hunter Henry', 'CeeDee Lamb',
                     'Antonio Gibson', 'Golden Tate', 'Anthony Miller',
                     'Anthony McFarland', 'Larry Fitzgerald', 'Dallas Goedert',
                     'Sammy Watkins']

num_pos = [v for k, v in league_info['pos_require'].items()]

my_p = sim.return_picks()
pick_num = my_p[len(to_add['players'])]

result = None
prob_filter = 1
while result is None and prob_filter < 20:
    try:
        result = sim.run_simulation(league_info, to_drop, to_add, pick_num=pick_num, prob_filter=prob_filter, iterations=iterations)
    except:
        prob_filter += 1
        print(prob_filter)

result
# %%
filt
# %%
