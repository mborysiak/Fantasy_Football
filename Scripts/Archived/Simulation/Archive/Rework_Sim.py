#%%

import pandas as pd
import sqlite3
from cvxopt import matrix
from cvxopt.glpk import ilp
import numpy as np

# connection for simulation and specific table
path = f'c:/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
table_vers = 'Version3'
set_year = 2020
league='nffc'

pick_prob_master = pd.read_sql_query('''SELECT * FROM PickProb''', conn_sim)

# create empty dataframe to store player point distributions
pos_update = {'QB': 'aQB', 'RB': 'bRB', 'WR': 'cWR', 'TE': 'dTE'}
data = pd.read_sql_query(f'''SELECT * FROM {table_vers}_{set_year}''', conn_sim)
data['pos'] = data['pos'].map(pos_update)

# pull in injury risk information
inj = pd.read_sql_query(f'''SELECT player, mean_risk 
                                    FROM Injuries 
                                    WHERE year={set_year} ''', conn_sim)

# add flex data
flex = data[data.pos.isin([ 'bRB', 'cWR', 'dTE'])]
flex.loc[:, 'pos'] = 'eFLEX'
data_master = pd.concat([data, flex], axis=0)

def get_picks(initial_pick, num_teams=12, third_reverse=True):

    teams = [i for i in range(1, num_teams+1)]
    teams_rev=list(reversed(teams))
        
    draft_order = teams.copy()
    draft_order.extend(teams_rev)
    if third_reverse:
        draft_order.extend(teams_rev)
    
    for i in range(10):
        draft_order.extend(teams)
        draft_order.extend(teams_rev)
    
    draft_order = pd.DataFrame(draft_order, columns=['team_num'])
    draft_order['pick_num'] = range(1, len(draft_order)+1)

    return list(draft_order.loc[draft_order.team_num==initial_pick, 'pick_num'].values)


#%%

#----------------
# Use Player Pick-Prob to adjust "Salaries"
#----------------

pos_left = {'aQB': 1,
            'bRB': 2,
            'cWR': 3,
            'dTE': 1,
            'eFLEX': 2}
pos_require = [v for k, v in pos_left.items() if v!=0]
pos_list = [k for k, v in pos_left.items() if v!=0]

data = data_master[data_master.pos.isin(pos_list)].copy().reset_index(drop=True)

# get the picks for the starting draft position
initial_pick=12
my_picks = get_picks(initial_pick)[(9-np.sum(pos_require)):]


def get_relevant_player_prob(pick_prob_master, data, my_picks):

    # create a dataframe of player probability for being drafted in the current slot
    player_prob_df = pick_prob_master[(pick_prob_master.pick >= my_picks[0]) & \
        (pick_prob_master.adp <= my_picks[0]*3)].groupby('player').agg({'pick_prob': 'sum'})

    # narrow down the player list for each potential selection based on their probability of being 
    # drafted near a specific slot. note the filter is determined empirically to prevent infeasible ilp solutions.
    pick_prob_df = pick_prob_master.copy()
    pick_prob_df = pick_prob_df.loc[(pick_prob_df.pick.isin(my_picks)) & \
        (pick_prob_df.pick_prob >= np.sum(pos_require)*1.5 - 3), 
        ['player', 'adp', 'pick']].reset_index(drop=True)

    # mrege with data to get the position for each player and sort by position / adp
    pick_prob_df = pd.merge(data[['player', 'pos']], pick_prob_df, on='player')
    pick_prob_df = pick_prob_df.sort_values(by=['pos', 'adp'], ascending=[True, True])

    # extract a dataframe of each player | pos| adp without duplicates for pick prob
    data_unique = pick_prob_df[['player', 'pos', 'adp']].drop_duplicates()

    # set a flag for creating the pick array sparse matrix to calculate Ax=b
    pick_prob_df['flag'] = 1
    pick_prob_df = pick_prob_df[['player', 'pos', 'pick', 'flag']]

    return pick_prob_df, data_unique



def create_pick_matrix(pick_prob_df, data_unique, my_picks):

    # intialize empty array for pick flags
    pick_array = np.empty(shape=(0,len(data_unique)))

    # loop through each pick and merge the unique player dataset
    # with players who have the potential to be selected at each pick
    for i in my_picks:
        tmp = pd.merge(data_unique, pick_prob_df.loc[pick_prob_df.pick==i, ['player', 'pos', 'flag']],
                    how='left', on=['player', 'pos']).fillna(0)
        to_add = tmp.flag.values#[:-1]
        # to_add = np.append(0, to_add)

        pick_array = np.vstack([pick_array, to_add])
        print(i, pick_array.shape)

    pick_array = pick_array.astype('int')

    return pick_array

pick_prob_df = pick_prob_master[pick_prob_master.adp >= my_picks[0]].copy().reset_index(drop=True)
pick_prob_df, data_unique = get_relevant_player_prob(pick_prob_master, data, my_picks)

# use the data unique to bring down data projections and properly order
data = pd.merge(data, data_unique, on=['player', 'pos']).set_index('player')
data = data.sort_values(by=['pos', 'adp'], ascending=[True, True])

# get the pick array matrix for Ax=B
pick_array = create_pick_matrix(pick_prob_df, data_unique, my_picks)

#%%
'''
This function creates the A matrix that is critical for the ILP solution being equal
to the positional constraints specified. I identified the given pattern empirically:
1. Repeat the vector [1, 0, 0, 0, ...] N times for each player for a given position.
    The number of trailing zeros is equal to p-1 positions to draft.
2. After the above0vector is repeated N times for a given player, append a 0 before
    repeating the same pattern for the next player. Repeat for all players up until the 
    last position.
3. for the last poition, repeat the pattern N-1 times and append a 1 at the end.
This pattern allows the b vector, e.g. [1, 2, 2, 1] to set the constraints on the positions
selected by the ILP solution.
'''


results = {}
for p in data.index.unique():
    results[p] = 0

for j in range(2,500):
   
    pos_counts = list(data.pos.value_counts().sort_index())
    points = -1*data.iloc[:, j]/100
    salaries = np.array([1 for p in points])

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
    A = np.array(A)
    A = np.vstack([A, pick_array])

    A = matrix(A, tc='d')

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
    h = matrix(500, size=(1,1), tc='d')

    # generate the b matrix with the number of position constraints
    b_vec = pos_require.copy()
    num_picks = len(my_picks)*[1]
    b_vec.extend(num_picks)
    b = matrix(b_vec, size=(len(b_vec), 1), tc='d')

    # solve the integer LP problem
    (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(points))))
    x = np.array(x)[:, 0]==1

    for p in data.index[x]:
        results[p] += 1

# %%
close_picks = list(data_unique.loc[data_unique.adp <= my_picks[0]*1.75, 'player'].values)
{k: v for k, v in sorted(results.items(), key=lambda item: item[1]) if k in close_picks}
# %%
data_unique.loc[data_unique.player.isin(close_picks), ['player', 'adp']].drop_duplicates().sort_values(by='adp').iloc[:50]
# %%
