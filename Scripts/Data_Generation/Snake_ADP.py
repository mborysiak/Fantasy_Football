#%%

from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def append_to_db(df, db_name='Season_Stats', table_name='NA', if_exist='append'):

    import sqlite3
    import os
    import datetime as dt
    from shutil import copyfile
    
    #--------
    # Append pandas df to database in Github
    #--------
    
    username = os.getlogin()
    
    # move into the local database directory
    os.chdir(f'/Users/{username}/Documents/Github/Fantasy_Football/Data/Databases/')
    
    # copy the current database over to new folder with timestamp appended
    today_time = dt.datetime.now().strftime('_%Y_%m_%d_%M')
    copyfile(db_name + '.sqlite3', 'DB_Versioning/' + db_name + today_time + '.sqlite3')

    conn = sqlite3.connect(db_name + '.sqlite3')

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )

    #--------
    # Append pandas df to database in OneDrive
    #--------

    os.chdir(f'/Users/{username}/OneDrive/FF/DataBase/')
    copyfile(db_name + '.sqlite3', 'DB_Versioning/' + db_name + today_time + '.sqlite3')

    conn = sqlite3.connect(db_name + '.sqlite3')
    
    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )

def name_clean(col):
    char_remove = re.sub('[\*\+\%\,]', '', str(col))
    char_remove = char_remove.replace('III', '')
    char_remove = char_remove.replace('II', '')
    char_remove = char_remove.replace('.', '')
    char_remove = char_remove.replace('Jr', '')
    char_remove = char_remove.replace('Jr.', '')
    char_remove = char_remove.rstrip().lstrip()

    return char_remove


#----------------
# Calculate the Pick Probability based on Initial Draft Choice
#----------------

def pick_likelihood(row):
    '''
    INPUT: A dataframe row containing ADP information and an index
    corresponding to the player.

    OUTPUT: A numpy matrix with a row for each pick probability of the player
    and columns indicating player index + pick prob
    '''

    # extract out the player index and ADP information from the row
    idx = row[0]
    min_adp = row[1]
    adp = row[2]
    max_adp = row[3]

    # calculate the approximate standard deviation using Range information
    adp_std = (max_adp - min_adp) / 4

    # calculate a truncated normal distribution based on min/max ADP
    trunc_dist = stats.truncnorm((min_adp - adp) /  adp_std, 
                                 (max_adp - adp) / adp_std, 
                                 loc=adp, 
                                 scale=adp_std)

    # create random distribution and round to whole numbers for each pick
    estimates = trunc_dist.rvs(1000)
    estimates = np.around(estimates, decimals=0)

    # get the counts for each draft pick and create an index array
    unique, counts = np.unique(estimates, return_counts=True)
    idx = np.full(len(counts), idx) 

    # return array with index, pick number, and pick counts
    return np.transpose(np.array([idx, unique, counts]))


#-----------------
# Get the Picks associated with each Start Draft Slot
#-----------------

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


#----------------
# Get player ADP
#----------------

# read in CSV of NFFC ADP and filter to position players + relevant columns
adp = pd.read_csv('c:/Users/mborysia/Downloads/ADP.tsv', delimiter='\t')
adp = adp[~adp['Position(s)'].isin([ 'TDSP', 'TK', 'K'])].reset_index(drop=True)
adp = adp[['Position(s)', 'Player', 'ADP', 'Min Pick', 'Max Pick']]
adp.columns = ['pos', 'player', 'adp', 'min_pick', 'max_pick']

# clean up the name
adp.player = adp.player.apply(lambda x: x.split(',')[1].rstrip() + ' ' + x.split(',')[0].lstrip())

# select top 240 players (20 rounds)
adp = adp.iloc[:240].reset_index()

# create an approximate salary curve for the each player based on ADP
adp['Salary'] = 100 / (1 + 0.1*adp.index)

# prepare salary output
output = adp[['player', 'Salary']].copy()
output['year'] = 2020
output['league'] = 'nffc'

append_to_db(output, 'Simulation', 'Salaries', 'append')

# #----------------
# # Get player ADP Distribution
# #----------------

# # calculate distribution for each pick for each player
# pick_prob_df = np.empty(shape=(1,3))
# for _, row in adp[['index', 'min_pick', 'adp', 'max_pick']].iterrows():
#     pick_prob_df = np.vstack((pick_prob_df, pick_likelihood(row)))

# #----------------
# # Use Player Pick-Prob to adjust "Salaries"
# #----------------

# my_picks = get_picks(5)

# # convert random distributions to likelihood of a player being picked at a given slot
# pick_prob_df = pd.DataFrame(pick_prob_df, columns=['index', 'pick', 'pick_prob'])
# pick_prob_df = pick_prob_df[pick_prob_df.pick.isin(my_picks)].groupby('index').agg({'pick_prob': 'sum'}).reset_index()

# # subtract prob from 1 so that very likely picks will drop the salary when multiplied
# pick_prob_df.pick_prob = 1 - (pick_prob_df.pick_prob / 1000)
# adp = pd.merge(adp, pick_prob_df, on='index')

# adp['SalaryAdj'] = adp.Salary * adp.pick_prob 
# salary_cap = adp.loc[adp.index.isin(my_picks), 'Salary'].sum() 



# %%
