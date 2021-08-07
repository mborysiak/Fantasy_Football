
# %%

from ff.db_operations import DataManage
from ff import general

# last year's statistics and adp to pull and append to database
year = 2020

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

import pandas as pd
import os
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np

#==========
# Clean the ADP data
#==========

'''
Cleaning the ADP data by selecting relevant features, and extracting the name and team
from the combined string column. Note that the year is not shifted back because the 
stats will be used to calculate FP/G for the rookie in that season, but will be removed
prior to training. Thus, the ADP should match the year from the stats.
'''

def clean_adp(data_adp, year):

    #--------
    # Select relevant columns and clean special figures
    #--------

    data_adp['year'] = year

    # set column names to what they are after pulling
    df_adp = data_adp.iloc[:, 1:].rename(columns={
        1: 'Player', 
        2: 'Avg Pick',
        3: 'Min Pick',
        4: 'Max Pick',
        5: '# Drafts Selected In'
    })

    # selecting relevant columns and dropping na
    df_adp = df_adp[['Player', 'year', 'Avg Pick']].dropna()

    # convert year to float and move back one year to match with stats
    df_adp['year'] = df_adp.year.astype('float')

    # selecting team and player name information from combined string
    df_adp['Tm'] = df_adp.Player.apply(team_select)
    df_adp['Player'] = df_adp.Player.apply(name_select)
    df_adp['Player'] = df_adp.Player.apply(name_clean)
    df_adp = df_adp[df_adp.Player != '1 Page:']
    
    # format and rename columns
    df_adp = df_adp[['Player', 'Tm', 'year', 'Avg Pick']]

    colnames_adp = {
        'Player': 'player',
        'Tm': 'team',
        'year': 'year',
        'Avg Pick': 'avg_pick'
    }

    df_adp = df_adp.rename(columns=colnames_adp)
    
    return df_adp


# %%
def get_adp(year, pos, rook=0):
    
    # get the dataset based on year + position
    URL = f'https://www71.myfantasyleague.com/{year+1}/reports?R=ADP&POS={pos}&ROOKIES={rook}&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=RECENT&PAGE=ALL'
    data = pd.read_html(URL)[1]

    # clean the dataset and print out check dataset
    df = clean_adp(data, year)[['player', 'avg_pick']]
    print(df.head(10))

    df = df[df.player!='Player Hint:'].reset_index(drop=True)
    
    # log the avg_pick to match existing
    df.avg_pick = np.log(df.avg_pick.astype('float'))

    return df


def merge_with_existing(df_adp, pos, year):     
    
    # read in existing dataset and extract columns
    if 'Rookie' in pos:
        df = pd.read_sql_query(f'''SELECT * FROM {pos}_Stats WHERE draft_year={year+1}''', conn)
    else:
        df = pd.read_sql_query(f'''SELECT * FROM {pos}_Stats WHERE year={year}''', conn)
    
    if pos == 'QB':
        df = df.rename(columns={'qb_avg_pick': 'avg_pick'})
    
    print('Initial Shape:', df.shape)
    cols = df.columns

    # find out which players no longer have ADPs and retain original ADP
    extra_player = df.loc[~df.player.isin(df_adp.player), ['player', 'avg_pick']]
    df_adp = pd.concat([df_adp, extra_player], axis=0)

    # drop old ADP and merge new ADP + reorder columns
    df = pd.merge(df.drop('avg_pick', axis=1), df_adp, on='player')
    df = df[cols]
    print('Final Shape:', df.shape)
    
    if pos=='QB':
        df = df.rename(columns={'avg_pick': 'qb_avg_pick'})
    
    return df


# %%
def db_overwrite(df, pos, year):
    dm.delete_from_db('Season_Stats', f'{pos}_Stats', f'''year={year}''')
    dm.write_to_db(df, db_name='Season_Stats', table_name=f'{pos}_Stats', if_exist='append')
    print(f'Successfully overwrote {pos}_Stats for Year {year}')

# %%
# pulling historical player adp for runningbacks
pos = 'RB'
rb_adp = get_adp(year, pos)
rb = merge_with_existing(rb_adp, pos, year)

#%%
db_overwrite(rb, pos, year)

# %%
# pulling historical player adp for runningbacks
pos = 'WR'
wr_adp = get_adp(year, pos)
wr = merge_with_existing(wr_adp, pos, year)

#%%
db_overwrite(wr, pos, year)


# %%
# pulling historical player adp for runningbacks
pos = 'QB'
qb_adp = get_adp(year, pos)
qb = merge_with_existing(qb_adp, pos, year)


# %%
db_overwrite(qb, pos, year)

# %%
# pulling historical player adp for runningbacks
pos = 'TE'
te_adp = get_adp(year, pos)
te = merge_with_existing(te_adp, pos, year)

# %%
db_overwrite(te, pos, year)

#%%
# pulling historical player adp for runningbacks
rookie_rb_adp = get_adp(year, 'RB', rook=1)
rookie_rb = merge_with_existing(rb_adp, 'Rookie_RB', year)

rookie_rb_adp['draft_year'] = year+1
rookie_rb_adp['pos'] = 'RB'
rookie_rb_adp['avg_pick'] = np.round(np.exp(rookie_rb_adp.avg_pick), 1)
rookie_rb_adp = rookie_rb_adp[['player', 'draft_year', 'pos', 'avg_pick']]

# %%
dm.delete_from_db('Season_Stats', 'Rookie_RB_Stats', f'''draft_year={year+1}''')
dm.write_to_db(rookie_rb, db_name='Season_Stats', table_name='Rookie_RB_Stats', if_exist='append')
print(f'Successfully overwrote Rookie_RB_Stats for Year {year+1}')

dm.delete_from_db('Season_Stats', 'Rookie_ADP', f'''draft_year={year+1} AND pos='RB' ''')
dm.write_to_db(rookie_rb_adp, db_name='Season_Stats', table_name='Rookie_ADP', if_exist='append')
print(f'Successfully overwrote Rookie_ADP for Year {year+1} AND Pos=RB')

# %%
# pulling historical player adp for runningbacks
rookie_wr_adp = get_adp(year, 'WR', rook=1)
rookie_wr = merge_with_existing(rb_adp, 'Rookie_WR', year)

rookie_wr_adp['draft_year'] = year+1
rookie_wr_adp['pos'] = 'WR'
rookie_wr_adp['avg_pick'] = np.round(np.exp(rookie_wr_adp.avg_pick), 1)
rookie_wr_adp = rookie_wr_adp[['player', 'draft_year', 'pos', 'avg_pick']]

# %%
dm.delete_from_db('Season_Stats', 'Rookie_WR_Stats', f'''draft_year={year+1}''')
dm.write_to_db(rookie_wr, db_name='Season_Stats', table_name='Rookie_WR_Stats', if_exist='append')
print(f'Successfully overwrote Rookie_WR_Stats for Year {year+1}')

dm.delete_from_db('Season_Stats', 'Rookie_ADP', f'''draft_year={year+1} AND pos='WR' ''')
dm.write_to_db(rookie_wr_adp, db_name='Season_Stats', table_name='Rookie_ADP', if_exist='append')
print(f'Successfully overwrote Rookie_ADP for Year {year+1} AND Pos=WR')

# %%
