
# %%
import sqlite3
import os

# last year's statistics and adp to pull and append to database
year = 2019

# name of database to append new data to
db_name = 'Season_Stats'
db_path = f'/Users/{os.getlogin()}/Documents/GitHub/Fantasy_Football/Data/Databases/'
conn = sqlite3.connect(db_path + db_name + '.sqlite3')

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
    URL = f'https://www71.myfantasyleague.com/{year+1}/reports?R=ADP&POS={pos}&ROOKIES={rook}&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=AUG15&PAGE=ALL'
    data = pd.read_html(URL)[1]

    # clean the dataset and print out check dataset
    df = clean_adp(data, year)[['player', 'avg_pick']]
    print(df.head(10))
    
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
def db_overwrite(df, pos, year, conn):
    conn.cursor().execute(f'''delete from {pos}_stats where year={year}''')
    conn.commit()
    append_to_db(df, db_name='Season_Stats', table_name=f'{pos}_Stats', if_exist='append')
    print(f'Successfully overwrote {pos}_Stats for Year {year}')

# %% [markdown]
# ## RB ADP

# %%
# pulling historical player adp for runningbacks
pos = 'RB'
rb_adp = get_adp(year, pos)
rb = merge_with_existing(rb_adp, pos, year)


# %%
db_overwrite(rb, pos, year, conn)

# %% [markdown]
# ## WR ADP

# %%
# pulling historical player adp for runningbacks
pos = 'WR'
wr_adp = get_adp(year, pos)
wr = merge_with_existing(wr_adp, pos, year)


# %%
db_overwrite(wr, pos, year, conn)

# %% [markdown]
# ## QB ADP

# %%
# pulling historical player adp for runningbacks
pos = 'QB'
qb_adp = get_adp(year, pos)
qb = merge_with_existing(qb_adp, pos, year)


# %%
db_overwrite(qb, pos, year, conn)

# %% [markdown]
# ## TE ADP

# %%
# pulling historical player adp for runningbacks
pos = 'TE'
te_adp = get_adp(year, pos)
te = merge_with_existing(te_adp, pos, year)


# %%
db_overwrite(te, pos, year, conn)

# %% [markdown]
# ## Rookie RB ADP

# %%
# pulling historical player adp for runningbacks
rookie_rb_adp = get_adp(year, 'RB', rook=1)
rookie_rb = merge_with_existing(rb_adp, 'Rookie_RB', year)


# %%
conn.cursor().execute(f'''delete from Rookie_RB_Stats where draft_year={year+1}''')
conn.commit()
append_to_db(rookie_rb, db_name='Season_Stats', table_name='Rookie_RB_Stats', if_exist='append')
print(f'Successfully overwrote Rookie_RB_Stats for Year {year+1}')

# %% [markdown]
# ## Rookie WR ADP

# %%
# pulling historical player adp for runningbacks
rookie_wr_adp = get_adp(year, 'WR', rook=1)
rookie_wr = merge_with_existing(rb_adp, 'Rookie_WR', year)


# %%
conn.cursor().execute(f'''delete from Rookie_WR_Stats where draft_year={year+1}''')
conn.commit()
append_to_db(rookie_wr, db_name='Season_Stats', table_name='Rookie_WR_Stats', if_exist='append')
print(f'Successfully overwrote Rookie_WR_Stats for Year {year+1}')


# %%



