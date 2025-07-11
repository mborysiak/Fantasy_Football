
#%%

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc

# set to last year
year = 2024

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

import pandas as pd
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np
import os

DB_NAME = 'Season_Stats_New'

#%%

def save_pff_stats(stat_type, set_year):
    if stat_type=='QB': fname = 'passing_summary'
    elif stat_type=='Rec': fname='receiving_summary'
    elif stat_type=='Rush': fname='rushing_summary'
    elif stat_type=='Oline': fname='offense_blocking'

    try:
        os.replace(f"/Users/borys/Downloads/{fname}.csv", 
                   f'{root_path}/Data/OtherData/PFF_Stats/{set_year}_{fname}.csv')
    except: 
        pass
    
    df = pd.read_csv(f'{root_path}/Data/OtherData/PFF_Stats/{set_year}_{fname}.csv')
    df.player = df.player.apply(dc.name_clean)
    df['year'] = set_year

    dm.delete_from_db(DB_NAME, f'PFF_{stat_type}_Stats', f"year={set_year}", create_backup=False)
    dm.write_to_db(df, DB_NAME, f'PFF_{stat_type}_Stats', 'append')

    return df

for stat_type in ['QB', 'Rec', 'Rush', 'Oline']:
    df = save_pff_stats(stat_type, year)

# %%

df = dm.read(f'''SELECT *
                    FROM PFF_Rush_Stats
                    ORDER BY year, targets DESC
                ''', 
                DB_NAME).drop(['player_id', 'position', 'team_name'], axis=1)

df
# %%
