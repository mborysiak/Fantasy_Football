#%%
# # User Inputs

# +
year = 2023

from ff.db_operations import DataManage
from ff import general, data_clean as dc

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

download_path = f'/Users/{general.get_username()}/Downloads/'
fo_path = f'{root_path}/Data/OtherData/Football_Outsiders/'


# # Load Packages

import pandas as pd
import numpy as np
from zData_Functions import *
import re
import os

def move_load(fname):
    try:
        os.replace(f'{download_path}/{fname}',
                   f'{fo_path}/{fname}')
        print(f'Moved {fname} to Football Outsiders Folder')
    except:
        print("File doesn't exist in downloads")

    return pd.read_csv(f'{fo_path}/{fname}')

#%%



fname = f'{year} Basic Offensive Line Stats.csv'
oline_df = move_load(fname)

oline_df.columns = range(oline_df.shape[1])

cols = {
    0: 'run_block_rank',
    1: 'team', 
    2: 'adjust_line_yds',
    3: 'rb_yds', 
    4: 'power_success_pct',
    5: 'power_rank',
    6: 'stuffed_pct',
    7: 'stuffed_rank',
    8: 'second_level_yds',
    9: 'second_level_rank',
    10: 'open_field_yds',
    11: 'open_field_rank',
    12: 'pass_block_rank', 
    13: 'sacks_allowed', 
    14: 'adjusted_sack_rate'
}

oline_df = oline_df.rename(columns=cols)

oline_df['year'] = year
        
oline_df['power_success_pct'] = oline_df.power_success_pct.apply(name_clean)
oline_df['stuffed_pct'] = oline_df.stuffed_pct.apply(name_clean)
oline_df['adjusted_sack_rate'] = oline_df.adjusted_sack_rate.apply(name_clean)

oline_df = oline_df[['team', 'year', 'adjust_line_yds', 'rb_yds', 'power_success_pct',
                    'stuffed_pct', 'second_level_yds', 'open_field_yds', 'pass_block_rank',
                    'sacks_allowed', 'adjusted_sack_rate']]

team_convert = {
    
    'ARI': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BUF': 'BUF',
    'CAR': 'CAR',
    'CHI': 'CHI',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'DAL': 'DAL',
    'DEN': 'DEN',
    'DET': 'DET',
    'GB': 'GNB',
    'HOU': 'HOU',
    'IND': 'IND',
    'JAX': 'JAX',
    'KC': 'KAN',
    'LAC': 'LAC',
    'SD': 'LAC',
    'LAR': 'LAR',
    'STL': 'LAR',
    'MIA': 'MIA',
    'MIN': 'MIN',
    'NE': 'NWE',
    'NO': 'NOR',
    'NYG': 'NYG',
    'NYJ': 'NYJ',
    'LVR': 'LVR',
    'LV': 'LVR',
    'OAK': 'LVR',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SF': 'SFO',
    'TB': 'TAM',
    'TEN': 'TEN',
    'WAS': 'WAS'
}

oline_df['team'] = oline_df.team.map(team_convert)
oline_df = dc.convert_to_float(oline_df)
oline_df = oline_df.dropna()

dm.delete_from_db('Season_Stats', 'OLine_Stats', f"year={year}")
dm.write_to_db(oline_df, 'Season_Stats', 'OLine_Stats', if_exist='append')


#%%

fname = f'{year} Team DVOA Ratings Offense.csv'
team_off = move_load(fname)

team_off.columns = ['team', 'dvoa_rank', 'total_dvoa', 'prev_year_rank',
                    'weighted_dvoa_rank', 'weighted_dvoa', 'pass_dvoa_rank',
                    'pass_dvoa', 'rush_dvoa_rank', 'rush_dvoa', 
                    'unadj_total_dvoa_rank', 'unadj_total_dvoa', 'unadj_pass_dvoa_rank',
                    'unadj_pass_dvoa', 'unadj_rush_dvoa_rank', 'unadj_rush_dvoa',
                    'dvoa_variance_rank', 'dvoa_variance', 'schedule_rank', 'schedule_dvoa']
team_off['year'] = year
    
cols = ['total_dvoa', 'weighted_dvoa', 'pass_dvoa', 'rush_dvoa', 'unadj_total_dvoa',
        'unadj_pass_dvoa', 'unadj_rush_dvoa', 'dvoa_variance', 'schedule_dvoa']
for col in cols:
    team_off[col] = team_off[col].apply(name_clean)

team_convert = {
    'ARI': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BUF': 'BUF',
    'CAR': 'CAR',
    'CHI': 'CHI',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'DAL': 'DAL',
    'DEN': 'DEN',
    'DET': 'DET',
    'GB': 'GNB',
    'HOU': 'HOU',
    'IND': 'IND',
    'JAX': 'JAX',
    'KC': 'KAN',
    'LACH': 'LAC',
    'SD': 'LAC',
    'LAC': 'LAC',
    'LAR': 'LAR', 
    'LARM': 'LAR',
    'STL': 'LAR',
    'MIA': 'MIA',
    'MIN': 'MIN',
    'NE': 'NWE',
    'NO': 'NOR',
    'NYG': 'NYG',
    'NYJ': 'NYJ',
    'OAK': 'LVR',
    'LVR': 'LVR',
    'LV': 'LVR',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SF': 'SFO',
    'TB': 'TAM',
    'TEN': 'TEN',
    'WAS': 'WAS'
}

team_off['team'] = team_off.team.map(team_convert)
team_off = dc.convert_to_float(team_off)

dm.delete_from_db('Season_Stats', 'Team_Offense_DVOA', f"year={year}")
dm.write_to_db(team_off, 'Season_Stats', 'Team_Offense_DVOA', if_exist='append')
# %%

