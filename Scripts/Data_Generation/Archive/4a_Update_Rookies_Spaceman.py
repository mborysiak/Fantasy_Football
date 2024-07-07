#%%
import pandas as pd
import os
import sqlite3
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np
import datetime as dt

from ff.db_operations import DataManage
from ff import general, data_clean as dc

# set the root path and database management object
root_path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football'
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set the database name
db_name = 'Season_Stats'
conn = sqlite3.connect(f'{root_path}/Data/Databases/{db_name}.sqlite3')

# set the year for current database download
set_year = 2023
rb_path = 'FF_Spaceman Raw Database 5_7_2023 Public - RB.csv'
wr_path = 'FF_Spaceman Raw Database 5_7_2023 Public - WR.csv'
# link to updated files: https://www.patreon.com/posts/free-college-for-81188980

# link to Peter Howard: https://www.patreon.com/pahoward

#%%

def move_load(download_path, move_path, fname):

    if not os.path.exists(move_path):
        os.makedirs(move_path)

    try:
        os.replace(f'{download_path}/{fname}',
                   f'{move_path}/{fname}')
        print(f'Moved {fname} to {move_path}')
    except:
        print("File doesn't exist in downloads")

    return pd.read_csv(f'{move_path}/{fname}')

def set_columns(df):
    first_row = df.iloc[0, :].fillna(method='ffill').values
    second_row = df.iloc[1, :].fillna(method='ffill').values
    cols = [f'{first_row[i]}_{second_row[i]}'.replace('nan_', '') for i in range(len(first_row))]
    cols = [c.lower().replace(' ', '_').replace('-', 'to').replace('/', 'per').replace(',', '') for c in cols]
    df.columns = cols
    df = df.iloc[2:, :]
    return df

def adjust_draft(df):
    df.draft_dr = df.draft_dr.apply(lambda x: x.replace('UDFA', '7')).astype('int')
    df.draft_dp = df.draft_dp.apply(lambda x: x.replace('UDFA', '32')).astype('int')
    df = df.assign(draft_pick=(df.draft_dr-1) * 32 + df.draft_dp)
    df = df.assign(log_draft_pick=np.log(df.draft_pick))
    return df


def convert_to_float(df):
    for c in df.columns:
        try: 
            df[c] = df[c].fillna('1000000').astype('str').apply(lambda x: x.replace('-', '1000000') \
                                                                           .replace('#DIV/0!', '1000000') \
                                                                           .replace('%', '') \
                                                                           .replace('#VALUE!', '1000000')).astype('float')
            if 'nfl_combine' in c or 'pro_day' in c:
                df.loc[df[c] >= 1000000, c] = np.nan
            else:
                df.loc[df[c] >= 1000000, c] = 0

        except: pass

    return df

def fix_combine_stats(df, ):
    for c in ['40_time', 'bench', 'vertical', 'broad','shuttle', '3_cone']:
        try: df.loc[df[f'nfl_combine_{c}'].isnull(), f'nfl_combine_{c}'] = df.loc[df[f'nfl_combine_{c}'].isnull(), f'pro_day_{c}']
        except: df.loc[df[f'combine_{c}'].isnull(), f'combine_{c}'] = df.loc[df[f'combine_{c}'].isnull(), f'pro_day_{c}']

    df = df.drop([c for c in df.columns if 'pro_day' in c], axis=1)
    return df

def add_power5(df):
    df['power5'] = 0
    try: df.loc[df.rb_conf.isin(['SEC', 'Pac-10', 'Big 12', 'Big Ten', 'Pac-12', 'ACC']), 'power5'] = 1
    except: df.loc[df.ff_spaceman_conf.isin(['SEC', 'Pac-10', 'Big 12', 'Big Ten', 'Pac-12', 'ACC']), 'power5'] = 1

    return df

download_path = f'/Users/{general.get_username()}/Downloads/'
move_path_path = f'{root_path}/Data/OtherData/Spaceman_College/{set_year}/'

rb = move_load(download_path, move_path_path, rb_path)
rb = set_columns(rb)
rb = convert_to_float(rb)
rb = adjust_draft(rb)
rb = fix_combine_stats(rb)
rb = add_power5(rb)
rb = rb.rename(columns={'rb_player': 'player', 'draft_draft_year': 'draft_year'})
rb.player = rb.player.apply(name_clean)

wr = move_load(download_path, move_path_path, wr_path)
wr = set_columns(wr)
wr = convert_to_float(wr)
wr = adjust_draft(wr)
wr = fix_combine_stats(wr)
wr = add_power5(wr)
wr = wr.rename(columns={'wr_player': 'player', 'draft_draft_year': 'draft_year'})
wr.player = wr.player.apply(name_clean)


# %%
dm.write_to_db(rb, 'Season_Stats','RB_Spaceman_Raw',  'replace')
dm.write_to_db(wr, 'Season_Stats', 'WR_Spaceman_Raw','replace')

# %%


def add_avg_pick(df):
    rookie_adp = dm.read('''SELECT * FROM Rookie_ADP''', 'Season_Stats').drop(['team', 'pos'], axis=1)
    df = pd.merge(df, rookie_adp, on=['player', 'draft_year'], how='left')
    df.avg_pick = df.avg_pick.fillna(250)
    df.loc[df.avg_pick > 250, 'avg_pick'] = 250
    df = df.assign(avg_pick = np.log(df.avg_pick))
    df['orig_avg_pick'] = np.exp(df.avg_pick)
    return df

def add_team(df, pos):
    team = dm.read(f'''SELECT player, year draft_year, team 
                      FROM Draft_Positions
                      WHERE pos='{pos}'
                      ''', 'Season_Stats')
    df = pd.merge(df, team, on=['player', 'draft_year'], how='left')
    return df

def add_stats(df, year_diff):
    target_data = pd.read_sql_query(f'''SELECT player, 
                                              games games_{year_diff},
                                              team team_fill, 
                                              year-{year_diff} draft_year, 
                                              rush_yd_per_game rush_yd_per_game_{year_diff},
                                              rec_per_game rec_per_game_{year_diff},
                                              rec_yd_per_game rec_yd_per_game_{year_diff}, 
                                              td_per_game td_per_game_{year_diff} 
                                   FROM RB_Stats''', conn)
    df = pd.merge(df, target_data, on=['player', 'draft_year'], how='left')
    df[f'fp_per_game_{year_diff}'] = (df[[f'rush_yd_per_game_{year_diff}', f'rec_per_game_{year_diff}', f'rec_yd_per_game_{year_diff}', 
                                            f'td_per_game_{year_diff}']]*[0.1, 0.5, 0.1, 7]).sum(axis=1)

    df.loc[df.team.isnull(), 'team'] = df.loc[df.team.isnull(), 'team_fill']
    df = df.drop(['team_fill', f'rec_per_game_{year_diff}', f'rec_yd_per_game_{year_diff}', f'td_per_game_{year_diff}', f'rush_yd_per_game_{year_diff}'], axis=1)
    df.loc[df[f'games_{year_diff}'].isnull(), f'fp_per_game_{year_diff}'] = np.nan
    
    return df

rb = dm.read(f'''SELECT * FROM RB_Spaceman_Raw''', 'Season_Stats')
rb = add_avg_pick(rb)
rb = add_team(rb, 'RB')
rb = add_stats(rb, 0)
rb = add_stats(rb, 1)
rb = add_stats(rb, 2)

rb = rb[ ((rb.games_0 > 6) \
         | (rb.games_1 > 6)
         | (rb.games_2 > 6)) \

         & ((rb.fp_per_game_0 > 0) 
            | (rb.fp_per_game_1 > 0)
            | (rb.fp_per_game_2 > 0)) \

         | (rb.draft_year==set_year)].reset_index(drop=True)

rb = rb.drop(['rb_school', 'rb_conf', 'rb_dob',
              'nfl_stats_finishes_and_milestones_top_5_rb', 
              'nfl_stats_finishes_and_milestones_top_24_rb',
              'nfl_stats_finishes_and_milestones_top_12_rb', 
              'nfl_stats_finishes_and_milestones_avg_ppg_yr_1to3'
              ], axis=1)

rb['fp_per_game_next'] = rb[['fp_per_game_0', 'fp_per_game_1', 'fp_per_game_2']].mean(axis=1)
rb['games_next'] = rb[['games_0', 'games_1', 'games_2']].mean(axis=1)

rb = rb.rename(columns={'fp_per_game_0': 'fp_per_game', 'games_0': 'games'})
rb = rb.drop(['fp_per_game_1', 'fp_per_game_2', 'games_1', 'games_2'], axis=1)

rb = rb.dropna(subset=['team']).reset_index(drop=True)
for c in rb.isnull().sum()[rb.isnull().sum()>0].index:
    if 'game' in c:
        rb[c] = rb[c].fillna(0)
    else:
        rb[c] = rb[c].fillna(rb[c].median())
rb=rb.assign(pos='RB')
dm.write_to_db(rb, 'Season_Stats', 'Rookie_RB_Stats', 'replace', create_backup=True)

#%%

def add_stats(df, year_diff):
    target_data = pd.read_sql_query(f'''SELECT player, 
                                              games games_{year_diff},
                                              team team_fill, 
                                              year-{year_diff} draft_year, 
                                              rec_per_game rec_per_game_{year_diff},
                                              rec_yd_per_game rec_yd_per_game_{year_diff}, 
                                              td_per_game td_per_game_{year_diff} 
                                   FROM WR_Stats''', conn)
    df = pd.merge(df, target_data, on=['player', 'draft_year'], how='left')
    df[f'fp_per_game_{year_diff}'] = (df[[f'rec_per_game_{year_diff}', f'rec_yd_per_game_{year_diff}', 
                                          f'td_per_game_{year_diff}']]*[0.5, 0.1, 7]).sum(axis=1)

    df.loc[df.team.isnull(), 'team'] = df.loc[df.team.isnull(), 'team_fill']
    df = df.drop(['team_fill', f'rec_per_game_{year_diff}', f'rec_yd_per_game_{year_diff}', f'td_per_game_{year_diff}'], axis=1)
    df.loc[df[f'games_{year_diff}'].isnull(), f'fp_per_game_{year_diff}'] = np.nan
    return df

wr = dm.read(f'''SELECT * FROM WR_Spaceman_Raw''', 'Season_Stats')
wr = add_avg_pick(wr)
wr = add_team(wr, 'WR')
wr = add_stats(wr, 0)
wr = add_stats(wr, 1)
wr = add_stats(wr, 2)

wr = wr[ ((wr.games_0 > 6) \
         | (wr.games_1 > 6)
         | (wr.games_2 > 6)) \

         & ((wr.fp_per_game_0 > 0) 
            | (wr.fp_per_game_1 > 0)
            | (wr.fp_per_game_2 > 0)) \

         | (wr.draft_year==set_year)].reset_index(drop=True)

wr = wr.drop(['ff_spaceman_school', 'ff_spaceman_conf', 'ff_spaceman_dob',
              'ms_rush_atts_19', 'career_average_kr_tds',
              'career_average_punt_returns', 'career_average_pr_yards',
              'career_average_pr_tds', 
               'nfl_stats_finishes_and_milestones_top_5_wr',
              'nfl_stats_finishes_and_milestones_top_12_wr',
              'nfl_stats_finishes_and_milestones_top_24_wr', 
              'nfl_stats_finishes_and_milestones_avg_ppg_yr_1to3'
       ], axis=1)

wr['fp_per_game_next'] = wr[['fp_per_game_0', 'fp_per_game_1', 'fp_per_game_2']].mean(axis=1)
wr['games_next'] = wr[['games_0', 'games_1', 'games_2']].mean(axis=1)
wr = wr.rename(columns={'fp_per_game_0': 'fp_per_game', 'games_0': 'games'})
wr = wr.drop(['fp_per_game_1', 'fp_per_game_2', 'games_1', 'games_2'], axis=1)

# wr = wr.drop('fp_per_game', axis=1).rename(columns={'nfl_stats_finishes_and_milestones_avg_ppg_yr_1to3':'fp_per_game'})

wr = wr.dropna(subset=['team']).reset_index(drop=True)

for c in wr.isnull().sum()[wr.isnull().sum()>0].index:
    if 'game' in c:
        wr[c] = wr[c].fillna(0)
    else:
        wr[c] = wr[c].fillna(wr[c].median())

wr = wr.assign(pos='WR')
dm.write_to_db(wr, 'Season_Stats', 'Rookie_WR_Stats', 'replace', create_backup=True)

# %%
corr_col = 'fp_per_game_next'
df = rb.copy()

display(df.corr()[[corr_col]].sort_values(by = corr_col).iloc[:40])
df.corr()[[corr_col]].sort_values(by = corr_col).iloc[-40:]
# %%
