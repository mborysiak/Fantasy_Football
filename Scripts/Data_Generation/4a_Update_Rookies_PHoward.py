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
root_path = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football'
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set the database name
db_name = 'Season_Stats'
conn = sqlite3.connect(f'{root_path}/Data/Databases/{db_name}.sqlite3')

# set the year for current database download
set_year = 2024
rb_path = "Pahowdy's College Database - RB.csv"
wr_path = "Pahowdy's College Database - WR.csv"

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
    top_cols = df.columns
    bottom_cols = df.iloc[0,:].values

    prefix = ''
    good_cols = []
    for tc, bc in zip(top_cols, bottom_cols):
        
        try: tc = tc.rstrip().lstrip()
        except: pass
        
        try: bc = bc.rstrip().lstrip()
        except: pass

        if 'Unnamed' not in tc: prefix = f'{tc}_'
        good_cols.append(f'{prefix}{bc}')

    good_cols = [c.lower().replace(' ', '_').replace('-', 'to').replace('/', 'per') \
                 .replace(',', '').replace('(', '').replace(')', '') for c in good_cols]
    
    df = df.iloc[1:, :]
    df.columns = good_cols
    return df

def adjust_draft(df):
    df = df.rename(columns={'dy': 'draft_year', 'dr': 'draft_dr', 'dp': 'draft_pick'})
    draft_positions = dm.read('''SELECT player, year draft_year, Round draft_dr_fill, Pick draft_pick_fill 
                                 FROM Draft_Positions''', 'Season_Stats')
    df = pd.merge(df, draft_positions, on=['player', 'draft_year'], how='left')
    
    df.loc[df.draft_dr=='UNK', 'draft_dr'] = df.loc[df.draft_dr=='UNK', 'draft_dr_fill']
    df.loc[df.draft_pick=='UNK', 'draft_pick'] = df.loc[df.draft_pick=='UNK', 'draft_pick_fill']
    df.loc[df.draft_dr=='UDFA', 'draft_dr'] = 7

    df = df[(df.draft_year <= set_year) & ~(df.draft_dr.isnull()) & ~(df.draft_pick.isnull())].reset_index(drop=True)
    df.draft_pick = df.draft_pick.astype('int')
    df = df.assign(log_draft_pick=np.log(df.draft_pick))

    return df


def convert_to_float(df):
    for c in df.columns:
        try: 
            df[c] = df[c].fillna('1000000').astype('str').apply(lambda x: x.replace('-', '1000000') \
                                                                           .replace('#DIV/0!', '1000000') \
                                                                           .replace('%', '') \
                                                                           .replace('#VALUE!', '1000000')
                                                                           ).astype('float')
            if 'nfl_combine' in c or 'pro_day' in c:
                df.loc[df[c] >= 1000000, c] = np.nan
            else:
                df.loc[df[c] >= 1000000, c] = 0

        except: pass

    return df

def fix_combine_stats(df):
    combine_filled = dm.read('''SELECT * FROM Combine_Data_Filled''', 'Season_Stats')
    combine_filled = combine_filled.rename(columns={'year': 'draft_year'})
    
    col_rename = {
        'height': 'combine_heightlns',
        'weight': 'combine_weightlb',
        'forty': 'combine_40_time',
        'bench_press': 'combine_bench',
        'three_cone': 'combine_3cone',
        'shuffle_20_yd': 'combine_shuttle',
        'vertical': 'combine_vert',
        'broad_jump': 'combine_broad_in',
        'hand_size': 'combine_hand_size',
        'speed_score': 'combine_hass',
        'speed_score': 'combine_wass',
        'bmi': 'combine_bmi'
        }

    df = pd.merge(df, combine_filled, on=['player', 'draft_year'], how='left')

    for k, v in col_rename.items():
        df[v] = df[v].fillna('1000000').apply(lambda x: x.replace('UNK', '1000000').replace('-',  '1000000')).astype('float')
        df.loc[df[v] >= 1000000, v] = np.nan
        df[v] = df[v].fillna(df[k])

    df = df.drop([k for k in col_rename.keys()], axis=1)
    df = df.drop(['QB', 'RB', 'WR', 'TE'], axis=1)

    return df

def add_power5(df):
    df['power5'] = 0
    df.loc[df.conference.isin(['SEC', 'Pac-10', 'Big 12', 'Big Ten', 'Pac-12', 'ACC']), 'power5'] = 1
    
    df['power2'] = 0
    df.loc[df.conference.isin(['SEC', 'Big Ten']), 'power2'] = 1

    return df

def drop_pahowdy_cals(df):
    drop_cols = [c for c in df.columns if 'adj_completion_%_pahowdy_calc_pff_stats' in c]
    df = df.drop(drop_cols, axis=1)
    return df

download_path = f'/Users/{general.get_username()}/Downloads/'
move_path_path = f'{root_path}/Data/OtherData/Peter_Howard_College/{set_year}/'

rb = move_load(download_path, move_path_path, rb_path)
rb = set_columns(rb)
rb = rb.rename(columns={'name': 'player'})
rb.player = rb.player.apply(name_clean)
rb = convert_to_float(rb)
rb = adjust_draft(rb)
rb = fix_combine_stats(rb)
rb = add_power5(rb)
rb = drop_pahowdy_cals(rb)
rb = rb.loc[:, ~rb.columns.duplicated()].copy()

wr = move_load(download_path, move_path_path, wr_path)
wr = set_columns(wr)
wr = wr.rename(columns={'name': 'player'})
wr.player = wr.player.apply(name_clean)
wr = convert_to_float(wr)
wr = adjust_draft(wr)
wr = fix_combine_stats(wr)
wr = add_power5(wr)
wr = drop_pahowdy_cals(wr)
wr = wr.loc[:, ~wr.columns.duplicated()].copy()

# %%
dm.write_to_db(rb, 'Season_Stats','RB_PHoward_Raw',  'replace')
dm.write_to_db(wr, 'Season_Stats', 'WR_PHoward_Raw','replace')

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

rb = dm.read(f'''SELECT * FROM RB_PHoward_Raw''', 'Season_Stats')
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

drop_cols = [c for c in rb.columns if 'nfl_stats' in c or 'conference' in c]
drop_cols.extend(['cfb_id', 'nflfastr', 'pff', 'pfr_id', 'pos', 'dob', 'landing_team', 'college'])
rb = rb.drop(drop_cols, axis=1)

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

wr = dm.read(f'''SELECT * FROM WR_PHoward_Raw''', 'Season_Stats')
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

drop_cols = [c for c in wr.columns if 'nfl_stats' in c or 'conference' in c]
drop_cols.extend(['cfb_id', 'nflfastr', 'pff', 'pfr_id', 'pos', 'dob', 'landing_team', 'college'])
wr = wr.drop(drop_cols, axis=1)

wr['fp_per_game_next'] = wr[['fp_per_game_0', 'fp_per_game_1', 'fp_per_game_2']].mean(axis=1)
wr['games_next'] = wr[['games_0', 'games_1', 'games_2']].mean(axis=1)
wr = wr.rename(columns={'fp_per_game_0': 'fp_per_game', 'games_0': 'games'})
wr = wr.drop(['fp_per_game_1', 'fp_per_game_2', 'games_1', 'games_2'], axis=1)

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
df = wr.copy()

display(df.corr()[[corr_col]].sort_values(by = corr_col).iloc[:40])
df.corr()[[corr_col]].sort_values(by = corr_col).iloc[-40:]
# %%
