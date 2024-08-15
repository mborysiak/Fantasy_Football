#%%

import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
import warnings
from zData_Functions import *
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm_daily = DataManage(db_path)

root_path = ffgeneral.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm_ff = DataManage(db_path)

pd.set_option('display.max_columns', 999)


#%%

for pos in ['QB', 'RB', 'WR', 'TE']:

    df = dm_daily.read(f'''SELECT * 
                           FROM {pos}_Stats
                           WHERE (season >= 2021 AND week <= 17)
                                  OR (season < 2021 AND week <= 16)
                        ''', 'FastR_Beta')

    df = df[~((df.player == 'Adrian Peterson') & (df.team=='CHI'))].reset_index(drop=True)
    df = df[~((df.player == 'Steve Smith') & (df.team.isin(['NYG', 'PHI', 'LAR'])))].reset_index(drop=True)
    df = df[~((df.player == 'Mike Williams') & (df.season < 2017))].reset_index(drop=True)

    df = df[~((df.player=='Trey Mcbride') & (df.season==2023) & (df.week < 8))].reset_index(drop=True)
    df = df[~((df.player=='Devon Achane') & (df.season==2023) & (df.week.isin([2,11])))].reset_index(drop=True)

    if pos == 'QB':
        df['total_plays'] = df.pass_qb_dropback_sum + df.rush_rush_attempt_sum
        df = df[df.total_plays > 15].reset_index(drop=True)
    
    stat_cols = [c for c in df.columns if c not in ('player', 'team', 'season', 'week', 'position')]
    df_all = df.groupby(['player', 'season']).agg({'week': 'count'}).reset_index()
    df_all = df_all.rename(columns={'week': 'games'})

    for agg_type in ['sum', 'mean', 'max']:
        print(agg_type)
        agg_stats = {c: agg_type for c in stat_cols}
        df_agg = df.groupby(['player', 'season']).agg(agg_stats).reset_index()
        df_agg = df_agg.rename(columns={c: f'{agg_type}_{c}' for c in df_agg.columns if c not in ('player', 'season')})
        df_all = pd.merge(df_all, df_agg, on=['player', 'season'], how='left')

    # df_all = df_all[df_all.games > 8].sort_values(by=['player', 'season'])
    df_all['fantasy_pts_per_game'] = (df_all['sum_fantasy_pts']/df_all['games'])
    df_all = df_all.sort_values(by=['player', 'season']).reset_index(drop=True)

    df_all['y_act'] = df_all.groupby('player')['fantasy_pts_per_game'].shift(-1)
    df_all['games_next'] = df_all.groupby('player')['games'].shift(-1)

    if pos == 'QB':
        df_all['fantasy_pts_rush_per_game'] = (df_all['sum_fantasy_pts_rush']/df_all['games'])
        df_all['fantasy_pts_pass_per_game'] = (df_all['sum_fantasy_pts_pass']/df_all['games'])

        df_all['y_act_rush'] = df_all.groupby('player')['fantasy_pts_rush_per_game'].shift(-1)
        df_all['y_act_pass'] = df_all.groupby('player')['fantasy_pts_pass_per_game'].shift(-1)

    if pos == 'RB':
        df_all['sum_fantasy_pts_rush'] = df_all.sum_rush_yards_gained_sum*0.1 + df_all.sum_rush_rush_touchdown_sum*7
        df_all['sum_fantasy_pts_rec'] = df_all.sum_rec_yards_gained_sum*0.1 + df_all.sum_rec_touchdown_sum*7 + df_all.sum_rec_complete_pass_sum*0.5
        df_all['fantasy_pts_rush_per_game'] = (df_all['sum_fantasy_pts_rush']/df_all['games'])
        df_all['fantasy_pts_rec_per_game'] = (df_all['sum_fantasy_pts_rec']/df_all['games'])

        df_all['y_act_rush'] = df_all.groupby('player')['fantasy_pts_rush_per_game'].shift(-1)
        df_all['y_act_rec'] = df_all.groupby('player')['fantasy_pts_rec_per_game'].shift(-1)

    # df_all['y_act_plusone'] = df_all.groupby('player')['fantasy_pts_per_game'].shift(-2)
    # df_all['games_next_plusone'] = df_all.groupby('player')['games'].shift(-2)

    df_all['year'] = df_all.season+1
    df_all = df_all.sort_values(by=['year', 'fantasy_pts_per_game'], ascending=[True, False])
    cols = ['player', 'year', 'season', 'games', 'games_next', 'fantasy_pts_per_game']
    cols.extend([c for c in df_all.columns if 'y_act' in c])
    cols.extend([c for c in df_all.columns if c not in cols])
    df_all = df_all[cols]

    df_all = df_all.sort_values(by=['year', 'fantasy_pts_per_game'], ascending=[True, False]).reset_index(drop=True)
    df_all.player = df_all.player.apply(dc.name_clean)
    dm_ff.write_to_db(df_all, 'Season_Stats_New', f'{pos}_Stats', if_exist='replace')

#%%

rec_cols = ['fantasy_pts', 'rec_yards_gained_sum', 'rec_first_down_sum', 'rec_pass_attempt_sum',
            'rec_ep_sum', 'rec_touchdown_sum', 'rec_air_yards_sum', 'rec_third_fourth_cp_sum', 
            'rec_red_zone_ep_sum', 'rec_yards_after_catch_sum', 'rec_xyac_median_yardage_sum',
            'rec_yac_epa_sum']

rush_cols = ['rush_yards_gained_sum', 'rush_first_down_rush_sum', 'rush_rush_attempt_sum', 'rush_ep_sum',
            'rush_red_zone_yards_gained_sum', 'rush_red_zone_rush_attempt_sum', 'rush_rush_touchdown_sum',
            'rush_third_fourth_ep_sum', 'rush_goalline_rush_attempt_sum']

cols = rec_cols + rush_cols

df_all_pos = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:

    df = dm_daily.read(f'''SELECT * 
                           FROM {pos}_Stats
                           WHERE (season >= 2021 AND week <= 17)
                                  OR (season < 2021 AND week <= 16)
                        ''', 'FastR_Beta')

    df = df[~((df.player == 'Adrian Peterson') & (df.team=='CHI'))].reset_index(drop=True)
    df = df[~((df.player == 'Steve Smith') & (df.team.isin(['NYG', 'PHI', 'LAR'])))].reset_index(drop=True)
    df = df[~((df.player == 'Mike Williams') & (df.season < 2017))].reset_index(drop=True)

    df = df[~((df.player=='Trey Mcbride') & (df.season==2023) & (df.week < 8))].reset_index(drop=True)
    df_all_pos = pd.concat([df_all_pos, df], axis=0)

df_all_pos.team = df_all_pos.team.map(team_map)
df_all_pos['year'] = df_all_pos.season+1
df_agg_all = df_all_pos[['team','season', 'year']].drop_duplicates()
for agg_type in ['sum', 'mean']:
    
    agg_stats = {c: agg_type for c in cols}
    df_agg = df_all_pos.groupby(['team', 'season']).agg(agg_stats).reset_index()
    df_agg = df_agg.rename(columns={c: f'team_{agg_type}_{c}' for c in df_agg.columns if c not in ('team', 'season')})
    df_agg_all = pd.merge(df_agg_all, df_agg, on=['team', 'season'])


dm_ff.write_to_db(df_agg_all, 'Season_Stats_New', 'Team_Stats', if_exist='replace')
df_agg_all.groupby('season').agg({'team': 'count'})
# %%
