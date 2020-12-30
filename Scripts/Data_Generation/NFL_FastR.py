#%%
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq
import datetime as dt
pd.set_option('display.max_columns', 999)

#---------------
# Functions
#---------------

def one_hot_col(df, col):
    return pd.concat([df, pd.get_dummies(df[col])], axis=1).drop(col, axis=1)

def get_agg_stats(data, gcols, stat_cols, agg_type, prefix=''):

    agg_stat = {c: agg_type for c in stat_cols}
    df = data.groupby(gcols).agg(agg_stat)
    df.columns = [f'{prefix}_{c}_{agg_type}' for c in df.columns]

    return df.reset_index()


#---------------
# Load Data
#---------------

# set the filepath and name for NFL Fast R data saved from R script
DATA_PATH = 'c:/Users/mborysia/Documents/Github/Fantasy_Football/Data/OtherData/NFL_FastR/'
FNAME = 'raw_data20201224.parquet'

# read in the data, filter to real players, and sort by value
data = pq.read_table(f'{DATA_PATH}/{FNAME}').to_pandas()
data = data.sort_values(by='epa', ascending=False).reset_index(drop=True)

#---------------
# CLean Data
#---------------

cols = ['season', 'week', 'season_type', 'game_id', 'spread_line', 'total_line', 'location', 'roof', 'surface',
        'vegas_wp', 'temp', 'defteam', 'posteam', 'home_team', 'away_team', 'home_coach', 'away_coach',
        'desc', 'play', 'down', 'drive', 'away_score', 'home_score', 'half_seconds_remaining',
        'passer_player_id', 'rusher_player_id', 'receiver_player_id',
        'receiver_player_name', 'rusher_player_name', 'passer_player_name',
        'receiver_player_position',  'rusher_player_position', 'passer_player_position',
        'play_type', 'play_type_nfl', 'shotgun', 'no_huddle', 
        'pass_attempt', 'rush_attempt', 'rush', 'qb_dropback', 'qb_scramble', 'penalty',
        'first_down', 'first_down_pass', 'first_down_rush', 'fourth_down_converted',
        'fourth_down_failed', 'third_down_converted', 'goal_to_go', 'td_prob', 'drive',
        'drive_ended_with_score', 'drive_first_downs', 'drive_play_count', 'drive_inside20',
        'drive_game_clock_start', 'drive_game_clock_end', 'drive_yards_penalized',
        'drive_start_yard_line', 'drive_end_yard_line', 'drive_time_of_possession',
        'run_gap', 'run_location','rush_touchdown', 'tackled_for_loss',
        'pass_touchdown', 'pass_length', 'pass_location', 'qb_epa', 'qb_hit',
        'cp', 'cpoe', 'air_epa', 'air_wpa', 'air_yards', 'comp_air_epa',
        'comp_air_wpa', 'comp_yac_epa', 'comp_yac_wpa','complete_pass', 'incomplete_pass',
        'interception', 'ep', 'epa', 'touchdown',
        'home_wp', 'home_wp_post', 'away_wp', 'away_wp_post',
        'fumble', 'fumble_lost',
        'wp', 'wpa', 'xyac_epa', 'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage',
        'xyac_success', 'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
        'yards_gained', 'ydsnet', 'ydstogo',
        'receiver_player_age', 'receiver_player_college_name', 
        'receiver_player_height', 'receiver_player_weight',
        'rusher_player_age', 'rusher_player_college_name',
        'rusher_player_height', 'rusher_player_weight',
        'passer_player_age', 'passer_player_college_name', 
        'passer_player_height', 'passer_player_weight']

data = data.loc[data.season_type=='REG', cols]

data.loc[data.run_location.isin(['left', 'right']), 'run_location'] = 'run_outside'
data.loc[data.run_location=='middle', 'run_location'] = 'run_middle'

data.loc[data.pass_location.isin(['left', 'right']), 'pass_location'] = 'pass_outside'
data.loc[data.pass_location=='middle', 'pass_location'] = 'pass_middle'

data.loc[(data.surface != 'grass') & ~(data.surface.isnull()), 'surface'] = 'synthetic'
data.loc[data.roof.isin(['outdoors', 'open']), 'roof'] = 'outdoors'
data.loc[data.roof.isin(['dome', 'closed']), 'roof'] = 'indoors'

for c in ['run_location', 'pass_location', 'surface', 'roof']:
    data = one_hot_col(data, c)


def time_convert(t):
    return int(t.split(':')[0]) + float(t.split(':')[1])/ 60

data.drive_time_of_possession = data.drive_time_of_possession.fillna('0:0').apply(time_convert)
data.drive_yards_penalized = data.drive_yards_penalized.fillna(0)

data.loc[data.posteam==data.home_team, 'spread_line'] = -data.loc[data.posteam==data.home_team, 'spread_line']

#%%

#===================
# Team, Coach, and QB Stats
#===================

#--------------
# Aggregate columns
#--------------



sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 'rush_attempt',
            'qb_dropback', 'qb_scramble', 'penalty', 'first_down', 'first_down_pass',
            'first_down_rush', 'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go', 'run_middle', 'run_outside', 'pass_middle',
            'pass_outside', 'rush_touchdown', 'tackled_for_loss', 'pass_touchdown',
            'qb_hit', 'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
            'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
            'comp_yac_wpa', 'complete_pass', 'incomplete_pass', 'interception',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost',  'xyac_epa',
            'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
            'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['spread_line', 'total_line', 'vegas_wp', 'temp', 'td_prob', 'qb_epa',
             'grass', 'synthetic', 'indoors', 'outdoors', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo']

gcols = ['season', 'week', 'posteam']

team_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='team')
team_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='team')

team = data[gcols].drop_duplicates()
team = pd.merge(team, team_sum, on=gcols)
team = pd.merge(team, team_mean, on=gcols)
team = team.sort_values(by=['season', 'posteam', 'week'])
# this is close but there seems to be some sort of duplicates

# drive_cols = ['season', 'week', 'posteam',
#               'drive', 'drive_ended_with_score', 'drive_first_downs', 'drive_play_count',
#               'drive_inside20', 'drive_yards_penalized', 'drive_time_of_possession']
# drive_agg_cols = ['drive_ended_with_score', 'drive_first_downs', 'drive_play_count',
#                   'drive_inside20', 'drive_yards_penalized', 'drive_time_of_possession']

# get_agg_stats(data[drive_cols].drop_duplicates(), gcols, drive_agg_cols, 'sum', prefix='team')             

#%%

from sklearn.preprocessing import StandardScaler

sum_cols = ['qb_epa', 'complete_pass', 'yards_gained',  'pass_touchdown',  'air_epa', 'air_yards', 
            'air_wpa',  'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa', 'comp_yac_wpa']
sum_stats = {c: 'sum' for c in sum_cols}

mean_cols = ['receiver_player_age', 'qb_epa',  'air_epa', 'air_wpa', 'air_yards', 'comp_air_epa', 
             'comp_air_wpa', 'comp_yac_epa',  'comp_yac_wpa', 'cp', 'cpoe']
mean_stats = {c: 'mean' for c in mean_cols}

count_stats = {'pass_attempt': 'count'}

gcols = ['season', 'season_type', 'receiver_player_id', 'receiver_player_name']


# met['receiver_player_age_mean'] = StandardScaler().fit_transform(met[['receiver_player_age_mean']])
# met['receiver_player_age_mean'] = met['receiver_player_age_mean'] + abs(met['receiver_player_age_mean'].min()) + 1

met = met[met.pass_attempt_count>50].reset_index()

met['qb_epa_per_att'] = met['qb_epa_sum'] / met['pass_attempt_count']
met['fantasy_pts'] = (met[['complete_pass_sum', 'yards_gained_sum', 'pass_touchdown_sum']] * [0.5, 0.1, 7]).sum(axis=1)
met['y_act'] = met.groupby('receiver_player_name')['fantasy_pts'].shift(-1)
met = met.dropna()

for c in ['qb_epa_per_att', 'fantasy_pts', 'air_yards_sum', 'pass_attempt_count', 'air_epa_sum', 'comp_air_epa_sum', 'air_epa_sum']:
    met[f'{c}_age'] = met[c] / met['receiver_player_age_mean']

met = met.sort_values(by=['receiver_player_name', 'season']).reset_index(drop=True)
met = round(met, 2)
met.iloc[:30]

