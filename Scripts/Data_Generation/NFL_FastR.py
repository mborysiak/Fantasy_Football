#%%
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq
import datetime as dt
import numpy as np
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


def get_coaches():
    hc = data[['season', 'week', 'home_team', 'home_coach']].drop_duplicates()
    ac = data[['season', 'week', 'away_team', 'away_coach']].drop_duplicates()
    
    new_cols = ['season', 'week', 'posteam', 'coach']
    hc.columns = new_cols
    ac.columns = new_cols

    return pd.concat([hc, ac], axis=0).reset_index(drop=True)


def window_max(df, w_col, gcols, agg_met, agg_type):
    
    gcols.append(w_col)
    agg_df = df.groupby(gcols, as_index=False).agg({agg_met: agg_type})
    
    gcols.remove(w_col)
    
    max_df = agg_df.groupby(gcols, as_index=False).agg({agg_met: 'max'})

    gcols.append(agg_met)
    return pd.merge(agg_df, max_df, on=gcols) 


def calc_fp(df, cols, pts):
    df['fantasy_pts'] = (df[cols] * pts).sum(axis=1)
    return df


def rolling_stats(df, gcols, rcols, period, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    rolls = df.groupby(['player_name'])[rcols].rolling(3).agg(agg_type).reset_index(drop=True)
    rolls.columns = [f'r{agg_type}{period}_{c}' for c in rolls.columns]

    return rolls


def rolling_expand(df, gcols, rcols, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    # if agg type is in form of percentile (e.g. p80) then use quantile
    if agg_type[0]=='p':

        # pull out perc amount and convert to decimal float to calculate quantile rolling
        perc_amt = float(agg_type[1:])/100
        rolls =  df.groupby(gcols)[rcols].apply(lambda x: x.expanding().quantile(perc_amt))
    # otherwise, use the string argument of aggregation
    else:
        rolls = df.groupby(gcols)[rcols].apply(lambda x: x.expanding().agg(agg_type))
    
    # clean up the rolled dataset indices and column name prior to returning 
    rolls = rolls.reset_index(drop=True)
    rolls.columns = [f'{agg_type}all_{c}' for c in rolls.columns]

    return rolls


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
        'pass_touchdown', 'pass_length', 'pass_location', 'qb_epa', 'qb_hit','sack',
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
data['pos_is_home'] = np.where(data.posteam==data.home_team, 1, 0)

# temperature data doesn't exists for 2020
data = data.drop('temp', axis=1)

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
            'qb_hit', 'sack', 'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
            'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
            'comp_yac_wpa', 'complete_pass', 'incomplete_pass', 'interception',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost',  'xyac_epa',
            'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
            'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
             'grass', 'synthetic', 'indoors', 'outdoors',
             'td_prob', 'qb_epa', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo', 'pos_is_home']

#--------------
# Team Stats
#--------------

gcols = ['season', 'week', 'posteam']

team_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='team')
team_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='team')

team = data[gcols].drop_duplicates()
team = pd.merge(team, team_sum, on=gcols)
team = pd.merge(team, team_mean, on=gcols)
team = team.sort_values(by=['season', 'posteam', 'week'])

#--------------
# Coach Stats
#--------------

gcols = ['season', 'week', 'coach']

coach_labels = get_coaches()
coach_data = pd.merge(data, coach_labels, on=['season', 'week', 'posteam'])

coach_sum = get_agg_stats(coach_data, gcols, sum_cols, 'sum', prefix='coach')
coach_mean = get_agg_stats(coach_data, gcols, mean_cols, 'mean', prefix='coach')

coaches = coach_data[gcols].drop_duplicates()
coaches = pd.merge(coaches, coach_sum, on=gcols)
coaches = pd.merge(coaches, coach_mean, on=gcols)
coaches = coaches.sort_values(by=['season', 'coach', 'week'])

#-------------
# QB Stats
#-------------

# find who the QB was on a given week
data['yards_gained_random'] = data.yards_gained.apply(lambda x: x + np.random.random(1))
w_grp = ['season', 'week', 'posteam']
(w_col, w_met, w_agg) = ('passer_player_name', 'yards_gained_random', 'sum')
qbs = window_max(data[data.sack==0], w_col, w_grp, w_met, w_agg).drop('yards_gained_random', axis=1)


#--------------
# Receiving Stats
#--------------

sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 
            'qb_dropback', 'qb_scramble', 'penalty', 'first_down', 'first_down_pass',
            'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go','pass_middle',
            'pass_outside', 'pass_touchdown',
            'qb_hit',  'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
            'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
            'comp_yac_wpa', 'complete_pass', 'incomplete_pass', 'interception',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost',  'xyac_epa',
            'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
            'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
             'grass', 'synthetic', 'indoors', 'outdoors',
             'td_prob', 'qb_epa', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo', 'pos_is_home']

gcols =  ['week', 'season', 'posteam', 'receiver_player_name']
rec_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='rec')
rec_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='rec')
rec = pd.merge(rec_sum, rec_mean, on=gcols)

fp_cols = ['rec_complete_pass_sum', 'rec_yards_gained_sum',  'rec_pass_touchdown_sum']
rec = calc_fp(rec, fp_cols, [0.5, 0.1, 7])

rec = pd.merge(rec, team, on=['week', 'season', 'posteam'])
rec = rec.sort_values(by=['receiver_player_name', 'season', 'week']).reset_index(drop=True)
rec['y_act'] = rec.groupby('receiver_player_name')['fantasy_pts'].shift(-1)

rec = rec.rename(columns={'receiver_player_name': 'player_name'})

#--------------
# Rushing Stats Stats
#--------------

sum_cols = ['shotgun', 'no_huddle', 'rush_attempt', 'first_down',
            'first_down_rush', 'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go', 'run_middle', 'run_outside',
             'rush_touchdown', 'tackled_for_loss',
            'ep', 'epa', 'touchdown', 'fumble','yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['td_prob', 'wp', 'wpa', 'ep', 'epa', 'yardline_100',
             'yards_gained', 'ydstogo']

gcols =  ['week', 'season', 'posteam', 'rusher_player_name']
rush_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='rush')
rush_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='rush')
rush = pd.merge(rush_sum, rush_mean, on=gcols)

rush = rush.rename(columns={'rusher_player_name': 'player_name'})
rush = pd.merge(rush, rec, on=['player_name', 'posteam', 'week', 'season'], how='left')
rush = rush.fillna(0)

fp_cols = ['rec_complete_pass_sum', 'rec_yards_gained_sum',
           'rush_yards_gained_sum',  'rec_pass_touchdown_sum', 'rush_rush_touchdown_sum']
rush = calc_fp(rush, fp_cols, [0.5, 0.1, 0.1, 7, 7])

rush = rush.sort_values(by=['player_name', 'season', 'week']).reset_index(drop=True)
rush['y_act'] = rush.groupby('player_name')['fantasy_pts'].shift(-1)

#%%

df = rush.copy()

#--------------
# Rolling Stats
#--------------

rcols = [ 'fantasy_pts', 'team_rush_touchdown_sum', 'team_tackled_for_loss_sum',
         'team_pass_touchdown_sum', 'team_qb_hit_sum', 'team_sack_sum',
         'team_qb_epa_sum', 'team_cp_sum', 'team_cpoe_sum', 'team_air_epa_sum',
         'team_air_wpa_sum', 'team_air_yards_sum', 'team_comp_air_epa_sum',
         'team_comp_air_wpa_sum', 'team_comp_yac_epa_sum',
         'team_comp_yac_wpa_sum', 'team_complete_pass_sum',
         'team_incomplete_pass_sum', 'team_interception_sum', 'team_ep_sum',
         'team_epa_sum', 'team_touchdown_sum', 'team_fumble_sum',
         'team_fumble_lost_sum', 'team_xyac_epa_sum'
       ]

rcols.extend([c for c in rec.columns if 'rec_' in c])
rcols.extend([c for c in rec.columns if 'rush_' in c])

# filter to 2010 since we're only interested in 2020
df = df[rec.season >= 2010].reset_index(drop=True)

gcols = ['player_name']

print('Calculating Rolling Stats 3 window')
rolls_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
rolls_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')
rolls_med = rolling_stats(df, gcols, rcols, 3, agg_type='median')

print('Calculating Expanding Stats')
hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p80')
hist_p20 = rolling_expand(df, gcols, rcols, agg_type='p20')

df = pd.concat([df, 
                hist_mean, hist_std, hist_p80, hist_p20, 
                rolls_mean, rolls_max, rolls_med], axis=1)

df = df[df.season==2020].reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

#%%

#=====================
# Regression Baseline Model
#=====================

from ff.db_operations import DataManage
from ff import general as ffgeneral
from skmodel.run_models import SciKitModel

#-----------------
# Format Data for Baseline Projections
#-----------------

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

baseline = dm.read('''SELECT playerName player_name,
                             week,
                             fantasyPoints, 
                             fantasyPointsRank,
                             salary,
                             `Proj Pts` ProjPts,
                             expertConsensus,
                             expertNathanJahnke,
                             expertKevinCole,
                             expertAndrewErickson,
                             expertIanHartitz
                      FROM PFF_Proj_Ranks a
                      JOIN (SELECT Name playerName, *
                            FROM PFF_Expert_Ranks )
                            USING (playerName, week)
                      WHERE a.position='rb' 
                      ''', 'Pre_PlayerData')

baseline = baseline.fillna(baseline.median())

baseline_m = pd.merge(baseline, 
                      df[['week', 'player_name', 'y_act']], 
                      on=['week', 'player_name'])

#-----------------
# Run Baseline Model
#-----------------

skm = SciKitModel(baseline_m)
X_base, y_base = skm.Xy_split(y_metric='y_act', to_drop=['player_name'])
cv_time = skm.cv_time_splits('week', X_base, 2)

model = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('k_best'),
                        skm.piece('ridge')])

params = skm.default_params(model)
params['k_best__k'] = range(1, X_base.shape[1])

best_model, scores, oof_data = skm.time_series_cv(model, X_base, y_base, params, 
                                                  col_split='week', time_split=3)

outliers = (oof_data['actual'] > oof_data['combined']*1.15).astype('int')
outliers = pd.Series(outliers, name='y_act')

#%%

df_m = pd.merge(df, baseline, on=['player_name', 'week'])

skm = SciKitModel(df_m)
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player_name', 'posteam'])
cv_time = skm.cv_time_splits('week', X, 6)

model = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc'),
                        # skm.feature_union([skm.piece('agglomeration'), skm.piece('k_best'), skm.piece('pca')]),
                        # skm.piece('k_best', label_rename='k_best2'),
                        skm.piece('lgbm')])

params = skm.default_params(model)
best_model = skm.random_search(model, X, y, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model, X, y, cv_time)

imp_cols = X.columns[best_model['select_perc'].get_support()]
skm.print_coef(best_model, imp_cols)

#%%







#%%

# this is close but there seems to be some sort of duplicates for getting drive info
# drive_cols = ['season', 'week', 'posteam',
#               'drive', 'drive_ended_with_score', 'drive_first_downs', 'drive_play_count',
#               'drive_inside20', 'drive_yards_penalized', 'drive_time_of_possession']
# drive_agg_cols = ['drive_ended_with_score', 'drive_first_downs', 'drive_play_count',
#                   'drive_inside20', 'drive_yards_penalized', 'drive_time_of_possession']

# get_agg_stats(data[drive_cols].drop_duplicates(), gcols, drive_agg_cols, 'sum', prefix='team')