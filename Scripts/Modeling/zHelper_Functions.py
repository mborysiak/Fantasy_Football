import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning


# ==========
# Dictionary for position relevant metrics
# ==========

# initialize full position dictionary
pos = {}

# ---------
# RB dictionary
# ---------

# initilize RB dictionary
pos['RB'] = {}

# total touch filter name
pos['RB']['touch_filter'] = 'total_touches'

# metrics to be predicted for fantasy point generation
pos['RB']['metrics'] = ['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['RB']['med_features'] = ['fp', 'tgt', 'receptions', 'total_touches', 'rush_yds', 'rec_yds', 
                           'rush_yd_per_game', 'rec_yd_per_game', 'rush_td', 'games_started', 
                           'qb_rating', 'qb_yds', 'pass_dvoa', 'tm_rush_td', 'tm_rush_yds', 
                           'tm_rush_att', 'adjust_line_yds', 'ms_rush_yd', 'ms_rec_yd', 'ms_rush_td',
                           'avg_pick', 'fp_per_touch', 'team_rush_avg_att',
                            'rz_20_rush_att', 'rz_20_rush_yds', 'rz_20_rush_td', 'rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                            'rz_10_rush_att', 'rz_10_rush_yds', 'rz_10_rush_td', 'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',
                            'rz_5_rush_att', 'rz_5_rush_yds', 'rz_5_rush_td',
                            'rec_yd_per_game_exp', 'rec_yd_per_game_exp_diff',
                           'rec_yd_per_game_exp_div', 'rec_per_game_exp', 'rec_per_game_exp_diff',
                           'rec_per_game_exp_div', 'td_per_game_exp', 'td_per_game_exp_diff',
                           'td_per_game_exp_div', 'rz_20_rush_att_exp', 'rz_20_rush_att_exp_diff',
                           'rz_20_rush_att_exp_div', 'rz_5_rush_att_exp', 'rz_5_rush_att_exp_diff',
                           'rz_5_rush_att_exp_div', 'rz_20_rush_pct_exp',
                           'rz_20_rush_pct_exp_diff', 'rz_20_rush_pct_exp_div',
                           'rz_5_rush_pct_exp', 'rz_5_rush_pct_exp_diff', 'rz_5_rush_pct_exp_div',
                           'ms_rush_att_exp', 'ms_rush_att_exp_diff', 'ms_rush_att_exp_div',
                           'ms_rush_yd_exp', 'ms_rush_yd_exp_diff', 'ms_rush_yd_exp_div',
                           'ms_rush_td_exp', 'ms_rush_td_exp_diff', 'ms_rush_td_exp_div',
                           'ms_rec_yd_exp', 'ms_rec_yd_exp_diff', 'ms_rec_yd_exp_div',
                           'ms_tgts_exp', 'ms_tgts_exp_diff', 'ms_tgts_exp_div',
                           'rush_rec_ratio_exp', 'rush_rec_ratio_exp_diff',
                           'rush_rec_ratio_exp_div', 'rz_20_tgt_exp', 'rz_20_tgt_exp_diff',
                           'rz_20_tgt_exp_div', 'rz_20_receptions_exp',
                           'rz_20_receptions_exp_diff', 'rz_20_receptions_exp_div', 'avg_pick_exp',
                           'avg_pick_exp_diff', 'avg_pick_exp_div', 'teammate_diff_min_exp',
                           'teammate_diff_min_exp_diff', 'teammate_diff_min_exp_div',
                           'teammate_diff_avg_exp', 'teammate_diff_avg_exp_diff',
                           'teammate_diff_avg_exp_div']

# sum feature categories
pos['RB']['sum_features'] = ['total_touches', 'att', 'total_yds', 'rush_td', 'fp', 'rec_yds', 
                             'rush_yds', 'qb_yds', 'avg_pick']

# max feature categories
pos['RB']['max_features'] = ['fp', 'rush_td', 'tgt', 'rush_yds', 'rec_yds', 'total_yds', 
                             'rush_yd_per_game', 'rec_yd_per_game', 'ms_rush_yd']

# age feature categories
pos['RB']['age_features'] = ['fp', 'rush_yd_per_game', 'rec_yd_per_game', 'total_touches', 'receptions', 'tgt',
                             'ms_rush_yd', 'ms_rec_yd', 'available_rush_att', 'available_tgt', 'total_touches_sum',
                             'total_yds_sum', 'avg_pick', 'fp_per_touch', 'ms_rush_yd_per_att', 'ms_tgts',
                            'rz_20_rush_att', 'rz_20_rush_yds', 'rz_20_rush_td', 'rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                            'rz_10_rush_att', 'rz_10_rush_yds', 'rz_10_rush_td', 'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',
                            'rz_5_rush_att', 'rz_5_rush_yds', 'rz_5_rush_td',
                            'rec_yd_per_game_exp', 'rec_yd_per_game_exp_diff',
                           'rec_yd_per_game_exp_div', 'rec_per_game_exp', 'rec_per_game_exp_diff',
                           'rec_per_game_exp_div', 'td_per_game_exp', 'td_per_game_exp_diff',
                           'td_per_game_exp_div', 'rz_20_rush_att_exp', 'rz_20_rush_att_exp_diff',
                           'rz_20_rush_att_exp_div', 'rz_5_rush_att_exp', 'rz_5_rush_att_exp_diff',
                           'rz_5_rush_att_exp_div', 'rz_20_rush_pct_exp',
                           'rz_20_rush_pct_exp_diff', 'rz_20_rush_pct_exp_div',
                           'rz_5_rush_pct_exp', 'rz_5_rush_pct_exp_diff', 'rz_5_rush_pct_exp_div',
                           'ms_rush_att_exp', 'ms_rush_att_exp_diff', 'ms_rush_att_exp_div',
                           'ms_rush_yd_exp', 'ms_rush_yd_exp_diff', 'ms_rush_yd_exp_div',
                           'ms_rush_td_exp', 'ms_rush_td_exp_diff', 'ms_rush_td_exp_div',
                           'ms_rec_yd_exp', 'ms_rec_yd_exp_diff', 'ms_rec_yd_exp_div',
                           'ms_tgts_exp', 'ms_tgts_exp_diff', 'ms_tgts_exp_div',
                           'rush_rec_ratio_exp', 'rush_rec_ratio_exp_diff',
                           'rush_rec_ratio_exp_div', 'rz_20_tgt_exp', 'rz_20_tgt_exp_diff',
                           'rz_20_tgt_exp_div', 'rz_20_receptions_exp',
                           'rz_20_receptions_exp_diff', 'rz_20_receptions_exp_div', 'avg_pick_exp',
                           'avg_pick_exp_diff', 'avg_pick_exp_div', 'teammate_diff_min_exp',
                           'teammate_diff_min_exp_diff', 'teammate_diff_min_exp_div',
                           'teammate_diff_avg_exp', 'teammate_diff_avg_exp_diff',
                           'teammate_diff_avg_exp_div']
#---------
# WR dictionary
#---------

# initilize RB dictionary
pos['WR'] = {}

# total touch filter name
pos['WR']['touch_filter'] = 'tgt'

# metrics to calculate stats for
pos['WR']['metrics'] = ['rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['WR']['med_features'] = ['fp', 'tgt', 'receptions', 'rec_yds', 'rec_yd_per_game', 'rec_td', 'games_started', 
                             'qb_rating', 'qb_yds', 'pass_dvoa', 'ms_tgts', 'ms_rec_yd', 
                             'tm_net_pass_yds', 'avg_pick',  'rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                             'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',
                            'rec_yd_per_game_exp_diff', 'rec_per_game_exp_diff', 'td_per_game_exp_diff', 
                             'rz_20_tgt_exp_diff', 'rz_20_receptions_exp_diff'
                            ]

# sum feature categories
pos['WR']['sum_features'] = ['receptions', 'rec_yds', 'tgt']

# max feature categories
pos['WR']['max_features'] = ['fp', 'rec_td', 'tgt', 'ms_tgts', 'ms_rec_yd', 'rec_yd_per_game',
                             'rz_20_tgt', 'rz_20_receptions', 
                             'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',]

# age feature categories
pos['WR']['age_features'] = ['fp', 'rec_yd_per_game', 'receptions', 'tgt', 'ms_tgts', 'ms_rec_yd', 
                             'avg_pick', 'ms_yds_per_tgts', 'rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                             'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',
                            'rec_yd_per_game_exp_diff', 'rec_per_game_exp_diff', 'td_per_game_exp_diff',
                            'rec_yd_per_game_exp_div', 'rec_per_game_exp_div', 'td_per_game_exp_div',
                             'rz_20_tgt_exp_diff', 'rz_20_receptions_exp_diff'
                            ]


# ---------
# QB dictionary
# ---------

# initilize RB dictionary
pos['QB'] = {}

# total touch filter name
pos['QB']['touch_filter'] = 'qb_att'

# metrics to calculate stats for
pos['QB']['metrics'] = ['qb_yd_per_game', 'pass_td_per_game','rush_yd_per_game', 
                        'rush_td_per_game' ,'int_per_game', 'sacks_per_game' ]

pos['QB']['med_features'] = ['fp', 'qb_tds','qb_rating', 'qb_yds', 'pass_dvoa', 'qb_complete_pct', 'qb_td_pct', 
                             'sack_pct', 'avg_pick', 'sacks_allowed', 'qbr', 'adj_yd_per_att', 'adj_net_yd_per_att',
                             'int', 'int_pct', 'rush_att', 'rush_yds', 'rush_td', 'rush_yd_per_game', 'rush_yd_per_att',
                             'rz_20_pass_complete', 'rz_20_pass_att',
                               'rz_20_complete_pct', 'rz_20_pass_yds', 'rz_20_pass_td', 'rz_20_int',
                               'rz_10_pass_complete', 'rz_10_pass_att', 'rz_10_complete_pct',
                               'rz_10_pass_yds', 'rz_10_pass_td', 'rz_10_int', 'rz_20_rush_att',
                               'rz_20_rush_yds', 'rz_20_rush_td', 'rz_20_rush_pct', 'rz_10_rush_att',
                               'rz_10_rush_yds', 'rz_10_rush_td', 'rz_10_rush_pct', 'rz_5_rush_att',
                               'rz_5_rush_yds', 'rz_5_rush_td', 'rz_5_rush_pct']
pos['QB']['max_features'] = ['fp', 'qb_rating', 'qb_yds', 'qb_tds', 'int', 'int_pct', 'sack_pct', 'rush_yd_per_att']
pos['QB']['age_features'] = ['fp', 'qb_rating', 'qb_yds', 'qb_complete_pct', 'qb_td_pct', 'sack_pct', 'rush_yd_per_game', 
                             'avg_pick', 'qbr', 'int', 'int_pct', 'rush_att', 'rush_yds', 'rush_td', 'rush_yd_per_att']
pos['QB']['sum_features'] = ['qb_tds', 'qb_yds', 'fourth_qt_comeback', 'game_winning_drives', 'fp']

# ---------
# WR dictionary
# ---------

# initilize RB dictionary
pos['TE'] = {}

# total touch filter name
pos['TE']['touch_filter'] = 'tgt'

# metrics to calculate stats for
pos['TE']['metrics'] = ['rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['TE']['med_features'] = ['fp', 'tgt', 'receptions', 'rec_yds', 'rec_yd_per_game', 'rec_td', 'games_started', 
                             'qb_rating', 'qb_yds', 'pass_dvoa', 'ms_tgts', 'ms_rec_yd', 
                             'tm_net_pass_yds', 'avg_pick','rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                             'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',
                            'rec_yd_per_game_exp_diff', 'rec_per_game_exp_diff', 'td_per_game_exp_diff', 
                             'rz_20_tgt_exp_diff', 'rz_20_receptions_exp_diff']
# sum feature categories
pos['TE']['sum_features'] = ['receptions', 'rec_yds', 'tgt', 'rec_td', 'qb_yds']

# max feature categories
pos['TE']['max_features'] = ['fp', 'rec_td', 'tgt', 'ms_tgts', 'rec_yds', 'ms_rec_yd', 'rec_yd_per_game',
                             'rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                             'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',]

# age feature categories
pos['TE']['age_features'] = ['fp', 'rec_yd_per_game', 'receptions', 'tgt', 'ms_tgts', 'ms_rec_yd', 
                             'avg_pick', 'ms_yds_per_tgts','rz_20_tgt', 'rz_20_receptions', 
                            'rz_20_catch_pct', 'rz_20_rec_yds', 'rz_20_rec_tds',
                             'rz_10_tgt', 'rz_10_receptions', 
                            'rz_10_catch_pct', 'rz_10_rec_yds', 'rz_10_rec_tds',
                            'rec_yd_per_game_exp_diff', 'rec_per_game_exp_diff', 'td_per_game_exp_diff', 
                             'rz_20_tgt_exp_diff', 'rz_20_receptions_exp_diff']


# initilize RB dictionary
pos['Rookie_RB'] = {}

# total touch filter name
pos['Rookie_RB']['touch_filter'] = 'rush_yd_per_game'

# metrics to be predicted for fantasy point generation
pos['Rookie_RB']['metrics'] = ['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['Rookie_RB']['med_features'] = []
# sum feature categories
pos['Rookie_RB']['sum_features'] = []

# max feature categories
pos['Rookie_RB']['max_features'] = []

# age feature categories
pos['Rookie_RB']['age_features'] = []


# initilize RB dictionary
pos['Rookie_WR'] = {}

# total touch filter name
pos['Rookie_WR']['touch_filter'] = 'rec_yd_per_game'

# metrics to be predicted for fantasy point generation
pos['Rookie_WR']['metrics'] = ['rec_yd_per_game', 'rec_per_game', 'td_per_game']

# median feature categories
pos['Rookie_WR']['med_features'] = []
# sum Rookie_WR categories
pos['Rookie_WR']['sum_features'] = []

# max feature categories
pos['Rookie_WR']['max_features'] = []

# age feature categories
pos['Rookie_WR']['age_features'] = []



# ==========
# Data Preparation Functions
# ==========


def touch_game_filter(df, pos, set_pos, set_year):
    '''
    Apply filters to the touches and games required to be in the dataset
    '''
    # split old and new to filter past years based on touches.
    # leave all new players in to ensure everyone gets a prediction
    old = df[((df[pos[set_pos]['touch_filter']] > pos[set_pos]['req_touch']) | \
             (df.games > pos[set_pos]['req_games'])) & \
             (df.year < set_year)].reset_index(drop=True)
    this_year = df[df.year>=set_year]

    # merge old and new back together after filtering
    df = pd.concat([old, this_year], axis=0)

    # create dataframes to store results
    df_train_results = pd.DataFrame([old.player, old.year]).T
    df_test_results = pd.DataFrame([this_year.player, this_year.year]).T
    
    return df, df_train_results, df_test_results


def add_exp_metrics(df, set_pos):
    '''
    Function to add how a players stats compare to the number of years that they have
    been in the league (years experience)
    '''
    if set_pos in ['WR', 'TE']:
        if pos[set_pos]['use_ay']:
            cols = ['rec_yd_per_game', 'rec_per_game', 'td_per_game',
                    'rz_20_tgt', 'rz_20_receptions', 'avg_pick', 'min_teammate']
            cols.extend(['ay_per_game', 'yac_per_game', 'racr', 'ay_per_tar', 'ay_per_rec',
               'ay_converted', 'yac_per_ay', 'air_yd_mkt_share', 'wopr', 'rec_yds_per_ay',
               'yac_plus_ay', 'team_yac', 'tm_air_per_att', 'tm_ay_converted', 'tm_rec_yds_per_ay',
               'tm_yac_per_ay', 'yac_mkt_share', 'yac_wopr', 'total_tgt_mkt_share'])
        else:
            cols = ['rec_yd_per_game', 'rec_per_game', 'td_per_game', 
                    'rz_20_tgt', 'rz_20_receptions', 'avg_pick', 'min_teammate']
           
    elif set_pos == 'QB':
        cols = ['fp', 'qb_tds','qb_rating', 'qb_yds', 'pass_dvoa', 'qb_complete_pct', 'qb_td_pct', 
                 'sack_pct', 'avg_pick', 'sacks_allowed', 'qbr', 'adj_yd_per_att', 'adj_net_yd_per_att',
                 'int', 'int_pct', 'rush_att', 'rush_yds', 'rush_yd_per_game', 'rush_td', 'rush_yd_per_att']
    
    elif set_pos == 'RB':
        cols = ['rush_yd_per_game','rec_yd_per_game', 'rec_per_game', 'td_per_game', 'rz_20_rush_att', 
                'rz_5_rush_att', 'rz_20_rush_pct', 'rz_5_rush_pct','ms_rush_att',
                'ms_rush_yd', 'ms_rush_td', 'ms_rec_yd', 'ms_tgts', 'rush_rec_ratio',
                'rz_20_tgt', 'rz_20_receptions', 'avg_pick', 'teammate_diff_min', 'teammate_diff_avg']
    
    for lab in cols:
        d = df.groupby('year_exp').agg('mean')[lab].reset_index()
        d = d.rename(columns={lab: lab + '_exp'})
        df = pd.merge(df, d, how='inner', left_on='year_exp', right_on='year_exp')
        df[lab + '_exp_diff'] = df[lab] - df[lab + '_exp']
        df[lab + '_exp_div'] = df[lab] / df[lab + '_exp']
        
    return df


def calculate_fp(df, pts, pos):
    
    # calculate fantasy points for QB's associated with a given RB or WR
    if pos == 'RB' or 'WR' or 'Rookie_RB' or 'Rookie_WR':
        df['qb_fp'] = \
        pts['QB'][0]*df['qb_yds'] + \
        pts['QB'][1]*df['qb_tds'] + \
        pts['QB'][4]*df['int'] + \
        pts['QB'][5]*df['qb_sacks']
    
    # calculate fantasy points for RB's
    if pos == 'RB':
        
        df['fp'] = \
        pts['RB'][0]*df['rush_yds'] + \
        pts['RB'][1]*df['rec_yds'] + \
        pts['RB'][3]*df['rush_td'] + \
        pts['RB'][4]*df['rec_td'] + \
        pts['RB'][2]*df['receptions']
        
        # calculate fantasy points per touch
        df['fp_per_touch'] = df['fp'] / df['total_touches']
        
        # calculate fantasy points per target
        df['fp_per_tgt'] = df['fp'] / df['tgt']
    
    if pos == 'WR':
        
        df['fp'] = \
        pts['WR'][0]*df['rec_yds'] + \
        pts['WR'][2]*df['rec_td'] + \
        pts['WR'][1]*df['receptions']
        
        # calculate fantasy points per touch
        df['fp_per_tgt'] = df['fp'] / df['tgt']
        
    if pos == 'TE':
        
        df['fp'] = \
        pts['WR'][0]*df['rec_yds'] + \
        pts['WR'][2]*df['rec_td'] + \
        pts['WR'][1]*df['receptions']
        
        # calculate fantasy points per touch
        df['fp_per_tgt'] = df['fp'] / df['tgt']
        
    if pos == 'QB':
        
        df['fp'] = \
        pts['QB'][0]*df['qb_yds'] + \
        pts['QB'][1]*df['qb_tds'] + \
        pts['QB'][2]*df['rush_yds'] + \
        pts['QB'][3]*df['rush_td'] + \
        pts['QB'][4]*df['int'] + \
        pts['QB'][5]*df['qb_sacks']
        
    if pos not in ['Rookie_RB', 'Rookie_WR']:
        # calculate fantasy points per game
        df['fp_per_game'] = df['fp'] / df['games']
    
    return df


def get_train_predict(df, target, pos, set_pos, year_split, early_year, vers):

    if vers == 'v1':
        # create training and prediction dataframes
        df_train, df_predict = features_target(df, set_pos,
                                               early_year, year_split,
                                               pos[set_pos]['med_features'],
                                               pos[set_pos]['sum_features'],
                                               pos[set_pos]['max_features'],
                                               pos[set_pos]['age_features'],
                                               target_feature=target)
    elif vers == 'v2':
        # create training and prediction dataframes
        df_train, df_predict = features_target_v2(df, set_pos,
                                               early_year, year_split,
                                               pos[set_pos]['med_features'],
                                               pos[set_pos]['sum_features'],
                                               pos[set_pos]['max_features'],
                                               pos[set_pos]['age_features'],
                                               target_feature=target)
    else:
        print('Specify v1 or v2 for features_target')

    df_train = convert_to_float(df_train)
    df_predict = convert_to_float(df_predict)

    # drop any rows that have a null target value (likely due to injuries or other missed season)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)
    df_train = df_train.fillna(df_train.mean())
#     df_predict = df_predict.dropna().reset_index(drop=True)
    
    return df_train, df_predict


def features_target(df, set_pos, year_start, year_split, median_features, sum_features, max_features, 
                    age_features, target_feature):
    
    import pandas as pd

    new_df = pd.DataFrame()
    years = range(year_start+1, int(df.year.max()+1))

    for year in years:
        
        # adding the median features
        past = df[df['year'] <= year]
        past = add_exp_metrics(past, set_pos)
        past = past.join(past.groupby('player')[median_features].median(),on='player', rsuffix='_median')
        past = past.join(past.groupby('player')[max_features].max(),on='player', rsuffix='_max')
        past = past.join(past.groupby('player')[sum_features].sum(),on='player', rsuffix='_sum')
            
        # adding the age features
        suffix = '/ age'
        for feature in age_features:
            feature_label = ' '.join([feature, suffix])
            past[feature_label] = past[feature] / np.exp(past.age)

        suffix = '_inverse_age'
        for feature in age_features:
            feature_label = ''.join([feature, suffix])
            if 'avg_pick' in feature_label:
                past[feature_label] = (past[feature].max() - past[feature]) * (np.max(np.exp(past.age))-past.age)
            else:
                past[feature_label] = past[feature] * (np.max(np.exp(past.age))-past.age)

        # adding the values for target feature
        year_n = past[past["year"] == year]
        year_n_plus_one = df[df['year'] == year+1][['player', target_feature]].rename(columns={target_feature: 'y_act'})
        year_n = pd.merge(year_n, year_n_plus_one, how='left', left_on='player', right_on='player')
        new_df = new_df.append(year_n)
    
    # creating dataframes to export
    new_df = new_df.sort_values(by=['year', 'fp'], ascending=[False, False])
  #  new_df = pd.concat([new_df, pd.get_dummies(new_df.year)], axis=1, sort=False)
    
    df_train = new_df[new_df.year < year_split].reset_index(drop=True)
    df_predict = new_df[new_df.year >= year_split].reset_index(drop=True)
    df_train = df_train.sort_values(['year', 'fp_per_game'], ascending=True).reset_index(drop=True)
    
    return df_train, df_predict


def features_target_v2(df, set_pos, year_start, year_split, median_features, sum_features, max_features, 
                       age_features, target_feature):
    
    import pandas as pd

    new_df = pd.DataFrame()
    years = range(year_start+1, int(df.year.max()+1))

    for year in years:

        # adding the median features
        past = df[df['year'] <= year]
        past = add_exp_metrics(past, set_pos)

        # join in the median, max, and sum features
        past = past.join(past.groupby('player')[median_features].median(), on='player', rsuffix='_median')
        past = past.join(past.groupby('player')[max_features].max(),on='player', rsuffix='_max')
        past = past.join(past.groupby('player')[sum_features].sum(),on='player', rsuffix='_sum')

        for mf in median_features:
            baseline = 0.01*past[mf].mean()
            past[mf + '_over_median'] = (past[mf]-past[mf + '_median']) / (past[mf + '_median'] + baseline)

        # adding the age features
        suffix = '_times_age'
        for feature in age_features:
            feature_label = ''.join([feature, suffix])
            past[feature_label] = past[feature] * (np.exp(past['age']) - np.min(np.exp(past.age)))

        suffix = '_inverse_age'
        for feature in age_features:
            feature_label = ''.join([feature, suffix])
            if 'avg_pick' in feature_label:
                past[feature_label] = (past[feature].max() - past[feature]) * (np.max(np.exp(past.age))-np.exp(past.age))
            else:
                past[feature_label] = past[feature] * (np.max(np.exp(past.age))-np.exp(past.age))

        # adding the values for target feature
        year_n = past[past["year"] == year]
        year_n_plus_one = df[df['year'] == year+1][['player', target_feature]].rename(columns={target_feature: 'y_act'})
        year_n = pd.merge(year_n, year_n_plus_one, how='left', left_on='player', right_on='player')
        new_df = new_df.append(year_n)

    # creating dataframes to export
    new_df = new_df.sort_values(by=['player', 'year']).reset_index(drop=True)
    rolling_stats = new_df.groupby('player')[median_features].rolling(3).mean().fillna(-1).reset_index(drop=True)
    rolling_stats.columns = [c + '_rolling' for c in rolling_stats.columns]
    new_df = pd.concat([new_df, rolling_stats], axis=1)

    df_train = new_df[new_df.year < year_split].reset_index(drop=True)
    df_predict = new_df[new_df.year >= year_split].reset_index(drop=True)
    df_train = df_train.sort_values(['year', 'fp_per_game'], ascending=True).reset_index(drop=True)
    
    return df_train, df_predict


# ===========
# Saving out Data to DB
# ===========

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


# ================
# Regression Functions
# ================

def corr_collinear_removal(df_train, corr_cutoff, collinear_cutoff, good_cols_ext=['player', 'avg_pick', 'year']):
    '''
    Function that removes low correlation features, followed by looping through
    the highest correlations in order and removing any features that are above
    a minimum collinear cutoff.
    '''
    
    # get the initial number of features to show the end difference
    init_features = df_train.shape[1]

    # find out the correlation with the target
    df_corr = df_train.corr()
    corr = df_corr['y_act']
    
    # pull out the features with correlation above the cutoff and sort by absolute correlation
    corr_sort = list(abs(corr[abs(corr) > corr_cutoff]).sort_values(ascending=False).index)
    
    # create lists to store good and bad features
    good_cols = ['y_act']
    bad_cols = []
    
    # loop through each sorted feature (skipping y_act, which is index 0)
    for feature in corr_sort[1:]:
        
        # if the feature has already been removed, move to next feature
        if feature in bad_cols:
            continue
            
        else:
            # if the feature hasn't been removed, append it to the good features list
            good_cols.append(feature)
            
            # find all features that are above the collinear cutoff for the current feature
            collinear = df_corr[feature]
            collinear_cols = list(collinear[abs(collinear) > collinear_cutoff].index)
            
            # add these extra features to the bad_cols list
            bad_cols.extend(collinear_cols)

    # add player, ADP, and year back to the features list and subset df_train
    good_cols.extend(good_cols_ext)
    df_train = df_train[good_cols]
    df_train = df_train.loc[:,~df_train.columns.duplicated()]
    
    return df_train


def X_y_split(train, val, scale=True, pca=False, n_components=0.5):
    '''
    input: train and validation or test datasets
    output: datasets split into X features and y response for train / validation or test
    '''
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X_train = train.select_dtypes(exclude=['object']).drop('y_act', axis=1)
    y_train = train.y_act
    
    try:    
        X_val = val.select_dtypes(exclude=['object']).drop('y_act', axis=1).copy()
        y_val = val.y_act
    except:
        X_val = val.select_dtypes(exclude=['object']).copy()
        y_val = None
    
    if scale == True:
        scale = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scale.transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(scale.transform(X_val), columns=X_val.columns)
    else:
        pass
    
    if pca == True:
        y_corr = pd.concat([X_train, y_train], axis=1).corr()['y_act']
        y_index = y_corr[y_corr < 0].index
        X_train.loc[:, y_index] = -1. * X_train.loc[:, y_index] 
        X_val.loc[:, y_index] = -1. * X_val.loc[:, y_index] 

        n_comp = np.max([2, int(X_train.shape[1] * n_components)])
        n_comp = np.min([n_comp, X_train.shape[0]])
        pca = PCA(n_components=n_comp, random_state=1234, svd_solver='randomized')
        pca.fit(X_train)

        X_train = pd.DataFrame(pca.transform(X_train), columns=['pca' + str(i) for i in range(n_comp)])
        X_val = pd.DataFrame(pca.transform(X_val), columns=['pca' + str(i) for i in range(n_comp)])
    else:
        pass
        
    return X_train, X_val, y_train, y_val


def error_compare(pred, adp, y_act):
    
    from sklearn.metrics import mean_squared_error, r2_score

    r_sq_pred = round(r2_score(y_act, pred), 3)
    r_sq_adp = round(r2_score(y_act, adp), 3)

    rmse_pred = round(np.sqrt(mean_squared_error(pred, y_act)), 3)
    rmse_adp = round(np.sqrt(mean_squared_error(adp, y_act)), 3)

    return [r_sq_pred, r_sq_adp, rmse_pred, rmse_adp]


def validation(estimator, df_train_orig, df_predict, corr_cutoff, collinear_cutoff, 
               skip_years=2, scale=False, pca=False, n_components=0.5):

    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    
    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------
    
    # return the df_train with only relevant features remaining
    df_train = corr_collinear_removal(df_train_orig, corr_cutoff, collinear_cutoff)
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    val_predictions_adp = np.array([])
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]

    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        # fit training data and creating prediction based on validation data
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)
        
        # fit and predict ADP on the dataset
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale=False, pca=False)

        lr.fit(X_train.avg_pick.values.reshape(-1,1), y_train)
        val_predict_adp = lr.predict(X_val.avg_pick.values.reshape(-1,1))

        # skip over the first N years of predictions due to high error for xgb / lgbm
        val_predictions = np.append(val_predictions, val_predict, axis=0)
        val_predictions_adp = np.append(val_predictions_adp, val_predict_adp, axis=0)
            
    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    # store the current predictions in the results tracker
    val_df = df_train.loc[df_train.year.isin(years), 
                          ['player', 'year', 'avg_pick', 'y_act']].reset_index(drop=True)
    val_pred = pd.concat([val_df, pd.Series(val_predictions, name='pred')], axis=1)
    val_pred = pd.concat([val_pred, pd.Series(val_predictions_adp, name='pred_adp')], axis=1)

    result = error_compare(val_pred)
    
    #==========
    # Create Predictions for Current Year
    #==========
    
    pred_cols = list(df_train.columns)
    pred_cols.remove('y_act')
    X_train, X_val, y_train, _ = X_y_split(df_train, df_predict[pred_cols], scale, pca, n_components)
    output_cols = list(X_train.columns)

    # fit training data and creating prediction based on validation data
    estimator.fit(X_train, y_train)
    ty_pred = pd.Series(estimator.predict(X_val), name='pred')
    ty_pred = pd.concat([df_predict[['player', 'year', 'avg_pick']], ty_pred], axis=1)

    return result, val_pred, ty_pred, estimator, output_cols


def create_summary(results, sort_col, keep_first=False):
    
    # create summary dataframe
    summary = pd.DataFrame(results['summary']).T.reset_index()
    scol = ['Iteration', 'Model', 'PredR2', 'AvgPickR2', 'PredRMSE', 'AvgPickRMSE']
    summary.columns = scol 
    
    # drop any duplicate model entries
    scol.remove('Iteration')
    summary = summary.drop_duplicates(subset=scol)

    # sort results by a given metric
    if sort_col == 'PredR2':
        summary = summary.sort_values(by=[sort_col], ascending=[False]).reset_index(drop=True)
    else:
        summary = summary.sort_values(by=[sort_col], ascending=[True]).reset_index(drop=True)

    # if wanting to get the best model for each unique model type
    if keep_first:
        summary = summary.drop_duplicates(subset=['Model'], keep='first')

    return summary


def test_ensemble(summary, results, n, iters, agg_type):
    '''
    Function that accepts multiple model results and averages them together to create an ensemble prediction.
    It then calculates metrics to validate that the ensemble is more accurate than individual models.
    '''

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # initialize Series to store results
    ensemble = pd.Series(dtype='float')
    ty_ensemble = pd.Series(dtype='float')

    # for each row in the summary dataframe, append the result
    for i, row in summary.iterrows():
        if i == 0:
            ensemble = pd.concat([ensemble, results['val_pred'][row.Iteration]], axis=1)
            ty_ensemble = pd.concat([ty_ensemble, results['ty_pred'][row.Iteration]], axis=1)
        else:
            ensemble = pd.concat([ensemble, results['val_pred'][row.Iteration]['pred' + str(row.Iteration)]], axis=1)
            ty_ensemble = pd.concat([ty_ensemble, results['ty_pred'][row.Iteration]['pred' + str(row.Iteration)]], axis=1)

    # get the median prediction from each of the models
    ensemble = ensemble.drop(0, axis=1)
    ty_ensemble = ty_ensemble.drop(0, axis=1)
    if agg_type=='mean':
        ensemble['pred'] = ensemble.iloc[:, 4:].mean(axis=1)
        ty_ensemble['pred'] = ty_ensemble.iloc[:, 3:].mean(axis=1)
    elif agg_type=='median':
        ensemble['pred'] = ensemble.iloc[:, 4:].median(axis=1)
        ty_ensemble['pred'] = ty_ensemble.iloc[:, 3:].median(axis=1)

    # store the predictions in the results dictionary
    results['val_pred'][int(n)] = ensemble
    results['ty_pred'][int(n)] = ty_ensemble

    # remove the +4 amount and iteration number from the n to properly
    # name the ensemble model with number of included models
    if agg_type=='median': j=str(int(n)-4-iters) 
    else: j=str(int(n) - iters)

    # create the output list to append error metrics
    ens_error = [n, 'Ensemble'+j]
    ens_error.extend(error_compare(ensemble))
    
    # create dataframe of results
    ens_error = pd.DataFrame(ens_error).T
    ens_error.columns = summary.columns
    
    return results, ens_error


def convert_to_float(df):
    for c in df.columns:
        try:
            df[c] = df[c].astype('float')
        except:
            pass
    
    return df


def adp_filter(df, adp_ppg_low, adp_ppg_high):
    '''Filters the dataframe to the specified adp cutoffs'''
    
    return df[(eval(f'df.avg_pick_pred{adp_ppg_low}')) & \
               (eval(f'df.avg_pick_pred{adp_ppg_high}'))].reset_index(drop=True)


#
# # Breakout Classification Functions


def get_adp_predictions(df_train, df_predict, year_min_int):
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()

    output = pd.DataFrame()
    for yy in range(int(df_train.year.min()+year_min_int), int(df_train.year.max()+1)):

        X = df_train[df_train.year < yy].avg_pick.values.reshape(-1, 1)
        y = df_train[df_train.year < yy].y_act

        lr.fit(X, y)
        pred = lr.predict(df_train[df_train.year == yy].avg_pick.values.reshape(-1, 1))
        output_tmp = df_train.loc[df_train.year == yy, ['player', 'year', 'avg_pick', 'y_act']].reset_index(drop=True)
        output_tmp = pd.concat([output_tmp, pd.Series(pred)], axis=1)

        output = pd.concat([output, output_tmp], axis=0)

    output = output.rename(columns={0: 'avg_pick_pred'})
    output['pct_off'] = (output.y_act - output.avg_pick_pred) / output.avg_pick_pred
    output = output.drop(['y_act', 'avg_pick'], axis=1)
    
    df_train = pd.merge(df_train, output, on=['player', 'year'])
    df_predict['avg_pick_pred'] = lr.predict(df_predict.avg_pick.values.reshape(-1,1))
    try: df_predict['pct_off'] = (df_predict.y_act - df_predict.avg_pick_pred) / df_predict.avg_pick_pred
    except: print('No y_act in df_predict')
    
    return df_train, df_predict, lr


def class_label(df, pct_off, act_ppg):
    '''create the label and filter based on inputs'''
    
    df['label'] = 0
    df.loc[(eval(f'df.pct_off{pct_off}')) & (eval(f'df.y_act{act_ppg}')), 'label'] = 1
    df = df.drop(['y_act', 'pct_off'], axis=1).rename(columns={'label': 'y_act'})
    
    return df


def remove_classification_collinear(df, corr_cut, collinear_cutoff, keep_cols):
    
    means = df.groupby('y_act').agg('median').T
    means['mean_diff'] = abs((means[0] - means[1]) / np.mean([abs(means[0]), abs(means[1])]))
    means = means[means.mean_diff > corr_cut]
    col_order = list(means.sort_values(by='mean_diff', ascending=False).index)
    df_cor = df.corr().loc[means.index, means.index]

    all_bad_cols = []
    all_good_cols = keep_cols
    for col in col_order:

        if col in all_bad_cols:
            continue
        else:
            all_good_cols.append(col)
            cor = df_cor[col]
            bad_cols = cor[abs(cor) > collinear_cutoff].index
            all_bad_cols.extend(bad_cols)

    df = df[all_good_cols]
    df = df.loc[:,~df.columns.duplicated()]
    
    return df


def class_validation(estimator, df_train_orig, df_predict, corr_cut, collinear_cutoff, use_smote,
                     skip_years=2, scale=False, pca=False, n_components=0.5):  
    
    from sklearn.metrics import matthews_corrcoef
    from imblearn.over_sampling import SMOTE

    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------

    # remove collinear variables based on difference of means between the 0 and 1 labeled groups
    df_train = remove_classification_collinear(df_train_orig, corr_cut, collinear_cutoff, ['player', 'avg_pick', 'year', 'y_act'])

    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]

    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        val_split = df_train[df_train.year == m]

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)
            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_val = X_val.values

        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

        # skip over the first two year of predictions due to high error for xgb / lgbm
        val_predictions = np.append(val_predictions, val_predict, axis=0)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========

    # store the current predictions in the results tracker
    val_df = df_train.loc[df_train.year.isin(years), 
                          ['player', 'year', 'avg_pick', 'y_act']].reset_index(drop=True)
    val_pred = pd.concat([val_df, pd.Series(val_predictions, name='pred')], axis=1)

    # calculate the RMSE and MAE of the ensemble predictions
    result = round(matthews_corrcoef(val_pred.pred, val_df.y_act), 3)
    
    #==========
    # Create Predictions for Current Year
    #==========
    
    pred_cols = list(df_train.columns)
    pred_cols.remove('y_act')
    X_train, X_val, y_train, _ = X_y_split(df_train, df_predict[pred_cols], scale, pca, n_components)
    output_cols = list(X_train.columns)

    # fit training data and creating prediction based on validation data
    if use_smote:
        knn = int(len(y_train[y_train==1])*0.5)
        smt = SMOTE(k_neighbors=knn, random_state=1234)
        X_train, y_train = smt.fit_resample(X_train.values, y_train)
        X_val = X_val.values

    estimator.fit(X_train, y_train)
    ty_pred = pd.Series(estimator.predict(X_val), name='pred')
    ty_pred = pd.concat([df_predict[['player', 'year', 'avg_pick']], ty_pred], axis=1)

    try:
        ty_pred_proba = pd.Series(estimator.predict_proba(X_val)[:,1], name='proba')
        ty_pred_proba = pd.concat([df_predict[['player', 'year', 'avg_pick']], ty_pred_proba], axis=1)
    except:
        ty_pred_proba = pd.Series(estimator.predict(X_val), name='proba')*0.5 + 0.25
        ty_pred_proba = pd.concat([df_predict[['player', 'year', 'avg_pick']], ty_pred_proba], axis=1)
    
    return result, val_pred, ty_pred, ty_pred_proba, estimator, output_cols


def class_ensemble(summary, results, n, iters, agg_type):
    '''
    Function that accepts multiple model results and averages them together to create an ensemble prediction.
    It then calculates metrics to validate that the ensemble is more accurate than individual models.
    '''

    from sklearn.metrics import matthews_corrcoef

    # initialize Series to store results
    ensemble = pd.Series(dtype='float')
    ty_ensemble = pd.Series(dtype='float')
    ty_proba = pd.Series(dtype='float')

    # for each row in the summary dataframe, append the result
    for i, row in summary.iterrows():
        if i == 0:
            ensemble = pd.concat([ensemble, results['val_pred'][row.Iteration]], axis=1)
            ty_ensemble = pd.concat([ty_ensemble, results['ty_pred'][row.Iteration]], axis=1)
            ty_proba = pd.concat([ty_proba, results['ty_proba'][row.Iteration]], axis=1)
        else:
            ensemble = pd.concat([ensemble, results['val_pred'][row.Iteration]['pred' + str(row.Iteration)]], axis=1)
            ty_ensemble = pd.concat([ty_ensemble, results['ty_pred'][row.Iteration]['pred' + str(row.Iteration)]], axis=1)
            ty_proba = pd.concat([ty_proba, results['ty_proba'][row.Iteration]['proba' + str(row.Iteration)]], axis=1)

    # get the median prediction from each of the models
    ensemble = ensemble.drop(0, axis=1)
    ty_ensemble = ty_ensemble.drop(0, axis=1)
    ty_proba = ty_proba.drop(0, axis=1)

    if agg_type=='mean':
        ensemble['pred'] = ensemble.iloc[:, 4:].mean(axis=1)
        ty_ensemble['pred'] = ty_ensemble.iloc[:, 3:].mean(axis=1)
        ty_proba['proba'] = ty_proba.iloc[:, 3:].mean(axis=1)
    elif agg_type=='median':
        ensemble['pred'] = ensemble.iloc[:, 4:].median(axis=1)
        ty_ensemble['pred'] = ty_ensemble.iloc[:, 3:].median(axis=1)
        ty_proba['proba'] = ty_proba.iloc[:, 3:].median(axis=1)

    ensemble.loc[ensemble.pred >= 0.5, 'pred'] = 1
    ensemble.loc[ensemble.pred < 0.5, 'pred'] = 0

    # store the predictions in the results dictionary
    results['val_pred'][int(n)] = ensemble
    results['ty_pred'][int(n)] = ty_ensemble
    results['ty_proba'][int(n)] = ty_proba
    
    # remove the +4 amount and iteration number from the n to properly
    # name the ensemble model with number of included models
    if agg_type=='median': j=str(int(n)-4-iters) 
    else: j=str(int(n) - iters)

    # create the output list to append error metrics
    ens_error = [n, 'Ensemble'+j]
    ens_error.append(round(matthews_corrcoef(ensemble.y_act, ensemble.pred), 3))
    
    # create dataframe of results
    ens_error = pd.DataFrame(ens_error).T
    ens_error.columns = summary.columns
    
    return results, ens_error


def class_create_summary(results, keep_first=False):
    
    # create summary dataframe
    summary = pd.DataFrame(results['summary']).T.reset_index()
    scol = ['Iteration', 'Model', 'F1Score']
    summary.columns = scol 

    # sort results by a given metric
    summary = summary.sort_values(by=['F1Score'], ascending=[False]).reset_index(drop=True)

    # if wanting to get the best model for each unique model type
    if keep_first:
        summary = summary.drop_duplicates(subset=['Model'], keep='first')

    return summary

# ================
# Combining Regression and Classification Models
# ================


# ---------
# Set the RF search params for each position
# ---------

pos['QB']['tree_params'] = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 18, 21, 25, 30],
    'splitter': ['random']
}

pos['RB']['tree_params'] = {
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 18, 21, 25],
    'splitter': ['random']
}

pos['WR']['tree_params'] = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 20, 25, 30, 35],
    'splitter': ['random']
}


pos['TE']['tree_params'] = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2],
    'min_samples_leaf': [15, 18, 22, 25, 30],
    'splitter': ['random']
}


def searching(est, params, X_grid, y_grid, n_jobs=3, print_results=True):
    '''
    Function to perform GridSearchCV and return the test RMSE, as well as the 
    optimized and fitted model
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    
    Search = GridSearchCV(estimator=est,
                          param_grid=params,
                          scoring='neg_mean_squared_error',
                          n_jobs=n_jobs,
                          cv=5,
                          return_train_score=True,
                          iid=False)
   
    search_results = Search.fit(X_grid, y_grid)
   
    best_params = search_results.cv_results_['params'][search_results.best_index_]
    est.set_params(**best_params)
    
    test_rmse = cross_val_score(est, X_grid, y_grid, scoring='neg_mean_squared_error', cv=5)
    test_rmse = np.mean(np.sqrt(np.abs(test_rmse)))
    
    if print_results==True:
        print(best_params)
        print('Best RMSE: ', round(test_rmse, 3))
   
    est.fit(X_grid, y_grid)
       
    return est


def create_distributions(self, prior_repeats=15, dist_size=1000, show_plots=False):
    
    # historical standard deviation and mean for actual results
    hist_std = self.df_train.groupby('player').agg('std').dropna()
    hist_mean = self.df_train.groupby('player').agg('mean').dropna()
    
    # merge historicaly mean and standard deviations
    hist_mean_std = pd.merge(hist_std, hist_mean, how='inner', left_index=True, right_index=True)
    
    # calculate global coefficient of variance for players that don't have enough historical results
    global_cv = (hist_mean_std.y_act_x / hist_mean_std.y_act_y).mean()
    
    #==========
    # Loop to Create Prior and Posterior Distributions
    #==========

    self.df_test = self.df_test.sort_values(by='pred', ascending=False)

    results = pd.DataFrame()

    for player in self.df_test.player[0:]:

        # set seed
        np.random.seed(1234)

        # create list for results
        results_list = [player]

        #==========
        # Pull Out Predictions and Actual Results for Given Player to Create Prior
        #==========

        #--------
        # Extract this year's results and multiply by prior_repeats
        #--------

        # extract predictions from ensemble and updated predictions based on cluster fit
        ty = self.df_test.loc[self.df_test.player == player, ['player', 'pred']]
        #ty_c = self.df_test.loc[self.df_test.player == player, ['player', 'cluster_pred']]

        # replicate the predictions to increase n_0 for the prior
        ty = pd.concat([ty]*prior_repeats, ignore_index=True)
        #ty_c = pd.concat([ty_c]*prior_repeats, ignore_index=True)

        # rename the prediction columns to 'points'
        ty = ty.rename(columns={'pred': 'points'})
        #ty_c = ty_c.rename(columns={'cluster_pred': 'points'})

        #--------
        # Extract previous year's results, if available
        #--------

        # pull out the most recent 5 years worth of performance, if available
        py = self.df_train.loc[self.df_train.player == player, ['player', 'y_act']].reset_index(drop=True)[0:5]

        # convert y_act to points name
        py = py.rename(columns={'y_act': 'points'})

        #--------
        # Create Prior Distribution and Conjugant Hyperparameters
        #--------

        # combine this year's prediction, the cluster prediction, and previous year actual, if available
        priors = pd.concat([ty, py], axis=0)

        # set m_0 to the priors mean
        m_0 = priors.points.mean()

        # Create the prior variance through a weighted average of the actual previous year
        # performance and a global coefficient of variance multiple by the prior mean.
        # If there is not at least 3 years of previous data, simply use the global cv.
        if py.shape[0] >= 3:
            s2_0 = ((py.shape[0]*py.points.std()**2) + (2*prior_repeats*(m_0 * global_cv)**2)) / (py.shape[0] + 2*prior_repeats)
        else:
            s2_0 = (m_0 * global_cv)**2

        # set the prior sample size and degrees of freedom
        n_0 = priors.shape[0]
        v_0 = n_0 - 1

        # calculate the prior distribution
        prior_y = np.random.normal(loc=m_0, scale=np.sqrt(s2_0), size=dist_size)

        #--------
        # Create the Data and Data Hyperparameters
        #--------

        # pull out the cluster for the current player
        ty_cluster = self.df_test[self.df_test.player == player].cluster.values[0]

        # create a list of the actual points scored to be used as updating data
        update_data = self.df_train[self.df_train.cluster == ty_cluster].y_act

        # set ybar to the mean of the update data
        ybar = update_data.mean()

        # calculate the standard deviation based on the 5th and 95th percentiles
        s2 = ((np.percentile(update_data, q=95)-np.percentile(update_data, q=5)) / 4.0)**2

        # determine the n as the number of data points
        n = len(update_data)

        #--------
        # Create the Posterior Distribution 
        #--------

        # set the poster n samples
        n_n = n_0 + n 

        # update the poster mean
        m_n = (n*ybar + n_0*m_0) / n_n 

        # update the posterior degrees of freedom
        v_n = v_0 + n 

        # update the posterior variance
        s2_n = ((n-1)*s2 + v_0*s2_0 + (n_0*n*(m_0 - ybar)**2)/n_n)/v_n

        # calculate the gamma distribution and convert to sigma
        phi = np.random.gamma(shape=v_n/2, scale=2/(s2_n*v_n), size=dist_size)
        sigma = 1/np.sqrt(phi)

        # calculate the posterior mean
        post_mu = np.random.normal(loc=m_n, scale=sigma/(np.sqrt(n_n)), size=dist_size)

        # create the posterior distribution
        pred_y =  np.random.normal(loc=post_mu, scale=sigma, size=dist_size)

        results_list.extend(pred_y*16)
        results = pd.concat([results, pd.DataFrame(results_list).T], axis=0)

        if show_plots == True:

            # set plot bins based on the width of the distribution
            bins = int((np.percentile(pred_y, 97.5)*16 - np.percentile(pred_y, 2.55)*16) / 10)

            # print the player name
            print(player)

            # create a plot of the prior distribution and a line for the mean
            pd.Series(prior_y*16, name='Prior').plot.hist(alpha=0.4, color='grey', bins=bins, legend=True, 
                                                            xlim=[0, self.df_test.pred.max()*16*1.75])
            plt.axvline(x=prior_y.mean()*16, alpha=0.8, linestyle='--', color='grey')

            # create a plot of the posterior distribution and a line for the mean
            pd.Series(pred_y*16, name='Posterior').plot.hist(alpha=0.4, color='teal', bins=bins, legend=True,
                                                            edgecolor='black', linewidth=1)
            plt.axvline(x=pred_y.mean()*16, alpha=1, linestyle='--', color='black')

            # show the plots
            plt.show();

    return results.reset_index(drop=True)
