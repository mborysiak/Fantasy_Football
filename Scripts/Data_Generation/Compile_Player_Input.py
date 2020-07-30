# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

year = 2019

import sqlite3
import os
import pandas as pd
import numpy as np
from zData_Functions import *


# # Helper Functions

def adp_groupby(df, position):

    df_adp = pd.read_sql_query('''
        SELECT team, year, avg_pick FROM {0}_Stats
        UNION
        SELECT team, draft_year-1 year, log_avg_pick avg_pick from Rookie_{0}_Stats
    '''.format(position), conn)

    # create teammate ADP metrics to see if top ranked player
    min_teammate = df_adp.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.min).reset_index().rename(columns={'avg_pick': 'min_teammate'})
    max_teammate = df_adp.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.max).reset_index().rename(columns={'avg_pick': 'max_teammate'})
    avg_teammate = df_adp.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.mean).reset_index().rename(columns={'avg_pick': 'avg_teammate'})

    for d in [min_teammate, max_teammate, avg_teammate]:
        df = pd.merge(df, d, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])

    df['teammate_diff_min'] = df.avg_pick - df['min_teammate'] 
    df['teammate_diff_avg'] = df.avg_pick - df['avg_teammate']
    df['teammate_diff_max'] = df.avg_pick - df['max_teammate'] 
    df['teammate_diff_min_div'] = df.avg_pick / df['min_teammate'] 
    df['teammate_diff_avg_dv'] = df.avg_pick / df['avg_teammate']

    return df


def year_exp(df):

    # adding years of experience
    min_year = df.groupby('player').agg('min')['year'].reset_index()
    min_year = min_year.rename(columns={'year': 'min_year'})
    df = pd.merge(df, min_year, how='left', left_on='player', right_on='player')
    df['year_exp'] = df.year - df.min_year
    
    return df


def qb_run(df):

    qb_run = pd.read_sql_query('SELECT * FROM QB_Stats', con=conn)
    qb_run = qb_run[['team', 'qb_avg_pick', 'year', 'rush_att', 'rush_yds', 'rush_td', 'long_rush', 'rush_yd_per_att',
                    'rz_20_pass_complete', 'rz_20_pass_att','rz_20_complete_pct', 
                    'rz_20_pass_yds', 'rz_20_pass_td', 'rz_20_int',
                    'rz_10_pass_complete', 'rz_10_pass_att', 'rz_10_complete_pct',
                    'rz_10_pass_yds', 'rz_10_pass_td', 'rz_10_int', 'rz_20_rush_att',
                    'rz_20_rush_yds', 'rz_20_rush_td', 'rz_20_rush_pct', 'rz_10_rush_att',
                    'rz_10_rush_yds', 'rz_10_rush_td', 'rz_10_rush_pct', 'rz_5_rush_att',
                    'rz_5_rush_yds', 'rz_5_rush_td', 'rz_5_rush_pct']]
    max_qb_pick = qb_run.groupby(['team','year']).agg({'qb_avg_pick':'max'}).reset_index()
    qb_run = pd.merge(qb_run, max_qb_pick, how='inner', 
                      left_on=['team', 'year', 'qb_avg_pick'],
                      right_on=['team', 'year', 'qb_avg_pick']).drop('qb_avg_pick', axis=1)
    cols = ['team', 'year']
    cols.extend(['qb_' + c for c in qb_run.columns[2:]])
    qb_run.columns = cols

    qb_run = qb_run.groupby(['team', 'year']).agg('max').reset_index()

    df = pd.merge(df, qb_run, how='left', on=['team', 'year'])
    print(df.isnull().sum()[df.isnull().sum()>0])
    df.fillna(0,inplace=True)
    print('')
    print(df.isnull().sum()[df.isnull().sum()>0])
    
    return df


def draft_value(df, pos):
    
    early_draft = pd.read_sql_query(f'''SELECT player, year min_year, nfl_draft_value 
                                         FROM Draft_Positions a
                                         JOIN (SELECT Pick, Value nfl_draft_value 
                                               FROM Draft_Values) b
                                               ON a.Pick=b.Pick
                                         WHERE pos='{pos}'
                                               AND year <= 1998 ''', conn)

    earliest = early_draft.groupby('player').agg({'min_year': 'min',
                                                  'nfl_draft_value': 'max'}).reset_index()
    early_draft = pd.merge(early_draft, earliest, on=['player', 'min_year', 'nfl_draft_value'])

    # adding years of experience
    early_min = df.loc[df.year == 1998, ['player']]
    early_min = pd.merge(early_min, early_draft, on='player')


    draft_val = pd.read_sql_query(f'''SELECT player, team, year min_year, nfl_draft_value 
                                     FROM Draft_Positions a
                                     JOIN (SELECT Pick, Value nfl_draft_value 
                                           FROM Draft_Values) b
                                           ON a.Pick=b.Pick
                                     WHERE pos='{pos}'
                                           and year >= 1999''', conn)


    dups = draft_val.groupby('player').agg('count')
    dup_player = list(dups[dups.min_year > 1].index)

    later_min = df[(df.year > 1998) & ~(df.player.isin(early_min.player))]
    later_min = later_min.groupby('player').agg({'year': 'min'}).reset_index()
    later_min = pd.merge(later_min, df[['player', 'year', 'team']], on=['player', 'year'])

    later_min_dup = later_min[later_min.player.isin(dup_player)].reset_index(drop=True)
    later_min = later_min[~later_min.player.isin(dup_player)].reset_index(drop=True).drop('team', axis=1)

    later_min = pd.merge(later_min, draft_val, on=['player'], how='left').drop('team', axis=1)
    later_min.loc[later_min.min_year.isnull(), 'min_year'] = later_min.loc[later_min.min_year.isnull(), 'year']
    later_min = later_min.fillna(1)
    later_min = later_min.drop('year', axis=1)

    later_min_dup = pd.merge(later_min_dup, draft_val, on=['player', 'team'], how='left').drop(['team', 'year'], axis=1)

    mins = pd.concat([early_min, later_min, later_min_dup], axis=0).reset_index(drop=True)
    df2 = pd.merge(df, mins, on='player')

    # check fo duplicates()
    df_cnts = df.groupby('player').agg(year1=('year', 'count')).reset_index()
    df2_cnts = df2.groupby('player').agg(year2=('year', 'count')).reset_index()

    check_dup = pd.merge(df_cnts, df2_cnts, on='player')
    print('Duplicates:', list(check_dup[check_dup.year1 != check_dup.year2].player))

    df = pd.merge(df, mins, on='player')
    df['year_exp'] = (df.year - df.min_year)
    df['nfl_draft_value_decay'] = df.nfl_draft_value * np.exp(-df.year_exp)
    df['year_exp'] = df['year_exp']+1
    
    return df

# # Compile RB

# +
#==========
# Load team-based statistics
#==========

'''
Pull in the oline, quarterback, and overall team offense statistics, and join them 
to the player data. This will provide team-based context for the players, as well as
allow for grouped statistics generation.
'''

# load prepared data
conn = sqlite3.connect(f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/Data/Databases/Season_Stats.sqlite3')
query = ''' 
    SELECT * 
    FROM RB_Stats A 
    INNER JOIN OLine_Stats B ON A.team = B.team AND A.year = B.year
    INNER JOIN Team_Efficiency C ON A.team = C.team AND A.year = C.year
    INNER JOIN Team_Offensive_Stats D ON A.team = D.team AND A.year = D.year
    INNER JOIN QB_PosPred E ON A.team = E.team AND A.year = E.year
    LEFT JOIN Team_Drafts F ON A.team = F.team AND A.year = F.year 
    '''

df = pd.read_sql_query(query, con=conn)

# remove duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# ensure everything is numeric
for col in df.columns:
    try:
        df[col] = df[col].astype('float')
    except:
        pass

# +
#==========
# Creating team based grouped statistics
#==========

'''
Create grouped statistics based on the team and teammates. For example,
create total touches by team feature, as well as how the average, min, and max
teammate adps compare to the current player.
'''

# groupby team and year to get total rb touches for each team
team_touches = df.groupby(['team', 'year'], group_keys=False)['rush_att'].agg(np.sum).reset_index().rename(columns={'rush_att': 'rb_att_on_team'})
df = pd.merge(df, team_touches, how='left', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_rush_att'] = 1-(df['rb_att_on_team'] / df['tm_rush_att'])
df['available_rush_att_2'] = 1-((df['rb_att_on_team'] - df['att']) / df['tm_rush_att'])

team_tgts = df.groupby(['team', 'year'], group_keys=False)['tgt'].agg(np.sum).reset_index().rename(columns={'tgt': 'tgt_on_team'})
df = pd.merge(df, team_tgts, how='left', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_tgt'] = 1-(df['tgt_on_team'] / df['tm_pass_att'])
df['available_tgt_2'] = 1-((df['tgt_on_team'] - df['tgt']) / df['tm_pass_att'])

# create market share statistics
df['ms_rush_att'] = df['att'] / df['tm_rush_att']
df['ms_rush_yd'] = df['rush_yds'] / df['tm_rush_yds']
df['ms_rush_td'] = df['rush_td'] / df['tm_rush_td']
df['ms_rec_yd'] = df['rec_yds'] / df['tm_pass_yds']
df['ms_tgts'] = df['tgt'] / df['tm_pass_att']
df['rush_rec_ratio'] = df.rush_yds / df.rec_yds
df['ms_rec_td'] = df['rec_td'] / df['tm_pass_td']

df['ms_rush_yd_per_att'] = df['ms_rush_yd'] / df['ms_rush_att']
df['avail_x_newteam'] = df['available_rush_att'] * df['new_team']

df['total_rz_rush_att'] = df.rz_20_rush_att + df.rz_10_rush_att + df.rz_5_rush_att +1
df['rz_rush_td_ratio'] = df.rush_td / df.total_rz_rush_att

df['total_rz_rec_att'] = df.rz_20_tgt + df.rz_10_tgt + 1
df['rz_rec_td_ratio'] = df.rec_td / df.total_rz_rec_att

df['total_rz_att'] = df.rz_20_rush_att + df.rz_10_rush_att + df.rz_5_rush_att + df.rz_20_tgt + df.rz_10_tgt + 1
df['rz_total_td_ratio'] = df.rec_td / df.total_rz_att
# -

df = adp_groupby(df, 'RB')
df = draft_value(df, 'RB')
df = qb_run(df)


append_to_db(df, db_name='Model_Inputs', table_name='RB_' + str(year), if_exist='replace')

# # Compile WR

# +
#==========
# Load team-based statistics
#==========

'''
Pull in the oline, quarterback, and overall team offense statistics, and join them 
to the player data. This will provide team-based context for the players, as well as
allow for grouped statistics generation.
'''

# load prepared data
conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Databases/Season_Stats.sqlite3')
query = ''' 
    SELECT * 
    FROM WR_Stats A 
    INNER JOIN OLine_Stats B ON A.team = B.team AND A.year = B.year
    INNER JOIN Team_Efficiency C ON A.team = C.team AND A.year = C.year
    INNER JOIN Team_Offensive_Stats D ON A.team = D.team AND A.year = D.year
    INNER JOIN QB_PosPred E ON A.team = E.team AND A.year = E.year
    INNER JOIN Team_Drafts F ON A.team = F.team AND A.year = F.year'''

df = pd.read_sql_query(query, con=conn)

# remove duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# ensure everything is numeric
for col in df.columns:
    try:
        df[col] = df[col].astype('float')
    except:
        pass

# +
#==========
# Creating team based grouped statistics
#==========

'''
Create grouped statistics based on the team and teammates. For example,
create total touches by team feature, as well as how the average, min, and max
teammate adps compare to the current player.
'''

# groupby team and year to get total wr yard and targets for each team
team_tgts = df.groupby(['team', 'year'], group_keys=False)['tgt'].agg(np.sum).reset_index().rename(columns={'tgt': 'tgt_on_team'})
df = pd.merge(df, team_tgts, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_tgt'] = 1-(df['tgt_on_team'] / df['tm_pass_att'])
df['available_tgt_2'] = 1-((df['tgt_on_team'] - df['tgt']) / df['tm_pass_att'])

team_yds = df.groupby(['team', 'year'], group_keys=False)['rec_yds'].agg(np.sum).reset_index().rename(columns={'rec_yds': 'yds_on_team'})
df = pd.merge(df, team_yds, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_yds'] = 1-(df['yds_on_team'] / df['tm_pass_yds'])
df['available_yds_2'] = 1-((df['yds_on_team']-df['rec_yds']) / df['tm_pass_yds'])

# create marketshare statistics
df['ms_rec_yd'] = df['rec_yds'] / df['tm_pass_yds']
df['ms_tgts'] = df['tgt'] / df['tm_pass_att']
df['ms_yds_per_tgts'] = df['ms_rec_yd'] / df['ms_tgts']

df['total_rz_rec_att'] = df.rz_20_tgt + df.rz_10_tgt + 1
df['rz_rec_td_ratio'] = df.rec_td / df.total_rz_rec_att

df['avail_tgt_x_newteam'] = df['available_tgt'] * df['new_team']
df['avail_yds_x_newteam'] = df['available_yds'] * df['new_team']
# -

df = adp_groupby(df, 'WR')
df = year_exp(df)
df = qb_run(df)
df = draft_value(df)

append_to_db(df, db_name='Model_Inputs.sqlite3', table_name='WR_' + str(year), if_exist='replace')

# # Compile QB

# +
#==========
# Load team-based statistics
#==========

'''
Pull in the oline, quarterback, and overall team offense statistics, and join them 
to the player data. This will provide team-based context for the players, as well as
allow for grouped statistics generation.
'''

# load prepared data
conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Databases/Season_Stats.sqlite3')
query = ''' 
    SELECT * 
    FROM QB_Stats A 
    INNER JOIN OLine_Stats B ON A.team = B.team AND A.year = B.year
    INNER JOIN Team_Efficiency C ON A.team = C.team AND A.year = C.year
    INNER JOIN Team_Offensive_Stats D ON A.team = D.team AND A.year = D.year
    INNER JOIN QB_PosPred E ON A.team = E.team AND A.year = E.year
     INNER JOIN Team_Drafts F ON A.team = F.team AND A.year = F.year
    '''

df = pd.read_sql_query(query, con=conn)

# remove duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# rename games column to fit into model script
df = df.rename(columns={'qb_games': 'games'})
df = df.rename(columns={'qb_avg_pick': 'avg_pick'})
df = df.rename(columns={'qb_age': 'age'})

# ensure everything is numeric
for col in df.columns:
    try:
        df[col] = df[col].astype('float')
    except:
        pass
# -

df = year_exp(df)
df = draft_value(df)

append_to_db(df, db_name='Model_Inputs.sqlite3', table_name='QB_' + str(year), if_exist='replace')

# # Compile TE

# +
#==========
# Load team-based statistics
#==========

'''
Pull in the oline, quarterback, and overall team offense statistics, and join them 
to the player data. This will provide team-based context for the players, as well as
allow for grouped statistics generation.
'''

# load prepared data
conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Databases/Season_Stats.sqlite3')
query = ''' 
    SELECT * 
    FROM TE_Stats A 
    INNER JOIN OLine_Stats B ON A.team = B.team AND A.year = B.year
    INNER JOIN Team_Efficiency C ON A.team = C.team AND A.year = C.year
    INNER JOIN Team_Offensive_Stats D ON A.team = D.team AND A.year = D.year
    INNER JOIN QB_PosPred E ON A.team = E.team AND A.year = E.year
    INNER JOIN Team_Drafts F ON A.team = F.team AND A.year = F.year'''

df = pd.read_sql_query(query, con=conn)

# remove duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# ensure everything is numeric
for col in df.columns:
    try:
        df[col] = df[col].astype('float')
    except:
        pass

# +
#==========
# Creating team based grouped statistics
#==========

'''
Create grouped statistics based on the team and teammates. For example,
create total touches by team feature, as well as how the average, min, and max
teammate adps compare to the current player.
'''

# groupby team and year to get total wr yard and targets for each team
team_tgts = df.groupby(['team', 'year'], group_keys=False)['tgt'].agg(np.sum).reset_index().rename(columns={'tgt': 'tgt_on_team'})
df = pd.merge(df, team_tgts, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_tgt'] = 1-(df['tgt_on_team'] / df['tm_pass_att'])
df['available_tgt_2'] = 1-((df['tgt_on_team'] - df['tgt']) / df['tm_pass_att'])

team_yds = df.groupby(['team', 'year'], group_keys=False)['rec_yds'].agg(np.sum).reset_index().rename(columns={'rec_yds': 'yds_on_team'})
df = pd.merge(df, team_yds, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_yds'] = 1-(df['yds_on_team'] / df['tm_pass_yds'])
df['available_yds_2'] = 1-((df['yds_on_team']-df['rec_yds']) / df['tm_pass_yds'])

# create marketshare statistics
df['ms_rec_yd'] = df['rec_yds'] / df['tm_pass_yds']
df['ms_tgts'] = df['tgt'] / df['tm_pass_att']
df['ms_yds_per_tgts'] = df['ms_rec_yd'] / df['ms_tgts']

df['total_rz_rec_att'] = df.rz_20_tgt + df.rz_10_tgt + 1
df['rz_rec_td_ratio'] = df.rec_td / df.total_rz_rec_att

df['avail_tgt_x_newteam'] = df['available_tgt'] * df['new_team']
df['avail_yds_x_newteam'] = df['available_yds'] * df['new_team']
# -

df = adp_groupby(df, 'WR')
df = year_exp(df)
df = qb_run(df)
df = draft_value(df)

append_to_db(df, db_name='Model_Inputs.sqlite3', table_name='TE_' + str(year), if_exist='replace')

# # Compile Rookie RB

# +
#==========
# Create team based grouped statistics
#==========

'''
Create grouped statistics based on the team and teammates. For example,
create total touches by team feature, as well as how the average, min, and max
teammate adps compare to the current player.
'''

#--------
# Pull in the saved out RB dataset and combine with team stats
#--------

# load prepared data
conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Season_Stats.sqlite3')

rookie_rb = pd.read_sql_query("select * from rookie_rb_stats", conn)
rookie_rb = rookie_rb.rename(columns={'draft_year': 'year'})
rookie_rb['year'] = rookie_rb['year'] - 1

team_stats_q = '''SELECT * FROM OLine_Stats A 
    INNER JOIN Team_Efficiency C ON A.team = C.team AND A.year = C.year
    INNER JOIN Team_Offensive_Stats D ON A.team = D.team AND A.year = D.year
    INNER JOIN QB_PosPred E ON A.team = E.team AND A.year = E.year'''
team_stats = pd.read_sql_query(team_stats_q, conn)

# remove duplicated columns
team_stats = team_stats.loc[:, ~team_stats.columns.duplicated()]
team_stats = team_stats.drop_duplicates()

rookie_rb = pd.merge(rookie_rb, team_stats, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])

# +
df = pd.read_sql_query('''select * from rb_stats a
                        inner join team_offensive_stats b on a.team=b.team and a.year=b.year''', con=conn)

# remove duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# ensure everything is numeric
for col in df.columns:
    try:
        df[col] = df[col].astype('float')
    except:
        pass
    
#--------
# Create team based stats
#--------

# groupby team and year to get total rb touches for each team
team_touches = df.groupby(['team', 'year'], group_keys=False)['rush_att'].agg(np.sum).reset_index().rename(columns={'rush_att': 'rb_att_on_team'})
df = pd.merge(df, team_touches, how='left', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_rush_att'] = 1-(df['rb_att_on_team'] / df['tm_rush_att'])

team_tgts = df.groupby(['team', 'year'], group_keys=False)['tgt'].agg(np.sum).reset_index().rename(columns={'tgt': 'tgt_on_team'})
df = pd.merge(df, team_tgts, how='left', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_tgt'] = 1-(df['tgt_on_team'] / df['tm_pass_att'])

# create teammate ADP metrics to see if top ranked player
min_teammate = df.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.min).reset_index().rename(columns={'avg_pick': 'min_teammate'})
max_teammate = df.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.max).reset_index().rename(columns={'avg_pick': 'max_teammate'})
avg_teammate = df.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.mean).reset_index().rename(columns={'avg_pick': 'avg_teammate'})

available = df[['team', 'year', 'available_rush_att', 'available_tgt']]
for d in [min_teammate, max_teammate, avg_teammate]:
    available = pd.merge(available, d, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])

available = available.drop_duplicates().reset_index(drop=True)

rookie_rb = pd.merge(rookie_rb, available, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
rookie_rb['teammate_diff_min'] = rookie_rb.avg_pick - rookie_rb['min_teammate'] 
rookie_rb['teammate_diff_avg'] = rookie_rb.avg_pick - rookie_rb['avg_teammate']
rookie_rb['teammate_diff_max'] = rookie_rb.avg_pick - rookie_rb['max_teammate'] 
rookie_rb['teammate_diff_min_div'] = rookie_rb.avg_pick / rookie_rb['min_teammate'] 
rookie_rb['teammate_diff_avg_dv'] = rookie_rb.avg_pick / rookie_rb['avg_teammate']
# -

rookie_rb = qb_run(rookie_rb)

append_to_db(rookie_rb, db_name='Model_Inputs.sqlite3', table_name='Rookie_RB_' + str(year), if_exist='replace')

# # Compile Rookie WR

# +
#==========
# Create team based grouped statistics
#==========

'''
Create grouped statistics based on the team and teammates. For example,
create total touches by team feature, as well as how the average, min, and max
teammate adps compare to the current player.
'''

#--------
# Pull in the saved out RB dataset and combine with team stats
#--------

# load prepared data
conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Season_Stats.sqlite3')

rookie_wr = pd.read_sql_query("select * from rookie_wr_stats", conn)
rookie_wr = rookie_wr.rename(columns={'draft_year': 'year'})
rookie_wr['year'] = rookie_wr['year'] - 1

team_stats_q = '''SELECT * FROM OLine_Stats A 
    INNER JOIN Team_Efficiency C ON A.team = C.team AND A.year = C.year
    INNER JOIN Team_Offensive_Stats D ON A.team = D.team AND A.year = D.year
    INNER JOIN QB_PosPred E ON A.team = E.team AND A.year = E.year'''
team_stats = pd.read_sql_query(team_stats_q, conn)

# remove duplicated columns
team_stats = team_stats.loc[:, ~team_stats.columns.duplicated()]
team_stats = team_stats.drop_duplicates()

rookie_wr = pd.merge(rookie_wr, team_stats, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])

# +
df = pd.read_sql_query('''select * from wr_stats a
                        inner join team_offensive_stats b on a.team=b.team and a.year=b.year''', con=conn)

# remove duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# ensure everything is numeric
for col in df.columns:
    try:
        df[col] = df[col].astype('float')
    except:
        pass
    
#--------
# Create team based stats
#--------

# get available targets by team
team_tgts = df.groupby(['team', 'year'], group_keys=False)['tgt'].agg(np.sum).reset_index().rename(columns={'tgt': 'tgt_on_team'})
df = pd.merge(df, team_tgts, how='left', left_on=['team', 'year'], right_on=['team', 'year'])
df['available_tgt'] = 1-(df['tgt_on_team'] / df['tm_pass_att'])

# create teammate ADP metrics to see if top ranked player
min_teammate = df.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.min).reset_index().rename(columns={'avg_pick': 'min_teammate'})
max_teammate = df.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.max).reset_index().rename(columns={'avg_pick': 'max_teammate'})
avg_teammate = df.groupby(['team', 'year'], group_keys=False)['avg_pick'].agg(np.mean).reset_index().rename(columns={'avg_pick': 'avg_teammate'})
available = df[['team', 'year', 'available_tgt']]

for d in [min_teammate, max_teammate, avg_teammate]:
    available = pd.merge(available, d, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
    
available = available.drop_duplicates().reset_index(drop=True)

rookie_wr = pd.merge(rookie_wr, available, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
rookie_wr['teammate_diff_min'] = rookie_wr.avg_pick - rookie_wr['min_teammate'] 
rookie_wr['teammate_diff_avg'] = rookie_wr.avg_pick - rookie_wr['avg_teammate']
rookie_wr['teammate_diff_max'] = rookie_wr.avg_pick - rookie_wr['max_teammate'] 
rookie_wr['teammate_diff_min_div'] = rookie_wr.avg_pick / rookie_wr['min_teammate'] 
rookie_wr['teammate_diff_avg_dv'] = rookie_wr.avg_pick / rookie_wr['avg_teammate']
# -

append_to_db(rookie_wr, db_name='Model_Inputs.sqlite3', table_name='Rookie_WR_' + str(year), if_exist='replace')
