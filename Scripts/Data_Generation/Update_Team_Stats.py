# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # User Inputs

# +
year = 2018

path = '/Users/Mark/Documents/Github/Fantasy_Football/'
db_name = 'Season_Stats.sqlite3'
# -

# # Load Packages

import pandas as pd
import numpy as np
import os
import sqlite3
from zData_Functions import *
import re

# # Update Team Efficiency

# +
team_off = pd.read_html('https://www.footballoutsiders.com/stats/teamoff')[0].iloc[:, 1:]

colnames_team_off = {
    1: 'team',
    2: 'overall_dvoa',
    3: 'last_year',
    4: 'weighted',
    5: 'rank',
    6: 'pass_off',
    7: 'pass_rank',
    8: 'rush_off',
    9: 'rush_rank',
    10: 'non_adj',
    11: 'non_adj_2',
    12: 'non_adj_3',
    13: 'variance',
    14: 'var_rank',
    15: 'schedule_diff',
    16: 'sched_rank'
}

team_off.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
team_off = team_off.rename(columns=colnames_team_off).dropna().reset_index(drop=True)
team_off['year'] = year
    
    
cols = ['overall_dvoa', 'pass_off', 'rush_off', 'variance' , 'schedule_diff']
for col in cols:
    team_off[col] = team_off[col].apply(name_clean)
    
team_off = team_off[['team', 'year', 'overall_dvoa', 'last_year', 'pass_off', 
                     'rush_off', 'variance', 'schedule_diff']]

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
    'OAK': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SF': 'SFO',
    'TB': 'TAM',
    'TEN': 'TEN',
    'WAS': 'WAS'
}

team_off['team'] = team_off.team.map(team_convert)
# -

# append_to_db(team_off, db_name=db_name, table_name='Team_Efficiency', if_exist='append')

# # Update Traditional Team Stats

# +
#==========
# Team Passing Stats
#==========

url_team = 'http://www.footballdb.com/stats/teamstat.html?lg=NFL&yr=' + str(year) + '&type=reg&cat=P&group=O&conf='
data_team = pd.read_html(url_team)[0]

cols= {
    'Team': 'team', 
    'Att': 'tm_pass_att',
    'Cmp': 'tm_pass_complete', 
    'Pct': 'tm_pass_complete_pct',
    'Yds': 'tm_pass_yds',
    'TD': 'tm_pass_td',
    'Int': 'tm_pass_int',
    'Sack': 'tm_sack',
    'Loss': 'tm_loss_sack_yds',
    'Rate': 'tm_qb_rate',
    'NetYds': 'tm_net_pass_yds',
    'Yds/G': 'tm_pass_yperg',
    'year': 'year'
}

df_team = data_team.rename(columns=cols).drop(['Gms', 'YPA'], axis=1)


# +
def unique_list(list_strings):
    ulist = []
    [ulist.append(x) for x in list_strings if x not in ulist]
    return ulist

team_names = []
for team in df_team.team:
    name_separate = re.sub(r"(\w)([A-Z])", r"\1 \2", team)
    team_name = ' '.join(unique_list(name_separate.split()))
    team_names.append(team_name)
    
df_team['team'] = team_names

# +
team_to_abb = {
    
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GNB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KAN',
    'Los Angeles Chargers LA': 'LAC',
    'San Diego Chargers': 'LAC',
    'Los Angeles Rams LA': 'LAR',
    'Los Angeles Rams': 'LAR',
    'St. Louis Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NWE',
    'New Orleans Saints': 'NOR',
    'New York Giants NY': 'NYG',
    'New York Jets NY': 'NYJ',
    'Oakland Raiders': 'OAK',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'Seattle Seahawks': 'SEA',
    'San Francisco 49ers': 'SFO',
    'Tampa Bay Buccaneers': 'TAM',
    'Tennessee Titans': 'TEN',
    'Tennessee Oilers': 'TEN',
    'Washington Redskins': 'WAS'
}

df_team.team = df_team.team.map(team_to_abb)
df_team['year'] = year
dfteam = df_team.reset_index(drop=True)

# +
#==========
# Team Rushing Stats
#==========
url_team = 'https://www.footballdb.com/stats/teamstat.html?lg=NFL&yr=' + str(year) + '&type=reg&cat=R&group=O&conf='
data_team_rush = pd.read_html(url_team)[0]

cols= {
    'Team': 'team', 
    'Att': 'tm_rush_att',
    'Yds': 'tm_rush_yds',
    'Avg': 'team_rush_avg_att',
    'TD': 'tm_rush_td',
    'FD': 'tm_rush_first_downs',
    'Yds/G': 'tm_rush_yperg',
    'year': 'year'
}

df_rush = data_team_rush.rename(columns=cols).drop('Gms', axis=1)

team_names = []
for team in df_rush.team:
    name_separate = re.sub(r"(\w)([A-Z])", r"\1 \2", team)
    team_name = ' '.join(unique_list(name_separate.split()))
    team_names.append(team_name)
    
df_rush['team'] = team_names

df_rush.team = df_rush.team.map(team_to_abb)

df_rush['year'] = year
df_rush = df_rush.reset_index(drop=True)

# +
#==========
# Overall Team Stats
#==========

url_team = 'https://www.footballdb.com/stats/teamstat.html?lg=NFL&yr=' + str(year) + '&type=reg&cat=W&group=O&conf='
data_team_fd = pd.read_html(url_team)

df_fd = data_team_fd[0].T.reset_index(drop=True).T

cols= {
    0: 'team', 
    1: 'games',
    2: 'tm_rush_fd',
    3: 'tm_pass_fd',
    4: 'tm_pen_fd',
    5: 'tm_tot_fd',
    6: 'tm_third_down_att',
    7: 'tm_third_down_made',
    8: 'tm_third_down_pct',
    9: 'tm_fourth_down_att',
    10: 'tm_fourth_down_made',
    11: 'tm_fourth_down_pct'
}

df_fd = df_fd.rename(columns=cols)
df_fd = df_fd[['team', 'tm_rush_fd', 'tm_pass_fd', 'tm_pen_fd', 'tm_tot_fd', 'tm_third_down_att',
               'tm_third_down_made', 'tm_third_down_pct']]

df_fd['year'] = year

team_names = []
for team in df_fd.team:
    name_separate = re.sub(r"(\w)([A-Z])", r"\1 \2", team)
    team_name = ' '.join(unique_list(name_separate.split()))
    team_names.append(team_name)
    
df_fd['team'] = team_names

df_fd.team = df_fd.team.map(team_to_abb)

# +
#==========
# Merging All Team Data
#==========

df_team = pd.merge(df_team, df_rush, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])
df_team = pd.merge(df_team, df_fd, how='inner', left_on=['team', 'year'], right_on=['team', 'year'])

df_team['total_tm_td'] = df_team.tm_pass_td + df_team.tm_rush_td
df_team['total_tm_yds'] = df_team.tm_pass_yds + df_team.tm_rush_yds

df_team.iloc[:, 1:] = df_team.iloc[:, 1:].astype('float')
# -

# append_to_db(df_team, db_name=db_name, table_name='Team_Offensive_Stats', if_exist='append')

# # Update Offensive Line

# +
oline = pd.read_html('https://www.footballoutsiders.com/stats/ol')[0]

oline_df = oline.T.reset_index(drop=True).T

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
    'OAK': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SF': 'SFO',
    'TB': 'TAM',
    'TEN': 'TEN',
    'WAS': 'WAS'
}

oline_df['team'] = oline_df.team.map(team_convert)
for col in oline_df.columns:
    try:
        oline_df[col] = oline_df[col].astype('float')
    except:
        pass
    
oline_df = oline_df.dropna()
# -

# append_to_db(oline_df, db_name=db_name, table_name='OLine_Stats', if_exist='append')

# # QB for Position Players

# +
#==========
# Scraping Statistical and ADP Data
#==========

# pulling passing statistics
url_player = 'https://www.pro-football-reference.com/years/' + str(year) + '/passing.htm'
data_player = pd.read_html(url_player)[0]

# pulling historical player adp
url_adp_qb = 'http://www03.myfantasyleague.com/' + str(year+1) + '/adp?COUNT=100&POS=QB&ROOKIES=0&INJURED=1&CUTOFF=1&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=-1&IS_MOCK=-1&TIME='
data_adp_qb = pd.read_html(url_adp_qb)[1]

# +
#--------
# Cleaning Player Statistical Data
#--------

df_player = data_player.iloc[:, 1:]
df_player['year'] = year

colnames_pass = {
    'Player': 'player',
    'Tm': 'team',
    'Pos': 'position',
    'Age': 'qb_age',
    'G': 'qb_games',
    'GS': 'qb_games_started',
    'QBrec': 'qb_record',
    'Cmp': 'qb_complete',
    'Att': 'qb_att',
    'Cmp%': 'qb_complete_pct',
    'Yds': 'qb_yds',
    'TD': 'qb_tds',
    'TD%': 'qb_td_pct',
    'Int': 'int',
    'Int%': 'int_pct',
    'Lng': 'qb_long',
    'Y/A': 'yd_per_att',
    'AY/A': 'adj_yd_per_att',
    'Y/C': 'yd_per_comp',
    'Y/G': 'qb_yd_per_game',
    'Rate': 'qb_rating',
    'QBR': 'qbr',
    'Sk': 'qb_sacks',
    'Yds.1': 'yds_lost_sack',
    'NY/A': 'net_yd_per_att',
    'ANY/A': 'adj_net_yd_per_att',
    'Sk%': 'sack_pct',
    '4QC': 'fourth_qt_comeback',
    'GWD': 'game_winning_drives'
}

df_player = df_player.rename(columns=colnames_pass)
df_player = df_player[df_player.player != 'Player']

df_player = df_player[['player', 'team', 'year', 'qb_age', 'qb_games_started', 'qb_att', 'qb_rating', 
                       'qbr', 'qb_complete', 'qb_complete_pct', 'qb_yds', 'qb_yd_per_game', 'qb_tds', 
                       'qb_td_pct', 'int', 'int_pct', 'qb_long', 'yd_per_att', 'yd_per_comp',
                       'adj_net_yd_per_att', 'adj_yd_per_att', 'net_yd_per_att', 'qb_sacks', 'sack_pct',
                       'fourth_qt_comeback', 'game_winning_drives']]

# cleaning player name and stat categories
df_player['player'] = df_player.player.apply(name_clean)

# fill missing values with zero
df_player = df_player.fillna(0)

for col in df_player.columns:
    try:
        df_player[col] = df_player[col].astype('float')
    except:
        pass

# +
#==========
# Cleaning Player ADP Data
#==========

df_adp = clean_adp(data_adp_qb, year)
df_adp = df_adp.rename(columns={'avg_pick': 'qb_avg_pick'})

# +
#==========
# Merging and Formatting All Player Data
#==========

# merge adp and player data
player_df = pd.merge(df_player, df_adp, how = 'left', left_on = ['player', 'year'], right_on = ['player', 'year'])

# filter players who threw more than 50 passes in season
player_df = player_df[player_df.qb_att > 50].reset_index(drop=True)

# ensure all teams have same abbreviations for matching
adp_to_player_teams = {
    
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
    'GBP': 'GNB',
    'HOU': 'HOU',
    'IND': 'IND',
    'JAC': 'JAX',
    'KCC': 'KAN',
    'LAC': 'LAC',
    'SDC': 'LAC',
    'LAR': 'LAR',
    'RAM': 'LAR',
    'MIA': 'MIA',
    'MIN': 'MIN',
    'NEP': 'NWE',
    'NOS': 'NOR',
    'NYG': 'NYG',
    'NYJ': 'NYJ',
    'OAK': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SFO': 'SFO',
    'TBB': 'TAM',
    'TEN': 'TEN',
    'WAS': 'WAS'
}

player_df['team_y'] = player_df['team_y'].map(adp_to_player_teams)

# update old team names to LA team names
la_update = {
    'STL': 'LAR',
    'SDG': 'LAC'
}

la_teams = player_df[(player_df.team_x == 'SDG') | (player_df.team_x == 'STL')]
la_teams['team_x'] = la_teams.team_x.map(la_update)
player_df.update(la_teams)

# fill in null teams with current team
player_df['team_y'] = player_df.team_y.fillna(player_df.team_x)

# create flag if player switched teams
player_df['qb_new_team'] = player_df['team_x'] != player_df['team_y']
player_df['qb_new_team'] = player_df.qb_new_team.map({True: 1, False: 0})

# keep current team 
player_df = player_df.drop('team_y', axis=1)
player_df = player_df.rename(columns = {'team_x': 'team'})

# fill null adp values
player_df['qb_avg_pick'] = player_df.qb_avg_pick.fillna(200)

# select columns
player_df = player_df[['player', 'team', 'year', 'qb_age', 'qb_games_started', 'qb_att', 'qb_rating', 
                       'qb_complete', 'qb_complete_pct', 'qb_yds', 'qb_yd_per_game', 'qb_tds', 
                       'qb_td_pct', 'int', 'int_pct', 'qb_long', 'yd_per_att', 'yd_per_comp', 'net_yd_per_att', 
                       'adj_net_yd_per_att', 'adj_yd_per_att', 'qb_sacks', 'sack_pct', 'qb_avg_pick']]

player_df.iloc[:, 2:] = player_df.iloc[:, 2:].astype('float')

# Define a lambda function to compute the weighted mean:
weighted_mean = lambda x: np.average(x, weights=player_df.loc[x.index, "qb_att"])

# Define a dictionary with the functions to apply for a given column:
functions = {
    'qb_age': weighted_mean,
    'qb_games_started': weighted_mean,
    'qb_att': ['sum'],
    'qb_rating': weighted_mean,
    'qb_complete': ['sum'],
    'qb_complete_pct': weighted_mean,
    'qb_yds': ['sum'],
    'qb_yd_per_game': weighted_mean,
    'qb_tds': ['sum'],
    'qb_td_pct': weighted_mean,
    'int': ['sum'],
    'int_pct': weighted_mean,
    'qb_long': weighted_mean,
    'yd_per_att': weighted_mean,
    'yd_per_comp': weighted_mean,
    'net_yd_per_att': weighted_mean,
    'adj_net_yd_per_att': weighted_mean,
    'adj_yd_per_att': weighted_mean,
    'net_yd_per_att': weighted_mean,
    'qb_sacks': ['sum'],
    'sack_pct': weighted_mean,
    'qb_avg_pick': weighted_mean
}

# group based on agg functions on the team and year levels for attachment to WR df
grouped_df = player_df.drop(['player'], axis=1)
grouped_df = grouped_df.groupby(['team', 'year'], group_keys=False).agg(functions).reset_index()
grouped_df.columns = grouped_df.columns.droplevel(1)
grouped_df = grouped_df.rename(columns={'qb_att': 'att'})
# -

# append_to_db(grouped_df, db_name=db_name, table_name='QB_PosPred', if_exist='append')

# # Coaching Data

# +
all_coaches = pd.DataFrame()
cols = ['coach', 'team', 'season_games', 'season_wins', 'season_losses', 'season_ties', 'games_w_team',
        'wins_w_team', 'losses_w_team', 'ties_w_team', 'career_games', 'career_wins', 'careers_losses',
        'career_ties', 'playoff_games', 'playoff_wins', 'playoff_losses',
        'playoff_games_w_team', 'playoff_wins_w_team', 'playoff_losses_w_team',
        'playoff_career_games', 'playoff_career_wins', 'playoff_career_losses', 'remark']

for i in range(1999, 2019):
    data = pd.read_html('https://www.pro-football-reference.com/years/{}/coaches.htm'.format(i))
    df = data[0].T.reset_index(drop=True).T
    df.columns = cols
    df['year'] = i
    
    all_coaches = pd.concat([all_coaches, df], axis=0)
    
all_coaches = all_coaches.reset_index(drop=True)
# -

all_coaches.head()

# # Draft Data by Team

# ## Pull Data From Pro Football Reference

# pulling in the draft value for each pick
draft_values = pd.read_html('https://www.pro-football-reference.com/draft/draft_trade_value.htm')[0]
draft_values = draft_values[['Round', 'Pick', 'Value']]
draft_values = draft_values.dropna()
draft_values = draft_values[draft_values.Round != 'Round'].reset_index(drop=True)
draft_values = convert_to_float(draft_values)

append_to_db(draft_values, table_name='Draft_Values')

# +
# create full positional list to loop through
draft_pos = pd.DataFrame()
for y in range(1985, 1995):
    
    # scrape in the results for each position
    DRAFT_URL = f'https://www.pro-football-reference.com/play-index/draft-finder.cgi?request=1&year_min={y}&year_max={y}&pick_type=overall&pos%5B%5D=qb&pos%5B%5D=rb&pos%5B%5D=wr&pos%5B%5D=te&pos%5B%5D=e&pos%5B%5D=t&pos%5B%5D=g&pos%5B%5D=c&pos%5B%5D=ol&pos%5B%5D=dt&pos%5B%5D=de&pos%5B%5D=dl&pos%5B%5D=ilb&pos%5B%5D=olb&pos%5B%5D=lb&pos%5B%5D=cb&pos%5B%5D=s&pos%5B%5D=db&pos%5B%5D=k&pos%5B%5D=p&conference=any&show=all&order_by=default'
    d = pd.read_html(DRAFT_URL)[0]
    
    # pull out the column names from multi column index
    good_cols = [c[1] for c in d.columns]
    d = d.T.reset_index(drop=True).T
    d.columns = good_cols

    # grab relevant columns and rename
    d = d[['Year', 'Rnd', 'Pick', 'Player', 'Pos', 'Tm', 'College/Univ']]
    d.columns = ['year', 'Round', 'Pick', 'player', 'pos', 'team', 'college']
    
    # concat current results to all results
    draft_pos = pd.concat([draft_pos, d], axis=0)
    
# ensure all positions are upper cased
draft_pos.pos = draft_pos.pos.apply(lambda x: x.upper())    
    
# drop duplicates if guy is in multiple positional pulls    
draft_pos = draft_pos.drop_duplicates()

# remove crap header rows and convert to float
draft_pos = draft_pos[draft_pos.year !='Year'].reset_index(drop=True)
draft_pos = convert_to_float(draft_pos)

# update the team names
draft_pos.loc[draft_pos.team == 'STL', 'team'] = 'LAR'
draft_pos.loc[draft_pos.team == 'SDG', 'team'] = 'LAC'
draft_pos.loc[draft_pos.team == 'LVR', 'team'] = 'OAK'
# -

append_to_db(draft_pos, table_name='Draft_Positions', if_exist='append')

# ## Roll up to Team Level

# +
# select all data from draft positions
draft_pos = pd.read_sql_query('''SELECT * FROM Draft_Positions''', conn)

# if a position is on defense then assign Def tag
check_d = ['DE', 'DT', 'LB', 'DB', 'NT', 'DL', 'OLB', 'CB', 'S', 'ILB', '']
draft_pos.loc[draft_pos.pos.isin(check_d), 'pos'] = 'Def'

# if a position is on oline then assign OL tag
check_ol = ['T', 'G', 'C', 'FB', 'OL', 'OT']
draft_pos.loc[draft_pos.pos.isin(check_ol), 'pos'] = 'OL'

# if a position is on ST then assign ST tag
check_st = ['P', 'K', 'LS']
draft_pos.loc[draft_pos.pos.isin(check_st), 'pos'] = 'ST'

# pull in the values for each draft pick
draft_values = pd.read_sql_query('''SELECT * FROM Draft_Values''', conn)
draft = pd.merge(draft_pos, draft_values, on=['Pick'], how='left').fillna(1)

# calculate the max, sum, and count of values
total_value = draft.groupby(['team', 'year', 'pos']).agg({'Value': 'sum'}).reset_index().rename(columns={'Value': 'total_draft_value'})
max_value = draft.groupby(['team', 'year', 'pos']).agg({'Value': 'max'}).reset_index().rename(columns={'Value': 'max_draft_value'})
value_cnts = draft.groupby(['team', 'year', 'pos']).agg({'Value': 'count'}).reset_index().rename(columns={'Value': 'count_picks'})

# join various value metrics together
team_value = pd.merge(total_value, max_value, on=['team', 'year', 'pos'])
team_value = pd.merge(team_value, value_cnts, on=['team', 'year', 'pos'])

# pivot tables out to wide format
total_value = pd.pivot_table(team_value, index=['team', 'year'], columns='pos', values='total_draft_value').reset_index().fillna(0)
cols = ['team', 'year']
cols.extend([c + '_draft_value_sum' for c in total_value.columns if c not in ('team', 'year')])
total_value.columns = cols

max_value = pd.pivot_table(team_value, index=['team', 'year'], columns='pos', values='max_draft_value').reset_index().fillna(0)
cols=['team', 'year']
cols.extend([c + '_draft_value_max' for c in max_value.columns if c not in ('team', 'year')])
max_value.columns = cols

value_cnts = pd.pivot_table(team_value, index=['team', 'year'], columns='pos', values='count_picks').reset_index().fillna(0)
cols=['team', 'year']
cols.extend([c + '_draft_count_picks' for c in value_cnts.columns if c not in ('team', 'year')])
value_cnts.columns = cols

# join pivoted values back together
team_values = pd.merge(total_value, max_value, on=['team', 'year'])
team_values = pd.merge(team_values, value_cnts, on=['team', 'year'])
team_values.year = team_values.year - 1
# -

append_to_db(team_values, table_name='Team_Drafts', if_exist='replace')


