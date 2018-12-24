
# coding: utf-8

# # User Inputs

# In[ ]:


# last year's statistics and adp to pull and append to database
year = 2017

# name of database to append new data to
db_name = 'Season_Stats.sqlite3'


# # Load Packages

# In[ ]:


import pandas as pd
import os
import sqlite3
from data_functions import *
pd.options.mode.chained_assignment = None
import numpy as np


# # Running Backs

# In[ ]:


#==========
# Scraping the statistical and ADP data
#==========

'''
Pull in statistical and ADP data for the given years using the custom data_load function.
'''

# pulling rushing statistics
url_player_rush = 'https://www.pro-football-reference.com/years/' + str(year) + '/rushing.htm'
data_player_rush = pd.read_html(url_player_rush)[0]

# pulling receiving statistics
url_player_rec = 'https://www.pro-football-reference.com/years/' + str(year) + '/receiving.htm'
data_player_rec = pd.read_html(url_player_rec)[0]

# pulling historical player adp
url_adp = 'http://www03.myfantasyleague.com/' + str(year+1) + '/adp?COUNT=100&POS=RB&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=0&IS_MOCK=-1&TIME='
data_adp = pd.read_html(url_adp)[1]


# In[ ]:


#==========
# Clean the Statistical Data
#==========

'''
Clean the statistical data by selecting the column names, formatting the column names,
and cleaning up any special characters associated with the player column.
'''

# Setting dates
data_player_rush['year'] = year
data_player_rec['year'] = year

#--------
# Setting the column names and removing multi-index
#--------

# cleaning up the rush columns
df_player_rush = data_player_rush.iloc[:, 1:]
df_player_rush = df_player_rush.T.reset_index(drop=True).T

# cleaning up the receiving columns
df_player_rec = data_player_rec.iloc[:, 1:]
df_player_rec = df_player_rec.T.reset_index(drop=True).T

# setting the column names for rushing data
colnames_player_rush = {
    0: 'player',
    1: 'team',
    2: 'age',
    3: 'pos',
    4: 'games',
    5: 'games_started',
    6: 'rush_att',
    7: 'rush_yds',
    8: 'rush_td',
    9: 'long_rush',
    10: 'rush_yd_per_att',
    11: 'rush_yd_per_game',
    12: 'fmb',
    13: 'year'
}

# setting the column names for receiving data
colnames_player_rec = {
    0: 'player',
    1: 'team',
    2: 'age',
    3: 'pos',
    4: 'games',
    5: 'games_started',
    6: 'tgt',
    7: 'receptions',
    8: 'catch_pct',
    9: 'rec_yds',
    10: 'yd_per_rec',
    11: 'rec_td',
    12: 'long_rec',
    13: 'rec_per_game',
    14: 'rec_yd_per_game',
    15: 'fmb',
    16: 'year'
}

# cleaning player name and stat categories
df_player_rush = df_player_rush.rename(columns = colnames_player_rush)
df_player_rush['player'] = df_player_rush.player.apply(name_clean)

# cleaning player name and stat categories
df_player_rec = df_player_rec.rename(columns = colnames_player_rec)
df_player_rec['player'] = df_player_rec.player.apply(name_clean)
df_player_rec['catch_pct'] = df_player_rec.catch_pct.apply(name_clean)

# removing nonsense rows
df_player_rush = df_player_rush[df_player_rush.games != 'G'].reset_index(drop=True)
df_player_rec = df_player_rec[df_player_rec.games != 'G'].reset_index(drop=True)

# convert data to numeric
for df in [df_player_rush, df_player_rec]:
    for col in df.columns:
        try:
            df[col] = df[col].astype('float')
        except:
            pass

#--------
# Merging rushing and receiving stats, as well as adding new stats
#--------

# merge together rushing and receiving stats, while removing duplicates
df_player = pd.merge(df_player_rush, df_player_rec.drop(['team', 'age', 'pos', 'games', 'games_started', 'fmb'], axis=1),
                     how='inner', left_on=['player', 'year'], right_on=['player', 'year'])

# add in new metrics based on rushing and receiving totals
df_player['total_touches'] = df_player.rush_att + df_player.receptions
df_player['total_td'] = df_player.rush_td + df_player.rec_td
df_player['td_per_game'] = df_player.total_td / df_player.games
df_player['total_yds'] = df_player.rush_yds + df_player.rec_yds
df_player['yds_per_touch'] = df_player.total_yds / df_player.total_touches
df_player['rush_att_per_game'] = df_player.rush_att / df_player.games


# In[ ]:


#==========
# Clean the ADP data
#==========

'''
Cleaning the ADP data by selecting relevant features, and extracting the name and team
from the combined string column. Note that the year is not shifted back because the
stats will be used to calculate FP/G for the rookie in that season, but will be removed
prior to training. Thus, the ADP should match the year from the stats.
'''

#--------
# Select relevant columns and clean special figures
#--------

data_adp['year'] = year

# set column names to what they are after pulling
data_adp = data_adp.iloc[1:, 1:].rename(columns={
    1: 'Player',
    2: 'Avg. Pick',
    3: 'Min. Pick',
    4: 'Max. Pick',
    5: '# Drafts Selected In'
})

# selecting relevant columns and dropping na
df_adp = data_adp[['Player', 'year', 'Avg. Pick']].dropna()

# convert year to float and move back one year to match with stats
df_adp['year'] = df_adp.year.astype('float')

# selecting team and player name information from combined string
df_adp['Tm'] = df_adp.Player.apply(team_select)
df_adp['Player'] = df_adp.Player.apply(name_select)
df_adp['Player'] = df_adp.Player.apply(name_clean)

# format and rename columns
df_adp = df_adp[['Player', 'Tm', 'year', 'Avg. Pick']]

colnames_adp = {
    'Player': 'player',
    'Tm': 'team',
    'year': 'year',
    'Avg. Pick': 'avg_pick'
}

df_adp = df_adp.rename(columns=colnames_adp)

#--------
# Removing duplicate players and updating teams
#--------

# updating FA* to KC for Kareem Hunt
df_adp.loc[(df_adp.player=='Kareem Hunt') & (df_adp.year==2017), 'team'] = 'KCC'


# In[ ]:


#==========
# Merging and formatting all player-based data.
#==========

'''
Join the statistical and adp data into a single, merged dataframe. Update the teams
to have a consistent abbreviation for later joining. Also, select only relevant columns,
as well as convert all numerical features to float.
'''

# merge adp and player data
df_merged = pd.merge(df_player, df_adp, how = 'inner', left_on = ['player', 'year'], right_on = ['player', 'year'])

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

df_merged['team_y'] = df_merged['team_y'].map(adp_to_player_teams)

# update old team names to LA team names
la_update = {
    'STL': 'LAR',
    'SDG': 'LAC'
}

la_teams = df_merged[(df_merged.team_x == 'SDG') | (df_merged.team_x == 'STL')]
la_teams['team_x'] = la_teams.team_x.map(la_update)
df_merged.update(la_teams)

# create flag if player switched teams
df_merged['team_y'] = df_merged.team_y.fillna('FA')
df_merged['new_team'] = df_merged['team_x'] != df_merged['team_y']
df_merged['new_team'] = df_merged.new_team.map({True: 1, False: 0})

# keep current team
df_merged = df_merged.drop('team_x', axis=1)
df_merged = df_merged.rename(columns = {'team_y': 'team'})

df_merged['pos'] = 'RB'


# In[ ]:


#==========
# Arranging statistical and ADP, sosorting
#==========

'''
Select and order relevant columns, followed by any remaining cleaning up of stats
and converting all numerical stats to float
'''

# rearrange columns
df_merged = df_merged[['player', 'pos', 'team', 'year', 'age', 'avg_pick',
                       'new_team', 'rush_att', 'rush_yds', 'rush_yd_per_att', 'rush_att_per_game', 'rush_yd_per_game',
                       'rush_td', 'tgt', 'receptions', 'rec_yds', 'yd_per_rec', 'rec_td',
                       'long_rec', 'long_rush', 'rec_per_game', 'rec_yd_per_game', 'catch_pct', 'total_yds',
                       'total_td', 'total_touches', 'td_per_game', 'yds_per_touch', 'fmb', 'games', 'games_started']]

# ensure all columns are numeric
df_merged[df_merged.columns[3:]] = df_merged[df_merged.columns[3:]].astype('float')

# log avg_pick and age
df_merged.loc[:, 'avg_pick'] = np.log(df_merged.avg_pick)
df_merged.loc[:, 'age'] = np.log(df_merged.age)

# sort values and reset index
df_merged = df_merged.sort_values(by=['year', 'rush_yds'], ascending=[False, False]).reset_index(drop=True)


# append_to_db(df_merged, db_name='Season_Stats.sqlite3', table_name='RB_Stats', if_exist='append')

# # Wide Receivers

# In[ ]:


#==========
# Scraping the statistical and ADP data
#==========

'''
Pull in statistical and ADP data for the given years using the custom data_load function.
'''

# pulling historical player adp
url_adp = 'http://www03.myfantasyleague.com/' + str(year+1) + '/adp?COUNT=100&POS=WR&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=0&IS_MOCK=-1&TIME='
data_adp = pd.read_html(url_adp)[1]


# In[ ]:


#==========
# Clean the ADP data
#==========

'''
Cleaning the ADP data by selecting relevant features, and extracting the name and team
from the combined string column. Note that the year is not shifted back because the
stats will be used to calculate FP/G for the rookie in that season, but will be removed
prior to training. Thus, the ADP should match the year from the stats.
'''

#--------
# Select relevant columns and clean special figures
#--------

data_adp['year'] = year

# set column names to what they are after pulling
data_adp = data_adp.iloc[1:, 1:].rename(columns={
    1: 'Player',
    2: 'Avg. Pick',
    3: 'Min. Pick',
    4: 'Max. Pick',
    5: '# Drafts Selected In'
})

# selecting relevant columns and dropping na
df_adp = data_adp[['Player', 'year', 'Avg. Pick']].dropna()

# convert year to float and move back one year to match with stats
df_adp['year'] = df_adp.year.astype('float')

# selecting team and player name information from combined string
df_adp['Tm'] = df_adp.Player.apply(team_select)
df_adp['Player'] = df_adp.Player.apply(name_select)
df_adp['Player'] = df_adp.Player.apply(name_clean)

# format and rename columns
df_adp = df_adp[['Player', 'Tm', 'year', 'Avg. Pick']]

colnames_adp = {
    'Player': 'player',
    'Tm': 'team',
    'year': 'year',
    'Avg. Pick': 'avg_pick'
}

df_adp = df_adp.rename(columns=colnames_adp)


# In[ ]:


#==========
# Merging and formatting all player-based data.
#==========

'''
Join the statistical and adp data into a single, merged dataframe. Update the teams
to have a consistent abbreviation for later joining. Also, select only relevant columns,
as well as convert all numerical features to float.
'''

# merge adp and player data
df_merged = pd.merge(df_player_rec, df_adp, how = 'inner', left_on = ['player', 'year'], right_on = ['player', 'year'])

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

df_merged['team_y'] = df_merged['team_y'].map(adp_to_player_teams)

# update old team names to LA team names
la_update = {
    'STL': 'LAR',
    'SDG': 'LAC'
}

la_teams = df_merged[(df_merged.team_x == 'SDG') | (df_merged.team_x == 'STL')]
la_teams['team_x'] = la_teams.team_x.map(la_update)
df_merged.update(la_teams)

# create flag if player switched teams
df_merged['team_y'] = df_merged.team_y.fillna('FA')
df_merged['new_team'] = df_merged['team_x'] != df_merged['team_y']
df_merged['new_team'] = df_merged.new_team.map({True: 1, False: 0})

# keep current team
df_merged = df_merged.drop('team_x', axis=1)
df_merged = df_merged.rename(columns = {'team_y': 'team'})

df_merged['pos'] = 'WR'


# In[ ]:


# ==========
# Arranging statistical and ADP columns prior to merging
#==========

'''
Select and order relevant columns, followed by any remaining cleaning up of stats
and converting all numerical stats to float
'''

# create new features
df_merged['yd_per_tgt'] = df_merged.rec_yds / df_merged.tgt
df_merged['td_per_rec'] = df_merged.rec_td / df_merged.receptions
df_merged['tgt_per_game'] = df_merged.tgt / df_merged.games
df_merged['td_per_game'] = df_merged.rec_td / df_merged.games

# rearrange columns
df_merged = df_merged[['player', 'pos', 'team', 'year', 'age', 'avg_pick',
                       'new_team', 'tgt', 'receptions', 'rec_yds', 'rec_td', 'td_per_rec',
                       'catch_pct', 'games', 'games_started', 'long_rec', 'rec_per_game',
                       'rec_yd_per_game', 'tgt_per_game', 'td_per_game', 'yd_per_rec',
                       'yd_per_tgt', 'fmb']]

# make all columns numeric
df_merged.iloc[:, 3:] = df_merged.iloc[:, 3:].astype('float')

# log avg_pick and age
df_merged.loc[:, 'avg_pick'] = np.log(df_merged.avg_pick)
df_merged.loc[:, 'age'] = np.log(df_merged.age)

# sort values and reset index
df_merged = df_merged.sort_values(by=['year', 'rec_yds'], ascending=[False, False]).reset_index(drop=True)


# append_to_db(df_merged, db_name='Season_Stats.sqlite3', table_name='WR_Stats', if_exist='append')
