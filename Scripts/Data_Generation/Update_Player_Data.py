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
import sqlite3

# last year's statistics and adp to pull and append to database
year = 2019

# update True means delete data associated with current year and re-write (e.g. update ADP)
update = True

# name of database to append new data to
db_name = 'Season_Stats.sqlite3'

conn = sqlite3.connect('/Users/Mark/Documents/GitHub/Fantasy_Football/Data/Season_Stats.sqlite3')
# -

# # Load Packages and Functions

import pandas as pd
import os
from data_functions import *
pd.options.mode.chained_assignment = None
import numpy as np

# +
#==========
# Clean the ADP data
#==========

'''
Cleaning the ADP data by selecting relevant features, and extracting the name and team
from the combined string column. Note that the year is not shifted back because the 
stats will be used to calculate FP/G for the rookie in that season, but will be removed
prior to training. Thus, the ADP should match the year from the stats.
'''

def clean_adp(data_adp, year):

    #--------
    # Select relevant columns and clean special figures
    #--------

    data_adp['year'] = year

    # set column names to what they are after pulling
    df_adp = data_adp.iloc[:, 1:].rename(columns={
        1: 'Player', 
        2: 'Avg Pick',
        3: 'Min Pick',
        4: 'Max Pick',
        5: '# Drafts Selected In'
    })

    # selecting relevant columns and dropping na
    df_adp = df_adp[['Player', 'year', 'Avg Pick']].dropna()

    # convert year to float and move back one year to match with stats
    df_adp['year'] = df_adp.year.astype('float')

    # selecting team and player name information from combined string
    df_adp['Tm'] = df_adp.Player.apply(team_select)
    df_adp['Player'] = df_adp.Player.apply(name_select)
    df_adp['Player'] = df_adp.Player.apply(name_clean)

    # format and rename columns
    df_adp = df_adp[['Player', 'Tm', 'year', 'Avg Pick']]

    colnames_adp = {
        'Player': 'player',
        'Tm': 'team',
        'year': 'year',
        'Avg Pick': 'avg_pick'
    }

    df_adp = df_adp.rename(columns=colnames_adp)
    
    return df_adp


# +
#==========
# Merging and formatting all player-based data.
#==========

'''
Join the statistical and adp data into a single, merged dataframe. Update the teams
to have a consistent abbreviation for later joining. Also, select only relevant columns, 
as well as convert all numerical features to float.
'''

def merge_stats_adp(df_stats, df_adp):

    # merge adp and player data
    df_merged = pd.merge(df_stats, df_adp, how = 'inner', left_on = ['player', 'year'], right_on = ['player', 'year'])

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

    return df_merged


# -

# # Running Backs

# +
#==========
# Scraping the statistical and ADP data
#==========

'''
Pull in statistical and ADP data for the given years using the custom data_load function.
'''

# pulling rushing statistics
url_rush = 'https://www.pro-football-reference.com/years/' + str(year) + '/rushing.htm'
data_rush = pd.read_html(url_rush)[0]

# pulling receiving statistics
url_rec = 'https://www.pro-football-reference.com/years/' + str(year) + '/receiving.htm'
data_rec = pd.read_html(url_rec)[0]

# pulling historical player adp for runningbacks
url_adp_rush = f'https://api.myfantasyleague.com/{year+1}/reports?R=ADP&POS=RB&ROOKIES=0&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=RECENT'
data_adp_rush = pd.read_html(url_adp_rush)[1]

# pulling historical redzone receiving data
url_rz_rec = 'https://www.pro-football-reference.com/years/' + str(year) + '/redzone-receiving.htm'
data_rz_rec = pd.read_html(url_rz_rec)[0]

# pulling historical redzone rushing data
url_rz_rush = 'https://www.pro-football-reference.com/years/' + str(year) + '/redzone-rushing.htm'
data_rz_rush = pd.read_html(url_rz_rush)[0]

# +
#==========
# Clean the Statistical Data
#==========

'''
Clean the statistical data by selecting the column names, formatting the column names,
and cleaning up any special characters associated with the player column.
'''

# Setting dates
data_rush['year'] = year
data_rec['year'] = year

#--------
# Setting the column names and removing multi-index
#--------

# cleaning up the rush columns
df_rush = data_rush.iloc[:, 1:]
df_rush = df_rush.T.reset_index(drop=True).T

# cleaning up the receiving columns
df_rec = data_rec.iloc[:, 1:].drop('Y/Tgt', axis=1)
df_rec = df_rec.T.reset_index(drop=True).T

# setting the column names for rushing data
colnames_rush = {
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
colnames_rec = {
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
df_rush = df_rush.rename(columns = colnames_rush)
df_rush['player'] = df_rush.player.apply(name_clean)

# cleaning player name and stat categories
df_rec = df_rec.rename(columns = colnames_rec)
df_rec['player'] = df_rec.player.apply(name_clean)
df_rec['catch_pct'] = df_rec.catch_pct.apply(name_clean)

# removing nonsense rows
df_rush = df_rush[df_rush.games != 'G'].reset_index(drop=True)
df_rec = df_rec[df_rec.games != 'G'].reset_index(drop=True)

# convert data to numeric
for df in [df_rush, df_rec]:
    for col in df.columns:
        try:
            df[col] = df[col].astype('float')
        except:
            pass

#--------
# Merging rushing and receiving stats, as well as adding new stats
#--------

# merge together rushing and receiving stats, while removing duplicates
df_rb = pd.merge(df_rush, df_rec.drop(['team', 'age', 'pos', 'games', 'games_started', 'fmb'], axis=1), 
                 how='inner', left_on=['player', 'year'], right_on=['player', 'year'])

# add in new metrics based on rushing and receiving totals
df_rb['total_touches'] = df_rb.rush_att + df_rb.receptions
df_rb['total_td'] = df_rb.rush_td + df_rb.rec_td
df_rb['td_per_game'] = df_rb.total_td / df_rb.games
df_rb['total_yds'] = df_rb.rush_yds + df_rb.rec_yds
df_rb['total_yds_per_game'] = df_rb.total_yds / df_rb.games
df_rb['yds_per_touch'] = df_rb.total_yds / df_rb.total_touches
df_rb['rush_att_per_game'] = df_rb.rush_att / df_rb.games
# -

df_rb

# +
#==========
# Clean the ADP data
#==========

df_adp_rush = clean_adp(data_adp_rush, year)

# +
conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/' + db_name)
all_rb = pd.read_sql_query("select * from rb_stats", con=conn)

leveon = all_rb[all_rb.player=="Le'Veon Bell"].mean()*0.9
leveon = leveon[df_rb.columns]
leveon.player="Le'Veon Bell"
leveon.team = 'NYJ'
leveon.age = 27
leveon.pos = 'RB'
leveon.games = 13
leveon.games_started = 13
leveon.year = 2018

jerick = all_rb[(all_rb.player=="Jerick McKinnon") | (all_rb.player=='Matt Breida')].mean()
jerick  = jerick[df_rb.columns]
jerick.player="Jerick McKinnon"
jerick.team = 'SFO'
jerick.age = 27
jerick.pos = 'RB'
jerick.games = 12
jerick.games_started = 12
jerick.year = 2018

guice = all_rb[(all_rb.age < np.log(22))].mean()
guice  = guice[df_rb.columns]
guice.player="Derrius Guice"
guice.team = 'WAS'
guice.age = 22
guice.pos = 'RB'
guice.games = 14
guice.games_started = 6
guice.year = 2018

df_rb = pd.concat([df_rb, pd.DataFrame(leveon).T, pd.DataFrame(jerick).T, pd.DataFrame(guice).T], axis=0).reset_index(drop=True)
# -

df_adp_rush

# +
#==========
# Merging and formatting all player-based data.
#==========

df_rb = merge_stats_adp(df_rb, df_adp_rush)

# +
#==========
# Arranging statistical and ADP, sosorting
#==========

'''
Select and order relevant columns, followed by any remaining cleaning up of stats
and converting all numerical stats to float
'''

df_rb['pos'] = 'RB'

# rearrange columns
df_rb = df_rb[['player', 'pos', 'team', 'year', 'age', 'avg_pick',
               'new_team', 'rush_att', 'rush_yds', 'rush_yd_per_att', 'rush_att_per_game', 'rush_yd_per_game',
               'rush_td', 'tgt', 'receptions', 'rec_yds', 'yd_per_rec', 'rec_td',
               'long_rec', 'long_rush', 'rec_per_game', 'rec_yd_per_game', 'catch_pct', 'total_yds',
               'total_td', 'total_touches', 'td_per_game', 'yds_per_touch', 'fmb', 'games', 'games_started']]

# +
#==========
# Formatting the Rushing and Receiving Red Zone Stats
#==========

# removing multi-index column headings
rz_rec = data_rz_rec.T.reset_index(drop=True).T.iloc[:, :]
rz_rush = data_rz_rush.T.reset_index(drop=True).T.iloc[:, :]

# receiving column names
col_names_rec = {
    0: 'player', 
    1: 'team',
    2: 'rz_20_tgt',
    3: 'rz_20_receptions',
    4: 'rz_20_catch_pct',
    5: 'rz_20_rec_yds',
    6: 'rz_20_rec_tds',
    7: 'rz_20_tgt_pct',
    8: 'rz_10_tgt',
    9: 'rz_10_receptions',
    10: 'rz_10_catch_pct',
    11: 'rz_10_rec_yds',
    12: 'rz_10_rec_tds',
    13: 'rz_10_tgt_pct', 
    14: 'link',
    15: 'year'
}

# rushing column names
col_names_rush = {
    0: 'player', 
    1: 'team',
    2: 'rz_20_rush_att',
    3: 'rz_20_rush_yds',
    4: 'rz_20_rush_td',
    5: 'rz_20_rush_pct',
    6: 'rz_10_rush_att',
    7: 'rz_10_rush_yds',
    8: 'rz_10_rush_td',
    9: 'rz_10_rush_pct',
    10: 'rz_5_rush_att',
    11: 'rz_5_rush_yds',
    12: 'rz_5_rush_td',
    13: 'rz_5_rush_pct',
    14: 'link',
    15: 'year'
}

# rename all columns
rz_rec = rz_rec.rename(columns=col_names_rec)
rz_rush = rz_rush.rename(columns=col_names_rush)

# remove percent signs from columns
for col in ['rz_20_catch_pct', 'rz_20_tgt_pct', 'rz_10_catch_pct', 'rz_10_tgt_pct']:
    rz_rec[col] = rz_rec[col].apply(name_clean)
    
# remove percent signs from columns
for col in ['rz_20_rush_pct', 'rz_10_rush_pct', 'rz_5_rush_pct']:
    rz_rush[col] = rz_rush[col].apply(name_clean)
    
# add year to the data
rz_rush['year'] = year
rz_rec ['year'] = year

# drop team prior to merging
rz_rush = rz_rush.drop(['team', 'link'], axis=1)
rz_rec = rz_rec.drop(['team', 'link'], axis=1)

# remove percent signs from columns
for df in [rz_rush, rz_rec]:
    for col in df.columns:
        try:
            df[col] = df[col].astype('float')
        except:
            pass

# +
cols = [c for c in all_rb.columns if c.startswith('rz')]
cols.append('player')
rz_fill = all_rb[cols]

leveon = rz_fill[rz_fill.player =="Le'Veon Bell"].mean()*0.8
leveon_rush = leveon[rz_rush.columns]
leveon_rec = leveon[rz_rec.columns]
leveon_rush.player = "Le'Veon Bell"
leveon_rush.year = 2018
leveon_rec.player = "Le'Veon Bell"
leveon_rec.year = 2018

jerick = rz_fill[(rz_fill.player=="Jerick McKinnon") | (rz_fill.player=='Matt Breida')].mean()
jerick_rush = jerick[rz_rush.columns]
jerick_rec = jerick[rz_rec.columns]
jerick_rush.player = "Jerick McKinnon"
jerick_rush.year = 2018
jerick_rec.player = "Jerick McKinnon"
jerick_rec.year = 2018

guice = rz_fill.mean()
guice_rush = guice[rz_rush.columns]
guice_rec = guice[rz_rec.columns]
guice_rush.player = "Derrius Guice"
guice_rush.year = 2018
guice_rec.player = "Derrius Guice"
guice_rec.year = 2018

rz_rush = pd.concat([rz_rush, pd.DataFrame(leveon_rush).T, 
                     pd.DataFrame(jerick_rush).T, pd.DataFrame(guice_rush).T], axis=0)
rz_rec = pd.concat([rz_rec, pd.DataFrame(leveon_rec).T, 
                    pd.DataFrame(jerick_rec).T, pd.DataFrame(guice_rec).T], axis=0)

# +
#==========
# Final preparation of data prior to uploading
#==========

# merge the red zone rushing stats with the player dataframe
df_rb = pd.merge(df_rb, rz_rush, how='left', left_on=['player', 'year'], right_on=['player', 'year'])

# merge the red zone receiving stats with the player dataframe
df_rb = pd.merge(df_rb, rz_rec, how='left', left_on=['player', 'year'], right_on=['player', 'year'])

# set everything to float
for col in df_rb.columns:
    try:
        df_rb[col] = df_rb[col].astype('float')
    except:
        pass
    
# fill nulls with zero--all should be RZ stats where they didn't accrue on the left join
df_rb = df_rb.fillna(0)

# log avg_pick and age
df_rb.loc[:, 'avg_pick'] = np.log(df_rb.avg_pick)
df_rb.loc[:, 'age'] = np.log(df_rb.age)

# sort values and reset index
df_rb = df_rb.sort_values(by=['year', 'avg_pick'], ascending=[False, True]).reset_index(drop=True)
# -

# conn.cursor().execute('''delete from rb_stats where year={}'''.format(year))
# conn.commit()
# append_to_db(df_rb, db_name='Season_Stats.sqlite3', table_name='RB_Stats', if_exist='append')

# # Wide Receivers

# +
#==========
# Scraping the statistical and ADP data
#==========

'''
Pull in statistical and ADP data for the given years using the custom data_load function.
'''

# pulling historical player adp
url_adp_rec = 'http://www03.myfantasyleague.com/' + str(year+1) + '/adp?COUNT=100&POS=WR&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=0&IS_MOCK=-1&TIME='
data_adp_rec = pd.read_html(url_adp_rec)[1]

# +
#==========
# Clean the ADP data
#==========

df_adp_rec = clean_adp(data_adp_rec, year)

# +
#==========
# Merging and formatting all player-based data.
#==========

df_wr = merge_stats_adp(df_rec, df_adp_rec)

# +
#==========
# Arranging statistical and ADP columns prior to merging
#==========

'''
Select and order relevant columns, followed by any remaining cleaning up of stats
and converting all numerical stats to float
'''

df_wr['pos'] = 'WR'

# create new features
df_wr['yd_per_tgt'] = df_wr.rec_yds / df_wr.tgt
df_wr['td_per_rec'] = df_wr.rec_td / df_wr.receptions
df_wr['tgt_per_game'] = df_wr.tgt / df_wr.games
df_wr['td_per_game'] = df_wr.rec_td / df_wr.games

# rearrange columns
df_wr = df_wr[['player', 'pos', 'team', 'year', 'age', 'avg_pick',
               'new_team', 'tgt', 'receptions', 'rec_yds', 'rec_td', 'td_per_rec',
               'catch_pct', 'games', 'games_started', 'long_rec', 'rec_per_game',
               'rec_yd_per_game', 'tgt_per_game', 'td_per_game', 'yd_per_rec', 
               'yd_per_tgt', 'fmb']]

# merge the red zone receiving stats with the player dataframe
df_wr = pd.merge(df_wr, rz_rec, how='left', left_on=['player', 'year'], right_on=['player', 'year'])

# set everything to float
for col in df_wr.columns:
    try:
        df_wr[col] = df_wr[col].astype('float')
    except:
        pass
    
# fill nulls with zero--all should be RZ stats where they didn't accrue on the left join
df_wr = df_wr.fillna(0)

# log avg_pick and age
df_wr.loc[:, 'avg_pick'] = np.log(df_wr.avg_pick)
df_wr.loc[:, 'age'] = np.log(df_wr.age)

# sort values and reset index
df_wr = df_wr.sort_values(by=['year', 'avg_pick'], ascending=[False, True]).reset_index(drop=True)
# -

# conn.cursor().execute('''delete from wr_stats where year={}'''.format(year))
# conn.commit()
# append_to_db(df_wr, db_name='Season_Stats.sqlite3', table_name='WR_Stats', if_exist='append')

# # Update QB

# +
#===========
# Scraping Statistical and ADP Data
#===========

# pulling passing statistics
url_qb = 'https://www.pro-football-reference.com/years/' + str(year) + '/passing.htm'
data_qb = pd.read_html(url_qb)[0]

# pulling historical player adp
url_adp_qb = 'http://www03.myfantasyleague.com/' + str(year+1) + '/adp?COUNT=100&POS=QB&ROOKIES=0&INJURED=1&CUTOFF=1&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=-1&IS_MOCK=-1&TIME='
data_adp_qb = pd.read_html(url_adp_qb)[1]

# pulling historical redzone passing data
url_rz_pass = 'https://www.pro-football-reference.com/years/' + str(year) + '/redzone-passing.htm'
data_rz_pass = pd.read_html(url_rz_pass)[0]

# +
#--------
# Cleaning Player Statistical Data
#--------

df_qb = data_qb
df_qb['year'] = year

colnames_pass = {
    'Player': 'player',
    'Tm': 'team',
    'Pos': 'pos',
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

# rename columns
df_qb = df_qb.rename(columns=colnames_pass)

# cleaning player name and stat categories
df_qb['player'] = df_qb.player.apply(name_clean)

# fill missing values with zero--almost all are game winning drives
df_qb = df_qb.fillna(0)

# +
#==========
# Clean the ADP data
#==========

df_adp_qb = clean_adp(data_adp_qb, year)
df_adp_qb = df_adp_qb.rename(columns={'avg_pick': 'qb_avg_pick'})

# +
#==========
# Merge the Stats and ADP Data
#==========

df_qb = merge_stats_adp(df_qb, df_adp_qb)

# +
#==========
# Merging and Formatting All Player Data
#==========

for col in df_qb.columns:
    try:
        df_qb[col] = df_qb[col].astype('float')
    except:
        pass

# filter players who threw more than 50 passes in season
df_qb = df_qb[df_qb.qb_att > 25].reset_index(drop=True)

# set position
df_qb['pos'] = 'QB'

# select columns
df_qb = df_qb[['player', 'pos', 'team', 'year', 'qb_avg_pick', 'qb_age', 'qb_games', 'qb_games_started', 'qb_att', 
               'qb_rating', 'qbr', 'qb_complete', 'qb_complete_pct', 'qb_yds', 'qb_yd_per_game', 'qb_tds', 
               'qb_td_pct', 'int', 'int_pct', 'qb_long', 'yd_per_att', 'yd_per_comp', 'net_yd_per_att', 
               'adj_net_yd_per_att', 'adj_yd_per_att', 'qb_sacks', 'sack_pct', 
               'fourth_qt_comeback', 'game_winning_drives']]

# merge rushing stats
df_qb = pd.merge(df_qb, df_rush.drop(['team', 'age', 'pos', 'games', 'games_started'], axis=1), 
                 how='inner', left_on=['player', 'year'], right_on=['player', 'year'])

# +
#==========
# Merging and Formatting RZ data
#==========

df_rz_pass = data_rz_pass.T.reset_index(drop=True).T.iloc[:, :]

df_rz_pass['year'] = year

rz_pass_cols = {
    0: 'player',
    1: 'team',
    2: 'rz_20_pass_complete',
    3: 'rz_20_pass_att',
    4: 'rz_20_complete_pct',
    5: 'rz_20_pass_yds',
    6: 'rz_20_pass_td',
    7: 'rz_20_int',
    8: 'rz_10_pass_complete',
    9: 'rz_10_pass_att',
    10: 'rz_10_complete_pct',
    11: 'rz_10_pass_yds',
    12: 'rz_10_pass_td',
    13: 'rz_10_int',
    14: 'link',
    15: 'year'
}

df_rz_pass = df_rz_pass.rename(columns=rz_pass_cols)

df_rz_pass = df_rz_pass.drop(['team', 'link'], axis=1)
df_rz_pass = df_rz_pass[df_rz_pass.player != 'Player']

# remove percent signs from columns
for col in ['rz_20_complete_pct', 'rz_10_complete_pct']:
    df_rz_pass[col] = df_rz_pass[col].apply(name_clean)
    
# remove percent signs from columns
for col in df_rz_pass.columns:
    try:
        df_rz_pass[col] = df_rz_pass[col].astype('float')
    except:
        pass
    
df_rz_pass = df_rz_pass.fillna(0)

# +
# merge QB passing RZ stats
df_qb = pd.merge(df_qb, df_rz_pass, how='inner', 
                 left_on=['player', 'year'], right_on=['player', 'year'])

# merge QB rushing RZ stats
df_qb = pd.merge(df_qb, rz_rush, how='left', 
                 left_on=['player', 'year'], right_on=['player', 'year'])

# fill NA with zero (e.g. no RZ rush yds)
df_qb = df_qb.fillna(0)

# ensure all new columns are numeric
for col in df_qb.columns:
    try:
        df_qb[col] = df_qb[col].astype('float')
    except:
        pass
    
# add new columns, specifically for target metrics
df_qb['pass_td_per_game'] = df_qb.qb_tds / df_qb.qb_games
df_qb['int_per_game'] = df_qb.int / df_qb.qb_games
df_qb['sacks_per_game'] = df_qb.qb_sacks / df_qb.qb_games
df_qb['rush_td_per_game'] = df_qb.rush_td / df_qb.qb_games

# sort columns
df_qb = df_qb.sort_values(['year', 'qb_avg_pick'], ascending=[False, True]).reset_index(drop=True)
# -

# conn.cursor().execute('''delete from qb_stats where year={}'''.format(year))
# conn.commit()
# append_to_db(df_qb, db_name='Season_Stats.sqlite3', table_name='QB_Stats', if_exist='append')

# # Update TE

# pull TE ADP
url_adp_te = 'http://www03.myfantasyleague.com/' + str(year+1) + '/adp?COUNT=64&POS=TE&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=0&IS_MOCK=-1&TIME='
data_adp_te = pd.read_html(url_adp_te)[1]

# +
#==========
# Clean the ADP data
#==========

df_adp_te = clean_adp(data_adp_te, year)

# +
hunter = df_rec[df_rec.pos == 'TE'].reset_index(drop=True).iloc[3:8].mean()
hunter['player'] = 'Hunter Henry'
hunter['team'] = 'LAC'
hunter['age'] = 24
hunter['pos'] = 'TE'
hunter = hunter[df_rec.columns]
df_rec = pd.concat([df_rec, pd.DataFrame(hunter).T], axis=0)

hunter_rz = rz_rec[rz_rec.player.isin(df_rec[df_rec.pos == 'TE'].player[3:8])].reset_index(drop=True)
for col in hunter_rz.columns:
    try:
        hunter_rz[col] = hunter_rz[col].astype('float')
    except:
        pass
    
hunter_rz = hunter_rz.mean()
hunter_rz['player'] = 'Hunter Henry'
hunter_rz = hunter_rz[rz_rec.columns]
rz_rec = pd.concat([rz_rec, pd.DataFrame(hunter_rz).T], axis=0)

# +
#==========
# Merging and formatting all player-based data.
#==========

df_te = merge_stats_adp(df_rec, df_adp_te)

# +
#==========
# Arranging statistical and ADP columns prior to merging
#==========

'''
Select and order relevant columns, followed by any remaining cleaning up of stats
and converting all numerical stats to float
'''

df_te['pos'] = 'TE'

# create new features
df_te['yd_per_tgt'] = df_te.rec_yds / df_te.tgt
df_te['td_per_rec'] = df_te.rec_td / df_te.receptions
df_te['tgt_per_game'] = df_te.tgt / df_te.games
df_te['td_per_game'] = df_te.rec_td / df_te.games

# rearrange columns
df_te = df_te[['player', 'pos', 'team', 'year', 'age', 'avg_pick',
               'new_team', 'tgt', 'receptions', 'rec_yds', 'rec_td', 'td_per_rec',
               'catch_pct', 'games', 'games_started', 'long_rec', 'rec_per_game',
               'rec_yd_per_game', 'tgt_per_game', 'td_per_game', 'yd_per_rec', 
               'yd_per_tgt', 'fmb']]

# merge the red zone receiving stats with the player dataframe
df_te = pd.merge(df_te, rz_rec, how='left', left_on=['player', 'year'], right_on=['player', 'year'])

# set everything to float
for col in df_te.columns:
    try:
        df_te[col] = df_te[col].astype('float')
    except:
        pass
    
# fill nulls with zero--all should be RZ stats where they didn't accrue on the left join
df_te = df_te.fillna(0)

# log avg_pick and age
df_te.loc[:, 'avg_pick'] = np.log(df_te.avg_pick)
df_te.loc[:, 'age'] = np.log(df_te.age)

# sort values and reset index
df_te = df_te.sort_values(by=['year', 'avg_pick'], ascending=[False, True]).reset_index(drop=True)
# -

conn.cursor().execute('''delete from te_stats where year={}'''.format(year))
conn.commit()
append_to_db(df_te, db_name='Season_Stats.sqlite3', table_name='TE_Stats', if_exist='append')
