
#%%

import sqlite3
import os

# last year's statistics and adp to pull and append to database
year = 2020

# update True means delete data associated with current year and re-write (e.g. update ADP)
update = True

# name of database to append new data to
db_name = 'Season_Stats'
db_path = f'/Users/{os.getlogin()}/Documents/GitHub/Fantasy_Football/Data/Databases/'
conn = sqlite3.connect(db_path + db_name + '.sqlite3')

# # Load Packages and Functions
import pandas as pd
import os
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np


#%%

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
    df_adp = df_adp[df_adp.Player != '1 Page:']
    
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
        'OAK': 'LVR',
        'LVR': 'LVR',
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


#%%

# # Running Backs

# +
#==========
# Scraping the statistical and ADP data
#==========

'''
Pull in statistical and ADP data for the given years using the custom data_load function.
'''

# pulling rushing statistics
url_rush = f'https://www.pro-football-reference.com/years/{year}/rushing.htm'
data_rush = pd.read_html(url_rush)[0]

# pulling receiving statistics
url_rec = f'https://www.pro-football-reference.com/years/{year}/receiving.htm'
data_rec = pd.read_html(url_rec)[0]

# pulling historical player adp for runningbacks
url_adp_rush = f'https://api.myfantasyleague.com/{year+1}/reports?R=ADP&POS=RB&ROOKIES=0&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=RECENT'
data_adp_rush = pd.read_html(url_adp_rush)[1]

# pulling historical redzone receiving data
url_rz_rec = f'https://www.pro-football-reference.com/years/{year}/redzone-receiving.htm'
data_rz_rec = pd.read_html(url_rz_rec)[0]

# pulling historical redzone rushing data
url_rz_rush = f'https://www.pro-football-reference.com/years/{year}/redzone-rushing.htm'
data_rz_rush = pd.read_html(url_rz_rush)[0]

#%%

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
good_cols = [c[1] for c in data_rush.columns if c[1] != '']
good_cols.append('year')
df_rush = data_rush.T.reset_index(drop=True).T
df_rush.columns = good_cols
df_rush = df_rush.drop('Rk', axis=1)

# cleaning up the receiving columns
df_rec = data_rec.copy()
df_rec = df_rec.drop('Rk', axis=1)

# setting the column names for rushing data
colnames_rush = {
    'Player': 'player', 
    'Tm': 'team', 
    'Age': 'age',
    'Pos': 'pos',
    'G': 'games',
    'GS': 'games_started',
    'Att': 'rush_att',
    'Yds': 'rush_yds',
    'TD': 'rush_td',
    '1D': 'first_downs',
    'Lng': 'long_rush',
    'Y/A': 'rush_yd_per_att',
    'Y/G': 'rush_yd_per_game',
    'Fmb': 'fmb',
    'year': 'year'
}

# setting the column names for receiving data
colnames_rec = {
    'Player': 'player',
    'Tm': 'team', 
    'Age': 'age',
    'Pos': 'pos',
    'G': 'games',
    'GS': 'games_started',
    'Tgt': 'tgt',
    'Rec': 'receptions',
    'Ctch%': 'catch_pct',
    'Yds': 'rec_yds', 
    'Y/R': 'yd_per_rec',
    'TD': 'rec_td',
    '1D': 'first_downs',
    'Lng': 'long_rec',
    'Y/Tgt': 'yards_per_tgt',
    'R/G': 'rec_per_game',
    'Y/G': 'rec_yd_per_game',
    'Fmb': 'fmb',
    'year': 'year'
}

# cleaning player name and stat categories
df_rush = df_rush.rename(columns = colnames_rush)
df_rush['player'] = df_rush.player.apply(name_clean)

# cleaning player name and stat categories
df_rec = df_rec.rename(columns = colnames_rec)
df_rec['player'] = df_rec.player.apply(name_clean)
df_rec['catch_pct'] = df_rec.catch_pct.apply(num_clean)

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
df_rb['total_yd_per_game'] = df_rb.total_yds / df_rb.games
df_rb['yds_per_touch'] = df_rb.total_yds / df_rb.total_touches
df_rb['rush_att_per_game'] = df_rb.rush_att / df_rb.games

#%%
#==========
# Find missing players
#==========

df_adp_rush = clean_adp(data_adp_rush, year)
missing = pd.merge(df_adp_rush[['player', 'avg_pick']], df_rb[['player', 'games']], on=['player'], how='left')
missing[(missing.games.isnull()) | (missing.games < 8)]

#%%

all_rb = pd.read_sql_query("select * from rb_stats", con=conn)

def add_player(d, df_rb):
    df = all_rb[eval(d['filter_q'])].mean()*d['frac']
    df = df[df_rb.columns]
    for c in ['player', 'team', 'age']:
        df[c] = d[c]
    df.games = round(df.games)
    df.games_started = df.games
    df.year = year
    df['total_yds_per_game'] = df.total_yds / df.games

    df_rb = df_rb[df_rb.player != d['player']]
    df_rb = pd.concat([df_rb, pd.DataFrame(df).T], axis=0).reset_index(drop=True)

    return df_rb

p1 = {
    'filter_q': '(all_rb.player=="Christian McCaffrey") & (all_rb.games > 12)',
    'frac': 0.95,
    'player': "Christian McCaffrey",
    'team': 'CAR',
    'age': 24,
}

p2 = {
    'filter_q': '(all_rb.player=="Saquon Barkley") & (all_rb.games > 12)',
    'frac': 0.95,
    'player': "Saquon Barkley",
    'team': 'NYG',
    'age': 23,
}

p3 = {
    'filter_q': '(all_rb.player=="Joe Mixon") & (all_rb.games > 12)',
    'frac': 0.95,
    'player': "Joe Mixon",
    'team': 'CIN',
    'age': 24,
}

df_rb = add_player(p1, df_rb)
df_rb = add_player(p2, df_rb)
df_rb = add_player(p3, df_rb)


#%%
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
               'total_td', 'total_touches', 'td_per_game', 'total_yd_per_game',
               'yds_per_touch', 'fmb', 'games', 'games_started']]

#%%
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
    rz_rec[col] = rz_rec[col].apply(num_clean)
    
# remove percent signs from columns
for col in ['rz_20_rush_pct', 'rz_10_rush_pct', 'rz_5_rush_pct']:
    rz_rush[col] = rz_rush[col].apply(num_clean)
    
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

#---------------
# Fill in RZ columns
#---------------

cols = [c for c in all_rb.columns if c.startswith('rz')]
cols.append('player')
rz_fill = all_rb[cols]

def add_player_rz(d, rz_rush, rz_rec):

    df = rz_fill[eval(d['filter_q'])].mean()*d['frac']
    df_rush = df[rz_rush.columns]
    df_rec = df[rz_rec.columns]
    df_rush.player = d['player']
    df_rush.year = year
    df_rec.player = d['player']
    df_rec.year = year

    rz_rush = rz_rush[rz_rush.player != d['player']]
    rz_rush = pd.concat([rz_rush, pd.DataFrame(df_rush).T], axis=0).reset_index(drop=True)

    rz_rec = rz_rec[rz_rec.player != d['player']]
    rz_rec = pd.concat([rz_rec, pd.DataFrame(df_rush).T], axis=0).reset_index(drop=True)

    return rz_rush, rz_rec

p1 = {
    'filter_q': '(all_rb.player=="Christian McCaffrey") & (all_rb.games > 12)',
    'frac': 0.95,
    'player': "Christian McCaffrey",
}

p2 = {
    'filter_q': '(all_rb.player=="Saquon Barkley") & (all_rb.games > 12)',
    'frac': 0.95,
    'player': "Saquon Barkley"
}

p3 = {
    'filter_q': '(all_rb.player=="Joe Mixon") & (all_rb.games > 12)',
    'frac': 0.95,
    'player': "Joe Mixon"
}

rz_rush, rz_rec = add_player_rz(p1, rz_rush, rz_rec)
rz_rush, rz_rec = add_player_rz(p2, rz_rush, rz_rec)
rz_rush, rz_rec = add_player_rz(p3, rz_rush, rz_rec)

#%%

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

find_missing = pd.merge(df_adp_rush, df_rb, on='player', how='left')
list(find_missing.loc[find_missing.team_y.isnull(), 'player'])

# conn.cursor().execute('''delete from rb_stats where year={}'''.format(year))
# conn.commit()
# append_to_db(df_rb, db_name='Season_Stats', table_name='RB_Stats', if_exist='append')


#%% Wide Receivers

#==========
# Scraping the statistical and ADP data
#==========

'''
Pull in statistical and ADP data for the given years using the custom data_load function.
'''

# pulling historical player adp
url_adp_rec = f'https://www71.myfantasyleague.com/{year+1}/reports?R=ADP&POS=WR&PERIOD=JUNE&CUTOFF=5&INJURED=1&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PAGE=ALL'
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

find_missing = pd.merge(df_adp_rec, df_wr, on='player', how='left')
list(find_missing.loc[find_missing.team_y.isnull(), 'player'])

# +
all_wr = pd.read_sql_query("select * from wr_stats", con=conn)
aj = all_wr[(all_wr.player=="AJ Green")].mean()*0.85
aj  = aj[df_wr.columns]
aj.player="AJ Green"
aj.team = 'CIN'
aj.age = 32
aj.pos = 'WR'
aj.games = 14
aj.games_started = 14
aj.year = 2019


df_wr = pd.concat([df_wr, pd.DataFrame(aj).T], axis=0)
# -

# conn.cursor().execute('''delete from wr_stats where year={}'''.format(year))
# conn.commit()
# append_to_db(df_wr, db_name='Season_Stats', table_name='WR_Stats', if_exist='append')

# # Update QB

# +
#===========
# Scraping Statistical and ADP Data
#===========

# pulling passing statistics
url_qb = 'https://www.pro-football-reference.com/years/' + str(year) + '/passing.htm'
data_qb = pd.read_html(url_qb)[0]

# pulling historical player adp
url_adp_qb = f'https://www71.myfantasyleague.com/{year+1}/reports?R=ADP&POS=QB&ROOKIES=0&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=JULY'
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
    '1D': 'first_downs',
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
df_qb = df_qb.drop('first_downs', axis=1)

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
df_qb = pd.merge(df_qb, df_rush.drop(['team', 'age', 'pos', 'games', 'games_started', 'first_downs'], axis=1), 
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
    df_rz_pass[col] = df_rz_pass[col].apply(num_clean)
    
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
# append_to_db(df_qb, db_name='Season_Stats', table_name='QB_Stats', if_exist='append')

# # Update TE

# pull TE ADP
url_adp_te = f'https://api.myfantasyleague.com/{year+1}/reports?R=ADP&POS=TE&ROOKIES=0&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=JULY'
data_adp_te = pd.read_html(url_adp_te)[1]

# +
#==========
# Clean the ADP data
#==========

df_adp_te = clean_adp(data_adp_te, year)
# -

# hunter = df_rec[df_rec.pos == 'TE'].reset_index(drop=True).iloc[3:8].mean()
# hunter['player'] = 'Hunter Henry'
# hunter['team'] = 'LAC'
# hunter['age'] = 24
# hunter['pos'] = 'TE'
# hunter = hunter[df_rec.columns]
# df_rec = pd.concat([df_rec, pd.DataFrame(hunter).T], axis=0)
#
# hunter_rz = rz_rec[rz_rec.player.isin(df_rec[df_rec.pos == 'TE'].player[3:8])].reset_index(drop=True)
# for col in hunter_rz.columns:
#     try:
#         hunter_rz[col] = hunter_rz[col].astype('float')
#     except:
#         pass
#     
# hunter_rz = hunter_rz.mean()
# hunter_rz['player'] = 'Hunter Henry'
# hunter_rz = hunter_rz[rz_rec.columns]
# rz_rec = pd.concat([rz_rec, pd.DataFrame(hunter_rz).T], axis=0)

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

# conn.cursor().execute('''delete from te_stats where year={}'''.format(year))
# conn.commit()
# append_to_db(df_te, db_name='Season_Stats', table_name='TE_Stats', if_exist='append')

# # Create Air Yards Table

# +
import requests

air_year = year
df_air_json = requests.get(f'http://api.airyards.com/{air_year}/weeks')

df_air = pd.DataFrame(df_air_json.json())

df_air = df_air[df_air.position.isin(['WR', 'RB', 'TE'])].reset_index(drop=True)
df_air = df_air[['full_name', 'position', 'team', 'week', 'air_yards', 'tar', 'rec', 'rec_yards', 'tm_att', 
                   'team_air', 'aypt', 'racr', 'ms_air_yards', 'target_share', 'wopr', 'yac']]
df_air['year'] = air_year
df_air = df_air.rename(columns={'week': 'games', 'full_name': 'player'})

col_agg = {'air_yards': 'sum',
           'rec_yards': 'sum',
           'rec': 'sum',
           'tar': 'sum',
           'tm_att': 'sum',
           'team_air': 'sum',
           'yac': 'sum',
           'games': 'count'}

df_air_player = df_air.groupby(['player', 'team', 'year', 'position']).agg(col_agg).reset_index()

dup_players = df_air_player.groupby('player').agg({'tar': 'max'}).reset_index()
dup_teams = pd.merge(df_air_player[['player', 'team', 'tar']], dup_players, on=['player', 'tar']).drop('tar', axis=1)

col_agg['games'] = 'sum'
df_air_player = df_air_player.groupby(['player', 'year', 'position']).agg(col_agg).reset_index()
df_air_player = pd.merge(df_air_player, dup_teams, on='player')

df_air_player['ay_per_game'] = df_air_player.air_yards / df_air_player.games
df_air_player['yac_per_game'] = df_air_player.yac / df_air_player.games
df_air_player['racr'] = df_air_player.rec_yards / (df_air_player.air_yards + 1.5)
df_air_player['ay_per_tar'] = df_air_player.air_yards / (df_air_player.tar + 1.5)
df_air_player['ay_per_rec'] = df_air_player.air_yards / (df_air_player.rec + 1.5)
df_air_player['tgt_mkt_share'] = df_air_player.tar / df_air_player.tm_att
df_air_player['ay_converted'] = (df_air_player.rec_yards - df_air_player.yac) / (df_air_player.air_yards+1.5)
df_air_player['yac_per_ay'] = df_air_player.yac / (df_air_player.air_yards+1.5)
df_air_player['air_yd_mkt_share'] = df_air_player.air_yards / df_air_player.team_air
df_air_player['wopr'] = 1.5*df_air_player.tgt_mkt_share + 0.7*df_air_player.air_yd_mkt_share
df_air_player['rec_yds_per_ay'] = df_air_player.rec_yards / (df_air_player.air_yards + 1)
df_air_player['yac_plus_ay'] = df_air_player.yac + df_air_player.air_yards

team_agg = {'tm_att': 'max',
            'team_air': 'max',
            'rec_yards': 'sum',
            'yac': 'sum'}

df_air_team = df_air.groupby(['team', 'games']).agg(team_agg)
df_air_team = df_air_team.groupby('team').agg({'tm_att': 'sum', 'team_air': 'sum', 'rec_yards': 'sum', 'yac': 'sum'})
df_air_team = df_air_team.rename(columns={'yac': 'team_yac'})
df_air_team['tm_air_per_att'] = df_air_team.team_air / df_air_team.tm_att
df_air_team['tm_ay_converted'] = (df_air_team.rec_yards - df_air_team.team_yac) / df_air_team.team_air
df_air_team['tm_rec_yds_per_ay'] = df_air_team.rec_yards / df_air_team.team_air
df_air_team['tm_yac_per_ay'] = df_air_team.team_yac / df_air_team.team_air
df_air_team = df_air_team.drop('rec_yards', axis=1)

df_air_player = df_air_player.drop(['tm_att', 'team_air', 'rec_yards', 'rec'], axis=1)
df_air_player = pd.merge(df_air_player, df_air_team, on=['team'])
df_air_player['tm_ay_per_game'] = df_air_player.team_air / df_air_player.games
df_air_player['total_tgt_mkt_share'] = df_air_player.tar / df_air_player.tm_att
df_air_player['yac_mkt_share'] = df_air_player.yac / df_air_player.team_yac
df_air_player['yac_wopr'] =  1.5*df_air_player.tgt_mkt_share + 0.7*df_air_player.air_yd_mkt_share + df_air_player.yac_mkt_share
df_air_player = df_air_player.drop(['tar', 'games', 'team'], axis=1)
# -

append_to_db(df_air_player, db_name='Season_Stats', table_name='AirYards', if_exist='append')
