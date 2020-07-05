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

# # Load Packages

set_year=2019

# +
import pandas as pd
import os
import sqlite3
from data_functions import *
pd.options.mode.chained_assignment = None
import numpy as np
from fractions import Fraction



# set core path
path = '/Users/Mark/Documents/Github/Fantasy_Football'

# set the database name
db_name = 'Season_Stats.sqlite3'

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
        2: 'Avg. Pick',
        3: 'Min. Pick',
        4: 'Max. Pick',
        5: '# Drafts Selected In'
    })

    # selecting relevant columns and dropping na
    df_adp = df_adp[['Player', 'year', 'Avg. Pick']].dropna()

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
    
    return df_adp


# -

# # Running Backs

# +
url_adp_rush = 'http://www03.myfantasyleague.com/{}/adp?COUNT=100&POS=RB&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=0&IS_MOCK=0&TIME='
data_adp_rush = pd.DataFrame()
rb_adp = pd.DataFrame()

for year in range(2004, set_year+1):
    url_year = url_adp_rush.format(str(year))
    f = pd.read_html(url_year, header=0)[1]
    f = f.assign(year=year)
    f = clean_adp(f, year)
    rb_adp = pd.concat([rb_adp, f], axis=0)
    
rb_adp = rb_adp.reset_index(drop=True)

# +
#==========
# Pulling in the Player Profiler statistics
#==========

'''
Pull in the player profiler statistics and clean up any formatting issues. Follow by
left joining the statistics to the existing player dataframe.
'''

full_path = path + '/Data/OtherData/PlayerProfiler/{}/RB/'.format(set_year)

# loop through each file and merge together into single file
data_pp = pd.DataFrame()
for file in os.listdir(full_path):
    f = pd.read_csv(full_path + file)
    
    try:
        data_pp = pd.merge(data_pp, f.drop('Position', axis=1), 
                           how='inner', left_on='Full Name', right_on='Full Name')
    except:
        data_pp = f
        
# convert all dashes to null
data_pp = data_pp.replace("-", float('nan'))

colnames = {
    'Full Name': 'player',
    'Position': 'position',
    'Draft Year': 'draft_year',
    '20-Yard Shuttle': 'shuffle_20_yd',
    'Athleticism Score': 'athlete_score',
    'SPARQ-x': 'sparq',
    '3-Cone Drill': 'three_cone',
    'Bench Press': 'bench_press',
    'Speed Score': 'speed_score',
    '40-Yard Dash': 'forty',
    'Broad Jump': 'broad_jump',
    'Vertical Jump': 'vertical',
    'Burst Score': 'burst_score',
    'Agility Score': 'agility_score',
    'Hand Size': 'hand_size',
    'Age': 'pp_age',
    'Arm Length': 'arm_length',
    'Height (Inches)': 'height',
    'Weight': 'weight',
    'Draft Pick': 'draft_pick', 
    'BMI': 'bmi',
    'Breakout Age': 'breakout_age',
    'College YPC': 'college_ypc',
    'Breakout Year': 'breakout_year',
    'College Dominator Rating': 'dominator_rating',
    'College Target Share': 'college_tgt_share'
}

# rename columns
data_pp = data_pp.rename(columns=colnames)

# replace undrafted players draft slot with 7.33
data_pp = data_pp.replace("Undrafted", 7.33)

def draft_pick(col):
    a = str(col).split('.')
    x = [float(val) for val in a]
    y = 32*x[0] + x[1] - 32
    return y

# create continuous draft pick number
data_pp['draft_pick'] = data_pp['draft_pick'].apply(draft_pick)

def weight_clean(col):
    y = str(col).split(' ')[0]
    y = float(y)
    return y

def arm_clean(x):
    try:
        return float(sum(Fraction(s) for s in x.split()))
    except:
        return x
    
def pct_clean(x):
    try:
        return float(x.replace('%', ''))
    except:
        return x

# clean up the weight to remove lbs
data_pp['weight'] = data_pp['weight'].apply(weight_clean)

data_pp.arm_length = data_pp.arm_length.apply(arm_clean)
data_pp.hand_size = data_pp.hand_size.apply(arm_clean)
data_pp.three_cone = data_pp.three_cone.apply(pct_clean)
data_pp.college_tgt_share = data_pp.college_tgt_share.apply(pct_clean)
data_pp.dominator_rating = data_pp.dominator_rating.apply(pct_clean)

# convert all columns to numeric
data_pp.iloc[:, 2:] = data_pp.iloc[:, 2:].astype('float')

# select only relevant columns before joining
data_pp = data_pp[['player', 'pp_age', 'draft_year', 'shuffle_20_yd', 'athlete_score', 
                   'sparq', 'three_cone', 'bench_press', 'speed_score', 'forty', 'broad_jump', 'vertical', 
                   'burst_score', 'agility_score',  'hand_size', 'arm_length', 'height', 'weight', 
                   'draft_pick', 'bmi', 'breakout_age', 'college_ypc', 'breakout_year', 'dominator_rating',
                   'college_tgt_share']]

new = data_pp[data_pp.draft_year == set_year]
old = data_pp[data_pp.draft_year != set_year].dropna(thresh=15, axis=0)

data_pp = pd.concat([new, old], axis=0).reset_index(drop=True)
data_pp.pp_age = data_pp.pp_age - (set_year - data_pp.draft_year)

# +
#===========
# Pull out Rookie seasons from training dataframe
#===========

'''
Loop through each player and select their minimum year, which will likely be their 
rookie season. Weird outliers will be removed later on.
'''

conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Season_Stats.sqlite3')
query = "SELECT * FROM RB_Stats A"
rb = pd.read_sql_query(query, con=conn)

rookies = pd.merge(data_pp, rb.drop('avg_pick', axis=1), how='left', left_on='player', right_on='player')
rookies = rookies[(rookies.draft_year == set_year) | (rookies.draft_year == rookies.year)].reset_index(drop=True)

rookies['total_td'] = rookies.rec_td + rookies.rush_td
rookies['rush_yd_per_game'] = rookies.rush_yds / rookies.games
rookies['rec_yd_per_game'] = rookies.rec_yds / rookies.games
rookies['rec_per_game'] = rookies.receptions / rookies.games
rookies['td_per_game'] = rookies.total_td / rookies.games

cols = list(data_pp.columns)
cols.extend(['rush_yd_per_game', 'rec_yd_per_game', 'rec_per_game', 'td_per_game'])

rookies = rookies[cols]
rookies.iloc[:, :-4] = rookies.iloc[:, :-4].fillna(rookies.iloc[:, :-4].median())

rookies = pd.merge(rookies, rb_adp, how='inner', left_on=['player', 'draft_year'], right_on=['player', 'year'])
rookies = rookies.drop('year', axis=1)

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
        'WAS': 'WAS', 
        'STL': 'LAR'
    }

rookies['team'] = rookies['team'].map(adp_to_player_teams)
# -

rookies.loc[:, 'log_draft_pick'] = np.log(rookies.draft_pick)
rookies.loc[:, 'log_avg_pick'] = np.log(rookies.avg_pick)
rookies.loc[:, 'speed_weight'] = rookies.speed_score * rookies.weight
rookies.loc[:, 'speed_weight_age'] = rookies.speed_score * rookies.weight / rookies.pp_age
rookies.loc[:, 'speed_catch'] = rookies.speed_score * rookies.college_tgt_share
rookies.loc[:, 'speed_catch_age'] = rookies.speed_score * rookies.college_tgt_share / rookies.pp_age
rookies.loc[:, 'draft_pick_age'] = rookies.draft_pick / rookies.pp_age

append_to_db(rookies, db_name='Season_Stats.sqlite3', table_name='Rookie_RB_Stats', if_exist='replace')

# # Wide Receivers

# +
#==========
# Pulling in the Player Profiler statistics
#==========

'''
Pull in the player profiler statistics and clean up any formatting issues. Follow by
left joining the statistics to the existing player dataframe.
'''

full_path = path + '/Data/OtherData/PlayerProfiler/{}/WR/'.format(set_year)

# loop through each file and merge together into single file
data_pp = pd.DataFrame()
for file in os.listdir(full_path):
    f = pd.read_csv(full_path + file)
    
    try:
        data_pp = pd.merge(data_pp, f.drop('Position', axis=1), 
                           how='inner', left_on='Full Name', right_on='Full Name')
    except:
        data_pp = f
        
# convert all dashes to null
data_pp = data_pp.replace("-", float('nan'))

colnames = {
    'Full Name': 'player',
    'Position': 'position',
    'Draft Year': 'draft_year',
    '20-Yard Shuttle': 'shuffle_20_yd',
    'Athleticism Score': 'athlete_score',
    'SPARQ-x': 'sparq',
    '3-Cone Drill': 'three_cone',
    'Bench Press': 'bench_press',
    'Height-adjusted Speed Score': 'speed_score',
    '40-Yard Dash': 'forty',
    'Broad Jump': 'broad_jump',
    'Vertical Jump': 'vertical',
    'Burst Score': 'burst_score',
    'Agility Score': 'agility_score',
    'Hand Size': 'hand_size',
    'Age': 'pp_age',
    'Arm Length': 'arm_length',
    'Height (Inches)': 'height',
    'Weight': 'weight',
    'Draft Pick': 'draft_pick', 
    'BMI': 'bmi',
    'Breakout Age': 'breakout_age',
    'College YPR': 'college_ypr',
    'Breakout Year': 'breakout_year',
    'College Dominator Rating': 'dominator_rating',
    'Catch Radius': 'catch_radius',
}

# rename columns
data_pp = data_pp.rename(columns=colnames)

# replace undrafted players draft slot with 7.33
data_pp = data_pp.replace("Undrafted", 7.33)
data_pp = data_pp.replace('Supplemental (2nd)', 2.15)


def draft_pick(col):
    a = str(col).split('.')
    x = [float(val) for val in a]
    y = 32*x[0] + x[1] - 32
    return y

# create continuous draft pick number
data_pp['draft_pick'] = data_pp['draft_pick'].apply(draft_pick)

def weight_clean(col):
    y = str(col).split(' ')[0]
    y = float(y)
    return y

def arm_clean(x):
    try:
        return float(sum(Fraction(s) for s in x.split()))
    except:
        return x
    
def pct_clean(x):
    try:
        return float(x.replace('%', ''))
    except:
        return x

# clean up the weight to remove lbs
data_pp['weight'] = data_pp['weight'].apply(weight_clean)

data_pp.arm_length = data_pp.arm_length.apply(arm_clean)
data_pp.hand_size = data_pp.hand_size.apply(arm_clean)
data_pp.three_cone = data_pp.three_cone.apply(pct_clean)
data_pp.dominator_rating = data_pp.dominator_rating.apply(pct_clean)

# convert all columns to numeric
try:
    data_pp.iloc[:, 2:] = data_pp.iloc[:, 2:].astype('float')
except:
    pass

# select only relevant columns before joining
data_pp = data_pp[['player', 'pp_age', 'draft_year', 'shuffle_20_yd', 'athlete_score', 'sparq', 
                   'three_cone', 'bench_press', 'draft_pick',
                   'speed_score', 'forty', 'broad_jump', 'vertical', 'burst_score', 'agility_score',
                   'hand_size', 'arm_length', 'height', 'weight',  'bmi', 'breakout_age' ,
                   'college_ypr', 'breakout_year', 'dominator_rating']]

new = data_pp[data_pp.draft_year == set_year]
old = data_pp[data_pp.draft_year != set_year].dropna(thresh=10, axis=0)

data_pp = pd.concat([new, old], axis=0).reset_index(drop=True)
data_pp.pp_age = data_pp.pp_age - (set_year - data_pp.draft_year)

# +
url_adp_rec = 'http://www03.myfantasyleague.com/{}/adp?COUNT=100&POS=WR&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=-1&IS_PPR=-1&IS_KEEPER=0&IS_MOCK=0&TIME='
data_adp_rec = pd.DataFrame()
wr_adp = pd.DataFrame()

for year in range(2004, set_year+1):
    url_year = url_adp_rec.format(str(year))
    f = pd.read_html(url_year, header=0)[1]
    f = f.assign(year=year)
    f = clean_adp(f, year)
    wr_adp = pd.concat([wr_adp, f], axis=0)
    
wr_adp = wr_adp.reset_index(drop=True)

# +
#===========
# Pull out Rookie seasons from training dataframe
#===========

'''
Loop through each player and select their minimum year, which will likely be their 
rookie season. Weird outliers will be removed later on.
'''

conn = sqlite3.connect('/Users/Mark/Documents/Github/Fantasy_Football/Data/Season_Stats.sqlite3')
query = "SELECT * FROM WR_Stats A"
wr = pd.read_sql_query(query, con=conn)

rookies = pd.merge(data_pp, wr.drop('avg_pick', axis=1), how='left', left_on='player', right_on='player')
rookies = rookies[(rookies.draft_year == set_year) | (rookies.draft_year == rookies.year)].reset_index(drop=True)

rookies['rec_yd_per_game'] = rookies.rec_yds / rookies.games
rookies['rec_per_game'] = rookies.receptions / rookies.games
rookies['td_per_game'] = rookies.rec_td / rookies.games

cols = list(data_pp.columns)
cols.extend(['rec_yd_per_game', 'rec_per_game', 'td_per_game'])

rookies = rookies[cols]
rookies.iloc[:, :-3] = rookies.iloc[:, :-3].fillna(rookies.iloc[:, :-3].median())

rookies = pd.merge(rookies, wr_adp, how='inner', left_on=['player', 'draft_year'], right_on=['player', 'year'])
rookies = rookies.drop('year', axis=1)


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
        'WAS': 'WAS', 
        'STL': 'LAR'
    }

rookies['team'] = rookies['team'].map(adp_to_player_teams)
# -

rookies.loc[:, 'log_draft_pick'] = np.log(rookies.draft_pick)
rookies.loc[:, 'log_avg_pick'] = np.log(rookies.avg_pick)
rookies.loc[:, 'draft_pick_age'] = rookies.draft_pick / rookies.pp_age
rookies.loc[:, 'hand_dominator'] = rookies.hand_size * rookies.dominator_rating

append_to_db(rookies, db_name='Season_Stats.sqlite3', table_name='Rookie_WR_Stats', if_exist='replace')


