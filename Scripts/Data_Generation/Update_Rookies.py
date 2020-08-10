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

# +
import requests
import json
import pandas as pd
import os
import sqlite3
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np
from fractions import Fraction
import datetime as dt

# set the year for last year's stats to pull
set_year=2019

# set core path
path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football'

# set the database name
db_name = 'Season_Stats'
conn = sqlite3.connect(f'{path}/Data/Databases/{db_name}.sqlite3')

# set the api path for collge football data
CS_API = 'https://api.collegefootballdata.com'
# -

# # Pull in College Stats

# +
# # player usage data--earliest 2013
# player_usage = requests.get(f'{CS_API}/player/usage?year={cstat_year}&excludeGarbageTime=true')

# # player data such as height and weight--need to loop through each letter in alphabet
# player_data = requests.get(f'{CS_API}/player/search?year={cstat_year}&searchTerm=A')

# # advanced stats data--earliest 2013
# player_adv = requests.get('https://api.collegefootballdata.com/ppa/players/season?year=2013') 

# standard player college stats
player_stats = requests.get(f'{CS_API}/stats/player/season?year={set_year}')

# +
c_stat = []
for i in player_stats.json():
    if i['category'] in ['receiving', 'rushing', 'passing']:
        c_stat.append(i)

c_stat = pd.DataFrame(c_stat)
c_stat['stat_name'] = c_stat.category + '_' + c_stat.statType
c_stat['stat'] = c_stat['stat'].astype('float')
c_stat = pd.pivot_table(c_stat, index=['player', 'team', 'conference'], 
                        columns='stat_name', values = 'stat').reset_index().fillna(0)
c_stat = c_stat[~c_stat.player.str.contains('Team')].reset_index(drop=True)
c_stat['rushing_ATT'] = c_stat.rushing_YDS / (c_stat.rushing_YPC + 0.1)

agg_cols = {
    'receiving_LONG': 'max', 
    'receiving_REC': 'sum', 
    'receiving_TD': 'sum',
    'receiving_YDS': 'sum',  
    'rushing_CAR': 'sum', 
    'rushing_LONG': 'max',
    'rushing_TD': 'sum', 
    'rushing_YDS': 'sum', 
    'rushing_ATT': 'sum',
    'passing_ATT': 'sum',
    'passing_COMPLETIONS': 'sum',
    'passing_INT': 'sum',
    'passing_PCT': 'sum',
    'passing_TD': 'sum',
    'passing_YDS': 'sum',
}
team_stat = c_stat.groupby('team').agg(agg_cols).reset_index()
team_stat.columns = ['team_' + c for c in team_stat.columns]
team_stat = team_stat.rename(columns={'team_team': 'team'})
team_stat['team_receiving_YPC'] = team_stat.team_passing_YDS / team_stat.team_passing_COMPLETIONS
team_stat['team_receiving_YPA'] = team_stat.team_passing_YDS / team_stat.team_passing_ATT
team_stat['team_complete_pct'] = team_stat.team_passing_COMPLETIONS / team_stat.team_passing_ATT

c_stat = pd.merge(c_stat, team_stat, on='team')
c_stat = pd.merge(c_stat, c_pstat, on='team')

c_stat['rec_yd_mkt_share'] = c_stat.receiving_YDS / c_stat.team_receiving_YDS
c_stat['rec_mkt_share'] = c_stat.receiving_REC / c_stat.team_receiving_REC
c_stat['rec_td_mkt_share'] = c_stat.receiving_TD / c_stat.team_receiving_TD

c_stat['rush_yd_mkt_share'] = c_stat.rushing_YDS / c_stat.team_rushing_YDS
c_stat['att_mkt_share'] = c_stat.rushing_ATT / c_stat.team_rushing_ATT
c_stat['rush_td_mkt_share'] = c_stat.rushing_TD / c_stat.team_rushing_TD

c_stat['year'] = cstat_year
c_stat = pd.concat([c_stat[['player', 'team', 'year', 'conference']], c_stat.iloc[:, 3:-1]], axis=1)
c_stat = c_stat[(c_stat.rushing_ATT >= 10) | (c_stat.receiving_REC >= 10) | (c_stat.passing_ATT > 25)].reset_index(drop=True)
# -

append_to_db(c_stat, db_name, 'College_Stats', if_exist='append')

# # Pull Combine Data

COMBINE_PATH = f'https://www.pro-football-reference.com/play-index/nfl-combine-results.cgi?request=1&year_min={set_year+1}&year_max={set_year+1}&height_min=0&height_max=100&weight_min=140&weight_max=400&pos%5B%5D=QB&pos%5B%5D=WR&pos%5B%5D=TE&pos%5B%5D=RB&show=all&order_by=year_id'
comb = pd.read_html(COMBINE_PATH)

comb_df = comb[0]
comb_df = comb_df[['Year', 'Player', 'Pos', 'Age', 'Height', 'Wt', '40YD', 
                   'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle']]
comb_df.columns = ['year', 'player', 'pos', 'pp_age', 'height', 'weight', 'forty',
                   'vertical', 'bench_press', 'broad_jump', 'three_cone', 'shuffle_20_yd']
comb_df = comb_df[comb_df.height!='Height'].reset_index(drop=True)
comb_df.height = comb_df.height.apply(lambda x: 12*int(x.split('-')[0]) + int(x.split('-')[1]))
comb_df = convert_to_float(comb_df)

append_to_db(comb_df, db_name, 'Combine_Data_Raw', if_exist='append')

# ## Predict Missing Values

# +
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

def fill_metrics(met, df, X_cols):
    
    print(f'============\n{met}\n------------')
    
    train = df[~df[met].isnull()]
    predict = df[df[met].isnull()]
    
    y = train[met]
    X = train[X_cols]
    
    sc = StandardScaler()

    X_sc = sc.fit_transform(X)
    pred_sc = sc.transform(predict[X_cols])
    pred_sc = pd.DataFrame(pred_sc, columns=X_cols)

    lr = LinearRegression()
    lr.fit(X_sc, y)
    print('R2 Score', round(lr.score(X_sc, y), 3))
    print(pd.Series(lr.coef_, index=X.columns))
    
    df.loc[df[met].isnull(), met] = lr.predict(pred_sc)
    
    return df


def reverse_metrics(met, train, predict, X_cols):
    
    print(f'============\n{met}\n------------')
    
    y = train[met]
    X = train[X_cols]
    
    sc = StandardScaler()

    X_sc = sc.fit_transform(X)
    pred_sc = sc.transform(predict[X_cols])
    pred_sc = pd.DataFrame(pred_sc, columns=X_cols)

    lr = LinearRegression()
    lr.fit(X_sc, y)
    print('R2 Score', round(lr.score(X_sc, y), 3))
    print(pd.Series(lr.coef_, index=X.columns))
    
    predict[met] = lr.predict(pred_sc)
    
    return predict


# +
comb_df = pd.read_sql_query('''SELECT * FROM combine_data_raw''', conn)
draft_info = pd.read_sql_query('''SELECT * FROM draft_positions''', conn)
draft_values = pd.read_sql_query('''SELECT * FROM draft_values''', conn)
draft_info = pd.merge(draft_info, draft_values, on=['Round', 'Pick'])

comb_df = pd.merge(comb_df, draft_info[['year', 'player', 'Value']], on=['year', 'player'], how='left')
comb_df.Value = comb_df.Value.fillna(1)
comb_df = pd.concat([comb_df, pd.get_dummies(comb_df.pos)], axis=1).drop('pos', axis=1)

pred_cols = ['height', 'weight', 'Value', 'QB', 'RB', 'TE', 'WR']
comb_df = fill_metrics('forty', comb_df, pred_cols)

pred_cols.append('forty')
comb_df = fill_metrics('vertical', comb_df, pred_cols)

pred_cols.append('vertical')
comb_df = fill_metrics('broad_jump', comb_df, pred_cols)

pred_cols.append('broad_jump')
comb_df = fill_metrics('three_cone', comb_df, pred_cols)

pred_cols.append('three_cone')
comb_df = fill_metrics('shuffle_20_yd', comb_df, pred_cols)

pred_cols.append('shuffle_20_yd')
comb_df = fill_metrics('bench_press', comb_df, pred_cols)
# -

# ## Predict Player Profiler Data

# +
rb = pd.read_sql_query('''SELECT * FROM Rookie_RB_Stats''', conn)
wr = pd.read_sql_query('''SELECT * FROM Rookie_WR_Stats''', conn)

pp = pd.concat([rb, wr], axis=0).reset_index(drop=True)
X_cols = [c for c in comb_df if c not in ('year', 'player', 'Value', 'pp_age', 'QB', 'RB', 'TE', 'WR')]
# -

# ## Get Player's Age as of Start of This Season



# +
birthdays = pd.read_html('https://nflbirthdays.com/')[0]
birthdays.columns = ['_', 'player', 'position', 'team', 'birthday', '__']
birthdays = birthdays[['player', 'birthday']]
birthdays = birthdays.iloc[1:, :]

run_date = dt.datetime(month=9, day=1, year=set_year+1)
birthdays.birthday = (run_date - pd.to_datetime(birthdays.birthday))
birthdays['age'] = birthdays.birthday.apply(lambda x: x.days / 365)

birthdays = birthdays.drop('birthday', axis=1)
birthdays['run_date'] = run_date.date()
# -

exist_age = pd.read_sql_query('''SELECT player, year, age FROM RB_Stats
                                 UNION
                                 SELECT player, year, age FROM WR_Stats
                                 UNION
                                 SELECT player, year, age FROM TE_Stats
                                 UNION
                                 SELECT player, year, qb_age age FROM QB_Stats''', conn)
exist_age.groupby('player')

birthdays.iloc[:20]



comb_df = reverse_metrics('hand_size', pp, comb_df, X_cols)
comb_df = reverse_metrics('arm_length', pp, comb_df, X_cols)
comb_df = reverse_metrics('speed_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('athlete_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('sparq', pp, comb_df, X_cols)
comb_df = reverse_metrics('bmi', pp, comb_df, X_cols)

pd.read_sql_query('''SELECT * FROM College_Stats''', conn)

# # User Inputs

# # Load Packages

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


