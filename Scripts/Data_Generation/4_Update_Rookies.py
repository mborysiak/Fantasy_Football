#%%
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

from ff.db_operations import DataManage
from ff import general, data_clean as dc

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


# set the year for last year's stats to pull
set_year=2021

# set core path
path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football'

# set the database name
db_name = 'Season_Stats'
conn = sqlite3.connect(f'{path}/Data/Databases/{db_name}.sqlite3')

# set the api path for collge football data
CS_API = 'https://api.collegefootballdata.com'

def adp_url(adp_pos, y=set_year+1):
    return f'https://www71.myfantasyleague.com/{y}/reports?R=ADP&POS={adp_pos}&ROOKIES=1&INJURED=1&CUTOFF=5&FCOUNT=0&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PERIOD=RECENT'

#%%
# -

# # Pull in Raw Data Sources

# ## Pull in College Stats

# +
# # player usage data--earliest 2013
# player_usage = requests.get(f'{CS_API}/player/usage?year={cstat_year}&excludeGarbageTime=true')

# # player data such as height and weight--need to loop through each letter in alphabet
# player_data = requests.get(f'{CS_API}/player/search?year={cstat_year}&searchTerm=A')

 # advanced stats data--earliest 2013
# player_adv = requests.get('https://api.collegefootballdata.com/ppa/players/season?year=2013') 

# standard player college stats
CFB_API_KEY = 'Bearer uknadItJAfZPhb8MtWCQ4QOPpihpGWmqkR9SXlgBu1mjhrW8YH7Nv7+IOCjMpWc9'
player_stats = requests.get(f'{CS_API}/stats/player/season?year={set_year}',
                            headers={'Authorization': CFB_API_KEY})

#%%
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

c_stat['rec_yd_mkt_share'] = c_stat.receiving_YDS / c_stat.team_receiving_YDS
c_stat['rec_mkt_share'] = c_stat.receiving_REC / c_stat.team_receiving_REC
c_stat['rec_td_mkt_share'] = c_stat.receiving_TD / c_stat.team_receiving_TD

c_stat['rush_yd_mkt_share'] = c_stat.rushing_YDS / c_stat.team_rushing_YDS
c_stat['att_mkt_share'] = c_stat.rushing_ATT / c_stat.team_rushing_ATT
c_stat['rush_td_mkt_share'] = c_stat.rushing_TD / c_stat.team_rushing_TD

c_stat['year'] = set_year
c_stat = pd.concat([c_stat[['player', 'team', 'year', 'conference']], c_stat.iloc[:, 3:-1]], axis=1)
c_stat = c_stat[(c_stat.rushing_ATT >= 10) | (c_stat.receiving_REC >= 10) | (c_stat.passing_ATT > 25)].reset_index(drop=True)

c_stat.player = c_stat.player.apply(name_clean)
#%%

dm.delete_from_db('Season_Stats', 'College_Stats', f"year={set_year}")
dm.write_to_db(c_stat, 'Season_Stats', 'College_Stats', if_exist='append')


#%%
# ## Get Player's Age as of Start of This Season

# +
birthdays = pd.read_html('https://nflbirthdays.com/')[0]
birthdays.columns = ['_', 'player', 'pos', 'team', 'birthday', '__']
birthdays = birthdays[['player', 'pos', 'birthday']]
birthdays = birthdays.iloc[1:, :]
birthdays.player = birthdays.player.apply(name_clean)    

run_date = dt.datetime(month=9, day=1, year=set_year+1)
birthdays.birthday = (run_date - pd.to_datetime(birthdays.birthday))
birthdays['age'] = birthdays.birthday.apply(lambda x: x.days / 365)

birthdays = birthdays.drop('birthday', axis=1)

exist_age = pd.read_sql_query('''SELECT player, year, pos, age FROM RB_Stats
                                 UNION
                                 SELECT player, year, pos, age FROM WR_Stats
                                 UNION
                                 SELECT player, year, pos, age FROM TE_Stats
                                 UNION
                                 SELECT player, year, pos, qb_age age FROM QB_Stats''', conn)

exist_year_max = exist_age.groupby(['player', 'pos']).agg({'year': 'max'}).reset_index()
exist_age = pd.merge(exist_age, exist_year_max, on=['player', 'pos', 'year'])
exist_age.loc[exist_age.age < 5, 'age'] = np.exp(exist_age.loc[exist_age.age < 5, 'age'])
exist_age['age'] = exist_age.age + (set_year+1)-exist_age.year

bday_check = pd.merge(birthdays, exist_age, on='player').sort_values(by='year')
bday_check[abs(bday_check.age_x - bday_check.age_y) > 1]

# +
exist_age = exist_age[~exist_age.player.isin(birthdays.player)]
birthdays = pd.concat([birthdays, exist_age[['player', 'pos', 'age']]], axis=0)

birthdays['run_date'] = run_date.date()
birthdays = birthdays.dropna()

dm.write_to_db(birthdays, 'Season_Stats', 'Player_Birthdays', if_exist='replace')

#%%
# ## Pull Combine Data

COMBINE_PATH = f'https://www.pro-football-reference.com/draft/{set_year+1}-combine.htm'
comb = pd.read_html(COMBINE_PATH)

# +
comb_df = comb[0]
comb_df['Year'] = set_year+1
comb_df = comb_df.rename(columns={'Year': 'year', 
                                  'Player': 'player', 
                                  'Pos':'pos', 
                                  'Ht':'height', 
                                  'Wt':'weight', 
                                  '40yd':'forty',
                                  'Vertical': 'vertical',
                                  'Bench': 'bench_press', 
                                  'Broad Jump': 'broad_jump', 
                                  '3Cone': 'three_cone', 
                                  'Shuttle': 'shuffle_20_yd'})

ages = dm.read('''SELECT player, age pp_age FROM Player_Birthdays''', 'Season_Stats')
comb_df = pd.merge(comb_df, ages, on='player')

comb_df = comb_df[['year', 'player', 'pos', 'pp_age', 'height', 'weight', 'forty',
                   'vertical', 'bench_press', 'broad_jump', 'three_cone', 'shuffle_20_yd']]

comb_df = comb_df[~comb_df.height.isnull()].reset_index(drop=True)
comb_df.height = comb_df.height.apply(lambda x: 12*int(x.split('-')[0]) + int(x.split('-')[1]))
comb_df = convert_to_float(comb_df)

comb_df.player = comb_df.player.apply(name_clean)
comb_df = comb_df[comb_df.pos.isin(['WR', 'RB', 'TE', 'QB'])].reset_index(drop=True)

#%%
dm.delete_from_db('Season_Stats', 'Combine_Data_Raw', f"year={set_year+1}")
dm.write_to_db(comb_df, 'Season_Stats', 'Combine_Data_Raw', if_exist='append')

#%%

# ## Get College Player ADP

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

# +

def rookie_adp(adp_pos, y):
    
    URL = adp_url(adp_pos, y)
    
    # pulling historical player adp for runningbacks
    data_adp = pd.read_html(URL)[1]
    df_adp = clean_adp(data_adp, y).rename(columns={'year': 'draft_year'})
    df_adp['pos'] = adp_pos
    df_adp['team'] = df_adp['team'].map(adp_to_player_teams)

    
    return df_adp

rb_adp = rookie_adp('RB', set_year+1)
wr_adp = rookie_adp('WR', set_year+1)
te_adp = rookie_adp('TE', set_year+1)
qb_adp = rookie_adp('QB', set_year+1)

rookie_adp = pd.concat([rb_adp, wr_adp, te_adp, qb_adp], axis=0)
rookie_adp = rookie_adp[['player', 'draft_year', 'pos', 'avg_pick', 'team']]
rookie_adp.player = rookie_adp.player.apply(name_clean)

#%%
dm.delete_from_db('Season_Stats', 'Rookie_ADP', f"draft_year={set_year+1}")
dm.write_to_db(rookie_adp, 'Season_Stats', 'Rookie_ADP', if_exist='append')

#%%
# Begin Creating Cleaned up Datasets
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
comb_df = dm.read('''SELECT * FROM combine_data_raw''', 'Season_Stats').drop('pp_age', axis=1)

for c in ['bench_press', 'three_cone']:
    comb_df.loc[comb_df[c]=='', c] = np.nan
    comb_df[c] = comb_df[c].astype('float')
    
draft_info = dm.read('''SELECT * FROM draft_positions''', 'Season_Stats')
draft_values = dm.read('''SELECT * FROM draft_values''', 'Season_Stats')

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
#%%

# ## Predict Player Profiler Data
rb = dm.read('''SELECT * FROM Rookie_RB_Stats_Old''', 'Season_Stats')
wr = dm.read('''SELECT * FROM Rookie_WR_Stats_Old''', 'Season_Stats')

pp = pd.concat([rb, wr], axis=0).reset_index(drop=True)
X_cols = [c for c in comb_df if c not in ('year', 'player', 'Value', 'pp_age', 'QB', 'RB', 'TE', 'WR')]

comb_df = reverse_metrics('hand_size', pp, comb_df, X_cols)
comb_df = reverse_metrics('arm_length', pp, comb_df, X_cols)
comb_df = reverse_metrics('speed_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('athlete_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('sparq', pp, comb_df, X_cols)
comb_df = reverse_metrics('bmi', pp, comb_df, X_cols)
comb_df = reverse_metrics('burst_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('agility_score', pp, comb_df, X_cols)


#%%
from sklearn.preprocessing import RobustScaler
from difflib import SequenceMatcher

comb_stats = dm.read('''SELECT player, year draft_year, pos FROM combine_data_raw''', 'Season_Stats')
draft = dm.read('''SELECT player, year draft_year, pos, college FROM draft_positions''', 'Season_Stats')
player_age = dm.read(f'''SELECT player, pos, age, {set_year+1} as 'run_date' FROM Player_Birthdays ''', 'Season_Stats')
rookie_adp = dm.read('''SELECT * FROM Rookie_ADP''', 'Season_Stats').drop('team', axis=1)

#%%
comb_stats = pd.merge(comb_stats, draft, on=['player', 'draft_year', 'pos'])

#%%
comb_stats = pd.merge(comb_stats, player_age, on=['player', 'pos'])
comb_stats = pd.merge(comb_stats, rookie_adp, on=['player', 'pos', 'draft_year'])

cstats = dm.read('''SELECT * FROM College_Stats''', 'Season_Stats')
comb_stats = pd.merge(comb_stats, cstats, on=['player'])
comb_stats = comb_stats[comb_stats.year < comb_stats.draft_year].reset_index(drop=True)


def match_seq(c):
    c1 = c[0]
    c2 = c[1]
    try:
        s = SequenceMatcher(None, c1, c2).quick_ratio()
    except:
        s = 0
    return s

old_name = ['UAB', 'UCF', 'Ole Miss']
new_name = ['Alabama Birmingham', 'Central Florida', 'Mississippi']
for o, n in zip(old_name, new_name):
    comb_stats.loc[comb_stats.team==o, 'team'] = n
comb_stats['team_match'] = comb_stats[['team', 'college']].apply(match_seq, axis=1)
comb_stats = comb_stats[(comb_stats.team_match > 0.5) | (comb_stats.draft_year == set_year+1)].reset_index(drop=True)

comb_stats = comb_stats[~((comb_stats.player=='Mike Williams') & (comb_stats.college=='Clemson'))]
comb_stats = comb_stats[~((comb_stats.player=='Mike Davis') & (comb_stats.college=='South Carolina'))]
comb_stats = comb_stats[~((comb_stats.player=='Steve Smith') & (comb_stats.college=='USC'))]
comb_stats = comb_stats[~((comb_stats.player=='Cedrick Wilson') & (comb_stats.college=='Boise St.'))]

comb_stats['season_age'] = comb_stats.age - (comb_stats.run_date - comb_stats.year)
comb_stats['season_age_scale'] = RobustScaler().fit_transform(comb_stats.season_age.values.reshape(-1,1))
comb_stats['season_age_scale'] = -comb_stats.season_age_scale.min() + comb_stats.season_age_scale + 1

comb_stats['power5'] = 0
comb_stats.loc[comb_stats.conference.isin(['SEC', 'Pac-10', 'Big 12', 'Big Ten', 'Pac-12', 'ACC']), 'power5'] = 1
comb_stats.groupby('player').agg({'year':'count'}).sort_values(by='year').iloc[-50:]
#%%

# # Split out and save WR

# +
wr = comb_stats[comb_stats.pos=='WR'].reset_index(drop=True).copy()
wr = wr[['player', 'draft_year', 'pos', 'college', 'conference', 'power5', 'age', 
         'year', 'season_age', 'season_age_scale',
         'receiving_LONG', 'receiving_REC', 'receiving_TD', 'receiving_YDS',
         'receiving_YPR', 'team_receiving_LONG', 'team_receiving_REC', 'team_receiving_TD', 
         'team_receiving_YDS','rec_yd_mkt_share', 'rec_mkt_share', 'rec_td_mkt_share']]

wr['rec_dominator'] = (wr.rec_mkt_share + (wr.rec_mkt_share * 1.5 * wr.power5)) / wr.season_age_scale
wr['rec_yd_dominator'] = (wr.rec_yd_mkt_share + (wr.rec_mkt_share* 1.5 * wr.power5)) / wr.season_age_scale
wr['rec_td_dominator'] = (wr.rec_td_mkt_share + (wr.rec_mkt_share* 1.5 * wr.power5)) / wr.season_age_scale

wr_stats = pd.DataFrame()
for m in ['mean', 'max']:

    agg_metr = {
        'rec_dominator': 'mean',
        'rec_yd_dominator': 'mean', 
        'rec_td_dominator': 'mean',
        'rec_mkt_share': 'mean',
        'rec_yd_mkt_share': 'mean',
        'rec_td_mkt_share': 'mean',
    }

    wr_agg = wr.groupby(['player', 'draft_year']).agg(agg_metr)
    wr_agg.columns = [f'{c}_{m}' for c in wr_agg.columns]
    wr_stats = pd.concat([wr_stats, wr_agg], axis=1)
    
wr_stats = wr_stats.reset_index()
# -

comb_df = comb_df.rename(columns={'year': 'draft_year'})
wr_stats = pd.merge(wr_stats, comb_df[comb_df.WR==1], on=['player', 'draft_year'])
wr_stats = wr_stats.drop(['RB', 'TE', 'QB', 'WR'], axis=1)
wr_stats = pd.merge(wr_stats, rookie_adp[rookie_adp.pos=='WR'], on=['player',  'draft_year'])

target_data = pd.read_sql_query('''SELECT player, games, year draft_year, rec_per_game, rec_yd_per_game, td_per_game 
                                   FROM WR_Stats''', conn)
target_data_next = pd.read_sql_query('''SELECT player, games games_next, year-1 draft_year, 
                                        rec_per_game rec_per_game_next, rec_yd_per_game rec_yd_per_game_next, 
                                        td_per_game td_per_game_next 
                                        FROM WR_Stats''', conn)                                  
wr_stats = pd.merge(wr_stats, target_data, on=['player', 'draft_year'], how='left')
wr_stats = pd.merge(wr_stats, target_data_next, on=['player', 'draft_year'], how='left')

wr_stats['fp_per_game'] = (wr_stats[['rec_per_game', 'rec_yd_per_game', 'td_per_game']]*[0.5, 0.1, 7]).sum(axis=1)
wr_stats['fp_per_game_next'] = (wr_stats[['rec_per_game_next', 'rec_yd_per_game_next', 'td_per_game_next']]*[0.5, 0.1, 7]).sum(axis=1)

wr_stats.avg_pick = np.log(wr_stats.avg_pick)

# +
draft = pd.read_sql_query('''SELECT player, pos, team FROM draft_positions''', conn)
draft = draft[~((draft.player=='Anthony Miller') & (draft.team=='LAC'))]
draft = draft[~(draft.player=='Laquon Treadwell')]
draft = draft[~(draft.player=='Tavon Austin')]

wr_stats = pd.merge(wr_stats, draft, on=['player', 'pos'])# -
wr_stats = wr_stats.loc[~(wr_stats.games.isnull()) | (wr_stats.draft_year==set_year+1), :].reset_index(drop=True)
wr_stats = wr_stats.loc[(wr_stats.games > 5) | (wr_stats.draft_year==set_year+1), :].reset_index(drop=True)
dm.write_to_db(wr_stats, 'Season_Stats', 'Rookie_WR_Stats', 'replace', create_backup=True)

# # Split out and Save RB

# +
rb = comb_stats[comb_stats.pos=='RB'].reset_index(drop=True).copy()
rb = rb[['player', 'draft_year', 'pos', 'college', 'conference', 'power5', 'age', 
         'year', 'season_age', 'season_age_scale','rec_yd_mkt_share', 'rec_mkt_share', 'rec_td_mkt_share',
         'rush_yd_mkt_share', 'rush_td_mkt_share', 'att_mkt_share']]
rb['total_yd_mkt_share'] = rb.rush_yd_mkt_share + rb.rec_yd_mkt_share
rb['total_td_mkt_share'] = rb.rush_td_mkt_share + rb.rec_td_mkt_share

rb['rec_dominator'] = (rb.rec_mkt_share + (rb.rec_mkt_share * 0.5 * rb.power5)) / rb.season_age_scale
rb['rec_yd_dominator'] = (rb.rec_yd_mkt_share + (rb.rec_mkt_share* 0.5 * rb.power5)) /rb.season_age_scale
rb['rec_td_dominator'] = (rb.rec_td_mkt_share + (rb.rec_mkt_share* 0.5 * rb.power5)) / rb.season_age_scale

rb['rush_yd_dominator'] = (rb.rush_yd_mkt_share + (rb.rush_yd_mkt_share * 0.5 * rb.power5)) / rb.season_age_scale
rb['rush_td_dominator'] = (rb.rush_td_mkt_share + (rb.rush_td_mkt_share * 0.5 * rb.power5)) / rb.season_age_scale

rb['total_yd_dominator'] = (rb.total_yd_mkt_share + (rb.total_yd_mkt_share * 0.5 * rb.power5)) / rb.season_age_scale
rb['total_td_dominator'] = (rb.total_td_mkt_share + (rb.total_td_mkt_share * 0.5 * rb.power5)) / rb.season_age_scale

agg_metr = {
    'rec_dominator': 'mean',
    'rec_yd_dominator': 'mean', 
    'rec_td_dominator': 'mean',
    'rec_mkt_share': 'mean',
    'rec_yd_mkt_share': 'mean',
    'rec_td_mkt_share': 'mean',
    'rush_yd_dominator': 'mean', 
    'rush_td_dominator': 'mean',
    'rush_yd_mkt_share': 'mean',
    'rush_td_mkt_share': 'mean',
    'total_yd_mkt_share': 'mean',
    'total_td_mkt_share': 'mean',
    'total_yd_dominator': 'mean',
    'total_td_dominator': 'mean',
    'att_mkt_share': 'mean'
}

rb_stats = rb.groupby(['player', 'draft_year']).agg(agg_metr)
rb_stats.columns = [f'{c}_mean' for c in rb_stats.columns]    
rb_stats = rb_stats.reset_index()

# +
try: comb_df = comb_df.rename(columns={'year': 'draft_year'})
except: pass

rb_stats = pd.merge(rb_stats, comb_df[comb_df.RB==1], on=['player', 'draft_year'])
rb_stats = rb_stats.drop(['RB', 'TE', 'QB', 'WR'], axis=1)
rb_stats = pd.merge(rb_stats, rookie_adp[rookie_adp.pos=='RB'], on=['player',  'draft_year'])

# +
target_data = pd.read_sql_query('''SELECT player, year draft_year, games, rush_yd_per_game, rec_per_game, 
                                          rec_yd_per_game, td_per_game 
                                   FROM rb_stats''', conn)
target_data_next = pd.read_sql_query('''SELECT player, year-1 draft_year, games games_next, 
                                        rush_yd_per_game rush_yd_per_game_next, rec_per_game rec_per_game_next, 
                                          rec_yd_per_game rec_yd_per_game_next, td_per_game td_per_game_next 
                                   FROM rb_stats''', conn)
rb_stats = pd.merge(rb_stats, target_data, on=['player', 'draft_year'], how='left')
rb_stats = pd.merge(rb_stats, target_data_next, on=['player', 'draft_year'], how='left')

rb_stats['fp_per_game'] = (rb_stats[['rush_yd_per_game', 'rec_per_game', 'rec_yd_per_game', 
                                     'td_per_game']]*[0.1, 0.5, 0.1, 7]).sum(axis=1)
rb_stats['fp_per_game_next'] = (rb_stats[['rush_yd_per_game_next', 'rec_per_game_next', 'rec_yd_per_game_next', 
                                          'td_per_game_next']]*[0.1, 0.5, 0.1, 7]).sum(axis=1)

rb_stats.avg_pick = np.log(rb_stats.avg_pick)
# -

draft = pd.read_sql_query('''SELECT player, pos, team FROM draft_positions''', conn)
draft = draft[~((draft.player=='Adrian Peterson') & (draft.team=='CHI'))]
rb_stats = pd.merge(rb_stats, draft, on=['player', 'pos'])

rb_stats = rb_stats.loc[~(rb_stats.games.isnull()) | (rb_stats.draft_year==set_year+1), :].reset_index(drop=True)
rb_stats = rb_stats.loc[(rb_stats.games > 5) | (rb_stats.draft_year==set_year+1), :].reset_index(drop=True)

dm.write_to_db(rb_stats, 'Season_Stats', 'Rookie_RB_Stats', 'replace', create_backup=True)

# %%

# %%
