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

import pandas as pd
import os
import sqlite3
from data_functions import *

# +
pff_path = '/Users/Mark/Documents/GitHub/Fantasy_Football/Data/PFF_2018/'

folder = ['receiving_summary', 'wr_deep', 'wr_slot_performance', 'wr_yprr']

data = pd.DataFrame()
for fold in folder:
    files = [f for f in os.listdir(pff_path + fold) if f != '.DS_Store']
    filename = files[0].split('-')[0]

    d_fold = pd.DataFrame()
    for i in range(1, len(files)+1):
        d = pd.read_csv(pff_path + fold + '/' + filename + '-{}.csv'.format(i))
        d['year'] = 2018 - i
        d_fold = pd.concat([d_fold, d], axis=0)
    
    if fold == folder[0]:
        data = d_fold
    else:
        data = pd.merge(data, d_fold, how='inner', left_on=['player', 'year'], right_on=['player', 'year'])
        data = data.drop([c for c in data.columns if c.endswith('_y')], axis=1)
        
data.columns = [c.strip('_x') for c in data.columns]
data = data.reset_index(drop=True)
# -

rec_keep = ['player', 'year', 'grades_offense', 'grades_pass_route', 'grades_hands_drop', 
            'grades_hands_fumble', 'yards_after_catch_per_reception', 'first_downs', 
            'interceptions', 'avoided_tackles', 'targeted_qb_rating','penalties', 'target_rate', 
            'catch_rate', 'drop_rate', 'deep_targets', 'deep_receptions', 'deep_drops', 'deep_yards',
            'deep_catchable', 'deep_touchdowns', 'route_snaps', 'slot_snaps', 'slot_targets', 
            'slot_target_percent', 'slot_receptions', 'slot_drops', 'slot_yards', 'slot_yprr', 
            'slot_touchdowns', 'slot_drop_rate', 'slot_catch_rate', 'yprr']
data = data[rec_keep]

append_to_db(data, db_name='Season_Stats.sqlite3', table_name='PFF_Receiving', if_exist='replace')

# # Fantasy Pros Data

cols = ['player', 'year', 'Rank', 'ADP', 'Best', 'Worst', 'Avg', 'Std Dev']

d2019 = pd.read_csv('/Users/Mark/Documents/GitHub/Fantasy_Football/Data/FantasyPros/FantasyPros2019.csv', header=None)
d2019 = d2019.iloc[1:, 3:]
d2019.columns = ['player', 'team', 'Position', 'Bye', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'ADPvsRank']
d2019['Rank'] = range(1, d2019.shape[0]+1)
d2019['year'] = 2019
d2019 = d2019[cols]

d2018 = pd.read_csv('/Users/Mark/Documents/GitHub/Fantasy_Football/Data/FantasyPros/FantasyPros2018.csv', header=None)
d2018 = d2018.dropna()
d2018.columns = ['Player_Team', 'Position', 'Bye', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'ADPvsRank']
d2018['player'] = d2018.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2018['Rank'] = range(1, d2018.shape[0]+1)
d2018['year'] = 2018
d2018 = d2018[cols]

d2018 = pd.read_csv('/Users/Mark/Documents/GitHub/Fantasy_Football/Data/FantasyPros/FantasyPros2018.csv', header=None)
d2018 = d2018.dropna()
d2018.columns = ['Player_Team', 'Position', 'Bye', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'ADPvsRank']
d2018['player'] = d2018.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2018['Rank'] = range(1, d2018.shape[0]+1)
d2018['year'] = 2018
d2018 = d2018[cols]

y2017 = pd.read_html('https://web.archive.org/web/20170709051007/https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php')
d2017 = y2017[2]
d2017 = d2017.iloc[1:, :9]
d2017 = d2017.dropna()
d2017.columns = ['Rank', 'Player_Team', 'Position', 'Bye', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP']
d2017['player'] = d2017.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2017['year'] = 2017
d2017 = d2017[cols]

y2016 = pd.read_html('https://web.archive.org/web/20160708123616/https://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php')
y2016[3]
d2016 = y2016[3]
d2016.columns = ['Rank', 'Player_Team', 'Position', 'Bye', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'RankvADP']
d2016['player'] = d2016.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2016['year'] = 2016
d2016 = d2016[cols]

y2015 = pd.read_html('https://web.archive.org/web/20150804045948/http://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php')
d2015 = y2015[3]
d2015.columns = ['Rank', 'Player_Team', 'Position', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'RankvADP']
d2015['player'] = d2015.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2015['year'] = 2015
d2015 = d2015[cols]

y2014 = pd.read_html('https://web.archive.org/web/20140804134756/http://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php')
d2014 = y2014[3]
d2014.columns = ['Rank', 'Player_Team', 'Position', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'RankvADP']
d2014['player'] = d2014.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2014['year'] = 2014
d2014 = d2014[cols]

y2013 = pd.read_html('https://web.archive.org/web/20130807045840/https://www.fantasypros.com/nfl/rankings/consensus-cheatsheets.php')
d2013 = y2013[3]
d2013.columns = ['Rank', 'Player_Team', 'Position', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'RankvADP']
d2013['player'] = d2013.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2013['year'] = 2013
d2013 = d2013[cols]

y2012 = pd.read_html('https://web.archive.org/web/20120804231119/http://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php')
d2012 = y2012[3]
d2012.columns = ['Rank', 'Player_Team', 'Best', 'Worst', 'Avg', 'Std Dev', 'ADP', 'RankvADP']
d2012['player'] = d2012.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
d2012['year'] = 2012
d2012 = d2012[cols]

fp_data = pd.concat([d2019, d2018, d2017, d2016, d2015, d2014, d2013, d2012], axis=0).reset_index(drop=True)

fp_data = fp_data.rename(columns={'Rank': 'rank', 'ADP': 'adp', 'Best':'best', 'Worst':'worst', 'Avg':'avg', 'Std Dev': 'std_dev'})

fp_data.adp = fp_data.adp.apply(lambda x: str(x).replace(',', ''))
fp_data = fp_data.dropna()

fp_data = fp_data[fp_data['rank'] != "googletag.cmd.push(function() { googletag.display('div-gpt-ad-1404326895972-0'); });"].reset_index(drop=True)
fp_data.year = fp_data.year.astype('int')
fp_data['rank'] = fp_data['rank'].astype('int')
fp_data.adp = fp_data.adp.astype('float')
fp_data.best = fp_data.best.astype('int')
fp_data.worst = fp_data.worst.astype('int')
fp_data.avg = fp_data.avg.astype('float')
fp_data.std_dev = fp_data.std_dev.astype('float')

append_to_db(fp_data, db_name='Season_Stats.sqlite3', table_name='FantasyPros', if_exist='replace')


