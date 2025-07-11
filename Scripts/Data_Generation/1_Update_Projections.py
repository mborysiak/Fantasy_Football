
# %%

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc

# set to this year
year = 2025

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

import pandas as pd
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np
import os

DB_NAME = 'Season_Stats_New'

#%%

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
    df_adp['Player'] = df_adp.Player.apply(dc.name_clean)
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

def get_adp(year, pos, source):
    
    if source == 'mfl':
        # get the dataset based on year + position
        URL = f'https://www45.myfantasyleague.com/{year}/reports?R=ADP&POS={pos}&PERIOD=RECENT&CUTOFF=5&FCOUNT=0&ROOKIES=0&INJURED=1&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PAGE=ALL'
        data = pd.read_html(URL)[1]

        # clean the dataset and print out check dataset
        df = clean_adp(data, year)[['player', 'avg_pick']]
        print(df.head(10))

        df = df[df.player!='Player Hint:'].reset_index(drop=True)

        # log the avg_pick to match existing
        df['avg_pick'] = df.avg_pick.astype('float')
        df = df.assign(pos=pos, year=year, source='mfl')
    
    elif source == 'fantasypros':
        df = pd.read_html("https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php")[0]
        df = df.rename(columns={'Player Team (Bye)': 'player', 'AVG': 'avg_pick', 'POS': 'pos'})
        
        df.player = df.player.apply(lambda x: x.split('(')[0].rstrip())
        df.player = df.player.apply(lambda x: x.split(' ')[:-1])
        df.player = df.player.apply(lambda x: ' '.join(x))
        df['player'] = df.player.apply(dc.name_clean)
        df['pos'] = df.pos.apply(lambda x: x[:2])
        df['avg_pick'] = df.avg_pick.astype('float')
        df = df.assign(year=year, source='fpros')
        df = df[['player', 'avg_pick', 'pos', 'year', 'source']]
        
    

    return df

def move_download_to_folder(root_path, folder, fname, set_year):
    try:
        os.replace(f"/Users/borys/Downloads/{fname}", 
                    f'{root_path}/Data/OtherData/{folder}/{set_year}{fname}')
    except:
        pass

    df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}{fname}')
    
    return df


def convert_to_float(df):
    for col in df.columns:
        try:
            df[col] = df[col].astype('float')
        except:
            pass
    return df

def pull_fftoday(pos, year):

    pos_ids = {
        'QB': 10,
        'RB': 20,
        'WR': 30,
        'TE': 40
    }

    num_pages = {
        'QB': [0],
        'RB': [0, 1],
        'WR': [0, 1, 2],
        'TE': [0]
        }

    cols = {
            'QB': ['player', 'team', 'bye', 'fft_pass_comp', 'fft_pass_att', 'fft_pass_yds', 'fft_pass_td',
                   'fft_pass_int', 'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 'fft_proj_pts'],
            'WR': ['player', 'team', 'bye', 'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 'fft_proj_pts'],
            'RB': ['player', 'team', 'bye', 'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 
                'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts'],
            'TE': ['player', 'team', 'bye', 'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts']
        }

    df = pd.DataFrame()
    for page_num in num_pages[pos]:
        try:
            fft_url = f"https://fftoday.com/rankings/playerproj.php?Season={year}&PosID={pos_ids[pos]}&LeagueID=193033&order_by=FFPts&sort_order=DESC&cur_page={page_num}"

            df_cur = pd.read_html(fft_url)[7]
            df_cur = df_cur.iloc[2:, 1:]
            df_cur.columns = cols[pos]

            df_cur = df_cur.assign(pos=pos, year=year)

            col_arr = ['player', 'pos', 'team', 'year']
            col_arr.extend([c for c in df_cur.columns if 'fft' in c])
            df_cur = df_cur[col_arr].drop('fft_proj_pts', axis=1, errors='ignore')
            
            df = pd.concat([df, df_cur], axis=0)
            
        except:
            print(pos,year, 'failed')

    return df


def predict_fft_sacks(df_ty):

    fft  = dm.read("SELECT * FROM FFToday_Projections", DB_NAME)
    qb_stats = dm.read("SELECT player, season year, sum_sack_sum FROM QB_Stats WHERE games>12", DB_NAME)
    fft = pd.merge(fft, qb_stats, on=['player', 'year']).dropna().sample(frac=1)

    X = fft[['fft_pass_comp', 'fft_pass_att', 'fft_pass_yds', 'fft_pass_td', 'fft_pass_int', 'fft_rush_att', 'fft_rush_yds']]
    X['fft_pass_yds_per_att'] = X.fft_pass_yds / (X.fft_pass_att+1)
    X['fft_pass_yds_per_cmp'] = X.fft_pass_yds / (X.fft_pass_comp+1)
    X['fft_pass_td_per_att'] = 100*X.fft_pass_td / (X.fft_pass_att+1)
    y = fft.sum_sack_sum

    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    pipe = make_pipeline(StandardScaler(),ElasticNet(alpha=0.01, l1_ratio=0.1))
    preds = cross_val_predict(pipe, X, y, cv=5)
    print(r2_score(y, preds))
    plt.plot(preds, y, 'o')

    pipe.fit(X, y)

    for c in df_ty.columns:
        try: df_ty[c] = df_ty[c].astype('float')
        except: pass

    df_ty['fft_pass_yds_per_att'] = df_ty.fft_pass_yds / (df_ty.fft_pass_att+1)
    df_ty['fft_pass_yds_per_cmp'] = df_ty.fft_pass_yds / (df_ty.fft_pass_comp+1)
    df_ty['fft_pass_td_per_att'] = 100*df_ty.fft_pass_td / (df_ty.fft_pass_att+1)
    df_ty['fft_sacks'] = pipe.predict(df_ty[X.columns])
    df_ty = df_ty.drop(['fft_pass_yds_per_att', 'fft_pass_yds_per_cmp', 'fft_pass_td_per_att'], axis=1, errors='ignore')

    return df_ty


def pull_fantasy_data(fname, set_year):

    # move fantasydata projections
    df = move_download_to_folder(root_path, 'FantasyData', fname, year)
    
    cols = {
            'rank': 'fdta_rank',
            'player': 'player', 
            'team': 'team', 
            'pos': 'pos',
            'pass_yds': 'fdta_pass_yds',
            'pass_td': 'fdta_pass_td',
            'pass_int': 'fdta_pass_int',
            'rush_yds': 'fdta_rush_yds',
            'rush_td': 'fdta_rush_td',
            'rec': 'fdta_rec',
            'rec_yds': 'fdta_rec_yds',
            'rec_td': 'fdta_rec_td',
            'def_sck': 'fdta_sack',
            'def_int': 'fdta_int',
            'fum_recovered': 'fdta_fum_rec',
            'fum_forced': 'fdta_fum_forced',
            'fpts_ppr_per_gp': 'fdta_fantasy_points_per_game',
            'fpts_ppr': 'fdta_fantasy_points_total',
            }
    
    df = df.rename(columns=cols)
    df = df[df.pos.isin(['QB', 'RB', 'WR', 'TE', 'DST'])].reset_index(drop=True)
    df = df.assign(year=set_year)

    df.player = df.player.apply(dc.name_clean)
    df.loc[df.pos=='DST', 'player'] = df.loc[df.pos=='DST', 'team']
    
    col_arr = ['player', 'pos', 'team', 'year']
    col_arr.extend([c for c in df.columns if 'fdta' in c])
    df = df[col_arr]
    
    return df


def format_ffa(df, table_name, set_year):
    df = df.dropna(subset=['player']).drop(['Unnamed: 0'], axis=1)
    df.player = df.player.apply(dc.name_clean)
    df.loc[df.position=='DST', 'player'] = df.loc[df.position=='DST', 'team']

    if table_name=='Projections': new_cols = ['player', 'position', 'team']
    elif table_name=='RawStats': new_cols = ['player', 'team', 'position']

    new_cols.extend(['ffa_' + c for c in df.columns if c not in ('player', 'position', 'team')])
    df.columns = new_cols

    df['year'] = set_year
    col_arr = ['player', 'position', 'team', 'year']
    col_arr.extend([c for c in df.columns if 'ffa' in c])
    df = df[col_arr]
    return df

#%%

for pos in ['QB', 'RB', 'WR', 'TE']:
    print(year, pos)
    mfl_adp = get_adp(year, pos, 'mfl')
    dm.delete_from_db(DB_NAME, 'ADP_Ranks', f"year={year} and pos='{pos}'", create_backup=False)
    dm.write_to_db(mfl_adp, DB_NAME, 'ADP_Ranks', 'append')

fp_adp = get_adp(year, 'all', 'fantasypros')
dm.write_to_db(fp_adp, DB_NAME, 'ADP_Ranks', 'append')


#%%

# pull fftoday rankings
output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    df = pull_fftoday(pos, year)
    output = pd.concat([output, df], axis=0, sort=False)

output = output.fillna(0)
output = convert_to_float(output)
output['player'] = output.player.apply(dc.name_clean)
output = predict_fft_sacks(output).round(1)
output.loc[output.pos.isin(['RB', 'WR', 'TE']), 'fft_sacks'] = 0

dm.delete_from_db(DB_NAME, 'FFToday_Projections', f"year={year}", create_backup=False)
dm.write_to_db(output, DB_NAME, 'FFToday_Projections', 'append')

#%%

# pull fantasydata projections
try:
    fdta_file = [f for f in os.listdir('c:/Users/borys/Downloads') if 'fantasy-football-weekly-projections' in f][0]
    new_fname = '-'.join(fdta_file.split('-')[:-1])+'.csv'
    os.rename(f'/Users/borys/Downloads/{fdta_file}', f'/Users/borys/Downloads/{new_fname}')
except: 
    print('No new Fantasy Data file found')

df = pull_fantasy_data(new_fname, year)

dm.delete_from_db(DB_NAME, 'FantasyData', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyData', 'append')



#%%
df = move_download_to_folder(root_path, 'FFA', f'projections_{year}_wk0.csv', year)
df = format_ffa(df, 'Projections', year)
df = df[~df.team.isnull()].reset_index(drop=True)

dm.delete_from_db(DB_NAME, 'FFA_Projections', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FFA_Projections', 'append')


df = move_download_to_folder(root_path, 'FFA', f'raw_stats_{year}_wk0.csv', year)
df = format_ffa(df, 'RawStats', year)
df = df[~df.team.isnull()].reset_index(drop=True)
df = df.drop([c for c in df.columns if '_idp_' in c], axis=1)
df = df.drop(['ffa_birthdate', 'ffa_draft_year'], axis=1)

dm.delete_from_db(DB_NAME, 'FFA_RawStats', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FFA_RawStats', 'append')


#%%
rename_cols = {
    'Player': 'player',
    'PASSING_ATT': 'fpros_pass_att',
    'PASSING_CMP': 'fpros_pass_cmp',
    'PASSING_YDS': 'fpros_pass_yds',
    'PASSING_TD': 'fpros_pass_td',
    'PASSING_TDS': 'fpros_pass_td',
    'PASSING_INT': 'fpros_pass_int',
    'PASSING_INTS': 'fpros_pass_int',
    'RUSHING_ATT': 'fpros_rush_att',
    'RUSHING_YDS': 'fpros_rush_yds',
    'RUSHING_TD': 'fpros_rush_td',
    'RUSHING_TDS': 'fpros_rush_td',
    'RECEIVING_REC': 'fpros_rec',
    'RECEIVING_YDS': 'fpros_rec_yds',
    'RECEIVING_TD': 'fpros_rec_td',
    'RECEIVING_TDS': 'fpros_rec_td',
    'MISC_FL': 'fpros_fum_lost',
    'FPTS': 'fpros_proj_pts',
    'MISC_FPTS': 'fpros_proj_pts',
}

df = pd.DataFrame()
for pos in ['qb', 'rb', 'wr', 'te']:
    print(pos, year)
    
    df_cur = pd.read_html(f'https://www.fantasypros.com/nfl/projections/{pos}.php?week=draft')[0]
    cols = [f'{c[0]}_{c[1]}' if 'Unnamed' not in c[0] else c[1] for c in df_cur.columns]

    df_cur.columns = cols
    df_cur = df_cur.rename(columns=rename_cols).assign(pos=pos.upper(), year=year)
    df_cur.player = df_cur.player.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
    df_cur.player = df_cur.player.apply(dc.name_clean)
    col_order = ['player', 'pos', 'year']
    col_order.extend([c for c in df_cur.columns if 'fpros' in c])
    df_cur = df_cur[col_order]
    df = pd.concat([df, df_cur], axis=0)

df = df.fillna(0)
df = df.round(2)

dm.delete_from_db(DB_NAME, 'FantasyPros_Projections', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyPros_Projections', 'append')


#%%

df = move_download_to_folder(root_path, 'PFF_Projections', f'projections.csv', year)

rename_cols = {
    'fantasyPointsRank': 'pff_rank', 
    'playerName': 'player', 
    'teamName': 'team', 
    'position': 'pos', 
    'games': 'pff_games', 
    'fantasyPoints': 'pff_proj_pts', 
    'auctionValue': 'pff_auction_value', 
    'passComp': 'pff_pass_comp', 
    'passAtt': 'pff_pass_att',
    'passYds': 'pff_pass_yds', 
    'passTd': 'pff_pass_td', 
    'passInt': 'pff_pass_int', 
    'passSacked': 'pff_pass_sacked', 
    'rushAtt': 'pff_rush_att', 
    'rushYds': 'pff_rush_yds',
    'rushTd': 'pff_rush_td', 
    'recvTargets': 'pff_rec_targets', 
    'recvReceptions': 'pff_rec_receptions', 
    'recvYds': 'pff_rec_yds', 
    'recvTd': 'pff_rec_td',
    'fumbles': 'pff_fumbles',
    'fumblesLost': 'pff_fumbles_lost',
}

df = df.rename(columns=rename_cols)
df = df.assign(year=year)

df.player = df.player.apply(dc.name_clean)
df.pos = df.pos.apply(lambda x: x.upper())
col_order = ['player', 'pos', 'team', 'year']
col_order.extend([c for c in df.columns if 'pff' in c])
df = df[col_order]
df = df.round(2)

dm.delete_from_db(DB_NAME, 'PFF_Projections', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'PFF_Projections', 'append')

#%%

df = move_download_to_folder(root_path, 'ETR', f'{year} Redraft Half PPR Rankings.csv', year)
df = df.rename(columns={'Name': 'player', 
                        'Team': 'team', 
                        'Pos': 'pos',
                        #'ADP': 'etr_adp', 
                        'ETR_Rank': 'etr_rank', 
                        'Pos_Rank': 'etr_pos_rank',
                        #'ADP Differential': 'etr_adp_diff', 
                        #'ADP Pos Rank': 'etr_adp_pos_rank'
                        })
df = df[~df.pos.isin(['K', 'DST'])].reset_index(drop=True)
# df = df[df.etr_adp!= 'Unranked'].reset_index(drop=True)
df.player = df.player.apply(dc.name_clean)
df.etr_pos_rank = df.etr_pos_rank.apply(lambda x: int(x[2:]))
# df.etr_adp_pos_rank = df.etr_adp_pos_rank.apply(lambda x: int(x[2:]))
# df.etr_adp = df.etr_adp.astype('float')
df = df.assign(year=year)

dm.delete_from_db(DB_NAME, 'ETR_Ranks', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'ETR_Ranks', 'append')

#%%

df = move_download_to_folder(root_path, 'FantasyPoints', f'{year} NFL Fantasy Football Season Rankings  Projections  Fantasy Points.csv', year)
df = df.drop(['POS.1', 'UP', 'DOWN','MOVE', 'TARGET', 'WIN'], axis=1)

df = df.rename(columns={
    'RK': 'fpts_overall_rank', 
    'Name': 'player', 
    'POS': 'pos', 
    'Team': 'team',
    'Bye': 'bye', 
    'ADP': 'fpts_adp',
    'FPTS': 'fpts_proj_points', 
    'G': 'fpts_games',
    'TIER': 'fpts_tier',
    'FPTS/G': 'fpts_proj_points_per_game',
    'ATT': 'fpts_pass_att', 
    'CMP': 'fpts_pass_cmp', 
    'YDS': 'fpts_pass_yds', 
    'TD': 'fpts_pass_td',
    'INT': 'fpts_pass_int', 
    'ATT.1': 'fpts_rush_att', 
    'YDS.1': 'fpts_rush_yds', 
    'TD.1': 'fpts_rush_td', 
    'REC': 'fpts_rec', 
    'YDS.2': 'fpts_rec_yds', 
    'TD.2': 'fpts_rec_td', 
})

for c in df.columns:
    try: df[c] = df[c].apply(lambda x: x.replace('-', '0')).astype('float')
    except: pass

df = df.assign(year=year)
df.player = df.player.apply(dc.name_clean)

cols = ['player', 'pos', 'team', 'year']
cols.extend([c for c in df.columns if 'fpts' in c])
df = df[cols]
dm.delete_from_db(DB_NAME, 'FantasyPoints_Projections', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyPoints_Projections', 'append')

#%%
# create full positional list to loop through
draft_pos = pd.DataFrame()

# scrape in the results for each position
DRAFT_URL = f'https://www.pro-football-reference.com/years/{year}/draft.htm'
d = pd.read_html(DRAFT_URL)[0]

# pull out the column names from multi column index
good_cols = [c[1] for c in d.columns]
d = d.T.reset_index(drop=True).T
d.columns = good_cols
d['Year'] = year

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
draft_pos = draft_pos[draft_pos.Pick !='Pick'].reset_index(drop=True)

draft_pos = convert_to_float(draft_pos)

# update the team names
draft_pos.loc[draft_pos.team == 'STL', 'team'] = 'LAR'
draft_pos.loc[draft_pos.team == 'SDG', 'team'] = 'LAC'
draft_pos.loc[draft_pos.team == 'OAK', 'team'] = 'LVR'
draft_pos.player = draft_pos.player.apply(dc.name_clean)
draft_pos
#%%
dm.delete_from_db(DB_NAME, 'Draft_Positions', f"year={year}")
dm.write_to_db(draft_pos, DB_NAME, table_name='Draft_Positions', if_exist='append')

#%%
# ## Roll up to Team Level

# +
# select all data from draft positions
draft_pos = dm.read('''SELECT * FROM Draft_Positions''', DB_NAME)

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
draft_values = dm.read('''SELECT * FROM Draft_Values''', DB_NAME)
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

#%%
dm.write_to_db(team_values, DB_NAME, table_name='Team_Drafts', if_exist='replace')

#%%

# rename_cols = {
#     'Player': 'player',
#     'PASSING_ATT': 'fpros_pass_att',
#     'PASSING_CMP': 'fpros_pass_cmp',
#     'PASSING_YDS': 'fpros_pass_yds',
#     'PASSING_TD': 'fpros_pass_td',
#     'PASSING_TDS': 'fpros_pass_td',
#     'PASSING_INT': 'fpros_pass_int',
#     'PASSING_INTS': 'fpros_pass_int',
#     'RUSHING_ATT': 'fpros_rush_att',
#     'RUSHING_YDS': 'fpros_rush_yds',
#     'RUSHING_TD': 'fpros_rush_td',
#     'RUSHING_TDS': 'fpros_rush_td',
#     'RECEIVING_REC': 'fpros_rec',
#     'RECEIVING_YDS': 'fpros_rec_yds',
#     'RECEIVING_TD': 'fpros_rec_td',
#     'RECEIVING_TDS': 'fpros_rec_td',
#     'MISC_FL': 'fpros_fum_lost',
#     'FPTS': 'fpros_proj_pts',
#     'MISC_FPTS': 'fpros_proj_pts',
# }


# qb = {'2014': 'https://web.archive.org/web/20140825162622/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2015': 'https://web.archive.org/web/20150820231912/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2016': 'https://web.archive.org/web/20160708123639/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2017': 'https://web.archive.org/web/20170815053138/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2018': 'https://web.archive.org/web/20180825080053/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2019': 'https://web.archive.org/web/20190826043354/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2020': 'https://web.archive.org/web/20200818160645/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2021': 'https://web.archive.org/web/20210817132727/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2022': 'https://web.archive.org/web/20220902170213/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       '2023': 'https://web.archive.org/web/20230906183344/https://www.fantasypros.com/nfl/projections/qb.php?week=draft',
#       }

# rb = {
# # '2015': 'https://web.archive.org/web/20150909144856/https://www.fantasypros.com/nfl/projections/rb.php?week=draft',
#       '2020': 'https://web.archive.org/web/20200811013535/https://www.fantasypros.com/nfl/projections/rb.php?week=draft',
#       '2022': 'https://web.archive.org/web/20220526044551/https://www.fantasypros.com/nfl/projections/rb.php?week=draft',
#       '2023': 'https://web.archive.org/web/20230811031041/https://www.fantasypros.com/nfl/projections/rb.php?week=draft',
#       }

# wr = {'2015': 'https://web.archive.org/web/20150908002158/https://www.fantasypros.com/nfl/projections/wr.php?week=draft',
#       '2016': 'https://web.archive.org/web/20180808115212/https://www.fantasypros.com/nfl/projections/wr.php?week=draft',
#       '2020': 'https://web.archive.org/web/20210728120136/https://www.fantasypros.com/nfl/projections/wr.php?week=draft',
#       '2022': 'https://web.archive.org/web/20220825210324/https://www.fantasypros.com/nfl/projections/wr.php?week=draft',
#       '2023': 'https://web.archive.org/web/20230811031238/https://www.fantasypros.com/nfl/projections/wr.php?week=draft',
#       }

# te = {
#     '2015': 'https://web.archive.org/web/20150908002135/https://www.fantasypros.com/nfl/projections/te.php?week=draft',
#     '2020': 'https://web.archive.org/web/20200811022718/https://www.fantasypros.com/nfl/projections/te.php?week=draft',
#     '2023': 'https://web.archive.org/web/20230811031159/https://www.fantasypros.com/nfl/projections/te.php?week=draft',
# }
# import time
# df_out = pd.DataFrame()
# for pos in ['QB', 'RB', 'WR', 'TE']:
#     if pos=='QB': pos_dict = qb
#     elif pos=='RB': pos_dict = rb
#     elif pos=='WR': pos_dict = wr
#     elif pos=='TE': pos_dict = te
#     time.sleep(5)
#     for year, url in pos_dict.items():
#         print(year)
        
#         df = pd.read_html(url)[1]
#         if len(df) < 10: df = pd.read_html(url)[2]
#         cols = [f'{c[0]}_{c[1]}' if 'Unnamed' not in c[0] else c[1] for c in df.columns]

#         df.columns = cols
#         df = df.rename(columns=rename_cols).assign(pos=pos.upper(), year=year)
#         df.player = df.player.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
#         df.player = df.player.apply(name_clean)
#         col_order = ['player', 'pos', 'year']
#         col_order.extend([c for c in df.columns if 'fpros' in c])
#         df = df[col_order]
#         df_out = pd.concat([df_out, df], axis=0, sort=False)    

# # dm.delete_from_db('Season_Stats', 'FantasyPros_Projections', f"year={year} and pos='{pos}'", create_backup=False)
# dm.write_to_db(df_out, 'Season_Stats', 'FantasyPros_Projections', 'replace')
