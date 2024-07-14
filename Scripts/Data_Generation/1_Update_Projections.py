
# %%

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc

# last year's statistics and adp to pull and append to database
year = 2024

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
        df['avg_pick_no_log'] = df.avg_pick
        df.avg_pick = np.log(df.avg_pick.astype('float'))
        df = df.assign(pos=pos, year=year, source='mfl')
    
    elif source == 'fantasypros':
        df = pd.read_html("https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php")[0]
        df = df.rename(columns={'Player Team (Bye)': 'player', 'AVG': 'avg_pick', 'POS': 'pos'})
        
        df.player = df.player.apply(lambda x: x.split('(')[0].rstrip())
        df.player = df.player.apply(lambda x: x.split(' ')[:-1])
        df.player = df.player.apply(lambda x: ' '.join(x))
        df['player'] = df.player.apply(dc.name_clean)
        df['pos'] = df.pos.apply(lambda x: x[:2])
        df['avg_pick_no_log'] = df.avg_pick
        df.avg_pick = np.log(df.avg_pick.astype('float'))
        df = df.assign(year=year, source='fpros')
        df = df[['player', 'avg_pick', 'avg_pick_no_log', 'pos', 'year', 'source']]
        
    

    return df

year=2024
fp_adp = get_adp(year, 'all', 'fantasypros')

for pos in ['QB', 'RB', 'WR', 'TE']:
    print(year, pos)
    mfl_adp = get_adp(year, pos, 'mfl')
    dm.delete_from_db(DB_NAME, 'ADP_Ranks', f"year={year} and pos='{pos}'", create_backup=False)
    dm.write_to_db(mfl_adp, DB_NAME, 'ADP_Ranks', 'append')

dm.write_to_db(fp_adp, DB_NAME, 'ADP_Ranks', 'append')


#%%

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
            fft_url = f"https://fftoday.com/rankings/playerproj.php?Season={year}&PosID={pos_ids[pos]}&LeagueID=&order_by=FFPts&sort_order=DESC&cur_page={page_num}"

            df_cur = pd.read_html(fft_url)[7]
            df_cur = df_cur.iloc[2:, 1:]
            df_cur.columns = cols[pos]

            df_cur = df_cur.assign(pos=pos, year=year)

            col_arr = ['player', 'pos', 'team', 'year']
            col_arr.extend([c for c in df_cur.columns if 'fft' in c])
            df_cur = df_cur[col_arr]
            
            df = pd.concat([df, df_cur], axis=0)
            
        except:
            print(pos,year, 'failed')

    return df


def pull_fantasy_data(fname, set_year):

    # move fantasydata projections
    df = move_download_to_folder(root_path, 'FantasyData', fname, year)
    

    cols = {
            'Rank': 'fdta_rank',
            'Name': 'player', 
            'Team': 'team', 
            'Position': 'pos',
            'PassingYards': 'fdta_pass_yds',
            'PassingTouchdowns': 'fdta_pass_td',
            'PassingInterceptions': 'fdta_pass_int',
            'RushingYards': 'fdta_rush_yds',
            'RushingTouchdowns': 'fdta_rush_td',
            'Receptions': 'fdta_rec',
            'ReceivingYards': 'fdta_rec_yds',
            'ReceivingTouchdowns': 'fdta_rec_td',
            'Sacks': 'fdta_sack',
            'Interceptions': 'fdta_int',
            'FumblesRecovered': 'fdta_fum_rec',
            'FumblesForced': 'fdta_fum_forced',
            'FantasyPointsPerGameHalfPointPPR': 'fdta_fantasy_points_per_game',
            'FantasyPointsHalfPointPpr': 'fdta_fantasy_points_total',
            }
    df = df[df.Position.isin(['QB', 'RB', 'WR', 'TE', 'DST'])].reset_index(drop=True)
    df = df.rename(columns=cols)
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

# pull fftoday rankings
output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    df = pull_fftoday(pos, year)
    output = pd.concat([output, df], axis=0, sort=False)

output = output.fillna(0)
output = convert_to_float(output)
df['player'] = df.player.apply(dc.name_clean)

dm.delete_from_db(DB_NAME, 'FFToday_Projections', f"year={year}", create_backup=False)
dm.write_to_db(output, DB_NAME, 'FFToday_Projections', 'append')

#%%

# pull fantasydata projections
df = pull_fantasy_data(f'fantasy-football-weekly-projections.csv', year)
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

for pos in ['qb', 'rb', 'wr', 'te']:
    print(pos, year)
    
    df = pd.read_html(f'https://www.fantasypros.com/nfl/projections/{pos}.php?week=draft')[0]
    cols = [f'{c[0]}_{c[1]}' if 'Unnamed' not in c[0] else c[1] for c in df.columns]

    df.columns = cols
    df = df.rename(columns=rename_cols).assign(pos=pos.upper(), year=year)
    df.player = df.player.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
    df.player = df.player.apply(dc.name_clean)
    col_order = ['player', 'pos', 'year']
    col_order.extend([c for c in df.columns if 'fpros' in c])
    df = df[col_order]

    dm.delete_from_db(DB_NAME, 'FantasyPros_Projections', f"year={year} and pos='{pos}'", create_backup=False)
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

dm.delete_from_db(DB_NAME, 'PFF_Projections', f"year={year}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'PFF_Projections', 'append')


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

# %%
