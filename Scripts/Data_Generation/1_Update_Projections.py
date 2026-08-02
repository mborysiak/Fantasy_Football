
# %%

import sys
import os
import re
from io import StringIO
from pathlib import Path

# Add Scripts directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
from config import YEAR, DB_NAME, POSITIONS, LEAGUE

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc
from Scripts.V2.refresh_dk_adp import (
    build_dk_adp_rows,
    parse_dk_payload,
    replace_current_dk_rows,
)

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

import pandas as pd
import requests
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np

#%%

def clean_adp(data_adp, year_val):

    #--------
    # Select relevant columns and clean special figures
    #--------

    data_adp['year'] = year_val

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
        'Avg Pick': 'pick'
    }

    df_adp = df_adp.rename(columns=colnames_adp)
    
    return df_adp

def pull_fantasypros_adp(year_val):
    # FantasyPros put ADP tables behind a login fence (July 2026), so scraping only
    # returns 5 rows. Instead, log in and click Export CSV on
    # https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php, then run this
    # to move the file from Downloads. Note: the best ball ADP page exports the same
    # filename, so download and process one at a time.
    fname = f'FantasyPros_{year_val}_Overall_ADP_Rankings.csv'
    df = move_download_to_folder(root_path, 'FantasyPros_ADP', fname, year_val)

    player_col = [c for c in df.columns if c.startswith('Player')][0]
    player_team = df[player_col].apply(split_fantasypros_best_ball_player)

    df = df.assign(player=player_team.player)
    df = df.rename(columns={'POS': 'pos', 'AVG': 'pick'})

    if df.player.isna().all():
        raise ValueError(f"No players parsed from {fname}; check the export format")

    return df[['player', 'pos', 'pick']]

def pull_draftkings_best_ball_adp():
    url = "https://www.occupyfantasyapi.com/best_ball/adps?site=draftkings&contest=all"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()

    data = response.json()
    rows = data.get('adps', [])
    if not rows:
        raise ValueError("DraftKings best ball ADP data was not found in the Occupy Fantasy API response")

    return parse_dk_payload(data)

def split_fantasypros_best_ball_player(player_value):
    if pd.isna(player_value):
        return pd.Series({'player': np.nan, 'team': np.nan})

    has_bye = '(' in str(player_value)
    player_team = str(player_value).split('(')[0].strip()
    player_team_split = player_team.split()

    if has_bye and len(player_team_split) > 2:
        player = ' '.join(player_team_split[:-1])
        team = player_team_split[-1]
    else:
        player = player_team
        team = np.nan

    return pd.Series({'player': player, 'team': team})

def get_adp(year_val, pos, source):
    
    if source == 'mfl':
        # get the dataset based on year + position
        URL = f'https://www45.myfantasyleague.com/{year_val}/reports?R=ADP&POS={pos}&PERIOD=RECENT&CUTOFF=5&FCOUNT=0&ROOKIES=0&INJURED=1&IS_PPR=3&IS_KEEPER=N&IS_MOCK=1&PAGE=ALL'
        data = pd.read_html(URL)[1]

        # clean the dataset and print out check dataset
        df = clean_adp(data, year_val)[['player', 'pick']]
        print(df.head(10))

        df = df[df.player!='Player Hint:'].reset_index(drop=True)

        # log the avg_pick to match existing
        df['pick'] = df.pick.astype('float')
        df = df.assign(pos=pos, year=year_val, source='mfl')
    
    elif source == 'fantasypros':
        df = pull_fantasypros_adp(year_val)
        df['player'] = df.player.apply(dc.name_clean)
        df['pos'] = df.pos.str.extract(r'^([A-Z]+)', expand=False)
        df['pick'] = df.pick.replace('-', np.nan).astype('float')
        df = df.dropna(subset=['player', 'pos', 'pick']).reset_index(drop=True)
        df = df.assign(year=year_val, source='fpros')
        df = df[['player', 'pick', 'pos', 'year', 'source']]
        
    

    return df

def move_download_to_folder(root_path, folder, fname, set_year, sep=','):

    output_folder = Path(root_path) / 'Data' / 'OtherData' / folder
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f'{set_year}{fname}'

    for download_folder in [Path.home() / 'Downloads', Path('/Users/borys/Downloads')]:
        download_path = download_folder / fname
        if download_path.exists():
            os.replace(download_path, output_path)
            break

    df = pd.read_csv(output_path, sep=sep, on_bad_lines='skip')

    return df

def find_data_file(folder, include_terms, set_year=None, exclude_terms=None):
    exclude_terms = exclude_terms or []
    include_terms = [term.lower() for term in include_terms]
    exclude_terms = [term.lower() for term in exclude_terms]
    search_folders = [Path.home() / 'Downloads', Path(root_path) / 'Data' / 'OtherData' / folder]

    matches = []
    for search_folder in search_folders:
        if not search_folder.exists():
            continue
        for file_path in search_folder.iterdir():
            if not file_path.is_file():
                continue
            file_name = file_path.name.lower()
            if all(term in file_name for term in include_terms) and not any(term in file_name for term in exclude_terms):
                matches.append(file_path)

    if not matches:
        raise FileNotFoundError(f"No file found for terms {include_terms} excluding {exclude_terms}")

    matches = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)
    file_name = matches[0].name
    year_prefix = str(set_year) if set_year is not None else ''
    if year_prefix and file_name.startswith(year_prefix):
        file_name = file_name[len(year_prefix):]

    return file_name

def first_existing_col(df, col_options, label):
    for col in col_options:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {label} column. Available columns: {list(df.columns)}")

def parse_pos_rank(pos_rank):
    if pd.isna(pos_rank):
        return np.nan

    match = re.search(r'\d+', str(pos_rank))
    if match is None:
        return np.nan

    return int(match.group())

def normalize_etr_rank_download(df, year_val, include_adp=False):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={
        first_existing_col(df, ['Player', 'Name', 'player'], 'player'): 'player',
        first_existing_col(df, ['Position', 'Pos', 'pos'], 'position'): 'pos',
        first_existing_col(df, ['Team', 'Tm', 'team'], 'team'): 'team',
        first_existing_col(df, ['ETR Rank', 'ETR_Rank', 'etr_rank'], 'ETR rank'): 'etr_rank',
        first_existing_col(df, ['ETR Pos Rank', 'Pos Rank ETR', 'Pos_Rank', 'etr_pos_rank'], 'ETR position rank'): 'etr_pos_rank',
    })

    df['pos'] = df.pos.astype(str).str.upper().str.strip()
    df = df[~df.pos.isin(['K', 'DST'])].reset_index(drop=True)
    df.player = df.player.apply(dc.name_clean)
    df.etr_rank = pd.to_numeric(df.etr_rank, errors='coerce')
    df.etr_pos_rank = df.etr_pos_rank.apply(parse_pos_rank)
    df = df.assign(year=year_val)

    if include_adp:
        optional_cols = {
            'etr_adp': ['ADP', 'etr_adp'],
            'etr_adp_pos_rank': ['ADP Pos Rank', 'Pos Rank ADP', 'etr_adp_pos_rank'],
            'etr_adp_diff': ['Ranking Diff', 'ADP Dif', 'ADP Delta', 'Delta', 'ADP Differential', 'etr_adp_diff'],
        }
        for output_col, source_cols in optional_cols.items():
            source_col = next((c for c in source_cols if c in df.columns), None)
            if source_col is None:
                df[output_col] = np.nan
            elif output_col == 'etr_adp_pos_rank':
                df[output_col] = df[source_col].apply(parse_pos_rank)
            else:
                df[output_col] = pd.to_numeric(df[source_col], errors='coerce')

        return df[['player', 'team', 'pos', 'etr_rank', 'etr_pos_rank', 'etr_adp', 'etr_adp_pos_rank', 'etr_adp_diff', 'year']]

    return df[['player', 'pos', 'team', 'year', 'etr_rank', 'etr_pos_rank']]

def normalize_silva_rank_download(df, year_val):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={
        first_existing_col(df, ['Player', 'Name', 'player'], 'player'): 'player',
        first_existing_col(df, ['Position', 'Pos', 'pos'], 'position'): 'pos',
        first_existing_col(df, ['Team', 'Tm', 'team'], 'team'): 'team',
        first_existing_col(df, ['Silva Rank', 'Rank', 'evan_silva_rank'], 'Silva rank'): 'evan_silva_rank',
        first_existing_col(df, ['Silva Pos Rank', 'Pos Rank Silva', 'Pos Rank', 'Position Rank', 'evan_silva_pos_rank'], 'Silva position rank'): 'evan_silva_pos_rank',
    })

    df['pos'] = df.pos.astype(str).str.upper().str.strip()
    df.player = df.player.apply(dc.name_clean)
    df.evan_silva_rank = pd.to_numeric(df.evan_silva_rank, errors='coerce')
    df.evan_silva_pos_rank = df.evan_silva_pos_rank.apply(parse_pos_rank)
    df = df.assign(year=year_val)

    return df[['player', 'pos', 'team', 'year', 'evan_silva_rank', 'evan_silva_pos_rank']]

def pull_draft_results(year_val):
    draft_url = f'https://www.pro-football-reference.com/years/{year_val}/draft.htm'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        response = requests.get(draft_url, headers=headers, timeout=30)
        response.raise_for_status()
        if 'Just a moment' in response.text[:5000]:
            raise ValueError("Pro Football Reference returned an anti-bot challenge page")

        df = pd.read_html(StringIO(response.text))[0]
        good_cols = [c[1] if isinstance(c, tuple) else c for c in df.columns]
        df = df.T.reset_index(drop=True).T
        df.columns = good_cols
        df['Year'] = year_val

        df = df[['Year', 'Rnd', 'Pick', 'Player', 'Pos', 'Tm', 'College/Univ']]
        df.columns = ['year', 'Round', 'Pick', 'player', 'pos', 'team', 'college']
        return df

    except Exception as pfr_error:
        nflverse_url = 'https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv'
        df = pd.read_csv(nflverse_url)
        df = df[df.season == year_val].copy()
        if df.empty:
            raise ValueError(f"No draft data found for {year_val} in the nflverse fallback source") from pfr_error

        df = df.rename(columns={
            'season': 'year',
            'round': 'Round',
            'pick': 'Pick',
            'pfr_player_name': 'player',
            'position': 'pos',
            'team': 'team',
            'college': 'college',
        })

        return df[['year', 'Round', 'Pick', 'player', 'pos', 'team', 'college']]


def convert_to_float(df):
    for col in df.columns:
        try:
            df[col] = df[col].astype('float')
        except:
            pass
    return df

def pull_fftoday(pos, year_val):

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
            fft_url = f"https://fftoday.com/rankings/playerproj.php?Season={year_val}&PosID={pos_ids[pos]}&LeagueID=193033&order_by=FFPts&sort_order=DESC&cur_page={page_num}"

            df_cur = pd.read_html(fft_url)[7]
            df_cur = df_cur.iloc[2:, 1:]
            df_cur.columns = cols[pos]

            df_cur = df_cur.assign(pos=pos, year=year_val)

            col_arr = ['player', 'pos', 'team', 'year']
            col_arr.extend([c for c in df_cur.columns if 'fft' in c])
            df_cur = df_cur[col_arr].drop('fft_proj_pts', axis=1, errors='ignore')
            
            df = pd.concat([df, df_cur], axis=0)
            
        except:
            print(pos,year_val, 'failed')

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
    df = move_download_to_folder(root_path, 'FantasyData', fname, set_year)
    
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
    df = df.drop(['fdta_fantasy_points_per_game', 'fdta_fantasy_points_total'], axis=1, errors='ignore')
    
    return df


def format_ffa(df, table_name, set_year):
    df = df.dropna(subset=['player'])
    try: df = df.drop(['Unnamed: 0'], axis=1)
    except: pass

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

for pos in POSITIONS:
    print(YEAR, pos)
    mfl_adp = get_adp(YEAR, pos, 'mfl')
    dm.delete_from_db(DB_NAME, 'ADP_Ranks', f"year={YEAR} and pos='{pos}'", create_backup=False)
    dm.write_to_db(mfl_adp, DB_NAME, 'ADP_Ranks', 'append')

fp_adp = get_adp(YEAR, 'all', 'fantasypros')
dm.write_to_db(fp_adp, DB_NAME, 'ADP_Ranks', 'append')

#%%

def pull_nffc(filename, label):

    df = move_download_to_folder(root_path, 'NFFC', filename, YEAR, sep='\t')
    df.Player = df.Player.apply(lambda x: x.split(',')[1] + ' ' + x.split(',')[0])
    df = df[['Player', 'Team', 'Position(s)', 'ADP', 'Min Pick', 'Max Pick']]
    df.columns = ['player', 'team', 'pos', 'pick_nffc', 'min_pick', 'max_pick']
    df['source'] = label
    df['year'] = YEAR
    df.player = df.player.apply(dc.name_clean)
    return df


df = pull_nffc('ADP.tsv', 'nffc_rotowire_online')
dm.delete_from_db(DB_NAME, 'NFFC_ADP', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'NFFC_ADP', 'append')

df = pull_nffc('ADP (1).tsv', 'nffc_best_ball_overall')
dm.write_to_db(df, DB_NAME, 'NFFC_ADP', 'append')

df = pull_nffc('ADP (2).tsv', 'nffc_best_ball_25s50s')
dm.write_to_db(df, DB_NAME, 'NFFC_ADP', 'append')

df = pull_nffc('ADP (3).tsv', 'nffc_cutline')
dm.write_to_db(df, DB_NAME, 'NFFC_ADP', 'append')

nffc_avg = dm.read(f'''SELECT player,
                                pos,
                                year,
                                avg(pick_nffc) avg_pick,
                                avg(min_pick) min_pick,
                                avg(max_pick) max_pick
                        FROM NFFC_ADP
                        WHERE year = {YEAR}
                        GROUP BY player, pos, year
                        ''', f'Season_Stats_New')
nffc_avg['std_dev'] = (nffc_avg['max_pick'] - nffc_avg['min_pick']) / 5
nffc_avg['league'] = 'nffc'
nffc_avg = nffc_avg.sort_values(by='avg_pick', ascending=True).reset_index(drop=True)

dm.delete_from_db(DB_NAME, 'ADP_Averages', f"year={YEAR} AND league='nffc'", create_backup=False)
dm.write_to_db(nffc_avg, DB_NAME, 'ADP_Averages', 'append')


nffc = dm.read(f'''SELECT *
                   FROM ADP_Averages
                   WHERE year = {YEAR}
                         AND league = 'nffc'
                ''', f'Season_Stats_New')

dk = pull_draftkings_best_ball_adp()
dk = build_dk_adp_rows(nffc, dk, year=YEAR)
replace_current_dk_rows(
    Path(db_path) / f"{DB_NAME}.sqlite3",
    dk,
    year=YEAR,
)

#%%

df = move_download_to_folder(root_path, 'FantasyPros_Best_Ball', f'FantasyPros_{YEAR}_Overall_ADP_Rankings.csv', YEAR)

if 'Player (Bye)' in df.columns:
    df = df.rename(columns={'Player (Bye)': 'Player'})
    player_team = df.Player.apply(split_fantasypros_best_ball_player)
    df = pd.concat([df.drop(columns=['Player']), player_team], axis=1)
elif 'Team' in df.columns:
    df = df.rename(columns={'Player': 'player', 'Team': 'team'})
else:
    df = df.rename(columns={'Player': 'player'})
    df['team'] = np.nan

df = df.dropna(subset=['player']).reset_index(drop=True)
df = df[['player', 'team', 'BB10', 'RTSports', 'Underdog', 'Drafters', 'AVG']]
df.columns = ['player', 'team', 'pick_bb10', 'pick_rtsports', 'pick_underdog', 'pick_drafters', 'pick_best_ball']
df.player = df.player.apply(dc.name_clean)
pick_cols = ['pick_bb10', 'pick_rtsports', 'pick_underdog', 'pick_drafters', 'pick_best_ball']
df[pick_cols] = df[pick_cols].replace({'-': np.nan, '—': np.nan}).apply(pd.to_numeric, errors='coerce')
df['year'] = YEAR

dm.delete_from_db(DB_NAME, 'FantasyPros_Best_Ball_ADP', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyPros_Best_Ball_ADP', 'append')

#%%

# pull fftoday rankings
output = pd.DataFrame()
for pos in POSITIONS:
    df = pull_fftoday(pos, YEAR)
    output = pd.concat([output, df], axis=0, sort=False)

output = output.fillna(0)
output = convert_to_float(output)
output['player'] = output.player.apply(dc.name_clean)
output = predict_fft_sacks(output).round(1)
output.loc[output.pos.isin(['RB', 'WR', 'TE']), 'fft_sacks'] = 0

dm.delete_from_db(DB_NAME, 'FFToday_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(output, DB_NAME, 'FFToday_Projections', 'append')

#%%

# pull fantasydata projections
try:
    fdta_file = [f for f in os.listdir('c:/Users/borys/Downloads') if 'fantasy-football-weekly-projections' in f][0]
    new_fname = '-'.join(fdta_file.split('-')[:-1])+'.csv'
    os.rename(f'/Users/borys/Downloads/{fdta_file}', f'/Users/borys/Downloads/{new_fname}')
except: 
    print('No new Fantasy Data file found')

df = pull_fantasy_data(new_fname, YEAR)

dm.delete_from_db(DB_NAME, 'FantasyData', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyData', 'append')



#%%
df = move_download_to_folder(root_path, 'FFA', f'projections_{YEAR}_wk0.csv', YEAR)
df = format_ffa(df, 'Projections', YEAR)
df = df[~df.team.isnull()].reset_index(drop=True)
df = df.drop(['ffa_bye_week', 'ffa_age', 'ffa_experience'], axis=1, errors='ignore')

dm.delete_from_db(DB_NAME, 'FFA_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FFA_Projections', 'append')


df = move_download_to_folder(root_path, 'FFA', f'raw_stats_{YEAR}_wk0.csv', YEAR)
df = format_ffa(df, 'RawStats', YEAR)
df = df[~df.team.isnull()].reset_index(drop=True)
df = df.drop([c for c in df.columns if '_idp_' in c], axis=1)
df = df.drop(['ffa_birthdate', 'ffa_draft_year'], axis=1)

dm.delete_from_db(DB_NAME, 'FFA_RawStats', f"year={YEAR}", create_backup=False)
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
    print(pos, YEAR)
    
    df_cur = pd.read_html(f'https://www.fantasypros.com/nfl/projections/{pos}.php?week=draft')[0]
    cols = [f'{c[0]}_{c[1]}' if 'Unnamed' not in c[0] else c[1] for c in df_cur.columns]

    df_cur.columns = cols
    df_cur = df_cur.rename(columns=rename_cols).assign(pos=pos.upper(), year=YEAR)
    df_cur.player = df_cur.player.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
    df_cur.player = df_cur.player.apply(dc.name_clean)
    col_order = ['player', 'pos', 'year']
    col_order.extend([c for c in df_cur.columns if 'fpros' in c])
    df_cur = df_cur[col_order]
    df = pd.concat([df, df_cur], axis=0)

df = df.fillna(0)
df = df.round(2)

dm.delete_from_db(DB_NAME, 'FantasyPros_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyPros_Projections', 'append')


#%%

df = move_download_to_folder(root_path, 'PFF_Projections', f'projections.csv', YEAR)

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
df = df.assign(year=YEAR)

df.player = df.player.apply(dc.name_clean)
df.pos = df.pos.apply(lambda x: x.upper())
col_order = ['player', 'pos', 'team', 'year']
col_order.extend([c for c in df.columns if 'pff' in c])
df = df[col_order]
df = df.round(2)

for c in ['pff_pass_comp', 'pff_pass_att', 'pff_pass_yds', 'pff_pass_td', 'pff_pass_int', 'pff_pass_sacked',
          'pff_rush_att', 'pff_rush_yds', 'pff_rush_td', 
          'pff_rec_targets', 'pff_rec_receptions', 'pff_rec_yds', 'pff_rec_td',
          'pff_fumbles', 'pff_fumbles_lost']:
    df[c] = df[c] * 17/df['pff_games']

dm.delete_from_db(DB_NAME, 'PFF_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'PFF_Projections', 'append')

#%%

etr_name = find_data_file('ETR', ['etr', 'half'], YEAR)

df = move_download_to_folder(root_path, 'ETR', etr_name, YEAR)
df = normalize_etr_rank_download(df, YEAR, include_adp=True)

dm.delete_from_db(DB_NAME, 'ETR_Ranks', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'ETR_Ranks', 'append')

#%%

etr_name = find_data_file('ETR', ['etr', 'ppr'], YEAR, exclude_terms=['half'])

df = move_download_to_folder(root_path, 'ETR', etr_name, YEAR)
df = normalize_etr_rank_download(df, YEAR)

dm.delete_from_db(DB_NAME, 'ETR_Ranks_PPR', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'ETR_Ranks_PPR', 'append')

#%%

etr_name = find_data_file('ETR', ['silva'], YEAR)

df = move_download_to_folder(root_path, 'ETR', etr_name, YEAR)
df = normalize_silva_rank_download(df, YEAR)

dm.delete_from_db(DB_NAME, 'Evan_Silva_Ranks', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'Evan_Silva_Ranks', 'append')

#%%

df = move_download_to_folder(root_path, 'FantasyPoints', f'{YEAR} NFL Fantasy Football Season Rankings  Projections  Fantasy Points.csv', YEAR)
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

df = df.assign(year=YEAR)
df.player = df.player.apply(dc.name_clean)

cols = ['player', 'pos', 'team', 'year']
cols.extend([c for c in df.columns if 'fpts' in c])
df = df[cols]
dm.delete_from_db(DB_NAME, 'FantasyPoints_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyPoints_Projections', 'append')

#%%

barret = f'Scott Barretts {YEAR} Redraft Fantasy Football Rankings  Fantasy Points'
df = move_download_to_folder(root_path, 'FantasyPoints', f'{barret}.csv', YEAR)
cols = {
    'Overall': 'barret_total_rank',
    'NAME':'player',
    'POS': 'pos',
    'TEAM': 'team',
    'EXODIA': 'exodia'
}
df = df.rename(columns=cols)
df = df[cols.values()]
df.player = df.player.apply(dc.name_clean)
df = df.assign(year=YEAR)
df.exodia = df.exodia.fillna('1').apply(lambda x: int(x.replace('-', '0')))

dm.delete_from_db(DB_NAME, 'Barret_Ranks', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'Barret_Ranks', 'append')

#%%

df = move_download_to_folder(root_path, 'FanduelResearch', 'REMAINING.csv', YEAR)

df['completions'] = df.completionsAttempts.apply(lambda x: float(x.split('/')[0]))
df['attempts'] = df.completionsAttempts.apply(lambda x: float(x.split('/')[1]))
cols = {
    'player': 'player', 
    'completions': 'fanduel_pass_cmp',
    'attempts': 'fanduel_pass_att',
    'passingYards': 'fanduel_pass_yds',
    'passingTouchdowns': 'fanduel_pass_td',
    'interceptionsThrown': 'fanduel_pass_int',
    'rushingAttempts': 'fanduel_rush_att',
    'rushingYards': 'fanduel_rush_yds',
    'rushingTouchdowns': 'fanduel_rush_td',
    'receptions': 'fanduel_rec',
    'targets': 'fanduel_rec_targets',
    'receivingYards': 'fanduel_rec_yds',
    'receivingTouchdowns': 'fanduel_rec_td',
}
df = (
    df
    .rename(columns=cols)
    .loc[:, cols.values()]
    .assign(year=YEAR)
)

df.player = df.player.apply(dc.name_clean)
dm.delete_from_db(DB_NAME, 'Fanduel_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'Fanduel_Projections', 'append')

#%%

fff_name = [f for f in os.listdir("/Users/borys/Downloads/") if '4for4' in f and 'proj' in f][0]
df = move_download_to_folder(root_path, '4for4', fff_name, YEAR)
cols = {
    'Player': 'player', 
    'Pos': 'pos',
    'Team': 'team',
    'Pass Comp': 'fff_pass_cmp',
    'Pass Att': 'fff_pass_att',
    'Pass Yds': 'fff_pass_yds',
    'Pass TD': 'fff_pass_td',
    'INT': 'fff_pass_int',
    'Rush Att': 'fff_rush_att',
    'Rush Yds': 'fff_rush_yds',
    'Rush TD': 'fff_rush_td',
    'Rec': 'fff_rec',
    'Rec Yds': 'fff_rec_yds',
    'Rec TD': 'fff_rec_td',
    'Pa1D': 'fff_pass_first_downs',
    'Ru1D': 'fff_rush_first_downs',
    'Rec1D': 'fff_rec_first_downs',
    'Health': 'fff_health'
}
df = (
    df
    .rename(columns=cols)
    .loc[:, cols.values()]
    .assign(year=YEAR)
)

df.player = df.player.apply(dc.name_clean)
dm.delete_from_db(DB_NAME, 'FFF_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FFF_Projections', 'append')


#%%
# create full positional list to loop through
draft_pos = pd.DataFrame()

# scrape in the results for each position
d = pull_draft_results(YEAR)

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
dm.delete_from_db(DB_NAME, 'Draft_Positions', f"year={YEAR}")
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



# %%
