
# %%

import sys
import os
import re
import hashlib
import time
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
from Scripts.Data_Generation.fantasypros_projection_csv import (
    FANTASYPROS_PROJECTION_POSITIONS,
    build_fantasypros_projection_rows,
    fantasypros_projection_filename,
)
from Scripts.Data_Generation.fantasypoints_projection_csv import (
    normalize_fantasypoints_projection_csv,
)
from Scripts.Data_Generation.adp_rank_ingest import (
    FANTASYPROS_ADP_POSITIONS,
    FANTASYPROS_MINIMUM_ROWS,
    MFL_MINIMUM_ROWS,
    replace_adp_rank_slice,
)
from Scripts.V2.adp_policy import (
    MFL_LAST_MODELED_SEASON,
    NFFC_DOWNLOADS,
    NFFC_MODELED_SOURCES,
    replace_current_nffc_policy_rows,
    utc_now as adp_utc_now,
)

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
adp_db_path = Path(db_path) / f'{DB_NAME}.sqlite3'

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
    fname = f'FantasyPros_{year_val}_Overall_ADP_Rankings (1).csv'
    df = move_download_to_folder(
        root_path,
        'FantasyPros_ADP',
        fname,
        year_val,
        archive_name='FantasyPros_Redraft_Half_PPR_ADP.csv',
    )

    best_ball_markers = {'BB10', 'Underdog', 'Drafters', 'DraftKings'}
    redraft_markers = {'ESPN', 'Sleeper', 'CBS', 'NFL', 'RTSports', 'Fantrax'}
    if best_ball_markers.intersection(df.columns) or not redraft_markers.intersection(df.columns):
        raise ValueError(
            f"{fname} is not the FantasyPros redraft ADP export; "
            f"available columns are {list(df.columns)}"
        )

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
        df = df[df.pos.isin(FANTASYPROS_ADP_POSITIONS)].reset_index(drop=True)
        df = df.assign(year=year_val, source='fpros')
        df = df[['player', 'pick', 'pos', 'year', 'source']]
        
    

    return df

def move_download_to_folder(
    root_path,
    folder,
    fname,
    set_year,
    sep=',',
    archive_name=None,
    header='infer',
):

    output_folder = Path(root_path) / 'Data' / 'OtherData' / folder
    output_folder.mkdir(parents=True, exist_ok=True)
    archived_filename = fname if archive_name is None else archive_name
    output_path = output_folder / f'{set_year}{archived_filename}'

    for download_folder in [Path.home() / 'Downloads', Path('/Users/borys/Downloads')]:
        download_path = download_folder / fname
        if download_path.exists():
            os.replace(download_path, output_path)
            break

    df = pd.read_csv(
        output_path,
        sep=sep,
        header=header,
        on_bad_lines='skip',
    )

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

FFTODAY_MINIMUM_ROWS = {
    'QB': 40,
    'RB': 80,
    'WR': 100,
    'TE': 40,
}


def pull_fftoday(pos, year_val, max_attempts=3):

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
        last_error = None
        for attempt in range(1, max_attempts + 1):
            fft_url = f"https://fftoday.com/rankings/playerproj.php?Season={year_val}&PosID={pos_ids[pos]}&LeagueID=193033&order_by=FFPts&sort_order=DESC&cur_page={page_num}"

            try:
                tables = pd.read_html(fft_url)
                matching_tables = [
                    table
                    for table in tables
                    if table.shape[0] >= 3
                    and table.shape[1] == len(cols[pos]) + 1
                ]
                if len(matching_tables) != 1:
                    raise ValueError(
                        f"expected one FFToday projection table, found "
                        f"{len(matching_tables)}"
                    )

                df_cur = matching_tables[0].iloc[2:, 1:].copy()
                df_cur.columns = cols[pos]
                df_cur = df_cur.assign(pos=pos, year=year_val)

                col_arr = ['player', 'pos', 'team', 'year']
                col_arr.extend([c for c in df_cur.columns if 'fft' in c])
                df_cur = df_cur[col_arr].drop(
                    'fft_proj_pts', axis=1, errors='ignore'
                )
                df = pd.concat([df, df_cur], axis=0)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(attempt)

        if last_error is not None:
            raise RuntimeError(
                f"FFToday {pos} {year_val} page {page_num} failed after "
                f"{max_attempts} attempts"
            ) from last_error

    return df


def validate_fftoday_rows(df, year_val):
    position_counts = df['pos'].value_counts().to_dict()
    failures = {
        pos: {'actual': int(position_counts.get(pos, 0)), 'minimum': minimum}
        for pos, minimum in FFTODAY_MINIMUM_ROWS.items()
        if position_counts.get(pos, 0) < minimum
    }
    if failures:
        raise ValueError(
            f"FFToday {year_val} coverage validation failed: {failures}"
        )
    return position_counts


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

if YEAR <= MFL_LAST_MODELED_SEASON:
    for pos in POSITIONS:
        print(YEAR, pos)
        mfl_adp = get_adp(YEAR, pos, 'mfl')
        replace_adp_rank_slice(
            adp_db_path,
            mfl_adp,
            year=YEAR,
            source='mfl',
            position=pos,
            allowed_positions=(pos,),
            minimum_rows_by_position={pos: MFL_MINIMUM_ROWS[pos]},
        )
else:
    print(
        f"Skipping MFL ADP for {YEAR}; governed model use ends after "
        f"{MFL_LAST_MODELED_SEASON}."
    )

fp_adp = get_adp(YEAR, 'all', 'fantasypros')
replace_adp_rank_slice(
    adp_db_path,
    fp_adp,
    year=YEAR,
    source='fpros',
    allowed_positions=FANTASYPROS_ADP_POSITIONS,
    minimum_rows_by_position=FANTASYPROS_MINIMUM_ROWS,
)
fp_position_counts = fp_adp['pos'].value_counts()
print(
    f"FantasyPros ADP save confirmed for {YEAR}: {len(fp_adp):,} rows "
    "saved to ADP_Ranks "
    f"({', '.join(f'{pos}={int(fp_position_counts.get(pos, 0)):,}' for pos in FANTASYPROS_ADP_POSITIONS)})."
)

#%%

def pull_nffc(filename, label, archive_name):

    df = move_download_to_folder(
        root_path,
        'NFFC',
        filename,
        YEAR,
        sep='\t',
        archive_name=archive_name,
    )
    required = {
        'Rank', 'Player', 'Team', 'Position(s)', 'ADP', 'Min Pick',
        'Max Pick', '# Picks',
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"NFFC {label} export is missing columns: {missing}"
        )
    snapshot_path = (
        Path(root_path)
        / 'Data'
        / 'OtherData'
        / 'NFFC'
        / f'{YEAR}{archive_name}'
    )
    snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    df.Player = df.Player.apply(lambda x: x.split(',')[1] + ' ' + x.split(',')[0])
    df = df[[
        'Player', 'Team', 'Position(s)', 'ADP', 'Min Pick', 'Max Pick',
        'Rank', '# Picks',
    ]]
    df.columns = [
        'player', 'team', 'pos', 'pick_nffc', 'min_pick', 'max_pick',
        'source_rank', 'draft_count',
    ]
    df['source'] = label
    df['year'] = YEAR
    df['snapshot_file'] = snapshot_path.name
    df['snapshot_sha256'] = snapshot_sha256
    df['ingested_at_utc'] = adp_utc_now()
    df.player = df.player.apply(dc.name_clean)
    return df


nffc_raw = []
for download_name, feed in NFFC_DOWNLOADS.items():
    nffc_raw.append(
        pull_nffc(
            download_name,
            feed['source'],
            feed['archive_name'],
        )
    )
nffc_raw = pd.concat(nffc_raw, ignore_index=True)
if set(nffc_raw.source.unique()) != set(NFFC_MODELED_SOURCES):
    raise ValueError('NFFC ingest did not produce the two governed feeds')
nffc_avg = replace_current_nffc_policy_rows(
    adp_db_path,
    nffc_raw,
    year=YEAR,
    rebuild_from_season=2025,
)
nffc_current_avg = nffc_avg.loc[
    pd.to_numeric(nffc_avg['year'], errors='coerce').eq(YEAR)
]
nffc_feed_counts = nffc_raw['source'].value_counts()
print(
    f"NFFC ADP save confirmed for {YEAR}: {len(nffc_raw):,} raw rows "
    f"({', '.join(f'{source}={int(nffc_feed_counts.get(source, 0)):,}' for source in NFFC_MODELED_SOURCES)}) "
    f"and {len(nffc_current_avg):,} aggregate rows saved."
)


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
dk_position_counts = dk['pos'].value_counts()
print(
    f"DraftKings ADP save confirmed for {YEAR}: {len(dk):,} rows saved "
    f"({', '.join(f'{pos}={int(dk_position_counts.get(pos, 0)):,}' for pos in POSITIONS)})."
)

#%%

df = move_download_to_folder(
    root_path,
    'FantasyPros_Best_Ball',
    f'FantasyPros_{YEAR}_Overall_ADP_Rankings.csv',
    YEAR,
    archive_name='FantasyPros_Best_Ball_ADP.csv',
)

required_best_ball_columns = {'BB10', 'Underdog', 'Drafters', 'AVG'}
missing_best_ball_columns = sorted(required_best_ball_columns.difference(df.columns))
if missing_best_ball_columns:
    raise ValueError(
        'FantasyPros best-ball export is missing columns '
        f'{missing_best_ball_columns}; available columns are {list(df.columns)}'
    )

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

fftoday_position_counts = validate_fftoday_rows(output, YEAR)
output = output.fillna(0)
output = convert_to_float(output)
output['player'] = output.player.apply(dc.name_clean)
output = predict_fft_sacks(output).round(1)
output.loc[output.pos.isin(['RB', 'WR', 'TE']), 'fft_sacks'] = 0

dm.delete_from_db(DB_NAME, 'FFToday_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(output, DB_NAME, 'FFToday_Projections', 'append')
saved_fftoday = dm.read(
    f"SELECT pos FROM FFToday_Projections WHERE year={YEAR}", DB_NAME
)
saved_fftoday_counts = saved_fftoday['pos'].value_counts().to_dict()
if saved_fftoday_counts != fftoday_position_counts:
    raise RuntimeError(
        f"FFToday save verification failed for {YEAR}: expected "
        f"{fftoday_position_counts}, found {saved_fftoday_counts}"
    )
print(
    f"FFToday projections save confirmed for {YEAR}: {len(saved_fftoday):,} "
    f"rows saved to FFToday_Projections {saved_fftoday_counts}."
)

#%%

# # pull fantasydata projections
# try:
#     fdta_file = [f for f in os.listdir('c:/Users/borys/Downloads') if 'fantasy-football-weekly-projections' in f][0]
#     new_fname = '-'.join(fdta_file.split('-')[:-1])+'.csv'
#     os.rename(f'/Users/borys/Downloads/{fdta_file}', f'/Users/borys/Downloads/{new_fname}')
# except:
#     print('No new Fantasy Data file found')

# df = pull_fantasy_data(new_fname, YEAR)

# dm.delete_from_db(DB_NAME, 'FantasyData', f"year={YEAR}", create_backup=False)
# dm.write_to_db(df, DB_NAME, 'FantasyData', 'append')



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
fantasypros_frames = {}
for pos in FANTASYPROS_PROJECTION_POSITIONS:
    fname = fantasypros_projection_filename(pos)
    print(f'FantasyPros CSV: {pos} {YEAR} ({fname})')
    fantasypros_frames[pos] = move_download_to_folder(
        root_path,
        'FantasyPros_Projections',
        fname,
        YEAR,
    )

df = build_fantasypros_projection_rows(fantasypros_frames, year=YEAR)
df['player'] = df.player.apply(dc.name_clean)
if df.duplicated(['player', 'pos']).any():
    duplicates = df.loc[
        df.duplicated(['player', 'pos'], keep=False),
        ['player', 'pos'],
    ].to_dict('records')
    raise ValueError(
        f'FantasyPros projections have duplicate cleaned player keys: {duplicates}'
    )
print(df.groupby('pos').size().to_dict())

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

df = move_download_to_folder(
    root_path,
    'FantasyPoints',
    'projections.season.csv',
    YEAR,
    header=1,
)
df = normalize_fantasypoints_projection_csv(df, year=YEAR)
df.player = df.player.apply(dc.name_clean)
duplicate_fpts_keys = df.duplicated(['player', 'pos'], keep=False)
if duplicate_fpts_keys.any():
    duplicates = df.loc[duplicate_fpts_keys, ['player', 'pos']]
    raise ValueError(
        'FantasyPoints projections have duplicate cleaned player-position '
        f"keys: {duplicates.head(20).to_dict('records')}"
    )
dm.delete_from_db(DB_NAME, 'FantasyPoints_Projections', f"year={YEAR}", create_backup=False)
dm.write_to_db(df, DB_NAME, 'FantasyPoints_Projections', 'append')
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





#%%

def save_pff_stats(stat_type, set_year):
    if stat_type=='QB': fname = 'passing_summary'
    elif stat_type=='Rec': fname='receiving_summary'
    elif stat_type=='Rush': fname='rushing_summary'
    elif stat_type=='Oline': fname='offense_blocking'

    try:
        os.replace(f"/Users/borys/Downloads/{fname}.csv", 
                   f'{root_path}/Data/OtherData/PFF_Stats/{set_year}_{fname}.csv')
    except: 
        pass
    
    df = pd.read_csv(f'{root_path}/Data/OtherData/PFF_Stats/{set_year}_{fname}.csv')
    df.player = df.player.apply(dc.name_clean)
    df['year'] = set_year

    cols = dm.read(f"SELECT * FROM PFF_{stat_type}_Stats WHERE year={set_year}", DB_NAME).columns
    df = df[[c for c in cols if c in df.columns]]

    dm.delete_from_db(DB_NAME, f'PFF_{stat_type}_Stats', f"year={set_year}", create_backup=False)
    dm.write_to_db(df, DB_NAME, f'PFF_{stat_type}_Stats', 'append')

    return df

for stat_type in ['QB', 'Rec', 'Rush', 'Oline']:
    df = save_pff_stats(stat_type, YEAR-1)

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
