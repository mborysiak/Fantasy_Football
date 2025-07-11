#%%

# set to this year for the projections
year = 2025

from ff.db_operations import DataManage
from ff import general

from skmodel import SciKitModel
import pandas as pd
from zData_Functions import *
pd.options.mode.chained_assignment = None
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', 999)


# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
DB_NAME = 'Season_Stats_New'

def name_cleanup(df):
    df.player = df.player.apply(name_clean)
    return df

def rolling_stats(df, gcols, rcols, period, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    rolls = df.groupby(gcols)[rcols].rolling(period, min_periods=1).agg(agg_type).reset_index(drop=True)
    rolls.columns = [f'r{agg_type}{period}_{c}' for c in rolls.columns]

    return rolls

def add_rolling_stats(df, gcols, rcols, perform_check=True):

    df = df.sort_values(by=[gcols[0], 'year']).reset_index(drop=True)

    if perform_check:
        cnt_check = df.groupby([gcols[0], 'year'])['year'].count()
        print(f'Counts of Groupby Category Over 17: {cnt_check[cnt_check>1]}')

    rolls3_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')

    df = pd.concat([df, rolls3_mean, rolls3_max,], axis=1)

    return df


def forward_fill(df, cols=None):
    
    if cols is None: cols = df.columns
    df = df.sort_values(by=['player', 'year'])
    df = df.groupby('player', as_index=False)[cols].fillna(method='ffill')
    df = df.sort_values(by=['player', 'year'])

    return df

#---------------
# Pre Game Data
#---------------

def calc_total_tds(df, prefix):
    df[f'{prefix}_total_tds'] = 0
    for c in ['pass_td', 'rush_td', 'rec_td']:
        try: df[f'{prefix}_total_tds'] += df[f'{prefix}_{c}']
        except: pass
    return df

def calc_proj_pts(df, pos, prefix):

    pt_values = {
        'pass_yds': 0.04,
        'pass_td': 5,
        'pass_int': -2,
        'pass_sacks': -1,
        'rush_yds': 0.1,
        'rush_td': 7,
        'rec_yds': 0.1,
        'rec_td': 7,
        'rec': 0.5
    }

    df[f'{prefix}_proj_points'] = 0

    if pos == 'QB': cols = ['pass_yds', 'pass_td', 'pass_int', 'pass_sacks', 'rush_yds', 'rush_td']
    else: cols = ['rush_yds', 'rush_td', 'rec_yds', 'rec_td', 'rec']
    for c in cols:
        try: df[f'{prefix}_proj_points'] += df[f'{prefix}_{c}'] * pt_values[c]
        except: df[f'{prefix}_proj_points'] += df[f'avg_proj_{c}'] * pt_values[c]

    return df

def calc_pos_rank(df, prefix):
    df[f'{prefix}_pos_rank'] = df.groupby(['year', 'pos'])[f'{prefix}_proj_points'].rank(ascending=False, method='min')
    df[f'{prefix}_pos_rank_log'] = np.log(df[f'{prefix}_pos_rank']+1)
    return df

def fftoday_proj(pos):
    df = dm.read(f'''SELECT * 
                     FROM FFToday_Projections 
                     WHERE pos='{pos}'
                           AND team!='FA'
                 ''', DB_NAME)
    df.team = df.team.map(team_map)
    df = df.rename(columns={'fft_sacks': 'fft_pass_sacks'})
    df = calc_total_tds(df, 'fft')
    df = calc_proj_pts(df, pos, 'fft')

    if pos == 'QB':
        df['fft_sacks_per_att'] = df.fft_pass_sacks / (df.fft_pass_att + 1)
        df.loc[df.fft_sacks_per_att>0.1, 'fft_pass_sacks'] = \
            df.loc[df.fft_sacks_per_att>0.1, 'fft_pass_att'] * 0.1 
        df = df.drop('fft_sacks_per_att', axis=1)

    return df

def fantasy_pros_new(df, pos):

    fp = dm.read(f'''SELECT * 
                     FROM FantasyPros_Projections
                     WHERE pos='{pos}' 
                         ''', DB_NAME)

    fp = fp.drop(['pos', 'fpros_fum_lost', 'fpros_proj_pts'], axis=1)
    fp = fp.rename(columns={'fpros_pass_cmp': 'fpros_pass_comp'})

    fp = calc_total_tds(fp, 'fpros')
    fp = name_cleanup(fp)
    fp.year = fp.year.astype('int')
    df = pd.merge(df, fp, on=['player', 'year'], how='left')
    return df

def ffa_compile_stats(df, pos):
    
    if pos == 'QB':
        cols = ['player', 'year',
                'ffa_pass_yds', 'ffa_pass_tds', 'ffa_pass_int',
                'ffa_rush_yds', 'ffa_rush_tds',
                ]
    elif pos in ('RB', 'WR', 'TE'):
        cols =  ['player', 'year', 
                 'ffa_rush_yds', 'ffa_rush_tds',
                 'ffa_rec_yds', 'ffa_rec_tds']

    ffa = dm.read(f"SELECT * FROM FFA_RawStats WHERE position='{pos}'", DB_NAME)
    ffa = ffa[cols].drop_duplicates()

    if pos=='QB': ffa = ffa.rename(columns={'ffa_pass_tds': 'ffa_pass_td'})
    ffa = ffa.rename(columns={'ffa_rush_tds': 'ffa_rush_td', 'ffa_rec_tds': 'ffa_rec_td'})

    ffa = calc_total_tds(ffa, 'ffa')
    ffa = ffa.round(2)
    df = pd.merge(df, ffa, on=['player', 'year'], how='left')


    return df

def fantasy_data_proj(df, pos):
    
    fd = dm.read(f'''SELECT * 
                     FROM FantasyData
                     WHERE pos='{pos}' 
                    ''', DB_NAME)
    
    fd = fd.drop(['team', 'pos', 'fdta_int', 'fdta_sack', 'fdta_fum_forced', 'fdta_fum_rec', 'fdta_rank'], axis=1)
    fd = calc_total_tds(fd, 'fdta')
    
    df = pd.merge(df, fd, on=['player', 'year'], how='left')

    return df


def get_pff_proj(df, pos):

    pff = dm.read(f'''SELECT *
                      FROM PFF_Projections
                      WHERE pos='{pos}'
                    ''', DB_NAME)
    pff = pff.drop(['team', 'pos', 'pff_games', 'pff_rank', 'pff_proj_pts', 'pff_auction_value',
                    'pff_fumbles', 'pff_fumbles_lost'], axis=1)
    pff = pff.rename(columns={'pff_rec_receptions': 'pff_rec', 'pff_pass_sacked': 'pff_pass_sacks'})
    pff = calc_total_tds(pff, 'pff')
    df = pd.merge(df, pff, on=['player', 'year'], how='left')

    return df

def fantasy_points_proj(df, pos):
    fpts = dm.read(f'''SELECT * 
                       FROM FantasyPoints_Projections
                       WHERE pos='{pos}' 
                        ''', DB_NAME)
    fpts = fpts.drop(['pos', 'team', 'fpts_overall_rank', 'fpts_adp', 'fpts_games',
                      'fpts_proj_points_per_game', 'fpts_proj_points', 'fpts_tier'], axis=1)
    fpts = fpts.rename(columns={'fpts_pass_cmp': 'fpts_pass_comp'})
    fpts = name_cleanup(fpts)
    fpts = calc_total_tds(fpts, 'fpts')
    df = pd.merge(df, fpts, on=['player', 'year'], how='left')

    return df


def add_adp(df, pos, source, bad_ty_adp=False):
    adp = dm.read(f'''SELECT player, year, avg_pick
                      FROM ADP_Ranks
                      WHERE pos='{pos}'
                            AND source = '{source}'
                      ''', DB_NAME)
    
    if source == 'fpros':
        adp = adp.rename(columns={'avg_pick': 'fpros_avg_pick'})
    
    # early in the year, the QB ADP is not accurate, so we remove it
    if source == 'mfl' and bad_ty_adp:
        adp.loc[adp.year==year, 'avg_pick'] = np.nan
    
    df = pd.merge(df, adp, on=['player', 'year'], how='left')

    return df

def add_etr_rank(df, pos):
    etr = dm.read(f'''SELECT player, year, etr_pos_rank
                      FROM ETR_Ranks
                      WHERE pos='{pos}'
                      ''', DB_NAME)
    
    df = pd.merge(df, etr, on=['player', 'year'], how='left')
    return df

#===================
# Fill Functions
#===================

def get_cols_to_fill(stat_name, data_sources, drop_sources=[]):
    cols = [f'{ds}_{stat_name}' for ds in data_sources if ds not in drop_sources]
    return cols

def consensus_fill(df, pos):

    sources = ['fpros', 'ffa', 'fft', 'fdta', 'pff', 'fpts']

    to_fill = {

        # passing stats
        'proj_pass_yds': get_cols_to_fill('pass_yds', sources),
        'proj_pass_td': get_cols_to_fill('pass_td', sources),
        'proj_pass_int': get_cols_to_fill('pass_int', sources),
        'proj_pass_comp': get_cols_to_fill('pass_comp', sources, drop_sources=['ffa', 'fdta']),
        'proj_pass_att':  get_cols_to_fill('pass_att', sources, drop_sources=['ffa', 'fdta']),
        'proj_pass_int': get_cols_to_fill('pass_int', sources),
        'proj_pass_sacks': get_cols_to_fill('pass_sacks', sources, drop_sources=['ffa', 'fdta', 'fpts', 'fpros']),

        # rushing stats
        'proj_rush_yds': get_cols_to_fill('rush_yds', sources),
        'proj_rush_td': get_cols_to_fill('rush_td', sources),
        'proj_rush_att': get_cols_to_fill('rush_att', sources, drop_sources=['ffa', 'fdta']),

        # receiving stats
        'proj_rec': get_cols_to_fill('rec', sources, drop_sources=['ffa']),
        'proj_rec_yds': get_cols_to_fill('rec_yds', sources),
        'proj_rec_td': get_cols_to_fill('rec_td', sources),
        
        # # touchdowns and proj
        'proj_total_tds': get_cols_to_fill('total_tds', sources),
        'proj_points': get_cols_to_fill('proj_points', sources),
        'pos_rank': get_cols_to_fill('pos_rank', sources) + ['etr_pos_rank'],

        'proj_avg_pick': ['avg_pick', 'fpros_avg_pick']
    }

    for k, tf in to_fill.items():

        if k == 'proj_points': 
            for c in sources: 
                df = calc_proj_pts(df, pos, c)
                df = calc_pos_rank(df, c)

        if pos == 'QB' and 'rec' in k:
            df = df.drop([c for c in tf if c in df.columns], axis=1)
            continue

        elif pos != 'QB' and 'pass' in k:
            df = df.drop([c for c in tf if c in df.columns], axis=1)
            continue

        # find columns that exist in dataset
        tf = [c for c in tf if c in df.columns]

        
        # fill in nulls based on available data
        for c in tf:
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), tf].mean(axis=1)

        # fill in the average for all cols
        df['avg_' + k] = df[tf].mean(axis=1, skipna=True)
        df['std_' + k] = df[tf].std(axis=1, skipna=True)

        global_cv = (df.loc[df['std_' + k] != 0, 'std_' + k] / df.loc[df['std_' + k] != 0, 'avg_' + k]).mean()
        df.loc[df['std_' + k] == 0, 'std_' + k] = df.loc[df['std_' + k] == 0, 'avg_' + k] * global_cv

        df['avg_' + k] = df['avg_' + k].round(2)
        df['std_' + k] = df['std_' + k].round(2)

    df['etr_pos_rank_log'] = np.log(df['etr_pos_rank'] + 1)

    return df

def calc_detailed_stats(df, pos):

    df['avg_proj_rush_points'] = df.avg_proj_rush_yds * 0.1 + df.avg_proj_rush_td * 7
    df['avg_proj_rush_yds_per_att'] = df.avg_proj_rush_yds / (df.avg_proj_rush_att + 1)
    df['avg_proj_rush_td_per_att'] = df.avg_proj_rush_td / (df.avg_proj_rush_att + 1)

    if pos == 'QB': 
        df['avg_proj_pass_points'] = df.avg_proj_pass_yds * 0.04 + df.avg_proj_pass_td * 4 - df.avg_proj_pass_int * 1
        df['avg_proj_pass_cmp_pct'] = df.avg_proj_pass_comp / (df.avg_proj_pass_att + 1)
        df['avg_proj_pass_td_per_att'] = df.avg_proj_pass_td / (df.avg_proj_pass_att + 1)
        df['avg_proj_pass_yds_per_att'] = df.avg_proj_pass_yds / (df.avg_proj_pass_att + 1)

        df['avg_proj_rush_pass_points_ratio'] = df.avg_proj_pass_points / (df.avg_proj_pass_points + df.avg_proj_rush_points + 1)
        df['avg_proj_rush_pass_att_ratio'] = df.avg_proj_rush_att / (df.avg_proj_pass_att + 10)
        df['avg_proj_rush_pass_yds_ratio'] = df.avg_proj_rush_yds / (df.avg_proj_pass_yds + 100)
        df['avg_proj_rush_pass_td_ratio'] = df.avg_proj_rush_td / (df.avg_proj_pass_td + 1)

    else: 
        df['avg_proj_rec_points'] = df.avg_proj_rec_yds * 0.1 + df.avg_proj_rec_td * 7 + df.avg_proj_rec * 0.5
        df['avg_proj_rec_yds_per_rec'] = df.avg_proj_rec_yds / (df.avg_proj_rec + 1)
        df['avg_proj_rec_td_per_rec'] = df.avg_proj_rec_td / (df.avg_proj_rec + 1)
        df['avg_proj_rush_ratio'] = df.avg_proj_rush_points / (df.avg_proj_rec_points + df.avg_proj_rush_points + 1)

    return df

def fill_avg_pick(df, col):
    from sklearn.linear_model import LinearRegression
    X = df.loc[(df[col].notnull()) & (df[col]<250), ['avg_pos_rank']].values
    y = df.loc[(df[col].notnull()) & (df[col]<250), col].values
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    df.loc[df[col].isnull(), col] = lr.predict(df.loc[df[col].isnull(), ['avg_pos_rank']].values)
    df.loc[df[col]<0, col] = df.loc[df[col]<0, 'avg_pos_rank']*2
    df[col+'_log'] = np.log(df[col] + 1)
    return df

def fill_pff_targets(df):
    X = df.loc[df.pff_rec_targets.notnull(), ['avg_proj_rec_yds', 'avg_proj_rec_td', 'avg_proj_rec']]
    y = df.loc[df.pff_rec_targets.notnull(), 'pff_rec_targets']
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X, y)
    df.loc[df.pff_rec_targets.isnull(), 'pff_rec_targets'] = lm.predict(df.loc[df.pff_rec_targets.isnull(), ['avg_proj_rec_yds', 'avg_proj_rec_td', 'avg_proj_rec']])
    return df

def rolling_proj_stats(df):
    df = forward_fill(df)
    proj_cols = [c for c in df.columns if 'proj_points' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)
    df = df.fillna(0)

    return df


def remove_non_uniques(df):
    cols = df.nunique()[df.nunique()==1].index
    cols = [c for c in cols if c != 'pos']
    df = df.drop(cols, axis=1)
    return df


def create_pos_rank(df, extra_pos=False):
    df = df.sort_values(by=['team', 'pos', 'year', 'avg_proj_points'],
                        ascending=[True, True, True, False]).reset_index(drop=True)

    df['pos_rank'] = df.pos + df.groupby(['team', 'pos', 'year']).cumcount().apply(lambda x: str(x))
    if extra_pos:
        df = df[df['pos_rank'].isin(['RB0', 'RB1', 'RB2', 'WR0', 'WR1', 'WR2', 'WR3', 'WR4', 'TE0', 'TE1'])].reset_index(drop=True)
    else:
        df = df[df['pos_rank'].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0'])].reset_index(drop=True)
    return df


def get_team_projections():

    team_proj = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE']:

        df = fftoday_proj(pos)
        df = drop_duplicate_players(df, 'fft_proj_points')
        df = fantasy_pros_new(df, pos)
        df = get_pff_proj(df, pos)
        df = ffa_compile_stats(df, pos)
        df = fantasy_data_proj(df, pos)
        df = fantasy_points_proj(df, pos)
        df = add_etr_rank(df, pos)

        df = consensus_fill(df, pos)

        df = add_adp(df, pos, 'mfl', bad_adps)
        df = add_adp(df, pos, 'fpros', bad_adps)
        df = fill_avg_pick(df, 'avg_pick')
        df = fill_avg_pick(df, 'fpros_avg_pick')

        if pos != 'QB': df = fill_pff_targets(df)
        df = df.dropna(axis=1)
        team_proj = pd.concat([team_proj, df], axis=0)

    team_proj = create_pos_rank(team_proj)
    team_proj = forward_fill(team_proj)
    team_proj = team_proj.fillna(0)

    cnts = team_proj.groupby(['team', 'year']).agg({'avg_proj_points': 'count'})
    print('Team counts that do not equal 7:', cnts[cnts.avg_proj_points!=7])

    sources = ['fft', 'fpros', 'ffa', 'fdta', 'pff', 'fpts', 'avg_proj']
    cols = []
    cols.extend(get_cols_to_fill('pass_yds', sources))
    cols.extend(get_cols_to_fill('pass_td', sources))
    cols.extend(get_cols_to_fill('pass_int', sources))
    cols.extend(get_cols_to_fill('pass_comp', sources, drop_sources=['ffa', 'fdta']))
    cols.extend(get_cols_to_fill('pass_att', sources, drop_sources=['ffa', 'fdta']))
    cols.extend(get_cols_to_fill('pass_sacks', sources, drop_sources=['ffa', 'fdta', 'fpts', 'fpros']))
    cols.extend(get_cols_to_fill('rush_yds', sources))
    cols.extend(get_cols_to_fill('rush_td', sources))
    cols.extend(get_cols_to_fill('rush_att', sources, drop_sources=['ffa', 'fdta']))
    cols.extend(get_cols_to_fill('rec', sources, drop_sources=['ffa']))
    cols.extend(get_cols_to_fill('rec_yds', sources))
    cols.extend(get_cols_to_fill('rec_td', sources))
    cols.extend(get_cols_to_fill('total_tds', sources))
    cols.extend(get_cols_to_fill('proj_points', sources))
    cols = [c.replace('avg_proj_proj_points', 'avg_proj_points') for c in cols]
    to_agg = {c: 'sum' for c in cols}


    # get the projections broken out by RB and WR/TE
    team_proj_pos = team_proj[team_proj.pos.isin(['RB', 'WR', 'TE'])].copy()
    team_proj_pos.loc[team_proj_pos.pos=='TE', 'pos'] = 'WR'
    team_proj_pos = team_proj_pos.groupby(['pos', 'team', 'year']).agg(to_agg)
    team_proj_pos.columns = [f'pos_proj_{c}' for c in team_proj_pos.columns]
    team_proj_pos = team_proj_pos.reset_index()
    team_proj_pos_te = team_proj_pos[team_proj_pos.pos=='WR'].copy()
    team_proj_pos_te['pos'] = 'TE'
    team_proj_pos = pd.concat([team_proj_pos, team_proj_pos_te], axis=0).reset_index(drop=True)
    
    # get the projections broken out by team
    team_proj = team_proj.groupby(['team', 'year']).agg(to_agg)
    team_proj.columns = [f'team_proj_{c}' for c in team_proj.columns]
    team_proj = team_proj.reset_index()

    return team_proj, team_proj_pos

def calc_market_share(df):
    
    player_cols = ['avg_proj_rush_att', 'avg_proj_rush_yds', 'avg_proj_rush_td', 
                   'avg_proj_rec', 'avg_proj_rec_yds', 'avg_proj_rec_td', 'avg_proj_points']

    team_cols = ['team_' + c for c in player_cols]

    for p, t in zip(player_cols, team_cols):
        df[p+'_share'] = df[p] / (df[t]+0.5)

    share_cols = [c+'_share' for c in player_cols]
    df[share_cols] = df[share_cols].fillna(0)
    df = add_rolling_stats(df, gcols=['player'], rcols=share_cols)

    df = forward_fill(df)
    share_cols = [c for c in df.columns if 'share' in c]
    df[share_cols] = df[share_cols].fillna(0)

    return df

def proj_market_share(df, proj_col_name):

    proj_cols = [c for c in df.columns if proj_col_name in c]

    for proj_col in proj_cols:
        orig_col = proj_col.replace(proj_col_name, '')
        if orig_col in df.columns:
            df[f'{proj_col_name}share_{orig_col}'] = df[orig_col] / (df[proj_col]+3)
            df[f'{proj_col_name}share_diff_{orig_col}'] = df[orig_col] - df[proj_col]
            df[[f'{proj_col_name}share_{orig_col}', f'{proj_col_name}share_diff_{orig_col}']] = \
                df[[f'{proj_col_name}share_{orig_col}', f'{proj_col_name}share_diff_{orig_col}']].fillna(0)
    return df

def get_pick_vs_team_stats(df):
    pick_rank_cols = ['avg_pick_log', 'avg_pos_rank', 'fpros_avg_pick_log']
    team_adp = df.groupby(['team', 'pos', 'year']).agg({r: ['sum', 'mean', 'min', 'max'] for r in pick_rank_cols}).reset_index()
    team_adp.columns = ['team_' + c[0] + '_' + c[1] if c[1]!='' else c[0] for c in team_adp.columns]
    df = pd.merge(df, team_adp, on=['team', 'pos', 'year'])

    for c in pick_rank_cols:
        df[f'mean_diff_{c}'] = df[c] - df[f'team_{c}_mean']
        df[f'min_diff_{c}'] = df[c] - df[f'team_{c}_min']
        df[f'sum_frac_{c}'] = df[c] / df[f'team_{c}_sum']

    return df


def get_max_qb():

    pos='QB'

    df = fftoday_proj(pos)
    df = drop_duplicate_players(df, 'fft_proj_points')
    df = fantasy_pros_new(df, pos)
    df = get_pff_proj(df, pos)
    df = ffa_compile_stats(df, pos)
    df = fantasy_data_proj(df, pos)
    df = fantasy_points_proj(df, pos)
    df = add_etr_rank(df, pos)

    df = consensus_fill(df, pos)

    df = add_adp(df, pos, 'mfl', bad_adps)
    df = add_adp(df, pos, 'fpros', bad_adps)
    df = fill_avg_pick(df, 'avg_pick')
    df = fill_avg_pick(df, 'fpros_avg_pick')
    df = df.dropna(axis=1)

    sources = ['avg_proj']
    cols = ['team', 'year']
    cols.extend(get_cols_to_fill('pass_yds', sources))
    cols.extend(get_cols_to_fill('pass_td', sources))
    cols.extend(get_cols_to_fill('pass_int', sources))
    cols.extend(get_cols_to_fill('pass_comp', sources, drop_sources=['ffa', 'fdta']))
    cols.extend(get_cols_to_fill('pass_att', sources, drop_sources=['ffa', 'fdta']))
    cols.extend(get_cols_to_fill('pass_sacks', sources, drop_sources=['ffa', 'fdta', 'fpts', 'fpros']))
    cols.extend(get_cols_to_fill('rush_yds', sources))
    cols.extend(get_cols_to_fill('rush_td', sources))
    cols.extend(get_cols_to_fill('rush_att', sources, drop_sources=['ffa', 'fdta']))
    cols.extend(get_cols_to_fill('total_tds', sources))
    cols.extend(get_cols_to_fill('proj_points', sources))
    cols = [c.replace('avg_proj_proj_points', 'avg_proj_points') for c in cols]

    df = df.sort_values(by=['team', 'year', 'avg_proj_points'],
                        ascending=[True, True, False])
    df = df.drop_duplicates(subset=['team', 'year'], keep='first').reset_index(drop=True)
    df = df[cols]
    df = df.dropna(axis=1)
    df = calc_detailed_stats(df, pos)
    df.columns = ['qb_'+c if c not in ('team', 'year') else c for c in df.columns]
    df = remove_non_uniques(df)

    return df

def add_pff_stats(df, pos, stat):

    if pos == 'RB': pos = 'HB'

    pff = dm.read(f'''SELECT *
                      FROM PFF_{stat}_Stats
                      WHERE position='{pos}'
                      ''', DB_NAME).drop(['player_id', 'position'], axis=1)
    
    pff = pff[~((pff.player == 'Adrian Peterson') & (pff.team_name=='CHI'))].reset_index(drop=True)
    pff = pff[~((pff.player == 'Steve Smith') & (pff.team_name.isin(['NYG', 'PHI', 'SL'])))].reset_index(drop=True)
    pff = pff[~((pff.player == 'Mike Williams') & (pff.year < 2017))].reset_index(drop=True)

    pff = pff.rename(columns={'year': 'season'}).drop('team_name', axis=1)
    pff.columns = [f'pff_{stat.lower()}_{c}' if c not in ('player', 'season') else c for c in pff.columns ]
    df = pd.merge(df, pff, on=['player', 'season'], how='left')
    df[[c for c in df.columns if f'pff_{stat.lower()}' in c]] = df[[c for c in df.columns if f'pff_{stat.lower()}' in c]].fillna(0)

    if stat == 'Rec':
        roll_cols = ['grades_offense', 'grades_pass_route', 'slot_rate', 'slot_snaps', 'wide_snaps', 'targeted_qb_rating',
                    'yprr', 'avg_depth_of_target', 'grades_hands_drop', 'route_rate', 'wide_rate']
    
    if stat == 'Rush':
        roll_cols = ['attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards',
                     'designed_yards', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive',
                     'first_downs', 'gap_attempts', 'grades_offense', 'grades_pass_block','grades_pass_route', 'grades_run',
                     'yards_after_contact', 'yco_attempt', 'ypa','zone_attempts']
    
    if stat == 'QB':
        roll_cols = ['accuracy_percent', 'avg_depth_of_target', 'avg_time_to_throw', 'bats', 'big_time_throws',
                     'btt_rate', 'grades_offense', 'def_gen_pressures', 'drop_rate', 'dropbacks', 'grades_pass',
                     'grades_run', 'hit_as_threw', 'pressure_to_sack_rate', 'qb_rating', 'sack_percent', 'scrambles',
                     'thrown_aways', 'turnover_worthy_plays', 'twp_rate', 'ypa']
        
    roll_cols = [f'pff_{stat.lower()}_{c}' for c in roll_cols]
    df = add_rolling_stats(df, ['player'], roll_cols)

    return df

def add_team_stat_share(df):
    team_stats = dm.read(f'SELECT * FROM Team_Stats', DB_NAME)
    df = pd.merge(df, team_stats, on=['team', 'season', 'year'])
    for c in team_stats.columns:
        if c not in ['team', 'year', 'season']:
            try:df[f'frac_{c}'] = df[c.replace('team_', '')] / (df[c] + 1.001)
            except: print(c)

    return df

def add_qb_yoy_stats(df):

    team_stats = dm.read(f'SELECT * FROM Team_Stats', DB_NAME)
    team_stats = team_stats[['year', 'team', 'team_sum_rec_yards_gained_sum', 'team_sum_rec_touchdown_sum']]
    team_stats.columns = ['year', 'last_year_team', 'last_year_team_rec_yards', 'last_year_team_rec_tds']

    df = df.sort_values(by=['player', 'year']).reset_index(drop=True)
    df.team = df.team.apply(lambda x: x.lstrip().rstrip())
    df['last_year_team'] = df.groupby('player').team.shift(1)
    df.loc[df.last_year_team.isnull(), 'last_year_team'] = df.loc[df.last_year_team.isnull(), 'team']

    df = pd.merge(df, team_stats, on=['last_year_team', 'year'], how='left')

    df['qb_this_year_vs_last_yards'] = df.qb_avg_proj_pass_yds - df.last_year_team_rec_yards
    df['qb_this_year_vs_last_tds'] = df.qb_avg_proj_pass_td - df.last_year_team_rec_tds
    df = df.drop(['last_year_team'], axis=1)

    return df

def add_draft_year_exp(df, pos):

    draft_year = dm.read(f'''
                        SELECT player, 
                                team, 
                                year draft_year, 
                                Round draft_round, 
                                Pick draft_pick 
                        FROM Draft_Positions 
                        WHERE pos='{pos}'
                        ''', DB_NAME)
    draft_year.team = draft_year.team.map(team_map)
    df = pd.merge(df, draft_year, on=['player', 'team'], how='left')
    df = df.sort_values(by=['player', 'year']).reset_index(drop=True)
    df[['draft_year', 'draft_round', 'draft_pick']] = df.groupby(['player'])[['draft_year', 'draft_round', 'draft_pick']].ffill().values

    min_year = df.groupby('player')['season'].min().reset_index().rename(columns={'season': 'min_year'})
    df = pd.merge(df, min_year, on='player')
    df = df.fillna({'draft_year': df.min_year, 'draft_round': 7, 'draft_pick': 300}).drop('min_year', axis=1)
    df['year_exp'] = df.year - df.draft_year

    df.loc[df.year_exp < 0, 'year_exp'] = 0
    high_yr = np.percentile(df.year_exp.dropna(), 95)
    df.loc[df.year_exp > high_yr, 'year_exp'] = high_yr

    return df

def drop_duplicate_players(df, order_col, rookie=False):
    df = df.sort_values(by=['player', 'year', order_col],
                    ascending=[True, True,  False])
    if rookie: df = df.drop_duplicates(subset=['player'], keep='first').reset_index(drop=True)
    else: df = df.drop_duplicates(subset=['player', 'year'], keep='first').reset_index(drop=True)
    
    return df

def remove_low_corrs(df, corr_cut = 3, collinear_cut = 0.995):
    obj_cols = list(df.dtypes[df.dtypes=='object'].index)
    obj_cols.extend(['year', 'pos', 'games', 'season', 'games_next', 'year_exp', 'avg_pick',
                     'avg_proj_points', 'avg_proj_rush_points', 'avg_proj_pass_points', 'avg_proj_rec_points',
                     'avg_proj_points_exp', 'avg_proj_points_exp_diff', 'avg_proj_points_exp_diff', 'avg_pick_exp', 'avg_pick_exp_diff',
                     'avg_proj_rank'])
    obj_cols.extend([c for c in df.columns if 'y_act_' in c])
    obj_cols = [c for c in obj_cols if c in df.columns]

    orig_shape = df.shape[1]
    skm = SciKitModel(df, model_obj='reg')
    X, y = skm.Xy_split('y_act', to_drop = obj_cols)
    corr_collin = skm.piece('corr_collinear')[-1]
    corr_collin.set_params(**{'corr_percentile': corr_cut, 'collinear_threshold': collinear_cut})
    X = corr_collin.fit_transform(X, y)
    new_shape = X.shape[1]

    print(f'Removed {orig_shape - new_shape} / {orig_shape} columns')
    df = pd.concat([df[obj_cols], X], axis=1)

    corrs = pd.DataFrame(np.corrcoef(pd.concat([X,y],axis=1).values, rowvar=False), 
                         columns=pd.concat([X,y],axis=1).columns,
                         index=pd.concat([X,y],axis=1).columns)
    
    corrs = corrs['y_act']
    corrs = corrs.dropna().sort_values()
    display(corrs.iloc[:20])
    display(corrs.iloc[-20:])
    
    return df

def drop_y_act_except_current(df, year):
    
    df = df[~(df.y_act.isnull()) | (df.year==year)].reset_index(drop=True)
    df.loc[(df.year==year), [c for c in df.columns if 'y_act' in c]] = 0

    return df

def drop_games(df, year, games=0, games_next=0):
    
    df = df[((df.games >= games) & \
             (df.games_next >= games_next)) | \
            #  (df.year==year)
             ((df.year==year) & \
              (df.games >= games))
              ].reset_index(drop=True)
    return df

def y_act_class(df, df_quant, rush_pass, gm_filter, proj_var_cut, y_act_cut, suffix):

    pos_cut = {
        'WR': 48,
        'RB': 36,
        'QB': 18,
        'TE': 18
    }

    if rush_pass == 'rush': 
        y_act = 'y_act_rush'
        suffix = suffix + '_rush'
        proj_col = 'avg_proj_rush_points'
    elif rush_pass == 'pass': 
        y_act = 'y_act_pass'
        suffix = suffix + '_pass'
        proj_col = 'avg_proj_pass_points'
        proj_var_cut -= 0.1
    elif rush_pass == 'rec': 
        y_act = 'y_act_rec'
        suffix = suffix + '_rec'
        proj_col = 'avg_proj_rec_points'
    else: 
        y_act = 'y_act'
        proj_col = 'avg_proj_points'
    
    df[f'{proj_col}_per_game'] = np.where(df.year >= 2021, df[proj_col] / 16, df[proj_col] / 15)
    df_quant[f'{proj_col}_per_game'] = np.where(df_quant.year >= 2021, df_quant[proj_col] / 16, df_quant[proj_col] / 15)

    df_quant = df_quant[df_quant[gm_filter] > 6].sort_values(by=['year', y_act], ascending=[True, False]).reset_index(drop=True)
    df_quant['include'] = df_quant.groupby('year').cumcount().values
    df_quant = df_quant[df_quant.include < pos_cut[pos]].reset_index(drop=True)
    df_quant['proj_var'] = df_quant[y_act] - df_quant[f'{proj_col}_per_game']   

    df_proj_quant = df_quant.groupby('year')['proj_var'].quantile(proj_var_cut).reset_index()
    df_act_quant = df_quant.loc[df_quant.include == y_act_cut-1, ['year', y_act]].rename(columns={y_act: 'y_act_quantile'})

    print(df_act_quant)
    print(df_proj_quant)

    df = pd.merge(df, df_proj_quant, on='year', how='left')
    df = pd.merge(df, df_act_quant, on='year', how='left')
    
    df[f'y_act_class_{suffix}'] = 0
    df.loc[(df[y_act] > df[f'{proj_col}_per_game'] + df.proj_var) & \
           (df[gm_filter] > 6) & \
           (df[y_act] >= df.y_act_quantile),
           f'y_act_class_{suffix}'] = 1
    df = df.drop(['proj_var', 'y_act_quantile'], axis=1)

    return df


def show_calibration_curve(y_true, y_pred, n_bins=10):
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    plt.plot(y, x, marker = '.', label = 'Quantile')

    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    plt.plot(y, x, marker = '+', label = 'Uniform')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()

    print('Brier Score:', brier_score_loss(y_true, y_pred))


def add_year_exp_compare(df):
    all_year_exp_df = pd.DataFrame()
    for i, yr in enumerate(df.year.sort_values().unique()):

        year_exp_cols = [
            'fpros_proj_points', 'pff_proj_points', 'avg_proj_points', 'fft_proj_points', 'fpts_proj_points', 'ffa_proj_points',
            'avg_pick_log', 'pff_pos_rank_log', 'ffa_pos_rank_log', 'fpros_avg_pick_log', 'etr_pos_rank_log', 'fdta_pos_rank_log', 
            'mean_diff_avg_pick_log', 'sum_frac_avg_pos_rank', 'min_diff_avg_pick_log',
            
            'fdta_rush_yds', 'pff_rush_yds', 
            'fpros_rush_yds', 'pff_rush_td','fpts_total_tds', 'ffa_total_tds',
            
            'fpros_rec_yds', 'fdta_rec_yds', 'pff_rec_yds', 'ffa_rec_yds', 'avg_proj_rec_yds',  'avg_proj_rec_points',
            'fpros_rec', 'fpts_rec', 'pff_rec',
            'fft_rec_td', 'pff_rec_td', 'fpros_rec_td', 'fdta_rec_td',
            
            'pff_pass_yds', 'fpros_pass_yds', 
            'pff_pass_td', 'avg_proj_pass_td',
            'fft_pass_yds_per_cmp','pff_pass_td_per_att',
            
            'team_proj_share_avg_proj_points', 'team_proj_share_pff_proj_points', 'team_proj_share_ffa_proj_points',
            'team_proj_share_fpros_proj_points', 'team_proj_share_fft_proj_points', 'team_proj_share_fpts_proj_points',
            
            'team_proj_share_avg_proj_rec_yds', 'team_proj_share_ffa_rec_yds', 'pos_proj_share_avg_proj_rec_yds', 
            'pos_proj_share_fpros_total_tds', 'team_proj_share_fpros_total_tds',
            
            'fantasy_pts_per_game', 'mean_fantasy_pts', 'mean_rec_first_down_sum', 'sum_pass_epa_sum', 'sum_pass_epa_mean']
        
        year_exp_cols = [c for c in year_exp_cols if c in df.columns]
        if i <= 1: year_exp = df[df.year<=yr+1].groupby('year_exp').agg({ye: 'mean' for ye in year_exp_cols}).reset_index()
        else: year_exp = df[df.year<=yr].groupby('year_exp').agg({ye: 'mean' for ye in year_exp_cols}).reset_index()
        year_exp.columns = [c+'_exp' if c != 'year_exp' else c for c in year_exp.columns]
        year_exp['year'] = yr
        all_year_exp_df = pd.concat([all_year_exp_df, year_exp], axis=0)
    
    df = pd.merge(df, all_year_exp_df, on=['year_exp', 'year'], how='left')

    # oldest players have no year_exp equivalent so forward fill
    col_order = df.columns
    year_exp_cols_fill = ['player', 'year']
    year_exp_cols_fill.extend([c for c in df.columns if '_exp' in c])
    df = forward_fill(df, df.columns)
    df = df[col_order]

    for c in year_exp_cols:
        c = c.replace('_exp', '')
        df[f'{c}_exp_diff'] = df[c] - df[f'{c}_exp']
        df[f'{c}_exp_frac'] = df[c] / (df[f'{c}_exp'] + 0.001)

    df = add_rolling_stats(df, ['player'], [c for c in df.columns if '_exp' in c])

    return df


def get_next_year_stats(df, stats_proj, ty_mean=False, is_rookie=False):

    # create next year's stats
    df_proj_next = df.copy()

    if pos == 'QB': 
        stats_proj = stats_proj[['player', 'year', 'y_act', 'y_act_rush', 'y_act_pass']]
        df_proj_next = df_proj_next.drop(['y_act_rush', 'y_act_pass'], axis=1)
    elif pos == 'RB' and not is_rookie:
        stats_proj = stats_proj[['player', 'year', 'y_act', 'y_act_rush', 'y_act_rec']]
        df_proj_next = df_proj_next.drop(['y_act_rush', 'y_act_rec'], axis=1)
    elif pos == 'RB' and is_rookie:
        stats_proj = stats_proj[['player', 'year', 'y_act']]
    else:
        stats_proj = stats_proj[['player', 'year', 'y_act']]

    stats_proj.year = stats_proj.year - 1
    stats_proj = stats_proj.rename(columns={'y_act': 'y_act_next'})
    df_proj_next = pd.merge(stats_proj, df_proj_next, on=['player', 'year'], how='right')

    
    if ty_mean:
        df_proj_next['y_act'] = df_proj_next[['y_act', 'y_act_next']].mean(axis=1)
        games = 4
        games_next = 4
    else:
        df_proj_next['y_act'] = df_proj_next['y_act_next']
        games = 0
        games_next = 4
    
    df_proj_next = df_proj_next.drop('y_act_next', axis=1)

    return df_proj_next, games, games_next


#%%

pos='WR'

bad_adps = True

class_cuts = {
    'WR': {
           'upside': {'y_act': 24, 'proj_var': 0.7},
           'top':  {'y_act': 12, 'proj_var': 0.45}
        },
    
    'RB': {
           'upside': {'y_act': 24, 'proj_var': 0.7},
           'top':  {'y_act': 12, 'proj_var': 0.45}
        },
    
    'QB': {
           'upside': {'y_act': 12, 'proj_var': 0.7},
           'top':  {'y_act': 6, 'proj_var': 0.45}
        },

    'TE': {
           'upside': {'y_act': 12, 'proj_var': 0.7},
           'top':  {'y_act': 6, 'proj_var': 0.45}
        },

    'Rookie_WR': {
           'upside': {'y_act': 24, 'proj_var': 0.7},
           'top':  {'y_act': 12, 'proj_var': 0.45}
        },

    'Rookie_RB': {
           'upside': {'y_act': 24, 'proj_var': 0.7},
           'top':  {'y_act': 12, 'proj_var': 0.45}
        },
}


#==============
# Get Projections and Stats
#==============

# pull all projections and ranks
df = fftoday_proj(pos); print(df.shape[0])
df = drop_duplicate_players(df, 'fft_proj_points')
df = fantasy_pros_new(df, pos); print(df.shape[0])
df = get_pff_proj(df, pos); print(df.shape[0])
df = ffa_compile_stats(df, pos); print(df.shape[0])
df = fantasy_data_proj(df, pos); print(df.shape[0])
df = fantasy_points_proj(df, pos); print(df.shape[0])
df = add_etr_rank(df, pos); print(df.shape[0])

#==============
# Add ADP and fill in consensus
#==============

df = add_adp(df, pos, 'mfl', bad_adps); print(df.shape[0])
df = add_adp(df, pos, 'fpros', bad_adps); print(df.shape[0])

df = consensus_fill(df, pos); print(df.shape[0])
df = calc_detailed_stats(df, pos)

df = fill_avg_pick(df, 'avg_pick'); print(df.shape[0])
df = fill_avg_pick(df, 'fpros_avg_pick'); print(df.shape[0])

if pos != 'QB': df = fill_pff_targets(df); print(df.shape[0])
else: df = df.drop('pff_rec_targets', axis=1, errors='ignore')

#==============
# Clean Up Data and Add Rolling
#==============

df = df.dropna(axis=1)
df = remove_non_uniques(df); print(df.shape[0])
df = drop_duplicate_players(df, 'avg_proj_points')
df = rolling_proj_stats(df); print(df.shape[0])

#==============
# Get Team Projections and Market Share
#==============
team_proj, team_proj_pos = get_team_projections()
df = pd.merge(df, team_proj, on=['team',  'year']); print( df.shape[0])

if pos != 'QB':
    df = pd.merge(df, team_proj_pos, on=['pos', 'team', 'year']); print( df.shape[0])
    df = proj_market_share(df, 'team_proj_'); print(df.shape[0])
    df = proj_market_share(df, 'pos_proj_'); print(df.shape[0])
    df = remove_non_uniques(df); print(df.shape[0])

    df = get_pick_vs_team_stats(df)
    max_qb = get_max_qb()
    df = pd.merge(df, max_qb, on=['team', 'year']); print(df.shape[0])

    df = add_qb_yoy_stats(df)

df = drop_duplicate_players(df, 'avg_proj_points')
df.sort_values(by=['year', 'avg_proj_points'],
                    ascending=[False, False]).reset_index(drop=True)

#%%

#-----------
# Save out Projection data
#-----------
stats_proj = dm.read(f'SELECT * FROM {pos}_Stats', DB_NAME)
stats_proj = drop_duplicate_players(stats_proj, 'sum_fantasy_pts')

if pos =='QB':
    stat_cols = ['player', 'season', 'year', 'games', 'games_next', 'fantasy_pts_per_game',
                  'fantasy_pts_rush_per_game', 'fantasy_pts_pass_per_game']
    stats_proj = stats_proj[stat_cols].rename(columns={'fantasy_pts_per_game': 'y_act', 
                                                       'fantasy_pts_rush_per_game': 'y_act_rush',
                                                       'fantasy_pts_pass_per_game': 'y_act_pass',
                                                       })
elif pos == 'RB':
    stat_cols = ['player', 'season', 'year', 'games', 'games_next', 'fantasy_pts_per_game',
                  'fantasy_pts_rush_per_game', 'fantasy_pts_rec_per_game']
    stats_proj = stats_proj[stat_cols].rename(columns={'fantasy_pts_per_game': 'y_act', 
                                                       'fantasy_pts_rush_per_game': 'y_act_rush',
                                                       'fantasy_pts_rec_per_game': 'y_act_rec',
                                                       })
else:
    stat_cols = ['player', 'season', 'year', 'games', 'games_next', 'fantasy_pts_per_game']
    stats_proj = stats_proj[stat_cols].rename(columns={'fantasy_pts_per_game': 'y_act'})

stats_proj[['season', 'year']] = stats_proj[['season', 'year']] - 1

df_proj = pd.merge(stats_proj, df, on=['player', 'year'], how='right')
df_proj.loc[(df_proj.year==year) & (df_proj.games.isnull()), ['season', 'games']] = [year-1, 16]


df_proj_next, games, games_next = get_next_year_stats(df_proj, stats_proj, ty_mean=False)

# fill in next years games that don't matter for this year after creating copy for next year
df_proj.loc[df_proj.games_next.isnull(), 'games_next'] = 16

# fill in last year's games for next year stats that don't matter
df_proj_next['games'] = 16
df_proj_next.loc[(df_proj_next.year==year) & (df_proj_next.games_next.isnull()), 'games_next'] = 16

df_proj = add_draft_year_exp(df_proj, pos)
df_proj = add_year_exp_compare(df_proj)

df_proj_next = add_draft_year_exp(df_proj_next, pos)
df_proj_next = add_year_exp_compare(df_proj_next)

for df_cur, lbl, gms, gms_next in ([df_proj, '', 4, 0], [df_proj_next, '_next', games, games_next]):
    # only drop games and not games_next, since the "season" was shifted to year
    # meaning that games will be for the games played in the projection year
    df_cur = drop_y_act_except_current(df_cur, year)
    df_cur = drop_games(df_cur, year, games=gms, games_next=gms_next)
    df_cur = remove_low_corrs(df_cur, corr_cut=3, collinear_cut=0.995)

    df_cur = y_act_class(df_cur, df_cur.copy(), 'both', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
    df_cur = y_act_class(df_cur, df_cur.copy(), 'both', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')
    if pos == 'QB':
        df_cur = y_act_class(df_cur, df_cur.copy(), 'rush', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
        df_cur = y_act_class(df_cur, df_cur.copy(), 'rush', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')
        df_cur = y_act_class(df_cur, df_cur.copy(), 'pass', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
        df_cur = y_act_class(df_cur, df_cur.copy(), 'pass', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')

    if pos == 'RB':
        df_cur = y_act_class(df_cur, df_cur.copy(), 'rush', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
        df_cur = y_act_class(df_cur, df_cur.copy(), 'rush', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')
        df_cur = y_act_class(df_cur, df_cur.copy(), 'rec', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
        df_cur = y_act_class(df_cur, df_cur.copy(), 'rec', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')

    df_cur = df_cur.loc[:,~df_cur.columns.duplicated()].copy()
    dm.write_to_db(df_cur, f'Model_Inputs{lbl}', f'{pos}_{year}_ProjOnly', if_exist='replace')

#------------
# Save out Full Data
#------------

stats = dm.read(f'SELECT * FROM {pos}_Stats', DB_NAME)
stats = drop_duplicate_players(stats, 'sum_fantasy_pts')

stat_roll_cols = [c for c in stats.columns if ('pass' in c or 'rush' in c or 'rec' in c) and 'y_act' not in c]
stats = add_rolling_stats(stats, ['player'], stat_roll_cols)
stats = remove_non_uniques(stats)
df_stats = pd.merge(stats, df, on=['player', 'year']); print(df.shape[0])
df_stats = add_team_stat_share(df_stats)

if pos == 'RB':
    df_stats = add_pff_stats(df_stats, pos, 'Rec')
    df_stats = add_pff_stats(df_stats, pos, 'Rush')
elif pos in ('WR', 'RB'):
    df_stats = add_pff_stats(df_stats, pos, 'Rec')
elif pos == 'QB':
    df_stats = add_pff_stats(df_stats, pos, 'QB')
    df_stats = add_pff_stats(df_stats, pos, 'Rush')

df_stats = drop_duplicate_players(df_stats, 'sum_fantasy_pts')

df_stats = add_draft_year_exp(df_stats, pos)
df_stats = add_year_exp_compare(df_stats)

# drop both last year's and next year's game limits
# since you need y_act to be relevant and last year's stats to be relvant
df_stats = drop_y_act_except_current(df_stats, year)
df_stats = drop_games(df_stats, year, games=4, games_next=4)
df_stats = remove_low_corrs(df_stats, corr_cut=4, collinear_cut=0.99)
df_stats = y_act_class(df_stats, df_stats.copy(), 'both', 'games_next', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
df_stats = y_act_class(df_stats, df_stats.copy(), 'both', 'games_next', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')

if pos == 'QB':
    df_stats = y_act_class(df_stats, df_stats.copy(), 'rush', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
    df_stats = y_act_class(df_stats, df_stats.copy(), 'rush', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')
    df_stats = y_act_class(df_stats, df_stats.copy(), 'pass', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
    df_stats = y_act_class(df_stats, df_stats.copy(), 'pass', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')

if pos == 'RB':
    df_stats = y_act_class(df_stats, df_stats.copy(), 'rush', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
    df_stats = y_act_class(df_stats, df_stats.copy(), 'rush', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')
    df_stats = y_act_class(df_stats, df_stats.copy(), 'rec', 'games', class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
    df_stats = y_act_class(df_stats, df_stats.copy(), 'rec', 'games', class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')

df_stats = df_stats.loc[:,~df_stats.columns.duplicated()].copy()
if df_stats.shape[1] > 2000:
    dm.write_to_db(df_stats.iloc[:,:2000], 'Model_Inputs', f'{pos}_{year}_Stats', if_exist='replace')
    dm.write_to_db(df_stats.iloc[:,2000:], 'Model_Inputs', f'{pos}_{year}_Stats_V2', if_exist='replace')
else:
    dm.write_to_db(df_stats, 'Model_Inputs', f'{pos}_{year}_Stats', if_exist='replace')

#------------
# Save out Rookie Data
#------------
if pos in ('RB', 'WR'):

    df_rookie = dm.read(f'''SELECT * 
                            FROM Rookie_{pos}_Stats
                            ''', DB_NAME).rename(columns={'draft_year': 'year'})

    stats = dm.read(f'''SELECT player, 
                                season year, 
                                games, 
                                games_next, 
                                fantasy_pts_per_game y_act 
                        FROM {pos}_Stats
                        ''', DB_NAME)
    
    # remove Marving Harrison Sr stats
    stats = stats[~((stats.player=='Marvin Harrison') & (stats.year < 2024))].reset_index(drop=True)
    stats = drop_duplicate_players(stats, 'y_act', rookie=True)

    
    df_rookie = pd.merge(stats, df_rookie, on=['player', 'year'], how='right')
    df_rookie = pd.merge(df_rookie, df, on=['player', 'year'])

    df_rookie['year_exp'] = 0
    df_rookie = add_year_exp_compare(df_rookie)
    df_rookie = df_rookie.drop([c for c in df_rookie.columns if 'rmean' in c or 'rmax' in c or 'std' in c], axis=1)
    df_rookie.loc[df_rookie.year==year, ['games','games_next']] = 16
    
    stats_next = dm.read(f'''SELECT player, 
                                season year, 
                                games, 
                                games_next, 
                                fantasy_pts_per_game y_act 
                        FROM {pos}_Stats''', DB_NAME)
    stats_next = drop_duplicate_players(stats_next, 'y_act', rookie=False)
    df_rookie_next, games, games_next = get_next_year_stats(df_rookie, stats_next, ty_mean=False, is_rookie=True)
    df_rookie_next = drop_duplicate_players(df_rookie_next, 'y_act', rookie=True)
    df_rookie_next.loc[df_rookie_next.year==year, ['games','games_next']] = 16
    
    df_rookie.games_next = df_rookie.games_next.fillna(16)

    for df_cur, lbl, gms, gms_next in ([df_rookie, '', 4, 0], [df_rookie_next, '_next', games, games_next]):

        df_cur = drop_y_act_except_current(df_cur, year)
        df_cur = drop_games(df_cur, year, games=games, games_next=games_next)

        df_cur = remove_low_corrs(df_cur, corr_cut=2, collinear_cut=0.998)
        df_cur = y_act_class(df_cur.fillna({'games_next': 16}), df_proj.copy(), 'both', 'games',
                                class_cuts[f'Rookie_{pos}']['upside']['proj_var'], class_cuts[f'Rookie_{pos}']['upside']['y_act'], 'upside')
        df_cur = y_act_class(df_cur.fillna({'games_next': 16}), df_proj.copy(), 'both', 'games',
                                class_cuts[f'Rookie_{pos}']['top']['proj_var'], class_cuts[f'Rookie_{pos}']['top']['y_act'], 'top')

        df_cur = drop_duplicate_players(df_cur, 'y_act', rookie=True)
        df_cur = df_cur.loc[:,~df_cur.columns.duplicated()].copy()

        dm.write_to_db(df_cur, f'Model_Inputs{lbl}', f'{pos}_{year}_Rookie', if_exist='replace')



# xx = pd.merge(stats.loc[stats.year>2007,['player', 'year', 'games', 'games_next', 'y_act']], 
#               df[['player', 'year', 'avg_proj_points']], on=['player', 'year'], how='outer'); print(df.shape[0])

# # xx[xx.avg_proj_points.isnull()].dropna(subset=['y_act']).sort_values(by='y_act').iloc[-50:]
# xx[xx.y_act.isnull()].dropna(subset=['avg_proj_points']).sort_values(by='avg_proj_points').iloc[-50:]


#%%

pos = 'WR'

from skmodel import SciKitModel
from hyperopt import Trials
from sklearn.metrics import r2_score
alpha = 0.8

model_obj = 'reg'
y_act_next = True

if y_act_next: lbl = '_next'
else: lbl = ''

class_metric = '_upside'

if model_obj =='class': proba = True
else: proba = False

Xy = dm.read(f"SELECT * FROM {pos}_{year}_ProjOnly WHERE pos='{pos}' ", f'Model_Inputs{lbl}')
# Xy = dm.read(f"SELECT * FROM {pos}_{year}_Stats WHERE pos='{pos}' ", 'Model_Inputs')
# if Xy.shape[1]==2000:
#     Xy = pd.concat([Xy, dm.read(f"SELECT * FROM {pos}_{year}_Stats_V2 ", 'Model_Inputs')], axis=1)
# Xy = dm.read(f"SELECT * FROM {pos}_{year}_Rookie ", f'Model_Inputs{lbl}')
if proba: Xy = Xy.drop('y_act', axis=1).rename(columns={f'y_act_class{class_metric}': 'y_act'})

Xy = Xy.sort_values(by='year').reset_index(drop=True)
Xy['team'] = 'team'
Xy['week'] = 1
Xy['game_date'] = Xy.year

Xy = Xy.drop([c for c in Xy.columns if 'y_act_' in c], axis=1)

# Xy = Xy[(Xy.year_exp >3) ].reset_index(drop=True)
# Xy = Xy[Xy.year_exp > 3].reset_index(drop=True)
# Xy = Xy[(Xy.year_exp >= 2) & (Xy.year_exp <= 3)].reset_index(drop=True)
# display(Xy.loc[Xy.y_act==1, ['player', 'year', 'y_act', 'avg_proj_points_per_game']].iloc[-50:])

pred = Xy[Xy.year==year].copy().reset_index(drop=True)
train = Xy[Xy.year<year].copy().reset_index(drop=True)
print(Xy.shape)

preds = []
actuals = []

skm = SciKitModel(train, model_obj=model_obj, alpha=alpha, hp_algo='atpe')
to_drop = list(train.dtypes[train.dtypes=='object'].index)
to_drop.extend(['games_next', 'games'])
X, y = skm.Xy_split('y_act', to_drop = to_drop)

if proba:
    p = 'select_perc_c'
    kb = 'k_best_c'
    m = 'cb_c'
else:
    p = 'select_perc'
    kb = 'k_best'
    m = 'enet'

pipe = skm.model_pipe([skm.piece('random_sample'),
                        skm.piece('std_scale'), 
                        skm.piece(p),
                        skm.feature_union([
                                       skm.piece('agglomeration'), 
                                        skm.piece(f'{kb}_fu'),
                                        skm.piece('pca')
                                        ]),
                        skm.piece(kb),
                        skm.piece(m)
                    ])

params = skm.default_params(pipe, 'bayes')

# pipe.steps[-1][-1].set_params(**{'loss_function': f'Quantile:alpha={alpha}'})
best_models, oof_data, param_scores, _ = skm.time_series_cv(pipe, X, y, params, n_iter=10,
                                                                col_split='year',n_splits=5,
                                                                time_split=2017, alpha=alpha,
                                                                bayes_rand='bayes', proba=proba,
                                                                sample_weight=False, trials=Trials(),
                                                                random_seed=64893)

print('R2 score:', r2_score(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred']))
oof_data['full_hold'].plot.scatter(x='pred', y='y_act')
try: show_calibration_curve(oof_data['full_hold'].y_act, oof_data['full_hold'].pred, n_bins=6)
except: pass

#%%

oof_data['full_hold'].sort_values(by='pred', ascending=False).iloc[:50]
# oof_data['full_hold'][(oof_data['full_hold'].pred >15) & (oof_data['full_hold'].y_act < 7)]

# %%

pred = pred.fillna({'games': 16})
try: pred['pred'] = best_models[-1].fit(X,y).predict_proba(pred[X.columns].fillna(pred.mean()))[:,1]
except: pred['pred'] = best_models[-1].fit(X,y).predict(pred[X.columns].fillna(pred.mean()))
pred[['player', 'year', 'pred']].sort_values(by='pred', ascending=False).iloc[:35]

# %%

import matplotlib.pyplot as plt

pipeline = best_models[2]
pipeline.fit(X,y)
# Extract the coefficients
log_reg = pipeline.named_steps[m]
try:
    log_reg.coef_.shape[1]
    coefficients = log_reg.coef_[0]
except: coefficients = log_reg.coef_

# Get the feature names from SelectKBest
selected_features = pipeline.named_steps[kb].get_support(indices=True)

coef = pd.Series(coefficients, index=X.columns[selected_features])
coef[np.abs(coef) > 0.01].sort_values().plot(kind = 'barh', figsize=(10, 10))
# %%
