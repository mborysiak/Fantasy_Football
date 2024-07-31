#%%

# set to this year for the projections
year = 2024

from ff.db_operations import DataManage
from ff import general

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


def rolling_expand(df, gcols, rcols, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    # if agg type is in form of percentile (e.g. p80) then use quantile
    if agg_type[0]=='p':

        # pull out perc amount and convert to decimal float to calculate quantile rolling
        perc_amt = float(agg_type[1:])/100
        rolls =  df.groupby(gcols)[rcols].apply(lambda x: x.expanding().quantile(perc_amt))

    # otherwise, use the string argument of aggregation
    else:
        rolls = df.groupby(gcols)[rcols].apply(lambda x: x.expanding().agg(agg_type))
    
    # clean up the rolled dataset indices and column name prior to returning 
    rolls = rolls.reset_index(drop=True)
    rolls.columns = [f'{agg_type}all_{c}' for c in rolls.columns]

    return rolls


def add_rolling_stats(df, gcols, rcols, perform_check=True):

    df = df.sort_values(by=[gcols[0], 'year']).reset_index(drop=True)

    if perform_check:
        cnt_check = df.groupby([gcols[0], 'year'])['year'].count()
        print(f'Counts of Groupby Category Over 17: {cnt_check[cnt_check>1]}')

    rolls3_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')

    # rolls8_mean = rolling_stats(df, gcols, rcols, 4, agg_type='mean')
    # rolls8_max = rolling_stats(df, gcols, rcols, 4, agg_type='max')
    # rolls8_std = rolling_stats(df, gcols, rcols, 4, agg_type='std')

    # hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
    # hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
    # hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p95')

    df = pd.concat([df, 
                   # rolls8_mean, rolls8_max, #rolls8_std,
                    rolls3_mean, rolls3_max,
                    #hist_mean, hist_std, hist_p80
                    ], axis=1)

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

def fftoday_proj(pos):
    df = dm.read(f'''SELECT * 
                     FROM FFToday_Projections 
                     WHERE pos='{pos}'
                           AND team!='FA'
                 ''', DB_NAME)
    df.team = df.team.map(team_map)
    df['fft_rank'] = df.groupby(['year'])['fft_proj_pts'].rank(ascending=False, method='min').values
    return df

def add_adp(df, pos, source):
    adp = dm.read(f'''SELECT player, year, avg_pick, avg_pick_no_log
                      FROM ADP_Ranks
                      WHERE pos='{pos}'
                            AND source = '{source}'
                      ''', DB_NAME)
    
    if source == 'fpros':
        adp = adp.rename(columns={'avg_pick': 'fpros_avg_pick', 
                                  'avg_pick_no_log': 'fpros_avg_pick_no_log'})
    df = pd.merge(df, adp, on=['player', 'year'], how='left')
    return df

def fantasy_pros_new(df, pos):

    fp = dm.read(f'''SELECT * 
                     FROM FantasyPros_Projections
                     WHERE pos='{pos}' 
                          
                         ''', DB_NAME).drop('pos', axis=1)
    fp = name_cleanup(fp)
    fp.year = fp.year.astype('int')
    df = pd.merge(df, fp, on=['player', 'year'], how='left')
    return df


def ffa_compile(df, table_name, pos):
    
    if table_name == 'FFA_Projections':
        cols = ['player', 'year', 'ffa_points', 'ffa_sd_pts',
                'ffa_dropoff', 'ffa_floor', 'ffa_ceiling', 'ffa_points_vor', 'ffa_floor_vor', 'ffa_ceiling_vor',
                'ffa_rank', 'ffa_floor_rank', 'ffa_ceiling_rank', 'ffa_position_rank', 'ffa_tier', 'ffa_uncertainty']
        
    elif table_name == 'FFA_RawStats':
        if pos == 'QB':
            cols = ['player', 'year', 'ffa_proj_points',
                    'ffa_pass_yds', 'ffa_pass_yds_sd', 'ffa_pass_tds', 'ffa_pass_tds_sd', 'ffa_pass_int',
                    'ffa_pass_int_sd', 'ffa_rush_yds', 'ffa_rush_yds_sd', 'ffa_rush_tds', 'ffa_rush_tds_sd',
                    ]
        elif pos in ('RB', 'WR', 'TE'):
            cols =  ['player', 'year', 'ffa_proj_points', 'ffa_rush_yds', 'ffa_rush_yds_sd', 'ffa_rush_tds', 'ffa_rush_tds_sd',
                      'ffa_rec_yds', 'ffa_rec_tds']

    ffa = dm.read(f"SELECT * FROM {table_name} WHERE position='{pos}'", DB_NAME)
    ffa = ffa[cols].drop_duplicates()

    df = pd.merge(df, ffa, on=['player', 'year'], how='left')

    return df

def fantasy_data_proj(df, pos):
    
    fd = dm.read(f'''SELECT * 
                     FROM FantasyData
                     WHERE pos='{pos}' 
                    ''', DB_NAME)
    
    fd = fd.drop(['team', 'pos'], axis=1)
    
    df = pd.merge(df, fd, on=['player', 'year'], how='left')

    return df


def pff_experts_new(df, pos):

    experts = dm.read(f'''SELECT player, week, year, a.defTeam,
                            fantasyPoints,  fantasyPointsRank,
                            `Proj Pts` ProjPts,
                            passComp, passAtt, passYds, passTd, passInt, passSacked,
                            rushAtt, rushYds, rushTd, recvTargets,
                            recvReceptions, recvYds, recvTd,
                            fumbles, fumblesLost, twoPt, returnYds, returnTd,
                            expertConsensus, expertNathanJahnke, expertIanHartitz,
                            rankadj_expertConsensus, rankadj_expertNathanJahnke,
                            playeradj_expertNathanJahnke,playeradj_expertConsensus 
                       
                    FROM PFF_Proj_Ranks a
                    JOIN (SELECT *
                            FROM PFF_Expert_Ranks 
                            WHERE Position='{pos}' )
                            USING (player, week, year)
                    ''', 'Pre_PlayerData')
    
    df = pd.merge(df, experts, on=['player', 'week', 'year'], how='left')

    return df
def get_pff_proj(df, pos):

    pff = dm.read(f'''SELECT *
                          FROM PFF_Projections
                            WHERE pos='{pos}'
                    ''', DB_NAME).drop(['team', 'pos'], axis=1)
    df = pd.merge(df, pff, on=['player', 'year'], how='left')

    return df



def consensus_fill(df):


    to_fill = {

        # stat fills
        'proj_pass_yds': ['fpros_pass_yds', 'ffa_pass_yds', 'fft_pass_yds', 'fdta_pass_yds', 'pff_pass_yds'],
        'proj_pass_td': ['fpros_pass_td', 'ffa_pass_tds', 'fft_pass_td',  'fdta_pass_td', 'pff_pass_td'],
        'proj_pass_int': ['fpros_pass_int', 'ffa_pass_int', 'fft_pass_int', 'fdta_int', 'fdta_pass_int', 'pff_pass_int'],
        'proj_pass_att': ['fft_pass_att', 'fpros_pass_att', 'pff_pass_att'],
        'proj_pass_comp': ['fft_pass_comp', 'fpros_pass_cmp', 'pff_pass_comp'],
        'proj_rush_yds': ['fpros_rush_yds' ,'ffa_rush_yds', 'fft_rush_yds', 'fdta_rush_yds', 'pff_rush_yds'],
        'proj_rush_att': [ 'fft_rush_att', 'fpros_rush_att', 'pff_rush_att'],
        'proj_rush_td': ['fpros_rush_td', 'ffa_rush_tds', 'fft_rush_td', 'fdta_rush_td', 'pff_rush_td'],
        'proj_rec': ['fpros_rec', 'fft_rec', 'fdta_rec', 'pff_rec_receptions'],
        'proj_rec_yds': ['fpros_rec_yds', 'ffa_rec_yds', 'fft_rec_yds', 'fdta_rec_yds', 'pff_rec_yds'],
        'proj_rec_td': ['fpros_rec_td', 'ffa_rec_tds', 'fft_rec_td','fdta_rec_td', 'pff_rec_td'],
        'proj_sacks': ['fft_sacks', 'pff_pass_sacked'],

        'proj_pass_yds_per_att': ['fft_pass_yds_per_att', 'fpros_pass_yds_per_att', 'pff_pass_yds_per_att'],
        'proj_rush_yds_per_att': ['fft_rush_yds_per_att', 'fpros_rush_yds_per_att', 'pff_rush_yds_per_att'],
        'proj_rec_yds_per_rec': ['fft_rec_yds_per_rec', 'fpros_rec_yds_per_rec', 'pff_rec_yds_per_rec'],
        'proj_pass_td_per_att': ['fft_pass_td_per_att', 'fpros_pass_td_per_att', 'pff_pass_td_per_att'],
        'proj_rush_td_per_att': ['fft_rush_td_per_att', 'fpros_rush_td_per_att', 'pff_rush_td_per_att'],
        'proj_rec_td_per_rec': ['fft_rec_td_per_rec', 'fpros_rec_td_per_rec', 'pff_rec_td_per_rec'],

        # point and rank fills
        'proj_points': ['ffa_proj_points', 'fft_proj_pts', 'fdta_fantasy_points_total', 'fpros_proj_pts_calc', 'pff_proj_pts_calc'],
        'proj_rank': ['fft_rank', 'ffa_rank', 'ffa_position_rank', 'fdta_rank', 'pff_rank'],
        'avg_pick': ['fpros_avg_pick', 'avg_pick'],
        }

    for k, tf in to_fill.items():

        # find columns that exist in dataset
        tf = [c for c in tf if c in df.columns]
        
        # fill in nulls based on available data
        for c in tf:
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), tf].mean(axis=1)
        
        # fill in the average for all cols
        df['avg_' + k] = df[tf].mean(axis=1)
        df['std_' + k] = df[tf].std(axis=1)

        if 'rank' in k:
            df['min' + k] = df[tf].min(axis=1)
        else:
            df['max_' + k] = df[tf].max(axis=1)
    
    return df


def log_rank_cols(df):
    rank_cols = [c for c in df.columns if 'rank' in c or 'expert' in c]
    for c in rank_cols:
        df['log_' + c] = np.log(df[c]+1)
    return df

def rolling_proj_stats(df):
    df = forward_fill(df)
    proj_cols = [c for c in df.columns if 'ffa' in c or 'rank' in c or 'fpros' in c or 'proj' in c \
                 or 'fft' in c or 'expert' in c or 'Pts' in c or 'Points' in c or 'points' in c or 'fdta' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)

    # for c in proj_cols:
    #     df[f'trend_diff_{c}'] = df[f'rmean2_{c}'] - df[f'rmean2_{c}']
    #     df[f'trend_chg_{c}'] = df[f'trend_diff_{c}'] / (df[f'rmean2_{c}']+5)

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

        # pull all projections and ranks
        df = fftoday_proj(pos); print(df.shape[0])
        df = fantasy_pros_new(df, pos); print(df.shape[0])
        df = get_pff_proj(df, pos); print(df.shape[0])
        df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
        df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
        df = fantasy_data_proj(df, pos); print(df.shape[0])

        df = consensus_fill(df); print(df.shape[0])
        df = df.dropna(axis=1)
        team_proj = pd.concat([team_proj, df], axis=0)

    team_proj = create_pos_rank(team_proj)
    team_proj = forward_fill(team_proj)
    team_proj = team_proj.fillna(0)

    cnts = team_proj.groupby(['team', 'year']).agg({'avg_proj_points': 'count'})
    print('Team counts that do not equal 7:', cnts[cnts.avg_proj_points!=7])

    cols = [
        'avg_proj_pass_yds', 'fpros_pass_yds', 'ffa_pass_yds', 'fft_pass_yds', 'fdta_pass_yds', 'pff_pass_yds',
        'avg_proj_pass_td', 'fpros_pass_td', 'ffa_pass_tds', 'fft_pass_td',  'fdta_pass_td', 'pff_pass_td',
        'avg_proj_rush_yds', 'fpros_rush_yds' ,'ffa_rush_yds', 'fft_rush_yds', 'fdta_rush_yds', 'pff_rush_yds',
        'avg_proj_rush_att', 'fft_rush_att', 'fpros_rush_att', 'pff_rush_att',
        'avg_proj_rush_td', 'fpros_rush_td', 'ffa_rush_tds', 'fft_rush_td', 'fdta_rush_td', 'pff_rush_td',
        'avg_proj_rec', 'fpros_rec', 'fft_rec', 'fdta_rec', 'pff_rec_receptions',
        'avg_proj_rec_yds', 'fpros_rec_yds', 'ffa_rec_yds', 'fft_rec_yds', 'fdta_rec_yds', 'pff_rec_yds',
        'avg_proj_rec_td', 'fpros_rec_td', 'ffa_rec_tds', 'fft_rec_td','fdta_rec_td', 'pff_rec_td',
        'avg_proj_points', 'ffa_proj_points', 'fft_proj_pts', 'fdta_fantasy_points_total', 'fpros_proj_pts_calc', 'pff_proj_pts_calc'
    ]

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
                   'avg_proj_rec', 'avg_proj_rec_yds', 'avg_proj_rec_td']

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
    pick_rank_cols = ['avg_pick', 'avg_proj_rank', 'fpros_avg_pick']
    team_adp = df.groupby(['team', 'pos', 'year']).agg({r: ['sum', 'mean', 'min'] for r in pick_rank_cols}).reset_index()
    team_adp.columns = ['team_' + c[0] + '_' + c[1] if c[1]!='' else c[0] for c in team_adp.columns]
    df = pd.merge(df, team_adp, on=['team', 'pos', 'year'])

    for c in pick_rank_cols:
        df[f'mean_diff_{c}'] = df[c] - df[f'team_{c}_mean']
        df[f'min_diff_{c}'] = df[c] - df[f'team_{c}_min']
        df[f'sum_frac_{c}'] = df[c] / df[f'team_{c}_sum']

    return df


def get_max_qb():

    pos='QB'

    # pull all projections and ranks
    df = fftoday_proj(pos); print(df.shape[0])
    df = fantasy_pros_new(df, pos); print(df.shape[0])
    df = get_pff_proj(df, pos); print(df.shape[0])
    df = add_adp(df, pos, 'mfl'); print(df.shape[0])
    df = add_adp(df, pos, 'fpros'); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])

    df = consensus_fill(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])

    df['log_avg_proj_rank'] = np.log(df[[c for c in df.columns if 'rank' in c]].mean(axis=1))
    df.loc[df.avg_pick.isnull(), 'avg_pick'] = df.loc[df.avg_pick.isnull(), 'log_avg_proj_rank'] * 0.676 + 1.71
    df.loc[df.avg_pick.isnull(), 'fpros_avg_pick'] = df.loc[df.avg_pick.isnull(), 'log_avg_proj_rank'] * 0.676 + 1.71

    qb_cols = [
               'team', 'year', 
               'avg_proj_pass_yds', 'std_proj_pass_yds', 'max_proj_pass_yds',
                'avg_proj_pass_td', 'std_proj_pass_td', 'max_proj_pass_td',
                'avg_proj_pass_int', 'std_proj_pass_int', 'max_proj_pass_int',
                'avg_proj_pass_att', 'std_proj_pass_att', 'max_proj_pass_att',
                'avg_proj_pass_comp', 'std_proj_pass_comp', 'max_proj_pass_comp',
                'avg_proj_rush_yds', 'std_proj_rush_yds', 'max_proj_rush_yds',
                'avg_proj_rush_att', 'std_proj_rush_att', 'max_proj_rush_att',
                'avg_proj_rush_td', 'std_proj_rush_td', 'max_proj_rush_td',
                 'avg_proj_points','std_proj_points', 'max_proj_points', 
                 'avg_proj_rank', 'std_proj_rank', 'minproj_rank',
                'avg_pick', 'fpros_avg_pick'
               ]
    df = df.sort_values(by=['team', 'year', 'avg_proj_points'],
                        ascending=[True, True, False])
    df = df.drop_duplicates(subset=['team', 'year'], keep='first').reset_index(drop=True)
    df = df[qb_cols]
    df = df.dropna(axis=1)
    df.columns = ['qb_'+c if c not in ('team', 'year') else c for c in df.columns]
    df = remove_non_uniques(df)

    df['rush_pass_att_ratio'] = df.qb_avg_proj_rush_att / (df.qb_avg_proj_pass_att + 10)
    df['rush_pass_yds_ratio'] = df.qb_avg_proj_rush_yds / (df.qb_avg_proj_pass_yds + 100)
    df['rush_pass_td_ratio'] = df.qb_avg_proj_rush_td / (df.qb_avg_proj_pass_td + 1)

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

    return df

def drop_duplicate_players(df, order_col, rookie=False):
    df = df.sort_values(by=['player', 'year', order_col],
                    ascending=[True, True,  False])
    if rookie: df = df.drop_duplicates(subset=['player'], keep='first').reset_index(drop=True)
    else: df = df.drop_duplicates(subset=['player', 'year'], keep='first').reset_index(drop=True)
    
    return df


def remove_low_corrs(df, corr_cut=0.015):
    obj_cols = df.dtypes[df.dtypes=='object'].index
    corrs = pd.DataFrame(np.corrcoef(df.drop(obj_cols, axis=1).values, rowvar=False), 
                         columns=[c for c in df.columns if c not in obj_cols],
                         index=[c for c in df.columns if c not in obj_cols])
    corrs = corrs['y_act']
    low_corrs = list(corrs[abs(corrs) < corr_cut].index)
    low_corrs = [c for c in low_corrs if c not in ('week', 'year', 'pos', 'games', 'games_next')]
    df = df.drop(low_corrs, axis=1)
    print(f'Removed {len(low_corrs)}/{df.shape[1]} columns')
    
    corrs = corrs.dropna().sort_values()
    display(corrs.iloc[:20])
    display(corrs.iloc[-20:])
    display(corrs[~corrs.index.str.contains('ffa|proj|expert|rank|Proj|fc|salary|Points|Rank|fdta|fft|proj') | \
                   corrs.index.str.contains('team') | \
                   corrs.index.str.contains('qb')].iloc[:20])
    display(corrs[~corrs.index.str.contains('ffa|proj|expert|rank|Proj|fc|salary|Points|Rank|fdta|fft|proj') | \
                   corrs.index.str.contains('team') | \
                   corrs.index.str.contains('qb')].iloc[-20:])
    return df

def drop_y_act_except_current(df, year):
    
    df = df[~(df.y_act.isnull()) | (df.year==year)].reset_index(drop=True)
    df.loc[(df.year==year), 'y_act'] = 0

    return df

def drop_games(df, year, games=0, games_next=0):
    df = df[((df.games >= games) & \
             (df.games >= games_next)) | \
             (df.year==year)].reset_index(drop=True)
    return df

def y_act_class(df, df_quant, proj_var_cut, y_act_cut, suffix):
    df['avg_proj_points_per_game'] = np.where(df.year >= 2021, df.avg_proj_points / 17, df.avg_proj_points / 16)
    df[f'y_act_class_{suffix}'] = 0

    df_quant['avg_proj_points_per_game'] = np.where(df_quant.year >= 2021, df_quant.avg_proj_points / 16, df_quant.avg_proj_points / 15)
    df_quant['proj_var'] = df_quant.y_act - df_quant.avg_proj_points_per_game   
    df_proj_quant = df_quant[df_quant.games_next > 4].groupby('year')['proj_var'].quantile(proj_var_cut).reset_index()
    df_act_quant = df_quant[df_quant.games_next > 4].groupby('year')['y_act'].quantile(y_act_cut).reset_index().rename(columns={'y_act': 'y_act_quantile'})
    
    df = pd.merge(df, df_proj_quant, on='year', how='left')
    df = pd.merge(df, df_act_quant, on='year', how='left')
    
    df.loc[(df.y_act > df.avg_proj_points_per_game + df.proj_var) & \
           (df.games_next > 8) & \
           (df.y_act >= df.y_act_quantile),
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

#%%

pos='TE'

class_cuts = {
    'WR': {
           'upside': {'y_act': 0.725, 'proj_var': 0.8},
           'top':  {'y_act': 0.825, 'proj_var': 0.6}
        },
    
    'RB': {
           'upside': {'y_act': 0.75, 'proj_var': 0.75},
           'top':  {'y_act': 0.85, 'proj_var': 0.6}
        },
    
    'QB': {
           'upside': {'y_act': 0.8, 'proj_var': 0.6},
           'top':  {'y_act': 0.825, 'proj_var': 0.5}
        },

    'TE': {
           'upside': {'y_act': 0.8, 'proj_var': 0.6},
           'top':  {'y_act': 0.825, 'proj_var': 0.5}
        },

    'Rookie_WR': {
           'upside': {'y_act': 0.7, 'proj_var': 0.7},
           'top':  {'y_act': 0.75, 'proj_var': 0.5}
        },

    'Rookie_RB': {
           'upside': {'y_act': 0.7, 'proj_var': 0.7},
           'top':  {'y_act': 0.75, 'proj_var': 0.5}
        },
}

# pull all projections and ranks
df = fftoday_proj(pos); print(df.shape[0])
df = drop_duplicate_players(df, 'fft_proj_pts')
df = add_adp(df, pos, 'mfl'); print(df.shape[0])
df = add_adp(df, pos, 'fpros'); print(df.shape[0])
df = fantasy_pros_new(df, pos); print(df.shape[0])
df = get_pff_proj(df, pos); print(df.shape[0])
df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
df = fantasy_data_proj(df, pos); print(df.shape[0])
if pos == 'QB':
    df = df.drop([c for c in df.columns if 'rec' in c], axis=1)

df['log_avg_proj_rank'] = np.log(df[[c for c in df.columns if 'rank' in c]].mean(axis=1))
df.loc[df.avg_pick.isnull(), 'avg_pick'] = df.loc[df.avg_pick.isnull(), 'log_avg_proj_rank'] * 0.676 + 1.71
df.loc[df.avg_pick.isnull(), 'fpros_avg_pick'] = df.loc[df.avg_pick.isnull(), 'log_avg_proj_rank'] * 0.676 + 1.71


df = consensus_fill(df); print(df.shape[0])
df = log_rank_cols(df); print(df.shape[0])

df = df.dropna(axis=1)
df = rolling_proj_stats(df); print(df.shape[0])
df = remove_non_uniques(df); print(df.shape[0])

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

df = drop_duplicate_players(df, 'avg_proj_points')

#-----------
# Save out Projection data
#-----------
stats = dm.read(f'SELECT * FROM {pos}_Stats', DB_NAME)
stats = drop_duplicate_players(stats, 'sum_fantasy_pts')
stats_proj = stats[['player', 'season', 'games', 'games_next', 'fantasy_pts_per_game']].copy()
stats_proj = stats_proj.rename(columns={'fantasy_pts_per_game': 'y_act', 'season': 'year'})
df_proj = pd.merge(stats_proj, df, on=['player', 'year'], how='right')
df_proj['season'] = df_proj.year - 1

# only drop next year's game limit since last year's are not being
# used for model predictions and don't matter if they are missing
df_proj = drop_y_act_except_current(df_proj, year)
df_proj = drop_games(df_proj, year, games=0, games_next=4)
df_proj = remove_low_corrs(df_proj.dropna(subset=['y_act']), corr_cut=0.02)
df_proj = add_draft_year_exp(df_proj, pos)

df_proj.loc[df_proj.year >= year-1, ['games_next']] = 16
df_proj = y_act_class(df_proj, df_proj.copy(), class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
df_proj = y_act_class(df_proj, df_proj.copy(), class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')

dm.write_to_db(df_proj, 'Model_Inputs', f'{pos}_{year}_ProjOnly', if_exist='replace')

#------------
# Save out Full Data
#------------

stats = add_rolling_stats(stats, ['player'], [c for c in stats.columns if 'pass' in c or 'rush' in c or 'rec' in c])
stats = remove_non_uniques(stats)
df_stats = pd.merge(stats, df, on=['player', 'year']); print(df.shape[0])

# drop both last year's and next year's game limits
# since you need y_act to be relevant and last year's stats to be relvant
df_stats = drop_y_act_except_current(df_stats, year)
df_stats = drop_games(df_stats, year, games=4, games_next=4)
df_stats = remove_low_corrs(df_stats.dropna(subset=['y_act']), corr_cut=0.02)
df_stats = add_draft_year_exp(df_stats, pos)
df_stats = y_act_class(df_stats, df_stats.copy(), class_cuts[pos]['upside']['proj_var'], class_cuts[pos]['upside']['y_act'], 'upside')
df_stats = y_act_class(df_stats, df_stats.copy(), class_cuts[pos]['top']['proj_var'], class_cuts[pos]['top']['y_act'], 'top')


dm.write_to_db(df_stats, 'Model_Inputs', f'{pos}_{year}_Stats', if_exist='replace')

#------------
# Save out Rookie Data
#------------
if pos in ('RB', 'WR'):

    df_rookie = dm.read(f'''SELECT * FROM Rookie_{pos}_Stats''', DB_NAME).rename(columns={'draft_year': 'year'})

    stats = dm.read(f'''SELECT player, 
                                season year, 
                                games, 
                                games_next, 
                                fantasy_pts_per_game y_act 
                        FROM {pos}_Stats''', DB_NAME)
    stats = drop_duplicate_players(stats, 'y_act', rookie=True)

    df_rookie = pd.merge(stats, df_rookie, on=['player', 'year'], how='right')
    df_rookie = pd.merge(df_rookie, df, on=['player', 'year'])
    df_rookie = drop_y_act_except_current(df_rookie, year)
    df_rookie = drop_games(df_rookie, year, games=4, games_next=0)

    df_rookie = remove_low_corrs(df_rookie, corr_cut=0.02)
    df_rookie = y_act_class(df_rookie.fillna({'games_next': 16}), df_proj.copy(), 
                            class_cuts[f'Rookie_{pos}']['upside']['proj_var'], class_cuts[f'Rookie_{pos}']['upside']['y_act'], 'upside')
    df_rookie = y_act_class(df_rookie.fillna({'games_next': 16}), df_proj.copy(), 
                            class_cuts[f'Rookie_{pos}']['top']['proj_var'], class_cuts[f'Rookie_{pos}']['top']['y_act'], 'top')

    df_rookie = drop_duplicate_players(df_rookie, 'y_act', rookie=True)

    dm.write_to_db(df_rookie, 'Model_Inputs', f'{pos}_{year}_Rookie', if_exist='replace')



# xx = pd.merge(stats.loc[stats.year>2007,['player', 'year', 'games', 'games_next', 'y_act']], 
#               df[['player', 'year', 'avg_proj_points']], on=['player', 'year'], how='outer'); print(df.shape[0])

# # xx[xx.avg_proj_points.isnull()].dropna(subset=['y_act']).sort_values(by='y_act').iloc[-50:]
# xx[xx.y_act.isnull()].dropna(subset=['avg_proj_points']).sort_values(by='avg_proj_points').iloc[-50:]


#%%

pos = 'RB'

from skmodel import SciKitModel
from hyperopt import Trials
from sklearn.metrics import r2_score
alpha = 0.8
model_obj = 'class'

class_metric = 'top'

if model_obj =='class': proba = True
else: False

# Xy = dm.read(f"SELECT * FROM {pos}_{year}_ProjOnly WHERE pos='{pos}' ", 'Model_Inputs')
# Xy = dm.read(f"SELECT * FROM {pos}_{year}_Stats WHERE pos='{pos}' ", 'Model_Inputs')
Xy = dm.read(f"SELECT * FROM {pos}_{year}_Rookie ", 'Model_Inputs')

Xy = Xy.sort_values(by='year').reset_index(drop=True)
Xy['team'] = 'team'
Xy['week'] = 1
Xy['game_date'] = Xy.year

if proba: Xy = Xy.drop('y_act', axis=1).rename(columns={f'y_act_class_{class_metric}': 'y_act'})
Xy = Xy.drop([c for c in Xy.columns if 'y_act_class' in c], axis=1)

pred = Xy[Xy.year==2024].copy().reset_index(drop=True)
train = Xy[Xy.year<2024].copy().reset_index(drop=True)
print(Xy.shape)

preds = []
actuals = []

skm = SciKitModel(train, model_obj=model_obj, alpha=alpha, hp_algo='atpe')
to_drop = list(train.dtypes[train.dtypes=='object'].index)
to_drop.extend(['games_next'])
X, y = skm.Xy_split('y_act', to_drop = to_drop)

if proba:
    p = 'select_perc_c'
    kb = 'k_best_c'
    m = 'lr_c'
else:
    p = 'select_perc'
    kb = 'k_best'
    m = 'lgbm'
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
                                                                time_split=2016, alpha=alpha,
                                                                bayes_rand='bayes', proba=proba,
                                                                sample_weight=False, trials=Trials(),
                                                                random_seed=12345)

print('R2 score:', r2_score(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred']))
oof_data['full_hold'].plot.scatter(x='pred', y='y_act')
try: show_calibration_curve(oof_data['full_hold'].y_act, oof_data['full_hold'].pred, n_bins=6)
except: pass

#%%

oof_data['full_hold'].sort_values(by='pred', ascending=False).iloc[:50]#[(oof_data['full_hold'].pred < 7.5) & (oof_data['full_hold'].y_act > 15)]#%%
# %%

pred = pred.fillna({'games': 16})
try: pred['pred'] = best_models[-1].fit(X,y).predict_proba(pred[X.columns].fillna(pred.mean()))[:,1]
except: pred['pred'] = best_models[-1].fit(X,y).predict(pred[X.columns].fillna(pred.mean()))
pred[['player', 'year', 'pred']].sort_values(by='pred', ascending=False).iloc[:30]

# %%
