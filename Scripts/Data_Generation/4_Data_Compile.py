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
        cnt_check = df.groupby([gcols[0], 'year'])['fft_proj_pts'].count()
        print(f'Counts of Groupby Category Over 17: {cnt_check[cnt_check>1]}')

    rolls3_mean = rolling_stats(df, gcols, rcols, 2, agg_type='mean')
    rolls3_max = rolling_stats(df, gcols, rcols, 2, agg_type='max')

    rolls8_mean = rolling_stats(df, gcols, rcols, 4, agg_type='mean')
    rolls8_max = rolling_stats(df, gcols, rcols, 4, agg_type='max')
    rolls8_std = rolling_stats(df, gcols, rcols, 4, agg_type='std')

    # hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
    # hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
    # hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p95')

    df = pd.concat([df, 
                    rolls8_mean, rolls8_max, rolls8_std,
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
                 ''', 'Season_Stats')
    df.team = df.team.map(team_map)
    df['fft_rank'] = df.groupby(['year'])['fft_proj_pts'].rank(ascending=False, method='min').values
    return df


def fantasy_pros_new(df, pos):

    fp = dm.read(f'''SELECT * 
                     FROM FantasyPros_Projections
                     WHERE pos='{pos}' 
                          
                         ''', 'Season_Stats').drop('pos', axis=1)
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
            cols = ['player', 'year',
                    'ffa_pass_yds', 'ffa_pass_yds_sd', 'ffa_pass_tds', 'ffa_pass_tds_sd', 'ffa_pass_int',
                    'ffa_pass_int_sd', 'ffa_rush_yds', 'ffa_rush_yds_sd', 'ffa_rush_tds', 'ffa_rush_tds_sd',
                    ]
        elif pos in ('RB', 'WR', 'TE'):
            cols =  ['player', 'year', 'ffa_rush_yds', 'ffa_rush_yds_sd', 'ffa_rush_tds', 'ffa_rush_tds_sd',
                      'ffa_rec_yds', 'ffa_rec_tds']

    ffa = dm.read(f"SELECT * FROM {table_name} WHERE position='{pos}'", 'Season_Stats')
    ffa = ffa[cols].drop_duplicates()

    df = pd.merge(df, ffa, on=['player', 'year'], how='left')

    return df

def fantasy_data_proj(df, pos):
    
    fd = dm.read(f'''SELECT * 
                     FROM FantasyData
                     WHERE pos='{pos}' 
                    ''', 'Season_Stats')
    
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
                    ''', 'Season_Stats').drop(['team', 'pos'], axis=1)
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
        # 'proj_rec_tgts': ['recvTargets', 'fc_proj_receiving_stats_tar'],

        # point and rank fills
        'proj_points': ['ffa_points', 'fpros_proj_pts', 'fft_proj_pts', 'fdta_fantasy_points_total', 'pff_proj_points'],
        'proj_rank': ['fft_rank', 'ffa_rank', 'ffa_position_rank', 'fdta_rank', 'pff_rank']
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


def fill_ratio_nulls(df):
    ratio_fill_cols = ['ffa_sd_pts', 'ffa_dropoff', 'ffa_floor', 'ffa_ceiling', 'ffa_points_vor', 'ffa_floor_vor',
                        'ffa_ceiling_vor', 'ffa_rank', 'ffa_floor_rank', 'ffa_ceiling_rank', 'ffa_rec_sd',
                        'ffa_tier', 'ffa_uncertainty','ffa_pass_yds_sd', 'ffa_pass_tds_sd', 'ffa_pass_int_sd',
                        'ffa_rush_yds_sd',  'ffa_rush_tds_sd', 'fc_proj_rushing_stats_pct', 'fc_proj_rushing_stats_att_tar', 
                        'fc_proj_receiving_stats_pct', 'fc_projected_values_floor', 'fc_projected_values_ceiling',
                        'ffa_sd_pts', 'ffa_uncertainty', 'ffa_dst_int_sd', 'ffa_dst_sacks_sd',
                        'ffa_dst_safety_sd', 'ffa_dst_td_sd', 'fc_proj_defensive_stats_pts']
    for c in ratio_fill_cols:
        if c in df.columns:
            fill_ratio = (df[c] / (df['ffa_points']+1)).mean()
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), 'ffa_points'] * fill_ratio + fill_ratio
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

    for c in proj_cols:
        df[f'trend_diff_{c}'] = df[f'rmean2_{c}'] - df[f'rmean2_{c}']
        df[f'trend_chg_{c}'] = df[f'trend_diff_{c}'] / (df[f'rmean2_{c}']+5)

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
        'avg_proj_points', 'ffa_points', 'fpros_proj_pts', 'fft_proj_pts', 'fdta_fantasy_points_total',# 'pff_proj_points',
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


def get_max_qb():

    pos='QB'

    # pull all projections and ranks
    df = fftoday_proj(pos); print(df.shape[0])
    df = fantasy_pros_new(df, pos); print(df.shape[0])
    df = get_pff_proj(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])

    df = consensus_fill(df); print(df.shape[0])
    # df = fill_ratio_nulls(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])
    df = df.dropna(axis=1)

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
                'avg_proj_rec', 'std_proj_rec', 'max_proj_rec', 'avg_proj_rec_yds',
                'std_proj_rec_yds', 'max_proj_rec_yds', 'avg_proj_rec_td',
                'std_proj_rec_td', 'max_proj_rec_td', 'avg_proj_points',
                'std_proj_points', 'max_proj_points', 'avg_proj_rank', 'std_proj_rank', 'minproj_rank'
               ]
    df = df.sort_values(by=['team', 'year', 'avg_proj_points'],
                        ascending=[True, True, False])
    df = df.drop_duplicates(subset=['team', 'year'], keep='first').reset_index(drop=True)
    df = df[qb_cols]
    df.columns = ['qb_'+c if c not in ('team', 'year') else c for c in df.columns]
    df = remove_non_uniques(df)

    return df


def remove_low_corrs(df, corr_cut=0.015):
    obj_cols = df.dtypes[df.dtypes=='object'].index
    corrs = pd.DataFrame(np.corrcoef(df.drop(obj_cols, axis=1).values, rowvar=False), 
                         columns=[c for c in df.columns if c not in obj_cols],
                         index=[c for c in df.columns if c not in obj_cols])
    corrs = corrs['y_act']
    low_corrs = list(corrs[abs(corrs) < corr_cut].index)
    low_corrs = [c for c in low_corrs if c not in ('week', 'year', 'fd_salary')]
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

#%%

pos='RB'


# pull all projections and ranks
df = fftoday_proj(pos); print(df.shape[0])
df = fantasy_pros_new(df, pos); print(df.shape[0])
df = get_pff_proj(df, pos); print(df.shape[0])
df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
df = fantasy_data_proj(df, pos); print(df.shape[0])
if pos == 'QB':
    df = df.drop([c for c in df.columns if 'rec' in c], axis=1)

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

    max_qb = get_max_qb()
    df = pd.merge(df, max_qb, on=['team', 'year']); print(df.shape[0])

#%%

# df = pd.merge(df, stats, on=['player', 'year'])
# df = remove_low_corrs(df, corr_cut=0.02)
# df
#%%
chk =  pd.merge(df[['player', 'year', 'avg_proj_points']], stats, on=['player', 'year'], how='outer')
chk[(chk.avg_proj_points.isnull()) & (chk.year >= 2008)].sort_values(by=['year', 'avg_proj_points'], ascending=[True, False]).iloc[:50]

#%%
chk[(chk.avg_proj_points.isnull()) & (chk.year >= 2008)].groupby('player').agg({'y_act': 'sum', 'year': 'count'}).sort_values(by='y_act', ascending=False).iloc[:50]

#%%
chk[(chk.avg_pick.isnull()) & (chk.year >= 2008)].groupby('player').agg({'avg_pick': 'mean', 'year': 'count'}).sort_values(by='year', ascending=False).iloc[:50]

# %%
