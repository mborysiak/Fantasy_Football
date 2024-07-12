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

#%%


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

get_team_projections()

#%%

pos='RB'

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
df = rolling_proj_stats(df); print(df.shape[0])
df = remove_non_uniques(df); print(df.shape[0])

df = calc_market_share(df); print(df.shape[0])
df

#%%




def add_player_comparison(df, cols):
    
    to_agg = {c: [np.mean, np.max, np.min] for c in cols}
    team_stats = df.groupby(['team', 'year']).agg(to_agg)

    diff_df = df[['player', 'team', 'year']].drop_duplicates()
    for c in cols:
        tmp_df = team_stats[c].reset_index()
        tmp_df = pd.merge(tmp_df, df[['player', 'team', 'year', c]], on=['team', 'year'])

        for a in ['mean', 'amin', 'amax']:
            tmp_df[f'{c}_{a}_diff'] = tmp_df[c] - tmp_df[a]
    
        tmp_df = tmp_df[['player', 'team', 'year', f'{c}_mean_diff', f'{c}_amax_diff', f'{c}_amin_diff']]
        diff_df = pd.merge(diff_df, tmp_df, on=['player', 'team', 'year'])
        
    diff_df = diff_df.drop_duplicates()
    team_stats.columns = [f'{c[0]}_{c[1]}' for c in team_stats.columns]
    team_stats = team_stats.reset_index().drop_duplicates()

    df = pd.merge(df, team_stats, on=['team', 'year'])
    df = pd.merge(df, diff_df, on=['player', 'team', 'year'])

    return df


def pos_rank_stats(df, team_pos_rank, pos):
    
    pos_stats = dm.read(f'''SELECT * 
                            FROM {pos}_Stats 
                            WHERE week < 17
                                  AND season >= 2020
                            --WHERE (season = 2020 AND week != 17)
                            --        OR (season >=2021 AND week != 18)
                            ''', 'FastR')
    pos_stats = pos_stats.rename(columns={'season': 'year'})

    pos_rank_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
                    'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
                    'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
                    'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
                    'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
                    'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
                    'rec_complete_pass_sum', 'rec_qb_dropback_sum'
    ]
    agg_cols = {c: 'sum' for c in pos_rank_cols}

    pos_stats = pd.merge(team_pos_rank, pos_stats, on=['player', 'team', 'week', 'year'], how='left')

    pos_stats = pos_stats.groupby(['pos_rank', 'team', 'week','year']).agg(agg_cols)
    pos_stats.columns = ['pos_rank_' + c for c in pos_stats.columns]
    pos_stats = pos_stats.reset_index()

    gcols = ['team', 'pos_rank']
    rcols=['pos_rank_' + c for c in pos_rank_cols]
    pos_stats = pos_stats.sort_values(by=['team', 'pos_rank', 'year', 'week']).reset_index(drop=True)

    rolls3_mean = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='max')

    rolls8_mean = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='mean')
    rolls8_max = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='max')

    pos_stats = pd.concat([pos_stats, rolls8_mean, rolls8_max, rolls3_mean, rolls3_max], axis=1)

    pos_stats = pd.merge(team_pos_rank, pos_stats, on=['pos_rank', 'team', 'week', 'year'])
    pos_stats = pos_stats.drop(['pos_rank_' + c for c in pos_rank_cols], axis=1)

    # remove cincy-buf due to cancellation
    pos_stats = pos_stats[~((pos_stats.team.isin(['BUF', 'CIN'])) & (pos_stats.week==17) & (pos_stats.year==2022))].reset_index(drop=True)

    pos_stats['week'] = pos_stats['week'] + 1
    pos_stats = switch_seasons(pos_stats)
    pos_stats = fix_bye_week(pos_stats)

    df = pd.merge(df, pos_stats, on=['player', 'team', 'pos', 'week', 'year'], how='left')

    return df
    



#-------------------
# Final Cleanup
#-------------------

def drop_y_act_except_current(df, week, year):
    
    
    df = df[~(df.y_act.isnull()) | ((df.week==week) & (df.year==year))].reset_index(drop=True)
    df.loc[((df.week==week) & (df.year==year)), 'y_act'] = 0

    return df

def remove_non_uniques(df):
    cols = df.nunique()[df.nunique()==1].index
    cols = [c for c in cols if c != 'pos']
    df = df.drop(cols, axis=1)
    return df

def drop_duplicate_players(df):
    df = df.sort_values(by=['player', 'year', 'week', 'projected_points', 'ffa_points'],
                    ascending=[True, True, True, False, False])
    df = df.drop_duplicates(subset=['player', 'year', 'week'], keep='first').reset_index(drop=True)
    return df

def one_qb_per_week(df):
    max_qb = df.groupby(['team', 'year', 'week']).agg({'projected_points': 'max',
                                                       'fp_rank': 'min'}).reset_index()
    cols = df.columns
    df = pd.merge(df, max_qb.drop('fp_rank', axis=1), on=['team', 'year', 'week', 'projected_points'])
    df = pd.merge(df, max_qb.drop('projected_points', axis=1), on=['team', 'year', 'week', 'fp_rank'])
    df = df[cols]

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


#--------------------
# Data to apply to all datasets
#--------------------

def get_max_qb():

    df = fantasy_pros_new('QB')
    df = pff_experts_new(df, 'QB')
    df = ffa_compile(df, 'FFA_Projections', 'QB')
    df = ffa_compile(df, 'FFA_RawStats', 'QB')
    df = fftoday_proj(df, 'QB')
    df = fantasy_cruncher(df, 'QB')
    df = get_salaries(df, 'QB')

    df = consensus_fill(df)
    df = fill_ratio_nulls(df)
    df = log_rank_cols(df)

    qb_cols = [
               'team', 'week', 'year', 
               'ffa_pass_yds', 'ffa_pass_tds','ffa_pass_int','ffa_rush_yds', 'ffa_rush_tds',
               'passComp', 'passAtt', 'passYds', 'passTd', 'passInt', 'passSacked', 'rushAtt', 'rushYds', 'rushTd',
               'fc_proj_passing_stats_att', 'fc_proj_passing_stats_yrds','fc_proj_passing_stats_tds', 'fc_proj_passing_stats_int',
               'fc_proj_rushing_stats_att',  'fc_proj_rushing_stats_yrds', 'fc_proj_rushing_stats_tds',
               'fft_proj_pts', 'fft_pass_att', 'fft_pass_int', 'fft_rush_yds', 'fft_rush_att', 'fft_pass_yds', 'fft_pass_td',
               'fft_rush_td', 'fft_pass_comp', 
               'avg_proj_pass_yds', 'avg_proj_pass_td',  'avg_proj_pass_int', 'avg_proj_pass_att',
               'avg_proj_rush_yds',  'avg_proj_rush_att', 'avg_proj_rush_td', 
               'ffa_points', 'projected_points', 'avg_proj_points', 'ProjPts', 
               'log_avg_proj_rank', 'log_playeradj_fp_rank', 'log_expertConsensus', 'avg_proj_rank',
               'fp_rank', 'log_ffa_position_rank', 'log_rankadj_fp_rank', 'dk_salary'
               ]
    df = df.sort_values(by=['team', 'year', 'week', 'projected_points', 'ffa_points'],
                        ascending=[True, True, True, False, False])
    df = df.drop_duplicates(subset=['team', 'year', 'week'], keep='first').reset_index(drop=True)
    df = df[qb_cols]
    df.columns = ['qb_'+c if c not in ('team', 'week', 'year') else c for c in df.columns]

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

def non_qb_team_pos_rank():
    team_pos_rank = pd.DataFrame()
    for pos in ['RB', 'WR', 'TE']:

        tp = fantasy_pros_new(pos)
        tp = pff_experts_new(tp, pos)
        tp = ffa_compile(tp, 'FFA_Projections', pos)
        tp = ffa_compile(tp, 'FFA_RawStats', pos)
        tp = fftoday_proj(tp, pos)
        tp = fantasy_cruncher(tp, pos)
        tp = consensus_fill(tp)
        tp = fill_ratio_nulls(tp)
        team_pos_rank = pd.concat([team_pos_rank, tp], axis=0)

    team_pos_rank = create_pos_rank(team_pos_rank, extra_pos=True)
    return  team_pos_rank[['player', 'pos', 'pos_rank', 'team', 'week', 'year']]

#%%

# create the scores and lines table
create_scores_lines_table(WEEK, YEAR)

# get datasets that will be used across positions
opp_defense = defense_for_pos()
team_proj, team_proj_pos = get_team_projections() # add injury removal here in the future
team_stats = get_team_stats()
team_qb = get_max_qb()
team_pos_rank = non_qb_team_pos_rank()
pff_oline = pff_oline_rollup()

#%%
pos = 'QB'
rush_or_pass = ''

def qb_pull(rush_or_pass):

    #-------------------
    # pre-game data
    #-------------------

    # pull all projections and ranks
    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fftoday_proj(df, pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])

    # clean up any missing values and engineer data
    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])
    df = rolling_proj_stats(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    df = add_weather(df); print(df.shape[0])

    #--------------------
    # Post-game data
    #--------------------

    # add player stats data
    df = get_player_data(df, pos); print(df.shape[0])

    df = add_rz_stats_qb(df); print(df.shape[0])
    df = add_qbr(df); print(df.shape[0])
    df = add_qb_adv(df); print(df.shape[0])
    df = add_next_gen(df, 'Passing'); print(df.shape[0])

    # merge self team and opposing team stats
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, opp_defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = def_pts_allowed(df); print(df.shape[0])
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    df = attach_y_act(df, pos, rush_or_pass=rush_or_pass)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna(axis=1, thresh=df.shape[0]-100).dropna().reset_index(drop=True); print(df.shape[0])

    df = one_qb_per_week(df); print(df.shape[0])

    df = remove_non_uniques(df)
    df = df[(df.ProjPts > 10) & (df.projected_points > 10)].reset_index(drop=True)
    df = drop_duplicate_players(df)
    df = remove_low_corrs(df, corr_cut=0.03)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df.iloc[:,:2000], 'Model_Features', f"QB_Data{rush_or_pass.replace('_', '')}", if_exist='replace', create_backup=True)
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:,2000:], 'Model_Features', f"QB_Data{rush_or_pass.replace('_', '')}2", if_exist='replace')

    return df

qb_both = qb_pull('')
# # qb_rush = qb_pull('_rush')
# # qb_pass = qb_pull('_pass')

#%%
for pos in ['RB', 'WR', 'TE']:

    #----------------
    # Pre game data
    #----------------

    # pull all projections and ranks
    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fftoday_proj(df, pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])

    # clean up any missing values and engineer data
    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])
    df = rolling_proj_stats(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    # add additional matchup and other pre-game data
    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    df = add_weather(df); print(df.shape[0])
    if pos == 'WR': df = cb_matchups(df); print(df.shape[0])
    if pos == 'TE': df = te_matchups(df); print(df.shape[0])

    #-----------------------
    # Post-Game Data
    #----------------------

    # add player stats
    df = get_player_data(df, pos); print(df.shape[0])

    # add advanced stats
    df = add_rz_stats(df); print(df.shape[0])
    df = advanced_rec_stats(df)
    if pos in ('WR', 'TE'):
        df = add_next_gen(df, 'Receiving'); print('next_gen', df.shape[0])
    if pos == 'RB': 
        df = advanced_rb_stats(df)
        df = add_next_gen(df, 'Rushing'); print(df.shape[0])

    df = def_pts_allowed(df); print(df.shape[0])

    # merge self team and opposing team stats
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj_pos, on=['pos', 'team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, opp_defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    # projection market share
    df = proj_market_share(df, 'team_proj_'); print(df.shape[0])
    df = proj_market_share(df, 'pos_proj_'); print(df.shape[0])
    
    # calc actual market share and trailing stats for pos ranks in a team
    df = calc_market_share(df); print(df.shape[0])
    df = pos_rank_stats(df, team_pos_rank, pos); print(df.shape[0])

    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df =  df.dropna(axis=1, thresh=df.shape[0]-100).dropna().reset_index(drop=True); print(df.shape[0])
    df = remove_non_uniques(df)
    df = drop_duplicate_players(df)
    df = remove_low_corrs(df, corr_cut=0.03)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df.iloc[:, :2000], 'Model_Features', f'{pos}_Data', if_exist='replace')
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:, 2000:], 'Model_Features', f'{pos}_Data2', if_exist='replace')

#%%

output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:

    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fftoday_proj(df, pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])

    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    df = get_salaries(df, pos); print(df.shape[0])
    df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = np.nan

    df = add_weather(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    
    # merge self team and opposing team stats
    if pos!= 'QB': df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
    
    if pos != 'QB': df = pd.merge(df, team_proj_pos, on=['pos', 'team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, opp_defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    # projection market share
    df = proj_market_share(df, 'team_proj_')
    if pos != 'QB': df = proj_market_share(df, 'pos_proj_')

    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df =  df.dropna(axis=1, thresh=df.shape[0]-100).dropna().reset_index(drop=True); print(df.shape[0])

    df['pos'] = pos
    if pos=='QB': df = one_qb_per_week(df); print(df.shape[0])
    
    df = drop_duplicate_players(df)
    df = remove_low_corrs(df, corr_cut=0.03)

    print('Data Size:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())
    
    output = pd.concat([output, df], axis=0)

output = def_pts_allowed(output); print(output.shape[0])
output = remove_non_uniques(output)

dm.write_to_db(output, 'Model_Features', 'Backfill', 'replace')

#%%

def create_self_cols(df):
    self_df = df.copy()
    self_cols = ['team', 'week','year']
    self_cols.extend(['self_'+c for c in self_df.columns if c not in ('team', 'week', 'year')])
    self_df.columns = self_cols
    return self_df

# defense stats that can be added to the offensive player data
defense = fantasy_pros_new('DST')
defense = add_fp_rolling(defense, 'Defense'); print(defense.shape[0])

defense = defense.rename(columns={'player': 'defTeam'})
defense = add_ffa_defense(defense)
defense = defense.drop(['pos', 'position', 'team'], axis=1)

pff_def = add_team_matchups()
defense = pd.merge(defense, pff_def, on=['defTeam', 'year', 'week'])

defense = fantasy_cruncher(defense.rename(columns={'defTeam': 'player'}), 'DST')
defense = defense.rename(columns={'player': 'defTeam'})

defense = fantasy_data_proj(defense.rename(columns={'defTeam': 'player'}), 'DST')
defense = defense.rename(columns={'player': 'defTeam'})

defense = consensus_fill(defense, is_dst=True)
defense = fill_ratio_nulls(defense)
defense = log_rank_cols(defense)
defense = rolling_proj_stats(defense.rename(columns={'defTeam':'player'}))
defense = defense.rename(columns={'player':'team'})

all_cols = [c for c in defense.columns if c != 'y_act']
defense = defense.dropna(subset=all_cols)

defense = add_gambling_lines(defense); print(defense.shape[0])
defense = add_weather(defense); print(defense.shape[0])


team_qb_self = create_self_cols(team_qb)
defense = pd.merge(defense, team_qb_self, on=['team', 'week', 'year'])
defense = pd.merge(defense, team_qb.rename(columns={'team': 'offTeam'}), 
                    on=['offTeam', 'week', 'year'], how='left')

team_proj_self = create_self_cols(team_proj)
defense = pd.merge(defense, team_proj_self, on=['team', 'week', 'year'])
defense = pd.merge(defense, team_proj.rename(columns={'team': 'offTeam'}), 
                   on=['offTeam', 'week', 'year']); print( defense.shape[0])

team_stats = get_team_stats()
defense = pd.merge(defense, team_stats.rename(columns={'team': 'offTeam'}), 
                   on=['offTeam', 'week', 'year']); print(defense.shape[0])

d_stats = get_defense_stats()
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')

pff_oline = pff_oline_rollup()
defense = pd.merge(defense, pff_oline.rename(columns={'team': 'offTeam'}), 
                    on=['offTeam', 'week', 'year']); print(defense.shape[0])

pff_def = pff_defense_rollup()
defense = pd.merge(defense, pff_def, on=['team', 'week', 'year'])

defense = defense.copy().rename(columns={'team': 'player'})
defense = forward_fill(defense)

defense = attach_y_act(defense, pos='Defense', defense=True)
defense = drop_y_act_except_current(defense, WEEK, YEAR); print(defense.shape[0])
defense = defense.dropna(axis=1, thresh=defense.shape[0]-100).dropna(); print(defense.shape[0])

print('Unique player-week-years:', defense[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', defense[['year', 'week', 'player']].drop_duplicates().groupby(['year', 'week'])['player'].count())

defense.columns = [c.replace('_dst', '') for c in defense.columns]
defense = remove_non_uniques(defense)
defense = remove_low_corrs(defense, corr_cut=0.02)

dm.write_to_db(defense, 'Model_Features', f'Defense_Data', if_exist='replace')


#%%

chk_week = 17
backfill_chk = dm.read(f"SELECT player FROM Backfill WHERE week={chk_week} AND year={YEAR}", 'Model_Features').player.values
sal = dm.read(f"SELECT player, salary FROM Salaries WHERE week={chk_week} AND year={YEAR}", 'Simulation')
sal[~sal.player.isin(backfill_chk)].sort_values(by='salary', ascending=False).iloc[:50]

#%%
chk_pos='WR'
backfill_chk = dm.read(f"SELECT player FROM {chk_pos}_Data WHERE week={WEEK-1} AND year={YEAR}", 'Model_Features').player.values
sal = dm.read(f'''SELECT player, salary 
                  FROM Salaries 
                  LEFT JOIN (SELECT DISTINCT player, pos FROM Model_Predictions WHERE year={YEAR}) USING (player)
                  WHERE league={WEEK-1} 
                        AND year={YEAR}
                        AND pos='{chk_pos}'
                  ''', 'Simulation')
sal[~sal.player.isin(backfill_chk)].sort_values(by='salary', ascending=False).iloc[:50]

#%%
count_chk = dm.read(f"SELECT player, week, year, count(*) cnts FROM Backfill GROUP BY player, week, year", 'Model_Features')
count_chk[count_chk.cnts > 1]

#%%

output.loc[(output.week==18) & (output.year==2022), ['player', 'y_act']]


# %%
# TO DO LIST
# - add in PFF scores
# - add in snaps and snap share

#%%

#==================
# Team Points Predictions
#==================
output['avg_pts'] = output[['ProjPts', 'fantasyPoints', 'projected_points']].mean(axis=1)
output = output.sort_values(by=['year', 'week', 'team', 'avg_pts'],
                            ascending=[True, True, True, False]).reset_index(drop=True)

team_pts = output.groupby(['year', 'week', 'team']).agg({'avg_pts': 'sum', 'y_act': 'sum'}).reset_index()

team_off = dm.read("SELECT * FROM Defense_Data", 'Model_Features').drop('y_act', axis=1)
team_off = team_off.rename(columns={'player': 'defTeam', 'offTeam': 'team'})
team_off = pd.merge(team_pts, team_off, on=['team', 'week', 'year'])
team_off = team_off.rename(columns={'team': 'player'})
team_off['team'] = team_off.player

print('Unique team-week-years:', team_off[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', team_off[['year', 'week', 'player']].drop_duplicates().groupby(['year', 'week'])['player'].count())

dm.write_to_db(team_off, 'Model_Features', f'Team_Offense_Data', if_exist='replace')

#%%

#==================
# Find missing players
#==================
cur_pos = 'RB'

dk_sal = dm.read('''SELECT player, team, week, year, dk_salary
                    FROM Daily_Salaries
                    WHERE dk_salary > 5500 
                          AND position='QB'
                    UNION
                    SELECT player, team, week, year, dk_salary
                    FROM Daily_Salaries
                    WHERE dk_salary > 4500 
                          AND position!='QB' ''', "Pre_PlayerData")

pff = dm.read('''SELECT player, offTeam team, week, year, expertConsensus, fantasyPoints, `Proj Pts` ProjPts
                    FROM PFF_Expert_Ranks
                    JOIN (SELECT player, week, year, fantasyPoints
                        FROM PFF_Proj_Ranks)
                        USING (player, week, year) ''', "Pre_PlayerData")

inj = dm.read('''SELECT player, week, year, 1 as is_out
                 FROM PlayerInjuries
                 WHERE game_status IN ('Out', 'Doubtful') 
                       AND pos in ('QB', 'RB', 'WR', 'TE')''', 'Pre_PlayerData')

data = pd.merge(dk_sal, pff, on=['player', 'team', 'week', 'year'], how='left')
data = pd.merge(data, inj, on=['player',  'week', 'year'], how='left')
data.is_out = data.is_out.fillna(0)
data
# missing_game = data.loc[(data.is_out==1) | (data.expertConsensus.isnull()),
#                         ['player', 'team', 'week', 'year', 'dk_salary']]

# pos = dm.read('''SELECT DISTINCT player, team, year, pos
#                  FROM FantasyPros
#                  ''', "Pre_PlayerData")

# missing_game = pd.merge(missing_game, pos, on=['player', 'team', 'year'])
# missing_game = missing_game.groupby(['team', 'pos', 'week', 'year']).agg({'dk_salary': 'sum'}).reset_index()
# missing_game = missing_game.rename(columns={'dk_salary': 'missing_salary'})
# missing_game_pos = missing_game[missing_game.pos==cur_pos].drop('pos', axis=1)

# xx = pd.merge(df, missing_game_pos, on=['team', 'week', 'year'], how='left').fillna({'missing_salary': 0})

# missing_game[missing_game.team=='SEA'].iloc[:50]
# %%

# Chase Brown
# Mitch Trubiskey
# Chris Rodriguez