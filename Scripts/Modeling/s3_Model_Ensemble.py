#%%
import datetime as dt
import pandas as pd
import numpy as np
import sqlite3
import sys
import os
from IPython.display import display
from sklearn.metrics import mean_squared_error, r2_score
# Add Scripts directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YEAR, LEAGUE, PRED_VERSION

from ff.db_operations import DataManage
from ff import general
from zProjection_Validation import build_final_validation_residuals


#==========
# General Setting
#==========

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

set_year = YEAR
vers = LEAGUE

#%%
playoff_pts = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    playoffs = dm.read(f'''SELECT player, season, fantasy_pts_per_game playoff_pts 
                           FROM {pos}_Stats
                           WHERE games =3''', 
                           'Season_Stats_Playoffs')
    playoff_pts = pd.concat([playoff_pts, playoffs], ignore_index=True)

proj_pts = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    df_pos = dm.read(f'''SELECT player, 
                                year+1 as season, 
                                avg_proj_points/16.5 as next_year_proj_points
                         FROM {pos}_{set_year}_ProjOnly 
                         WHERE pos='{pos}' ''', f'Model_Inputs')

    proj_pts = pd.concat([proj_pts, df_pos], ignore_index=True).fillna(0)


model_proj = dm.read(f'''
                SELECT player, 
                        year as season, 
                        current_or_next_year,
                        AVG(pred_fp_per_game) avg_pred
                FROM Model_Predictions
                WHERE version = '{vers}'
                      AND rush_pass NOT IN ('rush', 'pass', 'rec')
                      AND dataset='ProjOnly'
                GROUP BY player, season, current_or_next_year
''', 'Simulation')

avg_model_proj=model_proj.groupby(['player', 'season']).agg({'avg_pred': 'mean'}).reset_index()

for cur_next in ['current', 'next']:
    print("===================================")
    print('Running for:', cur_next)
    cur_df = pd.merge(proj_pts, model_proj, on=['player', 'season'])
    cur_df_proj = cur_df[cur_df.current_or_next_year == cur_next].reset_index()
    print('R2 Score Proj:', r2_score(cur_df_proj['next_year_proj_points'], cur_df_proj['avg_pred']))
    print('MSE Proj:', mean_squared_error(cur_df_proj['next_year_proj_points'], cur_df_proj['avg_pred']))
    cur_df_proj.plot.scatter(x='next_year_proj_points', y='avg_pred', title=f'Next Year Projections - {cur_next}')

    cur_df = pd.merge(model_proj, playoff_pts, on=['player', 'season'])
    cur_df_playoffs = cur_df[cur_df.current_or_next_year == cur_next].reset_index()
    print('R2 Score Playoffs:', r2_score(cur_df_playoffs['playoff_pts'], cur_df_playoffs['avg_pred']))
    print('MSE Playoffs:', mean_squared_error(cur_df_playoffs['playoff_pts'], cur_df_playoffs['avg_pred']))
    cur_df_playoffs.plot.scatter(x='playoff_pts', y='avg_pred', title=f'Playoff Points - {cur_next}')

print("===================================")
print('Running for: Combined')

cur_df = pd.merge(avg_model_proj, proj_pts, on=['player', 'season'])
print('R2 Score Playoffs:', r2_score(cur_df['next_year_proj_points'], cur_df['avg_pred']))
print('MSE Playoffs:', mean_squared_error(cur_df['next_year_proj_points'], cur_df['avg_pred']))
cur_df.plot.scatter(x='next_year_proj_points', y='avg_pred', title=f'Next Year Projections - Average')

cur_df = pd.merge(avg_model_proj, playoff_pts, on=['player', 'season'])
print('R2 Score Proj:', r2_score(cur_df['playoff_pts'], cur_df['avg_pred']))
print('MSE Proj:', mean_squared_error(cur_df['playoff_pts'], cur_df['avg_pred']))
cur_df.plot.scatter(x='playoff_pts', y='avg_pred', title=f'Playoff Points - Average')

#%%

model_proj = dm.read(f'''
                SELECT player, 
                        year as season, 
                        current_or_next_year,
                        AVG(pred_fp_per_game) avg_pred
                FROM Model_Predictions_Resid
                WHERE version = '{vers}'
                      AND rush_pass NOT IN ('rush', 'pass', 'rec')
                      AND dataset='ProjOnly'
                      AND pos!='QB'
                GROUP BY player, season, current_or_next_year
''', 'Simulation')

xx = pd.pivot(model_proj, columns=['current_or_next_year'], index=['player', 'season'])
xx.columns = [x[1] for x in xx.columns]
xx = xx.reset_index()
xx['pts_diff'] = xx['next'] - xx['current']
xx[(xx.season==2026) & (xx.current > 8)].sort_values(by='pts_diff', ascending=False).head(50)

#%%

#==========
# Check Rush Pass vs All Weighting
#==========
set_pos = 'QB'
current_or_next_year = 'current'
dataset = 'ProjOnly'
year_exp = 0
filter_data = 'greater_equal'

from sklearn.metrics import mean_squared_error, r2_score

rp = dm.read(f'''
                SELECT player, season, SUM(rp_pred) rp_pred, SUM(rp_y_act) rp_y_act
             FROM (
                SELECT player, 
                        season, 
                        rush_pass,
                        AVG(pred_fp_per_game) rp_pred, 
                        AVG(y_act) rp_y_act
                FROM Model_Validations_Resid
                WHERE rush_pass in ('rush', 'pass', 'rec')
                      AND pos = '{set_pos}'
                      AND year_exp={year_exp}
                      AND filter_data = '{filter_data}'
                      AND current_or_next_year = '{current_or_next_year}'
                      AND year = '{set_year}'
                      AND version = '{vers}'
                      AND dataset = '{dataset}'
                GROUP BY player, season, rush_pass
                )
                GROUP BY player, season
             ''', 'Validations')

both = dm.read(f'''SELECT player, 
                         season, 
                         AVG(pred_fp_per_game) both_pred, 
                         AVG(y_act) both_y_act
                FROM Model_Validations_Resid
                WHERE rush_pass NOT IN ('rush', 'pass', 'rec')
                      AND pos = '{set_pos}'
                      AND year_exp={year_exp}
                      AND filter_data = '{filter_data}'
                      AND current_or_next_year = '{current_or_next_year}'
                      AND year = '{set_year}'
                      AND version = '{vers}'
                      AND dataset = '{dataset}'
                GROUP BY player, season
                ''', 'Validations')

rp = pd.merge(rp, both, on=['player', 'season'])
rp['avg_pred'] = (rp.rp_pred + rp.both_pred) / 2
rp['y_act_avg'] = (rp.rp_y_act + rp.both_y_act) / 2
rp.plot.scatter(x='rp_pred', y='rp_y_act')
rp.plot.scatter(x='both_pred', y='both_y_act')
rp.plot.scatter(x='avg_pred', y='y_act_avg')

print('MSE Both:', mean_squared_error(rp.both_y_act, rp.both_pred))
print('R2 Both:', r2_score(rp.both_y_act, rp.both_pred))
print('MSE Rush/Pass:', mean_squared_error(rp.rp_y_act, rp.rp_pred))
print('R2 Rush/Pass:', r2_score(rp.rp_y_act, rp.rp_pred))
print('MSE Avg:', mean_squared_error(rp.y_act_avg, rp.avg_pred))
print('R2 Avg:', r2_score(rp.y_act_avg, rp.avg_pred))


#%%

# #===========
# # Rookie Val Ratios
# #===========

# def get_val_ratio(vers, set_year, pos, dataset):

#     val = dm.read(f'''SELECT player, 
#                             season, 
#                             dataset,
#                             AVG(pred_fp_per_game) pred, 
#                             AVG(y_act) y_act
#             FROM Model_Validations
#             WHERE version='{vers}'
#                 AND year = {set_year}
#                 AND dataset {dataset}
#                 AND rush_pass NOT IN ('rush', 'pass', 'rec')
#                 AND pos = '{pos}'
#                 AND current_or_next_year = 'current'
#             GROUP BY player, season
#         ''', 'Validations')
    

#     y_act_max = np.mean([np.percentile(val.y_act, 94), 
#                          np.percentile(val.y_act, 95), 
#                          np.percentile(val.y_act, 96), 
#                          np.percentile(val.y_act, 97),
#                          ])
#     pred_max = np.mean([np.percentile(val.pred, 94), 
#                         np.percentile(val.pred, 95), 
#                         np.percentile(val.pred, 96), 
#                         np.percentile(val.pred, 97)
#                         ])
#     return y_act_max/pred_max

# rookie_wr_ratio = []
# for pos in ['WR', 'RB']:
#     pos_val = get_val_ratio(vers, set_year, pos, 'NOT LIKE "%Rookie%"')
#     rookie_val = get_val_ratio(vers, set_year, 'WR', 'LIKE "%Rookie%"')
#     rookie_ratio_cur = rookie_val - pos_val + 1
#     rookie_wr_ratio.append(rookie_ratio_cur)

# rookie_wr_ratio = np.mean(rookie_wr_ratio)
# print('Rookie WR Ratio:', rookie_wr_ratio)


# rookie_rb_ratio = []
# for pos in ['RB']:
#     pos_val = get_val_ratio(vers, set_year, pos, 'NOT LIKE "%Rookie%"')
#     rookie_val = get_val_ratio(vers, set_year, 'RB', 'LIKE "%Rookie%"')
#     rookie_ratio_cur = rookie_val - pos_val + 1
#     rookie_rb_ratio.append(rookie_ratio_cur)

# rookie_rb_ratio = np.mean(rookie_rb_ratio)
# print('Rookie RB Ratio:', rookie_rb_ratio)



#%%

# rookies = dm.read(f'''SELECT player, 
#                              pos,
#                              rush_pass,
#                              AVG(pred_fp_per_game) pred_fp_per_game,
#                              AVG(pred_fp_per_game_upside) pred_prob_upside,
#                              AVG(pred_fp_per_game_top) pred_prob_top,
#                              AVG(std_dev) std_dev,
#                              AVG(min_score) min_score,   
#                              AVG(max_score) max_score
#                 FROM Model_Predictions
#                 WHERE version='{vers}'
#                        AND year = {set_year}
#                        AND dataset LIKE '%Rookie%'
#                 GROUP BY player, pos, rush_pass
#              ''', 'Simulation').sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

# rookies.loc[rookies.pos=='WR', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] = \
#     rookies.loc[rookies.pos=='WR', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] * 1.05#* rookie_wr_ratio

# rookies.loc[rookies.pos=='RB', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] = \
#     rookies.loc[rookies.pos=='RB', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] * 1.05#* rookie_rb_ratio

# display(rookies.iloc[:50])


#%%

resid_percentiles = [5, 10, 25, 75, 90, 95]
resid_cols = [f'pred_resid_{p}' for p in resid_percentiles]
resid_avg_sql = ',\n                        '.join([f'AVG({c}) {c}' for c in resid_cols])


def enforce_resid_order(df, cols):
    cols = [c for c in cols if c in df.columns]
    if cols:
        df[cols] = np.maximum.accumulate(df[cols].to_numpy(), axis=1)
    return df


def estimate_component_rho(default_rho=0.35, min_samples=50):
    val = dm.read(f'''SELECT player,
                             season,
                             pos,
                             rush_pass,
                             AVG(pred_fp_per_game) pred_fp_per_game,
                             AVG(y_act) y_act
                      FROM Model_Validations_Resid
                      WHERE rush_pass IN ('rush', 'pass', 'rec')
                            AND version='{vers}'
                            AND year = {set_year}
                            AND dataset NOT LIKE '%Rookie%'
                            AND current_or_next_year = 'current'
                      GROUP BY player, season, pos, rush_pass
                   ''', 'Validations')

    if len(val) == 0:
        return {}, pd.DataFrame()

    val['resid'] = val.y_act - val.pred_fp_per_game
    val = val.pivot_table(
        index=['player', 'season', 'pos'],
        columns='rush_pass',
        values='resid',
        aggfunc='mean'
    )

    rho_records = []
    for pos, pos_val in val.groupby(level='pos'):
        pos_val = pos_val.dropna(axis=1, how='all')
        comp_cols = [c for c in ['rush', 'pass', 'rec'] if c in pos_val.columns]
        pos_val = pos_val[comp_cols].dropna()

        rho = default_rho
        if len(comp_cols) > 1 and len(pos_val) >= min_samples:
            cov = pos_val.cov()
            std = pos_val.std(ddof=1)
            cov_sum = 0
            std_prod_sum = 0

            for i, c1 in enumerate(comp_cols):
                for c2 in comp_cols[i + 1:]:
                    cov_sum += cov.loc[c1, c2]
                    std_prod_sum += std[c1] * std[c2]

            if std_prod_sum > 0:
                rho = cov_sum / std_prod_sum
                rho = np.clip(rho, 0, 0.95)

        rho_records.append({
            'pos': pos,
            'rho': rho,
            'samples': len(pos_val),
            'components': ','.join(comp_cols),
        })

    rho_df = pd.DataFrame(rho_records)
    return dict(zip(rho_df.pos, rho_df.rho)), rho_df


def combine_component_residuals(group, lower_cols, upper_cols, default_rho=0.35):
    pos = group.name[1]
    rho = component_rho.get(pos, default_rho)
    out = {'pred_fp_per_game': group.pred_fp_per_game.sum()}

    for col in lower_cols + upper_cols:
        vals = group[col].dropna().to_numpy()
        if len(vals) == 0:
            out[col] = np.nan
            continue

        if len(vals) == 1:
            out[col] = vals[0]
            continue

        sign = -1 if col in lower_cols else 1
        vals = np.minimum(vals, 0) if sign < 0 else np.maximum(vals, 0)
        vals = np.abs(vals)
        resid_var = ((1 - rho) * np.sum(vals**2)) + (rho * (np.sum(vals) ** 2))
        out[col] = sign * np.sqrt(max(resid_var, 0))

    return pd.Series(out)


lower_resid_cols = [c for c in resid_cols if c in ['pred_resid_5', 'pred_resid_10', 'pred_resid_25']]
upper_resid_cols = [c for c in resid_cols if c in ['pred_resid_75', 'pred_resid_90', 'pred_resid_95']]
component_rho, component_rho_df = estimate_component_rho()
display(component_rho_df)


rp = dm.read(f'''SELECT player, 
                        pos,
                        rush_pass,
                        AVG(pred_fp_per_game) pred_fp_per_game,
                        {resid_avg_sql}
                FROM Model_Predictions_Resid
                WHERE rush_pass IN ('rush', 'pass', 'rec')
                      AND version='{vers}'
                      AND year = {set_year}
                      AND dataset NOT LIKE '%Rookie%'
                      AND current_or_next_year = 'current'
                GROUP BY player, pos, rush_pass
             ''', 'Simulation')

rp = (
    rp.groupby(['player', 'pos'])[['pred_fp_per_game', *resid_cols]]
      .apply(combine_component_residuals, lower_resid_cols, upper_resid_cols)
      .reset_index()
)
rp = enforce_resid_order(rp, resid_cols)
rp = rp.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
rp['ensemble_source'] = 'rush_pass_rec'
display(rp[((rp.pos=='QB'))].iloc[:15])
display(rp[((rp.pos!='QB'))].iloc[:50])

#%%


preds_ty = dm.read(f'''SELECT player, 
                        pos,
                        rush_pass,
                        AVG(pred_fp_per_game) pred_fp_per_game,
                        {resid_avg_sql}
                FROM Model_Predictions_Resid
                WHERE rush_pass NOT IN ('rush', 'pass', 'rec')
                       AND version='{vers}'
                       AND year = {set_year}
                       AND dataset NOT LIKE '%Rookie%'
                       AND current_or_next_year = 'current'
                GROUP BY player, pos, rush_pass
             ''', 'Simulation').sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
preds_ty['ensemble_source'] = 'all_current'

display(preds_ty[((preds_ty.pos=='QB'))].iloc[:15])
display(preds_ty[((preds_ty.pos!='QB'))].iloc[:50])

#%%

preds_ny = dm.read(f'''SELECT player, 
                        pos,
                        rush_pass,
                        AVG(pred_fp_per_game) pred_fp_per_game,
                        {resid_avg_sql}
                FROM Model_Predictions_Resid
                WHERE rush_pass NOT IN ('rush', 'pass', 'rec')
                       AND version='{vers}'
                       AND year = {set_year}
                       AND dataset NOT LIKE '%Rookie%'
                       AND current_or_next_year = 'next'
                       AND pos != 'QB'
                GROUP BY player, pos, rush_pass
             ''', 'Simulation').sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
preds_ny['ensemble_source'] = 'all_next'

display(preds_ny[((preds_ny.pos=='QB'))].iloc[:15])
display(preds_ny[((preds_ny.pos!='QB'))].iloc[:50])

#%%

ensemble_frames = [df for df in [rp, preds_ty, preds_ny] if len(df) > 0]
if not ensemble_frames:
    raise ValueError(f"No Model_Predictions_Resid rows found for version={vers} year={set_year}.")

preds = pd.concat(ensemble_frames, axis=0).reset_index(drop=True)
preds = preds.groupby(['player', 'pos'], as_index=False).agg({
    'pred_fp_per_game': 'mean',
    **{c: 'mean' for c in resid_cols}
})
preds = enforce_resid_order(preds, resid_cols)

preds = preds[preds.pred_fp_per_game > 0].reset_index(drop=True)

preds_ny_cpy = preds_ny[['player', 'pos', 'pred_fp_per_game', *resid_cols]].copy()
preds_ny_cpy = preds_ny_cpy.rename(columns={
    'pred_fp_per_game': 'pred_fp_per_game_ny',
    **{c: f'{c}_ny' for c in resid_cols}
})
preds = pd.merge(preds, preds_ny_cpy, on=['player', 'pos'], how='left')

ny_fill_cols = ['pred_fp_per_game_ny', *[f'{c}_ny' for c in resid_cols]]
current_fill_cols = ['pred_fp_per_game', *resid_cols]
preds.loc[preds.pred_fp_per_game_ny.isnull(), ny_fill_cols] = (
    preds.loc[preds.pred_fp_per_game_ny.isnull(), current_fill_cols].values
)
preds = enforce_resid_order(preds, [f'{c}_ny' for c in resid_cols])

preds['dataset'] = 'final_ensemble'
preds['version'] = vers
preds['year'] = set_year
preds = preds.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

preds['pos_rank'] = preds.groupby('pos')['pred_fp_per_game'].rank(ascending=False, method='first')

if vers == 'nv': 
    num_qb = 33
    num_te = 24
    num_rb = 60
    num_wr = 72
elif vers == 'beta': 
    num_qb = 24
    num_te = 24
    num_rb = 60
    num_wr = 72
elif vers == 'dk': 
    num_qb = 40
    num_te = 36
    num_rb = 84
    num_wr = 108
elif vers == 'nffc': 
    num_qb = 36
    num_te = 36
    num_rb = 96
    num_wr = 120

preds = preds[~((preds.pos=='QB') & (preds.pos_rank > num_qb))].reset_index(drop=True)
preds = preds[~((preds.pos=='TE') & (preds.pos_rank > num_te))].reset_index(drop=True)
preds = preds[~((preds.pos=='RB') & (preds.pos_rank > num_rb))].reset_index(drop=True)
preds = preds[~((preds.pos=='WR') & (preds.pos_rank > num_wr))].reset_index(drop=True).drop('pos_rank', axis=1)

display(preds[((preds.pos=='QB'))].iloc[:50])
display(preds[((preds.pos!='QB'))].iloc[:50])

#%%
downgrades = {
    'Zach Charbonnet': 0.7,
    'George Kittle': 0.85,
    'Travis Kelce': 0.85,
    'Sam Laporta': 0.9,
    'Derrick Henry': 0.9
}

for p, d in downgrades.items():
    adjust_cols = ['pred_fp_per_game', 'pred_fp_per_game_ny', *resid_cols, *[f'{c}_ny' for c in resid_cols]]
    preds.loc[preds.player==p, adjust_cols] = preds.loc[preds.player==p, adjust_cols] * d

preds = enforce_resid_order(preds, resid_cols)
preds = enforce_resid_order(preds, [f'{c}_ny' for c in resid_cols])

#%%

yoe = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    df_pos = dm.read(f'''SELECT player, year_exp Years_of_Experience
                         FROM {pos}_{set_year}_ProjOnly 
                         WHERE year={set_year}
                               AND pos='{pos}' ''', f'Model_Inputs')
    
    yoe = pd.concat([yoe, df_pos], ignore_index=True).fillna(0)

adps = dm.read(f"SELECT * FROM ADP_Averages WHERE year={set_year}", 'Season_Stats_New')
adps = pd.merge(adps, yoe, on='player', how='left')
adps = adps.drop('pos', axis=1)
dm.delete_from_db('Simulation', 'Avg_ADPs', f"year={set_year}", create_backup=True)
dm.write_to_db(adps, 'Simulation', 'Avg_ADPs', if_exist='append')

etr = dm.read(f'''SELECT player, etr_rank as avg_pick, 'etr' as league, year
                  FROM ETR_Ranks 
                  WHERE year={set_year}''', 
                  'Season_Stats_New')
dm.write_to_db(etr, 'Simulation', 'Avg_ADPs', if_exist='append')

# %%
import shutil

final_resid_table = 'Final_Predictions_Resid'
final_resid_exists = dm.read(f'''
    SELECT name
    FROM sqlite_master
    WHERE type='table'
          AND name='{final_resid_table}'
''', 'Simulation')

if len(final_resid_exists) > 0:
    existing_preds = dm.read(f'SELECT * FROM {final_resid_table}', 'Simulation')
    if {'version', 'year', 'dataset'}.issubset(existing_preds.columns):
        keep_preds = existing_preds[
            ~(
                (existing_preds.version == vers) &
                (existing_preds.year == set_year) &
                (existing_preds.dataset == 'final_ensemble')
            )
        ].copy()
    else:
        keep_preds = existing_preds.iloc[0:0].copy()
    final_preds = pd.concat([keep_preds, preds], ignore_index=True, sort=False)
    dm.write_to_db(final_preds, 'Simulation', final_resid_table, 'replace', create_backup=True)
else:
    dm.write_to_db(preds, 'Simulation', final_resid_table, 'replace')

validation_rows = dm.read(f'''
    SELECT *
    FROM Model_Validations_Resid
    WHERE version='{vers}'
          AND year={set_year}
''', 'Validations')
final_validation_rows = build_final_validation_residuals(validation_rows)
if not len(final_validation_rows):
    raise ValueError(
        f'No final validation rows built for version={vers}, year={set_year}.'
    )

final_validation_table = 'Final_Validations_Resid'
final_validation_exists = dm.read(f'''
    SELECT name
    FROM sqlite_master
    WHERE type='table'
          AND name='{final_validation_table}'
''', 'Validations')
if len(final_validation_exists):
    existing_validation_rows = dm.read(
        f'SELECT * FROM {final_validation_table}',
        'Validations',
    )
    keep_validation_rows = existing_validation_rows[
        ~(
            existing_validation_rows.version.eq(vers)
            & existing_validation_rows.model_spec_asof_year.eq(set_year)
        )
    ].copy()
    final_validation_rows = pd.concat(
        [keep_validation_rows, final_validation_rows],
        ignore_index=True,
        sort=False,
    )

dm.write_to_db(
    final_validation_rows,
    'Validations',
    final_validation_table,
    'replace',
    create_backup=bool(len(final_validation_exists)),
)
with sqlite3.connect(f'{db_path}/Validations.sqlite3') as validation_conn:
    validation_conn.execute(f'''
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_final_validations_resid_identity
        ON {final_validation_table}
           (version, model_spec_asof_year, season, player, pos)
    ''')

src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3'
shutil.copyfile(src, dst)


src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_Snake/app/Simulation.sqlite3'
shutil.copyfile(src, dst)

#%%


