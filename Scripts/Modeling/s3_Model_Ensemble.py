#%%
import datetime as dt
import pandas as pd
import numpy as np
from ff.db_operations import DataManage
from ff import general


#==========
# General Setting
#==========

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

set_year=2025
vers='beta'
date_mod = dt.date(2025, 7, 4)

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
                FROM Model_Validations
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
                FROM Model_Validations
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

#===========
# Rookie Val Ratios
#===========

def get_val_ratio(vers, set_year, pos, dataset):

    val = dm.read(f'''SELECT player, 
                            season, 
                            dataset,
                            AVG(pred_fp_per_game) pred, 
                            AVG(y_act) y_act
            FROM Model_Validations
            WHERE version='{vers}'
                AND year = {set_year}
                AND dataset {dataset}
                AND rush_pass NOT IN ('rush', 'pass', 'rec')
                AND pos = '{pos}'
                AND current_or_next_year = 'current'
            GROUP BY player, season
        ''', 'Validations')
    

    y_act_max = np.percentile(val.y_act, 96)
    pred_max = np.percentile(val.pred, 96)
    return y_act_max/pred_max

rookie_wr_ratio = []
for pos in ['WR', 'RB']:
    pos_val = get_val_ratio(vers, set_year, pos, 'NOT LIKE "%Rookie%"')
    rookie_val = get_val_ratio(vers, set_year, 'WR', 'LIKE "%Rookie%"')
    rookie_ratio_cur = rookie_val - pos_val + 1
    rookie_wr_ratio.append(rookie_ratio_cur)

rookie_wr_ratio = np.mean(rookie_wr_ratio)
print('Rookie WR Ratio:', rookie_wr_ratio)


rookie_rb_ratio = []
for pos in ['RB']:
    pos_val = get_val_ratio(vers, set_year, pos, 'NOT LIKE "%Rookie%"')
    rookie_val = get_val_ratio(vers, set_year, 'RB', 'LIKE "%Rookie%"')
    rookie_ratio_cur = rookie_val - pos_val + 1
    rookie_rb_ratio.append(rookie_ratio_cur)

rookie_rb_ratio = np.mean(rookie_rb_ratio)
print('Rookie RB Ratio:', rookie_rb_ratio)



#%%

rookies = dm.read(f'''SELECT player, 
                             pos,
                             rush_pass,
                             AVG(pred_fp_per_game) pred_fp_per_game,
                             AVG(pred_fp_per_game_upside) pred_prob_upside,
                             AVG(pred_fp_per_game_top) pred_prob_top,
                             AVG(std_dev) std_dev,
                             AVG(min_score) min_score,   
                             AVG(max_score) max_score,
                             MAX(date_modified) date_modified
                FROM Model_Predictions
                WHERE version='{vers}'
                       AND year = {set_year}
                       AND dataset LIKE '%Rookie%'
                GROUP BY player, pos, rush_pass
             ''', 'Simulation').sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

rookies.date_modified = pd.to_datetime(rookies.date_modified).apply(lambda x: x.date())
rookies = rookies[rookies.date_modified >= date_mod].reset_index(drop=True)
rookies.loc[rookies.pos=='WR', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] = \
    rookies.loc[rookies.pos=='WR', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] * 1.05#* rookie_wr_ratio

rookies.loc[rookies.pos=='RB', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] = \
    rookies.loc[rookies.pos=='RB', ['pred_fp_per_game', 'max_score', 'pred_prob_upside', 'pred_prob_top']] * 1.05#* rookie_rb_ratio

display(rookies.iloc[:50])


#%%

rp = dm.read(f'''SELECT player, 
                        pos,
                        rush_pass,
                        AVG(pred_fp_per_game) pred_fp_per_game,
                        AVG(pred_fp_per_game_upside) pred_prob_upside,
                        AVG(pred_fp_per_game_top) pred_prob_top,
                        AVG(std_dev) std_dev,
                        AVG(min_score) min_score,   
                        AVG(max_score) max_score, 
                        MAX(date_modified) date_modified
                FROM Model_Predictions
                WHERE rush_pass IN ('rush', 'pass', 'rec')
                      AND version='{vers}'
                      AND year = {set_year}
                GROUP BY player, pos, rush_pass
             ''', 'Simulation')

rp.date_modified = pd.to_datetime(rp.date_modified).apply(lambda x: x.date())
rp = rp[rp.date_modified >= date_mod].reset_index(drop=True)

wm = lambda x: np.average(x, weights=rp.loc[x.index, "pred_fp_per_game"])
rp = rp.assign(std_dev=rp.std_dev**2, max_score=rp.max_score**2)
rp = rp.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                        'pred_prob_upside': wm,
                                                        'pred_prob_top': wm,
                                                        'std_dev': 'sum',
                                                        'min_score': 'sum',
                                                        'max_score': 'sum'})
rp = rp.assign(std_dev=np.sqrt(rp.std_dev), max_score=np.sqrt(rp.max_score))
rp = rp.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
display(rp[((rp.pos=='QB'))].iloc[:15])
display(rp[((rp.pos!='QB'))].iloc[:50])

#%%


preds_ty = dm.read(f'''SELECT player, 
                        pos,
                        rush_pass,
                        AVG(pred_fp_per_game) pred_fp_per_game,
                        AVG(pred_fp_per_game_upside) pred_prob_upside,
                        AVG(pred_fp_per_game_top) pred_prob_top,
                        AVG(std_dev) std_dev,
                        AVG(min_score) min_score,   
                        AVG(max_score) max_score, 
                        MAX(date_modified) date_modified
                FROM Model_Predictions
                WHERE rush_pass NOT IN ('rush', 'pass', 'rec')
                       AND version='{vers}'
                       AND year = {set_year}
                       AND dataset NOT LIKE '%Rookie%'
                       AND current_or_next_year = 'current'
                GROUP BY player, pos, rush_pass
             ''', 'Simulation').sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

preds_ty.date_modified = pd.to_datetime(preds_ty.date_modified).apply(lambda x: x.date())
preds_ty = preds_ty[preds_ty.date_modified >= date_mod].reset_index(drop=True)
display(preds_ty[((preds_ty.pos=='QB'))].iloc[:15])
display(preds_ty[((preds_ty.pos!='QB'))].iloc[:50])

#%%

preds_ny = dm.read(f'''SELECT player, 
                        pos,
                        rush_pass,
                        AVG(pred_fp_per_game) pred_fp_per_game,
                        AVG(pred_fp_per_game_upside) pred_prob_upside,
                        AVG(pred_fp_per_game_top) pred_prob_top,
                        AVG(std_dev) std_dev,
                        AVG(min_score) min_score,   
                        AVG(max_score) max_score, 
                        MAX(date_modified) date_modified
                FROM Model_Predictions
                WHERE rush_pass NOT IN ('rush', 'pass', 'rec')
                       AND version='{vers}'
                       AND year = {set_year}
                       AND dataset NOT LIKE '%Rookie%'
                       AND current_or_next_year = 'next'
                       AND pos != 'QB'
                GROUP BY player, pos, rush_pass
             ''', 'Simulation').sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

preds_ny.date_modified = pd.to_datetime(preds_ny.date_modified).apply(lambda x: x.date())
preds_ny = preds_ny[preds_ny.date_modified >= date_mod].reset_index(drop=True)
display(preds_ny[((preds_ny.pos=='QB'))].iloc[:15])
display(preds_ny[((preds_ny.pos!='QB'))].iloc[:50])

#%%

preds = pd.concat([rp, rookies, preds_ty, preds_ny
                   ], axis=0).reset_index(drop=True)

preds.loc[preds.std_dev < 0, 'std_dev'] = 1

preds.loc[preds.max_score < preds.pred_fp_per_game, 'max_score'] = (
    preds.loc[preds.max_score < preds.pred_fp_per_game, 'pred_fp_per_game'] +
    preds.loc[preds.max_score < preds.pred_fp_per_game, 'std_dev'] * 1.5
)

preds.loc[preds.min_score > preds.pred_fp_per_game, 'min_score'] = (
    preds.loc[preds.min_score > preds.pred_fp_per_game, 'pred_fp_per_game'] -
    preds.loc[preds.min_score > preds.pred_fp_per_game, 'std_dev'] * 1.5
)

preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'mean', 
                                                              'pred_prob_upside': 'mean',
                                                              'pred_prob_top': 'mean',
                                                              'std_dev': 'mean',
                                                              'min_score': 'mean',
                                                              'max_score': 'mean'})

preds = preds[preds.pred_fp_per_game > 0].reset_index(drop=True)

preds_ny_cpy = preds_ny[['player', 'pred_fp_per_game', 'std_dev', 'min_score', 'max_score']].copy()
preds_ny_cpy.columns = ['player', 'pred_fp_per_game_ny', 'std_dev_ny', 'min_score_ny', 'max_score_ny']
preds = pd.merge(preds, preds_ny_cpy, on='player', how='left')

preds.loc[preds.std_dev_ny < 0, 'std_dev_ny'] = 1

preds.loc[preds.max_score_ny < preds.pred_fp_per_game_ny, 'max_score_ny'] = (
    preds.loc[preds.max_score_ny < preds.pred_fp_per_game_ny, 'pred_fp_per_game_ny'] +
    preds.loc[preds.max_score_ny < preds.pred_fp_per_game_ny, 'std_dev_ny'] * 1.5
)

preds.loc[preds.min_score_ny > preds.pred_fp_per_game_ny, 'min_score_ny'] = (
    preds.loc[preds.min_score_ny > preds.pred_fp_per_game_ny, 'pred_fp_per_game_ny'] -
    preds.loc[preds.min_score_ny > preds.pred_fp_per_game_ny, 'std_dev_ny'] * 1.5
)

preds['dataset'] = 'final_ensemble'
preds['version'] = vers
preds['year'] = set_year
preds = preds.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

preds['pos_rank'] = preds.groupby('pos')['pred_fp_per_game'].rank(ascending=False, method='first')

if vers == 'nv': num_qb = 33
elif vers == 'beta': num_qb = 24

preds = preds[~((preds.pos=='QB') & (preds.pos_rank > num_qb))].reset_index(drop=True)
preds = preds[~((preds.pos=='TE') & (preds.pos_rank > 24))].reset_index(drop=True)
preds = preds[~((preds.pos=='RB') & (preds.pos_rank > 60))].reset_index(drop=True)
preds = preds[~((preds.pos=='RB') & (preds.pos_rank > 72))].reset_index(drop=True).drop('pos_rank', axis=1)

preds.loc[preds.pred_fp_per_game_ny.isnull(), ['pred_fp_per_game_ny', 'std_dev_ny', 'min_score_ny', 'max_score_ny']] = \
    preds.loc[preds.pred_fp_per_game_ny.isnull(), ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']].values

display(preds[((preds.pos=='QB'))].iloc[:15])
display(preds[((preds.pos!='QB'))].iloc[:50])

#%%
downgrades = {
    'Anthony Richardson': 0.75,
    'Jalen Milroe': 0.5
}

for p, d in downgrades.items():
    preds.loc[preds.player==p, ['pred_prob_upside', 'pred_prob_top', 'pred_fp_per_game']] = \
        preds.loc[preds.player==p, ['pred_prob_upside', 'pred_prob_top', 'pred_fp_per_game']] * d

# %%
import shutil

dm.delete_from_db('Simulation', 'Final_Predictions', f"version='{vers}' AND year={set_year} AND dataset='final_ensemble'", create_backup=True)
dm.write_to_db(preds, 'Simulation', 'Final_Predictions', 'append')

src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3'
shutil.copyfile(src, dst)


#%%
