#%%

YEAR = 2023

import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm_daily = DataManage(db_path)


root_path = ffgeneral.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm_ff = DataManage(db_path)

pd.set_option('display.max_columns', 999)


#%%

for pos in ['QB', 'RB', 'WR', 'TE']:
    df = dm_daily.read(f'''SELECT * FROM {pos}_Stats''', 'FastR_Beta')
    stat_cols = [c for c in df.columns if c not in ('player', 'team', 'season', 'week', 'position')]

    def perc_agg80(x):
        return np.percentile(x, 80)

    def perc_agg20(x):
        return np.percentile(x, 20)

    df_all = df.groupby(['player', 'season']).agg({'week': 'count'}).reset_index()
    df_all = df_all.rename(columns={'week': 'games'})

    for agg_type in ['sum', 'mean', 'max']:
        print(agg_type)
        agg_stats = {}
        for c in stat_cols:
            agg_stats[c] = agg_type
        df_agg = df.groupby(['player', 'season']).agg(agg_stats).reset_index()
        if agg_type == perc_agg80: agg_type = 'p80'
        df_agg = df_agg.rename(columns={c: f'{agg_type}_{c}' for c in df_agg.columns if c not in ('player', 'season')})

        df_all = pd.merge(df_all, df_agg, on=['player', 'season'], how='left')

    # df_all = df_all[df_all.games > 8].sort_values(by=['player', 'season'])
    df_all['fantasy_pts_per_game'] = (df_all['sum_fantasy_pts']/df_all['games'])
    df_all = df_all.sort_values(by=['player', 'season']).reset_index(drop=True)

    df_all['y_act'] = df_all.groupby('player')['fantasy_pts_per_game'].shift(-1)
    df_all['games_next'] = df_all.groupby('player')['games'].shift(-1)
    df_all['year'] = df_all.season+1

    df_all = df_all.sort_values(by=['year', 'fantasy_pts_per_game'], ascending=[True, False])
    cols = ['player', 'year', 'season', 'games', 'games_next', 'fantasy_pts_per_game', 'y_act']
    cols.extend([c for c in df_all.columns if c not in cols])
    df_all = df_all[cols]

    df_all = df_all.sort_values(by=['year', 'fantasy_pts_per_game'], ascending=[True, False]).reset_index(drop=True)
    df_all.player = df_all.player.apply(dc.name_clean)
    dm_ff.write_to_db(df_all, 'Season_Stats_New', f'{pos}_Stats', if_exist='replace')


#%%

from skmodel import SciKitModel

Xy = df_all.drop([c for c in df_all if 'fantasy_pts' in c], axis=1)
Xy = Xy.sort_values(by='year').reset_index(drop=True)
Xy['team'] = 'team'
Xy['week'] = 1
Xy['game_date'] = Xy.year
pred = Xy[Xy.year==2024].copy().reset_index(drop=True)
Xy = Xy[(Xy.games >= 4) & (Xy.games_next >= 4)].dropna().reset_index(drop=True)

print(Xy.shape)

preds = []
actuals = []

skm = SciKitModel(Xy)
X, y = skm.Xy_split('y_act', to_drop = ['player', 'team', 'games_next'])

pipe = skm.model_pipe([ skm.piece('random_sample'),
                        skm.piece('std_scale'), 
                        skm.piece('select_perc'),
                        skm.feature_union([
                                        skm.piece('agglomeration'), 
                                        skm.piece('k_best'),
                                        skm.piece('pca')
                                        ]),
                        skm.piece('k_best'),
                        skm.piece('enet')
                    ])

params = skm.default_params(pipe, 'bayes')

best_models, oof_data, param_scores, _ = skm.time_series_cv(pipe, X, y, params, n_iter=20,
                                                                col_split='year',n_splits=5,
                                                                time_split=2014,
                                                                bayes_rand='bayes', proba=False,
                                                                sample_weight=False, trials=Trials(),
                                                                random_seed=12345)

#%%

oof_data['full_hold'].plot.scatter(x='pred', y='y_act')

#%%

oof_data['full_hold'][(oof_data['full_hold'].pred < 7.5) & (oof_data['full_hold'].y_act > 15)]
#%%

preds = best_models[0].fit(X,y).predict(pred[X.columns])
pred['pred'] = preds
pred[['player', 'year', 'pred']].sort_values(by='pred', ascending=False).iloc[:20]