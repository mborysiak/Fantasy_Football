#%%

from ff import db_operations, data_clean, ff_general
from ff.modeling import prepare

root_path = ff_general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'

dm = db_operations.DataManage(db_path)
rb = dm.read('''SELECT * FROM RB_Stats''', 'Season_Stats')

X_tr, _ = prepare.std_scale(rb[['rush_yds', 'rec_yds']])

#%%

##################################



# +
comb_df = pd.read_sql_query('''SELECT * FROM combine_data_raw''', conn).drop('pp_age', axis=1)
draft_info = pd.read_sql_query('''SELECT * FROM draft_positions''', conn)
draft_values = pd.read_sql_query('''SELECT * FROM draft_values''', conn)
draft_info = pd.merge(draft_info, draft_values, on=['Round', 'Pick'])

comb_df = pd.merge(comb_df, draft_info[['year', 'player', 'Value']], on=['year', 'player'], how='left')
comb_df.Value = comb_df.Value.fillna(1)
comb_df = pd.concat([comb_df, pd.get_dummies(comb_df.pos)], axis=1).drop('pos', axis=1)

pred_cols = ['height', 'weight', 'Value', 'QB', 'RB', 'TE', 'WR']
comb_df = fill_metrics('forty', comb_df, pred_cols)

pred_cols.append('forty')
comb_df = fill_metrics('vertical', comb_df, pred_cols)

pred_cols.append('vertical')
comb_df = fill_metrics('broad_jump', comb_df, pred_cols)

pred_cols.append('broad_jump')
comb_df = fill_metrics('three_cone', comb_df, pred_cols)

pred_cols.append('three_cone')
comb_df = fill_metrics('shuffle_20_yd', comb_df, pred_cols)

pred_cols.append('shuffle_20_yd')
comb_df = fill_metrics('bench_press', comb_df, pred_cols)
# -

# ## Predict Player Profiler Data

# +
rb = pd.read_sql_query('''SELECT * FROM Rookie_RB_Stats_Old''', conn)
wr = pd.read_sql_query('''SELECT * FROM Rookie_WR_Stats_Old''', conn)

pp = pd.concat([rb, wr], axis=0).reset_index(drop=True)
X_cols = [c for c in comb_df if c not in ('year', 'player', 'Value', 'pp_age', 'QB', 'RB', 'TE', 'WR')]
# -

comb_df = reverse_metrics('hand_size', pp, comb_df, X_cols)
comb_df = reverse_metrics('arm_length', pp, comb_df, X_cols)
comb_df = reverse_metrics('speed_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('athlete_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('sparq', pp, comb_df, X_cols)
comb_df = reverse_metrics('bmi', pp, comb_df, X_cols)
comb_df = reverse_metrics('burst_score', pp, comb_df, X_cols)
comb_df = reverse_metrics('agility_score', pp, comb_df, X_cols)

# %%
