#%%

from Stacking_Model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


set_year = 2024
show_plot = True
vers = 'beta'

runs = [
        # ['Rookie_RB', 'current', 'greater_equal', 0, ''],
        # ['Rookie_RB', 'next', 'greater_equal', 0, ''],
        # ['Rookie_WR', 'current', 'greater_equal', 0, ''],
        # ['Rookie_WR', 'next', 'greater_equal', 0, ''],
        # ['WR', 'current', 'greater_equal', 0, ''],
        # ['WR', 'current', 'less_equal', 4, ''],
        # ['WR', 'current', 'greater_equal', 5, ''],
        # ['WR', 'next', 'greater_equal', 0, ''],
        # ['WR', 'next', 'less_equal', 4, ''],
        # ['WR', 'next', 'greater_equal', 5, ''],
        # ['RB', 'current', 'greater_equal', 0, 'both'],
        # ['RB', 'current', 'less_equal', 3, ''],
        # ['RB', 'current', 'greater_equal', 4, ''],
        # ['RB', 'next', 'greater_equal', 0, ''],
        # ['RB', 'next', 'less_equal', 3, ''],
        # ['RB', 'next', 'greater_equal', 4, ''],
        # ['RB', 'current', 'greater_equal', 0, 'rush'],
        # ['RB', 'current', 'greater_equal', 0, 'rec'],
        # ['TE', 'current', 'greater_equal', 0, ''],
        # ['TE', 'next', 'greater_equal', 0, ''],
#         ['QB', 'current', 'greater_equal', 0, 'both'],
#         ['QB', 'next', 'greater_equal', 0, 'both'],
#         ['QB', 'current', 'greater_equal', 0, 'rush'],
#         ['QB', 'current', 'greater_equal', 0, 'pass'],
#         ['QB', 'next', 'greater_equal', 0, 'rush'],
#         ['QB', 'next', 'greater_equal', 0, 'pass'],
]

runs = [
        # ['RB', 'current', 'greater_equal', 0, '', 'Rookie'],
        # ['WR', 'current', 'greater_equal', 0, '', 'Rookie'],
    
        # ['WR', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        # ['WR', 'current', 'less_equal', 3, '', 'ProjOnly'],
        # ['WR', 'current', 'greater_equal', 4, '', 'ProjOnly'],

        # ['WR', 'current', 'greater_equal', 0, '', 'Stats'],
        # ['WR', 'current', 'less_equal', 3, '', 'Stats'],
        # ['WR', 'current', 'greater_equal', 4, '', 'Stats'],

        # ['RB', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        # ['RB', 'current', 'less_equal', 3, '', 'ProjOnly'],
        # ['RB', 'current', 'greater_equal', 4, '', 'ProjOnly'],

        # ['RB', 'current', 'greater_equal', 0, '', 'Stats'],
        # ['RB', 'current', 'less_equal', 3, '', 'Stats'],
        # ['RB', 'current', 'greater_equal', 4, '', 'Stats'],

        # ['TE', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        # ['TE', 'current', 'greater_equal', 0, '', 'Stats'],

        # ['QB', 'current', 'greater_equal', 0, 'both', 'ProjOnly'],
        # # ['QB', 'current', 'greater_equal', 0, 'rush', 'ProjOnly'],
        # # ['QB', 'current', 'greater_equal', 0, 'pass', 'ProjOnly'],

        ['QB', 'current', 'greater_equal', 0, 'both', 'Stats'],
        # ['QB', 'current', 'greater_equal', 0, 'rush', 'Stats'],
        # ['QB', 'current', 'greater_equal', 0, 'pass', 'Stats'],

]

print(vers)

#%%

for sp, cn, fd, ye, rp, dset in runs:

    set_pos = sp
    current_or_next_year = cn
    pos[set_pos]['filter_data'] = fd
    pos[set_pos]['year_exp'] = ye
    pos[set_pos]['rush_pass'] = rp

    #------------
    # Pull in the data and create train and predict sets
    #------------
    dataset = dset
    hp_algo = 'atpe'
    bayes_rand = 'bayes'

    model_output_path = create_pkey(pos, dataset, set_pos, current_or_next_year, bayes_rand, hp_algo)
    df = pull_data(set_pos, set_year, dataset)

    obj_cols = list(df.dtypes[df.dtypes=='object'].index)
    obj_cols = [c for c in obj_cols if c not in ['player', 'team', 'pos', 'games_next']]
    df= df.drop(obj_cols, axis=1)

    df, output_start = filter_df(df, pos, set_pos, set_year)
    df_train, df_predict, df_train_class, df_predict_class = get_train_predict(df, set_year)

    #------------
    # Run the Regression, Classification, and Quantiles
    #------------
    
    # # set up blank dictionaries for all metrics
    # out_dict_reg, out_dict_class, out_dict_quant = output_dict(), output_dict(), output_dict()

    # model_list = ['adp', 'lasso', 'lgbm', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'knn', 'ridge', 'bridge', 'enet']
    # results = Parallel(n_jobs=-1, verbose=50)(
    #                 delayed(get_model_output)
    #                 (m, df_train, 'reg', out_dict_reg, pos, set_pos, hp_algo, bayes_rand, i) \
    #                 for i, m in enumerate(model_list) 
    #                 )

    # out_dict_reg = extract_par_results(results, out_dict_reg)
    # save_output_dict(out_dict_reg, model_output_path, 'reg')

    # # run all other models
    # model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
    # results = Parallel(n_jobs=-1, verbose=50)(
    #                 delayed(get_model_output)
    #                 (m, df_train_class, 'class', out_dict_class, pos, set_pos, hp_algo, bayes_rand, i) \
    #                 for i, m in enumerate(model_list) 
    #                 )
    # out_dict_class = extract_par_results(results, out_dict_class)
    # save_output_dict(out_dict_class, model_output_path, 'class')

    # # run all other models
    # model_list = ['qr_q','lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']
    # models_q = [[alph, m] for alph in [0.6, 0.8] for m in model_list]
    # results = Parallel(n_jobs=-1, verbose=50)(
    #                 delayed(get_model_output)
    #                 (m[1], df_train, 'quantile', out_dict_quant, pos, set_pos, hp_algo, bayes_rand, i, alpha=m[0]) \
    #                 for i, m in enumerate(models_q) 
    #                 )

    # out_dict_quant = extract_par_results(results, out_dict_quant)
    # save_output_dict(out_dict_quant, model_output_path, 'quantile')

    #------------
    # Run the Stacking Models and Generate Output
    #------------
    run_params = {
        'stack_model': 'random_full_stack',
        'print_coef': False,
        'opt_type': 'optuna',
        'num_k_folds': 3,
        'n_iter': 50,

        'sd_metrics': {'pred_fp_per_game': 1, 'pred_fp_per_game_class': 1, 'pred_fp_per_game_quantile': 0.5}
    }


    # get the training data for stacking and prediction data after stacking
    X_stack_player, X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)
    _, X_predict = get_stack_predict_data(df_train, df_predict, df_train_class, df_predict_class, 
                                        models_reg, models_class, models_quant)

    #---------------
    # Regression
    #---------------
    final_models = ['bridge', 'enet', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', ]
    stack_val_pred = pd.DataFrame(); scores = []; best_models = []

    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack, i, 'reg', None, run_params, hp_algo) \
                    for i, fm in enumerate(final_models) 
                    )

    best_models, scores, stack_val_pred = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
    best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 'reg', show_plot=True, min_include=3)

    #---------------
    # Classification
    #---------------
    final_models_class = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', ]
    stack_val_class = pd.DataFrame(); scores_class = []; best_models_class = []
    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack_class, i, 'class', None, run_params, hp_algo) \
                    for i, fm in enumerate(final_models_class) 
                    )

    best_models_class, scores_class, stack_val_class = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions_class = stack_predictions(X_predict, best_models_class, final_models_class, 'class')
    best_val_class, best_predictions_class, _ = average_stack_models(df_train, scores_class, final_models_class, y_stack_class, stack_val_class, predictions_class, 'class', show_plot=True, min_include=2)

    #------------
    # Quantile
    #---------------

    final_models_quant = ['qr_q', 'gbm_q', 'gbmh_q', 'lgbm_q', 'rf_q', 'cb_q']
    stack_val_quant = pd.DataFrame(); scores_quant = []; best_models_quant = []

    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack, i, 'quantile', 0.8, run_params, hp_algo) \
                    for i, fm in enumerate(final_models_quant) 
    )

    best_models_quant, scores_quant, stack_val_quant = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions_quant = stack_predictions(X_predict, best_models_quant, final_models_quant, 'quantile')
    best_val_quant, best_predictions_quant, _ = average_stack_models(df_train, scores_quant, final_models_quant, y_stack, stack_val_quant, predictions_quant, 'quantile', show_plot=True, min_include=2)

    #---------------
    # Create Output
    #---------------

    if X_stack.shape[0] < 200: iso_spline = 'iso'
    else: iso_spline = 'spline'

    output = create_output(output_start, best_predictions, best_predictions_class, best_predictions_quant)
    df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_class, best_val_quant)
    output = val_std_dev(output, df_val_stack, metrics=run_params['sd_metrics'], iso_spline=iso_spline, show_plot=True)
    output.loc[output.max_score < (output.pred_fp_per_game + output.std_dev), 'max_score'] = output.pred_fp_per_game + output.std_dev*1.5
    y_max = df_train.y_act.max()
    output.loc[output.max_score > y_max, 'max_score'] = y_max + output.std_dev / 3

    display(output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50])

    # save out final results
    val_compare = validation_compare_df(model_output_path, best_val_reg)
    output = output.drop(['pred_fp_per_game_class', 'pred_fp_per_game_quantile'], axis=1)
    save_out_results(val_compare, 'Simulation', 'Model_Validations', pos, set_year, set_pos, dataset, current_or_next_year)
    save_out_results(output, 'Simulation', 'Model_Predictions', pos, set_year, set_pos, dataset, current_or_next_year)


# %%

# date_mod =  dt.date(2022,8,26)
# rp = dm.read(f'''SELECT player,
#                         current_or_next_year,
#                         pos,
#                         avg_pick,
#                         pred_fp_per_game,
#                         std_dev, 
#                         max_score,
#                         date_modified
#                 FROM Model_Predictions
#                 WHERE rush_pass IN ('rush', 'pass', 'rec')
#                       AND version='{pred_version}'
#                       AND year = {set_year}
#              ''', 'Simulation').sort_values(by='avg_pick')

# rp.date_modified = pd.to_datetime(rp.date_modified).apply(lambda x: x.date())
# rp = rp[rp.date_modified >= date_mod].reset_index(drop=True)

# # wm = lambda x: np.average(x, weights=rp.loc[x.index, "pred_fp_per_game"])
# rp = rp.assign(std_dev=rp.std_dev**2, max_score=rp.max_score**2)
# rp = rp.groupby(['player', 'current_or_next_year', 'pos','avg_pick']).agg({'pred_fp_per_game': 'sum', 
#                                                                            'std_dev': 'sum', 
#                                                                            'max_score': 'sum'}).reset_index()
# rp = rp.assign(std_dev=np.sqrt(rp.std_dev), max_score=np.sqrt(rp.max_score)).drop('current_or_next_year', axis=1)

# rp.std_dev = rp.std_dev / 1.4
# rp.max_score = rp.max_score / 1.3

import datetime as dt

date_mod =  dt.date(2024,7,20)

both = dm.read(f'''SELECT player, 
                          pos,
                          rush_pass,
                          pred_fp_per_game pred_fp_per_game,
                          std_dev,
                          min_score,   
                          max_score, date_modified
                FROM Model_Predictions
                WHERE (rush_pass NOT IN ('rush', 'pass', 'rec') OR rush_pass IS NULL)
                      AND version='{vers}'
                      AND year = {set_year}
             ''', 'Simulation')

both.date_modified = pd.to_datetime(both.date_modified).apply(lambda x: x.date())
both = both[both.date_modified >= date_mod].reset_index(drop=True)

# preds = pd.concat([rp, both], axis=0)
preds = both.copy()

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
                                                              'std_dev': 'mean',
                                                              'min_score': 'mean',
                                                              'max_score': 'mean'})
preds = preds[preds.pred_fp_per_game > 0].reset_index(drop=True)
preds['dataset'] = 'final_ensemble'
preds['version'] = vers
preds['year'] = set_year

display(preds[((preds.pos=='QB'))].sort_values(by='pred_fp_per_game', ascending=False).iloc[:15])
display(preds[((preds.pos!='QB'))].sort_values(by='pred_fp_per_game', ascending=False).iloc[:50])

# %%

dm.delete_from_db('Simulation', 'Final_Predictions', f"version='{vers}' AND year={set_year} AND dataset='final_ensemble'")
dm.write_to_db(preds, 'Simulation', 'Final_Predictions', 'append')
# %%
