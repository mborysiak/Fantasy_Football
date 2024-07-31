#%%

from Stacking_Model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

set_year = 2024
show_plot = True
vers = 'beta'
predict_only = False

runs = [
        # ['RB', 'current', 'greater_equal', 0, '', 'Rookie'],
        # ['WR', 'current', 'greater_equal', 0, '', 'Rookie'],
    
        # ['WR', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        # ['WR', 'current', 'less_equal', 3, '', 'ProjOnly'],
        # ['WR', 'current', 'greater_equal', 4, '', 'ProjOnly'],

        # ['WR', 'current', 'greater_equal', 0, '', 'Stats'],
        ['WR', 'current', 'less_equal', 3, '', 'Stats'],
        ['WR', 'current', 'greater_equal', 4, '', 'Stats'],

        ['RB', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        ['RB', 'current', 'less_equal', 3, '', 'ProjOnly'],
        ['RB', 'current', 'greater_equal', 4, '', 'ProjOnly'],

        ['RB', 'current', 'greater_equal', 0, '', 'Stats'],
        ['RB', 'current', 'less_equal', 3, '', 'Stats'],
        ['RB', 'current', 'greater_equal', 4, '', 'Stats'],

        ['TE', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        ['TE', 'current', 'greater_equal', 0, '', 'Stats'],

        ['QB', 'current', 'greater_equal', 0, 'both', 'ProjOnly'],
        # ['QB', 'current', 'greater_equal', 0, 'rush', 'ProjOnly'],
        # ['QB', 'current', 'greater_equal', 0, 'pass', 'ProjOnly'],

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
    hp_algo = 'tpe'
    bayes_rand = 'optuna'
    optuna_timeout = 60

    model_output_path, pkey = create_pkey(pos, dataset, set_pos,current_or_next_year, bayes_rand, hp_algo)
    df = pull_data(set_pos, set_year, dataset)

    obj_cols = list(df.dtypes[df.dtypes=='object'].index)
    obj_cols = [c for c in obj_cols if c not in ['player', 'team', 'pos']]
    df= df.drop(obj_cols, axis=1)

    df, output_start = filter_df(df, pos, set_pos, set_year)
    df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top = get_train_predict(df, set_year)

    #------------
    # Run the Regression, Classification, and Quantiles
    #------------

    if not predict_only:
            
        # set up blank dictionaries for all metrics
        out_dict_reg, out_dict_top, out_dict_upside, out_dict_quant = output_dict(), output_dict(), output_dict(), output_dict()

        model_list = ['adp', 'lasso', 'lgbm', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'knn', 'ridge', 'bridge', 'enet']
        results = Parallel(n_jobs=-1, verbose=1)(
                        delayed(get_model_output)
                        (m, df_train, 'reg', out_dict_reg, pos, set_pos, hp_algo, bayes_rand, i, optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(model_list) 
                        )

        out_dict_reg = extract_par_results(results, out_dict_reg)
        save_output_dict(out_dict_reg, model_output_path, 'reg')

        # run all other models
        model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
        results = Parallel(n_jobs=-1, verbose=1)(
                        delayed(get_model_output)
                        (m, df_train_top, 'class', out_dict_top, pos, set_pos, hp_algo, bayes_rand, i, '_top', optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(model_list) 
                        )
        out_dict_top = extract_par_results(results, out_dict_top)
        save_output_dict(out_dict_top, model_output_path, 'class_top')

        # run all other models
        model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
        results = Parallel(n_jobs=-1, verbose=1)(
                        delayed(get_model_output)
                        (m, df_train_upside, 'class', out_dict_upside, pos, set_pos, hp_algo, bayes_rand, i, '_upside', optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(model_list) 
                        )
        out_dict_upside = extract_par_results(results, out_dict_upside)
        save_output_dict(out_dict_upside, model_output_path, 'class_upside')

        # run all other models
        model_list = ['qr_q','lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']
        models_q = [[alph, m] for alph in [0.65, 0.85] for m in model_list]
        results = Parallel(n_jobs=-1, verbose=1)(
                        delayed(get_model_output)
                        (m[1], df_train, 'quantile', out_dict_quant, pos, set_pos, hp_algo, bayes_rand, i, alpha=m[0], optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(models_q) 
                        )

        out_dict_quant = extract_par_results(results, out_dict_quant)
        save_output_dict(out_dict_quant, model_output_path, 'quantile')

    #------------
    # Run the Stacking Models and Generate Output
    #------------
    run_params = {
        'stack_model': 'random_full_stack',
        'print_coef': False,
        'opt_type': 'optuna',
        'hp_algo': 'tpe',
        'optuna_timeout': 60*2,
        'num_k_folds': 3,
        'n_iter': 50,


        'sd_metrics': {'pred_fp_per_game': 1, 'pred_fp_per_game_upside': 1, 'pred_fp_per_game_top': 1, 'pred_fp_per_game_quantile': 1}
    }


    # get the training data for stacking and prediction data after stacking
    X_stack_player, X_stack, y_stack, y_stack_upside, y_stack_top, \
    models_reg, models_upside, models_top, models_quant = load_all_stack_pred(model_output_path)

    _, X_predict = get_stack_predict_data(df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top,
                                        models_reg, models_upside, models_top, models_quant)

    #--------------
    # Regression
    #---------------
    final_models = ['bridge', 'enet', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'lgbm', 'knn', 'ridge', 'lasso', 'xgb']
    stack_val_pred = pd.DataFrame(); scores = []; best_models = []

    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack, i, 'reg', None, run_params) \
                    for i, fm in enumerate(final_models) 
                    )

    best_models, scores, stack_val_pred = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
    best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 'reg', show_plot=True, min_include=3)

    #---------------
    # Classification Top
    #---------------
    final_models_top = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', ]
    stack_val_top = pd.DataFrame(); scores_top = []; best_models_top = []
    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack_top, i, 'class', None, run_params) \
                    for i, fm in enumerate(final_models_top) 
                    )

    best_models_top, scores_top, stack_val_top = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions_top = stack_predictions(X_predict, best_models_top, final_models_top, 'class')
    best_val_top, best_predictions_top, _ = average_stack_models(df_train, scores_top, final_models_top, y_stack_top, 
                                                                stack_val_top, predictions_top, 'class', show_plot=True, min_include=3)

    #---------------
    # Classification Upside
    #---------------
    final_models_upside = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', ]
    stack_val_upside = pd.DataFrame(); scores_upside = []; best_models_upside = []
    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack_upside, i, 'class', None, run_params) \
                    for i, fm in enumerate(final_models_upside) 
                    )

    best_models_upside, scores_upside, stack_val_upside = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions_upside = stack_predictions(X_predict, best_models_upside, final_models_upside, 'class')
    best_val_upside, best_predictions_upside, _ = average_stack_models(df_train, scores_upside, final_models_upside, y_stack_upside, 
                                                                    stack_val_upside, predictions_upside, 'class', show_plot=True, min_include=3)


    #------------
    # Quantile
    #---------------

    final_models_quant = ['qr_q', 'gbm_q', 'gbmh_q', 'rf_q', 'lgbm_q', 'cb_q']
    stack_val_quant = pd.DataFrame(); scores_quant = []; best_models_quant = []

    results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(run_stack_models)
                    (fm, X_stack, y_stack, i, 'quantile', 0.85, run_params) \
                    for i, fm in enumerate(final_models_quant) 
    )

    best_models_quant, scores_quant, stack_val_quant = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions_quant = stack_predictions(X_predict, best_models_quant, final_models_quant, 'quantile')
    best_val_quant, best_predictions_quant, _ = average_stack_models(df_train, scores_quant, final_models_quant, y_stack, stack_val_quant, predictions_quant, 'quantile', show_plot=True, min_include=3)

    #---------------
    # Create Output
    #---------------

    if X_stack.shape[0] < 150: iso_spline = 'iso'
    else: iso_spline = 'spline'
    output = create_output(output_start, best_predictions, best_predictions_upside, best_predictions_top, best_predictions_quant)
    df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_upside, best_val_top, best_val_quant)
    output = val_std_dev(output, df_val_stack, metrics=run_params['sd_metrics'], iso_spline=iso_spline, show_plot=True, max_grps_den=0.04, min_grps_den=0.08)
    output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]

    output.loc[output.std_dev < 1, 'std_dev'] = output.loc[output.std_dev < 1, 'pred_fp_per_game'] * 0.15
    output.loc[output.max_score < (output.pred_fp_per_game + output.std_dev), 'max_score'] = \
        output.loc[output.max_score < (output.pred_fp_per_game + output.std_dev), 'pred_fp_per_game'] + \
        output.loc[output.max_score < (output.pred_fp_per_game + output.std_dev), 'std_dev'] * 1.5
    
    output.loc[output.min_score > (output.pred_fp_per_game - output.std_dev), 'min_score'] = \
        output.loc[output.min_score > (output.pred_fp_per_game - output.std_dev), 'pred_fp_per_game'] - \
        output.loc[output.min_score > (output.pred_fp_per_game - output.std_dev), 'std_dev'] * 2

    y_max = df_train.y_act.max()
    output.loc[output.max_score > y_max, 'max_score'] = y_max + (output.loc[output.max_score > y_max, 'std_dev'] / 5)
    output = output.round(3)
    display(output.iloc[:50])
    # save out final results
    val_compare = validation_compare_df(model_output_path, best_val_reg)
    save_out_results(val_compare, 'Validations', 'Model_Validations', pos, set_year, set_pos, dataset, current_or_next_year)
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
import shutil

dm.delete_from_db('Simulation', 'Final_Predictions', f"version='{vers}' AND year={set_year} AND dataset='final_ensemble'")
dm.write_to_db(preds, 'Simulation', 'Final_Predictions', 'append')

src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3'
shutil.copyfile(src, dst)
# %%
