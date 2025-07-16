#%%

from s1_Stacking_Model import *

import warnings
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

set_year = 2025
show_plot = True
vers = 'nffc'
predict_only = True

runs = [
        ['WR', 'current', 'greater_equal', 0, '', 'Rookie'],
        ['RB', 'current', 'greater_equal', 0, '', 'Rookie'],

        ['WR', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        ['WR', 'current', 'less_equal', 3, '', 'ProjOnly'],
        ['WR', 'current', 'greater_equal', 4, '', 'ProjOnly'],

        ['WR', 'next', 'greater_equal', 0, '', 'ProjOnly'],
        ['WR', 'next', 'less_equal', 3, '', 'ProjOnly'],
        ['WR', 'next', 'greater_equal', 4, '', 'ProjOnly'],

        ['WR', 'current', 'greater_equal', 0, '', 'Stats'],
        ['WR', 'current', 'less_equal', 3, '', 'Stats'],
        ['WR', 'current', 'greater_equal', 4, '', 'Stats'],

        ['TE', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        ['TE', 'current', 'greater_equal', 0, '', 'Stats'],
        ['TE', 'next', 'greater_equal', 0, '', 'ProjOnly'],

        ['QB', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        ['QB', 'current', 'greater_equal', 0, 'rush', 'ProjOnly'],
        ['QB', 'current', 'greater_equal', 0, 'pass', 'ProjOnly'],
        ['QB', 'next', 'greater_equal', 0, '', 'ProjOnly'],

        ['QB', 'current', 'greater_equal', 0, '', 'Stats'],
        ['QB', 'current', 'greater_equal', 0, 'rush', 'Stats'],
        ['QB', 'current', 'greater_equal', 0, 'pass', 'Stats'],

        ['RB', 'current', 'greater_equal', 0, '', 'ProjOnly'],
        ['RB', 'current', 'less_equal', 3, '', 'ProjOnly'],
        ['RB', 'current', 'greater_equal', 4, '', 'ProjOnly'],
        ['RB', 'current', 'greater_equal', 0, 'rush', 'ProjOnly'],
        ['RB', 'current', 'greater_equal', 0, 'rec', 'ProjOnly'],
        ['RB', 'current', 'less_equal', 3, 'rush', 'ProjOnly'],
        ['RB', 'current', 'less_equal', 3, 'rec', 'ProjOnly'],
        ['RB', 'current', 'greater_equal', 4, 'rush', 'ProjOnly'],
        ['RB', 'current', 'greater_equal', 4, 'rec', 'ProjOnly'],

        ['RB', 'next', 'greater_equal', 0, '', 'ProjOnly'],
        ['RB', 'next', 'less_equal', 3, '', 'ProjOnly'],
        ['RB', 'next', 'greater_equal', 4, '', 'ProjOnly'],

        ['RB', 'current', 'greater_equal', 0, '', 'Stats'],
        ['RB', 'current', 'less_equal', 3, '', 'Stats'],
        ['RB', 'current', 'greater_equal', 4, '', 'Stats'],
        ['RB', 'current', 'greater_equal', 0, 'rush', 'Stats'],
        ['RB', 'current', 'greater_equal', 0, 'rec', 'Stats'],
        ['RB', 'current', 'less_equal', 3, 'rush', 'Stats'],
        ['RB', 'current', 'less_equal', 3, 'rec', 'Stats'],
        ['RB', 'current', 'greater_equal', 4, 'rush', 'Stats'],
        ['RB', 'current', 'greater_equal', 4, 'rec', 'Stats'],

     

]

print(vers)

#%%

for sp, cn, fd, ye, rp, dset in runs:

    set_pos = sp
    current_or_next_year = cn
    pos[set_pos]['filter_data'] = fd
    pos[set_pos]['year_exp'] = ye
    pos[set_pos]['rush_pass'] = rp

    print(f'{set_pos} - {current_or_next_year} - {pos[set_pos]["filter_data"]} - {pos[set_pos]["year_exp"]} - {pos[set_pos]["rush_pass"]} - {dset}')

    if dset=='Rookie': pos[set_pos]['n_splits'] = 4
    else: pos[set_pos]['n_splits'] = 5

    #------------
    # Pull in the data and create train and predict sets
    #------------

    dataset = dset
    hp_algo = 'tpe'
    bayes_rand = 'optuna'
    optuna_timeout = 45

    model_output_path, pkey = create_pkey(pos, dataset, set_pos,current_or_next_year, bayes_rand, hp_algo)
    df = pull_data(set_pos, set_year, dataset, current_or_next_year)

    obj_cols = list(df.dtypes[df.dtypes=='object'].index)
    obj_cols = [c for c in obj_cols if c not in ['player', 'team', 'pos']]
    df= df.drop(obj_cols, axis=1)

    df, output_start = filter_df(df, pos, set_pos, set_year)
    df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top = get_train_predict(df, set_year, pos[set_pos]['rush_pass'])

    #------------
    # Run the Regression, Classification, and Quantiles
    #------------

    if not predict_only:
            
        # set up blank dictionaries for all metrics
        out_dict_reg, out_dict_top, out_dict_upside, out_dict_quant = output_dict(), output_dict(), output_dict(), output_dict()

        model_list = ['adp', 'lasso', 'lgbm', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'knn', 'ridge', 'bridge', 'enet']
        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(get_model_output)
                        (m, df_train, 'reg', out_dict_reg, pos, set_pos, hp_algo, bayes_rand, i, optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(model_list) 
                        )

        out_dict_reg = extract_par_results(results, out_dict_reg)
        save_output_dict(out_dict_reg, model_output_path, 'reg')

        del results, out_dict_reg;gc.collect()

        # run all other models
        model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(get_model_output)
                        (m, df_train_top, 'class', out_dict_top, pos, set_pos, hp_algo, bayes_rand, i, '_top', optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(model_list) 
                        )
        out_dict_top = extract_par_results(results, out_dict_top)
        save_output_dict(out_dict_top, model_output_path, 'class_top')

        del results, out_dict_top; gc.collect()

        # run all other models
        model_list = ['lgbm_c', 'knn_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c']
        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(get_model_output)
                        (m, df_train_upside, 'class', out_dict_upside, pos, set_pos, hp_algo, bayes_rand, i, '_upside', optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(model_list) 
                        )
        out_dict_upside = extract_par_results(results, out_dict_upside)
        save_output_dict(out_dict_upside, model_output_path, 'class_upside')

        del results, out_dict_upside; gc.collect()

        # run all other models
        model_list = ['qr_q','lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']
        models_q = [[alph, m] for alph in [0.65, 0.85] for m in model_list]
        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(get_model_output)
                        (m[1], df_train, 'quantile', out_dict_quant, pos, set_pos, hp_algo, bayes_rand, i, alpha=m[0], optuna_timeout=optuna_timeout) \
                        for i, m in enumerate(models_q) 
                        )

        out_dict_quant = extract_par_results(results, out_dict_quant)
        save_output_dict(out_dict_quant, model_output_path, 'quantile')

        del results, out_dict_quant; gc.collect()

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

    if 'Rookie' in dataset: repeats = 3
    else: repeats = 1

    for i in range(repeats):

        # get the training data for stacking and prediction data after stacking
        X_stack_player, X_stack, y_stack, y_stack_upside, y_stack_top, \
        models_reg, models_upside, models_top, models_quant = load_all_stack_pred(model_output_path)

        _, X_predict = get_stack_predict_data(df_train, df_predict, df_train_upside, df_predict_upside, df_train_top, df_predict_top,
                                            models_reg, models_upside, models_top, models_quant)

        #--------------
        # Regression
        #---------------

        final_models = ['bridge', 'enet', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'lgbm', 'knn', 'ridge', 'lasso', 'xgb', 'et']
        stack_val_pred = pd.DataFrame(); scores = []; best_models = []

        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(run_stack_models)
                        (fm, X_stack, y_stack, i, 'reg', None, run_params) \
                        for i, fm in enumerate(final_models) 
                        )

        best_models, scores, stack_val_pred = unpack_stack_results(results)

        # get the best stack predictions and average
        predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
        best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, 
                                                                          predictions, 'reg', show_plot=True, min_include=3+i)

        #---------------
        # Classification Top
        #---------------
        final_models_top = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', 'et_c']
        stack_val_top = pd.DataFrame(); scores_top = []; best_models_top = []
        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(run_stack_models)
                        (fm, X_stack, y_stack_top, i, 'class', None, run_params) \
                        for i, fm in enumerate(final_models_top) 
                        )

        best_models_top, scores_top, stack_val_top = unpack_stack_results(results)

        # get the best stack predictions and average
        predictions_top = stack_predictions(X_predict, best_models_top, final_models_top, 'class')
        best_val_top, best_predictions_top, _ = average_stack_models(df_train, scores_top, final_models_top, y_stack_top, 
                                                                    stack_val_top, predictions_top, 'class', show_plot=True, min_include=3+i)

        #---------------
        # Classification Upside
        #---------------
        final_models_upside = [ 'lgbm_c', 'lr_c', 'rf_c', 'gbm_c', 'gbmh_c', 'mlp_c', 'cb_c', 'xgb_c', 'et_c']
        stack_val_upside = pd.DataFrame(); scores_upside = []; best_models_upside = []
        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(run_stack_models)
                        (fm, X_stack, y_stack_upside, i, 'class', None, run_params) \
                        for i, fm in enumerate(final_models_upside) 
                        )

        best_models_upside, scores_upside, stack_val_upside = unpack_stack_results(results)

        # get the best stack predictions and average
        predictions_upside = stack_predictions(X_predict, best_models_upside, final_models_upside, 'class')
        best_val_upside, best_predictions_upside, _ = average_stack_models(df_train, scores_upside, final_models_upside, y_stack_upside, 
                                                                           stack_val_upside, predictions_upside, 'class', show_plot=True, min_include=3+i)


        #------------
        # Quantile
        #---------------

        final_models_quant = ['qr_q', 'gbm_q', 'gbmh_q', 'rf_q', 'lgbm_q', 'cb_q']
        stack_val_quant = pd.DataFrame(); scores_quant = []; best_models_quant = []

        results = Parallel(n_jobs=8, verbose=1)(
                        delayed(run_stack_models)
                        (fm, X_stack, y_stack, i, 'quantile', 0.85, run_params) \
                        for i, fm in enumerate(final_models_quant) 
        )

        best_models_quant, scores_quant, stack_val_quant = unpack_stack_results(results)

        # get the best stack predictions and average
        predictions_quant = stack_predictions(X_predict, best_models_quant, final_models_quant, 'quantile')
        best_val_quant, best_predictions_quant, _ = average_stack_models(df_train, scores_quant, final_models_quant, y_stack, stack_val_quant, 
                                                                         predictions_quant, 'quantile', show_plot=True, min_include=3)

        #---------------
        # Create Output
        #---------------

        if X_stack.shape[0] < 150: iso_spline = 'iso'
        else: iso_spline = 'spline'
        output = create_output(output_start, best_predictions, best_predictions_upside, best_predictions_top, best_predictions_quant)
        df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_upside, best_val_top, best_val_quant)
        output = val_std_dev(output, df_val_stack, metrics=run_params['sd_metrics'], iso_spline=iso_spline, show_plot=True, max_grps_den=0.04, min_grps_den=0.08)

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

        if i > 0: dataset_out = f'{dataset}_{i}'
        else: dataset_out = dataset

        save_out_results(val_compare, 'Validations', 'Model_Validations', vers, pos, set_year, set_pos, dataset_out, current_or_next_year)
        save_out_results(output, 'Simulation', 'Model_Predictions', vers, pos, set_year, set_pos, dataset_out, current_or_next_year)

        del X_stack_player, X_stack, y_stack, y_stack_upside, y_stack_top, \
            models_reg, models_upside, models_top, models_quant
        gc.collect()


# %%


# df = dm.read('''SELECT * 
#                 FROM Model_Predictions
#                 WHERE year=2025
#                       AND version='dk' 
#                       AND pos!='QB'
#                       ''', 
#                       'Simulation')
# df.version = 'nffc'
# dm.write_to_db(df, 'Simulation', 'Model_Predictions', if_exist='append')

# df = dm.read('''SELECT * 
#                 FROM Model_Validations 
#                 WHERE year=2025
#                       AND version='dk' 
#                       AND pos!='QB'
#              ''', 
#                       'Validations')
# df.version = 'nffc'
# dm.write_to_db(df, 'Validations', 'Model_Validations', if_exist='append')
