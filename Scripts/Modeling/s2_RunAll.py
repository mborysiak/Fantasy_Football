#%%

import pandas as pd
import os
import sys
from joblib import delayed
import warnings
from IPython.display import display
from s1_Stacking_Model import (
    install_model_print_filter, 
    install_warning_filters,
    set_optuna_logging,
    create_pkey,
    pull_data,
    filter_df,
    get_train_predict,
    output_dict,
    extract_par_results,
    save_output_dict,
    load_all_stack_pred,
    unpack_stack_results,
    average_stack_models,
    stack_predictions,
    get_stack_predict_data,
    create_output,
    create_final_val_df,
    apply_empirical_resid_quantiles,
    cross_fit_empirical_resid_quantiles,
    eligible_empirical_resid_donors,
    validation_compare_df,
    save_out_results,
    get_model_output,
    run_parallel, 
    pos,
    run_stack_models,
    RESID_ALPHAS
)

import gc


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=PendingDeprecationWarning)
warnings.filterwarnings(
    'ignore',
    message=r'.*CatBoostRegressor.*__sklearn_tags__.*',
    category=DeprecationWarning,
)
warnings.filterwarnings('ignore', message='.*deprecated.*', category=UserWarning)

# Add Scripts directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YEAR, LEAGUE


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
install_warning_filters(include_pandas=True)
set_optuna_logging()
install_model_print_filter()

VERBOSE = 1
NUM_CORES = 4

# Use config settings
set_year = YEAR
show_plot = True
vers = LEAGUE
predict_only = True

runs = [
        ['RB', 'current', 'less_equal', 0, '', 'ProjOnly'],
        ['WR', 'current', 'less_equal', 0, '', 'ProjOnly'],

        ['RB', 'current', 'less_equal', 1, '', 'ProjOnly'],
        ['WR', 'current', 'less_equal', 1, '', 'ProjOnly'],

        ['RB', 'next', 'less_equal', 1, '', 'ProjOnly'],
        ['WR', 'next', 'less_equal', 1, '', 'ProjOnly'],

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

print(f'Running league {LEAGUE} for year {YEAR}')

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
    optuna_timeout = 30

    model_output_path, pkey = create_pkey(pos, dataset, set_pos, set_year,current_or_next_year, bayes_rand, hp_algo)
    df = pull_data(set_pos, set_year, dataset, current_or_next_year)

    obj_cols = list(df.dtypes[df.dtypes=='object'].index)
    obj_cols = [c for c in obj_cols if c not in ['player', 'team', 'pos']]
    df= df.drop(obj_cols, axis=1)

    df, output_start = filter_df(df, pos, set_pos, set_year)
    df_train, df_predict = get_train_predict(df, set_year, pos[set_pos]['rush_pass'])
    df_train_residual = df_train.copy()
    df_train_residual['y_act'] = df_train_residual.y_act - df_train_residual.avg_proj_points_per_game

    #------------
    # Run the Regression, Classification, and Quantiles
    #------------

    if not predict_only:

        # set up blank dictionaries for all metrics
        out_dict_reg, out_dict_quant = output_dict(), output_dict()

        model_list = ['adp', 'lasso', 'lgbm', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'ridge', 'bridge', 'enet']
        results = run_parallel(
            (delayed(get_model_output)
                (m, df_train, 'reg', out_dict_reg, pos, set_pos, hp_algo, bayes_rand, i, optuna_timeout=optuna_timeout)
                for i, m in enumerate(model_list)),
            n_jobs=NUM_CORES,
            verbose=1,
        )

        out_dict_reg = extract_par_results(results, out_dict_reg)
        save_output_dict(out_dict_reg, model_output_path, 'reg')
        del results, out_dict_reg

        model_list = ['qr_q','lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']
        models_q = [[alph, m] for alph in [0.75, 0.9] for m in model_list]
        results = run_parallel(
            (delayed(get_model_output)
                (m[1], df_train_residual, 'quantile', out_dict_quant, pos, set_pos, hp_algo, bayes_rand, i, alpha=m[0], optuna_timeout=optuna_timeout)
                for i, m in enumerate(models_q)),
            n_jobs=NUM_CORES,
            verbose=1,
        )

        out_dict_quant = extract_par_results(results, out_dict_quant)
        save_output_dict(out_dict_quant, model_output_path, 'quantile')
        del results, out_dict_quant
    
    #------------
    # Run the Stacking Models and Generate Output
    #------------

    run_params = {
        'stack_model': 'random_full_stack',
        'print_coef': False,
        'opt_type': 'optuna',
        'hp_algo': 'tpe',
        'num_k_folds': 3,
        'n_iter': 50,
        'optuna_timeout': 30,
    }

    if fd == 'less_equal' and ye==0:
        run_params['num_k_folds'] = 6

    # get the training data for stacking and prediction data after stacking
    X_stack_player, X_stack, y_stack, models_reg, models_quant = load_all_stack_pred(model_output_path)
    _, X_predict = get_stack_predict_data(df_train, df_train_residual, df_predict, models_reg, models_quant)

    #---------------
    # Regression
    #---------------
    final_models = ['bridge', 'enet', 'rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'lgbm', 'knn', 'ridge', 'lasso', 'xgb']
    stack_val_pred = pd.DataFrame(); scores = []; best_models = []

    results = run_parallel(
                    (delayed(run_stack_models)
                    (fm, X_stack, y_stack, i, 'reg', None, run_params) \
                    for i, fm in enumerate(final_models)),
                    n_jobs=NUM_CORES,
                    verbose=1,
                    )

    best_models, scores, stack_val_pred = unpack_stack_results(results)

    # get the best stack predictions and average
    predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
    best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 'reg', show_plot=True, min_include=3)

    #---------------
    # Create Output
    #---------------

    output = create_output(output_start, best_predictions)
    df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg)
    resid_outcome_horizon = int(current_or_next_year == 'next')
    resid_donors = eligible_empirical_resid_donors(
        df_val_stack,
        forecast_origin=set_year,
        origin_col='year',
        outcome_horizon=resid_outcome_horizon,
    )
    output, __builtins__ = apply_empirical_resid_quantiles(
        resid_donors,
        output,
        alphas=RESID_ALPHAS,
        n_bins=None,
        min_n=50,
        smooth=True,
        bootstrap_iters=50,
        bootstrap_replace=True,
    )
    display(output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50])
    # save out final results
    val_compare = validation_compare_df(model_output_path, best_val_reg)
    val_compare, val_resid_calibration_audit = cross_fit_empirical_resid_quantiles(
        val_compare,
        origin_col='season',
        outcome_horizon=resid_outcome_horizon,
        as_of_year=set_year,
        alphas=RESID_ALPHAS,
        min_training_rows=30,
        n_bins=None,
        min_n=50,
        smooth=True,
        bootstrap_iters=50,
        bootstrap_replace=True,
    )
    display(val_resid_calibration_audit)

    # # save out final results
    save_out_results(val_compare, 'Validations', 'Model_Validations_Resid', vers, pos, set_year, set_pos, dataset, current_or_next_year)
    save_out_results(output, 'Simulation', 'Model_Predictions_Resid', vers, pos, set_year, set_pos, dataset, current_or_next_year)

    del X_stack_player, X_stack, y_stack, models_reg, models_quant
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
