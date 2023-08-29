#%%
from Stacking_Model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


set_year = 2023
show_plot = True

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
        ['QB', 'current', 'greater_equal', 0, 'both'],
        ['QB', 'next', 'greater_equal', 0, 'both'],
        ['QB', 'current', 'greater_equal', 0, 'rush'],
        ['QB', 'current', 'greater_equal', 0, 'pass'],
        ['QB', 'next', 'greater_equal', 0, 'rush'],
        ['QB', 'next', 'greater_equal', 0, 'pass'],
]

print(vers)

for sp, cn, fd, ye, rp in runs:

    set_pos = sp
    current_or_next_year = cn
    pos[set_pos]['filter_data'] = fd
    pos[set_pos]['year_exp'] = ye
    pos[set_pos]['rush_pass'] = rp

    #------------
    # Pull in the data and create train and predict sets
    #------------

    pts_dict = create_pts_dict(pos, set_pos, vers)
    pos = class_cutoff(pos)
    model_output_path = create_pkey(pos, set_pos, cn)
    df = pull_data(pts_dict, set_pos, set_year)

    if 'Rookie' not in set_pos:
        df, output_start = filter_df(df, pos, set_pos, set_year)
        df_train, df_predict, min_samples = get_reg_data(df, pos, set_pos)
        df_train_class, df_predict_class = get_class_data(df, pos, set_pos)
    else:
        df_train, df_predict, df_train_class, df_predict_class, output_start, min_samples = prepare_rookie_data(df, set_pos, cn)

    if current_or_next_year == 'next' and 'Rookie' not in set_pos: 
        df_train, df_train_class = adjust_current_or_next(df_train, df_train_class)

    # #------------
    # # Run the Regression, Classification, and Quantiles
    # #------------
    # # set up blank dictionaries for all metrics
    # out_dict_reg, out_dict_class, out_dict_quant = output_dict(), output_dict(), output_dict()

    # # run all other models
    # model_list = ['adp', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'huber', 'bridge', 'gbmh', 'xgb', 'knn', 'gbm', 'rf']
    # for i, m in enumerate(model_list):
    #     out_dict_reg, _, _ = get_model_output(m, df_train, 'reg', out_dict_reg, pos, set_pos, i, min_samples)
    # save_output_dict(out_dict_reg, model_output_path, 'reg')

    # # run all other models
    # model_list = ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c','lgbm_c', 'knn_c', 'lr_c']
    # for i, m in enumerate(model_list):
    #     out_dict_class, _, _= get_model_output(m, df_train_class, 'class', out_dict_class, pos, set_pos, i, min_samples)
    # save_output_dict(out_dict_class, model_output_path, 'class')

    # # run all other models
    # for m in ['qr_q', 'gbm_q', 'gbmh_q', 'lgbm_q']:
    #     for alph in [0.65, 0.8]:
    #         out_dict_quant, _, _ = get_model_output(m, df_train, 'quantile', out_dict_quant, pos, set_pos, i, alpha=alph)
    # save_output_dict(out_dict_quant, model_output_path, 'quant')

    #------------
    # Run the Stacking Models and Generate Output
    #------------
    run_params = {
        'stack_model': 'random_kbest',
        'print_coef': True,
        'opt_type': 'rand',
        'num_k_folds': 3,
        'n_iter': 100,

        'sd_metrics': {'pred_fp_per_game': 1, 'pred_fp_per_game_class': 1, 'pred_fp_per_game_quantile': 0}
    }


    # get the training data for stacking and prediction data after stacking
    X_stack_player, X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)
    _, X_predict = get_stack_predict_data(df_train, df_predict, df_train_class, df_predict_class, 
                                        models_reg, models_class, models_quant)

    #---------------
    # Regression
    #---------------
    final_models = ['rf', 'gbm', 'gbmh', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 'bridge']
    stack_val_pred = pd.DataFrame(); scores = []; best_models = []
    for i, fm in enumerate(final_models):
        best_models, scores, stack_val_pred = run_stack_models(fm, X_stack, y_stack, best_models, scores, stack_val_pred, i, 'reg', None, run_params)

    # get the best stack predictions and average
    predictions = stack_predictions(X_predict, best_models, final_models, 'reg')
    best_val_reg, best_predictions, best_score = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, 'reg', show_plot=True, min_include=2)

    #---------------
    # Classification
    #---------------
    final_models_class = ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'lgbm_c', 'knn_c', 'lr_c']
    stack_val_class = pd.DataFrame(); scores_class = []; best_models_class = []
    for i, fm in enumerate(final_models_class):
        best_models_class, scores_class, stack_val_class = run_stack_models(fm, X_stack, y_stack_class, best_models_class, scores_class, stack_val_class, i, 'class', None, run_params)

    # get the best stack predictions and average
    predictions_class = stack_predictions(X_predict, best_models_class, final_models_class, 'class')
    best_val_class, best_predictions_class, _ = average_stack_models(df_train, scores_class, final_models_class, y_stack_class, stack_val_class, predictions_class, 'class', show_plot=True, min_include=2)

    #---------------
    # Quantile
    #---------------
    final_models_quant = ['qr_q', 'gbm_q', 'gbmh_q', 'lgbm_q']
    stack_val_quant = pd.DataFrame(); scores_quant = []; best_models_quant = []
    for i, fm in enumerate(final_models_quant):
        best_models_quant, scores_quant, stack_val_quant = run_stack_models(fm, X_stack, y_stack, best_models, scores_quant, stack_val_quant, i, 'quantile', 0.8, run_params)

    # get the best stack predictions and average
    predictions_quant = stack_predictions(X_predict, best_models_quant, final_models_quant, 'quantile')
    best_val_quant, best_predictions_quant, _ = average_stack_models(df_train, scores_quant, final_models_quant, y_stack, stack_val_quant, predictions_quant, 'quantile', show_plot=True, min_include=2)

    #---------------
    # Create Output
    #---------------
    output = create_output(output_start, best_predictions, best_predictions_class, best_predictions_quant)
    df_val_stack = create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_class, best_val_quant)
    output = val_std_dev(output, df_val_stack, metrics=run_params['sd_metrics'], iso_spline='iso', show_plot=True)
    display(output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50])

    # save out final results
    val_compare = validation_compare_df(model_output_path, best_val_reg)
    output = output.drop(['pred_fp_per_game_class', 'pred_fp_per_game_quantile'], axis=1)
    save_out_results(val_compare, 'Simulation', 'Model_Validations', pos, set_year, set_pos, current_or_next_year)
    save_out_results(output, 'Simulation', 'Model_Predictions', pos, set_year, set_pos, current_or_next_year)


# %%
 
# %%
from Stacking_Model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


set_year = 2023
show_plot = True

df = dm.read(f'''SELECT * 
                 FROM Model_Predictions
                 WHERE year={set_year}
                       AND version='beta' 
             ''', "Simulation")
df = df[df.pos!='QB'].reset_index(drop=True)
df['version'] = 'nv'

dm.delete_from_db('Simulation', 'Model_Predictions', f'''year={set_year} AND version='nv' ''')
dm.write_to_db(df, db_name='Simulation', table_name='Model_Predictions', if_exist='append')
# %%
