#%%
from Stacking_Model import *

runs = [
        # ['Rookie_RB', 'current', 'greater_equal', 0, ''],
        # ['Rookie_RB', 'next', 'greater_equal', 0, ''],
        # ['Rookie_WR', 'current', 'greater_equal', 0, ''],
     #   ['Rookie_WR', 'next', 'greater_equal', 0],
        # ['WR', 'current', 'greater_equal', 0, ''],
        # ['WR', 'current', 'less_equal', 3, ''],
        # ['WR', 'current', 'greater_equal', 4, ''],
        # ['WR', 'next', 'greater_equal', 0, ''],
        # ['WR', 'next', 'less_equal', 3, ''],
        # ['WR', 'next', 'greater_equal', 4, ''],
        # ['RB', 'current', 'greater_equal', 0, 'both'],
        # ['RB', 'current', 'less_equal', 3, ''],
        # ['RB', 'current', 'greater_equal', 4, ''],
        ['RB', 'next', 'greater_equal', 0, ''],
        ['RB', 'next', 'less_equal', 3, ''],
        ['RB', 'next', 'greater_equal', 4, ''],
        # ['RB', 'current', 'greater_equal', 0, 'rush'],
        # ['RB', 'current', 'greater_equal', 0, 'pass'],
        ['TE', 'current', 'greater_equal', 0, ''],
        ['TE', 'next', 'greater_equal', 0, ''],
        # ['QB', 'current', 'greater_equal', 0, 'both'],
        ['QB', 'next', 'greater_equal', 0, 'both'],
        # ['QB', 'current', 'greater_equal', 0, 'rush'],
        # ['QB', 'current', 'greater_equal', 0, 'pass'],
        
        
]

for sp, cn, fd, ye, rp in runs:

    set_pos = sp
    current_or_next_year = cn
    pos[set_pos]['filter_data'] = fd
    pos[set_pos]['year_exp'] = ye
    pos[set_pos]['rush_pass'] = rp

    #------------
    # Pull in the data and create train and predict sets
    #------------

    pts_dict = create_pts_dict(pos, set_pos)
    pos = class_cutoff(pos)
    model_output_path = create_pkey(pos, set_pos)
    df = pull_data(pts_dict, set_pos, set_year)

    if 'Rookie' not in set_pos:
        df, output_start = filter_df(df, pos, set_pos, set_year)
        df_train, df_predict, min_samples = get_reg_data(df, pos, set_pos)
        df_train_class, df_predict_class = get_class_data(df, pos, set_pos)
    else:
        df_train, df_predict, df_train_class, df_predict_class, output_start, min_samples = prepare_rookie_data(df, set_pos, cn)

    if current_or_next_year == 'next' and 'Rookie' not in set_pos: 
        df_train, df_train_class = adjust_current_or_next(df_train, df_train_class)

    #------------
    # Run the Regression, Classification, and Quantiles
    #------------

    # set up blank dictionaries for all metrics
    out_dict_reg, out_dict_class, out_dict_quant = output_dict(), output_dict(), output_dict()

    # run all other models
    model_list = ['adp', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf']
    for i, m in enumerate(model_list):
        out_dict_reg, _, _ = get_model_output(m, df_train, 'reg', out_dict_reg, pos, set_pos, i, min_samples)
    save_output_dict(out_dict_reg, model_output_path, 'reg')

    # run all other models
    model_list = ['lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c']
    for i, m in enumerate(model_list):
        out_dict_class, _, _= get_model_output(m, df_train_class, 'class', out_dict_class, pos, set_pos, i, min_samples)
    save_output_dict(out_dict_class, model_output_path, 'class')

    # run all other models
    for alph in [0.8, 0.95]:
        out_dict_quant, _, _ = get_model_output('gbm_q', df_train, 'quantile', out_dict_class, pos, set_pos, i, alpha=alph)
    save_output_dict(out_dict_quant, model_output_path, 'quant')

    #------------
    # Run the Stacking Models and Generate Output
    #------------

    # get the training data for stacking and prediction data after stacking
    X_stack, y_stack, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)
    X_predict = get_stack_predict_data(df_train, df_predict, df_train_class, df_predict_class, 
                                    models_reg, models_class, models_quant)

    # create the stacking models
    final_models = ['ridge', 'lasso', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm']
    stack_val_pred = pd.DataFrame(); scores = []; best_models = []
    for i, fm in enumerate(final_models):
        best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models, scores, stack_val_pred, show_plots=True)

    # get the best stack predictions and average
    predictions = mf.stack_predictions(X_predict, best_models, final_models)
    best_val, best_predictions = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, show_plot=True)

    # create the output and add standard devations / max scores
    output = mf.create_output(output_start, best_predictions)
    output = add_std_dev_max(df_train, df_predict, output, set_pos, num_cols=5)
    display(output.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50])

    # save out final results
    val_compare = validation_compare_df(model_output_path, best_val)
    save_out_results(val_compare, 'Simulation', 'Model_Validations', pos, set_year, set_pos, current_or_next_year)
    save_out_results(output, 'Simulation', 'Model_Predictions', pos, set_year, set_pos, current_or_next_year)


# %%


# %%
