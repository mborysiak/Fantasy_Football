# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # single random holdout

# +
@ignore_warnings(category=ConvergenceWarning)
def calc_f1_score(**args):
    
    globals()['cnter'] += 1
    i = globals()['cnter']
    if i % 25 == 0:
        print(f'\n---------- Run {i} Completed\n')
        
    from sklearn.metrics import f1_score
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    # remove the extra args not needed for modeling
    scale = True if args['scale'] == 1 else False
    pca = True if args['pca'] == 1 else False   
    use_smote = True if args['use_smote'] == 1 else False
    
    n_components = args['n_components']
    collinear_cutoff = args['collinear_cutoff']
    zero_weight = args['zero_weight']
    
    for arg in ['scale', 'pca', 'collinear_cutoff', 'use_smote', 'zero_weight', 'n_components']:
        del args[arg]

    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------

    # remove collinear variables based on difference of means between the 0 and 1 labeled groups
    df_train = remove_classification_collinear(globals()['df_train_orig'], collinear_cutoff, ['player', 'avg_pick', 'year', 'y_act'])
    df_predict = globals()['df_predict_orig'][df_train.columns]
    
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    opt_predictions = np.array([]) 
    y_vals = np.array([])
    y_opts = np.array([])
    
    # split train and holdout datasets
    train, holdout = train_test_split(df_train, test_size=0.2, random_state=i*4+i*9+13, 
                                      shuffle=True, stratify=df_train.y_act)
 
    train = train.sort_values(by='year').reset_index(drop=True)
    holdout = holdout.sort_values(by='year').reset_index(drop=True)

    #==========
    # Loop through years and complete a time-series validation of the given model
    #==========

    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = df_train[df_train.year < m]
        opt_split = df_train[df_train.year == m]

        # set up the estimator
        estimator.set_params(**args)
        estimator.class_weight = {0: zero_weight, 1: 1}

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_opt, y_train, y_opt = X_y_split(train_split, opt_split, scale, pca, n_components)

        if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)

            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_opt = X_opt.values

        # train the estimator and get predictions
        estimator.fit(X_train, y_train)
        opt_predict = estimator.predict(X_opt)

        train_split = pd.concat([train_split, holdout[holdout.year < m]], axis=0).reset_index(drop=True)
        val_split = pd.concat([opt_split, holdout[holdout.year == m]], axis=0).reset_index(drop=True)

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_val, y_train, y_val = X_y_split(train_split, val_split, scale, pca, n_components)

        if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)

            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_val = X_val.values

        # train the estimator and get predictions
        estimator.fit(X_train, y_train)
        val_predict = estimator.predict(X_val)

#         val_predict = opt_predict.copy()
#         y_val = y_opt.copy()

        # append the predictions
        val_predictions = np.append(val_predictions, val_predict, axis=0)
        opt_predictions = np.append(opt_predictions, opt_predict, axis=0)
        y_vals = np.append(y_vals, y_val, axis=0)
        y_opts = np.append(y_opts, y_opt, axis=0)



    df_train = pd.concat([train, holdout], axis=0).reset_index(drop=True)

    # splitting the train and validation sets into X_train, y_train, X_val and y_val
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)

    if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)
            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_test = X_test.values

    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========
    
    val_score = round(-matthews_corrcoef(val_predictions, y_vals), 3)
    test_score = round(-matthews_corrcoef(test_predict, y_test), 3)
    opt_score = round(-matthews_corrcoef(opt_predictions, y_opts), 3)
    
    # add in the opt predictions
    val_predictions = np.append(val_predictions, opt_predictions, axis=0)
    y_vals = np.append(y_vals, y_opts, axis=0)
    
    # set weights for validation to 1 and for test set to 2
    wts = [1]*len(val_predictions)
    wts.extend([1.5]*len(y_test))
    
    # append test to validation predictions for combined scoring
    val_predictions = np.append(val_predictions, test_predict, axis=0)
    y_vals = np.append(y_vals, y_test, axis=0)
    
   
    # calculate the RMSE and MAE of the ensemble predictions
    m_score = -round(matthews_corrcoef(val_predictions, y_vals, sample_weight=wts), 3)
    
    if i == 25:
        print('\nBest Random Scores:')
        print('OptScore:',globals()['best_opt'])
        print('CombinedScore:', globals()['best_combined'])
    
    if opt_score < globals()['best_opt']:
        globals()['best_opt'] = opt_score
        
        if i > 25:
            print('\nNew Best Opt Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
        
    if m_score < globals()['best_combined']:
        globals()['best_combined'] = m_score
        
        if i > 25:
            print('\nNew Best Combined Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
    
    globals()['opt_scores'].append(opt_score) 
    globals()['val_scores'].append(val_score)    
    globals()['test_scores'].append(test_score)
    globals()['combined_scores'].append(m_score)
    
    return opt_score


#================
# Run Optimization
#================
    
opt_scores = []
val_scores = []
test_scores = []
combined_scores = []
models_list = []

skip_years = pos[set_pos]['skip_years']

for m_num, model in enumerate(list(class_models.keys())):
    
    cnter = 1

    print(f'\n============= Running {model} =============\n')

    best_opt = 100
    best_combined = 100
    models_list.extend([model]*100)

    estimator = class_models[model]
    space = class_search_space[model]

    @use_named_args(space)
    def objective_class(**args):
        return calc_f1_score(**args)

    @use_named_args(space)
    def space_keys(**args):
        return list(args.keys())

    bayes_seed = 12345
    kappa = 3
    res_gp = eval(pos[set_pos]['minimizer'])(objective_class, space, n_calls=100, n_random_starts=25,
                                             random_state=bayes_seed, verbose=False, kappa=kappa, n_jobs=-1)

    output_class.loc[0, 'breakout_metric'] = breakout_metric
    output_class.loc[0, 'act_ppg'] = act_ppg
    output_class.loc[0, 'pct_off'] = pct_off
    output_class.loc[0, 'adp_ppg_high'] = adp_ppg_high
    output_class.loc[0, 'adp_ppg_low'] = adp_ppg_low
    output_class.loc[0, 'model'] = model
    output_class.loc[0, 'earliest_year'] = df_train_orig.year.min()
    output_class.loc[0, 'score'] = -best_combined
    
    _ = val_results(models_list, val_scores, combined_scores, test_scores)

    best_params_model = res_gp.x_iters[np.argmin(combined_scores[m_num*100:(m_num+1)*100])]
    
    print('Best Combined Score:', best_combined)
    params_output = dict(zip(space_keys(space), best_params_model))
    print(params_output)
    
    append_to_db(output_class, db_name='ParamTracking', table_name='ClassParamTracking', if_exist='append')
    max_pkey = pd.read_sql_query("SELECT max(pkey) FROM ClassParamTracking", param_conn).values[0][0]

    save_pickle(params_output, path + f'Data/Model_Params_Class/{max_pkey}.p')
    save_pickle(df_train_orig, path + f'Data/Model_Datasets_Class/{max_pkey}.p')    
    save_pickle(class_search_space[model], path + f'Data/Bayes_Space_Class/{max_pkey}.p')


# -

# # starting stratified cv with ts

# +
@ignore_warnings(category=ConvergenceWarning)
def calc_f1_score(**args):
    
    globals()['cnter'] += 1
    i = globals()['cnter']
    if i % 25 == 0:
        print(f'\n---------- Run {i} Completed\n')
        
    from sklearn.metrics import f1_score
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    # remove the extra args not needed for modeling
    scale = True if args['scale'] == 1 else False
    pca = True if args['pca'] == 1 else False   
    use_smote = True if args['use_smote'] == 1 else False
    
    n_components = args['n_components']
    collinear_cutoff = args['collinear_cutoff']
    zero_weight = args['zero_weight']
    
    for arg in ['scale', 'pca', 'collinear_cutoff', 'use_smote', 'zero_weight', 'n_components']:
        del args[arg]

    #----------
    # Filter down the features with a random correlation and collinear cutoff
    #----------

    # remove collinear variables based on difference of means between the 0 and 1 labeled groups
    df_train = remove_classification_collinear(globals()['df_train_orig'], collinear_cutoff, ['player', 'avg_pick', 'year', 'y_act'])
    df_predict = globals()['df_predict_orig'][df_train.columns]
    
    years = df_train_orig.year.unique()
    years = years[years > np.min(years) + skip_years]
    
    # set up array to save predictions and years to iterate through
    val_predictions = np.array([]) 
    opt_predictions = np.array([]) 
    y_vals = np.array([])
    y_opts = np.array([])
    
    #==========
    # K-Fold Validation Loop
    #==========
    
    skf = StratifiedKFold(n_splits=5, random_state=1234)
    for train_index, test_index in skf.split(df_train_orig, df_train_orig.year):
        
        train_fold, holdout = df_train.iloc[train_index, :], df_train.iloc[test_index, :]
        train_fold = train_fold.sort_values(by='year').reset_index(drop=True)

        for m in years:

            # create training set for all previous years and validation set for current year
            train_split = train_fold[train_fold.year < m]
            opt_split = train_fold[train_fold.year == m]

            # set up the estimator
            estimator.set_params(**args)
            estimator.class_weight = {0: zero_weight, 1: 1}

            # splitting the train and validation sets into X_train, y_train, X_val and y_val
            X_train, X_opt, y_train, y_opt = X_y_split(train_split, opt_split, scale, pca, n_components)

            if use_smote:
                knn = int(len(y_train[y_train==1])*0.5)
                smt = SMOTE(k_neighbors=knn, random_state=1234)

                X_train, y_train = smt.fit_resample(X_train.values, y_train)
                X_opt = X_opt.values

            # train the estimator and get predictions
            estimator.fit(X_train, y_train)
            opt_predict = estimator.predict(X_opt)

            # append the predictions
            opt_predictions = np.append(opt_predictions, opt_predict, axis=0)
            y_opts = np.append(y_opts, y_opt, axis=0)

    #------------
    # Full Validation Loop
    #------------
    
    for m in years:

        # create training set for all previous years and validation set for current year
        train_split = train_fold[train_fold.year < m]
        opt_split = train_fold[train_fold.year == m]

        # set up the estimator
        estimator.set_params(**args)
        estimator.class_weight = {0: zero_weight, 1: 1}

        # splitting the train and validation sets into X_train, y_train, X_val and y_val
        X_train, X_opt, y_train, y_opt = X_y_split(train_split, opt_split, scale, pca, n_components)

        if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)

            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_opt = X_opt.values

        # train the estimator and get predictions
        estimator.fit(X_train, y_train)
        opt_predict = estimator.predict(X_opt)

        # append the predictions
        val_predictions = np.append(val_predictions, val_predict, axis=0)
        opt_predictions = np.append(opt_predictions, opt_predict, axis=0)
        y_vals = np.append(y_vals, y_val, axis=0)
        y_opts = np.append(y_opts, y_opt, axis=0)

    # splitting the train and validation sets into X_train, y_train, X_val and y_val
    X_train, X_test, y_train, y_test = X_y_split(df_train, df_predict, scale, pca, n_components)

    if use_smote:
            knn = int(len(y_train[y_train==1])*0.5)
            smt = SMOTE(k_neighbors=knn, random_state=1234)
            X_train, y_train = smt.fit_resample(X_train.values, y_train)
            X_test = X_test.values

    estimator.fit(X_train, y_train)
    test_predict = estimator.predict(X_test)

    #==========
    # Calculate Error Metrics and Prepare Export
    #==========
    
    val_score = round(-matthews_corrcoef(val_predictions, y_vals), 3)
    test_score = round(-matthews_corrcoef(test_predict, y_test), 3)
    opt_score = round(-matthews_corrcoef(opt_predictions, y_opts), 3)
    
    # add in the opt predictions
    val_predictions = np.append(val_predictions, opt_predictions, axis=0)
    y_vals = np.append(y_vals, y_opts, axis=0)
    
    # set weights for validation to 1 and for test set to 2
    wts = [1]*len(val_predictions)
    wts.extend([1.5]*len(y_test))
    
    # append test to validation predictions for combined scoring
    val_predictions = np.append(val_predictions, test_predict, axis=0)
    y_vals = np.append(y_vals, y_test, axis=0)
    
   
    # calculate the RMSE and MAE of the ensemble predictions
    m_score = -round(matthews_corrcoef(val_predictions, y_vals, sample_weight=wts), 3)
    
    if i == 25:
        print('\nBest Random Scores:')
        print('OptScore:',globals()['best_opt'])
        print('CombinedScore:', globals()['best_combined'])
    
    if opt_score < globals()['best_opt']:
        globals()['best_opt'] = opt_score
        
        if i > 25:
            print('\nNew Best Opt Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
        
    if m_score < globals()['best_combined']:
        globals()['best_combined'] = m_score
        
        if i > 25:
            print('\nNew Best Combined Score Found:')
            print('OptScore:',opt_score)
            print('ValScore:',val_score)
            print('TestScore:', test_score)
            print('CombinedScore:', m_score)
    
    globals()['opt_scores'].append(opt_score) 
    globals()['val_scores'].append(val_score)    
    globals()['test_scores'].append(test_score)
    globals()['combined_scores'].append(m_score)
    
    return opt_score


#================
# Run Optimization
#================
    
opt_scores = []
val_scores = []
test_scores = []
combined_scores = []
models_list = []

skip_years = pos[set_pos]['skip_years']

for m_num, model in enumerate(list(class_models.keys())):
    
    cnter = 1

    print(f'\n============= Running {model} =============\n')

    best_opt = 100
    best_combined = 100
    models_list.extend([model]*100)

    estimator = class_models[model]
    space = class_search_space[model]

    @use_named_args(space)
    def objective_class(**args):
        return calc_f1_score(**args)

    @use_named_args(space)
    def space_keys(**args):
        return list(args.keys())

    bayes_seed = 12345
    kappa = 3
    res_gp = eval(pos[set_pos]['minimizer'])(objective_class, space, n_calls=100, n_random_starts=25,
                                             random_state=bayes_seed, verbose=False, kappa=kappa, n_jobs=-1)

    output_class.loc[0, 'breakout_metric'] = breakout_metric
    output_class.loc[0, 'act_ppg'] = act_ppg
    output_class.loc[0, 'pct_off'] = pct_off
    output_class.loc[0, 'adp_ppg_high'] = adp_ppg_high
    output_class.loc[0, 'adp_ppg_low'] = adp_ppg_low
    output_class.loc[0, 'model'] = model
    output_class.loc[0, 'earliest_year'] = df_train_orig.year.min()
    output_class.loc[0, 'score'] = -best_combined
    
    _ = val_results(models_list, val_scores, combined_scores, test_scores)

    best_params_model = res_gp.x_iters[np.argmin(combined_scores[m_num*100:(m_num+1)*100])]
    
    print('Best Combined Score:', best_combined)
    params_output = dict(zip(space_keys(space), best_params_model))
    print(params_output)
    
    append_to_db(output_class, db_name='ParamTracking', table_name='ClassParamTracking', if_exist='append')
    max_pkey = pd.read_sql_query("SELECT max(pkey) FROM ClassParamTracking", param_conn).values[0][0]

    save_pickle(params_output, path + f'Data/Model_Params_Class/{max_pkey}.p')
    save_pickle(df_train_orig, path + f'Data/Model_Datasets_Class/{max_pkey}.p')    
    save_pickle(class_search_space[model], path + f'Data/Bayes_Space_Class/{max_pkey}.p')
