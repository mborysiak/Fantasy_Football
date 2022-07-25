#%%

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

import zHelper_Functions as hf
pos = hf.pos

# set the root path and database management object
root_path = ffgeneral.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set the root path and database management object
root_path = ffgeneral.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


def create_sd_max_metrics(df, sd_cols, max_cols):

    sc = MinMaxScaler()
    X_sd = pd.DataFrame(sc.fit_transform(df[sd_cols]), columns=sd_cols)
    X_max = pd.DataFrame(sc.fit_transform(df[max_cols]), columns=max_cols)

    df['sd_metric'] = X_sd.mean(axis=1)
    df['max_metric'] = X_max.mean(axis=1)
    
    df = df[[c for c in df.columns if c not in max_cols and c not in sd_cols]]

    return df


def create_groups(df, num_grps):
    # create equal sizes groups going down the dataframe ordered by each metric
    df_len = len(df)
    repeats = math.ceil(df.shape[0] / num_grps)
    grps = np.repeat([i for i in range(num_grps)], repeats)
    df['grps'] = grps[:df_len]
    return df

def show_spline_fit(splines, met, X, y, X_max, y_max):
    print(met)
    X_pred = list(np.arange(0, 1, 0.1))
    plt.scatter(X, y)
    plt.scatter(X_max[met], y_max[met])
    plt.plot(X_pred, splines[met](X_pred), 'g', lw=3)
    plt.show()

def get_std_splines(df, sd_cols, max_cols, show_plot=False, k=2, s=2000, min_grps_den=100, max_grps_den=60):
    
    
    all_cols = list(set(sd_cols + max_cols  + ['player', 'year', 'y_act']))
    df = df[all_cols].dropna().reset_index(drop=True)

    # calculate sd and max metrics
    df = create_sd_max_metrics(df, sd_cols, max_cols)

    # create the groups    
    df = df.dropna()
    min_grps = int(df.shape[0] / min_grps_den)
    max_grps = int(df.shape[0] / max_grps_den)

    splines = {}; X_max = {}; y_max = {}; max_r2 = {}
    for x_val, met in zip(['sd_metric', 'max_metric'], ['std_dev', 'perc_99']):
        
        df = df.sort_values(by=x_val).reset_index(drop=True)

        max_r2[met] = 0
        for num_grps in range(min_grps, max_grps, 1):

            # create the groups to aggregate for std dev and max metrics
            df = create_groups(df, num_grps)

            # calculate the standard deviation and max of each group
            Xy = df.groupby('grps').agg({'y_act': [np.std, lambda x: np.percentile(x, 99)],
                                         'sd_metric': 'mean',
                                         'max_metric': 'mean',
                                         'player': 'count'})
            Xy.columns = ['std_dev', 'perc_99', 'sd_metric', 'max_metric', 'player_cnts']

            # fit a spline to the group datasets
            X = Xy[[x_val]]
            y = Xy[[met]]
            spl = UnivariateSpline(X, y, k=k, s=s)

            r2 = r2_score(y, spl(X))
            if r2 > max_r2[met]:
                max_r2[met] = r2
                splines[met] = spl
                X_max[met] = X
                y_max[met] = y

        if show_plot:
            show_spline_fit(splines, met, X, y, X_max, y_max)
            
    return splines['std_dev'], splines['perc_99']