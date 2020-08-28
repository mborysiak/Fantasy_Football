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

# # Reading in Old Salary Data

import pandas as pd
import numpy as np
import os
import sqlite3
from zData_Functions import *

# set core path
path = f'/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/Data/'

# # 2019 Salary + Actuals

results = pd.read_csv(f'{path}/OtherData/Salaries/salaries_2019_results_raw.csv', header=None)
results.columns = ['player', 'draft_amount']
results = results.dropna()
results = results[results.player != 'PLAYER'].reset_index(drop=True)
results.player = results.player.apply(lambda x: x.split(',')[0][:-3].rstrip().lstrip())
results.draft_amount = results.draft_amount.astype('float')

salaries = pd.read_csv(f'{path}/OtherData/Salaries/salaries_2019.csv')

keepers = ['Alvin Kamara', 'Nick Chubb', 'Michael Thomas', 'Damien Williams'
           'Mike Evans', 'Robert Woods', 'Tyler Lockett', 'Patrick Mahomes',
           'Adam Thielen', 'Ezekiel Elliott', 'Davante Adams', 'Adrian Peterson',
           'Devonta Freeman', 'Saquon Barkley', 'Tyreek Hill', 'Chris Godwin', 
           'Travis Kelce', 'Christian McCaffrey', 'James Conner', 'Joe Mixon', 
           'Tarik Cohen', 'Sony Michel']
results['keeper'] = 0

results

results.loc[results.player.isin(keepers), 'keeper'] = 1
salaries = pd.merge(salaries, results, on='player')
salaries['league'] = 'beta'

salaries.plot.scatter(x='salary', y='draft_amount')



# +
import pymc3 as pm

formula = 'draft_amount ~ salary'

# Context for the model
with pm.Model() as normal_model:
    
    # The prior for the data likelihood is a Normal Distribution
    family = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data = salaries, family = family)
    
    # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 500)

# +
from IPython.core.pylabtools import figsize
import seaborn as sns
import scipy.stats as stats

# Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation, max_bound):
    
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation)
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)
    
    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()
    
    # Add in intercept term
    test_observation['Intercept'] = 1
    
    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)
    
    # create truncated distribution
    lower, upper = 0,  max_bound * 1.2
    trunc_dist = stats.truncnorm((lower - mean_loc) / sd_value, (upper - mean_loc) / sd_value, 
                                  loc=mean_loc, scale=sd_value)
    estimates = trunc_dist.rvs(1000)

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                kde_kws = {'linewidth' : 4},
                label = 'Estimated Dist.')
    
    # Plot the mean estimate
    plt.vlines(x = mean_loc, ymin = 0, ymax = 0.1, 
               linestyles = '--', colors = 'red',
               label = 'Pred Estimate',
               linewidth = 2.5)
    
    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');
    
    # Prediction information
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                       np.percentile(estimates, 95)))
    
    plt.show()
    
    return estimates


# -

test_model(normal_trace, salaries.loc[1,['salary']], 105);

# # Pushing Salaries to Simulation Database

# +
year = 2020
league = 'nv'

salary_final = pd.read_csv(f'{path}/OtherData/Salaries/salaries_{year}_{league}.csv')
salary_final['year'] = year
salary_final['league'] = league
salary_final.player = salary_final.player.apply(name_clean)

# +
conn_sim = sqlite3.connect(f'{path}/Databases/Simulation.sqlite3')
conn_sim.cursor().execute(f'''DELETE FROM Salaries WHERE year={year} AND league='{league}' ''')
conn_sim.commit()
    
append_to_db(salary_final,'Simulation', 'Salaries', 'append')
# -

# # Pushing Injuries to Season_Stats Database

# +
from sklearn.preprocessing import StandardScaler

year = 2020

# read in the injury predictor file
inj = pd.read_csv(f'{path}/OtherData/InjuryPredictor/injury_predictor_{year}.csv', header=None)

# fix the columns and formatting
inj.columns = ['player', 'pct_miss_one', 'proj_games_missed', 'inj_pct_per_game', 'inj_risk', 'points']
inj.player = inj.player.apply(lambda x: x.split(',')[0])
inj.pct_miss_one = inj.pct_miss_one.apply(lambda x: float(x.strip('%')))
inj.inj_pct_per_game = inj.inj_pct_per_game.apply(lambda x: float(x.strip('%')))
inj = inj.drop(['points', 'inj_risk'], axis=1)

# scale the data and set minimum to 0
X = StandardScaler().fit_transform(inj.iloc[:, 1:])
inj = pd.concat([pd.DataFrame(inj.player),
                 pd.DataFrame(X, columns=['pct_miss_one', 'proj_games_missed', 'pct_per_game'])], 
                axis=1)
for col in ['pct_miss_one', 'proj_games_missed', 'pct_per_game']:
    inj[col] = inj[col] + abs(inj[col].min())
    
# take the mean risk value
inj['mean_risk'] = inj.iloc[:, 1:].mean(axis=1)
inj = inj[['player', 'mean_risk']].sort_values(by='mean_risk').reset_index(drop=True)

inj.player = inj.player.apply(name_clean)

# adjust specific players if needed
pts = [1, 1, 1, 1, 1]
players = ['Kerryon Johnson', 'Tyreek Hill', 'Todd Gurley', 'Deebo Samuel', 'Miles Sanders']
for pt, pl in zip(pts, players):
    inj.loc[inj.player==pl, 'mean_risk'] = inj.loc[inj.player==pl, 'mean_risk'] + pt
    
inj['year'] = year
# -

conn_sim = sqlite3.connect(f'{path}/Databases/Simulation.sqlite3')
conn_sim.cursor().execute(f'''DELETE FROM Injuries WHERE year={year}''')
conn_sim.commit()
append_to_db(inj, 'Simulation', 'Injuries', 'append')


