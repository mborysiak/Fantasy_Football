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

#%%

# import sim functions
from zSim_Helper import *
import seaborn as sns
from IPython.core.pylabtools import figsize

#===============
# Settings and User Inputs
#===============

np.random.seed(1234)

#--------
# League Settings
#--------

# connection for simulation and specific table
path = f'c:/Users/{os.getlogin()}/Documents/Github/Fantasy_Football/'
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
table_vers = 'Version2'
set_year = 2020
league='beta'

# number of iteration to run
iterations = 500

# define point values for all statistical categories
pass_yd_per_pt = 0.04 
pass_td_pt = 4
int_pts = -2
sacks = -1
rush_yd_per_pt = 0.1 
rec_yd_per_pt = 0.1
rush_rec_td = 7
ppr = .5

# creating dictionary containing point values for each position
pts_dict = {}
pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_rec_td, int_pts, sacks]
pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['WR'] = [rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['TE'] = [rec_yd_per_pt, ppr, rush_rec_td]


#==================
# Initialize the Simluation Class
#==================

# instantiate simulation class and add salary information to data
sim = FootballSimulation(pts_dict, conn_sim, table_vers, set_year, league, iterations)

# return the data and set up dataframe for proportion of salary across position
d = sim.return_data()
d = d.rename(columns={'pos': 'Position', 'salary': 'Salary'})
d.Position = d.Position.apply(lambda x: x[1:])
proport = d[d.Salary > 1].copy().reset_index(drop=True)
proport = proport.groupby('Position').agg({'Salary': 'mean'})

# get the proportion of salary by position
proport['total'] = proport.Salary.sum()
proport['Wts'] = proport.Salary / proport.total
proport = proport.reset_index()
proport = proport[['Position', 'Wts']]

# set league information, included position requirements, number of teams, and salary cap
league_info = {}
league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
league_info['num_teams'] = 12
league_info['initial_cap'] = 293
league_info['salary_cap'] = 293

player_list = []
for pl, row in d[['Position']].iterrows():
    if row.Position != 'FLEX':
        player_list.append([pl, row.Position, 0])

pick_df = pd.DataFrame(player_list, columns=['Player', 'Position', 'Salary'])
#------------------
# For Beta Keepers
#------------------

# input information for players and their associated salaries selected by other teams
keepers = {
    # 'Christian McCaffrey': 97,
    # 'Saquon Barkley': 126,
    # 'Dalvin Cook': 80,
    # 'Derrick Henry': 61,
    # 'Miles Sanders': 31,
    # 'Kenyan Drake': 23,
    # 'James Conner': 26,
    # 'Chris Godwin': 26,
    # 'AJ Brown': 33,
    # 'Terry McLaurin': 11,
    # 'Lamar Jackson': 11,
    # 'Patrick Mahomes': 26,
}

# # 2019 keepers
keepers = {
    # 'Patrick Mahomes': 11,
    # 'Christian McCaffrey': 82,
    # 'Saquon Barkley': 111,
    # 'Ezekiel Elliott': 100,
    # 'Miles Sanders': 31,
    # 'Joe Mixon': 55,
    # 'James Conner': 11,
    # 'Michael Thomas': 55,
    # 'Davante Adams': 60,
    # 'Tyreek Hill': 35,
    # 'Chris Godwin': 11,
    # 'Adam Thielen': 35,
    # 'Tyler Lockett': 11,
    # 'Travis Kelce': 55,
}

for p, s in keepers.items():
    pick_df.loc[pick_df.Player==p, 'Salary'] = s

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html 
import plotly.express as px
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

# set up dash with external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#==========================
# Submit Data Entry
#==========================

# input cells
player_input = dcc.Input(id='player-update', type='text', value=None)
salary_input = dcc.Input(id='salary-update', type='text', value=None)

# submit button
red_button_style = {'background-color': 'red',
                    'color': 'white',
                    'height': '150',
                    'width': '300',
                    'margin-left': '50px'}
submit_button = html.Button(id='submit-button-state', n_clicks=0, children='Refresh Top Picks', style=red_button_style)


#============================
# Build out Dash Tables
#============================

# set up all players drafted DataTable
subset_sal = pick_df[pick_df.Salary > 0]
drafted_player_table =  dash_table.DataTable(
                            id='draft-results-table',
                            columns=[{"name": i, "id": i} for i in pick_df.columns],
                            data=subset_sal.to_dict('records'),
                        )


# set up dataframe for Data Table for My Team
my_team_list = []
for k, v in league_info['pos_require'].items():
    for i in range(v):
        my_team_list.append([k, None, 0])
my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'Salary'])

 # set up my team  drafted DataTable
my_team_table =  dash_table.DataTable(
                            id='my-team-table',
                            columns=[{"name": i, "id": i} for i in my_team_df.columns],
                            data=my_team_df.to_dict('records'),
                            editable=True
                        )


#==========================
# Plotting Functions
#==========================

def create_bar(x_val, y_val, orient='h', color_str='rgba(50, 171, 96, 0.6)', text=None):
    
    marker_set = dict(color=color_str, line=dict(color=color_str, width=1))
    return go.Bar(x=x_val, y=y_val, marker=marker_set, orientation=orient, text=text)


def create_fig_layout(fig1, fig2):

    fig = go.Figure(data=[fig1, fig2])
    
    # Change the bar mode
    fig.update_layout(barmode='group', autosize=True, height=800, margin=dict(l=0, r=25, b=0, t=15, pad=0),
                      uniformtext_minsize=25, uniformtext_mode='hide'
                      )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    return fig


def create_pt_dist(my_team_df, proport, remaining_sal):

    # determine who is already selected and get their points
    selected = list(my_team_df.loc[~(my_team_df.Player.isnull()) | ~(my_team_df.Player==''), 'Player'].values)
    selected = d[(d.index.isin(selected)) & (d.Position!='FLEX')].drop(['Position', 'Salary'], axis=1)
    selected.columns = [int(c) for c in selected.columns]

    # get the remaining positions from my team and merge with proportion of salary to each
    remain = my_team_df.loc[(my_team_df.Player.isnull()) | (my_team_df.Player==''), 'Position']
    remain = pd.merge(remain, proport, on='Position')

    # readjust the weights to equal 1
    remain['TotalWt'] = remain.Wts.sum()
    remain['AdjustWt'] = remain.Wts / remain.TotalWt

    # multiply the weights for each position by remaining dollars
    remain['Salary'] = remain.AdjustWt * remaining_sal
    remain = remain[['Position', 'Salary']]

    # create dataset that models Pts and Std of Pts based on Salary + Position
    Xy = pd.concat([d.Position, d.Salary, d.iloc[:, 1:999].mean(axis=1),  d.iloc[:, 1:999].std(axis=1)], axis=1).reset_index(drop=True)
    Xy = pd.concat([Xy, remain], axis=0)
    Xy = pd.concat([pd.get_dummies(Xy.Position), Xy.Salary, Xy[0], Xy[1]], axis=1)

    # create X and y variables for training / predicting
    X_train = Xy.dropna().drop([0, 1], axis=1)
    X_test = Xy[Xy[0].isnull()].drop([0, 1], axis=1)
    y_mean = Xy[0].dropna()
    y_std = Xy[1].dropna()

    # generate model fit
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    if len(X_test) > 0:
        
        lr.fit(X_train, y_mean)
        remain_means = lr.predict(X_test)
        
        lr.fit(X_train, y_std)
        remain_std = lr.predict(X_test)
        
        team_dist = pd.DataFrame()
        team_dist = pd.concat([team_dist, selected], axis=0)
        np.random.seed(1234)
        for m, s in zip(remain_means, remain_std):
            team_dist = pd.concat([team_dist, pd.DataFrame(np.random.normal(m, s, 1000)).T], axis=0)
    else:
        team_dist = selected

    team_dist = team_dist.sum(axis=0)
    team_dist = team_dist * (11/16) + 150 + 275
    team_dist = pd.DataFrame(team_dist, columns=['projection'])

    pt_percentiles = {
        '20': int(np.percentile(team_dist, 20)),
        '50': int(team_dist.mean()),
        '80': int(np.percentile(team_dist, 80))
    }

    hist_data = [team_dist.projection.values]
    group_labels = [''] # name of the dataset

    fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=10)
    fig_hist.update_layout(autosize=True, height=250, margin=dict(l=0, r=0, b=0, t=15, pad=0), showlegend=False)

    return  fig_hist, pt_percentiles

#--------------
# Plot Creation
#--------------

# bar chart creation
pick_bar_init = create_bar( [100],['Mark'])
sal_bar_init = create_bar([100], ['Mark'])
fig = create_fig_layout(pick_bar_init, sal_bar_init)
gr = dcc.Graph(id='draft-results-graph', figure=fig)

# histogram creation
hist_fig = create_pt_dist(my_team_df, proport, remaining_sal=league_info['salary_cap'])
hist_gr = dcc.Graph(id='team-points-graph')


#============================
# Build out App Layout
#============================

app.layout = html.Div([

     html.Div([
         html.Div([
            html.H5('My Team'),
            my_team_table, html.Br(),
            html.Div(id='team-info'), hist_gr,  html.Hr(),
            html.H5("Other Team's Draft Picks"),
            player_input, salary_input,  html.Hr(),
           drafted_player_table,
            ], className="five columns"),

            html.Div([
                html.Div([
                    html.H5('Recommended Picks')
                 ], className='three columns'),
                 html.Div([
                     submit_button
                 ], className='two columnss'),
                gr, 
            ], className="seven columns")
       
       ], className="row2") ,        
         
])


#============================
# Update Functions
#============================

def update_to_drop(pick_df):

    to_drop = {}
    to_drop['players'] = []
    to_drop['salaries'] = []
    for _, row in pick_df.iterrows():
        if row.Salary > 0:
            to_drop['players'].append(row.Player)
            to_drop['salaries'].append(row.Salary)

    return to_drop


def update_to_add(my_team_df):
    
    to_add = {}
    to_add['players'] = []
    to_add['salaries'] = []
    for _, row in my_team_df.iterrows():
        if row.Player is not None:
            to_add['players'].append(row.Player)
            to_add['salaries'].append(row.Salary)

    return to_add



@app.callback([Output('draft-results-graph', 'figure'),
               Output('team-points-graph', 'figure'),
               Output('draft-results-table', 'data'),
               Output('team-info', 'children')
               ],
              [Input('submit-button-state', 'n_clicks')
               
               ],
              [State('my-team-table', 'data'),
               State('my-team-table', 'columns'),
                  State('player-update', 'value'),
               State('salary-update', 'value')]
)
def update_output(n_clicks, my_team_data, my_team_columns, p_update, s_update):

    my_team_df_raw = pd.DataFrame(my_team_data, columns=[c['name'] for c in my_team_columns])
    my_team_df = my_team_df_raw.copy().dropna()
    my_team_df = my_team_df[(my_team_df.Player != '')]
    my_team_df['Salary'] = my_team_df['Salary'].astype('int')
    
    # update the player + salary that have been picked
    if p_update is not None and s_update is not None:
        pick_df.loc[pick_df.Player == p_update, 'Salary'] = int(s_update)

    to_drop = update_to_drop(pick_df)
    to_add = update_to_add(my_team_df)

    # run the simulation
    _, inflation = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
    
    # get the results dataframe structured
    avg_sal = sim.show_most_selected(to_add, iterations, num_show=30)
    avg_sal = avg_sal.sort_values(by='Percent Drafted').reset_index()
    avg_sal = avg_sal.iloc[-20:]
    avg_sal.columns = ['Player', 'PercentDrafted', 'AverageSalary', 'ExpectedSalaryDiff']

    # Creating two subplots and merging into single figure
    pick_text = [str(c) for c in avg_sal.PercentDrafted]
    sal_text = [str(c) for c in avg_sal.AverageSalary]
    pick_bar = create_bar( list(avg_sal.PercentDrafted),list(avg_sal.Player), text=pick_text)
    sal_bar = create_bar(list(avg_sal.AverageSalary), list(avg_sal.Player), 
                         color_str='rgba(250, 190, 88, 1)', text=sal_text)
    fig = create_fig_layout(sal_bar, pick_bar)

    # show drafted players
    subset_sal = pick_df[pick_df.Salary > 0].copy()
    remain_sal = league_info['salary_cap'] - np.sum(to_add['salaries'])
    
    # histogram creation
    hist_fig, pt_perc = create_pt_dist(my_team_df_raw, proport, remaining_sal=remain_sal)
    output_str = f"Mean Team Pts: {pt_perc['50']}-----Inflation Percent: {round(inflation,2)}-----Remaining Salary:{remain_sal}"
    
    # save out csv of status
    pick_df.to_csv('c:/Users/mborysia/Desktop/Status_Save.csv', index=False)

    return fig, hist_fig, subset_sal.to_dict('records'), output_str

if __name__ == '__main__':
    app.run_server(debug=False)
