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


def _bar_center_zero(self, s, color_positive, width):

    # Either the min or the max should reach the edge (50%, centered on zero)
    m = max(abs(s.min()),abs(s.max()))
    normed = s * 60 * width / (100 * m)
    base = 'width: 10em; height: 80%;'

    attrs_pos = (base+ 'background: linear-gradient(90deg, transparent 0%, transparent 5%, {c} 0%, {c} {w}%, '
                'transparent {w}%)')

    return [attrs_pos.format(c=color_positive,  w=(5+x)) for x in normed]

def bar_excel(self, axis=0, color_positive='#5FBA7D', width=100):

    self.apply(self._bar_center_zero, axis=axis, color_positive=color_positive, width=width)
    
    return self

# create the bar charts within the
pd.io.formats.style.Styler._bar_center_zero = _bar_center_zero
pd.io.formats.style.Styler.bar_excel = bar_excel

#==================
# Initialize the Simluation Class
#==================

# instantiate simulation class and add salary information to data
sim = FootballSimulation(pts_dict, conn_sim, table_vers, set_year, iterations)
d = sim.return_data()

# set league information, included position requirements, number of teams, and salary cap
league_info = {}
league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
league_info['num_teams'] = 12
league_info['initial_cap'] = 293
league_info['salary_cap'] = 293

player_list = []
for pl, row in d[['pos']].iterrows():
    if row.pos != 'eFLEX':
        player_list.append([pl, row.pos[1:], 0])

pick_df = pd.DataFrame(player_list, columns=['Player', 'Position', 'Salary'])
#------------------
# For Beta Keepers
#------------------

# input information for players and their associated salaries selected by other teams
keepers = {
    'Christian McCaffrey': 97,
    'Saquon Barkley': 126,
    'Dalvin Cook': 80,
    'Derrick Henry': 61,
    'Miles Sanders': 31,
    'Clyde Edwards-Helaire': 99,
    'Kenyan Drake': 23,
    "Le'Veon Bell": 0,
    'James Conner': 26,
    'Chris Godwin': 26,
    'AJ Brown': 33,
    'Terry McLaurin': 11,
    'Lamar Jackson': 11,
    'Patrick Mahomes': 26,
    'Kyler Murray': 13,
}

# %%
# team_pts = d[(d.index.isin(team_picks.keys())) & (d.pos!='eFLEX')].drop(['pos', 'salary'], axis=1).sum(axis=0)
# team_pts = team_pts * (11/16) + 150 + 275
# print(f'20th Percentile: {np.percentile(team_pts, 20)}')
# print(f'Team Mean: {team_pts.mean()}')
# print(f'80th Percentile: {np.percentile(team_pts, 80)}')

# plt.figure(figsize(8, 8))
# sns.distplot(team_pts, hist = True, kde = True, bins = 19,
#                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
#             kde_kws = {'linewidth' : 4},
#             label = 'Estimated Dist.');


#%%
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html 
import plotly.express as px
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# set up dash with external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# set up dataframe for Data Table for My Team
my_team_list = []
for k, v in league_info['pos_require'].items():
    for i in range(v):
        my_team_list.append([k, None, None])
my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'Salary'])

# input cells
player_input = dcc.Input(id='player-update', type='text', value='Alvin Kamara')
salary_input = dcc.Input(id='salary-update', type='text', value='95')

# create radio button for my other or other team
radio_team = dcc.RadioItems(
                id='team-radio',
                options=[{'label': i, 'value': i} for i in ['My Team', 'Other Teams']],
                value='My Team',
                labelStyle={'display': 'inline-block'}
            )

# submit button
submit_button = html.Button(id='submit-button-state', n_clicks=0, children='Submit')

#---------------
# Plotting Functions
#---------------

# Creating two subplots
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False,
                    shared_yaxes=True, vertical_spacing=0.001)

def create_bar(x_val, y_val, orient='h', color_str='rgba(50, 171, 96, 0.6)'):
    
    marker_set = dict(color=color_str, line=dict(color=color_str, width=1))
    return go.Bar(x=x_val, y=y_val, marker=marker_set, orientation=orient)

def create_fig_layout(fig1, fig2):
    
    # Create the plot layout
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False,
                        shared_yaxes=True, vertical_spacing=0.001)

    fig.append_trace(fig1, 1, 1)
    fig.append_trace(fig2, 1, 2)

    fig.update_layout(autosize=True, height=800, margin=dict(l=0, r=25, b=0, t=0, pad=0))

    return fig

pick_bar_init = create_bar( [100],['Mark'])
sal_bar_init = create_bar([100], ['Mark'])
fig = create_fig_layout(pick_bar_init, sal_bar_init)

# player salary graph object
gr = dcc.Graph(id='draft-results-graph', figure=fig)

# set up all players drafted DataTable
subset_sal = pick_df[pick_df.Salary > 0]
drafted_player_table =  dash_table.DataTable(
                            id='draft-results-table',
                            columns=[{"name": i, "id": i} for i in pick_df.columns],
                            data=subset_sal.to_dict('records'),
                        )

 # set up my team  drafted DataTable
my_team_table =  dash_table.DataTable(
                            id='my-team-table',
                            columns=[{"name": i, "id": i} for i in my_team_df.columns],
                            data=my_team_df.to_dict('records'),
                        )

app.layout = html.Div([

     html.Div([
         html.Div([
            html.H4('Enter Draft Picks'),
            player_input, salary_input, html.Br(), html.Br(),
            radio_team, html.Br(),
            submit_button, html.Hr(),
            html.H4('My Team'),
            my_team_table, html.Br(), html.Hr(),
            html.H4('Drafted Players'), 
            drafted_player_table
            ], className="five columns"),

            html.Div([
                html.H4('Recommended Picks'),  
                gr
            ], className="seven columns")
       
       ], className="row2") ,        
         
])


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
               Output('draft-results-table', 'data'),
               Output('my-team-table', 'data')],
              [Input('submit-button-state', 'n_clicks'),
               Input('team-radio', 'value')],
              [State('player-update', 'value'),
               State('salary-update', 'value')]
)
def update_output(n_clicks, team_radio, p_update, s_update):
    
    if n_clicks == 0 or n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update

    else:
        
        # update the player + salary that have been picked
        if team_radio == 'Other Teams':
            pick_df.loc[pick_df.Player == p_update, 'Salary'] = int(s_update)

        elif team_radio == 'My Team': 
            pos_picked = list(pick_df.loc[pick_df.Player == p_update, 'Position'])[0]
            idx = my_team_df[(my_team_df.Position==pos_picked) & (my_team_df.Player.isnull())].index[0]
            my_team_df.loc[my_team_df.index==idx, ['Player', 'Salary']] = [p_update, int(s_update)]

        to_drop = update_to_drop(pick_df)
        to_add = update_to_add(my_team_df)

        # run the simulation
        _ = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
        
        # get the results dataframe structured
        avg_sal = sim.show_most_selected(to_add, iterations, num_show=30)
        avg_sal = avg_sal.sort_values(by='Percent Drafted').reset_index()
        avg_sal.columns = ['Player', 'PercentDrafted', 'AverageSalary', 'ExpectedSalaryDiff']

        # Creating two subplots and merging into single figure
        pick_bar = create_bar( list(avg_sal.PercentDrafted),list(avg_sal.Player))
        sal_bar = create_bar(list(avg_sal.AverageSalary), list(avg_sal.Player))
        fig = create_fig_layout(pick_bar, sal_bar)

        # show drafted players
        subset_sal = pick_df[pick_df.Salary > 0]

        return fig, subset_sal.to_dict('records'), my_team_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)
