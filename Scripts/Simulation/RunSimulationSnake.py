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
from zSim_Helper_Snake import *
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
table_vers = 'Version4'
set_year = 2020
league='nffc'
initial_pick = 1

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

# set league information, included position requirements, number of teams, and salary cap
initial_pick = 1

league_info = {}
league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
league_info['num_teams'] = 12
league_info['initial_cap'] = 400
league_info['salary_cap'] = 400

flex_pos = ['RB', 'WR', 'TE']

# creating dictionary containing point values for each position
pts_dict = {}
pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_rec_td, int_pts, sacks]
pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['WR'] = [rec_yd_per_pt, ppr, rush_rec_td]
pts_dict['TE'] = [rec_yd_per_pt, ppr, rush_rec_td]

# instantiate simulation class and add salary information to data
sim = FootballSimulation(pts_dict, conn_sim, table_vers, set_year, league, iterations, initial_pick)

# return the data and set up dataframe for proportion of salary across position
d = sim.return_data()
d = d.rename(columns={'pos': 'Position', 'salary': 'Salary'})
d.Position = d.Position.apply(lambda x: x[1:])

my_picks = sim.return_picks()

keepers = {}

#%%

################################################

#==========================
# Start of Dash App
#==========================

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


#==================
# Expected Points Functions
#==================

def csv_check(df, name):
    return df.to_csv(f'c:/Users/mborysia/Desktop/FF_App/{name}.csv', index=False)

#==========================
# Submit Data Entry
#==========================

# submit button
red_button_style = {'background-color': 'red',
                    'color': 'white',
                    'height': '20',
                    'width': '100%',
                    'fontSize': 16}
submit_button = html.Button(id='submit-button-state', n_clicks=0, 
                            children='Refresh Top Picks', style=red_button_style)


#============================
# Build out Dash Tables
#============================

main_color = '40, 110, 132'
main_color_rgb = f'rgb({main_color})'
main_color_rgba = f'rgba({main_color}, 0.8)'

#--------------
# Set up dataframe and Data Table for My Team
#--------------

# get the player ADP info and join back to the dataset to get a unique list of player info
player_adp = pd.read_sql_query('''SELECT DISTINCT player, adp from PickProb''', conn_sim)
d = pd.merge(d, player_adp, on='player').set_index('player')

# pull out the total number of players in the dataset to calculate picks used so far
total_player_num = len(d[d.Position!='FLEX'])

# create a table of Pos | Player | ADP | Is Selected | My Team
player_list = []
for pl, row in d.sort_values(by='adp')[['Position', 'adp']].iterrows():
    if row.Position != 'FLEX':
        player_list.append([row.Position, pl, row.adp,  'No'])
d = d.drop('adp', axis=1)

pick_df = pd.DataFrame(player_list, columns=['Position', 'Player', 'ADP', 'Is Selected'])
pick_df['My Team'] = 'No'


# set up all players drafted DataTable
drafted_player_table =  dash_table.DataTable(
                            id='draft-results-table',

                            columns=[{'id': c, 'name': c, 'editable': False} for c in pick_df.columns if c not in ('Is Selected', 'My Team')] +
                                     [{'id': c, 'name': c,'presentation': 'dropdown', 'editable': (c in ('Is Selected','My Team'))} for c in pick_df.columns if c in ('Is Selected', 'My Team')],
                            data=pick_df.to_dict('records'),
                            filter_action='native',
                            sort_action='native',
                            style_table={
                                            'height': '800px',
                                            'overflowY': 'auto',
                                        },
                            style_cell={'textAlign': 'left', 'fontSize':14, 'font-family':'sans-serif'},
                            dropdown={
                                   'Is Selected': {
                                        'options': [
                                            {'label': i, 'value': i} for i in ['Yes', 'No']
                                        ]
                                    },
                                    'My Team': {
                                        'options': [
                                            {'label': i, 'value': i} for i in ['Yes', 'No']
                                        ]
                                    }
                                    # 'style': {'backgroundColor': 'white', 'color': 'black'}
                                    },
                            style_data_conditional=[{
                                        'if': {'column_editable': False},
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                    }],
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    }
                        )


#--------------
# Set up dataframe and Data Table for My Team
#--------------

my_team_list = []
for k, v in league_info['pos_require'].items():
    for i in range(v):
        my_team_list.append([k, None, 0])
my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'ADP'])

 # set up my team  drafted DataTable
my_team_table =  dash_table.DataTable(
                            id='my-team-table',
                            columns=[{"name": i, "id": i, 'editable': False} for i in my_team_df.columns],
                            data=my_team_df.to_dict('records'),
                            style_cell={'textAlign': 'left', 'fontSize':14, 'font-family':'sans-serif'},
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    },
                            style_data_conditional=[{
                                        'if': {'column_editable': False},
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                    }],
                        )

#--------------
# Set up dataframe and Data Table for Team Info
#--------------

team_info = pd.DataFrame({'Mean Points': [None],
                          'Current Pick': [1],
                          'My Next Pick': [my_picks[0]]})

team_info_table =  dash_table.DataTable(
                            id='team-info-table',
                            columns=[{"name": i, "id": i, 'editable': False} for i in team_info.columns],
                            data=team_info.to_dict('records'),
                            style_cell={'textAlign': 'center', 'fontSize':14, 'font-family':'sans-serif'},
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    },
                            style_data_conditional=[{
                                        'if': {'column_editable': False},
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                    }],
                        )

#==========================
# Plotting Functions
#==========================

def create_bar(x_val, y_val, orient='h', color_str=main_color_rgba, text=None):
    '''
    Function to create a horizontal bar chart
    '''
    marker_set = dict(color=color_str, line=dict(color=color_str, width=1))
    return go.Bar(x=x_val, y=y_val, marker=marker_set, orientation=orient, text=text, showlegend=False)


def create_fig_layout(fig1, fig2):
    '''
    Function to combine bar charts into a single figure
    '''
    fig = go.Figure(data=[fig1, fig2])
    
    # Change the bar mode
    fig.update_layout(barmode='group', autosize=True, height=800, 
                      margin=dict(l=0, r=25, b=0, t=15, pad=0),
                      uniformtext_minsize=25, uniformtext_mode='hide')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    return fig

def create_hist(team_dist):
    
    hist_data = [team_dist.projection.values]
    group_labels = [''] # name of the dataset

    fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=10, show_rug=False, colors=[main_color_rgba])
    fig_hist.update_layout(autosize=True, height=200, margin=dict(l=0, r=0, b=0, t=0, pad=0), showlegend=False)

    return fig_hist


#--------------
# Plot Creation
#--------------

# bar chart creation
bar_gr = dcc.Graph(id='draft-results-graph')

# histogram creation
hist_gr = dcc.Graph(id='team-points-graph')


#============================
# Build out App Layout
#============================

app.layout = html.Div([

     html.Div([
         html.Div([
            html.H5("Enter Draft Pick Information"),
            drafted_player_table,
            ], className="four columns"),

            html.Div([
                html.H5('My Team'),
                my_team_table, html.Hr(),
                submit_button, html.Hr(),
                html.H5('Team Information'),
                team_info_table,
                html.Hr(),
                hist_gr
            ], className='four columns'),

            html.Div([
                html.H5('Recommended Picks'),
                bar_gr
            ], className="four columns")
       
       ], className="row2") ,        
         
])


#============================
# Update Functions
#============================

def update_to_drop(df):
    '''
    INPUT: Dataframe containing players + salaries to be dropped from selection

    OUTPUT: Dictionary containing dropped player + salaries for passing into simulation
    '''
    to_drop = {}
    to_drop['players'] = []
    for _, row in df.iterrows():
        to_drop['players'].append(row.Player)

    return to_drop


def update_to_add(df):
    '''
    INPUT: Dataframe containing players + salaries to be added to my team

    OUTPUT: Dictionary containing my team player + salaries for passing into simulation
    '''
    to_add = {}
    to_add['players'] = []
    for _, row in df.iterrows():
        if row.Player is not None and row.Player!='' and row['Is Selected'] == 'Yes':
            to_add['players'].append(row.Player)
    
    return to_add


def team_fill(df, df2):
    '''
    INPUT: df: blank team template to be filled in with chosen players
           df2: chosen players dataframe

    OUTPUT: Dataframe filled in with selected player information
    '''
    # loop through chosen players dataframe
    for _, row in df2.iterrows():

        # pull out the current position and find min position index not filled (if any)
        cur_pos = row.Position
        min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()

        # if position still needs to be filled, fill it
        if min_idx is not np.nan:
            df.loc[min_idx, ['Player', 'ADP']] = [row.Player, row.ADP]

        # if normal positions filled, fill in the FLEX if applicable
        elif cur_pos in ('RB', 'WR', 'TE'):
            cur_pos = 'FLEX'
            min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()
            if min_idx is not np.nan:
                df.loc[min_idx, ['Player', 'ADP']] = [row.Player, row.ADP]

            # otherwise, fill in the Bench
            else:
                bench = pd.DataFrame(['Bench', row.Player, row.ADP]).T
                bench.columns = ['Position', 'Player', 'ADP']
                df = pd.concat([df, bench], axis=0)
    return df



@app.callback([Output('draft-results-graph', 'figure'),
               Output('team-points-graph', 'figure'),
               Output('my-team-table', 'data'),
               Output('team-info-table', 'data')],
              [Input('submit-button-state', 'n_clicks')],
              [State('draft-results-table', 'data'),
               State('draft-results-table', 'columns')]
)
def update_output(n_clicks, drafted_data, drafted_columns):

    # get the list of drafted players
    drafted_df = pd.DataFrame(drafted_data, columns=[c['name'] for c in drafted_columns])

    # create a template of all team positions and players current selected for my team
    my_team_template = my_team_df.copy()
    my_team_select = drafted_df[drafted_df['My Team']=='Yes'].reset_index(drop=True)
    my_team_update = team_fill(my_team_template.copy(), my_team_select)

    # create a dataset of all other players that have been drafted
    drafted_df = drafted_df[(drafted_df['My Team']=='No') & (drafted_df['Is Selected']=='Yes')].reset_index(drop=True)
    
    # get lists of to_drop and to_add players and remaining salary
    to_drop = update_to_drop(drafted_df)
    to_add = update_to_add(my_team_select)
    next_pick = len(to_add['players']) + len(to_drop['players']) + 1

    # run the simulation
    result = None
    prob_filter = 1
    while result is None and prob_filter < 20:
        try:
            result = sim.run_simulation(league_info, to_drop, to_add, pick_num=next_pick, 
                                        prob_filter=prob_filter, iterations=iterations)
        except:
            prob_filter += 1

    
    # get the results dataframe structured
    result.columns = ['Player', 'PercentDrafted', 'ProbAvail', '_']
    result = result[['Player', 'PercentDrafted', 'ProbAvail']]
    result.PercentDrafted = result.PercentDrafted / iterations
    result.ProbAvail = result.ProbAvail / 1000
    result = result[result.PercentDrafted > 0]
    

    # Creating two subplots and merging into single figure
    (pl, pc_dr, pr_av) = result.Player, result.PercentDrafted, result.ProbAvail
    pick_bar = create_bar(pc_dr, pl, text=pc_dr)
    prob_bar = create_bar(pr_av, pl, color_str='rgba(250, 190, 88, 1)', text=pr_av)
    gr_fig = create_fig_layout(prob_bar, pick_bar)

    # # histogram creation
    # hist_fig = create_hist(cur_team_dist)
    hist_fig = dcc.Graph()

    # update team information table
    team_info_update = pd.DataFrame({'Mean Points': [100], 
                                     'Current Pick': [next_pick],
                                     'My Next Pick': [[p for p in my_picks if p>=next_pick][0]]})
    
    # # save out csv of status
    # drafted_df.to_csv('c:/Users/mborysia/Desktop/Status_Save.csv', index=False)

    return gr_fig, hist_fig, my_team_update.to_dict('records'), team_info_update.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=False)
