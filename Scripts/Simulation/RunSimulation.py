# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # User Inputs

# +
#==============
# Load Packages
#==============

# jupyter specifications
from IPython.core.interactiveshell import InteractiveShell
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from simulation import *
pd.options.mode.chained_assignment = None 

#===============
# Settings and User Inputs
#===============

#--------
# Database Login Info
#--------

# postgres login information
pg_log = {
    'USER': 'postgres',
    'PASSWORD': 'Ctdim#1bf!!!!!',
    'HOST': 'localhost',
    'PORT': '5432', 
    'DATABASE_NAME': 'fantasyfootball'
}

# create engine for connecting to database
engine = create_engine('postgres+psycopg2://{}:{}@{}:{}/{}'.format(pg_log['USER'], pg_log['PASSWORD'], pg_log['HOST'],
                                                                   pg_log['PORT'], pg_log['DATABASE_NAME']))

# specify schema and table to write out intermediate results
table_info = {
    'engine': engine,
    'schema': 'websitedev',
}

np.random.seed(123)

#--------
# League Settings
#--------

# set year
year = 2019

# set the number of simulation iterations
iterations = 1000

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


# +
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
# -

# # Pull in Data

# instantiate simulation class and add salary information to data
sim = FootballSimulation(pts_dict, table_info, 2019, iterations)
d = sim.return_data()

plt.style.use('classic')
plt.style.use('fivethirtyeight')
plt.style.use('seaborn-ticks')
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.linewidth'] = .1
plt.rcParams['grid.color'] = '0.4'
plt.rcParams['font.size'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.labelcolor'] = '0'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['xtick.major.top'] = False
plt.rcParams['ytick.major.right'] = False
plt.rcParams['figure.figsize'] = (6.4, 3.6)
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['axes.titlesize'] ='large'

# # Point Distributions

# +
names = ['Alvin Kamara', 'Nick Chubb', 'DeAndre Hopkins', 'JuJu Smith-Schuster', 
         'D.J. Moore', 'Jarvis Landry', 'Jared Cook']

pts_scored = [194, 247, 230, 95, 196, 189, 143]
pts_scored = [16/15*p for p in pts_scored]

for n in zip(names, pts_scored):
    d.loc[d.index == n[0]].iloc[0, 1:999].plot.hist(bins=25, linewidth=0.5, edgecolor='black', title=n[0]);
    plt.plot([n[1], n[1]], [0, 120])
    plt.savefig('output_' + n[0] + '.pdf')
    plt.show()
# -

# # Salary Distributions

sal = d[d.index=='Alvin Kamara'].iloc[0, -1]
plt.hist(sal*(1+(skewnorm.rvs(5, size=iterations*2)*.1)), bins=25);

# +
# set league information, included position requirements, number of teams, and salary cap
league_info = {}
league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
league_info['num_teams'] = 12
league_info['initial_cap'] = 293
league_info['salary_cap'] = 293

#------------------
# For Beta Keepers
#------------------

# input information for players and their associated salaries selected by other teams
to_drop = {}
to_drop['players'] = ['Michael Thomas', 'Damien Williams',
                      'Mike Evans', 'Robert Woods',
                      'Adam Thielen', 'Ezekiel Elliott', 'Davante Adams', 'Adrian Peterson',
                      'James Conner', 'Joe Mixon', 'Devonta Freeman', 'Saquon Barkley',
                      'Tyreek Hill', 'Chris Godwin', 'Patrick Mahomes', 'Tyler Lockett', 
                      'Christian McCaffrey', 'Travis Kelce', 'Tarik Cohen', 'Sony Michel',
                      'Kyle Rudolph', 'JuJu Smith-Schuster']
#                       'Dalvin Cook', 'LeSean McCoy',  'Zach Ertz', 'David Johnson',
#                       "Le'Veon Bell", 'Melvin Gordon', 'Odell Beckham', 'Amari Cooper',
#                       'Antonio Brown', 'Leonard Fournette', 'Julio Jones', 'Deshaun Watson',
#                        'Lamar Miller', 'Todd Gurley', 'Keenan Allen', 'Chris Carson',
#                       'A.J. Green', 'Aaron Rodgers', 'Derrius Guice', 'Curtis Samuel',
#                        'Derrick Henry', 'George Kittle', 'T.Y. Hilton', 'Kerryon Johnson',
#                        'Stefon Diggs', 'Cooper Kupp', 'Brandin Cooks', 'Marlon Mack',
#                       'Kenyan Drake', 'Tevin Coleman', 'Julian Edelman', 'Josh Jacobs',
#                       'James White', 'Mark Ingram', 'Aaron Jones', 'Kenny Golladay', 
#                        'Drew Brees', 'Matt Ryan', 'Phillip Lindsay', 'Royce Freeman', 
#                        'Tyler Boyd', 'Austin Hooper', 'Alshon Jeffery', 'Kareem Hunt',
#                        'Miles Sanders', 'Austin Ekeler'
#                     ]

to_drop['salaries'] = [
                        55, 16, 
                       65, 19,  
                       46, 114, 71, 11, 
                       11, 67, 11, 111, 
                       36, 11, 11, 11, 
                       77, 46, 12, 15,
                        1, 78
                        ]
#                        70, 13, 48, 94,
#                        96, 86, 95, 58,
#                        68, 72, 90, 22,
#                        12, 84, 67, 45,
#                         29, 10, 12, 22,
#                        52, 38, 52, 74,
#                         55, 23, 55, 42,
#                         13, 20, 55, 57,
#                        22, 36, 54, 40,
#                         8, 12, 35, 5, 
#                         23, 8, 20, 12,
#                         21, 12
#                       ]

# input information for players and their associated salaries selected by your team
to_add = {}
to_add['players'] = ['Alvin Kamara', 'Nick Chubb', 'DeAndre Hopkins', 'Jared Cook',
                     'Dalvin Cook', 'Lamar Jackson', 'Cooper Kupp', 'Tyler Boyd'
                     ]
to_add['salaries'] = [35, 18, 98, 6, 78, 1, 32, 25]

# to_drop = {}
# to_drop['players'] = []
# to_drop['salaries'] = []


results = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
avg_sal = sim.show_most_selected(to_add, iterations, num_show=30)
avg_sal.style.bar_excel(color_positive='#5FBA7D')
# -


