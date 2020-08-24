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
table_vers = 'Version1'
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
league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
league_info['num_teams'] = 12
league_info['initial_cap'] = 293
league_info['salary_cap'] = 293

pick_dict = {}
for p in d.index:
    pick_dict[p]=0

#------------------
# For Beta Keepers
#------------------

#%%

# input information for players and their associated salaries selected by other teams
qb_picked = {
    'Lamar Jackson': 0,
    'Patrick Mahomes': 0,
    'Deshaun Watson': 0,
    'Kyler Murray': 0,
    'Dak Prescott': 0,
    'Russell Wilson': 0,
    'Josh Allen': 0,
    'Drew Brees': 0,
    'Tom Brady': 0,
    'Matt Ryan': 0,
    'Aaron Rodgers': 0,
    'Carson Wentz': 0,
    'Matthew Stafford': 0,
    'Daniel Jones': 0,
    'Mitchell Trubisky': 0,
    'Sam Darnold': 0,
    'Baker Mayfield': 0,
    'Drew Lock': 0,
    'Dwayne Haskins': 0,
    'Kyle Allen': 0,
    'Mason Rudolph': 0,
    'Jared Goff': 0,
    'Kirk Cousins': 0,
    'Jameis Winston': 0,
    'Philip Rivers': 0,
    'Andy Dalton': 0,
    'Ryan Tannehill': 0,
    'Cam Newton': 0,
    'Derek Carr': 0,
    'Ryan Fitzpatrick': 0,
    'Jimmy Garoppolo': 0,
    'Marcus Mariota': 0,
    'Teddy Bridgewater': 0,
    'Jacoby Brissett': 0,
    'Nick Foles': 0,
    'Case Keenum': 0
}

rb_picked = {
    'Christian McCaffrey': 1,
    'Saquon Barkley': 1,
    'Ezekiel Elliott': 0,
    'Dalvin Cook': 0,
    'Alvin Kamara': 0,
    'Derrick Henry': 0,
    'Miles Sanders': 0,
    'Clyde Edwards-Helaire': 0,
    'Kenyan Drake': 0,
    'Nick Chubb': 0,
    'Aaron Jones': 0,
    'Joe Mixon': 0,
    'Austin Ekeler': 0,
    'Leonard Fournette': 0,
    'Chris Carson': 0,
    'David Johnson': 0,
    'Todd Gurley': 0,
    "Le'Veon Bell": 0,
    'Melvin Gordon': 0,
    'James Conner': 0,
    'Jonathan Taylor': 0,
    'Devin Singletary': 0,
    "D'Andre Swift": 0,
    'Kareem Hunt': 0,
    'Cam Akers': 0,
    'David Montgomery': 0,
    'Raheem Mostert': 0,
    'Tarik Cohen': 0,
    'Mark Ingram': 0,
    'J.K. Dobbins': 0,
    'Kerryon Johnson': 0,
    'Phillip Lindsay': 0,
    'Darrell Henderson': 0,
    'James White': 0,
    'Jordan Howard': 0,
    'Matt Breida': 0,
    'Zack Moss': 0,
    'Tevin Coleman': 0,
    'Alexander Mattison': 0,
    'Latavius Murray': 0,
    "Ke'Shawn Vaughn": 0,
    'AJ Dillon': 0,
    'Joshua Kelley': 0,
    'Eno Benjamin': 0,
    'DeeJay Dallas': 0,
    'Sony Michel': 0,
    'Dare Ogunbowale': 0,
    'Nyheim Hines': 0,
    'Boston Scott': 0,
    'Jaylen Samuels': 0,
    'Chase Edmonds': 0,
    'Ryquell Armstead': 0,
    'Royce Freeman': 0,
    'Rashaad Penny': 0,
    'Tony Pollard': 0,
    'Malcolm Brown': 0,
    'Justice Hill': 0,
    'Gus Edwards': 0,
    'Mike Boone': 0,
    'Darwin Thompson': 0,
    'Reggie Bonnafon': 0,
    'Damien Williams': 0,
    'Marlon Mack': 0,
    'Duke Johnson': 0,
    'Adrian Peterson': 0,
    'Carlos Hyde': 0,
    'Brian Hill': 0,
    'Jamaal Williams': 0,
    'Chris Thompson': 0,
    'Jerick McKinnon': 0,
    'Rex Burkhead': 0,
    'Frank Gore': 0,
    'DeAndre Washington': 0,
    'Giovani Bernard': 0,
    'Jalen Richard': 0,
    'Dion Lewis': 0,
    'Wayne Gallman': 0
}

wr_picked = {
    'Michael Thomas': 0,
    'Davante Adams': 0,
    'DeAndre Hopkins': 0,
    'Tyreek Hill': 0,
    'Julio Jones': 0,
    'Chris Godwin': 0,
    'Mike Evans': 0,
    'Kenny Golladay': 0,
    'D.J. Moore': 0,
    'Allen Robinson': 0,
    'Adam Thielen': 0,
    'JuJu Smith-Schuster': 0,
    'Amari Cooper': 0,
    'Courtland Sutton': 0,
    'Calvin Ridley': 0,
    'A.J. Brown': 0,
    'Robert Woods': 0,
    'Tyler Lockett': 0,
    'Cooper Kupp': 0,
    'Keenan Allen': 0,
    'T.Y. Hilton': 0,
    'Terry McLaurin': 0,
    'Jarvis Landry': 0,
    'Tyler Boyd': 0,
    'DeVante Parker': 0,
    'Marquise Brown': 0,
    'Michael Gallup': 0,
    'Stefon Diggs': 0,
    'Deebo Samuel': 0,
    'Marvin Jones': 0,
    'Diontae Johnson': 0,
    'Brandin Cooks': 0,
    'Jamison Crowder': 0,
    'Julian Edelman': 0,
    'Christian Kirk': 0,
    'Preston Williams': 0,
    'Sterling Shepard': 0,
    'Golden Tate': 0,
    'Robby Anderson': 0,
    'Darius Slayton': 0,
    'John Brown': 0,
    'Jerry Jeudy': 0,
    'Justin Jefferson': 0,
    'Mecole Hardman': 0,
    'Mike Williams': 0,
    'Emmanuel Sanders': 0,
    'Curtis Samuel': 0,
    'CeeDee Lamb': 0,
    'Tee Higgins': 0,
    'Quintez Cephus': 0,
    'Jalen Reagor': 0,
    'Denzel Mims': 0,
    'Brandon Aiyuk': 0,
    'Tyler Johnson': 0,
    'KJ Hamler': 0,
    'Devin Duvernay': 0,
    'Chase Claypool': 0,
    'Van Jefferson': 0,
    'Anthony Miller': 0,
    'James Washington': 0,
    'Allen Lazard': 0,
    'Steven Sims': 0,
    'Russell Gage': 0,
    'Auden Tate': 0,
    'Zach Pascal': 0,
    'Marquez Valdes-Scantling': 0,
    "Tre'Quan Smith": 0,
    'Kendrick Bourne': 0,
    'Kelvin Harmon': 0,
    'Miles Boykin': 0,
    'JJ Arcega-Whiteside': 0,
    'Andy Isabella': 0,
    'Olabisi Johnson': 0,
    'Dante Pettis': 0,
    'Jakobi Meyers': 0,
    'Keke Coutee': 0,
    'Trey Quinn': 0,
    'Justin Watson': 0,
    'Scott Miller': 0,
    'Will Fuller': 0,
    'Larry Fitzgerald': 0,
    'Sammy Watkins': 0,
    'Alshon Jeffery': 0,
    'Dede Westbrook': 0,
    'Breshad Perriman': 0,
    'John Ross': 0,
    'Randall Cobb': 0,
    'Corey Davis': 0,
    'Cole Beasley': 0,
    'Josh Reynolds': 0,
    'Mohamed Sanu': 0,
    'Chris Conley': 0,
    'Kenny Stills': 0,
    'Danny Amendola': 0,
    'Demarcus Robinson': 0,
    'Adam Humphries': 0,
    'Albert Wilson': 0,
    'Willie Snead': 0,
    'Phillip Dorsett': 0,
    'Cordarrelle Patterson': 0,
    'Marquise Goodwin': 0,
    'Allen Hurns': 0,
    'Tajae Sharpe': 0,
    'Rashard Higgins': 0
}

te_picked = {
    'Travis Kelce': 0,
    'George Kittle': 0,
    'Mark Andrews': 0,
    'Zach Ertz': 0,
    'Evan Engram': 0,
    'Tyler Higbee': 0,
    'Noah Fant': 0,
    'Hayden Hurst': 0,
    'Hunter Henry': 0,
    'Jared Cook': 0,
    'Dallas Goedert': 0,
    'Mike Gesicki': 0,
    'T.J. Hockenson': 0,
    'Jonnu Smith': 0,
    'Darren Fells': 0,
    'O.J. Howard': 0,
    'Dawson Knox': 0,
    'Blake Jarwin': 0,
    'Gerald Everett': 0,
    'Ian Thomas': 0,
    'Jacob Hollister': 0,
    'Kaden Smith': 0,
    'Jordan Akins': 0,
    'Logan Thomas': 0,
    'Jeremy Sprinkle': 0,
    'Ricky Seals-Jones': 0,
    'Drew Sample': 0,
    'Austin Hooper': 0,
    'Jack Doyle': 0,
    'Greg Olsen': 0,
    'Eric Ebron': 0,
    'Jimmy Graham': 0,
    'Kyle Rudolph': 0,
    'Tyler Eifert': 0,
    'Cameron Brate': 0,
    'Nick Boyle': 0,
    'C.J. Uzomah': 0,
    'Vance McDonald': 0,
    'Ryan Griffin': 0,
    'Maxx Williams': 0
}

to_drop = {}
to_drop['players'] = []
to_drop['salaries'] = []
for posit in ['qb', 'rb', 'wr', 'te']:
    for p, s in globals()[f'{posit}_picked'].items():
        if s > 0:
            to_drop['players'].append(p)
            to_drop['salaries'].append(s)

# input information for players and their associated salaries selected by your team
to_add = {}
to_add['players'] = []
to_add['salaries'] = []
team_picks = {
    'Michael Thomas': 1
    # 'Michael Thomas': 1,
    # 'Alvin Kamara': 50, 
    # 'Nick Chubb': 33,
    # 'Michael Thomas': 105,
    # 'Mark Andrews': 40,
    # 'Jonathan Taylor': 31,
    # 'Cam Akers': 20,
    # 'Deshaun Watson': 11,
    # 'CeeDee Lamb': 3
}

for p, s in team_picks.items():
    to_add['players'].append(p)
    to_add['salaries'].append(s)

results = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
avg_sal = sim.show_most_selected(to_add, iterations, num_show=30)
avg_sal.style.bar_excel(color_positive='#5FBA7D')


# %%
team_pts = d[(d.index.isin(team_picks.keys())) & (d.pos!='eFLEX')].drop(['pos', 'salary'], axis=1).sum(axis=0)
team_pts = team_pts * (11/16) + 150 + 275
print(f'20th Percentile: {np.percentile(team_pts, 20)}')
print(f'Team Mean: {team_pts.mean()}')
print(f'80th Percentile: {np.percentile(team_pts, 80)}')

plt.figure(figsize(8, 8))
sns.distplot(team_pts, hist = True, kde = True, bins = 19,
                hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
            kde_kws = {'linewidth' : 4},
            label = 'Estimated Dist.');
# %%