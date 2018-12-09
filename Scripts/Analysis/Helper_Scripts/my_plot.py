
class PrettyPlot():
    def __init__(self, plt):
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