#%%
import datetime as dt
import pandas as pd
import numpy as np
from ff.db_operations import DataManage
from ff import general
from scipy.interpolate import UnivariateSpline

#==========
# General Setting
#==========

# set the root path and database management object
root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


def calc_vor(df, year_col, pos_col, points_col, replacement_ranks, suffix):

    # Locate the replacement player's points for each season×position
    repl_pts = (
        df.sort_values([year_col, pos_col, points_col], ascending=[True, True, False])
          .groupby([year_col, pos_col])
          .apply(
              lambda g: g.iloc[
                  min(replacement_ranks.get(g.name[1], len(g)) - 1, len(g) - 1)
              ][points_col]
          )
          .rename("rep_pts")
          .reset_index()
    )

    df = df.merge(repl_pts, on=[year_col, pos_col])
    df[f"VOR_{suffix}"] = df[points_col] - df["rep_pts"]
    return df.drop('rep_pts', axis=1, errors='ignore')

def build_adp_value_table(
    df: pd.DataFrame,
    adp_col: str = "avg_pick",          # or "avg_pick_log"
    points_col: str = "y_act",
    pos_col: str = "pos",
    year_col: str = "year",
    max_pick: int = 300,    
                replacement_ranks = None,            # ignore ADP > max_pick
    spline_s: float = 1000,              # smoothing penalty (tune if curve too jagged / too flat)
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: adp_slot | cost
    ready to feed into a knapsack optimiser.
    """

    df = calc_vor(df, year_col, pos_col, points_col, replacement_ranks, 'ADP')

    # ── 2.  Bin by ADP slot & average VOR ─────────────────────────────────
    df["adp_slot"] = df[adp_col].round().clip(1, max_pick).astype(int)
    slot_mean = (
        df.groupby("adp_slot", as_index=False)["VOR_ADP"]
          .median()
          .sort_values("adp_slot")
    )

    # ── 3.  Smooth (optional) ─────────────────────────────────────────────
    x, y = slot_mean["adp_slot"].to_numpy(), slot_mean["VOR_ADP"].to_numpy()
    if len(x) > 4:                      # need ≥5 points for a stable spline
        spline = UnivariateSpline(x, y, s=spline_s)
        y = spline(x)
    slot_mean["vor_smooth"] = y

    # ── 4.  Enforce monotone ↓ with cumulative minimum (LEFT → RIGHT) ─────
    slot_mean["vor_monotone"] = np.minimum.accumulate(slot_mean["vor_smooth"])

    return df, slot_mean[["adp_slot", 'vor_monotone', 'vor_smooth']]

def plot_spline(df, chart_df):
    import matplotlib.pyplot as plt    

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(df["adp_slot"], df["VOR_ADP"], label="VOR", linewidths=0.5, alpha=0.5)
    ax.plot(chart_df["adp_slot"], chart_df["vor_smooth"], label="VOR (Monotone)", linestyle="--", color="orange")

    ax.set_title("Weekly Fantasy Points")
    ax.set_xlabel("Week")
    ax.set_ylabel("Points")
    ax.legend()
    ax.grid(True)
    plt.show()

def get_this_year_vor(df, year=2025):
    this_year = df.loc[df.year==year].copy()
    this_year.fillna({'vor_monotone': this_year.vor_monotone.min(), 'cost': 0}, inplace=True)
    this_year = this_year[['player', 'avg_pick', 'vor_monotone']]
    this_year.columns = ['Player', 'ADP_NFFC', 'VOR_ADP']
    return this_year.sort_values(by='ADP_NFFC', ascending=True).reset_index(drop=True)

league = 'dk'
pred_vers = 'final_ensemble'
year = 2025
replacement_ranks = {'QB': 12, 'RB': 28, 'WR': 44, 'TE': 12}

num_teams = 12
num_rounds = 25
max_pick = num_teams * num_rounds  # ignore ADP > max_pick

#%%

df = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    df_pos = dm.read(f'''SELECT player ,
                                pos, 
                                year,
                                pick_nffc avg_pick,
                                y_act
                     FROM {pos}_{year}_ProjOnly 
                     WHERE pos='{pos}' ''', f'Model_Inputs')
    

    df = pd.concat([df, df_pos], ignore_index=True)

df = df[df.avg_pick <= max_pick].reset_index(drop=True).copy()

df, chart_df = build_adp_value_table(df, max_pick=max_pick, replacement_ranks=replacement_ranks,)
df = df.merge(chart_df, on='adp_slot', how='left')
plot_spline(df, chart_df)
vor_adp = get_this_year_vor(df, year=2025)

#%%


df = pd.DataFrame()
cols = {'player': 'Player', 
        'year': 'Year',
        'pos': 'Position',
        'team': 'Team', 
        'fpros_pos_rank': 'Position_Rank', 
        'avg_proj_points': 'Consensus_Projected_Points',
        'year_exp': 'Years_of_Experience', 
        'avg_proj_pass_yds': 'Consensus_Passing_Yards',
        'avg_proj_pass_td': 'Consensus_Passing_Touchdowns',
        'avg_proj_rush_yds': 'Consensus_Rushing_Yards',
        'avg_proj_rush_td': 'Consensus_Rushing_Touchdowns',
        'avg_proj_rec': 'Consensus_Receptions',
        'avg_proj_rec_yds': 'Consensus_Receiving_Yards',
        'avg_rec_td': 'Consensus_Receiving_Touchdowns',
        'avg_proj_points_exp_diff': 'Projected_Points_Vs_Avg_Experience'
    }

for pos in ['QB', 'RB', 'WR', 'TE']:
    df_pos = dm.read(f'''SELECT *
                         FROM {pos}_{year}_ProjOnly 
                         WHERE year={year}
                               AND pos='{pos}' ''', f'Model_Inputs')
    
    df_pos = df_pos[[c for c in df_pos.columns if c in cols.keys()]]
    df_pos = df_pos.rename(columns=cols)
    df_pos.Consensus_Projected_Points = df_pos.Consensus_Projected_Points/16
    df = pd.concat([df, df_pos], ignore_index=True).fillna(0)

pred = dm.read(f'''SELECT player Player, 
                          pred_fp_per_game Predicted_PPG_Mark, 
                          pred_fp_per_game_ny Predicted_PPG_Next_Year_Mark,
                          pred_prob_upside Predicted_Breakout_Mark,
                          pred_prob_top Predicted_TopValue_Mark,
                          max_score Max_Score_Mark
                FROM Final_Predictions
                WHERE year={year}
                        AND dataset='{pred_vers}'
                        AND version='{league}'
            ''', 'Simulation')

df = pd.merge(df, vor_adp, on='Player', how='left')
df = pd.merge(df, pred, on=['Player'], how='inner')
df = df.sort_values(by='ADP_NFFC', ascending=True)

df = calc_vor(df, 'Year', 'Position', 'Predicted_PPG_Mark', replacement_ranks, 'Current_Year')
df = calc_vor(df, 'Year', 'Position', 'Consensus_Projected_Points', replacement_ranks, 'Consensus')
df = calc_vor(df, 'Year', 'Position', 'Predicted_TopValue_Mark', replacement_ranks, 'Top_Value')

df.VOR_Top_Value = df.VOR_Top_Value*5
df['VOR_Average'] = df[[c for c in df.columns if 'VOR_' in c]].mean(axis=1)
df = df.round(3)

df = df.sort_values(by='VOR_Average', ascending=False).reset_index(drop=True)
df.iloc[:50]

#%%
df.to_csv(f'{root_path}/Data/Projections/{year}_Projections_{league}_VOR.csv', index=False)

# %%

year = 2025
adp_league = 'nffc'

adps = dm.read(f'''SELECT player,
                          pos,
                          year,
                          avg(pick_nffc) avg_pick,
                          avg(min_pick) min_pick,
                          avg(max_pick) max_pick
                   FROM NFFC_ADP
                   WHERE year = {year}
                   GROUP BY player, pos, year
                ''', f'Season_Stats_New')
adps['std_dev'] = (adps['max_pick'] - adps['min_pick']) / 5
adps.sort_values(by='avg_pick', ascending=True).iloc[:50]
adps['league'] = adp_league

df = df.rename(columns={'Player': 'player'})
adps = pd.merge(adps, df[['player', 'Years_of_Experience']], on='player', how='left')

adps = adps[['player', 'Years_of_Experience', 'avg_pick', 'year', 'league', 'std_dev', 'min_pick', 'max_pick']]

dm.delete_from_db('Simulation', 'Avg_ADPs', f"year={year} AND league='{adp_league}'", create_backup=True)
dm.write_to_db(adps, 'Simulation', 'Avg_ADPs', 'append')

#%%

import shutil

src = f'{root_path}/Data/Databases/Simulation.sqlite3'
dst = f'/Users/borys/OneDrive/Documents/Github/Fantasy_Football_Snake/app/Simulation.sqlite3'
shutil.copyfile(src, dst)

# %%
# %%


# %%
