#%%
import os
import hashlib
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add Scripts directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YEAR, LEAGUE, PRED_VERSION, get_scoring_dict

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc


#==========
# Settings
#==========

POSITIONS = ["QB", "RB", "WR", "TE"]
WEEK_COUNT = 16
WEEKS = list(range(1, WEEK_COUNT + 1))

TEMPLATE_SEASON_MIN = 2008
MIN_TEMPLATE_POOL_SIZE = 40
PROJECTION_BUCKETS = 10
POOL_RANDOM_SEED = 20260702
VALIDATION_CURRENT_OR_NEXT_YEAR = "current"

TEMPLATE_TABLE = "Best_Ball_Weekly_Templates"
POOL_TABLE = "Best_Ball_Weekly_Template_Pools"
POOL_SUMMARY_TABLE = "Best_Ball_Weekly_Pool_Summary"
PLAYER_MAP_TABLE = "Best_Ball_Weekly_Player_Map"
TEMPLATE_AUDIT_TABLE = "Best_Ball_Weekly_Template_Audit"
PLAYER_POOL_AUDIT_TABLE = "Best_Ball_Weekly_Player_Pool_Audit"
BUCKET_AUDIT_TABLE = "Best_Ball_Weekly_Bucket_Audit"

LOW_ACTIVE_GAME_THRESHOLD = 2
HIGH_ZERO_TEMPLATE_POOL_SHARE = 0.10
HIGH_LOW_ACTIVE_TEMPLATE_POOL_SHARE = 0.20


#==========
# Paths / DB
#==========

root_path = general.get_main_path("Fantasy_Football")
db_path = f"{root_path}/Data/Databases/"
dm = DataManage(db_path)

daily_root_path = general.get_main_path("Daily_Fantasy_Data")
daily_db_path = f"{daily_root_path}/Databases/"
dm_daily = DataManage(daily_db_path)


#==========
# Helpers
#==========

def clean_player_names(df):
    df = df.copy()
    df["player"] = df["player"].apply(dc.name_clean)
    return df


def add_missing_cols(df, cols, fill_value=0):
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


def add_bonus_cols(df):
    df = df.copy()

    df = add_missing_cols(
        df,
        [
            "rush_yards_gained_sum",
            "rec_yards_gained_sum",
            "pass_yards_gained_sum",
            "rush_fumble_lost_sum",
            "rec_fumble_lost_sum",
            "pass_fumble_lost_sum",
        ],
    )

    df["rush_yd_100_bonus"] = np.where(df["rush_yards_gained_sum"] >= 100, 1, 0)
    df["rush_yd_200_bonus"] = np.where(df["rush_yards_gained_sum"] >= 200, 1, 0)
    df["rec_yd_100_bonus"] = np.where(df["rec_yards_gained_sum"] >= 100, 1, 0)
    df["rec_yd_200_bonus"] = np.where(df["rec_yards_gained_sum"] >= 200, 1, 0)
    df["pass_yd_300_bonus"] = np.where(df["pass_yards_gained_sum"] >= 300, 1, 0)
    df["pass_yd_400_bonus"] = np.where(df["pass_yards_gained_sum"] >= 400, 1, 0)

    df["fumble_lost"] = (
        df["rush_fumble_lost_sum"]
        + df["rec_fumble_lost_sum"]
        + df["pass_fumble_lost_sum"]
    )
    return df


def calc_fp(df, pts_dict, output_col):
    df = add_missing_cols(df, pts_dict.keys())
    cols = list(pts_dict.keys())
    pts = list(pts_dict.values())
    df[output_col] = (df[cols] * pts).sum(axis=1)
    return df


def add_fantasy_points(df, pos):
    df = add_bonus_cols(df)

    df = calc_fp(df, get_scoring_dict("rush"), "fantasy_pts_rush")

    if pos == "QB":
        df = calc_fp(df, get_scoring_dict("passing"), "fantasy_pts_pass")
        df["fantasy_pts"] = df["fantasy_pts_rush"] + df["fantasy_pts_pass"]

        df = add_missing_cols(df, ["pass_qb_dropback_sum", "rush_rush_attempt_sum"])
        df["total_plays"] = df["pass_qb_dropback_sum"] + df["rush_rush_attempt_sum"]
        df = df[df["total_plays"] > 15].reset_index(drop=True)
    else:
        df = calc_fp(df, get_scoring_dict("receiving"), "fantasy_pts_rec")
        df["fantasy_pts"] = df["fantasy_pts_rush"] + df["fantasy_pts_rec"]

    return df


def exp_bucket(year_exp):
    if pd.isna(year_exp):
        return "unknown"
    if year_exp <= 0:
        return "rookie"
    if year_exp <= 2:
        return "young"
    if year_exp <= 6:
        return "prime"
    return "veteran"


def year_exp_bucket(year_exp):
    if pd.isna(year_exp):
        return -1
    return int(max(0, round(float(year_exp))))


def qb_team_rank_bucket(rank):
    if pd.isna(rank) or rank < 0:
        return "unknown"
    if rank <= 1:
        return "qb1"
    if rank <= 2:
        return "qb2"
    return "qb3_plus"


def add_qb_team_rank_fields(df, year_col, projection_col):
    df = df.copy()
    df["qb_team_rank"] = -1
    df["qb_team_rank_bucket"] = "non_qb"

    if "team" not in df.columns:
        df.loc[df["pos"].eq("QB"), "qb_team_rank_bucket"] = "unknown"
        return df

    qb_mask = df["pos"].eq("QB") & df["team"].notnull()
    if qb_mask.any():
        qb_rank = (
            df.loc[qb_mask]
            .sort_values([year_col, "team", projection_col], ascending=[True, True, False])
            .groupby([year_col, "team"])
            .cumcount()
            + 1
        )
        df.loc[qb_rank.index, "qb_team_rank"] = qb_rank.astype(int)

    unknown_qb_mask = df["pos"].eq("QB") & df["qb_team_rank"].lt(0)
    df.loc[df["pos"].eq("QB"), "qb_team_rank_bucket"] = df.loc[
        df["pos"].eq("QB"), "qb_team_rank"
    ].apply(qb_team_rank_bucket)
    df.loc[unknown_qb_mask, "qb_team_rank_bucket"] = "unknown"
    return df


def add_exp_fields(df):
    df = df.copy()
    df["year_exp"] = pd.to_numeric(df["year_exp"], errors="coerce")
    df["year_exp_bucket"] = df["year_exp"].apply(year_exp_bucket)
    df["exp_bucket"] = df["year_exp"].apply(exp_bucket)
    return df


def add_projection_buckets(df, value_col, group_cols, pct_col="projection_rank_pct"):
    df = df.copy()
    df[pct_col] = (
        df.groupby(group_cols)[value_col]
        .rank(method="first", pct=True, ascending=True)
        .astype(float)
    )
    df["projection_decile"] = (
        np.ceil(df[pct_col] * PROJECTION_BUCKETS)
        .clip(1, PROJECTION_BUCKETS)
        .astype(int)
    )
    df["projection_tier"] = pd.cut(
        df[pct_col],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["fringe", "depth", "starter", "elite"],
        include_lowest=True,
    ).astype(str)
    return df


def safe_pool_part(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def player_pool_key(row):
    return "|".join(
        [
            str(row.year),
            str(row.version),
            str(row.dataset),
            str(row.pos),
            safe_pool_part(row.player),
        ]
    )


def stable_seed(*parts):
    seed_text = "|".join(str(p) for p in parts)
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + POOL_RANDOM_SEED) % (2**32)


def get_daily_max_template_season():
    max_seasons = []
    for pos in POSITIONS:
        max_season = dm_daily.read(
            f"SELECT MAX(season) max_season FROM {pos}_Stats WHERE week <= {WEEK_COUNT}",
            "FastR_Beta",
        ).max_season.iloc[0]
        max_seasons.append(int(max_season))
    return min(max_seasons)


def load_validation_ensemble_predictions(max_template_season):
    split = dm.read(
        f"""
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               rush_pass,
               AVG(pred_fp_per_game) pred_fp_per_game
        FROM Model_Validations_Resid
        WHERE version='{LEAGUE}'
              AND year={YEAR}
              AND dataset NOT LIKE '%Rookie%'
              AND current_or_next_year='{VALIDATION_CURRENT_OR_NEXT_YEAR}'
              AND rush_pass IN ('rush', 'pass', 'rec')
              AND season BETWEEN {TEMPLATE_SEASON_MIN} AND {max_template_season}
        GROUP BY player, season, pos, rush_pass
        """,
        "Validations",
    )

    if len(split) > 0:
        split = clean_player_names(split)
        split = (
            split.groupby(["player", "season", "pos"], as_index=False)
            .agg({"pred_fp_per_game": "sum"})
        )
        split["ensemble_source"] = "validation_rush_pass_rec"

    all_current = dm.read(
        f"""
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               rush_pass,
               AVG(pred_fp_per_game) pred_fp_per_game
        FROM Model_Validations_Resid
        WHERE version='{LEAGUE}'
              AND year={YEAR}
              AND dataset NOT LIKE '%Rookie%'
              AND current_or_next_year='{VALIDATION_CURRENT_OR_NEXT_YEAR}'
              AND rush_pass NOT IN ('rush', 'pass', 'rec')
              AND season BETWEEN {TEMPLATE_SEASON_MIN} AND {max_template_season}
        GROUP BY player, season, pos, rush_pass
        """,
        "Validations",
    )

    if len(all_current) > 0:
        all_current = clean_player_names(all_current)
        all_current = (
            all_current.groupby(["player", "season", "pos"], as_index=False)
            .agg({"pred_fp_per_game": "mean"})
        )
        all_current["ensemble_source"] = "validation_all_current"

    all_next = dm.read(
        f"""
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               rush_pass,
               AVG(pred_fp_per_game) pred_fp_per_game
        FROM Model_Validations_Resid
        WHERE version='{LEAGUE}'
              AND year={YEAR}
              AND dataset NOT LIKE '%Rookie%'
              AND current_or_next_year='next'
              AND rush_pass NOT IN ('rush', 'pass', 'rec')
              AND pos != 'QB'
              AND season BETWEEN {TEMPLATE_SEASON_MIN} AND {max_template_season}
        GROUP BY player, season, pos, rush_pass
        """,
        "Validations",
    )

    if len(all_next) > 0:
        all_next = clean_player_names(all_next)
        all_next = (
            all_next.groupby(["player", "season", "pos"], as_index=False)
            .agg({"pred_fp_per_game": "mean"})
        )
        all_next["ensemble_source"] = "validation_all_next"

    validation_preds = pd.concat([split, all_current, all_next], ignore_index=True)
    if len(validation_preds) == 0:
        return pd.DataFrame(
            columns=[
                "player",
                "season",
                "pos",
                "validation_pred_fp_per_game",
                "validation_ensemble_sources",
            ]
        )

    validation_preds = (
        validation_preds.groupby(["player", "season", "pos"], as_index=False)
        .agg(
            validation_pred_fp_per_game=("pred_fp_per_game", "mean"),
            validation_ensemble_sources=("ensemble_source", lambda x: ",".join(sorted(set(x)))),
        )
    )
    return validation_preds


def load_historical_projection_context(max_template_season):
    proj = pd.DataFrame()

    for pos in POSITIONS:
        df_pos = dm.read(
            f"""
            SELECT player,
                   pos,
                   team,
                   CAST(year AS INTEGER) season,
                   avg_proj_points,
                   avg_pick,
                   year_exp
            FROM {pos}_{YEAR}_ProjOnly
            WHERE pos='{pos}'
                  AND year BETWEEN {TEMPLATE_SEASON_MIN} AND {max_template_season}
            """,
            "Model_Inputs",
        )
        proj = pd.concat([proj, df_pos], ignore_index=True)

    proj = clean_player_names(proj)
    proj = proj.sort_values(
        ["season", "pos", "player", "avg_proj_points"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["season", "pos", "player"])
    proj = add_qb_team_rank_fields(
        proj,
        year_col="season",
        projection_col="avg_proj_points",
    )

    validation_preds = load_validation_ensemble_predictions(max_template_season)

    proj = add_exp_fields(proj)
    proj["preseason_proj_ppg"] = proj["avg_proj_points"] / WEEK_COUNT
    proj = proj.merge(
        validation_preds,
        on=["player", "season", "pos"],
        how="left",
    )
    proj["historical_pred_fp_per_game"] = proj["validation_pred_fp_per_game"].combine_first(
        proj["preseason_proj_ppg"]
    )
    proj["historical_projection_source"] = np.where(
        proj["validation_pred_fp_per_game"].notnull(),
        "validation_ensemble",
        "preseason_projection_fallback",
    )
    proj = add_projection_buckets(
        proj,
        value_col="historical_pred_fp_per_game",
        group_cols=["season", "pos"],
    )
    return proj.reset_index(drop=True)


def load_weekly_points(max_template_season):
    weekly = pd.DataFrame()

    for pos in POSITIONS:
        print(f"Loading weekly {pos} rows...")
        df_pos = dm_daily.read(
            f"""
            SELECT *
            FROM {pos}_Stats
            WHERE season BETWEEN {TEMPLATE_SEASON_MIN} AND {max_template_season}
                  AND week BETWEEN 1 AND {WEEK_COUNT}
            """,
            "FastR_Beta",
        )

        df_pos = df_pos[~((df_pos.player == "Adrian Peterson") & (df_pos.team == "CHI"))]
        df_pos = df_pos[
            ~((df_pos.player == "Steve Smith") & (df_pos.team.isin(["NYG", "PHI", "LAR"])))
        ]
        df_pos = df_pos[~((df_pos.player == "Mike Williams") & (df_pos.season < 2017))]
        df_pos = df_pos[
            ~((df_pos.player == "Trey Mcbride") & (df_pos.season == 2023) & (df_pos.week < 8))
        ]

        df_pos = clean_player_names(df_pos)
        df_pos = add_fantasy_points(df_pos, pos)
        df_pos["pos"] = pos
        df_pos = (
            df_pos.groupby(["player", "pos", "season", "week"], as_index=False)
            .agg({"fantasy_pts": "sum"})
        )

        weekly = pd.concat([weekly, df_pos], ignore_index=True)

    weekly["season"] = weekly["season"].astype(int)
    weekly["week"] = weekly["week"].astype(int)
    return weekly


def build_weekly_templates(proj, weekly):
    template_index = proj[
        [
            "player",
            "pos",
            "team",
            "season",
            "avg_proj_points",
            "preseason_proj_ppg",
            "validation_pred_fp_per_game",
            "historical_pred_fp_per_game",
            "historical_projection_source",
            "validation_ensemble_sources",
            "avg_pick",
            "year_exp",
            "year_exp_bucket",
            "exp_bucket",
            "qb_team_rank",
            "qb_team_rank_bucket",
            "projection_rank_pct",
            "projection_decile",
            "projection_tier",
        ]
    ].copy()
    template_index["template_id"] = np.arange(1, len(template_index) + 1)

    week_grid = pd.DataFrame({"week": WEEKS})
    expanded = template_index[["template_id", "player", "pos", "season"]].merge(
        week_grid, how="cross"
    )

    expanded = expanded.merge(
        weekly,
        on=["player", "pos", "season", "week"],
        how="left",
        indicator=True,
    )
    expanded["active_week"] = expanded["_merge"].eq("both")
    expanded["fantasy_pts"] = expanded["fantasy_pts"].fillna(0)

    season_stats = (
        expanded.groupby("template_id", as_index=False)
        .agg(
            active_games=("active_week", "sum"),
            season_points=("fantasy_pts", "sum"),
        )
    )
    season_stats["active_ppg"] = np.where(
        season_stats["active_games"] > 0,
        season_stats["season_points"] / season_stats["active_games"],
        0,
    )

    expanded = expanded.merge(
        season_stats[["template_id", "active_ppg"]],
        on="template_id",
        how="left",
    )
    expanded["week_profile"] = np.where(
        expanded["active_ppg"] > 0,
        expanded["fantasy_pts"] / expanded["active_ppg"],
        0,
    )

    week_profiles = expanded.pivot_table(
        index="template_id",
        columns="week",
        values="week_profile",
        aggfunc="sum",
        fill_value=0,
    )
    week_profiles = week_profiles.reindex(columns=WEEKS, fill_value=0)
    week_profiles.columns = [f"week_{w}" for w in WEEKS]
    week_profiles = week_profiles.reset_index()

    templates = (
        template_index.merge(season_stats, on="template_id", how="left")
        .merge(week_profiles, on="template_id", how="left")
    )
    templates["profile_total"] = templates[[f"week_{w}" for w in WEEKS]].sum(axis=1)

    front_cols = [
        "template_id",
        "player",
        "pos",
        "team",
        "season",
        "avg_proj_points",
        "preseason_proj_ppg",
        "validation_pred_fp_per_game",
        "historical_pred_fp_per_game",
        "historical_projection_source",
        "validation_ensemble_sources",
        "avg_pick",
        "year_exp",
        "year_exp_bucket",
        "exp_bucket",
        "qb_team_rank",
        "qb_team_rank_bucket",
        "projection_rank_pct",
        "projection_decile",
        "projection_tier",
        "active_games",
        "active_ppg",
        "season_points",
        "profile_total",
    ]
    return templates[front_cols + [f"week_{w}" for w in WEEKS]]


def select_player_template_pool(player_row, templates):
    pos_templates = templates[templates["pos"] == player_row.pos].copy()
    target_decile = int(player_row.projection_decile)
    target_exp = int(player_row.year_exp_bucket)
    is_qb = player_row.pos == "QB"
    target_qb_team_rank_bucket = getattr(player_row, "qb_team_rank_bucket", "non_qb")
    qb_rank_order = {
        "qb1": 1,
        "qb2": 2,
        "qb3_plus": 3,
        "unknown": 4,
        "non_qb": 4,
    }
    target_qb_rank_value = qb_rank_order.get(target_qb_team_rank_bucket, 4)
    group_cols = ["projection_decile", "year_exp_bucket", "qb_team_rank_bucket"]

    cells = (
        pos_templates.groupby(group_cols, as_index=False)
        .agg(template_count=("template_id", "nunique"))
    )
    cells["projection_distance"] = (cells["projection_decile"] - target_decile).abs()
    cells["exp_distance"] = (cells["year_exp_bucket"] - target_exp).abs()
    if is_qb:
        cells["qb_team_rank_distance"] = (
            cells["qb_team_rank_bucket"]
            .map(qb_rank_order)
            .fillna(4)
            .sub(target_qb_rank_value)
            .abs()
            .astype(int)
        )
    else:
        cells["qb_team_rank_bucket"] = "non_qb"
        cells["qb_team_rank_distance"] = 0
    cells["cell_distance"] = (
        cells["projection_distance"]
        + cells["exp_distance"]
        + (2 * cells["qb_team_rank_distance"])
    )

    rng = np.random.default_rng(
        stable_seed(
            player_row.player,
            player_row.pos,
            player_row.year,
            player_row.version,
            player_row.dataset,
        )
    )
    cells["tie_break"] = rng.random(len(cells))
    sort_cols = ["cell_distance", "projection_distance", "exp_distance", "tie_break"]
    if is_qb:
        sort_cols = [
            "qb_team_rank_distance",
            "cell_distance",
            "projection_distance",
            "exp_distance",
            "tie_break",
        ]
    cells = cells.sort_values(sort_cols).reset_index(drop=True)

    selected_cells = []
    selected_count = 0
    for cell in cells.itertuples(index=False):
        selected_cells.append(
            {
                "projection_decile": cell.projection_decile,
                "year_exp_bucket": cell.year_exp_bucket,
                "qb_team_rank_bucket": cell.qb_team_rank_bucket,
                "projection_distance": cell.projection_distance,
                "exp_distance": cell.exp_distance,
                "qb_team_rank_distance": cell.qb_team_rank_distance,
                "cell_distance": cell.cell_distance,
            }
        )
        selected_count += int(cell.template_count)
        if selected_count >= MIN_TEMPLATE_POOL_SIZE:
            break

    selected_cells = pd.DataFrame(selected_cells)
    selected_templates = pos_templates.merge(
        selected_cells,
        on=group_cols,
        how="inner",
    )
    template_pool_key = player_row.template_pool_key
    exact_mask = (
        (cells["projection_decile"] == target_decile)
        & (cells["year_exp_bucket"] == target_exp)
    )
    if is_qb:
        exact_mask &= cells["qb_team_rank_bucket"].eq(target_qb_team_rank_bucket)
    exact_templates = int(cells[exact_mask]["template_count"].sum())

    if exact_templates >= MIN_TEMPLATE_POOL_SIZE:
        pool_level = "exact_decile_exp_qb_role" if is_qb else "exact_decile_exp"
    else:
        pool_level = "expanded_decile_exp_qb_role" if is_qb else "expanded_decile_exp"

    pool_members = selected_templates[
        [
            "template_id",
            "pos",
            "projection_decile",
            "year_exp_bucket",
            "qb_team_rank_bucket",
            "projection_distance",
            "exp_distance",
            "qb_team_rank_distance",
            "cell_distance",
        ]
    ].copy()
    pool_members["template_pool_key"] = template_pool_key
    pool_members["pool_level"] = pool_level
    pool_members["pool_player"] = player_row.player
    pool_members["pool_year"] = player_row.year
    pool_members["pool_version"] = player_row.version
    pool_members["pool_dataset"] = player_row.dataset
    pool_members["target_projection_decile"] = target_decile
    pool_members["target_year_exp_bucket"] = target_exp
    pool_members["target_qb_team_rank_bucket"] = target_qb_team_rank_bucket

    pool_members = pool_members[
        [
            "template_pool_key",
            "pool_level",
            "pool_player",
            "pool_year",
            "pool_version",
            "pool_dataset",
            "pos",
            "template_id",
            "target_projection_decile",
            "target_year_exp_bucket",
            "target_qb_team_rank_bucket",
            "projection_decile",
            "year_exp_bucket",
            "qb_team_rank_bucket",
            "projection_distance",
            "exp_distance",
            "qb_team_rank_distance",
            "cell_distance",
        ]
    ]

    pool_summary = {
        "template_pool_key": template_pool_key,
        "pool_level": pool_level,
        "player": player_row.player,
        "pos": player_row.pos,
        "year": player_row.year,
        "version": player_row.version,
        "dataset": player_row.dataset,
        "target_projection_decile": target_decile,
        "target_year_exp_bucket": target_exp,
        "target_qb_team_rank_bucket": target_qb_team_rank_bucket,
        "exact_cell_templates": exact_templates,
        "template_count": int(selected_templates["template_id"].nunique()),
        "selected_cell_count": int(selected_cells.shape[0]),
        "min_projection_decile": int(selected_cells["projection_decile"].min()),
        "max_projection_decile": int(selected_cells["projection_decile"].max()),
        "min_year_exp_bucket": int(selected_cells["year_exp_bucket"].min()),
        "max_year_exp_bucket": int(selected_cells["year_exp_bucket"].max()),
        "min_qb_team_rank_distance": int(selected_cells["qb_team_rank_distance"].min()),
        "max_qb_team_rank_distance": int(selected_cells["qb_team_rank_distance"].max()),
        "max_projection_distance": int(selected_cells["projection_distance"].max()),
        "max_exp_distance": int(selected_cells["exp_distance"].max()),
        "max_cell_distance": int(selected_cells["cell_distance"].max()),
        "min_template_pool_size": MIN_TEMPLATE_POOL_SIZE,
    }

    return pool_members, pool_summary


def build_pool_tables(templates, player_map):
    all_members = []
    all_summaries = []

    for player_row in player_map.itertuples(index=False):
        pool_members, pool_summary = select_player_template_pool(player_row, templates)
        all_members.append(pool_members)
        all_summaries.append(pool_summary)

    return pd.concat(all_members, ignore_index=True), pd.DataFrame(all_summaries)


def build_template_join_audit(templates):
    audit = templates[
        [
            "template_id",
            "player",
            "pos",
            "team",
            "season",
            "avg_pick",
            "year_exp",
            "year_exp_bucket",
            "exp_bucket",
            "qb_team_rank",
            "qb_team_rank_bucket",
            "historical_pred_fp_per_game",
            "historical_projection_source",
            "validation_ensemble_sources",
            "projection_rank_pct",
            "projection_decile",
            "projection_tier",
            "active_games",
            "active_ppg",
            "season_points",
            "profile_total",
        ]
    ].copy()

    audit["zero_active_template"] = audit["active_games"].eq(0)
    audit["low_active_template"] = audit["active_games"].le(LOW_ACTIVE_GAME_THRESHOLD)
    audit["high_value_low_active_template"] = (
        audit["low_active_template"]
        & (
            audit["projection_decile"].ge(8)
            | audit["avg_pick"].le(120)
            | audit["historical_pred_fp_per_game"].ge(7)
        )
    )
    audit["profile_total_abs_error"] = (audit["profile_total"] - audit["active_games"]).abs()
    audit["profile_total_mismatch"] = audit["profile_total_abs_error"].gt(0.01)

    return audit.sort_values(
        ["zero_active_template", "high_value_low_active_template", "active_games", "historical_pred_fp_per_game"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def build_bucket_comparability_audit(proj, player_map):
    hist = proj[
        [
            "pos",
            "season",
            "projection_decile",
            "projection_tier",
            "historical_pred_fp_per_game",
            "projection_rank_pct",
        ]
    ].copy()
    hist["universe"] = "historical_proj"
    hist["year"] = hist["season"]
    hist["version"] = LEAGUE
    hist["dataset"] = "historical_projection_context"
    hist = hist.rename(
        columns={
            "historical_pred_fp_per_game": "projection_value",
            "projection_rank_pct": "rank_pct",
        }
    )

    current = player_map[
        [
            "pos",
            "year",
            "version",
            "dataset",
            "projection_decile",
            "projection_tier",
            "pred_fp_per_game",
            "prediction_rank_pct",
        ]
    ].copy()
    current["universe"] = "current_final_predictions"
    current["season"] = current["year"]
    current = current.rename(
        columns={
            "pred_fp_per_game": "projection_value",
            "prediction_rank_pct": "rank_pct",
        }
    )

    audit = pd.concat(
        [
            hist[
                [
                    "universe",
                    "year",
                    "season",
                    "version",
                    "dataset",
                    "pos",
                    "projection_decile",
                    "projection_tier",
                    "projection_value",
                    "rank_pct",
                ]
            ],
            current[
                [
                    "universe",
                    "year",
                    "season",
                    "version",
                    "dataset",
                    "pos",
                    "projection_decile",
                    "projection_tier",
                    "projection_value",
                    "rank_pct",
                ]
            ],
        ],
        ignore_index=True,
    )

    audit = (
        audit.groupby(
            [
                "universe",
                "year",
                "season",
                "version",
                "dataset",
                "pos",
                "projection_decile",
                "projection_tier",
            ],
            as_index=False,
        )
        .agg(
            player_count=("projection_value", "size"),
            min_projection_value=("projection_value", "min"),
            median_projection_value=("projection_value", "median"),
            max_projection_value=("projection_value", "max"),
            min_rank_pct=("rank_pct", "min"),
            max_rank_pct=("rank_pct", "max"),
        )
    )
    universe_sizes = (
        audit.groupby(["universe", "year", "season", "version", "dataset", "pos"], as_index=False)
        .agg(universe_player_count=("player_count", "sum"))
    )
    audit = audit.merge(
        universe_sizes,
        on=["universe", "year", "season", "version", "dataset", "pos"],
        how="left",
    )
    audit["decile_player_share"] = audit["player_count"] / audit["universe_player_count"]

    return audit.sort_values(
        ["universe", "pos", "year", "projection_decile"]
    ).reset_index(drop=True)


def build_player_pool_audit(player_map, pool_members, templates):
    template_quality = templates[
        [
            "template_id",
            "active_games",
            "active_ppg",
            "season_points",
            "profile_total",
            "projection_decile",
            "year_exp_bucket",
        ]
    ].copy()
    template_quality["zero_active_template"] = template_quality["active_games"].eq(0)
    template_quality["low_active_template"] = template_quality["active_games"].le(
        LOW_ACTIVE_GAME_THRESHOLD
    )

    pool_quality = pool_members.merge(template_quality, on="template_id", how="left")
    pool_audit = (
        pool_quality.groupby("template_pool_key", as_index=False)
        .agg(
            selected_template_count=("template_id", "nunique"),
            zero_active_templates=("zero_active_template", "sum"),
            low_active_templates=("low_active_template", "sum"),
            min_active_games=("active_games", "min"),
            median_active_games=("active_games", "median"),
            mean_active_games=("active_games", "mean"),
            max_active_games=("active_games", "max"),
            median_active_ppg=("active_ppg", "median"),
            mean_profile_total=("profile_total", "mean"),
            min_pool_projection_decile=("projection_decile_y", "min"),
            max_pool_projection_decile=("projection_decile_y", "max"),
            min_pool_year_exp_bucket=("year_exp_bucket_y", "min"),
            max_pool_year_exp_bucket=("year_exp_bucket_y", "max"),
        )
    )
    pool_audit["zero_active_template_share"] = (
        pool_audit["zero_active_templates"] / pool_audit["selected_template_count"]
    )
    pool_audit["low_active_template_share"] = (
        pool_audit["low_active_templates"] / pool_audit["selected_template_count"]
    )

    player_cols = [
        "template_pool_key",
        "player",
        "pos",
        "team",
        "year",
        "version",
        "dataset",
        "pred_fp_per_game",
        "current_avg_proj_points",
        "avg_pick",
        "qb_team_rank",
        "qb_team_rank_bucket",
        "projection_decile",
        "projection_tier",
        "year_exp_bucket",
        "exp_bucket",
        "template_pool_level",
        "template_pool_size",
        "selected_cell_count",
        "exact_cell_templates",
        "target_qb_team_rank_bucket",
        "min_projection_decile",
        "max_projection_decile",
        "min_year_exp_bucket",
        "max_year_exp_bucket",
        "min_qb_team_rank_distance",
        "max_qb_team_rank_distance",
    ]
    audit = player_map[player_cols].merge(
        pool_audit,
        on="template_pool_key",
        how="left",
    )
    audit["missing_template_pool"] = audit["selected_template_count"].isnull()
    audit["template_pool_below_min"] = audit["selected_template_count"].fillna(0).lt(
        MIN_TEMPLATE_POOL_SIZE
    )
    audit["high_zero_template_share"] = audit["zero_active_template_share"].fillna(0).gt(
        HIGH_ZERO_TEMPLATE_POOL_SHARE
    )
    audit["high_low_active_template_share"] = audit["low_active_template_share"].fillna(0).gt(
        HIGH_LOW_ACTIVE_TEMPLATE_POOL_SHARE
    )

    return audit.sort_values(
        [
            "missing_template_pool",
            "high_zero_template_share",
            "high_low_active_template_share",
            "zero_active_template_share",
            "pred_fp_per_game",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def validate_weekly_template_audits(player_pool_audit):
    missing_pool = player_pool_audit[player_pool_audit["missing_template_pool"]]
    below_min_pool = player_pool_audit[player_pool_audit["template_pool_below_min"]]

    if len(missing_pool) > 0:
        preview = ", ".join(missing_pool.player.head(10))
        raise ValueError(
            f"{len(missing_pool)} current players have no weekly template pool: {preview}"
        )

    if len(below_min_pool) > 0:
        preview = ", ".join(
            f"{row.player} ({int(row.selected_template_count)})"
            for row in below_min_pool.head(10).itertuples(index=False)
        )
        raise ValueError(
            f"{len(below_min_pool)} current players have template pools below "
            f"{MIN_TEMPLATE_POOL_SIZE}: {preview}"
        )


def load_current_player_context():
    current_context = pd.DataFrame()

    for pos in POSITIONS:
        df_pos = dm.read(
            f"""
            SELECT player,
                   pos,
                   team,
                   CAST(year AS INTEGER) year,
                   avg_proj_points current_avg_proj_points,
                   avg_pick model_input_avg_pick,
                   year_exp
            FROM {pos}_{YEAR}_ProjOnly
            WHERE pos='{pos}'
                  AND year={YEAR}
            """,
            "Model_Inputs",
        )
        current_context = pd.concat([current_context, df_pos], ignore_index=True)

    current_context = clean_player_names(current_context)
    current_context = current_context.sort_values(
        ["pos", "player", "current_avg_proj_points"],
        ascending=[True, True, False],
    ).drop_duplicates(["pos", "player"])
    current_context = add_qb_team_rank_fields(
        current_context,
        year_col="year",
        projection_col="current_avg_proj_points",
    )

    adp = dm.read(
        f"""
        SELECT player,
               CAST(year AS INTEGER) year,
               league,
               avg_pick adp_avg_pick,
               Years_of_Experience adp_year_exp
        FROM Avg_ADPs
        WHERE year={YEAR}
              AND league='{LEAGUE}'
        """,
        "Simulation",
    )
    adp = clean_player_names(adp)
    adp = adp.sort_values(["player", "adp_avg_pick"]).drop_duplicates(["player"])

    current_context = current_context.merge(
        adp[["player", "year", "adp_avg_pick", "adp_year_exp"]],
        on=["player", "year"],
        how="left",
    )
    current_context["year_exp"] = current_context["year_exp"].combine_first(
        current_context["adp_year_exp"]
    )
    current_context["avg_pick"] = current_context["model_input_avg_pick"].combine_first(
        current_context["adp_avg_pick"]
    )
    return current_context


def build_player_map_base():
    preds = dm.read(
        f"""
        SELECT *
        FROM Final_Predictions_Resid
        WHERE year={YEAR}
              AND version='{LEAGUE}'
              AND dataset='{PRED_VERSION}'
        """,
        "Simulation",
    )
    preds = clean_player_names(preds)
    preds = add_projection_buckets(
        preds,
        value_col="pred_fp_per_game",
        group_cols=["year", "version", "dataset", "pos"],
        pct_col="prediction_rank_pct",
    )

    current_context = load_current_player_context()
    player_map = preds.merge(
        current_context[
            [
                "player",
                "pos",
                "year",
                "team",
                "current_avg_proj_points",
                "avg_pick",
                "year_exp",
                "qb_team_rank",
                "qb_team_rank_bucket",
            ]
        ],
        on=["player", "pos", "year"],
        how="left",
    )
    player_map = add_exp_fields(player_map)
    player_map["template_pool_key"] = player_map.apply(player_pool_key, axis=1)

    cols = [
        "player",
        "pos",
        "year",
        "version",
        "dataset",
        "pred_fp_per_game",
        "current_avg_proj_points",
        "avg_pick",
        "year_exp",
        "year_exp_bucket",
        "exp_bucket",
        "team",
        "qb_team_rank",
        "qb_team_rank_bucket",
        "prediction_rank_pct",
        "projection_decile",
        "projection_tier",
        "template_pool_key",
    ]
    resid_cols = [c for c in player_map.columns if c.startswith("pred_resid_")]
    return player_map[cols + resid_cols].sort_values(
        ["pos", "pred_fp_per_game"],
        ascending=[True, False],
    )


def finalize_player_map(player_map, pool_summary):
    pool_cols = [
        "template_pool_key",
        "pool_level",
        "exact_cell_templates",
        "template_count",
        "selected_cell_count",
        "target_projection_decile",
        "target_year_exp_bucket",
        "target_qb_team_rank_bucket",
        "min_projection_decile",
        "max_projection_decile",
        "min_year_exp_bucket",
        "max_year_exp_bucket",
        "min_qb_team_rank_distance",
        "max_qb_team_rank_distance",
        "max_projection_distance",
        "max_exp_distance",
        "max_cell_distance",
        "min_template_pool_size",
    ]
    player_map = player_map.merge(
        pool_summary[pool_cols],
        on="template_pool_key",
        how="left",
    )
    player_map = player_map.rename(
        columns={
            "pool_level": "template_pool_level",
            "template_count": "template_pool_size",
        }
    )
    return player_map


def copy_simulation_db_to_apps():
    src = Path(root_path) / "Data" / "Databases" / "Simulation.sqlite3"
    app_dbs = [
        Path("/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3"),
        Path("/Users/borys/OneDrive/Documents/Github/Fantasy_Football_Snake/app/Simulation.sqlite3"),
    ]
    for dst in app_dbs:
        if dst.parent.exists():
            shutil.copyfile(src, dst)
            print(f"Copied Simulation.sqlite3 to {dst}")


def main():
    max_template_season = min(YEAR - 1, get_daily_max_template_season())
    print(
        f"Building weekly templates for {TEMPLATE_SEASON_MIN}-{max_template_season} "
        f"using {LEAGUE} scoring and {WEEK_COUNT} weeks..."
    )

    proj = load_historical_projection_context(max_template_season)
    weekly = load_weekly_points(max_template_season)

    templates = build_weekly_templates(proj, weekly)
    template_audit = build_template_join_audit(templates)
    player_map_base = build_player_map_base()
    bucket_audit = build_bucket_comparability_audit(proj, player_map_base)
    pool_members, pool_summary = build_pool_tables(templates, player_map_base)
    player_map = finalize_player_map(player_map_base, pool_summary)
    player_pool_audit = build_player_pool_audit(player_map, pool_members, templates)
    validate_weekly_template_audits(player_pool_audit)

    dm.write_to_db(templates, "Simulation", TEMPLATE_TABLE, "replace")
    dm.write_to_db(pool_members, "Simulation", POOL_TABLE, "replace")
    dm.write_to_db(pool_summary, "Simulation", POOL_SUMMARY_TABLE, "replace")
    dm.write_to_db(player_map, "Simulation", PLAYER_MAP_TABLE, "replace")
    dm.write_to_db(template_audit, "Simulation", TEMPLATE_AUDIT_TABLE, "replace")
    dm.write_to_db(player_pool_audit, "Simulation", PLAYER_POOL_AUDIT_TABLE, "replace")
    dm.write_to_db(bucket_audit, "Simulation", BUCKET_AUDIT_TABLE, "replace")

    copy_simulation_db_to_apps()

    print("\nTemplate count by position:")
    print(templates.groupby("pos").template_id.count())

    print("\nHistorical projection source:")
    print(templates.groupby(["pos", "historical_projection_source"]).template_id.count())

    print("\nHistorical template active-game audit:")
    template_summary = (
        template_audit.groupby("pos", as_index=False)
        .agg(
            templates=("template_id", "count"),
            zero_active_templates=("zero_active_template", "sum"),
            low_active_templates=("low_active_template", "sum"),
            high_value_low_active_templates=("high_value_low_active_template", "sum"),
        )
    )
    template_summary["zero_active_share"] = (
        template_summary["zero_active_templates"] / template_summary["templates"]
    )
    template_summary["low_active_share"] = (
        template_summary["low_active_templates"] / template_summary["templates"]
    )
    print(template_summary)

    suspicious_templates = template_audit[
        template_audit["zero_active_template"] | template_audit["high_value_low_active_template"]
    ]
    if len(suspicious_templates) > 0:
        print("\nSuspicious zero/low-active historical templates:")
        print(
            suspicious_templates[
                [
                    "player",
                    "pos",
                    "team",
                    "season",
                    "avg_pick",
                    "qb_team_rank",
                    "qb_team_rank_bucket",
                    "historical_pred_fp_per_game",
                    "projection_decile",
                    "projection_tier",
                    "active_games",
                    "historical_projection_source",
                ]
            ].head(30)
        )

    print("\nCurrent player template pool levels:")
    print(player_map.groupby(["pos", "template_pool_level"]).player.count())

    print("\nCurrent player pool injury/zero-template exposure:")
    pool_exposure = (
        player_pool_audit.groupby("pos", as_index=False)
        .agg(
            players=("player", "count"),
            high_zero_template_share=("high_zero_template_share", "sum"),
            high_low_active_template_share=("high_low_active_template_share", "sum"),
            max_zero_active_template_share=("zero_active_template_share", "max"),
            max_low_active_template_share=("low_active_template_share", "max"),
            min_template_pool_size=("selected_template_count", "min"),
        )
    )
    print(pool_exposure)

    flagged_pools = player_pool_audit[
        player_pool_audit["high_zero_template_share"]
        | player_pool_audit["high_low_active_template_share"]
    ]
    if len(flagged_pools) > 0:
        print("\nFlagged current-player template pools:")
        print(
            flagged_pools[
                [
                    "player",
                    "pos",
                    "team",
                    "pred_fp_per_game",
                    "current_avg_proj_points",
                    "avg_pick",
                    "qb_team_rank",
                    "qb_team_rank_bucket",
                    "projection_decile",
                    "template_pool_level",
                    "selected_template_count",
                    "zero_active_template_share",
                    "low_active_template_share",
                    "min_active_games",
                    "median_active_games",
                ]
            ].head(30)
        )

    bucket_universe_counts = (
        bucket_audit.groupby(["universe", "pos", "year"], as_index=False)
        .agg(universe_player_count=("universe_player_count", "max"))
        .sort_values(["universe", "pos", "year"])
    )
    print("\nCurrent bucket universe sizes:")
    print(bucket_universe_counts[bucket_universe_counts["universe"].eq("current_final_predictions")])

    print("\nHistorical bucket universe size ranges:")
    print(
        bucket_universe_counts[bucket_universe_counts["universe"].eq("historical_proj")]
        .groupby("pos", as_index=False)
        .agg(
            min_historical_players=("universe_player_count", "min"),
            median_historical_players=("universe_player_count", "median"),
            max_historical_players=("universe_player_count", "max"),
        )
    )

    print("\nSmallest selected pool sizes:")
    print(
        player_map[
            [
                "player",
                "pos",
                "team",
                "projection_decile",
                "year_exp_bucket",
                "qb_team_rank",
                "qb_team_rank_bucket",
                "projection_tier",
                "exp_bucket",
                "template_pool_level",
                "template_pool_size",
                "selected_cell_count",
                "min_projection_decile",
                "max_projection_decile",
                "min_year_exp_bucket",
                "max_year_exp_bucket",
                "min_qb_team_rank_distance",
                "max_qb_team_rank_distance",
            ]
        ]
        .sort_values("template_pool_size")
        .head(20)
    )


if __name__ == "__main__":
    main()

#%%
