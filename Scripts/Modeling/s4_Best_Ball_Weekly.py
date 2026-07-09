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
MAX_TEMPLATE_POOL_SIZE = 80
TEMPLATE_SAMPLE_TOP_TO_BOTTOM_RATIO = 2.
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
ADP_AUDIT_TABLE = "Best_Ball_ADP_Audit"

TEMPLATE_ID_OFFSET_STEP = 1_000_000
TEMPLATE_ID_LEAGUE_OFFSETS = {
    "beta": 1 * TEMPLATE_ID_OFFSET_STEP,
    "dk": 2 * TEMPLATE_ID_OFFSET_STEP,
    "nffc": 3 * TEMPLATE_ID_OFFSET_STEP,
    "nv": 4 * TEMPLATE_ID_OFFSET_STEP,
}

LOW_ACTIVE_GAME_THRESHOLD = 2
HIGH_ZERO_TEMPLATE_POOL_SHARE = 0.10
HIGH_LOW_ACTIVE_TEMPLATE_POOL_SHARE = 0.20
EXCLUDE_ZERO_ACTIVE_NON_QB_POOL_TEMPLATES = True
DEFAULT_ADP_PICK = 240
ADP_AUDIT_HIGH_IMPACT_PPG_MIN = 3.0
ADP_AUDIT_POS_RANK_LIMITS = {
    "QB": 40,
    "RB": 100,
    "WR": 120,
    "TE": 50,
}

PROJECTION_COMPONENT_COLS = [
    "avg_proj_rush_points",
    "avg_proj_rec_points",
    "avg_proj_pass_points",
    "qb_avg_proj_pass_points",
]
MATCH_FILL_VALUE = 0.5
QB_RANK_DISTANCE_ORDER = {
    "qb1": 0,
    "qb2": 1,
    "qb3_plus": 2,
    "unknown": 2,
    "non_qb": 2,
}
COMMON_MATCH_FEATURE_WEIGHTS = {
    "match_projection_rank_pct": 2.5,
    "year_exp_scaled": 2.0,
    "projection_x_exp": 1.0,
    "adp_rank_pct": 0.5,
}
POSITION_MATCH_FEATURE_WEIGHTS = {
    "QB": {
        "qb_team_rank_distance": 1.5,
        "qb_room_share": 1.25,
        "qb1_over_qb2_gap_pct": 0.75,
        "rush_share_of_own_points": 1.25,
        "rush_proj_rank_pct": 1.0,
        "pass_proj_rank_pct": 1.0,
    },
    "RB": {
        "rush_proj_rank_pct": 1.0,
        "rec_proj_rank_pct": 1.0,
        "rec_share_of_own_points": 1.0,
        "rb_rush_share_of_room": 1.25,
        "rb_rec_share_of_room": 0.75,
    },
    "WR": {
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
    "TE": {
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
}
MATCH_FEATURE_WEIGHTS = {
    pos: {**COMMON_MATCH_FEATURE_WEIGHTS, **POSITION_MATCH_FEATURE_WEIGHTS[pos]}
    for pos in POSITIONS
}
MATCH_FEATURE_COLS = sorted(
    {
        col
        for weights in MATCH_FEATURE_WEIGHTS.values()
        for col in weights
        if col != "qb_team_rank_distance"
    }
)
MATCH_OUTPUT_COLS = [
    "match_projection_rank_pct",
    "year_exp_scaled",
    "projection_x_exp",
    "adp_rank_pct",
    "rush_proj_rank_pct",
    "rec_proj_rank_pct",
    "pass_proj_rank_pct",
    "rush_share_of_own_points",
    "rec_share_of_own_points",
    "rb_rush_share_of_room",
    "rb_rec_share_of_room",
    "team_rec_share",
    "team_qb_proj_points",
    "qb_room_share",
    "team_qb1_proj_points",
    "team_qb2_proj_points",
    "qb1_over_qb2_gap_pct",
    "team_qb_pass_points",
    "team_qb_pass_proj_rank_pct",
] + PROJECTION_COMPONENT_COLS


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


def projection_select_cols(pos, year_alias, total_alias=None, avg_pick_alias=None):
    table = f"{pos}_{YEAR}_ProjOnly"
    available_cols = set(dm.read(f"SELECT * FROM {table} LIMIT 0", "Model_Inputs").columns)

    cols = [
        "player",
        "pos",
        "team",
        f"CAST(year AS INTEGER) {year_alias}",
    ]
    if total_alias is None:
        cols.append("avg_proj_points")
    else:
        cols.extend([f"avg_proj_points {total_alias}", "avg_proj_points"])
    if avg_pick_alias is None:
        cols.append("avg_pick")
    else:
        cols.append(f"avg_pick {avg_pick_alias}")
    cols.append("year_exp")
    cols.extend([col for col in PROJECTION_COMPONENT_COLS if col in available_cols])
    return cols


def add_projection_component_cols(df):
    df = df.copy()
    for col in PROJECTION_COMPONENT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def safe_ratio(numerator, denominator, fill_value=0):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ratio = numerator / denominator
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return ratio.clip(lower=0, upper=1)


def add_group_rank_pct(df, value_col, group_cols, output_col, ascending=True):
    df = df.copy()
    rank_col = f"__{output_col}_rank_value"
    df[rank_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[output_col] = (
        df.groupby(group_cols)[rank_col]
        .rank(method="first", pct=True, ascending=ascending)
        .astype(float)
        .fillna(MATCH_FILL_VALUE)
    )
    return df.drop(columns=[rank_col])


def add_template_match_features(df, group_cols, rank_pct_col, total_points_col):
    df = add_projection_component_cols(df)
    df = df.copy()

    df["match_projection_rank_pct"] = (
        pd.to_numeric(df[rank_pct_col], errors="coerce").fillna(MATCH_FILL_VALUE)
    )
    df["year_exp_scaled"] = (
        pd.to_numeric(df["year_exp"], errors="coerce")
        .clip(lower=0, upper=10)
        .div(10)
        .fillna(MATCH_FILL_VALUE)
    )
    df["projection_x_exp"] = df["match_projection_rank_pct"] * df["year_exp_scaled"]

    df = add_group_rank_pct(
        df,
        value_col="avg_pick",
        group_cols=group_cols,
        output_col="adp_rank_pct",
        ascending=False,
    )
    df = add_group_rank_pct(
        df,
        value_col="avg_proj_rush_points",
        group_cols=group_cols,
        output_col="rush_proj_rank_pct",
    )
    df = add_group_rank_pct(
        df,
        value_col="avg_proj_rec_points",
        group_cols=group_cols,
        output_col="rec_proj_rank_pct",
    )
    df = add_group_rank_pct(
        df,
        value_col="avg_proj_pass_points",
        group_cols=group_cols,
        output_col="pass_proj_rank_pct",
    )

    total_points = pd.to_numeric(df[total_points_col], errors="coerce")
    df["rush_share_of_own_points"] = safe_ratio(df["avg_proj_rush_points"], total_points)
    df["rec_share_of_own_points"] = safe_ratio(df["avg_proj_rec_points"], total_points)

    team_group_cols = [col for col in group_cols if col != "pos"] + ["team"]
    df["team_rb_rush_points"] = 0.0
    df["team_rb_rec_points"] = 0.0
    df["team_rec_points"] = 0.0

    rb_mask = df["pos"].eq("RB") & df["team"].notnull()
    if rb_mask.any():
        df.loc[rb_mask, "team_rb_rush_points"] = (
            df.loc[rb_mask].groupby(team_group_cols)["avg_proj_rush_points"].transform("sum")
        )
        df.loc[rb_mask, "team_rb_rec_points"] = (
            df.loc[rb_mask].groupby(team_group_cols)["avg_proj_rec_points"].transform("sum")
        )

    receiver_mask = df["pos"].isin(["RB", "WR", "TE"]) & df["team"].notnull()
    if receiver_mask.any():
        df.loc[receiver_mask, "team_rec_points"] = (
            df.loc[receiver_mask].groupby(team_group_cols)["avg_proj_rec_points"].transform("sum")
        )

    df["rb_rush_share_of_room"] = 0.0
    df["rb_rec_share_of_room"] = 0.0
    df.loc[rb_mask, "rb_rush_share_of_room"] = safe_ratio(
        df.loc[rb_mask, "avg_proj_rush_points"],
        df.loc[rb_mask, "team_rb_rush_points"],
    )
    df.loc[rb_mask, "rb_rec_share_of_room"] = safe_ratio(
        df.loc[rb_mask, "avg_proj_rec_points"],
        df.loc[rb_mask, "team_rb_rec_points"],
    )

    pass_catcher_mask = df["pos"].isin(["WR", "TE"])
    df["team_rec_share"] = 0.0
    df.loc[pass_catcher_mask, "team_rec_share"] = safe_ratio(
        df.loc[pass_catcher_mask, "avg_proj_rec_points"],
        df.loc[pass_catcher_mask, "team_rec_points"],
    )

    qb_mask = df["pos"].eq("QB") & df["team"].notnull()
    df["team_qb_proj_points"] = 0.0
    df["qb_room_share"] = 0.0
    df["team_qb1_proj_points"] = 0.0
    df["team_qb2_proj_points"] = 0.0
    df["qb1_over_qb2_gap_pct"] = 0.0
    if qb_mask.any():
        df.loc[qb_mask, "team_qb_proj_points"] = (
            df.loc[qb_mask].groupby(team_group_cols)["avg_proj_points"].transform("sum")
        )
        df.loc[qb_mask, "qb_room_share"] = safe_ratio(
            df.loc[qb_mask, "avg_proj_points"],
            df.loc[qb_mask, "team_qb_proj_points"],
        )

        qb_ranked = df.loc[qb_mask, team_group_cols + ["avg_proj_points"]].copy()
        qb_ranked = qb_ranked.sort_values(
            team_group_cols + ["avg_proj_points"],
            ascending=[True] * len(team_group_cols) + [False],
        )
        qb_ranked["qb_room_rank"] = qb_ranked.groupby(team_group_cols).cumcount() + 1
        qb1 = (
            qb_ranked[qb_ranked["qb_room_rank"].eq(1)][team_group_cols + ["avg_proj_points"]]
            .rename(columns={"avg_proj_points": "team_qb1_proj_points_calc"})
        )
        qb2 = (
            qb_ranked[qb_ranked["qb_room_rank"].eq(2)][team_group_cols + ["avg_proj_points"]]
            .rename(columns={"avg_proj_points": "team_qb2_proj_points_calc"})
        )
        df = df.merge(qb1, on=team_group_cols, how="left")
        df = df.merge(qb2, on=team_group_cols, how="left")
        df["team_qb1_proj_points"] = df["team_qb1_proj_points_calc"].fillna(0)
        df["team_qb2_proj_points"] = df["team_qb2_proj_points_calc"].fillna(0)
        df["qb1_over_qb2_gap_pct"] = safe_ratio(
            df["team_qb1_proj_points"] - df["team_qb2_proj_points"],
            df["team_qb1_proj_points"],
        )
        df = df.drop(
            columns=["team_qb1_proj_points_calc", "team_qb2_proj_points_calc"]
        )

    df["team_qb_pass_points"] = pd.to_numeric(
        df["qb_avg_proj_pass_points"], errors="coerce"
    ).fillna(0)
    qb_mask = df["pos"].eq("QB") & df["team"].notnull()
    if qb_mask.any():
        qb_pass = (
            df.loc[qb_mask, team_group_cols + ["avg_proj_pass_points"]]
            .groupby(team_group_cols, as_index=False)
            .agg(team_qb_pass_points_calc=("avg_proj_pass_points", "max"))
        )
        df = df.merge(qb_pass, on=team_group_cols, how="left")
        df["team_qb_pass_points"] = np.where(
            df["team_qb_pass_points"].gt(0),
            df["team_qb_pass_points"],
            df["team_qb_pass_points_calc"].fillna(0),
        )
        df = df.drop(columns=["team_qb_pass_points_calc"])

    df = add_group_rank_pct(
        df,
        value_col="team_qb_pass_points",
        group_cols=group_cols,
        output_col="team_qb_pass_proj_rank_pct",
    )

    for col in MATCH_FEATURE_COLS:
        if col not in df.columns:
            df[col] = MATCH_FILL_VALUE
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(MATCH_FILL_VALUE)

    return df


def safe_pool_part(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def player_join_key(values):
    return (
        pd.Series(values)
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


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


def template_id_offset(league):
    if league in TEMPLATE_ID_LEAGUE_OFFSETS:
        return TEMPLATE_ID_LEAGUE_OFFSETS[league]

    digest = hashlib.md5(str(league).encode("utf-8")).hexdigest()
    return (int(digest[:4], 16) + 10) * TEMPLATE_ID_OFFSET_STEP


def rows_matching(df, criteria):
    mask = pd.Series(True, index=df.index)
    for col, value in criteria.items():
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if isinstance(value, (int, float)):
            mask &= pd.to_numeric(df[col], errors="coerce").eq(value)
        else:
            mask &= df[col].fillna("").astype(str).eq(str(value))
    return mask


def db_table_exists(table_name):
    exists = dm.read(
        f"""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
              AND name='{table_name}'
        """,
        "Simulation",
    )
    return len(exists) > 0


def infer_existing_best_ball_league():
    candidates = [
        (PLAYER_MAP_TABLE, "version"),
        (POOL_SUMMARY_TABLE, "version"),
        (POOL_TABLE, "pool_version"),
        (ADP_AUDIT_TABLE, "version"),
    ]
    for table_name, col in candidates:
        if not db_table_exists(table_name):
            continue
        existing = dm.read(f"SELECT * FROM {table_name} LIMIT 0", "Simulation")
        if col not in existing.columns:
            continue
        values = dm.read(
            f"""
            SELECT DISTINCT {col} inferred_league
            FROM {table_name}
            WHERE {col} IS NOT NULL
            """,
            "Simulation",
        )
        values = values["inferred_league"].dropna().astype(str).tolist()
        if len(values) == 1:
            return values[0]
    return None


def prepare_existing_best_ball_table(existing, table_name):
    existing = existing.copy()

    if table_name in [TEMPLATE_TABLE, TEMPLATE_AUDIT_TABLE]:
        if "league" not in existing.columns:
            existing["league"] = infer_existing_best_ball_league() or LEAGUE
        if "template_local_id" not in existing.columns and "template_id" in existing.columns:
            existing["template_local_id"] = existing["template_id"]

    if table_name == POOL_TABLE:
        if "league" not in existing.columns and "pool_version" in existing.columns:
            existing["league"] = existing["pool_version"]
        if "template_league" not in existing.columns and "pool_version" in existing.columns:
            existing["template_league"] = existing["pool_version"]

    return existing


def replace_table_slice(table_name, new_df, keep_existing_mask_func):
    if db_table_exists(table_name):
        existing = dm.read(f"SELECT * FROM {table_name}", "Simulation")
        existing = prepare_existing_best_ball_table(existing, table_name)
        keep_existing = existing[keep_existing_mask_func(existing)].copy()
        combined = pd.concat([keep_existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()

    ordered_cols = list(new_df.columns) + [
        col for col in combined.columns if col not in new_df.columns
    ]
    combined = combined[ordered_cols]
    dm.write_to_db(combined, "Simulation", table_name, "replace")


def keep_not_current_league(df):
    return ~rows_matching(df, {"league": LEAGUE})


def keep_not_current_pool_slice(df):
    return ~rows_matching(
        df,
        {
            "pool_year": YEAR,
            "pool_version": LEAGUE,
            "pool_dataset": PRED_VERSION,
        },
    )


def keep_not_current_prediction_slice(df):
    return ~rows_matching(
        df,
        {
            "year": YEAR,
            "version": LEAGUE,
            "dataset": PRED_VERSION,
        },
    )


def keep_not_current_bucket_slice(df):
    return ~rows_matching(df, {"version": LEAGUE})


def write_best_ball_tables(
    templates,
    pool_members,
    pool_summary,
    player_map,
    template_audit,
    player_pool_audit,
    bucket_audit,
    adp_audit,
):
    table_writes = [
        (TEMPLATE_TABLE, templates, keep_not_current_league),
        (POOL_TABLE, pool_members, keep_not_current_pool_slice),
        (POOL_SUMMARY_TABLE, pool_summary, keep_not_current_prediction_slice),
        (PLAYER_MAP_TABLE, player_map, keep_not_current_prediction_slice),
        (TEMPLATE_AUDIT_TABLE, template_audit, keep_not_current_league),
        (PLAYER_POOL_AUDIT_TABLE, player_pool_audit, keep_not_current_prediction_slice),
        (BUCKET_AUDIT_TABLE, bucket_audit, keep_not_current_bucket_slice),
        (ADP_AUDIT_TABLE, adp_audit, keep_not_current_prediction_slice),
    ]

    for table_name, df, keep_existing in table_writes:
        replace_table_slice(table_name, df, keep_existing)


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
        select_cols = projection_select_cols(pos, year_alias="season")
        df_pos = dm.read(
            f"""
            SELECT {", ".join(select_cols)}
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
    proj = add_template_match_features(
        proj,
        group_cols=["season", "pos"],
        rank_pct_col="projection_rank_pct",
        total_points_col="avg_proj_points",
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
    base_cols = [
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
    template_cols = base_cols + [
        col for col in MATCH_OUTPUT_COLS if col in proj.columns and col not in base_cols
    ]
    template_index = proj[template_cols].copy()
    template_index["league"] = LEAGUE
    template_index["template_local_id"] = np.arange(1, len(template_index) + 1)
    template_index["template_id"] = (
        template_id_offset(LEAGUE) + template_index["template_local_id"]
    )

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
    templates["active_ppg_resid"] = (
        templates["active_ppg"] - templates["historical_pred_fp_per_game"]
    )
    templates["profile_total"] = templates[[f"week_{w}" for w in WEEKS]].sum(axis=1)

    front_cols = [
        "league",
        "template_id",
        "template_local_id",
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
        "active_ppg_resid",
        "season_points",
        "profile_total",
    ]
    match_cols = [
        col for col in MATCH_OUTPUT_COLS if col in templates.columns and col not in front_cols
    ]
    return templates[front_cols + match_cols + [f"week_{w}" for w in WEEKS]]


def select_player_template_pool(player_row, templates):
    pos_templates = templates[templates["pos"] == player_row.pos].copy()
    if EXCLUDE_ZERO_ACTIVE_NON_QB_POOL_TEMPLATES and player_row.pos != "QB":
        pos_templates = pos_templates[pos_templates["active_games"].gt(0)].copy()

    if len(pos_templates) == 0:
        raise ValueError(f"No eligible weekly templates found for position {player_row.pos}.")

    target_decile = int(player_row.projection_decile)
    target_exp = int(player_row.year_exp_bucket)
    is_qb = player_row.pos == "QB"
    target_qb_team_rank_bucket = getattr(player_row, "qb_team_rank_bucket", "non_qb")

    target_qb_rank_value = QB_RANK_DISTANCE_ORDER.get(target_qb_team_rank_bucket, 2)
    pos_templates["projection_distance"] = (
        pos_templates["projection_decile"] - target_decile
    ).abs()
    pos_templates["exp_distance"] = (pos_templates["year_exp_bucket"] - target_exp).abs()
    if is_qb:
        pos_templates["qb_team_rank_distance"] = (
            pos_templates["qb_team_rank_bucket"]
            .map(QB_RANK_DISTANCE_ORDER)
            .fillna(2)
            .sub(target_qb_rank_value)
            .abs()
            .astype(int)
        )
    else:
        pos_templates["qb_team_rank_distance"] = 0
    pos_templates["cell_distance"] = (
        pos_templates["projection_distance"]
        + pos_templates["exp_distance"]
        + (2 * pos_templates["qb_team_rank_distance"])
    )

    weights = MATCH_FEATURE_WEIGHTS[player_row.pos]
    pos_templates["template_distance"] = 0.0
    distance_cols = []
    for feature, weight in weights.items():
        dist_col = f"distance_{feature}"
        if feature == "qb_team_rank_distance":
            feature_distance = pos_templates["qb_team_rank_distance"].astype(float)
        else:
            template_values = pd.to_numeric(
                pos_templates[feature], errors="coerce"
            ).fillna(MATCH_FILL_VALUE)
            target_value = pd.to_numeric(
                pd.Series([getattr(player_row, feature, MATCH_FILL_VALUE)]),
                errors="coerce",
            ).fillna(MATCH_FILL_VALUE).iloc[0]
            feature_distance = (template_values - target_value).abs()

        pos_templates[dist_col] = feature_distance
        pos_templates["template_distance"] += weight * feature_distance
        distance_cols.append(dist_col)

    rng = np.random.default_rng(
        stable_seed(
            player_row.player,
            player_row.pos,
            player_row.year,
            player_row.version,
            player_row.dataset,
        )
    )
    pos_templates["tie_break"] = rng.random(len(pos_templates))
    pos_templates = pos_templates.sort_values(
        ["template_distance", "tie_break"]
    ).reset_index(drop=True)

    selected_count = min(MAX_TEMPLATE_POOL_SIZE, len(pos_templates))
    selected_templates = pos_templates.head(selected_count).copy()
    selected_templates["match_rank"] = np.arange(1, len(selected_templates) + 1)
    distance_min = selected_templates["template_distance"].min()
    distance_max = selected_templates["template_distance"].max()
    distance_range = distance_max - distance_min
    if distance_range > 0:
        selected_templates["template_sample_weight"] = np.exp(
            -np.log(TEMPLATE_SAMPLE_TOP_TO_BOTTOM_RATIO)
            * (selected_templates["template_distance"] - distance_min)
            / distance_range
        )
    else:
        selected_templates["template_sample_weight"] = 1.0
    selected_templates["template_sample_prob"] = (
        selected_templates["template_sample_weight"]
        / selected_templates["template_sample_weight"].sum()
    )

    template_pool_key = player_row.template_pool_key
    exact_mask = (
        pos_templates["projection_distance"].eq(0)
        & pos_templates["exp_distance"].eq(0)
        & pos_templates["qb_team_rank_distance"].eq(0)
    )
    exact_templates = int(exact_mask.sum())

    pool_level = "nearest_weighted_features"

    member_cols = [
        "template_id",
        "pos",
        "projection_decile",
        "year_exp_bucket",
        "qb_team_rank_bucket",
        "projection_distance",
        "exp_distance",
        "qb_team_rank_distance",
        "cell_distance",
        "template_distance",
        "template_sample_weight",
        "template_sample_prob",
        "match_rank",
    ] + distance_cols
    pool_members = selected_templates[member_cols].copy()
    pool_members["template_pool_key"] = template_pool_key
    pool_members["league"] = player_row.version
    pool_members["pool_level"] = pool_level
    pool_members["pool_player"] = player_row.player
    pool_members["pool_year"] = player_row.year
    pool_members["pool_version"] = player_row.version
    pool_members["pool_dataset"] = player_row.dataset
    pool_members["template_league"] = selected_templates["league"].to_numpy()
    pool_members["target_projection_decile"] = target_decile
    pool_members["target_year_exp_bucket"] = target_exp
    pool_members["target_qb_team_rank_bucket"] = target_qb_team_rank_bucket

    pool_members = pool_members[
        [
            "template_pool_key",
            "league",
            "pool_level",
            "pool_player",
            "pool_year",
            "pool_version",
            "pool_dataset",
            "pos",
            "template_id",
            "template_league",
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
            "template_distance",
            "template_sample_weight",
            "template_sample_prob",
            "match_rank",
        ]
        + distance_cols
    ]

    selected_count = int(selected_templates["template_id"].nunique())
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
        "template_count": selected_count,
        "selected_cell_count": selected_count,
        "selected_neighbor_count": selected_count,
        "min_projection_decile": int(selected_templates["projection_decile"].min()),
        "max_projection_decile": int(selected_templates["projection_decile"].max()),
        "min_year_exp_bucket": int(selected_templates["year_exp_bucket"].min()),
        "max_year_exp_bucket": int(selected_templates["year_exp_bucket"].max()),
        "min_qb_team_rank_distance": int(selected_templates["qb_team_rank_distance"].min()),
        "max_qb_team_rank_distance": int(selected_templates["qb_team_rank_distance"].max()),
        "max_projection_distance": int(selected_templates["projection_distance"].max()),
        "max_exp_distance": int(selected_templates["exp_distance"].max()),
        "max_cell_distance": int(selected_templates["cell_distance"].max()),
        "min_template_distance": float(selected_templates["template_distance"].min()),
        "median_template_distance": float(selected_templates["template_distance"].median()),
        "mean_template_distance": float(selected_templates["template_distance"].mean()),
        "max_template_distance": float(selected_templates["template_distance"].max()),
        "max_to_min_template_sample_weight_ratio": TEMPLATE_SAMPLE_TOP_TO_BOTTOM_RATIO,
        "min_template_sample_prob": float(selected_templates["template_sample_prob"].min()),
        "max_template_sample_prob": float(selected_templates["template_sample_prob"].max()),
        "min_template_pool_size": MIN_TEMPLATE_POOL_SIZE,
        "max_template_pool_size": MAX_TEMPLATE_POOL_SIZE,
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
            "league",
            "template_id",
            "template_local_id",
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
            "active_ppg_resid",
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
            "league",
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

    if "template_league" in pool_members.columns and "league" in template_quality.columns:
        pool_quality = pool_members.merge(
            template_quality,
            left_on=["template_league", "template_id"],
            right_on=["league", "template_id"],
            how="left",
        )
    else:
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
            min_member_template_distance=("template_distance", "min"),
            median_member_template_distance=("template_distance", "median"),
            mean_member_template_distance=("template_distance", "mean"),
            max_member_template_distance=("template_distance", "max"),
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
        "selected_neighbor_count",
        "exact_cell_templates",
        "target_qb_team_rank_bucket",
        "min_projection_decile",
        "max_projection_decile",
        "min_year_exp_bucket",
        "max_year_exp_bucket",
        "min_qb_team_rank_distance",
        "max_qb_team_rank_distance",
        "min_template_distance",
        "median_template_distance",
        "mean_template_distance",
        "max_template_distance",
    ]
    player_cols = [col for col in player_cols if col in player_map.columns]
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
        select_cols = projection_select_cols(
            pos,
            year_alias="year",
            total_alias="current_avg_proj_points",
            avg_pick_alias="model_input_avg_pick",
        )
        df_pos = dm.read(
            f"""
            SELECT {", ".join(select_cols)}
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
    context_cols = [
        "player",
        "pos",
        "year",
        "team",
        "current_avg_proj_points",
        "avg_proj_points",
        "avg_pick",
        "year_exp",
        "qb_team_rank",
        "qb_team_rank_bucket",
    ] + PROJECTION_COMPONENT_COLS
    context_cols = [col for col in context_cols if col in current_context.columns]
    player_map = preds.merge(
        current_context[context_cols],
        on=["player", "pos", "year"],
        how="left",
    )
    player_map = add_exp_fields(player_map)
    player_map = add_template_match_features(
        player_map,
        group_cols=["year", "version", "dataset", "pos"],
        rank_pct_col="prediction_rank_pct",
        total_points_col="current_avg_proj_points",
    )
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
    cols = cols + [col for col in MATCH_OUTPUT_COLS if col in player_map.columns and col not in cols]
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
        "selected_neighbor_count",
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
        "min_template_distance",
        "median_template_distance",
        "mean_template_distance",
        "max_template_distance",
        "max_to_min_template_sample_weight_ratio",
        "min_template_sample_prob",
        "max_template_sample_prob",
        "min_template_pool_size",
        "max_template_pool_size",
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


def build_adp_audit(player_map):
    audit_cols = [
        "player",
        "pos",
        "team",
        "year",
        "version",
        "dataset",
        "pred_fp_per_game",
        "current_avg_proj_points",
        "avg_pick",
        "prediction_rank_pct",
        "projection_decile",
        "projection_tier",
    ]
    audit_cols = [col for col in audit_cols if col in player_map.columns]
    audit = player_map[audit_cols].copy()
    audit = audit.rename(columns={"avg_pick": "player_map_avg_pick"})
    audit["player_join_key"] = player_join_key(audit["player"]).values
    audit["pos_pred_rank"] = (
        audit.groupby(["year", "version", "dataset", "pos"])["pred_fp_per_game"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    current_context = load_current_player_context()
    context_cols = [
        "player",
        "pos",
        "year",
        "model_input_avg_pick",
        "adp_avg_pick",
        "avg_pick",
        "year_exp",
        "adp_year_exp",
    ]
    context_cols = [col for col in context_cols if col in current_context.columns]
    current_context = current_context[context_cols].rename(
        columns={
            "model_input_avg_pick": "projection_avg_pick",
            "adp_avg_pick": "pipeline_exact_adp_avg_pick",
            "avg_pick": "pipeline_context_avg_pick",
            "year_exp": "pipeline_year_exp",
            "adp_year_exp": "avg_adp_year_exp",
        }
    )
    audit = audit.merge(
        current_context,
        on=["player", "pos", "year"],
        how="left",
    )

    avg_adp = dm.read(
        f"""
        SELECT player avg_adp_player,
               CAST(year AS INTEGER) year,
               league,
               avg_pick avg_adp_pick,
               std_dev avg_adp_std_dev,
               min_pick avg_adp_min_pick,
               max_pick avg_adp_max_pick,
               Years_of_Experience avg_adp_year_exp_app_match
        FROM Avg_ADPs
        WHERE year={YEAR}
              AND league='{LEAGUE}'
        """,
        "Simulation",
    )

    if len(avg_adp) == 0:
        avg_adp = pd.DataFrame(
            columns=[
                "avg_adp_player",
                "year",
                "league",
                "avg_adp_pick",
                "avg_adp_std_dev",
                "avg_adp_min_pick",
                "avg_adp_max_pick",
                "avg_adp_year_exp_app_match",
                "avg_adp_join_key",
                "avg_adp_join_key_match_count",
            ]
        )
    else:
        avg_adp["avg_adp_join_key"] = player_join_key(avg_adp["avg_adp_player"]).values
        avg_adp_counts = (
            avg_adp.groupby("avg_adp_join_key", as_index=False)
            .agg(avg_adp_join_key_match_count=("avg_adp_player", "count"))
        )
        avg_adp = avg_adp.merge(avg_adp_counts, on="avg_adp_join_key", how="left")
        avg_adp = (
            avg_adp.sort_values(["avg_adp_join_key", "avg_adp_pick"])
            .drop_duplicates("avg_adp_join_key")
        )

    audit = audit.merge(
        avg_adp.drop(columns=["year"], errors="ignore"),
        left_on="player_join_key",
        right_on="avg_adp_join_key",
        how="left",
    )

    for col in [
        "player_map_avg_pick",
        "projection_avg_pick",
        "pipeline_exact_adp_avg_pick",
        "pipeline_context_avg_pick",
        "avg_adp_pick",
        "avg_adp_std_dev",
        "avg_adp_min_pick",
        "avg_adp_max_pick",
    ]:
        if col not in audit.columns:
            audit[col] = np.nan
        audit[col] = pd.to_numeric(audit[col], errors="coerce")

    audit["has_avg_adp_match"] = audit["avg_adp_pick"].notnull()
    audit["has_player_map_fallback"] = audit["player_map_avg_pick"].notnull()
    audit["app_avg_pick"] = (
        audit["avg_adp_pick"]
        .combine_first(audit["player_map_avg_pick"])
        .fillna(DEFAULT_ADP_PICK)
    )
    audit["adp_source"] = np.select(
        [
            audit["has_avg_adp_match"],
            audit["has_player_map_fallback"],
        ],
        [
            "avg_adp",
            "player_map_fallback",
        ],
        default="default_240",
    )
    audit["player_map_avg_pick_source"] = np.select(
        [
            audit["projection_avg_pick"].notnull(),
            audit["pipeline_exact_adp_avg_pick"].notnull(),
            audit["player_map_avg_pick"].notnull(),
        ],
        [
            "model_input_projection",
            "pipeline_exact_avg_adp",
            "unknown_player_map",
        ],
        default="missing",
    )
    audit["missing_avg_adp_match"] = ~audit["has_avg_adp_match"]
    audit["using_player_map_fallback"] = audit["adp_source"].eq("player_map_fallback")
    audit["using_default_adp"] = audit["adp_source"].eq("default_240")
    audit["duplicate_avg_adp_join_key"] = (
        audit["avg_adp_join_key_match_count"].fillna(0).astype(int).gt(1)
    )
    audit["pos_rank_review_limit"] = (
        audit["pos"].map(ADP_AUDIT_POS_RANK_LIMITS).fillna(0).astype(int)
    )
    audit["high_projection_player"] = (
        audit["pred_fp_per_game"].ge(ADP_AUDIT_HIGH_IMPACT_PPG_MIN)
        | audit["pos_pred_rank"].le(audit["pos_rank_review_limit"])
    )
    audit["high_impact_missing_avg_adp"] = (
        audit["missing_avg_adp_match"] & audit["high_projection_player"]
    )
    audit["needs_review"] = (
        audit["high_impact_missing_avg_adp"]
        | (audit["using_default_adp"] & audit["pred_fp_per_game"].gt(0))
        | audit["duplicate_avg_adp_join_key"]
    )

    def issue_type(row):
        issues = []
        if row.using_default_adp:
            issues.append("missing_avg_adp_default_240")
        elif row.using_player_map_fallback:
            issues.append("missing_avg_adp_player_map_fallback")
        if row.duplicate_avg_adp_join_key:
            issues.append("duplicate_avg_adp_join_key")
        if len(issues) == 0:
            issues.append("ok")
        return ",".join(issues)

    audit["issue_type"] = audit.apply(issue_type, axis=1)

    ordered_cols = [
        "player",
        "pos",
        "team",
        "year",
        "version",
        "dataset",
        "pred_fp_per_game",
        "pos_pred_rank",
        "current_avg_proj_points",
        "projection_decile",
        "projection_tier",
        "app_avg_pick",
        "adp_source",
        "player_map_avg_pick",
        "player_map_avg_pick_source",
        "projection_avg_pick",
        "pipeline_exact_adp_avg_pick",
        "avg_adp_pick",
        "avg_adp_player",
        "avg_adp_std_dev",
        "avg_adp_min_pick",
        "avg_adp_max_pick",
        "pipeline_year_exp",
        "avg_adp_year_exp",
        "avg_adp_year_exp_app_match",
        "player_join_key",
        "avg_adp_join_key",
        "avg_adp_join_key_match_count",
        "missing_avg_adp_match",
        "using_player_map_fallback",
        "using_default_adp",
        "duplicate_avg_adp_join_key",
        "high_projection_player",
        "high_impact_missing_avg_adp",
        "needs_review",
        "issue_type",
    ]
    ordered_cols = [col for col in ordered_cols if col in audit.columns]
    return audit[ordered_cols].sort_values(
        [
            "needs_review",
            "high_impact_missing_avg_adp",
            "using_default_adp",
            "pred_fp_per_game",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


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
    adp_audit = build_adp_audit(player_map)
    validate_weekly_template_audits(player_pool_audit)

    write_best_ball_tables(
        templates,
        pool_members,
        pool_summary,
        player_map,
        template_audit,
        player_pool_audit,
        bucket_audit,
        adp_audit,
    )

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

    print("\nADP audit summary:")
    adp_summary = (
        adp_audit.groupby("adp_source", as_index=False)
        .agg(
            players=("player", "count"),
            needs_review=("needs_review", "sum"),
            high_impact_missing_avg_adp=("high_impact_missing_avg_adp", "sum"),
            default_240=("using_default_adp", "sum"),
        )
        .sort_values(["needs_review", "players"], ascending=[False, False])
    )
    print(adp_summary)

    review_adp = adp_audit[adp_audit["needs_review"]]
    if len(review_adp) > 0:
        print("\nADP audit rows needing review:")
        print(
            review_adp[
                [
                    "player",
                    "pos",
                    "team",
                    "pred_fp_per_game",
                    "pos_pred_rank",
                    "app_avg_pick",
                    "adp_source",
                    "player_map_avg_pick_source",
                    "avg_adp_player",
                    "issue_type",
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
