#%%
import os
import hashlib
import re
import shutil
import sqlite3
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
TEMPLATE_KERNEL_BANDWIDTH = {
    "QB": 0.55,
    "RB": 0.45,
    "WR": 0.35,
    "TE": 0.40,
}
TEMPLATE_MIN_LOCAL_WEIGHT = 0.35
TEMPLATE_LOCAL_DISTANCE_SCALE = 1.50
TEMPLATE_MAX_SAMPLE_PROBABILITY = 0.05
TEMPLATE_RECENCY_HALF_LIFE = 12.0
PROJECTION_BUCKETS = 10
POOL_RANDOM_SEED = 20260702
VALIDATION_CURRENT_OR_NEXT_YEAR = "current"
EXPERIENCE_SCALE_YEARS = 10.0
PROJECTION_PPG_SCALE = 10.0
MAX_PLAUSIBLE_TEMPLATE_EXPERIENCE = 25

# Keep structurally non-transferable outcomes in the source/audit table while
# preventing them from becoming generic football-performance templates.
TEMPLATE_OUTCOME_EXCLUSIONS = {
    ("Le'Veon Bell", "RB", 2018): "contract_holdout",
}

# Name-only historical sources contain a small number of overlapping NFL
# careers. Team-aware draft matching resolves most of them; these traded-career
# rows need an explicit identity anchor after their original draft team changed.
EXPERIENCE_DRAFT_YEAR_OVERRIDES = {
    ("Steve Smith", "WR", "CAR"): 2001,
    ("Steve Smith", "WR", "BAL"): 2001,
    ("Zach Miller", "TE", "LVR"): 2007,
    ("Zach Miller", "TE", "SEA"): 2007,
    ("Zach Miller", "TE", "JAX"): 2009,
    ("Zach Miller", "TE", "CHI"): 2009,
}
TEAM_ALIASES = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "GB": "GNB",
    "JAC": "JAX",
    "KC": "KAN",
    "LA": "LAR",
    "LV": "LVR",
    "NE": "NWE",
    "NO": "NOR",
    "OAK": "LVR",
    "SD": "LAC",
    "SF": "SFO",
    "STL": "LAR",
    "WSH": "WAS",
}

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
PROJECTION_UNCERTAINTY_SOURCE_COLS = [
    "std_proj_points",
    "std_pos_rank",
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
    "match_projection_ppg_scaled": 1.5,
    "year_exp_scaled": 2.0,
    "adp_rank_pct": 0.5,
    "market_projection_gap": 0.75,
    "projection_disagreement_frac": 0.75,
    "rank_disagreement_scaled": 0.50,
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
        "rb_combined_share_of_room": 1.0,
        "rb_room_rank_scaled": 0.75,
        "rb_gap_to_next_share": 0.75,
        "rb_room_concentration": 0.50,
    },
    "WR": {
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "pass_catcher_rank_scaled": 0.75,
        "pass_catcher_gap_to_next_share": 0.75,
        "pass_catcher_room_concentration": 0.50,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
    "TE": {
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "pass_catcher_rank_scaled": 0.75,
        "pass_catcher_gap_to_next_share": 0.75,
        "pass_catcher_room_concentration": 0.50,
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
    "match_projection_ppg_scaled",
    "year_exp_scaled",
    "projection_x_exp",
    "adp_rank_pct",
    "market_projection_gap",
    "projection_disagreement_frac",
    "rank_disagreement_scaled",
    "rush_proj_rank_pct",
    "rec_proj_rank_pct",
    "pass_proj_rank_pct",
    "rush_share_of_own_points",
    "rec_share_of_own_points",
    "rb_rush_share_of_room",
    "rb_rec_share_of_room",
    "rb_combined_share_of_room",
    "rb_room_rank_scaled",
    "rb_gap_to_next_share",
    "rb_room_concentration",
    "team_rec_share",
    "pass_catcher_rank_scaled",
    "pass_catcher_gap_to_next_share",
    "pass_catcher_room_concentration",
    "team_qb_proj_points",
    "qb_room_share",
    "team_qb1_proj_points",
    "team_qb2_proj_points",
    "qb1_over_qb2_gap_pct",
    "team_qb_pass_points",
    "team_qb_pass_proj_rank_pct",
] + PROJECTION_COMPONENT_COLS + PROJECTION_UNCERTAINTY_SOURCE_COLS


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


def canonical_team(team):
    if pd.isna(team):
        return None
    team = str(team).strip().upper()
    return TEAM_ALIASES.get(team, team)


def load_uncapped_experience_reference():
    drafts = dm.read(
        """
        SELECT player,
               pos,
               CAST(year AS INTEGER) draft_year,
               team draft_team
        FROM Draft_Positions
        WHERE pos IN ('QB', 'RB', 'WR', 'TE')
        """,
        "Season_Stats_New",
    )
    drafts = clean_player_names(drafts)
    drafts["draft_team_key"] = drafts["draft_team"].apply(canonical_team)
    drafts = drafts.drop_duplicates(
        ["player", "pos", "draft_year", "draft_team_key"]
    )

    debut_frames = []
    for pos in POSITIONS:
        debut = dm.read(
            f"""
            SELECT player,
                   '{pos}' pos,
                   MIN(CAST(season AS INTEGER)) debut_season
            FROM {pos}_Stats
            GROUP BY player
            """,
            "Season_Stats_New",
        )
        debut_frames.append(debut)
    debuts = clean_player_names(pd.concat(debut_frames, ignore_index=True))
    debuts = (
        debuts.groupby(["player", "pos"], as_index=False)
        .debut_season.min()
    )
    return drafts, debuts


def attach_uncapped_template_experience(df, season_col):
    """Replace capped model tenure with a collision-aware raw career year."""
    output = df.reset_index(drop=True).copy()
    output["source_year_exp"] = pd.to_numeric(
        output.get("year_exp"), errors="coerce"
    )
    output["_experience_row_id"] = np.arange(len(output))
    output["_experience_team_key"] = output.get(
        "team", pd.Series(index=output.index, dtype=object)
    ).apply(canonical_team)

    drafts, debuts = load_uncapped_experience_reference()
    candidates = output[
        [
            "_experience_row_id",
            "player",
            "pos",
            season_col,
            "_experience_team_key",
        ]
    ].merge(drafts, on=["player", "pos"], how="left")
    candidates = candidates[
        candidates.draft_year.le(candidates[season_col])
    ].copy()
    candidates["draft_team_match"] = (
        candidates.draft_team_key.eq(candidates._experience_team_key)
        & candidates.draft_team_key.notna()
    )
    chosen = (
        candidates.sort_values(
            ["_experience_row_id", "draft_team_match", "draft_year"],
            ascending=[True, False, False],
        )
        .drop_duplicates("_experience_row_id", keep="first")
        [["_experience_row_id", "draft_year", "draft_team_match"]]
    )
    output = output.merge(chosen, on="_experience_row_id", how="left")

    override_keys = list(
        zip(output.player, output.pos, output._experience_team_key)
    )
    override_year = pd.Series(
        [EXPERIENCE_DRAFT_YEAR_OVERRIDES.get(key) for key in override_keys],
        index=output.index,
        dtype=float,
    )
    override_valid = override_year.notna() & override_year.le(output[season_col])
    output.loc[override_valid, "draft_year"] = override_year.loc[override_valid]
    output.loc[override_valid, "draft_team_match"] = True

    output = output.merge(debuts, on=["player", "pos"], how="left")
    origin_year = output.draft_year.fillna(output.debut_season)
    reconstructed = pd.to_numeric(output[season_col], errors="coerce") - origin_year
    reconstructed = reconstructed.where(reconstructed.ge(0))
    implausible = reconstructed.gt(MAX_PLAUSIBLE_TEMPLATE_EXPERIENCE)
    reconstructed = reconstructed.mask(implausible)

    output["year_exp"] = reconstructed.combine_first(output.source_year_exp)
    draft_team_match = output.draft_team_match.eq(True)
    output["year_exp_source"] = np.select(
        [
            override_valid,
            output.draft_year.notna() & draft_team_match,
            output.draft_year.notna(),
            output.debut_season.notna(),
            output.source_year_exp.notna(),
        ],
        [
            "draft_identity_override",
            "draft_team_match",
            "draft_name_match",
            "debut_fallback",
            "model_input_fallback",
        ],
        default="missing",
    )
    output.loc[implausible & output.source_year_exp.notna(), "year_exp_source"] = (
        "implausible_reconstruction_model_fallback"
    )
    output["year_exp_uncapped_delta"] = (
        output.year_exp - output.source_year_exp
    )
    return output.drop(
        columns=[
            "_experience_row_id",
            "_experience_team_key",
            "draft_year",
            "draft_team_match",
            "debut_season",
        ]
    )


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


def add_fantasy_points(df, pos, filter_qb_workload=True):
    df = add_bonus_cols(df)

    df = calc_fp(df, get_scoring_dict("rush"), "fantasy_pts_rush")

    if pos == "QB":
        df = calc_fp(df, get_scoring_dict("passing"), "fantasy_pts_pass")
        df["fantasy_pts"] = df["fantasy_pts_rush"] + df["fantasy_pts_pass"]

        df = add_missing_cols(df, ["pass_qb_dropback_sum", "rush_rush_attempt_sum"])
        df["total_plays"] = df["pass_qb_dropback_sum"] + df["rush_rush_attempt_sum"]
        if filter_qb_workload:
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
    cols.extend(
        col
        for col in PROJECTION_UNCERTAINTY_SOURCE_COLS
        if col in available_cols
    )
    return cols


def add_projection_component_cols(df):
    df = df.copy()
    for col in PROJECTION_COMPONENT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def add_projection_uncertainty_cols(df, total_points_col, group_cols):
    """Create scale-free disagreement features from preseason sources."""
    df = df.copy()
    for col in PROJECTION_UNCERTAINTY_SOURCE_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total_points = pd.to_numeric(df[total_points_col], errors="coerce").abs()
    df["projection_disagreement_frac"] = safe_ratio(
        df["std_proj_points"],
        total_points,
        fill_value=np.nan,
    )
    group_size = df.groupby(group_cols)["player"].transform("size").clip(lower=1)
    df["rank_disagreement_scaled"] = (
        df["std_pos_rank"] / group_size
    ).clip(lower=0, upper=1)

    for col in ["projection_disagreement_frac", "rank_disagreement_scaled"]:
        group_median = df.groupby(group_cols)[col].transform("median")
        df[col] = df[col].fillna(group_median).fillna(MATCH_FILL_VALUE)
    return df


def add_room_structure_features(
    df,
    mask,
    team_group_cols,
    value_col,
    prefix,
):
    """Attach player share, within-room rank/gap, and room concentration."""
    df = df.copy()
    share_col = f"{prefix}_share_of_room"
    rank_col = f"{prefix}_rank_scaled"
    gap_col = f"{prefix}_gap_to_next_share"
    concentration_col = f"{prefix}_room_concentration"
    for col in [share_col, rank_col, gap_col, concentration_col]:
        df[col] = 0.0

    if not mask.any():
        return df

    room = df.loc[mask, team_group_cols].copy()
    room["__value"] = pd.to_numeric(
        df.loc[mask, value_col], errors="coerce"
    ).fillna(0).clip(lower=0).to_numpy()
    grouped = room.groupby(team_group_cols, sort=False)["__value"]
    room["__total"] = grouped.transform("sum")
    room["__share"] = safe_ratio(room["__value"], room["__total"])
    room["__size"] = grouped.transform("size")
    room["__rank"] = grouped.rank(method="min", ascending=False)
    room["__rank_scaled"] = np.where(
        room["__size"].gt(1),
        (room["__rank"] - 1) / (room["__size"] - 1),
        0.0,
    )
    room["__top"] = grouped.transform("max")
    room["__second"] = grouped.transform(
        lambda values: (
            values.nlargest(2).iloc[-1]
            if len(values) > 1 else 0.0
        )
    )
    room["__competitor"] = np.where(
        room["__value"].eq(room["__top"]),
        room["__second"],
        room["__top"],
    )
    room["__gap"] = (
        room["__value"] - room["__competitor"]
    ) / room["__total"].replace(0, np.nan)
    room["__gap"] = room["__gap"].replace([np.inf, -np.inf], np.nan).fillna(0)
    room["__share_sq"] = room["__share"] ** 2
    room["__concentration"] = room.groupby(
        team_group_cols, sort=False
    )["__share_sq"].transform("sum")

    df.loc[mask, share_col] = room["__share"].to_numpy()
    df.loc[mask, rank_col] = room["__rank_scaled"].to_numpy()
    df.loc[mask, gap_col] = room["__gap"].clip(-1, 1).to_numpy()
    df.loc[mask, concentration_col] = room["__concentration"].to_numpy()
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


def add_template_match_features(
    df,
    group_cols,
    rank_pct_col,
    total_points_col,
    projection_ppg_col,
):
    df = add_projection_component_cols(df)
    df = add_projection_uncertainty_cols(df, total_points_col, group_cols)
    df = df.copy()

    df["match_projection_rank_pct"] = (
        pd.to_numeric(df[rank_pct_col], errors="coerce").fillna(MATCH_FILL_VALUE)
    )
    df["year_exp_scaled"] = (
        pd.to_numeric(df["year_exp"], errors="coerce")
        .clip(lower=0)
        .div(EXPERIENCE_SCALE_YEARS)
        .fillna(MATCH_FILL_VALUE)
    )
    df["match_projection_ppg_scaled"] = (
        pd.to_numeric(df[projection_ppg_col], errors="coerce")
        .clip(lower=0)
        .div(PROJECTION_PPG_SCALE)
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
    df["market_projection_gap"] = (
        df["adp_rank_pct"] - df["match_projection_rank_pct"]
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

    df["__rb_combined_points"] = (
        df["avg_proj_rush_points"] + df["avg_proj_rec_points"]
    )
    df = add_room_structure_features(
        df,
        rb_mask,
        team_group_cols,
        "__rb_combined_points",
        "rb_combined",
    )
    df = df.rename(
        columns={
            "rb_combined_rank_scaled": "rb_room_rank_scaled",
            "rb_combined_gap_to_next_share": "rb_gap_to_next_share",
            "rb_combined_room_concentration": "rb_room_concentration",
        }
    )

    pass_catcher_mask = df["pos"].isin(["WR", "TE"])
    df["team_rec_share"] = 0.0
    df.loc[pass_catcher_mask, "team_rec_share"] = safe_ratio(
        df.loc[pass_catcher_mask, "avg_proj_rec_points"],
        df.loc[pass_catcher_mask, "team_rec_points"],
    )
    df = add_room_structure_features(
        df,
        receiver_mask,
        team_group_cols,
        "avg_proj_rec_points",
        "pass_catcher",
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

    return df.drop(columns=["__rb_combined_points"], errors="ignore")


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


def cap_probability_vector(probabilities, max_probability):
    """Cap any one donor while preserving a normalized probability vector."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    if len(probabilities) * max_probability < 1 - 1e-12:
        raise ValueError("Probability cap is too small for the number of donors.")

    capped = probabilities.copy()
    for _ in range(len(capped)):
        over = capped > max_probability
        if not over.any():
            break
        capped[over] = max_probability
        under = ~over
        remaining = 1.0 - capped[over].sum()
        under_total = capped[under].sum()
        if under_total <= 0:
            capped[under] = remaining / under.sum()
        else:
            capped[under] *= remaining / under_total
    return capped / capped.sum()


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
    proj = attach_uncapped_template_experience(proj, season_col="season")
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
        projection_ppg_col="historical_pred_fp_per_game",
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
        played_pos = df_pos[["player", "season", "week"]].drop_duplicates().copy()
        played_pos["pos"] = pos
        played_pos["played_week"] = True

        scored_pos = add_fantasy_points(
            df_pos,
            pos,
            filter_qb_workload=False,
        )
        scored_pos["pos"] = pos
        managed_pos = (
            scored_pos.groupby(["player", "pos", "season", "week"], as_index=False)
            .agg({"fantasy_pts": "sum"})
            .rename(columns={"fantasy_pts": "managed_fantasy_pts"})
        )
        if pos == "QB":
            scored_pos = scored_pos[scored_pos["total_plays"] > 15]
        scored_pos = (
            scored_pos.groupby(["player", "pos", "season", "week"], as_index=False)
            .agg({"fantasy_pts": "sum"})
        )
        # QB fantasy-point profiles intentionally retain the historical
        # >15-play workload filter, but participation must be derived before
        # that filter so short/injury-truncated appearances are not treated as
        # freely replaceable absences by managed-season scoring.
        df_pos = played_pos.merge(
            managed_pos,
            on=["player", "pos", "season", "week"],
            how="left",
            validate="one_to_one",
        ).merge(
            scored_pos,
            on=["player", "pos", "season", "week"],
            how="left",
            validate="one_to_one",
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
        "source_year_exp",
        "year_exp_source",
        "year_exp_uncapped_delta",
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
    expanded = template_index[
        ["template_id", "player", "pos", "season", "historical_pred_fp_per_game"]
    ].merge(
        week_grid, how="cross"
    )

    weekly = weekly.copy()
    if "played_week" not in weekly.columns:
        # Backward-compatible fixture/legacy input: any scored source row is
        # also evidence that the player participated.
        weekly["played_week"] = weekly["fantasy_pts"].notna()
    if "managed_fantasy_pts" not in weekly.columns:
        weekly["managed_fantasy_pts"] = weekly["fantasy_pts"]
    weekly_keys = ["player", "pos", "season", "week"]
    duplicate_weekly = weekly.duplicated(weekly_keys, keep=False)
    if duplicate_weekly.any():
        preview = weekly.loc[duplicate_weekly, weekly_keys].head(10).to_dict("records")
        raise ValueError(f"Weekly source rows are not unique by player-week: {preview}")

    expanded = expanded.merge(
        weekly,
        on=weekly_keys,
        how="left",
        validate="many_to_one",
    )
    expanded["active_week"] = expanded["fantasy_pts"].notna()
    expanded["played_week"] = expanded["played_week"].eq(True)
    active_without_played = expanded["active_week"] & ~expanded["played_week"]
    if active_without_played.any():
        preview = expanded.loc[active_without_played, weekly_keys].head(10).to_dict(
            "records"
        )
        raise ValueError(
            f"Weekly performance rows are missing played evidence: {preview}"
        )
    expanded["managed_fantasy_pts"] = expanded["managed_fantasy_pts"].combine_first(
        expanded["fantasy_pts"]
    )
    expanded["fantasy_pts"] = expanded["fantasy_pts"].fillna(0)
    expanded["managed_fantasy_pts"] = expanded["managed_fantasy_pts"].fillna(0)

    season_stats = (
        expanded.groupby("template_id", as_index=False)
        .agg(
            active_games=("active_week", "sum"),
            played_games=("played_week", "sum"),
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
    expanded["managed_profile_ppg"] = np.where(
        expanded["active_ppg"] > 0,
        expanded["active_ppg"],
        expanded["historical_pred_fp_per_game"],
    )
    expanded["managed_week_profile"] = np.where(
        expanded["managed_profile_ppg"] > 0,
        expanded["managed_fantasy_pts"] / expanded["managed_profile_ppg"],
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

    managed_week_profiles = expanded.pivot_table(
        index="template_id",
        columns="week",
        values="managed_week_profile",
        aggfunc="sum",
        fill_value=0,
    )
    managed_week_profiles = managed_week_profiles.reindex(columns=WEEKS, fill_value=0)
    managed_week_profiles.columns = [f"managed_week_{w}" for w in WEEKS]
    managed_week_profiles = managed_week_profiles.reset_index()

    played_profiles = expanded.pivot_table(
        index="template_id",
        columns="week",
        values="played_week",
        aggfunc="max",
        fill_value=False,
    )
    played_profiles = played_profiles.reindex(columns=WEEKS, fill_value=False)
    played_profiles.columns = [f"played_week_{w}" for w in WEEKS]
    played_profiles = played_profiles.astype(np.int8).reset_index()

    templates = (
        template_index.merge(season_stats, on="template_id", how="left")
        .merge(week_profiles, on="template_id", how="left")
        .merge(managed_week_profiles, on="template_id", how="left")
        .merge(played_profiles, on="template_id", how="left")
    )
    templates["active_ppg_resid"] = (
        templates["active_ppg"] - templates["historical_pred_fp_per_game"]
    )
    templates["profile_total"] = templates[[f"week_{w}" for w in WEEKS]].sum(axis=1)
    templates["managed_profile_total"] = templates[
        [f"managed_week_{w}" for w in WEEKS]
    ].sum(axis=1)
    exclusion_keys = zip(templates["player"], templates["pos"], templates["season"])
    templates["template_exclusion_reason"] = [
        TEMPLATE_OUTCOME_EXCLUSIONS.get(tuple(key), "")
        for key in exclusion_keys
    ]
    templates["template_eligible"] = templates[
        "template_exclusion_reason"
    ].eq("").astype(np.int8)

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
        "source_year_exp",
        "year_exp_source",
        "year_exp_uncapped_delta",
        "year_exp_bucket",
        "exp_bucket",
        "qb_team_rank",
        "qb_team_rank_bucket",
        "projection_rank_pct",
        "projection_decile",
        "projection_tier",
        "active_games",
        "played_games",
        "active_ppg",
        "active_ppg_resid",
        "season_points",
        "profile_total",
        "managed_profile_total",
        "template_eligible",
        "template_exclusion_reason",
    ]
    match_cols = [
        col for col in MATCH_OUTPUT_COLS if col in templates.columns and col not in front_cols
    ]
    week_cols = [f"week_{w}" for w in WEEKS]
    managed_week_cols = [f"managed_week_{w}" for w in WEEKS]
    played_cols = [f"played_week_{w}" for w in WEEKS]
    played_values = templates[played_cols].to_numpy()
    if np.any(pd.isna(played_values)):
        raise ValueError("Weekly template played masks contain missing values.")
    if not np.isin(played_values, [0, 1]).all():
        raise ValueError("Weekly template played masks must contain only 0 or 1.")
    played_games = played_values.sum(axis=1)
    if not np.array_equal(
        played_games.astype(int),
        templates["played_games"].to_numpy(dtype=int),
    ):
        raise ValueError(
            "Weekly template played-mask counts disagree with played_games."
        )
    active_values = templates["active_games"].to_numpy(dtype=int)
    played_values_count = templates["played_games"].to_numpy(dtype=int)
    if np.any(active_values > played_values_count):
        raise ValueError("Weekly templates contain more active than played games.")
    non_qb_mismatch = templates["pos"].ne("QB") & (
        templates["active_games"] != templates["played_games"]
    )
    if non_qb_mismatch.any():
        preview = ", ".join(templates.loc[non_qb_mismatch, "player"].head(10))
        raise ValueError(
            "Non-QB weekly templates disagree on active and played games: "
            f"{preview}"
        )
    managed_values = templates[managed_week_cols].to_numpy(dtype=float)
    if not np.isfinite(managed_values).all():
        raise ValueError("Managed weekly template profiles contain non-finite values.")
    return templates[
        front_cols + match_cols + week_cols + managed_week_cols + played_cols
    ]


def select_player_template_pool(player_row, templates):
    pos_templates = templates[
        templates["pos"].eq(player_row.pos)
        & templates["template_eligible"].eq(1)
    ].copy()

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
    distance_min = float(selected_templates["template_distance"].min())
    bandwidth = TEMPLATE_KERNEL_BANDWIDTH[player_row.pos]
    selected_templates["template_sample_weight"] = np.exp(
        -(selected_templates["template_distance"] - distance_min) / bandwidth
    )
    local_prob = (
        selected_templates["template_sample_weight"]
        / selected_templates["template_sample_weight"].sum()
    ).to_numpy(dtype=float)
    local_weight_fraction = max(
        TEMPLATE_MIN_LOCAL_WEIGHT,
        np.exp(-distance_min / TEMPLATE_LOCAL_DISTANCE_SCALE),
    )
    local_weight_fraction = min(float(local_weight_fraction), 1.0)
    uniform_prob = np.full(len(selected_templates), 1.0 / len(selected_templates))
    sample_prob = (
        local_weight_fraction * local_prob
        + (1.0 - local_weight_fraction) * uniform_prob
    )
    selected_templates["template_sample_prob"] = cap_probability_vector(
        sample_prob,
        TEMPLATE_MAX_SAMPLE_PROBABILITY,
    )
    selected_templates["template_season_gap"] = (
        int(player_row.year)
        - pd.to_numeric(selected_templates["season"], errors="raise").astype(int)
    )
    if selected_templates["template_season_gap"].le(0).any():
        raise ValueError(
            "Weekly template pools must contain only seasons before the target year."
        )
    selected_templates["template_recency_multiplier"] = np.power(
        0.5,
        selected_templates["template_season_gap"]
        / TEMPLATE_RECENCY_HALF_LIFE,
    )
    recency_prob = (
        selected_templates["template_sample_prob"]
        * selected_templates["template_recency_multiplier"]
    )
    selected_templates["template_sample_prob"] = cap_probability_vector(
        recency_prob.to_numpy(dtype=float),
        TEMPLATE_MAX_SAMPLE_PROBABILITY,
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
        "season",
        "template_season_gap",
        "template_recency_multiplier",
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
            "season",
            "template_season_gap",
            "template_recency_multiplier",
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
        "kernel_bandwidth": bandwidth,
        "local_weight_fraction": local_weight_fraction,
        "recency_half_life": TEMPLATE_RECENCY_HALF_LIFE,
        "weighted_template_season_gap": float(
            np.average(
                selected_templates["template_season_gap"],
                weights=selected_templates["template_sample_prob"],
            )
        ),
        "weight_last3_seasons": float(
            selected_templates.loc[
                selected_templates["template_season_gap"].le(3),
                "template_sample_prob",
            ].sum()
        ),
        "weight_10plus_seasons": float(
            selected_templates.loc[
                selected_templates["template_season_gap"].ge(10),
                "template_sample_prob",
            ].sum()
        ),
        "effective_sample_size": float(
            1.0 / np.square(selected_templates["template_sample_prob"]).sum()
        ),
        "max_to_min_template_sample_weight_ratio": float(
            selected_templates["template_sample_prob"].max()
            / selected_templates["template_sample_prob"].min()
        ),
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
            "source_year_exp",
            "year_exp_source",
            "year_exp_uncapped_delta",
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
            "played_games",
            "active_ppg",
            "active_ppg_resid",
            "season_points",
            "profile_total",
            "managed_profile_total",
            "template_eligible",
            "template_exclusion_reason",
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
    played_cols = [f"played_week_{w}" for w in WEEKS]
    score_cols = [f"managed_week_{w}" for w in WEEKS]
    played_values = templates[played_cols].to_numpy(dtype=np.int8)
    score_values = templates[score_cols].to_numpy(dtype=float)
    audit["played_mask_games"] = played_values.sum(axis=1)
    audit["played_mask_abs_error"] = (
        audit["played_mask_games"] - audit["played_games"]
    ).abs()
    audit["played_mask_mismatch"] = audit["played_mask_abs_error"].gt(0)
    audit["played_only_games"] = audit["played_games"] - audit["active_games"]
    audit["active_exceeds_played"] = audit["active_games"].gt(audit["played_games"])
    audit["non_qb_played_active_mismatch"] = (
        audit["pos"].ne("QB") & audit["played_only_games"].ne(0)
    )
    audit["played_zero_profile_weeks"] = (
        (played_values == 1) & np.isclose(score_values, 0)
    ).sum(axis=1)
    audit["played_negative_profile_weeks"] = (
        (played_values == 1) & (score_values < 0)
    ).sum(axis=1)

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


def validate_weekly_template_audits(player_pool_audit, template_audit=None):
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

    if template_audit is not None:
        excluded = template_audit[template_audit["template_eligible"].eq(0)]
        expected_exclusions = {
            key: reason
            for key, reason in TEMPLATE_OUTCOME_EXCLUSIONS.items()
            if key[2] <= template_audit["season"].max()
        }
        actual_exclusions = {
            (row.player, row.pos, int(row.season)): row.template_exclusion_reason
            for row in excluded.itertuples(index=False)
        }
        if actual_exclusions != expected_exclusions:
            raise ValueError(
                "Weekly template exclusions differ from the declared registry: "
                f"expected={expected_exclusions}, actual={actual_exclusions}"
            )
        mask_mismatch = template_audit[template_audit["played_mask_mismatch"]]
        if len(mask_mismatch) > 0:
            preview = ", ".join(mask_mismatch.player.head(10))
            raise ValueError(
                f"{len(mask_mismatch)} weekly templates have played-mask "
                f"counts that disagree with played_games: {preview}"
            )
        active_exceeds_played = template_audit[template_audit["active_exceeds_played"]]
        if len(active_exceeds_played) > 0:
            preview = ", ".join(active_exceeds_played.player.head(10))
            raise ValueError(
                f"{len(active_exceeds_played)} weekly templates have more active "
                f"than played games: {preview}"
            )
        non_qb_mismatch = template_audit[
            template_audit["non_qb_played_active_mismatch"]
        ]
        if len(non_qb_mismatch) > 0:
            preview = ", ".join(non_qb_mismatch.player.head(10))
            raise ValueError(
                f"{len(non_qb_mismatch)} non-QB templates disagree on active and "
                f"played games: {preview}"
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
    current_context = attach_uncapped_template_experience(
        current_context,
        season_col="year",
    )
    current_context = add_exp_fields(current_context)
    current_context["current_projection_ppg"] = (
        pd.to_numeric(
            current_context["current_avg_proj_points"], errors="coerce"
        ) / WEEK_COUNT
    )
    current_context = add_projection_buckets(
        current_context,
        value_col="current_projection_ppg",
        group_cols=["year", "pos"],
        pct_col="context_projection_rank_pct",
    )
    current_context = add_template_match_features(
        current_context,
        group_cols=["year", "pos"],
        rank_pct_col="context_projection_rank_pct",
        total_points_col="current_avg_proj_points",
        projection_ppg_col="current_projection_ppg",
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
        "source_year_exp",
        "year_exp_source",
        "year_exp_uncapped_delta",
        "qb_team_rank",
        "qb_team_rank_bucket",
    ] + MATCH_OUTPUT_COLS
    context_cols = [col for col in context_cols if col in current_context.columns]
    player_map = preds.merge(
        current_context[context_cols],
        on=["player", "pos", "year"],
        how="left",
    )
    player_map = add_exp_fields(player_map)
    # Team workload structure is computed on the complete preseason universe
    # before the final prediction table is pruned. Only projection-level fields
    # are rebased to the final model here.
    player_map["match_projection_rank_pct"] = player_map[
        "prediction_rank_pct"
    ]
    player_map["match_projection_ppg_scaled"] = (
        pd.to_numeric(player_map["pred_fp_per_game"], errors="coerce")
        .clip(lower=0)
        .div(PROJECTION_PPG_SCALE)
        .fillna(MATCH_FILL_VALUE)
    )
    player_map["projection_x_exp"] = (
        player_map["match_projection_rank_pct"]
        * player_map["year_exp_scaled"]
    )
    player_map["market_projection_gap"] = (
        player_map["adp_rank_pct"]
        - player_map["match_projection_rank_pct"]
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
        "source_year_exp",
        "year_exp_source",
        "year_exp_uncapped_delta",
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
        "kernel_bandwidth",
        "local_weight_fraction",
        "recency_half_life",
        "weighted_template_season_gap",
        "weight_last3_seasons",
        "weight_10plus_seasons",
        "effective_sample_size",
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
    generated_tables = [
        TEMPLATE_TABLE,
        POOL_TABLE,
        POOL_SUMMARY_TABLE,
        PLAYER_MAP_TABLE,
        TEMPLATE_AUDIT_TABLE,
        PLAYER_POOL_AUDIT_TABLE,
        BUCKET_AUDIT_TABLE,
        ADP_AUDIT_TABLE,
    ]
    required_template_cols = (
        [f"managed_week_{week}" for week in WEEKS]
        + [f"played_week_{week}" for week in WEEKS]
    )
    with sqlite3.connect(src) as source_conn:
        template_cols = {
            row[1]
            for row in source_conn.execute(
                f'PRAGMA table_info("{TEMPLATE_TABLE}")'
            )
        }
        missing_cols = sorted(set(required_template_cols) - template_cols)
        if missing_cols:
            print(
                "Skipped app database export because the weekly-template schema "
                f"is incomplete: {', '.join(missing_cols)}"
            )
            return
        null_predicate = " OR ".join(
            f'"{column}" IS NULL' for column in required_template_cols
        )
        incomplete_rows = source_conn.execute(
            f'SELECT COUNT(*) FROM "{TEMPLATE_TABLE}" WHERE {null_predicate}'
        ).fetchone()[0]
        if incomplete_rows:
            print(
                "Skipped app database export because "
                f"{incomplete_rows} retained template rows still have null "
                "played/managed-week fields. Rebuild every retained league slice."
            )
            return

    sibling_root = Path(root_path).resolve().parent
    auction_dst = sibling_root / "Fantasy_Football_App" / "app" / "Simulation.sqlite3"
    if not auction_dst.parent.exists():
        auction_dst = Path(
            "/Users/borys/OneDrive/Documents/Github/"
            "Fantasy_Football_App/app/Simulation.sqlite3"
        )
    if auction_dst.parent.exists():
        if not auction_dst.exists():
            shutil.copyfile(src, auction_dst)
            print(f"Copied Simulation.sqlite3 to {auction_dst}")
        else:
            with sqlite3.connect(auction_dst) as app_conn:
                app_conn.execute("ATTACH DATABASE ? AS source_db", (str(src),))
                app_conn.execute("BEGIN IMMEDIATE")
                try:
                    for table_name in generated_tables:
                        create_sql = app_conn.execute(
                            "SELECT sql FROM source_db.sqlite_master "
                            "WHERE type='table' AND name=?",
                            (table_name,),
                        ).fetchone()[0]
                        app_conn.execute(
                            f'DROP TABLE IF EXISTS main."{table_name}"'
                        )
                        app_conn.execute(create_sql)
                        app_conn.execute(
                            f'INSERT INTO main."{table_name}" '
                            f'SELECT * FROM source_db."{table_name}"'
                        )
                    app_conn.commit()
                except Exception:
                    app_conn.rollback()
                    raise
            print(
                f"Synchronized {len(generated_tables)} generated best-ball "
                f"tables to {auction_dst}"
            )

    snake_dst = sibling_root / "Fantasy_Football_Snake" / "app" / "Simulation.sqlite3"
    if not snake_dst.parent.exists():
        snake_dst = Path(
            "/Users/borys/OneDrive/Documents/Github/"
            "Fantasy_Football_Snake/app/Simulation.sqlite3"
        )
    if snake_dst.parent.exists():
        shutil.copyfile(src, snake_dst)
        print(f"Copied Simulation.sqlite3 to {snake_dst}")


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
    validate_weekly_template_audits(player_pool_audit, template_audit)

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

    print("\nUncapped historical template experience:")
    print(
        templates.groupby("pos", as_index=False).agg(
            templates=("template_id", "count"),
            min_year_exp=("year_exp", "min"),
            max_year_exp=("year_exp", "max"),
            above_ten=("year_exp", lambda values: int(values.gt(10).sum())),
            mean_uncapped_delta=("year_exp_uncapped_delta", "mean"),
        )
    )
    print("\nHistorical experience reconstruction sources:")
    print(templates.groupby(["pos", "year_exp_source"]).template_id.count())

    named_veterans = player_map[
        player_map.player.isin(
            ["Derrick Henry", "Alvin Kamara", "George Kittle", "Travis Kelce"]
        )
    ]
    if len(named_veterans) > 0:
        print("\nNamed current-player uncapped experience:")
        print(
            named_veterans[
                [
                    "player",
                    "pos",
                    "year_exp",
                    "source_year_exp",
                    "year_exp_source",
                    "year_exp_uncapped_delta",
                    "year_exp_scaled",
                    "min_year_exp_bucket",
                    "max_year_exp_bucket",
                ]
            ].sort_values(["pos", "player"])
        )

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
            played_only_games=("played_only_games", "sum"),
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
