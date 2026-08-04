#%%
import argparse
import json
import os
import hashlib
import re
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

# Add Scripts directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import YEAR, LEAGUE, PRED_VERSION, get_scoring_dict

from ff.db_operations import DataManage
from ff import general
import ff.data_clean as dc

from Scripts.V2.config import (
    OUTPUT_DB_PATH as V2_IDENTITY_DB_PATH,
    SOURCE_ROW_EXCLUSIONS,
)
from Scripts.V2.contracts import (
    publish_tables_atomic,
    scoring_hash,
    source_row_exclusion_policy_receipt,
)
from Scripts.V2.production_handoff import (
    AVG_ADP_AUDIT_TABLE,
    AVG_ADP_RECEIPT_TABLE,
    AVG_ADP_TABLE,
    V2_DATABASES,
    validate_avg_adp_publication,
)
from Scripts.V2.production_cycle import get_production_cycle
from Scripts.V2.template_identity import attach_v2_player_keys


#==========
# Settings
#==========

POSITIONS = ["QB", "RB", "WR", "TE"]
WEEK_COUNT_BY_LEAGUE = {
    "beta": 16,
    "dk": 16,
    "nffc": 17,
    "nv": 16,
}
TEMPLATE_SEASON_MIN_BY_LEAGUE = {
    "beta": 2008,
    "dk": 2008,
    # A 17-week NFFC profile cannot treat pre-2021 16-game schedules as if a
    # missing Week 17 were an observed zero. Use direct 17-week-era donors.
    "nffc": 2021,
    "nv": 2008,
}
_PRODUCTION_CYCLE = get_production_cycle(YEAR)
WEEK_COUNT_BY_LEAGUE.update(_PRODUCTION_CYCLE.weekly_horizons)
TEMPLATE_SEASON_MIN_BY_LEAGUE.update(
    _PRODUCTION_CYCLE.template_min_seasons
)
WEEK_COUNT = WEEK_COUNT_BY_LEAGUE.get(LEAGUE, 16)
WEEKS = list(range(1, WEEK_COUNT + 1))

TEMPLATE_SEASON_MIN = TEMPLATE_SEASON_MIN_BY_LEAGUE.get(LEAGUE, 2008)
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
    "BLT": "BAL",
    "CLV": "CLE",
    "GB": "GNB",
    "HST": "HOU",
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
    "TAM": "TB",
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
V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN = (
    "v2_template_center_unavailable_reason"
)
V2_TEMPLATE_CENTER_POSITION_COLUMN = "v2_template_center_position"
V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN = (
    "v2_template_center_position_mismatch"
)
V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN = (
    "v2_template_center_position_mismatch_reason"
)
BETA_2018_QB_CENTER_FALLBACK_EXCLUSION_ID = (
    "fftoday_qb_stored_2018_2019_vintage_quarantine_v1"
)
BETA_2018_QB_CENTER_FALLBACK_REASON = (
    "legacy_validated_oos_fallback:"
    f"{BETA_2018_QB_CENTER_FALLBACK_EXCLUSION_ID}:"
    "no_valid_beta_qb_sack_donor"
)
GOVERNED_V2_TEMPLATE_CENTER_POSITION_MISMATCHES = {
    (
        "d83694cb-a6dc-508c-b95c-c5b653de068a",
        2012,
        "RB",
        "WR",
    ): "canonical_hybrid_role_shift:dexter_mccluster",
    (
        "d83694cb-a6dc-508c-b95c-c5b653de068a",
        2013,
        "RB",
        "WR",
    ): "canonical_hybrid_role_shift:dexter_mccluster",
    (
        "8f059112-f544-512f-8b98-64e31802a4fc",
        2015,
        "RB",
        "WR",
    ): "canonical_hybrid_role_shift:deanthony_thomas",
    (
        "b16d3ba0-39d4-5a4b-bca8-ad15e147c96b",
        2019,
        "WR",
        "RB",
    ): "canonical_hybrid_role_shift:cordarrelle_patterson",
    (
        "b16d3ba0-39d4-5a4b-bca8-ad15e147c96b",
        2021,
        "WR",
        "RB",
    ): "canonical_hybrid_role_shift:cordarrelle_patterson",
    (
        "2f3a5f36-ad51-527b-8fdc-ca0a5e431ad6",
        2022,
        "RB",
        "WR",
    ): "canonical_hybrid_role_shift:ty_montgomery",
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
V2_SCORED_PROJECTION_CONTEXT_COLUMNS = [
    "expert_points_median",
    "expert_ppg_team_game_median",
    "expert_ppg_team_game_std",
    "projected_pass_point_share",
    "projected_rush_point_share",
    "projected_receiving_point_share",
    "team_qb1_ppg",
]
V2_SCORED_PROJECTION_CONTEXT_AUDIT_COLUMNS = [
    "projection_context_source",
    "projection_context_scoring_hash",
    "projection_context_run_id",
    "scoring_context_available",
    "scoring_context_unavailable_reason",
    "scoring_context_position",
    "scoring_context_position_mismatch",
    "scoring_context_position_mismatch_reason",
    "team_qb_scoring_context_available",
    "team_qb_scoring_context_unavailable_reason",
    "model_input_avg_proj_points",
    "model_input_preseason_proj_ppg",
    "model_input_avg_proj_pass_points",
    "model_input_avg_proj_rush_points",
    "model_input_avg_proj_rec_points",
    "model_input_qb_avg_proj_pass_points",
    "model_input_std_proj_points",
    "model_input_std_pos_rank",
    "projection_context_avg_proj_points_delta",
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
V2_SCORING_SENSITIVE_CURRENT_CONTEXT_COLS = {
    "current_avg_proj_points",
    "avg_proj_points",
    "avg_proj_pass_points",
    "avg_proj_rush_points",
    "avg_proj_rec_points",
    "qb_avg_proj_pass_points",
    "std_proj_points",
    *MATCH_OUTPUT_COLS,
}
V2_SCORING_CONTEXT_CAPABLE_LEAGUES = frozenset({"beta", "nffc"})
SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES = frozenset({"beta"})

# These fields are the minimum non-neutral context required to match a current
# production player to historical weekly templates. Position-specific room
# fields may be genuinely unavailable (for example, an unsigned player); those
# remain explicitly auditable through current_context_missing_optional_fields.
CURRENT_CONTEXT_REQUIRED_COLS = [
    "team",
    "current_avg_proj_points",
    "avg_pick",
    "year_exp",
]
CURRENT_CONTEXT_PROVENANCE_COLS = [
    "current_context_source",
    "current_context_match_method",
    "current_team_source",
    "current_adp_source",
    "current_context_fallback_fields",
    "current_context_missing_fields",
    "current_context_missing_optional_fields",
]


#==========
# Paths / DB
#==========

root_path = general.get_main_path("Fantasy_Football")
db_path = f"{root_path}/Data/Databases/"
dm = DataManage(db_path)
DEFAULT_SIMULATION_DB_PATH = (
    Path(root_path) / "Data" / "Databases" / "Simulation.sqlite3"
).resolve()
SIMULATION_DB_PATH = DEFAULT_SIMULATION_DB_PATH
SIMULATION_DB_NAME = SIMULATION_DB_PATH.stem
simulation_dm = DataManage(str(SIMULATION_DB_PATH.parent))

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


def resolve_league(league=None):
    league = LEAGUE if league is None else league
    league = str(league).strip().lower()
    if league not in TEMPLATE_ID_LEAGUE_OFFSETS:
        valid = ", ".join(sorted(TEMPLATE_ID_LEAGUE_OFFSETS))
        raise ValueError(f"League must be one of: {valid}")
    return league


def set_active_league(league):
    global LEAGUE, WEEK_COUNT, WEEKS, TEMPLATE_SEASON_MIN
    LEAGUE = resolve_league(league)
    WEEK_COUNT = WEEK_COUNT_BY_LEAGUE[LEAGUE]
    WEEKS = list(range(1, WEEK_COUNT + 1))
    TEMPLATE_SEASON_MIN = TEMPLATE_SEASON_MIN_BY_LEAGUE[LEAGUE]
    return LEAGUE


def resolve_scoring_matched_context(
    scoring_matched_context=None,
    *,
    league=None,
):
    """Resolve whether the active matcher must use league-scored V2 context."""

    league = resolve_league(LEAGUE if league is None else league)
    if scoring_matched_context is None:
        enabled = league in _PRODUCTION_CYCLE.template_context_sources
    else:
        enabled = bool(scoring_matched_context)
    if enabled and league not in V2_SCORING_CONTEXT_CAPABLE_LEAGUES:
        raise ValueError(
            f"League {league} has no governed V2 scoring-context implementation."
        )
    return enabled


def scoring_matched_context_source(league=None):
    league = resolve_league(LEAGUE if league is None else league)
    return _PRODUCTION_CYCLE.template_context_sources.get(
        league,
        f"v2_{league}_scoring_matched_preseason",
    )


def projection_schedule_games(seasons):
    """Return the NFL team-game projection horizon for each source season."""

    values = pd.to_numeric(seasons, errors="coerce")
    return pd.Series(
        np.where(values.ge(2021), 17.0, 16.0),
        index=values.index,
        dtype=float,
    )


def current_adp_source_league(league=None):
    """Return the market source that governs the active population policy."""
    league = resolve_league(LEAGUE if league is None else league)
    return "etr" if league == "beta" else league


def validate_published_avg_adp_keys(frame, source_name):
    """Validate keys supplied by the canonical Avg_ADPs publication."""

    if "player_key" not in frame:
        raise ValueError(
            f"{source_name} does not contain published player_key"
        )
    output = frame.copy()
    keys = output["player_key"].astype("string").str.strip()
    if keys.isna().any() or keys.eq("").any():
        raise ValueError(
            f"{source_name} contains missing published player_key"
        )
    output["player_key"] = keys
    if output["player_key"].duplicated().any():
        raise ValueError(
            f"{source_name} contains duplicate published player_key"
        )
    return output


def set_simulation_db(simulation_db):
    simulation_db = Path(simulation_db).expanduser().resolve()
    if simulation_db.suffix.lower() != ".sqlite3":
        raise ValueError("Simulation database must use the .sqlite3 extension.")
    if not simulation_db.is_file():
        raise FileNotFoundError(
            f"Simulation database does not exist: {simulation_db}"
        )

    global SIMULATION_DB_PATH, SIMULATION_DB_NAME, simulation_dm, db_path, dm
    SIMULATION_DB_PATH = simulation_db
    SIMULATION_DB_NAME = simulation_db.stem
    db_path = str(simulation_db.parent)
    simulation_dm = DataManage(db_path)
    # A staged Simulation database must consume the sibling staged
    # Model_Inputs/Validations databases as well.  Keeping ``dm`` pointed at
    # the live directory would silently mix refreshed and stale artifacts.
    dm = DataManage(db_path)
    return SIMULATION_DB_PATH


def resolve_v2_database(v2_database=None, league=None):
    league = resolve_league(league)
    if v2_database is None:
        v2_database = V2_DATABASES.get(league, V2_IDENTITY_DB_PATH)
    v2_database = Path(v2_database).expanduser().resolve()
    if not v2_database.is_file():
        raise FileNotFoundError(
            f"V2 database does not exist: {v2_database}"
        )

    required_tables = {"player_identity", "player_aliases"}
    if league in V2_DATABASES:
        required_tables.update(
            {"locked_candidate_runs", "locked_template_handoff"}
        )
    with sqlite3.connect(v2_database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing_tables = sorted(required_tables - tables)
        if missing_tables:
            raise ValueError(
                "V2 database is missing required tables: "
                + ", ".join(missing_tables)
            )
        if league in V2_DATABASES:
            locked_runs = connection.execute(
                """
                SELECT DISTINCT handoff.model_run_id,
                                runs.metadata_json
                FROM locked_template_handoff handoff
                LEFT JOIN locked_candidate_runs runs
                  ON runs.model_run_id=handoff.model_run_id
                """
            ).fetchall()
            if not locked_runs:
                raise ValueError(
                    f"V2 database has no active locked handoff: {v2_database}"
                )
            locked_objectives = set()
            for model_run_id, metadata_json in locked_runs:
                if metadata_json is None:
                    raise ValueError(
                        "V2 locked handoff has no candidate-run metadata for "
                        f"{model_run_id}: {v2_database}"
                    )
                try:
                    metadata = json.loads(metadata_json)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "V2 locked handoff has invalid candidate-run metadata "
                        f"for {model_run_id}: {v2_database}"
                    ) from exc
                scoring_objective = metadata.get("scoring_objective")
                if not scoring_objective:
                    raise ValueError(
                        "V2 locked handoff metadata has no scoring_objective "
                        f"for {model_run_id}: {v2_database}"
                    )
                locked_objectives.add(
                    str(scoring_objective).strip().lower()
                )
            if locked_objectives != {league}:
                raise ValueError(
                    "V2 locked handoff scoring objective "
                    f"{sorted(locked_objectives)} does not match {league}: "
                    f"{v2_database}"
                )
    return v2_database


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


def add_fantasy_points(df, pos, league, filter_qb_workload=True):
    league = resolve_league(league)
    df = add_bonus_cols(df)

    df = calc_fp(
        df,
        get_scoring_dict("rush", league=league),
        "fantasy_pts_rush",
    )

    if pos == "QB":
        df = calc_fp(
            df,
            get_scoring_dict("passing", league=league),
            "fantasy_pts_pass",
        )
        df["fantasy_pts"] = df["fantasy_pts_rush"] + df["fantasy_pts_pass"]

        df = add_missing_cols(df, ["pass_qb_dropback_sum", "rush_rush_attempt_sum"])
        df["total_plays"] = df["pass_qb_dropback_sum"] + df["rush_rush_attempt_sum"]
        if filter_qb_workload:
            df = df[df["total_plays"] > 15].reset_index(drop=True)
    else:
        df = calc_fp(
            df,
            get_scoring_dict("receiving", league=league),
            "fantasy_pts_rec",
        )
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


def has_current_team(values):
    teams = pd.Series(values).astype("string").str.strip().str.upper()
    return (
        teams.notna()
        & teams.ne("")
        & ~teams.isin({"FA", "UNK", "UNKNOWN", "NONE", "NAN", "NULL"})
    )


def fill_current_team_from_published_adp(
    frame,
    *,
    published_team_column,
    primary_source,
):
    """Fill only unassigned current teams from the canonical keyed ADP row."""

    frame = frame.copy()
    primary_team = frame.get(
        "team",
        pd.Series(pd.NA, index=frame.index, dtype="string"),
    )
    published_team = frame.get(
        published_team_column,
        pd.Series(pd.NA, index=frame.index, dtype="string"),
    )
    primary_valid = has_current_team(primary_team)
    published_valid = has_current_team(published_team)
    use_published = ~primary_valid & published_valid
    frame["team"] = primary_team.where(~use_published, published_team)
    frame["current_team_source"] = np.select(
        [primary_valid, use_published],
        [primary_source, "canonical_avg_adps"],
        default="unassigned",
    )
    return frame


def add_qb_team_rank_fields(df, year_col, projection_col):
    df = df.copy()
    df["qb_team_rank"] = -1
    df["qb_team_rank_bucket"] = "non_qb"

    if "team" not in df.columns:
        df.loc[df["pos"].eq("QB"), "qb_team_rank_bucket"] = "unknown"
        return df

    qb_mask = df["pos"].eq("QB") & has_current_team(df["team"]).to_numpy()
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
    *,
    preserve_signed_team_qb_context=False,
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

    valid_team = has_current_team(df["team"]).to_numpy()
    rb_mask = df["pos"].eq("RB") & valid_team
    if rb_mask.any():
        df.loc[rb_mask, "team_rb_rush_points"] = (
            df.loc[rb_mask].groupby(team_group_cols)["avg_proj_rush_points"].transform("sum")
        )
        df.loc[rb_mask, "team_rb_rec_points"] = (
            df.loc[rb_mask].groupby(team_group_cols)["avg_proj_rec_points"].transform("sum")
        )

    receiver_mask = df["pos"].isin(["RB", "WR", "TE"]) & valid_team
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

    qb_mask = df["pos"].eq("QB") & valid_team
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

    df["__provided_team_qb_pass_points"] = pd.to_numeric(
        df["qb_avg_proj_pass_points"], errors="coerce"
    )
    df["team_qb_pass_points"] = df[
        "__provided_team_qb_pass_points"
    ].fillna(0)
    qb_mask = df["pos"].eq("QB") & valid_team
    if qb_mask.any():
        qb_pass = (
            df.loc[qb_mask, team_group_cols + ["avg_proj_pass_points"]]
            .groupby(team_group_cols, as_index=False)
            .agg(team_qb_pass_points_calc=("avg_proj_pass_points", "max"))
        )
        df = df.merge(qb_pass, on=team_group_cols, how="left")
        provided_team_qb_context = df[
            "__provided_team_qb_pass_points"
        ].notna()
        if not preserve_signed_team_qb_context:
            provided_team_qb_context &= df[
                "__provided_team_qb_pass_points"
            ].gt(0)
        df["team_qb_pass_points"] = np.where(
            provided_team_qb_context,
            df["__provided_team_qb_pass_points"],
            df["team_qb_pass_points_calc"].fillna(0),
        )
        df = df.drop(columns=["team_qb_pass_points_calc"])

    if (
        preserve_signed_team_qb_context
        and "team_qb_scoring_context_available" in df
    ):
        unavailable_team_qb_context = pd.to_numeric(
            df["team_qb_scoring_context_available"],
            errors="coerce",
        ).eq(0)
        df.loc[
            unavailable_team_qb_context,
            "team_qb_pass_points",
        ] = np.nan

    df = df.drop(columns=["__provided_team_qb_pass_points"])

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
    exists = simulation_dm.read(
        f"""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
              AND name='{table_name}'
        """,
        SIMULATION_DB_NAME,
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
        existing = simulation_dm.read(
            f"SELECT * FROM {table_name} LIMIT 0",
            SIMULATION_DB_NAME,
        )
        if col not in existing.columns:
            continue
        values = simulation_dm.read(
            f"""
            SELECT DISTINCT {col} inferred_league
            FROM {table_name}
            WHERE {col} IS NOT NULL
            """,
            SIMULATION_DB_NAME,
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
        existing = simulation_dm.read(
            f"SELECT * FROM {table_name}",
            SIMULATION_DB_NAME,
        )
        existing = prepare_existing_best_ball_table(existing, table_name)
        keep_existing = existing[keep_existing_mask_func(existing)].copy()
        combined = pd.concat([keep_existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()

    ordered_cols = list(new_df.columns) + [
        col for col in combined.columns if col not in new_df.columns
    ]
    return combined[ordered_cols]


def keep_not_current_league(df):
    return ~rows_matching(df, {"league": LEAGUE})


def keep_not_current_pool_slice(df):
    return ~rows_matching(
        df,
        {
            "pool_version": LEAGUE,
            "pool_dataset": PRED_VERSION,
        },
    )


def keep_not_current_prediction_slice(df):
    return ~rows_matching(
        df,
        {
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

    tables = {
        table_name: replace_table_slice(
            table_name,
            df,
            keep_existing,
        )
        for table_name, df, keep_existing in table_writes
    }
    publish_tables_atomic(SIMULATION_DB_PATH, tables)


def validate_existing_v2_player_keys(
    frame,
    identity_database,
    *,
    position_column="pos",
):
    """Validate and preserve canonical keys already supplied by production.

    The production handoff owns current-player identity. Re-resolving those
    rows from a display name would recreate the handoff/player-map circularity
    and can fail for suffix or provider-name differences.
    """

    output = frame.copy()
    if "player_key" not in output.columns:
        raise ValueError("Current production rows do not contain player_key")
    present = output["player_key"].notna()
    if present.any() and not present.all():
        raise ValueError(
            "Current production rows mix keyed and unkeyed player identities"
        )
    if not present.any():
        raise ValueError("Current production rows contain no canonical player keys")

    output["player_key"] = output["player_key"].astype("string")
    with sqlite3.connect(identity_database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        identities = pd.read_sql_query(
            "SELECT player_key, position identity_position "
            "FROM player_identity",
            connection,
        )
        current_shadow_table = f"locked_{YEAR}_shadow_predictions"
        if current_shadow_table in tables:
            shadow_positions = pd.read_sql_query(
                "SELECT player_key, position shadow_position "
                f'FROM "{current_shadow_table}"',
                connection,
            )
            identities = identities.merge(
                shadow_positions,
                on="player_key",
                how="left",
                validate="one_to_one",
            )
            identities["position"] = identities["shadow_position"].combine_first(
                identities["identity_position"]
            )
        else:
            identities["position"] = identities["identity_position"]
    identities["player_key"] = identities["player_key"].astype("string")
    identities["position"] = identities["position"].astype("string").str.upper()
    if identities["player_key"].duplicated().any():
        raise ValueError("V2 identity validation contains duplicate player keys")

    validation = output[["player_key", position_column]].merge(
        identities,
        on="player_key",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = validation["_merge"].ne("both")
    if missing.any():
        preview = validation.loc[missing, "player_key"].drop_duplicates().head(10)
        raise ValueError(
            "Current production rows reference unknown V2 player keys: "
            f"{preview.tolist()}"
        )
    position_mismatch = (
        validation[position_column].astype("string").str.upper()
        != validation["position"]
    )
    if position_mismatch.any():
        preview = validation.loc[
            position_mismatch,
            ["player_key", position_column, "position"],
        ].head(10)
        raise ValueError(
            "Current production player positions disagree with V2 identity: "
            f"{preview.to_dict('records')}"
        )
    if "player_key_match_method" not in output.columns:
        output["player_key_match_method"] = "production_handoff_player_key"
    else:
        output["player_key_match_method"] = output[
            "player_key_match_method"
        ].fillna("production_handoff_player_key")
    return output


def attach_weekly_handoff_player_keys(
    templates,
    player_map,
    *,
    v2_database=None,
):
    v2_database = resolve_v2_database(v2_database)
    templates = attach_v2_player_keys(
        templates,
        v2_database,
        season_column="season",
    )
    if "player_key" in player_map and player_map["player_key"].notna().any():
        player_map = validate_existing_v2_player_keys(
            player_map,
            v2_database,
        )
    else:
        player_map = attach_v2_player_keys(
            player_map,
            v2_database,
            season_column="year",
        )
    return templates, player_map


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


def load_v2_scored_projection_context(
    v2_database,
    *,
    min_season,
    max_season,
):
    """Load leakage-safe, scoring-matched preseason context from the V2 mart."""

    if LEAGUE not in V2_SCORING_CONTEXT_CAPABLE_LEAGUES:
        raise ValueError(
            f"The V2 scored projection-context override is not governed for {LEAGUE}."
        )
    context_label = LEAGUE.upper()
    v2_database = resolve_v2_database(v2_database)
    required_columns = {
        "player_key",
        "season",
        "position",
        "team",
        "league",
        "scoring_hash",
        "run_id",
        "feature_cutoff_season",
        "preseason_source_season",
        *V2_SCORED_PROJECTION_CONTEXT_COLUMNS,
    }
    with sqlite3.connect(v2_database) as connection:
        available_columns = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("player_season_features")'
            )
        }
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            raise ValueError(
                f"V2 player_season_features lacks {context_label} scored projection "
                f"context columns: {missing_columns}"
            )
        context = pd.read_sql_query(
            """
            SELECT player_key,
                   CAST(season AS INTEGER) season,
                   position feature_context_position,
                   team feature_context_team,
                   league projection_context_league,
                   scoring_hash projection_context_scoring_hash,
                   run_id projection_context_run_id,
                   CAST(feature_cutoff_season AS INTEGER)
                       feature_context_cutoff_season,
                   CAST(preseason_source_season AS INTEGER)
                       feature_context_source_season,
                   expert_points_median,
                   expert_ppg_team_game_median,
                   expert_ppg_team_game_std,
                   projected_pass_point_share,
                   projected_rush_point_share,
                   projected_receiving_point_share,
                   team_qb1_ppg
            FROM player_season_features
            WHERE season BETWEEN ? AND ?
            """,
            connection,
            params=(int(min_season), int(max_season)),
        )
        if LEAGUE == "beta":
            projection_value_columns = {
                str(row[1])
                for row in connection.execute(
                    'PRAGMA table_info("player_season_projection_values")'
                )
            }
            required_projection_value_columns = {
                "player_key",
                "season",
                "provider",
                "position",
                "configured_points_complete",
                "provider_projected_points",
                "run_id",
            }
            missing_projection_value_columns = sorted(
                required_projection_value_columns - projection_value_columns
            )
            if missing_projection_value_columns:
                raise ValueError(
                    "V2 beta projection values lack scored rank-context "
                    f"columns: {missing_projection_value_columns}"
                )
            beta_rank_values = pd.read_sql_query(
                """
                SELECT player_key,
                       CAST(season AS INTEGER) season,
                       provider,
                       position,
                       provider_projected_points,
                       run_id
                FROM player_season_projection_values
                WHERE season BETWEEN ? AND ?
                      AND configured_points_complete=1
                      AND provider_projected_points IS NOT NULL
                """,
                connection,
                params=(int(min_season), int(max_season)),
            )
        locked_feature_runs = {
            str(row[0]).strip()
            for row in connection.execute(
                """
                SELECT DISTINCT runs.feature_run_id
                FROM locked_template_handoff handoff
                JOIN locked_candidate_runs runs
                  ON runs.model_run_id=handoff.model_run_id
                WHERE runs.feature_run_id IS NOT NULL
                """
            )
            if str(row[0]).strip()
        }

    if context.empty:
        raise ValueError(
            f"V2 player_season_features contains no {context_label} scored projection "
            f"context for {min_season}-{max_season}."
        )
    if context.duplicated(["player_key", "season"]).any():
        preview = context.loc[
            context.duplicated(["player_key", "season"], keep=False),
            ["player_key", "season"],
        ].head(10)
        raise ValueError(
            f"V2 {context_label} scored projection context contains duplicate keys: "
            f"{preview.to_dict('records')}"
        )

    expected_hash = scoring_hash(LEAGUE)
    context_league = (
        context["projection_context_league"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    context_hash = (
        context["projection_context_scoring_hash"]
        .astype("string")
        .str.strip()
    )
    context_run = (
        context["projection_context_run_id"]
        .astype("string")
        .str.strip()
    )
    if context_league.isna().any() or not context_league.eq(LEAGUE).all():
        observed_leagues = sorted(set(context_league.dropna().astype(str)))
        raise ValueError(
            f"V2 {context_label} scored projection context has unexpected leagues: "
            f"{observed_leagues}"
        )
    if context_hash.isna().any() or not context_hash.eq(expected_hash).all():
        observed_hashes = sorted(set(context_hash.dropna().astype(str)))
        raise ValueError(
            f"V2 {context_label} scored projection context has an unexpected scoring "
            f"hash: {observed_hashes}"
        )
    if context_run.isna().any() or context_run.eq("").any():
        raise ValueError(
            f"V2 {context_label} scored projection context contains a missing run_id."
        )
    observed_feature_runs = set(context_run.astype(str))
    if not locked_feature_runs or observed_feature_runs != locked_feature_runs:
        raise ValueError(
            f"V2 {context_label} scored projection context is not the exact feature run "
            "referenced by the active locked handoff: "
            f"context={sorted(observed_feature_runs)}, "
            f"locked={sorted(locked_feature_runs)}"
        )

    if LEAGUE == "beta":
        if beta_rank_values.empty:
            raise ValueError(
                "V2 beta projection values contain no scored provider ranks "
                f"for {min_season}-{max_season}."
            )
        if beta_rank_values.duplicated(
            ["player_key", "season", "provider"]
        ).any():
            preview = beta_rank_values.loc[
                beta_rank_values.duplicated(
                    ["player_key", "season", "provider"],
                    keep=False,
                ),
                ["player_key", "season", "provider"],
            ].head(10)
            raise ValueError(
                "V2 beta scored provider rank context contains duplicate "
                f"keys: {preview.to_dict('records')}"
            )
        beta_rank_values["provider_projected_points"] = pd.to_numeric(
            beta_rank_values["provider_projected_points"],
            errors="coerce",
        )
        invalid_rank_points = (
            beta_rank_values["provider_projected_points"].isna()
            | ~np.isfinite(beta_rank_values["provider_projected_points"])
        )
        if invalid_rank_points.any():
            raise ValueError(
                "V2 beta scored provider rank context contains invalid points."
            )
        beta_rank_runs = {
            str(run_id).strip()
            for run_id in beta_rank_values["run_id"].dropna()
            if str(run_id).strip()
        }
        if beta_rank_runs != observed_feature_runs:
            raise ValueError(
                "V2 beta scored provider ranks are not from the exact feature "
                "run used by player_season_features: "
                f"ranks={sorted(beta_rank_runs)}, "
                f"features={sorted(observed_feature_runs)}"
            )
        beta_rank_values["position"] = (
            beta_rank_values["position"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
        beta_rank_values["_beta_scored_position_rank"] = (
            beta_rank_values.groupby(
                ["season", "provider", "position"]
            )["provider_projected_points"]
            .rank(method="average", ascending=False)
        )
        beta_rank_context = (
            beta_rank_values.groupby(
                ["player_key", "season"],
                as_index=False,
            )
            .agg(
                beta_scored_position_rank_std=(
                    "_beta_scored_position_rank",
                    lambda values: (
                        float(values.std(ddof=0))
                        if values.notna().sum() >= 2
                        else np.nan
                    ),
                ),
                beta_scored_position_rank_source_count=(
                    "_beta_scored_position_rank",
                    "count",
                ),
            )
        )
        context = context.merge(
            beta_rank_context,
            on=["player_key", "season"],
            how="left",
            validate="one_to_one",
        )
        missing_rank_context = (
            pd.to_numeric(
                context["expert_points_median"], errors="coerce"
            ).notna()
            & context["beta_scored_position_rank_source_count"].isna()
        )
        if missing_rank_context.any():
            preview = context.loc[
                missing_rank_context,
                ["player_key", "season", "feature_context_position"],
            ].head(20)
            raise ValueError(
                "V2 beta scored provider-rank coverage is incomplete: "
                f"{preview.to_dict('records')}"
            )
    else:
        context["beta_scored_position_rank_std"] = np.nan
        context["beta_scored_position_rank_source_count"] = np.nan

    season = pd.to_numeric(context["season"], errors="coerce")
    cutoff = pd.to_numeric(
        context["feature_context_cutoff_season"], errors="coerce"
    )
    source_season = pd.to_numeric(
        context["feature_context_source_season"], errors="coerce"
    )
    invalid_time = (
        season.isna()
        | cutoff.ne(season - 1)
        | source_season.ne(season)
    )
    if invalid_time.any():
        preview = context.loc[
            invalid_time,
            [
                "player_key",
                "season",
                "feature_context_cutoff_season",
                "feature_context_source_season",
            ],
        ].head(10)
        raise ValueError(
            f"V2 {context_label} scored projection context violates the preseason "
            f"cutoff contract: {preview.to_dict('records')}"
        )

    # ``team_qb1_ppg`` is the QB1's total fantasy PPG.  Receiver matching
    # needs the team's projected QB passing points, so derive that value from
    # the actual QB1 row rather than relabeling total QB fantasy points.
    context["_feature_context_team_normalized"] = (
        context["feature_context_team"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(TEAM_ALIASES)
    )
    assigned_qb = (
        context["feature_context_position"].eq("QB")
        & has_current_team(context["_feature_context_team_normalized"])
    )
    qb_context = context.loc[
        assigned_qb,
        [
            "player_key",
            "season",
            "_feature_context_team_normalized",
            "expert_points_median",
            "expert_ppg_team_game_median",
            "projected_pass_point_share",
        ],
    ].copy()
    qb_context["_qb_total_points"] = pd.to_numeric(
        qb_context["expert_points_median"], errors="coerce"
    )
    qb_context["_qb_pass_share"] = pd.to_numeric(
        qb_context["projected_pass_point_share"], errors="coerce"
    )
    qb_context = qb_context[
        qb_context["_qb_total_points"].notna()
    ].copy()
    qb_context = qb_context.sort_values(
        [
            "season",
            "_feature_context_team_normalized",
            "_qb_total_points",
            "player_key",
        ],
        ascending=[True, True, False, True],
        na_position="last",
    ).drop_duplicates(
        ["season", "_feature_context_team_normalized"],
        keep="first",
    )
    invalid_qb1 = (
        qb_context["_qb_total_points"].isna()
        | ~np.isfinite(qb_context["_qb_total_points"])
        | qb_context["_qb_total_points"].lt(0)
        | (
            qb_context["_qb_total_points"].gt(0)
            & (
                qb_context["_qb_pass_share"].isna()
                | ~np.isfinite(qb_context["_qb_pass_share"])
            )
        )
    )
    if LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
        invalid_qb1 |= (
            qb_context["_qb_total_points"].gt(0)
            & (
                qb_context["_qb_pass_share"].lt(0)
                | qb_context["_qb_pass_share"].gt(1)
            )
        )
    if invalid_qb1.any():
        preview = qb_context.loc[
            invalid_qb1,
            [
                "player_key",
                "season",
                "_feature_context_team_normalized",
                "expert_points_median",
                "projected_pass_point_share",
            ],
        ].head(20)
        raise ValueError(
            f"V2 {context_label} scored projection context has an invalid team QB1 "
            f"passing share: {preview.to_dict('records')}"
        )
    qb_context["team_qb1_pass_points"] = np.where(
        qb_context["_qb_total_points"].gt(0),
        qb_context["_qb_total_points"] * qb_context["_qb_pass_share"],
        0.0,
    )
    qb_context["derived_team_qb1_ppg"] = pd.to_numeric(
        qb_context["expert_ppg_team_game_median"], errors="coerce"
    )
    context = context.merge(
        qb_context[
            [
                "season",
                "_feature_context_team_normalized",
                "team_qb1_pass_points",
                "derived_team_qb1_ppg",
            ]
        ],
        on=["season", "_feature_context_team_normalized"],
        how="left",
        validate="many_to_one",
    )
    return context


def apply_v2_scored_projection_context(
    projections,
    *,
    v2_database,
    season_column,
    use_expert_donor_center=False,
    use_expert_fallback_center=False,
):
    """Replace scoring-sensitive Model_Inputs fields by canonical V2 key."""

    if LEAGUE not in V2_SCORING_CONTEXT_CAPABLE_LEAGUES:
        raise ValueError(
            f"League {LEAGUE} has no governed V2 scoring-context implementation."
        )
    context_label = LEAGUE.upper()
    if use_expert_donor_center and use_expert_fallback_center:
        raise ValueError(
            "Expert donor-center policies must select either all rows or fallback rows."
        )
    projections = projections.copy()
    required_keys = {"player_key", "pos", "team", season_column}
    missing_keys = sorted(required_keys - set(projections.columns))
    if missing_keys:
        raise ValueError(
            f"{context_label} projection context lacks canonical join columns: "
            f"{missing_keys}"
        )
    if projections["player_key"].isna().any():
        raise ValueError(
            f"{context_label} projection context contains a null player_key before the "
            "scoring-context join."
        )
    if projections.duplicated(["player_key", season_column]).any():
        raise ValueError(
            f"{context_label} projection context contains duplicate player-season keys."
        )

    seasons = pd.to_numeric(projections[season_column], errors="coerce")
    if seasons.isna().any():
        raise ValueError(
            f"{context_label} projection context contains an invalid season."
        )
    scored = load_v2_scored_projection_context(
        v2_database,
        min_season=int(seasons.min()),
        max_season=int(seasons.max()),
    )
    scored_team_qb = scored.loc[
        scored["feature_context_position"].eq("QB")
        & pd.to_numeric(
            scored["expert_points_median"], errors="coerce"
        ).notna(),
        [
            "season",
            "_feature_context_team_normalized",
            "team_qb1_pass_points",
            "derived_team_qb1_ppg",
        ],
    ].drop_duplicates(
        ["season", "_feature_context_team_normalized"]
    )
    if season_column != "season":
        scored = scored.rename(columns={"season": season_column})
        scored_team_qb = scored_team_qb.rename(
            columns={"season": season_column}
        )
    projections["player_key"] = projections["player_key"].astype("string")
    scored["player_key"] = scored["player_key"].astype("string")
    projections = projections.merge(
        scored,
        on=["player_key", season_column],
        how="left",
        validate="one_to_one",
        indicator="_v2_scored_context_join",
    )
    missing_context = projections["_v2_scored_context_join"].ne("both")
    if missing_context.any():
        preview = projections.loc[
            missing_context,
            ["player", "player_key", "pos", season_column],
        ].head(20)
        raise ValueError(
            f"{context_label} scoring-matched projection context coverage is incomplete: "
            f"{preview.to_dict('records')}"
        )

    unavailable_reason = (
        projections.get(
            V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN,
            pd.Series("", index=projections.index),
        )
        .astype("string")
        .fillna("")
        .str.strip()
    )
    governed_context_unavailable_candidate = (
        (LEAGUE == "beta")
        & projections["pos"].eq("QB")
        & pd.to_numeric(
            projections[season_column], errors="coerce"
        ).eq(2018)
        & unavailable_reason.eq(BETA_2018_QB_CENTER_FALLBACK_REASON)
    )

    projection_team = (
        projections["team"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(TEAM_ALIASES)
    )
    projections["_projection_context_team_normalized"] = projection_team
    if LEAGUE == "beta":
        scored_team_qb = scored_team_qb.rename(
            columns={
                "_feature_context_team_normalized": (
                    "_projection_context_team_normalized"
                ),
                "team_qb1_pass_points": (
                    "_projection_team_qb1_pass_points"
                ),
                "derived_team_qb1_ppg": "_projection_team_qb1_ppg",
            }
        )
        projections = projections.merge(
            scored_team_qb,
            on=[season_column, "_projection_context_team_normalized"],
            how="left",
            validate="many_to_one",
        )
        assigned_projection_team = has_current_team(
            projections["_projection_context_team_normalized"]
        )
        projections["team_qb1_pass_points"] = projections[
            "_projection_team_qb1_pass_points"
        ].where(
            assigned_projection_team,
            projections["team_qb1_pass_points"],
        )
        projections["derived_team_qb1_ppg"] = projections[
            "_projection_team_qb1_ppg"
        ].where(
            assigned_projection_team,
            projections["derived_team_qb1_ppg"],
        )
        projections["team_qb1_ppg"] = projections[
            "_projection_team_qb1_ppg"
        ].where(
            assigned_projection_team,
            projections["team_qb1_ppg"],
        )
    feature_team = (
        projections["feature_context_team"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(TEAM_ALIASES)
    )
    team_mismatch = (
        has_current_team(projection_team)
        & has_current_team(feature_team)
        & projection_team.ne(feature_team)
        & ~governed_context_unavailable_candidate
    )
    if team_mismatch.any():
        preview = projections.loc[
            team_mismatch,
            [
                "player",
                "player_key",
                "pos",
                "team",
                "feature_context_team",
                season_column,
            ],
        ].head(20)
        raise ValueError(
            f"{context_label} scoring-matched projection context has unexpected team "
            f"mismatches: {preview.to_dict('records')}"
        )

    position_mismatch = projections["feature_context_position"].ne(
        projections["pos"]
    )
    projections["scoring_context_position"] = projections[
        "feature_context_position"
    ]
    projections["scoring_context_position_mismatch"] = (
        position_mismatch.astype(np.int8)
    )
    projections["scoring_context_position_mismatch_reason"] = ""
    governed_position_mismatch = pd.Series(
        False,
        index=projections.index,
        dtype=bool,
    )
    for mismatch_key, mismatch_reason in (
        GOVERNED_V2_TEMPLATE_CENTER_POSITION_MISMATCHES.items()
    ):
        player_key, season, template_position, feature_position = mismatch_key
        governed_row = (
            position_mismatch
            & projections["player_key"].astype(str).eq(player_key)
            & pd.to_numeric(
                projections[season_column], errors="coerce"
            ).eq(season)
            & projections["pos"].eq(template_position)
            & projections["feature_context_position"].eq(feature_position)
        )
        governed_position_mismatch |= governed_row
        projections.loc[
            governed_row,
            "scoring_context_position_mismatch_reason",
        ] = mismatch_reason
    unexpected_position_mismatch = (
        position_mismatch
        & ~governed_position_mismatch
        & ~governed_context_unavailable_candidate
    )
    if unexpected_position_mismatch.any():
        preview = projections.loc[
            unexpected_position_mismatch,
            [
                "player",
                "player_key",
                "pos",
                "feature_context_position",
                season_column,
            ],
        ].head(20)
        raise ValueError(
            f"{context_label} scoring-matched projection context has unexpected position "
            f"mismatches: {preview.to_dict('records')}"
        )

    required_numeric = [
        "expert_points_median",
        "expert_ppg_team_game_median",
        "expert_ppg_team_game_std",
    ]
    invalid_required_context = pd.Series(
        False,
        index=projections.index,
        dtype=bool,
    )
    for column in required_numeric:
        values = pd.to_numeric(projections[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)
        if (
            column == "expert_ppg_team_game_std"
            or LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES
        ):
            invalid |= values.lt(0)
        invalid_required_context |= invalid
        projections[column] = values

    governed_unavailable = governed_context_unavailable_candidate
    scoring_context_unavailable = (
        invalid_required_context & governed_unavailable
    )
    unexpected_invalid_context = (
        invalid_required_context & ~scoring_context_unavailable
    )
    if unexpected_invalid_context.any():
        preview = projections.loc[
            unexpected_invalid_context,
            [
                "player",
                "player_key",
                "pos",
                season_column,
                *required_numeric,
            ],
        ].head(20)
        raise ValueError(
            f"{context_label} scored projection context has invalid required "
            f"values: {preview.to_dict('records')}"
        )
    scoring_context_available = ~scoring_context_unavailable
    projections["scoring_context_available"] = (
        scoring_context_available.astype(np.int8)
    )
    projections["scoring_context_unavailable_reason"] = np.where(
        scoring_context_unavailable,
        unavailable_reason,
        "",
    )

    source_schedule_games = projection_schedule_games(
        projections[season_column]
    )
    expected_ppg = projections["expert_points_median"] / source_schedule_games
    inconsistent_ppg = scoring_context_available & ~np.isclose(
        expected_ppg,
        projections["expert_ppg_team_game_median"],
        rtol=0,
        atol=1e-9,
    )
    if inconsistent_ppg.any():
        preview = projections.loc[
            inconsistent_ppg,
            [
                "player",
                "player_key",
                season_column,
                "expert_points_median",
                "expert_ppg_team_game_median",
            ],
        ].head(20)
        raise ValueError(
            f"{context_label} scored projection total and team-game PPG disagree "
            "with the source-season schedule contract: "
            f"{preview.to_dict('records')}"
        )

    share_columns = [
        "projected_pass_point_share",
        "projected_rush_point_share",
        "projected_receiving_point_share",
    ]
    if LEAGUE in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
        component_total_available = scoring_context_available & ~np.isclose(
            projections["expert_points_median"],
            0.0,
            rtol=0,
            atol=1e-12,
        )
    else:
        component_total_available = (
            scoring_context_available
            & projections["expert_points_median"].gt(0)
        )
    for column in share_columns:
        values = pd.to_numeric(projections[column], errors="coerce")
        invalid = component_total_available & (
            values.isna()
            | ~np.isfinite(values)
        )
        if LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
            invalid |= component_total_available & (
                values.lt(0) | values.gt(1)
            )
        if invalid.any():
            preview = projections.loc[
                invalid,
                ["player", "player_key", "pos", season_column, column],
            ].head(20)
            raise ValueError(
                f"{context_label} scored projection context has invalid {column}: "
                f"{preview.to_dict('records')}"
            )
        projections[column] = values.where(component_total_available, 0.0)
    share_sum = projections[share_columns].sum(axis=1)
    inconsistent_shares = component_total_available & ~np.isclose(
        share_sum,
        1.0,
        rtol=0,
        atol=1e-9,
    )
    if inconsistent_shares.any():
        preview = projections.loc[
            inconsistent_shares,
            ["player", "player_key", "pos", season_column, *share_columns],
        ].head(20)
        raise ValueError(
            f"{context_label} scored projection component shares do not sum to one: "
            f"{preview.to_dict('records')}"
        )
    team_qb1_ppg = pd.to_numeric(
        projections["team_qb1_ppg"], errors="coerce"
    )
    team_qb1_pass_points = pd.to_numeric(
        projections["team_qb1_pass_points"], errors="coerce"
    )
    derived_team_qb1_ppg = pd.to_numeric(
        projections["derived_team_qb1_ppg"], errors="coerce"
    )
    assigned_team = has_current_team(projections["team"])
    invalid_team_qb_values = (
        (
            team_qb1_ppg.isna()
            | ~np.isfinite(team_qb1_ppg)
            | team_qb1_ppg.lt(0)
            | team_qb1_pass_points.isna()
            | ~np.isfinite(team_qb1_pass_points)
            | derived_team_qb1_ppg.isna()
            | ~np.isfinite(derived_team_qb1_ppg)
            | ~np.isclose(
                team_qb1_ppg,
                derived_team_qb1_ppg,
                rtol=0,
                atol=1e-9,
            )
        )
        & assigned_team
    )
    if LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
        invalid_team_qb_values |= (
            assigned_team & team_qb1_pass_points.lt(0)
        )
    governed_team_qb_unavailable = (
        (LEAGUE == "beta")
        & pd.to_numeric(
            projections[season_column], errors="coerce"
        ).eq(2018)
        & assigned_team
        & invalid_team_qb_values
    )
    invalid_team_qb = (
        invalid_team_qb_values
        & scoring_context_available
        & ~governed_team_qb_unavailable
    )
    if invalid_team_qb.any():
        preview = projections.loc[
            invalid_team_qb,
            [
                "player",
                "player_key",
                "pos",
                "team",
                season_column,
                "team_qb1_ppg",
                "team_qb1_pass_points",
                "derived_team_qb1_ppg",
            ],
        ].head(20)
        raise ValueError(
            f"{context_label} scored projection context lacks valid team-QB context for "
            f"an assigned player: {preview.to_dict('records')}"
        )
    team_qb_context_available = (
        ~invalid_team_qb_values | ~assigned_team
    )
    projections["team_qb_scoring_context_available"] = (
        team_qb_context_available.astype(np.int8)
    )
    projections["team_qb_scoring_context_unavailable_reason"] = np.where(
        governed_team_qb_unavailable,
        BETA_2018_QB_CENTER_FALLBACK_REASON,
        "",
    )
    projections["team_qb1_ppg"] = team_qb1_ppg
    projections["team_qb1_pass_points"] = team_qb1_pass_points

    original_columns = {
        "avg_proj_points": "model_input_avg_proj_points",
        "preseason_proj_ppg": "model_input_preseason_proj_ppg",
        "avg_proj_pass_points": "model_input_avg_proj_pass_points",
        "avg_proj_rush_points": "model_input_avg_proj_rush_points",
        "avg_proj_rec_points": "model_input_avg_proj_rec_points",
        "qb_avg_proj_pass_points": "model_input_qb_avg_proj_pass_points",
        "std_proj_points": "model_input_std_proj_points",
        "std_pos_rank": "model_input_std_pos_rank",
    }
    for source, audit_column in original_columns.items():
        projections[audit_column] = (
            projections[source] if source in projections else np.nan
        )

    projections["avg_proj_points"] = projections[
        "expert_points_median"
    ].where(
        scoring_context_available,
        projections["avg_proj_points"],
    )
    projections["preseason_proj_ppg"] = projections[
        "expert_ppg_team_game_median"
    ].where(
        scoring_context_available,
        projections["preseason_proj_ppg"],
    )
    for destination, share_column in (
        ("avg_proj_pass_points", "projected_pass_point_share"),
        ("avg_proj_rush_points", "projected_rush_point_share"),
        ("avg_proj_rec_points", "projected_receiving_point_share"),
    ):
        scored_component = (
            projections[share_column]
            * projections["expert_points_median"]
        )
        projections[destination] = scored_component.where(
            scoring_context_available,
            projections[destination],
        )
    projections["qb_avg_proj_pass_points"] = projections[
        "team_qb1_pass_points"
    ].where(
        scoring_context_available & team_qb_context_available,
        np.nan,
    )
    scored_std_points = (
        projections["expert_ppg_team_game_std"] * source_schedule_games
    )
    projections["std_proj_points"] = scored_std_points.where(
        scoring_context_available,
        projections["std_proj_points"],
    )
    if LEAGUE == "beta":
        scored_position_rank_std = pd.to_numeric(
            projections["beta_scored_position_rank_std"],
            errors="coerce",
        )
        projections["std_pos_rank"] = scored_position_rank_std.where(
            scoring_context_available,
            projections["std_pos_rank"],
        )
    projections["projection_context_source"] = np.where(
        scoring_context_available,
        scoring_matched_context_source(),
        "v2_beta_scoring_context_unavailable",
    )
    projections["projection_context_avg_proj_points_delta"] = (
        projections["avg_proj_points"]
        - pd.to_numeric(
            projections["model_input_avg_proj_points"],
            errors="coerce",
        )
    )
    if use_expert_donor_center:
        approved_policies = (
            _PRODUCTION_CYCLE.template_center_policies["nffc"]
        )
        if approved_policies != ("nffc_scored_expert_consensus",):
            raise ValueError(
                "The approved NFFC donor-center contract is not the "
                "scoring-matched expert consensus."
            )
        projections["historical_pred_fp_per_game"] = projections[
            "expert_ppg_team_game_median"
        ]
        projections["historical_projection_source"] = (
            "v2_nffc_expert_consensus"
        )
        projections["historical_center_policy"] = (
            approved_policies[0]
        )
        projections["v2_recenter_promoted"] = 0
    elif use_expert_fallback_center:
        if LEAGUE != "beta":
            raise ValueError(
                "The scoring-matched expert fallback center is governed only for beta."
            )
        approved_policies = set(
            _PRODUCTION_CYCLE.template_center_policies["beta"]
        )
        expected_policies = {
            "legacy_validated_oos",
            "beta_scored_expert_fallback",
        }
        if approved_policies != expected_policies:
            raise ValueError(
                "The approved beta donor-center contract does not permit "
                "the scoring-matched expert fallback."
            )
        fallback = (
            scoring_context_available
            & projections["historical_center_policy"].eq(
                "preseason_projection_fallback"
            )
        )
        projections.loc[
            fallback,
            "historical_pred_fp_per_game",
        ] = projections.loc[fallback, "expert_ppg_team_game_median"]
        projections.loc[
            fallback,
            "historical_projection_source",
        ] = "v2_beta_expert_consensus_fallback"
        projections.loc[
            fallback,
            "historical_center_policy",
        ] = "beta_scored_expert_fallback"

    return projections.drop(
        columns=[
            "_v2_scored_context_join",
            "feature_context_position",
            "projection_context_league",
            "feature_context_cutoff_season",
            "feature_context_source_season",
            "feature_context_team",
            "_feature_context_team_normalized",
            "_projection_context_team_normalized",
            "_projection_team_qb1_pass_points",
            "_projection_team_qb1_ppg",
            "team_qb1_pass_points",
            "derived_team_qb1_ppg",
            "beta_scored_position_rank_std",
            "beta_scored_position_rank_source_count",
            *V2_SCORED_PROJECTION_CONTEXT_COLUMNS,
        ],
        errors="ignore",
    )


def load_historical_projection_context(
    max_template_season,
    *,
    v2_database=None,
    scoring_matched_context=None,
    scoring_matched_fallback_center=None,
):
    use_scoring_context = resolve_scoring_matched_context(
        scoring_matched_context
    )
    if scoring_matched_fallback_center is None:
        use_scoring_fallback_center = (
            use_scoring_context
            and LEAGUE == "beta"
            and "beta_scored_expert_fallback"
            in _PRODUCTION_CYCLE.template_center_policies["beta"]
        )
    else:
        use_scoring_fallback_center = bool(
            scoring_matched_fallback_center
        )
    if use_scoring_fallback_center and not use_scoring_context:
        raise ValueError(
            "A scoring-matched fallback center requires scoring-matched context."
        )
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
    if not use_scoring_context:
        # Preserve the previously validated Model_Inputs matcher path exactly.
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
    proj = attach_locked_v2_historical_centers(
        proj,
        max_template_season=max_template_season,
        v2_database=v2_database,
    )
    if use_scoring_context:
        proj = apply_v2_scored_projection_context(
            proj,
            v2_database=v2_database,
            season_column="season",
            use_expert_donor_center=LEAGUE == "nffc",
            use_expert_fallback_center=use_scoring_fallback_center,
        )
        proj = add_qb_team_rank_fields(
            proj,
            year_col="season",
            projection_col="avg_proj_points",
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
        preserve_signed_team_qb_context=(
            use_scoring_context
            and LEAGUE in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES
        ),
    )
    return proj.reset_index(drop=True)


def _beta_2018_qb_center_fallback_proof(
    connection,
    model_run_ids,
):
    """Return the governed fallback reason only for an actively proven quarantine."""

    rules = [
        rule
        for rule in SOURCE_ROW_EXCLUSIONS
        if str(rule.get("exclusion_id", "")).strip()
        == BETA_2018_QB_CENTER_FALLBACK_EXCLUSION_ID
    ]
    if len(rules) != 1:
        return None
    rule = rules[0]
    expected_scope = ("FFToday_Projections", "QB", 2018)
    observed_scope = (
        str(rule.get("source_table", "")).strip(),
        str(rule.get("position", "")).strip().upper(),
        int(rule.get("stored_season", -1)),
    )
    reference = str(rule.get("reference", "")).strip()
    if observed_scope != expected_scope or not reference:
        return None

    model_run_ids = sorted(
        {str(model_run_id).strip() for model_run_id in model_run_ids}
    )
    if not model_run_ids:
        return None
    required_tables = {
        "locked_candidate_runs",
        "build_runs",
        "source_manifest",
    }
    available_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not required_tables.issubset(available_tables):
        return None

    placeholders = ", ".join("?" for _ in model_run_ids)
    candidate_rows = connection.execute(
        f"""
        SELECT DISTINCT model_run_id, feature_run_id
        FROM locked_candidate_runs
        WHERE model_run_id IN ({placeholders})
        """,
        model_run_ids,
    ).fetchall()
    candidate_features = {
        str(model_run_id): str(feature_run_id)
        for model_run_id, feature_run_id in candidate_rows
        if model_run_id is not None and feature_run_id is not None
    }
    if set(candidate_features) != set(model_run_ids):
        return None

    for feature_run_id in sorted(set(candidate_features.values())):
        feature_builds = connection.execute(
            """
            SELECT foundation_run_id
            FROM build_runs
            WHERE run_id=?
              AND component='milestone_3'
              AND league='beta'
              AND status='complete'
            """,
            (feature_run_id,),
        ).fetchall()
        if len(feature_builds) != 1 or feature_builds[0][0] is None:
            return None
        foundation_run_id = str(feature_builds[0][0])
        expected_policy = source_row_exclusion_policy_receipt(
            foundation_run_id
        )
        policy_rows = connection.execute(
            """
            SELECT source_uri, source_sha256, row_count
            FROM source_manifest
            WHERE run_id=?
              AND component=?
              AND source_name=?
            """,
            (
                foundation_run_id,
                expected_policy["component"],
                expected_policy["source_name"],
            ),
        ).fetchall()
        if len(policy_rows) != 1:
            return None
        policy_uri, policy_hash, policy_count = policy_rows[0]
        if (
            str(policy_uri) != expected_policy["source_uri"]
            or str(policy_hash) != expected_policy["source_sha256"]
            or policy_count is None
            or int(policy_count) != int(expected_policy["row_count"])
        ):
            return None

        quarantine_rows = connection.execute(
            """
            SELECT source_uri, row_count
            FROM source_manifest
            WHERE run_id=?
              AND component='source_quarantine'
              AND source_name=?
            """,
            (
                feature_run_id,
                BETA_2018_QB_CENTER_FALLBACK_EXCLUSION_ID,
            ),
        ).fetchall()
        if len(quarantine_rows) != 1:
            return None
        quarantine_uri, quarantine_count = quarantine_rows[0]
        if (
            str(quarantine_uri) != reference
            or quarantine_count is None
            or int(quarantine_count) <= 0
        ):
            return None
    return BETA_2018_QB_CENTER_FALLBACK_REASON


def attach_locked_v2_historical_centers(
    projections,
    *,
    max_template_season,
    v2_database=None,
):
    """Attach strict-OOS V2 centers without replacing validated donor centers.

    The 2026-07-29 rolling replay found that recentering the historical donor
    residuals on V2 predictions degraded PPG CRPS in both supported leagues.
    Keep those centers available for audit/research, but leave the production
    donor residuals centered on the previously validated OOS projections.
    """

    projections = projections.copy()
    projections["legacy_historical_pred_fp_per_game"] = projections[
        "historical_pred_fp_per_game"
    ]
    projections["legacy_historical_projection_source"] = projections[
        "historical_projection_source"
    ]
    projections["v2_point_center_source"] = pd.NA
    projections["v2_template_center_available"] = 0
    projections[V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN] = pd.NA
    projections[V2_TEMPLATE_CENTER_POSITION_COLUMN] = pd.NA
    projections[V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN] = 0
    projections[V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN] = pd.NA
    validated_center = (
        projections["validation_pred_fp_per_game"].notna()
        if "validation_pred_fp_per_game" in projections
        else ~projections["historical_projection_source"].eq(
            "preseason_projection_fallback"
        )
    )
    projections["historical_center_policy"] = np.where(
        validated_center,
        "legacy_validated_oos",
        "preseason_projection_fallback",
    )
    projections["v2_recenter_promoted"] = 0
    if LEAGUE not in V2_DATABASES:
        projections["historical_projection_source"] = (
            "legacy_non_v2_league"
        )
        return projections

    v2_database = resolve_v2_database(v2_database)
    projections = attach_v2_player_keys(
        projections,
        v2_database,
        season_column="season",
    )
    with sqlite3.connect(v2_database) as connection:
        centers = pd.read_sql_query(
            """
            SELECT model_run_id,
                   player_key,
                   CAST(season AS INTEGER) season,
                   position locked_v2_center_position,
                   historical_pred_fp_per_game v2_historical_pred_fp_per_game,
                   point_center_source v2_point_center_source,
                   template_center_available
            FROM locked_template_handoff
            WHERE season <= ?
            """,
            connection,
            params=(int(max_template_season),),
        )
    if centers.duplicated(["player_key", "season"]).any():
        raise ValueError(
            f"{LEAGUE} locked template handoff contains duplicate keys"
        )
    available = pd.to_numeric(
        centers["template_center_available"],
        errors="coerce",
    )
    if available.isna().any() or not available.isin([0, 1]).all():
        raise ValueError(
            f"{LEAGUE} locked template handoff has invalid "
            "template_center_available values"
        )
    center_present = centers["v2_historical_pred_fp_per_game"].notna()
    inconsistent_center = available.astype(int).ne(center_present.astype(int))
    if inconsistent_center.any():
        preview = centers.loc[
            inconsistent_center,
            [
                "player_key",
                "season",
                "v2_historical_pred_fp_per_game",
                "template_center_available",
            ],
        ].head(10)
        raise ValueError(
            f"{LEAGUE} locked template handoff center availability is "
            f"inconsistent: {preview.to_dict('records')}"
        )
    centers["template_center_available"] = available.astype(int)
    projections = projections.merge(
        centers,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_locked"),
        indicator="_v2_center_join",
    )
    if "v2_point_center_source_locked" in projections:
        projections["v2_point_center_source"] = projections.pop(
            "v2_point_center_source_locked"
        )
    v2_era = projections["season"].between(2017, max_template_season)
    missing_handoff = v2_era & projections["_v2_center_join"].ne("both")
    if missing_handoff.any():
        preview = projections.loc[
            missing_handoff,
            ["player", "pos", "season", "player_key"],
        ].head(10)
        raise ValueError(
            f"{LEAGUE} historical V2 point-center handoff coverage is "
            f"incomplete: {preview.to_dict('records')}"
        )
    position_mismatch = (
        v2_era
        & projections["_v2_center_join"].eq("both")
        & projections["locked_v2_center_position"].ne(projections["pos"])
    )
    governed_position_mismatch = pd.Series(
        False,
        index=projections.index,
        dtype=bool,
    )
    for mismatch_key, mismatch_reason in (
        GOVERNED_V2_TEMPLATE_CENTER_POSITION_MISMATCHES.items()
    ):
        player_key, season, template_position, locked_position = mismatch_key
        governed_row = (
            position_mismatch
            & projections["player_key"].astype(str).eq(player_key)
            & projections["season"].eq(season)
            & projections["pos"].eq(template_position)
            & projections["locked_v2_center_position"].eq(locked_position)
        )
        governed_position_mismatch |= governed_row
        projections.loc[
            governed_row,
            V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN,
        ] = mismatch_reason
    unexpected_position_mismatch = (
        position_mismatch & ~governed_position_mismatch
    )
    if unexpected_position_mismatch.any():
        preview = projections.loc[
            unexpected_position_mismatch,
            [
                "player",
                "player_key",
                "season",
                "pos",
                "locked_v2_center_position",
            ],
        ].head(10)
        raise ValueError(
            f"{LEAGUE} historical V2 point-center positions are "
            f"inconsistent: {preview.to_dict('records')}"
        )
    projections[V2_TEMPLATE_CENTER_POSITION_COLUMN] = projections[
        "locked_v2_center_position"
    ]
    projections[V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN] = (
        position_mismatch.astype(int)
    )
    missing_v2 = v2_era & projections[
        "v2_historical_pred_fp_per_game"
    ].isna()
    allowed_fallback = (
        missing_v2
        & projections["template_center_available"].eq(0)
        & projections["pos"].eq("QB")
        & projections["season"].eq(2018)
        & (LEAGUE == "beta")
    )
    unexpected_missing = missing_v2 & ~allowed_fallback
    if unexpected_missing.any():
        preview = projections.loc[
            unexpected_missing,
            ["player", "pos", "season", "player_key"],
        ].head(10)
        raise ValueError(
            f"{LEAGUE} historical V2 point-center coverage is incomplete: "
            f"{preview.to_dict('records')}"
        )
    if allowed_fallback.any():
        with sqlite3.connect(v2_database) as connection:
            fallback_proof = _beta_2018_qb_center_fallback_proof(
                connection,
                projections.loc[
                    allowed_fallback,
                    "model_run_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
            )
        if fallback_proof is None:
            raise ValueError(
                "beta 2018 QB historical V2 center fallback lacks the active "
                "FFToday quarantine proof"
            )
        projections.loc[
            allowed_fallback,
            V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN,
        ] = fallback_proof
    projections["v2_template_center_available"] = (
        projections["template_center_available"].fillna(0).astype(int)
    )
    projections.drop(
        columns=[
            "template_center_available",
            "model_run_id",
            "locked_v2_center_position",
            "_v2_center_join",
        ],
        inplace=True,
        errors="ignore",
    )
    return projections


def load_weekly_points(max_template_season, league=None):
    # Resolve at call time so research callers that temporarily set
    # builder.LEAGUE still receive the requested scoring rules.
    league = resolve_league(league)
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
            league=league,
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
    weekly["scoring_league"] = league
    return weekly


def resolve_template_scoring_league(weekly, league=None):
    requested_league = (
        resolve_league(league)
        if league is not None
        else None
    )
    weekly_leagues = []
    if "scoring_league" in weekly.columns:
        weekly_leagues = sorted(
            {
                resolve_league(value)
                for value in weekly["scoring_league"].dropna().unique()
            }
        )
    if len(weekly_leagues) > 1:
        raise ValueError(
            f"Weekly rows contain multiple scoring leagues: {weekly_leagues}"
        )
    if (
        requested_league is not None
        and weekly_leagues
        and weekly_leagues[0] != requested_league
    ):
        raise ValueError(
            "Weekly scoring league "
            f"{weekly_leagues[0]} does not match requested template league "
            f"{requested_league}."
        )
    if weekly_leagues:
        return weekly_leagues[0]
    return requested_league or resolve_league()


def build_weekly_templates(proj, weekly, league=None):
    league = resolve_template_scoring_league(weekly, league=league)
    base_cols = [
        "player",
        "player_key",
        "player_key_match_method",
        "pos",
        "team",
        "season",
        "avg_proj_points",
        "preseason_proj_ppg",
        "validation_pred_fp_per_game",
        "historical_pred_fp_per_game",
        "historical_projection_source",
        "legacy_historical_pred_fp_per_game",
        "legacy_historical_projection_source",
        "v2_historical_pred_fp_per_game",
        "v2_point_center_source",
        "v2_template_center_available",
        V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN,
        V2_TEMPLATE_CENTER_POSITION_COLUMN,
        V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN,
        V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN,
        "historical_center_policy",
        "v2_recenter_promoted",
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
    base_cols.extend(
        column
        for column in V2_SCORED_PROJECTION_CONTEXT_AUDIT_COLUMNS
        if column in proj.columns and column not in base_cols
    )
    template_cols = base_cols + [
        col for col in MATCH_OUTPUT_COLS if col in proj.columns and col not in base_cols
    ]
    template_index = proj[template_cols].copy()
    template_index["league"] = league
    template_index["template_local_id"] = np.arange(1, len(template_index) + 1)
    template_index["template_id"] = (
        template_id_offset(league) + template_index["template_local_id"]
    )

    week_grid = pd.DataFrame({"week": WEEKS})
    expanded = template_index[
        ["template_id", "player", "pos", "season", "historical_pred_fp_per_game"]
    ].merge(
        week_grid, how="cross"
    )

    weekly = weekly.drop(columns=["scoring_league"], errors="ignore").copy()
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
    if "scoring_context_available" in templates:
        unavailable_scoring_context = pd.to_numeric(
            templates["scoring_context_available"],
            errors="coerce",
        ).eq(0)
        unavailable_detail = templates.get(
            "scoring_context_unavailable_reason",
            pd.Series("", index=templates.index),
        ).astype("string").fillna("").str.strip()
        templates.loc[
            unavailable_scoring_context,
            "template_exclusion_reason",
        ] = (
            "scoring_context_unavailable:"
            + unavailable_detail.loc[unavailable_scoring_context]
        )
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
        "legacy_historical_pred_fp_per_game",
        "legacy_historical_projection_source",
        "v2_historical_pred_fp_per_game",
        "v2_point_center_source",
        "v2_template_center_available",
        V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN,
        V2_TEMPLATE_CENTER_POSITION_COLUMN,
        V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN,
        V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN,
        "historical_center_policy",
        "v2_recenter_promoted",
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
    front_cols.extend(
        column
        for column in V2_SCORED_PROJECTION_CONTEXT_AUDIT_COLUMNS
        if column in templates.columns and column not in front_cols
    )
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
        "pred_fp_per_game_ny",
        "pred_appear_current",
        "pred_appear_ny",
        "current_projection_model_version",
        "next_projection_model_version",
        "production_handoff_version",
        "current_projection_source",
        "current_uncertainty_source",
        "independent_current_residual_draw_allowed",
        "next_projection_source",
        "next_uncertainty_source",
        "v2_scoring_hash",
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
        observed_seasons = pd.to_numeric(
            template_audit["season"],
            errors="coerce",
        ).dropna()
        if observed_seasons.empty:
            raise ValueError("Weekly template audit contains no valid seasons")
        min_observed_season = int(observed_seasons.min())
        max_observed_season = int(observed_seasons.max())
        expected_exclusions = {
            key: reason
            for key, reason in TEMPLATE_OUTCOME_EXCLUSIONS.items()
            if min_observed_season <= key[2] <= max_observed_season
        }
        governed_beta_unavailable_reason = (
            "scoring_context_unavailable:"
            + BETA_2018_QB_CENTER_FALLBACK_REASON
        )
        governed_beta_unavailable = excluded[
            excluded["template_exclusion_reason"].eq(
                governed_beta_unavailable_reason
            )
        ]
        if not governed_beta_unavailable.empty:
            governed_seasons = pd.to_numeric(
                governed_beta_unavailable["season"],
                errors="coerce",
            )
            invalid_governed_unavailable = governed_beta_unavailable[
                governed_beta_unavailable["pos"].ne("QB")
                | governed_seasons.ne(2018)
            ]
            if LEAGUE != "beta" or not invalid_governed_unavailable.empty:
                preview = governed_beta_unavailable[
                    ["player", "pos", "season", "template_exclusion_reason"]
                ].head(10)
                raise ValueError(
                    "The governed beta scoring-context exclusion appeared "
                    "outside beta 2018 QB templates: "
                    f"{preview.to_dict('records')}"
                )
            expected_exclusions.update(
                {
                    (row.player, row.pos, int(row.season)):
                        governed_beta_unavailable_reason
                    for row in governed_beta_unavailable.itertuples(index=False)
                }
            )
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


def load_published_current_adp_context():
    """Load the canonical current feed used by both context routes."""

    adp_league = current_adp_source_league()
    adp = simulation_dm.read(
        f"""
        SELECT player_key,
               player,
               team adp_team,
               CAST(year AS INTEGER) year,
               league,
               avg_pick adp_avg_pick,
               Years_of_Experience adp_year_exp,
               identity_match_method
        FROM Avg_ADPs
        WHERE year={YEAR}
              AND league='{adp_league}'
              AND pos IN ('QB', 'RB', 'WR', 'TE')
        """,
        SIMULATION_DB_NAME,
    )
    adp = validate_published_avg_adp_keys(
        adp,
        f"{LEAGUE}_{adp_league}_weekly_context",
    )
    adp = (
        adp.sort_values(["player_key", "adp_avg_pick"])
        .drop_duplicates(["player_key"])
        .rename(columns={"player": "adp_source_player"})
    )
    return adp


def load_current_player_context(v2_database=None):
    v2_database = resolve_v2_database(v2_database)
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
    current_context = attach_v2_player_keys(
        current_context,
        v2_database,
        season_column="year",
        require_complete=True,
    ).rename(
        columns={
            "player_key_match_method": "current_context_match_method"
        }
    )
    current_context = add_qb_team_rank_fields(
        current_context,
        year_col="year",
        projection_col="current_avg_proj_points",
    )

    adp = load_published_current_adp_context()

    current_context = current_context.merge(
        adp[[
            "player_key",
            "year",
            "adp_source_player",
            "adp_team",
            "adp_avg_pick",
            "adp_year_exp",
        ]],
        on=["player_key", "year"],
        how="left",
        validate="one_to_one",
    )
    current_context = fill_current_team_from_published_adp(
        current_context,
        published_team_column="adp_team",
        primary_source="model_inputs",
    )
    current_context["year_exp"] = current_context["year_exp"].where(
        current_context["year_exp"].notna(),
        current_context["adp_year_exp"],
    )
    current_context["avg_pick"] = current_context["adp_avg_pick"].where(
        current_context["adp_avg_pick"].notna(),
        current_context["model_input_avg_pick"],
    )
    current_context["current_adp_source"] = np.select(
        [
            current_context["adp_avg_pick"].notna(),
            current_context["model_input_avg_pick"].notna(),
        ],
        [
            "canonical_avg_adps",
            "model_inputs_fallback",
        ],
        default="missing",
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
    current_context["current_context_source"] = (
        "model_inputs_projection_context"
    )
    current_context["current_context_missing_optional_fields"] = ""
    current_context = current_context.sort_values(
        ["player_key", "pos", "year", "current_avg_proj_points"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    current_context = current_context.drop_duplicates(
        ["player_key", "pos", "year"],
        keep="first",
    )
    return current_context


def attach_fallback_team_qb1_passing_context(fallback):
    """Derive team QB1 passing points from the full V2 feature population."""

    fallback = fallback.copy()
    required_columns = {
        "player_key",
        "season",
        "position",
        "team",
        "expert_points_median",
        "expert_ppg_team_game_median",
        "projected_pass_point_share",
        "team_qb1_ppg",
    }
    missing_columns = sorted(required_columns - set(fallback.columns))
    if missing_columns:
        raise ValueError(
            "V2 current fallback lacks team-QB context columns: "
            f"{missing_columns}"
        )

    normalized_team_column = "_fallback_team_normalized"
    fallback[normalized_team_column] = (
        fallback["team"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(TEAM_ALIASES)
    )
    assigned_qb = (
        fallback["position"].astype("string").str.upper().eq("QB")
        & has_current_team(fallback[normalized_team_column])
    )
    qb_context = fallback.loc[
        assigned_qb,
        [
            "player_key",
            "season",
            normalized_team_column,
            "expert_points_median",
            "expert_ppg_team_game_median",
            "projected_pass_point_share",
        ],
    ].copy()
    qb_context["_qb_total_points"] = pd.to_numeric(
        qb_context["expert_points_median"], errors="coerce"
    )
    qb_context["_qb_ppg"] = pd.to_numeric(
        qb_context["expert_ppg_team_game_median"], errors="coerce"
    )
    qb_context["_qb_pass_share"] = pd.to_numeric(
        qb_context["projected_pass_point_share"], errors="coerce"
    )
    qb_context = qb_context.sort_values(
        [
            "season",
            normalized_team_column,
            "_qb_total_points",
            "player_key",
        ],
        ascending=[True, True, False, True],
        na_position="last",
    ).drop_duplicates(
        ["season", normalized_team_column],
        keep="first",
    )
    invalid_qb1 = (
        qb_context["_qb_total_points"].isna()
        | ~np.isfinite(qb_context["_qb_total_points"])
        | qb_context["_qb_total_points"].lt(0)
        | qb_context["_qb_ppg"].isna()
        | ~np.isfinite(qb_context["_qb_ppg"])
        | qb_context["_qb_ppg"].lt(0)
        | (
            qb_context["_qb_total_points"].gt(0)
            & (
                qb_context["_qb_pass_share"].isna()
                | ~np.isfinite(qb_context["_qb_pass_share"])
                | qb_context["_qb_pass_share"].lt(0)
                | qb_context["_qb_pass_share"].gt(1)
            )
        )
    )
    if invalid_qb1.any():
        preview = qb_context.loc[
            invalid_qb1,
            [
                "player_key",
                "season",
                normalized_team_column,
                "expert_points_median",
                "expert_ppg_team_game_median",
                "projected_pass_point_share",
            ],
        ].head(20)
        raise ValueError(
            "V2 current fallback has an invalid team QB1 passing context: "
            f"{preview.to_dict('records')}"
        )
    qb_context["team_qb1_pass_points"] = np.where(
        qb_context["_qb_total_points"].gt(0),
        (
            qb_context["_qb_total_points"]
            * qb_context["_qb_pass_share"]
        ),
        0.0,
    )
    qb_context["derived_team_qb1_ppg"] = qb_context["_qb_ppg"]
    fallback = fallback.merge(
        qb_context[
            [
                "season",
                normalized_team_column,
                "team_qb1_pass_points",
                "derived_team_qb1_ppg",
            ]
        ],
        on=["season", normalized_team_column],
        how="left",
        validate="many_to_one",
    )

    assigned_team = has_current_team(fallback[normalized_team_column])
    expected_team_qb1_ppg = pd.to_numeric(
        fallback["team_qb1_ppg"], errors="coerce"
    )
    derived_team_qb1_ppg = pd.to_numeric(
        fallback["derived_team_qb1_ppg"], errors="coerce"
    )
    team_qb1_pass_points = pd.to_numeric(
        fallback["team_qb1_pass_points"], errors="coerce"
    )
    invalid_existing_team_qb1_ppg = expected_team_qb1_ppg.notna() & (
        ~np.isfinite(expected_team_qb1_ppg)
        | expected_team_qb1_ppg.lt(0)
        | ~np.isclose(
            expected_team_qb1_ppg,
            derived_team_qb1_ppg,
            rtol=0,
            atol=1e-9,
        )
    )
    invalid_assigned_team = assigned_team & (
        invalid_existing_team_qb1_ppg
        | derived_team_qb1_ppg.isna()
        | ~np.isfinite(derived_team_qb1_ppg)
        | team_qb1_pass_points.isna()
        | ~np.isfinite(team_qb1_pass_points)
        | team_qb1_pass_points.lt(0)
    )
    if invalid_assigned_team.any():
        preview = fallback.loc[
            invalid_assigned_team,
            [
                "player_key",
                "season",
                "position",
                "team",
                "team_qb1_ppg",
                "team_qb1_pass_points",
                "derived_team_qb1_ppg",
            ],
        ].head(20)
        raise ValueError(
            "V2 current fallback lacks consistent team-QB context for an "
            f"assigned player: {preview.to_dict('records')}"
        )
    fallback["team_qb1_ppg"] = expected_team_qb1_ppg.where(
        expected_team_qb1_ppg.notna(),
        derived_team_qb1_ppg,
    )
    fallback["team_qb1_pass_points"] = team_qb1_pass_points
    return fallback.drop(columns=[normalized_team_column])


def load_v2_current_player_context(
    v2_database=None,
    selected_player_keys=None,
    scoring_matched_context=None,
):
    """Build an auditable context fallback from the current V2 feature mart."""

    use_scoring_context = resolve_scoring_matched_context(
        scoring_matched_context
    )
    v2_database = resolve_v2_database(v2_database)
    required_columns = {
        "player_key",
        "display_name",
        "season",
        "position",
        "team",
        "league",
        "scoring_hash",
        "run_id",
        "feature_cutoff_season",
        "preseason_source_season",
        "expert_points_median",
        "expert_ppg_team_game_median",
        "expert_ppg_team_game_std",
        "expert_points_iqr",
        "adp_median",
        "year_exp",
        "projected_pass_point_share",
        "projected_rush_point_share",
        "projected_receiving_point_share",
        "team_qb1_ppg",
    }
    optional_columns = {
        "consensus_room_share",
        "consensus_room_gap_to_next",
        "consensus_room_hhi",
        "team_target_share",
        "team_reception_share",
        "team_rush_attempt_share",
        "pass_catcher_room_share",
    }
    with sqlite3.connect(v2_database) as connection:
        available_columns = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("player_season_features")'
            )
        }
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            raise ValueError(
                "V2 player_season_features lacks current-context fallback "
                f"columns: {missing_columns}"
            )
        select_columns = sorted(
            required_columns | optional_columns.intersection(available_columns)
        )
        fallback = pd.read_sql_query(
            "SELECT "
            + ", ".join(f'"{column}"' for column in select_columns)
            + ' FROM "player_season_features" WHERE season=?',
            connection,
            params=(int(YEAR),),
        )
    if fallback.empty:
        raise ValueError(
            f"V2 player_season_features has no current rows for {YEAR}"
        )
    if fallback["player_key"].duplicated().any():
        raise ValueError(
            f"V2 player_season_features has duplicate {YEAR} player keys"
        )
    if not use_scoring_context:
        # The V2 mart stores ``team_qb1_ppg`` as the QB1's total fantasy
        # scoring rate. Receiver matching needs the QB1's passing component,
        # so derive it from the full feature population before filtering to
        # the selected production universe. This keeps V2-only receivers
        # valid even when their quarterback is not itself selected.
        fallback = attach_fallback_team_qb1_passing_context(fallback)
    if selected_player_keys is not None:
        selected_player_keys = {
            str(player_key) for player_key in selected_player_keys
        }
        fallback = fallback[
            fallback["player_key"].astype(str).isin(selected_player_keys)
        ].copy()
        observed_keys = set(fallback["player_key"].astype(str))
        if observed_keys != selected_player_keys:
            raise ValueError(
                "Selected production keys are missing from the V2 current "
                "feature context: "
                f"{sorted(selected_player_keys - observed_keys)[:20]}"
            )
    if use_scoring_context:
        # Validate the table lineage against the active lock before any of its
        # fields can become authoritative in the league matcher.
        scored_context = load_v2_scored_projection_context(
            v2_database,
            min_season=YEAR,
            max_season=YEAR,
        )
        fallback = fallback.merge(
            scored_context[
                [
                    "player_key",
                    "season",
                    "team_qb1_pass_points",
                    "derived_team_qb1_ppg",
                    "beta_scored_position_rank_std",
                    "beta_scored_position_rank_source_count",
                ]
            ],
            on=["player_key", "season"],
            how="left",
            validate="one_to_one",
        )

        total = pd.to_numeric(
            fallback["expert_points_median"], errors="coerce"
        )
        ppg = pd.to_numeric(
            fallback["expert_ppg_team_game_median"], errors="coerce"
        )
        point_std = pd.to_numeric(
            fallback["expert_ppg_team_game_std"], errors="coerce"
        )
        invalid_required = (
            total.isna()
            | ppg.isna()
            | point_std.isna()
            | ~np.isfinite(total)
            | ~np.isfinite(ppg)
            | ~np.isfinite(point_std)
            | point_std.lt(0)
        )
        if LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
            invalid_required |= total.lt(0) | ppg.lt(0)
        source_schedule_games = projection_schedule_games(
            fallback["season"]
        )
        inconsistent_ppg = ~np.isclose(
            total / source_schedule_games,
            ppg,
            rtol=0,
            atol=1e-9,
        )
        if (invalid_required | inconsistent_ppg).any():
            preview = fallback.loc[
                invalid_required | inconsistent_ppg,
                [
                    "player_key",
                    "display_name",
                    "position",
                    "expert_points_median",
                    "expert_ppg_team_game_median",
                    "expert_ppg_team_game_std",
                ],
            ].head(20)
            raise ValueError(
                f"Selected {LEAGUE.upper()} current scoring context has invalid or "
                "inconsistent expert points: "
                f"{preview.to_dict('records')}"
            )
        share_columns = [
            "projected_pass_point_share",
            "projected_rush_point_share",
            "projected_receiving_point_share",
        ]
        shares = fallback[share_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if LEAGUE in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
            component_total_available = ~np.isclose(
                total,
                0.0,
                rtol=0,
                atol=1e-12,
            )
        else:
            component_total_available = total.gt(0)
        invalid_shares = component_total_available & (
            shares.isna().any(axis=1)
            | (~np.isfinite(shares)).any(axis=1)
            | ~np.isclose(
                shares.sum(axis=1),
                1.0,
                rtol=0,
                atol=1e-9,
            )
        )
        if LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
            invalid_shares |= component_total_available & (
                shares.lt(0).any(axis=1)
                | shares.gt(1).any(axis=1)
            )
        if invalid_shares.any():
            preview = fallback.loc[
                invalid_shares,
                [
                    "player_key",
                    "display_name",
                    "position",
                    "expert_points_median",
                    *share_columns,
                ],
            ].head(20)
            raise ValueError(
                f"Selected {LEAGUE.upper()} current scoring context has invalid component "
                f"shares: {preview.to_dict('records')}"
            )
        team_qb1_ppg = pd.to_numeric(
            fallback["team_qb1_ppg"], errors="coerce"
        )
        team_qb1_pass_points = pd.to_numeric(
            fallback["team_qb1_pass_points"], errors="coerce"
        )
        derived_team_qb1_ppg = pd.to_numeric(
            fallback["derived_team_qb1_ppg"], errors="coerce"
        )
        invalid_team_qb = (
            (
                ~np.isfinite(team_qb1_ppg)
                | team_qb1_ppg.lt(0)
                | ~np.isfinite(team_qb1_pass_points)
                | ~np.isfinite(derived_team_qb1_ppg)
                | ~np.isclose(
                    team_qb1_ppg,
                    derived_team_qb1_ppg,
                    rtol=0,
                    atol=1e-9,
                )
            )
            & has_current_team(fallback["team"])
        )
        if LEAGUE not in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES:
            invalid_team_qb |= (
                has_current_team(fallback["team"])
                & team_qb1_pass_points.lt(0)
            )
        if invalid_team_qb.any():
            preview = fallback.loc[
                invalid_team_qb,
                [
                    "player_key",
                    "display_name",
                    "position",
                    "team",
                    "team_qb1_ppg",
                    "team_qb1_pass_points",
                    "derived_team_qb1_ppg",
                ],
            ].head(20)
            raise ValueError(
                f"Selected {LEAGUE.upper()} current scoring context lacks valid team-QB "
                f"context for an assigned player: {preview.to_dict('records')}"
            )

    fallback = fallback.rename(
        columns={
            "display_name": "player",
            "position": "pos",
            "season": "year",
            "scoring_hash": "projection_context_scoring_hash",
            "run_id": "projection_context_run_id",
            "expert_points_median": "current_avg_proj_points",
            "adp_median": "feature_adp_median",
        }
    )
    fallback["player_key"] = fallback["player_key"].astype("string")
    fallback["pos"] = fallback["pos"].astype("string").str.upper()
    fallback["current_avg_proj_points"] = pd.to_numeric(
        fallback["current_avg_proj_points"],
        errors="coerce",
    )
    fallback["avg_proj_points"] = fallback["current_avg_proj_points"]
    fallback["model_input_avg_pick"] = np.nan
    published_adp = load_published_current_adp_context().rename(
        columns={
            "adp_team": "published_adp_team",
            "adp_avg_pick": "published_adp_avg_pick",
            "adp_year_exp": "published_adp_year_exp",
        }
    )
    fallback = fallback.merge(
        published_adp[
            [
                "player_key",
                "year",
                "published_adp_team",
                "published_adp_avg_pick",
                "published_adp_year_exp",
            ]
        ],
        on=["player_key", "year"],
        how="left",
        validate="one_to_one",
    )
    fallback = fill_current_team_from_published_adp(
        fallback,
        published_team_column="published_adp_team",
        primary_source="v2_player_season_features",
    )
    fallback["feature_adp_median"] = pd.to_numeric(
        fallback["feature_adp_median"],
        errors="coerce",
    )
    fallback["published_adp_avg_pick"] = pd.to_numeric(
        fallback["published_adp_avg_pick"],
        errors="coerce",
    )
    fallback["adp_avg_pick"] = fallback[
        "published_adp_avg_pick"
    ].where(
        fallback["published_adp_avg_pick"].notna(),
        fallback["feature_adp_median"],
    )
    fallback["current_adp_source"] = np.select(
        [
            fallback["published_adp_avg_pick"].notna(),
            fallback["feature_adp_median"].notna(),
        ],
        [
            "canonical_avg_adps",
            "v2_feature_mart_fallback",
        ],
        default="missing",
    )
    fallback["avg_pick"] = fallback["adp_avg_pick"]
    fallback["published_adp_year_exp"] = pd.to_numeric(
        fallback["published_adp_year_exp"],
        errors="coerce",
    )
    fallback["adp_year_exp"] = fallback[
        "published_adp_year_exp"
    ].where(
        fallback["published_adp_year_exp"].notna(),
        pd.to_numeric(fallback["year_exp"], errors="coerce"),
    )
    fallback["source_year_exp"] = fallback["adp_year_exp"]
    fallback["year_exp_source"] = "v2_player_season_features"
    fallback["year_exp_uncapped_delta"] = 0.0

    point_share_map = {
        "avg_proj_pass_points": "projected_pass_point_share",
        "avg_proj_rush_points": "projected_rush_point_share",
        "avg_proj_rec_points": "projected_receiving_point_share",
    }
    for destination, source in point_share_map.items():
        fallback[destination] = (
            pd.to_numeric(fallback[source], errors="coerce")
            * fallback["current_avg_proj_points"]
        )
    fallback["qb_avg_proj_pass_points"] = (
        pd.to_numeric(
            fallback["team_qb1_pass_points"], errors="coerce"
        )
    )
    fallback["std_proj_points"] = (
        pd.to_numeric(
            fallback["expert_ppg_team_game_std"],
            errors="coerce",
        )
        * projection_schedule_games(fallback["year"])
    )
    # Beta provider ranks are recomputed from the same configured, beta-scored
    # V2 points. Other scoring contexts retain the governed group fallback
    # until an equivalent league-specific rank contract is promoted.
    fallback["std_pos_rank"] = (
        pd.to_numeric(
            fallback["beta_scored_position_rank_std"],
            errors="coerce",
        )
        if LEAGUE == "beta" and use_scoring_context
        else np.nan
    )

    optional_raw_fields = sorted(
        {
            "expert_ppg_team_game_std",
            "expert_points_iqr",
            "projected_pass_point_share",
            "projected_rush_point_share",
            "projected_receiving_point_share",
            "team_qb1_ppg",
            "team_qb1_pass_points",
            *optional_columns,
        }.intersection(fallback.columns)
    )
    fallback["current_context_missing_optional_fields"] = fallback.apply(
        lambda row: ",".join(
            column
            for column in optional_raw_fields
            if pd.isna(row[column])
        ),
        axis=1,
    )

    # Rows without a projection center cannot supply a meaningful fallback.
    # Keep them out so a selected high-impact production row fails explicitly
    # in attach_current_context_by_player_key.
    fallback = fallback[
        fallback["current_avg_proj_points"].notna()
    ].reset_index(drop=True)
    fallback = add_qb_team_rank_fields(
        fallback,
        year_col="year",
        projection_col="current_avg_proj_points",
    )
    fallback = add_exp_fields(fallback)
    fallback["current_projection_ppg"] = (
        pd.to_numeric(
            fallback["expert_ppg_team_game_median"],
            errors="coerce",
        )
    )
    fallback = add_projection_buckets(
        fallback,
        value_col="current_projection_ppg",
        group_cols=["year", "pos"],
        pct_col="context_projection_rank_pct",
    )
    fallback = add_template_match_features(
        fallback,
        group_cols=["year", "pos"],
        rank_pct_col="context_projection_rank_pct",
        total_points_col="current_avg_proj_points",
        projection_ppg_col="current_projection_ppg",
        preserve_signed_team_qb_context=(
            use_scoring_context
            and LEAGUE in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES
        ),
    )
    fallback["current_context_source"] = (
        "v2_player_season_features_scoring_context"
        if use_scoring_context
        else "v2_player_season_features_fallback"
    )
    fallback["current_context_match_method"] = "v2_feature_player_key"
    return fallback


def _missing_context_fields(row, columns):
    missing = []
    for column in columns:
        value = row.get(column)
        if pd.isna(value) or (
            isinstance(value, str) and not value.strip()
        ):
            missing.append(column)
    return ",".join(missing)


def attach_current_context_by_player_key(
    predictions,
    model_input_context,
    v2_fallback_context,
    scoring_matched_context=None,
):
    """Attach current context without relying on the production display name."""

    predictions = predictions.copy()
    required_prediction_columns = {
        "player_key",
        "player",
        "pos",
        "year",
        "version",
        "dataset",
        "pred_fp_per_game",
    }
    missing_prediction_columns = sorted(
        required_prediction_columns - set(predictions.columns)
    )
    if missing_prediction_columns:
        raise ValueError(
            "Current predictions lack key-first context columns: "
            f"{missing_prediction_columns}"
        )
    if predictions["player_key"].isna().any():
        raise ValueError(
            "Current predictions contain null player_key before context join"
        )
    if predictions["player_key"].duplicated().any():
        raise ValueError(
            "Current predictions contain duplicate player_key before context join"
        )

    key_columns = ["player_key", "pos", "year"]
    context_value_columns = [
        "team",
        "current_avg_proj_points",
        "avg_proj_points",
        "model_input_avg_pick",
        "adp_avg_pick",
        "avg_pick",
        "year_exp",
        "adp_year_exp",
        "source_year_exp",
        "year_exp_source",
        "year_exp_uncapped_delta",
        "qb_team_rank",
        "qb_team_rank_bucket",
        "current_team_source",
        "current_adp_source",
        "current_context_missing_optional_fields",
        "projection_context_scoring_hash",
        "projection_context_run_id",
        *MATCH_OUTPUT_COLS,
    ]

    def prepare_context(frame, marker):
        frame = frame.copy()
        missing_keys = [column for column in key_columns if column not in frame]
        if missing_keys:
            raise ValueError(
                f"{marker} current context lacks key columns: {missing_keys}"
            )
        frame = frame[frame["player_key"].notna()].copy()
        frame["player_key"] = frame["player_key"].astype("string")
        frame["pos"] = frame["pos"].astype("string").str.upper()
        if frame.duplicated(key_columns).any():
            preview = frame.loc[
                frame.duplicated(key_columns, keep=False),
                key_columns,
            ].head(10)
            raise ValueError(
                f"{marker} current context has duplicate canonical keys: "
                f"{preview.to_dict('records')}"
            )
        keep_columns = key_columns + [
            column
            for column in context_value_columns
            if column in frame.columns
        ]
        for provenance_column in (
            "current_context_source",
            "current_context_match_method",
        ):
            if provenance_column in frame:
                keep_columns.append(provenance_column)
        prepared = frame[keep_columns].copy()
        prepared[f"_{marker}_context_match"] = 1
        return prepared

    model_context = prepare_context(model_input_context, "model_input")
    fallback_context = prepare_context(v2_fallback_context, "v2_fallback")
    output = predictions.merge(
        model_context,
        on=key_columns,
        how="left",
        validate="one_to_one",
        suffixes=("", "_model_input"),
    )
    # Ensure every model-input context value receives an unambiguous suffix
    # before the fallback merge, including columns absent from predictions.
    rename_model = {
        column: f"{column}_model_input"
        for column in context_value_columns
        if column in output.columns
    }
    rename_model.update(
        {
            column: f"{column}_model_input"
            for column in (
                "current_context_source",
                "current_context_match_method",
            )
            if column in output.columns
        }
    )
    output = output.rename(columns=rename_model)
    output = output.merge(
        fallback_context,
        on=key_columns,
        how="left",
        validate="one_to_one",
        suffixes=("", "_v2_fallback"),
    )
    rename_fallback = {
        column: f"{column}_v2_fallback"
        for column in context_value_columns
        if column in output.columns
    }
    rename_fallback.update(
        {
            column: f"{column}_v2_fallback"
            for column in (
                "current_context_source",
                "current_context_match_method",
            )
            if column in output.columns
        }
    )
    output = output.rename(columns=rename_fallback)

    fallback_fields = [[] for _ in range(len(output))]
    prefer_v2_scoring_context = resolve_scoring_matched_context(
        scoring_matched_context
    )
    used_v2_scoring_context = pd.Series(
        False,
        index=output.index,
        dtype=bool,
    )
    for column in context_value_columns:
        model_column = f"{column}_model_input"
        fallback_column = f"{column}_v2_fallback"
        model_values = (
            output[model_column]
            if model_column in output
            else pd.Series(pd.NA, index=output.index)
        )
        fallback_values = (
            output[fallback_column]
            if fallback_column in output
            else pd.Series(pd.NA, index=output.index)
        )
        if (
            prefer_v2_scoring_context
            and column in V2_SCORING_SENSITIVE_CURRENT_CONTEXT_COLS
        ):
            # Model_Inputs projections are DK-scored. They cannot fill or
            # override a scoring-matched league field, even when non-null.
            use_fallback = fallback_values.notna()
            output[column] = fallback_values
            used_v2_scoring_context |= use_fallback
        else:
            use_fallback = model_values.isna() & fallback_values.notna()
            output[column] = model_values.where(
                model_values.notna(),
                fallback_values,
            )
        for index in np.flatnonzero(use_fallback.to_numpy()):
            fallback_fields[index].append(column)

    model_match = output["_model_input_context_match"].eq(1)
    fallback_match = output["_v2_fallback_context_match"].eq(1)
    used_fallback = pd.Series(
        [bool(fields) for fields in fallback_fields],
        index=output.index,
    )
    output["current_context_source"] = np.select(
        [
            model_match & used_v2_scoring_context,
            ~model_match & fallback_match & prefer_v2_scoring_context,
            model_match & used_fallback,
            model_match,
            ~model_match & fallback_match,
        ],
        [
            "model_inputs_with_v2_scoring_context",
            "v2_player_season_features_scoring_context",
            "model_inputs_with_v2_feature_fill",
            "model_inputs_projection_context",
            "v2_player_season_features_fallback",
        ],
        default="missing",
    )
    model_method = output.get(
        "current_context_match_method_model_input",
        pd.Series(pd.NA, index=output.index),
    )
    fallback_method = output.get(
        "current_context_match_method_v2_fallback",
        pd.Series(pd.NA, index=output.index),
    )
    output["current_context_match_method"] = model_method.where(
        model_method.notna(),
        fallback_method,
    ).fillna("unresolved")
    output["current_context_fallback_fields"] = [
        ",".join(fields) for fields in fallback_fields
    ]
    output["current_context_missing_fields"] = output.apply(
        _missing_context_fields,
        columns=CURRENT_CONTEXT_REQUIRED_COLS,
        axis=1,
    )
    if "current_context_missing_optional_fields" not in output:
        output["current_context_missing_optional_fields"] = ""
    output["current_context_missing_optional_fields"] = output[
        "current_context_missing_optional_fields"
    ].fillna("")

    missing_required = output["current_context_missing_fields"].ne("")
    if missing_required.any():
        preview = output.loc[
            missing_required,
            [
                "player",
                "player_key",
                "pos",
                "pred_fp_per_game",
                "current_context_source",
                "current_context_missing_fields",
            ],
        ].head(20)
        raise ValueError(
            "Recommendation-eligible production rows lack required key-first "
            f"template context: {preview.to_dict('records')}"
        )

    drop_columns = [
        column
        for column in output.columns
        if column.endswith("_model_input")
        or column.endswith("_v2_fallback")
        or column
        in {
            "_model_input_context_match",
            "_v2_fallback_context_match",
        }
    ]
    return output.drop(columns=drop_columns)


def recompute_selected_universe_match_features(
    player_map,
    *,
    preserve_signed_team_qb_context=False,
):
    """Rebuild relative features with one canonical, temporary team key.

    Current source tables use a mix of provider team labels (for example,
    ``LA``/``LAR`` and ``ARZ``/``ARI``).  Team-room and quarterback-depth
    features must treat those aliases as one franchise, while the outward
    player-map contract continues to expose the source label.  Free-agent
    sentinels remain ineligible for team grouping through ``has_current_team``.
    """

    player_map = player_map.copy()
    regenerated_match_columns = sorted(
        set(MATCH_OUTPUT_COLS)
        - set(PROJECTION_COMPONENT_COLS)
        - set(PROJECTION_UNCERTAINTY_SOURCE_COLS)
    )
    regenerated_auxiliary_columns = [
        "team_rb_rush_points",
        "team_rb_rec_points",
        "team_rec_points",
        "pass_catcher_share_of_room",
    ]
    player_map = player_map.drop(
        columns=(
            regenerated_match_columns
            + regenerated_auxiliary_columns
        ),
        errors="ignore",
    )

    outward_team_column = "__outward_team_label"
    if outward_team_column in player_map:
        raise ValueError(
            f"Selected-universe context already contains {outward_team_column}"
        )
    player_map[outward_team_column] = player_map["team"]
    player_map["team"] = player_map["team"].map(canonical_team)

    player_map = add_qb_team_rank_fields(
        player_map,
        year_col="year",
        projection_col="current_avg_proj_points",
    )
    player_map["current_projection_ppg"] = (
        pd.to_numeric(
            player_map["current_avg_proj_points"],
            errors="coerce",
        )
        / WEEK_COUNT
    )
    player_map["context_projection_rank_pct"] = (
        player_map.groupby(["year", "pos"])[
            "current_projection_ppg"
        ]
        .rank(method="first", pct=True, ascending=True)
        .astype(float)
    )
    player_map = add_template_match_features(
        player_map,
        group_cols=["year", "pos"],
        rank_pct_col="context_projection_rank_pct",
        total_points_col="current_avg_proj_points",
        projection_ppg_col="current_projection_ppg",
        preserve_signed_team_qb_context=preserve_signed_team_qb_context,
    )
    player_map["team"] = player_map.pop(outward_team_column)
    return player_map


def build_player_map_base(
    v2_database=None,
    *,
    scoring_matched_context=None,
):
    preds = simulation_dm.read(
        f"""
        SELECT *
        FROM Final_Predictions_Resid
        WHERE year={YEAR}
              AND version='{LEAGUE}'
              AND dataset='{PRED_VERSION}'
        """,
        SIMULATION_DB_NAME,
    )
    v2_database = resolve_v2_database(v2_database)
    if "player_key" not in preds.columns or preds["player_key"].isna().all():
        # Name cleaning remains a compatibility path for legacy, unkeyed
        # publications. Keyed V2 rows already carry the canonical display name;
        # cleaning them would turn names such as Tetairoa McMillan back into a
        # source alias even though identity resolution is already complete.
        preds = clean_player_names(preds)
        preds = attach_v2_player_keys(
            preds,
            v2_database,
            season_column="year",
        )
    elif preds["player_key"].isna().any():
        raise ValueError(
            "Current production prediction slice mixes keyed and unkeyed rows"
        )
    preds["player_key"] = preds["player_key"].astype("string")
    if preds["player_key"].duplicated().any():
        raise ValueError(
            "Current production prediction slice has duplicate player_key"
        )
    preds = add_projection_buckets(
        preds,
        value_col="pred_fp_per_game",
        group_cols=["year", "version", "dataset", "pos"],
        pct_col="prediction_rank_pct",
    )

    current_context = load_current_player_context(
        v2_database=v2_database,
    )
    fallback_context = load_v2_current_player_context(
        v2_database=v2_database,
        selected_player_keys=preds["player_key"],
        scoring_matched_context=scoring_matched_context,
    )
    player_map = attach_current_context_by_player_key(
        preds,
        current_context,
        fallback_context,
        scoring_matched_context=scoring_matched_context,
    )
    player_map = add_exp_fields(player_map)
    # Recompute every relative rank and team-room field on the one governed
    # production universe. This keeps V2-only additions comparable with core
    # ProjOnly rows and prevents a larger fallback mart from changing their
    # percentile scale. Team aliases are canonicalized only while those
    # relative features are built; outward source labels remain unchanged.
    player_map = recompute_selected_universe_match_features(
        player_map,
        preserve_signed_team_qb_context=(
            resolve_scoring_matched_context(scoring_matched_context)
            and LEAGUE in SIGNED_PROJECTION_COMPONENT_SHARE_LEAGUES
        ),
    )
    # Projection-level matching is anchored to the final V2 center after the
    # shared-universe context has been rebuilt.
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
        "player_key",
        "player",
        "pos",
        "year",
        "version",
        "dataset",
        "pred_fp_per_game",
        "pred_fp_per_game_ny",
        "pred_appear_current",
        "pred_appear_ny",
        "current_projection_model_version",
        "next_projection_model_version",
        "production_handoff_version",
        "current_projection_source",
        "current_uncertainty_source",
        "independent_current_residual_draw_allowed",
        "next_projection_source",
        "next_uncertainty_source",
        "v2_scoring_hash",
        "projection_context_scoring_hash",
        "projection_context_run_id",
        "current_avg_proj_points",
        "avg_proj_points",
        "model_input_avg_pick",
        "adp_avg_pick",
        "adp_year_exp",
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
        *CURRENT_CONTEXT_PROVENANCE_COLS,
    ]
    cols = [column for column in cols if column in player_map]
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


def build_adp_audit(player_map, v2_database=None):
    audit_cols = [
        "player_key",
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
        "model_input_avg_pick",
        "adp_avg_pick",
        "adp_year_exp",
        "year_exp",
        *CURRENT_CONTEXT_PROVENANCE_COLS,
    ]
    audit_cols = [col for col in audit_cols if col in player_map.columns]
    audit = player_map[audit_cols].copy()
    audit = audit.rename(columns={"avg_pick": "player_map_avg_pick"})
    audit["player_join_key"] = audit["player_key"].astype("string")
    audit["pos_pred_rank"] = (
        audit.groupby(["year", "version", "dataset", "pos"])["pred_fp_per_game"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    audit["projection_avg_pick"] = audit.get(
        "model_input_avg_pick",
        np.nan,
    )
    audit["pipeline_exact_adp_avg_pick"] = audit.get(
        "adp_avg_pick",
        np.nan,
    )
    audit["pipeline_context_avg_pick"] = audit["player_map_avg_pick"]
    audit["pipeline_year_exp"] = audit.get("year_exp", np.nan)
    audit["avg_adp_year_exp"] = audit.get("adp_year_exp", np.nan)

    adp_league = current_adp_source_league()
    avg_adp = simulation_dm.read(
        f"""
        SELECT player_key,
               player avg_adp_player,
               CAST(year AS INTEGER) year,
               league,
               avg_pick avg_adp_pick,
               std_dev avg_adp_std_dev,
               min_pick avg_adp_min_pick,
               max_pick avg_adp_max_pick,
               Years_of_Experience avg_adp_year_exp_app_match,
               identity_match_method avg_adp_key_match_method
        FROM Avg_ADPs
        WHERE year={YEAR}
              AND league='{adp_league}'
              AND pos IN ('QB', 'RB', 'WR', 'TE')
        """,
        SIMULATION_DB_NAME,
    )

    if len(avg_adp) == 0:
        avg_adp = pd.DataFrame(
            columns=[
                "player_key",
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
                "avg_adp_key_match_method",
            ]
        )
    else:
        avg_adp = validate_published_avg_adp_keys(
            avg_adp,
            f"{LEAGUE}_{adp_league}_adp_audit",
        )
        avg_adp["avg_adp_join_key"] = avg_adp[
            "player_key"
        ].astype("string")
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
        avg_adp.drop(columns=["player_key"], errors="ignore"),
        left_on=["player_join_key", "year"],
        right_on=["avg_adp_join_key", "year"],
        how="left",
        validate="one_to_one",
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
    audit["governed_context_adp_fallback"] = (
        audit["using_player_map_fallback"]
        & audit["player_map_avg_pick"].gt(0)
        & audit["player_map_avg_pick_source"].isin(
            [
                "model_input_projection",
                "pipeline_exact_avg_adp",
            ]
        )
        & audit.get(
            "current_context_source",
            pd.Series("missing", index=audit.index),
        ).ne("missing")
    )
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
    audit["high_impact_unresolved_adp"] = (
        audit["high_impact_missing_avg_adp"]
        & ~audit["governed_context_adp_fallback"]
    )
    audit["needs_review"] = (
        audit["high_impact_unresolved_adp"]
        | (audit["using_default_adp"] & audit["pred_fp_per_game"].gt(0))
        | audit["duplicate_avg_adp_join_key"]
    )

    def issue_type(row):
        issues = []
        if row.using_default_adp:
            issues.append("missing_avg_adp_default_240")
        elif row.governed_context_adp_fallback:
            issues.append("governed_context_adp_fallback")
        elif row.using_player_map_fallback:
            issues.append("unresolved_player_map_adp_fallback")
        if row.duplicate_avg_adp_join_key:
            issues.append("duplicate_avg_adp_join_key")
        if len(issues) == 0:
            issues.append("ok")
        return ",".join(issues)

    audit["issue_type"] = audit.apply(issue_type, axis=1)

    ordered_cols = [
        "player_key",
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
        "avg_adp_key_match_method",
        "missing_avg_adp_match",
        "using_player_map_fallback",
        "using_default_adp",
        "governed_context_adp_fallback",
        "duplicate_avg_adp_join_key",
        "high_projection_player",
        "high_impact_missing_avg_adp",
        "high_impact_unresolved_adp",
        "needs_review",
        "issue_type",
        *CURRENT_CONTEXT_PROVENANCE_COLS,
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


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sqlite_file_integrity(database_path):
    database_path = Path(database_path).resolve()
    try:
        with closing(
            sqlite3.connect(
                f"file:{database_path.as_posix()}?mode=ro",
                uri=True,
            )
        ) as connection:
            results = [
                str(row[0])
                for row in connection.execute("PRAGMA integrity_check")
            ]
    except sqlite3.DatabaseError as exc:
        raise ValueError(
            f"SQLite integrity check failed for {database_path}: {exc}"
        ) from exc
    if results != ["ok"]:
        raise ValueError(
            f"SQLite integrity check failed for {database_path}: {results}"
        )


def _assert_no_active_sqlite_sidecars(database_path):
    database_path = Path(database_path).resolve()
    active = []
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{database_path}{suffix}")
        try:
            size_bytes = sidecar.stat().st_size
        except FileNotFoundError:
            continue
        if size_bytes > 0:
            active.append(f"{sidecar.name} ({size_bytes} bytes)")
    if active:
        raise ValueError(
            "Cannot byte-copy SQLite while an active sidecar exists: "
            + ", ".join(active)
        )


def copy_sqlite_database_atomic(source, destination):
    """Replace a live SQLite file only after an exact verified sibling copy."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("SQLite source and destination must differ")
    if not source.is_file():
        raise FileNotFoundError(f"SQLite source does not exist: {source}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            "SQLite destination directory does not exist: "
            f"{destination.parent}"
        )

    _assert_no_active_sqlite_sidecars(source)
    _assert_no_active_sqlite_sidecars(destination)
    _validate_sqlite_file_integrity(source)
    source_size_before = source.stat().st_size
    source_sha256_before = _file_sha256(source)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(source, temp_path)
        with temp_path.open("r+b") as copied_file:
            copied_file.flush()
            os.fsync(copied_file.fileno())

        copied_size = temp_path.stat().st_size
        copied_sha256 = _file_sha256(temp_path)
        source_size_after = source.stat().st_size
        source_sha256_after = _file_sha256(source)
        _assert_no_active_sqlite_sidecars(source)
        if not (
            source_size_before
            == copied_size
            == source_size_after
        ):
            raise ValueError(
                "SQLite copy size verification failed: "
                f"source_before={source_size_before}, copied={copied_size}, "
                f"source_after={source_size_after}"
            )
        if not (
            source_sha256_before
            == copied_sha256
            == source_sha256_after
        ):
            raise ValueError(
                "SQLite copy SHA-256 verification failed; the source changed "
                "during copy or the sibling copy differs"
            )
        _validate_sqlite_file_integrity(temp_path)
        _assert_no_active_sqlite_sidecars(destination)
        os.replace(temp_path, destination)
        return {
            "size_bytes": copied_size,
            "sha256": copied_sha256,
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()


def synchronize_sqlite_tables_atomic(source, destination, table_names):
    """Atomically replace selected app tables and verify exact row parity."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    table_names = tuple(dict.fromkeys(table_names))
    if not table_names:
        raise ValueError("At least one SQLite table is required for sync")
    if not source.is_file():
        raise FileNotFoundError(f"SQLite source does not exist: {source}")
    if not destination.is_file():
        raise FileNotFoundError(
            f"SQLite destination does not exist: {destination}"
        )

    _validate_sqlite_file_integrity(source)
    row_counts = {}
    with closing(sqlite3.connect(destination)) as app_conn:
        app_conn.execute("ATTACH DATABASE ? AS source_db", (str(source),))
        app_conn.execute("BEGIN IMMEDIATE")
        try:
            placeholders = ", ".join("?" for _ in table_names)
            source_table_rows = {
                str(row[0]): str(row[1])
                for row in app_conn.execute(
                    "SELECT name, sql FROM source_db.sqlite_master "
                    f"WHERE type='table' AND name IN ({placeholders})",
                    table_names,
                )
            }
            missing_tables = sorted(
                set(table_names).difference(source_table_rows)
            )
            if missing_tables:
                raise ValueError(
                    "SQLite table sync source is missing: "
                    + ", ".join(missing_tables)
                )
            source_index_rows = {
                str(row[0]): (str(row[1]), str(row[2]))
                for row in app_conn.execute(
                    "SELECT name, tbl_name, sql "
                    "FROM source_db.sqlite_master "
                    "WHERE type='index' AND sql IS NOT NULL "
                    f"AND tbl_name IN ({placeholders}) "
                    "ORDER BY name",
                    table_names,
                )
            }

            for table_name in table_names:
                create_sql = source_table_rows[table_name]
                app_conn.execute(
                    f'DROP TABLE IF EXISTS main."{table_name}"'
                )
                app_conn.execute(create_sql)
                app_conn.execute(
                    f'INSERT INTO main."{table_name}" '
                    f'SELECT * FROM source_db."{table_name}"'
                )

                source_count = int(
                    app_conn.execute(
                        f'SELECT COUNT(*) FROM source_db."{table_name}"'
                    ).fetchone()[0]
                )
                destination_count = int(
                    app_conn.execute(
                        f'SELECT COUNT(*) FROM main."{table_name}"'
                    ).fetchone()[0]
                )
                if destination_count != source_count:
                    raise ValueError(
                        "SQLite table sync row-count mismatch for "
                        f"{table_name}: source={source_count}, "
                        f"destination={destination_count}"
                    )

                columns = [
                    str(row[1])
                    for row in app_conn.execute(
                        f'PRAGMA source_db.table_info("{table_name}")'
                    )
                ]
                if not columns:
                    raise ValueError(
                        f"SQLite table sync source has no columns: {table_name}"
                    )
                quoted_columns = ", ".join(
                    '"' + column.replace('"', '""') + '"'
                    for column in columns
                )
                grouped_values = f"{quoted_columns}, COUNT(*)"
                mismatch = app_conn.execute(
                    "SELECT EXISTS("
                    f"SELECT {grouped_values} "
                    f'FROM source_db."{table_name}" '
                    f"GROUP BY {quoted_columns} "
                    "EXCEPT "
                    f"SELECT {grouped_values} "
                    f'FROM main."{table_name}" '
                    f"GROUP BY {quoted_columns}"
                    ")"
                ).fetchone()[0]
                if mismatch:
                    raise ValueError(
                        f"SQLite table sync content mismatch for {table_name}"
                    )
                row_counts[table_name] = source_count

            for _, index_sql in source_index_rows.values():
                app_conn.execute(index_sql)
            destination_index_rows = {
                str(row[0]): (str(row[1]), str(row[2]))
                for row in app_conn.execute(
                    "SELECT name, tbl_name, sql "
                    "FROM main.sqlite_master "
                    "WHERE type='index' AND sql IS NOT NULL "
                    f"AND tbl_name IN ({placeholders}) "
                    "ORDER BY name",
                    table_names,
                )
            }
            if destination_index_rows != source_index_rows:
                raise ValueError(
                    "SQLite table sync explicit-index parity mismatch: "
                    f"source={source_index_rows}, "
                    f"destination={destination_index_rows}"
                )

            integrity_results = [
                str(row[0])
                for row in app_conn.execute("PRAGMA main.integrity_check")
            ]
            if integrity_results != ["ok"]:
                raise ValueError(
                    "Auction SQLite integrity check failed before commit: "
                    f"{integrity_results}"
                )
            app_conn.commit()
        except Exception:
            app_conn.rollback()
            raise

    _validate_sqlite_file_integrity(destination)
    return row_counts


def _reserve_sibling_path(destination, marker):
    destination = Path(destination).resolve()
    descriptor, reserved_name = tempfile.mkstemp(
        prefix=f".{destination.name}.{marker}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    reserved_path = Path(reserved_name)
    reserved_path.unlink()
    return reserved_path


def _capture_sqlite_file_state(database_path):
    database_path = Path(database_path).resolve()
    if not database_path.exists():
        return {"destination_existed": False}
    if not database_path.is_file():
        raise ValueError(
            f"SQLite destination is not a regular file: {database_path}"
        )
    _assert_no_active_sqlite_sidecars(database_path)
    _validate_sqlite_file_integrity(database_path)
    size_before = database_path.stat().st_size
    sha256 = _file_sha256(database_path)
    size_after = database_path.stat().st_size
    _assert_no_active_sqlite_sidecars(database_path)
    if size_before != size_after:
        raise ValueError(
            f"SQLite destination changed while it was fingerprinted: "
            f"{database_path}"
        )
    return {
        "destination_existed": True,
        "prior_destination_size_bytes": size_after,
        "prior_destination_sha256": sha256,
    }


def _assert_sqlite_destination_unchanged(artifact):
    destination = artifact["destination"]
    label = artifact["label"]
    expected_exists = artifact["destination_existed"]
    if destination.exists() != expected_exists:
        raise ValueError(
            f"{label} live SQLite existence changed since staging"
        )
    if not expected_exists:
        return
    _assert_no_active_sqlite_sidecars(destination)
    actual_size = destination.stat().st_size
    actual_sha256 = _file_sha256(destination)
    _assert_no_active_sqlite_sidecars(destination)
    if (
        actual_size != artifact["prior_destination_size_bytes"]
        or actual_sha256 != artifact["prior_destination_sha256"]
    ):
        raise ValueError(
            f"{label} live SQLite changed since staging; refusing to "
            "overwrite newer app-owned state"
        )


def promote_sqlite_artifacts_with_rollback(artifacts):
    """Promote prepared app databases and restore earlier files on failure."""

    normalized = []
    for artifact in artifacts:
        label = str(artifact["label"])
        staged = Path(artifact["staged"]).resolve()
        destination = Path(artifact["destination"]).resolve()
        expected_size = int(artifact["size_bytes"])
        expected_sha256 = str(artifact["sha256"])
        if "destination_existed" not in artifact:
            raise ValueError(
                f"{label} artifact lacks its staged destination state"
            )
        destination_existed = bool(artifact["destination_existed"])
        prior_destination_size = artifact.get(
            "prior_destination_size_bytes"
        )
        prior_destination_sha256 = artifact.get(
            "prior_destination_sha256"
        )
        if destination_existed and (
            prior_destination_size is None
            or prior_destination_sha256 is None
        ):
            raise ValueError(
                f"{label} artifact lacks its prior destination fingerprint"
            )
        if not staged.is_file():
            raise FileNotFoundError(
                f"{label} staged SQLite artifact does not exist: {staged}"
            )
        _assert_no_active_sqlite_sidecars(staged)
        _validate_sqlite_file_integrity(staged)
        if staged.stat().st_size != expected_size:
            raise ValueError(
                f"{label} staged SQLite size changed before promotion"
            )
        if _file_sha256(staged) != expected_sha256:
            raise ValueError(
                f"{label} staged SQLite SHA-256 changed before promotion"
            )
        normalized.append(
            {
                "label": label,
                "staged": staged,
                "destination": destination,
                "size_bytes": expected_size,
                "sha256": expected_sha256,
                "destination_existed": destination_existed,
                "prior_destination_size_bytes": (
                    None
                    if prior_destination_size is None
                    else int(prior_destination_size)
                ),
                "prior_destination_sha256": (
                    None
                    if prior_destination_sha256 is None
                    else str(prior_destination_sha256)
                ),
            }
        )

    promoted = []
    try:
        # Preflight every target before changing either app, then repeat the
        # check immediately before each individual promotion to narrow the
        # optimistic-concurrency window.
        for artifact in normalized:
            _assert_sqlite_destination_unchanged(artifact)
        for artifact in normalized:
            label = artifact["label"]
            staged = artifact["staged"]
            destination = artifact["destination"]
            _assert_sqlite_destination_unchanged(artifact)
            backup = None
            if destination.exists():
                backup = _reserve_sibling_path(
                    destination,
                    "pre_release_backup",
                )
                os.replace(destination, backup)
                backup_size = backup.stat().st_size
                backup_sha256 = _file_sha256(backup)
                if (
                    backup_size
                    != artifact["prior_destination_size_bytes"]
                    or backup_sha256
                    != artifact["prior_destination_sha256"]
                ):
                    os.replace(backup, destination)
                    raise ValueError(
                        f"{label} live SQLite changed during promotion; "
                        "restored the newer app state"
                    )
            try:
                os.replace(staged, destination)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                raise
            promoted.append(
                {
                    **artifact,
                    "backup": backup,
                }
            )

        for artifact in promoted:
            destination = artifact["destination"]
            _validate_sqlite_file_integrity(destination)
            if destination.stat().st_size != artifact["size_bytes"]:
                raise ValueError(
                    f"{artifact['label']} promoted SQLite size mismatch"
                )
            if _file_sha256(destination) != artifact["sha256"]:
                raise ValueError(
                    f"{artifact['label']} promoted SQLite SHA-256 mismatch"
                )
    except Exception as promotion_error:
        rollback_errors = []
        for artifact in reversed(promoted):
            destination = artifact["destination"]
            backup = artifact["backup"]
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                elif destination.exists():
                    destination.unlink()
            except Exception as rollback_error:
                rollback_errors.append(
                    f"{artifact['label']}: {rollback_error}"
                )
        if rollback_errors:
            raise RuntimeError(
                "App database promotion failed and rollback was incomplete; "
                "preserved sibling backups require manual recovery: "
                + "; ".join(rollback_errors)
            ) from promotion_error
        raise

    for artifact in promoted:
        backup = artifact["backup"]
        if backup is not None and backup.exists():
            try:
                backup.unlink()
            except OSError as cleanup_error:
                print(
                    "Warning: app database promotion succeeded but a temporary "
                    f"backup could not be removed: {backup} ({cleanup_error})"
                )


def validate_weekly_template_export(connection):
    """Validate every retained league against its own weekly horizon."""

    template_cols = {
        row[1]
        for row in connection.execute(
            f'PRAGMA table_info("{TEMPLATE_TABLE}")'
        )
    }
    missing_base_cols = sorted({"league", "player_key"} - template_cols)
    if missing_base_cols:
        raise ValueError(
            "Production app export weekly-template schema is incomplete: "
            + ", ".join(missing_base_cols)
        )

    missing_league_rows = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM "{TEMPLATE_TABLE}"
            WHERE league IS NULL
               OR TRIM(CAST(league AS TEXT))=''
            """
        ).fetchone()[0]
    )
    if missing_league_rows:
        raise ValueError(
            "Production app export is incomplete because "
            f"{missing_league_rows} retained template rows lack league."
        )

    retained_leagues = {
        str(row[0]).strip().lower()
        for row in connection.execute(
            f'SELECT DISTINCT league FROM "{TEMPLATE_TABLE}"'
        )
    }
    unsupported_leagues = sorted(
        retained_leagues - set(WEEK_COUNT_BY_LEAGUE)
    )
    if unsupported_leagues:
        raise ValueError(
            "Production app export contains weekly templates for unsupported "
            "leagues: "
            + ", ".join(unsupported_leagues)
        )

    validated_horizons = {}
    for league in sorted(retained_leagues):
        horizon = int(WEEK_COUNT_BY_LEAGUE[league])
        week_cols = (
            [f"managed_week_{week}" for week in range(1, horizon + 1)]
            + [f"played_week_{week}" for week in range(1, horizon + 1)]
        )
        required_cols = ["player_key", *week_cols]
        missing_cols = sorted(set(required_cols) - template_cols)
        if missing_cols:
            raise ValueError(
                "Production app export weekly-template schema is incomplete "
                f"for {league}: "
                + ", ".join(missing_cols)
            )
        null_predicate = " OR ".join(
            f'"{column}" IS NULL' for column in week_cols
        )
        incomplete_rows = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM "{TEMPLATE_TABLE}"
                WHERE LOWER(TRIM(CAST(league AS TEXT)))=?
                  AND (
                      player_key IS NULL
                      OR TRIM(CAST(player_key AS TEXT))=''
                      OR {null_predicate}
                  )
                """,
                (league,),
            ).fetchone()[0]
        )
        if incomplete_rows:
            raise ValueError(
                "Production app export is incomplete because "
                f"{incomplete_rows} retained {league} template rows lack "
                f"canonical-key or played/managed-week fields through week "
                f"{horizon}."
            )
        validated_horizons[league] = horizon
    return validated_horizons


def copy_simulation_db_to_apps():
    src = SIMULATION_DB_PATH
    generated_tables = [
        AVG_ADP_TABLE,
        AVG_ADP_AUDIT_TABLE,
        AVG_ADP_RECEIPT_TABLE,
        "Final_Predictions_Resid",
        "V2_Production_Projection_Handoff",
        "V2_Production_Projection_Audit",
        "V2_Production_Eligibility_Audit",
        TEMPLATE_TABLE,
        POOL_TABLE,
        POOL_SUMMARY_TABLE,
        PLAYER_MAP_TABLE,
        TEMPLATE_AUDIT_TABLE,
        PLAYER_POOL_AUDIT_TABLE,
        BUCKET_AUDIT_TABLE,
        ADP_AUDIT_TABLE,
    ]
    with closing(sqlite3.connect(src)) as source_conn:
        source_tables = {
            row[0]
            for row in source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing_generated = sorted(
            set(generated_tables) - source_tables
        )
        if missing_generated:
            raise ValueError(
                "Production app export is missing required generated tables: "
                + ", ".join(missing_generated)
            )
        avg_adps = pd.read_sql_query(
            f'SELECT * FROM "{AVG_ADP_TABLE}"',
            source_conn,
        )
        validate_avg_adp_publication(avg_adps, year=YEAR)
        validate_weekly_template_export(source_conn)
        player_map_cols = {
            row[1]
            for row in source_conn.execute(
                f'PRAGMA table_info("{PLAYER_MAP_TABLE}")'
            )
        }
        if "player_key" not in player_map_cols:
            raise ValueError(
                "Production app export is incomplete because the current "
                "player map "
                "does not contain player_key."
            )
        missing_player_map_keys = source_conn.execute(
            f'SELECT COUNT(*) FROM "{PLAYER_MAP_TABLE}" '
            'WHERE "player_key" IS NULL'
        ).fetchone()[0]
        if missing_player_map_keys:
            raise ValueError(
                "Production app export is incomplete because "
                f"{missing_player_map_keys} current player-map rows have null "
                "player_key."
            )
        final_prediction_cols = {
            row[1]
            for row in source_conn.execute(
                'PRAGMA table_info("Final_Predictions_Resid")'
            )
        }
        required_prediction_cols = {
            "player_key",
            "pred_appear_ny",
            "current_uncertainty_source",
            "independent_current_residual_draw_allowed",
            "production_handoff_version",
        }
        missing_prediction_cols = sorted(
            required_prediction_cols - final_prediction_cols
        )
        if missing_prediction_cols:
            raise ValueError(
                "Production projection handoff schema is incomplete: "
                + ", ".join(missing_prediction_cols)
            )
        incomplete_projection_rows = source_conn.execute(
            """
            SELECT COUNT(*)
            FROM Final_Predictions_Resid
            WHERE year=?
              AND dataset=?
              AND version IN ('dk', 'nffc', 'beta')
              AND (
                    player_key IS NULL
                 OR pred_fp_per_game IS NULL
                 OR pred_fp_per_game_ny IS NULL
                 OR pred_appear_ny IS NULL
                 OR current_uncertainty_source != 'joint_weekly_template_only'
                 OR independent_current_residual_draw_allowed != 0
              )
            """,
            (YEAR, PRED_VERSION),
        ).fetchone()[0]
        if incomplete_projection_rows:
            raise ValueError(
                f"{incomplete_projection_rows} DK/NFFC/beta projection rows violate "
                "the V2 production handoff"
            )

    sibling_root = Path(root_path).resolve().parent
    auction_dst = sibling_root / "Fantasy_Football_App" / "app" / "Simulation.sqlite3"
    snake_dst = sibling_root / "Fantasy_Football_Snake" / "app" / "Simulation.sqlite3"
    app_targets = []
    staged_paths = []
    with tempfile.TemporaryDirectory(
        prefix="ff_simulation_release_"
    ) as snapshot_directory:
        source_snapshot = (
            Path(snapshot_directory) / "Simulation.sqlite3"
        )
        source_receipt = copy_sqlite_database_atomic(
            src,
            source_snapshot,
        )
        auction_row_counts = None
        try:
            if auction_dst.parent.exists():
                auction_stage = _reserve_sibling_path(
                    auction_dst,
                    "release_stage",
                )
                staged_paths.append(auction_stage)
                if auction_dst.exists():
                    prior_auction_receipt = copy_sqlite_database_atomic(
                        auction_dst,
                        auction_stage,
                    )
                    prior_auction_state = {
                        "destination_existed": True,
                        "prior_destination_size_bytes": (
                            prior_auction_receipt["size_bytes"]
                        ),
                        "prior_destination_sha256": (
                            prior_auction_receipt["sha256"]
                        ),
                    }
                    auction_row_counts = synchronize_sqlite_tables_atomic(
                        source_snapshot,
                        auction_stage,
                        generated_tables,
                    )
                else:
                    prior_auction_state = {
                        "destination_existed": False,
                    }
                    copy_sqlite_database_atomic(
                        source_snapshot,
                        auction_stage,
                    )
                app_targets.append(
                    {
                        "label": "Auction",
                        "staged": auction_stage,
                        "destination": auction_dst,
                        "size_bytes": auction_stage.stat().st_size,
                        "sha256": _file_sha256(auction_stage),
                        **prior_auction_state,
                    }
                )

            if snake_dst.parent.exists():
                prior_snake_state = _capture_sqlite_file_state(snake_dst)
                snake_stage = _reserve_sibling_path(
                    snake_dst,
                    "release_stage",
                )
                staged_paths.append(snake_stage)
                snake_receipt = copy_sqlite_database_atomic(
                    source_snapshot,
                    snake_stage,
                )
                app_targets.append(
                    {
                        "label": "Snake",
                        "staged": snake_stage,
                        "destination": snake_dst,
                        **snake_receipt,
                        **prior_snake_state,
                    }
                )

            # Probe/promote Snake first because it is a full-file replacement
            # and is the destination most likely to be held open by an app.
            app_targets.sort(
                key=lambda artifact: artifact["label"] != "Snake"
            )
            promote_sqlite_artifacts_with_rollback(app_targets)
        finally:
            for staged_path in staged_paths:
                if staged_path.exists():
                    staged_path.unlink()

    for artifact in app_targets:
        if artifact["label"] == "Auction" and auction_row_counts is not None:
            print(
                f"Synchronized {len(generated_tables)} generated production "
                f"tables ({sum(auction_row_counts.values())} rows) to "
                f"{artifact['destination']}"
            )
        else:
            print(
                "Copied and verified Simulation.sqlite3 to "
                f"{artifact['destination']} "
                f"({artifact['size_bytes']} bytes, "
                f"sha256={artifact['sha256']})"
            )
    if app_targets:
        print(
            "App release source snapshot: "
            f"{source_receipt['size_bytes']} bytes, "
            f"sha256={source_receipt['sha256']}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build league-specific best-ball weekly templates."
    )
    parser.add_argument(
        "--league",
        choices=sorted(TEMPLATE_ID_LEAGUE_OFFSETS),
        default=None,
        help=f"Scoring/output league (default: {LEAGUE}).",
    )
    parser.add_argument(
        "--simulation-db",
        type=Path,
        default=None,
        help="Existing Simulation.sqlite3-compatible staging target.",
    )
    parser.add_argument(
        "--v2-db",
        type=Path,
        default=None,
        help="V2 database used for both locked centers and player identity.",
    )
    parser.add_argument(
        "--no-app-sync",
        action="store_true",
        help="Do not synchronize generated tables to application databases.",
    )
    return parser.parse_args(argv)


def main(
    league=None,
    simulation_db=None,
    v2_database=None,
    sync_apps=True,
):
    active_league = set_active_league(LEAGUE if league is None else league)
    if simulation_db is not None:
        set_simulation_db(simulation_db)
    if SIMULATION_DB_PATH != DEFAULT_SIMULATION_DB_PATH and sync_apps:
        raise ValueError(
            "A custom simulation database requires sync_apps=False "
            "(CLI: --no-app-sync)."
        )
    if (
        SIMULATION_DB_PATH != DEFAULT_SIMULATION_DB_PATH
        and v2_database is None
    ):
        raise ValueError(
            "A custom simulation database requires an explicit v2_database "
            "(CLI: --v2-db) to prevent mixing staged and live inputs."
        )
    active_v2_database = resolve_v2_database(
        v2_database,
        league=active_league,
    )
    live_v2_databases = {
        Path(path).resolve()
        for path in [V2_IDENTITY_DB_PATH, *V2_DATABASES.values()]
    }
    configured_v2_database = Path(
        V2_DATABASES.get(active_league, V2_IDENTITY_DB_PATH)
    ).resolve()
    if (
        SIMULATION_DB_PATH == DEFAULT_SIMULATION_DB_PATH
        and active_v2_database != configured_v2_database
    ):
        raise ValueError(
            "The live Simulation database requires the configured "
            f"{active_league} V2 database: {configured_v2_database}"
        )
    if (
        SIMULATION_DB_PATH != DEFAULT_SIMULATION_DB_PATH
        and active_v2_database in live_v2_databases
    ):
        raise ValueError(
            "A custom simulation database requires a staged V2 database "
            "copy, not a live V2 database."
        )

    max_template_season = min(YEAR - 1, get_daily_max_template_season())
    print(
        f"Building weekly templates for {TEMPLATE_SEASON_MIN}-{max_template_season} "
        f"using {active_league} scoring and {WEEK_COUNT} weeks..."
    )

    proj = load_historical_projection_context(
        max_template_season,
        v2_database=active_v2_database,
    )
    weekly = load_weekly_points(
        max_template_season,
        league=active_league,
    )

    templates = build_weekly_templates(
        proj,
        weekly,
        league=active_league,
    )
    template_audit = build_template_join_audit(templates)
    player_map_base = build_player_map_base(
        v2_database=active_v2_database,
    )
    bucket_audit = build_bucket_comparability_audit(proj, player_map_base)
    pool_members, pool_summary = build_pool_tables(templates, player_map_base)
    player_map = finalize_player_map(player_map_base, pool_summary)
    templates, player_map = attach_weekly_handoff_player_keys(
        templates,
        player_map,
        v2_database=active_v2_database,
    )
    player_pool_audit = build_player_pool_audit(player_map, pool_members, templates)
    adp_audit = build_adp_audit(
        player_map,
        v2_database=active_v2_database,
    )
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

    if sync_apps:
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
            governed_fallback=("governed_context_adp_fallback", "sum"),
            unresolved=("high_impact_unresolved_adp", "sum"),
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
    args = parse_args()
    main(
        league=args.league,
        simulation_db=args.simulation_db,
        v2_database=args.v2_db,
        sync_apps=not args.no_app_sync,
    )

#%%
