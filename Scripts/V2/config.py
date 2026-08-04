"""Configuration for the isolated V2 projection pipeline."""

from pathlib import Path

from Scripts.config import LEAGUE, YEAR


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Season_Stats_New.sqlite3"
OUTPUT_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3"

POSITIONS = ("QB", "RB", "WR", "TE")
START_SEASON = 2006
COMPLETED_THROUGH_SEASON = YEAR - 1
PROJECTION_THROUGH_SEASON = YEAR

USEFUL_SEASON_MIN_GAMES = 4
QB_MIN_OFFENSIVE_PLAYS = 15

NFLVERSE_PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "players/players.csv"
)
NFLVERSE_WEEKLY_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)

# Declarative corrections for confirmed source-season labeling errors.  The
# shared contracts helper applies these rules to identity and feature rows,
# retains the stored season and rationale as audit metadata, and rejects a
# correction if native rows already exist for the effective table/position/year.
SOURCE_SEASON_OVERRIDES = (
    {
        "override_id": "fantasypros_wr_2016_to_2018_v1",
        "source_table": "FantasyPros_Projections",
        "position": "WR",
        "stored_season": 2016,
        "effective_season": 2018,
        "reason": (
            "The archived FantasyPros WR loader entry stored under 2016 "
            "contains the 2018 projection vintage."
        ),
        "reference": "wayback_timestamp=20180808115212",
    },
    {
        "override_id": "fantasypros_wr_2020_to_2021_v1",
        "source_table": "FantasyPros_Projections",
        "position": "WR",
        "stored_season": 2020,
        "effective_season": 2021,
        "reason": (
            "The archived FantasyPros WR loader entry stored under 2020 "
            "contains the 2021 projection vintage."
        ),
        "reference": "wayback_timestamp=20210728120136",
    },
)

# Declarative quarantines for source rows that must not enter any V2 identity,
# candidate, or feature lineage.  These are intentionally distinct from season
# overrides: a quarantined slice is dropped when the archived vintage is
# already represented by native rows under its correct season.
SOURCE_ROW_EXCLUSIONS = (
    {
        "exclusion_id": "fftoday_qb_stored_2018_2019_vintage_quarantine_v1",
        "source_table": "FFToday_Projections",
        "position": "QB",
        "stored_season": 2018,
        "reason": (
            "The FFToday QB rows stored under 2018 match the official 2019 "
            "projection archive, while native 2019 QB rows already exist."
        ),
        "reference": (
            "https://www.fftoday.com/rankings/playerproj.php"
            "?Season=2019&PosID=10"
        ),
    },
)

# Some provider archives retain valid historical projection/market values but
# expose a mutable player-team label that is refreshed after the source season.
# Keep those values while discarding only the contaminated historical team
# label.  A trusted same-season source may still supply team context downstream;
# otherwise the player-season remains deliberately unassigned to a team.
SOURCE_TEAM_TRUST_POLICIES = (
    {
        "policy_id": "ffa_raw_historical_team_labels_untrusted_v1",
        "source_table": "FFA_RawStats",
        "through_season": COMPLETED_THROUGH_SEASON,
        "action": "discard_team",
        "reason": (
            "Historical FFA raw-stat rows contain later-team backfills, "
            "including otherwise-identical 2019 rows that differ only by "
            "the player's later team."
        ),
        "reference": "internal_audit=2026-08-02_historical_team_backfill",
    },
    {
        "policy_id": "ffa_projection_historical_team_labels_untrusted_v1",
        "source_table": "FFA_Projections",
        "through_season": COMPLETED_THROUGH_SEASON,
        "action": "discard_team",
        "reason": (
            "Historical FFA projection rows contain later-team backfills, "
            "including otherwise-identical 2019 rows that differ only by "
            "the player's later team."
        ),
        "reference": "internal_audit=2026-08-02_historical_team_backfill",
    },
    {
        "policy_id": "fantasypros_best_ball_historical_team_labels_untrusted_v1",
        "source_table": "FantasyPros_Best_Ball_ADP",
        "through_season": COMPLETED_THROUGH_SEASON,
        "action": "discard_team",
        "reason": (
            "Historical FantasyPros best-ball ADP rows expose mutable team "
            "labels that reflect later destinations rather than the source "
            "season."
        ),
        "reference": "internal_audit=2026-08-02_historical_team_backfill",
    },
)

# These sources define who was knowable before a season. A row enters the
# projection spine through at least one of these sources; observed outcomes are
# deliberately not a candidate source.
CANDIDATE_SOURCE_TABLES = {
    "Draft_Positions": {
        "source": "nfl_draft",
        "source_kind": "draft",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
        "draft_year": "year",
        "draft_round": "Round",
        "draft_pick": "Pick",
        "college": "college",
    },
    "ADP_Ranks": {
        "source_column": "source",
        "source_prefix": "adp_",
        "source_kind": "market",
        "player": "player",
        "position": "pos",
        "season": "year",
    },
    "ADP_Averages": {
        "source_column": "league",
        "source_prefix": "adp_average_",
        "source_kind": "market",
        "player": "player",
        "position": "pos",
        "season": "year",
    },
    "DraftKings_ADP": {
        "source": "draftkings_adp",
        "source_kind": "market",
        "player": "player",
        "position": "pos",
        "season": "year",
    },
    "FantasyPros_Best_Ball_ADP": {
        "source": "fantasypros_best_ball_adp",
        "source_kind": "market",
        "allow_missing_position": True,
        "player": "player",
        "team": "team",
        "season": "year",
    },
    "FFA_RawStats": {
        "source": "ffa_raw",
        "source_kind": "projection",
        "source_player_id": "ffa_id",
        "player": "player",
        "position": "position",
        "team": "team",
        "season": "year",
    },
    "FFA_Projections": {
        "source": "ffa_projection",
        "source_kind": "projection",
        "player": "player",
        "position": "position",
        "team": "team",
        "season": "year",
    },
    "FFToday_Projections": {
        "source": "fftoday",
        "source_kind": "projection",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "FantasyData": {
        "source": "fantasydata",
        "source_kind": "projection",
        # The subscription is no longer being refreshed. Retain historical
        # evidence, but exclude the static 2026 snapshot from every current
        # candidate, projection-value, and market-value path.
        "through_season": 2025,
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "FantasyPoints_Projections": {
        "source": "fantasypoints",
        "source_kind": "projection",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "FantasyPros_Projections": {
        "source": "fantasypros",
        "source_kind": "projection",
        "player": "player",
        "position": "pos",
        "season": "year",
    },
    "PFF_Projections": {
        "source": "pff_projection",
        "source_kind": "projection",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "FFF_Projections": {
        "source": "fff",
        "source_kind": "projection",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "Fanduel_Projections": {
        "source": "fanduel",
        "source_kind": "projection",
        "allow_missing_position": True,
        "player": "player",
        "season": "year",
    },
    "NFFC_ADP": {
        "source_column": "source",
        "source_prefix": "",
        "source_kind": "market",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "ETR_Ranks": {
        "source": "etr_rank",
        "source_kind": "ranking",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "Evan_Silva_Ranks": {
        "source": "evan_silva_rank",
        "source_kind": "ranking",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "FFF_Ranks": {
        "source": "fff_rank",
        "source_kind": "ranking",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
    "Barret_Ranks": {
        "source": "barret_rank",
        "source_kind": "ranking",
        "player": "player",
        "position": "pos",
        "team": "team",
        "season": "year",
    },
}

# Legacy rookie tables remain identity evidence only. They are not allowed to
# establish projection-spine eligibility because their maintenance and
# historical reliability are not strong enough for that role.
IDENTITY_SOURCE_TABLES = {
    **CANDIDATE_SOURCE_TABLES,
    "Rookie_RB_Stats": {
        "source": "rookie_rb_legacy",
        "player": "player",
        "position_value": "RB",
        "team": "team",
        "season": "draft_year",
        "draft_year": "draft_year",
    },
    "Rookie_WR_Stats": {
        "source": "rookie_wr_legacy",
        "player": "player",
        "position_value": "WR",
        "team": "team",
        "season": "draft_year",
        "draft_year": "draft_year",
    },
}


def candidate_source_kind(source: str) -> str | None:
    """Return the declared candidate-source family for a normalized source."""
    source_value = str(source)
    if source_value.startswith("nffc_"):
        return "market"
    for spec in CANDIDATE_SOURCE_TABLES.values():
        source_kind = str(spec["source_kind"])
        constant = spec.get("source")
        if constant is not None and source_value == constant:
            return source_kind
        prefix = spec.get("source_prefix")
        if spec.get("source_column") and prefix:
            if source_value.startswith(str(prefix)):
                return source_kind
    return None


def candidate_source_through_season(source: str) -> int | None:
    """Return an optional final eligible season for a normalized source."""
    source_value = str(source)
    for spec in CANDIDATE_SOURCE_TABLES.values():
        constant = spec.get("source")
        prefix = spec.get("source_prefix")
        matches = constant is not None and source_value == str(constant)
        if spec.get("source_column") and prefix:
            matches = matches or source_value.startswith(str(prefix))
        if matches:
            through_season = spec.get("through_season")
            return int(through_season) if through_season is not None else None
    return None


TEAM_MAP = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "HST": "HOU",
    "IND": "IND",
    "JAC": "JAC",
    "JAX": "JAC",
    "KC": "KC",
    "KAN": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "STL": "LAR",
    "LAC": "LAC",
    "SD": "LAC",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "OAK": "LVR",
    "LVR": "LVR",
    "LV": "LVR",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WFT": "WAS",
}


# Provider component mappings for the Milestone 3 feature mart. Metrics are
# normalized before consensus so the V2 baseline is scored under one league
# definition rather than averaging provider point totals with unknown scoring.
PROJECTION_VALUE_SPECS = {
    "FFA_RawStats": {
        "source": "ffa_raw",
        "provider": "ffa",
        "metrics": {
            "passing_yards": "ffa_pass_yds",
            "passing_tds": "ffa_pass_tds",
            "interceptions": "ffa_pass_int",
            "rushing_yards": "ffa_rush_yds",
            "rushing_tds": "ffa_rush_tds",
            "receiving_yards": "ffa_rec_yds",
            "receiving_tds": "ffa_rec_tds",
        },
    },
    "FFA_Projections": {
        "source": "ffa_projection",
        "provider": "ffa",
        "metrics": {
            "raw_projected_points": "ffa_points",
            "source_floor_points": "ffa_floor",
            "source_ceiling_points": "ffa_ceiling",
            "source_uncertainty": "ffa_uncertainty",
        },
    },
    "FFToday_Projections": {
        "source": "fftoday",
        "provider": "fftoday",
        "metrics": {
            "pass_completions": "fft_pass_comp",
            "pass_attempts": "fft_pass_att",
            "passing_yards": "fft_pass_yds",
            "passing_tds": "fft_pass_td",
            "interceptions": "fft_pass_int",
            "sacks": "fft_sacks",
            "rush_attempts": "fft_rush_att",
            "rushing_yards": "fft_rush_yds",
            "rushing_tds": "fft_rush_td",
            "receptions": "fft_rec",
            "receiving_yards": "fft_rec_yds",
            "receiving_tds": "fft_rec_td",
        },
    },
    "FantasyData": {
        "source": "fantasydata",
        "provider": "fantasydata",
        "metrics": {
            "passing_yards": "fdta_pass_yds",
            "passing_tds": "fdta_pass_td",
            "interceptions": "fdta_pass_int",
            "rushing_yards": "fdta_rush_yds",
            "rushing_tds": "fdta_rush_td",
            "receptions": "fdta_rec",
            "receiving_yards": "fdta_rec_yds",
            "receiving_tds": "fdta_rec_td",
        },
    },
    "FantasyPoints_Projections": {
        "source": "fantasypoints",
        "provider": "fantasypoints",
        "metrics": {
            "projected_games": "fpts_games",
            "raw_projected_points": "fpts_proj_points",
            "raw_projected_ppg": "fpts_proj_points_per_game",
            "pass_completions": "fpts_pass_cmp",
            "pass_attempts": "fpts_pass_att",
            "passing_yards": "fpts_pass_yds",
            "passing_tds": "fpts_pass_td",
            "interceptions": "fpts_pass_int",
            "rush_attempts": "fpts_rush_att",
            "rushing_yards": "fpts_rush_yds",
            "rushing_tds": "fpts_rush_td",
            "receptions": "fpts_rec",
            "receiving_yards": "fpts_rec_yds",
            "receiving_tds": "fpts_rec_td",
        },
    },
    "FantasyPros_Projections": {
        "source": "fantasypros",
        "provider": "fantasypros",
        "metrics": {
            "raw_projected_points": "fpros_proj_pts",
            "pass_completions": "fpros_pass_cmp",
            "pass_attempts": "fpros_pass_att",
            "passing_yards": "fpros_pass_yds",
            "passing_tds": "fpros_pass_td",
            "interceptions": "fpros_pass_int",
            "rush_attempts": "fpros_rush_att",
            "rushing_yards": "fpros_rush_yds",
            "rushing_tds": "fpros_rush_td",
            "receptions": "fpros_rec",
            "receiving_yards": "fpros_rec_yds",
            "receiving_tds": "fpros_rec_td",
        },
    },
    "PFF_Projections": {
        "source": "pff_projection",
        "provider": "pff",
        "metrics": {
            "projected_games": "pff_games",
            "raw_projected_points": "pff_proj_pts",
            "pass_completions": "pff_pass_comp",
            "pass_attempts": "pff_pass_att",
            "passing_yards": "pff_pass_yds",
            "passing_tds": "pff_pass_td",
            "interceptions": "pff_pass_int",
            "sacks": "pff_pass_sacked",
            "rush_attempts": "pff_rush_att",
            "rushing_yards": "pff_rush_yds",
            "rushing_tds": "pff_rush_td",
            "targets": "pff_rec_targets",
            "receptions": "pff_rec_receptions",
            "receiving_yards": "pff_rec_yds",
            "receiving_tds": "pff_rec_td",
        },
    },
    "FFF_Projections": {
        "source": "fff",
        "provider": "fff",
        "metrics": {
            "pass_completions": "fff_pass_cmp",
            "pass_attempts": "fff_pass_att",
            "passing_yards": "fff_pass_yds",
            "passing_tds": "fff_pass_td",
            "interceptions": "fff_pass_int",
            "rush_attempts": "fff_rush_att",
            "rushing_yards": "fff_rush_yds",
            "rushing_tds": "fff_rush_td",
            "receptions": "fff_rec",
            "receiving_yards": "fff_rec_yds",
            "receiving_tds": "fff_rec_td",
        },
    },
    "Fanduel_Projections": {
        "source": "fanduel",
        "provider": "fanduel",
        "metrics": {
            "pass_completions": "fanduel_pass_cmp",
            "pass_attempts": "fanduel_pass_att",
            "passing_yards": "fanduel_pass_yds",
            "passing_tds": "fanduel_pass_td",
            "interceptions": "fanduel_pass_int",
            "rush_attempts": "fanduel_rush_att",
            "rushing_yards": "fanduel_rush_yds",
            "rushing_tds": "fanduel_rush_td",
            "targets": "fanduel_rec_targets",
            "receptions": "fanduel_rec",
            "receiving_yards": "fanduel_rec_yds",
            "receiving_tds": "fanduel_rec_td",
        },
    },
}


MARKET_VALUE_SPECS = {
    "ADP_Ranks": {
        "source_column": "source",
        "source_prefix": "adp_",
        "metrics": {"adp": "pick"},
    },
    "ADP_Averages": {
        "source_column": "league",
        "source_prefix": "adp_average_",
        "metrics": {"adp": "avg_pick"},
    },
    "DraftKings_ADP": {
        "source": "draftkings_adp",
        "metrics": {"adp": "avg_pick"},
    },
    "FantasyPros_Best_Ball_ADP": {
        "source": "fantasypros_best_ball_adp",
        "metrics": {"adp": "pick_best_ball"},
    },
    # ADP_Averages already contains one player-level NFFC composite built from
    # the four contest feeds in NFFC_ADP.  Keep those raw contest rows as
    # candidate/identity evidence above, but do not admit them here: ingesting
    # both the four rows and their composite would give one provider family
    # five votes in the cross-market ADP consensus.
    "FFA_Projections": {
        "source": "ffa_projection",
        "metrics": {
            "adp": "ffa_adp",
            "expert_rank": "ffa_rank",
            "source_position_rank": "ffa_position_rank",
        },
    },
    "FantasyPoints_Projections": {
        "source": "fantasypoints",
        "metrics": {
            "adp": "fpts_adp",
            "expert_rank": "fpts_overall_rank",
        },
    },
    "FantasyData": {
        "source": "fantasydata",
        "metrics": {"expert_rank": "fdta_rank"},
    },
    "PFF_Projections": {
        "source": "pff_projection",
        "metrics": {"expert_rank": "pff_rank"},
    },
    "ETR_Ranks": {
        "source": "etr_rank",
        "metrics": {
            "adp": "etr_adp",
            "expert_rank": "etr_rank",
            "source_position_rank": "etr_pos_rank",
        },
    },
    "Evan_Silva_Ranks": {
        "source": "evan_silva_rank",
        "metrics": {
            "expert_rank": "evan_silva_rank",
            "source_position_rank": "evan_silva_pos_rank",
        },
    },
    "FFF_Ranks": {
        "source": "fff_rank",
        "metrics": {"expert_rank": "fff_total_rank"},
    },
    "Barret_Ranks": {
        "source": "barret_rank",
        "metrics": {"expert_rank": "barret_total_rank"},
    },
}
