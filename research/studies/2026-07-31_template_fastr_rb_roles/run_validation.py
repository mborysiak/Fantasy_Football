"""Replay causal prior-season nflfastR RB role profiles in matching."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
REFERENCE_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_fastr_receiver_profiles"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fastr_rb_role_reference",
    REFERENCE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import replay reference from {REFERENCE_PATH}")
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

receiver_rate = reference.receiver_rate
pruning = reference.pruning
base = reference.base
builder = reference.builder

BASELINE_METHOD = "production"
PRIMARY_METHOD = "rb_dual_role_w050"
RECENCY_HALF_LIFE = 12.0
OPPORTUNITY_SHRINKAGE = 40.0
RED_ZONE_ROOM_SHRINKAGE = 10.0
GOAL_LINE_ROOM_SHRINKAGE = 5.0
THIRD_FOURTH_TARGET_ROOM_SHRINKAGE = 8.0
EXPANDED_TARGET_COUNTS = {"QB": 48, "RB": 90, "WR": 120, "TE": 48}
DEFAULT_FASTR_DATABASE = reference.DEFAULT_FASTR_DATABASE

RED_ZONE_CARRY_SHARE = "match_actual_rb_red_zone_carry_room_share_profile"
GOAL_LINE_CARRY_SHARE = "match_actual_rb_goal_line_carry_room_share_profile"
THIRD_FOURTH_TARGET_SHARE = (
    "match_actual_rb_third_fourth_target_room_share_profile"
)
FEATURE_COLUMNS = [
    RED_ZONE_CARRY_SHARE,
    GOAL_LINE_CARRY_SHARE,
    THIRD_FOURTH_TARGET_SHARE,
]
RAW_PROFILE_COLUMNS = [
    "actual_rb_profile_source_season",
    "actual_rb_profile_opportunities",
    "actual_rb_profile_opportunity_weeks",
    "actual_rb_red_zone_carries",
    "actual_rb_goal_line_carries",
    "actual_rb_third_fourth_targets",
    "actual_rb_red_zone_carry_room_share",
    "actual_rb_goal_line_carry_room_share",
    "actual_rb_third_fourth_target_room_share",
    "actual_rb_opportunity_reliability",
    "actual_rb_red_zone_reliability",
    "actual_rb_goal_line_reliability",
    "actual_rb_third_fourth_target_reliability",
    "actual_rb_role_profile_available",
    "actual_rb_identity_method",
]
PROFILE_COLUMNS = [*RAW_PROFILE_COLUMNS, *FEATURE_COLUMNS]

VARIANTS = {
    BASELINE_METHOD: {},
    "rb_scoring_role_w050": {
        RED_ZONE_CARRY_SHARE: 0.25,
        GOAL_LINE_CARRY_SHARE: 0.25,
    },
    "rb_passing_down_w050": {
        THIRD_FOURTH_TARGET_SHARE: 0.50,
    },
    PRIMARY_METHOD: {
        GOAL_LINE_CARRY_SHARE: 0.25,
        THIRD_FOURTH_TARGET_SHARE: 0.25,
    },
    "rb_dual_role_w100": {
        GOAL_LINE_CARRY_SHARE: 0.50,
        THIRD_FOURTH_TARGET_SHARE: 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--fastr-db", type=Path, default=DEFAULT_FASTR_DATABASE)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    methods = {}
    metadata = []
    for method, added in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        weights["RB"].update(added)
        methods[method] = {
            "weights": weights,
            "recency_half_life": RECENCY_HALF_LIFE,
            "variant": method,
            "removed_families": (),
        }
        metadata.append(
            {
                "method": method,
                "primary": int(method == PRIMARY_METHOD),
                "added_total_weight": float(sum(added.values())),
                "added_features": ",".join(added),
                **{
                    f"weight_{feature}": added.get(feature, 0.0)
                    for feature in FEATURE_COLUMNS
                },
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def configure_reference_globals() -> None:
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    base.TARGET_COUNTS = EXPANDED_TARGET_COUNTS


def load_actual_rb_role_profiles(
    fastr_database: Path,
    projections: pd.DataFrame,
    max_source_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not fastr_database.exists():
        raise FileNotFoundError(f"nflfastR database not found: {fastr_database}")
    query = """
        SELECT player,
               CAST(season AS INTEGER) season,
               team,
               CAST(week AS INTEGER) week,
               rush_rush_attempt_sum rush_attempts,
               rec_pass_attempt_sum targets,
               rush_red_zone_rush_attempt_sum red_zone_carries,
               rush_goalline_rush_attempt_sum goal_line_carries,
               rec_third_fourth_pass_attempt_sum third_fourth_targets
        FROM RB_Stats
        WHERE season <= ?
    """
    with sqlite3.connect(fastr_database) as connection:
        players = pd.read_sql_query(
            query,
            connection,
            params=(int(max_source_season),),
        )

    players["pos"] = "RB"
    players = builder.clean_player_names(players)
    players["team"] = players["team"].map(builder.canonical_team)
    count_columns = [
        "rush_attempts",
        "targets",
        "red_zone_carries",
        "goal_line_carries",
        "third_fourth_targets",
    ]
    for column in count_columns:
        players[column] = pd.to_numeric(
            players[column], errors="coerce"
        ).fillna(0.0)

    room = (
        players.groupby(["season", "team", "week"], as_index=False)
        .agg(
            room_red_zone_carries=("red_zone_carries", "sum"),
            room_goal_line_carries=("goal_line_carries", "sum"),
            room_third_fourth_targets=("third_fourth_targets", "sum"),
        )
    )
    global_map, exact_map = reference._unique_identity_maps(projections)
    players = players.merge(
        global_map.rename(columns={"player_key": "global_player_key"}),
        on=["player", "pos"],
        how="left",
        validate="many_to_one",
    ).merge(
        exact_map.rename(columns={"player_key": "exact_player_key"}),
        on=["player", "pos", "season", "team"],
        how="left",
        validate="many_to_one",
    )
    players["player_key"] = players.global_player_key.fillna(
        players.exact_player_key
    )
    players["actual_rb_identity_method"] = np.select(
        [players.global_player_key.notna(), players.exact_player_key.notna()],
        ["globally_unique_name_position", "season_team_disambiguation"],
        default="unresolved",
    )
    identity_audit = (
        players.groupby(
            ["season", "actual_rb_identity_method"],
            as_index=False,
        )
        .agg(
            rows=("player", "size"),
            rush_attempts=("rush_attempts", "sum"),
            targets=("targets", "sum"),
        )
    )
    identity_audit["opportunities"] = (
        identity_audit.rush_attempts + identity_audit.targets
    )
    players = players[players.player_key.notna()].copy()
    players = players.merge(
        room,
        on=["season", "team", "week"],
        how="left",
        validate="many_to_one",
    )
    players["opportunities"] = players.rush_attempts + players.targets
    players = players[players.opportunities.gt(0)].copy()

    weekly = (
        players.groupby(
            ["player_key", "season", "week", "actual_rb_identity_method"],
            as_index=False,
        )
        .agg(
            opportunities=("opportunities", "sum"),
            red_zone_carries=("red_zone_carries", "sum"),
            goal_line_carries=("goal_line_carries", "sum"),
            third_fourth_targets=("third_fourth_targets", "sum"),
            room_red_zone_carries=("room_red_zone_carries", "sum"),
            room_goal_line_carries=("room_goal_line_carries", "sum"),
            room_third_fourth_targets=("room_third_fourth_targets", "sum"),
        )
    )
    profiles = (
        weekly.groupby(
            ["player_key", "season", "actual_rb_identity_method"],
            as_index=False,
        )
        .agg(
            actual_rb_profile_opportunities=("opportunities", "sum"),
            actual_rb_profile_opportunity_weeks=("week", "nunique"),
            actual_rb_red_zone_carries=("red_zone_carries", "sum"),
            actual_rb_goal_line_carries=("goal_line_carries", "sum"),
            actual_rb_third_fourth_targets=("third_fourth_targets", "sum"),
            opportunity_room_red_zone_carries=(
                "room_red_zone_carries",
                "sum",
            ),
            opportunity_room_goal_line_carries=(
                "room_goal_line_carries",
                "sum",
            ),
            opportunity_room_third_fourth_targets=(
                "room_third_fourth_targets",
                "sum",
            ),
        )
    )
    profiles["actual_rb_red_zone_carry_room_share"] = (
        profiles.actual_rb_red_zone_carries
        / profiles.opportunity_room_red_zone_carries.replace(0, np.nan)
    ).clip(0, 1)
    profiles["actual_rb_goal_line_carry_room_share"] = (
        profiles.actual_rb_goal_line_carries
        / profiles.opportunity_room_goal_line_carries.replace(0, np.nan)
    ).clip(0, 1)
    profiles["actual_rb_third_fourth_target_room_share"] = (
        profiles.actual_rb_third_fourth_targets
        / profiles.opportunity_room_third_fourth_targets.replace(0, np.nan)
    ).clip(0, 1)
    profiles["actual_rb_opportunity_reliability"] = (
        profiles.actual_rb_profile_opportunities
        / (profiles.actual_rb_profile_opportunities + OPPORTUNITY_SHRINKAGE)
    ).fillna(0.0)
    profiles["actual_rb_red_zone_reliability"] = (
        profiles.actual_rb_opportunity_reliability
        * profiles.opportunity_room_red_zone_carries
        / (
            profiles.opportunity_room_red_zone_carries
            + RED_ZONE_ROOM_SHRINKAGE
        )
    ).fillna(0.0)
    profiles["actual_rb_goal_line_reliability"] = (
        profiles.actual_rb_opportunity_reliability
        * profiles.opportunity_room_goal_line_carries
        / (
            profiles.opportunity_room_goal_line_carries
            + GOAL_LINE_ROOM_SHRINKAGE
        )
    ).fillna(0.0)
    profiles["actual_rb_third_fourth_target_reliability"] = (
        profiles.actual_rb_opportunity_reliability
        * profiles.opportunity_room_third_fourth_targets
        / (
            profiles.opportunity_room_third_fourth_targets
            + THIRD_FOURTH_TARGET_ROOM_SHRINKAGE
        )
    ).fillna(0.0)

    raw_to_feature = {
        "actual_rb_red_zone_carry_room_share": (
            RED_ZONE_CARRY_SHARE,
            "actual_rb_red_zone_reliability",
        ),
        "actual_rb_goal_line_carry_room_share": (
            GOAL_LINE_CARRY_SHARE,
            "actual_rb_goal_line_reliability",
        ),
        "actual_rb_third_fourth_target_room_share": (
            THIRD_FOURTH_TARGET_SHARE,
            "actual_rb_third_fourth_target_reliability",
        ),
    }
    for raw_column, (feature, reliability_column) in raw_to_feature.items():
        percentile = profiles.groupby("season")[raw_column].rank(
            method="average",
            pct=True,
        )
        profiles[feature] = 0.5 + (
            percentile - 0.5
        ) * profiles[reliability_column]
        profiles[feature] = profiles[feature].fillna(0.5).clip(0, 1)

    profiles = profiles.sort_values(
        ["player_key", "season", "actual_rb_profile_opportunities"],
        ascending=[True, True, False],
    )
    duplicate = profiles.duplicated(["player_key", "season"], keep=False)
    if duplicate.any():
        raise ValueError(
            "RB role profiles are not unique by player_key/source season."
        )
    profiles = profiles.rename(
        columns={"season": "actual_rb_profile_source_season"}
    )
    profiles["season"] = profiles.actual_rb_profile_source_season + 1
    profiles["actual_rb_role_profile_available"] = 1
    return profiles[["player_key", "season", *PROFILE_COLUMNS]], identity_audit


def attach_actual_rb_role_profiles(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    overlap = sorted(set(PROFILE_COLUMNS).intersection(frame.columns))
    if overlap:
        raise ValueError("RB role columns already exist: " + ", ".join(overlap))
    output = frame.merge(
        profiles,
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    for feature in FEATURE_COLUMNS:
        output[feature] = pd.to_numeric(
            output[feature], errors="coerce"
        ).fillna(builder.MATCH_FILL_VALUE).clip(0, 1)
    output["actual_rb_role_profile_available"] = pd.to_numeric(
        output.actual_rb_role_profile_available,
        errors="coerce",
    ).fillna(0).astype(int)
    return output


def coverage_audit(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for population, frame in (
        ("historical_templates", templates),
        ("rolling_targets", targets),
    ):
        for position, group in frame.groupby("pos", sort=True):
            available = group.actual_rb_role_profile_available.eq(1)
            rows.append(
                {
                    "population": population,
                    "pos": position,
                    "rows": int(len(group)),
                    "available": int(available.sum()),
                    "coverage": float(available.mean()),
                    "red_zone_share_available": int(
                        group.actual_rb_red_zone_carry_room_share.notna().sum()
                    ),
                    "goal_line_share_available": int(
                        group.actual_rb_goal_line_carry_room_share.notna().sum()
                    ),
                    "third_fourth_target_share_available": int(
                        group.actual_rb_third_fourth_target_room_share.notna().sum()
                    ),
                    "mean_opportunities_when_available": float(
                        group.loc[
                            available, "actual_rb_profile_opportunities"
                        ].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def attach_prediction_metadata(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "player_key",
        "player",
        "pos",
        "season",
        "preseason_pos_rank",
        "qb_team_rank",
        "qb_team_rank_bucket",
        *PROFILE_COLUMNS,
    ]
    if "team" in targets:
        columns.append("team")
    profile = targets[columns].rename(
        columns={"player_key": "target_player_key"}
    )
    output = predictions.merge(
        profile,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )
    return receiver_rate.refresh_row_event_losses(output)


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else STUDY_DIR / f"results_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db is not None
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
    )
    profiles, identity_audit = load_actual_rb_role_profiles(
        args.fastr_db.resolve(),
        projections,
        max_season,
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    templates = receiver_rate.reattach_template_player_keys(templates, projections)
    templates = attach_actual_rb_role_profiles(templates, profiles)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(templates, forecasts)
    targets = base.build_targets(target_templates)
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = (
        targets.groupby(["season", "pos"]).cumcount() + 1
    )

    predictions = pruning.run_replay(templates, targets)
    expected_rows = len(targets) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} predictions; found {len(predictions)}."
        )
    predictions = attach_prediction_metadata(predictions, targets)
    coverage = coverage_audit(templates, targets)
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    profiles.to_csv(results_dir / "actual_rb_role_profiles.csv", index=False)
    identity_audit.to_csv(results_dir / "identity_audit.csv", index=False)
    coverage.to_csv(results_dir / "feature_coverage.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "fastr_database": str(args.fastr_db.resolve()),
        "max_template_season": int(max_season),
        "expanded_target_counts": EXPANDED_TARGET_COUNTS,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "primary_method": PRIMARY_METHOD,
        "runtime_seconds": runtime_seconds,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
