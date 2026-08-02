"""Replay causal prior-season nflfastR receiver profiles in matching."""

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
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_b_replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fastr_profile_replay_reference",
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
PRIMARY_METHOD = "usage_air_value_disp_w100"
RECENCY_HALF_LIFE = 12.0
TARGET_SHRINKAGE = 40.0
RED_ZONE_SHRINKAGE = 8.0
WEEK_SHRINKAGE = 8.0
HIGH_USE_THRESHOLD = 0.20
EXPANDED_TARGET_COUNTS = {"QB": 48, "RB": 90, "WR": 120, "TE": 48}
DEFAULT_FASTR_DATABASE = (
    REPO_ROOT.parent
    / "Daily_Fantasy_Data"
    / "Databases"
    / "FastR_Beta.sqlite3"
)

TARGET_SHARE = "match_actual_target_share_profile"
AIR_SHARE = "match_actual_air_yards_share_profile"
ADOT = "match_actual_adot_profile"
RED_ZONE_SHARE = "match_actual_red_zone_target_share_profile"
TARGET_SHARE_IQR = "match_actual_target_share_iqr_profile"
HIGH_USE_RATE = "match_actual_high_use_week_rate_profile"
FEATURE_COLUMNS = [
    TARGET_SHARE,
    AIR_SHARE,
    ADOT,
    RED_ZONE_SHARE,
    TARGET_SHARE_IQR,
    HIGH_USE_RATE,
]
RAW_PROFILE_COLUMNS = [
    "actual_profile_source_season",
    "actual_profile_source_position",
    "actual_profile_targets",
    "actual_profile_targeted_weeks",
    "actual_target_share",
    "actual_air_yards_share",
    "actual_adot",
    "actual_red_zone_target_share",
    "actual_target_share_iqr",
    "actual_high_use_week_rate",
    "actual_target_reliability",
    "actual_red_zone_reliability",
    "actual_dispersion_reliability",
    "actual_receiver_profile_available",
    "actual_receiver_identity_method",
]
PROFILE_COLUMNS = [*RAW_PROFILE_COLUMNS, *FEATURE_COLUMNS]

VARIANTS = {
    BASELINE_METHOD: {},
    "usage_depth_w100": {
        TARGET_SHARE: 0.50,
        ADOT: 0.50,
    },
    "usage_air_value_w100": {
        TARGET_SHARE: 0.30,
        AIR_SHARE: 0.25,
        ADOT: 0.20,
        RED_ZONE_SHARE: 0.25,
    },
    PRIMARY_METHOD: {
        TARGET_SHARE: 0.225,
        AIR_SHARE: 0.20,
        ADOT: 0.175,
        RED_ZONE_SHARE: 0.20,
        TARGET_SHARE_IQR: 0.10,
        HIGH_USE_RATE: 0.10,
    },
    "usage_air_value_disp_w150": {
        TARGET_SHARE: 0.3375,
        AIR_SHARE: 0.30,
        ADOT: 0.2625,
        RED_ZONE_SHARE: 0.30,
        TARGET_SHARE_IQR: 0.15,
        HIGH_USE_RATE: 0.15,
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
        for position in ("WR", "TE"):
            weights[position].update(added)
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
                **{f"weight_{feature}": added.get(feature, 0.0)
                   for feature in FEATURE_COLUMNS},
            }
        )
    return methods, pd.DataFrame(metadata)


METHODS, METHOD_METADATA = build_methods()


def configure_reference_globals() -> None:
    reference.METHODS = METHODS
    receiver_rate.METHODS = METHODS
    receiver_rate.METHOD_METADATA = METHOD_METADATA
    receiver_rate.BASELINE_METHOD = BASELINE_METHOD
    pruning.METHODS = METHODS
    pruning.METHOD_METADATA = METHOD_METADATA
    pruning.BASELINE_METHOD = BASELINE_METHOD
    base.TARGET_COUNTS = EXPANDED_TARGET_COUNTS


def _unique_identity_maps(projections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["player_key", "player", "pos", "season", "team"]
    identities = projections[columns].dropna(subset=["player_key"]).copy()
    identities["player"] = builder.clean_player_names(
        identities[["player"]]
    )["player"]
    identities["team"] = identities["team"].map(builder.canonical_team)
    identities["season"] = pd.to_numeric(
        identities["season"], errors="raise"
    ).astype(int)
    identities["player_key"] = identities["player_key"].astype("string")

    global_counts = (
        identities.groupby(["player", "pos"])["player_key"]
        .nunique()
        .rename("key_count")
        .reset_index()
    )
    global_map = (
        identities.merge(
            global_counts[global_counts.key_count.eq(1)],
            on=["player", "pos"],
            how="inner",
        )[["player", "pos", "player_key"]]
        .drop_duplicates()
    )

    exact_counts = (
        identities.groupby(["player", "pos", "season", "team"])[
            "player_key"
        ]
        .nunique()
        .rename("key_count")
        .reset_index()
    )
    exact_map = (
        identities.merge(
            exact_counts[exact_counts.key_count.eq(1)],
            on=["player", "pos", "season", "team"],
            how="inner",
        )[["player", "pos", "season", "team", "player_key"]]
        .drop_duplicates()
    )
    return global_map, exact_map


def load_actual_receiver_profiles(
    fastr_database: Path,
    projections: pd.DataFrame,
    max_source_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not fastr_database.exists():
        raise FileNotFoundError(f"nflfastR database not found: {fastr_database}")
    player_columns = """
        player, CAST(season AS INTEGER) season, team, CAST(week AS INTEGER) week,
        rec_pass_attempt_sum targets,
        rec_air_yards_sum air_yards,
        rec_red_zone_pass_attempt_sum red_zone_targets
    """
    team_query = """
        SELECT CAST(season AS INTEGER) season, team,
               CAST(week AS INTEGER) week,
               team_rec_pass_attempt_sum team_targets,
               team_rec_air_yards_sum team_air_yards,
               team_rec_red_zone_pass_attempt_sum team_red_zone_targets
        FROM Team_Stats
        WHERE season <= ?
    """
    with sqlite3.connect(fastr_database) as connection:
        frames = []
        for position in ("WR", "TE"):
            frame = pd.read_sql_query(
                f"SELECT {player_columns} FROM {position}_Stats WHERE season <= ?",
                connection,
                params=(int(max_source_season),),
            )
            frame["pos"] = position
            frames.append(frame)
        players = pd.concat(frames, ignore_index=True)
        teams = pd.read_sql_query(
            team_query,
            connection,
            params=(int(max_source_season),),
        )

    players = builder.clean_player_names(players)
    players["team"] = players["team"].map(builder.canonical_team)
    teams["team"] = teams["team"].map(builder.canonical_team)
    numeric = [
        "targets",
        "air_yards",
        "red_zone_targets",
        "team_targets",
        "team_air_yards",
        "team_red_zone_targets",
    ]
    for column in numeric[:3]:
        players[column] = pd.to_numeric(
            players[column], errors="coerce"
        ).fillna(0.0)
    for column in numeric[3:]:
        teams[column] = pd.to_numeric(teams[column], errors="coerce")
    if teams.duplicated(["season", "team", "week"]).any():
        raise ValueError("Team_Stats is not unique by season/team/week.")

    global_map, exact_map = _unique_identity_maps(projections)
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
    players["player_key"] = players["global_player_key"].fillna(
        players["exact_player_key"]
    )
    players["actual_receiver_identity_method"] = np.select(
        [
            players["global_player_key"].notna(),
            players["exact_player_key"].notna(),
        ],
        ["globally_unique_name_position", "season_team_disambiguation"],
        default="unresolved",
    )
    identity_audit = (
        players.groupby(
            ["season", "pos", "actual_receiver_identity_method"],
            as_index=False,
        )
        .agg(rows=("player", "size"), targets=("targets", "sum"))
    )
    players = players[players["player_key"].notna()].copy()
    players = players.merge(
        teams,
        on=["season", "team", "week"],
        how="left",
        validate="many_to_one",
    )
    if players["team_targets"].isna().mean() > 0.001:
        raise ValueError("More than 0.1% of mapped receiver rows miss team data.")

    weekly = (
        players.groupby(
            [
                "player_key",
                "pos",
                "season",
                "week",
                "actual_receiver_identity_method",
            ],
            as_index=False,
        )
        .agg(
            targets=("targets", "sum"),
            air_yards=("air_yards", "sum"),
            red_zone_targets=("red_zone_targets", "sum"),
            team_targets=("team_targets", "sum"),
            team_air_yards=("team_air_yards", "sum"),
            team_red_zone_targets=("team_red_zone_targets", "sum"),
        )
    )
    weekly = weekly[
        weekly.targets.gt(0) & weekly.team_targets.gt(0)
    ].copy()
    weekly["weekly_target_share"] = (
        weekly.targets / weekly.team_targets
    ).clip(lower=0, upper=1)

    grouped = weekly.groupby(
        ["player_key", "pos", "season", "actual_receiver_identity_method"],
        as_index=False,
    )
    profiles = grouped.agg(
        actual_profile_targets=("targets", "sum"),
        actual_profile_targeted_weeks=("week", "nunique"),
        player_air_yards=("air_yards", "sum"),
        player_red_zone_targets=("red_zone_targets", "sum"),
        opportunity_team_targets=("team_targets", "sum"),
        opportunity_team_air_yards=("team_air_yards", "sum"),
        opportunity_team_red_zone_targets=("team_red_zone_targets", "sum"),
        actual_target_share_iqr=(
            "weekly_target_share",
            lambda values: float(values.quantile(0.75) - values.quantile(0.25)),
        ),
        actual_high_use_week_rate=(
            "weekly_target_share",
            lambda values: float(values.ge(HIGH_USE_THRESHOLD).mean()),
        ),
    )
    profiles["actual_target_share"] = (
        profiles.actual_profile_targets / profiles.opportunity_team_targets
    ).clip(lower=0, upper=1)
    profiles["actual_air_yards_share"] = (
        profiles.player_air_yards
        / profiles.opportunity_team_air_yards.replace(0, np.nan)
    ).clip(lower=-0.5, upper=1.5)
    profiles["actual_adot"] = (
        profiles.player_air_yards
        / profiles.actual_profile_targets.replace(0, np.nan)
    ).clip(lower=-5, upper=30)
    profiles["actual_red_zone_target_share"] = (
        profiles.player_red_zone_targets
        / profiles.opportunity_team_red_zone_targets.replace(0, np.nan)
    ).clip(lower=0, upper=1)
    profiles["actual_target_reliability"] = (
        profiles.actual_profile_targets
        / (profiles.actual_profile_targets + TARGET_SHRINKAGE)
    ).fillna(0.0)
    profiles["actual_red_zone_reliability"] = (
        profiles.player_red_zone_targets
        / (profiles.player_red_zone_targets + RED_ZONE_SHRINKAGE)
    ).fillna(0.0)
    profiles["actual_dispersion_reliability"] = (
        profiles.actual_target_reliability
        * profiles.actual_profile_targeted_weeks
        / (profiles.actual_profile_targeted_weeks + WEEK_SHRINKAGE)
    ).fillna(0.0)

    raw_to_feature = {
        "actual_target_share": (TARGET_SHARE, "actual_target_reliability"),
        "actual_air_yards_share": (AIR_SHARE, "actual_target_reliability"),
        "actual_adot": (ADOT, "actual_target_reliability"),
        "actual_red_zone_target_share": (
            RED_ZONE_SHARE,
            "actual_red_zone_reliability",
        ),
        "actual_target_share_iqr": (
            TARGET_SHARE_IQR,
            "actual_dispersion_reliability",
        ),
        "actual_high_use_week_rate": (
            HIGH_USE_RATE,
            "actual_dispersion_reliability",
        ),
    }
    for raw_column, (feature, reliability_column) in raw_to_feature.items():
        percentile = profiles.groupby(["season", "pos"])[raw_column].rank(
            method="average",
            pct=True,
        )
        profiles[feature] = 0.5 + (
            percentile - 0.5
        ) * profiles[reliability_column]
        profiles[feature] = profiles[feature].fillna(0.5).clip(0, 1)

    profiles = profiles.sort_values(
        ["player_key", "season", "actual_profile_targets"],
        ascending=[True, True, False],
    )
    position_collisions = profiles.duplicated(
        ["player_key", "season"], keep=False
    )
    if position_collisions.any():
        collision_count = int(position_collisions.sum())
        identity_audit = pd.concat(
            [
                identity_audit,
                pd.DataFrame(
                    [{
                        "season": -1,
                        "pos": "ALL",
                        "actual_receiver_identity_method": "position_collision_rows",
                        "rows": collision_count,
                        "targets": float(
                            profiles.loc[
                                position_collisions, "actual_profile_targets"
                            ].sum()
                        ),
                    }]
                ),
            ],
            ignore_index=True,
        )
        profiles = profiles.drop_duplicates(["player_key", "season"])

    profiles = profiles.rename(
        columns={
            "season": "actual_profile_source_season",
            "pos": "actual_profile_source_position",
        }
    )
    profiles["season"] = profiles.actual_profile_source_season + 1
    profiles["actual_receiver_profile_available"] = 1
    keep = ["player_key", "season", *PROFILE_COLUMNS]
    return profiles[keep], identity_audit


def attach_actual_receiver_profiles(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    overlap = sorted(set(PROFILE_COLUMNS).intersection(frame.columns))
    if overlap:
        raise ValueError("Actual receiver columns already exist: " + ", ".join(overlap))
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
    output["actual_receiver_profile_available"] = pd.to_numeric(
        output["actual_receiver_profile_available"], errors="coerce"
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
            available = group.actual_receiver_profile_available.eq(1)
            rows.append(
                {
                    "population": population,
                    "pos": position,
                    "rows": int(len(group)),
                    "available": int(available.sum()),
                    "coverage": float(available.mean()),
                    "mean_targets_when_available": float(
                        group.loc[available, "actual_profile_targets"].mean()
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
    profile = targets[columns].rename(columns={"player_key": "target_player_key"})
    output = predictions.merge(
        profile,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )
    return receiver_rate.refresh_row_event_losses(output)


def current_ladd_audit(
    league: str,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulation = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
    with sqlite3.connect(simulation) as connection:
        templates = pd.read_sql_query(
            "SELECT * FROM Best_Ball_Weekly_Templates WHERE league = ?",
            connection,
            params=(league,),
        )
        target = pd.read_sql_query(
            """
            SELECT * FROM Best_Ball_Weekly_Player_Map
            WHERE version = ? AND player = 'Ladd McConkey'
            """,
            connection,
            params=(league,),
        )
    if templates.empty or target.empty:
        return pd.DataFrame(), pd.DataFrame()
    templates = attach_actual_receiver_profiles(templates, profiles)
    target["season"] = target["year"]
    target = attach_actual_receiver_profiles(target, profiles)
    player_row = next(target.itertuples(index=False))
    summaries = []
    top_rows = []
    original_weights = builder.MATCH_FEATURE_WEIGHTS
    try:
        for method, specification in METHODS.items():
            builder.MATCH_FEATURE_WEIGHTS = specification["weights"]
            members, _ = builder.select_player_template_pool(player_row, templates)
            pool = members.merge(
                templates[
                    [
                        "league",
                        "template_id",
                        "player",
                        "season",
                        "historical_pred_fp_per_game",
                        "active_ppg_resid",
                        "played_games",
                        *FEATURE_COLUMNS,
                    ]
                ],
                left_on=["template_league", "template_id"],
                right_on=["league", "template_id"],
                how="left",
                validate="one_to_one",
                suffixes=("", "_template"),
            )
            probabilities = pool.template_sample_prob.to_numpy(dtype=float)
            residuals = pool.active_ppg_resid.to_numpy(dtype=float)
            residual_mean = float(np.average(residuals, weights=probabilities))
            pryor = pool[pool.player.eq("Terrelle Pryor")]
            pryor_row = pryor.iloc[0] if not pryor.empty else None
            summaries.append(
                {
                    "league": league,
                    "method": method,
                    "weighted_abs_pred_ppg_gap": float(
                        np.average(
                            np.abs(
                                pool.historical_pred_fp_per_game.to_numpy(dtype=float)
                                - float(player_row.pred_fp_per_game)
                            ),
                            weights=probabilities,
                        )
                    ),
                    "expected_played": float(
                        np.average(pool.played_games, weights=probabilities)
                    ),
                    "pool_residual_sd": float(
                        np.sqrt(
                            np.average(
                                np.square(residuals - residual_mean),
                                weights=probabilities,
                            )
                        )
                    ),
                    "top12_weight": float(
                        pool.nsmallest(12, "match_rank").template_sample_prob.sum()
                    ),
                    "pryor_rank": (
                        int(pryor_row.match_rank) if pryor_row is not None else np.nan
                    ),
                    "pryor_distance": (
                        float(pryor_row.template_distance)
                        if pryor_row is not None else np.nan
                    ),
                    "pryor_weight": (
                        float(pryor_row.template_sample_prob)
                        if pryor_row is not None else 0.0
                    ),
                }
            )
            top = pool.nsmallest(12, "match_rank")[
                [
                    "match_rank",
                    "player",
                    "season_template",
                    "template_distance",
                    "template_sample_prob",
                    "historical_pred_fp_per_game",
                    "played_games",
                    *FEATURE_COLUMNS,
                ]
            ].copy()
            top.insert(0, "method", method)
            top.insert(0, "league", league)
            top_rows.append(top)
    finally:
        builder.MATCH_FEATURE_WEIGHTS = original_weights
    return pd.DataFrame(summaries), pd.concat(top_rows, ignore_index=True)


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
    profiles, identity_audit = load_actual_receiver_profiles(
        args.fastr_db.resolve(),
        projections,
        max_season,
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    templates = receiver_rate.reattach_template_player_keys(templates, projections)
    templates = attach_actual_receiver_profiles(templates, profiles)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(templates, forecasts)
    targets = base.build_targets(target_templates)
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = targets.groupby(["season", "pos"]).cumcount() + 1

    predictions = pruning.run_replay(templates, targets)
    expected_rows = len(targets) * len(METHODS)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} predictions; found {len(predictions)}."
        )
    predictions = attach_prediction_metadata(predictions, targets)
    coverage = coverage_audit(templates, targets)
    ladd_summary, ladd_top12 = current_ladd_audit(league, profiles)
    runtime_seconds = time.perf_counter() - started

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    profiles.to_csv(results_dir / "actual_receiver_profiles.csv", index=False)
    identity_audit.to_csv(results_dir / "identity_audit.csv", index=False)
    coverage.to_csv(results_dir / "feature_coverage.csv", index=False)
    ladd_summary.to_csv(results_dir / "current_ladd_pool_audit.csv", index=False)
    ladd_top12.to_csv(results_dir / "current_ladd_top12.csv", index=False)
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
