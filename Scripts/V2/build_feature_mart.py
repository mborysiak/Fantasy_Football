"""Build V2 consensus, history, lifecycle, team, and room features."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from Scripts.V2.adp_policy import canonical_adp_family_values
from Scripts.V2.config import (
    POSITIONS,
    PROJECTION_THROUGH_SEASON,
    PROJECTION_VALUE_SPECS,
    TEAM_MAP,
)
from Scripts.V2.contracts import (
    FEATURE_AUDIT_COLUMNS,
    FEATURE_CATALOG_COLUMNS,
    FEATURE_CORRELATION_COLUMNS,
    FEATURE_MANIFEST_COLUMNS,
    align_columns,
    require_columns,
)


PROJECTION_MEDIAN_FEATURES = {
    "projected_games": "proj_games",
    "pass_attempts": "proj_pass_attempts",
    "passing_yards": "proj_passing_yards",
    "passing_tds": "proj_passing_tds",
    "interceptions": "proj_interceptions",
    "rush_attempts": "proj_rush_attempts",
    "rushing_yards": "proj_rushing_yards",
    "rushing_tds": "proj_rushing_tds",
    "targets": "proj_targets",
    "receptions": "proj_receptions",
    "receiving_yards": "proj_receiving_yards",
    "receiving_tds": "proj_receiving_tds",
    "passing_points": "proj_passing_points",
    "rushing_points": "proj_rushing_points",
    "receiving_points": "proj_receiving_points",
}

PROJECTION_PROVIDERS = tuple(
    sorted({str(spec["provider"]) for spec in PROJECTION_VALUE_SPECS.values()})
)
# Provider-specific estimates remain masked until the provider has at least
# three prior projection seasons. All configured providers can still
# contribute equally to the consensus immediately; this gate only prevents a
# learned provider coefficient from being estimated from one recent season.
MIN_PROVIDER_HISTORY_SEASONS_FOR_SPECIFIC_FEATURE = 3
PROJECTION_PROVIDER_CHALLENGER_FEATURES = {
    f"provider_{provider}_ppg_team_game"
    for provider in PROJECTION_PROVIDERS
}
PROJECTION_SHAPE_CHALLENGER_FEATURES = {
    "proj_total_touches",
    "proj_total_opportunities",
    "proj_pass_yards_per_attempt",
    "proj_pass_td_rate",
    "proj_interception_rate",
    "proj_rush_yards_per_attempt",
    "proj_rush_td_rate",
    "proj_receiving_yards_per_reception",
    "proj_receiving_td_rate",
    "proj_catch_rate",
}
PROJECTION_DISAGREEMENT_FEATURES = {
    "proj_pass_attempts_std",
    "proj_rush_attempts_std",
    "proj_targets_std",
    "proj_receptions_std",
    "proj_passing_tds_std",
    "proj_rushing_tds_std",
    "proj_receiving_tds_std",
    "proj_receiving_yards_std",
}
PROJECTION_RESEARCH_CHALLENGER_FEATURES = (
    PROJECTION_PROVIDER_CHALLENGER_FEATURES
    | PROJECTION_SHAPE_CHALLENGER_FEATURES
    | PROJECTION_DISAGREEMENT_FEATURES
)

HISTORY_GAP_RELIABILITY_GAMES = 8.0
HISTORY_GAP_CHALLENGER_FEATURES = {
    "history_career_ppg_gap",
    "history_prior_year_ppg_gap",
    "history_prior_3year_ppg_gap",
    "history_career_ppg_gap_shrunk",
    "history_prior_year_ppg_gap_shrunk",
    "history_prior_3year_ppg_gap_shrunk",
    "history_career_opportunity_games_log",
    "history_prior_year_opportunity_games_log",
    "history_prior_3year_opportunity_games_log",
    "history_prior_year_ppg_available",
    "history_prior_3year_ppg_available",
    "history_prior_year_residual_neutral",
    "history_seasons_since_observed_neutral",
}

PROJECTION_TRAJECTORY_CHALLENGER_FEATURES = {
    "projection_trajectory_change_1year",
    "projection_trajectory_change_3year",
    "projection_trajectory_prior_year_available",
    "projection_trajectory_prior_3year_count",
    "projection_trajectory_prior_3year_std",
}

ADP_TRANSFORM_CHALLENGER_FEATURES = {"adp_log"}

TEAM_ENVIRONMENT_CHALLENGER_FEATURES = {
    "team_qb1_passing_yards",
    "team_qb1_passing_tds",
    "team_qb1_rushing_yards",
    "team_qb1_rushing_tds",
    "team_qb1_rush_point_share",
    "team_core_skill_points",
    "team_core_skill_projection_percentile",
    "team_supporting_cast_points",
    "team_projected_rushing_yards",
    "team_projected_rushing_tds",
    "team_projected_offensive_tds",
}

TEAM_CORE_POSITION_LIMITS = {
    "RB": 2,
    "WR": 3,
    "TE": 1,
}

RESIDUAL_CANDIDATE_FEATURES = {
    "expert_ppg_team_game_median",
    "expert_ppg_active_median",
    "expert_ppg_team_game_std",
    "expert_points_iqr",
    "projection_provider_count",
    "proj_games",
    "proj_pass_attempts",
    "proj_rush_attempts",
    "proj_targets",
    "projected_pass_point_share",
    "projected_rush_point_share",
    "projected_receiving_point_share",
    "adp_median",
    "projection_adp_percentile_diff",
    "age",
    "year_exp",
    "is_rookie",
    "draft_pick_log",
    "career_observed_seasons",
    "career_weighted_ppg",
    "prior_year_ppg",
    "prior_year_ppg_residual",
    "prior_3year_weighted_ppg",
    "prior_3year_ppg_std",
    "seasons_since_observed",
    "room_share_median",
    "room_gap_to_leader_median",
    "room_hhi_median",
    "consensus_room_share",
    "team_qb1_ppg",
    "team_changed_from_prior_candidate",
}

LEGACY_RESIDUAL_CHALLENGER_FEATURES = {
    "expert_ppg_exp_peer_mean",
    "expert_ppg_exp_diff",
    "expert_ppg_exp_percentile",
    "adp_best_teammate_gap",
    "adp_worst_teammate_gap",
    "adp_mean_teammate_gap",
    "adp_teammates_better_count",
    "adp_room_strength_share",
    "team_target_share",
    "team_reception_share",
    "team_rush_attempt_share",
    "team_receiving_yard_share",
}

PARTICIPATION_CANDIDATE_FEATURES = {
    "expert_ppg_team_game_median",
    "projection_provider_count",
    "proj_games",
    "adp_median",
    "adp_source_count",
    "age",
    "year_exp",
    "is_rookie",
    "experience_known",
    "draft_pick_log",
    "career_observed_seasons",
    "career_useful_seasons",
    "career_opportunity_games",
    "prior_year_candidate",
    "prior_year_appeared",
    "prior_year_opportunity_games",
    "seasons_since_observed",
    "last_observed_opportunity_games",
    "team_changed_from_prior_candidate",
}

TEMPLATE_CHALLENGER_FEATURES = {
    "expert_ppg_team_game_median",
    "proj_games",
    "year_exp",
    "projection_adp_percentile_diff",
    "expert_ppg_team_game_std",
    "projected_rush_point_share",
    "projected_receiving_point_share",
    "room_share_median",
    "room_gap_to_leader_median",
    "room_hhi_median",
    "team_qb1_ppg",
    "consensus_room_share",
}

TEMPLATE_FAMILY_BUDGETS = {
    "projection_level": 0.28,
    "availability": 0.08,
    "lifecycle": 0.10,
    "market": 0.14,
    "projection_uncertainty": 0.08,
    "role_composition": 0.10,
    "room": 0.17,
    "team": 0.05,
}

# The long source tables retain all normalized inputs. The wide mart is capped
# to a reviewed set so adding a derived column cannot silently expand a model's
# search space. Production manifests below remain materially smaller.
FEATURE_MART_AUDIT_FEATURES = {
    "candidate_source_count",
    "projection_source_count",
    "market_source_count",
    "ranking_source_count",
    "draft_source_count",
    "position_conflict",
    "team_conflict",
    "identity_is_confirmed",
    "expert_points_median",
    "proj_passing_yards",
    "proj_passing_tds",
    "proj_interceptions",
    "proj_rushing_yards",
    "proj_rushing_tds",
    "proj_receptions",
    "proj_receiving_yards",
    "proj_receiving_tds",
    "configured_projection_provider_count",
    "imputed_projection_provider_count",
    "expert_points_count",
    "expert_ppg_team_game_iqr",
    "source_uncertainty_median",
    "source_ceiling_floor_gap",
    "adp_std",
    "adp_iqr",
    "adp_position_percentile",
    "projection_position_percentile",
    "expert_rank_median",
    "age_known",
    "draft_capital_known",
    "draft_round",
    "draft_pick",
    "has_prior_outcome",
    "last_observed_season",
    "last_observed_ppg",
    "last_observed_points",
    "prior_year_outcome_observed",
    "prior_year_points",
    "prior_year_useful",
    "prior_year_pass_point_share",
    "prior_year_rush_point_share",
    "prior_year_receiving_point_share",
    "consensus_room_gap_to_leader",
    "consensus_room_gap_to_next",
    "consensus_room_hhi",
    "consensus_room_player_count",
    "room_points_median",
    "room_share_std",
    "consensus_team_skill_points",
    "team_points_median",
    "team_qb1_passing_yards",
    "team_qb_projection_gap",
    "pass_catcher_room_points",
    "pass_catcher_room_share",
}

FEATURE_MART_FEATURES = (
    RESIDUAL_CANDIDATE_FEATURES
    | LEGACY_RESIDUAL_CHALLENGER_FEATURES
    | PROJECTION_RESEARCH_CHALLENGER_FEATURES
    | HISTORY_GAP_CHALLENGER_FEATURES
    | PROJECTION_TRAJECTORY_CHALLENGER_FEATURES
    | ADP_TRANSFORM_CHALLENGER_FEATURES
    | TEAM_ENVIRONMENT_CHALLENGER_FEATURES
    | PARTICIPATION_CANDIDATE_FEATURES
    | TEMPLATE_CHALLENGER_FEATURES
    | FEATURE_MART_AUDIT_FEATURES
)


def _iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def _group_summary(
    frame: pd.DataFrame,
    value: str,
    prefix: str,
    stats: tuple[str, ...] = ("median",),
) -> pd.DataFrame:
    keys = ["player_key", "season"]
    grouped = frame.groupby(keys)[value]
    pieces: list[pd.Series] = []
    for stat in stats:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if stat == "median":
                series = grouped.median()
            elif stat == "mean":
                series = grouped.mean()
            elif stat == "std":
                series = grouped.std(ddof=0)
            elif stat == "iqr":
                series = grouped.apply(_iqr)
            elif stat == "count":
                series = grouped.count()
            else:
                raise ValueError(f"Unsupported summary statistic: {stat}")
        pieces.append(series.rename(f"{prefix}_{stat}"))
    return pd.concat(pieces, axis=1).reset_index()


def build_projection_consensus(
    projection_values: pd.DataFrame,
) -> pd.DataFrame:
    if projection_values.empty:
        return pd.DataFrame(columns=["player_key", "season"])
    keys = ["player_key", "season"]
    base = (
        projection_values.groupby(keys)["provider"]
        .nunique()
        .rename("projection_provider_count")
        .reset_index()
    )
    configured = (
        projection_values[
            projection_values["configured_points_complete"].eq(1)
        ]
        .groupby(keys)["provider"]
        .nunique()
        .rename("configured_projection_provider_count")
        .reset_index()
    )
    base = base.merge(configured, on=keys, how="left")
    base["configured_projection_provider_count"] = (
        base["configured_projection_provider_count"].fillna(0).astype(int)
    )
    imputed = projection_values[
        projection_values["configured_points_complete"].eq(1)
        & projection_values[
            "configured_points_imputed_component_count"
        ].gt(0)
    ]
    imputed = (
        imputed.groupby(keys)["provider"]
        .nunique()
        .rename("imputed_projection_provider_count")
        .reset_index()
    )
    base = base.merge(imputed, on=keys, how="left")
    base["imputed_projection_provider_count"] = (
        base["imputed_projection_provider_count"].fillna(0).astype(int)
    )
    consensus_values = projection_values.copy()
    consensus_point_row = consensus_values[
        "configured_points_complete"
    ].eq(1)
    for column in (
        "provider_projected_points",
        "provider_points_per_team_game",
        "provider_points_per_projected_game",
    ):
        consensus_values[column] = consensus_values[column].where(
            consensus_point_row
        )

    summaries = [
        (
            "provider_projected_points",
            "expert_points",
            ("median", "mean", "std", "iqr", "count"),
        ),
        (
            "provider_points_per_team_game",
            "expert_ppg_team_game",
            ("median", "std", "iqr", "count"),
        ),
        (
            "provider_points_per_projected_game",
            "expert_ppg_active",
            ("median", "std", "count"),
        ),
        (
            "source_uncertainty",
            "source_uncertainty",
            ("median", "count"),
        ),
        (
            "provider_room_share",
            "room_share",
            ("median", "std", "count"),
        ),
        (
            "provider_room_rank",
            "room_rank",
            ("median", "std"),
        ),
        (
            "provider_room_gap_to_leader",
            "room_gap_to_leader",
            ("median", "std"),
        ),
        (
            "provider_room_hhi",
            "room_hhi",
            ("median", "std"),
        ),
        (
            "provider_room_points",
            "room_points",
            ("median", "std"),
        ),
        (
            "provider_team_points",
            "team_points",
            ("median", "std"),
        ),
    ]
    for value, prefix, stats in summaries:
        base = base.merge(
            _group_summary(consensus_values, value, prefix, stats),
            on=keys,
            how="left",
        )

    for source_metric, feature_name in PROJECTION_MEDIAN_FEATURES.items():
        summary = _group_summary(
            projection_values,
            source_metric,
            feature_name,
            ("median",),
        ).rename(columns={f"{feature_name}_median": feature_name})
        base = base.merge(summary, on=keys, how="left")

    disagreement_features = {
        "pass_attempts": "proj_pass_attempts_std",
        "rush_attempts": "proj_rush_attempts_std",
        "targets": "proj_targets_std",
        "receptions": "proj_receptions_std",
        "passing_tds": "proj_passing_tds_std",
        "rushing_tds": "proj_rushing_tds_std",
        "receiving_tds": "proj_receiving_tds_std",
        "receiving_yards": "proj_receiving_yards_std",
    }
    for source_metric, feature_name in disagreement_features.items():
        summary = _group_summary(
            projection_values,
            source_metric,
            feature_name.removesuffix("_std"),
            ("std",),
        ).rename(
            columns={
                f"{feature_name.removesuffix('_std')}_std": feature_name
            }
        )
        base = base.merge(summary, on=keys, how="left")

    configured_provider_values = projection_values[
        projection_values["configured_points_complete"].eq(1)
    ].copy()
    provider_seasons = (
        configured_provider_values[["provider", "season"]]
        .drop_duplicates()
        .sort_values(["provider", "season"])
    )
    provider_seasons["_provider_prior_seasons"] = provider_seasons.groupby(
        "provider"
    ).cumcount()
    configured_provider_values = configured_provider_values.merge(
        provider_seasons,
        on=["provider", "season"],
        how="left",
        validate="many_to_one",
    )
    configured_provider_values = configured_provider_values[
        configured_provider_values["_provider_prior_seasons"].ge(
            MIN_PROVIDER_HISTORY_SEASONS_FOR_SPECIFIC_FEATURE
        )
    ]
    provider_ppg = (
        configured_provider_values
        .pivot_table(
            index=keys,
            columns="provider",
            values="provider_points_per_team_game",
            aggfunc="median",
        )
        .rename(
            columns={
                provider: f"provider_{provider}_ppg_team_game"
                for provider in PROJECTION_PROVIDERS
            }
        )
        .reset_index()
    )
    for feature in PROJECTION_PROVIDER_CHALLENGER_FEATURES:
        if feature not in provider_ppg:
            provider_ppg[feature] = np.nan
    base = base.merge(
        provider_ppg[
            [*keys, *sorted(PROJECTION_PROVIDER_CHALLENGER_FEATURES)]
        ],
        on=keys,
        how="left",
    )

    component_total = base[
        ["proj_passing_points", "proj_rushing_points", "proj_receiving_points"]
    ].fillna(0).sum(axis=1)
    component_denominator = component_total.where(
        ~np.isclose(component_total, 0.0, rtol=0, atol=1e-12)
    )
    for component, output in (
        ("proj_passing_points", "projected_pass_point_share"),
        ("proj_rushing_points", "projected_rush_point_share"),
        ("proj_receiving_points", "projected_receiving_point_share"),
    ):
        # Beta/NV sack deductions can make a fringe QB's projected passing
        # component and total negative. Preserve the signed composition as
        # long as the component sum is nonzero; requiring a positive total
        # silently erased exactly the scoring context the auction matchers need.
        base[output] = base[component] / component_denominator
    ceiling = _group_summary(
        projection_values,
        "source_ceiling_points",
        "source_ceiling_points",
        ("median",),
    )
    floor = _group_summary(
        projection_values,
        "source_floor_points",
        "source_floor_points",
        ("median",),
    )
    base = base.merge(ceiling, on=keys, how="left").merge(
        floor, on=keys, how="left"
    )
    base["source_ceiling_floor_gap"] = (
        base["source_ceiling_points_median"]
        - base["source_floor_points_median"]
    )
    return base


def add_projection_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(output[column], errors="coerce")

    def rate(numerator: str, denominator: str) -> pd.Series:
        denominator_values = numeric(denominator)
        return numeric(numerator) / denominator_values.where(
            denominator_values.gt(0)
        )

    output["proj_total_touches"] = pd.concat(
        [numeric("proj_rush_attempts"), numeric("proj_receptions")],
        axis=1,
    ).sum(axis=1, min_count=2)
    output["proj_total_opportunities"] = pd.concat(
        [numeric("proj_rush_attempts"), numeric("proj_targets")],
        axis=1,
    ).sum(axis=1, min_count=2)
    output["proj_pass_yards_per_attempt"] = rate(
        "proj_passing_yards", "proj_pass_attempts"
    )
    output["proj_pass_td_rate"] = rate(
        "proj_passing_tds", "proj_pass_attempts"
    )
    output["proj_interception_rate"] = rate(
        "proj_interceptions", "proj_pass_attempts"
    )
    output["proj_rush_yards_per_attempt"] = rate(
        "proj_rushing_yards", "proj_rush_attempts"
    )
    output["proj_rush_td_rate"] = rate(
        "proj_rushing_tds", "proj_rush_attempts"
    )
    output["proj_receiving_yards_per_reception"] = rate(
        "proj_receiving_yards", "proj_receptions"
    )
    output["proj_receiving_td_rate"] = rate(
        "proj_receiving_tds", "proj_receptions"
    )
    output["proj_catch_rate"] = rate("proj_receptions", "proj_targets")
    return output


def build_market_consensus(market_values: pd.DataFrame) -> pd.DataFrame:
    if market_values.empty:
        return pd.DataFrame(columns=["player_key", "season"])
    keys = ["player_key", "season"]
    family_values = canonical_adp_family_values(market_values)
    base = _group_summary(
        family_values,
        "adp",
        "adp",
        ("median", "std", "iqr", "count"),
    ).rename(columns={"adp_count": "adp_source_count"})
    expert = _group_summary(
        market_values,
        "expert_rank",
        "expert_rank",
        ("median", "std", "count"),
    )
    position = _group_summary(
        market_values,
        "source_position_rank",
        "source_position_rank",
        ("median", "std", "count"),
    )
    return base.merge(expert, on=keys, how="outer").merge(
        position,
        on=keys,
        how="outer",
    )


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.gt(0))


def add_consensus_room_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["team_normalized"] = (
        output["team"].astype("string").str.upper().map(TEAM_MAP)
    )
    eligible = (
        output["team_normalized"].notna()
        & output["position"].isin(POSITIONS)
        & output["expert_points_median"].notna()
    )
    context = output[eligible].copy()
    if context.empty:
        return output
    context["_feature_row_index"] = context.index
    team_keys = ["season", "team_normalized"]
    room_keys = [*team_keys, "position"]
    context["consensus_team_skill_points"] = context.groupby(team_keys)[
        "expert_points_median"
    ].transform("sum")
    context["consensus_room_points"] = context.groupby(room_keys)[
        "expert_points_median"
    ].transform("sum")
    context["consensus_room_share"] = _safe_share(
        context["expert_points_median"],
        context["consensus_room_points"],
    )
    context["consensus_room_rank"] = context.groupby(room_keys)[
        "expert_points_median"
    ].rank(method="min", ascending=False)
    leader = context.groupby(room_keys)["expert_points_median"].transform("max")
    context["consensus_room_gap_to_leader"] = (
        leader - context["expert_points_median"]
    )
    context["consensus_room_hhi"] = context.groupby(room_keys)[
        "consensus_room_share"
    ].transform(lambda values: float(np.square(values).sum()))
    context["consensus_room_player_count"] = context.groupby(room_keys)[
        "player_key"
    ].transform("nunique")

    sorted_context = context.sort_values(
        [*room_keys, "expert_points_median", "player_key"],
        ascending=[True, True, True, False, True],
    )
    sorted_context["next_points"] = sorted_context.groupby(room_keys)[
        "expert_points_median"
    ].shift(-1)
    sorted_context["consensus_room_gap_to_next"] = (
        sorted_context["expert_points_median"] - sorted_context["next_points"]
    )
    context["consensus_room_gap_to_next"] = sorted_context[
        "consensus_room_gap_to_next"
    ]

    qb = context[context["position"].eq("QB")].copy()
    qb = qb.sort_values(
        [*team_keys, "expert_points_median", "player_key"],
        ascending=[True, True, False, True],
    )
    qb["qb_rank"] = qb.groupby(team_keys).cumcount() + 1
    qb1 = qb[qb["qb_rank"].eq(1)][
        [
            *team_keys,
            "expert_ppg_team_game_median",
            "proj_passing_yards",
            "proj_passing_tds",
            "proj_rushing_yards",
            "proj_rushing_tds",
            "projected_rush_point_share",
        ]
    ].rename(
        columns={
            "expert_ppg_team_game_median": "team_qb1_ppg",
            "proj_passing_yards": "team_qb1_passing_yards",
            "proj_passing_tds": "team_qb1_passing_tds",
            "proj_rushing_yards": "team_qb1_rushing_yards",
            "proj_rushing_tds": "team_qb1_rushing_tds",
            "projected_rush_point_share": (
                "team_qb1_rush_point_share"
            ),
        }
    )
    qb2 = qb[qb["qb_rank"].eq(2)][
        [*team_keys, "expert_ppg_team_game_median"]
    ].rename(
        columns={"expert_ppg_team_game_median": "team_qb2_ppg"}
    )
    qb_context = qb1.merge(qb2, on=team_keys, how="left")
    qb_context["team_qb_projection_gap"] = (
        qb_context["team_qb1_ppg"] - qb_context["team_qb2_ppg"]
    )
    context = context.merge(qb_context, on=team_keys, how="left")

    skill_players = context[
        context["position"].isin(TEAM_CORE_POSITION_LIMITS)
    ].copy()
    skill_players = skill_players.sort_values(
        [*team_keys, "position", "expert_points_median", "player_key"],
        ascending=[True, True, True, False, True],
    )
    skill_players["team_position_rank"] = skill_players.groupby(
        [*team_keys, "position"]
    ).cumcount() + 1
    skill_players["team_core_member"] = skill_players[
        "team_position_rank"
    ].le(skill_players["position"].map(TEAM_CORE_POSITION_LIMITS))
    core_skill = skill_players[skill_players["team_core_member"]].copy()
    core_summary = (
        core_skill.groupby(team_keys)
        .agg(
            team_core_skill_points=(
                "expert_points_median",
                "sum",
            ),
            team_core_skill_rushing_yards=(
                "proj_rushing_yards",
                "sum",
            ),
            team_core_skill_rushing_tds=(
                "proj_rushing_tds",
                "sum",
            ),
        )
        .reset_index()
    )
    core_summary["team_core_skill_projection_percentile"] = (
        core_summary.groupby("season")["team_core_skill_points"].rank(
            method="average",
            pct=True,
        )
    )
    team_environment = core_summary.merge(
        qb1[
            [
                *team_keys,
                "team_qb1_passing_tds",
                "team_qb1_rushing_yards",
                "team_qb1_rushing_tds",
            ]
        ],
        on=team_keys,
        how="left",
    )
    team_environment["team_projected_rushing_yards"] = (
        team_environment["team_core_skill_rushing_yards"]
        + team_environment["team_qb1_rushing_yards"]
    )
    team_environment["team_projected_rushing_tds"] = (
        team_environment["team_core_skill_rushing_tds"]
        + team_environment["team_qb1_rushing_tds"]
    )
    team_environment["team_projected_offensive_tds"] = (
        team_environment["team_qb1_passing_tds"]
        + team_environment["team_projected_rushing_tds"]
    )
    context = context.merge(
        team_environment[
            [
                *team_keys,
                "team_core_skill_points",
                "team_core_skill_projection_percentile",
                "team_projected_rushing_yards",
                "team_projected_rushing_tds",
                "team_projected_offensive_tds",
            ]
        ],
        on=team_keys,
        how="left",
    )
    core_members = core_skill[
        ["player_key", "season"]
    ].drop_duplicates()
    core_members["team_core_member"] = 1
    context = context.merge(
        core_members,
        on=["player_key", "season"],
        how="left",
    )
    context["team_supporting_cast_points"] = (
        context["team_core_skill_points"]
        - context["expert_points_median"].where(
            context["team_core_member"].eq(1),
            0.0,
        )
    )

    pass_catchers = context[context["position"].isin(["WR", "TE"])].copy()
    pass_catchers["pass_catcher_room_points"] = pass_catchers.groupby(team_keys)[
        "expert_points_median"
    ].transform("sum")
    pass_catchers["pass_catcher_room_share"] = _safe_share(
        pass_catchers["expert_points_median"],
        pass_catchers["pass_catcher_room_points"],
    )
    context = context.merge(
        pass_catchers[
            ["player_key", "season", "pass_catcher_room_points", "pass_catcher_room_share"]
        ],
        on=["player_key", "season"],
        how="left",
    )

    context_columns = [
        "consensus_team_skill_points",
        "consensus_room_points",
        "consensus_room_share",
        "consensus_room_rank",
        "consensus_room_gap_to_leader",
        "consensus_room_gap_to_next",
        "consensus_room_hhi",
        "consensus_room_player_count",
        "team_qb1_ppg",
        "team_qb1_passing_yards",
        "team_qb1_passing_tds",
        "team_qb1_rushing_yards",
        "team_qb1_rushing_tds",
        "team_qb1_rush_point_share",
        "team_qb_projection_gap",
        "team_core_skill_points",
        "team_core_skill_projection_percentile",
        "team_supporting_cast_points",
        "team_projected_rushing_yards",
        "team_projected_rushing_tds",
        "team_projected_offensive_tds",
        "pass_catcher_room_points",
        "pass_catcher_room_share",
    ]
    for column in context_columns:
        output[column] = np.nan
    row_index = context["_feature_row_index"].astype(int).to_numpy()
    output.loc[row_index, context_columns] = context[
        context_columns
    ].to_numpy()
    return output


def add_experience_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare preseason PPG with same-season position/experience peers."""
    output = frame.copy()
    ppg = pd.to_numeric(
        output["expert_ppg_team_game_median"], errors="coerce"
    )
    experience = pd.to_numeric(output["year_exp"], errors="coerce")
    eligible = (
        ppg.notna()
        & experience.notna()
        & experience.ge(0)
        & output["position"].isin(POSITIONS)
    )
    output["expert_ppg_exp_peer_mean"] = np.nan
    output["expert_ppg_exp_diff"] = np.nan
    output["expert_ppg_exp_percentile"] = np.nan
    if not eligible.any():
        return output

    context = output.loc[
        eligible,
        ["season", "position", "player_key"],
    ].copy()
    context["_ppg"] = ppg.loc[eligible].astype(float)
    # Exact early-career seasons remain distinct. Sparse late-career seasons
    # share an 8+ bucket rather than borrowing a future season.
    context["_experience_bucket"] = (
        experience.loc[eligible].clip(upper=8).astype(int)
    )
    cohort_keys = ["season", "position", "_experience_bucket"]
    cohort_sum = context.groupby(cohort_keys)["_ppg"].transform("sum")
    cohort_count = context.groupby(cohort_keys)["_ppg"].transform("count")
    cohort_peer_mean = (cohort_sum - context["_ppg"]) / (
        cohort_count - 1
    ).where(cohort_count.gt(1))

    position_keys = ["season", "position"]
    position_sum = context.groupby(position_keys)["_ppg"].transform("sum")
    position_count = context.groupby(position_keys)["_ppg"].transform("count")
    position_peer_mean = (position_sum - context["_ppg"]) / (
        position_count - 1
    ).where(position_count.gt(1))
    context["expert_ppg_exp_peer_mean"] = cohort_peer_mean.fillna(
        position_peer_mean
    )
    context["expert_ppg_exp_diff"] = (
        context["_ppg"] - context["expert_ppg_exp_peer_mean"]
    )

    cohort_rank = context.groupby(cohort_keys)["_ppg"].rank(
        method="average", ascending=True
    )
    context["expert_ppg_exp_percentile"] = (
        (cohort_rank - 1) / (cohort_count - 1).where(cohort_count.gt(1))
    ).fillna(0.5)

    for column in (
        "expert_ppg_exp_peer_mean",
        "expert_ppg_exp_diff",
        "expert_ppg_exp_percentile",
    ):
        output.loc[context.index, column] = context[column].to_numpy()
    return output


def add_adp_room_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add self-excluded same-position teammate ADP comparisons."""
    output = frame.copy()
    columns = (
        "adp_best_teammate_gap",
        "adp_worst_teammate_gap",
        "adp_mean_teammate_gap",
        "adp_teammates_better_count",
        "adp_room_strength_share",
    )
    for column in columns:
        output[column] = np.nan

    adp = pd.to_numeric(output["adp_median"], errors="coerce")
    eligible = (
        adp.gt(0)
        & output["team_normalized"].notna()
        & output["position"].isin(POSITIONS)
    )
    if not eligible.any():
        return output

    context = output.loc[
        eligible,
        ["season", "team_normalized", "position", "player_key"],
    ].copy()
    context["_adp"] = adp.loc[eligible].astype(float)
    room_keys = ["season", "team_normalized", "position"]
    for _, room in context.groupby(room_keys, sort=False):
        if len(room) < 2:
            continue
        room_adp = room["_adp"]
        strength = 1 / np.sqrt(room_adp)
        strength_total = float(strength.sum())
        for row_index, own_adp in room_adp.items():
            teammates = room_adp.drop(index=row_index)
            output.loc[row_index, "adp_best_teammate_gap"] = (
                own_adp - teammates.min()
            )
            output.loc[row_index, "adp_worst_teammate_gap"] = (
                own_adp - teammates.max()
            )
            output.loc[row_index, "adp_mean_teammate_gap"] = (
                own_adp - teammates.mean()
            )
            output.loc[row_index, "adp_teammates_better_count"] = int(
                teammates.lt(own_adp).sum()
            )
            output.loc[row_index, "adp_room_strength_share"] = (
                float(strength.loc[row_index]) / strength_total
            )
    return output


def add_team_opportunity_share_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add same-season preseason opportunity shares within each NFL team."""
    output = frame.copy()
    eligible_team = output["team_normalized"].notna()
    team_keys = ["season", "team_normalized"]
    mappings = {
        "proj_targets": "team_target_share",
        "proj_receptions": "team_reception_share",
        "proj_rush_attempts": "team_rush_attempt_share",
        "proj_receiving_yards": "team_receiving_yard_share",
    }
    for source, target in mappings.items():
        output[target] = np.nan
        values = pd.to_numeric(output[source], errors="coerce")
        denominator = values.where(eligible_team).groupby(
            [output[key] for key in team_keys]
        ).transform("sum")
        output.loc[eligible_team, target] = _safe_share(
            values.loc[eligible_team],
            denominator.loc[eligible_team],
        )
    return output


def add_lifecycle_features(
    frame: pd.DataFrame,
    player_identity: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = [
        "player_key",
        "birth_date",
        "draft_round",
        "draft_pick",
    ]
    identity = player_identity[identity_columns].drop_duplicates("player_key")
    output = frame.merge(
        identity,
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    birth_date = pd.to_datetime(output["birth_date"], errors="coerce")
    season_date = pd.to_datetime(
        output["season"].astype(str) + "-09-01",
        errors="coerce",
    )
    output["age"] = (season_date - birth_date).dt.days / 365.25
    output["age_known"] = output["age"].notna().astype(int)
    output["draft_round"] = pd.to_numeric(
        output["draft_round"], errors="coerce"
    )
    output["draft_pick"] = pd.to_numeric(
        output["draft_pick"], errors="coerce"
    )
    drafted_as_of = output["draft_year"].notna() & output["draft_year"].le(
        output["season"]
    )
    output.loc[~drafted_as_of, ["draft_round", "draft_pick"]] = np.nan
    output["draft_capital_known"] = output["draft_pick"].notna().astype(int)
    output["draft_pick_log"] = np.log1p(output["draft_pick"])
    output["identity_is_confirmed"] = output["identity_status"].eq(
        "confirmed"
    ).astype(int)
    return output


def add_history_features(
    frame: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()
    history_columns = [
        "has_prior_outcome",
        "career_observed_seasons",
        "career_useful_seasons",
        "career_opportunity_games",
        "career_weighted_ppg",
        "last_observed_season",
        "seasons_since_observed",
        "last_observed_opportunity_games",
        "last_observed_ppg",
        "last_observed_points",
        "prior_year_outcome_observed",
        "prior_year_opportunity_games",
        "prior_year_ppg",
        "prior_year_points",
        "prior_year_useful",
        "prior_3year_opportunity_games",
        "prior_3year_weighted_ppg",
        "prior_3year_ppg_std",
        "prior_year_pass_point_share",
        "prior_year_rush_point_share",
        "prior_year_receiving_point_share",
    ]
    records: list[dict[str, object]] = []
    outcome_groups = {
        str(player_key): group.sort_values("season")
        for player_key, group in outcomes.groupby("player_key")
    }
    for row in output[["player_key", "season"]].itertuples(index=False):
        player_history = outcome_groups.get(str(row.player_key))
        if player_history is None:
            prior = pd.DataFrame()
        else:
            prior = player_history[player_history["season"].lt(row.season)]
        record: dict[str, object] = {
            "has_prior_outcome": int(not prior.empty),
            "career_observed_seasons": len(prior),
            "career_useful_seasons": (
                int(prior["useful_season"].sum()) if not prior.empty else 0
            ),
            "career_opportunity_games": (
                float(prior["opportunity_games"].sum()) if not prior.empty else 0
            ),
        }
        if prior.empty:
            for column in history_columns:
                record.setdefault(column, np.nan)
            record["has_prior_outcome"] = 0
            record["career_observed_seasons"] = 0
            record["career_useful_seasons"] = 0
            record["career_opportunity_games"] = 0
            records.append(record)
            continue

        total_games = prior["opportunity_games"].sum()
        record["career_weighted_ppg"] = (
            prior["season_points"].sum() / total_games
            if total_games > 0
            else np.nan
        )
        last = prior.iloc[-1]
        record["last_observed_season"] = int(last["season"])
        record["seasons_since_observed"] = int(row.season - last["season"])
        record["last_observed_opportunity_games"] = last["opportunity_games"]
        record["last_observed_ppg"] = last["conditional_ppg"]
        record["last_observed_points"] = last["season_points"]

        prior_year = prior[prior["season"].eq(row.season - 1)]
        record["prior_year_outcome_observed"] = int(not prior_year.empty)
        if not prior_year.empty:
            previous = prior_year.iloc[-1]
            record["prior_year_opportunity_games"] = previous[
                "opportunity_games"
            ]
            record["prior_year_ppg"] = previous["conditional_ppg"]
            record["prior_year_points"] = previous["season_points"]
            record["prior_year_useful"] = previous["useful_season"]
            component_total = previous["season_points"]
            if component_total != 0:
                record["prior_year_pass_point_share"] = (
                    previous["passing_points"] / component_total
                )
                record["prior_year_rush_point_share"] = (
                    previous["rushing_points"] / component_total
                )
                record["prior_year_receiving_point_share"] = (
                    previous["receiving_points"] / component_total
                )

        recent = prior[prior["season"].ge(row.season - 3)]
        recent_games = recent["opportunity_games"].sum()
        record["prior_3year_opportunity_games"] = recent_games
        record["prior_3year_weighted_ppg"] = (
            recent["season_points"].sum() / recent_games
            if recent_games > 0
            else np.nan
        )
        record["prior_3year_ppg_std"] = recent["conditional_ppg"].std(ddof=0)
        records.append(record)

    history = pd.DataFrame(records, index=output.index)
    for column in history_columns:
        if column not in history:
            history[column] = np.nan
    output[history_columns] = history[history_columns]

    prior_spine = output[
        [
            "player_key",
            "season",
            "appeared",
            "opportunity_games",
            "team_normalized",
            "expert_ppg_team_game_median",
            "conditional_ppg",
        ]
    ].copy()
    prior_spine["season"] = prior_spine["season"] + 1
    prior_spine = prior_spine.rename(
        columns={
            "appeared": "prior_year_appeared",
            "opportunity_games": "prior_year_spine_opportunity_games",
            "team_normalized": "prior_year_team",
            "expert_ppg_team_game_median": "prior_year_expert_ppg",
            "conditional_ppg": "prior_year_actual_ppg",
        }
    )
    prior_spine["prior_year_candidate"] = 1
    output = output.merge(
        prior_spine,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    output["prior_year_candidate"] = (
        output["prior_year_candidate"].fillna(0).astype(int)
    )
    output["prior_year_no_opportunity"] = (
        output["prior_year_candidate"].eq(1)
        & output["prior_year_appeared"].eq(0)
    ).astype(int)
    output["team_changed_from_prior_candidate"] = (
        output["team_normalized"].notna()
        & output["prior_year_team"].notna()
        & output["team_normalized"].ne(output["prior_year_team"])
    ).astype(int)
    output["prior_year_ppg_residual"] = (
        output["prior_year_actual_ppg"] - output["prior_year_expert_ppg"]
    )
    return output


def add_history_gap_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Express prior production as a neutral adjustment to current consensus."""
    require_columns(
        frame,
        (
            "expert_ppg_team_game_median",
            "expert_ppg_active_median",
            "career_weighted_ppg",
            "prior_year_ppg",
            "prior_3year_weighted_ppg",
            "career_opportunity_games",
            "prior_year_opportunity_games",
            "prior_3year_opportunity_games",
            "prior_year_ppg_residual",
            "seasons_since_observed",
        ),
        "history_gap_features",
    )
    output = frame.copy()
    team_game = pd.to_numeric(
        output["expert_ppg_team_game_median"], errors="coerce"
    )
    active_game = pd.to_numeric(
        output["expert_ppg_active_median"], errors="coerce"
    )
    anchor = active_game.combine_first(team_game)
    definitions = (
        (
            "career",
            "career_weighted_ppg",
            "career_opportunity_games",
        ),
        (
            "prior_year",
            "prior_year_ppg",
            "prior_year_opportunity_games",
        ),
        (
            "prior_3year",
            "prior_3year_weighted_ppg",
            "prior_3year_opportunity_games",
        ),
    )
    for prefix, value_column, games_column in definitions:
        value = pd.to_numeric(output[value_column], errors="coerce")
        games = (
            pd.to_numeric(output[games_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        gap = value - anchor
        gap = gap.mask(value.isna() & anchor.notna(), 0.0)
        reliability = games / (games + HISTORY_GAP_RELIABILITY_GAMES)
        output[f"history_{prefix}_ppg_gap"] = gap
        output[f"history_{prefix}_ppg_gap_shrunk"] = gap * reliability
        output[f"history_{prefix}_opportunity_games_log"] = np.log1p(games)

    output["history_prior_year_ppg_available"] = (
        pd.to_numeric(output["prior_year_ppg"], errors="coerce")
        .notna()
        .astype(int)
    )
    output["history_prior_3year_ppg_available"] = (
        pd.to_numeric(output["prior_3year_weighted_ppg"], errors="coerce")
        .notna()
        .astype(int)
    )
    output["history_prior_year_residual_neutral"] = pd.to_numeric(
        output["prior_year_ppg_residual"], errors="coerce"
    ).fillna(0.0)
    output["history_seasons_since_observed_neutral"] = (
        pd.to_numeric(output["seasons_since_observed"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return output


def add_projection_trajectory_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare current preseason PPG with the player's prior projections."""
    require_columns(
        frame,
        (
            "player_key",
            "season",
            "expert_ppg_team_game_median",
        ),
        "projection_trajectory_features",
    )
    output = frame.copy()
    projection_groups = {
        str(player_key): group[
            ["season", "expert_ppg_team_game_median"]
        ].sort_values("season")
        for player_key, group in output.groupby("player_key")
    }
    records: list[dict[str, object]] = []
    for row in output[
        ["player_key", "season", "expert_ppg_team_game_median"]
    ].itertuples(index=False):
        current = pd.to_numeric(
            pd.Series([row.expert_ppg_team_game_median]),
            errors="coerce",
        ).iloc[0]
        history = projection_groups[str(row.player_key)]
        prior = history[history["season"].lt(row.season)].copy()
        prior["expert_ppg_team_game_median"] = pd.to_numeric(
            prior["expert_ppg_team_game_median"],
            errors="coerce",
        )
        prior = prior.dropna(subset=["expert_ppg_team_game_median"])
        prior_year = prior[prior["season"].eq(row.season - 1)]
        recent = prior[prior["season"].ge(row.season - 3)]
        record: dict[str, object] = {
            "projection_trajectory_prior_year_available": int(
                not prior_year.empty
            ),
            "projection_trajectory_prior_3year_count": len(recent),
            "projection_trajectory_prior_3year_std": (
                float(
                    recent["expert_ppg_team_game_median"].std(ddof=0)
                )
                if not recent.empty
                else 0.0
            ),
        }
        if pd.isna(current):
            record["projection_trajectory_change_1year"] = np.nan
            record["projection_trajectory_change_3year"] = np.nan
        else:
            record["projection_trajectory_change_1year"] = (
                float(
                    current
                    - prior_year.iloc[-1][
                        "expert_ppg_team_game_median"
                    ]
                )
                if not prior_year.empty
                else 0.0
            )
            if recent.empty:
                record["projection_trajectory_change_3year"] = 0.0
            else:
                seasons_ago = row.season - recent["season"]
                weights = 4.0 - seasons_ago
                prior_weighted = np.average(
                    recent["expert_ppg_team_game_median"],
                    weights=weights,
                )
                record["projection_trajectory_change_3year"] = float(
                    current - prior_weighted
                )
        records.append(record)
    trajectory = pd.DataFrame(records, index=output.index)
    trajectory_columns = sorted(PROJECTION_TRAJECTORY_CHALLENGER_FEATURES)
    output[trajectory_columns] = trajectory[trajectory_columns]
    return output


def add_market_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["adp_log"] = np.log1p(output["adp_median"])
    group_keys = ["season", "position"]
    output["projection_position_rank"] = output.groupby(group_keys)[
        "expert_points_median"
    ].rank(method="min", ascending=False)
    output["adp_position_rank"] = output.groupby(group_keys)[
        "adp_median"
    ].rank(method="min", ascending=True)
    group_size = output.groupby(group_keys)["player_key"].transform("count")
    output["projection_position_percentile"] = 1 - (
        (output["projection_position_rank"] - 1)
        / (group_size - 1).where(group_size.gt(1))
    )
    output["adp_position_percentile"] = 1 - (
        (output["adp_position_rank"] - 1)
        / (group_size - 1).where(group_size.gt(1))
    )
    output["projection_adp_rank_diff"] = (
        output["projection_position_rank"] - output["adp_position_rank"]
    )
    output["projection_adp_percentile_diff"] = (
        output["projection_position_percentile"]
        - output["adp_position_percentile"]
    )
    return output


def _feature_family(feature: str) -> tuple[str, str]:
    if feature in PROJECTION_PROVIDER_CHALLENGER_FEATURES:
        return "provider_projection", "provider_projection"
    if feature in PROJECTION_SHAPE_CHALLENGER_FEATURES:
        return "projection_shape", "projection_shape"
    if feature in PROJECTION_DISAGREEMENT_FEATURES:
        return "projection_disagreement", "projection_disagreement"
    if feature.startswith("expert_ppg_exp_"):
        return "experience_context", "experience_context"
    if feature in {
        "adp_best_teammate_gap",
        "adp_worst_teammate_gap",
        "adp_mean_teammate_gap",
        "adp_teammates_better_count",
        "adp_room_strength_share",
    }:
        return "market_room", "market_room"
    if feature in {
        "team_target_share",
        "team_reception_share",
        "team_rush_attempt_share",
        "team_receiving_yard_share",
    }:
        return "opportunity_share", "opportunity_share"
    if feature == "proj_games":
        return "availability", "availability"
    if feature in {
        "expert_points_median",
        "expert_ppg_team_game_median",
        "expert_ppg_active_median",
    } or feature.startswith("proj_"):
        return "projection_level", "projection_level"
    if (
        feature.startswith("expert_")
        or feature.startswith("source_")
        or feature
        in {
            "projection_provider_count",
            "configured_projection_provider_count",
            "imputed_projection_provider_count",
        }
    ):
        return "projection_uncertainty", "projection_uncertainty"
    if feature.startswith("adp_") or feature.startswith("projection_adp"):
        return "market", "market"
    if feature.startswith("projection_trajectory_"):
        return "projection_trajectory", "projection_trajectory"
    if feature.startswith("room_") or feature.startswith("consensus_room"):
        return "room", "room"
    if (
        feature.startswith("team_")
        or feature.startswith("pass_catcher")
        or feature == "consensus_team_skill_points"
    ):
        return "team", "team"
    if feature.startswith("projection_position_"):
        return "projection_level", "projection_level"
    if feature.startswith("projected_") and feature.endswith("_share"):
        return "role_composition", "role_composition"
    if (
        feature.startswith("prior_")
        or feature.startswith("career_")
        or feature.startswith("last_")
        or feature.startswith("history_")
        or feature.startswith("seasons_since")
        or feature == "has_prior_outcome"
    ):
        return "history", "history"
    if feature in {
        "age",
        "age_known",
        "year_exp",
        "experience_known",
        "is_rookie",
        "draft_round",
        "draft_pick",
        "draft_pick_log",
        "draft_capital_known",
    }:
        return "lifecycle", "lifecycle"
    if feature in {
        "identity_is_confirmed",
        "position_conflict",
        "team_conflict",
        "candidate_source_count",
        "projection_source_count",
        "market_source_count",
        "ranking_source_count",
        "draft_source_count",
    }:
        return "availability", "availability"
    return "other", "other"


def build_feature_catalog(
    features: pd.DataFrame,
    feature_columns: list[str],
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        family, group = _feature_family(feature)
        residual = int(
            feature
            in (
                RESIDUAL_CANDIDATE_FEATURES
                | LEGACY_RESIDUAL_CHALLENGER_FEATURES
                | PROJECTION_RESEARCH_CHALLENGER_FEATURES
                | HISTORY_GAP_CHALLENGER_FEATURES
                | PROJECTION_TRAJECTORY_CHALLENGER_FEATURES
                | ADP_TRANSFORM_CHALLENGER_FEATURES
                | TEAM_ENVIRONMENT_CHALLENGER_FEATURES
            )
        )
        participation = int(feature in PARTICIPATION_CANDIDATE_FEATURES)
        template = int(feature in TEMPLATE_CHALLENGER_FEATURES)
        rows.append(
            {
                "feature_name": feature,
                "family": family,
                "description": feature.replace("_", " "),
                "dtype": str(features[feature].dtype),
                "collinearity_group": group,
                "residual_eligible": residual,
                "participation_eligible": participation,
                "template_eligible": template,
                "audit_only": int(not (residual or participation or template)),
                "run_id": run_id,
            }
        )
    return align_columns(
        pd.DataFrame(rows),
        FEATURE_CATALOG_COLUMNS,
        "feature_catalog",
    )


def build_feature_manifests(
    catalog: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manifest_specs = (
        ("residual_candidate_v1", RESIDUAL_CANDIDATE_FEATURES),
        (
            "residual_legacy_challenger_v1",
            LEGACY_RESIDUAL_CHALLENGER_FEATURES,
        ),
        (
            "residual_projection_challenger_v1",
            PROJECTION_RESEARCH_CHALLENGER_FEATURES,
        ),
        (
            "residual_history_gap_challenger_v1",
            HISTORY_GAP_CHALLENGER_FEATURES,
        ),
        (
            "residual_projection_trajectory_challenger_v1",
            PROJECTION_TRAJECTORY_CHALLENGER_FEATURES,
        ),
        (
            "residual_adp_transform_challenger_v1",
            ADP_TRANSFORM_CHALLENGER_FEATURES,
        ),
        (
            "residual_team_environment_challenger_v1",
            TEAM_ENVIRONMENT_CHALLENGER_FEATURES,
        ),
        ("participation_candidate_v1", PARTICIPATION_CANDIDATE_FEATURES),
        ("template_challenger_v1", TEMPLATE_CHALLENGER_FEATURES),
    )
    for manifest_name, feature_names in manifest_specs:
        selected = catalog[catalog["feature_name"].isin(feature_names)]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "manifest_name": manifest_name,
                    "feature_name": row.feature_name,
                    "family": row.family,
                    "status": "candidate",
                    "family_weight_budget": (
                        TEMPLATE_FAMILY_BUDGETS.get(row.family)
                        if manifest_name == "template_challenger_v1"
                        else np.nan
                    ),
                    "run_id": run_id,
                }
            )
    return align_columns(
        pd.DataFrame(rows),
        FEATURE_MANIFEST_COLUMNS,
        "feature_manifests",
    )


def build_feature_audit(
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    training = features["conditional_ppg_training_eligible"].eq(1)
    current = features["season"].eq(features["season"].max())
    rows: list[dict[str, object]] = []
    for row in catalog.itertuples(index=False):
        values = features[row.feature_name]
        training_values = values[training]
        current_values = values[current]
        rows.append(
            {
                "feature_name": row.feature_name,
                "family": row.family,
                "non_null_count": int(values.notna().sum()),
                "coverage_rate": float(values.notna().mean()),
                "training_non_null_count": int(training_values.notna().sum()),
                "training_coverage_rate": (
                    float(training_values.notna().mean())
                    if len(training_values)
                    else np.nan
                ),
                "current_non_null_count": int(current_values.notna().sum()),
                "current_coverage_rate": (
                    float(current_values.notna().mean())
                    if len(current_values)
                    else np.nan
                ),
                "unique_count": int(values.nunique(dropna=True)),
                "zero_variance": int(values.nunique(dropna=True) <= 1),
                "run_id": run_id,
            }
        )
    return align_columns(
        pd.DataFrame(rows),
        FEATURE_AUDIT_COLUMNS,
        "feature_audit",
    )


def build_feature_correlations(
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    run_id: str,
    threshold: float = 0.90,
    min_shared_rows: int = 100,
) -> pd.DataFrame:
    numeric_catalog = catalog[
        catalog["feature_name"].isin(
            features.select_dtypes(include=[np.number, "boolean"]).columns
        )
    ]
    rows: list[dict[str, object]] = []
    for family, group in numeric_catalog.groupby("collinearity_group"):
        columns = group["feature_name"].tolist()
        for feature_a, feature_b in combinations(columns, 2):
            pair = features[[feature_a, feature_b]].dropna()
            if len(pair) < min_shared_rows:
                continue
            if pair[feature_a].nunique() <= 1 or pair[feature_b].nunique() <= 1:
                continue
            correlation = pair[feature_a].corr(
                pair[feature_b], method="spearman"
            )
            if pd.isna(correlation) or abs(correlation) < threshold:
                continue
            rows.append(
                {
                    "family": family,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman": float(correlation),
                    "abs_spearman": float(abs(correlation)),
                    "shared_rows": len(pair),
                    "run_id": run_id,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=FEATURE_CORRELATION_COLUMNS)
    return align_columns(
        frame,
        FEATURE_CORRELATION_COLUMNS,
        "feature_correlations",
    ).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def build_feature_mart(
    spine: pd.DataFrame,
    player_identity: pd.DataFrame,
    outcomes: pd.DataFrame,
    projection_values: pd.DataFrame,
    market_values: pd.DataFrame,
    run_id: str,
    spine_run_id: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    require_columns(
        spine,
        (
            "player_key",
            "season",
            "position",
            "team",
            "feature_cutoff_season",
            "conditional_ppg_training_eligible",
        ),
        "player_season_spine",
    )
    keys = ["player_key", "season"]
    features = spine.copy()
    features = features.rename(columns={"run_id": "spine_run_id"})
    features = features.merge(
        build_projection_consensus(projection_values),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        build_market_consensus(market_values),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    features = add_projection_shape_features(features)
    features = add_consensus_room_features(features)
    features = add_experience_context_features(features)
    features = add_adp_room_features(features)
    features = add_team_opportunity_share_features(features)
    features = add_lifecycle_features(features, player_identity)
    features = add_history_features(features, outcomes)
    features = add_history_gap_features(features)
    features = add_projection_trajectory_features(features)
    features = add_market_rank_features(features)
    features["feature_foundation_run_id"] = spine_run_id
    features["run_id"] = run_id

    if len(features) != len(spine):
        raise ValueError("Feature mart must preserve every spine row")
    if features.duplicated(["player_key", "season", "league"]).any():
        raise ValueError("Feature mart contains duplicate player-seasons")
    if not features["feature_cutoff_season"].eq(
        features["season"] - 1
    ).all():
        raise ValueError("Feature history cutoff is not strictly prior")
    numeric = features.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Feature mart contains infinite numeric values")

    missing_features = sorted(FEATURE_MART_FEATURES.difference(features.columns))
    if missing_features:
        raise ValueError(
            "Reviewed feature mart contract is missing constructed columns: "
            f"{missing_features}"
        )
    feature_columns = [
        column for column in features.columns if column in FEATURE_MART_FEATURES
    ]
    base_columns = [
        column for column in spine.columns if column != "run_id"
    ] + ["spine_run_id", "feature_foundation_run_id", "run_id"]
    features = features.loc[
        :,
        list(dict.fromkeys([*base_columns, *feature_columns])),
    ].copy()
    catalog = build_feature_catalog(features, feature_columns, run_id)
    manifests = build_feature_manifests(catalog, run_id)
    audit = build_feature_audit(features, catalog, run_id)
    correlations = build_feature_correlations(features, catalog, run_id)
    return features, catalog, manifests, audit, correlations
