"""Build the leakage-safe V2 player-season projection spine."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from Scripts.V2.config import (
    COMPLETED_THROUGH_SEASON,
    POSITIONS,
    START_SEASON,
    candidate_source_kind,
)
from Scripts.V2.contracts import (
    PLAYER_SEASON_SOURCE_COLUMNS,
    PLAYER_SEASON_SPINE_COLUMNS,
    align_columns,
    apply_source_row_exclusions,
    require_columns,
    scoring_hash,
)


def _clean_text(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _deterministic_mode(
    values: Iterable[object],
    preferred: object = None,
) -> object:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return pd.NA
    counts = pd.Series(cleaned).value_counts()
    leaders = sorted(counts[counts.eq(counts.max())].index.tolist())
    if preferred is not None and pd.notna(preferred) and str(preferred) in leaders:
        return str(preferred)
    return leaders[0]


def build_player_season_sources(
    player_aliases: pd.DataFrame,
    player_identity: pd.DataFrame,
    run_id: str,
    start_season: int = START_SEASON,
    projection_through_season: int | None = None,
) -> pd.DataFrame:
    """Collapse auditable preseason aliases to one source observation per player-year."""
    require_columns(
        player_aliases,
        (
            "player_key",
            "source",
            "source_name",
            "position",
            "team",
            "season",
            "match_method",
        ),
        "player_aliases",
    )
    require_columns(
        player_identity,
        ("player_key", "position"),
        "player_identity",
    )

    aliases = player_aliases.copy()
    if "source_table" not in aliases:
        aliases["source_table"] = pd.NA
    aliases = apply_source_row_exclusions(
        aliases,
        "player_aliases for projection spine",
    )
    aliases["season"] = pd.to_numeric(
        aliases["season"], errors="coerce"
    ).astype("Int64")
    aliases["source_kind"] = aliases["source"].map(candidate_source_kind)
    aliases = aliases[
        aliases["source_kind"].notna()
        & aliases["season"].notna()
        & aliases["season"].ge(start_season)
    ].copy()
    if projection_through_season is not None:
        aliases = aliases[
            aliases["season"].le(projection_through_season)
        ].copy()
    if aliases.empty:
        return pd.DataFrame(columns=PLAYER_SEASON_SOURCE_COLUMNS)

    identity_position = (
        player_identity.drop_duplicates("player_key")
        .set_index("player_key")["position"]
        .astype("string")
        .str.upper()
        .to_dict()
    )
    aliases["source_position"] = (
        aliases["position"].astype("string").str.upper()
    )
    missing_position = ~aliases["source_position"].isin(POSITIONS)
    aliases.loc[missing_position, "source_position"] = aliases.loc[
        missing_position, "player_key"
    ].map(identity_position)
    aliases.loc[
        ~aliases["source_position"].isin(POSITIONS),
        "source_position",
    ] = pd.NA
    aliases["source_team"] = aliases["team"].map(_clean_text)
    aliases["source_player_name"] = aliases["source_name"].map(_clean_text)

    rows: list[dict[str, object]] = []
    keys = ["player_key", "season", "source", "source_kind"]
    for key, group in aliases.groupby(keys, dropna=False, sort=True):
        player_key, season, source, source_kind = key
        rows.append(
            {
                "player_key": player_key,
                "season": int(season),
                "source": source,
                "source_kind": source_kind,
                "source_player_name": _deterministic_mode(
                    group["source_player_name"]
                ),
                "source_position": _deterministic_mode(
                    group["source_position"]
                ),
                "source_team": _deterministic_mode(group["source_team"]),
                "match_method": "|".join(
                    sorted(
                        {
                            str(value)
                            for value in group["match_method"]
                            if pd.notna(value)
                        }
                    )
                ),
                "record_count": len(group),
                "run_id": run_id,
            }
        )

    sources = pd.DataFrame(rows)
    eligible_player_seasons = set(
        sources.loc[
            sources["source_position"].isin(POSITIONS),
            ["player_key", "season"],
        ].itertuples(index=False, name=None)
    )
    sources = sources[
        [
            (player_key, season) in eligible_player_seasons
            for player_key, season in sources[
                ["player_key", "season"]
            ].itertuples(index=False, name=None)
        ]
    ].copy()
    sources = align_columns(
        sources,
        PLAYER_SEASON_SOURCE_COLUMNS,
        "player_season_sources",
    )
    return sources.sort_values(
        ["season", "player_key", "source"]
    ).reset_index(drop=True)


def _candidate_summary(
    player_sources: pd.DataFrame,
    player_identity: pd.DataFrame,
) -> pd.DataFrame:
    identity = player_identity.drop_duplicates("player_key").set_index(
        "player_key"
    )
    rows: list[dict[str, object]] = []
    for (player_key, season), group in player_sources.groupby(
        ["player_key", "season"],
        sort=True,
    ):
        identity_row = identity.loc[player_key]
        canonical_position = identity_row["position"]
        position = _deterministic_mode(
            group["source_position"],
            preferred=(
                canonical_position
                if canonical_position in POSITIONS
                else None
            ),
        )
        if pd.isna(position) or position not in POSITIONS:
            continue
        team = _deterministic_mode(group["source_team"])
        source_kinds = group["source_kind"].value_counts()
        non_draft_count = int(
            source_kinds.get("projection", 0)
            + source_kinds.get("market", 0)
            + source_kinds.get("ranking", 0)
        )
        position_values = {
            str(value)
            for value in group["source_position"]
            if pd.notna(value)
        }
        team_values = {
            str(value)
            for value in group["source_team"]
            if pd.notna(value)
        }
        rows.append(
            {
                "player_key": player_key,
                "season": int(season),
                "position": position,
                "team": team,
                "candidate_rule": (
                    "preseason_evidence"
                    if non_draft_count > 0
                    else "drafted_rookie_only"
                ),
                "candidate_source_count": int(group["source"].nunique()),
                "projection_source_count": int(
                    source_kinds.get("projection", 0)
                ),
                "market_source_count": int(source_kinds.get("market", 0)),
                "ranking_source_count": int(source_kinds.get("ranking", 0)),
                "draft_source_count": int(source_kinds.get("draft", 0)),
                "candidate_sources": "|".join(sorted(set(group["source"]))),
                "position_conflict": int(len(position_values) > 1),
                "team_conflict": int(len(team_values) > 1),
            }
        )
    return pd.DataFrame(rows)


def build_player_season_spine(
    player_sources: pd.DataFrame,
    player_identity: pd.DataFrame,
    player_outcomes: pd.DataFrame,
    league: str,
    run_id: str,
    foundation_run_id: str,
    completed_through_season: int = COMPLETED_THROUGH_SEASON,
) -> pd.DataFrame:
    """Join preseason candidates to explicit participation and PPG targets."""
    if player_sources.empty:
        return pd.DataFrame(columns=PLAYER_SEASON_SPINE_COLUMNS)
    require_columns(
        player_outcomes,
        (
            "player_key",
            "season",
            "league",
            "opportunity_games",
            "season_points",
            "conditional_ppg",
            "appeared",
            "useful_season",
            "target_available",
        ),
        "player_season_outcomes",
    )

    spine = _candidate_summary(player_sources, player_identity)
    identity_columns = [
        "player_key",
        "gsis_id",
        "display_name",
        "identity_status",
        "identity_source",
        "draft_year",
        "rookie_season",
    ]
    spine = spine.merge(
        player_identity[identity_columns].drop_duplicates("player_key"),
        on="player_key",
        how="left",
        validate="many_to_one",
    )

    for column in ("draft_year", "rookie_season"):
        spine[column] = pd.to_numeric(
            spine[column], errors="coerce"
        ).astype("Int64")
        spine.loc[spine[column].gt(spine["season"]), column] = pd.NA
    experience_start = spine["rookie_season"].fillna(spine["draft_year"])
    spine["year_exp"] = (
        spine["season"].astype("Int64") - experience_start
    ).astype("Int64")
    spine.loc[spine["year_exp"].lt(0), "year_exp"] = pd.NA
    experience_known = experience_start.notna()
    spine["experience_known"] = experience_known.astype(int)
    spine["is_rookie"] = pd.Series(
        pd.NA,
        index=spine.index,
        dtype="Int64",
    )
    spine.loc[experience_known, "is_rookie"] = (
        experience_start[experience_known]
        .eq(spine.loc[experience_known, "season"])
        .astype(int)
    )

    outcome_columns = [
        "player_key",
        "season",
        "league",
        "opportunity_games",
        "season_points",
        "conditional_ppg",
        "appeared",
        "useful_season",
        "target_available",
    ]
    outcomes = player_outcomes[
        player_outcomes["league"].eq(league)
    ][outcome_columns].copy()
    outcomes = outcomes.rename(
        columns={"season_points": "observed_season_points"}
    )
    outcomes["outcome_observed"] = 1
    spine = spine.merge(
        outcomes,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )

    spine["outcome_complete"] = spine["season"].le(
        completed_through_season
    ).astype(int)
    complete = spine["outcome_complete"].eq(1)
    observed = spine["outcome_observed"].eq(1)
    pending = ~complete
    identity_resolved = spine["identity_status"].eq("confirmed")
    participation_resolved = identity_resolved | observed
    participation_available = complete & participation_resolved
    unresolved_identity = complete & ~participation_resolved

    spine["outcome_observed"] = (
        spine["outcome_observed"].fillna(0).astype(int)
    )
    spine["outcome_join_status"] = np.select(
        [pending, observed, unresolved_identity],
        ["pending", "observed_opportunity", "unresolved_identity"],
        default="no_opportunity",
    )
    spine["active_target_available"] = participation_available.astype(int)

    for column in ("appeared", "opportunity_games", "useful_season"):
        spine.loc[participation_available, column] = spine.loc[
            participation_available, column
        ].fillna(0)
        spine.loc[~participation_available, column] = pd.NA
        spine[column] = pd.to_numeric(
            spine[column], errors="coerce"
        ).astype("Int64")

    spine["unconditional_season_points"] = np.where(
        participation_available,
        spine["observed_season_points"].fillna(0.0),
        np.nan,
    )
    spine.loc[~observed, "conditional_ppg"] = np.nan
    spine.loc[pending, "observed_season_points"] = np.nan
    spine["conditional_ppg_target_available"] = (
        complete & observed & spine["target_available"].eq(1)
    ).astype(int)
    spine["conditional_ppg_training_eligible"] = (
        spine["conditional_ppg_target_available"].eq(1)
        & spine["useful_season"].eq(1)
    ).astype(int)
    spine = spine.drop(columns="target_available")

    spine["feature_cutoff_season"] = spine["season"] - 1
    spine["preseason_source_season"] = spine["season"]
    spine["league"] = league
    spine["scoring_hash"] = scoring_hash(league)
    spine["foundation_run_id"] = foundation_run_id
    spine["run_id"] = run_id
    spine = align_columns(
        spine,
        PLAYER_SEASON_SPINE_COLUMNS,
        "player_season_spine",
    )
    return spine.sort_values(
        ["season", "position", "display_name", "player_key"]
    ).reset_index(drop=True)


def validate_projection_spine(
    player_sources: pd.DataFrame,
    spine: pd.DataFrame,
) -> None:
    if player_sources.empty or spine.empty:
        raise ValueError("Projection spine and source observations cannot be empty")
    if player_sources.duplicated(
        ["player_key", "season", "source"]
    ).any():
        raise ValueError(
            "player_season_sources contains duplicate player-season sources"
        )
    if spine.duplicated(["player_key", "season", "league"]).any():
        raise ValueError("player_season_spine contains duplicate player-seasons")
    if not spine["position"].isin(POSITIONS).all():
        raise ValueError("Projection spine contains an ineligible position")
    if not spine["feature_cutoff_season"].eq(spine["season"] - 1).all():
        raise ValueError("Feature cutoffs must be strictly before target seasons")
    if not spine["preseason_source_season"].eq(spine["season"]).all():
        raise ValueError("Preseason evidence must align to the projected season")
    unknown_experience = spine["experience_known"].eq(0)
    if spine.loc[unknown_experience, ["year_exp", "is_rookie"]].notna().any().any():
        raise ValueError("Unknown experience cannot be encoded as veteran experience")
    if spine.loc[
        ~unknown_experience, ["year_exp", "is_rookie"]
    ].isna().any().any():
        raise ValueError("Known experience must expose year_exp and is_rookie")

    source_counts = player_sources.groupby(
        ["player_key", "season"]
    )["source"].nunique()
    spine_counts = spine.set_index(
        ["player_key", "season"]
    )["candidate_source_count"]
    if not source_counts.equals(spine_counts.reindex(source_counts.index)):
        raise ValueError("Candidate source counts do not reconcile")

    complete = spine["outcome_complete"].eq(1)
    pending = ~complete
    participation_available = spine["active_target_available"].eq(1)
    participation_unavailable = ~participation_available
    if spine.loc[
        participation_available,
        ["appeared", "opportunity_games", "useful_season"],
    ].isna().any().any():
        raise ValueError("Available candidate seasons need participation labels")
    if spine.loc[
        participation_unavailable,
        [
            "appeared",
            "opportunity_games",
            "useful_season",
            "unconditional_season_points",
            "conditional_ppg",
        ],
    ].notna().any().any():
        raise ValueError("Unavailable rows cannot expose outcome labels")
    expected_active = (
        complete
        & (
            spine["identity_status"].eq("confirmed")
            | spine["outcome_observed"].eq(1)
        )
    ).astype(int)
    if not spine["active_target_available"].eq(expected_active).all():
        raise ValueError(
            "Active-target availability requires a completed season and "
            "resolved identity"
        )

    no_opportunity = spine["outcome_join_status"].eq("no_opportunity")
    if not spine.loc[no_opportunity, "appeared"].eq(0).all():
        raise ValueError("Completed source candidates without outcomes must be inactive")
    if spine.loc[no_opportunity, "conditional_ppg"].notna().any():
        raise ValueError("Inactive candidates cannot receive conditional PPG")
    if not spine.loc[
        no_opportunity, "unconditional_season_points"
    ].eq(0).all():
        raise ValueError("Inactive candidates must have zero unconditional points")
    unresolved = spine["outcome_join_status"].eq("unresolved_identity")
    if spine.loc[
        unresolved,
        [
            "appeared",
            "opportunity_games",
            "useful_season",
            "unconditional_season_points",
            "conditional_ppg",
        ],
    ].notna().any().any():
        raise ValueError("Unresolved identities cannot receive outcome labels")

    expected_ppg_available = (
        complete & spine["outcome_observed"].eq(1)
    ).astype(int)
    if not spine["conditional_ppg_target_available"].eq(
        expected_ppg_available
    ).all():
        raise ValueError("Conditional-PPG availability does not match observations")
    expected_training = (
        expected_ppg_available.eq(1) & spine["useful_season"].eq(1)
    ).astype(int)
    if not spine["conditional_ppg_training_eligible"].eq(
        expected_training
    ).all():
        raise ValueError("Conditional-PPG training eligibility is inconsistent")
