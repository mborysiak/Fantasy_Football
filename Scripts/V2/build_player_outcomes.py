"""Build exact-season V2 player outcomes from canonical nflverse weekly stats."""

from __future__ import annotations

import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import pandas as pd

from Scripts.V2.config import (
    COMPLETED_THROUGH_SEASON,
    NFLVERSE_WEEKLY_STATS_URL,
    POSITIONS,
    QB_MIN_OFFENSIVE_PLAYS,
    USEFUL_SEASON_MIN_GAMES,
)
from Scripts.V2.contracts import (
    PLAYER_OUTCOME_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    align_columns,
    bytes_sha256,
    configured_scoring,
    scoring_hash,
)


WEEKLY_REQUIRED_COLUMNS = (
    "player_id",
    "player_display_name",
    "position_group",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "fumbles_lost_total",
    "special_teams_tds",
)


def _read_csv_payload(url: str) -> tuple[pd.DataFrame, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Fantasy-Football-V2/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    frame = pd.read_csv(
        io.BytesIO(payload),
        usecols=lambda column: column in WEEKLY_REQUIRED_COLUMNS,
        low_memory=False,
    )
    return frame, bytes_sha256(payload)


def fetch_weekly_stats(
    seasons: Iterable[int],
    url_template: str = NFLVERSE_WEEKLY_STATS_URL,
    max_workers: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    manifest: list[dict[str, object]] = []

    def fetch(season: int) -> tuple[int, str, pd.DataFrame, str]:
        url = url_template.format(season=season)
        frame, checksum = _read_csv_payload(url)
        return season, url, frame, checksum

    season_values = sorted({int(season) for season in seasons})
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, season): season for season in season_values}
        for future in as_completed(futures):
            season, url, frame, checksum = future.result()
            frames[season] = frame
            manifest.append(
                {
                    "component": "outcomes",
                    "source_name": f"nflverse_weekly_stats_{season}",
                    "source_uri": url,
                    "source_sha256": checksum,
                    "row_count": len(frame),
                }
            )

    weekly = pd.concat([frames[season] for season in season_values], ignore_index=True)
    return weekly, pd.DataFrame(manifest).sort_values("source_name").reset_index(
        drop=True
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def score_weekly_stats(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    """Score weekly nflverse rows using the repository's configured league rules."""
    missing = [column for column in WEEKLY_REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"weekly stats is missing required columns: {missing}")

    scored = frame.copy()
    rules = configured_scoring(league)
    passing = rules["passing"]
    rushing = rules["rushing"]
    receiving = rules["receiving"]

    pass_yards = _numeric(scored, "passing_yards")
    rush_yards = _numeric(scored, "rushing_yards")
    rec_yards = _numeric(scored, "receiving_yards")

    scored["passing_points"] = (
        pass_yards * passing.get("pass_yards_gained_sum", 0.0)
        + _numeric(scored, "passing_tds")
        * passing.get("pass_pass_touchdown_sum", 0.0)
        + _numeric(scored, "passing_interceptions")
        * passing.get("pass_interception_sum", 0.0)
        + _numeric(scored, "sacks_suffered") * passing.get("sack_sum", 0.0)
        + pass_yards.ge(300).astype(float)
        * passing.get("pass_yd_300_bonus", 0.0)
        + pass_yards.ge(400).astype(float)
        * passing.get("pass_yd_400_bonus", 0.0)
    )
    scored["rushing_points"] = (
        rush_yards * rushing.get("rush_yards_gained_sum", 0.0)
        + _numeric(scored, "rushing_tds")
        * rushing.get("rush_rush_touchdown_sum", 0.0)
        + rush_yards.ge(100).astype(float)
        * rushing.get("rush_yd_100_bonus", 0.0)
        + rush_yards.ge(200).astype(float)
        * rushing.get("rush_yd_200_bonus", 0.0)
    )
    scored["receiving_points"] = (
        _numeric(scored, "receptions")
        * receiving.get("rec_complete_pass_sum", 0.0)
        + rec_yards * receiving.get("rec_yards_gained_sum", 0.0)
        + _numeric(scored, "receiving_tds")
        * receiving.get("rec_pass_touchdown_sum", 0.0)
        + rec_yards.ge(100).astype(float)
        * receiving.get("rec_yd_100_bonus", 0.0)
        + rec_yards.ge(200).astype(float)
        * receiving.get("rec_yd_200_bonus", 0.0)
    )

    scored["fumble_points"] = _numeric(
        scored, "fumbles_lost_total"
    ) * rushing.get("fumble_lost", 0.0)

    # The current repository scoring dictionaries do not expose two-point or
    # return-TD settings. Keep the components explicit and zero unless those
    # settings are later added, rather than silently changing historical labels.
    scored["two_point_points"] = (
        _numeric(scored, "passing_2pt_conversions")
        * passing.get("pass_two_point_conv", 0.0)
        + _numeric(scored, "rushing_2pt_conversions")
        * rushing.get("rush_two_point_conv", 0.0)
        + _numeric(scored, "receiving_2pt_conversions")
        * receiving.get("rec_two_point_conv", 0.0)
    )
    scored["special_teams_points"] = _numeric(
        scored, "special_teams_tds"
    ) * receiving.get("special_teams_touchdown", 0.0)

    component_columns = [
        "passing_points",
        "rushing_points",
        "receiving_points",
        "fumble_points",
        "two_point_points",
        "special_teams_points",
    ]
    scored["fantasy_points_configured"] = scored[component_columns].sum(axis=1)
    return scored


def _fantasy_week_mask(frame: pd.DataFrame) -> pd.Series:
    last_week = np.where(_numeric(frame, "season").ge(2021), 17, 16)
    return _numeric(frame, "week").le(last_week)


def _opportunity_mask(frame: pd.DataFrame) -> pd.Series:
    position = (
        frame["position_group"].astype("string").str.upper().fillna("")
    )
    qb_plays = (
        _numeric(frame, "attempts")
        + _numeric(frame, "sacks_suffered")
        + _numeric(frame, "carries")
    )
    skill_opportunities = _numeric(frame, "targets") + _numeric(frame, "carries")
    return pd.Series(
        np.where(
            position.eq("QB").to_numpy(dtype=bool),
            qb_plays.gt(QB_MIN_OFFENSIVE_PLAYS).to_numpy(dtype=bool),
            skill_opportunities.gt(0).to_numpy(dtype=bool),
        ),
        index=frame.index,
    )


def _alias_position_map(
    player_aliases: pd.DataFrame | None,
) -> dict[tuple[str, int], str]:
    if player_aliases is None or player_aliases.empty:
        return {}
    aliases = player_aliases.copy()
    aliases["season"] = pd.to_numeric(
        aliases["season"], errors="coerce"
    ).astype("Int64")
    aliases["position"] = aliases["position"].astype("string").str.upper()
    aliases = aliases[
        aliases["player_key"].notna()
        & aliases["season"].notna()
        & aliases["position"].isin(POSITIONS)
    ].drop_duplicates(["player_key", "season", "source", "position"])
    if aliases.empty:
        return {}

    votes = (
        aliases.groupby(["player_key", "season", "position"], as_index=False)
        .size()
        .sort_values(
            ["player_key", "season", "size", "position"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["player_key", "season"])
    )
    return {
        (str(row.player_key), int(row.season)): str(row.position)
        for row in votes.itertuples(index=False)
    }


def aggregate_player_outcomes(
    weekly: pd.DataFrame,
    player_identity: pd.DataFrame,
    league: str,
    run_id: str,
    player_aliases: pd.DataFrame | None = None,
    completed_through_season: int = COMPLETED_THROUGH_SEASON,
    useful_season_min_games: int = USEFUL_SEASON_MIN_GAMES,
) -> pd.DataFrame:
    """Aggregate exact calendar-season outcomes without shifts or forward fills."""
    scored = score_weekly_stats(weekly, league)
    scored["position_group"] = (
        scored["position_group"].astype("string").str.upper()
    )
    scored["season"] = pd.to_numeric(scored["season"], errors="coerce").astype(
        "Int64"
    )
    identity_map = (
        player_identity[player_identity["gsis_id"].notna()]
        .drop_duplicates("gsis_id")
        .set_index("gsis_id")["player_key"]
        .to_dict()
    )
    scored["player_key"] = scored["player_id"].astype(str).map(identity_map)
    alias_positions = _alias_position_map(player_aliases)
    fallback_positions = [
        alias_positions.get((str(player_key), int(season)))
        if pd.notna(player_key) and pd.notna(season)
        else None
        for player_key, season in scored[
            ["player_key", "season"]
        ].itertuples(index=False, name=None)
    ]
    scored["resolved_position"] = scored["position_group"]
    non_fantasy_position = ~scored["resolved_position"].isin(POSITIONS)
    scored.loc[non_fantasy_position, "resolved_position"] = pd.Series(
        fallback_positions, index=scored.index
    )[non_fantasy_position]
    scored["position_group"] = scored["resolved_position"]

    eligible = (
        scored["position_group"].isin(POSITIONS)
        & scored["season_type"].eq("REG")
        & _fantasy_week_mask(scored)
        & _opportunity_mask(scored)
        & scored["player_id"].notna()
    )
    missing_ids = scored.loc[
        eligible & scored["player_key"].isna(),
        ["player_id", "player_display_name"],
    ].drop_duplicates()
    if not missing_ids.empty:
        raise ValueError(
            "Weekly outcomes contain gsis_id values missing from player_identity:\n"
            f"{missing_ids.head(10).to_string(index=False)}"
        )
    scored = scored[eligible & scored["player_key"].notna()].copy()
    if scored.empty:
        return pd.DataFrame(columns=PLAYER_OUTCOME_COLUMNS)

    numeric_aggregations = {
        "fantasy_points_configured": "sum",
        "passing_points": "sum",
        "rushing_points": "sum",
        "receiving_points": "sum",
        "fumble_points": "sum",
        "two_point_points": "sum",
        "special_teams_points": "sum",
        "attempts": "sum",
        "carries": "sum",
        "targets": "sum",
        "receptions": "sum",
        "passing_yards": "sum",
        "rushing_yards": "sum",
        "receiving_yards": "sum",
        "passing_tds": "sum",
        "rushing_tds": "sum",
        "receiving_tds": "sum",
    }
    grouped = scored.groupby(["player_id", "season"], as_index=False).agg(
        {
            "player_key": "last",
            "player_display_name": "last",
            "position_group": "last",
            "game_id": "nunique",
            **numeric_aggregations,
        }
    )
    teams = (
        scored.groupby(["player_id", "season"])["team"]
        .agg(
            lambda values: "|".join(
                sorted({str(value) for value in values if pd.notna(value)})
            )
        )
        .rename("teams")
        .reset_index()
    )
    grouped = grouped.merge(teams, on=["player_id", "season"], how="left")

    grouped = grouped.rename(
        columns={
            "player_id": "gsis_id",
            "player_display_name": "display_name",
            "position_group": "position",
            "game_id": "opportunity_games",
            "fantasy_points_configured": "season_points",
            "attempts": "pass_attempts",
            "carries": "rush_attempts",
        }
    )
    grouped["conditional_ppg"] = (
        grouped["season_points"] / grouped["opportunity_games"]
    )
    grouped["appeared"] = grouped["opportunity_games"].gt(0).astype(int)
    grouped["useful_season"] = (
        grouped["opportunity_games"].ge(useful_season_min_games).astype(int)
    )
    grouped["outcome_complete"] = (
        grouped["season"].le(completed_through_season).astype(int)
    )
    grouped["target_available"] = grouped["outcome_complete"]
    grouped.loc[grouped["target_available"].eq(0), "conditional_ppg"] = np.nan
    grouped["league"] = league
    grouped["scoring_hash"] = scoring_hash(league)
    grouped["run_id"] = run_id
    grouped = align_columns(grouped, PLAYER_OUTCOME_COLUMNS, "player_outcomes")
    return grouped.sort_values(
        ["season", "position", "season_points"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def validate_outcomes(
    outcomes: pd.DataFrame,
    useful_season_min_games: int = USEFUL_SEASON_MIN_GAMES,
) -> None:
    if outcomes.empty:
        raise ValueError("player_season_outcomes cannot be empty")
    key_columns = ["player_key", "season", "league"]
    if outcomes[key_columns].isna().any().any():
        raise ValueError("Outcome identity and season columns cannot be null")
    if outcomes.duplicated(key_columns).any():
        raise ValueError("player_season_outcomes contains duplicate player-seasons")

    components = outcomes[
        [
            "passing_points",
            "rushing_points",
            "receiving_points",
            "fumble_points",
            "two_point_points",
            "special_teams_points",
        ]
    ].sum(axis=1)
    if not np.allclose(
        components.to_numpy(), outcomes["season_points"].to_numpy(), atol=1e-9
    ):
        raise ValueError("Outcome components do not reconcile to season_points")

    expected_useful = outcomes["opportunity_games"].ge(
        useful_season_min_games
    ).astype(int)
    if not expected_useful.equals(outcomes["useful_season"].astype(int)):
        raise ValueError("useful_season does not match the configured threshold")

    invalid_target = outcomes["target_available"].eq(0) & outcomes[
        "conditional_ppg"
    ].notna()
    if invalid_target.any():
        raise ValueError("Incomplete outcomes cannot expose conditional_ppg targets")


def build_player_outcome_frames(
    player_identity: pd.DataFrame,
    seasons: Iterable[int],
    league: str,
    run_id: str,
    player_aliases: pd.DataFrame | None = None,
    completed_through_season: int = COMPLETED_THROUGH_SEASON,
    useful_season_min_games: int = USEFUL_SEASON_MIN_GAMES,
    max_workers: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly, manifest = fetch_weekly_stats(seasons, max_workers=max_workers)
    outcomes = aggregate_player_outcomes(
        weekly,
        player_identity,
        league=league,
        run_id=run_id,
        player_aliases=player_aliases,
        completed_through_season=completed_through_season,
        useful_season_min_games=useful_season_min_games,
    )
    validate_outcomes(outcomes, useful_season_min_games)
    manifest["run_id"] = run_id
    manifest = align_columns(manifest, SOURCE_MANIFEST_COLUMNS, "source_manifest")
    return outcomes, manifest
