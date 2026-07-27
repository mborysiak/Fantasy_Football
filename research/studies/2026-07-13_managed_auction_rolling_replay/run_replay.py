"""Leakage-safe rolling-origin replay of managed auction roster construction.

The study intentionally imports only the current pure simulation helper. It does
not import the production salary/model builders because those modules mutate
production SQLite tables at import/run time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
GITHUB_ROOT = ROOT.parent
APP_ROOT = GITHUB_ROOT / "Fantasy_Football_App"
APP_HELPER = APP_ROOT / "app" / "zSim_Helper.py"
DAILY_DB = GITHUB_ROOT / "Daily_Fantasy_Data" / "Databases" / "FastR_Beta.sqlite3"
SIM_DB = ROOT / "Data" / "Databases" / "Simulation.sqlite3"

sys.path.insert(0, str(APP_ROOT / "app"))
sys.path.insert(0, str(ROOT / "Scripts"))

from config import get_scoring_dict  # noqa: E402
from zSim_Helper import (  # noqa: E402
    FootballSimulation,
    MANAGED_LINEUP_REQUIRE,
    MANAGED_POS_MAX,
    WAIVER_PLAYER_COUNTS,
)


POSITIONS = ("QB", "RB", "WR", "TE")
WEEKS = tuple(range(1, 17))
WEEK_COLS = [f"week_{week}" for week in WEEKS]
PLAYED_COLS = [f"played_week_{week}" for week in WEEKS]
ROSTER_SIZE = 13
SALARY_CAP = 298.0
NUM_TEAMS = 12
TOTAL_MARKET_BUDGET = NUM_TEAMS * SALARY_CAP
TOTAL_MARKET_SLOTS = NUM_TEAMS * ROSTER_SIZE
TOP_N = 12
LINEUP_REQUIRE = dict(MANAGED_LINEUP_REQUIRE)
POS_MIN = {pos: int(LINEUP_REQUIRE[pos]) for pos in POSITIONS}
POS_MAX = dict(MANAGED_POS_MAX)
ZERO_WAIVERS = {pos: 0.0 for pos in POSITIONS}
MATCH_FILL_VALUE = 0.5
MAX_TEMPLATE_POOL_SIZE = 80
TEMPLATE_TOP_TO_BOTTOM_RATIO = 2.0
QB_RANK_DISTANCE_ORDER = {
    "qb1": 0,
    "qb2": 1,
    "qb3_plus": 2,
    "unknown": 2,
    "non_qb": 2,
}
MATCH_FEATURE_WEIGHTS = {
    "QB": {
        "match_projection_rank_pct": 2.5,
        "year_exp_scaled": 2.0,
        "projection_x_exp": 1.0,
        "adp_rank_pct": 0.5,
        "qb_team_rank_distance": 1.5,
        "qb_room_share": 1.25,
        "qb1_over_qb2_gap_pct": 0.75,
        "rush_share_of_own_points": 1.25,
        "rush_proj_rank_pct": 1.0,
        "pass_proj_rank_pct": 1.0,
    },
    "RB": {
        "match_projection_rank_pct": 2.5,
        "year_exp_scaled": 2.0,
        "projection_x_exp": 1.0,
        "adp_rank_pct": 0.5,
        "rush_proj_rank_pct": 1.0,
        "rec_proj_rank_pct": 1.0,
        "rec_share_of_own_points": 1.0,
        "rb_rush_share_of_room": 1.25,
        "rb_rec_share_of_room": 0.75,
    },
    "WR": {
        "match_projection_rank_pct": 2.5,
        "year_exp_scaled": 2.0,
        "projection_x_exp": 1.0,
        "adp_rank_pct": 0.5,
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
    "TE": {
        "match_projection_rank_pct": 2.5,
        "year_exp_scaled": 2.0,
        "projection_x_exp": 1.0,
        "adp_rank_pct": 0.5,
        "rec_proj_rank_pct": 1.0,
        "team_rec_share": 1.25,
        "team_qb_pass_proj_rank_pct": 0.5,
    },
}


@dataclass(frozen=True)
class FrozenSource:
    year: int
    local_path: Path | None = None
    git_repo: Path | None = None
    git_commit: str | None = None
    git_path: str | None = None
    fallback_git_repo: Path | None = None
    fallback_git_commit: str | None = None
    fallback_git_path: str | None = None


FROZEN_SOURCES = {
    2022: FrozenSource(
        year=2022,
        git_repo=ROOT,
        git_commit="fea8ab4845dd0c5efb26292cb59007c865a3a003",
        git_path="Data/Databases/Simulation.sqlite3",
    ),
    2023: FrozenSource(
        year=2023,
        local_path=ROOT / "Data" / "Databases" / "DB_Versioning"
        / "Simulation__2023_08_28_52.sqlite3",
        fallback_git_repo=APP_ROOT,
        fallback_git_commit="89b0fb32fef39e57e257e1a0af35b2ff03a71593",
        fallback_git_path="app/Simulation.sqlite3",
    ),
    2024: FrozenSource(
        year=2024,
        local_path=ROOT / "Data" / "Databases" / "DB_Versioning"
        / "Simulation__2024_08_26_48.sqlite3",
        fallback_git_repo=APP_ROOT,
        fallback_git_commit="eae29b1a0d3e867028d9c428c30902d95033272f",
        fallback_git_path="app/Simulation.sqlite3",
    ),
    2025: FrozenSource(
        year=2025,
        local_path=ROOT / "Data" / "Databases" / "DB_Versioning"
        / "Simulation__2025_08_24_55.sqlite3",
        fallback_git_repo=APP_ROOT,
        fallback_git_commit="dbe4fb6e0922522a3476435e72d0054cf2f6a491",
        fallback_git_path="app/Simulation.sqlite3",
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str, binary: bool = False) -> bytes | str:
    value = subprocess.check_output(["git", *args], cwd=repo)
    return value if binary else value.decode("utf-8").strip()


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def open_frozen_source(source: FrozenSource) -> tuple[sqlite3.Connection, dict[str, Any]]:
    if source.local_path is not None and source.local_path.exists():
        conn = sqlite3.connect(f"file:{source.local_path}?mode=ro", uri=True)
        manifest = {
            "year": source.year,
            "source_type": "local_db_version",
            "path": str(source.local_path),
            "bytes": source.local_path.stat().st_size,
            "sha256": sha256_file(source.local_path),
            "last_write_time": source.local_path.stat().st_mtime,
        }
        return conn, manifest

    repo = source.git_repo or source.fallback_git_repo
    commit = source.git_commit or source.fallback_git_commit
    db_path = source.git_path or source.fallback_git_path
    if repo is None or commit is None or db_path is None:
        raise FileNotFoundError(f"No usable frozen source configured for {source.year}.")

    full_commit = str(git_output(repo, "rev-parse", commit))
    blob = git_output(repo, "show", f"{full_commit}:{db_path}", binary=True)
    assert isinstance(blob, bytes)
    conn = sqlite3.connect(":memory:")
    conn.deserialize(blob)
    blob_sha = str(git_output(repo, "rev-parse", f"{full_commit}:{db_path}"))
    commit_meta = str(
        git_output(repo, "show", "-s", "--format=%cI|%s", full_commit)
    )
    manifest = {
        "year": source.year,
        "source_type": "git_blob",
        "repo": str(repo),
        "commit": full_commit,
        "commit_meta": commit_meta,
        "path": db_path,
        "blob_sha": blob_sha,
        "bytes": len(blob),
        "sha256": sha256_bytes(blob),
    }
    return conn, manifest


def clean_display_name(value: Any) -> str:
    name = "" if value is None else str(value)
    name = name.replace("`", "'").replace("-", " ")
    name = re.sub(r"[.*+%,]", "", name)
    name = re.sub(r"\b(?:Jr|Sr|II|III)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip().title()
    aliases = {
        "Gabe Davis": "Gabriel Davis",
        "Eli Mitchell": "Elijah Mitchell",
        "Ken Walker": "Kenneth Walker",
        "Jeffery Wilson": "Jeff Wilson",
        "Josh Palmer": "Joshua Palmer",
        "Tank Dell": "Nathaniel Dell",
        "De'Von Achane": "Devon Achane",
        "Hollywood Brown": "Marquise Brown",
        "Chig Okonkwo": "Chigoziem Okonkwo",
    }
    return aliases.get(name, name)


def player_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_display_name(value).lower())


def normalize_pos(value: Any) -> str:
    return str(value).replace("Rookie_", "").upper()


def add_identity(df: pd.DataFrame, player_col: str = "player") -> pd.DataFrame:
    df = df.copy()
    df["player"] = df[player_col].map(clean_display_name)
    df["player_key"] = df[player_col].map(player_key)
    if "pos" in df:
        df["pos"] = df["pos"].map(normalize_pos)
    return df


def legacy_truncnorm_draws(
    means: np.ndarray,
    stds: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    num_draws: int,
    seed: int,
    floor: float,
) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(1e-6, 1 - 1e-6, size=(len(means), num_draws))
    draws = np.repeat(means[:, None], num_draws, axis=1)
    valid = (
        np.isfinite(means)
        & np.isfinite(stds)
        & np.isfinite(mins)
        & np.isfinite(maxs)
        & (stds > 1e-8)
        & (maxs > mins)
    )
    for idx in np.flatnonzero(valid):
        lower = (mins[idx] - means[idx]) / stds[idx]
        upper = (maxs[idx] - means[idx]) / stds[idx]
        draws[idx] = stats.truncnorm.ppf(
            uniforms[idx],
            lower,
            upper,
            loc=means[idx],
            scale=stds[idx],
        )
    return np.maximum(np.nan_to_num(draws, nan=floor), floor).astype(np.float32)


def stable_seed(*parts: Any) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.md5(text).hexdigest()[:8], 16)


def add_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df:
            df[column] = 0.0
    return df


def add_bonus_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "rush_yards_gained_sum",
        "rec_yards_gained_sum",
        "pass_yards_gained_sum",
        "rush_fumble_lost_sum",
        "rec_fumble_lost_sum",
        "pass_fumble_lost_sum",
    ]
    df = add_missing_columns(df, needed)
    df["rush_yd_100_bonus"] = (df["rush_yards_gained_sum"] >= 100).astype(float)
    df["rush_yd_200_bonus"] = (df["rush_yards_gained_sum"] >= 200).astype(float)
    df["rec_yd_100_bonus"] = (df["rec_yards_gained_sum"] >= 100).astype(float)
    df["rec_yd_200_bonus"] = (df["rec_yards_gained_sum"] >= 200).astype(float)
    df["pass_yd_300_bonus"] = (df["pass_yards_gained_sum"] >= 300).astype(float)
    df["pass_yd_400_bonus"] = (df["pass_yards_gained_sum"] >= 400).astype(float)
    df["fumble_lost"] = (
        df["rush_fumble_lost_sum"]
        + df["rec_fumble_lost_sum"]
        + df["pass_fumble_lost_sum"]
    )
    return df


def score_component(df: pd.DataFrame, scoring: dict[str, float]) -> np.ndarray:
    df = add_missing_columns(df, list(scoring))
    return df[list(scoring)].to_numpy(dtype=float) @ np.asarray(list(scoring.values()))


def load_raw_weekly(min_year: int = 2008, max_year: int = 2025) -> pd.DataFrame:
    if not DAILY_DB.exists():
        raise FileNotFoundError(f"Raw weekly database not found: {DAILY_DB}")
    pieces = []
    with sqlite3.connect(f"file:{DAILY_DB}?mode=ro", uri=True) as conn:
        for pos in POSITIONS:
            frame = pd.read_sql_query(
                f"SELECT * FROM {pos}_Stats "
                "WHERE season BETWEEN ? AND ? AND week BETWEEN 1 AND 16",
                conn,
                params=(min_year, max_year),
            )
            frame = frame[
                ~((frame.player == "Adrian Peterson") & (frame.team == "CHI"))
            ]
            frame = frame[
                ~(
                    (frame.player == "Steve Smith")
                    & frame.team.isin(["NYG", "PHI", "LAR"])
                )
            ]
            frame = frame[~((frame.player == "Mike Williams") & (frame.season < 2017))]
            frame = frame[
                ~(
                    (frame.player.str.lower() == "trey mcbride")
                    & (frame.season == 2023)
                    & (frame.week < 8)
                )
            ]
            frame = add_identity(frame)
            frame["pos"] = pos
            frame = add_bonus_columns(frame)
            fantasy = score_component(frame, get_scoring_dict("rush", league="beta"))
            if pos == "QB":
                fantasy = fantasy + score_component(
                    frame,
                    get_scoring_dict("passing", league="beta"),
                )
                frame = add_missing_columns(
                    frame,
                    ["pass_qb_dropback_sum", "rush_rush_attempt_sum"],
                )
                frame["active_row"] = (
                    frame["pass_qb_dropback_sum"]
                    + frame["rush_rush_attempt_sum"]
                    > 15
                )
            else:
                fantasy = fantasy + score_component(
                    frame,
                    get_scoring_dict("receiving", league="beta"),
                )
                frame["active_row"] = True
            frame["managed_score"] = fantasy
            frame["active_score"] = np.where(
                frame["active_row"],
                frame["managed_score"],
                0.0,
            )
            grouped = (
                frame.groupby(
                    ["player_key", "pos", "season", "week"],
                    as_index=False,
                )
                .agg(
                    player=("player", "first"),
                    managed_score=("managed_score", "sum"),
                    active_score=("active_score", "sum"),
                    active=("active_row", "max"),
                )
            )
            grouped["played"] = True
            pieces.append(grouped)
    weekly = pd.concat(pieces, ignore_index=True)
    weekly["season"] = weekly.season.astype(int)
    weekly["week"] = weekly.week.astype(int)
    weekly["active"] = weekly.active.astype(bool)
    weekly["played"] = weekly.played.astype(bool)
    return weekly


def load_feature_templates() -> pd.DataFrame:
    with sqlite3.connect(f"file:{SIM_DB}?mode=ro", uri=True) as conn:
        features = pd.read_sql_query(
            "SELECT * FROM Best_Ball_Weekly_Templates WHERE league='beta'",
            conn,
        )
    features = add_identity(features)
    features["season"] = pd.to_numeric(features.season, errors="coerce").astype(int)
    features["preseason_proj_ppg"] = pd.to_numeric(
        features.preseason_proj_ppg,
        errors="coerce",
    ).fillna(0.0)
    features = features.sort_values(
        ["season", "pos", "player_key", "preseason_proj_ppg"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["season", "pos", "player_key"])

    features["projection_rank_pct"] = (
        features.groupby(["season", "pos"])["preseason_proj_ppg"]
        .rank(method="first", pct=True, ascending=True)
        .astype(float)
    )
    features["projection_decile"] = np.ceil(
        10 * features.projection_rank_pct
    ).clip(1, 10).astype(int)
    features["match_projection_rank_pct"] = features.projection_rank_pct
    features["year_exp_scaled"] = (
        pd.to_numeric(features.year_exp, errors="coerce")
        .clip(lower=0, upper=10)
        .div(10)
        .fillna(MATCH_FILL_VALUE)
    )
    features["projection_x_exp"] = (
        features.match_projection_rank_pct * features.year_exp_scaled
    )
    for pos, weights in MATCH_FEATURE_WEIGHTS.items():
        del pos
        for feature in weights:
            if feature == "qb_team_rank_distance":
                continue
            if feature not in features:
                features[feature] = MATCH_FILL_VALUE
            features[feature] = pd.to_numeric(
                features[feature], errors="coerce"
            ).fillna(MATCH_FILL_VALUE)
    return features.reset_index(drop=True)


def load_actual_salaries() -> pd.DataFrame:
    with sqlite3.connect(f"file:{SIM_DB}?mode=ro", uri=True) as conn:
        actual = pd.read_sql_query(
            "SELECT player, actual_salary, is_keeper, year, league "
            "FROM Actual_Salaries WHERE league='beta'",
            conn,
        )
    actual = add_identity(actual)
    actual["year"] = pd.to_numeric(actual.year, errors="coerce").astype(int)
    actual["actual_salary"] = pd.to_numeric(
        actual.actual_salary, errors="coerce"
    ).fillna(1.0)
    actual["is_keeper"] = pd.to_numeric(actual.is_keeper, errors="coerce").fillna(0)
    return actual


def choose_duplicate_forecasts(
    forecast: pd.DataFrame,
    target_features: pd.DataFrame,
) -> pd.DataFrame:
    target_pos = (
        target_features[["player_key", "pos"]]
        .drop_duplicates("player_key")
        .set_index("player_key")
        .pos.to_dict()
    )
    forecast = forecast.copy()
    forecast["target_pos_match"] = forecast.apply(
        lambda row: int(target_pos.get(row.player_key) == row.pos),
        axis=1,
    )
    forecast = forecast.sort_values(
        ["player_key", "target_pos_match", "pred_fp_per_game"],
        ascending=[True, False, False],
    ).drop_duplicates("player_key")
    return forecast.drop(columns="target_pos_match").reset_index(drop=True)


def load_frozen_forecast(
    year: int,
    conn: sqlite3.Connection,
    target_features: pd.DataFrame,
    num_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    if year in (2022, 2023):
        table = f"Versionbeta_{year}"
        forecast = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        forecast = add_identity(forecast)
        forecast["pos"] = forecast.pos.map(normalize_pos)
        sample_cols = sorted(
            [column for column in forecast if str(column).isdigit()],
            key=lambda value: int(value),
        )
        if not sample_cols:
            raise ValueError(f"{table} has no empirical draw columns.")
        take = min(num_draws, len(sample_cols))
        draws = forecast[sample_cols[:take]].to_numpy(dtype=np.float32) / 16.0
        if take < num_draws:
            draws = np.tile(draws, (1, math.ceil(num_draws / take)))[:, :num_draws]
        forecast["pred_fp_per_game"] = draws.mean(axis=1)
        forecast["draw_row"] = np.arange(len(forecast))
        forecast = choose_duplicate_forecasts(forecast, target_features)
        draws = draws[forecast.draw_row.to_numpy(dtype=int)]
        source_meta = {
            "projection_table": table,
            "projection_distribution": "frozen_1000_draw_matrix_divided_by_16",
        }
    else:
        forecast = pd.read_sql_query(
            "SELECT * FROM Final_Predictions "
            "WHERE year=? AND version='beta' AND dataset='final_ensemble'",
            conn,
            params=(year,),
        )
        forecast = add_identity(forecast)
        forecast["pos"] = forecast.pos.map(normalize_pos)
        forecast = forecast[forecast.pos.isin(POSITIONS)].copy()
        forecast = choose_duplicate_forecasts(forecast, target_features)
        draws = legacy_truncnorm_draws(
            forecast.pred_fp_per_game.to_numpy(float),
            forecast.std_dev.to_numpy(float),
            forecast.min_score.to_numpy(float),
            forecast.max_score.to_numpy(float),
            num_draws=num_draws,
            seed=seed + year,
            floor=0.0,
        )
        source_meta = {
            "projection_table": "Final_Predictions",
            "projection_distribution": "frozen_legacy_truncated_normal",
        }

    forecast = forecast[forecast.pos.isin(POSITIONS)].reset_index(drop=True)
    if len(forecast) != len(draws):
        raise ValueError("Frozen forecast labels and draw matrix do not align.")
    if forecast.player_key.duplicated().any():
        raise ValueError("Frozen forecast still contains duplicate player keys.")
    source_meta.update(
        {
            "forecast_players": int(len(forecast)),
            "forecast_draws": int(draws.shape[1]),
            "position_counts": forecast.pos.value_counts().sort_index().to_dict(),
        }
    )
    return forecast, draws.astype(np.float32), source_meta


def load_target_salary_rows(
    year: int,
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, str]:
    candidates = []
    if table_exists(conn, "Salaries_Pred"):
        candidates.append("Salaries_Pred")
    if table_exists(conn, "Salaries"):
        candidates.append("Salaries")
    for table in candidates:
        salary = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE year=? AND league='betapred'",
            conn,
            params=(year,),
        )
        if len(salary):
            salary = add_identity(salary)
            salary = salary.sort_values("salary", ascending=False).drop_duplicates(
                "player_key"
            )
            return salary.reset_index(drop=True), table
    raise ValueError(f"No frozen beta salary forecast found for {year}.")


def load_prior_salary_residuals(
    year: int,
    conn: sqlite3.Connection,
    actual: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    for priority, table in enumerate(("Salaries_Pred", "Salaries")):
        if not table_exists(conn, table):
            continue
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if not {"player", "salary", "year", "league"}.issubset(columns):
            continue
        frame = pd.read_sql_query(
            f"SELECT player, salary, year, league FROM {table} "
            "WHERE year<? AND league='betapred'",
            conn,
            params=(year,),
        )
        if len(frame):
            frame = add_identity(frame)
            frame["priority"] = priority
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=["pos", "salary", "residual"])
    predicted = pd.concat(pieces, ignore_index=True)
    predicted["year"] = pd.to_numeric(predicted.year, errors="coerce").astype(int)
    predicted["salary"] = pd.to_numeric(predicted.salary, errors="coerce")
    predicted = predicted.sort_values("priority").drop_duplicates(["year", "player_key"])
    prior_actual = actual[(actual.year < year) & actual.is_keeper.eq(0)].copy()
    prior_actual = prior_actual.sort_values("actual_salary", ascending=False).drop_duplicates(
        ["year", "player_key"]
    )
    pos_map = (
        features[["season", "player_key", "pos", "preseason_proj_ppg"]]
        .sort_values("preseason_proj_ppg", ascending=False)
        .drop_duplicates(["season", "player_key"])
        .rename(columns={"season": "year"})
    )
    residuals = predicted.merge(
        prior_actual[["year", "player_key", "actual_salary"]],
        on=["year", "player_key"],
        how="inner",
    ).merge(
        pos_map[["year", "player_key", "pos"]],
        on=["year", "player_key"],
        how="left",
    )
    residuals = residuals[residuals.pos.isin(POSITIONS)].copy()
    residuals["residual"] = residuals.actual_salary - residuals.salary
    return residuals[["year", "player_key", "pos", "salary", "residual"]]


def rolling_residual_salary_draws(
    forecast: pd.DataFrame,
    residuals: pd.DataFrame,
    num_draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((len(forecast), num_draws), dtype=np.float32)
    global_residual = residuals.residual.to_numpy(dtype=float)
    for idx, row in forecast.iterrows():
        if not row.salary_source_matched:
            output[idx] = legacy_truncnorm_draws(
                np.array([2.0]),
                np.array([0.5]),
                np.array([1.0]),
                np.array([5.0]),
                num_draws,
                seed + idx,
                floor=1.0,
            )[0]
            continue
        pool = residuals[residuals.pos == row.pos].copy()
        if len(pool) < 10:
            pool = residuals.copy()
        if len(pool) == 0:
            samples = np.zeros(num_draws)
        else:
            distance = np.abs(
                np.log1p(pool.salary.to_numpy(dtype=float))
                - math.log1p(float(row.salary))
            )
            take = np.argsort(distance)[: min(60, len(pool))]
            selected = pool.iloc[take]
            selected_distance = distance[take]
            weights = np.exp(-2.0 * selected_distance)
            weights = weights / weights.sum()
            samples = rng.choice(
                selected.residual.to_numpy(dtype=float),
                size=num_draws,
                replace=True,
                p=weights,
            )
        output[idx] = np.maximum(float(row.salary) + samples, 1.0)
    return output


def build_salary_forecast(
    year: int,
    conn: sqlite3.Connection,
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    features: pd.DataFrame,
    num_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    salary_rows, salary_table = load_target_salary_rows(year, conn)
    salary_rows = salary_rows.rename(
        columns={
            "std_dev": "salary_std_dev",
            "min_score": "salary_min_score",
            "max_score": "salary_max_score",
        }
    )
    keep = [
        column
        for column in [
            "player_key",
            "salary",
            "salary_std_dev",
            "salary_min_score",
            "salary_max_score",
        ]
        if column in salary_rows
    ]
    salary_rows = salary_rows[keep].copy()
    merged = forecast.merge(salary_rows, on="player_key", how="left")
    merged["salary_source_matched"] = merged.salary.notna()
    merged["salary"] = pd.to_numeric(merged.salary, errors="coerce").fillna(2.0)

    if year == 2022:
        residuals = load_prior_salary_residuals(year, conn, actual, features)
        draws = rolling_residual_salary_draws(
            merged,
            residuals,
            num_draws=num_draws,
            seed=seed + 2200,
        )
        distribution = "prior_only_nearest_salary_residuals"
        residual_rows = int(len(residuals))
    else:
        for column, default in (
            ("salary_std_dev", 0.5),
            ("salary_min_score", 1.0),
            ("salary_max_score", 5.0),
        ):
            if column not in merged:
                merged[column] = default
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        missing = ~merged.salary_source_matched
        merged.loc[missing, "salary_std_dev"] = 0.5
        merged.loc[missing, "salary_min_score"] = 1.0
        merged.loc[missing, "salary_max_score"] = 5.0
        draws = legacy_truncnorm_draws(
            merged.salary.to_numpy(float),
            merged.salary_std_dev.to_numpy(float),
            merged.salary_min_score.to_numpy(float),
            merged.salary_max_score.to_numpy(float),
            num_draws=num_draws,
            seed=seed + 2200 + year,
            floor=1.0,
        )
        distribution = "frozen_legacy_truncated_normal"
        residual_rows = 0

    draws = np.rint(draws).clip(1).astype(np.float32)
    metadata = {
        "salary_table": salary_table,
        "salary_rows": int(len(salary_rows)),
        "salary_matches_to_forecast": int(merged.salary_source_matched.sum()),
        "salary_missing_fallback": int((~merged.salary_source_matched).sum()),
        "salary_distribution": distribution,
        "prior_salary_residual_rows": residual_rows,
    }
    return merged.reset_index(drop=True), draws, metadata


def raw_week_matrices(
    labels: pd.DataFrame,
    weekly: pd.DataFrame,
    year_column: str = "season",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return managed score, played, active-score matrices aligned to labels."""
    scores = np.zeros((len(labels), len(WEEKS)), dtype=np.float32)
    played = np.zeros((len(labels), len(WEEKS)), dtype=np.int8)
    active_scores = np.zeros((len(labels), len(WEEKS)), dtype=np.float32)
    match_kind = []
    exact = {
        key: group.set_index("week")
        for key, group in weekly.groupby(["season", "pos", "player_key"], sort=False)
    }
    by_key: dict[tuple[int, str], list[pd.DataFrame]] = {}
    for (season, _, key), group in weekly.groupby(
        ["season", "pos", "player_key"], sort=False
    ):
        by_key.setdefault((int(season), key), []).append(group.set_index("week"))

    for idx, row in labels.reset_index(drop=True).iterrows():
        season = int(row[year_column])
        group = exact.get((season, row.pos, row.player_key))
        kind = "exact_pos"
        if group is None:
            alternatives = by_key.get((season, row.player_key), [])
            if len(alternatives) == 1:
                group = alternatives[0]
                kind = "key_only_pos_fallback"
            else:
                kind = "no_raw_rows"
        if group is not None:
            week_idx = group.index.to_numpy(dtype=int) - 1
            scores[idx, week_idx] = group.managed_score.to_numpy(dtype=np.float32)
            played[idx, week_idx] = 1
            active_scores[idx, week_idx] = group.active_score.to_numpy(dtype=np.float32)
        match_kind.append(kind)
    return scores, played, active_scores, match_kind


def build_target_feature_rows(
    year: int,
    forecast: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    target_features = features[features.season == year].copy()
    target_features = target_features.sort_values(
        "preseason_proj_ppg", ascending=False
    ).drop_duplicates(["player_key", "pos"])
    target = forecast[["player", "player_key", "pos", "pred_fp_per_game"]].merge(
        target_features.drop(columns=["player"], errors="ignore"),
        on=["player_key", "pos"],
        how="left",
        suffixes=("", "_feature"),
    )
    target["match_projection_rank_pct"] = (
        target.groupby("pos")["pred_fp_per_game"]
        .rank(method="first", pct=True, ascending=True)
        .astype(float)
    )
    target["projection_decile"] = np.ceil(
        target.match_projection_rank_pct * 10
    ).clip(1, 10).astype(int)
    if "year_exp" not in target:
        target["year_exp"] = np.nan
    target["year_exp_scaled"] = (
        pd.to_numeric(target.year_exp, errors="coerce")
        .clip(lower=0, upper=10)
        .div(10)
        .fillna(MATCH_FILL_VALUE)
    )
    target["projection_x_exp"] = (
        target.match_projection_rank_pct * target.year_exp_scaled
    )
    if "qb_team_rank_bucket" not in target:
        target["qb_team_rank_bucket"] = np.where(
            target.pos.eq("QB"), "unknown", "non_qb"
        )
    qb_bucket_fallback = pd.Series(
        np.where(target.pos.eq("QB"), "unknown", "non_qb"),
        index=target.index,
    )
    target["qb_team_rank_bucket"] = target.qb_team_rank_bucket.where(
        target.qb_team_rank_bucket.notna(),
        qb_bucket_fallback,
    )
    for weights in MATCH_FEATURE_WEIGHTS.values():
        for feature in weights:
            if feature == "qb_team_rank_distance":
                continue
            if feature not in target:
                target[feature] = MATCH_FILL_VALUE
            target[feature] = pd.to_numeric(
                target[feature], errors="coerce"
            ).fillna(MATCH_FILL_VALUE)
    return target.reset_index(drop=True)


def build_template_cache(
    year: int,
    forecast: pd.DataFrame,
    features: pd.DataFrame,
    raw_weekly: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    donors = features[features.season < year].copy().reset_index(drop=True)
    donor_scores, donor_played, active_scores, donor_match = raw_week_matrices(
        donors,
        raw_weekly,
    )
    active_games = (active_scores != 0).sum(axis=1)
    # A played active week can score exactly zero, so derive active counts from
    # the raw active flag rather than from point values.
    raw_active = raw_weekly[raw_weekly.active].copy()
    active_count_map = (
        raw_active.groupby(["season", "pos", "player_key"]).week.nunique().to_dict()
    )
    active_games = np.array(
        [
            active_count_map.get((int(row.season), row.pos, row.player_key), 0)
            for row in donors.itertuples()
        ],
        dtype=int,
    )
    active_points = active_scores.sum(axis=1)
    active_ppg = np.divide(
        active_points,
        active_games,
        out=np.zeros(len(donors), dtype=float),
        where=active_games > 0,
    )
    donors["active_games_safe"] = active_games
    donors["active_ppg_safe"] = active_ppg
    donors["active_ppg_resid_safe"] = (
        donors.active_ppg_safe - donors.preseason_proj_ppg
    )
    denominator = np.where(
        active_ppg > 0,
        active_ppg,
        donors.preseason_proj_ppg.to_numpy(dtype=float),
    )
    donor_profiles = np.divide(
        donor_scores,
        denominator[:, None],
        out=np.zeros_like(donor_scores),
        where=denominator[:, None] > 0,
    ).astype(np.float32)
    donors["profile_row"] = np.arange(len(donors))
    donors["raw_match_kind"] = donor_match

    targets = build_target_feature_rows(year, forecast, features)
    cache_profiles: dict[str, np.ndarray] = {}
    cache_probs: dict[str, np.ndarray] = {}
    cache_residuals: dict[str, np.ndarray] = {}
    cache_played: dict[str, np.ndarray] = {}
    audit_rows = []

    for target in targets.itertuples(index=False):
        eligible = donors[donors.pos == target.pos].copy()
        if target.pos != "QB":
            eligible = eligible[eligible.active_games_safe > 0].copy()
        if len(eligible) == 0:
            raise ValueError(f"No causal template donors for {target.player} ({target.pos}).")

        target_qb_bucket = getattr(target, "qb_team_rank_bucket", "unknown")
        target_qb_rank = QB_RANK_DISTANCE_ORDER.get(str(target_qb_bucket), 2)
        if target.pos == "QB":
            qb_distance = (
                eligible.qb_team_rank_bucket.fillna("unknown")
                .map(QB_RANK_DISTANCE_ORDER)
                .fillna(2)
                .sub(target_qb_rank)
                .abs()
                .to_numpy(dtype=float)
            )
        else:
            qb_distance = np.zeros(len(eligible), dtype=float)

        distance = np.zeros(len(eligible), dtype=float)
        for feature, weight in MATCH_FEATURE_WEIGHTS[target.pos].items():
            if feature == "qb_team_rank_distance":
                feature_distance = qb_distance
            else:
                target_value = float(getattr(target, feature, MATCH_FILL_VALUE))
                feature_values = pd.to_numeric(
                    eligible[feature], errors="coerce"
                ).fillna(MATCH_FILL_VALUE).to_numpy(dtype=float)
                feature_distance = np.abs(feature_values - target_value)
            distance += float(weight) * feature_distance

        eligible["template_distance_safe"] = distance
        eligible["tie_break"] = [
            stable_seed(year, target.player_key, row.season, row.player_key)
            for row in eligible.itertuples()
        ]
        selected = eligible.sort_values(
            ["template_distance_safe", "tie_break"]
        ).head(MAX_TEMPLATE_POOL_SIZE)
        selected_distance = selected.template_distance_safe.to_numpy(dtype=float)
        distance_range = float(selected_distance.max() - selected_distance.min())
        if distance_range > 0:
            weights = np.exp(
                -math.log(TEMPLATE_TOP_TO_BOTTOM_RATIO)
                * (selected_distance - selected_distance.min())
                / distance_range
            )
        else:
            weights = np.ones(len(selected), dtype=float)
        probs = weights / weights.sum()
        rows = selected.profile_row.to_numpy(dtype=int)
        cache_profiles[target.player] = donor_profiles[rows]
        cache_probs[target.player] = np.cumsum(probs)
        cache_probs[target.player][-1] = 1.0
        cache_residuals[target.player] = selected.active_ppg_resid_safe.to_numpy(
            dtype=np.float32
        )
        cache_played[target.player] = donor_played[rows].astype(np.int8)
        audit_rows.append(
            {
                "year": year,
                "player": target.player,
                "player_key": target.player_key,
                "pos": target.pos,
                "target_feature_match": bool(
                    pd.notna(getattr(target, "season", np.nan))
                ),
                "pool_size": int(len(selected)),
                "min_donor_season": int(selected.season.min()),
                "max_donor_season": int(selected.season.max()),
                "zero_active_share": float(selected.active_games_safe.eq(0).mean()),
                "mean_distance": float(selected_distance.mean()),
            }
        )

    cache = {
        "week_cols": WEEK_COLS,
        "profiles": cache_profiles,
        "cum_probs": cache_probs,
        "active_residuals": cache_residuals,
        "played": cache_played,
    }
    return cache, pd.DataFrame(audit_rows)


def make_simulation(
    year: int,
    player_data: pd.DataFrame,
    cache: dict[str, Any],
) -> FootballSimulation:
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.set_year = year
    sim.pos_require_start = dict(LINEUP_REQUIRE)
    sim.pred_vers = "rolling_replay"
    sim.league = "beta"
    sim.conn = None
    sim.salary_cap = SALARY_CAP
    sim.sal_pred_actual = "pred"
    sim.weekly_template_profiles = None
    sim.weekly_template_played_masks = None
    sim.weekly_template_week_cols = None
    sim.weekly_template_cum_probs = None
    sim.weekly_template_active_ppg_resids = None
    sim.weekly_template_centered_active_ppg_resids = None
    sim.weekly_template_active_ppg_resid_sds = None
    sim.player_data = player_data.copy()
    sim.set_weekly_template_profile_cache(
        cache["week_cols"],
        cache["profiles"],
        cache["cum_probs"],
        cache["active_residuals"],
        weekly_template_played_masks=cache["played"],
    )
    return sim


def build_predictions(
    forecast: pd.DataFrame,
    ppg_draws: np.ndarray,
) -> pd.DataFrame:
    labels = forecast[["player", "player_key", "pos", "salary"]].copy()
    labels = labels.drop(columns="player_key")
    samples = pd.DataFrame(
        ppg_draws,
        columns=[f"draw_{idx}" for idx in range(ppg_draws.shape[1])],
    )
    return pd.concat([labels.reset_index(drop=True), samples], axis=1)


def build_waiver_pool(
    season: int,
    raw_weekly: pd.DataFrame,
    features: pd.DataFrame,
    recorded_auction_keys: set[str],
) -> dict[str, Any]:
    season_raw = raw_weekly[raw_weekly.season == season].copy()
    labels = (
        season_raw[["player_key", "player", "pos"]]
        .sort_values(["pos", "player_key"])
        .drop_duplicates(["player_key", "pos"])
    )
    labels = labels[~labels.player_key.isin(recorded_auction_keys)].reset_index(drop=True)
    labels["season"] = season
    scores, played, _, match_kind = raw_week_matrices(labels, season_raw)
    feature_ppg = (
        features[features.season == season][
            ["player_key", "pos", "preseason_proj_ppg"]
        ]
        .sort_values("preseason_proj_ppg", ascending=False)
        .drop_duplicates(["player_key", "pos"])
    )
    labels = labels.merge(feature_ppg, on=["player_key", "pos"], how="left")
    labels["preseason_proj_ppg"] = labels.preseason_proj_ppg.fillna(0.0)
    decisions = FootballSimulation.build_managed_decision_scores(
        scores,
        preseason_ppg=labels.preseason_proj_ppg.to_numpy(dtype=float),
        played_mask=played,
    )
    labels["raw_match_kind"] = match_kind
    rankings: dict[tuple[str, int], np.ndarray] = {}
    for pos in POSITIONS:
        pos_idx = np.flatnonzero(labels.pos.to_numpy() == pos)
        for week_idx in range(len(WEEKS)):
            available = pos_idx[played[pos_idx, week_idx] > 0]
            rankings[(pos, week_idx)] = available[
                np.argsort(decisions[available, week_idx])[::-1]
            ]
    return {
        "labels": labels,
        "scores": scores,
        "played": played,
        "decisions": decisions,
        "rankings": rankings,
    }


def dynamic_waiver_slots(
    waiver_pool: dict[str, Any],
    excluded_keys: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    excluded_keys = excluded_keys or set()
    labels = waiver_pool["labels"]
    source_scores = waiver_pool["scores"]
    source_decisions = waiver_pool["decisions"]
    slot_scores = []
    slot_decisions = []
    slot_played = []
    slot_positions = []
    slot_names = []
    keys = labels.player_key.to_numpy()
    for pos in POSITIONS:
        count = int(WAIVER_PLAYER_COUNTS[pos])
        scores = np.zeros((count, len(WEEKS)), dtype=np.float32)
        decisions = np.zeros((count, len(WEEKS)), dtype=np.float32)
        played = np.zeros((count, len(WEEKS)), dtype=np.int8)
        for week_idx in range(len(WEEKS)):
            ranked = waiver_pool["rankings"][(pos, week_idx)]
            chosen = [idx for idx in ranked if keys[idx] not in excluded_keys][:count]
            for slot_idx, source_idx in enumerate(chosen):
                scores[slot_idx, week_idx] = source_scores[source_idx, week_idx]
                decisions[slot_idx, week_idx] = source_decisions[source_idx, week_idx]
                played[slot_idx, week_idx] = 1
        slot_scores.append(scores)
        slot_decisions.append(decisions)
        slot_played.append(played)
        slot_positions.extend([pos] * count)
        slot_names.extend([f"WW_ACTUAL_{pos}_{idx + 1}" for idx in range(count)])
    return (
        np.vstack(slot_scores),
        np.vstack(slot_decisions),
        np.vstack(slot_played),
        np.asarray(slot_positions, dtype=object),
        np.asarray(slot_names, dtype=object),
    )


def realized_rostered_keys(actual: pd.DataFrame, season: int) -> set[str]:
    season_actual = actual[actual.year == season].copy()
    season_actual = season_actual.sort_values(
        ["actual_salary", "player_key"],
        ascending=[False, True],
    ).drop_duplicates("player_key")
    keepers = season_actual[season_actual.is_keeper.ne(0)]
    nonkeepers = season_actual[season_actual.is_keeper.eq(0)]
    open_slots = max(TOTAL_MARKET_SLOTS - len(keepers), 0)
    return set(keepers.player_key) | set(nonkeepers.head(open_slots).player_key)


def empirical_waiver_baselines(
    origin_year: int,
    raw_weekly: pd.DataFrame,
    features: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    detail = []
    all_scores: dict[str, list[float]] = {pos: [] for pos in POSITIONS}
    for season in range(2019, origin_year):
        rostered = realized_rostered_keys(actual, season)
        pool = build_waiver_pool(season, raw_weekly, features, rostered)
        slot_scores, _, slot_played, slot_pos, _ = dynamic_waiver_slots(pool)
        for pos in POSITIONS:
            mask = slot_pos == pos
            values = slot_scores[mask].reshape(-1)
            # There is one value per supported replacement slot-week. Missing
            # options remain zero, matching a failed waiver replacement.
            all_scores[pos].extend(values.tolist())
            detail.append(
                {
                    "origin_year": origin_year,
                    "source": "prior_empirical",
                    "source_season": season,
                    "pos": pos,
                    "baseline": float(values.mean()) if len(values) else 0.0,
                    "played_slot_share": float(slot_played[mask].mean()) if mask.any() else 0.0,
                    "observations": int(len(values)),
                }
            )
    baselines = {
        pos: float(np.round(np.mean(values), 1)) if values else 0.0
        for pos, values in all_scores.items()
    }
    return baselines, pd.DataFrame(detail)


def build_actual_environment(
    year: int,
    forecast: pd.DataFrame,
    raw_weekly: pd.DataFrame,
    features: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target_actual = actual[actual.year == year].copy()
    target_actual = target_actual.sort_values(
        "actual_salary", ascending=False
    ).drop_duplicates("player_key")
    actual_salary_map = target_actual.set_index("player_key").actual_salary.to_dict()
    keeper_keys = set(target_actual.loc[target_actual.is_keeper.ne(0), "player_key"])
    recorded_keys = set(target_actual.player_key)
    rostered_keys = realized_rostered_keys(actual, year)

    labels = forecast[["player", "player_key", "pos", "pred_fp_per_game"]].copy()
    labels["season"] = year
    scores, played, _, raw_match = raw_week_matrices(
        labels,
        raw_weekly[raw_weekly.season == year],
    )
    decisions = FootballSimulation.build_managed_decision_scores(
        scores,
        preseason_ppg=labels.pred_fp_per_game.to_numpy(dtype=float),
        played_mask=played,
    )
    actual_cost = np.array(
        [actual_salary_map.get(key, 1.0) for key in labels.player_key],
        dtype=float,
    )
    labels["actual_salary"] = actual_cost
    labels["actual_salary_matched"] = labels.player_key.isin(recorded_keys)
    labels["is_keeper"] = labels.player_key.isin(keeper_keys)
    labels["raw_match_kind"] = raw_match
    waiver_pool = build_waiver_pool(year, raw_weekly, features, rostered_keys)
    slot_scores, _, _, slot_pos, _ = dynamic_waiver_slots(waiver_pool)
    realized_baseline = {
        pos: float(np.round(slot_scores[slot_pos == pos].mean(), 1))
        for pos in POSITIONS
    }
    environment = {
        "year": year,
        "labels": labels,
        "scores": scores,
        "played": played,
        "decisions": decisions,
        "actual_cost": actual_cost,
        "keeper_keys": keeper_keys,
        "recorded_result_keys": recorded_keys,
        "realized_rostered_keys": rostered_keys,
        "keeper_count": int(target_actual.is_keeper.ne(0).sum()),
        "keeper_spend": float(
            target_actual.loc[target_actual.is_keeper.ne(0), "actual_salary"].sum()
        ),
        "waiver_pool": waiver_pool,
        "realized_waiver_baseline": realized_baseline,
        "score_cache": {},
    }
    return environment, labels


def score_actual_roster(
    environment: dict[str, Any],
    roster_players: tuple[str, ...],
) -> dict[str, Any]:
    cache_key = tuple(sorted(roster_players))
    if cache_key in environment["score_cache"]:
        return environment["score_cache"][cache_key]
    labels = environment["labels"]
    mask = labels.player.isin(roster_players).to_numpy()
    if int(mask.sum()) != len(roster_players):
        missing = sorted(set(roster_players) - set(labels.loc[mask, "player"]))
        raise ValueError(f"Actual outcome labels missing roster players: {missing}")
    roster_scores = environment["scores"][mask]
    roster_decisions = environment["decisions"][mask]
    roster_played = environment["played"][mask]
    roster_pos = labels.loc[mask, "pos"].to_numpy()
    roster_names = labels.loc[mask, "player"].to_numpy()
    roster_keys = set(labels.loc[mask, "player_key"])

    drafted_weekly = FootballSimulation.managed_lineup_weekly_scores(
        roster_scores,
        roster_pos,
        decision_scores=roster_decisions,
        player_names=roster_names,
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=ZERO_WAIVERS,
        played_mask=roster_played,
    )
    waiver_scores, waiver_decisions, waiver_played, waiver_pos, waiver_names = (
        dynamic_waiver_slots(environment["waiver_pool"], roster_keys)
    )
    weekly, details = FootballSimulation.managed_lineup_weekly_scores(
        np.vstack([roster_scores, waiver_scores]),
        np.concatenate([roster_pos, waiver_pos]),
        decision_scores=np.vstack([roster_decisions, waiver_decisions]),
        player_names=np.concatenate([roster_names, waiver_names]),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=ZERO_WAIVERS,
        played_mask=np.vstack([roster_played, waiver_played]),
        return_details=True,
    )
    result = {
        "actual_points": float(weekly.sum()),
        "drafted_only_points": float(np.asarray(drafted_weekly).sum()),
        "actual_waiver_starts": int(details.waiver_starts.sum()),
        "actual_salary_spend": float(environment["actual_cost"][mask].sum()),
        "actual_salary_missing_players": int(
            (~labels.loc[mask, "actual_salary_matched"]).sum()
        ),
        "raw_outcome_missing_players": int(
            labels.loc[mask, "raw_match_kind"].eq("no_raw_rows").sum()
        ),
    }
    environment["score_cache"][cache_key] = result
    return result


def generate_construction_contexts(
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    num_contexts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds = np.random.SeedSequence(seed).generate_state(
        num_contexts,
        dtype=np.uint32,
    )
    weekly, played = sim.sample_seeded_template_weekly_contexts(
        predictions,
        [int(value) for value in seeds],
        num_weeks=16,
        return_played_masks=True,
    )
    preseason = predictions[sim.sample_value_columns(predictions)].mean(axis=1)
    decisions = np.stack(
        [
            sim.build_managed_decision_scores(
                weekly[idx],
                preseason_ppg=preseason,
                played_mask=played[idx],
            )
            for idx in range(num_contexts)
        ]
    )
    return weekly, decisions, played


def managed_value_banks(
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    predictions: pd.DataFrame,
    waiver_options: dict[str, dict[str, float]],
) -> dict[tuple[str, float], np.ndarray]:
    banks: dict[tuple[str, float], np.ndarray] = {}
    for waiver_name, waiver_baseline in waiver_options.items():
        for bench_weight in (0.0, 0.25):
            values = []
            for idx in range(len(weekly)):
                values.append(
                    FootballSimulation.managed_marginal_values(
                        weekly[idx],
                        predictions.pos.to_numpy(),
                        decisions[idx],
                        predictions.player.to_numpy(),
                        base_players=[],
                        waiver_baselines=waiver_baseline,
                        lineup_require=LINEUP_REQUIRE,
                        bench_upside_weight=bench_weight,
                        played_mask=played[idx],
                    )
                )
            banks[(waiver_name, bench_weight)] = np.column_stack(values).astype(
                np.float32
            )
    return banks


def normalize_market_draws(
    sim: FootballSimulation,
    raw_draws: np.ndarray,
    remaining_budget: float,
    remaining_slots: int,
) -> np.ndarray:
    normalized = np.empty_like(raw_draws, dtype=np.float32)
    available = np.ones(raw_draws.shape[0], dtype=bool)
    for draw_idx in range(raw_draws.shape[1]):
        normalized[:, draw_idx] = sim.normalize_salary_market_values(
            raw_draws[:, draw_idx],
            available,
            remaining_market_budget=remaining_budget,
            remaining_market_slots=remaining_slots,
        )
    return normalized


def crps_from_samples(samples: np.ndarray, observed: float) -> float:
    values = np.sort(np.asarray(samples, dtype=float))
    n = len(values)
    if n == 0:
        return float("nan")
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    pairwise_mean = 2.0 * np.sum(coefficients * values) / (n * n)
    return float(np.mean(np.abs(values - observed)) - 0.5 * pairwise_mean)


def weighted_interval_score(samples: np.ndarray, observed: float) -> float:
    samples = np.asarray(samples, dtype=float)
    median = float(np.median(samples))
    numerator = 0.5 * abs(median - observed)
    alphas = (0.5, 0.2, 0.1)
    for alpha in alphas:
        lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
        interval_score = upper - lower
        if observed < lower:
            interval_score += (2 / alpha) * (lower - observed)
        elif observed > upper:
            interval_score += (2 / alpha) * (observed - upper)
        weight = alpha / 2
        numerator += weight * interval_score
    return float(numerator / (len(alphas) + 0.5))


def salary_calibration(
    year: int,
    sim: FootballSimulation,
    forecast: pd.DataFrame,
    salary_draws: np.ndarray,
    environment: dict[str, Any],
    num_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = environment["labels"]
    keeper_mask = labels.is_keeper.to_numpy(dtype=bool)
    candidate_idx = np.flatnonzero(~keeper_mask)
    candidate = forecast.iloc[candidate_idx].reset_index(drop=True)
    candidate_draws = salary_draws[candidate_idx]
    actual_cost = environment["actual_cost"][candidate_idx]
    actual_matched = labels.actual_salary_matched.to_numpy(dtype=bool)[candidate_idx]
    remaining_budget = TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = TOTAL_MARKET_SLOTS - environment["keeper_count"]

    rng = np.random.default_rng(seed + year * 13)
    plan = rng.integers(
        0,
        candidate_draws.shape[1],
        size=(num_samples, 5),
    )
    metric_rows = []
    player_rows = []
    for draw_count in (1, 5):
        raw_market = np.stack(
            [candidate_draws[:, row[:draw_count]].mean(axis=1) for row in plan],
            axis=1,
        )
        market = normalize_market_draws(
            sim,
            raw_market,
            remaining_budget,
            remaining_slots,
        )
        matched_idx = np.flatnonzero(actual_matched)
        per_player = []
        for idx in matched_idx:
            samples = market[idx]
            observed = float(actual_cost[idx])
            quantiles = np.quantile(samples, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
            pit = float(np.mean(samples <= observed))
            row = {
                "year": year,
                "salary_draw_count": draw_count,
                "player": candidate.player.iloc[idx],
                "pos": candidate.pos.iloc[idx],
                "salary_source_matched": bool(candidate.salary_source_matched.iloc[idx]),
                "actual_salary": observed,
                "forecast_mean": float(samples.mean()),
                "forecast_sd": float(samples.std()),
                "p05": float(quantiles[0]),
                "p10": float(quantiles[1]),
                "p25": float(quantiles[2]),
                "p50": float(quantiles[3]),
                "p75": float(quantiles[4]),
                "p90": float(quantiles[5]),
                "p95": float(quantiles[6]),
                "pit": pit,
                "crps": crps_from_samples(samples, observed),
                "wis": weighted_interval_score(samples, observed),
            }
            row["covered_50"] = row["p25"] <= observed <= row["p75"]
            row["covered_80"] = row["p10"] <= observed <= row["p90"]
            row["covered_90"] = row["p05"] <= observed <= row["p95"]
            per_player.append(row)
        player_frame = pd.DataFrame(per_player)
        player_rows.extend(per_player)
        actual_top = set(
            player_frame.nlargest(min(TOP_N, len(player_frame)), "actual_salary").player
        )
        forecast_top = set(
            player_frame.nlargest(min(TOP_N, len(player_frame)), "forecast_mean").player
        )
        error = player_frame.forecast_mean - player_frame.actual_salary
        metric_rows.append(
            {
                "year": year,
                "salary_draw_count": draw_count,
                "players": int(len(player_frame)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(error.mean()),
                "spearman": float(
                    player_frame.forecast_mean.corr(
                        player_frame.actual_salary,
                        method="spearman",
                    )
                ),
                "top12_recall": float(len(actual_top & forecast_top) / max(len(actual_top), 1)),
                "coverage_50": float(player_frame.covered_50.mean()),
                "coverage_80": float(player_frame.covered_80.mean()),
                "coverage_90": float(player_frame.covered_90.mean()),
                "mean_width_50": float((player_frame.p75 - player_frame.p25).mean()),
                "mean_width_80": float((player_frame.p90 - player_frame.p10).mean()),
                "mean_width_90": float((player_frame.p95 - player_frame.p05).mean()),
                "mean_forecast_sd": float(player_frame.forecast_sd.mean()),
                "mean_crps": float(player_frame.crps.mean()),
                "mean_wis": float(player_frame.wis.mean()),
                "pit_mean": float(player_frame.pit.mean()),
                "pit_extreme_share": float(
                    ((player_frame.pit < 0.1) | (player_frame.pit > 0.9)).mean()
                ),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(player_rows)


def forecast_roster_ev(
    roster_players: tuple[str, ...],
    waiver_name: str,
    waiver_baseline: dict[str, float],
    predictions: pd.DataFrame,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    cache: dict[tuple[tuple[str, ...], str], float],
) -> float:
    key = (tuple(sorted(roster_players)), waiver_name)
    if key in cache:
        return cache[key]
    mask = predictions.player.isin(roster_players).to_numpy()
    scores, _ = FootballSimulation.managed_lineup_multi_context_scores(
        weekly[:, mask, :],
        predictions.loc[mask, "pos"].to_numpy(),
        decisions[:, mask, :],
        predictions.loc[mask, "player"].to_numpy(),
        lineup_require=LINEUP_REQUIRE,
        waiver_baselines=waiver_baseline,
        played_mask=played[:, mask, :],
    )
    value = float(np.mean(scores))
    cache[key] = value
    return value


def solve_actual_oracle(
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    environment: dict[str, Any],
    candidate_full_idx: np.ndarray,
) -> dict[str, Any]:
    actual_predictions = predictions.copy()
    actual_cost = environment["actual_cost"][candidate_full_idx]
    actual_predictions["salary"] = actual_cost
    weekly = environment["scores"][candidate_full_idx]
    decisions = environment["decisions"][candidate_full_idx]
    played = environment["played"][candidate_full_idx]
    baseline = environment["realized_waiver_baseline"]
    values = FootballSimulation.managed_marginal_values(
        weekly,
        actual_predictions.pos.to_numpy(),
        decisions,
        actual_predictions.player.to_numpy(),
        base_players=[],
        waiver_baselines=baseline,
        lineup_require=LINEUP_REQUIRE,
        bench_upside_weight=0.0,
        played_mask=played,
    )
    static = sim.build_managed_ilp_static_matrices(
        actual_predictions,
        {},
        [],
        [],
        ROSTER_SIZE,
        POS_MIN,
        POS_MAX,
        enforce_top_n=False,
    )
    result = sim._solve_managed_scenario(
        actual_predictions,
        values,
        weekly,
        decisions,
        static,
        [],
        {},
        [],
        ROSTER_SIZE,
        POS_MIN,
        POS_MAX,
        baseline,
        LINEUP_REQUIRE,
        False,
        refine_roster=True,
        score_roster=False,
        salary_values=actual_cost,
        played_mask=played,
    )
    if result is None:
        raise RuntimeError("Actual-price hindsight roster was infeasible.")
    roster = tuple(sorted(result["selected_players"]))
    score = score_actual_roster(environment, roster)
    return {
        "selected_players": roster,
        **score,
        "approximation": "actual marginal objective plus current one-swap refinement",
    }


def run_variant_trials(
    year: int,
    sim: FootballSimulation,
    predictions: pd.DataFrame,
    salary_draws: np.ndarray,
    environment: dict[str, Any],
    candidate_full_idx: np.ndarray,
    weekly: np.ndarray,
    decisions: np.ndarray,
    played: np.ndarray,
    evaluation_weekly: np.ndarray,
    evaluation_decisions: np.ndarray,
    evaluation_played: np.ndarray,
    value_banks: dict[tuple[str, float], np.ndarray],
    waiver_options: dict[str, dict[str, float]],
    trials: int,
    context_draws: int,
    seed: int,
    refine_roster: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    remaining_budget = TOTAL_MARKET_BUDGET - environment["keeper_spend"]
    remaining_slots = TOTAL_MARKET_SLOTS - environment["keeper_count"]
    top_n = (
        predictions.nlargest(min(TOP_N, len(predictions)), "salary").player.tolist()
    )
    static = {
        enforce: sim.build_managed_ilp_static_matrices(
            predictions,
            {},
            [],
            top_n,
            ROSTER_SIZE,
            POS_MIN,
            POS_MAX,
            enforce_top_n=enforce,
        )
        for enforce in (False, True)
    }
    ref_weekly = weekly.mean(axis=0)
    ref_decisions = decisions.mean(axis=0)
    ref_played = np.where(
        np.any(played >= 0, axis=0),
        np.any(played > 0, axis=0).astype(np.int8),
        -1,
    ).astype(np.int8)
    rng = np.random.default_rng(seed + year * 101)
    salary_plan = rng.integers(0, salary_draws.shape[1], size=(trials, 5))
    context_plan = rng.integers(0, weekly.shape[0], size=(trials, context_draws))
    markets: dict[int, np.ndarray] = {}
    for count in (1, 5):
        raw = np.column_stack(
            [salary_draws[:, row[:count]].mean(axis=1) for row in salary_plan]
        )
        markets[count] = normalize_market_draws(
            sim,
            raw,
            remaining_budget,
            remaining_slots,
        )

    forecast_cache: dict[tuple[tuple[str, ...], str], float] = {}
    rows = []
    variants = list(product((1, 5), (False, True), waiver_options, (0.0, 0.25)))
    start = time.perf_counter()
    for trial in range(trials):
        context_idx = context_plan[trial]
        for salary_count, enforce_top, waiver_name, bench_weight in variants:
            waiver = waiver_options[waiver_name]
            values = value_banks[(waiver_name, bench_weight)][:, context_idx].mean(axis=1)
            market = markets[salary_count][:, trial]
            predictions["salary"] = market
            solved = sim._solve_managed_scenario(
                predictions,
                values,
                ref_weekly,
                ref_decisions,
                static[enforce_top],
                [],
                {},
                top_n,
                ROSTER_SIZE,
                POS_MIN,
                POS_MAX,
                waiver,
                LINEUP_REQUIRE,
                enforce_top,
                refine_roster=refine_roster,
                score_roster=False,
                salary_values=market,
                played_mask=ref_played,
            )
            if solved is None:
                rows.append(
                    {
                        "year": year,
                        "trial": trial,
                        "salary_draw_count": salary_count,
                        "enforce_top_n": enforce_top,
                        "waiver_source": waiver_name,
                        "bench_upside_weight": bench_weight,
                        "solve_status": "infeasible",
                    }
                )
                continue
            roster = tuple(sorted(solved["selected_players"]))
            selected = predictions.player.isin(roster).to_numpy()
            actual_score = score_actual_roster(environment, roster)
            forecast_ev = forecast_roster_ev(
                roster,
                waiver_name,
                waiver,
                predictions,
                evaluation_weekly,
                evaluation_decisions,
                evaluation_played,
                forecast_cache,
            )
            pos_counts = predictions.loc[selected, "pos"].value_counts().to_dict()
            actual_feasible = actual_score["actual_salary_spend"] <= SALARY_CAP + 1e-8
            rows.append(
                {
                    "year": year,
                    "trial": trial,
                    "salary_draw_count": salary_count,
                    "enforce_top_n": enforce_top,
                    "waiver_source": waiver_name,
                    "bench_upside_weight": bench_weight,
                    "solve_status": "optimal",
                    "variant": (
                        f"d{salary_count}_top{int(enforce_top)}_"
                        f"waiver{waiver_name}_bench{int(bench_weight * 100):02d}"
                    ),
                    "roster": "|".join(roster),
                    "forecast_salary_spend": float(market[selected].sum()),
                    "actual_cap_feasible": bool(actual_feasible),
                    "actual_cap_overage": float(
                        max(actual_score["actual_salary_spend"] - SALARY_CAP, 0.0)
                    ),
                    "forecast_ev": forecast_ev,
                    "forecast_error": actual_score["actual_points"] - forecast_ev,
                    "contains_top_n": bool(set(roster) & set(top_n)),
                    "qb_count": int(pos_counts.get("QB", 0)),
                    "rb_count": int(pos_counts.get("RB", 0)),
                    "wr_count": int(pos_counts.get("WR", 0)),
                    "te_count": int(pos_counts.get("TE", 0)),
                    **actual_score,
                }
            )
        if (trial + 1) % max(1, min(25, trials)) == 0:
            elapsed = time.perf_counter() - start
            print(
                f"{year}: completed {trial + 1}/{trials} paired trials "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    return frame, {
        "top_n_players": top_n,
        "remaining_market_budget": remaining_budget,
        "remaining_market_slots": remaining_slots,
        "variant_runtime_seconds": time.perf_counter() - start,
    }


FACTOR_COMPARISONS = {
    "salary_draws_1_minus_5": ("salary_draw_count", 5, 1),
    "top_n_off_minus_on": ("enforce_top_n", True, False),
    "prior_waiver_minus_projected": (
        "waiver_source",
        "current_projected",
        "prior_empirical",
    ),
    "bench_0_minus_025": ("bench_upside_weight", 0.25, 0.0),
}
CURRENT_PROFILE = {
    "salary_draw_count": 5,
    "enforce_top_n": True,
    "waiver_source": "current_projected",
    "bench_upside_weight": 0.25,
}
VARIANT_DIMS = [
    "salary_draw_count",
    "enforce_top_n",
    "waiver_source",
    "bench_upside_weight",
]


def roster_jaccard(left: str, right: str) -> tuple[float, int]:
    left_set = set(str(left).split("|")) if pd.notna(left) else set()
    right_set = set(str(right).split("|")) if pd.notna(right) else set()
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 1.0
    changed = len(left_set - right_set)
    return float(jaccard), int(changed)


def build_paired_effects(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trials = trials[trials.solve_status == "optimal"].copy()
    trials["absolute_forecast_error"] = trials.forecast_error.abs()
    metrics = [
        "actual_points",
        "drafted_only_points",
        "actual_cap_feasible",
        "actual_cap_overage",
        "actual_salary_spend",
        "absolute_forecast_error",
        "hindsight_heuristic_gap",
        "actual_waiver_starts",
    ]
    paired_rows = []
    for factor_name, (column, default_value, candidate_value) in FACTOR_COMPARISONS.items():
        other = [dim for dim in VARIANT_DIMS if dim != column]
        join_cols = ["year", "trial", *other]
        keep = [*join_cols, "roster", "contains_top_n", *metrics]
        default = trials[trials[column] == default_value][keep].copy()
        candidate = trials[trials[column] == candidate_value][keep].copy()
        merged = default.merge(
            candidate,
            on=join_cols,
            suffixes=("_default", "_candidate"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            values = row._asdict()
            jaccard, changed = roster_jaccard(
                values["roster_default"], values["roster_candidate"]
            )
            output = {
                "factor": factor_name,
                "default_value": str(default_value),
                "candidate_value": str(candidate_value),
                "year": values["year"],
                "trial": values["trial"],
                "roster_jaccard": jaccard,
                "roster_slots_changed": changed,
                "roster_changed": changed > 0,
                "top_n_binding": bool(
                    factor_name == "top_n_off_minus_on"
                    and values["contains_top_n_default"]
                    and not values["contains_top_n_candidate"]
                ),
                "both_actual_cap_feasible": bool(
                    values["actual_cap_feasible_default"]
                    and values["actual_cap_feasible_candidate"]
                ),
            }
            for metric in metrics:
                default_metric = values[f"{metric}_default"]
                candidate_metric = values[f"{metric}_candidate"]
                output[f"{metric}_effect"] = candidate_metric - default_metric
            output["joint_feasible_actual_points_effect"] = (
                values["actual_points_candidate"] - values["actual_points_default"]
                if output["both_actual_cap_feasible"]
                else np.nan
            )
            paired_rows.append(output)
    paired = pd.DataFrame(paired_rows)
    return summarize_paired_effects(paired)


def build_current_profile_effects(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pair each one-knob alternative with the exact current app profile."""
    trials = trials[trials.solve_status == "optimal"].copy()
    trials["absolute_forecast_error"] = trials.forecast_error.abs()
    metrics = [
        "actual_points",
        "drafted_only_points",
        "actual_cap_feasible",
        "actual_cap_overage",
        "actual_salary_spend",
        "absolute_forecast_error",
        "hindsight_heuristic_gap",
        "actual_waiver_starts",
    ]
    baseline_mask = pd.Series(True, index=trials.index)
    for column, value in CURRENT_PROFILE.items():
        baseline_mask &= trials[column].eq(value)
    baseline = trials.loc[baseline_mask].copy()
    paired_rows = []
    for factor_name, (column, default_value, candidate_value) in FACTOR_COMPARISONS.items():
        if CURRENT_PROFILE[column] != default_value:
            raise AssertionError(f"Current profile disagrees with {factor_name} default.")
        candidate_mask = pd.Series(True, index=trials.index)
        for profile_column, profile_value in CURRENT_PROFILE.items():
            expected = candidate_value if profile_column == column else profile_value
            candidate_mask &= trials[profile_column].eq(expected)
        candidate = trials.loc[candidate_mask].copy()
        keep = [
            "year",
            "trial",
            "roster",
            "contains_top_n",
            *metrics,
        ]
        merged = baseline[keep].merge(
            candidate[keep],
            on=["year", "trial"],
            suffixes=("_default", "_candidate"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            values = row._asdict()
            jaccard, changed = roster_jaccard(
                values["roster_default"], values["roster_candidate"]
            )
            both_feasible = bool(
                values["actual_cap_feasible_default"]
                and values["actual_cap_feasible_candidate"]
            )
            output = {
                "factor": factor_name,
                "default_value": str(default_value),
                "candidate_value": str(candidate_value),
                "year": values["year"],
                "trial": values["trial"],
                "roster_jaccard": jaccard,
                "roster_slots_changed": changed,
                "roster_changed": changed > 0,
                "top_n_binding": bool(
                    factor_name == "top_n_off_minus_on"
                    and values["contains_top_n_default"]
                    and not values["contains_top_n_candidate"]
                ),
                "both_actual_cap_feasible": both_feasible,
            }
            for metric in metrics:
                output[f"{metric}_effect"] = (
                    values[f"{metric}_candidate"] - values[f"{metric}_default"]
                )
            output["joint_feasible_actual_points_effect"] = (
                values["actual_points_candidate"] - values["actual_points_default"]
                if both_feasible
                else np.nan
            )
            paired_rows.append(output)
    paired = pd.DataFrame(paired_rows)
    return summarize_paired_effects(paired)


def summarize_paired_effects(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize paired effects, treating each trial—not each background—as independent."""
    effect_columns = [column for column in paired if column.endswith("_effect")]
    by_year = (
        paired.groupby(["factor", "default_value", "candidate_value", "year"], as_index=False)
        .agg(
            comparisons=("trial", "size"),
            roster_changed_share=("roster_changed", "mean"),
            mean_roster_jaccard=("roster_jaccard", "mean"),
            mean_roster_slots_changed=("roster_slots_changed", "mean"),
            top_n_binding_share=("top_n_binding", "mean"),
            feasible_pair_share=("both_actual_cap_feasible", "mean"),
            feasible_pair_count=("both_actual_cap_feasible", "sum"),
            **{column: (column, "mean") for column in effect_columns},
        )
    )
    mcse_rows = []
    for keys, raw_group in paired.groupby(
        ["factor", "default_value", "candidate_value", "year"]
    ):
        row = dict(zip(["factor", "default_value", "candidate_value", "year"], keys))
        row["paired_trials"] = int(raw_group.trial.nunique())
        for column in effect_columns:
            values = raw_group.groupby("trial")[column].mean()
            # With missing conditional pairs, the displayed pair-weighted mean and an
            # equal-trial MCSE have different estimands. Leave that MCSE undefined.
            row[f"{column}_mcse"] = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1 and not raw_group[column].isna().any()
                else np.nan
            )
        mcse_rows.append(row)
    by_year = by_year.merge(
        pd.DataFrame(mcse_rows),
        on=["factor", "default_value", "candidate_value", "year"],
        how="left",
        validate="one_to_one",
    )
    across = []
    for factor, group in by_year.groupby("factor"):
        row: dict[str, Any] = {
            "factor": factor,
            "default_value": group.default_value.iloc[0],
            "candidate_value": group.candidate_value.iloc[0],
            "seasons": int(group.year.nunique()),
            "mean_roster_changed_share": float(group.roster_changed_share.mean()),
            "mean_roster_jaccard": float(group.mean_roster_jaccard.mean()),
            "mean_feasible_pair_share": float(group.feasible_pair_share.mean()),
        }
        for column in effect_columns:
            values = group.set_index("year")[column]
            row[f"mean_{column}"] = float(values.mean())
            row[f"development_2022_2024_{column}"] = float(
                values.loc[values.index <= 2024].mean()
            )
            row[f"temporal_check_2025_{column}"] = float(
                values.get(2025, np.nan)
            )
            row[f"positive_seasons_{column}"] = int((values > 0).sum())
        across.append(row)
    return paired, by_year, pd.DataFrame(across)


def build_variant_summaries(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimal = trials[trials.solve_status == "optimal"].copy()
    optimal["absolute_forecast_error"] = optimal.forecast_error.abs()
    optimal["actual_points_feasible"] = optimal.actual_points.where(
        optimal.actual_cap_feasible
    )
    by_year = (
        optimal.groupby(["year", "variant", *VARIANT_DIMS], as_index=False)
        .agg(
            trials=("trial", "size"),
            unique_rosters=("roster", "nunique"),
            actual_points=("actual_points", "mean"),
            actual_points_feasible=("actual_points_feasible", "mean"),
            drafted_only_points=("drafted_only_points", "mean"),
            cap_feasible_rate=("actual_cap_feasible", "mean"),
            feasible_trials=("actual_cap_feasible", "sum"),
            cap_overage=("actual_cap_overage", "mean"),
            actual_salary_spend=("actual_salary_spend", "mean"),
            forecast_ev=("forecast_ev", "mean"),
            absolute_forecast_error=("absolute_forecast_error", "mean"),
            hindsight_heuristic_gap=("hindsight_heuristic_gap", "mean"),
            actual_waiver_starts=("actual_waiver_starts", "mean"),
            qb_count=("qb_count", "mean"),
            rb_count=("rb_count", "mean"),
            wr_count=("wr_count", "mean"),
            te_count=("te_count", "mean"),
        )
    )
    across = (
        by_year.groupby(["variant", *VARIANT_DIMS], as_index=False)
        .agg(
            seasons=("year", "nunique"),
            total_trials=("trials", "sum"),
            feasible_trials=("feasible_trials", "sum"),
            actual_points=("actual_points", "mean"),
            actual_points_feasible=("actual_points_feasible", "mean"),
            drafted_only_points=("drafted_only_points", "mean"),
            cap_feasible_rate=("cap_feasible_rate", "mean"),
            cap_overage=("cap_overage", "mean"),
            absolute_forecast_error=("absolute_forecast_error", "mean"),
            hindsight_heuristic_gap=("hindsight_heuristic_gap", "mean"),
            actual_waiver_starts=("actual_waiver_starts", "mean"),
        )
        .sort_values(["cap_feasible_rate", "actual_points_feasible"], ascending=False)
    )
    return by_year, across


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 2) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    headers = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, divider, *rows])


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    salary_metrics: pd.DataFrame,
    current_profile_effects: pd.DataFrame,
    factorial_effects: pd.DataFrame,
    variants_across: pd.DataFrame,
    join_audit: pd.DataFrame,
) -> None:
    salary_wide = salary_metrics.pivot(
        index="year", columns="salary_draw_count", values="mean_forecast_sd"
    )
    salary_ratio = (
        salary_wide[5] / salary_wide[1]
        if 1 in salary_wide and 5 in salary_wide
        else pd.Series(dtype=float)
    )
    salary_display = salary_metrics[
        [
            "year",
            "salary_draw_count",
            "coverage_80",
            "coverage_90",
            "mean_forecast_sd",
            "mean_crps",
            "mae",
        ]
    ].copy()
    effect_columns = [
        "factor",
        "mean_actual_points_effect",
        "mean_joint_feasible_actual_points_effect",
        "development_2022_2024_actual_points_effect",
        "temporal_check_2025_actual_points_effect",
        "mean_actual_cap_feasible_effect",
        "mean_feasible_pair_share",
        "mean_absolute_forecast_error_effect",
        "mean_roster_changed_share",
    ]
    best = variants_across.head(5).copy()
    match_summary = (
        join_audit.groupby("year", as_index=False)
        .agg(
            forecast_players=("player", "size"),
            salary_forecast_matches=("salary_source_matched", "sum"),
            actual_salary_matches=("actual_salary_matched", "sum"),
            raw_outcome_matches=("raw_outcome_matched", "sum"),
            excluded_keepers=("is_keeper", "sum"),
        )
    )
    lines = [
        "# Managed Auction Rolling-Origin Replay Results",
        "",
        f"Run: {args.trials} paired trials per cell, {args.contexts} prior-only "
        f"construction contexts plus an independently seeded evaluation bank per "
        f"origin, seed {args.seed}.",
        "",
        "This is a replay of empty-roster Target/look-ahead construction. It is not "
        "an end-to-end Current Nomination replay because historical nomination order "
        "and auction-state logs do not exist.",
        "",
        "## Salary calibration",
        "",
        markdown_table(
            salary_display,
            [
                "year",
                "salary_draw_count",
                "coverage_80",
                "coverage_90",
                "mean_forecast_sd",
                "mean_crps",
                "mae",
            ],
            digits=3,
        ),
        "",
        "Average-five / one-draw forecast-SD ratios by year: "
        + ", ".join(
            f"{year}: {value:.3f}" for year, value in salary_ratio.items()
        )
        + ".",
        "",
        "## One-at-a-time changes from the current app profile",
        "",
        "Positive point effects favor the candidate setting named in the factor; "
        "positive feasibility effects mean more rosters fit the realized $298 cap. "
        "The unqualified point effect scores every selected roster even when its "
        "realized price exceeded the cap. The joint-feasible point effect uses only "
        "pairs where both settings fit; `mean_feasible_pair_share` reports that "
        "coverage. Each candidate changes exactly one setting from 5 salary draws, "
        "Top-N on, projected waivers, and bench weight 0.25.",
        "",
        markdown_table(current_profile_effects, effect_columns, digits=3),
        "",
        "## Factorial marginal effects",
        "",
        "These candidate-minus-default effects average over all eight combinations "
        "of the other settings. They are a robustness view, not a direct replay of "
        "a one-knob change from the current profile.",
        "",
        markdown_table(factorial_effects, effect_columns, digits=3),
        "",
        "## Highest cap-feasibility variants",
        "",
        "Variants are ordered first by realized cap-feasible rate. Point totals are "
        "the equal-season average of each year's conditional mean among trials that "
        "fit the realized cap; `feasible_trials` is the pooled count. These should not "
        "be compared without the feasibility rate and count.",
        "",
        markdown_table(
            best,
            [
                "variant",
                "actual_points_feasible",
                "cap_feasible_rate",
                "feasible_trials",
                "drafted_only_points",
                "absolute_forecast_error",
                "hindsight_heuristic_gap",
            ],
            digits=2,
        ),
        "",
        "## Join and survivorship audit",
        "",
        markdown_table(
            match_summary,
            [
                "year",
                "forecast_players",
                "salary_forecast_matches",
                "actual_salary_matches",
                "raw_outcome_matches",
                "excluded_keepers",
            ],
            digits=0,
        ),
        "",
        "Raw FastR weekly rows, not the survivorship-filtered target template table, "
        "supply realized scores and played evidence. Construction donors are capped "
        "at origin year minus one and use preseason consensus features only.",
        "",
        "## Interpretation limits",
        "",
        "- Four seasons are four independent outcome units; Monte Carlo trials measure "
        "simulation stability, not additional seasons.",
        "- Recorded keepers are removed from this empty-roster replay. Their recorded "
        "prices remain deterministic, but no historical owner mapping exists for a "
        "specific keeper-choice replay.",
        "- Historical final auction prices are treated as exogenous. The replay cannot "
        "model how a different roster, nomination order, or bidding path would change "
        "those prices, so realized-cap feasibility is diagnostic rather than causal.",
        "- The common realized waiver stream is intentionally optimistic: ranking is "
        "causal, but eligibility is hindsight availability-filtered using target-week "
        "played evidence. It also omits opponent competition, transaction limits, and "
        "roster persistence. Because it is shared, paired differences remain useful.",
        "- The hindsight roster is an approximation using the current marginal objective "
        "plus one-swap refinement, not a global nonlinear season optimum. The reported "
        "gap is therefore not guaranteed to be nonnegative and is not labeled regret.",
        "- The shared one-swap refinement averages weekly scores across contexts but ORs "
        "their played masks. That mirrors the app, but can treat missed-game probability "
        "as a played zero and understate the value of depth or waiver cover.",
        "- Frozen files precede target result imports, but exact first-nomination "
        "timestamps were not recorded and preseason feature revisions are not fully "
        "independently timestamped.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--contexts", type=int, default=250)
    parser.add_argument("--context-draws", type=int, default=5)
    parser.add_argument("--projection-draws", type=int, default=1000)
    parser.add_argument("--salary-draws", type=int, default=5000)
    parser.add_argument("--salary-calibration-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(STUDY_DIR / "results"),
    )
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(FROZEN_SOURCES))
    if invalid:
        parser.error(f"Unsupported replay years: {invalid}")
    if min(args.trials, args.contexts, args.context_draws) <= 0:
        parser.error("Trials, contexts, and context draws must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Loading raw weekly outcomes and preseason feature rows...", flush=True)
    raw_weekly = load_raw_weekly(max_year=max(args.years))
    features = load_feature_templates()
    actual = load_actual_salaries()

    manifest: dict[str, Any] = {
        "study": STUDY_DIR.name,
        "config": vars(args),
        "simulation_helper": {
            "path": str(APP_HELPER),
            "sha256": sha256_file(APP_HELPER),
            "git_head": str(git_output(APP_ROOT, "rev-parse", "HEAD")),
        },
        "current_outcome_sources": {
            "raw_weekly_db": str(DAILY_DB),
            "raw_weekly_sha256": sha256_file(DAILY_DB),
            "simulation_db": str(SIM_DB),
            "simulation_db_sha256": sha256_file(SIM_DB),
            "raw_weekly_rows": int(len(raw_weekly)),
            "feature_rows": int(len(features)),
        },
        "origins": {},
        "method_boundary": {
            "construction_template_max_season": "origin_year - 1",
            "target_outcome_source": "raw FastR_Beta weeks 1-16",
            "current_nomination_replay": False,
            "reason": "nomination order and auction-state logs are unavailable",
        },
    }

    all_trials = []
    all_salary_metrics = []
    all_salary_players = []
    all_join_audit = []
    all_template_audit = []
    all_waiver_detail = []
    oracle_rows = []

    for year in args.years:
        year_started = time.perf_counter()
        print(f"\n=== Origin {year} ===", flush=True)
        conn, source_manifest = open_frozen_source(FROZEN_SOURCES[year])
        try:
            target_features = features[features.season == year].copy()
            forecast, ppg_draws, projection_meta = load_frozen_forecast(
                year,
                conn,
                target_features,
                args.projection_draws,
                args.seed,
            )
            forecast, salary_draws, salary_meta = build_salary_forecast(
                year,
                conn,
                forecast,
                actual,
                features,
                args.salary_draws,
                args.seed,
            )
        finally:
            conn.close()

        environment, outcome_labels = build_actual_environment(
            year,
            forecast,
            raw_weekly,
            features,
            actual,
        )
        cache, template_audit = build_template_cache(
            year,
            forecast,
            features,
            raw_weekly,
        )
        template_audit["max_donor_is_causal"] = template_audit.max_donor_season < year
        if not template_audit.max_donor_is_causal.all():
            raise AssertionError("Construction template pool crossed the replay origin.")
        all_template_audit.append(template_audit)

        player_data = forecast[
            ["player", "player_key", "pos", "pred_fp_per_game", "salary"]
        ].copy()
        sim = make_simulation(year, player_data, cache)
        current_waiver = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )
        prior_waiver, waiver_detail = empirical_waiver_baselines(
            year,
            raw_weekly,
            features,
            actual,
        )
        waiver_options = {
            "current_projected": current_waiver,
            "prior_empirical": prior_waiver,
        }
        for source_name, values in {
            "current_projected": current_waiver,
            "target_realized_diagnostic": environment["realized_waiver_baseline"],
        }.items():
            for pos, value in values.items():
                all_waiver_detail.append(
                    pd.DataFrame(
                        [
                            {
                                "origin_year": year,
                                "source": source_name,
                                "source_season": np.nan,
                                "pos": pos,
                                "baseline": value,
                                "played_slot_share": np.nan,
                                "observations": np.nan,
                            }
                        ]
                    )
                )
        all_waiver_detail.append(waiver_detail)

        keeper_mask = outcome_labels.is_keeper.to_numpy(dtype=bool)
        candidate_full_idx = np.flatnonzero(~keeper_mask)
        candidate_forecast = forecast.iloc[candidate_full_idx].reset_index(drop=True)
        candidate_ppg = ppg_draws[candidate_full_idx]
        candidate_salary_draws = salary_draws[candidate_full_idx]
        predictions = build_predictions(candidate_forecast, candidate_ppg)

        print(
            f"{year}: {len(forecast)} frozen players, {len(predictions)} selectable "
            f"after {environment['keeper_count']} recorded keepers; building "
            f"{args.contexts} prior-only contexts...",
            flush=True,
        )
        weekly, decisions, played = generate_construction_contexts(
            sim,
            predictions,
            args.contexts,
            args.seed + year,
        )
        evaluation_weekly, evaluation_decisions, evaluation_played = (
            generate_construction_contexts(
                sim,
                predictions,
                args.contexts,
                args.seed + 100_000 + year,
            )
        )
        value_banks = managed_value_banks(
            weekly,
            decisions,
            played,
            predictions,
            waiver_options,
        )
        salary_metrics, salary_players = salary_calibration(
            year,
            sim,
            forecast,
            salary_draws,
            environment,
            args.salary_calibration_draws,
            args.seed,
        )
        all_salary_metrics.append(salary_metrics)
        all_salary_players.append(salary_players)

        oracle = solve_actual_oracle(
            sim,
            predictions,
            environment,
            candidate_full_idx,
        )
        oracle_rows.append(
            {
                "year": year,
                "roster": "|".join(oracle.pop("selected_players")),
                **oracle,
            }
        )
        trials, run_meta = run_variant_trials(
            year,
            sim,
            predictions,
            candidate_salary_draws,
            environment,
            candidate_full_idx,
            weekly,
            decisions,
            played,
            evaluation_weekly,
            evaluation_decisions,
            evaluation_played,
            value_banks,
            waiver_options,
            args.trials,
            args.context_draws,
            args.seed,
            refine_roster=not args.no_refinement,
        )
        trials["hindsight_heuristic_gap"] = np.where(
            trials.get("actual_cap_feasible", False),
            oracle_rows[-1]["actual_points"] - trials.get("actual_points", np.nan),
            np.nan,
        )
        all_trials.append(trials)

        join_audit = forecast[
            [
                "player",
                "player_key",
                "pos",
                "pred_fp_per_game",
                "salary",
                "salary_source_matched",
            ]
        ].copy()
        join_audit["year"] = year
        join_audit["actual_salary"] = outcome_labels.actual_salary.to_numpy()
        join_audit["actual_salary_matched"] = outcome_labels.actual_salary_matched.to_numpy()
        join_audit["is_keeper"] = outcome_labels.is_keeper.to_numpy()
        join_audit["raw_match_kind"] = outcome_labels.raw_match_kind.to_numpy()
        join_audit["raw_outcome_matched"] = join_audit.raw_match_kind.ne("no_raw_rows")
        join_audit = join_audit.merge(
            template_audit[
                ["player", "target_feature_match", "pool_size", "max_donor_season"]
            ],
            on="player",
            how="left",
        )
        join_audit["selectable"] = ~join_audit.is_keeper
        all_join_audit.append(join_audit)

        source_manifest.update(projection_meta)
        source_manifest.update(salary_meta)
        source_manifest.update(run_meta)
        source_manifest.update(
            {
                "keeper_count": environment["keeper_count"],
                "keeper_spend": environment["keeper_spend"],
                "projected_waiver_baseline": current_waiver,
                "prior_empirical_waiver_baseline": prior_waiver,
                "realized_waiver_diagnostic": environment["realized_waiver_baseline"],
                "runtime_seconds": time.perf_counter() - year_started,
            }
        )
        manifest["origins"][str(year)] = source_manifest

        pd.concat(all_trials, ignore_index=True).to_csv(
            output_dir / "roster_trials.csv", index=False
        )
        print(
            f"{year}: complete in {time.perf_counter() - year_started:.1f}s; "
            f"oracle={oracle_rows[-1]['actual_points']:.1f} points.",
            flush=True,
        )

    trials = pd.concat(all_trials, ignore_index=True)
    salary_metrics = pd.concat(all_salary_metrics, ignore_index=True)
    salary_players = pd.concat(all_salary_players, ignore_index=True)
    join_audit = pd.concat(all_join_audit, ignore_index=True)
    template_audit = pd.concat(all_template_audit, ignore_index=True)
    waiver_detail = pd.concat(all_waiver_detail, ignore_index=True)
    oracle_frame = pd.DataFrame(oracle_rows)
    expected_trial_rows = len(args.years) * args.trials * 16
    if len(trials) != expected_trial_rows:
        raise AssertionError(
            f"Expected {expected_trial_rows} variant rows, found {len(trials)}."
        )
    trial_key = ["year", "trial", *VARIANT_DIMS]
    if trials.duplicated(trial_key).any():
        raise AssertionError("Replay contains duplicate factorial cells.")
    cell_counts = trials.groupby(["year", "trial"]).size()
    if not cell_counts.eq(16).all():
        raise AssertionError("Replay does not contain all 16 cells for every trial.")
    if not trials.solve_status.eq("optimal").all():
        failures = trials.loc[trials.solve_status.ne("optimal"), ["year", "trial", "variant"]]
        raise AssertionError(f"Replay contains failed solves: {failures.head().to_dict('records')}")
    roster_sizes = trials.roster.str.count(r"\|").add(1)
    if not roster_sizes.eq(ROSTER_SIZE).all():
        raise AssertionError("A solved replay roster does not contain 13 players.")
    if (trials.forecast_salary_spend > SALARY_CAP + 1e-4).any():
        raise AssertionError("A solved replay roster exceeds the forecast salary cap.")
    if (~trials.loc[trials.enforce_top_n, "contains_top_n"]).any():
        raise AssertionError("A Top-N constrained replay roster violates the constraint.")
    factorial_paired, factorial_by_year, factorial_across = build_paired_effects(trials)
    current_paired, current_by_year, current_across = build_current_profile_effects(trials)
    variants_by_year, variants_across = build_variant_summaries(trials)

    outputs = {
        "roster_trials.csv": trials,
        "variant_summary_by_year.csv": variants_by_year,
        "variant_summary_across_years.csv": variants_across,
        "paired_current_profile_effects.csv": current_paired,
        "current_profile_effects_by_year.csv": current_by_year,
        "current_profile_effects_across_years.csv": current_across,
        "paired_factorial_marginal_effects.csv": factorial_paired,
        "factorial_marginal_effects_by_year.csv": factorial_by_year,
        "factorial_marginal_effects_across_years.csv": factorial_across,
        "salary_calibration_by_year.csv": salary_metrics,
        "salary_calibration_players.csv": salary_players,
        "join_audit.csv": join_audit,
        "template_pool_audit.csv": template_audit,
        "waiver_baselines.csv": waiver_detail,
        "hindsight_oracle_rosters.csv": oracle_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["validation"] = {
        "expected_trial_rows": expected_trial_rows,
        "all_solves_optimal": True,
        "all_rosters_size_13": True,
        "all_forecast_spend_within_cap": True,
        "all_top_n_constraints_satisfied": True,
        "all_template_donors_pre_origin": bool(
            template_audit.max_donor_is_causal.all()
        ),
    }
    manifest["output_rows"] = {
        filename: int(len(frame)) for filename, frame in outputs.items()
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_summary(
        output_dir,
        args,
        salary_metrics,
        current_across,
        factorial_across,
        variants_across,
        join_audit,
    )
    print(
        f"\nReplay complete in {time.perf_counter() - started:.1f}s. "
        f"Results: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
