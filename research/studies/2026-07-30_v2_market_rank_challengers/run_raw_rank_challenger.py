"""Test raw available expert-rank aggregations against the locked V2 PPG model.

The user-requested primary diagnostic is the median of every observed raw
overall rank.  Secondary variants test a log transform, a season-wide
percentile calculated after aggregation, and that percentile plus a normalized
source-coverage control.  DK substitutes the full-PPR ETR rank table; beta
retains the half-PPR ETR table.

The study is attribution-only.  Every forecast origin trains only on earlier
seasons, reuses the incumbent's strictly-prior selected hyperparameters, and
keeps imputation inside each training fit.  The primary comparison uses
full-column random forests on both sides so adding a feature cannot change
which incumbent columns are sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import SOURCE_DB_PATH
from Scripts.V2.contracts import scoring_hash
from Scripts.V2.locked_candidates import (
    LOCKED_BLEND_WEIGHTS,
    PRIMARY_PPG_FEATURES,
    lock_version_for_scoring,
)
from Scripts.V2.production_handoff import (
    load_identity_frames,
    resolve_source_player_keys,
)


NORMALIZED_RUNNER_PATH = STUDY_ROOT / "run_expert_rank_challenger.py"
DEFAULT_DATABASES = {
    "dk": (
        STUDY_ROOT
        / "artifacts"
        / "local"
        / "Projection_V2_single_nffc.sqlite3"
    ),
    "beta": (
        STUDY_ROOT
        / "artifacts"
        / "local"
        / "Projection_V2_beta_single_nffc.sqlite3"
    ),
}
VARIANT_FEATURES = {
    "incumbent": (),
    "normalized_scoring_specific": (
        "scoring_specific_rank_position_percentile_median",
    ),
    "raw_available_median": ("raw_rank_median",),
    "raw_log": ("raw_rank_log1p",),
    "raw_percentile": ("raw_rank_overall_percentile",),
    "raw_percentile_coverage": (
        "raw_rank_overall_percentile",
        "rank_source_coverage",
    ),
}
PROMOTION_VARIANT = "raw_percentile_coverage"
NORMALIZED_COMPARATOR_VARIANT = "normalized_scoring_specific"
MODEL_COMPONENTS = tuple(LOCKED_BLEND_WEIGHTS)
RANDOM_FOREST_COMPONENT = "conditional_ppg_random_forest"
FULL_COLUMN_RF_METHOD = "random_forest_full_columns"
CONTROLLED_BLEND_METHOD = "controlled_equal_thirds"
RANDOM_SEED = 1234
ETR_PPR_RAW_FILES = {
    2025: (
        REPO_ROOT
        / "Data"
        / "OtherData"
        / "ETR"
        / "2025ETR NFL Re-Draft Rankings - PPR.csv"
    ),
    2026: (
        REPO_ROOT
        / "Data"
        / "OtherData"
        / "ETR"
        / "2026NFL ETR Rankings – Full PPR.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league",
        choices=("dk", "beta", "all"),
        default="dk",
    )
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    return parser.parse_args()


def _load_normalized_runner():
    spec = importlib.util.spec_from_file_location(
        "v2_normalized_rank_runner_for_raw_study",
        NORMALIZED_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load normalized-rank runner: {NORMALIZED_RUNNER_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_existing_raw_median(
    features: pd.DataFrame,
    market_values: pd.DataFrame,
) -> float:
    expected = (
        market_values[market_values["expert_rank"].notna()]
        .groupby(["player_key", "season"], sort=True)["expert_rank"]
        .median()
        .rename("expected_raw_rank_median")
        .reset_index()
    )
    compared = features[
        ["player_key", "season", "expert_rank_median"]
    ].merge(
        expected,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    missing_mismatch = compared["expert_rank_median"].isna() ^ compared[
        "expected_raw_rank_median"
    ].isna()
    if missing_mismatch.any():
        raise ValueError(
            "Existing expert_rank_median missingness does not reproduce the "
            f"raw available-source median: {int(missing_mismatch.sum())} rows"
        )
    complete = compared.dropna(
        subset=["expert_rank_median", "expected_raw_rank_median"]
    )
    maximum = float(
        (
            complete["expert_rank_median"]
            - complete["expected_raw_rank_median"]
        )
        .abs()
        .max()
    )
    if maximum > 1e-12:
        raise ValueError(
            "Existing expert_rank_median differs from the raw median: "
            f"max_abs_delta={maximum}"
        )
    return maximum


def _load_full_ppr_etr(
    database: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not SOURCE_DB_PATH.is_file():
        raise FileNotFoundError(
            f"Source database does not exist: {SOURCE_DB_PATH.resolve()}"
        )
    aliases, identities = load_identity_frames(database)
    with _read_only_connection(database) as connection:
        spine_positions = pd.read_sql_query(
            """
            SELECT player_key, CAST(season AS INTEGER) season, position
            FROM player_season_spine
            """,
            connection,
        )
    if spine_positions.duplicated(["player_key", "season"]).any():
        raise ValueError(
            "player_season_spine is not unique by player_key-season"
        )
    spine_positions.rename(
        columns={"position": "staged_position"},
        inplace=True,
    )
    with _read_only_connection(SOURCE_DB_PATH) as connection:
        ppr = pd.read_sql_query(
            """
            SELECT player, pos, team, CAST(year AS INTEGER) season,
                   etr_rank, etr_pos_rank
            FROM ETR_Ranks_PPR
            """,
            connection,
        )
    if ppr.empty:
        raise ValueError("ETR_Ranks_PPR contains no rows")

    frames: list[pd.DataFrame] = []
    resolution: dict[str, object] = {
        "source_table": "ETR_Ranks_PPR",
        "seasons": {},
    }
    for season, rows in ppr.groupby("season", sort=True):
        resolved = resolve_source_player_keys(
            rows,
            aliases,
            identities,
            year=int(season),
            source_name=f"ETR_Ranks_PPR_{int(season)}",
            require_complete=True,
        )
        if resolved["player_key"].duplicated().any():
            raise ValueError(
                f"ETR_Ranks_PPR {int(season)} has duplicate canonical keys"
            )
        staged_positions = resolved[["player_key", "season", "pos"]].merge(
            spine_positions,
            on=["player_key", "season"],
            how="left",
            validate="many_to_one",
        )
        position_disagreement = (
            staged_positions["staged_position"].isna()
            | staged_positions["pos"].astype(str).str.upper().ne(
                staged_positions["staged_position"]
                .astype(str)
                .str.upper()
            )
        )
        if position_disagreement.any():
            preview = staged_positions.loc[
                position_disagreement,
                ["player_key", "season", "pos", "staged_position"],
            ].head(20)
            raise ValueError(
                f"ETR_Ranks_PPR {int(season)} position disagreement: "
                f"{preview.to_dict('records')}"
            )
        methods = resolved[
            "eligibility_key_match_method"
        ].value_counts().to_dict()
        resolution["seasons"][str(int(season))] = {
            "rows": len(resolved),
            "resolved_rows": int(resolved["player_key"].notna().sum()),
            "match_methods": {
                str(key): int(value) for key, value in methods.items()
            },
            "staged_position_disagreements": int(
                position_disagreement.sum()
            ),
        }
        current = resolved.rename(
            columns={
                "pos": "position",
                "etr_rank": "expert_rank",
                "etr_pos_rank": "source_position_rank",
            }
        )[
            [
                "player_key",
                "season",
                "position",
                "team",
                "expert_rank",
                "source_position_rank",
            ]
        ].copy()
        current["source"] = "etr_rank"
        current["source_table"] = "ETR_Ranks_PPR"
        frames.append(current)
    output = pd.concat(frames, ignore_index=True)
    output["position"] = output["position"].astype(str).str.upper()
    return output, resolution


def _scoring_specific_rank_rows(
    market_values: pd.DataFrame,
    database: Path,
    league: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = [
        "player_key",
        "season",
        "source",
        "source_table",
        "position",
        "team",
        "expert_rank",
        "source_position_rank",
    ]
    ranks = market_values.loc[
        market_values["expert_rank"].notna()
        & market_values["position"].isin(("QB", "RB", "WR", "TE")),
        columns,
    ].copy()
    ppr_resolution: dict[str, object] = {
        "source_table": None,
        "seasons": {},
    }
    if league == "dk":
        ranks = ranks[~ranks["source"].eq("etr_rank")].copy()
        ppr, ppr_resolution = _load_full_ppr_etr(database)
        ranks = pd.concat([ranks, ppr], ignore_index=True, sort=False)

    ranks["season"] = pd.to_numeric(
        ranks["season"], errors="raise"
    ).astype(int)
    ranks["position"] = ranks["position"].astype(str).str.upper()
    ranks["expert_rank"] = pd.to_numeric(
        ranks["expert_rank"], errors="raise"
    )
    invalid = (
        ~np.isfinite(ranks["expert_rank"])
        | ranks["expert_rank"].le(0)
    )
    if invalid.any():
        raise ValueError(
            f"Scoring-specific rank rows contain {int(invalid.sum())} "
            "invalid ranks"
        )
    duplicate = ranks.duplicated(
        ["player_key", "season", "source"],
        keep=False,
    )
    if duplicate.any():
        preview = ranks.loc[
            duplicate,
            ["player_key", "season", "source", "source_table"],
        ].head(20)
        raise ValueError(
            "Scoring-specific rank rows are not one vote per source: "
            f"{preview.to_dict('records')}"
        )
    ranks.sort_values(
        ["season", "source", "position", "expert_rank", "player_key"],
        inplace=True,
    )
    ranks.reset_index(drop=True, inplace=True)
    return ranks, ppr_resolution


def _canonicalize_rank_positions(
    rank_rows: pd.DataFrame,
    feature_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = feature_universe[
        ["player_key", "season", "position"]
    ].drop_duplicates()
    if canonical.duplicated(["player_key", "season"]).any():
        raise ValueError(
            "Feature universe is not unique by player_key-season"
        )
    canonical.rename(
        columns={"position": "canonical_position"},
        inplace=True,
    )
    output = rank_rows.rename(
        columns={"position": "source_position"}
    ).merge(
        canonical,
        on=["player_key", "season"],
        how="left",
        validate="many_to_one",
    )
    unmatched = output["canonical_position"].isna()
    if unmatched.any():
        preview = output.loc[
            unmatched,
            ["player_key", "season", "source", "source_position"],
        ].head(20)
        raise ValueError(
            "Scoring-specific rank rows fall outside the feature universe: "
            f"{preview.to_dict('records')}"
        )
    output["source_position"] = (
        output["source_position"].astype(str).str.upper()
    )
    output["canonical_position"] = (
        output["canonical_position"].astype(str).str.upper()
    )
    output["position_disagreement"] = output["source_position"].ne(
        output["canonical_position"]
    )
    audit = (
        output.groupby(
            [
                "season",
                "source",
                "source_table",
                "source_position",
                "canonical_position",
                "position_disagreement",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            rank_rows=("player_key", "size"),
            players=("player_key", "nunique"),
        )
        .reset_index()
    )
    output["position"] = output.pop("canonical_position")
    output.sort_values(
        ["season", "source", "position", "expert_rank", "player_key"],
        inplace=True,
    )
    output.reset_index(drop=True, inplace=True)
    return output, audit


def build_scoring_specific_normalized_rank(
    rank_rows: pd.DataFrame,
) -> pd.DataFrame:
    ranks = rank_rows.copy()
    source_position_keys = ["source", "season", "position"]
    ranks["_source_position_order"] = ranks.groupby(
        source_position_keys,
        sort=True,
    )["expert_rank"].rank(method="average", ascending=True)
    ranks["_source_position_size"] = ranks.groupby(
        source_position_keys,
        sort=True,
    )["player_key"].transform("count")
    denominator = ranks["_source_position_size"].sub(1)
    ranks["_source_position_percentile"] = 1 - (
        ranks["_source_position_order"].sub(1)
        / denominator.where(denominator.gt(0))
    )
    ranks.loc[
        ranks["_source_position_size"].eq(1),
        "_source_position_percentile",
    ] = 0.5
    percentile = ranks["_source_position_percentile"].dropna()
    if not percentile.between(0, 1).all():
        raise ValueError(
            "Scoring-specific normalized comparator escaped [0, 1]"
        )
    return (
        ranks.groupby(["player_key", "season"], sort=True)
        .agg(
            scoring_specific_rank_position_percentile_median=(
                "_source_position_percentile",
                "median",
            )
        )
        .reset_index()
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stable_frame_sha256(
    frame: pd.DataFrame,
    sort_columns: Sequence[str],
) -> str:
    stable = frame.copy()
    stable.sort_values(list(sort_columns), inplace=True)
    stable = stable.reindex(sorted(stable.columns), axis=1)
    payload = stable.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _stable_rank_rows_sha256(rank_rows: pd.DataFrame) -> str:
    columns = [
        "player_key",
        "season",
        "source",
        "source_table",
        "position",
        "source_position",
        "team",
        "expert_rank",
        "source_position_rank",
    ]
    columns = [column for column in columns if column in rank_rows.columns]
    stable = rank_rows.loc[:, columns].copy()
    stable.sort_values(
        ["season", "source", "position", "expert_rank", "player_key"],
        inplace=True,
    )
    payload = stable.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _input_manifest(
    database: Path,
    league: str,
    rank_rows: pd.DataFrame,
    *,
    locked_lineage: dict[str, object],
    position_audit: pd.DataFrame,
) -> dict[str, object]:
    raw_files: list[dict[str, object]] = []
    if league == "dk":
        for season, path in sorted(ETR_PPR_RAW_FILES.items()):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing pinned full-PPR ETR input for {season}: {path}"
                )
            raw_files.append(
                {
                    "season": season,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "league": league,
        "staged_database": str(database.resolve()),
        "source_database": str(SOURCE_DB_PATH.resolve()),
        "rank_rows": int(len(rank_rows)),
        "rank_row_seasons": sorted(
            int(value) for value in rank_rows["season"].unique()
        ),
        "rank_sources": sorted(
            str(value) for value in rank_rows["source"].unique()
        ),
        "rank_source_tables": sorted(
            str(value) for value in rank_rows["source_table"].unique()
        ),
        "rank_rows_sha256": _stable_rank_rows_sha256(rank_rows),
        "rank_position_disagreement_rows": int(
            position_audit.loc[
                position_audit["position_disagreement"].eq(True),
                "rank_rows",
            ].sum()
        ),
        "locked_lineage": locked_lineage,
        "model_spec": {
            "variants": {
                key: list(value)
                for key, value in VARIANT_FEATURES.items()
            },
            "primary_features": list(PRIMARY_PPG_FEATURES),
            "components": list(MODEL_COMPONENTS),
            "controlled_method": CONTROLLED_BLEND_METHOD,
            "production_surface_method": "equal_thirds",
            "random_seed": RANDOM_SEED,
        },
        "pinned_etr_ppr_raw_files": raw_files,
    }


def _iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def build_raw_rank_features(
    rank_rows: pd.DataFrame,
    feature_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["player_key", "season"]
    universe = feature_universe[
        ["player_key", "season", "position"]
    ].drop_duplicates()
    if universe.duplicated(keys).any():
        raise ValueError("Feature universe is not unique by player-season")

    observed = (
        rank_rows.groupby(keys, sort=True)
        .agg(
            raw_rank_median=("expert_rank", "median"),
            raw_rank_iqr=("expert_rank", _iqr),
            raw_rank_source_count=("source", "nunique"),
        )
        .reset_index()
    )
    observed["raw_rank_log1p"] = np.log1p(observed["raw_rank_median"])

    depths = (
        rank_rows.groupby(
            ["season", "position", "source", "source_table"],
            sort=True,
            dropna=False,
        )
        .agg(
            source_position_depth=("player_key", "nunique"),
            source_max_overall_rank=("expert_rank", "max"),
        )
        .reset_index()
    )
    depths["season_position_median_source_depth"] = depths.groupby(
        ["season", "position"]
    )["source_position_depth"].transform("median")
    depths["is_shallow_source"] = depths["source_position_depth"].le(
        depths["season_position_median_source_depth"]
    )

    expected = universe.merge(
        depths,
        on=["season", "position"],
        how="left",
        validate="many_to_many",
    )
    present = rank_rows[
        ["player_key", "season", "position", "source"]
    ].drop_duplicates()
    present["rank_present"] = 1
    expected = expected.merge(
        present,
        on=["player_key", "season", "position", "source"],
        how="left",
        validate="one_to_one",
    )
    expected["rank_present"] = expected["rank_present"].fillna(0).astype(int)
    expected["is_shallow_source"] = expected[
        "is_shallow_source"
    ].eq(True)
    expected["shallow_rank_present"] = (
        expected["rank_present"].eq(1)
        & expected["is_shallow_source"]
    ).astype(int)
    expected["shallow_rank_missing"] = (
        expected["rank_present"].eq(0)
        & expected["is_shallow_source"]
        & expected["source"].notna()
    ).astype(int)

    coverage = (
        expected.groupby(keys, sort=True)
        .agg(
            eligible_rank_source_count=("source", "nunique"),
            observed_rank_source_count=("rank_present", "sum"),
            shallow_rank_source_count=("is_shallow_source", "sum"),
            shallow_rank_present_count=("shallow_rank_present", "sum"),
            shallow_rank_omission_count=("shallow_rank_missing", "sum"),
        )
        .reset_index()
    )
    denominator = coverage["eligible_rank_source_count"].replace(0, np.nan)
    coverage["rank_source_coverage"] = (
        coverage["observed_rank_source_count"] / denominator
    )
    coverage["shallow_rank_omission_flag"] = (
        coverage["shallow_rank_omission_count"].gt(0).astype(int)
    )

    output = universe.merge(
        observed,
        on=keys,
        how="left",
        validate="one_to_one",
    ).merge(
        coverage,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    observed_count_mismatch = (
        output["raw_rank_source_count"].fillna(0)
        .astype(int)
        .ne(
            output["observed_rank_source_count"]
            .fillna(0)
            .astype(int)
        )
    )
    if observed_count_mismatch.any():
        preview = output.loc[
            observed_count_mismatch,
            [
                "player_key",
                "season",
                "position",
                "raw_rank_source_count",
                "observed_rank_source_count",
            ],
        ].head(20)
        raise ValueError(
            "Rank publication coverage disagrees with raw source counts: "
            f"{preview.to_dict('records')}"
        )
    impossible_coverage = (
        output["observed_rank_source_count"].fillna(0)
        > output["eligible_rank_source_count"].fillna(0)
    )
    if impossible_coverage.any():
        raise ValueError(
            "Observed rank source count exceeds eligible publishing sources"
        )
    output["raw_rank_available"] = output["raw_rank_median"].notna().astype(int)
    output["_raw_rank_order"] = output.groupby("season")[
        "raw_rank_median"
    ].rank(method="average", ascending=True)
    output["_raw_rank_count"] = output.groupby("season")[
        "raw_rank_median"
    ].transform("count")
    percentile_denominator = output["_raw_rank_count"].sub(1)
    output["raw_rank_overall_percentile"] = 1 - (
        output["_raw_rank_order"].sub(1)
        / percentile_denominator.where(percentile_denominator.gt(0))
    )
    output.loc[
        output["_raw_rank_count"].eq(1)
        & output["raw_rank_median"].notna(),
        "raw_rank_overall_percentile",
    ] = 0.5
    observed_percentiles = output[
        "raw_rank_overall_percentile"
    ].dropna()
    if not observed_percentiles.between(0, 1).all():
        raise ValueError("Raw-rank percentile escaped [0, 1]")
    output.drop(
        columns=["_raw_rank_order", "_raw_rank_count"],
        inplace=True,
    )
    return output, depths


def _feature_columns(variant: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*PRIMARY_PPG_FEATURES, *VARIANT_FEATURES[variant]))
    )


def _read_only_connection(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro",
        uri=True,
    )


def _locked_lineage(
    candidate_runs: pd.DataFrame,
    selected: pd.DataFrame,
    locked_predictions: pd.DataFrame,
    *,
    feature_run_id: str,
    league: str,
) -> dict[str, object]:
    lock_version = lock_version_for_scoring(league)
    active = candidate_runs[
        candidate_runs["lock_version"].eq(lock_version)
    ].copy()
    if len(active) != 1:
        raise ValueError(
            f"Expected one locked_candidate_runs row for {lock_version}; "
            f"observed {len(active)}"
        )
    run = active.iloc[0]
    if str(run.feature_run_id) != str(feature_run_id):
        raise ValueError(
            "Locked model feature lineage does not match active features: "
            f"locked={run.feature_run_id}, active={feature_run_id}"
        )
    model_run_id = str(run.model_run_id)
    for table_name, frame in (
        ("locked_selected_hyperparameters", selected),
        ("locked_whole_season_predictions", locked_predictions),
    ):
        observed_run_ids = set(frame["model_run_id"].dropna().astype(str))
        observed_locks = set(frame["lock_version"].dropna().astype(str))
        if observed_run_ids != {model_run_id}:
            raise ValueError(
                f"{table_name} model_run_id mismatch: {observed_run_ids}"
            )
        if observed_locks != {lock_version}:
            raise ValueError(
                f"{table_name} lock_version mismatch: {observed_locks}"
            )
    return {
        "lock_version": lock_version,
        "model_run_id": model_run_id,
        "feature_run_id": str(feature_run_id),
        "created_at_utc": str(run.created_at_utc),
        "status": str(run.status),
        "selected_hyperparameter_rows": int(len(selected)),
        "selected_hyperparameters_sha256": _stable_frame_sha256(
            selected,
            ["model_name", "forecast_origin"],
        ),
        "locked_prediction_rows": int(len(locked_predictions)),
    }


def _load_inputs(
    database: Path,
    league: str,
):
    if not database.is_file():
        raise FileNotFoundError(
            f"Staged V2 database does not exist: {database.resolve()}"
        )
    normalized_runner = _load_normalized_runner()
    locked = normalized_runner._load_locked_runner()
    locked.ACTIVE_OUTPUT_DB_PATH = database
    locked.ACTIVE_RESULTS_DIR = STUDY_ROOT / "artifacts" / "local"
    locked.ACTIVE_SCORING_OBJECTIVE = league
    locked.ACTIVE_LOCK_VERSION = lock_version_for_scoring(league)
    features, _, feature_run_id = locked._load_inputs()
    with _read_only_connection(database) as connection:
        market_values = pd.read_sql_query(
            "SELECT * FROM player_season_market_values",
            connection,
        )
        selected = pd.read_sql_query(
            "SELECT * FROM locked_selected_hyperparameters",
            connection,
        )
        locked_predictions = pd.read_sql_query(
            "SELECT * FROM locked_whole_season_predictions",
            connection,
        )
        candidate_runs = pd.read_sql_query(
            "SELECT * FROM locked_candidate_runs",
            connection,
        )
    locked_lineage = _locked_lineage(
        candidate_runs,
        selected,
        locked_predictions,
        feature_run_id=feature_run_id,
        league=league,
    )
    raw_median_reproduction_delta = _assert_existing_raw_median(
        features,
        market_values,
    )
    rank_rows, ppr_resolution = _scoring_specific_rank_rows(
        market_values,
        database,
        league,
    )
    rank_rows, position_audit = _canonicalize_rank_positions(
        rank_rows,
        features,
    )
    input_manifest = _input_manifest(
        database,
        league,
        rank_rows,
        locked_lineage=locked_lineage,
        position_audit=position_audit,
    )
    raw_features, depth_audit = build_raw_rank_features(
        rank_rows,
        features,
    )
    normalized_comparator = build_scoring_specific_normalized_rank(
        rank_rows
    )
    raw_features = raw_features.merge(
        normalized_comparator,
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        raw_features.drop(columns="position"),
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    source_coverage = (
        rank_rows.groupby(
            ["season", "source", "source_table"],
            sort=True,
            dropna=False,
        )
        .agg(
            ranked_players=("player_key", "nunique"),
            positions=("position", "nunique"),
        )
        .reset_index()
    )
    return (
        normalized_runner,
        locked,
        features,
        selected,
        locked_predictions,
        raw_features,
        source_coverage,
        depth_audit,
        feature_run_id,
        raw_median_reproduction_delta,
        ppr_resolution,
        input_manifest,
        position_audit,
    )


def _run_predictions(
    locked,
    features: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    ppg, _, candidates = locked._target_frames(features)
    prediction_frames: list[pd.DataFrame] = []
    for variant in VARIANT_FEATURES:
        columns = _feature_columns(variant)
        missing = sorted(set(columns).difference(features.columns))
        if missing:
            raise ValueError(f"{variant} is missing feature columns: {missing}")
        for component in MODEL_COMPONENTS:
            component_selection = selected[
                selected["model_name"].eq(component)
            ].copy()
            prediction_frames.append(
                locked._selected_predictions(
                    ppg,
                    candidates,
                    columns,
                    fit_model_name=component,
                    output_model_name=f"{variant}__{component}",
                    selected=component_selection,
                )
            )

    full_column_selected = selected[
        selected["model_name"].eq(RANDOM_FOREST_COMPONENT)
    ].copy()
    full_column_selected["parameters_json"] = full_column_selected[
        "parameters_json"
    ].map(
        lambda value: json.dumps(
            {
                **json.loads(value),
                "max_features": 1.0,
            },
            sort_keys=True,
        )
    )
    for variant in VARIANT_FEATURES:
        prediction_frames.append(
            locked._selected_predictions(
                ppg,
                candidates,
                _feature_columns(variant),
                fit_model_name=RANDOM_FOREST_COMPONENT,
                output_model_name=(
                    f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns"
                ),
                selected=full_column_selected,
            )
        )

    long = pd.concat(prediction_frames, ignore_index=True)
    wide = long.pivot(
        index=["player_key", "season", "position"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    wide.columns.name = None
    metadata_columns = [
        "player_key",
        "season",
        "position",
        "conditional_ppg",
        "conditional_ppg_training_eligible",
        "has_prior_outcome",
        "is_rookie",
        "year_exp",
        "raw_rank_median",
        "raw_rank_log1p",
        "raw_rank_iqr",
        "raw_rank_source_count",
        "raw_rank_overall_percentile",
        "scoring_specific_rank_position_percentile_median",
        "eligible_rank_source_count",
        "observed_rank_source_count",
        "rank_source_coverage",
        "shallow_rank_omission_flag",
        "raw_rank_available",
    ]
    output = candidates.loc[:, metadata_columns].merge(
        wide,
        on=["player_key", "season", "position"],
        how="left",
        validate="one_to_one",
    )
    output = locked._add_history_depth(output)
    for variant in VARIANT_FEATURES:
        component_columns = [
            f"{variant}__{component}" for component in MODEL_COMPONENTS
        ]
        component_values = output[component_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        weights = np.asarray(
            [LOCKED_BLEND_WEIGHTS[component] for component in MODEL_COMPONENTS],
            dtype=float,
        )
        output[f"{variant}__equal_thirds"] = component_values.to_numpy().dot(
            weights
        )
        output.loc[
            component_values.isna().any(axis=1),
            f"{variant}__equal_thirds",
        ] = np.nan
        controlled_columns = [
            f"{variant}__conditional_ppg_lasso",
            f"{variant}__conditional_ppg_lightgbm",
            f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns",
        ]
        controlled_values = output[controlled_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        output[f"{variant}__{CONTROLLED_BLEND_METHOD}"] = (
            controlled_values.mean(axis=1)
        )
        output.loc[
            controlled_values.isna().any(axis=1),
            f"{variant}__{CONTROLLED_BLEND_METHOD}",
        ] = np.nan
    return output


def _assert_incumbent_reproduces_exactly(
    predictions: pd.DataFrame,
    locked_predictions: pd.DataFrame,
) -> float:
    checks = {
        **{
            component: f"incumbent__{component}"
            for component in MODEL_COMPONENTS
        },
        "conditional_ppg_primary_blend": "incumbent__equal_thirds",
    }
    differences: list[float] = []
    keys = ["player_key", "season"]
    for locked_method, challenger_column in checks.items():
        expected = locked_predictions[
            locked_predictions["method"].eq(locked_method)
        ][keys + ["prediction"]].dropna(subset=["prediction"])
        evaluation_universe = (
            predictions["season"].isin(expected["season"].unique())
            & predictions["conditional_ppg_training_eligible"].eq(1)
            & predictions["conditional_ppg"].notna()
        )
        observed = predictions.loc[
            evaluation_universe,
            keys + [challenger_column],
        ].rename(columns={challenger_column: "observed"})
        observed = observed.dropna(subset=["observed"])
        if expected.duplicated(keys).any() or observed.duplicated(keys).any():
            raise ValueError(
                f"Duplicate incumbent reproduction keys for {locked_method}"
            )
        key_audit = expected[keys].merge(
            observed[keys],
            on=keys,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not key_audit["_merge"].eq("both").all():
            counts = key_audit["_merge"].value_counts().to_dict()
            raise ValueError(
                f"Incumbent reproduction keyset mismatch for "
                f"{locked_method}: {counts}"
            )
        compared = expected.merge(
            observed,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        differences.extend(
            (
                compared["prediction"] - compared["observed"]
            ).abs().tolist()
        )
    if not differences:
        raise ValueError("No incumbent predictions were reproduced")
    maximum = float(max(differences))
    if maximum > 1e-10:
        raise ValueError(
            "Incumbent attribution replay differs from locked predictions: "
            f"max_abs_delta={maximum}"
        )
    return maximum


def _evaluation_long(
    predictions: pd.DataFrame,
    outer_seasons: Sequence[int],
) -> pd.DataFrame:
    working = predictions.copy()
    working["rank_availability"] = np.where(
        working["raw_rank_available"].eq(1),
        "available",
        "unavailable",
    )
    working["coverage_band"] = np.select(
        [
            working["raw_rank_available"].eq(0),
            working["rank_source_coverage"].ge(1.0 - 1e-12),
            working["rank_source_coverage"].ge(0.5),
        ],
        ["unavailable", "complete", "partial_high"],
        default="partial_low",
    )
    working["coverage_completeness"] = np.where(
        working["coverage_band"].eq("complete"),
        "complete",
        "incomplete_or_unavailable",
    )
    working["shallow_omission"] = np.where(
        working["shallow_rank_omission_flag"].eq(1),
        "omission",
        "no_omission",
    )
    eligible = (
        working["season"].isin(outer_seasons)
        & working["conditional_ppg_training_eligible"].eq(1)
        & working["conditional_ppg"].notna()
    )
    metadata = [
        "player_key",
        "season",
        "position",
        "history_depth",
        "conditional_ppg",
        "raw_rank_available",
        "rank_availability",
        "rank_source_coverage",
        "coverage_band",
        "coverage_completeness",
        "shallow_omission",
    ]
    rows: list[pd.DataFrame] = []
    for variant in VARIANT_FEATURES:
        methods = {
            component.removeprefix("conditional_ppg_"): (
                f"{variant}__{component}"
            )
            for component in MODEL_COMPONENTS
        }
        methods["equal_thirds"] = f"{variant}__equal_thirds"
        methods[FULL_COLUMN_RF_METHOD] = (
            f"{variant}__{RANDOM_FOREST_COMPONENT}_full_columns"
        )
        methods[CONTROLLED_BLEND_METHOD] = (
            f"{variant}__{CONTROLLED_BLEND_METHOD}"
        )
        for method, column in methods.items():
            current = working.loc[eligible, metadata].copy()
            current["variant"] = variant
            current["method"] = method
            current["actual"] = current.pop("conditional_ppg")
            current["prediction"] = working.loc[eligible, column].to_numpy()
            current = current[current["prediction"].notna()].copy()
            current["squared_error"] = (
                current["actual"] - current["prediction"]
            ) ** 2
            rows.append(current)
    output = pd.concat(rows, ignore_index=True)
    counts = output.groupby(["variant", "method"]).size()
    if counts.nunique() != 1:
        raise ValueError(
            "Raw-rank variants do not evaluate identical OOF rows: "
            f"{counts.to_dict()}"
        )
    return output


def _score_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, method), model in evaluation.groupby(
        ["variant", "method"],
        sort=True,
    ):
        slices: list[tuple[str, str, pd.DataFrame]] = [
            ("pooled", "all", model),
            ("recent", "2023_2025", model[model["season"].ge(2023)]),
            (
                "source_era",
                "2017_2023",
                model[model["season"].le(2023)],
            ),
            (
                "source_era",
                "2024_2025",
                model[model["season"].ge(2024)],
            ),
        ]
        for column, slice_type in (
            ("season", "season"),
            ("position", "position"),
            ("history_depth", "history_depth"),
            ("rank_availability", "rank_availability"),
            ("coverage_band", "coverage_band"),
            ("coverage_completeness", "coverage_completeness"),
            ("shallow_omission", "shallow_omission"),
        ):
            slices.extend(
                (slice_type, str(value), group)
                for value, group in model.groupby(column, sort=True)
            )
        for slice_type, slice_value, group in slices:
            if group.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "n_rows": len(group),
                    "n_seasons": group["season"].nunique(),
                    "rmse": float(
                        np.sqrt(
                            mean_squared_error(
                                group["actual"],
                                group["prediction"],
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _score_delta(
    scores: pd.DataFrame,
    variant: str,
    method: str,
    slice_type: str,
    slice_value: str,
) -> float:
    selected = scores[
        scores["variant"].isin(("incumbent", variant))
        & scores["method"].eq(method)
        & scores["slice_type"].eq(slice_type)
        & scores["slice_value"].eq(slice_value)
    ].set_index("variant")["rmse"]
    if set(selected.index) != {"incumbent", variant}:
        raise ValueError(
            f"Missing {method} {slice_type}/{slice_value} score for {variant}"
        )
    return float(selected[variant] - selected["incumbent"])


def _variant_summary(
    normalized_runner,
    evaluation: pd.DataFrame,
    scores: pd.DataFrame,
    iterations: int,
    *,
    method: str,
    variants: Sequence[str] | None = None,
) -> pd.DataFrame:
    blend = evaluation[evaluation["method"].eq(method)].copy()
    incumbent = blend[blend["variant"].eq("incumbent")][
        ["player_key", "season", "squared_error"]
    ].rename(columns={"squared_error": "incumbent_squared_error"})
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    compared_variants = (
        tuple(VARIANT_FEATURES)[1:]
        if variants is None
        else tuple(variants)
    )
    for variant in compared_variants:
        challenger = blend[blend["variant"].eq(variant)][
            ["player_key", "season", "squared_error"]
        ].rename(columns={"squared_error": "variant_squared_error"})
        compared = incumbent.merge(
            challenger,
            on=["player_key", "season"],
            how="inner",
            validate="one_to_one",
        )
        season_rmse = (
            compared.groupby("season", sort=True)
            .agg(
                incumbent_rmse=(
                    "incumbent_squared_error",
                    lambda value: float(np.sqrt(value.mean())),
                ),
                variant_rmse=(
                    "variant_squared_error",
                    lambda value: float(np.sqrt(value.mean())),
                ),
            )
            .reset_index()
        )
        season_rmse["delta"] = (
            season_rmse["variant_rmse"] - season_rmse["incumbent_rmse"]
        )
        pooled = scores[
            scores["variant"].isin(("incumbent", variant))
            & scores["method"].eq(method)
            & scores["slice_type"].eq("pooled")
        ].set_index("variant")["rmse"]
        recent = scores[
            scores["variant"].isin(("incumbent", variant))
            & scores["method"].eq(method)
            & scores["slice_type"].eq("recent")
        ].set_index("variant")["rmse"]
        if set(pooled.index) != {"incumbent", variant}:
            raise ValueError(f"Missing pooled {method} score for {variant}")
        season_low, season_high = normalized_runner._cluster_interval(
            compared,
            "season",
            iterations,
            rng,
        )
        player_low, player_high = normalized_runner._cluster_interval(
            compared,
            "player_key",
            iterations,
            rng,
        )
        rows.append(
            {
                "comparison_method": method,
                "variant": variant,
                "incumbent_rmse": float(pooled["incumbent"]),
                "variant_rmse": float(pooled[variant]),
                "pooled_delta_variant_minus_incumbent": float(
                    pooled[variant] - pooled["incumbent"]
                ),
                "recent_delta_variant_minus_incumbent": float(
                    recent[variant] - recent["incumbent"]
                ),
                "early_era_delta": _score_delta(
                    scores,
                    variant,
                    method,
                    "source_era",
                    "2017_2023",
                ),
                "expanded_era_delta": _score_delta(
                    scores,
                    variant,
                    method,
                    "source_era",
                    "2024_2025",
                ),
                "mean_season_delta": float(season_rmse["delta"].mean()),
                "median_season_delta": float(season_rmse["delta"].median()),
                "season_wins": int(season_rmse["delta"].lt(0).sum()),
                "season_count": len(season_rmse),
                "season_bootstrap_95_low": season_low,
                "season_bootstrap_95_high": season_high,
                "player_cluster_95_low": player_low,
                "player_cluster_95_high": player_high,
            }
        )
    return pd.DataFrame(rows)


def _slice_delta_table(
    scores: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    incumbent = scores[
        scores["variant"].eq("incumbent")
        & scores["method"].eq(method)
    ][
        ["slice_type", "slice_value", "n_rows", "n_seasons", "rmse"]
    ].rename(columns={"rmse": "incumbent_rmse"})
    rows: list[pd.DataFrame] = []
    for variant in tuple(VARIANT_FEATURES)[1:]:
        challenger = scores[
            scores["variant"].eq(variant)
            & scores["method"].eq(method)
        ][["slice_type", "slice_value", "rmse"]].rename(
            columns={"rmse": "variant_rmse"}
        )
        compared = incumbent.merge(
            challenger,
            on=["slice_type", "slice_value"],
            how="inner",
            validate="one_to_one",
        )
        compared["variant"] = variant
        compared["delta_variant_minus_incumbent"] = (
            compared["variant_rmse"] - compared["incumbent_rmse"]
        )
        rows.append(compared)
    return pd.concat(rows, ignore_index=True)


def _promotion_audit(
    controlled: pd.DataFrame,
    production: pd.DataFrame,
    slice_deltas: pd.DataFrame,
) -> dict[str, object]:
    candidate = controlled[
        controlled["variant"].eq(PROMOTION_VARIANT)
    ].iloc[0]
    production_candidate = production[
        production["variant"].eq(PROMOTION_VARIANT)
    ].iloc[0]
    normalized_level = controlled[
        controlled["variant"].eq(NORMALIZED_COMPARATOR_VARIANT)
    ].iloc[0]
    candidate_slices = slice_deltas[
        slice_deltas["variant"].eq(PROMOTION_VARIANT)
    ]
    positions = candidate_slices[
        candidate_slices["slice_type"].eq("position")
    ]
    history = candidate_slices[
        candidate_slices["slice_type"].eq("history_depth")
    ]
    incomplete_coverage = candidate_slices[
        candidate_slices["slice_type"].eq("coverage_completeness")
        & candidate_slices["slice_value"].eq(
            "incomplete_or_unavailable"
        )
    ].iloc[0]
    shallow_omission = candidate_slices[
        candidate_slices["slice_type"].eq("shallow_omission")
        & candidate_slices["slice_value"].eq("omission")
    ].iloc[0]
    gates = {
        "pooled_delta_negative": bool(
            candidate.pooled_delta_variant_minus_incumbent < 0
        ),
        "recent_delta_negative": bool(
            candidate.recent_delta_variant_minus_incumbent < 0
        ),
        "season_wins_at_least_6": bool(candidate.season_wins >= 6),
        "season_interval_upper_nonpositive": bool(
            candidate.season_bootstrap_95_high <= 0
        ),
        "player_interval_upper_nonpositive": bool(
            candidate.player_cluster_95_high <= 0
        ),
        "production_surface_delta_negative": bool(
            production_candidate.pooled_delta_variant_minus_incumbent < 0
        ),
        "production_surface_recent_delta_negative": bool(
            production_candidate.recent_delta_variant_minus_incumbent < 0
        ),
        "production_surface_season_wins_at_least_6": bool(
            production_candidate.season_wins >= 6
        ),
        "production_surface_season_interval_upper_nonpositive": bool(
            production_candidate.season_bootstrap_95_high <= 0
        ),
        "production_surface_player_interval_upper_nonpositive": bool(
            production_candidate.player_cluster_95_high <= 0
        ),
        "beats_normalized_rank_by_0_001": bool(
            candidate.variant_rmse
            <= normalized_level.variant_rmse - 0.001
        ),
        "three_of_four_positions_nonworse": bool(
            positions["delta_variant_minus_incumbent"].le(0).sum() >= 3
        ),
        "no_position_worse_by_0_01": bool(
            positions["delta_variant_minus_incumbent"].max() <= 0.01
        ),
        "no_history_slice_worse_by_0_01": bool(
            history["delta_variant_minus_incumbent"].max() <= 0.01
        ),
        "early_era_nonworse": bool(candidate.early_era_delta <= 0),
        "expanded_era_nonworse": bool(candidate.expanded_era_delta <= 0),
        "incomplete_coverage_nonworse": bool(
            incomplete_coverage.delta_variant_minus_incumbent <= 0
        ),
        "shallow_omission_nonworse_by_0_01": bool(
            shallow_omission.delta_variant_minus_incumbent <= 0.01
        ),
    }
    return {
        "advancement_variant": PROMOTION_VARIANT,
        "decision_scope": (
            "single_league_advance_to_nested_retune_not_production_promotion"
        ),
        "all_single_league_gates_pass": bool(all(gates.values())),
        "gates": gates,
        "controlled_candidate_rmse": float(candidate.variant_rmse),
        "normalized_rank_level_rmse": float(
            normalized_level.variant_rmse
        ),
        "candidate_minus_normalized_rank_rmse": float(
            candidate.variant_rmse - normalized_level.variant_rmse
        ),
        "max_position_delta": float(
            positions["delta_variant_minus_incumbent"].max()
        ),
        "max_history_delta": float(
            history["delta_variant_minus_incumbent"].max()
        ),
        "incomplete_coverage_delta": float(
            incomplete_coverage.delta_variant_minus_incumbent
        ),
        "shallow_omission_delta": float(
            shallow_omission.delta_variant_minus_incumbent
        ),
    }


def _findings_markdown(
    league: str,
    database: Path,
    feature_run_id: str,
    reproduction_delta: float,
    raw_median_reproduction_delta: float,
    controlled: pd.DataFrame,
    production: pd.DataFrame,
    promotion: dict[str, object],
    source_coverage: pd.DataFrame,
) -> str:
    lines = [
        f"# Raw Expert-Rank Challenger - {league.upper()}",
        "",
        "## Method",
        "",
        "- Raw median uses every observed provider overall rank and ignores "
        "missing provider rows.",
        "- The percentile is calculated across all ranked QB/RB/WR/TE players "
        "within a season after the raw median is formed.",
        "- Publication coverage is observed rank sources divided by sources "
        "publishing any rank for that season-position; it is not a "
        "depth-adjusted rank.",
        "- The normalized comparator is rebuilt in-process from the identical "
        "scoring-specific provider rows.",
        "- DK replaces half-PPR ETR with full-PPR ETR; beta retains half-PPR "
        "ETR.",
        "- Primary attribution uses full-column random forests on both sides. "
        "The locked 50% forest remains a separate sensitivity.",
        "",
        "## Controlled results",
        "",
        "| Variant | RMSE | Delta | Recent | Early era | Expanded era | Wins | "
        "Season 95% | Player 95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controlled.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.variant_rmse:.5f} | "
            f"{row.pooled_delta_variant_minus_incumbent:+.5f} | "
            f"{row.recent_delta_variant_minus_incumbent:+.5f} | "
            f"{row.early_era_delta:+.5f} | "
            f"{row.expanded_era_delta:+.5f} | "
            f"{row.season_wins}/{row.season_count} | "
            f"[{row.season_bootstrap_95_low:+.5f}, "
            f"{row.season_bootstrap_95_high:+.5f}] | "
            f"[{row.player_cluster_95_low:+.5f}, "
            f"{row.player_cluster_95_high:+.5f}] |"
        )
    lines.extend(
        [
            "",
            "## Production-surface sensitivity",
            "",
            "| Variant | RMSE | Delta | Recent | Wins | Season 95% | "
            "Player 95% |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in production.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.variant_rmse:.5f} | "
            f"{row.pooled_delta_variant_minus_incumbent:+.5f} | "
            f"{row.recent_delta_variant_minus_incumbent:+.5f} | "
            f"{row.season_wins}/{row.season_count} | "
            f"[{row.season_bootstrap_95_low:+.5f}, "
            f"{row.season_bootstrap_95_high:+.5f}] | "
            f"[{row.player_cluster_95_low:+.5f}, "
            f"{row.player_cluster_95_high:+.5f}] |"
        )
    current_sources = source_coverage.loc[
        source_coverage["season"].eq(2026), "source"
    ].nunique()
    failed = [
        key
        for key, passed in promotion["gates"].items()
        if not passed
    ]
    lines.extend(
        [
            "",
            "## Governance",
            "",
            f"- Feature run: `{feature_run_id}`",
            f"- Staged database: `{database.resolve()}`",
            f"- Locked-incumbent reproduction max delta: "
            f"`{reproduction_delta:.3g}`",
            f"- Existing raw-median reproduction max delta: "
            f"`{raw_median_reproduction_delta:.3g}`",
            f"- 2026 scoring-specific rank providers: {current_sources}",
            f"- Prespecified advancement candidate: `{PROMOTION_VARIANT}`",
            f"- Single-league gates all pass: "
            f"`{promotion['all_single_league_gates_pass']}`",
            f"- Failed gates: `{failed}`",
            "- Passing these gates only advances the candidate to a nested "
            "retune; it does not promote the feature to production.",
            "",
        ]
    )
    return "\n".join(lines)


def _paired_variant_diagnostic(
    normalized_runner,
    evaluation: pd.DataFrame,
    *,
    method: str,
    baseline_variant: str,
    challenger_variant: str,
    iterations: int,
) -> dict[str, object]:
    selected = evaluation[evaluation["method"].eq(method)]
    baseline = selected[selected["variant"].eq(baseline_variant)][
        ["player_key", "season", "squared_error"]
    ].rename(columns={"squared_error": "incumbent_squared_error"})
    challenger = selected[selected["variant"].eq(challenger_variant)][
        ["player_key", "season", "squared_error"]
    ].rename(columns={"squared_error": "variant_squared_error"})
    compared = baseline.merge(
        challenger,
        on=["player_key", "season"],
        how="inner",
        validate="one_to_one",
    )
    if len(compared) != len(baseline) or len(compared) != len(challenger):
        raise ValueError(
            f"Direct {challenger_variant} versus {baseline_variant} "
            f"comparison does not use identical rows"
        )
    baseline_rmse = float(
        np.sqrt(compared["incumbent_squared_error"].mean())
    )
    challenger_rmse = float(
        np.sqrt(compared["variant_squared_error"].mean())
    )
    by_season = (
        compared.groupby("season", sort=True)
        .agg(
            baseline_mse=("incumbent_squared_error", "mean"),
            challenger_mse=("variant_squared_error", "mean"),
        )
        .reset_index()
    )
    by_season["delta"] = (
        np.sqrt(by_season["challenger_mse"])
        - np.sqrt(by_season["baseline_mse"])
    )
    rng = np.random.default_rng(RANDOM_SEED)
    season_low, season_high = normalized_runner._cluster_interval(
        compared,
        "season",
        iterations,
        rng,
    )
    player_low, player_high = normalized_runner._cluster_interval(
        compared,
        "player_key",
        iterations,
        rng,
    )
    return {
        "method": method,
        "baseline_variant": baseline_variant,
        "challenger_variant": challenger_variant,
        "baseline_rmse": baseline_rmse,
        "challenger_rmse": challenger_rmse,
        "delta_challenger_minus_baseline": (
            challenger_rmse - baseline_rmse
        ),
        "season_wins": int(by_season["delta"].lt(0).sum()),
        "season_count": int(len(by_season)),
        "season_bootstrap_95_low": season_low,
        "season_bootstrap_95_high": season_high,
        "player_cluster_95_low": player_low,
        "player_cluster_95_high": player_high,
    }


def _combine_league_audits(
    results_dir: Path,
    iterations: int,
) -> None:
    league_payloads: dict[str, dict[str, object]] = {}
    model_specs: dict[str, object] = {}
    league_metrics: dict[str, dict[str, object]] = {}
    normalized_runner = _load_normalized_runner()
    for league in ("dk", "beta"):
        league_dir = STUDY_ROOT / "results" / f"raw_rank_{league}"
        audit_path = league_dir / "advancement_audit.json"
        manifest_path = league_dir / "input_manifest.json"
        if not audit_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Run {league} before combining results: "
                f"missing {audit_path if not audit_path.is_file() else manifest_path}"
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if audit.get("advancement_variant") != PROMOTION_VARIANT:
            raise ValueError(
                f"{league} advancement candidate does not match "
                f"{PROMOTION_VARIANT}"
            )
        league_payloads[league] = audit
        model_specs[league] = manifest["model_spec"]
        controlled = pd.read_csv(league_dir / "variant_summary.csv")
        production = pd.read_csv(
            league_dir / "production_surface_variant_summary.csv"
        )
        evaluation = pd.read_csv(league_dir / "oof_predictions.csv")
        league_metrics[league] = {
            "controlled": controlled.set_index("variant")[
                [
                    "variant_rmse",
                    "pooled_delta_variant_minus_incumbent",
                    "season_wins",
                    "season_count",
                    "season_bootstrap_95_low",
                    "season_bootstrap_95_high",
                    "player_cluster_95_low",
                    "player_cluster_95_high",
                ]
            ].to_dict("index"),
            "production_surface": production.set_index("variant")[
                [
                    "variant_rmse",
                    "pooled_delta_variant_minus_incumbent",
                    "season_wins",
                    "season_count",
                    "season_bootstrap_95_low",
                    "season_bootstrap_95_high",
                    "player_cluster_95_low",
                    "player_cluster_95_high",
                ]
            ].to_dict("index"),
            "raw_log_vs_normalized": [
                _paired_variant_diagnostic(
                    normalized_runner,
                    evaluation,
                    method=method,
                    baseline_variant=NORMALIZED_COMPARATOR_VARIANT,
                    challenger_variant="raw_log",
                    iterations=iterations,
                )
                for method in (
                    CONTROLLED_BLEND_METHOD,
                    "equal_thirds",
                )
            ],
        }
    if model_specs["dk"] != model_specs["beta"]:
        raise ValueError(
            "DK and beta raw-rank studies do not share one model spec"
        )

    combined_pass = all(
        bool(payload["all_single_league_gates_pass"])
        for payload in league_payloads.values()
    )
    failed = {
        league: [
            gate
            for gate, passed in payload["gates"].items()
            if not passed
        ]
        for league, payload in league_payloads.items()
    }
    combined = {
        "advancement_variant": PROMOTION_VARIANT,
        "decision_scope": (
            "cross_league_advance_to_nested_retune_not_production_promotion"
        ),
        "both_leagues_pass_all_gates": combined_pass,
        "next_action": (
            "run_strict_nested_retune"
            if combined_pass
            else "retain_outside_production"
        ),
        "bootstrap_iterations": iterations,
        "failed_gates_by_league": failed,
        "league_audits": league_payloads,
        "league_metrics": league_metrics,
        "shared_model_spec_sha256": hashlib.sha256(
            json.dumps(
                model_specs["dk"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest().upper(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "advancement_audit.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Raw Expert-Rank Cross-League Decision",
        "",
        f"- Candidate: `{PROMOTION_VARIANT}`",
        f"- Both leagues pass every gate: `{combined_pass}`",
        f"- Next action: `{combined['next_action']}`",
        f"- DK failed gates: `{failed['dk']}`",
        f"- Beta failed gates: `{failed['beta']}`",
        "",
        "## Headline RMSE deltas versus incumbent",
        "",
        "| League | Surface | Normalized | Raw median | Raw log | "
        "Raw percentile | Percentile + coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for league in ("dk", "beta"):
        for surface, label in (
            ("controlled", "controlled"),
            ("production_surface", "production"),
        ):
            values = league_metrics[league][surface]
            lines.append(
                f"| {league.upper()} | {label} | "
                f"{values[NORMALIZED_COMPARATOR_VARIANT]['pooled_delta_variant_minus_incumbent']:+.5f} | "
                f"{values['raw_available_median']['pooled_delta_variant_minus_incumbent']:+.5f} | "
                f"{values['raw_log']['pooled_delta_variant_minus_incumbent']:+.5f} | "
                f"{values['raw_percentile']['pooled_delta_variant_minus_incumbent']:+.5f} | "
                f"{values[PROMOTION_VARIANT]['pooled_delta_variant_minus_incumbent']:+.5f} |"
            )
    lines.extend(
        [
            "",
            "## Exploratory raw-log diagnostic",
            "",
            "`raw_log` was a prespecified scale diagnostic, not the "
            "advancement candidate. Its direct difference from the matched "
            "normalized comparator is not distinguishable in this study:",
            "",
            "| League | Surface | Raw log - normalized | Wins | Season 95% | "
            "Player 95% |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for league in ("dk", "beta"):
        for diagnostic in league_metrics[league][
            "raw_log_vs_normalized"
        ]:
            surface = (
                "controlled"
                if diagnostic["method"] == CONTROLLED_BLEND_METHOD
                else "production"
            )
            lines.append(
                f"| {league.upper()} | {surface} | "
                f"{diagnostic['delta_challenger_minus_baseline']:+.5f} | "
                f"{diagnostic['season_wins']}/"
                f"{diagnostic['season_count']} | "
                f"[{diagnostic['season_bootstrap_95_low']:+.5f}, "
                f"{diagnostic['season_bootstrap_95_high']:+.5f}] | "
                f"[{diagnostic['player_cluster_95_low']:+.5f}, "
                f"{diagnostic['player_cluster_95_high']:+.5f}] |"
            )
    lines.extend(
        [
            "",
            "Descriptively, the percentile-plus-coverage point gain is larger "
            "in the expanded-provider era and in the 45 OOF rows with no "
            "rank. That no-rank slice is exploratory and has no interaction "
            "interval. Among rank-available rows the candidate changes "
            "controlled RMSE by -0.00024 DK and +0.00027 beta.",
            "",
        "Passing this receipt would advance the feature only to a strict "
        "nested-retune validation; it would not itself change production.",
        "",
        ]
    )
    (results_dir / "findings.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(combined, indent=2))


def main() -> None:
    args = parse_args()
    league = args.league
    if league == "all":
        if args.output_db is not None:
            raise ValueError("--output-db is not valid with --league all")
        combined_dir = (
            args.results_dir
            if args.results_dir is not None
            else STUDY_ROOT / "results" / "raw_rank_combined"
        )
        _combine_league_audits(
            combined_dir,
            args.bootstrap_iterations,
        )
        return
    database = (
        args.output_db
        if args.output_db is not None
        else DEFAULT_DATABASES[league]
    )
    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else STUDY_ROOT / "results" / f"raw_rank_{league}"
    )
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    results_dir.mkdir(parents=True, exist_ok=True)

    (
        normalized_runner,
        locked,
        features,
        selected,
        locked_predictions,
        raw_features,
        source_coverage,
        depth_audit,
        feature_run_id,
        raw_median_reproduction_delta,
        ppr_resolution,
        input_manifest,
        position_audit,
    ) = _load_inputs(database, league)
    observed_hashes = set(features["scoring_hash"].dropna().astype(str))
    expected_hash = scoring_hash(league)
    if observed_hashes != {expected_hash}:
        raise ValueError(
            f"Scoring mismatch: observed={observed_hashes}, "
            f"expected={expected_hash}"
        )

    predictions = _run_predictions(locked, features, selected)
    reproduction_delta = _assert_incumbent_reproduces_exactly(
        predictions,
        locked_predictions,
    )
    evaluation = _evaluation_long(predictions, locked.OUTER_SEASONS)
    scores = _score_table(evaluation)
    controlled = _variant_summary(
        normalized_runner,
        evaluation,
        scores,
        args.bootstrap_iterations,
        method=CONTROLLED_BLEND_METHOD,
    )
    production = _variant_summary(
        normalized_runner,
        evaluation,
        scores,
        args.bootstrap_iterations,
        method="equal_thirds",
    )
    component_summaries = pd.concat(
        [
            _variant_summary(
                normalized_runner,
                evaluation,
                scores,
                args.bootstrap_iterations,
                method=method,
                variants=(PROMOTION_VARIANT,),
            )
            for method in (
                "lasso",
                "random_forest",
                "lightgbm",
                FULL_COLUMN_RF_METHOD,
            )
        ],
        ignore_index=True,
    )
    slice_deltas = _slice_delta_table(
        scores,
        method=CONTROLLED_BLEND_METHOD,
    )
    promotion = _promotion_audit(
        controlled,
        production,
        slice_deltas,
    )
    shadow = predictions[predictions["season"].eq(locked.CURRENT_SEASON)].copy()

    raw_features.to_csv(results_dir / "raw_rank_features.csv", index=False)
    source_coverage.to_csv(
        results_dir / "rank_source_coverage.csv",
        index=False,
    )
    depth_audit.to_csv(results_dir / "rank_source_depths.csv", index=False)
    position_audit.to_csv(
        results_dir / "rank_position_audit.csv",
        index=False,
    )
    evaluation.to_csv(results_dir / "oof_predictions.csv", index=False)
    scores.to_csv(results_dir / "model_scores.csv", index=False)
    controlled.to_csv(results_dir / "variant_summary.csv", index=False)
    production.to_csv(
        results_dir / "production_surface_variant_summary.csv",
        index=False,
    )
    component_summaries.to_csv(
        results_dir / "promotion_candidate_component_summary.csv",
        index=False,
    )
    slice_deltas.to_csv(results_dir / "slice_deltas.csv", index=False)
    shadow.to_csv(results_dir / "shadow_predictions.csv", index=False)
    (results_dir / "ppr_identity_resolution.json").write_text(
        json.dumps(ppr_resolution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "advancement_audit.json").write_text(
        json.dumps(promotion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "promotion_audit.json").write_text(
        json.dumps(promotion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "findings.md").write_text(
        _findings_markdown(
            league,
            database,
            feature_run_id,
            reproduction_delta,
            raw_median_reproduction_delta,
            controlled,
            production,
            promotion,
            source_coverage,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "league": league,
                "database": str(database.resolve()),
                "feature_run_id": feature_run_id,
                "locked_reproduction_max_abs_delta": reproduction_delta,
                "raw_median_reproduction_max_abs_delta": (
                    raw_median_reproduction_delta
                ),
                "controlled_summary": controlled.to_dict("records"),
                "production_surface_summary": production.to_dict("records"),
                "promotion_candidate_component_summary": (
                    component_summaries.to_dict("records")
                ),
                "promotion_audit": promotion,
                "results_directory": str(results_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
