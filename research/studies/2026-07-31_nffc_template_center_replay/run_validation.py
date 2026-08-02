"""Strict modern-era NFFC replay of expert versus locked donor centers."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRUNING_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-23_template_feature_pruning"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "nffc_template_center_pruning_reference",
    PRUNING_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import template replay from {PRUNING_PATH}")
pruning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pruning
SPEC.loader.exec_module(pruning)
base = pruning.base
builder = pruning.builder

from Scripts.V2.contracts import scoring_hash
from Scripts.V2.production_cycle import get_production_cycle


LEAGUE = "nffc"
CURRENT_SEASON = 2026
DONOR_START = 2021
ORIGIN_START = 2023
ORIGIN_END = 2025
WEEK_COUNT = 17
RECENCY_HALF_LIFE = 12.0
BOOTSTRAP_SAMPLES = 2_000
PLAYER_BOOTSTRAP_SEED = 20260731
SEASON_BOOTSTRAP_SEED = 20260732
EXPERT_POLICY = "expert_donor_center"
LOCKED_POLICY = "locked_oof_donor_center"
DEFAULT_RESULTS = STUDY_DIR / "results"

LOWER_IS_BETTER = [
    "ppg_crps",
    "contribution_crps",
    "played_crps",
    "plus3_brier_row",
    "plus5_brier_row",
    "impact_brier_row",
    "zero_brier_row",
    "extended_absence_brier_row",
]
CRPS_METRICS = ["ppg_crps", "contribution_crps", "played_crps"]
EVENT_SPECS = [
    ("prob_plus3", "observed_plus3", "plus3"),
    ("prob_plus5", "observed_plus5", "plus5"),
    ("prob_impact", "observed_impact", "impact"),
    (
        "prob_zero_contribution",
        "observed_zero_contribution",
        "zero_contribution",
    ),
    (
        "prob_extended_absence",
        "observed_extended_absence",
        "extended_absence",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-db", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )
    return parser.parse_args()


def configure_replay(season: int) -> None:
    cycle = get_production_cycle(season)
    if season != CURRENT_SEASON:
        raise ValueError(
            f"This frozen study is registered for {CURRENT_SEASON}, not {season}"
        )
    if builder.YEAR != season:
        raise ValueError(
            "Builder season does not match --season. Set "
            f"FF_CURRENT_SEASON={season} before running."
        )
    builder.set_active_league(LEAGUE)
    if builder.WEEK_COUNT != WEEK_COUNT:
        raise ValueError(
            f"NFFC must use {WEEK_COUNT} weeks; found {builder.WEEK_COUNT}"
        )
    if builder.TEMPLATE_SEASON_MIN != DONOR_START:
        raise ValueError(
            "NFFC donor era mismatch: "
            f"{builder.TEMPLATE_SEASON_MIN} != {DONOR_START}"
        )
    if cycle.weekly_horizons[LEAGUE] != WEEK_COUNT:
        raise ValueError("Approved cycle NFFC horizon is inconsistent")
    if cycle.template_min_seasons[LEAGUE] != DONOR_START:
        raise ValueError("Approved cycle NFFC donor era is inconsistent")

    base.builder.LEAGUE = LEAGUE
    base.ORIGIN_START = ORIGIN_START
    base.CORE_ORIGIN_START = ORIGIN_START
    base.RECENT_ORIGIN_START = ORIGIN_START
    base.WEEK_COLS = [
        f"managed_week_{week}" for week in range(1, WEEK_COUNT + 1)
    ]
    pruning.METHODS = {
        "production": {
            "weights": deepcopy(builder.MATCH_FEATURE_WEIGHTS),
            "recency_half_life": RECENCY_HALF_LIFE,
            "variant": "production",
            "removed_families": (),
        }
    }
    pruning.BASELINE_METHOD = "production"
    pruning.PERIODS = {
        "modern_2023_2025": (ORIGIN_START, ORIGIN_END),
    }


def load_v2_projection_context(
    v2_database: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = [
        "player_key",
        "display_name",
        "season",
        "position",
        "team",
        "feature_cutoff_season",
        "preseason_source_season",
        "league",
        "scoring_hash",
        "run_id",
        "expert_points_median",
        "expert_ppg_team_game_median",
        "expert_ppg_team_game_std",
        "expert_points_iqr",
        "projected_pass_point_share",
        "projected_rush_point_share",
        "projected_receiving_point_share",
        "team_qb1_ppg",
    ]
    with sqlite3.connect(v2_database) as connection:
        context = pd.read_sql_query(
            "SELECT "
            + ", ".join(f'"{column}"' for column in columns)
            + ' FROM "player_season_features" '
            + "WHERE season BETWEEN ? AND ?",
            connection,
            params=(DONOR_START, ORIGIN_END),
        )
    scored_context = builder.load_v2_scored_projection_context(
        v2_database,
        min_season=DONOR_START,
        max_season=ORIGIN_END,
    )
    context = context.merge(
        scored_context[
            ["player_key", "season", "team_qb1_pass_points"]
        ],
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    if context.empty:
        raise ValueError("NFFC V2 feature context is empty")
    if context.duplicated(["player_key", "season"]).any():
        raise ValueError("NFFC V2 feature context has duplicate player seasons")
    expected_hash = scoring_hash(LEAGUE)
    if set(context["league"].dropna().astype(str)) != {LEAGUE}:
        raise ValueError("V2 feature context contains a non-NFFC league")
    if set(context["scoring_hash"].dropna().astype(str)) != {expected_hash}:
        raise ValueError("V2 feature context scoring hash is not NFFC")
    if not context["feature_cutoff_season"].eq(
        context["season"] - 1
    ).all():
        raise ValueError("V2 feature context violates the prior-season cutoff")
    if not context["preseason_source_season"].eq(context["season"]).all():
        raise ValueError("V2 feature context is not same-season preseason data")
    run_ids = sorted(context["run_id"].dropna().astype(str).unique())
    if len(run_ids) != 1:
        raise ValueError(f"Expected one V2 feature run; found {run_ids}")
    receipt = {
        "feature_run_id": run_ids[0],
        "league": LEAGUE,
        "scoring_hash": expected_hash,
        "rows": int(len(context)),
        "min_season": int(context["season"].min()),
        "max_season": int(context["season"].max()),
    }
    return context, receipt


def load_locked_centers(
    v2_database: Path,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cycle = get_production_cycle(season)
    with sqlite3.connect(v2_database) as connection:
        centers = pd.read_sql_query(
            """
            SELECT lock_version,
                   model_run_id,
                   player_key,
                   CAST(season AS INTEGER) season,
                   position locked_position,
                   historical_pred_fp_per_game locked_oof_ppg,
                   template_center_available
            FROM locked_template_handoff
            WHERE season BETWEEN ? AND ?
            """,
            connection,
            params=(DONOR_START, ORIGIN_END),
        )
    if centers.empty:
        raise ValueError("Locked NFFC template handoff is empty")
    if centers.duplicated(["player_key", "season"]).any():
        raise ValueError("Locked NFFC centers have duplicate player seasons")
    expected_version = cycle.locked_versions[LEAGUE]
    versions = sorted(centers["lock_version"].dropna().astype(str).unique())
    if versions != [expected_version]:
        raise ValueError(
            f"Locked NFFC version mismatch: {versions} != {expected_version}"
        )
    run_ids = sorted(centers["model_run_id"].dropna().astype(str).unique())
    if len(run_ids) != 1:
        raise ValueError(f"Expected one locked model run; found {run_ids}")
    available = pd.to_numeric(
        centers["template_center_available"], errors="coerce"
    )
    if available.isna().any() or not available.isin([0, 1]).all():
        raise ValueError("Locked center availability is invalid")
    centers["template_center_available"] = available.astype(int)
    inconsistent = centers["template_center_available"].ne(
        centers["locked_oof_ppg"].notna().astype(int)
    )
    if inconsistent.any():
        raise ValueError("Locked center availability disagrees with its value")
    receipt = {
        "lock_version": expected_version,
        "model_run_id": run_ids[0],
        "rows": int(len(centers)),
        "available_rows": int(
            centers["template_center_available"].sum()
        ),
    }
    return centers, receipt


def scoring_context_audit(
    projections: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = projections.copy()
    required_production_columns = {
        "avg_proj_points",
        "preseason_proj_ppg",
        "avg_proj_pass_points",
        "avg_proj_rush_points",
        "avg_proj_rec_points",
        "qb_avg_proj_pass_points",
        "std_proj_points",
        "historical_pred_fp_per_game",
        "historical_projection_source",
        "historical_center_policy",
        "projection_context_source",
        "projection_context_scoring_hash",
        "projection_context_run_id",
        "projection_context_avg_proj_points_delta",
        "model_input_avg_proj_points",
        "model_input_preseason_proj_ppg",
        "model_input_avg_proj_pass_points",
        "model_input_avg_proj_rush_points",
        "model_input_avg_proj_rec_points",
        "model_input_qb_avg_proj_pass_points",
        "model_input_std_proj_points",
        *builder.MATCH_OUTPUT_COLS,
    }
    missing_production_columns = sorted(
        required_production_columns - set(output.columns)
    )
    if missing_production_columns:
        raise ValueError(
            "Historical loader did not apply the production NFFC scoring "
            f"context: {missing_production_columns}"
        )

    context_columns = [
        "player_key",
        "season",
        "position",
        "feature_cutoff_season",
        "preseason_source_season",
        "league",
        "scoring_hash",
        "run_id",
        "expert_points_median",
        "expert_ppg_team_game_median",
        "expert_ppg_team_game_std",
        "expert_points_iqr",
        "projected_pass_point_share",
        "projected_rush_point_share",
        "projected_receiving_point_share",
        "team_qb1_ppg",
        "team_qb1_pass_points",
    ]
    output = output.merge(
        context[context_columns].rename(
            columns={"position": "v2_context_position"}
        ),
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
        indicator="_v2_context_join",
    )
    missing_context = output["_v2_context_join"].ne("both")
    if missing_context.any():
        preview = output.loc[
            missing_context,
            ["player_key", "player", "pos", "season"],
        ].head(20)
        raise ValueError(
            "Historical rows lack NFFC V2 scoring context: "
            f"{preview.to_dict('records')}"
        )

    position_mismatch = output["pos"].ne(output["v2_context_position"])
    governed = pd.Series(False, index=output.index, dtype=bool)
    for mismatch_key in (
        builder.GOVERNED_V2_TEMPLATE_CENTER_POSITION_MISMATCHES
    ):
        player_key, mismatch_season, template_pos, locked_pos = mismatch_key
        governed |= (
            output["player_key"].astype(str).eq(player_key)
            & output["season"].eq(mismatch_season)
            & output["pos"].eq(template_pos)
            & output["v2_context_position"].eq(locked_pos)
        )
    unexpected_mismatch = position_mismatch & ~governed
    if unexpected_mismatch.any():
        preview = output.loc[
            unexpected_mismatch,
            [
                "player_key",
                "player",
                "season",
                "pos",
                "v2_context_position",
            ],
        ].head(20)
        raise ValueError(
            "Unexpected V2 context position mismatch: "
            f"{preview.to_dict('records')}"
        )

    total = pd.to_numeric(output["expert_points_median"], errors="coerce")
    ppg = pd.to_numeric(
        output["expert_ppg_team_game_median"], errors="coerce"
    )
    invalid_total = total.isna() | ~np.isfinite(total) | total.lt(0)
    invalid_ppg = ppg.isna() | ~np.isfinite(ppg) | ppg.lt(0)
    if invalid_total.any() or invalid_ppg.any():
        raise ValueError(
            "Historical NFFC V2 context lacks a valid expert point center"
        )
    if not np.allclose(total / WEEK_COUNT, ppg, atol=1e-10):
        raise ValueError("NFFC expert total does not reconcile to 17-week PPG")

    share_columns = [
        "projected_pass_point_share",
        "projected_rush_point_share",
        "projected_receiving_point_share",
    ]
    shares = output[share_columns].apply(pd.to_numeric, errors="coerce")
    missing_share = shares.isna().any(axis=1) & total.gt(0)
    if missing_share.any():
        preview = output.loc[
            missing_share,
            ["player_key", "player", "pos", "season", *share_columns],
        ].head(20)
        raise ValueError(
            "Positive-point historical context lacks component shares: "
            f"{preview.to_dict('records')}"
        )
    shares = shares.fillna(0.0)
    if ((shares < -1e-10) | (shares > 1 + 1e-10)).any().any():
        raise ValueError("NFFC projection component shares are outside [0, 1]")
    share_sum = shares.sum(axis=1)
    if not np.allclose(
        share_sum[total.gt(0)],
        1.0,
        atol=1e-8,
    ):
        raise ValueError("NFFC projection component shares do not sum to one")

    expected_components = {
        "avg_proj_pass_points": (
            total * shares["projected_pass_point_share"]
        ),
        "avg_proj_rush_points": (
            total * shares["projected_rush_point_share"]
        ),
        "avg_proj_rec_points": (
            total * shares["projected_receiving_point_share"]
        ),
    }
    expected_values = {
        "avg_proj_points": total,
        "preseason_proj_ppg": ppg,
        "historical_pred_fp_per_game": ppg,
        "std_proj_points": (
            pd.to_numeric(
                output["expert_ppg_team_game_std"],
                errors="coerce",
            ).fillna(0.0)
            * WEEK_COUNT
        ),
        **expected_components,
    }
    for column, expected in expected_values.items():
        actual = pd.to_numeric(output[column], errors="coerce")
        if not np.allclose(
            actual,
            expected,
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                f"Production NFFC scoring context disagrees with {column}"
            )

    expected_qb_context = pd.to_numeric(
        output["team_qb1_pass_points"], errors="coerce"
    )
    if not np.allclose(
        pd.to_numeric(
            output["qb_avg_proj_pass_points"],
            errors="coerce",
        ),
        expected_qb_context,
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    ):
        raise ValueError(
            "Production NFFC team-QB context is not scoring-matched"
        )

    expected_hash = scoring_hash(LEAGUE)
    expected_run_ids = set(context["run_id"].dropna().astype(str))
    provenance_checks = {
        "projection_context_source": {
            "v2_nffc_scoring_matched_preseason"
        },
        "projection_context_scoring_hash": {expected_hash},
        "projection_context_run_id": expected_run_ids,
        "historical_projection_source": {
            "v2_nffc_expert_consensus"
        },
        "historical_center_policy": {
            "nffc_scored_expert_consensus"
        },
    }
    for column, expected in provenance_checks.items():
        observed = set(output[column].dropna().astype(str))
        if observed != expected:
            raise ValueError(
                f"Production NFFC {column} mismatch: "
                f"{sorted(observed)} != {sorted(expected)}"
            )

    expected_delta = total - pd.to_numeric(
        output["model_input_avg_proj_points"],
        errors="coerce",
    )
    if not np.allclose(
        pd.to_numeric(
            output["projection_context_avg_proj_points_delta"],
            errors="coerce",
        ),
        expected_delta,
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    ):
        raise ValueError(
            "Production NFFC scoring-context delta receipt is inconsistent"
        )

    audit = (
        output.assign(
            point_delta=(
                output["avg_proj_points"]
                - pd.to_numeric(
                    output["model_input_avg_proj_points"],
                    errors="coerce",
                )
            ),
            ppg_delta=(
                output["historical_pred_fp_per_game"]
                - pd.to_numeric(
                    output["model_input_avg_proj_points"],
                    errors="coerce",
                )
                / WEEK_COUNT
            ),
            position_mismatch=position_mismatch.astype(int),
        )
        .groupby(["season", "pos"], as_index=False)
        .agg(
            rows=("player_key", "size"),
            v2_context_rows=("_v2_context_join", lambda values: int(
                values.eq("both").sum()
            )),
            governed_position_mismatches=("position_mismatch", "sum"),
            expert_center_rows=("expert_points_median", "count"),
            component_complete_rows=(
                "projected_rush_point_share",
                "count",
            ),
            mean_point_delta=("point_delta", "mean"),
            mean_abs_point_delta=("point_delta", lambda values: float(
                values.abs().mean()
            )),
            mean_ppg_delta=("ppg_delta", "mean"),
            mean_abs_ppg_delta=("ppg_delta", lambda values: float(
                values.abs().mean()
            )),
        )
    )
    output = output.drop(columns=["_v2_context_join"])
    return output, audit


def build_locked_target_templates(
    expert_templates: pd.DataFrame,
    centers: pd.DataFrame,
) -> pd.DataFrame:
    target_templates = expert_templates[
        expert_templates["season"].between(ORIGIN_START, ORIGIN_END)
    ].copy()
    target_templates = target_templates.rename(
        columns={
            "historical_pred_fp_per_game": "expert_historical_pred_fp_per_game",
            "historical_projection_source": "expert_historical_projection_source",
        }
    ).merge(
        centers[
            [
                "player_key",
                "season",
                "locked_position",
                "locked_oof_ppg",
                "template_center_available",
            ]
        ],
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    missing = (
        target_templates["template_center_available"].ne(1)
        | target_templates["locked_oof_ppg"].isna()
    )
    if missing.any():
        preview = target_templates.loc[
            missing,
            ["player_key", "player", "pos", "season"],
        ].head(20)
        raise ValueError(
            "Held-out targets lack locked OOF centers: "
            f"{preview.to_dict('records')}"
        )
    target_templates["historical_pred_fp_per_game"] = pd.to_numeric(
        target_templates["locked_oof_ppg"], errors="raise"
    )
    target_templates["historical_projection_source"] = (
        "locked_nffc_oof_target_center"
    )
    target_templates = target_templates.drop(
        columns=["projection_rank_pct", "projection_decile", "projection_tier"],
        errors="ignore",
    )
    target_templates = builder.add_projection_buckets(
        target_templates,
        value_col="historical_pred_fp_per_game",
        group_cols=["season", "pos"],
    )
    target_templates["match_projection_rank_pct"] = target_templates[
        "projection_rank_pct"
    ]
    target_templates["match_projection_ppg_scaled"] = (
        target_templates["historical_pred_fp_per_game"]
        .clip(lower=0)
        .div(builder.PROJECTION_PPG_SCALE)
    )
    target_templates["projection_x_exp"] = (
        target_templates["match_projection_rank_pct"]
        * target_templates["year_exp_scaled"]
    )
    target_templates["market_projection_gap"] = (
        target_templates["adp_rank_pct"]
        - target_templates["match_projection_rank_pct"]
    )
    return target_templates


def reattach_template_player_keys(
    templates: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Restore the exact projection key dropped by template construction."""

    if "player_key" in templates:
        return templates
    key_map = projections[
        ["player_key", "player", "pos", "season"]
    ].drop_duplicates()
    if key_map.duplicated(["player", "pos", "season"]).any():
        raise ValueError(
            "Projection keys are ambiguous on the weekly-template grain"
        )
    output = templates.merge(
        key_map,
        on=["player", "pos", "season"],
        how="left",
        validate="one_to_one",
    )
    if output["player_key"].isna().any():
        preview = output.loc[
            output["player_key"].isna(),
            ["player", "pos", "season"],
        ].head(20)
        raise ValueError(
            "Weekly templates lost canonical projection keys: "
            f"{preview.to_dict('records')}"
        )
    return output


def build_center_arms(
    expert_templates: pd.DataFrame,
    centers: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    expert = expert_templates.copy()
    expert["donor_center_policy"] = EXPERT_POLICY
    expert["donor_point_center"] = expert[
        "historical_pred_fp_per_game"
    ]

    locked = expert_templates.merge(
        centers[
            [
                "player_key",
                "season",
                "locked_oof_ppg",
                "template_center_available",
            ]
        ],
        on=["player_key", "season"],
        how="left",
        validate="one_to_one",
    )
    missing = (
        locked["template_center_available"].ne(1)
        | locked["locked_oof_ppg"].isna()
    )
    if missing.any():
        preview = locked.loc[
            missing,
            ["player_key", "player", "pos", "season"],
        ].head(20)
        raise ValueError(
            "Historical donors lack locked OOF centers: "
            f"{preview.to_dict('records')}"
        )
    locked["historical_pred_fp_per_game"] = pd.to_numeric(
        locked["locked_oof_ppg"], errors="raise"
    )
    locked["active_ppg_resid"] = (
        locked["active_ppg"] - locked["historical_pred_fp_per_game"]
    )
    locked["historical_projection_source"] = (
        "locked_nffc_oof_donor_center"
    )
    locked["donor_center_policy"] = LOCKED_POLICY
    locked["donor_point_center"] = locked[
        "historical_pred_fp_per_game"
    ]

    match_columns = sorted(
        set(builder.MATCH_FEATURE_COLS)
        | {"qb_team_rank_bucket"}
    )
    for column in match_columns:
        left = expert[column]
        right = locked[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                equal_nan=True,
            ):
                raise AssertionError(
                    f"Center arms differ on match feature {column}"
                )
        elif not left.fillna("").astype(str).equals(
            right.fillna("").astype(str)
        ):
            raise AssertionError(
                f"Center arms differ on match feature {column}"
            )
    return {
        EXPERT_POLICY: expert,
        LOCKED_POLICY: locked,
    }


def validate_target_counts(targets: pd.DataFrame) -> pd.DataFrame:
    counts = (
        targets.groupby(["season", "pos"], as_index=False)
        .size()
        .rename(columns={"size": "target_rows"})
    )
    counts["expected_rows"] = counts["pos"].map(base.TARGET_COUNTS)
    expected_cells = {
        (season, pos)
        for season in range(ORIGIN_START, ORIGIN_END + 1)
        for pos in builder.POSITIONS
    }
    observed_cells = set(
        counts[["season", "pos"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise ValueError(
            "Held-out target cells are incomplete: "
            f"{sorted(expected_cells - observed_cells)}"
        )
    if not counts["target_rows"].eq(counts["expected_rows"]).all():
        raise ValueError(
            "Held-out target counts do not match the frozen position counts"
        )
    return counts


def run_arm(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    predictions = pruning.run_replay(templates, targets)
    if not predictions["method"].eq("production").all():
        raise AssertionError("Unexpected replay method")
    predictions["center_policy"] = policy
    key_map = targets[["player_key", "player", "pos", "season"]]
    if key_map.duplicated(["player", "pos", "season"]).any():
        raise ValueError("Target display keys are ambiguous")
    predictions = predictions.merge(
        key_map,
        on=["player", "pos", "season"],
        how="left",
        validate="one_to_one",
    )
    if predictions["player_key"].isna().any():
        raise ValueError("Replay predictions lost canonical player keys")
    return predictions


def paired_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["player_key", "player", "pos", "season"]
    fields = keys + [
        *LOWER_IS_BETTER,
        "ppg_bias_row",
        "ppg_80_covered",
        "contribution_80_covered",
        "played_80_covered",
    ]
    frame = predictions.copy()
    frame["ppg_bias_row"] = frame["ppg_mean"] - frame["observed_ppg"]
    expert = frame[frame["center_policy"].eq(EXPERT_POLICY)][fields]
    locked = frame[frame["center_policy"].eq(LOCKED_POLICY)][fields]
    paired = locked.merge(
        expert,
        on=keys,
        suffixes=("_locked", "_expert"),
        validate="one_to_one",
    )
    for metric in fields[len(keys):]:
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_locked"]
            - paired[f"{metric}_expert"]
        )
    return paired


def comparison_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in LOWER_IS_BETTER:
        expert = paired[f"{metric}_expert"]
        locked = paired[f"{metric}_locked"]
        rows.append(
            {
                "metric": metric,
                "n": int(len(paired)),
                "expert": float(expert.mean()),
                "locked": float(locked.mean()),
                "locked_minus_expert": float((locked - expert).mean()),
                "relative_delta": float(
                    (locked.mean() / expert.mean()) - 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def clustered_bootstrap(
    paired: pd.DataFrame,
    *,
    cluster_column: str,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    delta_columns = [f"{metric}_delta" for metric in LOWER_IS_BETTER]
    cluster_values = sorted(paired[cluster_column].astype(str).unique())
    cluster_frames = [
        paired.loc[
            paired[cluster_column].astype(str).eq(cluster),
            delta_columns,
        ].to_numpy(dtype=float)
        for cluster in cluster_values
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(delta_columns)), dtype=float)
    for draw in range(samples):
        sampled = rng.integers(
            0,
            len(cluster_frames),
            size=len(cluster_frames),
        )
        values = np.concatenate(
            [cluster_frames[index] for index in sampled],
            axis=0,
        )
        draws[draw] = values.mean(axis=0)
    output = []
    for index, metric in enumerate(LOWER_IS_BETTER):
        values = draws[:, index]
        output.append(
            {
                "cluster": cluster_column,
                "metric": metric,
                "n_rows": int(len(paired)),
                "n_clusters": int(len(cluster_frames)),
                "samples": int(samples),
                "locked_minus_expert": float(
                    paired[f"{metric}_delta"].mean()
                ),
                "bootstrap_p025": float(np.quantile(values, 0.025)),
                "bootstrap_p975": float(np.quantile(values, 0.975)),
                "probability_locked_better": float(
                    np.mean(values < 0)
                ),
            }
        )
    return pd.DataFrame(output)


def event_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["method"] = frame["center_policy"]
    return pd.concat(
        [
            base.probability_calibration(
                frame,
                probability_column,
                outcome_column,
                label,
            )
            for probability_column, outcome_column, label in EVENT_SPECS
        ],
        ignore_index=True,
    ).rename(columns={"method": "center_policy"})


def pit_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    labels = [f"{start / 10:.1f}-{(start + 1) / 10:.1f}" for start in range(10)]
    for policy, frame in predictions.groupby("center_policy", sort=True):
        for metric in ["ppg", "contribution"]:
            pit = pd.to_numeric(frame[f"{metric}_pit"], errors="raise")
            bins = pd.cut(
                pit,
                bins=np.linspace(0, 1, 11),
                labels=labels,
                include_lowest=True,
            )
            counts = bins.value_counts(sort=False)
            for label, count in counts.items():
                output.append(
                    {
                        "center_policy": policy,
                        "metric": metric,
                        "pit_bin": str(label),
                        "n": int(count),
                        "share": float(count / len(frame)),
                        "uniform_share": 0.10,
                    }
                )
    return pd.DataFrame(output)


def donor_origin_audit(
    arms: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for policy, templates in arms.items():
        eligible = templates[templates["template_eligible"].eq(1)]
        for season in sorted(targets["season"].unique()):
            for pos in builder.POSITIONS:
                donors = eligible[
                    eligible["pos"].eq(pos)
                    & eligible["season"].between(DONOR_START, season - 1)
                ]
                rows.append(
                    {
                        "center_policy": policy,
                        "target_season": int(season),
                        "pos": pos,
                        "eligible_prior_donors": int(len(donors)),
                        "min_donor_season": int(donors["season"].min()),
                        "max_donor_season": int(donors["season"].max()),
                        "future_donor_rows": int(
                            donors["season"].ge(season).sum()
                        ),
                        "week_17_nonnull_rows": int(
                            donors["week_17"].notna().sum()
                        ),
                    }
                )
    audit = pd.DataFrame(rows)
    if audit["eligible_prior_donors"].lt(
        builder.MIN_TEMPLATE_POOL_SIZE
    ).any():
        raise ValueError("A rolling origin lacks the minimum donor pool")
    if audit["future_donor_rows"].sum():
        raise ValueError("A rolling origin contains future donors")
    if not audit["week_17_nonnull_rows"].eq(
        audit["eligible_prior_donors"]
    ).all():
        raise ValueError("A rolling donor pool has missing Week 17 values")
    return audit


def decision_table(
    summaries: pd.DataFrame,
    positions: pd.DataFrame,
    seasons: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    by_policy = summaries.set_index("center_policy")
    expert = by_policy.loc[EXPERT_POLICY]
    locked = by_policy.loc[LOCKED_POLICY]
    player_ppg = bootstrap[
        bootstrap["cluster"].eq("player_key")
        & bootstrap["metric"].eq("ppg_crps")
    ].iloc[0]
    season_ppg = seasons.pivot(
        index="season",
        columns="center_policy",
        values="ppg_crps",
    )
    season_wins = int(
        (
            season_ppg[LOCKED_POLICY]
            < season_ppg[EXPERT_POLICY]
        ).sum()
    )

    position_metrics = positions.set_index(
        ["center_policy", "pos"]
    )[CRPS_METRICS]
    expert_position = position_metrics.loc[EXPERT_POLICY]
    locked_position = position_metrics.loc[LOCKED_POLICY]
    ratios = locked_position / expert_position
    composite = ratios.mean(axis=1) - 1.0
    individual = ratios - 1.0

    gates = [
        (
            "pooled_ppg_crps_improves",
            float(locked["ppg_crps"]) < float(expert["ppg_crps"]),
            float(locked["ppg_crps"] - expert["ppg_crps"]),
            0.0,
        ),
        (
            "player_cluster_ppg_upper_at_or_below_zero",
            float(player_ppg["bootstrap_p975"]) <= 0.0,
            float(player_ppg["bootstrap_p975"]),
            0.0,
        ),
        (
            "ppg_season_wins_at_least_two",
            season_wins >= 2,
            float(season_wins),
            2.0,
        ),
        (
            "contribution_crps_within_0_25pct",
            float(locked["contribution_crps"])
            <= float(expert["contribution_crps"]) * 1.0025,
            float(
                locked["contribution_crps"]
                / expert["contribution_crps"]
                - 1
            ),
            0.0025,
        ),
        (
            "played_crps_within_0_25pct",
            float(locked["played_crps"])
            <= float(expert["played_crps"]) * 1.0025,
            float(locked["played_crps"] / expert["played_crps"] - 1),
            0.0025,
        ),
        (
            "all_coverage_within_one_point",
            all(
                float(locked[f"{metric}_80_coverage"])
                >= float(expert[f"{metric}_80_coverage"]) - 0.01
                for metric in ("ppg", "contribution", "played")
            ),
            float(
                min(
                    locked[f"{metric}_80_coverage"]
                    - expert[f"{metric}_80_coverage"]
                    for metric in ("ppg", "contribution", "played")
                )
            ),
            -0.01,
        ),
        (
            "all_event_briers_within_0_001",
            all(
                float(locked[f"{event}_brier"])
                <= float(expert[f"{event}_brier"]) + 0.001
                for event in (
                    "plus3",
                    "plus5",
                    "impact",
                    "zero",
                    "extended_absence",
                )
            ),
            float(
                max(
                    locked[f"{event}_brier"]
                    - expert[f"{event}_brier"]
                    for event in (
                        "plus3",
                        "plus5",
                        "impact",
                        "zero",
                        "extended_absence",
                    )
                )
            ),
            0.001,
        ),
        (
            "absolute_ppg_bias_within_0_10",
            abs(float(locked["ppg_bias"]))
            <= abs(float(expert["ppg_bias"])) + 0.10,
            float(
                abs(float(locked["ppg_bias"]))
                - abs(float(expert["ppg_bias"]))
            ),
            0.10,
        ),
        (
            "position_composite_within_0_5pct",
            float(composite.max()) <= 0.005,
            float(composite.max()),
            0.005,
        ),
        (
            "position_metric_within_1pct",
            float(individual.max().max()) <= 0.01,
            float(individual.max().max()),
            0.01,
        ),
    ]
    decision = pd.DataFrame(
        [
            {
                "gate": gate,
                "passed": int(passed),
                "observed": observed,
                "threshold": threshold,
            }
            for gate, passed, observed, threshold in gates
        ]
    )
    recommendation = (
        LOCKED_POLICY
        if decision["passed"].eq(1).all()
        else EXPERT_POLICY
    )
    return decision, recommendation


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_findings(
    results_dir: Path,
    summaries: pd.DataFrame,
    comparison: pd.DataFrame,
    positions: pd.DataFrame,
    seasons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: pd.DataFrame,
    recommendation: str,
    runtime_seconds: float,
) -> None:
    summary_focus = summaries[
        [
            "center_policy",
            "n",
            "ppg_crps",
            "ppg_bias",
            "ppg_80_coverage",
            "contribution_crps",
            "contribution_80_coverage",
            "played_crps",
            "played_80_coverage",
            "extended_absence_brier",
        ]
    ]
    comparison_focus = comparison[
        comparison["metric"].isin(CRPS_METRICS)
    ]
    bootstrap_focus = bootstrap[
        bootstrap["metric"].isin(CRPS_METRICS)
    ]
    position_focus = positions[
        [
            "center_policy",
            "pos",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
        ]
    ]
    season_focus = seasons[
        [
            "center_policy",
            "season",
            "ppg_crps",
            "contribution_crps",
            "played_crps",
        ]
    ]
    conclusion = (
        "Promote the locked OOF donor center for the NFFC candidate."
        if recommendation == LOCKED_POLICY
        else "Retain the scoring-matched expert donor center; do not promote "
        "the locked OOF donor center."
    )
    text = f"""# NFFC Template Center Replay Findings

## Conclusion

{conclusion}

The recommendation follows the prespecified gate without post-hoc weighting.
All target centers are the same locked OOF NFFC forecasts, and both arms use
identical scoring-matched V2 match context and donor pools.

## Pooled calibration

{markdown_table(summary_focus)}

## Locked minus expert

Negative CRPS deltas favor the locked donor center.

{markdown_table(comparison_focus)}

## Clustered uncertainty

{markdown_table(bootstrap_focus)}

## Position safety

{markdown_table(position_focus)}

## Season consistency

{markdown_table(season_focus)}

## Prespecified gates

{markdown_table(decision)}

## Scope

- Target seasons: 2023-2025.
- Donors: 2021 through the season immediately before each target.
- Horizon: 17 weeks.
- Positions: QB, RB, WR, TE.
- Contribution uses the inherited managed-auction replacement baselines and is
  secondary to PPG and played-games calibration.
- Three target-season clusters are not enough for strong season-level
  asymptotics; player-cluster uncertainty and directional safety are reported.
- Production code and live databases were not changed.

Runtime: {runtime_seconds:.1f} seconds.
"""
    (results_dir / "findings.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    v2_database = args.v2_db.resolve()
    results_dir = args.results_dir.resolve()
    if not v2_database.is_file():
        raise FileNotFoundError(f"NFFC V2 database not found: {v2_database}")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    configure_replay(args.season)
    context, context_receipt = load_v2_projection_context(v2_database)
    centers, center_receipt = load_locked_centers(
        v2_database,
        args.season,
    )

    max_season = min(
        ORIGIN_END,
        builder.get_daily_max_template_season(),
    )
    if max_season != ORIGIN_END:
        raise ValueError(
            f"Expected weekly outcomes through {ORIGIN_END}; found {max_season}"
        )
    print(
        f"Building NFFC {DONOR_START}-{max_season} templates with "
        f"{WEEK_COUNT} weeks",
        flush=True,
    )
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
    )
    projections, context_audit = scoring_context_audit(
        projections,
        context,
    )
    weekly = builder.load_weekly_points(max_season, league=LEAGUE)
    expert_templates = builder.build_weekly_templates(
        projections,
        weekly,
        league=LEAGUE,
    )
    expert_templates = reattach_template_player_keys(
        expert_templates,
        projections,
    )
    if not expert_templates["season"].between(
        DONOR_START,
        ORIGIN_END,
    ).all():
        raise ValueError("Template bank violates the NFFC donor-era contract")
    for week in range(1, WEEK_COUNT + 1):
        for prefix in ("week", "managed_week", "played_week"):
            column = f"{prefix}_{week}"
            if column not in expert_templates:
                raise ValueError(f"Template bank lacks {column}")
            if expert_templates[column].isna().any():
                raise ValueError(f"Template bank has null {column}")

    target_templates = build_locked_target_templates(
        expert_templates,
        centers,
    )
    targets = base.build_targets(target_templates)
    target_counts = validate_target_counts(targets)
    arms = build_center_arms(expert_templates, centers)
    donor_audit = donor_origin_audit(arms, targets)

    prediction_frames = []
    for policy in (EXPERT_POLICY, LOCKED_POLICY):
        print(f"Running {policy}", flush=True)
        prediction_frames.append(
            run_arm(arms[policy], targets, policy)
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_predictions = len(targets) * 2
    if len(predictions) != expected_predictions:
        raise AssertionError(
            f"Expected {expected_predictions} predictions; "
            f"found {len(predictions)}"
        )

    distance_check = predictions.pivot(
        index=["player_key", "pos", "season"],
        columns="center_policy",
        values=["pool_size", "min_template_distance", "median_template_distance"],
    )
    for metric in (
        "pool_size",
        "min_template_distance",
        "median_template_distance",
    ):
        if not np.allclose(
            distance_check[(metric, EXPERT_POLICY)],
            distance_check[(metric, LOCKED_POLICY)],
        ):
            raise AssertionError(
                f"Center arms selected different pools for {metric}"
            )

    summaries = pruning.grouped_summary(
        predictions,
        ["center_policy"],
    )
    positions = pruning.grouped_summary(
        predictions,
        ["center_policy", "pos"],
    )
    seasons = pruning.grouped_summary(
        predictions,
        ["center_policy", "season"],
    )
    paired = paired_predictions(predictions)
    comparison = comparison_summary(paired)
    bootstrap = pd.concat(
        [
            clustered_bootstrap(
                paired,
                cluster_column="player_key",
                samples=args.bootstrap_samples,
                seed=PLAYER_BOOTSTRAP_SEED,
            ),
            clustered_bootstrap(
                paired,
                cluster_column="season",
                samples=args.bootstrap_samples,
                seed=SEASON_BOOTSTRAP_SEED,
            ),
        ],
        ignore_index=True,
    )
    calibration = event_calibration(predictions)
    pit = pit_calibration(predictions)
    decision, recommendation = decision_table(
        summaries,
        positions,
        seasons,
        bootstrap,
    )
    runtime_seconds = time.perf_counter() - started

    center_audit = (
        expert_templates[
            [
                "player_key",
                "player",
                "pos",
                "season",
                "historical_pred_fp_per_game",
            ]
        ]
        .rename(
            columns={
                "historical_pred_fp_per_game": "expert_donor_center"
            }
        )
        .merge(
            centers[
                ["player_key", "season", "locked_oof_ppg"]
            ],
            on=["player_key", "season"],
            how="left",
            validate="one_to_one",
        )
    )
    center_audit["locked_minus_expert"] = (
        center_audit["locked_oof_ppg"]
        - center_audit["expert_donor_center"]
    )
    center_summary = (
        center_audit.groupby(["season", "pos"], as_index=False)
        .agg(
            rows=("player_key", "size"),
            mean_expert_center=("expert_donor_center", "mean"),
            mean_locked_center=("locked_oof_ppg", "mean"),
            mean_locked_minus_expert=("locked_minus_expert", "mean"),
            mean_abs_locked_minus_expert=(
                "locked_minus_expert",
                lambda values: float(values.abs().mean()),
            ),
        )
    )

    predictions.to_csv(
        results_dir / "target_predictions.csv",
        index=False,
    )
    paired.to_csv(
        results_dir / "paired_target_deltas.csv",
        index=False,
    )
    summaries.to_csv(
        results_dir / "summary_overall.csv",
        index=False,
    )
    positions.to_csv(
        results_dir / "summary_by_position.csv",
        index=False,
    )
    seasons.to_csv(
        results_dir / "summary_by_season.csv",
        index=False,
    )
    comparison.to_csv(
        results_dir / "comparison_summary.csv",
        index=False,
    )
    bootstrap.to_csv(
        results_dir / "clustered_bootstrap.csv",
        index=False,
    )
    calibration.to_csv(
        results_dir / "event_calibration.csv",
        index=False,
    )
    pit.to_csv(
        results_dir / "pit_calibration.csv",
        index=False,
    )
    context_audit.to_csv(
        results_dir / "scoring_context_audit.csv",
        index=False,
    )
    center_summary.to_csv(
        results_dir / "center_distribution_audit.csv",
        index=False,
    )
    target_counts.to_csv(
        results_dir / "target_count_audit.csv",
        index=False,
    )
    donor_audit.to_csv(
        results_dir / "donor_origin_audit.csv",
        index=False,
    )
    decision.to_csv(
        results_dir / "decision_gates.csv",
        index=False,
    )
    metadata = {
        "league": LEAGUE,
        "current_season": int(args.season),
        "v2_database": str(v2_database),
        "feature_context": context_receipt,
        "locked_centers": center_receipt,
        "week_count": WEEK_COUNT,
        "donor_start": DONOR_START,
        "origin_start": ORIGIN_START,
        "origin_end": ORIGIN_END,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "center_policies": [EXPERT_POLICY, LOCKED_POLICY],
        "match_feature_count": {
            position: len(builder.MATCH_FEATURE_WEIGHTS[position])
            for position in builder.POSITIONS
        },
        "recency_half_life": RECENCY_HALF_LIFE,
        "minimum_pool_size": builder.MIN_TEMPLATE_POOL_SIZE,
        "maximum_pool_size": builder.MAX_TEMPLATE_POOL_SIZE,
        "maximum_sample_probability": (
            builder.TEMPLATE_MAX_SAMPLE_PROBABILITY
        ),
        "bootstrap_samples": int(args.bootstrap_samples),
        "recommendation": recommendation,
        "all_decision_gates_passed": bool(
            decision["passed"].eq(1).all()
        ),
        "runtime_seconds": runtime_seconds,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_findings(
        results_dir,
        summaries,
        comparison,
        positions,
        seasons,
        bootstrap,
        decision,
        recommendation,
        runtime_seconds,
    )

    print(
        summaries[
            [
                "center_policy",
                "ppg_crps",
                "contribution_crps",
                "played_crps",
                "ppg_bias",
                "ppg_80_coverage",
                "played_80_coverage",
            ]
        ]
        .round(6)
        .to_string(index=False),
        flush=True,
    )
    print(
        f"Recommendation: {recommendation}; "
        f"gates passed={int(decision['passed'].sum())}/{len(decision)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
