"""Fresh role-tiered replay of overall log-ADP matcher challengers."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    "overall_log_adp_role_tier_reference",
    REFERENCE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import role-tier replay from {REFERENCE_PATH}")
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

receiver_rate = reference.receiver_rate
pruning = reference.pruning
base = reference.base
builder = reference.builder

BASELINE_METHOD = "production"
RECENCY_HALF_LIFE = 12.0
EXPANDED_TARGET_COUNTS = {"QB": 48, "RB": 90, "WR": 120, "TE": 48}
LOG_ADP_FEATURE = "match_log_adp_scaled"
LOG_ADP_WEIGHT = 0.50
LOG_ADP_CAP = 300.0
VARIANTS = {
    BASELINE_METHOD: {
        "retain_adp_rank": True,
        "add_log_adp": False,
    },
    "replace_adp_rank_with_log_adp": {
        "retain_adp_rank": False,
        "add_log_adp": True,
    },
    "add_log_adp": {
        "retain_adp_rank": True,
        "add_log_adp": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def add_overall_log_adp(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a fixed-scale overall-pick feature without position ranking."""
    output = frame.copy()
    adp = pd.to_numeric(output["avg_pick"], errors="coerce")
    valid = adp.gt(0) & np.isfinite(adp)
    output[LOG_ADP_FEATURE] = np.nan
    output.loc[valid, LOG_ADP_FEATURE] = (
        np.log1p(adp.loc[valid].clip(lower=1.0, upper=LOG_ADP_CAP))
        / np.log1p(LOG_ADP_CAP + 1.0)
    )
    output["overall_log_adp_available"] = valid.astype(int)
    finite = output.loc[valid, LOG_ADP_FEATURE]
    if finite.empty or not finite.between(0, 1).all():
        raise ValueError("Overall log-ADP transform is outside [0, 1].")
    return output


def build_methods() -> tuple[dict[str, dict], pd.DataFrame]:
    methods: dict[str, dict] = {}
    metadata = []
    for method, variant in VARIANTS.items():
        weights = deepcopy(builder.MATCH_FEATURE_WEIGHTS)
        for position in builder.POSITIONS:
            if not variant["retain_adp_rank"]:
                removed = weights[position].pop("adp_rank_pct", None)
                if removed is None:
                    raise KeyError(f"{position} production weights lack adp_rank_pct")
            if variant["add_log_adp"]:
                weights[position][LOG_ADP_FEATURE] = LOG_ADP_WEIGHT
        methods[method] = {
            "weights": weights,
            "recency_half_life": RECENCY_HALF_LIFE,
            "variant": method,
            "removed_families": (
                ("adp_rank",) if not variant["retain_adp_rank"] else ()
            ),
        }
        metadata.append(
            {
                "method": method,
                **variant,
                "log_adp_weight": (
                    LOG_ADP_WEIGHT if variant["add_log_adp"] else 0.0
                ),
                "log_adp_cap": LOG_ADP_CAP,
                "recency_half_life": RECENCY_HALF_LIFE,
                "feature_count_total": sum(len(weights[pos]) for pos in builder.POSITIONS),
                "total_match_weight": sum(
                    sum(position_weights.values())
                    for position_weights in weights.values()
                ),
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
            values = pd.to_numeric(group[LOG_ADP_FEATURE], errors="coerce")
            available = values.notna()
            rows.append(
                {
                    "population": population,
                    "pos": position,
                    "rows": len(group),
                    "available": int(available.sum()),
                    "missing": int((~available).sum()),
                    "coverage": float(available.mean()),
                    "min_scaled": float(values[available].min()),
                    "median_scaled": float(values[available].median()),
                    "max_scaled": float(values[available].max()),
                }
            )
    return pd.DataFrame(rows)


def add_target_metadata(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    output = reference.add_target_metadata(predictions, targets)
    market = targets[
        ["player", "pos", "season", LOG_ADP_FEATURE]
    ].drop_duplicates(["player", "pos", "season"])
    return output.merge(
        market,
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )


def current_bowers_audit(
    league: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = builder.simulation_dm.read(
        f"""
        SELECT *
        FROM Best_Ball_Weekly_Player_Map
        WHERE year={builder.YEAR}
              AND version='{league}'
              AND player='Brock Bowers'
        """,
        builder.SIMULATION_DB_NAME,
    )
    if current.empty:
        return pd.DataFrame(), pd.DataFrame()
    templates = builder.simulation_dm.read(
        f"""
        SELECT *
        FROM Best_Ball_Weekly_Templates
        WHERE league='{league}'
              AND template_eligible=1
        """,
        builder.SIMULATION_DB_NAME,
    )
    templates = add_overall_log_adp(templates)
    current = current.rename(columns={"year": "season"})
    current = add_overall_log_adp(current)
    target = current.iloc[0]
    target_tuple = next(current.itertuples(index=False))
    templates = templates[templates.pos.eq(target.pos)].reset_index(drop=True)
    summaries = []
    details = []
    residual_column = (
        builder.MANAGED_ACTIVE_PPG_RESID_COLUMN
        if builder.MANAGED_ACTIVE_PPG_RESID_COLUMN in templates.columns
        else "active_ppg_resid"
    )
    for method, specification in METHODS.items():
        pool = pruning.selected_pool(target_tuple, templates, specification)
        donors = pool["donors"].reset_index(drop=True).copy()
        probabilities = pool["probabilities"]
        distances = pool["distances"]
        residuals = pd.to_numeric(
            donors[residual_column], errors="raise"
        ).to_numpy(dtype=float)
        raw_mean = float(np.average(residuals, weights=probabilities))
        centered = residuals - raw_mean
        donor_log_adp = pd.to_numeric(
            donors[LOG_ADP_FEATURE], errors="coerce"
        ).to_numpy(dtype=float)
        target_log_adp = float(target[LOG_ADP_FEATURE])
        for rank, donor in donors.iterrows():
            details.append(
                {
                    "league": league,
                    "method": method,
                    "match_rank": rank + 1,
                    "player": donor.player,
                    "season": int(donor.season),
                    "year_exp": float(donor.year_exp),
                    "avg_pick": float(donor.avg_pick),
                    "template_distance": float(distances[rank]),
                    "template_sample_prob": float(probabilities[rank]),
                    "managed_residual": float(residuals[rank]),
                    "centered_managed_residual": float(centered[rank]),
                    "distance_overall_log_adp": float(
                        abs(donor_log_adp[rank] - target_log_adp)
                    ),
                }
            )
        avg_pick = pd.to_numeric(donors.avg_pick, errors="coerce").to_numpy()
        names = donors.player.astype(str).to_numpy()
        played = pd.to_numeric(
            donors.played_games, errors="raise"
        ).to_numpy(dtype=float)
        summaries.append(
            {
                "league": league,
                "method": method,
                "target_pred_ppg": float(target.pred_fp_per_game),
                "target_avg_pick": float(target.avg_pick),
                "target_log_adp_scaled": target_log_adp,
                "pool_size": len(donors),
                "effective_sample_size": float(
                    1.0 / np.square(probabilities).sum()
                ),
                "top12_weight": float(probabilities[:12].sum()),
                "adp_35_or_earlier_weight": float(
                    probabilities[np.isfinite(avg_pick) & (avg_pick <= 35)].sum()
                ),
                "kelce_weight": float(
                    probabilities[names == "Travis Kelce"].sum()
                ),
                "kelce_gronk_graham_weight": float(
                    probabilities[
                        np.isin(
                            names,
                            ["Travis Kelce", "Rob Gronkowski", "Jimmy Graham"],
                        )
                    ].sum()
                ),
                "expected_played": float(np.average(played, weights=probabilities)),
                "centered_residual_q10": float(
                    base.weighted_quantile(centered, probabilities, 0.10)
                ),
                "centered_residual_q90": float(
                    base.weighted_quantile(centered, probabilities, 0.90)
                ),
                "prob_centered_plus3": float(probabilities[centered >= 3].sum()),
                "prob_centered_plus5": float(probabilities[centered >= 5].sum()),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(details)


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else STUDY_DIR / f"results_phase_b_{league}"
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
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(
        projections,
        weekly,
        league=league,
    )
    templates = add_overall_log_adp(templates)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    target_templates = add_overall_log_adp(target_templates)
    targets = base.build_targets(target_templates)
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = (
        targets.groupby(["season", "pos"]).cumcount() + 1
    )

    predictions = pruning.run_replay(templates, targets)
    predictions = receiver_rate.refresh_row_event_losses(predictions)
    predictions = add_target_metadata(predictions, targets)
    coverage = coverage_audit(templates, targets)
    cohort = (
        targets.groupby(["season", "pos"], as_index=False)
        .agg(
            targets=("player", "size"),
            team_qb1=("qb_team_rank", lambda values: int(values.eq(1).sum())),
        )
    )
    summaries = pd.concat(
        [
            receiver_rate.grouped_period_summaries(predictions, scope)
            for scope in ("all", "wr", "te", "rb", "qb")
        ],
        ignore_index=True,
    )
    bowers_summary, bowers_details = current_bowers_audit(league)

    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    summaries.to_csv(results_dir / "summary_by_period.csv", index=False)
    coverage.to_csv(results_dir / "overall_log_adp_coverage.csv", index=False)
    cohort.to_csv(results_dir / "target_cohort.csv", index=False)
    bowers_summary.to_csv(results_dir / "current_bowers_pool_summary.csv", index=False)
    bowers_details.to_csv(results_dir / "current_bowers_pool_members.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_template_season": int(max_season),
        "expanded_target_counts": EXPANDED_TARGET_COUNTS,
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(len(METHODS)),
        "baseline_method": BASELINE_METHOD,
        "log_adp_feature": LOG_ADP_FEATURE,
        "log_adp_weight": LOG_ADP_WEIGHT,
        "log_adp_cap": LOG_ADP_CAP,
        "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
