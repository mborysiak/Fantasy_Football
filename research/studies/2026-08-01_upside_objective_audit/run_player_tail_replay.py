"""Replay template finalists against causal rare-upside objectives."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "Scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

PHASE_B_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_b_replay.py"
)
SPEC = importlib.util.spec_from_file_location("upside_phase_b_reference", PHASE_B_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import Phase-B replay from {PHASE_B_PATH}")
phase_b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase_b
SPEC.loader.exec_module(phase_b)

receiver_rate = phase_b.receiver_rate
pruning = phase_b.pruning
base = phase_b.base
builder = phase_b.builder

BASELINE_METHOD = phase_b.BASELINE_METHOD
TARGET_COUNTS = phase_b.EXPANDED_TARGET_COUNTS
CORE_COUNTS = {"QB": 18, "RB": 36, "WR": 48, "TE": 18}
LOOKBACK_SEASONS = 5
RESIDUAL_STRIKE = 5.0
PERIODS = {
    "all_2017_2025": (2017, 2025),
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
WEEK_COLS = list(base.WEEK_COLS)
ORIGINAL_EVALUATE_DISTRIBUTION = base.evaluate_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta"), required=True)
    parser.add_argument("--v2-db", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def historical_target_observations(templates: pd.DataFrame) -> pd.DataFrame:
    frames = []
    eligible = templates[templates.template_eligible.eq(1)]
    for (season, pos), group in eligible.groupby(["season", "pos"], sort=True):
        cohort = group.sort_values(
            ["historical_pred_fp_per_game", "avg_pick", "player"],
            ascending=[False, True, True],
        ).head(TARGET_COUNTS[pos])
        observations = pd.DataFrame(
            [base.target_observation(row) for _, row in cohort.iterrows()]
        )
        observations["season"] = int(season)
        observations["pos"] = pos
        frames.append(observations)
    output = pd.concat(frames, ignore_index=True)
    return output[np.isfinite(output.observed_contribution)].reset_index(drop=True)


def causal_thresholds(
    templates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    history = historical_target_observations(templates)
    rows = []
    for season in sorted(targets.season.unique()):
        for pos in builder.POSITIONS:
            prior = history[
                history.pos.eq(pos)
                & history.season.between(
                    int(season) - LOOKBACK_SEASONS,
                    int(season) - 1,
                )
            ]
            if len(prior) < TARGET_COUNTS[pos]:
                raise ValueError(
                    f"Only {len(prior)} prior comparable {pos} rows for {season}."
                )
            rows.append(
                {
                    "season": int(season),
                    "pos": pos,
                    "threshold_history_start": int(prior.season.min()),
                    "threshold_history_end": int(prior.season.max()),
                    "threshold_n": int(len(prior)),
                    "league_winner_contribution_q90": float(
                        prior.observed_contribution.quantile(0.90)
                    ),
                    "league_winner_contribution_q95": float(
                        prior.observed_contribution.quantile(0.95)
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_tail_targets(
    targets: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    output = targets.merge(
        thresholds,
        on=["season", "pos"],
        how="left",
        validate="many_to_one",
    )
    if output.league_winner_contribution_q90.isna().any():
        raise ValueError("Causal tail thresholds are missing from held-out targets.")
    for severity in (90, 95):
        threshold = output[f"league_winner_contribution_q{severity}"]
        output[f"observed_league_winner_q{severity}"] = (
            output.observed_residual.ge(RESIDUAL_STRIKE)
            & output.observed_contribution.ge(threshold)
        ).astype(int)
        output[f"observed_tail_utility_q{severity}"] = np.where(
            output.observed_residual.ge(RESIDUAL_STRIKE),
            np.maximum(output.observed_contribution - threshold, 0.0),
            0.0,
        )
    return output


def evaluate_tail_distribution(target, donors, probabilities, specification):
    evaluation = ORIGINAL_EVALUATE_DISTRIBUTION(
        target,
        donors,
        probabilities,
        specification,
    )
    prediction = float(target.historical_pred_fp_per_game)
    donor_residuals = donors.active_ppg_resid.to_numpy(dtype=float)
    applied_residuals = donor_residuals.copy()
    if specification["center"]:
        applied_residuals -= base.weighted_mean(donor_residuals, probabilities)
    predicted_ppg = np.maximum(prediction + applied_residuals, 0.0)
    profiles = donors[WEEK_COLS].to_numpy(dtype=float)
    weekly_scores = predicted_ppg[:, None] * profiles
    contribution = np.maximum(
        weekly_scores - base.WAIVER_BASELINES[target.pos],
        0.0,
    ).sum(axis=1)

    for severity in (90, 95):
        threshold = float(
            getattr(target, f"league_winner_contribution_q{severity}")
        )
        event = (
            (applied_residuals >= RESIDUAL_STRIKE)
            & (contribution >= threshold)
        )
        utility = np.where(
            applied_residuals >= RESIDUAL_STRIKE,
            np.maximum(contribution - threshold, 0.0),
            0.0,
        )
        observed_utility = float(
            getattr(target, f"observed_tail_utility_q{severity}")
        )
        evaluation.update(
            {
                f"prob_league_winner_q{severity}": float(
                    probabilities[event].sum()
                ),
                f"tail_utility_q{severity}_mean": base.weighted_mean(
                    utility,
                    probabilities,
                ),
                f"tail_utility_q{severity}_crps": base.weighted_crps(
                    utility,
                    probabilities,
                    observed_utility,
                ),
            }
        )
    return evaluation


def safe_event_metrics(outcome: pd.Series, score: pd.Series) -> dict[str, float]:
    y = outcome.to_numpy(dtype=int)
    probability = np.clip(score.to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    event_rate = float(y.mean())
    if y.min() == y.max():
        return {
            "event_rate": event_rate,
            "predicted_rate": float(probability.mean()),
            "calibration_bias": float(probability.mean() - event_rate),
            "brier": float(np.square(probability - y).mean()),
            "log_loss": np.nan,
            "average_precision": np.nan,
            "roc_auc": np.nan,
            "top_decile_lift": np.nan,
            "top_decile_recall": np.nan,
        }
    take = max(1, int(np.ceil(0.10 * len(y))))
    top = np.argsort(-probability, kind="stable")[:take]
    return {
        "event_rate": event_rate,
        "predicted_rate": float(probability.mean()),
        "calibration_bias": float(probability.mean() - event_rate),
        "brier": float(np.square(probability - y).mean()),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "top_decile_lift": float(y[top].mean() / event_rate),
        "top_decile_recall": float(y[top].sum() / y.sum()),
    }


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scoped_frames = {
        "all": predictions,
        "core": predictions[predictions.is_core],
        "depth": predictions[~predictions.is_core],
    }
    for period, (start, end) in PERIODS.items():
        for scope, scope_frame in scoped_frames.items():
            period_frame = scope_frame[scope_frame.season.between(start, end)]
            for method, group in period_frame.groupby("method", sort=True):
                common = {
                    "period": period,
                    "scope": scope,
                    "method": method,
                    "n": int(len(group)),
                    "ppg_crps": float(group.ppg_crps.mean()),
                    "contribution_crps": float(group.contribution_crps.mean()),
                }
                for severity in (90, 95):
                    metrics = safe_event_metrics(
                        group[f"observed_league_winner_q{severity}"],
                        group[f"prob_league_winner_q{severity}"],
                    )
                    rows.append(
                        {
                            **common,
                            "severity": f"q{severity}",
                            **metrics,
                            "tail_utility_crps": float(
                                group[f"tail_utility_q{severity}_crps"].mean()
                            ),
                            "observed_tail_utility_mean": float(
                                group[f"observed_tail_utility_q{severity}"].mean()
                            ),
                            "predicted_tail_utility_mean": float(
                                group[f"tail_utility_q{severity}_mean"].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ppg_crps",
        "contribution_crps",
        "brier",
        "log_loss",
        "average_precision",
        "top_decile_lift",
        "top_decile_recall",
        "tail_utility_crps",
    ]
    keys = ["period", "scope", "severity"]
    baseline = summary[summary.method.eq(BASELINE_METHOD)][keys + metric_cols]
    baseline = baseline.rename(
        columns={metric: f"{metric}_baseline" for metric in metric_cols}
    )
    output = summary.merge(baseline, on=keys, how="left", validate="many_to_one")
    for metric in metric_cols:
        output[f"{metric}_delta"] = (
            output[metric] - output[f"{metric}_baseline"]
        )
    return output


def calibration_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in (
        ("all", predictions),
        ("core", predictions[predictions.is_core]),
    ):
        for method, group in frame.groupby("method", sort=True):
            for severity in (90, 95):
                ordered = group.sort_values(
                    f"prob_league_winner_q{severity}", kind="stable"
                ).copy()
                ordered["probability_bin"] = pd.qcut(
                    np.arange(len(ordered)),
                    q=10,
                    labels=False,
                )
                for probability_bin, bin_frame in ordered.groupby(
                    "probability_bin", sort=True
                ):
                    rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "severity": f"q{severity}",
                            "probability_bin": int(probability_bin) + 1,
                            "n": int(len(bin_frame)),
                            "predicted_rate": float(
                                bin_frame[
                                    f"prob_league_winner_q{severity}"
                                ].mean()
                            ),
                            "observed_rate": float(
                                bin_frame[
                                    f"observed_league_winner_q{severity}"
                                ].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    league = args.league
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else STUDY_DIR / f"results_player_{league}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phase_b.configure_reference_globals()
    builder.set_active_league(league)
    base.builder.LEAGUE = league
    base.evaluate_distribution = evaluate_tail_distribution
    v2_database = (
        args.v2_db.resolve()
        if args.v2_db is not None
        else Path(builder.resolve_v2_database(league=league)).resolve()
    )
    max_season = builder.get_daily_max_template_season()
    rates = receiver_rate.load_receiver_rate_features(v2_database, max_season)
    projections = builder.load_historical_projection_context(
        max_season,
        v2_database=v2_database,
    )
    weekly = builder.load_weekly_points(max_season, league=league)
    templates = builder.build_weekly_templates(projections, weekly, league=league)
    templates = receiver_rate.reattach_template_player_keys(templates, projections)
    templates = receiver_rate.attach_receiver_rate_features(templates, rates)
    forecasts = base.load_production_oos_forecasts(max_season)
    target_templates = base.build_production_oos_target_templates(
        templates,
        forecasts,
    )
    targets = base.build_targets(target_templates)
    targets = targets.sort_values(
        ["season", "pos", "historical_pred_fp_per_game", "avg_pick", "player"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    targets["preseason_pos_rank"] = targets.groupby(["season", "pos"]).cumcount() + 1
    thresholds = causal_thresholds(templates, targets)
    targets = add_tail_targets(targets, thresholds)

    predictions = pruning.run_replay(templates, targets)
    predictions = phase_b.add_target_metadata(predictions, targets)
    tail_columns = [
        "player",
        "pos",
        "season",
        "threshold_history_start",
        "threshold_history_end",
        "threshold_n",
        "league_winner_contribution_q90",
        "league_winner_contribution_q95",
        "observed_league_winner_q90",
        "observed_league_winner_q95",
        "observed_tail_utility_q90",
        "observed_tail_utility_q95",
    ]
    predictions = predictions.merge(
        targets[tail_columns].drop_duplicates(["player", "pos", "season"]),
        on=["player", "pos", "season"],
        how="left",
        validate="many_to_one",
    )
    predictions["is_core"] = predictions.preseason_pos_rank.le(
        predictions.pos.map(CORE_COUNTS)
    )
    for severity in (90, 95):
        predictions[f"league_winner_q{severity}_brier_row"] = np.square(
            predictions[f"prob_league_winner_q{severity}"]
            - predictions[f"observed_league_winner_q{severity}"]
        )

    summary = add_baseline_deltas(summarize_predictions(predictions))
    bins = calibration_bins(predictions)
    predictions.to_csv(results_dir / "target_predictions.csv", index=False)
    thresholds.to_csv(results_dir / "causal_thresholds.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    bins.to_csv(results_dir / "calibration_bins.csv", index=False)
    phase_b.METHOD_METADATA.to_csv(results_dir / "method_metadata.csv", index=False)
    metadata = {
        "league": league,
        "v2_database": str(v2_database),
        "max_template_season": int(max_season),
        "target_rows": int(len(targets)),
        "prediction_rows": int(len(predictions)),
        "method_count": int(predictions.method.nunique()),
        "residual_strike": RESIDUAL_STRIKE,
        "threshold_lookback_seasons": LOOKBACK_SEASONS,
        "threshold_target_counts": TARGET_COUNTS,
        "production_changed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

