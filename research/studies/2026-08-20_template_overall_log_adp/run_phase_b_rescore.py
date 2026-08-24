"""Apply the frozen role-tier policy to the overall log-ADP replay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = (
    STUDY_DIR.parent
    / "2026-07-31_template_role_tiered_validation"
    / "run_phase_b_rescore.py"
)
SPEC = importlib.util.spec_from_file_location(
    "overall_log_adp_role_tier_rescore_reference",
    REFERENCE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import role-tier scorer from {REFERENCE_PATH}")
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)
phase_a = reference.phase_a

RESULTS = STUDY_DIR / "results_phase_b"
SOURCES = (
    phase_a.Source(
        STUDY_DIR.name,
        "results_phase_b_dk",
        "dk",
        "production",
        "fresh_expanded",
    ),
    phase_a.Source(
        STUDY_DIR.name,
        "results_phase_b_beta",
        "beta",
        "production",
        "fresh_expanded",
    ),
)
PLAYER_BOOTSTRAP_SAMPLES = 5_000
PLAYER_BOOTSTRAP_SEED = 20260820


def player_cluster_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PLAYER_BOOTSTRAP_SEED)
    output = []
    for source in SOURCES:
        experiment = source.study + "__" + source.result_dir
        source_frame = predictions[predictions.experiment.eq(experiment)].copy()
        source_frame = source_frame[
            phase_a.tier_masks(source_frame)["core_main"]
        ]
        for period in ("development_2017_2022", "temporal_2023_2025"):
            start, end = phase_a.PERIODS[period]
            scoped = source_frame[source_frame.season.between(start, end)]
            keys = ["player", "pos", "season"]
            fields = keys + list(phase_a.METRICS)
            baseline = scoped[scoped.method.eq(source.baseline)][fields]
            for method in sorted(scoped.method.unique()):
                if method == source.baseline:
                    continue
                candidate = scoped[scoped.method.eq(method)][fields]
                paired = candidate.merge(
                    baseline,
                    on=keys,
                    suffixes=("_candidate", "_baseline"),
                    validate="one_to_one",
                )
                paired["player_cluster"] = paired.pos + "|" + paired.player
                clusters = sorted(paired.player_cluster.unique())
                for metric in ("ppg_crps", "contribution_crps"):
                    paired["delta"] = (
                        paired[f"{metric}_candidate"]
                        - paired[f"{metric}_baseline"]
                    )
                    cluster_stats = (
                        paired.groupby("player_cluster").delta.agg(["sum", "size"])
                    )
                    sums = cluster_stats["sum"].to_numpy(dtype=float)
                    sizes = cluster_stats["size"].to_numpy(dtype=float)
                    sampled = rng.integers(
                        0,
                        len(clusters),
                        size=(PLAYER_BOOTSTRAP_SAMPLES, len(clusters)),
                    )
                    draws = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
                    output.append(
                        {
                            "experiment": experiment,
                            "league": source.league,
                            "candidate_method": method,
                            "baseline_method": source.baseline,
                            "tier": "core_main",
                            "period": period,
                            "metric": metric,
                            "n": len(paired),
                            "player_clusters": len(clusters),
                            "candidate_minus_baseline": float(paired.delta.mean()),
                            "bootstrap_p025": float(np.quantile(draws, 0.025)),
                            "bootstrap_p975": float(np.quantile(draws, 0.975)),
                            "probability_candidate_better": float(
                                np.mean(draws < 0)
                            ),
                        }
                    )
    return pd.DataFrame(output)


def combine_bowers_audits() -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        path = STUDY_DIR / source.result_dir / "current_bowers_pool_summary.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_findings(
    finalists: pd.DataFrame,
    screen: pd.DataFrame,
    player_bootstrap: pd.DataFrame,
    bowers: pd.DataFrame,
) -> None:
    selected = finalists[finalists.phase_b_finalist]
    selected_methods = set(selected.method)
    intervals = player_bootstrap[
        player_bootstrap.candidate_method.isin(selected_methods)
        & player_bootstrap.metric.eq("ppg_crps")
    ]
    bowers_columns = [
        "league",
        "method",
        "adp_35_or_earlier_weight",
        "kelce_weight",
        "kelce_gronk_graham_weight",
        "expected_played",
        "centered_residual_q10",
        "centered_residual_q90",
        "prob_centered_plus3",
        "prob_centered_plus5",
    ]
    bowers_view = bowers[bowers_columns] if not bowers.empty else bowers
    lines = [
        "# Overall log-ADP Phase B findings",
        "",
        "This file is generated by `run_phase_b_rescore.py`.",
        "",
        f"- Frozen challengers evaluated: {screen.method.nunique()}.",
        f"- Mechanical Phase B finalists: {len(selected)}.",
        "- The replacement arm removes only direct positional ADP rank; the "
        "existing market/projection-gap feature remains unchanged.",
        "- The addition arm retains every production feature and adds the "
        "fixed-scale overall log-ADP distance at weight 0.50.",
        "",
        "## Cross-league decision table",
        "",
        phase_a.markdown_table(finalists),
        "",
        "## Finalist player-cluster PPG intervals",
        "",
        phase_a.markdown_table(intervals),
        "",
        "## Current Brock Bowers pool diagnostics",
        "",
        phase_a.markdown_table(bowers_view),
        "",
    ]
    (RESULTS / "phase_b_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(
        [phase_a.read_source(source) for source in SOURCES],
        ignore_index=True,
    )
    metrics = phase_a.metric_table(predictions, SOURCES)
    deltas = phase_a.add_baseline_deltas(metrics)
    season_bootstrap = phase_a.bootstrap_intervals(predictions, SOURCES)
    position_guardrails = phase_a.position_ppg_guardrails(predictions, SOURCES)
    screen = phase_a.screen_candidates(
        deltas,
        season_bootstrap,
        position_guardrails,
    )
    cross_league = phase_a.cross_league_screen(screen)
    sensitivity = reference.tier_sensitivity(deltas)
    finalists = reference.select_finalists(cross_league, sensitivity)
    player_bootstrap = player_cluster_bootstrap(predictions)
    bowers = combine_bowers_audits()

    metrics.to_csv(RESULTS / "role_tier_metrics.csv", index=False)
    deltas.to_csv(RESULTS / "candidate_deltas.csv", index=False)
    season_bootstrap.to_csv(
        RESULTS / "season_cluster_bootstrap.csv", index=False
    )
    player_bootstrap.to_csv(
        RESULTS / "player_cluster_bootstrap.csv", index=False
    )
    position_guardrails.to_csv(
        RESULTS / "position_ppg_guardrails.csv", index=False
    )
    screen.to_csv(RESULTS / "candidate_screen.csv", index=False)
    cross_league.to_csv(RESULTS / "cross_league_screen.csv", index=False)
    sensitivity.to_csv(RESULTS / "tier_sensitivity.csv", index=False)
    finalists.to_csv(RESULTS / "finalist_decisions.csv", index=False)
    bowers.to_csv(RESULTS / "current_bowers_pool_summary.csv", index=False)
    write_findings(finalists, screen, player_bootstrap, bowers)
    metadata = {
        "leagues": [source.league for source in SOURCES],
        "targets_per_league": int(
            predictions[predictions.league.eq("dk")][
                ["player", "pos", "season"]
            ].drop_duplicates().shape[0]
        ),
        "challenger_count": int(screen.method.nunique()),
        "phase_b_finalists": finalists.loc[
            finalists.phase_b_finalist, "method"
        ].tolist(),
        "season_bootstrap_samples": phase_a.BOOTSTRAP_SAMPLES,
        "player_bootstrap_samples": PLAYER_BOOTSTRAP_SAMPLES,
        "production_changed": False,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

