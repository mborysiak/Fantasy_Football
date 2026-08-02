"""Apply the frozen role-tier policy to the nflfastR receiver replay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROLE_DIR = STUDY_DIR.parent / "2026-07-31_template_role_tiered_validation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase_a = load_module("fastr_role_phase_a", ROLE_DIR / "run_phase_a_rescore.py")
phase_b = load_module("fastr_role_phase_b", ROLE_DIR / "run_phase_b_rescore.py")

RESULTS = STUDY_DIR / "results_role_tier"
SOURCES = (
    phase_a.Source(STUDY_DIR.name, "results_dk", "dk", "production", "fresh_expanded"),
    phase_a.Source(STUDY_DIR.name, "results_beta", "beta", "production", "fresh_expanded"),
)


def receiver_position_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report the directly affected positions without changing selection gates."""
    masks = phase_a.tier_masks(predictions)
    scopes = {
        "wr_core_main": predictions.pos.eq("WR") & masks["core_main"],
        "te_core_main": predictions.pos.eq("TE") & masks["core_main"],
        "wr_all_saved": predictions.pos.eq("WR"),
        "te_all_saved": predictions.pos.eq("TE"),
    }
    rows = []
    for source in SOURCES:
        experiment = source.study + "__" + source.result_dir
        source_frame = predictions[predictions.experiment.eq(experiment)]
        for scope, mask in scopes.items():
            scoped = source_frame.loc[mask.loc[source_frame.index]]
            for period, (start, end) in phase_a.PERIODS.items():
                period_frame = scoped[scoped.season.between(start, end)]
                baseline = period_frame[
                    period_frame.method.eq(source.baseline)
                ]
                baseline_ppg = float(baseline.ppg_crps.mean())
                for method, group in period_frame.groupby("method"):
                    if method == source.baseline:
                        continue
                    ppg = float(group.ppg_crps.mean())
                    rows.append(
                        {
                            "league": source.league,
                            "scope": scope,
                            "period": period,
                            "method": method,
                            "n": int(len(group)),
                            "ppg_crps_delta": ppg - baseline_ppg,
                            "ppg_crps_relative_delta": ppg / baseline_ppg - 1,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    phase_b.SOURCES = SOURCES
    predictions = pd.concat(
        [phase_a.read_source(source) for source in SOURCES],
        ignore_index=True,
    )
    metrics = phase_a.metric_table(predictions, SOURCES)
    deltas = phase_a.add_baseline_deltas(metrics)
    season_bootstrap = phase_a.bootstrap_intervals(predictions, SOURCES)
    position_guardrails = phase_a.position_ppg_guardrails(predictions, SOURCES)
    screen = phase_a.screen_candidates(deltas, season_bootstrap, position_guardrails)
    cross_league = phase_a.cross_league_screen(screen)
    sensitivity = phase_b.tier_sensitivity(deltas)
    finalists = phase_b.select_finalists(cross_league, sensitivity)
    player_bootstrap = phase_b.player_cluster_bootstrap(predictions)
    position_slices = receiver_position_slices(predictions)

    metrics.to_csv(RESULTS / "role_tier_metrics.csv", index=False)
    deltas.to_csv(RESULTS / "candidate_deltas.csv", index=False)
    season_bootstrap.to_csv(RESULTS / "season_cluster_bootstrap.csv", index=False)
    player_bootstrap.to_csv(RESULTS / "player_cluster_bootstrap.csv", index=False)
    position_guardrails.to_csv(RESULTS / "position_ppg_guardrails.csv", index=False)
    screen.to_csv(RESULTS / "candidate_screen.csv", index=False)
    cross_league.to_csv(RESULTS / "cross_league_screen.csv", index=False)
    sensitivity.to_csv(RESULTS / "tier_sensitivity.csv", index=False)
    position_slices.to_csv(RESULTS / "receiver_position_slices.csv", index=False)
    finalists.to_csv(RESULTS / "finalist_decisions.csv", index=False)
    selected = finalists[finalists.phase_b_finalist]
    metadata = {
        "leagues": [source.league for source in SOURCES],
        "targets_per_league": int(
            predictions[predictions.league.eq("dk")][
                ["player", "pos", "season"]
            ].drop_duplicates().shape[0]
        ),
        "challenger_count": int(screen.method.nunique()),
        "phase_b_finalists": selected.method.tolist(),
        "production_changed": False,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    focus = finalists[
        [
            "method",
            "mean_development_ppg_relative_delta",
            "worst_temporal_ppg_relative_delta",
            "all_leagues_screen_pass",
            "all_leagues_one_se_near_best",
            "tier_sensitivity_guardrail",
            "phase_b_finalist",
        ]
    ]
    slice_focus = position_slices[
        position_slices.scope.isin(["wr_core_main", "te_core_main"])
        & position_slices.period.isin(
            ["development_2017_2022", "temporal_2023_2025"]
        )
    ]
    (RESULTS / "findings.md").write_text(
        "# Role-tier findings\n\n"
        + phase_a.markdown_table(focus)
        + "\n\n## Directly affected position slices\n\n"
        + phase_a.markdown_table(slice_focus)
        + "\n",
        encoding="utf-8",
    )
    print(focus.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
