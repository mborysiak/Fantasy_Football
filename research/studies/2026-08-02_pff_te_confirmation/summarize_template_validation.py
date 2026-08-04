"""Apply role-tier, TE-slice, and rare-upside summaries to PFF TE replays."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
ROLE_DIR = REPO_ROOT / "research" / "studies" / "2026-07-31_template_role_tiered_validation"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase_a = _load("pff_te_phase_a", ROLE_DIR / "run_phase_a_rescore.py")
phase_b = _load("pff_te_phase_b", ROLE_DIR / "run_phase_b_rescore.py")

RESULTS = STUDY_DIR / "results_template_summary"
BASELINE = "production"
PRIMARY = "te_pff_mtf_w025"
SOURCES = (
    phase_a.Source(STUDY_DIR.name, "results_template_dk", "dk", BASELINE, "fresh_expanded"),
    phase_a.Source(STUDY_DIR.name, "results_template_beta", "beta", BASELINE, "fresh_expanded"),
)
PERIODS = {
    "development_2017_2022": (2017, 2022),
    "temporal_2023_2025": (2023, 2025),
}
BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 1234


def _te_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        experiment = f"{source.study}__{source.result_dir}"
        frame = predictions[predictions["experiment"].eq(experiment)].copy()
        masks = phase_a.tier_masks(frame)
        scopes = {
            "te_core": frame["pos"].eq("TE") & masks["core_main"],
            "te_all": frame["pos"].eq("TE"),
            "all_core": masks["core_main"],
        }
        for scope, mask in scopes.items():
            scoped = frame[mask]
            for period, (start, end) in PERIODS.items():
                period_frame = scoped[scoped["season"].between(start, end)]
                baseline = period_frame[period_frame["method"].eq(BASELINE)]
                base_values = {
                    "ppg_crps": float(baseline["ppg_crps"].mean()),
                    "contribution_crps": float(baseline["contribution_crps"].mean()),
                    "played_crps": float(baseline["played_crps"].mean()),
                    "q90_brier": float(baseline["league_winner_q90_brier_row"].mean()),
                }
                for method, group in period_frame.groupby("method", sort=True):
                    values = {
                        "ppg_crps": float(group["ppg_crps"].mean()),
                        "contribution_crps": float(group["contribution_crps"].mean()),
                        "played_crps": float(group["played_crps"].mean()),
                        "q90_brier": float(group["league_winner_q90_brier_row"].mean()),
                    }
                    rows.append(
                        {
                            "league": source.league,
                            "scope": scope,
                            "period": period,
                            "method": method,
                            "rows": len(group),
                            **values,
                            **{f"{metric}_delta": values[metric] - base_values[metric] for metric in values},
                        }
                    )
    return pd.DataFrame(rows)


def _season_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        experiment = f"{source.study}__{source.result_dir}"
        frame = predictions[
            predictions["experiment"].eq(experiment)
            & predictions["pos"].eq("TE")
        ].copy()
        masks = phase_a.tier_masks(frame)
        frame = frame[masks["core_main"]]
        baseline = frame[frame["method"].eq(BASELINE)][
            ["player", "pos", "season", "ppg_crps", "league_winner_q90_brier_row"]
        ]
        for method in sorted(set(frame["method"]) - {BASELINE}):
            challenger = frame[frame["method"].eq(method)][
                ["player", "pos", "season", "ppg_crps", "league_winner_q90_brier_row"]
            ]
            joined = baseline.merge(
                challenger,
                on=["player", "pos", "season"],
                suffixes=("_base", "_challenger"),
                validate="one_to_one",
            )
            for metric in ("ppg_crps", "league_winner_q90_brier_row"):
                joined["delta"] = joined[f"{metric}_challenger"] - joined[f"{metric}_base"]
                per_season = joined.groupby("season")["delta"].mean()
                rng = np.random.default_rng(RANDOM_SEED + len(rows))
                draws = rng.choice(
                    per_season.to_numpy(float),
                    size=(BOOTSTRAP_DRAWS, len(per_season)),
                    replace=True,
                ).mean(axis=1)
                rows.append(
                    {
                        "league": source.league,
                        "scope": "te_core",
                        "method": method,
                        "metric": metric,
                        "mean_season_delta": float(per_season.mean()),
                        "season_wins": int(per_season.lt(0).sum()),
                        "seasons": len(per_season),
                        "bootstrap_low": float(np.quantile(draws, 0.025)),
                        "bootstrap_high": float(np.quantile(draws, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def _decision(slices: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = sorted(set(slices["method"]) - {BASELINE})
    for method in methods:
        candidate = slices[slices["method"].eq(method)]
        for league in ("dk", "beta"):
            lookup = candidate[candidate["league"].eq(league)].set_index(["scope", "period"])
            dev_te = lookup.loc[("te_core", "development_2017_2022")]
            recent_te = lookup.loc[("te_core", "temporal_2023_2025")]
            dev_all = lookup.loc[("all_core", "development_2017_2022")]
            recent_all = lookup.loc[("all_core", "temporal_2023_2025")]
            dev_relative = dev_all["ppg_crps_delta"] / (
                dev_all["ppg_crps"] - dev_all["ppg_crps_delta"]
            )
            recent_relative = recent_all["ppg_crps_delta"] / (
                recent_all["ppg_crps"] - recent_all["ppg_crps_delta"]
            )
            rows.append(
                {
                    "league": league,
                    "method": method,
                    "prespecified_primary": method == PRIMARY,
                    "te_development_ppg_delta": dev_te["ppg_crps_delta"],
                    "te_recent_ppg_delta": recent_te["ppg_crps_delta"],
                    "te_development_q90_brier_delta": dev_te["q90_brier_delta"],
                    "te_recent_q90_brier_delta": recent_te["q90_brier_delta"],
                    "all_core_development_ppg_relative_delta": dev_relative,
                    "all_core_recent_ppg_relative_delta": recent_relative,
                    "te_ppg_both_periods_improve": bool(dev_te["ppg_crps_delta"] < 0 and recent_te["ppg_crps_delta"] < 0),
                    "te_q90_both_periods_improve": bool(dev_te["q90_brier_delta"] < 0 and recent_te["q90_brier_delta"] < 0),
                    "aggregate_guardrail_pass": bool(
                        dev_relative <= 0.0025 and recent_relative <= 0.0025
                    ),
                }
            )
    decision = pd.DataFrame(rows)
    decision["cross_league_mechanical_pass"] = decision.groupby("method")[
        "te_ppg_both_periods_improve"
    ].transform("all") & decision.groupby("method")[
        "aggregate_guardrail_pass"
    ].transform("all")
    decision["cross_league_upside_pass"] = decision.groupby("method")[
        "te_q90_both_periods_improve"
    ].transform("all")
    decision["advance_to_roster"] = (
        decision["cross_league_mechanical_pass"]
        & decision["cross_league_upside_pass"]
    )
    return decision


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    phase_b.SOURCES = SOURCES
    predictions = pd.concat([phase_a.read_source(source) for source in SOURCES], ignore_index=True)
    metrics = phase_a.metric_table(predictions, SOURCES)
    deltas = phase_a.add_baseline_deltas(metrics)
    season_cluster = phase_a.bootstrap_intervals(predictions, SOURCES)
    position_guardrails = phase_a.position_ppg_guardrails(predictions, SOURCES)
    generic_screen = phase_a.screen_candidates(deltas, season_cluster, position_guardrails)
    cross_league = phase_a.cross_league_screen(generic_screen)
    te_slices = _te_slices(predictions)
    te_bootstrap = _season_bootstrap(predictions)
    decision = _decision(te_slices, te_bootstrap)

    metrics.to_csv(RESULTS / "role_tier_metrics.csv", index=False)
    deltas.to_csv(RESULTS / "candidate_deltas.csv", index=False)
    season_cluster.to_csv(RESULTS / "season_cluster_bootstrap.csv", index=False)
    position_guardrails.to_csv(RESULTS / "position_ppg_guardrails.csv", index=False)
    generic_screen.to_csv(RESULTS / "generic_candidate_screen.csv", index=False)
    cross_league.to_csv(RESULTS / "generic_cross_league_screen.csv", index=False)
    te_slices.to_csv(RESULTS / "te_slices.csv", index=False)
    te_bootstrap.to_csv(RESULTS / "te_season_bootstrap.csv", index=False)
    decision.to_csv(RESULTS / "primary_decision.csv", index=False)
    metadata = {
        "primary_method": PRIMARY,
        "primary_cross_league_mechanical_pass": bool(
            decision.loc[decision["method"].eq(PRIMARY), "cross_league_mechanical_pass"].all()
        ),
        "primary_cross_league_upside_pass": bool(
            decision.loc[decision["method"].eq(PRIMARY), "cross_league_upside_pass"].all()
        ),
        "roster_finalists": sorted(
            decision.loc[decision["advance_to_roster"], "method"].unique()
        ),
        "production_changed": False,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
