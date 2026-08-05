"""Confirm raw logged expert-rank level against normalized rank level."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parent
RAW_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-07-30_v2_market_rank_challengers"
    / "run_raw_rank_challenger.py"
)
HELPER_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "studies"
    / "2026-08-04_v2_logged_rank_disagreement"
    / "run_study.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.contracts import scoring_hash


DATABASES = {
    "dk": REPO_ROOT / "Data" / "Databases" / "Projection_V2.sqlite3",
    "beta": REPO_ROOT / "Data" / "Databases" / "Projection_V2_beta.sqlite3",
}
VARIANT_FEATURES = {
    "incumbent": (),
    "normalized_rank": (
        "scoring_specific_rank_position_percentile_median",
    ),
    "raw_log": ("raw_rank_log1p",),
}
COMPARISONS = {
    "normalized_rank": "incumbent",
    "raw_log": "normalized_rank",
}
CONTROLLED_METHOD = "controlled_equal_thirds"
PRODUCTION_METHOD = "equal_thirds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("dk", "beta", "all"), default="all")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--combine-existing", action="store_true")
    return parser.parse_args()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load study module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runners():
    raw = _load_module(
        RAW_RUNNER_PATH,
        "v2_raw_rank_runner_for_log_rank_confirmation",
    )
    helper = _load_module(
        HELPER_RUNNER_PATH,
        "v2_logged_disagreement_runner_for_log_rank_confirmation",
    )
    raw.VARIANT_FEATURES = VARIANT_FEATURES.copy()
    helper.POINT_COMPARISONS = COMPARISONS.copy()
    return raw, helper


def _gate_audit(summary: pd.DataFrame) -> dict[str, object]:
    controlled = summary[
        summary["method"].eq(CONTROLLED_METHOD)
        & summary["challenger_variant"].eq("raw_log")
    ].iloc[0]
    production = summary[
        summary["method"].eq(PRODUCTION_METHOD)
        & summary["challenger_variant"].eq("raw_log")
    ].iloc[0]
    gates = {
        "controlled_pooled_improvement_at_least_0_001": bool(
            controlled.pooled_delta <= -0.001
        ),
        "controlled_recent_nonworse": bool(controlled.recent_delta <= 0),
        "controlled_at_least_6_season_wins": bool(controlled.season_wins >= 6),
        "controlled_season_interval_upper_nonpositive": bool(
            controlled.season_95_high <= 0
        ),
        "controlled_player_interval_upper_nonpositive": bool(
            controlled.player_95_high <= 0
        ),
        "production_pooled_nonworse": bool(production.pooled_delta <= 0),
        "production_recent_nonworse": bool(production.recent_delta <= 0),
        "controlled_at_least_3_positions_nonworse": bool(
            controlled.nonworse_positions >= 3
        ),
    }
    return {
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "next_action": (
            "prefer_raw_log_for_future_nested_validation"
            if all(gates.values())
            else "retain_normalized_rank_as_challenger"
        ),
    }


def _findings_markdown(
    league: str,
    summary: pd.DataFrame,
    audit: dict[str, object],
    feature_run_id: str,
    database_hash: str,
    missingness_mismatch_rows: int,
) -> str:
    lines = [
        f"# Logged Expert-Rank Level Confirmation - {league.upper()}",
        "",
        "| Surface | Baseline | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Positions nonworse |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | `{row.baseline_variant}` | "
            f"`{row.challenger_variant}` | {row.pooled_delta:+.5f} | "
            f"{row.recent_delta:+.5f} | {row.season_wins}/{row.season_count} | "
            f"[{row.season_95_low:+.5f}, {row.season_95_high:+.5f}] | "
            f"[{row.player_95_low:+.5f}, {row.player_95_high:+.5f}] | "
            f"{row.nonworse_positions}/{row.position_count} |"
        )
    failed = [key for key, passed in audit["gates"].items() if not passed]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Raw log passes every gate: `{audit['all_gates_pass']}`.",
            f"- Next action: `{audit['next_action']}`.",
            f"- Failed gates: `{failed}`.",
            "- A tie retains normalized rank because it is depth- and QB-placement robust.",
            "- No production feature, lock, template, or table changed.",
            "",
            "## Lineage",
            "",
            f"- Feature run: `{feature_run_id}`",
            f"- Database SHA-256 before/after: `{database_hash}`",
            f"- Rank representation missingness mismatches: `{missingness_mismatch_rows}`",
            "",
        ]
    )
    return "\n".join(lines)


def _run_league(
    league: str,
    database: Path,
    results_dir: Path,
    iterations: int,
) -> dict[str, object]:
    if not database.is_file():
        raise FileNotFoundError(database)
    raw, helper = _load_runners()
    before_hash = helper._file_sha256(database)
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
    ) = raw._load_inputs(database, league)
    observed_hashes = set(features["scoring_hash"].dropna().astype(str))
    expected_hash = scoring_hash(league)
    if observed_hashes != {expected_hash}:
        raise ValueError(
            f"Scoring mismatch: observed={observed_hashes}, expected={expected_hash}"
        )
    missingness_mismatch = features["raw_rank_log1p"].isna() ^ features[
        "scoring_specific_rank_position_percentile_median"
    ].isna()
    missingness_mismatch_rows = int(missingness_mismatch.sum())
    if missingness_mismatch_rows:
        raise ValueError(
            "Raw-log and normalized rank representations do not cover the same rows: "
            f"{missingness_mismatch_rows}"
        )
    predictions = raw._run_predictions(locked, features, selected)
    reproduction_delta = raw._assert_incumbent_reproduces_exactly(
        predictions, locked_predictions
    )
    evaluation = raw._evaluation_long(predictions, locked.OUTER_SEASONS)
    scores = raw._score_table(evaluation)
    summary = helper._point_comparison_summary(
        normalized_runner, evaluation, iterations
    )
    audit = _gate_audit(summary)
    after_hash = helper._file_sha256(database)
    if after_hash != before_hash:
        raise RuntimeError("V2 database changed during read-only confirmation")

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_features.to_csv(results_dir / "raw_rank_features.csv", index=False)
    evaluation.to_csv(results_dir / "point_oof_predictions.csv", index=False)
    scores.to_csv(results_dir / "point_scores.csv", index=False)
    summary.to_csv(results_dir / "comparison_summary.csv", index=False)
    source_coverage.to_csv(results_dir / "rank_source_coverage.csv", index=False)
    depth_audit.to_csv(results_dir / "rank_source_depths.csv", index=False)
    position_audit.to_csv(results_dir / "rank_position_audit.csv", index=False)
    (results_dir / "ppr_identity_resolution.json").write_text(
        json.dumps(ppr_resolution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **input_manifest,
        "database_sha256_before": before_hash,
        "database_sha256_after": after_hash,
        "rank_representation_missingness_mismatch_rows": missingness_mismatch_rows,
        "incumbent_reproduction_max_abs_delta": reproduction_delta,
        "raw_median_reproduction_max_abs_delta": raw_median_reproduction_delta,
        "bootstrap_iterations": iterations,
    }
    (results_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "gate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "findings.md").write_text(
        _findings_markdown(
            league,
            summary,
            audit,
            feature_run_id,
            before_hash,
            missingness_mismatch_rows,
        ),
        encoding="utf-8",
    )
    payload = {
        "league": league,
        "database": str(database.resolve()),
        "database_sha256": before_hash,
        "feature_run_id": feature_run_id,
        "summary": summary.to_dict("records"),
        "audit": audit,
    }
    (results_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_existing(league: str) -> dict[str, object]:
    path = STUDY_ROOT / "results" / league / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _combine(
    payloads: Sequence[dict[str, object]],
    results_dir: Path,
) -> dict[str, object]:
    by_league = {str(payload["league"]): payload for payload in payloads}
    if set(by_league) != {"dk", "beta"}:
        raise ValueError("Combined confirmation requires DK and beta")
    both_pass = all(
        bool(by_league[league]["audit"]["all_gates_pass"])
        for league in ("dk", "beta")
    )
    decision = {
        "both_leagues_pass": both_pass,
        "next_action": (
            "prefer_raw_log_for_future_nested_validation"
            if both_pass
            else "retain_normalized_rank_as_challenger"
        ),
        "league_results": by_league,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Logged Expert-Rank Level Cross-League Confirmation",
        "",
        f"- Raw log passes both leagues: `{both_pass}`.",
        f"- Next action: `{decision['next_action']}`.",
        "",
        "| League | Controlled raw-log minus normalized | Recent | Wins | Production delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for league in ("dk", "beta"):
        summary = pd.DataFrame(by_league[league]["summary"])
        controlled = summary[
            summary["method"].eq(CONTROLLED_METHOD)
            & summary["challenger_variant"].eq("raw_log")
        ].iloc[0]
        production = summary[
            summary["method"].eq(PRODUCTION_METHOD)
            & summary["challenger_variant"].eq("raw_log")
        ].iloc[0]
        lines.append(
            f"| {league.upper()} | {controlled.pooled_delta:+.5f} | "
            f"{controlled.recent_delta:+.5f} | "
            f"{controlled.season_wins}/{controlled.season_count} | "
            f"{production.pooled_delta:+.5f} |"
        )
    lines.extend(
        [
            "",
            "A tie retains normalized rank because it is more robust to provider depth and overall QB placement.",
            "No production feature, model lock, template, or SQLite table changed.",
            "",
        ]
    )
    (results_dir / "findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return decision


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    if args.league == "all" and args.database is not None:
        raise ValueError("--database cannot be used with --league all")
    if args.combine_existing and args.league != "all":
        raise ValueError("--combine-existing requires --league all")
    if args.league == "all":
        if args.combine_existing:
            payloads = [_load_existing(league) for league in ("dk", "beta")]
        else:
            payloads = [
                _run_league(
                    league,
                    DATABASES[league],
                    STUDY_ROOT / "results" / league,
                    args.bootstrap_iterations,
                )
                for league in ("dk", "beta")
            ]
        decision = _combine(
            payloads, args.results_dir or STUDY_ROOT / "results"
        )
        print(json.dumps(decision, indent=2))
        return
    payload = _run_league(
        args.league,
        args.database or DATABASES[args.league],
        args.results_dir or STUDY_ROOT / "results" / args.league,
        args.bootstrap_iterations,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
