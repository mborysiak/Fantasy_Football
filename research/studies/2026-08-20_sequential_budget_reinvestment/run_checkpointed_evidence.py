"""Accumulate a bounded target-board batch in short, player-level processes."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

import run_bounded_target_board as board


EVIDENCE_COLUMNS = (
    "_CurrentBlockEvidence",
    "_KeeperBlockEvidence",
    "_CandidateBenchFlags",
    "_CandidateKeeperWinRates",
)
ADDITIVE_SUMMARY_FIELDS = (
    "screen_paths",
    "confirm_paths",
    "screen_validation_contexts_total",
    "confirm_validation_contexts_total",
    "construction_contexts_total",
    "policy_refreshes",
    "roster_score_computations",
)


def fresh_curve_evidence(prior_curves, combined_curves):
    prior_lookup = {
        (row.Player, int(row.Price)): row
        for _, row in prior_curves.iterrows()
    }
    fresh_rows = []
    for _, combined in combined_curves.iterrows():
        key = (combined.Player, int(combined.Price))
        prior = prior_lookup[key]
        row = combined.copy()
        for column in EVIDENCE_COLUMNS:
            if column not in combined.index:
                continue
            old_values = tuple(prior.get(column, tuple()))
            combined_values = tuple(combined.get(column, tuple()))
            row[column] = combined_values[len(old_values):]
        row["PolicyRefreshes"] = max(
            int(combined.get("PolicyRefreshes", 0) or 0)
            - int(prior.get("PolicyRefreshes", 0) or 0),
            0,
        )
        fresh_rows.append(row)
    return pd.DataFrame(fresh_rows)


def run_player_partial(
    prior_prefix,
    output_prefix,
    player,
    variation,
    compute_budget,
    timeout_seconds,
):
    command = [
        sys.executable,
        str(Path(board.__file__).resolve()),
        "--variation",
        str(int(variation)),
        "--compute-budget",
        str(int(compute_budget)),
        "--prior-prefix",
        prior_prefix,
        "--output-prefix",
        output_prefix,
        "--only-player",
        player,
    ]
    for attempt in range(1, 4):
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            print(
                f"  {player}: timeout on attempt {attempt}/3",
                flush=True,
            )
            continue
        if completed.returncode == 0:
            print(f"  {player}: complete", flush=True)
            return
        print(
            f"  {player}: failed attempt {attempt}/3\n{completed.stderr[-1200:]}",
            flush=True,
        )
    raise RuntimeError(f"Unable to complete evidence for {player}")


def run_batch(prior_prefix, output_prefix, variation, compute_budget, timeout_seconds):
    prior_results, prior_summary, prior_curves = board.load_evidence_state(
        prior_prefix
    )
    prior_results = prior_results.loc[
        prior_results.EvidenceStage.astype(str).str.lower().eq("confirmed")
    ].copy()
    players = prior_results.sort_values(
        ["TargetRank", "Player"],
        kind="mergesort",
    ).Player.tolist()
    fresh_parts = []
    partial_summaries = []
    for index, player in enumerate(players, start=1):
        partial_prefix = (
            f"checkpoints/{output_prefix}_player_{index:02d}"
        )
        print(f"[{index:02d}/{len(players):02d}] {player}", flush=True)
        run_player_partial(
            prior_prefix,
            partial_prefix,
            player,
            variation,
            compute_budget,
            timeout_seconds,
        )
        _, partial_summary, partial_curves = board.load_evidence_state(
            partial_prefix
        )
        player_prior_curves = prior_curves.loc[
            prior_curves.Player.eq(player)
        ].copy()
        fresh_parts.append(fresh_curve_evidence(
            player_prior_curves,
            partial_curves,
        ))
        partial_summaries.append(partial_summary)

    fresh_curves = pd.concat(fresh_parts, ignore_index=True, sort=False)
    results, curves = board.sequential.accumulate_sequential_target_board(
        prior_results,
        prior_curves,
        pd.DataFrame(),
        fresh_curves,
    )
    summary = dict(prior_summary)
    first_summary = partial_summaries[0]
    summary.update({
        "target_seed": int(first_summary["target_seed"]),
        "evidence_seeds": list(first_summary["evidence_seeds"]),
        "evidence_batches": int(prior_summary.get("evidence_batches", 1)) + 1,
        "evidence_blocks": int(prior_summary.get("evidence_blocks", 4))
        + int(prior_summary.get("evidence_blocks_per_batch", 4)),
        "confirmed_count": int(len(results)),
        "discovery_watchlist_count": 0,
        "candidate_count": int(len(results)),
    })
    for field in ADDITIVE_SUMMARY_FIELDS:
        prior_value = int(prior_summary.get(field, 0) or 0)
        added = sum(
            max(int(item.get(field, prior_value) or 0) - prior_value, 0)
            for item in partial_summaries
        )
        summary[field] = prior_value + added
    summary["runtime_seconds"] = float(prior_summary.get("runtime_seconds", 0.0)) + sum(
        max(
            float(item.get("runtime_seconds", 0.0))
            - float(prior_summary.get("runtime_seconds", 0.0)),
            0.0,
        )
        for item in partial_summaries
    )
    board.save_board_state(results, summary, curves, output_prefix)
    confirmed = results.sort_values("TargetRank", kind="mergesort")
    print("\nAccumulated top ten", flush=True)
    print(confirmed[[
        "TargetRank",
        "Player",
        "Pos",
        "MarketPrice",
        "Recommendation",
        "SequentialGain",
        "SequentialLCB80",
        "EvidenceBlocks",
        "BlockPositiveRate",
    ]].head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--variation", type=int, required=True)
    parser.add_argument("--compute-budget", type=int, default=320)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    run_batch(
        args.prior_prefix,
        args.output_prefix,
        args.variation,
        args.compute_budget,
        args.timeout_seconds,
    )
