"""Run and summarize the matched 12-slot NFFC preference study."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
WORKER = STUDY_DIR / "slot_worker.py"
DEFAULT_RESULTS = STUDY_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--rooms", type=int, default=256)
    parser.add_argument("--audit-samples", type=int, default=512)
    parser.add_argument("--rank-draws", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def two_way_se(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    room_component = np.var(values.mean(axis=1), ddof=1) / values.shape[0]
    season_component = np.var(values.mean(axis=0), ddof=1) / values.shape[1]
    return float(np.sqrt(room_component + season_component))


def nearest_psd(covariance: np.ndarray) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(values, 1e-12)
    return (vectors * clipped) @ vectors.T


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    slot_dir = results_dir / "slots"
    slot_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    matrices = []
    for slot in range(1, 13):
        json_path = slot_dir / f"slot_{slot:02d}.json"
        npz_path = slot_dir / f"slot_{slot:02d}.npz"
        if args.force or not (json_path.exists() and npz_path.exists()):
            command = [
                sys.executable,
                str(WORKER),
                "--slot",
                str(slot),
                "--seed",
                str(args.seed),
                "--rooms",
                str(args.rooms),
                "--audit-samples",
                str(args.audit_samples),
                "--output-json",
                str(json_path),
                "--output-npz",
                str(npz_path),
            ]
            completed = subprocess.run(
                command,
                cwd=STUDY_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Slot {slot} failed with code {completed.returncode}:\n"
                    f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            print(completed.stdout.strip(), flush=True)
        receipt = json.loads(json_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != 2:
            raise RuntimeError(
                f"Slot {slot} has a stale receipt; rerun with --force"
            )
        with np.load(npz_path) as data:
            matrices.append(
                {
                    "rooms": data["audit_rooms"].astype(np.int64),
                    "values": data["audit_values"].astype(np.float64),
                }
            )
        receipts.append(receipt)

    reference = receipts[0]
    frozen_fields = ["database_sha256", "helper_sha256", "scenario_banks"]
    for receipt in receipts[1:]:
        for field in frozen_fields:
            if receipt[field] != reference[field]:
                raise RuntimeError(f"Cross-slot {field} mismatch")

    common_rooms = sorted(
        set.intersection(*(set(matrix["rooms"].tolist()) for matrix in matrices))
    )
    if len(common_rooms) != args.rooms:
        raise RuntimeError(
            f"Only {len(common_rooms)} of {args.rooms} rooms completed for every slot"
        )
    aligned = []
    for matrix in matrices:
        locations = [int(np.where(matrix["rooms"] == room)[0][0]) for room in common_rooms]
        aligned.append(matrix["values"][locations])
    values = np.stack(aligned)
    if values.shape != (12, args.rooms, args.audit_samples):
        raise RuntimeError(f"Unexpected aligned score shape: {values.shape}")

    means = values.mean(axis=(1, 2))
    best_index = int(np.argmax(means))
    best_values = values[best_index]
    pairwise_se = np.asarray(
        [two_way_se(slot_values - best_values) for slot_values in values]
    )
    differences = means - means[best_index]

    room_means = values.mean(axis=2).T
    season_means = values.mean(axis=1).T
    covariance = (
        np.cov(room_means, rowvar=False, ddof=1) / args.rooms
        + np.cov(season_means, rowvar=False, ddof=1) / args.audit_samples
    )
    covariance = nearest_psd(covariance)
    rng = np.random.default_rng(20260803)
    draws = rng.multivariate_normal(means, covariance, size=args.rank_draws)
    draw_order = np.argsort(-draws, axis=1)
    draw_ranks = np.empty_like(draw_order)
    draw_ranks[
        np.arange(args.rank_draws)[:, None],
        draw_order,
    ] = np.arange(1, 13)
    first_probability = (draw_ranks == 1).mean(axis=0)
    expected_rank = draw_ranks.mean(axis=0)

    rows = []
    for index, receipt in enumerate(receipts):
        rows.append(
            {
                "slot": int(receipt["slot"]),
                "first_six_picks": ",".join(map(str, receipt["first_six_picks"])),
                "most_common_first_pick": receipt["most_common_first_pick"],
                "most_common_first_pick_share": float(
                    receipt["most_common_first_pick_share"]
                ),
                "heldout_ev": float(means[index]),
                "difference_vs_best": float(differences[index]),
                "paired_se_vs_best": float(pairwise_se[index]),
                "ci95_low_vs_best": float(differences[index] - 1.96 * pairwise_se[index]),
                "ci95_high_vs_best": float(differences[index] + 1.96 * pairwise_se[index]),
                "first_place_probability": float(first_probability[index]),
                "expected_rank": float(expected_rank[index]),
                "rooms": int(receipt["audit_room_count"]),
                "audit_seasons": int(receipt["audit_season_count"]),
                "elapsed_seconds": float(receipt["elapsed_seconds"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["heldout_ev", "slot"], ascending=[False, True]
    ).reset_index(drop=True)
    frame.insert(0, "preference_order", np.arange(1, len(frame) + 1))
    frame["top_equivalent_95"] = frame["ci95_high_vs_best"].ge(0.0)
    frame.to_csv(results_dir / "slot_ranking.csv", index=False)

    ordering = frame["slot"].astype(int).tolist()
    top_equivalent = frame.loc[frame["top_equivalent_95"], "slot"].astype(int).tolist()
    summary = {
        "schema_version": 1,
        "design": {
            "league": "nffc",
            "year": 2026,
            "dataset": "final_ensemble",
            "teams": 12,
            "rounds": 20,
            "draft_order": "third_round_reversal",
            "rooms": args.rooms,
            "construction_samples": 16,
            "audit_samples": args.audit_samples,
            "seed": args.seed,
            "stack_preference": True,
            "method": "ex_ante_sequential_rollout_v1",
            "primary_outcome": "heldout_raw_17_week_best_ball_points",
        },
        "preference_order": ordering,
        "best_slot": int(ordering[0]),
        "top_equivalent_95": top_equivalent,
        "score_range": float(means.max() - means.min()),
        "database_sha256": reference["database_sha256"],
        "helper_sha256": reference["helper_sha256"],
        "common_rooms": common_rooms,
        "rank_draws": args.rank_draws,
        "all_slots_complete": True,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    display = frame[
        [
            "preference_order",
            "slot",
            "most_common_first_pick",
            "most_common_first_pick_share",
            "heldout_ev",
            "difference_vs_best",
            "ci95_low_vs_best",
            "ci95_high_vs_best",
            "first_place_probability",
            "expected_rank",
        ]
    ].copy()
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Preference order: {ordering}")
    print(f"Top-equivalent at 95%: {top_equivalent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
