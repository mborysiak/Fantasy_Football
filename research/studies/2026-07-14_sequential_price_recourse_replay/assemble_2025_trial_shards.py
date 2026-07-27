"""Validate isolated 2025 trial shards and commit one resumable year checkpoint.

The 2025 full-process replay exposed intermittent native CVXOPT/GLPK exits.
Each order-regime/trial cell was therefore rerun in a fresh process with the
same full eight-trial plans, original trial index, seeds, and policy grid.
This script refuses to assemble anything except the exact 256-path grid and
writes the checkpoint metadata last as the commit marker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
SHARD_ROOT = STUDY_DIR / "artifacts" / "local" / "2025_trial_shards"
CHECKPOINT_DIR = STUDY_DIR / "artifacts" / "local" / "full_checkpoints_pickle"
PREFIX_MANIFEST = (
    STUDY_DIR
    / "artifacts"
    / "local"
    / "2025_prefix_validation"
    / "source_manifest.json"
)
RUNNER = STUDY_DIR / "run_recourse_replay.py"
YEAR = 2025
TRIALS = tuple(range(8))
ORDER_REGIMES = ("tier_early", "uniform", "position_run", "star_late")
PRICE_RULES = ("clearing", "plus_one")
POLICY_MODES = ("strict", "operational")
BUFFERS = (5.0, 10.0)
SALARY_CAP = 298.0
SOLVER_TOLERANCE = 1e-3

FRAME_SOURCES = {
    "paths": "policy_paths.csv",
    "events": "event_decisions.csv",
    "raw_rosters": "raw_roster_rows.csv",
    "owner_audit": "raw_owner_audit.csv",
    "tape_audit": "price_tape_audit.csv",
    "tape_events": "price_tape_events.csv",
    "template_audit": "template_pool_audit.csv",
}
REPEATED_AUDIT_FRAMES = tuple(FRAME_SOURCES)[2:]
REQUIRED_VALIDATION_FLAGS = (
    "unique_path_cells",
    "all_template_donors_pre_origin",
    "k_dst_excluded_from_tape",
    "all_completed_paid_spend_within_298",
    "all_attempted_events_in_ledger",
    "all_branch_sampled_spend_within_298",
    "all_branch_nominal_spend_within_exported_stage",
    "strict_paths_used_no_fallback",
    "runtime_generic_repair_disabled",
    "incomplete_paths_not_target_scored",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_origin(origin: dict[str, Any]) -> dict[str, Any]:
    result = dict(origin)
    result.pop("runtime_seconds", None)
    return result


def main() -> None:
    prefix_manifest = json.loads(PREFIX_MANIFEST.read_text(encoding="utf-8"))
    if prefix_manifest["validation"].get("prefix_invariance") is not True:
        raise AssertionError("The standalone 2025 prefix-invariance test did not pass.")

    path_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    shard_records: list[dict[str, Any]] = []
    audit_hashes: dict[str, set[str]] = {
        name: set() for name in REPEATED_AUDIT_FRAMES
    }
    first_frames: dict[str, pd.DataFrame] = {}
    canonical_origin: dict[str, Any] | None = None
    total_runtime = 0.0

    for regime in ORDER_REGIMES:
        for trial in TRIALS:
            shard_dir = SHARD_ROOT / f"{regime}_trial{trial}"
            manifest_path = shard_dir / "source_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest["config"]
            if (
                config.get("years") != [YEAR]
                or config.get("trials") != len(TRIALS)
                or config.get("trial_indices") != [trial]
                or config.get("order_regimes") != [regime]
                or config.get("price_rules") != list(PRICE_RULES)
                or config.get("policy_modes") != list(POLICY_MODES)
                or config.get("buffers") != list(BUFFERS)
                or config.get("skip_prefix_check") is not True
            ):
                raise AssertionError(f"Unexpected shard config: {shard_dir}")
            for flag in REQUIRED_VALIDATION_FLAGS:
                if manifest["validation"].get(flag) is not True:
                    raise AssertionError(f"{shard_dir}: validation failed for {flag}.")

            paths = pd.read_csv(shard_dir / FRAME_SOURCES["paths"])
            events = pd.read_csv(shard_dir / FRAME_SOURCES["events"])
            if len(paths) != 8 or paths.path_id.nunique() != 8:
                raise AssertionError(f"{shard_dir}: expected eight unique policy paths.")
            if (
                set(paths.year) != {YEAR}
                or set(paths.trial) != {trial}
                or set(paths.order_regime) != {regime}
            ):
                raise AssertionError(f"{shard_dir}: shard identity columns drifted.")
            expected_cells = {
                (price, mode, buffer)
                for price in PRICE_RULES
                for mode in POLICY_MODES
                for buffer in BUFFERS
            }
            actual_cells = set(
                paths[["price_rule", "policy_mode", "nominal_buffer"]]
                .itertuples(index=False, name=None)
            )
            if actual_cells != expected_cells:
                raise AssertionError(f"{shard_dir}: policy cells are incomplete.")

            origin = manifest["origins"][str(YEAR)]
            if canonical_origin is None:
                canonical_origin = normalized_origin(origin)
            elif normalized_origin(origin) != canonical_origin:
                raise AssertionError(f"{shard_dir}: frozen origin metadata drifted.")
            total_runtime += float(origin["runtime_seconds"])

            for name in REPEATED_AUDIT_FRAMES:
                frame_path = shard_dir / FRAME_SOURCES[name]
                audit_hashes[name].add(sha256_file(frame_path))
                if name not in first_frames:
                    first_frames[name] = pd.read_csv(frame_path)

            path_frames.append(paths)
            event_frames.append(events)
            shard_records.append(
                {
                    "order_regime": regime,
                    "trial": trial,
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "runner_sha256": manifest["runner_sha256"],
                    "paths_sha256": sha256_file(
                        shard_dir / FRAME_SOURCES["paths"]
                    ),
                    "events_sha256": sha256_file(
                        shard_dir / FRAME_SOURCES["events"]
                    ),
                }
            )

    if any(len(hashes) != 1 for hashes in audit_hashes.values()):
        raise AssertionError(f"Repeated shard audit frames drifted: {audit_hashes}")
    if canonical_origin is None:
        raise AssertionError("No shard origins were loaded.")

    paths = pd.concat(path_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    key = [
        "year",
        "trial",
        "order_regime",
        "price_rule",
        "policy_mode",
        "nominal_buffer",
    ]
    if len(paths) != 256 or paths.duplicated(key).any() or paths.path_id.nunique() != 256:
        raise AssertionError("The assembled 2025 path grid is not exactly 256 cells.")
    expected_ledger = paths.set_index("path_id").events_seen.astype(int)
    actual_ledger = events.groupby("path_id").size().reindex(expected_ledger.index)
    if not actual_ledger.eq(expected_ledger).all():
        raise AssertionError("The assembled event ledger does not reconcile.")
    if not set(events.path_id).issubset(set(paths.path_id)):
        raise AssertionError("The assembled ledger contains an unknown path ID.")
    completed = paths.complete.astype(bool)
    if paths.loc[completed, "paid_spend"].gt(SALARY_CAP + 1e-8).any():
        raise AssertionError("A completed assembled path exceeds the paid-price cap.")
    if paths.loc[completed, "roster_size"].ne(13).any():
        raise AssertionError("A completed assembled path has the wrong roster size.")
    if paths.loc[~completed, "actual_points"].notna().any():
        raise AssertionError("An incomplete assembled path accessed target outcomes.")
    strict = paths.policy_mode.eq("strict")
    if (
        paths.loc[strict, "nominal_relaxation_events"].ne(0).any()
        or paths.loc[strict, "top_n_relaxation_events"].ne(0).any()
        or paths.generic_repair_buys.ne(0).any()
    ):
        raise AssertionError("An assembled strict path used a forbidden relaxation.")
    for column in ("buy_sampled_spend", "pass_sampled_spend"):
        spend = pd.to_numeric(events[column], errors="coerce")
        if spend.gt(SALARY_CAP + SOLVER_TOLERANCE).any():
            raise AssertionError(f"{column} exceeds the numerical solver tolerance.")
    nominal_stage = pd.to_numeric(events.nominal_stage, errors="coerce")
    for column in ("buy_nominal_spend", "pass_nominal_spend"):
        spend = pd.to_numeric(events[column], errors="coerce")
        constrained = spend.notna() & nominal_stage.notna()
        if (
            spend[constrained]
            > SALARY_CAP + nominal_stage[constrained] + SOLVER_TOLERANCE
        ).any():
            raise AssertionError(f"{column} exceeds its assembled nominal stage.")

    frames = {
        "paths": paths,
        "events": events,
        **first_frames,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths: dict[str, Path] = {}
    for name, frame in frames.items():
        frame_path = CHECKPOINT_DIR / f"{YEAR}_{name}.pkl"
        frame.to_pickle(frame_path)
        frame_paths[name] = frame_path

    prior_meta = json.loads(
        (CHECKPOINT_DIR / "2022_meta.json").read_text(encoding="utf-8")
    )
    origin_manifest = dict(canonical_origin)
    origin_manifest.update(
        {
            "runtime_seconds": total_runtime,
            "execution_mode": "isolated_order_trial_shards",
            "sharded_execution": {
                "reason": "intermittent native CVXOPT/GLPK process exits",
                "method_change": False,
                "shard_count": len(shard_records),
                "full_trial_plan_preserved": True,
                "prefix_validation_manifest": str(PREFIX_MANIFEST),
                "prefix_validation_manifest_sha256": sha256_file(PREFIX_MANIFEST),
                "assembler": str(Path(__file__).resolve()),
                "assembler_sha256": sha256_file(Path(__file__).resolve()),
                "shards": shard_records,
            },
        }
    )
    meta = {
        "config": prior_meta["config"],
        "runner_sha256": sha256_file(RUNNER),
        "simulation_helper_sha256": prior_meta["simulation_helper_sha256"],
        "current_outcome_sources": prior_meta["current_outcome_sources"],
        "year": YEAR,
        "prefix_invariance": True,
        "origin_manifest": origin_manifest,
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "frame_sha256": {
            name: sha256_file(frame_path) for name, frame_path in frame_paths.items()
        },
    }
    meta_path = CHECKPOINT_DIR / f"{YEAR}_meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(
        f"Committed {len(paths)} paths and {len(events)} events to {meta_path}."
    )


if __name__ == "__main__":
    main()
