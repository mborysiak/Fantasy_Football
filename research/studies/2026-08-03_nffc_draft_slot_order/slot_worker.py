"""Simulate one ex-ante NFFC draft slot in a fresh Python process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
MODEL_REPO = STUDY_DIR.parents[2]
DEFAULT_SNAKE_REPO = MODEL_REPO.parent / "Fantasy_Football_Snake"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--rooms", type=int, default=256)
    parser.add_argument("--audit-samples", type=int, default=512)
    parser.add_argument("--snake-repo", type=Path, default=DEFAULT_SNAKE_REPO)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snake_repo = args.snake_repo.resolve()
    app_dir = snake_repo / "app"
    database_path = app_dir / "Simulation.sqlite3"
    helper_path = app_dir / "zSim_Helper.py"
    sys.path.insert(0, str(app_dir))

    from zSim_Helper import FootballSimulation

    connection = sqlite3.connect(
        f"{database_path.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    started = time.perf_counter()
    try:
        sim = FootballSimulation(
            conn=connection,
            set_year=2026,
            pos_require_start={"QB": 3, "RB": 6, "WR": 8, "TE": 3},
            num_teams=12,
            num_rounds=20,
            my_pick_position=args.slot,
            pred_vers="final_ensemble",
            league="nffc",
            use_ownership=0,
            position_ranges=FootballSimulation.default_best_ball_position_ranges(),
            use_stack_bonus=True,
            stack_bonus_pct=0.25,
            stack_pair_cap=12.0,
            stack_team_cap=18.0,
        )

        with sim.temp_seed(args.seed):
            ppg_pred = sim.get_predictions("pred_fp_per_game", num_options=1000)
            adp_samples = sim.get_adp_samples(num_options=1000)

        player_names = ppg_pred.player.astype(str).to_numpy()
        player_ids = sim.identity_values(ppg_pred, validate_unique=True)
        player_positions = ppg_pred.pos.astype(str).to_numpy()
        player_teams = ppg_pred.team.fillna("").astype(str).str.strip().to_numpy()
        player_ppg = ppg_pred["base_pred_fp_per_game"].to_numpy(dtype=np.float32)

        adp_ids = sim.identity_values(adp_samples, validate_unique=True)
        adp_values = adp_samples[sim.sample_value_columns(adp_samples)].copy()
        adp_values.index = adp_ids
        adp_values = adp_values.reindex(player_ids)
        if adp_values.isna().any().any():
            raise RuntimeError("Could not align NFFC ADP samples to players")
        adp_matrix = adp_values.to_numpy(dtype=np.float32)

        construction_columns, audit_columns = sim.select_disjoint_policy_ppg_columns(
            len(sim.sample_value_columns(ppg_pred)),
            16,
            args.audit_samples,
            construction_seed=args.seed + 101,
            evaluation_seed=args.seed + 505,
        )
        construction_bank = sim.sample_sequential_policy_score_bank(
            ppg_pred,
            None,
            16,
            17,
            args.seed + 101,
            construction_columns,
        )
        audit_bank = sim.sample_sequential_policy_score_bank(
            ppg_pred,
            None,
            args.audit_samples,
            17,
            args.seed + 505,
            audit_columns,
        )
        draft_orders, adp_columns = sim.build_sequential_draft_orders(
            adp_matrix,
            args.rooms,
            seed=args.seed + 303,
        )
        survival_table = sim.build_sequential_survival_table(
            adp_matrix,
            sim.my_picks,
        )
        empty_mask = np.zeros(len(player_names), dtype=bool)
        position_ranges = sim.best_ball_position_ranges(
            player_positions,
            empty_mask,
        )

        audit_values = []
        stack_utilities = []
        first_picks = []
        first_pick_positions = []
        paths = []
        pre_pick_opponents = []
        for room_index in range(args.rooms):
            remaining = np.ones(len(player_names), dtype=bool)
            selected_indices: list[int] = []
            order_pointer, drafted_before_first = sim.advance_sequential_opponents(
                remaining,
                draft_orders[room_index],
                0,
                sim.my_picks[0] - 1,
            )
            if len(drafted_before_first) != args.slot - 1:
                raise RuntimeError(
                    f"Slot {args.slot}, room {room_index}: incomplete pre-pick draft"
                )
            pre_pick_opponents.append(player_names[drafted_before_first].tolist())

            for pick_index, current_pick in enumerate(sim.my_picks):
                picks_left = len(sim.my_picks) - pick_index
                legal = sim.sequential_legal_candidate_indices(
                    remaining,
                    player_positions,
                    selected_indices,
                    picks_left,
                    pos_ranges=position_ranges,
                )
                if len(legal) == 0:
                    raise RuntimeError(
                        f"Slot {args.slot}, room {room_index}, pick {pick_index}: "
                        "no legal candidate"
                    )
                _, immediate = sim.marginal_best_ball_values_bank(
                    construction_bank,
                    player_positions,
                    selected_indices,
                    legal,
                )
                stack = sim.sequential_stack_marginal_utilities(
                    selected_indices,
                    legal,
                    player_positions,
                    player_teams,
                    player_ppg,
                    sim.stack_bonus_pct,
                    sim.stack_pair_cap,
                    sim.stack_team_cap,
                )
                next_pick = (
                    sim.my_picks[pick_index + 1]
                    if pick_index + 1 < len(sim.my_picks)
                    else None
                )
                policy_scores, _, _, _, _ = sim.sequential_policy_scores(
                    legal,
                    immediate + stack,
                    player_positions,
                    adp_matrix,
                    current_pick,
                    next_pick,
                    survival_probabilities=survival_table[pick_index],
                )
                chosen = int(legal[int(np.argmax(policy_scores))])
                if pick_index == 0 and not remaining[chosen]:
                    raise RuntimeError("First pick was already drafted by an opponent")
                selected_indices.append(chosen)
                remaining[chosen] = False

                if next_pick is not None:
                    opponent_count = next_pick - current_pick - 1
                    order_pointer, opponent_picks = sim.advance_sequential_opponents(
                        remaining,
                        draft_orders[room_index],
                        order_pointer,
                        opponent_count,
                    )
                    if len(opponent_picks) != opponent_count:
                        raise RuntimeError(
                            f"Slot {args.slot}, room {room_index}: opponent draft exhausted"
                        )

            roster = np.asarray(selected_indices, dtype=np.int64)
            if len(roster) != 20 or len(np.unique(roster)) != 20:
                raise RuntimeError(f"Slot {args.slot}, room {room_index}: illegal roster")
            counts = Counter(player_positions[roster])
            if not all(
                minimum <= counts.get(position, 0) <= maximum
                for position, (minimum, maximum) in position_ranges.items()
            ):
                raise RuntimeError(
                    f"Slot {args.slot}, room {room_index}: position bounds failed"
                )

            first_picks.append(str(player_names[roster[0]]))
            first_pick_positions.append(str(player_positions[roster[0]]))
            paths.append(player_names[roster].tolist())
            audit_values.append(
                sim.best_ball_roster_scores_bank(
                    audit_bank,
                    player_positions,
                    roster,
                )
            )
            stack_utilities.append(
                sim.sequential_stack_roster_utility(
                    roster,
                    player_positions,
                    player_teams,
                    player_ppg,
                    sim.stack_bonus_pct,
                    sim.stack_pair_cap,
                    sim.stack_team_cap,
                )
            )
    finally:
        connection.close()

    audit_values_array = np.stack(audit_values).astype(np.float64)
    if audit_values_array.shape != (args.rooms, args.audit_samples):
        raise RuntimeError(f"Unexpected audit shape: {audit_values_array.shape}")
    first_pick_counts = Counter(first_picks)
    first_position_counts = Counter(first_pick_positions)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        audit_values=audit_values_array,
        audit_rooms=np.arange(args.rooms, dtype=np.int64),
        stack_utilities=np.asarray(stack_utilities, dtype=np.float64),
    )
    payload = {
        "schema_version": 2,
        "method": "ex_ante_sequential_rollout_v1",
        "slot": args.slot,
        "first_six_picks": [int(value) for value in sim.my_picks[:6]],
        "first_pick_counts": dict(first_pick_counts.most_common()),
        "first_pick_position_counts": dict(first_position_counts.most_common()),
        "most_common_first_pick": first_pick_counts.most_common(1)[0][0],
        "most_common_first_pick_share": (
            first_pick_counts.most_common(1)[0][1] / args.rooms
        ),
        "audit_mean": float(audit_values_array.mean()),
        "audit_se": float(sim.approximate_two_way_se(audit_values_array)),
        "audit_room_count": int(audit_values_array.shape[0]),
        "audit_season_count": int(audit_values_array.shape[1]),
        "mean_stack_utility": float(np.mean(stack_utilities)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "scenario_banks": {
            "construction_ppg_columns": construction_columns.tolist(),
            "audit_ppg_columns": audit_columns.tolist(),
            "disjoint": True,
        },
        "draft_room_adp_columns": adp_columns.tolist(),
        "pre_first_opponent_count": args.slot - 1,
        "database_path": str(database_path.resolve()),
        "database_sha256": sha256_file(database_path),
        "helper_sha256": sha256_file(helper_path),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "paths": paths,
        "pre_pick_opponents": pre_pick_opponents,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "slot": args.slot,
                "audit_mean": payload["audit_mean"],
                "first_pick": payload["most_common_first_pick"],
                "first_pick_share": payload["most_common_first_pick_share"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
