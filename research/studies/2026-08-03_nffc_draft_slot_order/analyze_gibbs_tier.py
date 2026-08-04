"""Fixed-policy sensitivity for a larger Jahmyr Gibbs projection edge."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
MODEL_REPO = STUDY_DIR.parents[2]
SNAKE_REPO = MODEL_REPO.parent / "Fantasy_Football_Snake"
APP_DIR = SNAKE_REPO / "app"
RESULTS_DIR = STUDY_DIR / "results"
sys.path.insert(0, str(APP_DIR))

from zSim_Helper import FootballSimulation


def main() -> int:
    database = APP_DIR / "Simulation.sqlite3"
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        sim = FootballSimulation(
            connection,
            2026,
            {"QB": 3, "RB": 6, "WR": 8, "TE": 3},
            12,
            20,
            1,
            pred_vers="final_ensemble",
            league="nffc",
            use_ownership=0,
            position_ranges=FootballSimulation.default_best_ball_position_ranges(),
        )
        with sim.temp_seed(20260719):
            predictions = sim.get_predictions("pred_fp_per_game", num_options=1000)
        _, audit_columns = sim.select_disjoint_policy_ppg_columns(
            len(sim.sample_value_columns(predictions)),
            16,
            512,
            construction_seed=20260719 + 101,
            evaluation_seed=20260719 + 505,
        )
        player_names = predictions.player.astype(str).to_numpy()
        player_positions = predictions.pos.astype(str).to_numpy()
        player_index = {player: index for index, player in enumerate(player_names)}
        gibbs_mask = predictions.player.astype(str).eq("Jahmyr Gibbs")
        if gibbs_mask.sum() != 1:
            raise RuntimeError("Expected exactly one Jahmyr Gibbs row")

        paths = {}
        for slot in (1, 3, 4):
            receipt = json.loads(
                (RESULTS_DIR / "slots" / f"slot_{slot:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            paths[slot] = [
                np.asarray([player_index[player] for player in roster], dtype=np.int64)
                for roster in receipt["paths"]
            ]

        rows = []
        for bump in np.arange(0.0, 2.01, 0.25):
            adjusted = predictions.copy()
            adjusted.loc[gibbs_mask, "base_pred_fp_per_game"] += float(bump)
            audit_bank = sim.sample_sequential_policy_score_bank(
                adjusted,
                None,
                512,
                17,
                20260719 + 505,
                audit_columns,
            )
            slot_means = {}
            for slot, rosters in paths.items():
                values = np.stack(
                    [
                        sim.best_ball_roster_scores_bank(
                            audit_bank,
                            player_positions,
                            roster,
                        )
                        for roster in rosters
                    ]
                )
                slot_means[slot] = float(values.mean())
            rows.append(
                {
                    "gibbs_ppg_bump": float(bump),
                    "gibbs_modeled_ppg": float(
                        predictions.loc[gibbs_mask, "base_pred_fp_per_game"].iloc[0]
                        + bump
                    ),
                    "slot_1_ev": slot_means[1],
                    "slot_3_ev": slot_means[3],
                    "slot_4_ev": slot_means[4],
                    "slot_1_minus_3": slot_means[1] - slot_means[3],
                    "slot_1_minus_4": slot_means[1] - slot_means[4],
                }
            )
    finally:
        connection.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "sensitivity_gibbs_tier.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
