"""Audit the locked V2 point-center handoff against production template rows."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.V2.config import OUTPUT_DB_PATH
from Scripts.V2.contracts import publish_tables_atomic


RESULTS_DIR = STUDY_DIR / "results"
SIMULATION_DB_PATH = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
KEY_COLUMNS = ("player_key", "season")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-db",
        type=Path,
        default=OUTPUT_DB_PATH,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--simulation-db",
        type=Path,
        default=SIMULATION_DB_PATH,
    )
    parser.add_argument("--league", default="dk")
    return parser.parse_args()


def _load_handoff(output_database: Path) -> pd.DataFrame:
    with sqlite3.connect(output_database) as connection:
        handoff = pd.read_sql_query(
            "SELECT * FROM locked_template_handoff", connection
        )
    rename = {}
    if "active_ppg" in handoff:
        rename["active_ppg"] = "v2_conditional_ppg_actual"
    if "active_ppg_resid" in handoff:
        rename[
            "active_ppg_resid"
        ] = "v2_conditional_ppg_training_residual"
    handoff.rename(columns=rename, inplace=True)
    handoff["template_active_ppg_resid_recompute_required"] = 1
    if handoff.duplicated(list(KEY_COLUMNS)).any():
        duplicates = handoff.loc[
            handoff.duplicated(list(KEY_COLUMNS), keep=False),
            ["player_key", "display_name", "position", "season"],
        ]
        raise ValueError(
            "V2 handoff is ambiguous on canonical player_key/season:\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )
    return handoff


def _load_production_templates(
    simulation_database: Path,
    league: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(simulation_database) as connection:
        templates = pd.read_sql_query(
            """
            SELECT league, template_id, player_key, player,
                   pos AS position, season,
                   historical_pred_fp_per_game AS production_point_center,
                   active_ppg AS template_active_ppg,
                   active_ppg_resid AS production_active_ppg_resid,
                   template_eligible
            FROM Best_Ball_Weekly_Templates
            WHERE season BETWEEN 2017 AND 2025
                  AND league=?
            """,
            connection,
            params=(league,),
        )
        player_map = pd.read_sql_query(
            """
            SELECT version AS league, player_key, player,
                   pos AS position, year AS season, pred_fp_per_game
            FROM Best_Ball_Weekly_Player_Map
            WHERE year=2026 AND version=?
            """,
            connection,
            params=(league,),
        )
    for frame_name, frame in (
        ("Best_Ball_Weekly_Templates", templates),
        ("Best_Ball_Weekly_Player_Map", player_map),
    ):
        if frame["player_key"].isna().any():
            raise ValueError(f"{frame_name} contains null canonical player_key")
    return templates, player_map


def build_audit(
    handoff: pd.DataFrame,
    templates: pd.DataFrame,
    player_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = handoff[handoff["season"].between(2017, 2025)][
        [
            *KEY_COLUMNS,
            "display_name",
            "position",
            "historical_pred_fp_per_game",
            "v2_conditional_ppg_actual",
        ]
    ].rename(columns={"position": "v2_position"})
    joined = templates.merge(
        historical,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    joined["v2_template_active_ppg_resid"] = (
        joined["template_active_ppg"]
        - joined["historical_pred_fp_per_game"]
    )
    joined["v2_target_minus_template_active_ppg"] = (
        joined["v2_conditional_ppg_actual"]
        - joined["template_active_ppg"]
    )

    current = handoff[handoff["season"].eq(2026)][
        [
            *KEY_COLUMNS,
            "position",
            "historical_pred_fp_per_game",
        ]
    ].rename(columns={"position": "v2_position"})
    current_joined = player_map.merge(
        current,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "audit": "v2_handoff_duplicate_player_key_season_rows",
            "league": "all",
            "n_rows": len(handoff),
            "value": float(
                handoff.duplicated(list(KEY_COLUMNS), keep=False).sum()
            ),
        }
    )
    for league, group in joined.groupby("league"):
        matched = group["display_name"].notna()
        center = group["historical_pred_fp_per_game"].notna()
        target_comparable = group[
            group["v2_target_minus_template_active_ppg"].notna()
        ]
        rows.extend(
            [
                {
                    "audit": "historical_identity_join_rate",
                    "league": league,
                    "n_rows": len(group),
                    "value": float(matched.mean()),
                },
                {
                    "audit": "historical_position_mismatch_rate",
                    "league": league,
                    "n_rows": int(matched.sum()),
                    "value": float(
                        (
                            group.loc[matched, "position"]
                            != group.loc[matched, "v2_position"]
                        ).mean()
                    ),
                },
                {
                    "audit": "historical_point_center_coverage",
                    "league": league,
                    "n_rows": len(group),
                    "value": float(center.mean()),
                },
                {
                    "audit": "v2_target_vs_template_active_ppg_mae",
                    "league": league,
                    "n_rows": len(target_comparable),
                    "value": float(
                        target_comparable[
                            "v2_target_minus_template_active_ppg"
                        ].abs().mean()
                    ),
                },
                {
                    "audit": "v2_target_vs_template_active_ppg_max_abs",
                    "league": league,
                    "n_rows": len(target_comparable),
                    "value": float(
                        target_comparable[
                            "v2_target_minus_template_active_ppg"
                        ].abs().max()
                    ),
                },
                {
                    "audit": "recomputed_residual_identity_max_abs",
                    "league": league,
                    "n_rows": int(center.sum()),
                    "value": float(
                        (
                            group.loc[center, "historical_pred_fp_per_game"]
                            + group.loc[
                                center, "v2_template_active_ppg_resid"
                            ]
                            - group.loc[center, "template_active_ppg"]
                        )
                        .abs()
                        .max()
                    ),
                },
            ]
        )
    for league, group in current_joined.groupby("league"):
        rows.append(
            {
                "audit": "current_player_map_identity_join_rate",
                "league": league,
                "n_rows": len(group),
                "value": float(group["v2_position"].notna().mean()),
            }
        )
        matched = group["v2_position"].notna()
        rows.append(
            {
                "audit": "current_player_map_position_mismatch_rate",
                "league": league,
                "n_rows": int(matched.sum()),
                "value": float(
                    (
                        group.loc[matched, "position"]
                        != group.loc[matched, "v2_position"]
                    ).mean()
                ),
            }
        )
        rows.append(
            {
                "audit": "current_player_map_point_center_coverage",
                "league": league,
                "n_rows": len(group),
                "value": float(
                    group["historical_pred_fp_per_game"].notna().mean()
                ),
            }
        )

    historical_unmatched = joined[joined["display_name"].isna()][
        [
            "league",
            "template_id",
            "player_key",
            "player",
            "position",
            "season",
        ]
    ].drop_duplicates()
    current_unmatched = current_joined[current_joined["v2_position"].isna()][
        [
            "league",
            "player_key",
            "player",
            "position",
            "season",
        ]
    ].drop_duplicates()
    current_unmatched.insert(1, "template_id", pd.NA)
    unmatched = pd.concat(
        [historical_unmatched, current_unmatched],
        ignore_index=True,
    )
    unmatched["reason"] = np.where(
        unmatched["season"].eq(2026),
        "current_player_key_season_not_found_in_v2_handoff",
        "player_key_season_not_found_in_v2_handoff",
    )
    audit = pd.DataFrame(rows)
    run_id = str(handoff["model_run_id"].dropna().iloc[0])
    lock_version = str(handoff["lock_version"].dropna().iloc[0])
    for frame in (audit, unmatched):
        frame.insert(0, "model_run_id", run_id)
        frame.insert(0, "lock_version", lock_version)
    return audit, unmatched


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    handoff = _load_handoff(args.output_db)
    templates, player_map = _load_production_templates(
        args.simulation_db,
        str(args.league),
    )
    audit, unmatched = build_audit(handoff, templates, player_map)

    persisted_handoff = handoff.copy()
    persisted_handoff.to_csv(
        args.results_dir / "locked_template_handoff.csv", index=False
    )
    audit.to_csv(
        args.results_dir / "locked_template_production_join_audit.csv",
        index=False,
    )
    unmatched.to_csv(
        args.results_dir / "locked_template_production_unmatched.csv",
        index=False,
    )
    publish_tables_atomic(
        args.output_db,
        {
            "locked_template_handoff": persisted_handoff,
            "locked_template_production_join_audit": audit,
            "locked_template_production_unmatched": unmatched,
        },
    )
    print(audit.to_string(index=False))
    print(f"\nUnmatched production template rows: {len(unmatched)}")


if __name__ == "__main__":
    main()
