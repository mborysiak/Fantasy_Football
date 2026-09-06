"""Validate and summarize the paired breakout review publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "Data" / "Databases" / "Simulation.sqlite3"
DEFAULT_APP = REPO_ROOT.parent / "Fantasy_Football_App" / "app" / "Simulation.sqlite3"
RESULTS = Path(__file__).resolve().parent / "results"
TABLES = (
    "Breakout_Paired_Templates",
    "Breakout_Paired_Template_Pools",
    "Breakout_Paired_Player_Map",
    "Breakout_Paired_Template_Audit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--app-db", type=Path, default=DEFAULT_APP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_db.resolve()
    app = args.app_db.resolve()
    RESULTS.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(source) as connection:
        connection.execute("ATTACH DATABASE ? AS app_db", (str(app),))
        parity = {}
        counts = {}
        for table in TABLES:
            source_count = int(
                connection.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0]
            )
            app_count = int(
                connection.execute(f'SELECT COUNT(*) FROM app_db."{table}"').fetchone()[0]
            )
            different = bool(
                connection.execute(
                    "SELECT "
                    f'EXISTS(SELECT * FROM main."{table}" EXCEPT '
                    f'SELECT * FROM app_db."{table}") OR '
                    f'EXISTS(SELECT * FROM app_db."{table}" EXCEPT '
                    f'SELECT * FROM main."{table}")'
                ).fetchone()[0]
            )
            counts[table] = {"source": source_count, "app": app_count}
            parity[table] = not different and source_count == app_count

        audit = pd.read_sql_query(
            "SELECT * FROM Breakout_Paired_Template_Audit ORDER BY league",
            connection,
        )
        probability = pd.read_sql_query(
            """
            SELECT target_league league,
                   COUNT(*) pool_count,
                   MIN(pool_size) min_pool_size,
                   MAX(pool_size) max_pool_size,
                   MIN(probability_sum) min_probability_sum,
                   MAX(probability_sum) max_probability_sum
            FROM (
                SELECT target_league, target_player_key,
                       COUNT(*) pool_size,
                       SUM(template_sample_prob) probability_sum
                FROM Breakout_Paired_Template_Pools
                GROUP BY target_league, target_player_key
            )
            GROUP BY target_league
            ORDER BY target_league
            """,
            connection,
        )
        top_beta = pd.read_sql_query(
            """
            SELECT player, pos, avg_pick, pred_fp_per_game,
                   breakout_signed_next_growth,
                   template_current_breakout_rate,
                   template_playoff_hit_rate,
                   template_future_high_performer_rate,
                   template_current_and_future_rate,
                   template_playoff_and_future_rate
            FROM Breakout_Paired_Player_Map
            WHERE profile_version='paired_breakout_v1'
              AND year=2026 AND league='beta' AND dataset='final_ensemble'
              AND is_keeper=0 AND (avg_pick >= 50 OR avg_pick IS NULL)
            ORDER BY template_current_and_future_rate DESC,
                     pred_fp_per_game DESC, player
            LIMIT 40
            """,
            connection,
        )
        keepers = pd.read_sql_query(
            """
            SELECT league, player, pos, keeper_salary,
                   breakout_signed_next_growth,
                   template_current_and_future_rate
            FROM Breakout_Paired_Player_Map
            WHERE profile_version='paired_breakout_v1' AND is_keeper=1
            ORDER BY league, player
            """,
            connection,
        )
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA main.integrity_check")
        ]
        app_integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA app_db.integrity_check")
        ]

    audit.to_csv(RESULTS / "generation_audit.csv", index=False)
    probability.to_csv(RESULTS / "pool_probability_audit.csv", index=False)
    top_beta.to_csv(RESULTS / "beta_late_market_review.csv", index=False)
    keepers.to_csv(RESULTS / "keeper_diagnostic_rows.csv", index=False)
    summary = {
        "source_db": str(source),
        "app_db": str(app),
        "table_counts": counts,
        "exact_table_parity": parity,
        "source_integrity": integrity,
        "app_integrity": app_integrity,
        "all_probability_sums_one": bool(
            probability.min_probability_sum.between(1 - 1e-8, 1 + 1e-8).all()
            and probability.max_probability_sum.between(1 - 1e-8, 1 + 1e-8).all()
        ),
        "all_pools_eighty": bool(
            probability.min_pool_size.eq(80).all()
            and probability.max_pool_size.eq(80).all()
        ),
        "salary_match_feature_count": int(audit.salary_match_feature_count.sum()),
        "invalid_appearance_rows": int(audit.invalid_appearance_rows.sum()),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not all(parity.values()):
        raise ValueError(f"Source/app table parity failed: {parity}")
    if integrity != ["ok"] or app_integrity != ["ok"]:
        raise ValueError("SQLite integrity failed")
    if not summary["all_probability_sums_one"] or not summary["all_pools_eighty"]:
        raise ValueError("Pool probability or size contract failed")
    if summary["salary_match_feature_count"] != 0:
        raise ValueError("Salary entered paired breakout matching")
    if summary["invalid_appearance_rows"] != 0:
        raise ValueError("Paired N+1 appearance contract failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
