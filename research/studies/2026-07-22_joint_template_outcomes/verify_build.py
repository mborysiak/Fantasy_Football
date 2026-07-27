"""Verify adaptive weekly-template matching and cross-repo synchronization."""

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DB = ROOT / "Data" / "Databases" / "Simulation.sqlite3"
APP_DB = ROOT.parent / "Fantasy_Football_App" / "app" / "Simulation.sqlite3"


def query_metrics(database):
    with sqlite3.connect(database) as conn:
        bell = conn.execute(
            """
            SELECT t.template_eligible,
                   t.template_exclusion_reason,
                   COUNT(p.template_id)
            FROM Best_Ball_Weekly_Templates t
            LEFT JOIN Best_Ball_Weekly_Template_Pools p
              ON p.template_id = t.template_id
             AND p.template_league = t.league
            WHERE t.league = 'beta'
              AND t.player = "Le'Veon Bell"
              AND t.pos = 'RB'
              AND t.season = 2018
            GROUP BY 1, 2
            """
        ).fetchone()
        pool_metrics = conn.execute(
            """
            SELECT pos,
                   COUNT(*),
                   MIN(effective_sample_size),
                   AVG(effective_sample_size),
                   MAX(max_template_sample_prob),
                   AVG(local_weight_fraction)
            FROM Best_Ball_Weekly_Pool_Summary
            WHERE year = 2026
              AND version = 'beta'
              AND dataset = 'final_ensemble'
            GROUP BY pos
            ORDER BY pos
            """
        ).fetchall()
        pool_sums = conn.execute(
            """
            SELECT COUNT(*), MAX(ABS(sample_prob_sum - 1.0))
            FROM (
                SELECT template_pool_key,
                       SUM(template_sample_prob) sample_prob_sum
                FROM Best_Ball_Weekly_Template_Pools
                WHERE pool_year = 2026
                  AND pool_version = 'beta'
                  AND pool_dataset = 'final_ensemble'
                GROUP BY template_pool_key
            )
            """
        ).fetchone()
        eligible_zero_active = conn.execute(
            """
            SELECT COUNT(DISTINCT t.template_id), COUNT(*)
            FROM Best_Ball_Weekly_Templates t
            INNER JOIN Best_Ball_Weekly_Template_Pools p
              ON p.template_id = t.template_id
             AND p.template_league = t.league
            WHERE t.league = 'beta'
              AND t.active_games = 0
              AND t.template_eligible = 1
            """
        ).fetchone()
        missing_features = conn.execute(
            """
            SELECT COUNT(*)
            FROM Best_Ball_Weekly_Player_Map
            WHERE year = 2026
              AND version = 'beta'
              AND dataset = 'final_ensemble'
              AND (
                  match_projection_ppg_scaled IS NULL
                  OR projection_disagreement_frac IS NULL
                  OR rank_disagreement_scaled IS NULL
                  OR market_projection_gap IS NULL
                  OR (pos = 'RB' AND rb_room_rank_scaled IS NULL)
                  OR (
                      pos IN ('WR', 'TE')
                      AND pass_catcher_rank_scaled IS NULL
                  )
              )
            """
        ).fetchone()[0]
    return bell, pool_metrics, pool_sums, eligible_zero_active, missing_features


def main():
    source = query_metrics(SOURCE_DB)
    app = query_metrics(APP_DB)
    assert source == app, "Source and auction-app weekly tables disagree."

    bell, pool_metrics, pool_sums, eligible_zero_active, missing_features = source
    assert bell == (0, "contract_holdout", 0)
    assert pool_sums[0] == 180
    assert pool_sums[1] < 1e-10
    assert eligible_zero_active[0] > 0 and eligible_zero_active[1] > 0
    assert missing_features == 0
    assert all(row[4] <= 0.0500001 for row in pool_metrics)
    assert all(row[2] >= 40 for row in pool_metrics)

    print("Bell exclusion:", bell)
    print("Eligible zero-active donors/uses:", eligible_zero_active)
    print("Pool probability sums:", pool_sums)
    print("Pool metrics by position:")
    for row in pool_metrics:
        print(row)


if __name__ == "__main__":
    main()

