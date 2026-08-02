"""Read-only release gates for the 2026 V2 population cutover."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


YEAR = 2026
DATASET = "final_ensemble"
EXPECTED_POSITION_COUNTS = {
    "dk": {"QB": 55, "RB": 100, "WR": 143, "TE": 53},
    "beta": {"QB": 50, "RB": 95, "WR": 133, "TE": 50},
}
EXPECTED_GOVERNED_ADP_FALLBACKS = {"dk": 14, "beta": 91}
EXPECTED_DK_EXCLUSIONS = {
    "3f0b675d-ef58-5606-8f9e-73bc2a9b4118",  # Kareem Hunt
    "677b8fa5-8879-5913-8a35-9a71859ab8a3",  # Austin Ekeler
    "7ae33581-c9ae-51b6-a8d5-fe24f3e5615a",  # Tyreek Hill
    "e492c31b-21c9-55b9-b007-4dd0d8fd1ad4",  # Nick Chubb
    "f973b1c8-3470-57f5-bc68-42e35a830411",  # Joe Mixon
}
SALARY_METHOD = "current_locked_spec_v6_v2_population_11f"
PREMIUM_TRANSFER_POLICY = (
    "historical_v5_selection_surface_to_current_v6_v1"
)


def connect_read_only(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
    )


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def read_slice(
    connection: sqlite3.Connection,
    table: str,
    league: str,
) -> pd.DataFrame:
    league_column = (
        "pool_version"
        if table == "Best_Ball_Weekly_Template_Pools"
        else "version"
    )
    return pd.read_sql_query(
        f'''SELECT *
              FROM "{table}"
             WHERE year=? AND {league_column}=? AND dataset=?''',
        connection,
        params=(YEAR, league, DATASET),
    )


def validate_population_and_weekly(
    connection: sqlite3.Connection,
) -> dict[str, object]:
    results: dict[str, object] = {}
    require(
        connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
        "Simulation.sqlite3 failed integrity_check",
    )
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    require(not foreign_keys, f"Simulation foreign-key errors: {foreign_keys[:10]}")

    eligibility = pd.read_sql_query(
        """SELECT *
             FROM V2_Production_Eligibility_Audit
            WHERE year=? AND dataset=?""",
        connection,
        params=(YEAR, DATASET),
    )
    excluded = eligibility.loc[eligibility.governed_excluded.eq(1)]
    require(
        set(excluded.player_key) == EXPECTED_DK_EXCLUSIONS,
        "Governed DK exclusion set changed",
    )
    require(
        excluded.league.eq("dk").all()
        and excluded.governed_exclusion_reason.eq(
            "market_only_without_current_projection_center"
        ).all(),
        "Governed exclusions lost their reviewed policy",
    )

    for league, expected_by_position in EXPECTED_POSITION_COUNTS.items():
        expected_count = sum(expected_by_position.values())
        final = pd.read_sql_query(
            """SELECT *
                 FROM Final_Predictions_Resid
                WHERE year=? AND version=? AND dataset=?""",
            connection,
            params=(YEAR, league, DATASET),
        )
        require(len(final) == expected_count, f"{league} Final population changed")
        require(
            final.player_key.notna().all() and not final.player_key.duplicated().any(),
            f"{league} Final player keys are incomplete",
        )
        require(
            final.groupby("pos").size().to_dict() == expected_by_position,
            f"{league} position counts changed",
        )
        require(
            final.player.eq("Tetairoa McMillan").sum() == 1,
            f"{league} Tetairoa display identity regressed",
        )
        require(
            final.player.eq("Amon-Ra St. Brown").sum() == 1,
            f"{league} Amon-Ra display identity regressed",
        )

        player_map = pd.read_sql_query(
            """SELECT *
                 FROM Best_Ball_Weekly_Player_Map
                WHERE year=? AND version=? AND dataset=?""",
            connection,
            params=(YEAR, league, DATASET),
        )
        require(
            set(player_map.player_key) == set(final.player_key),
            f"{league} player-map population differs from Final",
        )
        require(
            player_map.current_context_missing_fields.fillna("").eq("").all(),
            f"{league} has required current-context gaps",
        )

        adp = pd.read_sql_query(
            """SELECT *
                 FROM Best_Ball_ADP_Audit
                WHERE year=? AND version=? AND dataset=?""",
            connection,
            params=(YEAR, league, DATASET),
        )
        require(len(adp) == expected_count, f"{league} ADP audit coverage changed")
        require(
            int(adp.using_default_adp.sum()) == 0
            and int(adp.needs_review.sum()) == 0
            and int(adp.high_impact_unresolved_adp.sum()) == 0,
            f"{league} has unresolved/default ADP rows",
        )
        require(
            int(adp.governed_context_adp_fallback.sum())
            == EXPECTED_GOVERNED_ADP_FALLBACKS[league],
            f"{league} governed ADP fallback count changed",
        )

        pools = pd.read_sql_query(
            """SELECT *
                 FROM Best_Ball_Weekly_Template_Pools
                WHERE pool_year=? AND pool_version=? AND pool_dataset=?""",
            connection,
            params=(YEAR, league, DATASET),
        )
        pool_summary = pools.groupby("template_pool_key").agg(
            donors=("template_id", "size"),
            unique_donors=("template_id", "nunique"),
            probability=("template_sample_prob", "sum"),
        )
        require(
            len(pool_summary) == expected_count,
            f"{league} template-pool population changed",
        )
        require(
            pool_summary.donors.eq(80).all()
            and pool_summary.unique_donors.eq(80).all(),
            f"{league} template pools are not 80 unique donors",
        )
        require(
            np.allclose(pool_summary.probability, 1.0, atol=1e-10),
            f"{league} template probabilities do not sum to one",
        )
        require(
            pools.template_sample_prob.gt(0).all()
            and pools.template_sample_prob.le(0.05 + 1e-12).all(),
            f"{league} template probability bounds failed",
        )

        results[f"{league}_players"] = expected_count
        results[f"{league}_governed_adp_fallbacks"] = int(
            adp.governed_context_adp_fallback.sum()
        )

    dk_map = pd.read_sql_query(
        """SELECT *
             FROM Best_Ball_Weekly_Player_Map
            WHERE year=? AND version='dk' AND dataset=?""",
        connection,
        params=(YEAR, DATASET),
    ).set_index("player")
    require(
        dk_map.loc["Ty Simpson", "qb_team_rank"] == 2
        and np.isclose(
            dk_map.loc["Ty Simpson", "team_qb_proj_points"],
            dk_map.loc["Matthew Stafford", "team_qb_proj_points"],
        ),
        "LA/LAR quarterback room is fragmented",
    )
    require(
        0 < dk_map.loc["Jarquez Hunter", "rb_combined_share_of_room"] < 1,
        "LA/LAR running-back room is fragmented",
    )
    require(
        0 < dk_map.loc["Elijah Higgins", "team_rec_share"] < 1,
        "ARI/ARZ pass-catcher room is fragmented",
    )
    fa_rows = dk_map.loc[dk_map.team.eq("FA")]
    room_columns = [
        "rb_rush_share_of_room",
        "rb_rec_share_of_room",
        "rb_combined_share_of_room",
        "rb_room_rank_scaled",
        "rb_gap_to_next_share",
        "rb_room_concentration",
        "team_rec_share",
        "pass_catcher_rank_scaled",
        "pass_catcher_gap_to_next_share",
        "pass_catcher_room_concentration",
        "team_qb_proj_points",
        "qb_room_share",
        "team_qb1_proj_points",
        "team_qb2_proj_points",
        "qb1_over_qb2_gap_pct",
        "team_qb_pass_points",
    ]
    require(
        not fa_rows.empty
        and np.allclose(
            fa_rows[room_columns].fillna(0).to_numpy(dtype=float),
            0,
        ),
        "Free agents participate in a synthetic team room",
    )
    results["governed_dk_exclusions"] = len(excluded)
    return results


def validate_salary_and_reserve(
    simulation: sqlite3.Connection,
    validations: sqlite3.Connection,
) -> dict[str, object]:
    final = pd.read_sql_query(
        """SELECT player_key, player
             FROM Final_Predictions_Resid
            WHERE year=? AND version='beta' AND dataset=?""",
        simulation,
        params=(YEAR, DATASET),
    )
    salaries = pd.read_sql_query(
        """SELECT *
             FROM Salaries_Pred
            WHERE year=? AND league='betapred'""",
        simulation,
        params=(YEAR,),
    )
    keepers = pd.read_sql_query(
        """SELECT *
             FROM League_Keepers
            WHERE year=? AND league='beta'""",
        simulation,
        params=(YEAR,),
    )
    require(
        len(salaries) == len(final) == 328
        and set(salaries.player_key) == set(final.player_key),
        "Salary population differs from beta production",
    )
    require(
        salaries.player_key.notna().all()
        and not salaries.player_key.duplicated().any(),
        "Salary player keys are incomplete",
    )
    require(
        salaries.salary_method_version.eq(SALARY_METHOD).all(),
        "Salary method version changed",
    )
    require(
        salaries[
            [
                "salary",
                "std_dev",
                "min_score",
                "max_score",
                "salary_resid_5",
                "salary_resid_95",
            ]
        ]
        .notna()
        .all()
        .all(),
        "Salary output contains missing required values",
    )
    require(
        len(keepers) == 14
        and keepers.player_key.notna().all()
        and set(keepers.player_key).issubset(set(final.player_key)),
        "Keeper key contract failed",
    )
    nonkeeper = salaries.loc[
        ~salaries.player_key.isin(keepers.player_key)
    ]
    require(
        len(nonkeeper) == 314
        and np.isclose(
            nonkeeper.nlargest(142, "salary").salary.sum(),
            3071.0,
            atol=1e-7,
        ),
        "Salary top-slot market does not reconcile to remaining budget",
    )
    require(
        salaries.salary_population_source.value_counts().to_dict()
        == {
            "model_inputs_projonly": 326,
            "v2_player_season_features_fallback": 2,
        },
        "Salary population provenance changed",
    )

    premiums = pd.read_sql_query(
        """SELECT *
             FROM Salary_Selection_Premium
            WHERE year=? AND league='beta'""",
        simulation,
        params=(YEAR,),
    )
    require(
        len(premiums) == 314
        and set(premiums.player_key) == set(nonkeeper.player_key),
        "Reserve-premium population differs from non-keeper salary surface",
    )
    require(
        premiums.salary_method_version.eq(SALARY_METHOD).all()
        and premiums.seed_method_version.eq(
            "app_target_selection_only_keeper_portfolio_v3"
        ).all()
        and premiums.calibration_transfer_policy.eq(
            PREMIUM_TRANSFER_POLICY
        ).all(),
        "Reserve-premium provenance changed",
    )
    require(
        premiums.seed_trials.eq(1000).all()
        and premiums.seed_success_trials.eq(1000).all(),
        "Reserve refresh is not the required fresh 1000/1000 seed",
    )

    seeds = pd.read_sql_query(
        """SELECT *
             FROM Salary_Selection_Seeds
            WHERE year=? AND league='beta'""",
        validations,
        params=(YEAR,),
    )
    require(
        len(seeds) == 314
        and set(seeds.player_key) == set(nonkeeper.player_key)
        and seeds.salary_method_version.eq(SALARY_METHOD).all()
        and seeds.seed_trials.eq(1000).all()
        and seeds.seed_success_trials.eq(1000).all(),
        "Persisted current reserve seed failed",
    )
    return {
        "salary_players": len(salaries),
        "salary_v2_fallback_players": int(
            salaries.salary_population_source.eq(
                "v2_player_season_features_fallback"
            ).sum()
        ),
        "keepers": len(keepers),
        "reserve_players": len(premiums),
        "reserve_expected_roster_dollars": float(
            (premiums.selection_rate * premiums.applied_premium).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    simulation_path = args.database_dir / "Simulation.sqlite3"
    validations_path = args.database_dir / "Validations.sqlite3"

    with connect_read_only(simulation_path) as simulation:
        results = validate_population_and_weekly(simulation)
        with connect_read_only(validations_path) as validations:
            require(
                validations.execute("PRAGMA integrity_check").fetchone()[0]
                == "ok",
                "Validations.sqlite3 failed integrity_check",
            )
            results.update(
                validate_salary_and_reserve(simulation, validations)
            )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
