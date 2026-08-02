"""Refresh the annual auction optimizer-selection decision-price reserve.

This is intentionally a lightweight second-stage workflow:

1. run one premium-free preseason Target seed for the active season;
2. fit the strictly prior-season ridge calibration to saved seed rates and
   actual-minus-point salary residuals; and
3. publish a static per-player premium for Target and Nomination ILPs.

The coherent salary prediction remains the displayed market price.  The
premium is a separate affordability reserve and is never applied to keepers,
entered auction purchases, or an explicitly priced nominee.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Scripts.config import YEAR as DEFAULT_YEAR  # noqa: E402

APP_ROOT = ROOT.parent / "Fantasy_Football_App"
APP_DIR = APP_ROOT / "app"
SIMULATION_DB = ROOT / "Data" / "Databases" / "Simulation.sqlite3"
VALIDATIONS_DB = ROOT / "Data" / "Databases" / "Validations.sqlite3"
APP_SIMULATION_DB = APP_DIR / "Simulation.sqlite3"

BOOTSTRAP_ROSTERS = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-16_optimizer_selection_surcharge"
    / "results"
    / "roster_trials.csv"
)
BOOTSTRAP_CANDIDATES = (
    ROOT
    / "research"
    / "studies"
    / "2026-07-16_salary_v5_replay"
    / "results"
    / "selected_residuals_v5"
    / "candidate_diagnostic.csv"
)

SEED_TABLE = "Salary_Selection_Seeds"
CALIBRATOR_TABLE = "Salary_Selection_Calibrator"
PREMIUM_TABLE = "Salary_Selection_Premium"
SALARY_METHOD_VERSION = "current_locked_spec_v6_v2_population_11f"
HISTORICAL_SALARY_METHOD_VERSION = (
    "current_locked_spec_v5_compact_salary_features"
)
HISTORICAL_SEED_METHOD = "target_managed_baseline_298_reconstructed_v1"
CURRENT_SEED_METHOD = "app_target_selection_only_keeper_portfolio_v3"
PREMIUM_METHOD_VERSION = "ridge_a100_positive_cap10_v1"
CALIBRATION_TRANSFER_POLICY = (
    "historical_v5_selection_surface_to_current_v6_v1"
)
RIDGE_ALPHA = 100.0
PREMIUM_CAP = 10.0

NUM_TEAMS = 12
TEAM_BUDGET = 298
ROSTER_SIZE = 13
LINEUP_REQUIRE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
POS_MIN = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
POS_MAX = {"QB": 1, "RB": 7, "WR": 7, "TE": 2}
TOTAL_BUDGET = NUM_TEAMS * TEAM_BUDGET
TOTAL_SLOTS = NUM_TEAMS * ROSTER_SIZE


def player_key(value: object) -> str:
    """Match the compact alphanumeric keys used by the replay diagnostics."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if not table_exists(conn, table):
        return pd.DataFrame()
    return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def replace_table(conn: sqlite3.Connection, table: str, rows: pd.DataFrame) -> None:
    rows.to_sql(table, conn, if_exists="replace", index=False)


def replace_active_slice(
    conn: sqlite3.Connection,
    table: str,
    rows: pd.DataFrame,
    year: int,
    league: str,
) -> None:
    existing = read_table(conn, table)
    if not existing.empty and {"year", "league"}.issubset(existing.columns):
        keep = ~(
            pd.to_numeric(existing.year, errors="coerce").eq(int(year))
            & existing.league.astype(str).eq(str(league))
        )
        rows = pd.concat([existing.loc[keep], rows], ignore_index=True, sort=False)
    replace_table(conn, table, rows)


def clear_active_slice(
    conn: sqlite3.Connection,
    table: str,
    year: int,
    league: str,
) -> int:
    """Remove a generated active slice before a deliberately clean rebuild."""
    if not table_exists(conn, table):
        return 0
    columns = {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    }
    missing = {"year", "league"} - columns
    if missing:
        raise ValueError(
            f"{table} cannot be cleared by active slice; missing columns: "
            f"{sorted(missing)}."
        )
    cursor = conn.execute(
        f'DELETE FROM "{table}" WHERE year = ? AND league = ?',
        (int(year), str(league)),
    )
    return max(int(cursor.rowcount), 0)


def replace_league_slice(
    conn: sqlite3.Connection,
    table: str,
    rows: pd.DataFrame,
    league: str,
) -> None:
    existing = read_table(conn, table)
    if not existing.empty and "league" in existing.columns:
        rows = pd.concat(
            [existing.loc[~existing.league.astype(str).eq(str(league))], rows],
            ignore_index=True,
            sort=False,
        )
    replace_table(conn, table, rows)


def reconstruct_historical_seeds(league: str) -> pd.DataFrame:
    """Bootstrap the durable seed history from the validated Target replay."""
    if league != "beta":
        raise ValueError(
            "The bundled historical Target bootstrap is beta-specific. "
            f"Persist a validated {league!r} seed history before refreshing that league."
        )
    if not BOOTSTRAP_ROSTERS.exists() or not BOOTSTRAP_CANDIDATES.exists():
        raise FileNotFoundError(
            "Historical selection seeds are absent and the validated bootstrap "
            "study outputs could not be found."
        )

    rosters = pd.read_csv(BOOTSTRAP_ROSTERS)
    rosters = rosters[
        rosters.variant.eq("baseline_298") & rosters.status.eq("optimal")
    ].copy()
    if rosters.empty:
        raise ValueError("No optimal baseline_298 bootstrap rosters were found.")
    if rosters.duplicated(["year", "trial"]).any():
        raise ValueError("Bootstrap roster trials are not unique by year/trial.")

    trial_counts = rosters.groupby("year").trial.nunique().rename("seed_trials")
    roster_players = rosters[["year", "trial", "roster"]].copy()
    roster_players["player"] = roster_players.roster.str.split("|")
    roster_players = roster_players.explode("player", ignore_index=True)
    if roster_players.duplicated(["year", "trial", "player"]).any():
        raise ValueError("A bootstrap roster contains a duplicated player.")
    selected = (
        roster_players.groupby(["year", "player"])
        .size()
        .rename("selection_slots")
        .reset_index()
    )

    candidates = pd.read_csv(BOOTSTRAP_CANDIDATES)
    required = {
        "year",
        "player",
        "player_key",
        "pos",
        "point_salary",
        "actual_salary",
        "actual_salary_recorded",
        "salary_residual",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Bootstrap candidate diagnostics lack columns: {missing}")
    candidates = candidates[list(required)].copy()
    candidates["year"] = candidates.year.astype(int)
    if candidates.duplicated(["year", "player_key"]).any():
        raise ValueError("Bootstrap candidates duplicate a player origin.")

    seeds = candidates.merge(
        selected,
        on=["year", "player"],
        how="left",
        validate="one_to_one",
    )
    seeds["selection_slots"] = seeds.selection_slots.fillna(0).astype(int)
    seeds = seeds.merge(
        trial_counts.reset_index(),
        on="year",
        how="left",
        validate="many_to_one",
    )
    if seeds.seed_trials.isna().any():
        raise ValueError("Bootstrap candidates do not all map to a trial count.")
    seeds["seed_trials"] = seeds.seed_trials.astype(int)
    seeds["seed_success_trials"] = seeds.seed_trials
    seeds["selection_rate"] = seeds.selection_slots / seeds.seed_success_trials
    seeds["actual_salary_recorded"] = (
        seeds.actual_salary_recorded.fillna(0).astype(int)
    )
    seeds["league"] = league
    # The bundled replay was generated from the v5 salary surface. Preserve
    # that provenance; relabeling these rows as v6 would make the calibration
    # look same-surface when it is an explicit prior-surface transfer.
    seeds["salary_method_version"] = HISTORICAL_SALARY_METHOD_VERSION
    seeds["seed_method_version"] = HISTORICAL_SEED_METHOD
    seeds["seed_random_seed"] = np.nan
    seeds["generated_at"] = datetime.now(timezone.utc).isoformat()
    return seeds[
        [
            "year",
            "league",
            "player",
            "player_key",
            "pos",
            "point_salary",
            "selection_rate",
            "selection_slots",
            "seed_trials",
            "seed_success_trials",
            "actual_salary",
            "actual_salary_recorded",
            "salary_residual",
            "salary_method_version",
            "seed_method_version",
            "seed_random_seed",
            "generated_at",
        ]
    ].sort_values(["year", "player"]).reset_index(drop=True)


def load_or_bootstrap_seeds(league: str) -> pd.DataFrame:
    with sqlite3.connect(VALIDATIONS_DB) as conn:
        seeds = read_table(conn, SEED_TABLE)
    if seeds.empty:
        seeds = reconstruct_historical_seeds(league)
    else:
        seeds = seeds[seeds.league.eq(league)].copy()
        if seeds.empty:
            seeds = reconstruct_historical_seeds(league)
    return seeds


def refresh_realized_salaries(
    seeds: pd.DataFrame,
    target_year: int,
    league: str,
) -> pd.DataFrame:
    """Attach newly available prior-season non-keeper auction outcomes."""
    seeds = seeds.copy()
    with sqlite3.connect(SIMULATION_DB) as conn:
        actual = pd.read_sql_query(
            """SELECT year, player, actual_salary
                 FROM Actual_Salaries
                WHERE league=? AND year<? AND COALESCE(is_keeper, 0)=0""",
            conn,
            params=(league, int(target_year)),
        )
    if actual.empty:
        return seeds
    if actual.duplicated(["year", "player"]).any():
        raise ValueError("Actual_Salaries duplicates a non-keeper player origin.")
    actual = actual.rename(columns={"actual_salary": "refreshed_actual_salary"})
    seeds = seeds.merge(
        actual,
        on=["year", "player"],
        how="left",
        validate="one_to_one",
    )
    has_actual = seeds.refreshed_actual_salary.notna() & seeds.year.lt(target_year)
    seeds.loc[has_actual, "actual_salary"] = seeds.loc[
        has_actual, "refreshed_actual_salary"
    ]
    seeds.loc[has_actual, "actual_salary_recorded"] = 1
    seeds.loc[has_actual, "salary_residual"] = (
        pd.to_numeric(seeds.loc[has_actual, "actual_salary"], errors="coerce")
        - pd.to_numeric(seeds.loc[has_actual, "point_salary"], errors="coerce")
    )
    return seeds.drop(columns="refreshed_actual_salary")


def calibration_features(frame: pd.DataFrame) -> pd.DataFrame:
    salary = pd.to_numeric(frame.point_salary, errors="coerce").clip(lower=1.0)
    selection = (
        pd.to_numeric(frame.selection_rate, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    output = pd.DataFrame(index=frame.index)
    output["pos"] = frame.pos.fillna("UNK").astype(str)
    output["pred_salary"] = salary
    output["pred_salary_sq"] = (salary / 25.0) ** 2
    output["selection_rate"] = selection
    output["selection_x_salary"] = selection * salary
    for pos in ("QB", "RB", "TE"):
        output[f"selection_x_{pos}"] = selection * frame.pos.eq(pos).astype(float)
    return output


def fit_calibrator(
    seeds: pd.DataFrame,
    target: pd.DataFrame,
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    training = seeds[
        seeds.year.lt(target_year)
        & pd.to_numeric(seeds.actual_salary_recorded, errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(seeds.salary_residual, errors="coerce").notna()
    ].copy()
    if training.empty:
        raise ValueError(f"No observed prior-origin premium training rows for {target_year}.")
    if training.year.max() >= target_year:
        raise AssertionError("Premium calibration crossed the target season.")
    training_salary_methods = sorted(
        training.salary_method_version.dropna().astype(str).unique().tolist()
    )
    training_seed_methods = sorted(
        training.seed_method_version.dropna().astype(str).unique().tolist()
    )
    if training_salary_methods != [HISTORICAL_SALARY_METHOD_VERSION]:
        raise ValueError(
            "Premium calibration expected the governed historical v5 salary "
            f"surface, found {training_salary_methods}."
        )
    if training_seed_methods != [HISTORICAL_SEED_METHOD]:
        raise ValueError(
            "Premium calibration expected the governed historical Target "
            f"seed method, found {training_seed_methods}."
        )

    x_train = calibration_features(training)
    x_target = calibration_features(target)
    numeric = [column for column in x_train.columns if column != "pos"]
    preprocessing = ColumnTransformer(
        [
            ("pos", OneHotEncoder(handle_unknown="ignore"), ["pos"]),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(x_train, training.salary_residual.to_numpy(dtype=float))
    predicted = model.predict(x_target)
    output = target.copy()
    output["predicted_salary_residual"] = predicted
    output["full_premium"] = np.clip(predicted, 0.0, PREMIUM_CAP)
    output["half_premium"] = output.full_premium * 0.5

    coefficient_rows = pd.DataFrame(
        {
            "target_year": int(target_year),
            "training_through_year": int(training.year.max()),
            "feature": model.named_steps["preprocessing"].get_feature_names_out(),
            "coefficient": model.named_steps["ridge"].coef_,
        }
    )
    coefficient_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "target_year": [int(target_year)],
                    "training_through_year": [int(training.year.max())],
                    "feature": ["intercept"],
                    "coefficient": [model.named_steps["ridge"].intercept_],
                }
            ),
            coefficient_rows,
        ],
        ignore_index=True,
    )
    coefficient_rows["ridge_alpha"] = RIDGE_ALPHA
    coefficient_rows["premium_cap"] = PREMIUM_CAP
    coefficient_rows["premium_method_version"] = PREMIUM_METHOD_VERSION
    coefficient_rows["calibration_transfer_policy"] = (
        CALIBRATION_TRANSFER_POLICY
    )
    coefficient_rows["training_salary_method_versions"] = ",".join(
        training_salary_methods
    )
    coefficient_rows["training_seed_method_versions"] = ",".join(
        training_seed_methods
    )
    coefficient_rows["generated_at"] = datetime.now(timezone.utc).isoformat()
    metadata = {
        "training_rows": int(len(training)),
        "training_through_year": int(training.year.max()),
        "training_origins": sorted(training.year.unique().astype(int).tolist()),
        "training_salary_method_versions": training_salary_methods,
        "training_seed_method_versions": training_seed_methods,
        "calibration_transfer_policy": CALIBRATION_TRANSFER_POLICY,
    }
    return output, coefficient_rows, metadata


def load_simulation_class():
    if not APP_DIR.exists():
        raise FileNotFoundError(f"Auction app directory not found: {APP_DIR}")
    sys.path.insert(0, str(APP_DIR))
    from zSim_Helper import FootballSimulation  # noqa: PLC0415

    return FootballSimulation


def keeper_state(year: int, league: str) -> pd.DataFrame:
    with sqlite3.connect(SIMULATION_DB) as conn:
        if not table_exists(conn, "League_Keepers"):
            return pd.DataFrame(
                columns=["player_key", "player", "keeper_salary"]
            )
        keeper_columns = {
            row[1]
            for row in conn.execute('PRAGMA table_info("League_Keepers")')
        }
        player_key_select = (
            "player_key," if "player_key" in keeper_columns else ""
        )
        keepers = pd.read_sql_query(
            f"""SELECT {player_key_select}
                        player, keeper_salary
                 FROM League_Keepers
                WHERE year=? AND league=?""",
            conn,
            params=(int(year), league),
        )
    if keepers.player.duplicated().any():
        raise ValueError("League_Keepers duplicates a player in the active slice.")
    return keepers


def run_current_seed(
    year: int,
    league: str,
    trials: int,
    workers: int,
    random_seed: int,
) -> tuple[pd.DataFrame, int, dict[str, float]]:
    FootballSimulation = load_simulation_class()
    keepers = keeper_state(year, league)
    keeper_players = keepers.player.tolist()
    keeper_spend = float(
        pd.to_numeric(keepers.keeper_salary, errors="coerce").fillna(0).sum()
    )
    remaining_budget = TOTAL_BUDGET - keeper_spend
    remaining_slots = TOTAL_SLOTS - len(keepers)

    conn = sqlite3.connect(SIMULATION_DB)
    try:
        sim = FootballSimulation(
            conn,
            int(year),
            LINEUP_REQUIRE,
            TEAM_BUDGET,
            "final_ensemble",
            league,
            sal_pred_actual="pred",
        )
        sim.load_weekly_template_profiles()
        waiver_baselines = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )

        if (
            "player_key" in keepers
            and keepers.player_key.notna().all()
        ):
            keeper_keys = keepers.player_key.astype(str).tolist()
            key_to_player = sim.player_data.set_index("player_key").player
            unknown_keepers = set(keeper_keys) - set(
                sim.player_data.player_key.astype(str)
            )
            if unknown_keepers:
                raise ValueError(
                    "Keeper keys are outside the canonical simulation pool: "
                    f"{sorted(unknown_keepers)}"
                )
            keeper_players = key_to_player.loc[keeper_keys].tolist()
            available_mask = ~sim.player_data.player_key.astype(str).isin(
                keeper_keys
            ).to_numpy()
        else:
            all_players = sim.player_data.player.to_numpy(dtype=object)
            available_mask = ~np.isin(all_players, keeper_players)
        point_salary = sim.normalize_salary_market_values(
            sim.player_data.salary.to_numpy(dtype=float),
            available_mask,
            remaining_market_budget=remaining_budget,
            remaining_market_slots=remaining_slots,
        )
        current_surface = sim.player_data.loc[
            available_mask, ["player_key", "player", "pos"]
        ].copy()
        current_surface["point_salary"] = point_salary[available_mask]

        selection = sim.run_sim_parallel(
            {"players": [], "salaries": []},
            keeper_players,
            int(trials),
            max_workers=int(workers),
            block_size=50,
            random_seed=int(random_seed),
            require_top_n=12,
            num_avg_pts=5,
            next_year_frac=0,
            enforce_top_n=True,
            scoring_mode="managed",
            roster_size=ROSTER_SIZE,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts=POS_MIN,
            pos_max_counts=POS_MAX,
            waiver_baselines=waiver_baselines,
            bench_upside_weight=0.0,
            managed_value_options=50,
            managed_context_draws=5,
            managed_context_refresh_interval=50,
            managed_roster_refinement=True,
            use_keeper_portfolio=True,
            remaining_market_budget=remaining_budget,
            remaining_market_slots=remaining_slots,
            selection_only=True,
            use_selection_premium=False,
        )
        success_trials = int(getattr(sim, "last_success_trials", 0))
    finally:
        conn.close()

    if success_trials <= 0:
        raise RuntimeError("The current Target seed produced no optimal rosters.")
    selection = selection[["player", "SelectionCounts"]].copy()
    selection["selection_rate"] = (
        pd.to_numeric(selection.SelectionCounts, errors="coerce").fillna(0.0)
        / 100.0
    )
    current_surface = current_surface.merge(
        selection[["player", "selection_rate"]],
        on="player",
        how="left",
        validate="one_to_one",
    )
    current_surface["selection_rate"] = current_surface.selection_rate.fillna(0.0)
    current_surface["selection_slots"] = np.rint(
        current_surface.selection_rate * success_trials
    ).astype(int)
    if (
        current_surface.player_key.isna().any()
        or current_surface.player_key.duplicated().any()
    ):
        duplicated = current_surface.loc[
            current_surface.player_key.isna()
            | current_surface.player_key.duplicated(False),
            "player",
        ].tolist()
        raise ValueError(
            f"Current seed canonical player keys are incomplete: {duplicated}"
        )
    market = {
        "keeper_count": int(len(keepers)),
        "keeper_spend": keeper_spend,
        "remaining_budget": float(remaining_budget),
        "remaining_slots": int(remaining_slots),
    }
    return current_surface, success_trials, market


def current_seed_rows(
    surface: pd.DataFrame,
    year: int,
    league: str,
    requested_trials: int,
    success_trials: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = surface.copy()
    rows["year"] = int(year)
    rows["league"] = league
    rows["seed_trials"] = int(requested_trials)
    rows["seed_success_trials"] = int(success_trials)
    rows["actual_salary"] = np.nan
    rows["actual_salary_recorded"] = 0
    rows["salary_residual"] = np.nan
    rows["salary_method_version"] = SALARY_METHOD_VERSION
    rows["seed_method_version"] = CURRENT_SEED_METHOD
    rows["seed_random_seed"] = int(random_seed)
    rows["generated_at"] = datetime.now(timezone.utc).isoformat()
    return rows[
        [
            "year",
            "league",
            "player",
            "player_key",
            "pos",
            "point_salary",
            "selection_rate",
            "selection_slots",
            "seed_trials",
            "seed_success_trials",
            "actual_salary",
            "actual_salary_recorded",
            "salary_residual",
            "salary_method_version",
            "seed_method_version",
            "seed_random_seed",
            "generated_at",
        ]
    ]


def build_premium_rows(
    calibrated: pd.DataFrame,
    year: int,
    league: str,
    premium_strength: float,
    requested_trials: int,
    success_trials: int,
    metadata: dict[str, object],
) -> pd.DataFrame:
    output = calibrated.copy()
    output["year"] = int(year)
    output["league"] = league
    output["applied_premium"] = output.full_premium * float(premium_strength)
    output["premium_strength"] = float(premium_strength)
    output["training_through_year"] = metadata["training_through_year"]
    output["training_rows"] = metadata["training_rows"]
    output["ridge_alpha"] = RIDGE_ALPHA
    output["premium_cap"] = PREMIUM_CAP
    output["salary_method_version"] = SALARY_METHOD_VERSION
    output["seed_method_version"] = CURRENT_SEED_METHOD
    output["premium_method_version"] = PREMIUM_METHOD_VERSION
    output["calibration_transfer_policy"] = metadata[
        "calibration_transfer_policy"
    ]
    output["training_salary_method_versions"] = ",".join(
        metadata["training_salary_method_versions"]
    )
    output["training_seed_method_versions"] = ",".join(
        metadata["training_seed_method_versions"]
    )
    output["seed_trials"] = int(requested_trials)
    output["seed_success_trials"] = int(success_trials)
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    columns = [
        "year",
        "league",
        "player",
        "player_key",
        "pos",
        "point_salary",
        "selection_rate",
        "selection_slots",
        "predicted_salary_residual",
        "full_premium",
        "half_premium",
        "applied_premium",
        "premium_strength",
        "training_through_year",
        "training_rows",
        "ridge_alpha",
        "premium_cap",
        "salary_method_version",
        "seed_method_version",
        "premium_method_version",
        "calibration_transfer_policy",
        "training_salary_method_versions",
        "training_seed_method_versions",
        "seed_trials",
        "seed_success_trials",
        "generated_at",
    ]
    output = output[columns].sort_values(
        ["applied_premium", "selection_rate", "point_salary"],
        ascending=False,
    )
    if output.duplicated(["year", "league", "player"]).any():
        raise ValueError("Premium output duplicates a player in the active slice.")
    return output.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--league", default="beta")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260716)
    parser.add_argument("--premium-strength", type=float, default=0.5)
    parser.add_argument(
        "--simulation-db",
        type=Path,
        default=SIMULATION_DB,
        help="Simulation database to read and update.",
    )
    parser.add_argument(
        "--validations-db",
        type=Path,
        default=VALIDATIONS_DB,
        help="Validation database to read and update.",
    )
    parser.add_argument(
        "--app-simulation-db",
        type=Path,
        default=APP_SIMULATION_DB,
        help="Auction app database updated unless --no-app-sync is set.",
    )
    parser.add_argument(
        "--reuse-current-seed",
        action="store_true",
        help="Refit/re-publish from the saved active-season seed without rerunning Target.",
    )
    parser.add_argument(
        "--no-app-sync",
        action="store_true",
        help="Write source databases only.",
    )
    return parser.parse_args()


def main() -> None:
    global SIMULATION_DB
    global VALIDATIONS_DB
    global APP_SIMULATION_DB

    args = parse_args()
    default_simulation_db = SIMULATION_DB.resolve()
    default_validations_db = VALIDATIONS_DB.resolve()
    SIMULATION_DB = args.simulation_db.resolve()
    VALIDATIONS_DB = args.validations_db.resolve()
    APP_SIMULATION_DB = args.app_simulation_db.resolve()
    custom_simulation = SIMULATION_DB != default_simulation_db
    custom_validations = VALIDATIONS_DB != default_validations_db
    if custom_simulation != custom_validations:
        raise ValueError(
            "Custom reserve builds must provide both --simulation-db and "
            "--validations-db so live and staged surfaces cannot be mixed."
        )
    if custom_simulation and SIMULATION_DB.parent != VALIDATIONS_DB.parent:
        raise ValueError(
            "Custom Simulation and Validations databases must share one "
            "staging directory."
        )
    if custom_simulation and not args.no_app_sync:
        raise ValueError(
            "Custom staged reserve builds require --no-app-sync."
        )
    for database_name, database_path in (
        ("Simulation", SIMULATION_DB),
        ("Validations", VALIDATIONS_DB),
    ):
        if not database_path.exists():
            raise FileNotFoundError(
                f"{database_name} database not found: {database_path}"
            )
    if args.trials <= 0 or args.workers <= 0:
        raise ValueError("Trials and workers must be positive integers.")
    if not 0.0 <= args.premium_strength <= 1.0:
        raise ValueError("Premium strength must be between 0 and 1.")

    seeds = refresh_realized_salaries(
        load_or_bootstrap_seeds(args.league),
        args.year,
        args.league,
    )
    saved_current = seeds[
        seeds.year.eq(args.year) & seeds.league.eq(args.league)
    ].copy()
    if args.reuse_current_seed:
        if saved_current.empty:
            raise ValueError("No saved current-season seed exists to reuse.")
        saved_salary_methods = set(
            saved_current.salary_method_version.dropna().astype(str)
        )
        saved_seed_methods = set(
            saved_current.seed_method_version.dropna().astype(str)
        )
        if saved_salary_methods != {SALARY_METHOD_VERSION}:
            raise ValueError(
                "Saved current-season seed does not match the active salary "
                f"method: {sorted(saved_salary_methods)}."
            )
        if saved_seed_methods != {CURRENT_SEED_METHOD}:
            raise ValueError(
                "Saved current-season seed does not match the active Target "
                f"method: {sorted(saved_seed_methods)}."
            )
        current_surface = saved_current[
            [
                "player",
                "player_key",
                "pos",
                "point_salary",
                "selection_rate",
                "selection_slots",
            ]
        ].copy()
        success_trials = int(saved_current.seed_success_trials.max())
        requested_trials = int(saved_current.seed_trials.max())
        market = {}
    else:
        # FootballSimulation loads the published premium table during
        # construction, even though this seed later requests
        # use_selection_premium=False.  Clear only the staged active slice so
        # a stale player surface cannot contaminate or block the clean seed.
        with sqlite3.connect(SIMULATION_DB) as conn:
            cleared_premium_rows = clear_active_slice(
                conn,
                PREMIUM_TABLE,
                args.year,
                args.league,
            )
        print(
            f"Cleared {cleared_premium_rows} stale {PREMIUM_TABLE} rows for "
            f"{args.year} {args.league} before the premium-free seed."
        )
        current_surface, success_trials, market = run_current_seed(
            args.year,
            args.league,
            args.trials,
            args.workers,
            args.random_seed,
        )
        requested_trials = int(args.trials)
        new_seed = current_seed_rows(
            current_surface,
            args.year,
            args.league,
            requested_trials,
            success_trials,
            args.random_seed,
        )
        seeds = seeds[
            ~(seeds.year.eq(args.year) & seeds.league.eq(args.league))
        ]
        seeds = pd.concat([seeds, new_seed], ignore_index=True, sort=False)

    calibrated, coefficients, metadata = fit_calibrator(
        seeds,
        current_surface,
        args.year,
    )
    premium_rows = build_premium_rows(
        calibrated,
        args.year,
        args.league,
        args.premium_strength,
        requested_trials,
        success_trials,
        metadata,
    )

    with sqlite3.connect(VALIDATIONS_DB) as conn:
        replace_league_slice(
            conn,
            SEED_TABLE,
            seeds.sort_values(["year", "league", "player"]).reset_index(drop=True),
            args.league,
        )
        replace_active_slice(
            conn,
            CALIBRATOR_TABLE,
            coefficients.assign(year=int(args.year), league=args.league),
            args.year,
            args.league,
        )
    with sqlite3.connect(SIMULATION_DB) as conn:
        replace_active_slice(
            conn,
            PREMIUM_TABLE,
            premium_rows,
            args.year,
            args.league,
        )
    if not args.no_app_sync:
        if not APP_SIMULATION_DB.exists():
            raise FileNotFoundError(f"Auction app database not found: {APP_SIMULATION_DB}")
        with sqlite3.connect(APP_SIMULATION_DB) as conn:
            replace_active_slice(
                conn,
                PREMIUM_TABLE,
                premium_rows,
                args.year,
                args.league,
            )

    weighted_premium = float(
        np.average(
            premium_rows.applied_premium,
            weights=np.maximum(premium_rows.selection_rate, 1e-12),
        )
    )
    expected_roster_premium = float(
        np.sum(premium_rows.selection_rate * premium_rows.applied_premium)
    )
    print(
        f"Published {len(premium_rows)} {args.year} {args.league} premiums; "
        f"Target success={success_trials}/{requested_trials}; "
        f"training through {metadata['training_through_year']} "
        f"({metadata['training_rows']} rows)."
    )
    if market:
        print(
            "Keeper/market state: "
            f"{market['keeper_count']} keepers, ${market['keeper_spend']:.0f} spent, "
            f"{market['remaining_slots']} slots, ${market['remaining_budget']:.0f} available."
        )
    print(
        f"Selection-weighted applied premium: ${weighted_premium:.2f} per player; "
        f"expected roster reserve: ${expected_roster_premium:.2f}; "
        f"maximum: ${premium_rows.applied_premium.max():.2f}."
    )
    print(
        premium_rows[
            ["player", "pos", "point_salary", "selection_rate", "applied_premium"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
