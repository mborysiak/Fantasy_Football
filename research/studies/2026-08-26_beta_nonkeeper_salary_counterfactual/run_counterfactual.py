"""Run an isolated 2026 beta salary rebuild with four non-keeper candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


YEAR = 2026
LEAGUE = "beta"
NUM_TEAMS = 12
TEAM_BUDGET = 298
TEAM_ROSTER_SIZE = 13
TARGET_PLAYERS = (
    "Chase Brown",
    "Bhayshul Tuten",
    "Luther Burden III",
    "Colston Loveland",
)
DATABASE_NAMES = (
    "Simulation.sqlite3",
    "Validations.sqlite3",
    "Model_Inputs.sqlite3",
    "Season_Stats_New.sqlite3",
    "Projection_V2_beta.sqlite3",
)

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
RESULTS_DIR = STUDY_DIR / "results"
LIVE_DB_DIR = REPO_ROOT / "Data" / "Databases"
LIVE_KEEPER_FILE = (
    REPO_ROOT
    / "Data"
    / "OtherData"
    / "Keepers"
    / f"keepers_{YEAR}_{LEAGUE}.csv"
)
SALARY_SCRIPT = REPO_ROOT / "Scripts" / "Modeling" / "s4_Salaries_Injuries.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sqlite(path: Path, query: str, params: tuple = ()) -> pd.DataFrame:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return pd.read_sql_query(query, connection, params=params)


def build_counterfactual_keepers() -> tuple[pd.DataFrame, list[str]]:
    keepers = pd.read_csv(LIVE_KEEPER_FILE)
    required = {"player", "keeper_salary"}
    if not required.issubset(keepers.columns):
        raise ValueError(
            f"Keeper file lacks required columns: {sorted(required - set(keepers.columns))}"
        )
    target_lookup = {player.casefold(): player for player in TARGET_PLAYERS}
    keeper_lookup = keepers.player.astype(str).str.strip().str.casefold()
    removed = keepers.loc[keeper_lookup.isin(target_lookup), "player"].tolist()
    counterfactual = keepers.loc[~keeper_lookup.isin(target_lookup)].copy()
    if set(removed) != {"Chase Brown", "Bhayshul Tuten"}:
        raise ValueError(
            "Expected the active beta keeper file to contain exactly Chase Brown "
            f"and Bhayshul Tuten among the four candidates; removed={removed}"
        )
    return counterfactual, removed


def load_salary_slice(database: Path) -> pd.DataFrame:
    return read_sqlite(
        database,
        """
        SELECT player_key,
               player,
               salary,
               std_dev,
               min_score,
               max_score,
               salary_resid_5,
               salary_resid_10,
               salary_resid_25,
               salary_resid_75,
               salary_resid_90,
               salary_resid_95,
               salary_population_source,
               ensemble_uncertainty_feature_source,
               salary_method_version
          FROM Salaries_Pred
         WHERE year=? AND league=?
        """,
        (YEAR, f"{LEAGUE}pred"),
    )


def load_keeper_slice(database: Path) -> pd.DataFrame:
    return read_sqlite(
        database,
        """
        SELECT year, league, player_key, player, keeper_salary
          FROM League_Keepers
         WHERE year=? AND league=?
         ORDER BY player
        """,
        (YEAR, LEAGUE),
    )


def write_markdown_summary(
    comparison: pd.DataFrame,
    metadata: dict,
    top_movers: pd.DataFrame,
) -> None:
    lines = [
        "# 2026 Beta Non-Keeper Salary Counterfactual",
        "",
        (
            "The governed salary model was rebuilt with Chase Brown, Bhayshul "
            "Tuten, Luther Burden III, and Colston Loveland all on the open market."
        ),
        "",
        (
            f"The active keeper pool falls from {metadata['baseline_keeper_count']} "
            f"keepers spending ${metadata['baseline_keeper_spend']:.0f} to "
            f"{metadata['counterfactual_keeper_count']} keepers spending "
            f"${metadata['counterfactual_keeper_spend']:.0f}. The modeled open "
            f"market therefore has {metadata['available_slots']} slots and "
            f"${metadata['available_budget']:.0f}."
        ),
        "",
        "## Candidate salaries",
        "",
        "| Player | Pos | Pred PPG | Current | Non-keeper model | Change | P10-P90 | Min-Max | ESPN source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        source_salary = (
            f"${row.source_salary:.0f}"
            if pd.notna(row.source_salary)
            else "missing"
        )
        lines.append(
            f"| {row.player} | {row.pos} | {row.pred_fp_per_game:.1f} | "
            f"${row.current_salary:.1f} | ${row.counterfactual_salary:.1f} | "
            f"{row.salary_change:+.1f} | ${row.counterfactual_p10:.1f}-${row.counterfactual_p90:.1f} | "
            f"${row.counterfactual_min:.1f}-${row.counterfactual_max:.1f} | {source_salary} |"
        )
    lines.extend(
        [
            "",
            "`P10-P90` is the modeled salary center plus the stored 10th/90th "
            "residual quantiles, floored at $1. `Min-Max` is the app's legacy "
            "uncertainty range.",
            "",
            "## Largest whole-market center moves",
            "",
            "| Player | Current | Counterfactual | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_movers.itertuples(index=False):
        lines.append(
            f"| {row.player} | ${row.current_salary:.1f} | "
            f"${row.counterfactual_salary:.1f} | {row.salary_change:+.1f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Projection/salary key parity: `{metadata['key_parity']}`.",
            (
                f"- Top-{metadata['available_slots']} non-keeper salary total: "
                f"${metadata['top_nonkeeper_salary_total']:.6f} versus "
                f"${metadata['available_budget']:.6f} available."
            ),
            f"- Live input hashes unchanged after the run: `{metadata['live_inputs_unchanged']}`.",
            f"- Salary method: `{metadata['salary_method_version']}`.",
            "- Production keeper CSV and all production/app databases were left unchanged.",
            "",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-stage",
        type=Path,
        help=(
            "Extract and validate a previously completed isolated stage instead "
            "of rerunning the salary fit. Intended only for recovery after a "
            "post-build wrapper failure."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counterfactual_keepers, removed = build_counterfactual_keepers()
    counterfactual_keeper_path = RESULTS_DIR / "counterfactual_keepers.csv"
    counterfactual_keepers.to_csv(counterfactual_keeper_path, index=False)

    guarded_inputs = [LIVE_KEEPER_FILE, *(LIVE_DB_DIR / name for name in DATABASE_NAMES)]
    pre_hashes = {str(path): sha256_file(path) for path in guarded_inputs}

    live_simulation = LIVE_DB_DIR / "Simulation.sqlite3"
    current_salaries = load_salary_slice(live_simulation)
    current_keepers = load_keeper_slice(live_simulation)

    reusable_stage = args.reuse_stage
    with tempfile.TemporaryDirectory(prefix="beta-nonkeeper-salary-") as stage_name:
        stage_dir = Path(stage_name)
        database_source_dir = (
            reusable_stage.expanduser().resolve()
            if reusable_stage is not None
            else LIVE_DB_DIR
        )
        for name in DATABASE_NAMES:
            source = database_source_dir / name
            if reusable_stage is not None and not source.is_file():
                source = LIVE_DB_DIR / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, stage_dir / name)

        environment = os.environ.copy()
        environment.update(
            {
                "FF_CURRENT_SEASON": str(YEAR),
                "FF_MODEL_DATABASE_DIR": str(stage_dir),
                "FF_AUCTION_LEAGUE": LEAGUE,
                "FF_V2_BETA_DATABASE": str(stage_dir / "Projection_V2_beta.sqlite3"),
                "FF_V2_AUCTION_DATABASE": str(stage_dir / "Projection_V2_beta.sqlite3"),
                "FF_KEEPERS_FILE": str(counterfactual_keeper_path),
                "MPLBACKEND": "Agg",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (
                str(REPO_ROOT),
                environment.get("PYTHONPATH", ""),
            )
            if item
        )
        if reusable_stage is None:
            completed = subprocess.run(
                [sys.executable, str(SALARY_SCRIPT)],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Salary rebuild failed with exit code {completed.returncode}"
                )

        staged_simulation = stage_dir / "Simulation.sqlite3"
        counterfactual_salaries = load_salary_slice(staged_simulation)
        counterfactual_keeper_slice = load_keeper_slice(staged_simulation)
        production_keys = read_sqlite(
            staged_simulation,
            """
            SELECT player_key, player, pos, pred_fp_per_game
              FROM Final_Predictions_Resid
             WHERE year=? AND version=? AND dataset='final_ensemble'
            """,
            (YEAR, LEAGUE),
        )
        source_salaries = read_sqlite(
            staged_simulation,
            """
            SELECT player, salary AS source_salary
              FROM Salaries
             WHERE year=? AND league=?
            """,
            (YEAR, LEAGUE),
        )

    post_hashes = {str(path): sha256_file(path) for path in guarded_inputs}
    live_inputs_unchanged = pre_hashes == post_hashes
    if not live_inputs_unchanged:
        changed = [path for path in pre_hashes if pre_hashes[path] != post_hashes[path]]
        raise RuntimeError(f"Live inputs changed during the isolated run: {changed}")

    expected_keeper_count = len(current_keepers) - len(removed)
    expected_keeper_spend = float(
        current_keepers.keeper_salary.sum()
        - current_keepers.loc[current_keepers.player.isin(removed), "keeper_salary"].sum()
    )
    if len(counterfactual_keeper_slice) != expected_keeper_count:
        raise ValueError(
            f"Expected {expected_keeper_count} counterfactual keepers; "
            f"found {len(counterfactual_keeper_slice)}"
        )
    if not np.isclose(counterfactual_keeper_slice.keeper_salary.sum(), expected_keeper_spend):
        raise ValueError("Counterfactual keeper spend does not match the expected spend")
    if set(counterfactual_keeper_slice.player).intersection(TARGET_PLAYERS):
        raise ValueError("At least one candidate remained in the keeper table")

    projection_key_set = set(production_keys.player_key)
    salary_key_set = set(counterfactual_salaries.player_key)
    key_parity = projection_key_set == salary_key_set
    if not key_parity or counterfactual_salaries.player_key.duplicated().any():
        raise ValueError("Counterfactual salary output lacks exact unique key parity")

    keeper_key_set = set(counterfactual_keeper_slice.player_key)
    nonkeeper_salaries = counterfactual_salaries.loc[
        ~counterfactual_salaries.player_key.isin(keeper_key_set)
    ]
    available_slots = NUM_TEAMS * TEAM_ROSTER_SIZE - len(counterfactual_keeper_slice)
    available_budget = NUM_TEAMS * TEAM_BUDGET - float(
        counterfactual_keeper_slice.keeper_salary.sum()
    )
    top_nonkeeper_total = float(
        nonkeeper_salaries.nlargest(available_slots, "salary").salary.sum()
    )
    if not np.isclose(top_nonkeeper_total, available_budget, atol=1e-6):
        raise ValueError(
            "Counterfactual top non-keeper market misses the available budget: "
            f"{top_nonkeeper_total} versus {available_budget}"
        )

    salary_columns = [
        "player_key",
        "player",
        "salary",
        "std_dev",
        "min_score",
        "max_score",
        "salary_resid_10",
        "salary_resid_90",
        "salary_method_version",
        "salary_population_source",
        "ensemble_uncertainty_feature_source",
    ]
    comparison = (
        current_salaries[salary_columns]
        .rename(
            columns={
                column: f"current_{column}"
                for column in salary_columns
                if column != "player_key"
            }
        )
        .merge(
            counterfactual_salaries[salary_columns].rename(
                columns={
                    column: f"counterfactual_{column}"
                    for column in salary_columns
                    if column != "player_key"
                }
            ),
            on="player_key",
            how="inner",
            validate="one_to_one",
        )
        .merge(production_keys, on="player_key", how="left", validate="one_to_one")
    )
    comparison["salary_change"] = (
        comparison.counterfactual_salary - comparison.current_salary
    )
    comparison["counterfactual_p10"] = np.maximum(
        1.0,
        comparison.counterfactual_salary + comparison.counterfactual_salary_resid_10,
    )
    comparison["counterfactual_p90"] = np.maximum(
        1.0,
        comparison.counterfactual_salary + comparison.counterfactual_salary_resid_90,
    )
    comparison["counterfactual_min"] = comparison.counterfactual_min_score
    comparison["counterfactual_max"] = comparison.counterfactual_max_score
    comparison = comparison.merge(
        source_salaries,
        left_on="player",
        right_on="player",
        how="left",
        validate="one_to_one",
    )
    comparison["current_is_keeper"] = comparison.player.isin(current_keepers.player)
    comparison["counterfactual_is_keeper"] = comparison.player.isin(
        counterfactual_keeper_slice.player
    )

    target_comparison = comparison.loc[
        comparison.player.isin(TARGET_PLAYERS)
    ].copy()
    if set(target_comparison.player) != set(TARGET_PLAYERS):
        raise ValueError("Not every target player appeared in the rebuilt salary surface")
    if target_comparison.counterfactual_is_keeper.any():
        raise ValueError("A target player still has a counterfactual keeper override")
    uncertainty_columns = [
        "counterfactual_std_dev",
        "counterfactual_salary_resid_10",
        "counterfactual_salary_resid_90",
    ]
    if (target_comparison[uncertainty_columns].abs().sum(axis=1) <= 0.1).any():
        raise ValueError("A candidate lacks model-derived counterfactual uncertainty")

    target_order = {player: index for index, player in enumerate(TARGET_PLAYERS)}
    target_comparison["target_order"] = target_comparison.player.map(target_order)
    target_comparison = target_comparison.sort_values("target_order").drop(
        columns="target_order"
    )
    all_deltas = comparison.sort_values("salary_change", ascending=False)
    top_movers = pd.concat(
        [
            all_deltas.head(8),
            all_deltas.tail(8),
        ],
        ignore_index=True,
    ).drop_duplicates("player_key")

    target_comparison.to_csv(RESULTS_DIR / "candidate_salary_comparison.csv", index=False)
    all_deltas.to_csv(RESULTS_DIR / "all_salary_deltas.csv", index=False)
    counterfactual_keeper_slice.to_csv(
        RESULTS_DIR / "published_counterfactual_keeper_slice.csv", index=False
    )

    method_versions = counterfactual_salaries.salary_method_version.dropna().unique()
    if len(method_versions) != 1:
        raise ValueError(f"Expected one salary method version; found {method_versions}")
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "year": YEAR,
        "league": LEAGUE,
        "target_players": list(TARGET_PLAYERS),
        "active_target_keepers_removed": removed,
        "already_nonkeepers": sorted(set(TARGET_PLAYERS) - set(removed)),
        "baseline_keeper_count": int(len(current_keepers)),
        "baseline_keeper_spend": float(current_keepers.keeper_salary.sum()),
        "counterfactual_keeper_count": int(len(counterfactual_keeper_slice)),
        "counterfactual_keeper_spend": float(
            counterfactual_keeper_slice.keeper_salary.sum()
        ),
        "available_slots": int(available_slots),
        "available_budget": float(available_budget),
        "top_nonkeeper_salary_total": top_nonkeeper_total,
        "projection_population": int(len(production_keys)),
        "salary_population": int(len(counterfactual_salaries)),
        "key_parity": bool(key_parity),
        "live_inputs_unchanged": bool(live_inputs_unchanged),
        "salary_method_version": str(method_versions[0]),
        "live_input_sha256": pre_hashes,
        "counterfactual_keeper_sha256": sha256_file(counterfactual_keeper_path),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown_summary(target_comparison, metadata, top_movers)

    print("COUNTERFACTUAL_RESULT=" + json.dumps(metadata, sort_keys=True))
    print(
        target_comparison[
            [
                "player",
                "pos",
                "pred_fp_per_game",
                "current_salary",
                "counterfactual_salary",
                "salary_change",
                "counterfactual_p10",
                "counterfactual_p90",
                "source_salary",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
