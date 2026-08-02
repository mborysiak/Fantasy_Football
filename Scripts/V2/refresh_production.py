"""Build and validate every downstream draft artifact after raw-source ingest.

The manual boundary remains ``Scripts/Data_Generation/1_Update_Projections.py``.
Once that notebook has populated ``Season_Stats_New.sqlite3``, this command
rebuilds the canonical inputs, all registered locked V2 scoring objectives, weekly
templates, auction salaries/reserve inputs, and staged app databases.

By default nothing live is replaced.  Pass ``--promote`` only after the staged
release and both app smoke tests complete successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Scripts.V2.contracts import scoring_hash
from Scripts.V2.production_cycle import (
    APPROVED_PRODUCTION_CYCLES,
    DEFAULT_PRODUCTION_YEAR,
    PRODUCTION_LEAGUES,
    ProductionCycle,
    get_production_cycle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = REPO_ROOT / "Data" / "Databases"
PRODUCTION_BACKUP_ROOT = REPO_ROOT / "Data" / "Production_Refresh_Backups"
SIBLING_ROOT = REPO_ROOT.parent
AUCTION_DB = (
    SIBLING_ROOT
    / "Fantasy_Football_App"
    / "app"
    / "Simulation.sqlite3"
)
SNAKE_DB = (
    SIBLING_ROOT
    / "Fantasy_Football_Snake"
    / "app"
    / "Simulation.sqlite3"
)

DEFAULT_DATASET = "final_ensemble"
DEFAULT_MAX_WORKERS = 6
DEFAULT_SELECTION_TRIALS = 1000
DEFAULT_SELECTION_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_APP_TIMEOUT = 90
MANIFEST_SCHEMA_VERSION = 3
NATIVE_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
WINDOWS_ACCESS_VIOLATION_RETURN_CODES = frozenset(
    {
        3221225477,
        -1073741819,
    }
)
MAX_NATIVE_CRASH_ATTEMPTS = 2

PIPELINE_CODE_ROOTS = {
    "main_scripts": REPO_ROOT / "Scripts",
    "auction_app": SIBLING_ROOT / "Fantasy_Football_App" / "app",
    "snake_app": SIBLING_ROOT / "Fantasy_Football_Snake" / "app",
    "ff_package": SIBLING_ROOT / "ff" / "ff",
    "scikit_model_package": SIBLING_ROOT / "Scikit_Model" / "skmodel",
}
for _cycle_year, _cycle in APPROVED_PRODUCTION_CYCLES.items():
    PIPELINE_CODE_ROOTS.update(
        {
            f"cycle_{_cycle_year}_locked_study": (
                _cycle.locked_runner.parent
            ),
            f"cycle_{_cycle_year}_next_study": (
                _cycle.next_year_runner.parent
            ),
        }
    )
PIPELINE_CODE_SUFFIXES = frozenset({".py"})
EXTERNAL_SQLITE_INPUTS = {
    "weekly_history": (
        SIBLING_ROOT
        / "Daily_Fantasy_Data"
        / "Databases"
        / "FastR_Beta.sqlite3"
    ),
}
STATIC_EXTERNAL_FILE_INPUTS = {
    "selection_bootstrap_rosters": (
        REPO_ROOT
        / "research"
        / "studies"
        / "2026-07-16_optimizer_selection_surcharge"
        / "results"
        / "roster_trials.csv"
    ),
    "selection_bootstrap_candidates": (
        REPO_ROOT
        / "research"
        / "studies"
        / "2026-07-16_salary_v5_replay"
        / "results"
        / "selected_residuals_v5"
        / "candidate_diagnostic.csv"
    ),
}


def external_file_inputs(year: int) -> dict[str, Path]:
    """Resolve every mutable non-SQLite input for the selected cycle."""

    return {
        "current_auction_salaries": (
            REPO_ROOT
            / "Data"
            / "OtherData"
            / "Salaries"
            / f"salaries_{int(year)}_beta.csv"
        ),
        "current_beta_keepers": (
            REPO_ROOT
            / "Data"
            / "OtherData"
            / "Keepers"
            / f"keepers_{int(year)}_beta.csv"
        ),
        **STATIC_EXTERNAL_FILE_INPUTS,
    }

DATABASE_FILES = {
    "source": "Season_Stats_New.sqlite3",
    "model_inputs": "Model_Inputs.sqlite3",
    "model_inputs_next": "Model_Inputs_next.sqlite3",
    "v2_dk": "Projection_V2.sqlite3",
    "v2_nffc": "Projection_V2_nffc.sqlite3",
    "v2_beta": "Projection_V2_beta.sqlite3",
    "simulation": "Simulation.sqlite3",
    "validations": "Validations.sqlite3",
}
PROMOTED_DATABASES = (
    "model_inputs",
    "model_inputs_next",
    "v2_dk",
    "v2_nffc",
    "v2_beta",
    "simulation",
    "validations",
)
BOOTSTRAPPABLE_DATABASES = frozenset({"v2_nffc"})
GOVERNED_HANDOFF_TABLES = (
    "Avg_ADPs",
    "Avg_ADPs_Publication_Audit",
    "Avg_ADPs_Publication_Receipt",
    "Final_Predictions_Resid",
    "V2_Production_Projection_Handoff",
    "V2_Production_Projection_Audit",
    "V2_Production_Eligibility_Audit",
    "V2_Projection_Legacy_Backup",
)
WEEKLY_TABLES = (
    "Best_Ball_Weekly_Templates",
    "Best_Ball_Weekly_Template_Pools",
    "Best_Ball_Weekly_Pool_Summary",
    "Best_Ball_Weekly_Player_Map",
    "Best_Ball_Weekly_Template_Audit",
    "Best_Ball_Weekly_Player_Pool_Audit",
    "Best_Ball_Weekly_Bucket_Audit",
    "Best_Ball_ADP_Audit",
)
SALARY_TABLES = (
    "Salaries",
    "Salaries_Pred",
    "League_Keepers",
    "Salary_Selection_Premium",
)
GENERATED_AUCTION_TABLES = (
    *GOVERNED_HANDOFF_TABLES,
    *WEEKLY_TABLES,
    *SALARY_TABLES,
)

PIPELINE_STEPS = (
    "snapshot",
    "model_inputs",
    "v2_dk",
    "v2_nffc",
    "v2_beta",
    "locked_dk",
    "locked_nffc",
    "locked_beta",
    "next_dk",
    "next_nffc",
    "next_beta",
    "keepers",
    "handoff",
    "weekly_dk",
    "weekly_nffc",
    "weekly_beta",
    "template_audit_dk",
    "template_audit_nffc",
    "template_audit_beta",
    "salary",
    "selection_premium",
    "validate",
    "prepare_apps",
    "app_smoke",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file_state(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required refresh input not found: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def pipeline_code_fingerprint() -> dict[str, Any]:
    """Hash every local code/config file that can affect this release."""

    files: dict[str, dict[str, Any]] = {}
    for label, root in sorted(PIPELINE_CODE_ROOTS.items()):
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"Required refresh code root not found: {root}"
            )
        for path in sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in PIPELINE_CODE_SUFFIXES
        ):
            key = f"{label}/{path.relative_to(root).as_posix()}"
            files[key] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not files:
        raise RuntimeError("No pipeline code files were found to fingerprint")
    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def assert_pipeline_code_unchanged(
    manifest: Mapping[str, Any],
) -> None:
    expected = manifest.get("code_fingerprint")
    if not expected:
        raise ValueError("Refresh manifest lacks a pipeline code fingerprint")
    actual = pipeline_code_fingerprint()
    if (
        str(actual["sha256"]) != str(expected.get("sha256"))
        or int(actual["file_count"]) != int(expected.get("file_count", -1))
    ):
        expected_files = expected.get("files", {})
        actual_files = actual.get("files", {})
        changed = sorted(
            key
            for key in set(expected_files) | set(actual_files)
            if expected_files.get(key) != actual_files.get(key)
        )
        raise RuntimeError(
            "Pipeline/app code changed after this refresh started; start a "
            f"new refresh. Changed files: {changed[:20]}"
        )


def production_cycle_receipt(year: int) -> dict[str, Any]:
    cycle = get_production_cycle(year)
    return {
        **cycle.receipt(),
        "contract_sha256": cycle.contract_sha256(),
    }


def assert_production_cycle_unchanged(
    manifest: Mapping[str, Any],
) -> None:
    expected = manifest.get("production_cycle")
    if not expected:
        raise ValueError("Refresh manifest lacks a production-cycle receipt")
    actual = production_cycle_receipt(int(manifest["options"]["year"]))
    if actual != expected:
        raise RuntimeError(
            "The approved production-cycle contract changed after this "
            "refresh started; start a new refresh."
        )


def _sqlite_sidecars(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in (
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
            Path(f"{path}-journal"),
        )
        if candidate.exists()
    ]


def validate_sqlite(path: Path, *, foreign_keys: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    sidecars = _sqlite_sidecars(path)
    if sidecars:
        raise RuntimeError(
            "Close any process writing this database before refresh; active "
            f"SQLite sidecars were found for {path}: "
            + ", ".join(str(item.name) for item in sidecars)
        )
    with closing(
        sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    ) as conn:
        integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise ValueError(f"SQLite integrity failed for {path}: {integrity}")
        fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys and fk_rows:
            raise ValueError(
                f"SQLite foreign-key check failed for {path}: {fk_rows[:10]}"
            )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "integrity": "ok",
        "foreign_keys": "ok" if not fk_rows else f"{len(fk_rows)} violations",
    }


def database_state(path: Path) -> dict[str, Any]:
    return validate_sqlite(path)


def sqlite_backup(source: Path, destination: Path) -> dict[str, Any]:
    """Create a consistent SQLite backup without mutating the source."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    source_before = database_state(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with closing(sqlite3.connect(source)) as source_conn:
            with closing(sqlite3.connect(temporary)) as destination_conn:
                source_conn.backup(destination_conn)
        copied = validate_sqlite(temporary)
        source_after = database_state(source)
        if (
            source_before["size_bytes"] != source_after["size_bytes"]
            or source_before["sha256"] != source_after["sha256"]
        ):
            raise RuntimeError(
                f"SQLite source changed while it was staged: {source}"
            )
        os.replace(temporary, destination)
        copied["path"] = str(destination)
        return {
            "source": source_before,
            "staged": copied,
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def _require_tables(path: Path, tables: Iterable[str]) -> None:
    with closing(sqlite3.connect(path)) as connection:
        missing = [
            table for table in tables if not _table_exists(connection, table)
        ]
    if missing:
        raise ValueError(f"{path} is missing required tables: {missing}")


def database_tables(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as connection:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]


def stable_table_digest(path: Path, table: str) -> str:
    """Digest a SQLite table as an unordered row multiset plus its schema."""

    digest = hashlib.sha256()
    with sqlite3.connect(path) as connection:
        schema = connection.execute(
            "SELECT COALESCE(sql, '') FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if schema is None:
            raise ValueError(f"{path} is missing table {table}")
        digest.update(str(schema[0]).encode("utf-8"))
        columns = [
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        digest.update("\x1f".join(columns).encode("utf-8"))
        indexes = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='index' AND tbl_name=? AND sql IS NOT NULL "
            "ORDER BY name",
            (table,),
        ).fetchall()
        for name, statement in indexes:
            digest.update(str(name).encode("utf-8"))
            digest.update(str(statement).encode("utf-8"))
        rows = sorted(
            repr(tuple(row)).encode("utf-8")
            for row in connection.execute(f'SELECT * FROM "{table}"')
        )
    for row in rows:
        digest.update(row)
        digest.update(b"\n")
    return digest.hexdigest()


def table_digests(path: Path, tables: Iterable[str]) -> dict[str, str]:
    return {
        table: stable_table_digest(path, table)
        for table in sorted(set(tables))
    }


def _manifest_path(stage_dir: Path) -> Path:
    return Path(stage_dir).resolve() / "refresh_manifest.json"


def _path_map(stage_dir: Path) -> dict[str, dict[str, str]]:
    stage_dir = Path(stage_dir).resolve()
    staged_db_dir = stage_dir / "databases"
    live = {
        key: str((DATABASE_DIR / filename).resolve())
        for key, filename in DATABASE_FILES.items()
    }
    staged = {
        key: str((staged_db_dir / filename).resolve())
        for key, filename in DATABASE_FILES.items()
    }
    live.update(
        {
            "auction_app": str(AUCTION_DB.resolve()),
            "snake_app": str(SNAKE_DB.resolve()),
        }
    )
    return {
        "live": live,
        "staged": staged,
        "app_bases": {
            "auction": str(
                (stage_dir / "app_bases" / "Auction_Simulation.sqlite3").resolve()
            ),
            "snake": str(
                (stage_dir / "app_bases" / "Snake_Simulation.sqlite3").resolve()
            ),
        },
        "app_artifacts": {
            "auction": str(
                (
                    stage_dir
                    / "app_artifacts"
                    / "Auction_Simulation.sqlite3"
                ).resolve()
            ),
            "snake": str(
                (
                    stage_dir
                    / "app_artifacts"
                    / "Snake_Simulation.sqlite3"
                ).resolve()
            ),
        },
    }


def _new_manifest(stage_dir: Path, options: Mapping[str, Any]) -> dict[str, Any]:
    cycle_receipt = production_cycle_receipt(int(options["year"]))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        ),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "stage_dir": str(Path(stage_dir).resolve()),
        "options": dict(options),
        "production_cycle": cycle_receipt,
        "code_fingerprint": pipeline_code_fingerprint(),
        "paths": _path_map(stage_dir),
        "baseline": {},
        "external_inputs": {},
        "steps": {
            step: {"status": "pending"}
            for step in PIPELINE_STEPS
        },
        "promotion": {"status": "not_requested"},
    }


def _save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(
        _manifest_path(Path(manifest["stage_dir"])),
        manifest,
    )


def _load_manifest(stage_dir: Path) -> dict[str, Any]:
    path = _manifest_path(stage_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Refresh manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported refresh manifest schema: "
            f"{manifest.get('schema_version')}"
        )
    if Path(manifest["repo_root"]).resolve() != REPO_ROOT:
        raise ValueError("Refresh manifest belongs to a different repository")
    assert_production_cycle_unchanged(manifest)
    return manifest


def _resolved_paths(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Path]]:
    return {
        section: {
            key: Path(value).resolve()
            for key, value in values.items()
        }
        for section, values in manifest["paths"].items()
    }


def _subprocess_environment(
    staged_paths: Mapping[str, Path],
    *,
    year: int,
) -> dict[str, str]:
    environment = _native_thread_limited_environment()
    for pipeline_flag in (
        "FF_CANONICAL_INPUTS_ONLY",
        "SALARY_KEEPERS_ONLY",
        "SALARY_VALIDATION_DATASETS_ONLY",
    ):
        environment.pop(pipeline_flag, None)
    environment.update(
        {
            "FF_CURRENT_SEASON": str(int(year)),
            "FF_MODEL_DATABASE_DIR": str(staged_paths["simulation"].parent),
            "FF_V2_BETA_DATABASE": str(staged_paths["v2_beta"]),
            "FF_KEEPERS_FILE": str(
                external_file_inputs(year)["current_beta_keepers"]
            ),
            "MPLBACKEND": "Agg",
        }
    )
    python_path = [
        str(REPO_ROOT),
        environment.get("PYTHONPATH", ""),
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in python_path if item
    )
    return environment


def _native_thread_limited_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a complete subprocess environment with deterministic native caps."""

    limited = (
        os.environ.copy()
        if environment is None
        else dict(environment)
    )
    limited.update(
        {
            variable: "1"
            for variable in NATIVE_THREAD_ENVIRONMENT_VARIABLES
        }
    )
    return limited


def run_logged_command(
    command: Sequence[str],
    *,
    step: str,
    stage_dir: Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    log_path = Path(stage_dir) / "logs" / f"{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline([str(item) for item in command])
    print(f"\n[{step}] {printable}", flush=True)
    command_started = utc_now()
    attempt_receipts: list[dict[str, Any]] = []
    process_environment = _native_thread_limited_environment(environment)
    with log_path.open("a", encoding="utf-8", newline="") as log:
        for attempt in range(1, MAX_NATIVE_CRASH_ATTEMPTS + 1):
            started = utc_now()
            attempt_label = (
                f"attempt {attempt}/{MAX_NATIVE_CRASH_ATTEMPTS}"
            )
            log.write(
                f"\n[{started}] {attempt_label}: {printable}\n"
            )
            log.flush()
            if attempt > 1:
                print(f"[{step}] {attempt_label}", flush=True)
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=REPO_ROOT,
                env=process_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            console_available = True
            try:
                for line in process.stdout:
                    if console_available:
                        try:
                            print(line, end="", flush=True)
                        except OSError:
                            # The build log remains authoritative if a detached
                            # terminal or redirected console closes mid-run.
                            console_available = False
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover - OS timing
                    process.kill()
                    process.wait()
                raise

            completed = utc_now()
            native_crash = (
                return_code in WINDOWS_ACCESS_VIOLATION_RETURN_CODES
            )
            will_retry = (
                native_crash
                and attempt < MAX_NATIVE_CRASH_ATTEMPTS
            )
            if return_code == 0:
                outcome = "completed"
            elif will_retry:
                outcome = "retryable_native_failure"
            elif native_crash:
                outcome = "native_failure_exhausted"
            else:
                outcome = "failed"
            receipt = {
                "attempt": attempt,
                "started_at_utc": started,
                "completed_at_utc": completed,
                "return_code": return_code,
                "outcome": outcome,
                "will_retry": will_retry,
            }
            if native_crash:
                receipt["failure_class"] = "windows_access_violation"
            attempt_receipts.append(receipt)
            log.write(
                f"[{completed}] {attempt_label} exited {return_code}; "
                f"outcome={outcome}\n"
            )
            log.flush()

            if return_code == 0:
                return {
                    "command": [str(item) for item in command],
                    "log": str(log_path.resolve()),
                    "started_at_utc": command_started,
                    "completed_at_utc": completed,
                    "attempts": attempt_receipts,
                }
            if not will_retry:
                error = subprocess.CalledProcessError(
                    return_code,
                    list(command),
                )
                error.attempt_receipts = attempt_receipts
                error.log_path = str(log_path.resolve())
                raise error
            retry_message = (
                f"[{step}] Windows access violation on {attempt_label}; "
                "retrying"
            )
            print(retry_message, flush=True)
            log.write(f"{retry_message}\n")
            log.flush()

    raise AssertionError("Subprocess retry loop exited unexpectedly")


def _python(manifest: Mapping[str, Any]) -> str:
    return str(manifest["options"]["python"])


def _app_python(manifest: Mapping[str, Any]) -> str:
    return str(manifest["options"]["app_python"])


def step_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    for directory in (
        Path(manifest["stage_dir"]) / "databases",
        Path(manifest["stage_dir"]) / "results",
        Path(manifest["stage_dir"]) / "logs",
        Path(manifest["stage_dir"]) / "app_bases",
        Path(manifest["stage_dir"]) / "app_artifacts",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    receipts: dict[str, Any] = {}
    for key in DATABASE_FILES:
        live_path = paths["live"][key]
        staged_path = paths["staged"][key]
        if not live_path.is_file() and key in BOOTSTRAPPABLE_DATABASES:
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(staged_path)):
                pass
            receipt = {
                "source": {
                    "path": str(live_path),
                    "exists": False,
                },
                "staged": database_state(staged_path),
                "bootstrap": True,
            }
        else:
            receipt = sqlite_backup(live_path, staged_path)
        receipts[key] = receipt
        manifest["baseline"][key] = receipt["source"]
    for app in ("auction", "snake"):
        live_key = f"{app}_app"
        receipt = sqlite_backup(
            paths["live"][live_key],
            paths["app_bases"][app],
        )
        receipts[live_key] = receipt
        manifest["baseline"][live_key] = receipt["source"]
    external_inputs = {
        "sqlite": {
            key: database_state(path)
            for key, path in EXTERNAL_SQLITE_INPUTS.items()
        },
        "files": {
            key: regular_file_state(path)
            for key, path in external_file_inputs(
                int(manifest["options"]["year"])
            ).items()
        },
    }
    manifest["external_inputs"] = external_inputs
    source_market_counts = _validate_source_markets(
        paths["staged"]["source"],
        int(manifest["options"]["year"]),
    )
    return {
        "database_count": len(receipts),
        "receipts": receipts,
        "external_inputs": external_inputs,
        "source_market_counts": source_market_counts,
    }


def step_model_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    environment = _subprocess_environment(
        paths["staged"],
        year=int(manifest["options"]["year"]),
    )
    environment["FF_CANONICAL_INPUTS_ONLY"] = "1"
    result = run_logged_command(
        [
            _python(manifest),
            str(REPO_ROOT / "Scripts" / "Data_Generation" / "4_Data_Compile.py"),
        ],
        step="model_inputs",
        stage_dir=Path(manifest["stage_dir"]),
        environment=environment,
    )
    required = [
        f"{position}_{manifest['options']['year']}_ProjOnly"
        for position in ("QB", "RB", "WR", "TE")
    ]
    _require_tables(paths["staged"]["model_inputs"], required)
    _require_tables(paths["staged"]["model_inputs_next"], required)
    return result


def step_v2(manifest: dict[str, Any], league: str) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    step = f"v2_{league}"
    year = int(manifest["options"]["year"])
    return run_logged_command(
        [
            _python(manifest),
            "-m",
            "Scripts.V2.build_milestone_3",
            "--source-db",
            str(paths["staged"]["source"]),
            "--output-db",
            str(paths["staged"][step]),
            "--league",
            league,
            "--completed-through",
            str(year - 1),
            "--projection-through",
            str(year),
            "--max-workers",
            str(manifest["options"]["max_workers"]),
        ],
        step=step,
        stage_dir=Path(manifest["stage_dir"]),
        environment=_subprocess_environment(
            paths["staged"],
            year=int(manifest["options"]["year"]),
        ),
    )


def step_locked(manifest: dict[str, Any], league: str) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    step = f"locked_{league}"
    cycle = get_production_cycle(int(manifest["options"]["year"]))
    return run_logged_command(
        [
            _python(manifest),
            str(cycle.locked_runner),
            "--league",
            league,
            "--output-db",
            str(paths["staged"][f"v2_{league}"]),
            "--results-dir",
            str(Path(manifest["stage_dir"]) / "results" / step),
        ],
        step=step,
        stage_dir=Path(manifest["stage_dir"]),
        environment=_subprocess_environment(
            paths["staged"],
            year=int(manifest["options"]["year"]),
        ),
    )


def step_next(manifest: dict[str, Any], league: str) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    step = f"next_{league}"
    cycle = get_production_cycle(int(manifest["options"]["year"]))
    return run_logged_command(
        [
            _python(manifest),
            str(cycle.next_year_runner),
            "--league",
            league,
            "--output-db",
            str(paths["staged"][f"v2_{league}"]),
            "--results-dir",
            str(Path(manifest["stage_dir"]) / "results" / step),
        ],
        step=step,
        stage_dir=Path(manifest["stage_dir"]),
        environment=_subprocess_environment(
            paths["staged"],
            year=int(manifest["options"]["year"]),
        ),
    )


def step_keepers(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    environment = _subprocess_environment(
        paths["staged"],
        year=int(manifest["options"]["year"]),
    )
    environment["SALARY_KEEPERS_ONLY"] = "1"
    return run_logged_command(
        [
            _python(manifest),
            str(REPO_ROOT / "Scripts" / "Modeling" / "s4_Salaries_Injuries.py"),
        ],
        step="keepers",
        stage_dir=Path(manifest["stage_dir"]),
        environment=environment,
    )


def _handoff_command(
    manifest: Mapping[str, Any],
    staged: Mapping[str, Path],
) -> list[str]:
    return [
        _python(manifest),
        "-m",
        "Scripts.V2.production_handoff",
        "--simulation-db",
        str(staged["simulation"]),
        "--model-inputs-db",
        str(staged["model_inputs"]),
        "--market-source-db",
        str(staged["source"]),
        "--dk-v2-db",
        str(staged["v2_dk"]),
        "--nffc-v2-db",
        str(staged["v2_nffc"]),
        "--beta-v2-db",
        str(staged["v2_beta"]),
        "--year",
        str(manifest["options"]["year"]),
        "--dataset",
        str(manifest["options"]["dataset"]),
    ]


def step_handoff(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    environment = _subprocess_environment(
        paths["staged"],
        year=int(manifest["options"]["year"]),
    )
    first = run_logged_command(
        _handoff_command(manifest, paths["staged"]),
        step="handoff",
        stage_dir=Path(manifest["stage_dir"]),
        environment=environment,
    )
    first_digests = table_digests(
        paths["staged"]["simulation"],
        GOVERNED_HANDOFF_TABLES,
    )
    second = run_logged_command(
        _handoff_command(manifest, paths["staged"]),
        step="handoff_idempotence",
        stage_dir=Path(manifest["stage_dir"]),
        environment=environment,
    )
    second_digests = table_digests(
        paths["staged"]["simulation"],
        GOVERNED_HANDOFF_TABLES,
    )
    if first_digests != second_digests:
        changed = sorted(
            table
            for table in GOVERNED_HANDOFF_TABLES
            if first_digests.get(table) != second_digests.get(table)
        )
        raise ValueError(
            f"Production handoff is not idempotent; changed tables: {changed}"
        )
    return {
        "first_run": first,
        "second_run": second,
        "governed_table_digests": second_digests,
        "idempotent": True,
    }


def step_weekly(manifest: dict[str, Any], league: str) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    step = f"weekly_{league}"
    return run_logged_command(
        [
            _python(manifest),
            "-m",
            "Scripts.Modeling.s4_Best_Ball_Weekly",
            "--league",
            league,
            "--simulation-db",
            str(paths["staged"]["simulation"]),
            "--v2-db",
            str(paths["staged"][f"v2_{league}"]),
            "--no-app-sync",
        ],
        step=step,
        stage_dir=Path(manifest["stage_dir"]),
        environment=_subprocess_environment(
            paths["staged"],
            year=int(manifest["options"]["year"]),
        ),
    )


def step_template_audit(
    manifest: dict[str, Any],
    league: str,
) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    step = f"template_audit_{league}"
    cycle = get_production_cycle(int(manifest["options"]["year"]))
    return run_logged_command(
        [
            _python(manifest),
            str(cycle.template_audit_runner),
            "--league",
            league,
            "--output-db",
            str(paths["staged"][f"v2_{league}"]),
            "--simulation-db",
            str(paths["staged"]["simulation"]),
            "--results-dir",
            str(Path(manifest["stage_dir"]) / "results" / step),
        ],
        step=step,
        stage_dir=Path(manifest["stage_dir"]),
        environment=_subprocess_environment(
            paths["staged"],
            year=int(manifest["options"]["year"]),
        ),
    )


def step_salary(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    environment = _subprocess_environment(
        paths["staged"],
        year=int(manifest["options"]["year"]),
    )
    environment.pop("SALARY_KEEPERS_ONLY", None)
    return run_logged_command(
        [
            _python(manifest),
            str(REPO_ROOT / "Scripts" / "Modeling" / "s4_Salaries_Injuries.py"),
        ],
        step="salary",
        stage_dir=Path(manifest["stage_dir"]),
        environment=environment,
    )


def step_selection_premium(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    command = [
        _python(manifest),
        str(
            REPO_ROOT
            / "Scripts"
            / "Modeling"
            / "s5_Auction_Selection_Premium.py"
        ),
        "--year",
        str(manifest["options"]["year"]),
        "--league",
        "beta",
        "--trials",
        str(manifest["options"]["selection_trials"]),
        "--workers",
        str(manifest["options"]["selection_workers"]),
        "--simulation-db",
        str(paths["staged"]["simulation"]),
        "--validations-db",
        str(paths["staged"]["validations"]),
        "--no-app-sync",
    ]
    return run_logged_command(
        command,
        step="selection_premium",
        stage_dir=Path(manifest["stage_dir"]),
        environment=_subprocess_environment(
            paths["staged"],
            year=int(manifest["options"]["year"]),
        ),
    )


def _fetch_set(
    path: Path,
    query: str,
    parameters: Sequence[Any] = (),
) -> set[tuple[Any, ...]]:
    with sqlite3.connect(path) as connection:
        return {tuple(row) for row in connection.execute(query, parameters)}


def _query_row_count(path: Path, query: str) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM ({query})"
            ).fetchone()[0]
        )


def _assert_cross_league_identity(
    databases: Mapping[str, Path] | Path,
    beta_database: Path | None = None,
) -> dict[str, int]:
    if isinstance(databases, (str, Path)):
        if beta_database is None:
            raise ValueError("A beta database is required")
        league_databases = {
            "dk": Path(databases),
            "beta": Path(beta_database),
        }
    else:
        league_databases = {
            str(league): Path(path)
            for league, path in databases.items()
        }
    if len(league_databases) < 2:
        raise ValueError("Cross-league identity requires at least two builds")
    reference_league = next(iter(league_databases))
    reference_database = league_databases[reference_league]
    exact_tables = {
        "player_identity": "SELECT * FROM player_identity",
        "player_aliases": "SELECT * FROM player_aliases",
    }
    results: dict[str, int] = {}
    for table, query in exact_tables.items():
        reference_rows = _fetch_set(reference_database, query)
        reference_count = _query_row_count(reference_database, query)
        if reference_count != len(reference_rows):
            raise ValueError(
                f"{reference_league} {table} contains duplicate rows"
            )
        for league, database in league_databases.items():
            rows = _fetch_set(database, query)
            count = _query_row_count(database, query)
            if count != len(rows):
                raise ValueError(f"{league} {table} contains duplicate rows")
            if rows != reference_rows:
                raise ValueError(
                    f"Cross-league {table} drifted for {league} versus "
                    f"{reference_league}: "
                    f"{reference_league}_only="
                    f"{len(reference_rows - rows)}, "
                    f"{league}_only={len(rows - reference_rows)}"
                )
        results[table] = len(reference_rows)

    key_queries = {
        "player_season_spine": (
            "SELECT player_key, season, gsis_id, display_name, position, team "
            "FROM player_season_spine"
        ),
        "player_season_features": (
            "SELECT player_key, season FROM player_season_features"
        ),
    }
    for table, query in key_queries.items():
        reference_rows = _fetch_set(reference_database, query)
        reference_count = _query_row_count(reference_database, query)
        if reference_count != len(reference_rows):
            raise ValueError(
                f"{reference_league} {table} contains duplicate keys"
            )
        for league, database in league_databases.items():
            rows = _fetch_set(database, query)
            count = _query_row_count(database, query)
            if count != len(rows):
                raise ValueError(f"{league} {table} contains duplicate keys")
            if rows != reference_rows:
                raise ValueError(
                    f"Cross-league {table} key drifted for {league} versus "
                    f"{reference_league}: "
                    f"{reference_league}_only="
                    f"{len(reference_rows - rows)}, "
                    f"{league}_only={len(rows - reference_rows)}"
                )
        results[table] = len(reference_rows)
    return results


def _active_foundation_run_id(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        milestone_3 = connection.execute(
            """
            SELECT foundation_run_id
            FROM build_runs
            WHERE component='milestone_3' AND status='complete'
            """
        ).fetchall()
        if len(milestone_3) != 1 or not milestone_3[0][0]:
            raise ValueError(
                f"{path} must contain exactly one complete milestone_3 run"
            )
        milestone_2 = connection.execute(
            """
            SELECT foundation_run_id, status
            FROM build_runs
            WHERE run_id=? AND component='milestone_2'
            """,
            (str(milestone_3[0][0]),),
        ).fetchall()
        if (
            len(milestone_2) != 1
            or str(milestone_2[0][1]) != "complete"
            or not milestone_2[0][0]
        ):
            raise ValueError(
                f"{path} milestone_3 does not resolve to one complete "
                "milestone_2 foundation"
            )
        milestone_1 = connection.execute(
            """
            SELECT run_id, status
            FROM build_runs
            WHERE run_id=? AND component='milestone_1'
            """,
            (str(milestone_2[0][0]),),
        ).fetchall()
        if len(milestone_1) != 1 or str(milestone_1[0][1]) != "complete":
            raise ValueError(
                f"{path} milestone_2 does not resolve to one complete "
                "milestone_1 foundation"
            )
    return str(milestone_1[0][0])


def _validate_cross_league_remote_sources(
    databases: Mapping[str, Path] | Path,
    beta_database: Path | None = None,
) -> dict[str, Any]:
    """Require every scoring build to consume identical remote payloads."""

    if isinstance(databases, (str, Path)):
        if beta_database is None:
            raise ValueError("A beta database is required")
        league_databases = {
            "dk": Path(databases),
            "beta": Path(beta_database),
        }
    else:
        league_databases = {
            str(league): Path(path)
            for league, path in databases.items()
        }
    if len(league_databases) < 2:
        raise ValueError("Cross-league source validation requires two builds")

    receipts: dict[str, set[tuple[Any, ...]]] = {}
    foundation_runs: dict[str, str] = {}
    for league, database in league_databases.items():
        foundation_run = _active_foundation_run_id(database)
        foundation_runs[league] = foundation_run
        rows = _fetch_set(
            database,
            """
            SELECT component,
                   source_name,
                   source_uri,
                   source_sha256,
                   row_count
            FROM source_manifest
            WHERE run_id=? AND source_sha256 IS NOT NULL
            """,
            (foundation_run,),
        )
        if not rows:
            raise ValueError(
                f"{league} active foundation has no hashed remote sources"
            )
        with closing(sqlite3.connect(database)) as connection:
            active_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM source_manifest
                    WHERE run_id=? AND source_sha256 IS NOT NULL
                    """,
                    (foundation_run,),
                ).fetchone()[0]
            )
        if active_count != len(rows):
            raise ValueError(
                f"{league} active foundation duplicates remote source receipts"
            )
        receipts[league] = rows
    reference_league = next(iter(receipts))
    reference_receipts = receipts[reference_league]
    for league, rows in receipts.items():
        if rows != reference_receipts:
            raise ValueError(
                "Scoring builds used different remote source payloads: "
                f"{reference_league}_only="
                f"{len(reference_receipts - rows)}, "
                f"{league}_only={len(rows - reference_receipts)}"
            )
    return {
        "receipt_count": len(reference_receipts),
        "foundation_run_ids": foundation_runs,
    }


def _validate_model_inputs(
    path: Path,
    year: int,
    *,
    minimums: Mapping[str, int] | None = None,
) -> dict[str, int]:
    if minimums is None:
        minimums = get_production_cycle(
            year
        ).model_input_position_minimums
    counts: dict[str, int] = {}
    with sqlite3.connect(path) as connection:
        for position in ("QB", "RB", "WR", "TE"):
            table = f"{position}_{year}_ProjOnly"
            if not _table_exists(connection, table):
                raise ValueError(f"Model inputs are missing {table}")
            count = int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}" '
                    "WHERE CAST(year AS INTEGER)=?",
                    (year,),
                ).fetchone()[0]
            )
            duplicates = int(
                connection.execute(
                    "SELECT COUNT(*) FROM ("
                    f'SELECT player, year FROM "{table}" '
                    "WHERE CAST(year AS INTEGER)=? "
                    "GROUP BY player, year HAVING COUNT(*) > 1)",
                    (year,),
                ).fetchone()[0]
            )
            if count <= 0 or duplicates:
                raise ValueError(
                    f"Invalid {table}: rows={count}, duplicate_keys={duplicates}"
                )
            minimum = int(minimums[position])
            if count < minimum:
                raise ValueError(
                    f"{table} has only {count} rows; the {year} release "
                    f"floor is {minimum}"
                )
            counts[position] = count
    return counts


def _validate_model_acceptance(
    database: Path,
    *,
    league: str,
) -> dict[str, Any]:
    """Require the locked models to beat their prespecified expert baselines."""

    required = (
        "locked_model_comparisons",
        "next_year_model_comparisons",
    )
    _require_tables(database, required)
    with closing(sqlite3.connect(database)) as connection:
        current = connection.execute(
            """
            SELECT seasons,
                   mean_season_rmse_delta,
                   recent_mean_delta,
                   bootstrap_95_upper,
                   season_wins
            FROM locked_model_comparisons
            WHERE comparison='primary_vs_expert'
            """
        ).fetchall()
        next_ppg = connection.execute(
            """
            SELECT origins,
                   mean_origin_delta,
                   recent_mean_delta,
                   bootstrap_95_upper,
                   origin_wins
            FROM next_year_model_comparisons
            WHERE comparison='primary_vs_expert_carry'
              AND metric='rmse'
            """
        ).fetchall()
        next_participation = connection.execute(
            """
            SELECT origins,
                   mean_origin_delta,
                   recent_mean_delta,
                   bootstrap_95_upper,
                   origin_wins
            FROM next_year_model_comparisons
            WHERE comparison='participation_lgbm_vs_prior'
              AND metric='brier'
            """
        ).fetchall()

    comparisons = {
        "current_ppg_vs_expert": current,
        "next_ppg_vs_expert_carry": next_ppg,
        "next_participation_vs_prior": next_participation,
    }
    receipt: dict[str, Any] = {}
    for name, rows in comparisons.items():
        if len(rows) != 1:
            raise ValueError(
                f"{league} {name} acceptance evidence must contain exactly "
                f"one row; observed {len(rows)}"
            )
        periods, mean_delta, recent_delta, bootstrap_upper, wins = rows[0]
        numeric = (
            float(mean_delta),
            float(recent_delta),
            float(bootstrap_upper),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"{league} {name} has non-finite evidence")
        periods = int(periods)
        wins = int(wins)
        if periods < 8 or wins <= periods / 2:
            raise ValueError(
                f"{league} {name} lacks broad temporal support: "
                f"periods={periods}, wins={wins}"
            )
        if any(value > 0 for value in numeric):
            raise ValueError(
                f"{league} {name} did not clear the locked baseline: "
                f"mean_delta={mean_delta}, recent_delta={recent_delta}, "
                f"bootstrap_95_upper={bootstrap_upper}"
            )
        receipt[name] = {
            "periods": periods,
            "wins": wins,
            "mean_delta": float(mean_delta),
            "recent_delta": float(recent_delta),
            "bootstrap_95_upper": float(bootstrap_upper),
        }
    return receipt


def _validate_source_markets(path: Path, year: int) -> dict[str, Any]:
    cycle = get_production_cycle(year)
    minimums = cycle.source_market_minimums
    specs = {
        "dk": ("ADP_Averages", "year=? AND league='dk'"),
        "nffc": ("NFFC_ADP", "year=?"),
        "etr": ("ETR_Ranks", "year=?"),
    }
    results: dict[str, Any] = {}
    with sqlite3.connect(path) as connection:
        for source, (table, predicate) in specs.items():
            if not _table_exists(connection, table):
                raise ValueError(f"Raw source database is missing {table}")
            count = int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE {predicate}',
                    (year,),
                ).fetchone()[0]
            )
            minimum = int(minimums[source])
            if count < minimum:
                raise ValueError(
                    f"Raw source {source} has only {count} rows for {year}; "
                    f"the {year} release floor is {minimum}"
                )
            results[source] = count

        nffc_feed_counts = {
            str(label): int(count)
            for label, count in connection.execute(
                """
                SELECT COALESCE(source, '<NULL>'), COUNT(*)
                FROM NFFC_ADP
                WHERE year=?
                GROUP BY COALESCE(source, '<NULL>')
                """,
                (year,),
            )
        }
        expected_labels = set(cycle.nffc_source_feed_minimums)
        actual_labels = set(nffc_feed_counts)
        if actual_labels != expected_labels:
            raise ValueError(
                f"NFFC_ADP source labels for {year} do not match the annual "
                "contract: "
                f"missing={sorted(expected_labels - actual_labels)}, "
                f"unexpected={sorted(actual_labels - expected_labels)}"
            )
        for label, minimum in cycle.nffc_source_feed_minimums.items():
            count = nffc_feed_counts[label]
            if count < int(minimum):
                raise ValueError(
                    f"Raw NFFC feed {label} has only {count} rows for {year}; "
                    f"the {year} release floor is {minimum}"
                )
        results["nffc_feed_counts"] = dict(sorted(nffc_feed_counts.items()))
    return results


def _count_invalid_nffc_template_context_rows(
    connection: sqlite3.Connection,
    *,
    expected_center_policy: str,
    expected_context_source: str,
    expected_scoring_hash: str,
    expected_horizon: int,
) -> int:
    """Count NFFC template rows that violate scored-context provenance."""

    return int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM Best_Ball_Weekly_Templates
            WHERE league='nffc'
              AND (
                  historical_projection_source IS NULL
                  OR historical_projection_source<>
                      'v2_nffc_expert_consensus'
                  OR historical_center_policy IS NULL
                  OR historical_center_policy<>?
                  OR projection_context_source IS NULL
                  OR projection_context_source<>?
                  OR projection_context_scoring_hash IS NULL
                  OR projection_context_scoring_hash<>?
                  OR projection_context_run_id IS NULL
                  OR TRIM(projection_context_run_id)=''
                  OR model_input_avg_proj_points IS NULL
                  OR projection_context_avg_proj_points_delta IS NULL
                  OR avg_proj_points IS NULL
                  OR historical_pred_fp_per_game IS NULL
                  OR avg_proj_pass_points IS NULL
                  OR avg_proj_rush_points IS NULL
                  OR avg_proj_rec_points IS NULL
                  OR ABS(
                      avg_proj_points / ? -
                      historical_pred_fp_per_game
                  ) > 0.000000001
                  OR ABS(
                      avg_proj_pass_points +
                      avg_proj_rush_points +
                      avg_proj_rec_points -
                      avg_proj_points
                  ) > 0.000000001
                  OR ABS(
                      avg_proj_points -
                      model_input_avg_proj_points -
                      projection_context_avg_proj_points_delta
                  ) > 0.000000001
                  OR COALESCE(v2_recenter_promoted, -1)<>0
              )
            """,
            (
                expected_center_policy,
                expected_context_source,
                expected_scoring_hash,
                expected_horizon,
            ),
        ).fetchone()[0]
    )


def _count_invalid_nffc_player_map_context_rows(
    connection: sqlite3.Connection,
    *,
    year: int,
    dataset: str,
    expected_scoring_hash: str,
    expected_feature_run_id: str,
) -> int:
    """Count current NFFC player-map rows with invalid scored context."""

    return int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM Best_Ball_Weekly_Player_Map
            WHERE version='nffc'
              AND year=?
              AND dataset=?
              AND (
                  current_context_source IS NULL
                  OR current_context_source NOT IN (
                      'model_inputs_with_v2_scoring_context',
                      'v2_player_season_features_scoring_context'
                  )
                  OR projection_context_scoring_hash IS NULL
                  OR projection_context_scoring_hash<>?
                  OR projection_context_run_id IS NULL
                  OR projection_context_run_id<>?
                  OR current_avg_proj_points IS NULL
                  OR avg_proj_pass_points IS NULL
                  OR avg_proj_rush_points IS NULL
                  OR avg_proj_rec_points IS NULL
                  OR ABS(
                      avg_proj_pass_points +
                      avg_proj_rush_points +
                      avg_proj_rec_points -
                      current_avg_proj_points
                  ) > 0.000000001
              )
            """,
            (
                year,
                dataset,
                expected_scoring_hash,
                expected_feature_run_id,
            ),
        ).fetchone()[0]
    )


def _validate_simulation(
    path: Path,
    *,
    year: int,
    dataset: str,
    selection_trials: int,
) -> dict[str, Any]:
    cycle = get_production_cycle(year)
    _require_tables(
        path,
        (*GOVERNED_HANDOFF_TABLES, *WEEKLY_TABLES, *SALARY_TABLES),
    )
    results: dict[str, Any] = {}
    with sqlite3.connect(path) as connection:
        import pandas as pd

        from Scripts.V2.production_handoff import (
            validate_avg_adp_publication,
        )

        published_avg_adps = pd.read_sql_query(
            "SELECT * FROM Avg_ADPs",
            connection,
        )
        validate_avg_adp_publication(published_avg_adps, year=year)
        avg_duplicate_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT COALESCE(player_key, draft_entity_key) entity_key,
                           year,
                           league
                    FROM Avg_ADPs
                    WHERE CAST(year AS INTEGER)=?
                    GROUP BY entity_key, year, league
                    HAVING COUNT(*) > 1
                )
                """,
                (year,),
            ).fetchone()[0]
        )
        if avg_duplicate_count:
            raise ValueError(
                f"Avg_ADPs has {avg_duplicate_count} duplicate current keys"
            )
        missing_avg_key_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM Avg_ADPs
                WHERE CAST(year AS INTEGER)=?
                  AND player_key IS NULL
                  AND draft_entity_key IS NULL
                """,
                (year,),
            ).fetchone()[0]
        )
        if missing_avg_key_count:
            raise ValueError(
                f"Avg_ADPs has {missing_avg_key_count} unkeyed current rows"
            )
        avg_counts = dict(
            connection.execute(
                """
                SELECT league, COUNT(*)
                FROM Avg_ADPs
                WHERE CAST(year AS INTEGER)=?
                  AND league IN ('dk', 'nffc', 'etr')
                  AND player_key IS NOT NULL
                  AND avg_pick IS NOT NULL
                GROUP BY league
                """,
                (year,),
            ).fetchall()
        )
        if any(int(avg_counts.get(league, 0)) <= 0 for league in ("dk", "nffc", "etr")):
            raise ValueError(f"Canonical current ADP feeds are incomplete: {avg_counts}")
        results["avg_adp_counts"] = avg_counts

        prediction_keys: dict[str, set[str]] = {}
        prediction_position_counts: dict[str, dict[str, int]] = {}
        map_keys: dict[str, set[str]] = {}
        for league in cycle.leagues:
            prediction_rows = connection.execute(
                """
                SELECT player_key,
                       pred_fp_per_game,
                       pred_fp_per_game_ny,
                       pred_appear_current,
                       pred_appear_ny
                FROM Final_Predictions_Resid
                WHERE year=? AND dataset=? AND version=?
                """,
                (year, dataset, league),
            ).fetchall()
            if not prediction_rows:
                raise ValueError(f"No production predictions for {league}")
            keys = [row[0] for row in prediction_rows]
            if any(key is None or str(key).strip() == "" for key in keys):
                raise ValueError(f"{league} production has missing player keys")
            if len(keys) != len(set(keys)):
                raise ValueError(f"{league} production has duplicate player keys")
            if any(
                value is None
                for row in prediction_rows
                for value in row[1:]
            ):
                raise ValueError(f"{league} production projections are incomplete")
            if any(
                not math.isfinite(float(value))
                for row in prediction_rows
                for value in row[1:]
            ):
                raise ValueError(
                    f"{league} production projections contain non-finite values"
                )
            if any(
                not 0.0 <= float(row[index]) <= 1.0
                for row in prediction_rows
                for index in (3, 4)
            ):
                raise ValueError(
                    f"{league} production appearance probabilities are invalid"
                )
            population_minimum = int(
                cycle.production_population_minimums[league]
            )
            if len(prediction_rows) < population_minimum:
                raise ValueError(
                    f"{league} production has only {len(prediction_rows)} "
                    f"players; the {year} release floor is "
                    f"{population_minimum}"
                )
            prediction_keys[league] = {str(key) for key in keys}
            position_counts = {
                str(position): int(count)
                for position, count in connection.execute(
                    """
                    SELECT pos, COUNT(*)
                    FROM Final_Predictions_Resid
                    WHERE year=? AND dataset=? AND version=?
                    GROUP BY pos
                    """,
                    (year, dataset, league),
                ).fetchall()
            }
            below_floor = {
                position: {
                    "rows": int(position_counts.get(position, 0)),
                    "minimum": int(minimum),
                }
                for position, minimum in (
                    cycle.production_position_minimums[league].items()
                )
                if int(position_counts.get(position, 0)) < int(minimum)
            }
            if below_floor:
                raise ValueError(
                    f"{league} production population is materially truncated "
                    f"by position: {below_floor}"
                )
            prediction_position_counts[league] = position_counts

            rows = connection.execute(
                """
                SELECT player_key
                FROM Best_Ball_Weekly_Player_Map
                WHERE year=? AND dataset=? AND version=?
                """,
                (year, dataset, league),
            ).fetchall()
            map_values = [row[0] for row in rows]
            if len(map_values) != len(set(map_values)):
                raise ValueError(f"{league} weekly map has duplicate player keys")
            map_keys[league] = {str(key) for key in map_values if key is not None}
            if map_keys[league] != prediction_keys[league]:
                raise ValueError(
                    f"{league} weekly/projection population mismatch: "
                    f"projection_only={len(prediction_keys[league] - map_keys[league])}, "
                    f"weekly_only={len(map_keys[league] - prediction_keys[league])}"
                )

            pool_sizes = connection.execute(
                """
                SELECT template_pool_key, COUNT(*)
                FROM Best_Ball_Weekly_Template_Pools
                WHERE pool_year=? AND pool_dataset=? AND pool_version=?
                GROUP BY template_pool_key
                """,
                (year, dataset, league),
            ).fetchall()
            if len(pool_sizes) != len(prediction_keys[league]):
                raise ValueError(
                    f"{league} weekly pool count does not match production"
                )
            invalid_pool_sizes = [
                (key, count) for key, count in pool_sizes if int(count) != 80
            ]
            if invalid_pool_sizes:
                raise ValueError(
                    f"{league} weekly pools do not contain exactly 80 donors: "
                    f"{invalid_pool_sizes[:10]}"
                )
            expected_horizon = int(cycle.weekly_horizons[league])
            required_week_column = f"week_{expected_horizon}"
            template_columns = {
                str(row[1])
                for row in connection.execute(
                    'PRAGMA table_info("Best_Ball_Weekly_Templates")'
                )
            }
            if required_week_column not in template_columns:
                raise ValueError(
                    f"{league} weekly templates lack "
                    f"{required_week_column}"
                )
            invalid_horizon_rows = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM Best_Ball_Weekly_Templates
                    WHERE league=?
                      AND (
                          season < ?
                          OR "{required_week_column}" IS NULL
                      )
                    """,
                    (
                        league,
                        int(cycle.template_min_seasons[league]),
                    ),
                ).fetchone()[0]
            )
            if invalid_horizon_rows:
                raise ValueError(
                    f"{league} has {invalid_horizon_rows} template rows "
                    "outside its governed season/horizon contract"
                )
            observed_center_policies = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT historical_center_policy
                    FROM Best_Ball_Weekly_Templates
                    WHERE league=?
                    """,
                    (league,),
                )
                if row[0] is not None and str(row[0]).strip()
            }
            approved_center_policies = set(
                cycle.template_center_policies[league]
            )
            if (
                not observed_center_policies
                or not observed_center_policies.issubset(
                    approved_center_policies
                )
            ):
                raise ValueError(
                    f"{league} template center policies are not approved: "
                    f"observed={sorted(observed_center_policies)}, "
                    f"approved={sorted(approved_center_policies)}"
                )
            if league == "nffc":
                required_nffc_context_columns = {
                    "projection_context_source",
                    "projection_context_scoring_hash",
                    "projection_context_run_id",
                    "model_input_avg_proj_points",
                    "projection_context_avg_proj_points_delta",
                }
                missing_nffc_context_columns = sorted(
                    required_nffc_context_columns - template_columns
                )
                if missing_nffc_context_columns:
                    raise ValueError(
                        "NFFC templates lack scored-context audit columns: "
                        f"{missing_nffc_context_columns}"
                    )
                expected_context_source = (
                    cycle.template_context_sources["nffc"]
                )
                expected_scoring_hash = scoring_hash("nffc")
                invalid_nffc_context = (
                    _count_invalid_nffc_template_context_rows(
                        connection,
                        expected_center_policy=(
                            cycle.template_center_policies["nffc"][0]
                        ),
                        expected_context_source=expected_context_source,
                        expected_scoring_hash=expected_scoring_hash,
                        expected_horizon=expected_horizon,
                    )
                )
                if invalid_nffc_context:
                    raise ValueError(
                        "NFFC scored template context/center contract failed "
                        f"for {invalid_nffc_context} rows"
                    )
                nffc_context_runs = {
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT DISTINCT projection_context_run_id
                        FROM Best_Ball_Weekly_Templates
                        WHERE league='nffc'
                        """
                    )
                }
                if len(nffc_context_runs) != 1:
                    raise ValueError(
                        "NFFC templates do not have one feature-context run: "
                        f"{sorted(nffc_context_runs)}"
                    )
                player_map_columns = {
                    str(row[1])
                    for row in connection.execute(
                        'PRAGMA table_info("Best_Ball_Weekly_Player_Map")'
                    )
                }
                required_nffc_map_columns = {
                    "current_context_source",
                    "projection_context_scoring_hash",
                    "projection_context_run_id",
                }
                missing_nffc_map_columns = sorted(
                    required_nffc_map_columns - player_map_columns
                )
                if missing_nffc_map_columns:
                    raise ValueError(
                        "NFFC player map lacks scored-context audit columns: "
                        f"{missing_nffc_map_columns}"
                    )
                expected_feature_run_id = next(iter(nffc_context_runs))
                invalid_nffc_map_context = (
                    _count_invalid_nffc_player_map_context_rows(
                        connection,
                        year=year,
                        dataset=dataset,
                        expected_scoring_hash=expected_scoring_hash,
                        expected_feature_run_id=expected_feature_run_id,
                    )
                )
                if invalid_nffc_map_context:
                    raise ValueError(
                        "NFFC current player-map scoring context failed for "
                        f"{invalid_nffc_map_context} rows"
                    )
                results["nffc_template_context"] = {
                    "source": expected_context_source,
                    "scoring_hash": expected_scoring_hash,
                    "feature_run_id": expected_feature_run_id,
                    "center_policy": (
                        cycle.template_center_policies["nffc"][0]
                    ),
                }

        league_placeholders = ", ".join("?" for _ in cycle.leagues)
        bad_adp = connection.execute(
            f"""
            SELECT COALESCE(SUM(CASE WHEN needs_review THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN using_default_adp THEN 1 ELSE 0 END), 0),
                   COALESCE(
                       SUM(CASE WHEN high_impact_unresolved_adp THEN 1 ELSE 0 END),
                       0
                   )
            FROM Best_Ball_ADP_Audit
            WHERE year=? AND dataset=?
              AND version IN ({league_placeholders})
            """,
            (year, dataset, *cycle.leagues),
        ).fetchone()
        if any(int(value) for value in bad_adp):
            raise ValueError(
                "Weekly ADP audit failed: "
                f"needs_review={bad_adp[0]}, defaults={bad_adp[1]}, "
                f"high_impact_unresolved={bad_adp[2]}"
            )

        salary_rows = connection.execute(
            """
            SELECT player_key
            FROM Salaries_Pred
            WHERE year=? AND league='betapred'
            """,
            (year,),
        ).fetchall()
        salary_keys = [row[0] for row in salary_rows]
        if not salary_keys or any(key is None for key in salary_keys):
            raise ValueError("Current salary predictions lack canonical keys")
        if len(salary_keys) != len(set(salary_keys)):
            raise ValueError("Current salary predictions duplicate player keys")
        salary_key_set = {str(key) for key in salary_keys}
        if salary_key_set != prediction_keys["beta"]:
            raise ValueError(
                "Current salary/projection population mismatch: "
                f"projection_only="
                f"{len(prediction_keys['beta'] - salary_key_set)}, "
                f"salary_only="
                f"{len(salary_key_set - prediction_keys['beta'])}"
            )

        keeper_keys = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT player_key
                FROM League_Keepers
                WHERE year=? AND league='beta'
                """,
                (year,),
            )
            if row[0] is not None
        }
        premium_keys = [
            row[0]
            for row in connection.execute(
                """
                SELECT player_key
                FROM Salary_Selection_Premium
                WHERE year=? AND league='beta'
                """,
                (year,),
            )
        ]
        premium_row = connection.execute(
            """
            SELECT COUNT(*),
                   COUNT(DISTINCT player_key),
                   SUM(player_key IS NULL),
                   MIN(seed_trials),
                   MIN(seed_success_trials)
            FROM Salary_Selection_Premium
            WHERE year=? AND league='beta'
            """,
            (year,),
        ).fetchone()
        if (
            int(premium_row[0] or 0) <= 0
            or int(premium_row[0]) != int(premium_row[1])
            or int(premium_row[2] or 0)
            or int(premium_row[3] or 0) != int(selection_trials)
            or int(premium_row[4] or 0) != int(selection_trials)
        ):
            raise ValueError(
                "Current auction selection-premium surface is invalid: "
                f"{premium_row}"
            )
        premium_key_set = {
            str(key) for key in premium_keys if key is not None
        }
        expected_premium_keys = prediction_keys["beta"] - keeper_keys
        if premium_key_set != expected_premium_keys:
            raise ValueError(
                "Selection-premium population does not equal beta "
                "non-keepers: "
                f"missing={len(expected_premium_keys - premium_key_set)}, "
                f"extra={len(premium_key_set - expected_premium_keys)}"
            )
        results.update(
            {
                "production_counts": {
                    league: len(keys)
                    for league, keys in prediction_keys.items()
                },
                "production_position_counts": prediction_position_counts,
                "weekly_population_matches": True,
                "weekly_pool_donors": 80,
                "weekly_horizons": dict(cycle.weekly_horizons),
                "weekly_adp_audit": {
                    "needs_review": int(bad_adp[0]),
                    "defaults": int(bad_adp[1]),
                    "high_impact_unresolved": int(bad_adp[2]),
                },
                "salary_prediction_count": len(salary_keys),
                "selection_premium_count": int(premium_row[0]),
                "selection_success_trials": int(premium_row[4]),
            }
        )
    return results


def validate_release(manifest: Mapping[str, Any]) -> dict[str, Any]:
    assert_production_cycle_unchanged(manifest)
    cycle = get_production_cycle(int(manifest["options"]["year"]))
    paths = _resolved_paths(manifest)
    staged = paths["staged"]
    integrity = {
        key: validate_sqlite(path)
        for key, path in staged.items()
    }
    source = _validate_source_markets(
        staged["source"],
        int(manifest["options"]["year"]),
    )
    model_inputs = _validate_model_inputs(
        staged["model_inputs"],
        int(manifest["options"]["year"]),
    )
    model_inputs_next = _validate_model_inputs(
        staged["model_inputs_next"],
        int(manifest["options"]["year"]),
    )
    identity = _assert_cross_league_identity(
        {
            league: staged[f"v2_{league}"]
            for league in cycle.leagues
        },
    )
    remote_sources = _validate_cross_league_remote_sources(
        {
            league: staged[f"v2_{league}"]
            for league in cycle.leagues
        },
    )
    from Scripts.V2.production_handoff import (
        load_validated_shadow_predictions,
    )

    lineage: dict[str, dict[str, Any]] = {}
    for league in cycle.leagues:
        database = staged[f"v2_{league}"]
        current_shadow, next_shadow = load_validated_shadow_predictions(
            database,
            league=league,
            year=int(manifest["options"]["year"]),
        )
        lineage[league] = {
            "current_shadow_rows": len(current_shadow),
            "next_shadow_rows": len(next_shadow),
            "acceptance": _validate_model_acceptance(
                database,
                league=league,
            ),
        }
        _require_tables(
            database,
            (
                cycle.current_shadow_table,
                cycle.next_shadow_table,
                "locked_template_production_unmatched",
            ),
        )
        with sqlite3.connect(database) as connection:
            unmatched = int(
                connection.execute(
                    "SELECT COUNT(*) "
                    "FROM locked_template_production_unmatched"
                ).fetchone()[0]
            )
            active_feature_runs = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT runs.feature_run_id
                    FROM locked_template_handoff handoff
                    JOIN locked_candidate_runs runs
                      ON runs.model_run_id=handoff.model_run_id
                    WHERE runs.feature_run_id IS NOT NULL
                    """
                )
            }
        if unmatched:
            raise ValueError(
                f"{league} template handoff has {unmatched} unmatched rows"
            )
        if len(active_feature_runs) != 1:
            raise ValueError(
                f"{league} active handoff does not reference exactly one "
                f"feature run: {sorted(active_feature_runs)}"
            )
        lineage[league]["feature_run_id"] = next(
            iter(active_feature_runs)
        )
    simulation = _validate_simulation(
        staged["simulation"],
        year=int(manifest["options"]["year"]),
        dataset=str(manifest["options"]["dataset"]),
        selection_trials=int(manifest["options"]["selection_trials"]),
    )
    nffc_template_context = simulation.get("nffc_template_context", {})
    if nffc_template_context.get("feature_run_id") != lineage["nffc"].get(
        "feature_run_id"
    ):
        raise ValueError(
            "NFFC template context does not reference the active locked "
            "feature run: "
            f"template={nffc_template_context.get('feature_run_id')!r}, "
            f"locked={lineage['nffc'].get('feature_run_id')!r}"
        )
    return {
        "integrity": integrity,
        "source_market_counts": source,
        "model_input_counts": model_inputs,
        "next_model_input_counts": model_inputs_next,
        "cross_league_identity_counts": identity,
        "cross_league_remote_sources": remote_sources,
        "shadow_lineage": lineage,
        "simulation": simulation,
    }


def step_validate(manifest: dict[str, Any]) -> dict[str, Any]:
    return validate_release(manifest)


def synchronize_sqlite_tables(
    source: Path,
    destination: Path,
    tables: Sequence[str],
) -> dict[str, int]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    validate_sqlite(source)
    validate_sqlite(destination)
    row_counts: dict[str, int] = {}
    with sqlite3.connect(destination) as connection:
        connection.execute("ATTACH DATABASE ? AS generated_source", (str(source),))
        connection.execute("BEGIN IMMEDIATE")
        try:
            for table in tables:
                schema = connection.execute(
                    "SELECT sql FROM generated_source.sqlite_master "
                    "WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if schema is None or not schema[0]:
                    raise ValueError(f"Generated source is missing {table}")
                indexes = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT sql FROM generated_source.sqlite_master "
                        "WHERE type='index' AND tbl_name=? "
                        "AND sql IS NOT NULL ORDER BY name",
                        (table,),
                    )
                ]
                connection.execute(f'DROP TABLE IF EXISTS main."{table}"')
                connection.execute(str(schema[0]))
                connection.execute(
                    f'INSERT INTO main."{table}" '
                    f'SELECT * FROM generated_source."{table}"'
                )
                for statement in indexes:
                    connection.execute(statement)
                row_counts[table] = int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM main."{table}"'
                    ).fetchone()[0]
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    validate_sqlite(destination)
    return row_counts


def step_prepare_apps(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    auction_owned_tables = sorted(
        set(database_tables(paths["app_bases"]["auction"]))
        - set(GENERATED_AUCTION_TABLES)
    )
    auction_owned_before = table_digests(
        paths["app_bases"]["auction"],
        auction_owned_tables,
    )
    auction_receipt = sqlite_backup(
        paths["app_bases"]["auction"],
        paths["app_artifacts"]["auction"],
    )
    auction_counts = synchronize_sqlite_tables(
        paths["staged"]["simulation"],
        paths["app_artifacts"]["auction"],
        GENERATED_AUCTION_TABLES,
    )
    auction_owned_after = table_digests(
        paths["app_artifacts"]["auction"],
        auction_owned_tables,
    )
    if auction_owned_before != auction_owned_after:
        changed = sorted(
            table
            for table in auction_owned_tables
            if auction_owned_before.get(table) != auction_owned_after.get(table)
        )
        raise ValueError(
            f"Auction app-owned tables changed during staging: {changed}"
        )
    source_generated = table_digests(
        paths["staged"]["simulation"],
        GENERATED_AUCTION_TABLES,
    )
    auction_generated = table_digests(
        paths["app_artifacts"]["auction"],
        GENERATED_AUCTION_TABLES,
    )
    if source_generated != auction_generated:
        changed = sorted(
            table
            for table in GENERATED_AUCTION_TABLES
            if source_generated.get(table) != auction_generated.get(table)
        )
        raise ValueError(
            f"Auction generated-table parity failed: {changed}"
        )
    snake_receipt = sqlite_backup(
        paths["staged"]["simulation"],
        paths["app_artifacts"]["snake"],
    )
    simulation_tables = database_tables(paths["staged"]["simulation"])
    snake_tables = database_tables(paths["app_artifacts"]["snake"])
    if simulation_tables != snake_tables:
        raise ValueError("Snake candidate table inventory differs from staging")
    if table_digests(
        paths["staged"]["simulation"],
        simulation_tables,
    ) != table_digests(
        paths["app_artifacts"]["snake"],
        snake_tables,
    ):
        raise ValueError("Snake candidate table content differs from staging")
    return {
        "auction": {
            "copy": auction_receipt,
            "generated_table_counts": auction_counts,
            "app_owned_table_count": len(auction_owned_tables),
            "app_owned_tables_unchanged": True,
            "generated_tables_match_staging": True,
            "final_state": database_state(paths["app_artifacts"]["auction"]),
        },
        "snake": {
            "copy": snake_receipt,
            "table_count": len(snake_tables),
            "tables_match_staging": True,
            "final_state": database_state(paths["app_artifacts"]["snake"]),
        },
    }


def step_app_smoke(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = _resolved_paths(manifest)
    results = {}
    for app in ("auction", "snake"):
        step = f"app_smoke_{app}"
        # AppTest may use the app runtime when the modeling virtualenv does not
        # contain Streamlit.
        command = [
            _app_python(manifest),
            "-m",
            "Scripts.V2.app_smoke",
            app,
            "--database",
            str(paths["app_artifacts"][app]),
            "--timeout-seconds",
            str(manifest["options"]["app_timeout"]),
            "--expected-year",
            str(manifest["options"]["year"]),
        ]
        if app == "snake":
            for league in ("dk", "nffc"):
                command.extend(["--require-league", league])
        results[app] = run_logged_command(
            command,
            step=step,
            stage_dir=Path(manifest["stage_dir"]),
            environment=_subprocess_environment(
                paths["staged"],
                year=int(manifest["options"]["year"]),
            ),
        )
        results[app]["database_state"] = database_state(
            paths["app_artifacts"][app]
        )
    return results


def execute_step(step: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if step == "snapshot":
        return step_snapshot(manifest)
    if step == "model_inputs":
        return step_model_inputs(manifest)
    if (
        step.startswith("v2_")
        and step.removeprefix("v2_") in PRODUCTION_LEAGUES
    ):
        return step_v2(manifest, step.removeprefix("v2_"))
    if (
        step.startswith("locked_")
        and step.removeprefix("locked_") in PRODUCTION_LEAGUES
    ):
        return step_locked(manifest, step.removeprefix("locked_"))
    if (
        step.startswith("next_")
        and step.removeprefix("next_") in PRODUCTION_LEAGUES
    ):
        return step_next(manifest, step.removeprefix("next_"))
    if step == "keepers":
        return step_keepers(manifest)
    if step == "handoff":
        return step_handoff(manifest)
    if (
        step.startswith("weekly_")
        and step.removeprefix("weekly_") in PRODUCTION_LEAGUES
    ):
        return step_weekly(manifest, step.removeprefix("weekly_"))
    if (
        step.startswith("template_audit_")
        and step.removeprefix("template_audit_") in PRODUCTION_LEAGUES
    ):
        return step_template_audit(
            manifest,
            step.removeprefix("template_audit_"),
        )
    if step == "salary":
        return step_salary(manifest)
    if step == "selection_premium":
        return step_selection_premium(manifest)
    if step == "validate":
        return step_validate(manifest)
    if step == "prepare_apps":
        return step_prepare_apps(manifest)
    if step == "app_smoke":
        return step_app_smoke(manifest)
    raise ValueError(f"Unsupported refresh step: {step}")


def assert_live_state_unchanged(
    manifest: Mapping[str, Any],
    keys: Iterable[str],
) -> None:
    paths = _resolved_paths(manifest)
    for key in keys:
        expected = manifest["baseline"].get(key)
        if not expected:
            raise ValueError(f"Manifest lacks baseline state for {key}")
        if expected.get("exists", True) is False:
            if paths["live"][key].exists():
                raise RuntimeError(
                    f"Live {key} was created after staging; refusing to "
                    "overwrite newer state. Start a new refresh."
                )
            continue
        actual = database_state(paths["live"][key])
        if (
            int(actual["size_bytes"]) != int(expected["size_bytes"])
            or str(actual["sha256"]) != str(expected["sha256"])
        ):
            raise RuntimeError(
                f"Live {key} changed after staging; refusing to continue or "
                "overwrite newer state. Start a new refresh."
            )


def assert_external_inputs_unchanged(
    manifest: Mapping[str, Any],
) -> None:
    expected_inputs = manifest.get("external_inputs")
    if not expected_inputs:
        raise ValueError("Refresh manifest lacks external-input snapshots")
    for category, state_function in (
        ("sqlite", database_state),
        ("files", regular_file_state),
    ):
        expected_category = expected_inputs.get(category)
        if not expected_category:
            raise ValueError(
                f"Refresh manifest lacks {category} external-input snapshots"
            )
        for label, expected in expected_category.items():
            actual = state_function(Path(expected["path"]))
            if (
                int(actual["size_bytes"]) != int(expected["size_bytes"])
                or str(actual["sha256"]) != str(expected["sha256"])
            ):
                raise RuntimeError(
                    f"External refresh input {label} changed after staging; "
                    "start a new refresh."
                )


def assert_staged_release_unchanged(
    manifest: Mapping[str, Any],
) -> None:
    """Reject changes made after validation or either candidate app smoke."""

    paths = _resolved_paths(manifest)
    validation = manifest["steps"].get("validate", {})
    if validation.get("status") != "completed":
        raise ValueError("The staged validation step is not complete")
    expected_databases = validation["result"]["integrity"]
    for key, path in paths["staged"].items():
        expected = expected_databases.get(key)
        if not expected:
            raise ValueError(f"Validation receipt lacks staged state for {key}")
        actual = database_state(path)
        if (
            int(actual["size_bytes"]) != int(expected["size_bytes"])
            or str(actual["sha256"]) != str(expected["sha256"])
        ):
            raise RuntimeError(
                f"Staged {key} changed after validation; rerun from validate"
            )

    smoke = manifest["steps"].get("app_smoke", {})
    if smoke.get("status") != "completed":
        raise ValueError("The staged app smoke step is not complete")
    for app, path in paths["app_artifacts"].items():
        expected = smoke["result"].get(app, {}).get("database_state")
        if not expected:
            raise ValueError(f"App smoke receipt lacks {app} database state")
        actual = database_state(path)
        if (
            int(actual["size_bytes"]) != int(expected["size_bytes"])
            or str(actual["sha256"]) != str(expected["sha256"])
        ):
            raise RuntimeError(
                f"Staged {app} app database changed after smoke; rerun app_smoke"
            )


def _promotion_sources(
    manifest: Mapping[str, Any],
) -> list[tuple[str, Path, Path]]:
    paths = _resolved_paths(manifest)
    artifacts = [
        (
            key,
            paths["staged"][key],
            paths["live"][key],
        )
        for key in PROMOTED_DATABASES
    ]
    artifacts.extend(
        [
            (
                "auction_app",
                paths["app_artifacts"]["auction"],
                paths["live"]["auction_app"],
            ),
            (
                "snake_app",
                paths["app_artifacts"]["snake"],
                paths["live"]["snake_app"],
            ),
        ]
    )
    return artifacts


def promote_release(manifest: dict[str, Any]) -> dict[str, Any]:
    """Promote all files as one rollback set after optimistic concurrency checks."""

    if any(
        manifest["steps"][step]["status"] != "completed"
        for step in PIPELINE_STEPS
    ):
        raise ValueError("Every staged refresh step must complete before promotion")
    all_live_keys = ("source", *PROMOTED_DATABASES, "auction_app", "snake_app")
    assert_production_cycle_unchanged(manifest)
    assert_pipeline_code_unchanged(manifest)
    assert_external_inputs_unchanged(manifest)
    assert_live_state_unchanged(manifest, all_live_keys)
    assert_staged_release_unchanged(manifest)
    validate_release(manifest)

    backup_dir = PRODUCTION_BACKUP_ROOT / str(manifest["run_id"])
    backup_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    try:
        for label, staged, destination in _promotion_sources(manifest):
            staged_state = database_state(staged)
            expected_live = manifest["baseline"][label]
            destination_existed = expected_live.get("exists", True) is not False
            durable_backup = None
            if destination_existed:
                durable_backup = backup_dir / f"{label}.pre_refresh.sqlite3"
                shutil.copy2(destination, durable_backup)
                backup_state = validate_sqlite(durable_backup)
                if (
                    backup_state["size_bytes"] != expected_live["size_bytes"]
                    or backup_state["sha256"] != expected_live["sha256"]
                ):
                    raise RuntimeError(
                        f"Durable pre-refresh backup differs for {label}"
                    )
            prepared = destination.with_name(
                f".{destination.name}.{manifest['run_id']}.release_stage"
            )
            if prepared.exists():
                prepared.unlink()
            shutil.copy2(staged, prepared)
            prepared_state = database_state(prepared)
            if prepared_state["sha256"] != staged_state["sha256"]:
                raise ValueError(f"Prepared promotion copy differs for {label}")
            artifacts.append(
                {
                    "label": label,
                    "staged": staged,
                    "destination": destination,
                    "prepared": prepared,
                    "staged_state": staged_state,
                    "durable_backup": durable_backup,
                    "destination_existed": destination_existed,
                }
            )
    except Exception:
        for artifact in artifacts:
            if artifact["prepared"].exists():
                artifact["prepared"].unlink()
        raise

    promoted: list[dict[str, Any]] = []
    try:
        assert_pipeline_code_unchanged(manifest)
        assert_external_inputs_unchanged(manifest)
        assert_live_state_unchanged(manifest, all_live_keys)
        assert_staged_release_unchanged(manifest)
        for artifact in artifacts:
            destination = artifact["destination"]
            assert_live_state_unchanged(
                manifest,
                (artifact["label"],),
            )
            temporary_backup = destination.with_name(
                f".{destination.name}.{manifest['run_id']}.rollback"
            )
            if temporary_backup.exists():
                temporary_backup.unlink()
            if artifact["destination_existed"]:
                os.replace(destination, temporary_backup)
            try:
                os.replace(artifact["prepared"], destination)
            except Exception:
                if temporary_backup.exists():
                    os.replace(temporary_backup, destination)
                raise
            artifact["temporary_backup"] = temporary_backup
            promoted.append(artifact)

        for artifact in promoted:
            actual = database_state(artifact["destination"])
            expected = artifact["staged_state"]
            if actual["sha256"] != expected["sha256"]:
                raise ValueError(
                    f"Promoted database differs for {artifact['label']}"
                )
    except Exception as promotion_error:
        rollback_errors = []
        for artifact in reversed(promoted):
            try:
                if artifact["temporary_backup"].exists():
                    os.replace(
                        artifact["temporary_backup"],
                        artifact["destination"],
                    )
                elif (
                    not artifact["destination_existed"]
                    and artifact["destination"].exists()
                ):
                    artifact["destination"].unlink()
            except Exception as rollback_error:  # pragma: no cover - OS failure
                rollback_errors.append(
                    f"{artifact['label']}: {rollback_error}"
                )
        if rollback_errors:
            raise RuntimeError(
                "Promotion failed and rollback was incomplete. Durable backups "
                f"remain in {backup_dir}: {rollback_errors}"
            ) from promotion_error
        raise
    finally:
        for artifact in artifacts:
            prepared = artifact["prepared"]
            if prepared.exists():
                prepared.unlink()

    cleanup_warnings = []
    for artifact in promoted:
        temporary_backup = artifact["temporary_backup"]
        if temporary_backup.exists():
            try:
                temporary_backup.unlink()
            except OSError as error:
                # Installation and post-install hash verification have already
                # succeeded.  A Windows handle can delay cleanup without
                # invalidating the release; retain the recoverable file and
                # record it rather than reporting a false promotion failure.
                cleanup_warnings.append(
                    {
                        "label": artifact["label"],
                        "path": str(temporary_backup),
                        "error": str(error),
                    }
                )
    return {
        "promoted_at_utc": utc_now(),
        "cleanup_warnings": cleanup_warnings,
        "artifacts": {
            artifact["label"]: {
                "destination": str(artifact["destination"]),
                "sha256": artifact["staged_state"]["sha256"],
                "size_bytes": artifact["staged_state"]["size_bytes"],
                "durable_backup": (
                    str(artifact["durable_backup"])
                    if artifact["durable_backup"] is not None
                    else None
                ),
            }
            for artifact in promoted
        },
    }


def run_pipeline(
    manifest: dict[str, Any],
    *,
    through: str,
) -> dict[str, Any]:
    if manifest.get("code_fingerprint"):
        assert_production_cycle_unchanged(manifest)
        assert_pipeline_code_unchanged(manifest)
    source_step = manifest["steps"]["snapshot"]["status"]
    if source_step == "completed":
        assert_live_state_unchanged(manifest, ("source",))
        if manifest.get("external_inputs"):
            assert_external_inputs_unchanged(manifest)
    final_index = PIPELINE_STEPS.index(through)
    for step in PIPELINE_STEPS[: final_index + 1]:
        state = manifest["steps"][step]
        if state["status"] == "completed":
            print(f"[{step}] already complete; skipping", flush=True)
            continue
        state.clear()
        state.update(
            {
                "status": "running",
                "started_at_utc": utc_now(),
            }
        )
        _save_manifest(manifest)
        try:
            result = execute_step(step, manifest)
        except BaseException as error:
            attempt_receipts = getattr(error, "attempt_receipts", None)
            state.update(
                {
                    "status": "failed",
                    "failed_at_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            if attempt_receipts is not None:
                state["attempts"] = attempt_receipts
            log_path = getattr(error, "log_path", None)
            if log_path is not None:
                state["log"] = log_path
            _save_manifest(manifest)
            raise
        state.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "result": result,
            }
        )
        _save_manifest(manifest)
    return manifest


def _default_stage_dir() -> Path:
    return (
        Path(tempfile.gettempdir())
        / "fantasy-football-production-refresh"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--stage-dir",
        type=Path,
        help="New staging directory (default: a unique local temp directory).",
    )
    location.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing staging directory from its manifest.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_PRODUCTION_YEAR,
        help=(
            "Registered current production season "
            f"(default: {DEFAULT_PRODUCTION_YEAR})."
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--selection-trials",
        type=int,
        default=DEFAULT_SELECTION_TRIALS,
    )
    parser.add_argument(
        "--selection-workers",
        type=int,
        default=DEFAULT_SELECTION_WORKERS,
    )
    parser.add_argument(
        "--app-timeout",
        type=int,
        default=DEFAULT_APP_TIMEOUT,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--app-python",
        type=Path,
        default=None,
        help=(
            "Python with Streamlit installed (default: the first working "
            "system/current interpreter)."
        ),
    )
    parser.add_argument(
        "--through",
        choices=PIPELINE_STEPS,
        default=PIPELINE_STEPS[-1],
        help="Stop after this staged step; useful for diagnostics.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Replace live model/app databases after every gate passes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered plan without creating or changing files.",
    )
    return parser.parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cycle = get_production_cycle(int(args.year))
    if args.dataset != DEFAULT_DATASET:
        raise ValueError(
            f"The production apps require dataset={DEFAULT_DATASET!r}"
        )
    for name in (
        "max_workers",
        "selection_trials",
        "selection_workers",
        "app_timeout",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")
    app_python_candidates = []
    if args.app_python is not None:
        app_python_candidates.append(args.app_python.expanduser().resolve())
    else:
        system_python = shutil.which("python")
        if system_python:
            app_python_candidates.append(Path(system_python).resolve())
        base_executable = getattr(sys, "_base_executable", None)
        if base_executable:
            app_python_candidates.append(Path(base_executable).resolve())
        app_python_candidates.append(python)
    app_python = None
    checked: list[str] = []
    for candidate in dict.fromkeys(app_python_candidates):
        if not candidate.is_file():
            checked.append(f"{candidate} (missing)")
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import streamlit"],
            env=_native_thread_limited_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        checked.append(f"{candidate} (exit {probe.returncode})")
        if probe.returncode == 0:
            app_python = candidate
            break
    if app_python is None:
        raise RuntimeError(
            "No Python interpreter with Streamlit is available for app smoke "
            f"tests. Checked: {checked}. Pass --app-python explicitly."
        )
    return {
        "year": int(args.year),
        "dataset": str(args.dataset),
        "max_workers": int(args.max_workers),
        "selection_trials": int(args.selection_trials),
        "selection_workers": int(args.selection_workers),
        "app_timeout": int(args.app_timeout),
        "python": str(python),
        "app_python": str(app_python),
        "production_leagues": list(cycle.leagues),
    }


def _dry_run_payload(
    *,
    stage_dir: Path,
    year: int,
    through: str,
    promote: bool,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cycle = get_production_cycle(int(year))
    final_index = PIPELINE_STEPS.index(through)
    completed = set()
    if manifest is not None:
        completed = {
            step
            for step, state in manifest["steps"].items()
            if state["status"] == "completed"
        }
    return {
        "stage_dir": str(stage_dir.resolve()),
        "steps": [
            {
                "step": step,
                "action": "skip_completed" if step in completed else "run",
            }
            for step in PIPELINE_STEPS[: final_index + 1]
        ],
        "promotion_requested": bool(promote),
        "production_cycle": production_cycle_receipt(
            cycle.current_season
        ),
        "live_writes_before_promotion": False,
        "manual_prerequisite": (
            "Finish Scripts/Data_Generation/1_Update_Projections.py and verify "
            "Season_Stats_New.sqlite3 before starting."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.promote and args.through != PIPELINE_STEPS[-1]:
        raise ValueError(
            "--promote requires the complete staged pipeline; omit --through "
            f"or use --through {PIPELINE_STEPS[-1]}"
        )

    if args.resume is not None:
        stage_dir = args.resume.expanduser().resolve()
        manifest = _load_manifest(stage_dir)
        assert_production_cycle_unchanged(manifest)
        assert_pipeline_code_unchanged(manifest)
        requested_options = _options_from_args(args)
        for key in ("year", "dataset"):
            if requested_options[key] != manifest["options"][key]:
                raise ValueError(
                    f"Resume {key} differs from the existing manifest"
                )
        # Runtime-only changes may make a retry practical without changing the
        # data/model contract.
        manifest["options"]["app_timeout"] = requested_options["app_timeout"]
    else:
        stage_dir = (
            args.stage_dir.expanduser().resolve()
            if args.stage_dir is not None
            else _default_stage_dir().resolve()
        )
        if stage_dir.exists() and any(stage_dir.iterdir()):
            raise FileExistsError(
                f"New staging directory must be empty: {stage_dir}"
            )
        manifest = _new_manifest(stage_dir, _options_from_args(args))

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(
                    stage_dir=stage_dir,
                    year=int(args.year),
                    through=args.through,
                    promote=args.promote,
                    manifest=manifest if args.resume is not None else None,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    stage_dir.mkdir(parents=True, exist_ok=True)
    _save_manifest(manifest)
    print(f"Refresh staging directory: {stage_dir}", flush=True)
    run_pipeline(manifest, through=args.through)
    assert_production_cycle_unchanged(manifest)
    assert_pipeline_code_unchanged(manifest)
    if manifest["steps"]["snapshot"]["status"] == "completed":
        assert_external_inputs_unchanged(manifest)

    if args.promote:
        manifest["promotion"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        _save_manifest(manifest)
        try:
            receipt = promote_release(manifest)
        except BaseException as error:
            manifest["promotion"] = {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _save_manifest(manifest)
            raise
        manifest["promotion"] = {
            "status": "completed",
            **receipt,
        }
        _save_manifest(manifest)
        print("Production refresh promoted successfully.", flush=True)
    else:
        print(
            "Staged refresh complete. Live databases were not changed. "
            f"Review {stage_dir} and resume with --promote when ready.",
            flush=True,
        )


if __name__ == "__main__":
    main()
