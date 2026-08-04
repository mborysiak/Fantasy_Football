from __future__ import annotations

import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from Scripts.V2 import refresh_production as refresh


def _database(path: Path, statements: list[str]) -> None:
    with closing(sqlite3.connect(path)) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()


def _write_source_market_database(
    path: Path,
    *,
    nffc_feed_counts: dict[str, int],
) -> None:
    statements = [
        "CREATE TABLE ADP_Averages (year INTEGER, league TEXT)",
        """
        WITH RECURSIVE sequence(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1 FROM sequence WHERE value < 300
        )
        INSERT INTO ADP_Averages
        SELECT 2026, 'dk' FROM sequence
        """,
        "CREATE TABLE NFFC_ADP (year INTEGER, source TEXT)",
        "CREATE TABLE ETR_Ranks (year INTEGER)",
        """
        WITH RECURSIVE sequence(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1 FROM sequence WHERE value < 180
        )
        INSERT INTO ETR_Ranks
        SELECT 2026 FROM sequence
        """,
    ]
    for label, count in nffc_feed_counts.items():
        if count <= 0:
            continue
        statements.append(
            f"""
            WITH RECURSIVE sequence(value) AS (
                SELECT 1
                UNION ALL
                SELECT value + 1 FROM sequence WHERE value < {int(count)}
            )
            INSERT INTO NFFC_ADP
            SELECT 2026, '{label}' FROM sequence
            """
        )
    _database(path, statements)


def test_release_plan_covers_every_downstream_surface():
    assert refresh.PIPELINE_STEPS == (
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
    assert len(refresh.GENERATED_AUCTION_TABLES) == 20
    assert len(set(refresh.GENERATED_AUCTION_TABLES)) == 20
    assert "V2_Projection_Legacy_Backup" in refresh.GENERATED_AUCTION_TABLES
    assert "Salary_Selection_Premium" in refresh.GENERATED_AUCTION_TABLES
    assert refresh.MANIFEST_SCHEMA_VERSION == 5
    assert refresh.DATABASE_FILES["parameter_cache"] == (
        "V2_Parameter_Cache.sqlite3"
    )
    assert "parameter_cache" in refresh.PROMOTED_DATABASES
    assert "parameter_cache" in refresh.BOOTSTRAPPABLE_DATABASES
    assert refresh.GITHUB_BLOB_LIMIT_BYTES == 100 * 1024 * 1024


def test_refresh_fingerprints_every_governed_salary_export():
    inputs = refresh.external_file_inputs(2026)

    assert {
        key: path.name
        for key, path in inputs.items()
        if "auction_salaries" in key
    } == {
        "historical_auction_salaries_2025_beta": "salaries_2025_beta.csv",
        "historical_auction_salaries_2025_nv": "salaries_2025_nv.csv",
        "current_auction_salaries": "salaries_2026_beta.csv",
    }
    for key in (
        "historical_auction_salaries_2025_beta",
        "historical_auction_salaries_2025_nv",
        "current_auction_salaries",
    ):
        state = refresh.regular_file_state(inputs[key])
        assert state["sha256"]
        assert state["size_bytes"] > 0


def test_vacuum_sqlite_reclaims_pages_without_changing_content(tmp_path):
    database = tmp_path / "compact.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE TABLE values_table (value INTEGER, payload BLOB)"
        )
        connection.executemany(
            "INSERT INTO values_table VALUES (?, ?)",
            [(index, b"x" * 8192) for index in range(200)],
        )
        connection.execute("DELETE FROM values_table WHERE value > 0")
        connection.commit()
    before_digest = refresh.stable_table_digest(database, "values_table")

    receipt = refresh.vacuum_sqlite(database)

    assert receipt["reclaimed_bytes"] > 0
    assert receipt["after"]["size_bytes"] < receipt["before"]["size_bytes"]
    assert receipt["after_pages"]["freelist_count"] == 0
    assert refresh.stable_table_digest(database, "values_table") == before_digest


def test_app_artifact_size_gate_rejects_oversized_database(tmp_path):
    database = tmp_path / "oversized.sqlite3"
    _database(
        database,
        [
            "CREATE TABLE values_table (value INTEGER)",
            "INSERT INTO values_table VALUES (1)",
        ],
    )

    accepted = refresh.validate_app_artifact_size(
        database,
        app="snake",
        limit_bytes=database.stat().st_size,
    )
    assert accepted["within_limit"] is True

    with pytest.raises(ValueError, match="after VACUUM"):
        refresh.validate_app_artifact_size(
            database,
            app="snake",
            limit_bytes=database.stat().st_size - 1,
        )


def test_prepare_apps_vacuums_both_candidates(tmp_path, monkeypatch):
    simulation = tmp_path / "staged" / "Simulation.sqlite3"
    auction_base = tmp_path / "bases" / "Auction.sqlite3"
    auction_artifact = tmp_path / "artifacts" / "Auction.sqlite3"
    snake_artifact = tmp_path / "artifacts" / "Snake.sqlite3"
    simulation.parent.mkdir(parents=True)
    auction_base.parent.mkdir(parents=True)
    with closing(sqlite3.connect(simulation)) as connection:
        connection.execute("CREATE TABLE generated (value INTEGER, payload BLOB)")
        connection.executemany(
            "INSERT INTO generated VALUES (?, ?)",
            [(index, b"x" * 8192) for index in range(200)],
        )
        connection.execute("DELETE FROM generated WHERE value > 0")
        connection.commit()
    _database(
        auction_base,
        [
            "CREATE TABLE generated (value INTEGER, payload BLOB)",
            "INSERT INTO generated VALUES (999, X'01')",
            "CREATE TABLE app_owned (value TEXT)",
            "INSERT INTO app_owned VALUES ('preserve')",
        ],
    )
    paths = {
        "staged": {"simulation": simulation},
        "app_bases": {"auction": auction_base},
        "app_artifacts": {
            "auction": auction_artifact,
            "snake": snake_artifact,
        },
    }
    monkeypatch.setattr(refresh, "_resolved_paths", lambda _manifest: paths)
    monkeypatch.setattr(refresh, "GENERATED_AUCTION_TABLES", ("generated",))

    result = refresh.step_prepare_apps({})

    assert result["auction"]["compaction"]["after_pages"]["freelist_count"] == 0
    assert result["snake"]["compaction"]["reclaimed_bytes"] > 0
    assert result["snake"]["compaction"]["after_pages"]["freelist_count"] == 0
    assert result["auction"]["github_size_gate"]["within_limit"] is True
    assert result["snake"]["github_size_gate"]["within_limit"] is True
    assert refresh.table_digests(simulation, ["generated"]) == refresh.table_digests(
        snake_artifact,
        ["generated"],
    )
    assert refresh.table_digests(auction_base, ["app_owned"]) == refresh.table_digests(
        auction_artifact,
        ["app_owned"],
    )


def test_source_market_gate_returns_each_governed_nffc_feed(tmp_path):
    database = tmp_path / "source.sqlite3"
    expected_feeds = {
        "nffc_rotowire_online": 400,
        "nffc_best_ball_overall": 400,
        "nffc_best_ball_25s50s": 400,
        "nffc_cutline": 250,
    }
    _write_source_market_database(
        database,
        nffc_feed_counts=expected_feeds,
    )

    result = refresh._validate_source_markets(database, 2026)

    assert result == {
        "dk": 300,
        "nffc": 1450,
        "etr": 180,
        "nffc_feed_counts": dict(sorted(expected_feeds.items())),
    }


@pytest.mark.parametrize(
    "feed_counts, expected_fragment",
    [
        (
            {
                "nffc_rotowire_online": 400,
                "nffc_best_ball_overall": 400,
                "nffc_best_ball_25s50s": 400,
            },
            "missing=['nffc_cutline']",
        ),
        (
            {
                "nffc_rotowire_online": 400,
                "nffc_best_ball_overall": 400,
                "nffc_best_ball_25s50s": 400,
                "nffc_cutline": 250,
                "nffc_unreviewed_feed": 1,
            },
            "unexpected=['nffc_unreviewed_feed']",
        ),
    ],
)
def test_source_market_gate_requires_exact_nffc_feed_labels(
    tmp_path,
    feed_counts,
    expected_fragment,
):
    database = tmp_path / "source.sqlite3"
    _write_source_market_database(
        database,
        nffc_feed_counts=feed_counts,
    )

    with pytest.raises(ValueError, match="do not match the annual contract") as error:
        refresh._validate_source_markets(database, 2026)

    assert expected_fragment in str(error.value)


def test_source_market_gate_enforces_each_nffc_feed_floor(tmp_path):
    database = tmp_path / "source.sqlite3"
    _write_source_market_database(
        database,
        nffc_feed_counts={
            "nffc_rotowire_online": 399,
            "nffc_best_ball_overall": 400,
            "nffc_best_ball_25s50s": 400,
            "nffc_cutline": 250,
        },
    )

    with pytest.raises(
        ValueError,
        match="nffc_rotowire_online has only 399 rows",
    ):
        refresh._validate_source_markets(database, 2026)


def test_snapshot_validates_source_markets_before_modeling(
    monkeypatch,
    tmp_path,
):
    live = {
        key: tmp_path / "live" / filename
        for key, filename in refresh.DATABASE_FILES.items()
    }
    live.update(
        {
            "auction_app": tmp_path / "live" / "auction.sqlite3",
            "snake_app": tmp_path / "live" / "snake.sqlite3",
        }
    )
    staged = {
        key: tmp_path / "stage" / "databases" / filename
        for key, filename in refresh.DATABASE_FILES.items()
    }
    paths = {
        "live": live,
        "staged": staged,
        "model_input_bases": {
            key: tmp_path / "stage" / "model_input_bases" / filename
            for key, filename in refresh.DATABASE_FILES.items()
            if key in refresh.MODEL_INPUT_BASE_KEYS
        },
        "app_bases": {
            "auction": tmp_path / "stage" / "auction_base.sqlite3",
            "snake": tmp_path / "stage" / "snake_base.sqlite3",
        },
        "app_artifacts": {},
    }
    for path in live.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_backup(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
        return {
            "source": {"path": str(source), "exists": True},
            "staged": {"path": str(destination), "integrity": "ok"},
        }

    calls = []

    def fake_validate(path, year):
        calls.append((path, year))
        return {"nffc_feed_counts": {"nffc_cutline": 250}}

    monkeypatch.setattr(refresh, "_resolved_paths", lambda _manifest: paths)
    monkeypatch.setattr(refresh, "sqlite_backup", fake_backup)
    monkeypatch.setattr(
        refresh,
        "atomic_validated_sqlite_copy",
        fake_backup,
    )
    monkeypatch.setattr(refresh, "EXTERNAL_SQLITE_INPUTS", {})
    monkeypatch.setattr(refresh, "external_file_inputs", lambda _year: {})
    monkeypatch.setattr(refresh, "_validate_source_markets", fake_validate)
    manifest = {
        "stage_dir": str(tmp_path / "stage"),
        "options": {"year": 2026},
        "baseline": {},
    }

    result = refresh.step_snapshot(manifest)

    assert calls == [(staged["source"], 2026)]
    assert result["source_market_counts"] == {
        "nffc_feed_counts": {"nffc_cutline": 250}
    }
    assert set(result["model_input_bases"]) == set(
        refresh.MODEL_INPUT_BASE_KEYS
    )


def test_path_map_separates_model_input_retry_bases(tmp_path):
    paths = refresh._path_map(tmp_path)

    for key in refresh.MODEL_INPUT_BASE_KEYS:
        retry_base = Path(paths["model_input_bases"][key])
        assert retry_base.parent == (tmp_path / "model_input_bases").resolve()
        assert retry_base != Path(paths["staged"][key])
        assert retry_base != Path(paths["live"][key])


def test_future_year_fails_closed_until_model_rollover():
    args = refresh.parse_args(["--dry-run", "--year", "2027"])
    with pytest.raises(ValueError, match="not an approved production cycle"):
        refresh._options_from_args(args)


def test_subprocess_environment_propagates_cycle_year_and_keeper_input(
    monkeypatch,
    tmp_path,
):
    for variable in refresh.NATIVE_THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(variable, "99")
    staged = {
        "simulation": tmp_path / "Simulation.sqlite3",
        "v2_beta": tmp_path / "Projection_V2_beta.sqlite3",
    }

    environment = refresh._subprocess_environment(staged, year=2026)

    assert environment["FF_CURRENT_SEASON"] == "2026"
    assert environment["FF_MODEL_DATABASE_DIR"] == str(tmp_path)
    assert environment["FF_V2_BETA_DATABASE"] == str(staged["v2_beta"])
    assert environment["FF_KEEPERS_FILE"].endswith(
        "keepers_2026_beta.csv"
    )
    assert {
        variable: environment[variable]
        for variable in refresh.NATIVE_THREAD_ENVIRONMENT_VARIABLES
    } == {
        variable: "1"
        for variable in refresh.NATIVE_THREAD_ENVIRONMENT_VARIABLES
    }
    assert environment["PYTHONFAULTHANDLER"] == "1"


class _FakeLoggedProcess:
    def __init__(self, return_code: int, output: str = "") -> None:
        self.return_code = return_code
        self.stdout = iter(output.splitlines(keepends=True))

    def wait(self, timeout=None):
        return self.return_code

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def _install_fake_logged_processes(
    monkeypatch,
    return_codes: list[int],
) -> list[dict]:
    calls: list[dict] = []
    pending_codes = iter(return_codes)

    def fake_popen(command, **kwargs):
        return_code = next(pending_codes)
        calls.append(
            {
                "command": command,
                "environment": kwargs["env"],
                "return_code": return_code,
            }
        )
        return _FakeLoggedProcess(
            return_code,
            output=f"native process returned {return_code}\n",
        )

    monkeypatch.setattr(refresh.subprocess, "Popen", fake_popen)
    return calls


def test_logged_command_caps_native_threads_and_retries_access_violation(
    monkeypatch,
    tmp_path,
):
    calls = _install_fake_logged_processes(
        monkeypatch,
        [3221225477, 0],
    )
    preparations: list[int] = []

    def prepare_attempt(attempt: int) -> dict[str, int]:
        preparations.append(attempt)
        return {"prepared_attempt": attempt}

    receipt = refresh.run_logged_command(
        ["python", "model.py"],
        step="locked_nffc",
        stage_dir=tmp_path,
        environment={
            variable: "8"
            for variable in refresh.NATIVE_THREAD_ENVIRONMENT_VARIABLES
        },
        before_attempt=prepare_attempt,
    )

    assert len(calls) == 2
    assert preparations == [1, 2]
    for call in calls:
        assert {
            variable: call["environment"][variable]
            for variable in refresh.NATIVE_THREAD_ENVIRONMENT_VARIABLES
        } == {
            variable: "1"
            for variable in refresh.NATIVE_THREAD_ENVIRONMENT_VARIABLES
        }
    assert [
        attempt["outcome"]
        for attempt in receipt["attempts"]
    ] == ["retryable_native_failure", "completed"]
    assert [
        attempt["return_code"]
        for attempt in receipt["attempts"]
    ] == [3221225477, 0]
    assert [
        attempt["preparation"]["prepared_attempt"]
        for attempt in receipt["attempts"]
    ] == [1, 2]
    log_text = Path(receipt["log"]).read_text(encoding="utf-8")
    assert (
        f"attempt 1/{refresh.MAX_NATIVE_CRASH_ATTEMPTS} "
        "exited 3221225477"
    ) in log_text
    assert (
        "Windows access violation on attempt "
        f"1/{refresh.MAX_NATIVE_CRASH_ATTEMPTS}; retrying"
    ) in log_text
    assert (
        f"attempt 2/{refresh.MAX_NATIVE_CRASH_ATTEMPTS} exited 0"
        in log_text
    )


@pytest.mark.parametrize("return_code", (3221226356, -1073740940))
def test_logged_command_retries_windows_heap_corruption(
    monkeypatch,
    tmp_path,
    return_code,
):
    calls = _install_fake_logged_processes(
        monkeypatch,
        [return_code, 0],
    )

    receipt = refresh.run_logged_command(
        ["python", "model.py"],
        step="v2_nffc",
        stage_dir=tmp_path,
    )

    assert len(calls) == 2
    assert [
        attempt["failure_class"]
        for attempt in receipt["attempts"]
        if "failure_class" in attempt
    ] == ["windows_heap_corruption"]
    log_text = Path(receipt["log"]).read_text(encoding="utf-8")
    assert (
        "Windows heap corruption on attempt "
        f"1/{refresh.MAX_NATIVE_CRASH_ATTEMPTS}; retrying"
    ) in log_text


def test_v2_step_isolates_foundation_from_feature_mart(monkeypatch, tmp_path):
    calls = []
    staged = {
        "source": tmp_path / "Season_Stats_New.sqlite3",
        "v2_dk": tmp_path / "Projection_V2.sqlite3",
    }

    monkeypatch.setattr(
        refresh,
        "_resolved_paths",
        lambda _manifest: {"staged": staged},
    )
    monkeypatch.setattr(refresh, "_python", lambda _manifest: "python")
    monkeypatch.setattr(
        refresh,
        "_subprocess_environment",
        lambda _paths, year: {"FF_CURRENT_SEASON": str(year)},
    )

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return {"command": list(command), "step": kwargs["step"]}

    monkeypatch.setattr(refresh, "run_logged_command", fake_run)
    result = refresh.step_v2(
        {
            "stage_dir": str(tmp_path),
            "options": {"year": 2026, "max_workers": 6},
        },
        "dk",
    )

    assert result["process_isolation"] == (
        "milestone_2_then_milestone_3_reuse"
    )
    assert len(calls) == 2
    assert calls[0][0][1:3] == ["-m", "Scripts.V2.build_milestone_2"]
    assert calls[0][1]["step"] == "v2_dk_foundation"
    assert calls[1][0][1:3] == ["-m", "Scripts.V2.build_milestone_3"]
    assert calls[1][0][-1] == "--reuse-foundation"
    assert calls[1][1]["step"] == "v2_dk"


def test_logged_command_exhausts_native_retry_and_records_manifest_attempts(
    monkeypatch,
    tmp_path,
):
    calls = _install_fake_logged_processes(
        monkeypatch,
        [-1073741819] * refresh.MAX_NATIVE_CRASH_ATTEMPTS,
    )
    manifest = {
        "stage_dir": str(tmp_path),
        "steps": {
            step: {"status": "pending"}
            for step in refresh.PIPELINE_STEPS
        },
    }
    monkeypatch.setattr(refresh, "_save_manifest", lambda _manifest: None)
    monkeypatch.setattr(
        refresh,
        "execute_step",
        lambda step, _manifest: refresh.run_logged_command(
            ["python", "model.py"],
            step=step,
            stage_dir=tmp_path,
        ),
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        refresh.run_pipeline(manifest, through="snapshot")

    assert error.value.returncode == -1073741819
    assert len(calls) == refresh.MAX_NATIVE_CRASH_ATTEMPTS
    attempts = manifest["steps"]["snapshot"]["attempts"]
    assert len(attempts) == refresh.MAX_NATIVE_CRASH_ATTEMPTS
    assert [
        attempt["outcome"]
        for attempt in attempts
    ] == [
        *(
            ["retryable_native_failure"]
            * (refresh.MAX_NATIVE_CRASH_ATTEMPTS - 1)
        ),
        "native_failure_exhausted",
    ]
    assert manifest["steps"]["snapshot"]["status"] == "failed"
    assert Path(manifest["steps"]["snapshot"]["log"]) == (
        tmp_path / "logs" / "snapshot.log"
    )


def test_logged_command_does_not_retry_ordinary_failure(
    monkeypatch,
    tmp_path,
):
    calls = _install_fake_logged_processes(monkeypatch, [1])

    with pytest.raises(subprocess.CalledProcessError) as error:
        refresh.run_logged_command(
            ["python", "model.py"],
            step="locked_nffc",
            stage_dir=tmp_path,
        )

    assert error.value.returncode == 1
    assert len(calls) == 1
    assert len(error.value.attempt_receipts) == 1
    receipt = error.value.attempt_receipts[0]
    assert receipt["attempt"] == 1
    assert receipt["return_code"] == 1
    assert receipt["outcome"] == "failed"
    assert receipt["will_retry"] is False
    assert "failure_class" not in receipt


def test_sqlite_backup_closes_handles_and_preserves_content(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    _database(
        source,
        [
            "CREATE TABLE rows (row_id INTEGER PRIMARY KEY, value TEXT)",
            "INSERT INTO rows VALUES (1, 'one')",
            "INSERT INTO rows VALUES (2, 'two')",
        ],
    )

    receipt = refresh.sqlite_backup(source, destination)

    assert receipt["source"]["integrity"] == "ok"
    assert receipt["staged"]["integrity"] == "ok"
    assert destination.is_file()
    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute(
            "SELECT * FROM rows ORDER BY row_id"
        ).fetchall() == [(1, "one"), (2, "two")]
    # Windows will fail this replacement if validation leaked a connection.
    replacement = tmp_path / "replacement.sqlite3"
    destination.replace(replacement)
    assert replacement.is_file()


def test_atomic_sqlite_copy_is_exact_and_removes_stale_sidecars(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    _database(
        source,
        [
            "CREATE TABLE rows (row_id INTEGER PRIMARY KEY, value TEXT)",
            "INSERT INTO rows VALUES (1, 'source')",
        ],
    )
    _database(
        destination,
        [
            "CREATE TABLE rows (row_id INTEGER PRIMARY KEY, value TEXT)",
            "INSERT INTO rows VALUES (1, 'old')",
        ],
    )
    sidecars = [Path(f"{destination}{suffix}") for suffix in ("-wal", "-shm", "-journal")]
    for sidecar in sidecars:
        sidecar.write_bytes(b"stale")

    receipt = refresh.atomic_validated_sqlite_copy(source, destination)

    assert receipt["source"]["sha256"] == receipt["staged"]["sha256"]
    assert refresh.sha256_file(source) == refresh.sha256_file(destination)
    assert not any(sidecar.exists() for sidecar in sidecars)
    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute("SELECT * FROM rows").fetchall() == [
            (1, "source")
        ]


def _write_model_input_retry_base(path: Path, marker: str) -> None:
    statements = [
        "CREATE TABLE marker (value TEXT)",
        f"INSERT INTO marker VALUES ('{marker}')",
    ]
    statements.extend(
        f"CREATE TABLE {position}_2026_ProjOnly (player TEXT)"
        for position in ("QB", "RB", "WR", "TE")
    )
    _database(path, statements)


def test_model_inputs_retry_restores_both_databases_before_each_attempt(
    monkeypatch,
    tmp_path,
):
    staged = {
        "model_inputs": tmp_path / "databases" / "Model_Inputs.sqlite3",
        "model_inputs_next": (
            tmp_path / "databases" / "Model_Inputs_next.sqlite3"
        ),
        "simulation": tmp_path / "databases" / "Simulation.sqlite3",
        "v2_beta": tmp_path / "databases" / "Projection_V2_beta.sqlite3",
    }
    retry_bases = {
        "model_inputs": tmp_path / "bases" / "Model_Inputs.sqlite3",
        "model_inputs_next": tmp_path / "bases" / "Model_Inputs_next.sqlite3",
    }
    for key in refresh.MODEL_INPUT_BASE_KEYS:
        retry_bases[key].parent.mkdir(parents=True, exist_ok=True)
        _write_model_input_retry_base(retry_bases[key], f"base-{key}")
        refresh.atomic_validated_sqlite_copy(retry_bases[key], staged[key])
        with closing(sqlite3.connect(staged[key])) as connection:
            connection.execute("UPDATE marker SET value='dirty-before-run'")
            connection.commit()

    paths = {
        "staged": staged,
        "model_input_bases": retry_bases,
    }
    base_states = {
        key: refresh.database_state(path)
        for key, path in retry_bases.items()
    }
    manifest = {
        "stage_dir": str(tmp_path),
        "options": {
            "year": 2026,
            "python": "python",
        },
        "steps": {
            "snapshot": {
                "status": "completed",
                "result": {"model_input_bases": base_states},
            }
        },
    }
    monkeypatch.setattr(refresh, "_resolved_paths", lambda _manifest: paths)
    return_codes = iter([3221225477, 0])
    observed: list[dict[str, str]] = []

    def fake_popen(command, **kwargs):
        values = {}
        for key in refresh.MODEL_INPUT_BASE_KEYS:
            with closing(sqlite3.connect(staged[key])) as connection:
                values[key] = connection.execute(
                    "SELECT value FROM marker"
                ).fetchone()[0]
        observed.append(values)
        if len(observed) == 1:
            for key in refresh.MODEL_INPUT_BASE_KEYS:
                with closing(sqlite3.connect(staged[key])) as connection:
                    connection.execute(
                        "UPDATE marker SET value='partial-after-crash'"
                    )
                    connection.commit()
        return _FakeLoggedProcess(next(return_codes))

    monkeypatch.setattr(refresh.subprocess, "Popen", fake_popen)

    result = refresh.step_model_inputs(manifest)

    expected = {
        key: f"base-{key}"
        for key in refresh.MODEL_INPUT_BASE_KEYS
    }
    assert observed == [expected, expected]
    assert [
        attempt["outcome"]
        for attempt in result["attempts"]
    ] == ["retryable_native_failure", "completed"]
    assert all("preparation" in attempt for attempt in result["attempts"])
    assert {
        key: refresh.database_state(path)["sha256"]
        for key, path in retry_bases.items()
    } == {
        key: state["sha256"]
        for key, state in base_states.items()
    }


def _write_identity_database(path: Path, *, alias_name: str = "player") -> None:
    _database(
        path,
        [
            """
            CREATE TABLE player_identity (
                player_key TEXT,
                gsis_id TEXT,
                display_name TEXT
            )
            """,
            "INSERT INTO player_identity VALUES ('key-1', 'gsis-1', 'Player')",
            """
            CREATE TABLE player_aliases (
                player_key TEXT,
                source TEXT,
                source_name TEXT
            )
            """,
            (
                "INSERT INTO player_aliases VALUES "
                f"('key-1', 'provider', '{alias_name}')"
            ),
            """
            CREATE TABLE player_season_spine (
                player_key TEXT,
                season INTEGER,
                gsis_id TEXT,
                display_name TEXT,
                position TEXT,
                team TEXT
            )
            """,
            (
                "INSERT INTO player_season_spine VALUES "
                "('key-1', 2026, 'gsis-1', 'Player', 'WR', 'DET')"
            ),
            """
            CREATE TABLE player_season_features (
                player_key TEXT,
                season INTEGER
            )
            """,
            "INSERT INTO player_season_features VALUES ('key-1', 2026)",
        ],
    )


def test_cross_league_identity_gate_accepts_equal_builds(tmp_path):
    dk = tmp_path / "dk.sqlite3"
    beta = tmp_path / "beta.sqlite3"
    _write_identity_database(dk)
    _write_identity_database(beta)

    result = refresh._assert_cross_league_identity(dk, beta)

    assert result == {
        "player_identity": 1,
        "player_aliases": 1,
        "player_season_spine": 1,
        "player_season_features": 1,
    }


def test_cross_league_identity_gate_rejects_alias_drift(tmp_path):
    dk = tmp_path / "dk.sqlite3"
    beta = tmp_path / "beta.sqlite3"
    _write_identity_database(dk)
    _write_identity_database(beta, alias_name="different")

    with pytest.raises(ValueError, match="player_aliases drifted"):
        refresh._assert_cross_league_identity(dk, beta)


def _write_remote_source_database(
    path: Path,
    *,
    checksum: str,
) -> None:
    _database(
        path,
        [
            """
            CREATE TABLE build_runs (
                run_id TEXT,
                component TEXT,
                foundation_run_id TEXT,
                status TEXT
            )
            """,
            """
            INSERT INTO build_runs VALUES
                ('m1', 'milestone_1', 'm1', 'complete'),
                ('m2', 'milestone_2', 'm1', 'complete'),
                ('m3', 'milestone_3', 'm2', 'complete')
            """,
            """
            CREATE TABLE source_manifest (
                run_id TEXT,
                component TEXT,
                source_name TEXT,
                source_uri TEXT,
                source_sha256 TEXT,
                row_count INTEGER
            )
            """,
            (
                "INSERT INTO source_manifest VALUES "
                "('m1', 'identity', 'remote_players', "
                f"'https://example.test/players.csv', '{checksum}', 100)"
            ),
        ],
    )


def test_cross_league_remote_source_gate_requires_equal_payloads(tmp_path):
    dk = tmp_path / "dk.sqlite3"
    beta = tmp_path / "beta.sqlite3"
    _write_remote_source_database(dk, checksum="same")
    _write_remote_source_database(beta, checksum="same")

    result = refresh._validate_cross_league_remote_sources(dk, beta)

    assert result == {
        "receipt_count": 1,
        "foundation_run_ids": {"dk": "m1", "beta": "m1"},
    }


def test_cross_league_remote_source_gate_rejects_payload_drift(tmp_path):
    dk = tmp_path / "dk.sqlite3"
    beta = tmp_path / "beta.sqlite3"
    _write_remote_source_database(dk, checksum="dk")
    _write_remote_source_database(beta, checksum="beta")

    with pytest.raises(ValueError, match="different remote source payloads"):
        refresh._validate_cross_league_remote_sources(dk, beta)


def test_generated_table_sync_preserves_app_owned_tables(tmp_path):
    source = tmp_path / "source.sqlite3"
    app = tmp_path / "app.sqlite3"
    _database(
        source,
        [
            "CREATE TABLE generated (row_id INTEGER, value TEXT)",
            "INSERT INTO generated VALUES (1, 'new')",
        ],
    )
    _database(
        app,
        [
            "CREATE TABLE generated (row_id INTEGER, value TEXT)",
            "INSERT INTO generated VALUES (1, 'old')",
            "CREATE TABLE app_owned (setting TEXT)",
            "INSERT INTO app_owned VALUES ('keep-me')",
        ],
    )

    counts = refresh.synchronize_sqlite_tables(
        source,
        app,
        ("generated",),
    )

    assert counts == {"generated": 1}
    with closing(sqlite3.connect(app)) as connection:
        assert connection.execute("SELECT * FROM generated").fetchall() == [
            (1, "new")
        ]
        assert connection.execute("SELECT * FROM app_owned").fetchall() == [
            ("keep-me",)
        ]


def test_resume_skips_completed_steps(monkeypatch, tmp_path):
    manifest = {
        "stage_dir": str(tmp_path),
        "steps": {
            step: {"status": "pending"}
            for step in refresh.PIPELINE_STEPS
        },
    }
    calls: list[str] = []

    monkeypatch.setattr(refresh, "_save_manifest", lambda _manifest: None)
    monkeypatch.setattr(
        refresh,
        "assert_live_state_unchanged",
        lambda _manifest, _keys: None,
    )
    monkeypatch.setattr(
        refresh,
        "execute_step",
        lambda step, _manifest: calls.append(step) or {"step": step},
    )

    refresh.run_pipeline(manifest, through="model_inputs")
    refresh.run_pipeline(manifest, through="model_inputs")

    assert calls == ["snapshot", "model_inputs"]
    assert manifest["steps"]["snapshot"]["status"] == "completed"
    assert manifest["steps"]["model_inputs"]["status"] == "completed"


def test_source_change_blocks_resume(tmp_path):
    source = tmp_path / "source.sqlite3"
    _database(
        source,
        [
            "CREATE TABLE values_table (value INTEGER)",
            "INSERT INTO values_table VALUES (1)",
        ],
    )
    baseline = refresh.database_state(source)
    manifest = {
        "paths": {"live": {"source": str(source)}},
        "baseline": {"source": baseline},
    }
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("INSERT INTO values_table VALUES (2)")
        connection.commit()

    with pytest.raises(RuntimeError, match="changed after staging"):
        refresh.assert_live_state_unchanged(manifest, ("source",))


def test_pipeline_code_change_blocks_resume(monkeypatch):
    manifest = {
        "code_fingerprint": {
            "sha256": "before",
            "file_count": 1,
            "files": {
                "main_scripts/example.py": {
                    "size_bytes": 1,
                    "sha256": "before",
                }
            },
        }
    }
    monkeypatch.setattr(
        refresh,
        "pipeline_code_fingerprint",
        lambda: {
            "sha256": "after",
            "file_count": 1,
            "files": {
                "main_scripts/example.py": {
                    "size_bytes": 2,
                    "sha256": "after",
                }
            },
        },
    )

    with pytest.raises(RuntimeError, match="code changed"):
        refresh.assert_pipeline_code_unchanged(manifest)


def test_external_input_change_blocks_resume(tmp_path):
    weekly = tmp_path / "weekly.sqlite3"
    salaries = tmp_path / "salaries.csv"
    _database(
        weekly,
        [
            "CREATE TABLE values_table (value INTEGER)",
            "INSERT INTO values_table VALUES (1)",
        ],
    )
    salaries.write_text("player,salary\nOne,1\n", encoding="utf-8")
    manifest = {
        "external_inputs": {
            "sqlite": {
                "weekly_history": refresh.database_state(weekly),
            },
            "files": {
                "current_auction_salaries": refresh.regular_file_state(
                    salaries
                ),
            },
        }
    }
    refresh.assert_external_inputs_unchanged(manifest)
    salaries.write_text(
        "player,salary\nOne,1\nTwo,2\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="changed after staging"):
        refresh.assert_external_inputs_unchanged(manifest)


def test_model_input_population_floor_rejects_truncation(tmp_path):
    database = tmp_path / "model_inputs.sqlite3"
    _database(
        database,
        [
            """
            CREATE TABLE QB_2026_ProjOnly (
                player TEXT,
                year INTEGER
            )
            """,
            "INSERT INTO QB_2026_ProjOnly VALUES ('Only Player', 2026)",
        ],
    )

    with pytest.raises(ValueError, match="release floor"):
        refresh._validate_model_inputs(database, 2026)


def _write_model_acceptance_database(
    path: Path,
    *,
    current_upper: float = -0.01,
) -> None:
    _database(
        path,
        [
            """
            CREATE TABLE locked_model_comparisons (
                comparison TEXT,
                seasons INTEGER,
                mean_season_rmse_delta REAL,
                recent_mean_delta REAL,
                bootstrap_95_upper REAL,
                season_wins INTEGER
            )
            """,
            (
                "INSERT INTO locked_model_comparisons VALUES "
                f"('primary_vs_expert', 9, -0.08, -0.07, "
                f"{current_upper}, 9)"
            ),
            """
            CREATE TABLE next_year_model_comparisons (
                comparison TEXT,
                metric TEXT,
                origins INTEGER,
                mean_origin_delta REAL,
                recent_mean_delta REAL,
                bootstrap_95_upper REAL,
                origin_wins INTEGER
            )
            """,
            """
            INSERT INTO next_year_model_comparisons VALUES
                ('primary_vs_expert_carry', 'rmse', 8, -1.0, -0.8, -0.5, 8),
                ('participation_lgbm_vs_prior', 'brier', 8, -0.1, -0.08, -0.05, 8)
            """,
        ],
    )


def test_model_acceptance_requires_supported_baseline_improvement(tmp_path):
    accepted = tmp_path / "accepted.sqlite3"
    rejected = tmp_path / "rejected.sqlite3"
    _write_model_acceptance_database(accepted)
    _write_model_acceptance_database(rejected, current_upper=0.01)

    receipt = refresh._validate_model_acceptance(
        accepted,
        league="nffc",
    )
    assert set(receipt) == {
        "current_ppg_vs_expert",
        "next_ppg_vs_expert_carry",
        "next_participation_vs_prior",
    }
    with pytest.raises(ValueError, match="did not clear the locked baseline"):
        refresh._validate_model_acceptance(
            rejected,
            league="nffc",
        )


@pytest.mark.parametrize(
    "nullable_column",
    [
        "historical_projection_source",
        "historical_center_policy",
        "projection_context_source",
        "projection_context_scoring_hash",
        "projection_context_run_id",
        "model_input_avg_proj_points",
        "projection_context_avg_proj_points_delta",
        "avg_proj_points",
        "historical_pred_fp_per_game",
        "avg_proj_pass_points",
        "avg_proj_rush_points",
        "avg_proj_rec_points",
        "v2_recenter_promoted",
    ],
)
def test_nffc_template_context_rejects_null_required_fields(
    nullable_column,
):
    cycle = refresh.get_production_cycle(2026)
    expected_center_policy = cycle.template_center_policies["nffc"][0]
    expected_context_source = cycle.template_context_sources["nffc"]
    expected_scoring_hash = refresh.scoring_hash("nffc")
    columns = [
        "league",
        "historical_projection_source",
        "historical_center_policy",
        "projection_context_source",
        "projection_context_scoring_hash",
        "projection_context_run_id",
        "model_input_avg_proj_points",
        "projection_context_avg_proj_points_delta",
        "avg_proj_points",
        "historical_pred_fp_per_game",
        "avg_proj_pass_points",
        "avg_proj_rush_points",
        "avg_proj_rec_points",
        "v2_recenter_promoted",
    ]
    values = [
        "nffc",
        "v2_nffc_expert_consensus",
        expected_center_policy,
        expected_context_source,
        expected_scoring_hash,
        "feature-run",
        160.0,
        10.0,
        170.0,
        10.0,
        100.0,
        20.0,
        50.0,
        0,
    ]
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            """
            CREATE TABLE Best_Ball_Weekly_Templates (
                league TEXT,
                historical_projection_source TEXT,
                historical_center_policy TEXT,
                projection_context_source TEXT,
                projection_context_scoring_hash TEXT,
                projection_context_run_id TEXT,
                model_input_avg_proj_points REAL,
                projection_context_avg_proj_points_delta REAL,
                avg_proj_points REAL,
                historical_pred_fp_per_game REAL,
                avg_proj_pass_points REAL,
                avg_proj_rush_points REAL,
                avg_proj_rec_points REAL,
                v2_recenter_promoted INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO Best_Ball_Weekly_Templates
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        assert refresh._count_invalid_nffc_template_context_rows(
            connection,
            expected_center_policy=expected_center_policy,
            expected_context_source=expected_context_source,
            expected_scoring_hash=expected_scoring_hash,
            expected_horizon=17,
        ) == 0

        connection.execute(
            f'UPDATE Best_Ball_Weekly_Templates '
            f'SET "{nullable_column}"=NULL'
        )

        assert refresh._count_invalid_nffc_template_context_rows(
            connection,
            expected_center_policy=expected_center_policy,
            expected_context_source=expected_context_source,
            expected_scoring_hash=expected_scoring_hash,
            expected_horizon=17,
        ) == 1


def test_beta_template_context_accepts_promoted_and_quarantined_rows():
    expected_hash = refresh.scoring_hash("beta")
    expected_source = "v2_beta_scoring_matched_preseason"
    columns = (
        "league, season, pos, historical_projection_source, "
        "historical_center_policy, projection_context_source, "
        "projection_context_scoring_hash, projection_context_run_id, "
        "scoring_context_available, scoring_context_unavailable_reason, "
        "team_qb_scoring_context_available, "
        "team_qb_scoring_context_unavailable_reason, "
        "team_qb_pass_proj_rank_pct, model_input_avg_proj_points, "
        "projection_context_avg_proj_points_delta, avg_proj_points, "
        "historical_pred_fp_per_game, avg_proj_pass_points, "
        "avg_proj_rush_points, avg_proj_rec_points, "
        "v2_recenter_promoted, template_eligible, "
        "template_exclusion_reason"
    )
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            """
            CREATE TABLE Best_Ball_Weekly_Templates (
                league TEXT, season INTEGER, pos TEXT,
                historical_projection_source TEXT,
                historical_center_policy TEXT,
                projection_context_source TEXT,
                projection_context_scoring_hash TEXT,
                projection_context_run_id TEXT,
                scoring_context_available INTEGER,
                scoring_context_unavailable_reason TEXT,
                team_qb_scoring_context_available INTEGER,
                team_qb_scoring_context_unavailable_reason TEXT,
                team_qb_pass_proj_rank_pct REAL,
                model_input_avg_proj_points REAL,
                projection_context_avg_proj_points_delta REAL,
                avg_proj_points REAL,
                historical_pred_fp_per_game REAL,
                avg_proj_pass_points REAL,
                avg_proj_rush_points REAL,
                avg_proj_rec_points REAL,
                v2_recenter_promoted INTEGER,
                template_eligible INTEGER,
                template_exclusion_reason TEXT
            )
            """
        )
        connection.execute(
            f"INSERT INTO Best_Ball_Weekly_Templates ({columns}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "beta", 2025, "WR",
                "v2_beta_expert_consensus_fallback",
                "beta_scored_expert_fallback",
                expected_source, expected_hash, "feature-run", 1, None,
                1, None, 0.4, 100.0, 104.0, 204.0, 12.0,
                -20.4, 102.0, 122.4, 0, 1, "",
            ),
        )
        connection.execute(
            f"INSERT INTO Best_Ball_Weekly_Templates ({columns}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "beta", 2018, "QB", "validation_ensemble",
                "legacy_validated_oos",
                "v2_beta_scoring_context_unavailable", expected_hash,
                "feature-run", 0,
                refresh.BETA_SCORING_CONTEXT_UNAVAILABLE_REASON,
                0, refresh.BETA_SCORING_CONTEXT_UNAVAILABLE_REASON, 0.5,
                200.0, 0.0, 200.0, 12.0, None, None, None, 0, 0,
                "scoring_context_unavailable:"
                + refresh.BETA_SCORING_CONTEXT_UNAVAILABLE_REASON,
            ),
        )

        assert refresh._count_invalid_beta_template_context_rows(
            connection,
            expected_context_source=expected_source,
            expected_scoring_hash=expected_hash,
        ) == 0

        connection.execute(
            "UPDATE Best_Ball_Weekly_Templates "
            "SET projection_context_source='model_inputs' "
            "WHERE scoring_context_available=1"
        )
        assert refresh._count_invalid_beta_template_context_rows(
            connection,
            expected_context_source=expected_source,
            expected_scoring_hash=expected_hash,
        ) == 1


@pytest.mark.parametrize(
    "nullable_column",
    [
        "current_context_source",
        "projection_context_scoring_hash",
        "projection_context_run_id",
        "current_avg_proj_points",
        "avg_proj_pass_points",
        "avg_proj_rush_points",
        "avg_proj_rec_points",
    ],
)
def test_nffc_player_map_context_rejects_null_required_fields(
    nullable_column,
):
    expected_scoring_hash = refresh.scoring_hash("nffc")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            """
            CREATE TABLE Best_Ball_Weekly_Player_Map (
                version TEXT,
                year INTEGER,
                dataset TEXT,
                current_context_source TEXT,
                projection_context_scoring_hash TEXT,
                projection_context_run_id TEXT,
                current_avg_proj_points REAL,
                avg_proj_pass_points REAL,
                avg_proj_rush_points REAL,
                avg_proj_rec_points REAL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO Best_Ball_Weekly_Player_Map
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "nffc",
                2026,
                "backfill",
                "model_inputs_with_v2_scoring_context",
                expected_scoring_hash,
                "feature-run",
                170.0,
                100.0,
                20.0,
                50.0,
            ),
        )
        assert refresh._count_invalid_nffc_player_map_context_rows(
            connection,
            year=2026,
            dataset="backfill",
            expected_scoring_hash=expected_scoring_hash,
            expected_feature_run_id="feature-run",
        ) == 0

        connection.execute(
            f'UPDATE Best_Ball_Weekly_Player_Map '
            f'SET "{nullable_column}"=NULL'
        )

        assert refresh._count_invalid_nffc_player_map_context_rows(
            connection,
            year=2026,
            dataset="backfill",
            expected_scoring_hash=expected_scoring_hash,
            expected_feature_run_id="feature-run",
        ) == 1


def test_staged_change_after_app_smoke_blocks_promotion(tmp_path):
    staged = tmp_path / "staged.sqlite3"
    auction = tmp_path / "auction.sqlite3"
    snake = tmp_path / "snake.sqlite3"
    for path in (staged, auction, snake):
        _database(
            path,
            [
                "CREATE TABLE values_table (value INTEGER)",
                "INSERT INTO values_table VALUES (1)",
            ],
        )
    manifest = {
        "paths": {
            "staged": {"simulation": str(staged)},
            "app_artifacts": {
                "auction": str(auction),
                "snake": str(snake),
            },
        },
        "steps": {
            "validate": {
                "status": "completed",
                "result": {
                    "integrity": {
                        "simulation": refresh.database_state(staged),
                    }
                },
            },
            "app_smoke": {
                "status": "completed",
                "result": {
                    "auction": {
                        "database_state": refresh.database_state(auction),
                    },
                    "snake": {
                        "database_state": refresh.database_state(snake),
                    },
                },
            },
        },
    }
    refresh.assert_staged_release_unchanged(manifest)
    with closing(sqlite3.connect(auction)) as connection:
        connection.execute("INSERT INTO values_table VALUES (2)")
        connection.commit()

    with pytest.raises(RuntimeError, match="changed after smoke"):
        refresh.assert_staged_release_unchanged(manifest)


def test_promotion_failure_rolls_back_already_replaced_destinations(
    monkeypatch,
    tmp_path,
):
    live_first = tmp_path / "live_first.sqlite3"
    live_second = tmp_path / "live_second.sqlite3"
    staged_first = tmp_path / "staged_first.sqlite3"
    staged_second = tmp_path / "staged_second.sqlite3"
    for path, value in (
        (live_first, "live-first"),
        (live_second, "live-second"),
        (staged_first, "staged-first"),
        (staged_second, "staged-second"),
    ):
        _database(
            path,
            [
                "CREATE TABLE values_table (value TEXT)",
                f"INSERT INTO values_table VALUES ('{value}')",
            ],
        )

    original_states = {
        "first": refresh.database_state(live_first),
        "second": refresh.database_state(live_second),
    }
    manifest = {
        "run_id": "rollback-test",
        "baseline": original_states,
        "steps": {
            step: {"status": "completed"}
            for step in refresh.PIPELINE_STEPS
        },
    }
    monkeypatch.setattr(
        refresh,
        "_promotion_sources",
        lambda _manifest: [
            ("first", staged_first, live_first),
            ("second", staged_second, live_second),
        ],
    )
    monkeypatch.setattr(
        refresh,
        "assert_live_state_unchanged",
        lambda _manifest, _keys: None,
    )
    monkeypatch.setattr(
        refresh,
        "assert_staged_release_unchanged",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        refresh,
        "assert_pipeline_code_unchanged",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        refresh,
        "assert_production_cycle_unchanged",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        refresh,
        "assert_external_inputs_unchanged",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        refresh,
        "validate_release",
        lambda _manifest: {},
    )
    backup_root = tmp_path / "durable_backups"
    monkeypatch.setattr(refresh, "PRODUCTION_BACKUP_ROOT", backup_root)

    real_replace = refresh.os.replace
    failed = False

    def fail_second_install(source, destination):
        nonlocal failed
        source = Path(source)
        destination = Path(destination)
        if (
            not failed
            and source.name
            == f".{live_second.name}.{manifest['run_id']}.release_stage"
            and destination == live_second
        ):
            failed = True
            raise OSError("injected second-artifact install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(refresh.os, "replace", fail_second_install)

    with pytest.raises(
        OSError,
        match="injected second-artifact install failure",
    ):
        refresh.promote_release(manifest)

    assert failed
    assert refresh.database_state(live_first) == original_states["first"]
    assert refresh.database_state(live_second) == original_states["second"]
    assert not list(tmp_path.glob("*.rollback"))
    assert not list(tmp_path.glob("*.release_stage"))
    for label in ("first", "second"):
        durable_backup = (
            backup_root
            / manifest["run_id"]
            / f"{label}.pre_refresh.sqlite3"
        )
        backup_state = refresh.database_state(durable_backup)
        assert backup_state["sha256"] == original_states[label]["sha256"]
        assert (
            backup_state["size_bytes"]
            == original_states[label]["size_bytes"]
        )


def test_promotion_can_atomically_create_registered_bootstrap_database(
    monkeypatch,
    tmp_path,
):
    staged = tmp_path / "staged_nffc.sqlite3"
    live = tmp_path / "live_nffc.sqlite3"
    _database(
        staged,
        [
            "CREATE TABLE values_table (value TEXT)",
            "INSERT INTO values_table VALUES ('nffc')",
        ],
    )
    manifest = {
        "run_id": "bootstrap-test",
        "baseline": {
            "v2_nffc": {
                "path": str(live),
                "exists": False,
            }
        },
        "steps": {
            step: {"status": "completed"}
            for step in refresh.PIPELINE_STEPS
        },
    }
    monkeypatch.setattr(
        refresh,
        "_promotion_sources",
        lambda _manifest: [("v2_nffc", staged, live)],
    )
    for guard in (
        "assert_live_state_unchanged",
        "assert_staged_release_unchanged",
        "assert_pipeline_code_unchanged",
        "assert_external_inputs_unchanged",
        "assert_production_cycle_unchanged",
    ):
        monkeypatch.setattr(refresh, guard, lambda *_args: None)
    monkeypatch.setattr(refresh, "validate_release", lambda _manifest: {})
    monkeypatch.setattr(
        refresh,
        "PRODUCTION_BACKUP_ROOT",
        tmp_path / "backups",
    )

    receipt = refresh.promote_release(manifest)

    assert live.is_file()
    assert refresh.database_state(live)["sha256"] == (
        refresh.database_state(staged)["sha256"]
    )
    assert receipt["artifacts"]["v2_nffc"]["durable_backup"] is None


def test_successful_promotion_reports_deferred_rollback_cleanup(
    monkeypatch,
    tmp_path,
):
    live = tmp_path / "live.sqlite3"
    staged = tmp_path / "staged.sqlite3"
    _database(
        live,
        [
            "CREATE TABLE values_table (value TEXT)",
            "INSERT INTO values_table VALUES ('old')",
        ],
    )
    _database(
        staged,
        [
            "CREATE TABLE values_table (value TEXT)",
            "INSERT INTO values_table VALUES ('new')",
        ],
    )
    manifest = {
        "run_id": "cleanup-warning-test",
        "baseline": {"artifact": refresh.database_state(live)},
        "steps": {
            step: {"status": "completed"}
            for step in refresh.PIPELINE_STEPS
        },
    }
    monkeypatch.setattr(
        refresh,
        "_promotion_sources",
        lambda _manifest: [("artifact", staged, live)],
    )
    for guard in (
        "assert_live_state_unchanged",
        "assert_staged_release_unchanged",
        "assert_pipeline_code_unchanged",
        "assert_external_inputs_unchanged",
        "assert_production_cycle_unchanged",
    ):
        monkeypatch.setattr(refresh, guard, lambda *_args: None)
    monkeypatch.setattr(refresh, "validate_release", lambda _manifest: {})
    monkeypatch.setattr(
        refresh,
        "PRODUCTION_BACKUP_ROOT",
        tmp_path / "backups",
    )

    real_unlink = Path.unlink

    def retain_rollback(path, *args, **kwargs):
        if path.name.endswith(".rollback"):
            raise PermissionError("injected Windows handle")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", retain_rollback)

    receipt = refresh.promote_release(manifest)

    assert refresh.database_state(live)["sha256"] == (
        refresh.database_state(staged)["sha256"]
    )
    assert len(receipt["cleanup_warnings"]) == 1
    warning = receipt["cleanup_warnings"][0]
    assert warning["label"] == "artifact"
    assert "injected Windows handle" in warning["error"]
    assert Path(warning["path"]).is_file()


def test_promote_requires_the_complete_staged_plan():
    with pytest.raises(ValueError, match="complete staged pipeline"):
        refresh.main(
            [
                "--dry-run",
                "--through",
                "snapshot",
                "--promote",
            ]
        )
