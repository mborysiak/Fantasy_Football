import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from Scripts.Modeling import s4_Best_Ball_Weekly as weekly_builder


def _write_database(path, rows, table_name="generated"):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            f'CREATE TABLE "{table_name}" '
            "(row_id INTEGER, value TEXT)"
        )
        connection.executemany(
            f'INSERT INTO "{table_name}" VALUES (?, ?)',
            rows,
        )
        connection.commit()


def _temp_copies(destination):
    return list(
        destination.parent.glob(f".{destination.name}.*.tmp")
    )


def test_atomic_sqlite_copy_replaces_only_after_exact_verification(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    _write_database(source, [(1, "new"), (2, "rows")])
    _write_database(destination, [(1, "old")])
    expected_bytes = source.read_bytes()

    receipt = weekly_builder.copy_sqlite_database_atomic(
        source,
        destination,
    )

    assert destination.read_bytes() == expected_bytes
    assert receipt == {
        "size_bytes": len(expected_bytes),
        "sha256": weekly_builder._file_sha256(source),
    }
    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute(
            "SELECT * FROM generated ORDER BY row_id"
        ).fetchall() == [(1, "new"), (2, "rows")]
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
    assert _temp_copies(destination) == []


def test_atomic_sqlite_copy_integrity_failure_preserves_live_file(tmp_path):
    source = tmp_path / "corrupt.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    source.write_bytes(b"this is not a SQLite database")
    _write_database(destination, [(1, "live")])
    original_bytes = destination.read_bytes()

    with pytest.raises(ValueError, match="integrity check failed"):
        weekly_builder.copy_sqlite_database_atomic(
            source,
            destination,
        )

    assert destination.read_bytes() == original_bytes
    assert _temp_copies(destination) == []


def test_atomic_sqlite_copy_hash_failure_preserves_live_file(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.sqlite3"
    substituted_source = tmp_path / "substituted.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    _write_database(source, [(1, "source")])
    _write_database(substituted_source, [(1, "change")])
    _write_database(destination, [(1, "live")])
    original_bytes = destination.read_bytes()
    real_copyfile = shutil.copyfile

    def substitute_copy(_source, target):
        return real_copyfile(substituted_source, target)

    monkeypatch.setattr(
        weekly_builder.shutil,
        "copyfile",
        substitute_copy,
    )

    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        weekly_builder.copy_sqlite_database_atomic(
            source,
            destination,
        )

    assert destination.read_bytes() == original_bytes
    assert _temp_copies(destination) == []


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_atomic_sqlite_copy_active_sidecar_preserves_live_file(
    suffix,
    tmp_path,
):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    _write_database(source, [(1, "source")])
    _write_database(destination, [(1, "live")])
    original_bytes = destination.read_bytes()
    source.with_name(source.name + suffix).write_bytes(b"active")

    with pytest.raises(ValueError, match="active sidecar exists"):
        weekly_builder.copy_sqlite_database_atomic(
            source,
            destination,
        )

    assert destination.read_bytes() == original_bytes
    assert _temp_copies(destination) == []


def test_selected_table_sync_is_atomic_exact_and_preserves_app_tables(
    tmp_path,
):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "auction.sqlite3"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute(
            "CREATE TABLE generated "
            "(row_id INTEGER, value TEXT)"
        )
        connection.executemany(
            "INSERT INTO generated VALUES (?, ?)",
            [(1, "new"), (1, "new"), (2, "rows")],
        )
        connection.execute(
            "CREATE INDEX ix_generated_value "
            "ON generated(value)"
        )
        connection.execute(
            "CREATE TABLE generated_second "
            "(key TEXT, amount REAL)"
        )
        connection.executemany(
            "INSERT INTO generated_second VALUES (?, ?)",
            [("a", 1.5), ("b", None)],
        )
        connection.commit()
    with closing(sqlite3.connect(destination)) as connection:
        connection.execute(
            "CREATE TABLE generated "
            "(row_id INTEGER, value TEXT)"
        )
        connection.execute(
            "INSERT INTO generated VALUES (9, 'stale')"
        )
        connection.execute(
            "CREATE TABLE app_owned "
            "(scenario TEXT, budget INTEGER)"
        )
        connection.execute(
            "INSERT INTO app_owned VALUES ('keeper-state', 298)"
        )
        connection.commit()

    counts = weekly_builder.synchronize_sqlite_tables_atomic(
        source,
        destination,
        ["generated", "generated_second"],
    )

    assert counts == {
        "generated": 3,
        "generated_second": 2,
    }
    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute(
            "SELECT * FROM generated ORDER BY row_id, value"
        ).fetchall() == [
            (1, "new"),
            (1, "new"),
            (2, "rows"),
        ]
        assert connection.execute(
            "SELECT * FROM generated_second ORDER BY key"
        ).fetchall() == [
            ("a", 1.5),
            ("b", None),
        ]
        assert connection.execute(
            "SELECT * FROM app_owned"
        ).fetchall() == [("keeper-state", 298)]
        assert connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master "
            "WHERE type='index' AND name='ix_generated_value'"
        ).fetchone() == (
            "generated",
            "CREATE INDEX ix_generated_value ON generated(value)",
        )
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"


def test_selected_table_sync_failure_rolls_back_existing_app_tables(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "auction.sqlite3"
    _write_database(source, [(1, "new")])
    _write_database(destination, [(1, "live")])

    with pytest.raises(ValueError, match="source is missing: absent"):
        weekly_builder.synchronize_sqlite_tables_atomic(
            source,
            destination,
            ["generated", "absent"],
        )

    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute(
            "SELECT * FROM generated"
        ).fetchall() == [(1, "live")]
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"


def test_selected_table_index_failure_rolls_back_all_app_changes(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "auction.sqlite3"
    _write_database(source, [(1, "new")])
    with closing(sqlite3.connect(source)) as connection:
        connection.execute(
            "CREATE INDEX shared_index_name ON generated(value)"
        )
        connection.commit()
    with closing(sqlite3.connect(destination)) as connection:
        connection.execute(
            "CREATE TABLE generated "
            "(row_id INTEGER, value TEXT)"
        )
        connection.execute(
            "INSERT INTO generated VALUES (1, 'live')"
        )
        connection.execute(
            "CREATE TABLE app_owned "
            "(scenario TEXT, budget INTEGER)"
        )
        connection.execute(
            "INSERT INTO app_owned VALUES ('keeper-state', 298)"
        )
        connection.execute(
            "CREATE INDEX shared_index_name ON app_owned(scenario)"
        )
        connection.commit()

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        weekly_builder.synchronize_sqlite_tables_atomic(
            source,
            destination,
            ["generated"],
        )

    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute(
            "SELECT * FROM generated"
        ).fetchall() == [(1, "live")]
        assert connection.execute(
            "SELECT * FROM app_owned"
        ).fetchall() == [("keeper-state", 298)]
        assert connection.execute(
            "SELECT tbl_name FROM sqlite_master "
            "WHERE type='index' AND name='shared_index_name'"
        ).fetchone() == ("app_owned",)
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"


def _promotion_artifact(label, staged, destination):
    return {
        "label": label,
        "staged": staged,
        "destination": destination,
        "size_bytes": staged.stat().st_size,
        "sha256": weekly_builder._file_sha256(staged),
        **weekly_builder._capture_sqlite_file_state(destination),
    }


def test_multi_app_promotion_replaces_both_verified_artifacts(tmp_path):
    snake_live = tmp_path / "snake.sqlite3"
    auction_live = tmp_path / "auction.sqlite3"
    snake_stage = tmp_path / "snake.stage.sqlite3"
    auction_stage = tmp_path / "auction.stage.sqlite3"
    _write_database(snake_live, [(1, "old-snake")])
    _write_database(auction_live, [(1, "old-auction")])
    _write_database(snake_stage, [(1, "new-snake")])
    _write_database(auction_stage, [(1, "new-auction")])
    expected_snake = snake_stage.read_bytes()
    expected_auction = auction_stage.read_bytes()

    weekly_builder.promote_sqlite_artifacts_with_rollback(
        [
            _promotion_artifact(
                "Snake",
                snake_stage,
                snake_live,
            ),
            _promotion_artifact(
                "Auction",
                auction_stage,
                auction_live,
            ),
        ]
    )

    assert snake_live.read_bytes() == expected_snake
    assert auction_live.read_bytes() == expected_auction
    assert not snake_stage.exists()
    assert not auction_stage.exists()
    assert list(tmp_path.glob(".*.pre_release_backup.*.tmp")) == []


def test_multi_app_second_promotion_failure_restores_both_live_files(
    monkeypatch,
    tmp_path,
):
    snake_live = tmp_path / "snake.sqlite3"
    auction_live = tmp_path / "auction.sqlite3"
    snake_stage = tmp_path / "snake.stage.sqlite3"
    auction_stage = tmp_path / "auction.stage.sqlite3"
    _write_database(snake_live, [(1, "old-snake")])
    _write_database(auction_live, [(1, "old-auction")])
    _write_database(snake_stage, [(1, "new-snake")])
    _write_database(auction_stage, [(1, "new-auction")])
    original_snake = snake_live.read_bytes()
    original_auction = auction_live.read_bytes()
    real_replace = os.replace

    def fail_auction_promotion(source, destination):
        if (
            Path(source).resolve() == auction_stage.resolve()
            and Path(destination).resolve() == auction_live.resolve()
        ):
            raise PermissionError("simulated locked Auction database")
        return real_replace(source, destination)

    monkeypatch.setattr(
        weekly_builder.os,
        "replace",
        fail_auction_promotion,
    )

    with pytest.raises(
        PermissionError,
        match="simulated locked Auction",
    ):
        weekly_builder.promote_sqlite_artifacts_with_rollback(
            [
                _promotion_artifact(
                    "Snake",
                    snake_stage,
                    snake_live,
                ),
                _promotion_artifact(
                    "Auction",
                    auction_stage,
                    auction_live,
                ),
            ]
        )

    assert snake_live.read_bytes() == original_snake
    assert auction_live.read_bytes() == original_auction
    assert list(tmp_path.glob(".*.pre_release_backup.*.tmp")) == []


def test_multi_app_promotion_refuses_newer_auction_app_state(tmp_path):
    snake_live = tmp_path / "snake.sqlite3"
    auction_live = tmp_path / "auction.sqlite3"
    snake_stage = tmp_path / "snake.stage.sqlite3"
    auction_stage = tmp_path / "auction.stage.sqlite3"
    _write_database(snake_live, [(1, "old-snake")])
    _write_database(auction_live, [(1, "old-auction")])
    _write_database(snake_stage, [(1, "new-snake")])
    _write_database(auction_stage, [(1, "new-auction")])
    original_snake = snake_live.read_bytes()
    artifacts = [
        _promotion_artifact(
            "Snake",
            snake_stage,
            snake_live,
        ),
        _promotion_artifact(
            "Auction",
            auction_stage,
            auction_live,
        ),
    ]

    with closing(sqlite3.connect(auction_live)) as connection:
        connection.execute(
            "UPDATE generated SET value='newer-app-write'"
        )
        connection.commit()
    newer_auction = auction_live.read_bytes()

    with pytest.raises(
        ValueError,
        match="Auction live SQLite changed since staging",
    ):
        weekly_builder.promote_sqlite_artifacts_with_rollback(
            artifacts
        )

    assert snake_live.read_bytes() == original_snake
    assert auction_live.read_bytes() == newer_auction
    assert snake_stage.exists()
    assert auction_stage.exists()
    assert list(tmp_path.glob(".*.pre_release_backup.*.tmp")) == []


def test_multi_app_promotion_detects_write_during_live_backup_move(
    monkeypatch,
    tmp_path,
):
    snake_live = tmp_path / "snake.sqlite3"
    auction_live = tmp_path / "auction.sqlite3"
    snake_stage = tmp_path / "snake.stage.sqlite3"
    auction_stage = tmp_path / "auction.stage.sqlite3"
    _write_database(snake_live, [(1, "old-snake")])
    _write_database(auction_live, [(1, "old-auction")])
    _write_database(snake_stage, [(1, "new-snake")])
    _write_database(auction_stage, [(1, "new-auction")])
    original_snake = snake_live.read_bytes()
    artifacts = [
        _promotion_artifact(
            "Snake",
            snake_stage,
            snake_live,
        ),
        _promotion_artifact(
            "Auction",
            auction_stage,
            auction_live,
        ),
    ]
    real_replace = os.replace
    injected_write = False

    def write_then_move_live_to_backup(source, destination):
        nonlocal injected_write
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if (
            not injected_write
            and source_path == auction_live.resolve()
            and "pre_release_backup" in destination_path.name
        ):
            with closing(sqlite3.connect(auction_live)) as connection:
                connection.execute(
                    "UPDATE generated SET value='last-moment-app-write'"
                )
                connection.commit()
            injected_write = True
        return real_replace(source, destination)

    monkeypatch.setattr(
        weekly_builder.os,
        "replace",
        write_then_move_live_to_backup,
    )

    with pytest.raises(
        ValueError,
        match="Auction live SQLite changed during promotion",
    ):
        weekly_builder.promote_sqlite_artifacts_with_rollback(
            artifacts
        )

    assert injected_write
    assert snake_live.read_bytes() == original_snake
    with closing(sqlite3.connect(auction_live)) as connection:
        assert connection.execute(
            "SELECT * FROM generated"
        ).fetchall() == [(1, "last-moment-app-write")]
    assert list(tmp_path.glob(".*.pre_release_backup.*.tmp")) == []


def test_requested_app_export_fails_closed_on_incomplete_schema(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "Simulation.sqlite3"
    required_tables = [
        weekly_builder.AVG_ADP_TABLE,
        weekly_builder.AVG_ADP_AUDIT_TABLE,
        weekly_builder.AVG_ADP_RECEIPT_TABLE,
        "Final_Predictions_Resid",
        "V2_Production_Projection_Handoff",
        "V2_Production_Projection_Audit",
        "V2_Production_Eligibility_Audit",
        weekly_builder.TEMPLATE_TABLE,
        weekly_builder.POOL_TABLE,
        weekly_builder.POOL_SUMMARY_TABLE,
        weekly_builder.PLAYER_MAP_TABLE,
        weekly_builder.TEMPLATE_AUDIT_TABLE,
        weekly_builder.PLAYER_POOL_AUDIT_TABLE,
        weekly_builder.BUCKET_AUDIT_TABLE,
        weekly_builder.ADP_AUDIT_TABLE,
    ]
    with closing(sqlite3.connect(source)) as connection:
        for table_name in required_tables:
            connection.execute(
                f'CREATE TABLE "{table_name}" (placeholder TEXT)'
            )
        connection.commit()

    monkeypatch.setattr(
        weekly_builder,
        "SIMULATION_DB_PATH",
        source,
    )
    monkeypatch.setattr(
        weekly_builder,
        "validate_avg_adp_publication",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        ValueError,
        match="weekly-template schema is incomplete",
    ):
        weekly_builder.copy_simulation_db_to_apps()


def test_weekly_template_export_validates_each_league_horizon(tmp_path):
    source = tmp_path / "Simulation.sqlite3"
    managed_columns = [
        f"managed_week_{week}" for week in range(1, 18)
    ]
    played_columns = [
        f"played_week_{week}" for week in range(1, 18)
    ]
    week_columns = managed_columns + played_columns
    with closing(sqlite3.connect(source)) as connection:
        connection.execute(
            f"""
            CREATE TABLE "{weekly_builder.TEMPLATE_TABLE}" (
                league TEXT,
                player_key TEXT,
                {", ".join(f'"{column}" REAL' for column in week_columns)}
            )
            """
        )
        placeholders = ", ".join("?" for _ in range(2 + len(week_columns)))
        connection.executemany(
            f'INSERT INTO "{weekly_builder.TEMPLATE_TABLE}" '
            f"VALUES ({placeholders})",
            [
                (
                    "dk",
                    "dk-player",
                    *([1.0] * 16),
                    None,
                    *([1.0] * 16),
                    None,
                ),
                (
                    "nffc",
                    "nffc-player",
                    *([1.0] * 17),
                    *([1.0] * 17),
                ),
            ],
        )
        connection.commit()

        assert weekly_builder.validate_weekly_template_export(connection) == {
            "dk": 16,
            "nffc": 17,
        }

        connection.execute(
            f'UPDATE "{weekly_builder.TEMPLATE_TABLE}" '
            "SET played_week_17=NULL WHERE league='nffc'"
        )
        connection.commit()

        with pytest.raises(
            ValueError,
            match="retained nffc template rows.*through week 17",
        ):
            weekly_builder.validate_weekly_template_export(connection)


def test_app_export_prepares_both_apps_from_one_source_snapshot(
    monkeypatch,
    tmp_path,
):
    main_root = tmp_path / "Fantasy_Football"
    auction_dir = tmp_path / "Fantasy_Football_App" / "app"
    snake_dir = tmp_path / "Fantasy_Football_Snake" / "app"
    main_root.mkdir()
    auction_dir.mkdir(parents=True)
    snake_dir.mkdir(parents=True)
    source = main_root / "Simulation.sqlite3"
    auction_live = auction_dir / "Simulation.sqlite3"
    snake_live = snake_dir / "Simulation.sqlite3"
    generated_tables = [
        weekly_builder.AVG_ADP_TABLE,
        weekly_builder.AVG_ADP_AUDIT_TABLE,
        weekly_builder.AVG_ADP_RECEIPT_TABLE,
        "Final_Predictions_Resid",
        "V2_Production_Projection_Handoff",
        "V2_Production_Projection_Audit",
        "V2_Production_Eligibility_Audit",
        weekly_builder.TEMPLATE_TABLE,
        weekly_builder.POOL_TABLE,
        weekly_builder.POOL_SUMMARY_TABLE,
        weekly_builder.PLAYER_MAP_TABLE,
        weekly_builder.TEMPLATE_AUDIT_TABLE,
        weekly_builder.PLAYER_POOL_AUDIT_TABLE,
        weekly_builder.BUCKET_AUDIT_TABLE,
        weekly_builder.ADP_AUDIT_TABLE,
    ]
    template_columns = [
        "league TEXT",
        "player_key TEXT",
        *[
            f"managed_week_{week} REAL"
            for week in weekly_builder.WEEKS
        ],
        *[
            f"played_week_{week} REAL"
            for week in weekly_builder.WEEKS
        ],
    ]
    prediction_columns = [
        "year INTEGER",
        "dataset TEXT",
        "version TEXT",
        "player_key TEXT",
        "pred_fp_per_game REAL",
        "pred_fp_per_game_ny REAL",
        "pred_appear_ny REAL",
        "current_uncertainty_source TEXT",
        "independent_current_residual_draw_allowed INTEGER",
        "production_handoff_version TEXT",
    ]
    with closing(sqlite3.connect(source)) as connection:
        for table_name in generated_tables:
            if table_name == weekly_builder.TEMPLATE_TABLE:
                columns = template_columns
            elif table_name == weekly_builder.PLAYER_MAP_TABLE:
                columns = ["player_key TEXT"]
            elif table_name == "Final_Predictions_Resid":
                columns = prediction_columns
            else:
                columns = ["placeholder TEXT"]
            connection.execute(
                f'CREATE TABLE "{table_name}" ({", ".join(columns)})'
            )
        connection.execute(
            f'INSERT INTO "{weekly_builder.AVG_ADP_TABLE}" '
            "VALUES ('source-release')"
        )
        connection.execute(
            f'CREATE INDEX ix_test_avg_adp_release '
            f'ON "{weekly_builder.AVG_ADP_TABLE}"(placeholder)'
        )
        connection.commit()
    with closing(sqlite3.connect(auction_live)) as connection:
        connection.execute(
            "CREATE TABLE app_owned "
            "(scenario TEXT, budget INTEGER)"
        )
        connection.execute(
            "INSERT INTO app_owned VALUES ('keeper-state', 298)"
        )
        connection.commit()
    _write_database(snake_live, [(1, "old-snake")])
    expected_source_bytes = source.read_bytes()

    monkeypatch.setattr(weekly_builder, "root_path", str(main_root))
    monkeypatch.setattr(
        weekly_builder,
        "SIMULATION_DB_PATH",
        source,
    )
    monkeypatch.setattr(
        weekly_builder,
        "validate_avg_adp_publication",
        lambda *_args, **_kwargs: None,
    )

    weekly_builder.copy_simulation_db_to_apps()

    assert snake_live.read_bytes() == expected_source_bytes
    with closing(sqlite3.connect(auction_live)) as connection:
        assert connection.execute(
            "SELECT * FROM app_owned"
        ).fetchall() == [("keeper-state", 298)]
        assert connection.execute(
            f'SELECT * FROM "{weekly_builder.AVG_ADP_TABLE}"'
        ).fetchall() == [("source-release",)]
        assert connection.execute(
            "SELECT tbl_name FROM sqlite_master "
            "WHERE type='index' AND name='ix_test_avg_adp_release'"
        ).fetchone() == (weekly_builder.AVG_ADP_TABLE,)
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
    assert list(auction_dir.glob(".*.tmp")) == []
    assert list(snake_dir.glob(".*.tmp")) == []
