from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Scripts.V2 import refresh_production as refresh
from Scripts.V2.release_change_report import (
    REPORT_VERSION,
    build_release_change_report,
    load_verified_release_change_report,
    load_weighted_template_residual_rows,
    render_release_change_report,
    write_release_change_report,
)


def _write_release_database(
    path: Path,
    players: dict[str, tuple[str, str, float, float]],
) -> None:
    """Write one-league projections and 80 equiprobable donors per player."""

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE Final_Predictions_Resid (
                version TEXT,
                player_key TEXT,
                player TEXT,
                pos TEXT,
                pred_fp_per_game REAL,
                year INTEGER,
                dataset TEXT
            );
            CREATE TABLE Best_Ball_Weekly_Player_Map (
                version TEXT,
                player_key TEXT,
                player TEXT,
                pos TEXT,
                pred_fp_per_game REAL,
                year INTEGER,
                dataset TEXT,
                template_pool_key TEXT
            );
            CREATE TABLE Best_Ball_Weekly_Template_Pools (
                template_pool_key TEXT,
                template_id INTEGER,
                template_league TEXT,
                template_sample_prob REAL
            );
            CREATE TABLE Best_Ball_Weekly_Templates (
                league TEXT,
                template_id INTEGER,
                active_ppg_resid REAL
            );
            """
        )
        next_template_id = 1
        for player_key, (player, pos, projection, residual) in players.items():
            pool_key = f"2026|dk|final_ensemble|{pos}|{player_key}"
            connection.execute(
                "INSERT INTO Final_Predictions_Resid VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "dk",
                    player_key,
                    player,
                    pos,
                    projection,
                    2026,
                    "final_ensemble",
                ),
            )
            connection.execute(
                """
                INSERT INTO Best_Ball_Weekly_Player_Map
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "dk",
                    player_key,
                    player,
                    pos,
                    projection,
                    2026,
                    "final_ensemble",
                    pool_key,
                ),
            )
            for _ in range(80):
                template_id = next_template_id
                next_template_id += 1
                connection.execute(
                    "INSERT INTO Best_Ball_Weekly_Templates VALUES (?, ?, ?)",
                    ("dk", template_id, residual),
                )
                connection.execute(
                    "INSERT INTO Best_Ball_Weekly_Template_Pools VALUES (?, ?, ?, ?)",
                    (pool_key, template_id, "dk", 1.0 / 80.0),
                )
        connection.commit()


def test_release_change_report_captures_movers_population_and_residuals(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    _write_release_database(
        baseline,
        {
            "alpha": ("Alpha Player", "WR", 10.0, 1.0),
            "bravo": ("Bravo Player", "RB", 8.0, -2.0),
            "dropped": ("Dropped Player", "TE", 5.0, 0.0),
        },
    )
    _write_release_database(
        candidate,
        {
            "alpha": ("Alpha Player", "WR", 12.0, 3.0),
            "bravo": ("Bravo Player", "RB", 7.0, -4.0),
            "added": ("Added Player", "QB", 6.0, 5.0),
        },
    )

    report = build_release_change_report(
        baseline,
        candidate,
        year=2026,
        dataset="final_ensemble",
        run_id="report-test",
        leagues=("dk",),
        top_n=2,
        generated_at_utc="2026-08-08T00:00:00+00:00",
    )

    assert report["report_version"] == REPORT_VERSION
    assert report["population"]["added_count"] == 1
    assert report["population"]["dropped_count"] == 1
    assert {row["player"] for row in report["population"]["changes"]} == {
        "Added Player",
        "Dropped Player",
    }
    assert report["projections"]["increases"][0]["player"] == "Alpha Player"
    assert report["projections"]["increases"][0][
        "projection_delta_ppg"
    ] == pytest.approx(2.0)
    assert report["projections"]["decreases"][0]["player"] == "Bravo Player"
    assert report["weighted_template_residuals"]["positive"][0][
        "player"
    ] == "Added Player"
    assert report["weighted_template_residuals"]["positive"][0][
        "old_weighted_template_residual"
    ] is None
    assert report["weighted_template_residuals"]["negative"][0][
        "player"
    ] == "Bravo Player"
    assert report["weighted_template_residuals"]["negative"][0][
        "weighted_template_residual_delta"
    ] == pytest.approx(-2.0)
    markdown = render_release_change_report(report)
    assert "Top 2 projection increases" in markdown
    assert "Added Player" in markdown
    assert "Bravo Player" in markdown


def test_saved_release_change_report_is_hash_verified(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    players = {"alpha": ("Alpha Player", "WR", 10.0, 1.0)}
    _write_release_database(baseline, players)
    _write_release_database(candidate, players)
    report = build_release_change_report(
        baseline,
        candidate,
        year=2026,
        dataset="final_ensemble",
        run_id="verified-report-test",
        leagues=("dk",),
        generated_at_utc="2026-08-08T00:00:00+00:00",
    )
    receipt = write_release_change_report(tmp_path / "report", report)

    loaded = load_verified_release_change_report(
        receipt,
        baseline_database=baseline,
        candidate_database=candidate,
    )
    assert loaded == report

    Path(receipt["markdown"]["path"]).write_text(
        "changed after review\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="changed after review"):
        load_verified_release_change_report(
            receipt,
            baseline_database=baseline,
            candidate_database=candidate,
        )


def test_weighted_template_report_rejects_probability_drift(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bad_probability.sqlite3"
    _write_release_database(
        database,
        {"alpha": ("Alpha Player", "WR", 10.0, 1.0)},
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE Best_Ball_Weekly_Template_Pools "
            "SET template_sample_prob=0.02"
        )
        connection.commit()

    with pytest.raises(ValueError, match="probabilities sum"):
        load_weighted_template_residual_rows(
            database,
            year=2026,
            dataset="final_ensemble",
            leagues=("dk",),
        )


def test_reviewed_report_is_archived_with_promotion_backups(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    players = {"alpha": ("Alpha Player", "WR", 10.0, 1.0)}
    _write_release_database(baseline, players)
    _write_release_database(candidate, players)
    report = build_release_change_report(
        baseline,
        candidate,
        year=2026,
        dataset="final_ensemble",
        run_id="archive-report-test",
        leagues=("dk",),
        generated_at_utc="2026-08-08T00:00:00+00:00",
    )
    receipt = write_release_change_report(tmp_path / "stage", report)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    manifest = {
        "paths": {
            "live": {"simulation": str(baseline)},
            "staged": {"simulation": str(candidate)},
        },
        "release_change_report": receipt,
    }

    archived = refresh._archive_release_change_report(
        manifest,
        backup_dir,
    )

    assert Path(archived["json"]["path"]) == (
        backup_dir / "release_change_report.json"
    )
    assert Path(archived["markdown"]["path"]) == (
        backup_dir / "release_change_report.md"
    )
    assert archived["json"]["sha256"] == receipt["json"]["sha256"]
    assert archived["markdown"]["sha256"] == receipt["markdown"]["sha256"]
