"""Build a human-readable prediction/template change receipt for a release.

The report compares the live Simulation database captured at refresh start with
the fully validated staged candidate.  Projection movement comes from the
published ``Final_Predictions_Resid.pred_fp_per_game`` center.  Template
residuals use the same joint donor path as the apps: the probability-weighted
mean of ``Best_Ball_Weekly_Templates.active_ppg_resid`` under each player's
``template_sample_prob`` vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_VERSION = "production_release_change_report_v1"
DEFAULT_TOP_N = 10
EXPECTED_TEMPLATE_DONORS = 80
PROJECTION_TABLE = "Final_Predictions_Resid"
PLAYER_MAP_TABLE = "Best_Ball_Weekly_Player_Map"
POOL_TABLE = "Best_Ball_Weekly_Template_Pools"
TEMPLATE_TABLE = "Best_Ball_Weekly_Templates"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def _require_columns(
    connection: sqlite3.Connection,
    table: str,
    required: Iterable[str],
) -> None:
    columns = _table_columns(connection, table)
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"{table} is missing report columns: {missing}")


def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, received {value!r}")
    return number


def load_projection_rows(
    database: Path,
    *,
    year: int,
    dataset: str,
    leagues: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load one unique published point center per league/player key."""

    database = Path(database).resolve()
    allowed_leagues = {str(league) for league in leagues}
    required = {
        "version",
        "player_key",
        "player",
        "pos",
        "pred_fp_per_game",
        "year",
        "dataset",
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        _require_columns(connection, PROJECTION_TABLE, required)
        query = f"""
            SELECT version AS league, player_key, player, pos,
                   pred_fp_per_game
            FROM {PROJECTION_TABLE}
            WHERE year=? AND dataset=?
            ORDER BY version, player_key
        """
        for source in connection.execute(query, (int(year), str(dataset))):
            league = str(source["league"])
            if league not in allowed_leagues:
                continue
            player_key = str(source["player_key"] or "").strip()
            if not player_key:
                raise ValueError(
                    f"{database} has a blank published player_key for {league}"
                )
            key = (league, player_key)
            if key in rows:
                raise ValueError(
                    f"{database} has duplicate published projection key {key}"
                )
            rows[key] = {
                "league": league,
                "player_key": player_key,
                "player": str(source["player"]),
                "pos": str(source["pos"]),
                "projection_ppg": _finite_float(
                    source["pred_fp_per_game"],
                    label=f"{key} projection_ppg",
                ),
            }
    missing_leagues = sorted(
        league
        for league in allowed_leagues
        if not any(key[0] == league for key in rows)
    )
    if missing_leagues:
        raise ValueError(
            f"{database} has no published projection rows for {missing_leagues}"
        )
    return rows


def load_weighted_template_residual_rows(
    database: Path,
    *,
    year: int,
    dataset: str,
    leagues: Sequence[str],
    expected_donors: int = EXPECTED_TEMPLATE_DONORS,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the exact expected donor residual under app sampling weights."""

    database = Path(database).resolve()
    allowed_leagues = {str(league) for league in leagues}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        _require_columns(
            connection,
            PLAYER_MAP_TABLE,
            {
                "version",
                "player_key",
                "player",
                "pos",
                "year",
                "dataset",
                "pred_fp_per_game",
                "template_pool_key",
            },
        )
        _require_columns(
            connection,
            POOL_TABLE,
            {
                "template_pool_key",
                "template_id",
                "template_league",
                "template_sample_prob",
            },
        )
        _require_columns(
            connection,
            TEMPLATE_TABLE,
            {"league", "template_id", "active_ppg_resid"},
        )
        query = f"""
            SELECT
                player_map.version AS league,
                player_map.player_key,
                player_map.player,
                player_map.pos,
                player_map.pred_fp_per_game,
                COUNT(*) AS donor_count,
                COUNT(DISTINCT pool.template_id) AS distinct_donor_count,
                SUM(
                    CASE
                        WHEN pool.template_sample_prob IS NULL
                          OR pool.template_sample_prob < 0
                          OR template.active_ppg_resid IS NULL
                        THEN 1 ELSE 0
                    END
                ) AS invalid_donor_rows,
                SUM(pool.template_sample_prob) AS probability_sum,
                SUM(
                    pool.template_sample_prob * template.active_ppg_resid
                ) / SUM(pool.template_sample_prob) AS weighted_template_residual
            FROM {PLAYER_MAP_TABLE} AS player_map
            JOIN {POOL_TABLE} AS pool
              ON pool.template_pool_key=player_map.template_pool_key
            JOIN {TEMPLATE_TABLE} AS template
              ON template.league=pool.template_league
             AND template.template_id=pool.template_id
            WHERE player_map.year=? AND player_map.dataset=?
            GROUP BY
                player_map.version,
                player_map.player_key,
                player_map.player,
                player_map.pos,
                player_map.pred_fp_per_game
            ORDER BY player_map.version, player_map.player_key
        """
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for source in connection.execute(query, (int(year), str(dataset))):
            league = str(source["league"])
            if league not in allowed_leagues:
                continue
            player_key = str(source["player_key"] or "").strip()
            key = (league, player_key)
            if not player_key or key in result:
                raise ValueError(
                    f"{database} has invalid weighted-residual key {key}"
                )
            donor_count = int(source["donor_count"])
            distinct_donor_count = int(source["distinct_donor_count"])
            invalid_donor_rows = int(source["invalid_donor_rows"])
            probability_sum = _finite_float(
                source["probability_sum"],
                label=f"{key} template probability sum",
            )
            if donor_count != int(expected_donors):
                raise ValueError(
                    f"{key} has {donor_count} weighted-template donors; "
                    f"expected {expected_donors}"
                )
            if distinct_donor_count != donor_count:
                raise ValueError(f"{key} has duplicate weighted-template donors")
            if invalid_donor_rows:
                raise ValueError(
                    f"{key} has {invalid_donor_rows} invalid donor residual rows"
                )
            if not math.isclose(
                probability_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{key} template probabilities sum to {probability_sum}"
                )
            result[key] = {
                "league": league,
                "player_key": player_key,
                "player": str(source["player"]),
                "pos": str(source["pos"]),
                "projection_ppg": _finite_float(
                    source["pred_fp_per_game"],
                    label=f"{key} template-map projection_ppg",
                ),
                "weighted_template_residual": _finite_float(
                    source["weighted_template_residual"],
                    label=f"{key} weighted_template_residual",
                ),
                "donor_count": donor_count,
                "probability_sum": probability_sum,
            }
    missing_leagues = sorted(
        league
        for league in allowed_leagues
        if not any(key[0] == league for key in result)
    )
    if missing_leagues:
        raise ValueError(
            f"{database} has no weighted residual rows for {missing_leagues}"
        )
    return result


def _top_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    column: str,
    count: int,
    descending: bool,
) -> list[dict[str, Any]]:
    direction = -1.0 if descending else 1.0
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            direction * float(row[column]),
            str(row["league"]),
            str(row["player_key"]),
        ),
    )
    return ordered[: int(count)]


def build_release_change_report(
    baseline_database: Path,
    candidate_database: Path,
    *,
    year: int,
    dataset: str,
    run_id: str,
    leagues: Sequence[str],
    top_n: int = DEFAULT_TOP_N,
    generated_at_utc: str | None = None,
    retrospective: bool = False,
) -> dict[str, Any]:
    """Compare one staged Simulation release with its unchanged live baseline."""

    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    baseline_database = Path(baseline_database).resolve()
    candidate_database = Path(candidate_database).resolve()
    baseline_projections = load_projection_rows(
        baseline_database,
        year=year,
        dataset=dataset,
        leagues=leagues,
    )
    candidate_projections = load_projection_rows(
        candidate_database,
        year=year,
        dataset=dataset,
        leagues=leagues,
    )
    baseline_residuals = load_weighted_template_residual_rows(
        baseline_database,
        year=year,
        dataset=dataset,
        leagues=leagues,
    )
    candidate_residuals = load_weighted_template_residual_rows(
        candidate_database,
        year=year,
        dataset=dataset,
        leagues=leagues,
    )
    if set(baseline_projections) != set(baseline_residuals):
        raise ValueError(
            "Baseline projection and weighted-template populations differ"
        )
    if set(candidate_projections) != set(candidate_residuals):
        raise ValueError(
            "Candidate projection and weighted-template populations differ"
        )

    baseline_keys = set(baseline_projections)
    candidate_keys = set(candidate_projections)
    common_keys = sorted(baseline_keys.intersection(candidate_keys))
    added_keys = sorted(candidate_keys.difference(baseline_keys))
    dropped_keys = sorted(baseline_keys.difference(candidate_keys))

    projection_changes = []
    for key in common_keys:
        old_projection = baseline_projections[key]
        new_projection = candidate_projections[key]
        projection_changes.append(
            {
                "league": key[0],
                "player_key": key[1],
                "player": new_projection["player"],
                "pos": new_projection["pos"],
                "old_projection_ppg": old_projection["projection_ppg"],
                "new_projection_ppg": new_projection["projection_ppg"],
                "projection_delta_ppg": (
                    new_projection["projection_ppg"]
                    - old_projection["projection_ppg"]
                ),
            }
        )
    candidate_residual_rows = []
    for key in sorted(candidate_keys):
        new_projection = candidate_projections[key]
        new_residual = candidate_residuals[key]
        old_residual = baseline_residuals.get(key)
        old_value = (
            old_residual["weighted_template_residual"]
            if old_residual is not None
            else None
        )
        candidate_residual_rows.append(
            {
                "league": key[0],
                "player_key": key[1],
                "player": new_projection["player"],
                "pos": new_projection["pos"],
                "new_projection_ppg": new_projection["projection_ppg"],
                "old_weighted_template_residual": old_value,
                "new_weighted_template_residual": new_residual[
                    "weighted_template_residual"
                ],
                "weighted_template_residual_delta": (
                    new_residual["weighted_template_residual"] - old_value
                    if old_value is not None
                    else None
                ),
                "donor_count": new_residual["donor_count"],
            }
        )

    def population_row(
        key: tuple[str, str],
        source: Mapping[tuple[str, str], Mapping[str, Any]],
        status: str,
    ) -> dict[str, Any]:
        row = source[key]
        return {
            "status": status,
            "league": key[0],
            "player_key": key[1],
            "player": row["player"],
            "pos": row["pos"],
            "projection_ppg": row["projection_ppg"],
        }

    population_changes = [
        *(population_row(key, candidate_projections, "added") for key in added_keys),
        *(population_row(key, baseline_projections, "dropped") for key in dropped_keys),
    ]
    population_changes = sorted(
        population_changes,
        key=lambda row: (
            str(row["status"]),
            str(row["league"]),
            str(row["pos"]),
            str(row["player"]),
        ),
    )
    league_summary = []
    for league in leagues:
        old = {key for key in baseline_keys if key[0] == league}
        new = {key for key in candidate_keys if key[0] == league}
        league_summary.append(
            {
                "league": str(league),
                "old_count": len(old),
                "new_count": len(new),
                "common_count": len(old.intersection(new)),
                "added_count": len(new.difference(old)),
                "dropped_count": len(old.difference(new)),
            }
        )

    return {
        "report_version": REPORT_VERSION,
        "run_id": str(run_id),
        "generated_at_utc": generated_at_utc or utc_now(),
        "retrospective": bool(retrospective),
        "year": int(year),
        "dataset": str(dataset),
        "leagues": [str(league) for league in leagues],
        "top_n": int(top_n),
        "baseline_simulation": _file_receipt(baseline_database),
        "candidate_simulation": _file_receipt(candidate_database),
        "population": {
            "old_count": len(baseline_keys),
            "new_count": len(candidate_keys),
            "common_count": len(common_keys),
            "added_count": len(added_keys),
            "dropped_count": len(dropped_keys),
            "by_league": league_summary,
            "changes": population_changes,
        },
        "projections": {
            "increases": _top_rows(
                (
                    row
                    for row in projection_changes
                    if row["projection_delta_ppg"] > 0
                ),
                column="projection_delta_ppg",
                count=top_n,
                descending=True,
            ),
            "decreases": _top_rows(
                (
                    row
                    for row in projection_changes
                    if row["projection_delta_ppg"] < 0
                ),
                column="projection_delta_ppg",
                count=top_n,
                descending=False,
            ),
        },
        "weighted_template_residuals": {
            "definition": (
                "sum(template_sample_prob * active_ppg_resid) / "
                "sum(template_sample_prob)"
            ),
            "positive": _top_rows(
                (
                    row
                    for row in candidate_residual_rows
                    if row["new_weighted_template_residual"] > 0
                ),
                column="new_weighted_template_residual",
                count=top_n,
                descending=True,
            ),
            "negative": _top_rows(
                (
                    row
                    for row in candidate_residual_rows
                    if row["new_weighted_template_residual"] < 0
                ),
                column="new_weighted_template_residual",
                count=top_n,
                descending=False,
            ),
        },
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> str:
    headers = [header for _, header in columns]
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        output.append(
            "| " + " | ".join(["None", *([""] * (len(headers) - 1))]) + " |"
        )
        return "\n".join(output)
    for row in rows:
        output.append(
            "| "
            + " | ".join(_format_cell(row.get(key)) for key, _ in columns)
            + " |"
        )
    return "\n".join(output)


def render_release_change_report(report: Mapping[str, Any]) -> str:
    """Render the concise review artifact printed before promotion."""

    population = report["population"]
    projections = report["projections"]
    residuals = report["weighted_template_residuals"]
    projection_columns = (
        ("league", "League"),
        ("player", "Player"),
        ("pos", "Pos"),
        ("old_projection_ppg", "Old PPG"),
        ("new_projection_ppg", "New PPG"),
        ("projection_delta_ppg", "Delta"),
    )
    residual_columns = (
        ("league", "League"),
        ("player", "Player"),
        ("pos", "Pos"),
        ("new_projection_ppg", "Projection"),
        ("old_weighted_template_residual", "Old Resid"),
        ("new_weighted_template_residual", "New Resid"),
        ("weighted_template_residual_delta", "Resid Delta"),
    )
    population_columns = (
        ("status", "Status"),
        ("league", "League"),
        ("player", "Player"),
        ("pos", "Pos"),
        ("projection_ppg", "PPG"),
    )
    summary_columns = (
        ("league", "League"),
        ("old_count", "Old"),
        ("new_count", "New"),
        ("common_count", "Common"),
        ("added_count", "Added"),
        ("dropped_count", "Dropped"),
    )
    retrospective = " (retrospective)" if report.get("retrospective") else ""
    sections = [
        f"# Production Release Change Report{retrospective}",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Slice: `{report['year']} / {report['dataset']}`",
        (
            "- Population: "
            f"{population['old_count']} old, {population['new_count']} new, "
            f"{population['added_count']} added, "
            f"{population['dropped_count']} dropped"
        ),
        "",
        "## Population by league",
        "",
        _markdown_table(population["by_league"], summary_columns),
        "",
        "## Added and dropped players",
        "",
        _markdown_table(population["changes"], population_columns),
        "",
        f"## Top {report['top_n']} projection increases",
        "",
        _markdown_table(projections["increases"], projection_columns),
        "",
        f"## Top {report['top_n']} projection decreases",
        "",
        _markdown_table(projections["decreases"], projection_columns),
        "",
        f"## Top {report['top_n']} positive weighted-template residuals",
        "",
        _markdown_table(residuals["positive"], residual_columns),
        "",
        f"## Top {report['top_n']} negative weighted-template residuals",
        "",
        _markdown_table(residuals["negative"], residual_columns),
        "",
        (
            "Weighted-template residual = probability-weighted donor "
            "`active_ppg_resid` under the exact app sampling probabilities."
        ),
        "",
    ]
    return "\n".join(sections)


def write_release_change_report(
    output_directory: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist JSON lineage plus the concise Markdown preview."""

    output_directory = Path(output_directory).resolve()
    json_path = output_directory / "release_change_report.json"
    markdown_path = output_directory / "release_change_report.md"
    _atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(markdown_path, render_release_change_report(report))
    return {
        "report_version": REPORT_VERSION,
        "run_id": str(report["run_id"]),
        "generated_at_utc": str(report["generated_at_utc"]),
        "baseline_simulation_sha256": str(
            report["baseline_simulation"]["sha256"]
        ),
        "candidate_simulation_sha256": str(
            report["candidate_simulation"]["sha256"]
        ),
        "json": _file_receipt(json_path),
        "markdown": _file_receipt(markdown_path),
    }


def load_verified_release_change_report(
    receipt: Mapping[str, Any],
    *,
    baseline_database: Path,
    candidate_database: Path,
) -> dict[str, Any]:
    """Verify a saved report against both databases before reusing it."""

    if receipt.get("report_version") != REPORT_VERSION:
        raise ValueError("Release change report version is missing or stale")
    baseline_sha256 = sha256_file(Path(baseline_database))
    candidate_sha256 = sha256_file(Path(candidate_database))
    if receipt.get("baseline_simulation_sha256") != baseline_sha256:
        raise RuntimeError("Release change report baseline no longer matches live")
    if receipt.get("candidate_simulation_sha256") != candidate_sha256:
        raise RuntimeError("Release change report candidate no longer matches staging")
    for label in ("json", "markdown"):
        state = receipt.get(label)
        if not isinstance(state, Mapping):
            raise ValueError(f"Release change report lacks {label} receipt")
        path = Path(str(state["path"])).resolve()
        if not path.is_file() or sha256_file(path) != state.get("sha256"):
            raise RuntimeError(f"Saved release change {label} changed after review")
    report = json.loads(Path(receipt["json"]["path"]).read_text(encoding="utf-8"))
    if report.get("report_version") != REPORT_VERSION:
        raise ValueError("Saved release change JSON has a stale version")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--dataset", default="final_ensemble")
    parser.add_argument("--league", action="append", dest="leagues")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--retrospective", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    leagues = tuple(args.leagues or ("dk", "nffc", "beta", "nv"))
    report = build_release_change_report(
        args.baseline,
        args.candidate,
        year=args.year,
        dataset=args.dataset,
        run_id=args.run_id,
        leagues=leagues,
        top_n=args.top_n,
        retrospective=args.retrospective,
    )
    write_release_change_report(args.output_dir, report)
    print(render_release_change_report(report), flush=True)


if __name__ == "__main__":
    main()
