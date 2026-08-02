"""Run a Streamlit smoke test against an explicit staged app database."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_SPECS = {
    "auction": {
        "root": REPO_ROOT.parent / "Fantasy_Football_App" / "app",
        "entrypoint": "ffapp.py",
        "database_environment": "AUCTION_SIMULATION_DB",
    },
    "snake": {
        "root": REPO_ROOT.parent / "Fantasy_Football_Snake" / "app",
        "entrypoint": "snake_draft_app.py",
        "database_environment": "SNAKE_SIMULATION_DB",
    },
}


def validate_database(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Staged app database not found: {path}")
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"Staged app database failed integrity: {path}")


def run_app_smoke(
    app_name: str,
    database: Path,
    *,
    timeout_seconds: int = 60,
    expected_year: int | None = None,
    required_leagues: tuple[str, ...] = (),
) -> dict[str, object]:
    app_name = str(app_name).strip().lower()
    if app_name not in APP_SPECS:
        raise ValueError(f"Unsupported app: {app_name}")
    database = Path(database).expanduser().resolve()
    validate_database(database)

    spec = APP_SPECS[app_name]
    app_root = Path(spec["root"]).resolve()
    entrypoint = app_root / str(spec["entrypoint"])
    if not entrypoint.is_file():
        raise FileNotFoundError(f"App entrypoint not found: {entrypoint}")

    os.environ[str(spec["database_environment"])] = str(database)
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(entrypoint),
        default_timeout=timeout_seconds,
    ).run(timeout=timeout_seconds)
    def assert_clean(active_app) -> None:
        exceptions = [
            str(element.value) for element in active_app.exception
        ]
        errors = [str(element.value) for element in active_app.error]
        if exceptions or errors:
            raise RuntimeError(
                f"{app_name} staged AppTest failed: "
                f"exceptions={exceptions}, errors={errors}"
            )

    def selectbox(active_app, label: str):
        matches = [
            element
            for element in active_app.selectbox
            if str(element.label) == label
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{app_name} staged AppTest expected one {label!r} "
                f"selectbox; observed {len(matches)}"
            )
        return matches[0]

    assert_clean(app)
    year_label = "Prediction Year" if app_name == "snake" else "Year"
    if expected_year is not None:
        observed_year = int(selectbox(app, year_label).value)
        if observed_year != int(expected_year):
            raise RuntimeError(
                f"{app_name} staged AppTest opened year {observed_year}, "
                f"expected {expected_year}"
            )

    tested_leagues: list[str] = []
    if required_leagues:
        league_label = "League Type" if app_name == "snake" else "League"
        league_widget = selectbox(app, league_label)
        options = {
            str(option).strip().lower()
            for option in league_widget.options
        }
        missing = sorted(set(required_leagues).difference(options))
        if missing:
            raise RuntimeError(
                f"{app_name} staged AppTest lacks required league options: "
                f"{missing}; available={sorted(options)}"
            )
        for league in required_leagues:
            app = selectbox(app, league_label).select(league).run(
                timeout=timeout_seconds
            )
            assert_clean(app)
            if expected_year is not None:
                observed_year = int(selectbox(app, year_label).value)
                if observed_year != int(expected_year):
                    raise RuntimeError(
                        f"{app_name} {league} context opened year "
                        f"{observed_year}, expected {expected_year}"
                    )
            tested_leagues.append(league)
    surface_counts = {
        "dataframes": len(app.dataframe),
        "selectboxes": len(app.selectbox),
        "buttons": len(app.button),
    }
    empty_surfaces = [
        surface
        for surface, count in surface_counts.items()
        if int(count) <= 0
    ]
    if empty_surfaces:
        raise RuntimeError(
            f"{app_name} staged AppTest rendered an incomplete draft surface: "
            f"missing={empty_surfaces}, counts={surface_counts}"
        )

    return {
        "app": app_name,
        "database": str(database),
        "exceptions": 0,
        "errors": 0,
        "expected_year": expected_year,
        "tested_leagues": tested_leagues,
        "titles": len(app.title),
        **surface_counts,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", choices=sorted(APP_SPECS))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--expected-year", type=int)
    parser.add_argument(
        "--require-league",
        action="append",
        default=[],
        help="League selector option that must render and rerun cleanly.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    result = run_app_smoke(
        args.app,
        args.database,
        timeout_seconds=args.timeout_seconds,
        expected_year=args.expected_year,
        required_leagues=tuple(args.require_league),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
